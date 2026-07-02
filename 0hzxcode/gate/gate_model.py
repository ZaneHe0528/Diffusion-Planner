#!/usr/bin/env python3
"""BLUE 式轻量 gate：原始观测 -> 帧间变化距离 d_hat + 变化档位 level。

结构（对齐 BLUE"冻结主干 + 单隐层 MLP gate"的精神，参数 ~0.2M）：
  每个特征组一个极小编码分支（token 级共享 MLP + masked mean/max 池化），
  concat 各组 embedding -> LayerNorm -> 单隐层 MLP trunk -> 双头：
    - reg 头：回归标准化 log1p(d)（Huber）
    - cls 头：K 档 level 分类（加权 CE），档位边界（米）存于 checkpoint

特征组开关：
  - 训练期 group-dropout 随机置零各组 embedding，使单个模型支持推理期任意关组；
  - 推理期 forward(..., disabled_groups={...}) 即时关闭；
  - 训练期彻底排除某组用 GateConfig.enabled_groups（重训消融）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from gate_features import FEATURE_GROUPS, GROUP_BY_NAME

WHEEL_BASE = 3.089  # pacifica，与 data_process 一致
DT = 0.1            # 闭环仿真步长 [s]


@dataclass
class GateConfig:
    enabled_groups: list[str] = field(default_factory=lambda: [g.name for g in FEATURE_GROUPS])
    embed_dim: int = 64          # 每组 embedding 维度
    token_hidden: int = 128      # token 级共享 MLP 隐层
    trunk_hidden: int = 128      # trunk 单隐层宽度（对齐 BLUE hidden=128）
    num_levels: int = 4          # level 档数（低/中/高/满）
    dropout: float = 0.1
    group_dropout: float = 0.15  # 训练期整组置零概率
    use_encoding: bool = False   # 可选：外挂 planner encoding 作对照组
    encoding_dim: int = 192


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


def derive_ego_kinematics(ego: torch.Tensor) -> torch.Tensor:
    """从 ego_history [B,21,7]（x,y,heading,vx,vy,ax,ay，当前帧系）派生运动学特征。

    这些量闭环推理时同样可从 history buffer 现成算出。
    """
    xy, h = ego[..., 0:2], ego[..., 2]
    speed = torch.hypot(ego[..., 3], ego[..., 4])          # [B,21]
    accel = torch.hypot(ego[..., 5], ego[..., 6])          # [B,21]
    dh = h[:, 1:] - h[:, :-1]
    dh = torch.atan2(torch.sin(dh), torch.cos(dh))
    yaw_rate = dh / DT                                     # [B,20]

    cur_speed = speed[:, -1]
    cur_yaw_rate = yaw_rate[:, -1]
    curvature = cur_yaw_rate / cur_speed.clamp_min(0.5)
    steering = torch.atan(curvature * WHEEL_BASE)
    dist_2s = torch.hypot(xy[:, 0, 0], xy[:, 0, 1])        # 当前点即原点，首帧位置≈2s 位移
    net_heading = h[:, -1] - h[:, 0]
    net_heading = torch.atan2(torch.sin(net_heading), torch.cos(net_heading))

    return torch.stack(
        [
            cur_speed,
            speed.mean(dim=1),
            speed.amax(dim=1),
            speed[:, -1] - speed[:, 0],
            accel[:, -1],
            accel.mean(dim=1),
            accel.amax(dim=1),
            cur_yaw_rate,
            yaw_rate.abs().amax(dim=1),
            curvature,
            steering,
            dist_2s,
            net_heading,
            (cur_speed < 0.1).float(),
        ],
        dim=1,
    )  # [B,14]


EGO_DERIVED_DIM = 14


class TokenBranch(nn.Module):
    """token 集合分支：共享 MLP + masked mean/max 池化 -> 组 embedding。"""

    def __init__(self, token_dim: int, cfg: GateConfig):
        super().__init__()
        self.phi = _mlp(token_dim, cfg.token_hidden, cfg.embed_dim, cfg.dropout)
        self.proj = nn.Linear(2 * cfg.embed_dim, cfg.embed_dim)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # tokens [B,N,F]，valid [B,N]
        feat = self.phi(tokens)
        mask = valid.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        mean_pool = (feat * mask).sum(dim=1) / denom
        neg_inf = torch.finfo(feat.dtype).min
        max_pool = feat.masked_fill(~valid.unsqueeze(-1), neg_inf).amax(dim=1)
        max_pool = torch.where(valid.any(dim=1, keepdim=True), max_pool, torch.zeros_like(max_pool))
        return self.proj(torch.cat([mean_pool, max_pool], dim=1))


class RawObsGate(nn.Module):
    """原始观测 gate。输入为数据集/推理时的原始数组字典（float32）。"""

    def __init__(self, cfg: GateConfig):
        super().__init__()
        self.cfg = cfg
        self.group_order = [g.name for g in FEATURE_GROUPS if g.name in cfg.enabled_groups]
        if cfg.use_encoding:
            self.group_order.append("encoding")

        branches: dict[str, nn.Module] = {}
        for name in self.group_order:
            if name == "ego_history":
                branches[name] = _mlp(21 * 7 + EGO_DERIVED_DIM, cfg.token_hidden, cfg.embed_dim, cfg.dropout)
            elif name == "neighbor_agents":
                branches[name] = TokenBranch(21 * 11, cfg)
            elif name == "lanes":
                branches[name] = TokenBranch(20 * 12 + 2, cfg)
            elif name == "route_lanes":
                branches[name] = TokenBranch(20 * 12 + 2, cfg)
            elif name == "static_objects":
                branches[name] = TokenBranch(10, cfg)
            elif name == "prev_d":
                branches[name] = _mlp(3, 32, cfg.embed_dim, cfg.dropout)
            elif name == "encoding":
                branches[name] = _mlp(cfg.encoding_dim, cfg.token_hidden, cfg.embed_dim, cfg.dropout)
            else:
                raise ValueError(f"no branch for group {name}")
        self.branches = nn.ModuleDict(branches)

        trunk_in = cfg.embed_dim * len(self.group_order)
        self.trunk_norm = nn.LayerNorm(trunk_in)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, cfg.trunk_hidden), nn.SiLU(), nn.Dropout(cfg.dropout)
        )
        self.head_reg = nn.Linear(cfg.trunk_hidden, 1)
        self.head_cls = nn.Linear(cfg.trunk_hidden, cfg.num_levels)

        # 每组扁平特征的标准化统计（fit_normalization 填充）
        for name in self.group_order:
            dim = self._group_feat_dim(name)
            self.register_buffer(f"norm_mean_{name}", torch.zeros(dim))
            self.register_buffer(f"norm_std_{name}", torch.ones(dim))
        # 回归目标 z = (log1p(d) - mean) / std
        self.register_buffer("target_mean", torch.zeros(1))
        self.register_buffer("target_std", torch.ones(1))
        # level 档位边界（米），训练时按 train 分位数或 CLI 指定
        self.register_buffer("level_edges_m", torch.zeros(cfg.num_levels - 1))

    def _group_feat_dim(self, name: str) -> int:
        return {
            "ego_history": 21 * 7 + EGO_DERIVED_DIM,
            "neighbor_agents": 21 * 11,
            "lanes": 20 * 12 + 2,
            "route_lanes": 20 * 12 + 2,
            "static_objects": 10,
            "prev_d": 3,
            "encoding": self.cfg.encoding_dim,
        }[name]

    # ------------------------------------------------------------------
    # 原始数组 -> (扁平特征, 有效 mask)
    # ------------------------------------------------------------------
    def group_features(self, name: str, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        if name == "ego_history":
            ego = batch["ego_history"].float()
            flat = torch.cat([ego.reshape(ego.shape[0], -1), derive_ego_kinematics(ego)], dim=1)
            return flat, None
        if name == "neighbor_agents":
            x = batch["neighbor_agents_past"].float()                    # [B,32,21,11]
            valid = x.abs().sum(dim=(2, 3)) > 0
            return x.reshape(x.shape[0], x.shape[1], -1), valid
        if name in ("lanes", "route_lanes"):
            x = batch[name].float()                                      # [B,N,20,12]
            valid = x.abs().sum(dim=(2, 3)) > 0
            flat = x.reshape(x.shape[0], x.shape[1], -1)
            sl = batch[f"{name}_speed_limit"].float().reshape(x.shape[0], x.shape[1], 1)
            has_sl = batch[f"{name}_has_speed_limit"].float().reshape(x.shape[0], x.shape[1], 1)
            return torch.cat([flat, sl, has_sl], dim=2), valid
        if name == "static_objects":
            x = batch["static_objects"].float()                          # [B,5,10]
            valid = x.abs().sum(dim=2) > 0
            return x, valid
        if name == "prev_d":
            return batch["prev_d"].float(), None
        if name == "encoding":
            return batch["encoding"].float(), None
        raise ValueError(name)

    def _normalize(self, name: str, feat: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        mean = getattr(self, f"norm_mean_{name}")
        std = getattr(self, f"norm_std_{name}")
        out = (feat - mean) / std
        if valid is not None:
            out = out * valid.unsqueeze(-1).float()  # padding 归零，避免标准化后引入偏移
        return out

    # ------------------------------------------------------------------
    def forward(
        self,
        batch: dict[str, torch.Tensor],
        disabled_groups: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        disabled = disabled_groups or set()
        embeds = []
        for name in self.group_order:
            feat, valid = self.group_features(name, batch)
            feat = self._normalize(name, feat, valid)
            if isinstance(self.branches[name], TokenBranch):
                emb = self.branches[name](feat, valid)
            else:
                emb = self.branches[name](feat)
            if name in disabled:
                emb = torch.zeros_like(emb)
            elif self.training and self.cfg.group_dropout > 0 and len(self.group_order) > 1:
                drop = torch.rand(emb.shape[0], 1, device=emb.device) < self.cfg.group_dropout
                emb = emb.masked_fill(drop, 0.0)
            embeds.append(emb)

        h = self.trunk(self.trunk_norm(torch.cat(embeds, dim=1)))
        z = self.head_reg(h).squeeze(-1)
        logits = self.head_cls(h)
        return {"z": z, "logits": logits, "d_hat": self.z_to_d(z)}

    # ------------------------------------------------------------------
    def z_to_d(self, z: torch.Tensor) -> torch.Tensor:
        log_d = z * self.target_std + self.target_mean
        return torch.expm1(log_d).clamp_min(0.0)

    def d_to_z(self, d: torch.Tensor) -> torch.Tensor:
        return (torch.log1p(d.clamp_min(0.0)) - self.target_mean) / self.target_std

    def d_to_level(self, d: torch.Tensor) -> torch.Tensor:
        return torch.bucketize(d, self.level_edges_m)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit_normalization(
        self,
        iter_batches,
        level_edges_m: list[float],
        d_train: np.ndarray,
    ) -> None:
        """用 train 集统计填充各组标准化 buffer 与目标 z 标准化、level 边界。"""
        sums = {n: None for n in self.group_order}
        sqs = {n: None for n in self.group_order}
        cnt = {n: 0.0 for n in self.group_order}
        for batch in iter_batches:
            for name in self.group_order:
                feat, valid = self.group_features(name, batch)
                if valid is None:
                    f2 = feat
                    c = float(feat.shape[0])
                else:
                    m = valid.unsqueeze(-1).float()
                    f2 = feat * m
                    c = float(valid.sum().item())
                s = f2.sum(dim=tuple(range(f2.dim() - 1)))
                q = (f2 * f2).sum(dim=tuple(range(f2.dim() - 1)))
                sums[name] = s if sums[name] is None else sums[name] + s
                sqs[name] = q if sqs[name] is None else sqs[name] + q
                cnt[name] += c
        for name in self.group_order:
            c = max(cnt[name], 1.0)
            mean = sums[name] / c
            var = (sqs[name] / c - mean * mean).clamp_min(1e-12)
            std = var.sqrt()
            std[std < 1e-6] = 1.0
            getattr(self, f"norm_mean_{name}").copy_(mean)
            getattr(self, f"norm_std_{name}").copy_(std)

        log_d = np.log1p(np.clip(d_train, 0.0, None))
        t_std = float(log_d.std()) or 1.0
        self.target_mean.fill_(float(log_d.mean()))
        self.target_std.fill_(t_std)
        self.level_edges_m.copy_(torch.tensor(level_edges_m, dtype=torch.float32))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_gate(path, device: str = "cpu") -> tuple[RawObsGate, dict]:
    """加载训练好的 gate（闭环集成入口）。返回 (model, checkpoint dict)。"""
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = GateConfig(**state["config"])
    model = RawObsGate(cfg).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state
