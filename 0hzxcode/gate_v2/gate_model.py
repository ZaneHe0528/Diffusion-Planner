#!/usr/bin/env python3
"""gate_v2 轻量门控：ego + 邻车 -> d_hat（仅回归头）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from gate_features import FEATURE_GROUPS, GROUP_BY_NAME

WHEEL_BASE = 3.089
DT = 0.1


@dataclass
class GateConfig:
    enabled_groups: list[str] = field(default_factory=lambda: [g.name for g in FEATURE_GROUPS])
    embed_dim: int = 64
    token_hidden: int = 128
    trunk_hidden: int = 128
    num_levels: int = 4
    dropout: float = 0.1
    group_dropout: float = 0.15


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


def derive_ego_kinematics(ego: torch.Tensor) -> torch.Tensor:
    xy, h = ego[..., 0:2], ego[..., 2]
    speed = torch.hypot(ego[..., 3], ego[..., 4])
    accel = torch.hypot(ego[..., 5], ego[..., 6])
    dh = h[:, 1:] - h[:, :-1]
    dh = torch.atan2(torch.sin(dh), torch.cos(dh))
    yaw_rate = dh / DT

    cur_speed = speed[:, -1]
    cur_yaw_rate = yaw_rate[:, -1]
    curvature = cur_yaw_rate / cur_speed.clamp_min(0.5)
    steering = torch.atan(curvature * WHEEL_BASE)
    dist_2s = torch.hypot(xy[:, 0, 0], xy[:, 0, 1])
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
    )


EGO_DERIVED_DIM = 14


class TokenBranch(nn.Module):
    def __init__(self, token_dim: int, cfg: GateConfig):
        super().__init__()
        self.phi = _mlp(token_dim, cfg.token_hidden, cfg.embed_dim, cfg.dropout)
        self.proj = nn.Linear(2 * cfg.embed_dim, cfg.embed_dim)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        feat = self.phi(tokens)
        mask = valid.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        mean_pool = (feat * mask).sum(dim=1) / denom
        neg_inf = torch.finfo(feat.dtype).min
        max_pool = feat.masked_fill(~valid.unsqueeze(-1), neg_inf).amax(dim=1)
        max_pool = torch.where(valid.any(dim=1, keepdim=True), max_pool, torch.zeros_like(max_pool))
        return self.proj(torch.cat([mean_pool, max_pool], dim=1))


class LiteGate(nn.Module):
    """ego + 邻车 -> d_hat；level 由 d_hat bucketize 得到。"""

    def __init__(self, cfg: GateConfig):
        super().__init__()
        self.cfg = cfg
        self.group_order = [g.name for g in FEATURE_GROUPS if g.name in cfg.enabled_groups]

        branches: dict[str, nn.Module] = {}
        for name in self.group_order:
            if name == "ego_history":
                branches[name] = _mlp(21 * 7 + EGO_DERIVED_DIM, cfg.token_hidden, cfg.embed_dim, cfg.dropout)
            elif name == "neighbor_agents":
                branches[name] = TokenBranch(21 * 11, cfg)
            elif name == "prev_d":
                branches[name] = _mlp(3, 32, cfg.embed_dim, cfg.dropout)
            else:
                raise ValueError(f"unknown group {name}")
        self.branches = nn.ModuleDict(branches)

        trunk_in = cfg.embed_dim * len(self.group_order)
        self.trunk_norm = nn.LayerNorm(trunk_in)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, cfg.trunk_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
        )
        self.head_reg = nn.Linear(cfg.trunk_hidden, 1)

        for name in self.group_order:
            dim = self._group_feat_dim(name)
            self.register_buffer(f"norm_mean_{name}", torch.zeros(dim))
            self.register_buffer(f"norm_std_{name}", torch.ones(dim))
        self.register_buffer("target_mean", torch.zeros(1))
        self.register_buffer("target_std", torch.ones(1))
        self.register_buffer("level_edges_m", torch.zeros(cfg.num_levels - 1))
        self.register_buffer("hard_threshold_m", torch.zeros(1))
        self.register_buffer("score_threshold_m", torch.zeros(1))

    def _group_feat_dim(self, name: str) -> int:
        return {
            "ego_history": 21 * 7 + EGO_DERIVED_DIM,
            "neighbor_agents": 21 * 11,
            "prev_d": 3,
        }[name]

    def group_features(self, name: str, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        if name == "ego_history":
            ego = batch["ego_history"].float()
            flat = torch.cat([ego.reshape(ego.shape[0], -1), derive_ego_kinematics(ego)], dim=1)
            return flat, None
        if name == "neighbor_agents":
            x = batch["neighbor_agents_past"].float()
            valid = x.abs().sum(dim=(2, 3)) > 0
            return x.reshape(x.shape[0], x.shape[1], -1), valid
        if name == "prev_d":
            return batch["prev_d"].float(), None
        raise ValueError(name)

    def _normalize(self, name: str, feat: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        mean = getattr(self, f"norm_mean_{name}")
        std = getattr(self, f"norm_std_{name}")
        out = (feat - mean) / std
        if valid is not None:
            out = out * valid.unsqueeze(-1).float()
        return out

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
        d_hat = self.z_to_d(z)
        return {"z": z, "d_hat": d_hat, "level": self.d_to_level(d_hat)}

    def z_to_d(self, z: torch.Tensor) -> torch.Tensor:
        log_d = z * self.target_std + self.target_mean
        return torch.expm1(log_d).clamp_min(0.0)

    def d_to_z(self, d: torch.Tensor) -> torch.Tensor:
        return (torch.log1p(d.clamp_min(0.0)) - self.target_mean) / self.target_std

    def d_to_level(self, d: torch.Tensor) -> torch.Tensor:
        return torch.bucketize(d, self.level_edges_m)

    @torch.no_grad()
    def fit_normalization(
        self,
        iter_batches,
        level_edges_m: list[float],
        d_train: np.ndarray,
        hard_threshold_m: float,
        score_threshold_m: float,
    ) -> None:
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
        self.hard_threshold_m.fill_(hard_threshold_m)
        self.score_threshold_m.fill_(score_threshold_m)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_gate(path, device: str = "cpu") -> tuple[LiteGate, dict]:
    key = (str(path), device)
    if not hasattr(load_gate, "_cache"):
        load_gate._cache = {}
    cache = load_gate._cache
    if key in cache:
        return cache[key]

    state = torch.load(path, map_location=device, weights_only=False)
    cfg = GateConfig(**state["config"])
    model = LiteGate(cfg).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    cache[key] = (model, state)
    return cache[key]
