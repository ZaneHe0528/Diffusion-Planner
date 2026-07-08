#!/usr/bin/env python3
"""Warm-start 工具：ego-shift、邻车 token 匹配、匀速外推、再加噪、帧间缓存。"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear

PLAN_DT_S = 0.1


@dataclass
class WarmStartCache:
    x0_norm: torch.Tensor | None = None
    anchor_xyh: torch.Tensor | None = None
    neighbor_tokens: list[str] = field(default_factory=list)


def _rotate_xy(xy: torch.Tensor, dh: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(dh), torch.sin(dh)
    x, y = xy[..., 0], xy[..., 1]
    return torch.stack([c * x - s * y, s * x + c * y], dim=-1)


def ego_shift_trajectory(
    trajectory: torch.Tensor,
    prev_anchor_xyh: torch.Tensor,
    cur_anchor_xyh: torch.Tensor,
) -> torch.Tensor:
    """把上一帧物理坐标轨迹对齐到当前 ego 帧。

    trajectory: [B, P, T*4] 或 [B, P, T, 4]
    anchor: [B, 3] = x, y, heading
    """
    orig_shape = trajectory.shape
    if trajectory.dim() == 3 and orig_shape[-1] % 4 == 0:
        b, p, flat = trajectory.shape
        t = flat // 4
        x = trajectory.reshape(b, p, t, 4)
    else:
        x = trajectory
        b, p, t, _ = x.shape

    dx = prev_anchor_xyh[:, 0] - cur_anchor_xyh[:, 0]
    dy = prev_anchor_xyh[:, 1] - cur_anchor_xyh[:, 1]
    dh = prev_anchor_xyh[:, 2] - cur_anchor_xyh[:, 2]
    dh = torch.atan2(torch.sin(dh), torch.cos(dh))

    xy = x[..., :2]
    trans = torch.stack([dx, dy], dim=-1).view(b, 1, 1, 2)
    xy_shift = _rotate_xy(xy + trans, dh.view(b, 1, 1))
    cos_h = x[..., 2]
    sin_h = x[..., 3]
    new_cos = cos_h * torch.cos(dh).view(b, 1, 1) - sin_h * torch.sin(dh).view(b, 1, 1)
    new_sin = sin_h * torch.cos(dh).view(b, 1, 1) + cos_h * torch.sin(dh).view(b, 1, 1)
    out = torch.stack([xy_shift[..., 0], xy_shift[..., 1], new_cos, new_sin], dim=-1)

    if len(orig_shape) == 3:
        return out.reshape(b, p, -1)
    return out


def _slot_mean_std(state_normalizer, slot: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    mean = state_normalizer.mean[slot].to(device=device, dtype=dtype).reshape(1, 1, 1, 4)
    std = state_normalizer.std[slot].to(device=device, dtype=dtype).reshape(1, 1, 1, 4)
    return mean, std


def denormalize_slot_trajectory(x0_norm: torch.Tensor, state_normalizer, slot: int) -> torch.Tensor:
    orig_shape = x0_norm.shape
    if x0_norm.dim() == 3:
        b, p, flat = x0_norm.shape
        x = x0_norm.reshape(b, p, flat // 4, 4)
    else:
        x = x0_norm
    mean, std = _slot_mean_std(state_normalizer, slot, x.device, x.dtype)
    out = x * std + mean
    if len(orig_shape) == 3:
        return out.reshape(orig_shape)
    return out


def normalize_slot_trajectory(x0_phys: torch.Tensor, state_normalizer, slot: int) -> torch.Tensor:
    orig_shape = x0_phys.shape
    if x0_phys.dim() == 3:
        b, p, flat = x0_phys.shape
        x = x0_phys.reshape(b, p, flat // 4, 4)
    else:
        x = x0_phys
    mean, std = _slot_mean_std(state_normalizer, slot, x.device, x.dtype)
    out = (x - mean) / std
    if len(orig_shape) == 3:
        return out.reshape(orig_shape)
    return out


def ego_shift_normalized_trajectory(
    x0_norm: torch.Tensor,
    prev_anchor_xyh: torch.Tensor,
    cur_anchor_xyh: torch.Tensor,
    state_normalizer,
    prev_slot: int,
    cur_slot: int,
) -> torch.Tensor:
    x0_phys = denormalize_slot_trajectory(x0_norm, state_normalizer, prev_slot)
    shifted_phys = ego_shift_trajectory(x0_phys, prev_anchor_xyh, cur_anchor_xyh)
    return normalize_slot_trajectory(shifted_phys, state_normalizer, cur_slot)


def constant_velocity_neighbor_x0(
    neighbor_slot: int,
    neighbor_past: torch.Tensor,
    future_len: int,
    state_normalizer,
    device: torch.device,
) -> torch.Tensor:
    """用当前观测匀速外推构造邻车归一化 x0 行 [flat]."""
    flat = (1 + future_len) * 4
    if neighbor_past.abs().sum() < 1e-6:
        return torch.zeros(flat, device=device)

    cur = neighbor_past[-1]
    x, y = float(cur[0]), float(cur[1])
    cos_h, sin_h = float(cur[2]), float(cur[3])
    vx, vy = float(cur[4]), float(cur[5])

    traj = torch.zeros(1 + future_len, 4, device=device)
    for k in range(1 + future_len):
        traj[k, 0] = x + k * PLAN_DT_S * vx
        traj[k, 1] = y + k * PLAN_DT_S * vy
        traj[k, 2] = cos_h
        traj[k, 3] = sin_h

    agent_idx = neighbor_slot
    mean, std = _slot_mean_std(state_normalizer, agent_idx, device, traj.dtype)
    normed = (traj.view(1, 1, 1 + future_len, 4) - mean) / std
    return normed.reshape(-1)


def build_neighbor_x0_row(
    slot: int,
    cache: WarmStartCache,
    cur_anchor_xyh: torch.Tensor,
    current_neighbor_tokens: list[str] | None,
    neighbor_agents_past: torch.Tensor | None,
    future_len: int,
    state_normalizer,
    device: torch.device,
) -> torch.Tensor:
    """构造单个邻车行的归一化 x0 [B, 1, flat]。"""
    b = 1
    flat = (1 + future_len) * 4
    cur_tok = ""
    if current_neighbor_tokens and slot - 1 < len(current_neighbor_tokens):
        cur_tok = current_neighbor_tokens[slot - 1]

    token_to_prev_slot: dict[str, int] = {}
    for i, tok in enumerate(cache.neighbor_tokens):
        if tok:
            token_to_prev_slot[tok] = i

    if cur_tok and cur_tok in token_to_prev_slot:
        prev_row = token_to_prev_slot[cur_tok] + 1
        nbr_cache = cache.x0_norm[:, prev_row : prev_row + 1, :]
        return ego_shift_normalized_trajectory(
            nbr_cache,
            cache.anchor_xyh,
            cur_anchor_xyh,
            state_normalizer,
            prev_slot=prev_row,
            cur_slot=slot,
        )

    if neighbor_agents_past is not None and state_normalizer is not None:
        nbr_past = neighbor_agents_past[0, slot - 1]
        row = constant_velocity_neighbor_x0(slot, nbr_past, future_len, state_normalizer, device)
        return row.view(1, 1, flat)

    return torch.zeros(b, 1, flat, device=device)


def build_warmstart_init(
    cache: WarmStartCache,
    current_states: torch.Tensor,
    cur_anchor_xyh: torch.Tensor,
    t_start: float,
    *,
    current_neighbor_tokens: list[str] | None = None,
    neighbor_agents_past: torch.Tensor | None = None,
    future_len: int = 80,
    state_normalizer=None,
    sde: VPSDE_linear | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """从缓存构造 warm-start 初始噪声态与参考 x0；无缓存则返回 (None, None)。"""
    if cache.x0_norm is None or cache.anchor_xyh is None:
        return None, None

    b, p, _ = current_states.shape
    device = current_states.device

    if state_normalizer is None:
        return None, None

    ego_shifted = ego_shift_normalized_trajectory(
        cache.x0_norm[:, 0:1, :],
        cache.anchor_xyh,
        cur_anchor_xyh,
        state_normalizer,
        prev_slot=0,
        cur_slot=0,
    )
    rows = [ego_shifted]
    for slot in range(1, p):
        rows.append(
            build_neighbor_x0_row(
                slot,
                cache,
                cur_anchor_xyh,
                current_neighbor_tokens,
                neighbor_agents_past,
                future_len,
                state_normalizer,
                device,
            )
        )

    x0 = torch.cat(rows, dim=1)
    x0 = x0.reshape(b, p, -1, 4)
    x0[:, :, 0, :] = current_states
    ref_x0_norm = x0.reshape(b, p, -1).clone()
    x_init = renoise_to_t(ref_x0_norm, t_start, sde=sde)
    return x_init, ref_x0_norm


def renoise_to_t(
    x0: torch.Tensor,
    t_start: float,
    sde: VPSDE_linear | None = None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """x_{t_s} = alpha(t_s)*x0 + sigma(t_s)*noise。"""
    sde = sde or VPSDE_linear()
    t = torch.full((x0.shape[0],), float(t_start), device=x0.device, dtype=x0.dtype)
    mean, sigma = sde.marginal_prob(x0, t)
    if noise is None:
        noise = torch.randn_like(x0)
    return mean + sigma * noise


def update_cache(
    cache: WarmStartCache,
    x0_norm: torch.Tensor,
    anchor_xyh: torch.Tensor,
    neighbor_tokens: list[str] | None = None,
) -> None:
    cache.x0_norm = x0_norm.detach()
    cache.anchor_xyh = anchor_xyh.detach()
    cache.neighbor_tokens = list(neighbor_tokens) if neighbor_tokens else []


def ego_max_distance_m(
    pred_norm: torch.Tensor,
    ref_norm: torch.Tensor,
    state_normalizer,
    ego_slot: int = 0,
) -> float:
    """重叠段 ego 轨迹逐点欧氏距离 max（物理米）。"""
    pred = state_normalizer.inverse(pred_norm.reshape(1, 1, -1, 4))[:, ego_slot, :, :2]
    ref = state_normalizer.inverse(ref_norm.reshape(1, 1, -1, 4))[:, ego_slot, :, :2]
    dist = torch.linalg.norm(pred - ref, dim=-1)
    return float(dist.max().item())
