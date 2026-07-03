#!/usr/bin/env python3
"""Warm-start 工具：ego-shift、再加噪、帧间缓存。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear


@dataclass
class WarmStartCache:
    x0_norm: torch.Tensor | None = None
    anchor_xyh: torch.Tensor | None = None


def _rotate_xy(xy: torch.Tensor, dh: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(dh), torch.sin(dh)
    x, y = xy[..., 0], xy[..., 1]
    return torch.stack([c * x - s * y, s * x + c * y], dim=-1)


def ego_shift_trajectory(
    x0_norm: torch.Tensor,
    prev_anchor_xyh: torch.Tensor,
    cur_anchor_xyh: torch.Tensor,
) -> torch.Tensor:
    """把上一帧归一化轨迹 x0 对齐到当前 ego 帧。

    x0_norm: [B, P, T*4] 或 [B, P, T, 4]
    anchor: [B, 3] = x, y, heading
    """
    orig_shape = x0_norm.shape
    if x0_norm.dim() == 3 and orig_shape[-1] % 4 == 0:
        b, p, flat = x0_norm.shape
        t = flat // 4
        x = x0_norm.reshape(b, p, t, 4)
    else:
        x = x0_norm
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


def renoise_to_t(
    x0: torch.Tensor,
    t_start: float,
    sde: VPSDE_linear | None = None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """x_{t_s} = alpha(t_s)*x0 + sigma(t_s)*noise。"""
    sde = sde or VPSDE_linear()
    t = torch.full((x0.shape[0],), float(t_start), device=x0.device, dtype=x0.dtype)
    alpha, sigma = sde.marginal_prob(x0, t)
    while alpha.dim() < x0.dim():
        alpha = alpha.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
    if noise is None:
        noise = torch.randn_like(x0)
    return alpha * x0 + sigma * noise


def build_warmstart_init(
    cache: WarmStartCache,
    current_states: torch.Tensor,
    cur_anchor_xyh: torch.Tensor,
    t_start: float,
    sde: VPSDE_linear | None = None,
) -> torch.Tensor | None:
    """从缓存构造 warm-start 初始噪声态 x_T；无缓存则返回 None。"""
    if cache.x0_norm is None or cache.anchor_xyh is None:
        return None

    x0 = ego_shift_trajectory(cache.x0_norm, cache.anchor_xyh, cur_anchor_xyh)
    b, p, _ = current_states.shape
    x0 = x0.reshape(b, p, -1, 4)
    x0[:, :, 0, :] = current_states
    x0 = x0.reshape(b, p, -1)
    return renoise_to_t(x0, t_start, sde=sde)


def update_cache(
    cache: WarmStartCache,
    x0_norm: torch.Tensor,
    anchor_xyh: torch.Tensor,
) -> None:
    cache.x0_norm = x0_norm.detach()
    cache.anchor_xyh = anchor_xyh.detach()
