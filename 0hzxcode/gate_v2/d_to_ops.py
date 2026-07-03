#!/usr/bin/env python3
"""d_hat -> warm-start 操作点 (t_s, steps, level)。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import diffusion_planner.model.diffusion_utils.dpm_solver_pytorch as dpm

# 全量 normalized-L2 分位（米/步，perstep_max_m 口径）
DEFAULT_LEVEL_EDGES_M = (0.275, 0.696, 1.385)

# 每档 (t_s 占 T 的比例, 去噪步数)
DEFAULT_LEVEL_OPS = (
    (0.12, 2),
    (0.35, 4),
    (0.65, 7),
    (1.00, 10),
)


@dataclass(frozen=True)
class WarmStartOps:
    level: int
    d_hat_m: float
    t_start: float
    steps: int
    reuse_ratio: float


def d_to_level(d_hat_m: float, level_edges_m: tuple[float, ...] = DEFAULT_LEVEL_EDGES_M) -> int:
    level = 0
    for edge in level_edges_m:
        if d_hat_m > edge:
            level += 1
    return min(level, len(DEFAULT_LEVEL_OPS) - 1)


def meters_to_sigma_norm(d_hat_m: float, xy_std_m: float = 20.0) -> float:
    """把米制 d 折算到归一化状态空间的噪声尺度（近似）。"""
    return max(d_hat_m / max(xy_std_m, 1e-6), 0.0)


def sigma_to_t_start(
    sigma_target: float,
    *,
    schedule: str = "linear",
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    t_max: float = 1.0,
    device: str = "cpu",
) -> float:
    """反解 VP 调度：marginal_std(t) ~= sigma_target。"""
    if sigma_target <= 1e-6:
        return 1e-3
    if sigma_target >= 0.999:
        return t_max

    ns = dpm.NoiseScheduleVP(
        schedule=schedule,
        continuous_beta_0=beta_min,
        continuous_beta_1=beta_max,
    )
    lo, hi = 1e-3, t_max
    t = torch.tensor([0.5 * (lo + hi)], device=device, dtype=torch.float32)
    for _ in range(32):
        sig = ns.marginal_std(t).item()
        if sig < sigma_target:
            lo = t.item()
        else:
            hi = t.item()
        t = torch.tensor([0.5 * (lo + hi)], device=device, dtype=torch.float32)
    return float(t.item())


def level_to_ops(level: int, base_steps: int = 10, t_max: float = 1.0) -> tuple[float, int]:
    level = int(max(0, min(level, len(DEFAULT_LEVEL_OPS) - 1)))
    t_frac, steps = DEFAULT_LEVEL_OPS[level]
    return t_frac * t_max, min(steps, base_steps)


def d_hat_to_warmstart_ops(
    d_hat_m: float,
    *,
    level_edges_m: tuple[float, ...] = DEFAULT_LEVEL_EDGES_M,
    xy_std_m: float = 20.0,
    base_steps: int = 10,
    t_max: float = 1.0,
    use_continuous_ts: bool = True,
    device: str = "cpu",
) -> WarmStartOps:
    level = d_to_level(d_hat_m, level_edges_m)
    if use_continuous_ts and level < len(DEFAULT_LEVEL_OPS) - 1:
        sigma = meters_to_sigma_norm(d_hat_m, xy_std_m)
        t_start = sigma_to_t_start(sigma, device=device, t_max=t_max)
        _, steps = level_to_ops(level, base_steps=base_steps, t_max=t_max)
        t_start = min(t_start, level_to_ops(level + 1, base_steps, t_max)[0])
    else:
        t_start, steps = level_to_ops(level, base_steps=base_steps, t_max=t_max)

    t_start = float(max(1e-3, min(t_start, t_max)))
    reuse_ratio = float(max(0.0, 1.0 - t_start / t_max))
    return WarmStartOps(
        level=level,
        d_hat_m=float(d_hat_m),
        t_start=t_start,
        steps=int(max(2, steps)),
        reuse_ratio=reuse_ratio,
    )
