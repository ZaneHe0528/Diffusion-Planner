#!/usr/bin/env python3
"""安全兜底：hard recall 阈值 + 邻车强交互保守上调。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from d_to_ops import WarmStartOps, d_hat_to_warmstart_ops, level_edges_from_checkpoint


@dataclass(frozen=True)
class SafetyConfig:
    hard_threshold_m: float
    score_threshold_m: float
    level_edges_m: tuple[float, ...]
    neighbor_close_m: float = 12.0
    neighbor_fast_mps: float = 3.0
    neighbor_bump_levels: int = 1
    max_level: int = 3
    base_steps: int = 10


def neighbor_interaction_score(neighbor_agents_past: torch.Tensor | np.ndarray) -> float:
    if isinstance(neighbor_agents_past, torch.Tensor):
        x = neighbor_agents_past.detach().float().cpu().numpy()
    else:
        x = np.asarray(neighbor_agents_past, dtype=np.float32)
    if x.ndim == 4:
        x = x[0]
    cur = x[:, -1, :]
    valid = np.abs(cur).sum(axis=-1) > 0
    if not valid.any():
        return 0.0
    cur = cur[valid]
    dist = np.hypot(cur[:, 0], cur[:, 1])
    speed = np.hypot(cur[:, 4], cur[:, 5])
    close = dist < 12.0
    fast = speed > 3.0
    score = float((close & fast).mean())
    score = max(score, float(close.mean()) * 0.5)
    return score


def apply_safety(
    d_hat_m: float,
    neighbor_agents_past: torch.Tensor | np.ndarray | None,
    cfg: SafetyConfig,
    *,
    device: str = "cpu",
) -> tuple[WarmStartOps, dict]:
    meta: dict = {
        "d_hat_m": float(d_hat_m),
        "forced_full": False,
        "score_alarm": bool(d_hat_m >= cfg.score_threshold_m),
        "neighbor_score": 0.0,
        "level_bump": 0,
    }
    edges = cfg.level_edges_m

    if d_hat_m >= cfg.hard_threshold_m:
        meta["forced_full"] = True
        return WarmStartOps(
            level=cfg.max_level,
            d_hat_m=float(d_hat_m),
            t_start=1.0,
            steps=cfg.base_steps,
            reuse_ratio=0.0,
        ), meta

    ops = d_hat_to_warmstart_ops(
        d_hat_m,
        level_edges_m=edges,
        base_steps=cfg.base_steps,
        device=device,
    )
    level = ops.level

    if neighbor_agents_past is not None:
        nscore = neighbor_interaction_score(neighbor_agents_past)
        meta["neighbor_score"] = nscore
        if nscore >= 0.25:
            bump = cfg.neighbor_bump_levels
            meta["level_bump"] = bump
            level = min(level + bump, cfg.max_level)
            ops = d_hat_to_warmstart_ops(
                edges[min(level, len(edges) - 1)],
                level_edges_m=edges,
                base_steps=cfg.base_steps,
                use_continuous_ts=False,
                device=device,
            )
            ops = WarmStartOps(
                level=level,
                d_hat_m=ops.d_hat_m,
                t_start=ops.t_start,
                steps=ops.steps,
                reuse_ratio=ops.reuse_ratio,
            )

    if level >= cfg.max_level:
        ops = WarmStartOps(
            level=cfg.max_level,
            d_hat_m=ops.d_hat_m,
            t_start=1.0,
            steps=cfg.base_steps,
            reuse_ratio=0.0,
        )

    meta["level"] = ops.level
    meta["t_start"] = ops.t_start
    meta["steps"] = ops.steps
    return ops, meta


def safety_config_from_checkpoint(state: dict[str, Any]) -> SafetyConfig:
    edges = level_edges_from_checkpoint(state)
    hard = float(state.get("hard_threshold_m", edges[-1] if edges else 1.0))
    score = float(state.get("score_threshold_m", hard))
    return SafetyConfig(
        hard_threshold_m=hard,
        score_threshold_m=score,
        level_edges_m=edges,
        max_level=len(edges),
    )
