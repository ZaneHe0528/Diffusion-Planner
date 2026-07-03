#!/usr/bin/env python3
"""gate_v2 单元测试：d->ops、warmstart、safety。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from d_to_ops import d_hat_to_warmstart_ops, d_to_level
from safety import SafetyConfig, apply_safety, neighbor_interaction_score
from warmstart import WarmStartCache, build_warmstart_init, ego_shift_trajectory, renoise_to_t


def test_d_to_level():
    assert d_to_level(0.1) == 0
    assert d_to_level(0.5) == 1
    assert d_to_level(1.0) == 2
    assert d_to_level(2.0) == 3


def test_d_hat_monotonic_ops():
    ops_low = d_hat_to_warmstart_ops(0.05, use_continuous_ts=True)
    ops_high = d_hat_to_warmstart_ops(1.5, use_continuous_ts=False)
    assert ops_low.t_start < ops_high.t_start
    assert ops_low.steps <= ops_high.steps


def test_safety_force_full():
    cfg = SafetyConfig(hard_threshold_m=1.0, score_threshold_m=0.8, level_edges_m=(0.275, 0.696, 1.385))
    ops, meta = apply_safety(0.9, None, cfg)
    assert meta["forced_full"] is True
    assert ops.t_start == 1.0
    assert ops.steps == 10


def test_neighbor_bump():
    nbr = np.zeros((32, 21, 11), dtype=np.float32)
    nbr[0, -1, 0:2] = [5.0, 1.0]
    nbr[0, -1, 4:6] = [4.0, 0.0]
    assert neighbor_interaction_score(nbr) > 0.2
    cfg = SafetyConfig(hard_threshold_m=5.0, score_threshold_m=5.0, level_edges_m=(0.275, 0.696, 1.385))
    ops, meta = apply_safety(0.1, nbr, cfg)
    assert meta["level_bump"] >= 1


def test_ego_shift_and_renoise():
    b, p, t = 1, 2, 3
    x0 = torch.randn(b, p, t, 4)
    prev = torch.tensor([[1.0, 2.0, 0.1]], dtype=torch.float32)
    cur = torch.tensor([[1.5, 2.0, 0.1]], dtype=torch.float32)
    shifted = ego_shift_trajectory(x0.reshape(b, p, -1), prev, cur)
    assert shifted.shape == (b, p, t * 4)
    x_ts = renoise_to_t(x0.reshape(b, p, -1), 0.3)
    assert x_ts.shape == (b, p, t * 4)


def test_build_warmstart_init():
    cache = WarmStartCache()
    b, p, flat = 1, 3, 4 * 5
    cache.x0_norm = torch.randn(b, p, flat)
    cache.anchor_xyh = torch.tensor([[0.0, 0.0, 0.0]])
    cur = torch.tensor([[0.5, 0.0, 0.0]])
    current = torch.zeros(b, p, 4)
    x_init = build_warmstart_init(cache, current, cur, t_start=0.2)
    assert x_init is not None
    assert x_init.shape == (b, p, flat)


if __name__ == "__main__":
    test_d_to_level()
    test_d_hat_monotonic_ops()
    test_safety_force_full()
    test_neighbor_bump()
    test_ego_shift_and_renoise()
    test_build_warmstart_init()
    print("gate_v2 tests passed")
