#!/usr/bin/env python3
"""gate_v2 单元测试：d->ops、warmstart、safety、token 匹配、首步注入。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))

from d_to_ops import d_hat_to_warmstart_ops, d_to_level
from gate_model import GateConfig, LiteGate
from inference import model_inputs_to_gate_batch
from safety import SafetyConfig, apply_safety, neighbor_interaction_score
from warmstart import (
    WarmStartCache,
    build_warmstart_init,
    constant_velocity_neighbor_x0,
    ego_shift_trajectory,
    renoise_to_t,
    update_cache,
)
from diffusion_planner.model.diffusion_utils.sampling import build_dpm_solver_bundle, dpm_sampler
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.utils.normalizer import StateNormalizer


EDGES = (0.275, 0.696, 1.385)


class _DummyModel(torch.nn.Module):
    model_type = "x_start"

    def forward(self, x, t, **kwargs):
        return x


def _state_normalizer(p: int = 3) -> StateNormalizer:
    mean = [[[10.0, 0.0, 0.0, 0.0]] for _ in range(p)]
    std = [[[20.0, 20.0, 1.0, 1.0]] for _ in range(p)]
    return StateNormalizer(mean, std)


def test_d_to_level():
    assert d_to_level(0.1, EDGES) == 0
    assert d_to_level(0.5, EDGES) == 1
    assert d_to_level(1.0, EDGES) == 2
    assert d_to_level(2.0, EDGES) == 3


def test_d_hat_monotonic_ops():
    ops_low = d_hat_to_warmstart_ops(0.05, level_edges_m=EDGES, use_continuous_ts=True)
    ops_high = d_hat_to_warmstart_ops(1.5, level_edges_m=EDGES, use_continuous_ts=False)
    assert ops_low.t_start < ops_high.t_start
    assert ops_low.steps <= ops_high.steps


def test_safety_force_full():
    cfg = SafetyConfig(hard_threshold_m=1.0, score_threshold_m=0.8, level_edges_m=(0.275, 0.696, 1.385))
    ops, meta = apply_safety(0.9, None, cfg)
    assert meta["score_alarm"] is True
    assert meta["forced_full"] is False
    assert ops.t_start < 1.0

    ops, meta = apply_safety(1.1, None, cfg)
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


def test_planner_batched_gate_input_shape():
    batch = model_inputs_to_gate_batch(
        {"neighbor_agents_past": torch.zeros(1, 32, 21, 11)},
        ego_history=np.zeros((21, 7), dtype=np.float32),
    )
    assert batch["neighbor_agents_past"].shape == (1, 32, 21, 11)
    model = LiteGate(GateConfig(enabled_groups=["ego_history", "neighbor_agents"]))
    out = model(batch)
    assert out["d_hat"].shape == (1,)


def test_ego_shift_and_renoise():
    b, p, t = 1, 2, 3
    x0 = torch.randn(b, p, t, 4)
    prev = torch.tensor([[1.0, 2.0, 0.1]], dtype=torch.float32)
    cur = torch.tensor([[1.5, 2.0, 0.1]], dtype=torch.float32)
    shifted = ego_shift_trajectory(x0.reshape(b, p, -1), prev, cur)
    assert shifted.shape == (b, p, t * 4)
    x_ts = renoise_to_t(x0.reshape(b, p, -1), 0.3)
    assert x_ts.shape == (b, p, t * 4)
    sde = VPSDE_linear()
    x_known = torch.full((1, 1, 1), 2.0)
    t_known = torch.tensor([0.3])
    mean, _ = sde.marginal_prob(x_known, t_known)
    out = renoise_to_t(x_known, 0.3, sde=sde, noise=torch.zeros_like(x_known))
    assert torch.allclose(out, mean)


def test_normalized_shift_uses_physical_units():
    cache = WarmStartCache()
    normalizer = _state_normalizer(p=1)
    # Physical x=5m is normalized as (5 - 10) / 20 = -0.25.
    cache.x0_norm = torch.tensor([[[-0.25, 0.0, 1.0, 0.0, -0.25, 0.0, 1.0, 0.0]]])
    cache.anchor_xyh = torch.tensor([[2.0, 0.0, 0.0]])
    current = torch.zeros(1, 1, 4)
    _, ref = build_warmstart_init(
        cache,
        current,
        torch.tensor([[0.0, 0.0, 0.0]]),
        t_start=0.2,
        future_len=1,
        state_normalizer=normalizer,
    )
    # Previous anchor is 2m ahead of current, so physical x=5m becomes 7m:
    # normalized x=(7 - 10) / 20 = -0.15. The old normalized-space shift gave 1.75.
    assert torch.allclose(ref[0, 0, 4], torch.tensor(-0.15), atol=1e-6)


def test_constant_velocity_uses_raw_units_then_normalizes():
    normalizer = _state_normalizer(p=2)
    past = torch.zeros(21, 11)
    past[-1, 0] = 6.0
    past[-1, 2] = 1.0
    past[-1, 4] = 2.0
    row = constant_velocity_neighbor_x0(1, past, future_len=1, state_normalizer=normalizer, device=torch.device("cpu"))
    assert torch.allclose(row[0], torch.tensor(-0.2), atol=1e-6)
    assert torch.allclose(row[4], torch.tensor(-0.19), atol=1e-6)


def test_build_warmstart_init():
    cache = WarmStartCache()
    b, p, t_pts = 1, 3, 5
    flat = t_pts * 4
    cache.x0_norm = torch.randn(b, p, flat)
    cache.anchor_xyh = torch.tensor([[0.0, 0.0, 0.0]])
    cache.neighbor_tokens = ["tok_a", "tok_b"]
    cur = torch.tensor([[0.5, 0.0, 0.0]])
    current = torch.zeros(b, p, 4)
    x_init, ref = build_warmstart_init(
        cache,
        current,
        cur,
        t_start=0.2,
        current_neighbor_tokens=["tok_a", "tok_b"],
        future_len=t_pts - 1,
        state_normalizer=_state_normalizer(p=p),
    )
    assert x_init is not None
    assert ref is not None
    assert x_init.shape == (b, p, flat)


def test_token_match_vs_cv():
    cache = WarmStartCache()
    b, p, t_pts = 1, 3, 5
    flat = t_pts * 4
    cache.x0_norm = torch.arange(b * p * flat, dtype=torch.float32).reshape(b, p, flat)
    cache.anchor_xyh = torch.zeros(1, 3)
    cache.neighbor_tokens = ["tok_a", "tok_b"]
    cur = torch.zeros(1, 3)
    current = torch.zeros(b, p, 4)

    _, ref_match = build_warmstart_init(
        cache,
        current,
        cur,
        0.2,
        current_neighbor_tokens=["tok_a", "unknown"],
        future_len=t_pts - 1,
        state_normalizer=_state_normalizer(p=p),
    )
    assert ref_match[0, 1].abs().sum() > 0
    assert not torch.allclose(ref_match[0, 1], ref_match[0, 2])


def test_first_model_output_injection_nfe():
    model = _DummyModel()
    x = torch.randn(1, 4)
    nfe_plain = {}
    plain = dpm_sampler(model, x, diffusion_steps=4, sample_params={"nfe_holder": nfe_plain})
    _, _, solver = build_dpm_solver_bundle(model)
    first = solver.model_fn(x, torch.ones(1))
    nfe_inj = {}
    inj = dpm_sampler(
        model,
        x,
        diffusion_steps=4,
        sample_params={"first_model_output": first, "nfe_holder": nfe_inj},
    )
    assert nfe_plain["nfe"] == 5
    assert nfe_inj["nfe"] == nfe_plain["nfe"] - 1
    assert torch.allclose(inj, plain, atol=1e-5)


if __name__ == "__main__":
    test_d_to_level()
    test_d_hat_monotonic_ops()
    test_safety_force_full()
    test_neighbor_bump()
    test_planner_batched_gate_input_shape()
    test_ego_shift_and_renoise()
    test_normalized_shift_uses_physical_units()
    test_constant_velocity_uses_raw_units_then_normalizes()
    test_build_warmstart_init()
    test_token_match_vs_cv()
    test_first_model_output_injection_nfe()
    print("gate_v2 tests passed")
