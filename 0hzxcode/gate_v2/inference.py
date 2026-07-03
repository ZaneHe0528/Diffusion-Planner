#!/usr/bin/env python3
"""闭环推理：从 planner 输入构建 gate batch 并预测 d_hat。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from d_to_ops import WarmStartOps
from gate_model import LiteGate, load_gate
from safety import SafetyConfig, apply_safety, safety_config_from_checkpoint
from warmstart import WarmStartCache, build_warmstart_init, update_cache


def build_ego_history_from_ego_states(ego_states: list[Any]) -> np.ndarray:
    """从 nuPlan ego state 列表构建 [21,7] 当前帧系 ego_history。"""
    from diffusion_planner.data_process.ego_process import sampled_past_ego_states_to_array
    from diffusion_planner.data_process.utils import convert_absolute_quantities_to_relative

    ego_state = ego_states[-1]
    anchor = np.array(
        [ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading],
        dtype=np.float64,
    )
    ego_abs = sampled_past_ego_states_to_array(ego_states[-21:])
    if ego_abs.shape[0] < 21:
        pad = np.zeros((21 - ego_abs.shape[0], 7), dtype=np.float64)
        ego_abs = np.concatenate([pad, ego_abs], axis=0)
    return convert_absolute_quantities_to_relative(ego_abs, anchor.copy(), "ego").astype(np.float32)


def anchor_from_ego_state(ego_state: Any) -> np.ndarray:
    return np.array(
        [ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading],
        dtype=np.float32,
    )


def model_inputs_to_gate_batch(
    model_inputs: dict[str, torch.Tensor],
    ego_history: np.ndarray | None = None,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    batch = {
        "neighbor_agents_past": model_inputs["neighbor_agents_past"].float().unsqueeze(0).to(device),
    }
    if ego_history is not None:
        batch["ego_history"] = torch.from_numpy(ego_history).float().unsqueeze(0).to(device)
    elif "ego_history" in model_inputs:
        batch["ego_history"] = model_inputs["ego_history"].float().unsqueeze(0).to(device)
    else:
        raise KeyError("ego_history required: pass from build_ego_history_from_ego_states")
    return batch


class GateWarmStartController:
    """gate 预测 + 安全兜底 + warm-start 缓存。"""

    def __init__(
        self,
        gate_ckpt: str | Path,
        device: str = "cpu",
        base_steps: int = 10,
        enabled: bool = True,
    ):
        self.device = device
        self.base_steps = base_steps
        self.enabled = enabled
        self.model: LiteGate | None = None
        self.safety: SafetyConfig | None = None
        self.cache = WarmStartCache()
        if enabled and gate_ckpt is not None:
            self.model, state = load_gate(str(gate_ckpt), device=device)
            self.safety = safety_config_from_checkpoint(state)
            self.safety = SafetyConfig(
                hard_threshold_m=self.safety.hard_threshold_m,
                score_threshold_m=float(state.get("score_threshold_m", self.safety.score_threshold_m)),
                level_edges_m=self.safety.level_edges_m,
                base_steps=base_steps,
            )

    @torch.no_grad()
    def predict_ops(
        self,
        model_inputs: dict[str, torch.Tensor],
        ego_history: np.ndarray | None = None,
    ) -> tuple[WarmStartOps | None, dict]:
        if not self.enabled or self.model is None or self.safety is None:
            return None, {"enabled": False}

        batch = model_inputs_to_gate_batch(model_inputs, ego_history=ego_history, device=self.device)
        out = self.model(batch)
        d_hat = float(out["d_hat"].item())
        ops, meta = apply_safety(
            d_hat,
            batch.get("neighbor_agents_past"),
            self.safety,
            device=self.device,
        )
        meta["enabled"] = True
        meta["level_from_gate"] = int(out["level"].item())
        return ops, meta

    def prepare_warmstart(
        self,
        ops: WarmStartOps | None,
        current_states: torch.Tensor,
        anchor_xyh: np.ndarray | torch.Tensor,
        sde=None,
    ) -> dict | None:
        if not self.enabled or ops is None or ops.reuse_ratio <= 0:
            return None
        if isinstance(anchor_xyh, np.ndarray):
            anchor = torch.from_numpy(anchor_xyh).float().to(current_states.device).unsqueeze(0)
        else:
            anchor = anchor_xyh.float()
            if anchor.dim() == 1:
                anchor = anchor.unsqueeze(0)

        x_init = build_warmstart_init(self.cache, current_states, anchor, ops.t_start, sde=sde)
        if x_init is None:
            return None
        return {
            "x_init": x_init,
            "t_start": ops.t_start,
            "steps": ops.steps,
            "return_x0_norm": True,
            "meta": {"level": ops.level, "d_hat_m": ops.d_hat_m, "reuse_ratio": ops.reuse_ratio},
        }

    def on_decode(
        self,
        x0_norm: torch.Tensor,
        anchor_xyh: np.ndarray | torch.Tensor,
    ) -> None:
        if isinstance(anchor_xyh, np.ndarray):
            anchor = torch.from_numpy(anchor_xyh).float().to(x0_norm.device).unsqueeze(0)
        else:
            anchor = anchor_xyh.float()
            if anchor.dim() == 1:
                anchor = anchor.unsqueeze(0)
        update_cache(self.cache, x0_norm, anchor)

    def reset(self) -> None:
        self.cache = WarmStartCache()
