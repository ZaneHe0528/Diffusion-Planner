#!/usr/bin/env python3
"""不依赖 nuPlan 数据集的 planner+gate 集成冒烟：加载 checkpoint 跑单步 decoder。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
GATE_ROOT = REPO / "0hzxcode" / "gate_v2"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(GATE_ROOT))

from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from inference import GateWarmStartController


def main():
    args_file = REPO / "checkpoints" / "args.json"
    ckpt = REPO / "checkpoints" / "model.pth"
    gate_ckpt = REPO / "0hzxcode" / "gate_v2_output" / "runs" / "ego_nbr_perstep" / "best.pt"
    assert args_file.exists(), args_file
    assert ckpt.exists(), ckpt
    assert gate_ckpt.exists(), gate_ckpt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(str(args_file), guidance_fn=None)
    model = Diffusion_Planner(config).to(device)
    state = torch.load(ckpt, map_location=device)
    state = state.get("ema_state_dict", state)
    model.load_state_dict({k[len("module.") :]: v for k, v in state.items() if k.startswith("module.")})
    model.eval()

    b = 1
    p = 1 + config.predicted_neighbor_num
    t_flat = (1 + config.future_len) * 4
    inputs = {
        "ego_current_state": torch.zeros(b, 10, device=device),
        "neighbor_agents_past": torch.randn(b, config.agent_num, 21, 11, device=device) * 0.01,
        "static_objects": torch.zeros(b, config.static_objects_num, 10, device=device),
        "lanes": torch.zeros(b, config.lane_num, config.lane_len, 12, device=device),
        "route_lanes": torch.zeros(b, config.route_num, config.route_len, 12, device=device),
    }
    config.observation_normalizer(inputs)

    controller = GateWarmStartController(
        gate_ckpt,
        device=device,
        future_len=config.future_len,
        predicted_neighbor_num=config.predicted_neighbor_num,
    )
    controller.set_state_normalizer(config.state_normalizer)

    ego_hist = torch.zeros(21, 7)
    ops, meta = controller.predict_ops(inputs, ego_history=ego_hist.numpy())
    assert meta["enabled"], meta

    ego_current = inputs["ego_current_state"][:, None, :4]
    nbr_current = inputs["neighbor_agents_past"][:, : config.predicted_neighbor_num, -1, :4]
    current_states = torch.cat([ego_current, nbr_current], dim=1)
    anchor = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    tokens = [f"tok_{i}" for i in range(config.agent_num)]

    with torch.no_grad():
        _, out0 = model(inputs)
        assert "prediction" in out0
        controller.on_decode(out0["x0_norm"], anchor, neighbor_tokens=tokens)

        warm = controller.prepare_warmstart(
            ops,
            current_states,
            anchor,
            neighbor_tokens=tokens,
            neighbor_agents_past=inputs["neighbor_agents_past"],
            sde=model.sde,
        )
        assert warm is not None or ops.reuse_ratio <= 0
        if warm is not None:
            _, out1 = model(inputs, warmstart=warm)
            assert "warmstart_meta" in out1
            assert out1["warmstart_meta"].get("nfe") is not None
            print("warmstart nfe", out1["warmstart_meta"]["nfe"])

    print("planner+gate integration smoke passed")


if __name__ == "__main__":
    main()
