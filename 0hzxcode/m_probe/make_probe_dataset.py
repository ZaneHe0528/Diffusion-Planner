#!/usr/bin/env python3
"""Export frozen encoder features for the M-probe dataset.

This script only materializes features. The label d should come from the
full-denoise offline replay and must match the runtime fallback distance
definition. Provide it with --labels-csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_planner.model.diffusion_planner import Diffusion_Planner_Encoder
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.train_utils import openjson


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--data-list", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "model.pth")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "checkpoints" / "args.json")
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "0hzxcode" / "m_probe_output" / "probe_dataset.npz")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-ema", action="store_true")
    return parser.parse_args()


def load_config(path: Path, device: str) -> Namespace:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["device"] = device
    args = Namespace(**raw)
    args.normalization_file_path = str(REPO_ROOT / "normalization.json")
    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)
    return args


def load_encoder(config: Namespace, checkpoint: Path, use_ema: bool) -> Diffusion_Planner_Encoder:
    model = Diffusion_Planner_Encoder(config)
    state = torch.load(checkpoint, map_location=config.device)
    if use_ema and isinstance(state, dict) and "ema_state_dict" in state:
        state = state["ema_state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if not key.startswith("encoder."):
            continue
        cleaned[key[len("encoder.") :]] = value
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    model.to(config.device)
    return model


def load_labels(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    if not path.exists():
        raise SystemExit(
            f"labels CSV not found: {path}\n"
            "This file is required when --labels-csv is provided. Generate d labels first, "
            "or omit --labels-csv to export encoding-only data with d=NaN. "
            "Encoding-only data is useful for checking export, but cannot run the final probe."
        )
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not sample_id:
            raise ValueError("labels CSV must contain sample_id")
        if "d" not in row:
            raise ValueError("labels CSV must contain d")
        labels[sample_id] = row
        labels[Path(sample_id).name] = row
    return labels


def valid_token_mask(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    neighbor_valid = torch.sum(torch.ne(inputs["neighbor_agents_past"][..., :8], 0), dim=(-1, -2)) != 0
    static_valid = torch.sum(torch.ne(inputs["static_objects"][..., :10], 0), dim=-1) != 0
    lane_valid = torch.sum(torch.ne(inputs["lanes"][..., :, :8], 0), dim=(-1, -2)) != 0
    return torch.cat([neighbor_valid, static_valid, lane_valid], dim=1)


def masked_mean_encoding(encoding: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=encoding.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (encoding * weights).sum(dim=1) / denom


def ego_features_from_current_state(ego_current_state: torch.Tensor) -> torch.Tensor:
    # ego_current_state: x, y, cos, sin, vx, vy, ax, ay, steering, yaw_rate, ...
    vx = ego_current_state[:, 4]
    vy = ego_current_state[:, 5]
    ax = ego_current_state[:, 6]
    ay = ego_current_state[:, 7]
    steering = ego_current_state[:, 8]
    yaw_rate = ego_current_state[:, 9]
    speed = torch.hypot(vx, vy)
    accel = torch.hypot(ax, ay)
    return torch.stack([speed, accel, steering, yaw_rate, vx, vy, ax, ay], dim=1)


def collate_inputs(batch: tuple[torch.Tensor, ...], device: str) -> dict[str, torch.Tensor]:
    return {
        "ego_current_state": batch[0].to(device),
        "neighbor_agents_past": batch[2].to(device),
        "lanes": batch[4].to(device),
        "lanes_speed_limit": batch[5].to(device),
        "lanes_has_speed_limit": batch[6].to(device),
        "route_lanes": batch[7].to(device),
        "route_lanes_speed_limit": batch[8].to(device),
        "route_lanes_has_speed_limit": batch[9].to(device),
        "static_objects": batch[10].to(device),
    }


def scenario_from_sample_id(sample_id: str) -> str:
    path = Path(sample_id)
    if len(path.parts) >= 2:
        return path.parts[-2]
    return path.stem


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available; use --device cpu")

    config = load_config(args.config, args.device)
    labels = load_labels(args.labels_csv)
    sample_ids_all = openjson(str(args.data_list))
    if args.max_samples is not None:
        sample_ids_all = sample_ids_all[: args.max_samples]

    dataset = DiffusionPlannerData(
        str(args.data_dir),
        str(args.data_list),
        config.agent_num,
        config.predicted_neighbor_num,
        config.future_len,
    )
    if args.max_samples is not None:
        dataset.data_list = dataset.data_list[: args.max_samples]

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = load_encoder(config, args.checkpoint, use_ema=not args.no_ema)

    encodings: list[np.ndarray] = []
    ego_features: list[np.ndarray] = []
    sample_ids: list[str] = []
    scenario_ids: list[str] = []
    d_values: list[float] = []

    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = batch[0].shape[0]
            batch_sample_ids = sample_ids_all[offset : offset + batch_size]
            offset += batch_size

            inputs = collate_inputs(batch, args.device)
            raw_ego_current = inputs["ego_current_state"].detach().clone()
            token_mask = valid_token_mask(inputs)
            norm_inputs = config.observation_normalizer(inputs)
            encoder_outputs = model(norm_inputs)
            pooled = masked_mean_encoding(encoder_outputs["encoding"], token_mask)
            ego = ego_features_from_current_state(raw_ego_current)

            for i, sample_id in enumerate(batch_sample_ids):
                label = labels.get(sample_id) or labels.get(Path(sample_id).name)
                if labels and label is None:
                    continue
                sample_ids.append(sample_id)
                scenario_ids.append(label.get("scenario_id") if label and label.get("scenario_id") else scenario_from_sample_id(sample_id))
                d_values.append(float(label["d"]) if label else float("nan"))
                encodings.append(pooled[i].detach().cpu().numpy())
                ego_features.append(ego[i].detach().cpu().numpy())

    if not encodings:
        raise SystemExit("no rows exported; check --labels-csv sample_id matching")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        encoding=np.asarray(encodings, dtype=np.float32),
        d=np.asarray(d_values, dtype=np.float32),
        scenario_id=np.asarray(scenario_ids, dtype=str),
        sample_id=np.asarray(sample_ids, dtype=str),
        ego_features=np.asarray(ego_features, dtype=np.float32),
    )
    print(f"wrote: {args.output}")
    print(f"rows: {len(encodings)}")
    print(f"labels: {'yes' if labels else 'no'}")


if __name__ == "__main__":
    main()
