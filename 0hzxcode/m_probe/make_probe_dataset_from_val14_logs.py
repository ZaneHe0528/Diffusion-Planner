#!/usr/bin/env python3
"""Export (encoding, d) probe data from val14 simulation logs.

This is Step 2 for the val14 M-probe flow:

1. Read d_labels.csv generated from analyze_adjacent_traj_l2.py output.
2. For the same closed-loop log frame sample_id, rebuild planner inputs from
   SimulationHistorySample history.
3. Run the frozen Diffusion-Planner encoder.
4. Save probe_dataset.npz with encoding, d, scenario_id, ego_features, sample_id.

Rows with fewer than 21 history frames are skipped because the encoder was
trained with time_len=21.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

REPO_ROOT = Path(__file__).resolve().parents[2]
NUPLAN_DEVKIT_ROOT = REPO_ROOT / "nuplan-devkit"
for _path in (REPO_ROOT, NUPLAN_DEVKIT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.simulation_log import SimulationLog

from diffusion_planner.data_process.data_processor import DataProcessor
from make_probe_dataset import (
    ego_features_from_current_state,
    load_config,
    load_encoder,
    masked_mean_encoding,
    valid_token_mask,
)


DEFAULT_LABELS = REPO_ROOT / "0hzxcode" / "m_probe_output" / "d_labels.csv"
DEFAULT_OUTPUT = REPO_ROOT / "0hzxcode" / "m_probe_output" / "probe_dataset.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "model.pth")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "checkpoints" / "args.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--history-size", type=int, default=21)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--max-logs", type=int, default=None, help="Optional cap on distinct simulation logs.")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--cpu-log-load", action="store_true", help="Force CUDA tensors stored in logs to CPU while loading.")
    parser.add_argument(
        "--unsafe-cuda-log-load",
        action="store_true",
        help="Allow loading simulation logs in the same process after CUDA is initialized. This can abort with double-free.",
    )
    return parser.parse_args()


def patch_torch_load_from_bytes_to_cpu() -> None:
    """Simulation logs may contain CUDA tensors; load them on CPU in CPU-only processes."""
    import torch.storage

    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")


def load_labels(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"labels CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"sample_id", "d", "scenario_id", "log_file", "new_iteration"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise SystemExit(f"labels CSV missing required columns: {sorted(missing)}")
    for row in rows:
        row["d"] = float(row["d"])
        row["new_iteration"] = int(row["new_iteration"])
    return rows


def rows_by_log(rows: list[dict[str, Any]], max_rows: int | None, max_logs: int | None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if max_rows is not None and sum(len(v) for v in grouped.values()) >= max_rows:
            break
        if row["log_file"] not in grouped and max_logs is not None and len(grouped) >= max_logs:
            continue
        grouped[row["log_file"]].append(row)
    for log_rows in grouped.values():
        log_rows.sort(key=lambda r: r["new_iteration"])
    return dict(grouped)


def build_history_buffer(samples: list[Any], iteration: int, history_size: int) -> SimulationHistoryBuffer | None:
    start = iteration - history_size + 1
    if start < 0:
        return None
    window = samples[start : iteration + 1]
    if len(window) != history_size:
        return None
    return SimulationHistoryBuffer.initialize_from_list(
        buffer_size=history_size,
        ego_states=[sample.ego_state for sample in window],
        observations=[sample.observation for sample in window],
        sample_interval=0.1,
    )


def export_row(
    row: dict[str, Any],
    simulation_log: Any,
    processor: DataProcessor,
    encoder: torch.nn.Module,
    config: Any,
    history_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    samples = simulation_log.simulation_history.data
    iteration = row["new_iteration"]
    if iteration >= len(samples):
        return None

    history_buffer = build_history_buffer(samples, iteration, history_size)
    if history_buffer is None:
        return None

    sample = samples[iteration]
    inputs = processor.observation_adapter(
        history_buffer=history_buffer,
        traffic_light_data=list(sample.traffic_light_status),
        map_api=simulation_log.scenario.map_api,
        route_roadblock_ids=simulation_log.scenario.get_route_roadblock_ids(),
        device=config.device,
    )
    raw_ego_current = inputs["ego_current_state"].detach().clone()
    token_mask = valid_token_mask(inputs)
    norm_inputs = config.observation_normalizer(inputs)

    with torch.no_grad():
        encoder_outputs = encoder(norm_inputs)
        pooled = masked_mean_encoding(encoder_outputs["encoding"], token_mask)
        ego_features = ego_features_from_current_state(raw_ego_current)

    return pooled[0].detach().cpu().numpy(), ego_features[0].detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and args.cpu_log_load and not args.unsafe_cuda_log_load:
        raise SystemExit(
            "Refusing unsafe mode: --device cuda together with --cpu-log-load can crash while unpickling "
            "nuPlan SimulationLog objects that contain torch/CUDA state (observed: free(): double free).\n"
            "Use the stable command with --device cpu for this log-based export. The exported probe_dataset.npz "
            "is identical in format and can still be used by run_encoding_d_probe.py.\n"
            "If you intentionally want to experiment with the unsafe single-process CUDA path, add "
            "--unsafe-cuda-log-load."
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False; use --device cpu")
    if args.cpu_log_load or args.device == "cpu":
        patch_torch_load_from_bytes_to_cpu()

    labels = load_labels(args.labels_csv)
    grouped = rows_by_log(labels, args.max_rows, args.max_logs)
    config = load_config(args.config, args.device)
    encoder = load_encoder(config, args.checkpoint, use_ema=not args.no_ema)
    processor = DataProcessor(config)

    encodings: list[np.ndarray] = []
    ego_features: list[np.ndarray] = []
    d_values: list[float] = []
    sample_ids: list[str] = []
    scenario_ids: list[str] = []

    skipped_short_history = 0
    skipped_missing_iteration = 0
    failed_rows: list[dict[str, str]] = []
    processed_logs = 0

    for log_file, log_rows in grouped.items():
        processed_logs += 1
        print(f"[{processed_logs}/{len(grouped)}] load log: {log_file} rows={len(log_rows)}", flush=True)
        try:
            simulation_log = SimulationLog.load_data(Path(log_file))
        except Exception as exc:  # noqa: BLE001
            failed_rows.append({"log_file": log_file, "sample_id": "", "reason": f"log_load_failed: {exc}"})
            continue

        for row in log_rows:
            iteration = row["new_iteration"]
            if iteration < args.history_size - 1:
                skipped_short_history += 1
                continue
            if iteration >= len(simulation_log.simulation_history.data):
                skipped_missing_iteration += 1
                continue
            try:
                result = export_row(row, simulation_log, processor, encoder, config, args.history_size)
            except Exception as exc:  # noqa: BLE001
                failed_rows.append({"log_file": log_file, "sample_id": row["sample_id"], "reason": str(exc)})
                continue
            if result is None:
                skipped_short_history += 1
                continue
            encoding, ego = result
            encodings.append(encoding.astype(np.float32))
            ego_features.append(ego.astype(np.float32))
            d_values.append(float(row["d"]))
            sample_ids.append(row["sample_id"])
            scenario_ids.append(row["scenario_id"])

    if not encodings:
        raise SystemExit("no rows exported; check labels, logs, and history-size")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        encoding=np.asarray(encodings, dtype=np.float32),
        d=np.asarray(d_values, dtype=np.float32),
        scenario_id=np.asarray(scenario_ids, dtype=str),
        sample_id=np.asarray(sample_ids, dtype=str),
        ego_features=np.asarray(ego_features, dtype=np.float32),
    )

    summary = {
        "labels_csv": str(args.labels_csv),
        "output": str(args.output),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "device": args.device,
        "history_size": args.history_size,
        "input_label_rows_considered": sum(len(v) for v in grouped.values()),
        "logs_considered": len(grouped),
        "rows_exported": len(encodings),
        "skipped_short_history": skipped_short_history,
        "skipped_missing_iteration": skipped_missing_iteration,
        "failed_rows": failed_rows[:100],
        "num_failed_rows": len(failed_rows),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote: {args.output}", flush=True)
    print(f"wrote: {summary_path}", flush=True)
    print(f"rows_exported: {len(encodings)}", flush=True)
    print(f"skipped_short_history: {skipped_short_history}", flush=True)
    print(f"num_failed_rows: {len(failed_rows)}", flush=True)


if __name__ == "__main__":
    main()
