#!/usr/bin/env python3
"""Export (encoding, d) probe data from val14 nuPlan DB, not simulation logs.

This avoids SimulationLog unpickle crashes:
  free(): double free detected in tcache 2

Inputs:
  - d_labels.csv generated from adjacent_traj_l2_output
  - val14 / nuPlan DB path
  - map path
  - frozen Diffusion-Planner checkpoint

Output:
  - probe_dataset.npz with encoding, d, scenario_id, sample_id, ego_features
"""

from __future__ import annotations

import argparse
import csv
import faulthandler
import gc
import json
import os
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

# Native BLAS/OpenMP thread pools must be pinned BEFORE importing numpy/torch,
# otherwise OpenBLAS oversubscription corrupts the heap during map/encoder
# processing and aborts with "free(): double free detected in tcache 2"
# (or SIGILL inside a numpy ufunc under a debugger).
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

REPO_ROOT = Path(__file__).resolve().parents[2]
NUPLAN_DEVKIT_ROOT = REPO_ROOT / "nuplan-devkit"
for p in (REPO_ROOT, NUPLAN_DEVKIT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.utils.multithreading.worker_sequential import Sequential

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

faulthandler.enable(all_threads=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "model.pth")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "checkpoints" / "args.json")
    parser.add_argument("--data-path", type=Path, required=True, help="nuPlan DB dir, e.g. .../nuplan-v1.1/trainval")
    parser.add_argument("--map-path", type=Path, required=True, help="nuPlan maps dir")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--history-size", type=int, default=21)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Workers for scenario discovery when --scenario-worker is thread/process. <=1 uses sequential.",
    )
    parser.add_argument(
        "--scenario-worker",
        choices=("sequential", "thread", "process"),
        default="sequential",
        help="Worker used by NuPlanScenarioBuilder. Sequential is safest; process can segfault with torch/CUDA.",
    )
    parser.add_argument("--gc-every-scenarios", type=int, default=10)
    parser.add_argument(
        "--chunk-size-scenarios",
        type=int,
        default=5,
        help="Run DB export through short-lived child processes of this many scenarios. 0 disables supervision.",
    )
    parser.add_argument("--scenario-start-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--scenario-count", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--unsafe-cuda-db-export",
        action="store_true",
        help="Keep encoder inference on CUDA. This can segfault with nuPlan map/shapely native code.",
    )
    parser.add_argument("--no-ema", action="store_true")
    return parser.parse_args()


def load_labels(path: Path, max_rows: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"labels CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    required = {"sample_id", "d", "scenario_id", "log_name", "scenario_name", "new_iteration"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise SystemExit(f"labels CSV missing required columns: {sorted(missing)}")

    out = []
    for row in rows:
        row["d"] = float(row["d"])
        row["new_iteration"] = int(row["new_iteration"])
        out.append(row)
        if max_rows is not None and len(out) >= max_rows:
            break
    return out


def group_labels_by_scenario(
    labels: list[dict[str, Any]],
    max_scenarios: int | None,
    scenario_start_index: int = 0,
    scenario_count: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        by_scenario[row["scenario_id"]].append(row)

    scenario_ids = list(by_scenario.keys())
    if max_scenarios is not None:
        scenario_ids = scenario_ids[:max_scenarios]

    if scenario_start_index < 0:
        raise SystemExit("--scenario-start-index must be >= 0")
    if scenario_start_index or scenario_count is not None:
        end = len(scenario_ids) if scenario_count is None else scenario_start_index + scenario_count
        scenario_ids = scenario_ids[scenario_start_index:end]

    keep = set(scenario_ids)
    filtered = [row for row in labels if row["scenario_id"] in keep]
    filtered_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        filtered_by_scenario[row["scenario_id"]].append(row)

    return filtered, filtered_by_scenario


def chunk_output_path(output: Path, start: int, end: int) -> Path:
    chunk_dir = output.parent / f".{output.stem}_chunks"
    return chunk_dir / f"chunk_{start:04d}_{end:04d}.npz"


def append_optional_arg(cmd: list[str], flag: str, value: Any | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def child_command(args: argparse.Namespace, start: int, end: int, output: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--labels-csv",
        str(args.labels_csv),
        "--data-path",
        str(args.data_path),
        "--map-path",
        str(args.map_path),
        "--checkpoint",
        str(args.checkpoint),
        "--config",
        str(args.config),
        "--output",
        str(output),
        "--device",
        args.device,
        "--history-size",
        str(args.history_size),
        "--num-workers",
        str(args.num_workers),
        "--scenario-worker",
        args.scenario_worker,
        "--gc-every-scenarios",
        str(args.gc_every_scenarios),
        "--chunk-size-scenarios",
        "0",
        "--scenario-start-index",
        str(start),
        "--scenario-count",
        str(end - start),
        "--worker-child",
    ]
    append_optional_arg(cmd, "--max-rows", args.max_rows)
    append_optional_arg(cmd, "--max-scenarios", args.max_scenarios)
    if args.no_ema:
        cmd.append("--no-ema")
    if args.unsafe_cuda_db_export:
        cmd.append("--unsafe-cuda-db-export")
    return cmd


def merge_chunk_outputs(chunk_paths: list[Path], output: Path, summary: dict[str, Any]) -> None:
    arrays: dict[str, list[np.ndarray]] = {
        "encoding": [],
        "d": [],
        "scenario_id": [],
        "sample_id": [],
        "ego_features": [],
    }
    rows_exported = 0

    for chunk_path in chunk_paths:
        with np.load(chunk_path, allow_pickle=False) as data:
            chunk_rows = int(data["d"].shape[0])
            rows_exported += chunk_rows
            for key in arrays:
                arrays[key].append(np.asarray(data[key]))

    if rows_exported == 0:
        raise SystemExit("no rows exported from successful chunks")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        encoding=np.concatenate(arrays["encoding"], axis=0).astype(np.float32),
        d=np.concatenate(arrays["d"], axis=0).astype(np.float32),
        scenario_id=np.concatenate(arrays["scenario_id"], axis=0).astype(str),
        sample_id=np.concatenate(arrays["sample_id"], axis=0).astype(str),
        ego_features=np.concatenate(arrays["ego_features"], axis=0).astype(np.float32),
    )

    summary["rows_exported"] = rows_exported
    summary["chunks_merged"] = len(chunk_paths)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote: {output}", flush=True)
    print(f"wrote: {summary_path}", flush=True)
    print(f"rows_exported: {rows_exported}", flush=True)


def run_supervised_export(args: argparse.Namespace, labels: list[dict[str, Any]], by_scenario: dict[str, list[dict[str, Any]]]) -> None:
    total_scenarios = len(by_scenario)
    chunk_size = max(1, args.chunk_size_scenarios)
    chunk_dir = args.output.parent / f".{args.output.stem}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    successful_chunks: list[Path] = []
    failed_chunks: list[dict[str, Any]] = []

    def run_range(start: int, end: int) -> list[Path]:
        chunk_path = chunk_output_path(args.output, start, end)
        summary_path = chunk_path.with_suffix(".summary.json")
        if chunk_path.exists() and summary_path.exists():
            print(f"reuse chunk [{start}/{total_scenarios}:{end}/{total_scenarios}] {chunk_path}", flush=True)
            return [chunk_path]

        print(f"run chunk [{start}/{total_scenarios}:{end}/{total_scenarios}]", flush=True)
        proc = subprocess.run(child_command(args, start, end, chunk_path))
        if proc.returncode == 0:
            return [chunk_path]

        print(f"chunk failed rc={proc.returncode}: [{start}:{end}]", flush=True)
        if end - start <= 1:
            scenario_id = list(by_scenario.keys())[start] if start < total_scenarios else None
            failed_chunks.append({"start": start, "end": end, "scenario_id": scenario_id, "returncode": proc.returncode})
            return []

        mid = start + (end - start) // 2
        return run_range(start, mid) + run_range(mid, end)

    for start in range(0, total_scenarios, chunk_size):
        successful_chunks.extend(run_range(start, min(start + chunk_size, total_scenarios)))

    summary = {
        "labels_csv": str(args.labels_csv),
        "output": str(args.output),
        "data_path": str(args.data_path),
        "map_path": str(args.map_path),
        "device": args.device,
        "history_size": args.history_size,
        "scenario_worker": args.scenario_worker,
        "num_workers": args.num_workers,
        "unsafe_cuda_db_export": args.unsafe_cuda_db_export,
        "chunk_size_scenarios": args.chunk_size_scenarios,
        "input_label_rows": len(labels),
        "input_scenarios": total_scenarios,
        "failed_chunks": failed_chunks,
        "num_failed_chunks": len(failed_chunks),
    }
    merge_chunk_outputs(successful_chunks, args.output, summary)
    if failed_chunks:
        print(f"WARNING: skipped {len(failed_chunks)} crashing scenario chunk(s); see summary JSON", flush=True)


def make_filter(log_names: list[str], scenario_tokens: list[str]) -> ScenarioFilter:
    return ScenarioFilter(
        scenario_types=None,
        scenario_tokens=scenario_tokens,
        log_names=log_names,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=False,
        ego_start_speed_threshold=None,
        ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
    )


def build_scenarios(
    data_path: Path,
    map_path: Path,
    labels: list[dict[str, Any]],
    num_workers: int,
    scenario_worker: str,
) -> dict[str, Any]:
    log_names = sorted({r["log_name"] for r in labels})
    scenario_tokens = sorted({r["scenario_name"] for r in labels})

    builder = NuPlanScenarioBuilder(
        str(data_path),
        str(map_path),
        sensor_root=None,
        db_files=None,
        map_version="nuplan-maps-v1.0",
    )
    scenario_filter = make_filter(log_names, scenario_tokens)
    if scenario_worker == "sequential" or num_workers <= 1:
        worker = Sequential()
    else:
        worker = SingleMachineParallelExecutor(
            use_process_pool=scenario_worker == "process",
            max_workers=num_workers,
        )
    scenarios = builder.get_scenarios(scenario_filter, worker)

    result = {}
    for scenario in scenarios:
        key = f"{scenario.log_name}__{scenario.scenario_name}"
        result[key] = scenario
    return result


def build_history_buffer(scenario: Any, iteration: int, history_size: int) -> SimulationHistoryBuffer | None:
    if iteration < history_size - 1:
        return None

    ego_states = [
        scenario.get_ego_state_at_iteration(i)
        for i in range(iteration - history_size + 1, iteration + 1)
    ]
    observations = [
        scenario.get_tracked_objects_at_iteration(i)
        for i in range(iteration - history_size + 1, iteration + 1)
    ]

    return SimulationHistoryBuffer.initialize_from_list(
        buffer_size=history_size,
        ego_states=ego_states,
        observations=observations,
        sample_interval=scenario.database_interval,
    )


def export_one(
    row: dict[str, Any],
    scenario: Any,
    processor: DataProcessor,
    encoder: torch.nn.Module,
    config: Any,
    history_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    iteration = row["new_iteration"]
    history_buffer = build_history_buffer(scenario, iteration, history_size)
    if history_buffer is None:
        return None

    traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(iteration))
    inputs = processor.observation_adapter(
        history_buffer=history_buffer,
        traffic_light_data=traffic_light_data,
        map_api=scenario.map_api,
        route_roadblock_ids=scenario.get_route_roadblock_ids(),
        device=config.device,
    )

    raw_ego_current = inputs["ego_current_state"].detach().clone()
    token_mask = valid_token_mask(inputs)
    norm_inputs = config.observation_normalizer(inputs)

    with torch.no_grad():
        encoder_outputs = encoder(norm_inputs)
        pooled = masked_mean_encoding(encoder_outputs["encoding"], token_mask)
        ego = ego_features_from_current_state(raw_ego_current)

    return pooled[0].detach().cpu().numpy(), ego[0].detach().cpu().numpy()


def export_dataset(args: argparse.Namespace, labels: list[dict[str, Any]], by_scenario: dict[str, list[dict[str, Any]]]) -> None:
    print(f"labels: {len(labels)} rows, {len(by_scenario)} scenarios", flush=True)
    scenarios = build_scenarios(args.data_path, args.map_path, labels, args.num_workers, args.scenario_worker)
    loaded_scenario_count = len(scenarios)
    print(f"loaded scenarios from DB: {loaded_scenario_count}", flush=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False; use --device cpu")

    config = load_config(args.config, args.device)
    encoder = load_encoder(config, args.checkpoint, use_ema=not args.no_ema)
    processor = DataProcessor(config)

    encodings, ego_features, d_values, sample_ids, scenario_ids = [], [], [], [], []
    skipped_short_history = 0
    skipped_missing_scenario = 0
    failed_rows = []

    for idx, (scenario_id, rows) in enumerate(by_scenario.items(), start=1):
        scenario = scenarios.pop(scenario_id, None)
        if scenario is None:
            skipped_missing_scenario += len(rows)
            continue

        rows.sort(key=lambda r: r["new_iteration"])
        print(f"[{idx}/{len(by_scenario)}] {scenario_id} rows={len(rows)}", flush=True)

        for row in rows:
            try:
                result = export_one(row, scenario, processor, encoder, config, args.history_size)
            except Exception as exc:
                failed_rows.append({"sample_id": row["sample_id"], "reason": str(exc)})
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

        del scenario
        if args.gc_every_scenarios > 0 and idx % args.gc_every_scenarios == 0:
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

    if not encodings:
        raise SystemExit("no rows exported; check DB path, labels, and scenario matching")

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
        "data_path": str(args.data_path),
        "map_path": str(args.map_path),
        "device": args.device,
        "history_size": args.history_size,
        "scenario_worker": args.scenario_worker,
        "num_workers": args.num_workers,
        "unsafe_cuda_db_export": args.unsafe_cuda_db_export,
        "chunk_size_scenarios": args.chunk_size_scenarios,
        "input_label_rows": len(labels),
        "input_scenarios": len(by_scenario),
        "db_scenarios_loaded": loaded_scenario_count,
        "rows_exported": len(encodings),
        "skipped_short_history": skipped_short_history,
        "skipped_missing_scenario": skipped_missing_scenario,
        "num_failed_rows": len(failed_rows),
        "failed_rows": failed_rows[:100],
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote: {args.output}", flush=True)
    print(f"wrote: {summary_path}", flush=True)
    print(f"rows_exported: {len(encodings)}", flush=True)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not args.unsafe_cuda_db_export:
        print(
            "WARNING: --device cuda is unsafe for DB export in this environment; "
            "falling back to --device cpu. Use --unsafe-cuda-db-export to force CUDA.",
            file=sys.stderr,
            flush=True,
        )
        args.device = "cpu"

    labels = load_labels(args.labels_csv, args.max_rows)
    labels, by_scenario = group_labels_by_scenario(
        labels,
        args.max_scenarios,
        args.scenario_start_index,
        args.scenario_count,
    )

    if (
        not args.worker_child
        and args.chunk_size_scenarios > 0
        and len(by_scenario) > args.chunk_size_scenarios
    ):
        print(
            f"supervised export: {len(by_scenario)} scenarios, "
            f"chunk_size={args.chunk_size_scenarios}",
            flush=True,
        )
        run_supervised_export(args, labels, by_scenario)
        return

    export_dataset(args, labels, by_scenario)


if __name__ == "__main__":
    main()
