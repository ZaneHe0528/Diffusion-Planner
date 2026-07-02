#!/usr/bin/env python3
"""从 val14 闭环仿真日志导出 gate 训练数据集：每帧**原始观测** + 帧间变化距离 d。

为什么读仿真日志而不是 nuPlan DB：
  - d 标签（d_labels.csv）本来就由 analyze_adjacent_traj_l2.py 从这些日志计算，
    sample_id 的 new_iteration 就是日志里 SimulationHistorySample 的下标，
    帧号/时间戳天然对齐；
  - 旧 make_probe_dataset_from_val14_db.py 重建场景时没有传 ScenarioMapping，
    与闭环仿真（15s、-3s 偏移、10Hz）相比帧错位约 3s 且采样率不同（20Hz），
    旧 probe 数据集的 encoding 与 d 并不对应同一帧；
  - 日志里的 ego_state / observation 是闭环下 planner 真实看到的输入
    （ego 走的是 planner 轨迹），与 gate 部署时的分布一致。

崩溃防护（旧日志导出方案的教训）：
  - 不加载 encoder、不初始化 CUDA；BLAS/OMP 线程在 import numpy 前钉死；
  - torch.storage 反序列化强制 map 到 CPU（日志里可能 pickle 了 CUDA 张量）；
  - chunk 子进程隔离 + 失败二分，个别坏场景自动跳过并记录。

输出：--output-dir 下若干 chunk_XXXX_YYYY.npz，键 = gate_features.FEATURE_GROUPS
的数组 + d / sample_id / scenario_id / scenario_type，另有 export_summary.json。
"""

from __future__ import annotations

import argparse
import csv
import faulthandler
import gc
import io
import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

# BLAS/OpenMP 线程必须在 import numpy/torch 前钉死，否则并行时堆损坏（沿用旧脚本教训）。
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
NUPLAN_DEVKIT_ROOT = REPO_ROOT / "nuplan-devkit"
for _p in (str(REPO_ROOT), str(NUPLAN_DEVKIT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_LABELS = REPO_ROOT / "0hzxcode" / "m_probe_output" / "d_labels.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "0hzxcode" / "gate_output" / "gate_dataset_chunks"

faulthandler.enable(all_threads=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "checkpoints" / "args.json")
    parser.add_argument("--history-size", type=int, default=21)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--chunk-size-scenarios", type=int, default=5)
    parser.add_argument("--max-parallel", type=int, default=6, help="并行 chunk 子进程数")
    # 子进程内部参数
    parser.add_argument("--scenario-start-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--scenario-count", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--chunk-output", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 标签
# ---------------------------------------------------------------------------

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


def group_labels_by_scenario(
    labels: list[dict[str, Any]],
    max_scenarios: int | None,
    start_index: int = 0,
    count: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        by_scenario[row["scenario_id"]].append(row)

    scenario_ids = list(by_scenario.keys())
    if max_scenarios is not None:
        scenario_ids = scenario_ids[:max_scenarios]
    end = len(scenario_ids) if count is None else start_index + count
    scenario_ids = scenario_ids[start_index:end]
    return {sid: by_scenario[sid] for sid in scenario_ids}


# ---------------------------------------------------------------------------
# 子进程：真正的导出逻辑
# ---------------------------------------------------------------------------

def patch_torch_load_to_cpu() -> None:
    """日志里可能 pickle 了 CUDA 张量；强制在 CPU 反序列化。"""
    import torch
    import torch.storage

    torch.set_num_threads(1)
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")


def observation_arrays(
    ego_states: list[Any],
    observations: list[Any],
    traffic_light_data: list[Any],
    map_api: Any,
    route_roadblock_ids: list[str],
    config: Any,
) -> dict[str, np.ndarray]:
    """复刻 DataProcessor.observation_adapter 的处理，但输出 numpy（不建 torch 张量）。

    额外输出 ego_history（闭环 ego 过去 21 帧，当前帧系），修复旧脚本从推理态
    ego_current_state 里读到全零运动学的问题。
    """
    from nuplan.common.actor_state.state_representation import Point2D

    from diffusion_planner.data_process.agent_process import (
        agent_past_process,
        sampled_static_objects_to_array_list,
        sampled_tracked_objects_to_array_list,
    )
    from diffusion_planner.data_process.ego_process import sampled_past_ego_states_to_array
    from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
    from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
    from diffusion_planner.data_process.utils import convert_absolute_quantities_to_relative

    ego_state = ego_states[-1]
    ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
    anchor_ego_state = np.array(
        [ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64
    )

    # ego 过去 21 帧 -> 当前帧系 [21,7]（x,y,heading,vx,vy,ax,ay）
    ego_abs = sampled_past_ego_states_to_array(ego_states)
    ego_history = convert_absolute_quantities_to_relative(ego_abs, anchor_ego_state.copy(), "ego").astype(np.float32)

    # 邻车 / 静态障碍（与 observation_adapter 相同）
    neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(observations)
    static_objects, static_objects_types = sampled_static_objects_to_array_list(observations[-1])
    _, neighbor_agents_past, _, static_objects = agent_past_process(
        None,
        neighbor_agents_past,
        neighbor_agents_types,
        config.agent_num,
        static_objects,
        static_objects_types,
        config.static_objects_num,
        10,  # max_ped_bike，与 DataProcessor 一致
        anchor_ego_state,
    )

    # 地图（与 observation_adapter 相同）
    route_roadblock_ids = route_roadblock_correction(ego_state, map_api, route_roadblock_ids)
    map_features = ["LANE", "LEFT_BOUNDARY", "RIGHT_BOUNDARY", "ROUTE_LANES"]
    max_elements = {
        "LANE": config.lane_num,
        "LEFT_BOUNDARY": config.lane_num,
        "RIGHT_BOUNDARY": config.lane_num,
        "ROUTE_LANES": config.route_num,
    }
    max_points = {
        "LANE": config.lane_len,
        "LEFT_BOUNDARY": config.lane_len,
        "RIGHT_BOUNDARY": config.lane_len,
        "ROUTE_LANES": config.route_len,
    }
    coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
        map_api, map_features, ego_coords, 100, traffic_light_data
    )
    vector_map = map_process(
        route_roadblock_ids,
        anchor_ego_state,
        coords,
        traffic_light_data,
        speed_limit,
        lane_route,
        map_features,
        max_elements,
        max_points,
    )

    return {
        "ego_history": ego_history,                                             # f32 [21,7]
        "neighbor_agents_past": neighbor_agents_past[:, -21:].astype(np.float16),
        "static_objects": static_objects.astype(np.float16),
        "lanes": vector_map["lanes"].astype(np.float16),
        "lanes_speed_limit": vector_map["lanes_speed_limit"].astype(np.float16),
        "lanes_has_speed_limit": vector_map["lanes_has_speed_limit"].astype(np.uint8),
        "route_lanes": vector_map["route_lanes"].astype(np.float16),
        "route_lanes_speed_limit": vector_map["route_lanes_speed_limit"].astype(np.float16),
        "route_lanes_has_speed_limit": vector_map["route_lanes_has_speed_limit"].astype(np.uint8),
    }


def run_child(args: argparse.Namespace) -> None:
    patch_torch_load_to_cpu()

    from argparse import Namespace

    from nuplan.planning.simulation.simulation_log import SimulationLog

    config = Namespace(**json.loads(args.config.read_text(encoding="utf-8")))

    labels = load_labels(args.labels_csv)
    by_scenario = group_labels_by_scenario(labels, args.max_scenarios, args.scenario_start_index, args.scenario_count)

    columns: dict[str, list[np.ndarray]] = defaultdict(list)
    d_values: list[float] = []
    sample_ids: list[str] = []
    scenario_ids: list[str] = []
    scenario_types: list[str] = []
    skipped_short_history = 0
    skipped_missing_iteration = 0
    failed_rows: list[dict[str, str]] = []

    for idx, (scenario_id, rows) in enumerate(by_scenario.items(), start=1):
        log_file = rows[0]["log_file"]
        print(f"[{idx}/{len(by_scenario)}] {scenario_id} rows={len(rows)}", flush=True)
        try:
            simulation_log = SimulationLog.load_data(Path(log_file))
        except Exception as exc:
            failed_rows.append({"sample_id": f"{scenario_id}/*", "reason": f"log_load_failed: {exc}"})
            continue

        samples = simulation_log.simulation_history.data
        scenario = simulation_log.scenario
        map_api = scenario.map_api
        route_roadblock_ids = scenario.get_route_roadblock_ids()

        rows.sort(key=lambda r: r["new_iteration"])
        for row in rows:
            iteration = row["new_iteration"]
            if iteration < args.history_size - 1:
                skipped_short_history += 1
                continue
            if iteration >= len(samples):
                skipped_missing_iteration += 1
                continue
            window = samples[iteration - args.history_size + 1 : iteration + 1]
            try:
                frame = observation_arrays(
                    ego_states=[s.ego_state for s in window],
                    observations=[s.observation for s in window],
                    traffic_light_data=list(samples[iteration].traffic_light_status),
                    map_api=map_api,
                    route_roadblock_ids=route_roadblock_ids,
                    config=config,
                )
            except Exception as exc:  # 单帧失败不拖垮整个 chunk
                failed_rows.append({"sample_id": row["sample_id"], "reason": str(exc)})
                continue
            for key, value in frame.items():
                columns[key].append(value)
            d_values.append(float(row["d"]))
            sample_ids.append(row["sample_id"])
            scenario_ids.append(row["scenario_id"])
            scenario_types.append(row.get("scenario_type", ""))

        del simulation_log, samples, scenario, map_api
        gc.collect()

    if not d_values:
        raise SystemExit("no rows exported in this chunk")

    output: Path = args.chunk_output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        d=np.asarray(d_values, dtype=np.float32),
        sample_id=np.asarray(sample_ids, dtype=str),
        scenario_id=np.asarray(scenario_ids, dtype=str),
        scenario_type=np.asarray(scenario_types, dtype=str),
        **{key: np.stack(vals, axis=0) for key, vals in columns.items()},
    )
    summary = {
        "rows_exported": len(d_values),
        "skipped_short_history": skipped_short_history,
        "skipped_missing_iteration": skipped_missing_iteration,
        "num_failed_rows": len(failed_rows),
        "failed_rows": failed_rows[:50],
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote: {output} rows={len(d_values)}", flush=True)


# ---------------------------------------------------------------------------
# 父进程：并行调度 chunk 子进程（失败自动二分，坏场景跳过）
# ---------------------------------------------------------------------------

def chunk_output_path(output_dir: Path, start: int, end: int) -> Path:
    return output_dir / f"chunk_{start:04d}_{end:04d}.npz"


def child_command(args: argparse.Namespace, start: int, end: int, output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--labels-csv", str(args.labels_csv),
        "--config", str(args.config),
        "--history-size", str(args.history_size),
        "--scenario-start-index", str(start),
        "--scenario-count", str(end - start),
        "--chunk-output", str(output),
        "--worker-child",
    ] + (["--max-scenarios", str(args.max_scenarios)] if args.max_scenarios is not None else [])


def run_supervisor(args: argparse.Namespace, total_scenarios: int) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = max(1, args.chunk_size_scenarios)
    lock = threading.Lock()
    done_chunks: list[Path] = []
    failed_units: list[dict[str, Any]] = []
    t0 = time.time()

    def run_range(start: int, end: int) -> None:
        chunk_path = chunk_output_path(args.output_dir, start, end)
        if chunk_path.exists() and chunk_path.with_suffix(".summary.json").exists():
            with lock:
                done_chunks.append(chunk_path)
            print(f"reuse chunk [{start}:{end}]", flush=True)
            return
        proc = subprocess.run(child_command(args, start, end, chunk_path), stdout=subprocess.DEVNULL)
        if proc.returncode == 0:
            with lock:
                done_chunks.append(chunk_path)
                n_done = len(done_chunks)
            print(f"chunk [{start}:{end}] done ({n_done} chunks, {(time.time()-t0)/60:.1f} min elapsed)", flush=True)
            return
        print(f"chunk failed rc={proc.returncode}: [{start}:{end}]", flush=True)
        if end - start <= 1:
            with lock:
                failed_units.append({"start": start, "end": end, "returncode": proc.returncode})
            return
        mid = start + (end - start) // 2
        run_range(start, mid)
        run_range(mid, end)

    ranges = [(s, min(s + chunk_size, total_scenarios)) for s in range(0, total_scenarios, chunk_size)]
    print(f"exporting {total_scenarios} scenarios in {len(ranges)} chunks, {args.max_parallel} parallel workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        list(pool.map(lambda r: run_range(*r), ranges))

    rows_total = 0
    for p in sorted(set(done_chunks)):
        try:
            rows_total += json.loads(p.with_suffix(".summary.json").read_text())["rows_exported"]
        except Exception:
            pass
    summary = {
        "labels_csv": str(args.labels_csv),
        "output_dir": str(args.output_dir),
        "source": "closed-loop simulation logs (log_file column of labels csv)",
        "history_size": args.history_size,
        "chunk_size_scenarios": args.chunk_size_scenarios,
        "max_parallel": args.max_parallel,
        "input_scenarios": total_scenarios,
        "chunks_done": len(set(done_chunks)),
        "rows_exported": rows_total,
        "failed_units": failed_units,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    (args.output_dir / "export_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    if args.worker_child:
        if args.chunk_output is None:
            raise SystemExit("--worker-child requires --chunk-output")
        run_child(args)
        return
    labels = load_labels(args.labels_csv)
    by_scenario = group_labels_by_scenario(labels, args.max_scenarios)
    run_supervisor(args, len(by_scenario))


if __name__ == "__main__":
    main()
