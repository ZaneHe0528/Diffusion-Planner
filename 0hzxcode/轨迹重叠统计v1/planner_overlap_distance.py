#!/usr/bin/env python3
"""Measure adjacent planner trajectory overlap distances from nuPlan simulation logs.


python hzxcode/planner_overlap_distance.py \
  /home/ubuntu/code/hezexiang/Diffusion-Planner/exp/exp/simulation/closed_loop_nonreactive_agents/diffusion_planner/val14/diffusion_planner_release/model_2026-06-15-16-34-20 \
  --output-dir hzxcode/planner_overlap_distance_output \
  --max-logs 10 \
  --plot




python hzxcode/planner_overlap_distance.py \
  /home/ubuntu/code/hezexiang/Diffusion-Planner/exp/exp/simulation/closed_loop_nonreactive_agents/diffusion_planner/val14/diffusion_planner_release/model_2026-06-15-16-34-20 \
  --output-dir hzxcode/planner_overlap_distance_output \
  --plot




"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
NUPLAN_DEVKIT_ROOT = REPO_ROOT / "nuplan-devkit"
if str(NUPLAN_DEVKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(NUPLAN_DEVKIT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


@dataclass(frozen=True)
class DistanceRow:
    """One aligned timestamp comparison between two adjacent planner outputs."""

    log_file: str
    scenario_name: str
    log_name: str
    scenario_type: str
    pair_index: int
    old_iteration: int
    new_iteration: int
    time_us: int
    horizon_time_s: float
    old_x: float
    old_y: float
    new_x: float
    new_y: float
    distance_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Euclidean distance distribution between the unexecuted overlap "
            "of one planner trajectory and the next planner trajectory."
        )
    )
    parser.add_argument(
        "simulation_log_root",
        type=Path,
        help="A simulation log file, or a directory containing *.msgpack.xz / *.pkl.xz logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "hzxcode" / "planner_overlap_distance_output",
        help="Directory for CSV/JSON/plot outputs.",
    )
    parser.add_argument(
        "--pair-stride",
        type=int,
        default=1,
        help="Compare sample i with sample i + pair_stride. Use 1 for adjacent planning steps.",
    )
    parser.add_argument(
        "--pose",
        choices=("rear_axle", "center"),
        default="rear_axle",
        help="Which ego pose point to compare in global coordinates.",
    )
    parser.add_argument(
        "--include-current",
        action="store_true",
        help=(
            "Include the new trajectory's first state. By default it is excluded because "
            "nuPlan stores the current ego state there, not a newly predicted future point."
        ),
    )
    parser.add_argument(
        "--max-logs",
        type=int,
        default=None,
        help="Optional cap for quick experiments on a few scenarios.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also save a histogram if matplotlib is available.",
    )
    return parser.parse_args()


def find_simulation_logs(path: Path, max_logs: int | None) -> list[Path]:
    if path.is_file():
        logs = [path]
    else:
        logs = sorted(path.rglob("*.msgpack.xz")) + sorted(path.rglob("*.pkl.xz"))

    if max_logs is not None:
        logs = logs[:max_logs]
    return logs


def state_xy(state: "EgoState", pose: str) -> tuple[float, float]:
    if pose == "center":
        return float(state.center.x), float(state.center.y)
    return float(state.rear_axle.x), float(state.rear_axle.y)


def scenario_attr(simulation_log: object, attr: str) -> str:
    scenario = getattr(simulation_log, "scenario", None)
    value = getattr(scenario, attr, "")
    return str(value() if callable(value) else value)


def sampled_times_us(trajectory: "AbstractTrajectory", include_current: bool) -> list[int]:
    sampled_states = trajectory.get_sampled_trajectory()
    start_time_us = trajectory.start_time.time_us
    times = [state.time_us for state in sampled_states]
    if include_current:
        return [time_us for time_us in times if time_us >= start_time_us]
    return [time_us for time_us in times if time_us > start_time_us]


def rows_for_log(log_path: Path, pose: str, pair_stride: int, include_current: bool) -> Iterator[DistanceRow]:
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.planning.simulation.simulation_log import SimulationLog

    simulation_log = SimulationLog.load_data(log_path)
    history = simulation_log.simulation_history.data
    scenario_name = scenario_attr(simulation_log, "scenario_name")
    log_name = scenario_attr(simulation_log, "log_name")
    scenario_type = scenario_attr(simulation_log, "scenario_type")

    for pair_index, old_index in enumerate(range(0, len(history) - pair_stride)):
        old_sample = history[old_index]
        new_sample = history[old_index + pair_stride]
        old_trajectory = old_sample.trajectory
        new_trajectory = new_sample.trajectory

        overlap_times_us = [
            time_us
            for time_us in sampled_times_us(new_trajectory, include_current)
            if old_trajectory.start_time.time_us <= time_us <= old_trajectory.end_time.time_us
        ]

        for time_us in overlap_times_us:
            old_state = old_trajectory.get_state_at_time(TimePoint(time_us))
            new_state = new_trajectory.get_state_at_time(TimePoint(time_us))
            old_x, old_y = state_xy(old_state, pose)
            new_x, new_y = state_xy(new_state, pose)
            distance_m = float(np.hypot(old_x - new_x, old_y - new_y))
            horizon_time_s = (time_us - new_trajectory.start_time.time_us) / 1e6

            yield DistanceRow(
                log_file=str(log_path),
                scenario_name=scenario_name,
                log_name=log_name,
                scenario_type=scenario_type,
                pair_index=pair_index,
                old_iteration=int(old_sample.iteration.index),
                new_iteration=int(new_sample.iteration.index),
                time_us=int(time_us),
                horizon_time_s=float(horizon_time_s),
                old_x=old_x,
                old_y=old_y,
                new_x=new_x,
                new_y=new_y,
                distance_m=distance_m,
            )


def write_rows_csv(path: Path, rows: Sequence[DistanceRow]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(DistanceRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def percentile_dict(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    percentiles = np.percentile(values, [50, 75, 90, 95, 99])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p50": float(percentiles[0]),
        "p75": float(percentiles[1]),
        "p90": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "max": float(np.max(values)),
    }


def write_horizon_stats_csv(path: Path, rows: Sequence[DistanceRow]) -> None:
    grouped: dict[float, list[float]] = {}
    for row in rows:
        grouped.setdefault(round(row.horizon_time_s, 6), []).append(row.distance_m)

    fieldnames = ["horizon_time_s", "count", "mean", "std", "min", "p50", "p75", "p90", "p95", "p99", "max"]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for horizon_time_s in sorted(grouped):
            stats = percentile_dict(np.asarray(grouped[horizon_time_s], dtype=np.float64))
            writer.writerow({"horizon_time_s": horizon_time_s, **stats})


def write_summary_json(path: Path, rows: Sequence[DistanceRow], logs: Sequence[Path], args: argparse.Namespace) -> None:
    distances = np.asarray([row.distance_m for row in rows], dtype=np.float64)
    summary = {
        "simulation_log_root": str(args.simulation_log_root),
        "num_logs_found": len(logs),
        "num_distance_points": len(rows),
        "pair_stride": args.pair_stride,
        "pose": args.pose,
        "include_current": args.include_current,
        "distance_m": percentile_dict(distances),
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def maybe_write_plot(path: Path, rows: Sequence[DistanceRow]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip plot output.")
        return

    distances = np.asarray([row.distance_m for row in rows], dtype=np.float64)
    if distances.size == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(distances, bins=80)
    plt.xlabel("Overlap Euclidean distance (m)")
    plt.ylabel("Count")
    plt.title("Adjacent planner trajectory overlap distance")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def collect_rows(logs: Iterable[Path], args: argparse.Namespace) -> list[DistanceRow]:
    rows: list[DistanceRow] = []
    for index, log_path in enumerate(logs, start=1):
        print(f"[{index}] loading {log_path}")
        rows.extend(rows_for_log(log_path, args.pose, args.pair_stride, args.include_current))
    return rows


def main() -> None:
    args = parse_args()
    if args.pair_stride < 1:
        raise ValueError("--pair-stride must be >= 1")

    logs = find_simulation_logs(args.simulation_log_root, args.max_logs)
    if not logs:
        raise FileNotFoundError(f"No simulation logs found under {args.simulation_log_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(logs, args)

    write_rows_csv(args.output_dir / "overlap_distances.csv", rows)
    write_horizon_stats_csv(args.output_dir / "overlap_distance_by_horizon.csv", rows)
    write_summary_json(args.output_dir / "summary.json", rows, logs, args)
    if args.plot:
        maybe_write_plot(args.output_dir / "overlap_distance_hist.png", rows)

    print(f"Wrote {len(rows)} distance points from {len(logs)} logs to {args.output_dir}")


if __name__ == "__main__":
    main()
