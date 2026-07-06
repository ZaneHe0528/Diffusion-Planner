#!/usr/bin/env python3
"""Export raw-observation caches for adaptive gate training.

This script is intentionally separate from data_process.py.  The normal
preprocessor writes samples for Diffusion-Planner training.  Gate training also
needs sequence metadata and the ego history so model-generated adjacent
trajectories can later be joined back as d labels.

The script does not compute d.  d must be computed from model/planner predicted
trajectories, after aligning adjacent frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

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
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
NUPLAN_DEVKIT_ROOT = REPO_ROOT / "nuplan-devkit"
for _path in (str(REPO_ROOT), str(NUPLAN_DEVKIT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

def load_runtime_dependencies() -> None:
    global Point2D
    global NuPlanScenarioBuilder, ScenarioFilter, ScenarioMapping
    global SingleMachineParallelExecutor, Sequential
    global convert_absolute_to_relative_poses
    global agent_future_process, agent_past_process
    global sampled_static_objects_to_array_list, sampled_tracked_objects_to_array_list
    global calculate_additional_ego_states, sampled_past_ego_states_to_array
    global get_neighbor_vector_set_map, map_process, route_roadblock_correction

    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
    from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    from diffusion_planner.data_process.agent_process import (
        agent_future_process,
        agent_past_process,
        sampled_static_objects_to_array_list,
        sampled_tracked_objects_to_array_list,
    )
    from diffusion_planner.data_process.ego_process import (
        calculate_additional_ego_states,
        sampled_past_ego_states_to_array,
    )
    from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
    from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def load_json_list(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"JSON list not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [str(x) for x in data]


def make_sample_id(log_name: str, scenario_name: str, iteration: int) -> str:
    return f"{log_name}__{scenario_name}__iter_{int(iteration):06d}"


def make_scenario_id(log_name: str, scenario_name: str) -> str:
    return f"{log_name}__{scenario_name}"


def ego_state_to_array(ego_state: EgoState) -> np.ndarray:
    return np.asarray(
        [
            ego_state.rear_axle.x,
            ego_state.rear_axle.y,
            ego_state.rear_axle.heading,
            ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
            ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
            ego_state.dynamic_car_state.rear_axle_acceleration_2d.y,
        ],
        dtype=np.float32,
    )


def time_points_to_array(time_points: list[TimePoint]) -> np.ndarray:
    return np.asarray([t.time_us for t in time_points], dtype=np.int64)


def get_ego_past_array_from_scenario_at_iteration(
    scenario: Any,
    iteration: int,
    num_past_poses: int,
    past_time_horizon: float,
) -> tuple[np.ndarray, np.ndarray]:
    current_ego_state = scenario.get_ego_state_at_iteration(iteration)
    past_ego_states = list(
        scenario.get_ego_past_trajectory(
            iteration=iteration,
            num_samples=num_past_poses,
            time_horizon=past_time_horizon,
        )
    )
    past_time_stamps = list(
        scenario.get_past_timestamps(
            iteration=iteration,
            num_samples=num_past_poses,
            time_horizon=past_time_horizon,
        )
    )

    sampled_past_ego_states = past_ego_states + [current_ego_state]
    sampled_time_stamps = past_time_stamps + [scenario.get_time_point(iteration)]
    if len(sampled_past_ego_states) != num_past_poses + 1:
        raise ValueError(f"expected {num_past_poses + 1} ego history states, got {len(sampled_past_ego_states)}")
    if len(sampled_time_stamps) != num_past_poses + 1:
        raise ValueError(f"expected {num_past_poses + 1} history timestamps, got {len(sampled_time_stamps)}")

    return sampled_past_ego_states_to_array(sampled_past_ego_states), time_points_to_array(sampled_time_stamps)


def get_ego_future_array_from_scenario_at_iteration(
    scenario: Any,
    iteration: int,
    current_ego_state: EgoState,
    num_future_poses: int,
    future_time_horizon: float,
) -> np.ndarray:
    future_trajectory_absolute_states = list(
        scenario.get_ego_future_trajectory(
            iteration=iteration,
            num_samples=num_future_poses,
            time_horizon=future_time_horizon,
        )
    )
    if len(future_trajectory_absolute_states) != num_future_poses:
        raise ValueError(f"expected {num_future_poses} future ego states, got {len(future_trajectory_absolute_states)}")
    return convert_absolute_to_relative_poses(
        current_ego_state.rear_axle,
        [state.rear_axle for state in future_trajectory_absolute_states],
    )


def get_filter_parameters(
    num_scenarios_per_type: int | None = None,
    limit_total_scenarios: int | float | None = None,
    shuffle: bool = False,
    scenario_tokens: list[str] | None = None,
    log_names: list[str] | None = None,
):
    scenario_types = None
    map_names = None
    timestamp_threshold_s = None
    ego_displacement_minimum_m = None
    expand_scenarios = False
    remove_invalid_goals = False
    ego_start_speed_threshold = None
    ego_stop_speed_threshold = None
    speed_noise_tolerance = None

    return (
        scenario_types,
        scenario_tokens,
        log_names,
        map_names,
        num_scenarios_per_type,
        limit_total_scenarios,
        timestamp_threshold_s,
        ego_displacement_minimum_m,
        expand_scenarios,
        remove_invalid_goals,
        shuffle,
        ego_start_speed_threshold,
        ego_stop_speed_threshold,
        speed_noise_tolerance,
    )


class GateDataProcessor:
    def __init__(self, config: argparse.Namespace):
        self._save_dir = Path(config.save_path)
        self._compressed = config.compressed
        self._include_future_gt = not config.no_future_gt

        self.past_time_horizon = float(config.past_time_horizon)
        self.num_past_poses = int(config.num_past_poses)
        self.future_time_horizon = float(config.future_time_horizon)
        self.num_future_poses = int(config.num_future_poses)

        self.num_agents = int(config.agent_num)
        self.num_static = int(config.static_objects_num)
        self.max_ped_bike = int(config.max_ped_bike)
        self._radius = float(config.radius)

        self._map_features = ["LANE", "LEFT_BOUNDARY", "RIGHT_BOUNDARY", "ROUTE_LANES"]
        self._max_elements = {
            "LANE": int(config.lane_num),
            "LEFT_BOUNDARY": int(config.lane_num),
            "RIGHT_BOUNDARY": int(config.lane_num),
            "ROUTE_LANES": int(config.route_num),
        }
        self._max_points = {
            "LANE": int(config.lane_len),
            "LEFT_BOUNDARY": int(config.lane_len),
            "RIGHT_BOUNDARY": int(config.lane_len),
            "ROUTE_LANES": int(config.route_len),
        }

    def process_frame(self, scenario: Any, iteration: int) -> dict[str, np.ndarray]:
        map_name = getattr(scenario, "_map_name", "")
        map_api = scenario.map_api

        ego_state = scenario.get_ego_state_at_iteration(iteration)
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.asarray(
            [ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading],
            dtype=np.float64,
        )

        ego_history_global, history_timestamps_us = get_ego_past_array_from_scenario_at_iteration(
            scenario,
            iteration,
            self.num_past_poses,
            self.past_time_horizon,
        )

        present_tracked_objects = scenario.get_tracked_objects_at_iteration(iteration).tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_past_tracked_objects(
                iteration=iteration,
                time_horizon=self.past_time_horizon,
                num_samples=self.num_past_poses,
            )
        ]
        if len(past_tracked_objects) != self.num_past_poses:
            raise ValueError(f"expected {self.num_past_poses} past object frames, got {len(past_tracked_objects)}")

        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(sampled_past_observations)
        static_objects, static_objects_types = sampled_static_objects_to_array_list(present_tracked_objects)
        ego_history, neighbor_agents_past, neighbor_indices, static_objects = agent_past_process(
            ego_history_global.copy(),
            neighbor_agents_past,
            neighbor_agents_types,
            self.num_agents,
            static_objects,
            static_objects_types,
            self.num_static,
            self.max_ped_bike,
            anchor_ego_state,
        )

        route_roadblock_ids = list(scenario.get_route_roadblock_ids())
        traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(iteration))
        if route_roadblock_ids and route_roadblock_ids != [""]:
            route_roadblock_ids = route_roadblock_correction(ego_state, map_api, route_roadblock_ids)

        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api,
            self._map_features,
            ego_coords,
            self._radius,
            traffic_light_data,
        )
        vector_map = map_process(
            route_roadblock_ids,
            anchor_ego_state,
            coords,
            traffic_light_data,
            speed_limit,
            lane_route,
            self._map_features,
            self._max_elements,
            self._max_points,
        )

        ego_current_state = calculate_additional_ego_states(ego_history, history_timestamps_us)
        data: dict[str, np.ndarray] = {
            # Main model inputs.
            "ego_current_state": ego_current_state.astype(np.float32),
            "neighbor_agents_past": neighbor_agents_past.astype(np.float32),
            "static_objects": static_objects.astype(np.float32),
            # Gate-specific ego input.
            "ego_history": ego_history.astype(np.float32),
            # Metadata needed to join model trajectory outputs and d labels.
            "anchor_ego_pose": anchor_ego_state.astype(np.float32),
            "ego_current_global": ego_state_to_array(ego_state),
            "ego_history_global": ego_history_global.astype(np.float32),
            "history_timestamps_us": history_timestamps_us,
            "timestamp_us": np.asarray(scenario.get_time_point(iteration).time_us, dtype=np.int64),
            "iteration": np.asarray(iteration, dtype=np.int64),
            "database_interval_s": np.asarray(float(scenario.database_interval), dtype=np.float32),
            "route_roadblock_ids": np.asarray(route_roadblock_ids, dtype=str),
            "map_name": np.asarray(map_name, dtype=str),
            "log_name": np.asarray(scenario.log_name, dtype=str),
            "scenario_name": np.asarray(scenario.scenario_name, dtype=str),
            "scenario_id": np.asarray(make_scenario_id(scenario.log_name, scenario.scenario_name), dtype=str),
            "scenario_type": np.asarray(scenario.scenario_type, dtype=str),
            "sample_id": np.asarray(make_sample_id(scenario.log_name, scenario.scenario_name, iteration), dtype=str),
        }
        data.update(vector_map)

        if self._include_future_gt:
            ego_agent_future = get_ego_future_array_from_scenario_at_iteration(
                scenario,
                iteration,
                ego_state,
                self.num_future_poses,
                self.future_time_horizon,
            )
            future_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_future_tracked_objects(
                    iteration=iteration,
                    time_horizon=self.future_time_horizon,
                    num_samples=self.num_future_poses,
                )
            ]
            if len(future_tracked_objects) != self.num_future_poses:
                raise ValueError(f"expected {self.num_future_poses} future object frames, got {len(future_tracked_objects)}")
            sampled_future_observations = [present_tracked_objects] + future_tracked_objects
            future_tracked_objects_array_list, _ = sampled_tracked_objects_to_array_list(sampled_future_observations)
            neighbor_agents_future = agent_future_process(
                anchor_ego_state,
                future_tracked_objects_array_list,
                self.num_agents,
                neighbor_indices,
            )
            data["ego_agent_future"] = ego_agent_future.astype(np.float32)
            data["neighbor_agents_future"] = neighbor_agents_future.astype(np.float32)

        return data

    def save_frame(self, data: dict[str, np.ndarray]) -> str:
        sample_id = str(data["sample_id"].item())
        filename = f"{sample_id}.npz"
        path = self._save_dir / filename
        if self._compressed:
            np.savez_compressed(path, **data)
        else:
            np.savez(path, **data)
        return filename


def iter_frame_indices(num_iterations: int, frame_stride: int, max_frames: int | None) -> list[int]:
    indices = list(range(0, num_iterations, frame_stride))
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def resolve_db_files_for_limited_run(
    data_path: str,
    log_names: list[str] | None,
    explicit_db_files: str | None,
    total_scenarios: int | None,
    limited_db_files: int,
) -> str | list[str] | None:
    if explicit_db_files:
        return explicit_db_files
    if total_scenarios is None or log_names is None or limited_db_files <= 0:
        return None

    root = Path(data_path)
    paths: list[str] = []
    for log_name in log_names[:limited_db_files]:
        candidates = [root / log_name]
        if not log_name.endswith(".db"):
            candidates.insert(0, root / f"{log_name}.db")
        for candidate in candidates:
            if candidate.exists():
                paths.append(str(candidate))
                break

    if paths:
        print(
            f"limited run: scanning {len(paths)} db file(s) from {data_path} "
            f"before applying total_scenarios={total_scenarios}",
            flush=True,
        )
        return paths
    print("limited run: no matching db files found from log_names_json; falling back to full discovery", flush=True)
    return None


def build_scenarios(args: argparse.Namespace) -> list[Any]:
    log_names = None if args.all_logs else load_json_list(args.log_names_json)
    scenario_tokens = load_json_list(args.scenario_tokens_json)
    total_scenarios = None if args.total_scenarios <= 0 else args.total_scenarios
    db_files = resolve_db_files_for_limited_run(
        args.data_path,
        log_names,
        args.db_files,
        total_scenarios,
        args.limited_db_files,
    )
    scenario_mapping = ScenarioMapping({}, args.scenario_subsample_ratio)
    builder = NuPlanScenarioBuilder(
        args.data_path,
        args.map_path,
        sensor_root=None,
        db_files=db_files,
        map_version=args.map_version,
        scenario_mapping=scenario_mapping,
    )
    scenario_filter = ScenarioFilter(
        *get_filter_parameters(
            args.scenarios_per_type,
            total_scenarios,
            args.shuffle_scenarios,
            scenario_tokens=scenario_tokens,
            log_names=log_names,
        )
    )
    if args.num_workers <= 1:
        worker = Sequential()
    else:
        worker = SingleMachineParallelExecutor(
            use_process_pool=args.use_process_pool,
            max_workers=args.num_workers,
        )
    return builder.get_scenarios(scenario_filter, worker)


def write_metadata_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "file",
        "sample_id",
        "scenario_id",
        "log_name",
        "scenario_name",
        "scenario_type",
        "map_name",
        "iteration",
        "timestamp_us",
        "anchor_x",
        "anchor_y",
        "anchor_heading",
        "database_interval_s",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-frame raw observation data for adaptive gate training.")
    parser.add_argument("--data_path", default="/data/nuplan-v1.1/trainval", type=str, help="path to raw nuPlan data")
    parser.add_argument("--map_path", default="/data/nuplan-v1.1/maps", type=str, help="path to map data")
    parser.add_argument("--db_files", default=None, type=str, help="optional db file, db directory, or list accepted by nuPlan")
    parser.add_argument("--map_version", default="nuplan-maps-v1.0", type=str)
    parser.add_argument("--save_path", default="./gate_cache", type=str, help="directory for exported npz files")
    parser.add_argument("--data_list", default="./diffusion_planner_gate.json", type=str, help="output JSON file list")
    parser.add_argument("--metadata_csv", default="./diffusion_planner_gate_metadata.csv", type=str)
    parser.add_argument("--summary_json", default="./diffusion_planner_gate_summary.json", type=str)

    parser.add_argument("--log_names_json", type=Path, default=Path("./nuplan_train.json"))
    parser.add_argument("--scenario_tokens_json", type=Path, default=None)
    parser.add_argument("--all_logs", action="store_true", help="ignore --log_names_json and scan all db files under data_path")
    parser.add_argument("--scenarios_per_type", type=int, default=None)
    parser.add_argument("--total_scenarios", type=int, default=10, help="0 means no total-scenario limit")
    parser.add_argument("--shuffle_scenarios", type=parse_bool, default=False)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_process_pool", type=parse_bool, default=True)
    parser.add_argument(
        "--limited_db_files",
        type=int,
        default=16,
        help="when total_scenarios > 0 and --db_files is unset, scan only this many log-name dbs first",
    )

    # 0.5 turns the 20Hz DB stream into 10Hz scenario iterations, matching the planner loop used by d analysis.
    parser.add_argument("--scenario_subsample_ratio", type=float, default=0.5)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames_per_scenario", type=int, default=None)

    parser.add_argument("--agent_num", type=int, default=32)
    parser.add_argument("--static_objects_num", type=int, default=5)
    parser.add_argument("--max_ped_bike", type=int, default=10)
    parser.add_argument("--lane_len", type=int, default=20)
    parser.add_argument("--lane_num", type=int, default=70)
    parser.add_argument("--route_len", type=int, default=20)
    parser.add_argument("--route_num", type=int, default=25)
    parser.add_argument("--radius", type=float, default=100.0)

    parser.add_argument("--past_time_horizon", type=float, default=2.0)
    parser.add_argument("--num_past_poses", type=int, default=20)
    parser.add_argument("--future_time_horizon", type=float, default=8.0)
    parser.add_argument("--num_future_poses", type=int, default=80)
    parser.add_argument("--no_future_gt", action="store_true", help="skip ego_agent_future / neighbor_agents_future export")
    parser.add_argument("--compressed", action="store_true", help="write npz with compression")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame_stride must be positive")
    if not (0.0 < args.scenario_subsample_ratio <= 1.0):
        raise ValueError("--scenario_subsample_ratio must be in (0, 1]")

    load_runtime_dependencies()

    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_scenarios(args)
    print(f"Total number of scenarios: {len(scenarios)}")

    processor = GateDataProcessor(args)
    data_list: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    skipped_short_future = 0

    for scenario in tqdm(scenarios, desc="scenarios"):
        frame_indices = iter_frame_indices(
            scenario.get_number_of_iterations(),
            args.frame_stride,
            args.max_frames_per_scenario,
        )
        for iteration in frame_indices:
            if not args.no_future_gt and iteration + args.num_future_poses >= scenario.get_number_of_iterations():
                skipped_short_future += 1
                continue
            try:
                data = processor.process_frame(scenario, iteration)
                filename = processor.save_frame(data)
            except Exception as exc:
                failed_rows.append(
                    {
                        "scenario_id": make_scenario_id(scenario.log_name, scenario.scenario_name),
                        "iteration": int(iteration),
                        "reason": str(exc),
                    }
                )
                continue

            anchor = data["anchor_ego_pose"]
            data_list.append(filename)
            metadata_rows.append(
                {
                    "file": filename,
                    "sample_id": str(data["sample_id"].item()),
                    "scenario_id": str(data["scenario_id"].item()),
                    "log_name": str(data["log_name"].item()),
                    "scenario_name": str(data["scenario_name"].item()),
                    "scenario_type": str(data["scenario_type"].item()),
                    "map_name": str(data["map_name"].item()),
                    "iteration": int(data["iteration"].item()),
                    "timestamp_us": int(data["timestamp_us"].item()),
                    "anchor_x": float(anchor[0]),
                    "anchor_y": float(anchor[1]),
                    "anchor_heading": float(anchor[2]),
                    "database_interval_s": float(data["database_interval_s"].item()),
                }
            )

    data_list_path = Path(args.data_list)
    data_list_path.parent.mkdir(parents=True, exist_ok=True)
    with data_list_path.open("w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4)
    write_metadata_csv(Path(args.metadata_csv), metadata_rows)

    summary = {
        "save_path": str(save_dir),
        "data_list": str(args.data_list),
        "metadata_csv": str(args.metadata_csv),
        "rows_exported": len(data_list),
        "scenarios_loaded": len(scenarios),
        "failed_rows": failed_rows[:100],
        "num_failed_rows": len(failed_rows),
        "skipped_short_future": skipped_short_future,
        "scenario_subsample_ratio": args.scenario_subsample_ratio,
        "frame_stride": args.frame_stride,
        "note": "d is not computed here; join model-generated adjacent-trajectory labels by sample_id.",
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
