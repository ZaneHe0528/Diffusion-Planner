#!/usr/bin/env python3
"""离线分析:统计 Diffusion-Planner 相邻规划帧之间「重叠未来轨迹」的差异。

目的(Falcon 可行性分析,非实现复用):
    判断相邻规划帧输出的未来轨迹是否足够相似,从而评估 Falcon 式
    (partial-denoising / 历史轨迹复用)在 Diffusion-Planner 上是否可行。

数据来源(方案 A,不改训练 / 不改 planner 推理):
    直接解析 nuPlan 闭环仿真日志 (*.msgpack.xz / *.pkl.xz)。每个
    SimulationHistorySample.trajectory 就是「那一帧规划输出的 ego 未来轨迹」。

================== 关键假设(同时写入注释与脚本输出) ==================
[A1] 轨迹形状:Diffusion-Planner 经 transform_predictions_to_states(include_ego_state=True)
     生成轨迹,采样点数 = 81(index 0 = 当前帧 ego 状态,index 1..80 = 未来 80 个预测点),
     时间跨度 8s,步长 dt = 0.1s。脚本会实际读取并打印真实值,不硬编码。
[A2] 坐标系:轨迹已由 relative_to_absolute_poses 转到 **global / 绝对地图坐标**,
     因此相邻帧可以直接比较,无需再做坐标变换。(若未来发现是 ego frame,需先对齐。)
[A3] 字段:每个采样点取 EgoState 的 (x, y, heading)。x,y 单位米,heading 单位弧度。
[A4] gap:相邻规划帧的时间间隔 / 轨迹步长。脚本从 timestamp 自动推断;
     若无法推断则默认 gap = 1 并在输出中显式标注。闭环仿真为 10Hz,轨迹步长 0.1s,
     故正常情况下 gap == 1。
[A5] 对齐方式(按时间偏移,Falcon 风格):
       previous_overlap = traj[i][gap : T]      # 上一帧「未执行」的尾部
       current_overlap  = traj[i+1][0 : T-gap]  # 新一帧对应同一段时间的头部
     仅在两段长度一致时计算距离。脚本额外校验两段对应采样点的绝对时间确实对齐。

================== 指标定义 ==================
主判据(只用位置 x, y):
    overall_l2     = || (prev_xy - cur_xy).flatten() ||_2          # 整段一个标量
    normalized_l2  = overall_l2 / sqrt(overlap_len)                 # = 每步位置差的 RMS(米)
                     便于不同 overlap 长度 / horizon 互相比较。
辅助(不作为主判据):
    per-step Euclidean distance 的 mean / max / P50 / P90 / P95 / P99(米)。
heading 单独处理(绝不与米混入同一个 L2):
    yaw 误差 = wrap_to_pi(prev_yaw - cur_yaw),单独输出 mean / max(弧度)。

================== 输出 ==================
output-dir 下:
    per_pair_overlap_l2.csv      每个相邻帧对一行(主数据)
    per_scenario_summary.csv     按 scenario 分组的汇总
    summary.json                 全局统计 + epsilon 候选 + 假设记录
    hist_normalized_l2.png / cdf_normalized_l2.png
    hist_overall_l2.png   / cdf_overall_l2.png
    hist_perstep_dist.png / cdf_perstep_dist.png

运行示例:
    conda activate dp
    python hzxcode/analyze_adjacent_traj_l2.py \\
      exp/exp/simulation/closed_loop_nonreactive_agents/diffusion_planner/val14/diffusion_planner_release/model_2026-06-15-16-34-20 \\
      --output-dir hzxcode/adjacent_traj_l2_output \\
      --max-logs 50          # 先跑小批验证,去掉则跑全部
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
NUPLAN_DEVKIT_ROOT = REPO_ROOT / "nuplan-devkit"
for _p in (NUPLAN_DEVKIT_ROOT, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if TYPE_CHECKING:
    from nuplan.common.actor_state.ego_state import EgoState
    from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

DEFAULT_LOG_ROOT = (
    REPO_ROOT
    / "exp/exp/simulation/closed_loop_nonreactive_agents/diffusion_planner/"
    "val14/diffusion_planner_release/model_2026-06-15-16-34-20"
)

# 推断不出 gap 时的默认值(假设 A4)。
DEFAULT_GAP_ASSUMPTION = 1


@dataclass(frozen=True)
class PairRow:
    """一对相邻规划帧 (i, i+1) 在其重叠未来轨迹上的差异统计。"""

    log_file: str
    scenario_type: str
    log_name: str
    scenario_name: str
    pair_index: int
    old_iteration: int
    new_iteration: int
    gap_steps: int
    overlap_len: int
    time_aligned: bool  # 两段重叠的绝对时间是否真正对齐(健全性检查)
    # --- 主判据:位置 (x, y) ---
    overall_l2_xy: float
    normalized_l2_xy: float
    # --- 辅助:per-step 欧氏距离(米) ---
    perstep_mean_m: float
    perstep_max_m: float
    perstep_p50_m: float
    perstep_p90_m: float
    perstep_p95_m: float
    perstep_p99_m: float
    # --- heading 单独处理(弧度) ---
    yaw_mean_err_rad: float
    yaw_max_err_rad: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "simulation_log_root",
        type=Path,
        nargs="?",
        default=DEFAULT_LOG_ROOT,
        help="仿真日志文件,或包含 *.msgpack.xz / *.pkl.xz 的目录。默认指向已有的 val14 运行。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "hzxcode" / "adjacent_traj_l2_output",
        help="CSV / JSON / 图输出目录。",
    )
    parser.add_argument(
        "--pose",
        choices=("rear_axle", "center"),
        default="rear_axle",
        help="比较哪个 ego 位姿点(global 坐标)。planner 输出的是 rear_axle,默认 rear_axle。",
    )
    parser.add_argument(
        "--pair-stride",
        type=int,
        default=1,
        help="比较 sample i 与 sample i+pair_stride。相邻规划帧用 1。",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=None,
        help="覆盖 gap(轨迹 step 偏移)。默认 None = 从 timestamp 自动推断,推断失败则用 1。",
    )
    parser.add_argument("--max-logs", type=int, default=None, help="只处理前 N 个日志,便于快速验证。")
    parser.add_argument("--no-plot", action="store_true", help="不画图(只产出 CSV / JSON)。")
    parser.add_argument(
        "--perstep-cap",
        type=int,
        default=3_000_000,
        help="用于分布图/分位数的 per-step 距离采样上限,超过则随机下采样(辅助指标,不影响主判据)。",
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


def scenario_attr(simulation_log: object, attr: str) -> str:
    scenario = getattr(simulation_log, "scenario", None)
    value = getattr(scenario, attr, "")
    try:
        return str(value() if callable(value) else value)
    except Exception:
        return ""


def _state_to_xyh(state: "EgoState", pose: str) -> tuple[float, float, float]:
    """提取 (x, y, heading)(global, 弧度)。"""
    point = state.center if pose == "center" else state.rear_axle
    return float(point.x), float(point.y), float(point.heading)


def trajectory_to_array(trajectory: "AbstractTrajectory", pose: str) -> tuple[np.ndarray, np.ndarray]:
    """把一条规划轨迹转成 (t_us[N], xyh[N, 3])。"""
    states = trajectory.get_sampled_trajectory()
    t_us = np.asarray([int(s.time_point.time_us) for s in states], dtype=np.int64)
    xyh = np.asarray([_state_to_xyh(s, pose) for s in states], dtype=np.float64)
    return t_us, xyh


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def infer_gap(old_t_us: np.ndarray, new_t_us: np.ndarray, override: int | None) -> tuple[int, bool]:
    """从 timestamp 推断 gap(轨迹 step 偏移)。

    返回 (gap, inferred):inferred=False 表示退回默认假设 gap。
    """
    if override is not None:
        return max(int(override), 1), True
    if old_t_us.size < 2:
        return DEFAULT_GAP_ASSUMPTION, False
    dt_us = float(np.median(np.diff(old_t_us)))
    if dt_us <= 0:
        return DEFAULT_GAP_ASSUMPTION, False
    gap = int(round((new_t_us[0] - old_t_us[0]) / dt_us))
    if gap < 1:
        return DEFAULT_GAP_ASSUMPTION, False
    return gap, True


def compute_pair(
    old_t: np.ndarray,
    old_xyh: np.ndarray,
    new_t: np.ndarray,
    new_xyh: np.ndarray,
    gap: int,
) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """按假设 A5 做时间偏移对齐,返回 (per_step_dist[L], yaw_err[L], time_aligned)。"""
    t = min(old_xyh.shape[0], new_xyh.shape[0])
    overlap_len = t - gap
    if overlap_len <= 0:
        return None

    prev = old_xyh[gap:t]          # traj[i][gap : T]
    cur = new_xyh[0 : t - gap]     # traj[i+1][0 : T-gap]
    if prev.shape[0] != cur.shape[0]:
        return None

    # 健全性检查:两段对应采样点的绝对时间是否对齐(容忍半个 dt)。
    prev_t = old_t[gap:t]
    cur_t = new_t[0 : t - gap]
    dt_us = float(np.median(np.diff(old_t))) if old_t.size >= 2 else 1e5
    time_aligned = bool(np.all(np.abs(prev_t - cur_t) <= 0.5 * abs(dt_us)))

    dpos = prev[:, :2] - cur[:, :2]
    per_step_dist = np.hypot(dpos[:, 0], dpos[:, 1])
    yaw_err = np.abs(wrap_to_pi(prev[:, 2] - cur[:, 2]))
    return per_step_dist, yaw_err, time_aligned


def rows_for_log(
    log_path: Path,
    pose: str,
    pair_stride: int,
    gap_override: int | None,
) -> Iterator[tuple[PairRow, np.ndarray]]:
    """逐日志产出 (PairRow, per_step_dist 数组)。"""
    from nuplan.planning.simulation.simulation_log import SimulationLog

    simulation_log = SimulationLog.load_data(log_path)
    history = simulation_log.simulation_history.data
    scenario_type = scenario_attr(simulation_log, "scenario_type")
    log_name = scenario_attr(simulation_log, "log_name")
    scenario_name = scenario_attr(simulation_log, "scenario_name")

    # 预先把每帧轨迹转成数组,避免重复解析。
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def get_arr(idx: int) -> tuple[np.ndarray, np.ndarray]:
        if idx not in cache:
            cache[idx] = trajectory_to_array(history[idx].trajectory, pose)
        return cache[idx]

    for pair_index, old_index in enumerate(range(0, len(history) - pair_stride)):
        new_index = old_index + pair_stride
        old_t, old_xyh = get_arr(old_index)
        new_t, new_xyh = get_arr(new_index)

        gap, _inferred = infer_gap(old_t, new_t, gap_override)
        result = compute_pair(old_t, old_xyh, new_t, new_xyh, gap)
        if result is None:
            continue
        per_step_dist, yaw_err, time_aligned = result

        overlap_len = int(per_step_dist.size)
        overall_l2 = float(np.linalg.norm((old_xyh[gap : gap + overlap_len, :2] - new_xyh[:overlap_len, :2]).ravel()))
        normalized_l2 = overall_l2 / float(np.sqrt(overlap_len))
        pcts = np.percentile(per_step_dist, [50, 90, 95, 99])

        row = PairRow(
            log_file=str(log_path),
            scenario_type=scenario_type,
            log_name=log_name,
            scenario_name=scenario_name,
            pair_index=pair_index,
            old_iteration=int(history[old_index].iteration.index),
            new_iteration=int(history[new_index].iteration.index),
            gap_steps=int(gap),
            overlap_len=overlap_len,
            time_aligned=time_aligned,
            overall_l2_xy=overall_l2,
            normalized_l2_xy=normalized_l2,
            perstep_mean_m=float(np.mean(per_step_dist)),
            perstep_max_m=float(np.max(per_step_dist)),
            perstep_p50_m=float(pcts[0]),
            perstep_p90_m=float(pcts[1]),
            perstep_p95_m=float(pcts[2]),
            perstep_p99_m=float(pcts[3]),
            yaw_mean_err_rad=float(np.mean(yaw_err)),
            yaw_max_err_rad=float(np.max(yaw_err)),
        )
        yield row, per_step_dist


def percentile_dict(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"count": 0}
    pcts = np.percentile(values, [50, 75, 90, 95, 99])
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p50": float(pcts[0]),
        "p75": float(pcts[1]),
        "p90": float(pcts[2]),
        "p95": float(pcts[3]),
        "p99": float(pcts[4]),
        "max": float(np.max(values)),
    }


def write_pair_csv(path: Path, rows: Sequence[PairRow]) -> None:
    fieldnames = list(PairRow.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_scenario_csv(path: Path, rows: Sequence[PairRow]) -> None:
    grouped: dict[tuple[str, str, str], list[PairRow]] = {}
    for row in rows:
        grouped.setdefault((row.scenario_type, row.log_name, row.scenario_name), []).append(row)

    fieldnames = [
        "scenario_type",
        "log_name",
        "scenario_name",
        "num_pairs",
        "norm_l2_mean",
        "norm_l2_p50",
        "norm_l2_p90",
        "norm_l2_p95",
        "norm_l2_max",
        "perstep_mean_m",
        "perstep_p95_m",
        "yaw_mean_err_rad",
        "yaw_max_err_rad",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (s_type, l_name, s_name), grp in sorted(grouped.items()):
            norm = np.asarray([r.normalized_l2_xy for r in grp])
            perstep_mean = np.asarray([r.perstep_mean_m for r in grp])
            perstep_p95 = np.asarray([r.perstep_p95_m for r in grp])
            yaw_mean = np.asarray([r.yaw_mean_err_rad for r in grp])
            yaw_max = np.asarray([r.yaw_max_err_rad for r in grp])
            writer.writerow(
                {
                    "scenario_type": s_type,
                    "log_name": l_name,
                    "scenario_name": s_name,
                    "num_pairs": len(grp),
                    "norm_l2_mean": float(np.mean(norm)),
                    "norm_l2_p50": float(np.percentile(norm, 50)),
                    "norm_l2_p90": float(np.percentile(norm, 90)),
                    "norm_l2_p95": float(np.percentile(norm, 95)),
                    "norm_l2_max": float(np.max(norm)),
                    "perstep_mean_m": float(np.mean(perstep_mean)),
                    "perstep_p95_m": float(np.mean(perstep_p95)),
                    "yaw_mean_err_rad": float(np.mean(yaw_mean)),
                    "yaw_max_err_rad": float(np.mean(yaw_max)),
                }
            )


def make_plots(output_dir: Path, name: str, values: np.ndarray, xlabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if values.size == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=100)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(f"Histogram: {name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"hist_{name}.png", dpi=150)
    plt.close()

    ordered = np.sort(values)
    cdf = np.arange(1, ordered.size + 1) / ordered.size
    plt.figure(figsize=(8, 5))
    plt.plot(ordered, cdf)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(f"CDF: {name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"cdf_{name}.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.pair_stride < 1:
        raise ValueError("--pair-stride 必须 >= 1")

    logs = find_simulation_logs(args.simulation_log_root, args.max_logs)
    if not logs:
        raise FileNotFoundError(f"在 {args.simulation_log_root} 下找不到仿真日志")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[PairRow] = []
    perstep_chunks: list[np.ndarray] = []
    perstep_total = 0
    rng = np.random.default_rng(0)

    for index, log_path in enumerate(logs, start=1):
        print(f"[{index}/{len(logs)}] {log_path.name}")
        try:
            for row, per_step_dist in rows_for_log(log_path, args.pose, args.pair_stride, args.gap):
                rows.append(row)
                perstep_total += per_step_dist.size
                if perstep_total <= args.perstep_cap:
                    perstep_chunks.append(per_step_dist)
                else:
                    # 超过上限后随机保留一部分,保证分布图仍有代表性。
                    keep = rng.random(per_step_dist.size) < (args.perstep_cap / max(perstep_total, 1))
                    if keep.any():
                        perstep_chunks.append(per_step_dist[keep])
        except Exception as exc:  # noqa: BLE001 - 单个损坏日志不应中断整体分析
            print(f"  跳过(load/parse 失败): {exc}")

    if not rows:
        raise RuntimeError("没有产出任何相邻帧对,检查日志内容或 gap 设置。")

    norm_l2 = np.asarray([r.normalized_l2_xy for r in rows], dtype=np.float64)
    overall_l2 = np.asarray([r.overall_l2_xy for r in rows], dtype=np.float64)
    yaw_mean = np.asarray([r.yaw_mean_err_rad for r in rows], dtype=np.float64)
    yaw_max = np.asarray([r.yaw_max_err_rad for r in rows], dtype=np.float64)
    perstep = np.concatenate(perstep_chunks) if perstep_chunks else np.empty(0)

    gaps = np.asarray([r.gap_steps for r in rows])
    aligned_ratio = float(np.mean([r.time_aligned for r in rows]))

    # epsilon 候选:Falcon 复用阈值取 normalized L2 的分位数。
    eps = np.percentile(norm_l2, [50, 75, 90, 95])
    epsilon_candidates = {
        "normalized_l2_p50": float(eps[0]),
        "normalized_l2_p75": float(eps[1]),
        "normalized_l2_p90": float(eps[2]),
        "normalized_l2_p95": float(eps[3]),
    }

    write_pair_csv(args.output_dir / "per_pair_overlap_l2.csv", rows)
    write_scenario_csv(args.output_dir / "per_scenario_summary.csv", rows)

    summary = {
        "assumptions": {
            "A1_traj_points": "transform_predictions_to_states(include_ego_state=True) -> 81 点, dt=0.1s, horizon=8s (脚本读取真实值)",
            "A2_frame": "global / absolute map coordinates (相邻帧可直接比较)",
            "A3_fields": "(x, y, heading); xy=米, heading=弧度",
            "A4_gap_default_when_uninferable": DEFAULT_GAP_ASSUMPTION,
            "A5_alignment": "prev=traj[i][gap:T], cur=traj[i+1][0:T-gap]; 仅等长时计算",
            "pose_point": args.pose,
            "pair_stride": args.pair_stride,
            "gap_override": args.gap,
        },
        "data_health": {
            "num_logs": len(logs),
            "num_pairs": len(rows),
            "gap_value_counts": {int(g): int(c) for g, c in zip(*np.unique(gaps, return_counts=True))},
            "time_aligned_ratio": aligned_ratio,
            "perstep_points_total": int(perstep_total),
            "perstep_points_used_for_dist": int(perstep.size),
        },
        "main_metric_normalized_l2_xy_m": percentile_dict(norm_l2),
        "overall_l2_xy_m": percentile_dict(overall_l2),
        "aux_perstep_distance_m": percentile_dict(perstep),
        "yaw_mean_err_rad": percentile_dict(yaw_mean),
        "yaw_max_err_rad": percentile_dict(yaw_max),
        "epsilon_candidates_normalized_l2_m": epsilon_candidates,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    if not args.no_plot:
        make_plots(args.output_dir, "normalized_l2", norm_l2, "Normalized L2 over overlap (m, RMS per step)")
        make_plots(args.output_dir, "overall_l2", overall_l2, "Overall L2 over overlap (m)")
        make_plots(args.output_dir, "perstep_dist", perstep, "Per-step Euclidean distance (m)")

    # 关键结果 + 假设打印到 stdout(假设 A4/A5 显式标注)。
    print("\n================== 结果摘要 ==================")
    print(f"日志数: {len(logs)} | 相邻帧对数: {len(rows)} | gap 分布: {summary['data_health']['gap_value_counts']}")
    if aligned_ratio < 0.999:
        print(f"[警告] 仅 {aligned_ratio:.3%} 的帧对时间严格对齐,其余结果仅作粗略参考。")
    print(f"[主判据] normalized L2 (米/步 RMS): "
          f"mean={summary['main_metric_normalized_l2_xy_m']['mean']:.3f} "
          f"P50={epsilon_candidates['normalized_l2_p50']:.3f} "
          f"P90={epsilon_candidates['normalized_l2_p90']:.3f} "
          f"P95={epsilon_candidates['normalized_l2_p95']:.3f}")
    print(f"[辅助] per-step 距离(米): mean={summary['aux_perstep_distance_m'].get('mean', float('nan')):.3f} "
          f"P95={summary['aux_perstep_distance_m'].get('p95', float('nan')):.3f}")
    print(f"[heading] yaw_mean_err(rad): mean={summary['yaw_mean_err_rad']['mean']:.4f} | "
          f"yaw_max_err(rad): P95={summary['yaw_max_err_rad']['p95']:.4f}")
    print(f"[epsilon 候选 normalized L2] {epsilon_candidates}")
    print(f"\n输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
