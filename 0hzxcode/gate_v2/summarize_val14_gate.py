#!/usr/bin/env python3
"""汇总 gate warm-start val14 闭环诊断与仿真分数。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "0hzxcode" / "gate_v2_output" / "val14_closedloop" / "frames.jsonl"
BASELINE_SCORE = 89.59
BASELINE_NFE = 11
BASELINE_DECODER_MS = 19.0


def load_frames(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_latest_aggregator(exp_root: Path, allow_any: bool = False):
    candidates = sorted(
        exp_root.glob("**/val14_gate_warmstart/**/aggregator_metric/*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates and allow_any:
        candidates = sorted(exp_root.glob("**/aggregator_metric/*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def score_to_points(score: float) -> float:
    return score * 100.0 if score <= 1.5 else score


def read_score_points(path: Path) -> float | None:
    import pyarrow.parquet as pq

    schema_names = set(pq.read_schema(path).names)
    if "score" not in schema_names:
        return None
    columns = ["score"]
    if "scenario_type" in schema_names:
        columns.append("scenario_type")
    if "scenario" in schema_names:
        columns.append("scenario")
    table = pq.read_table(path, columns=columns)
    scores = table["score"].to_pylist()
    scenario_types = table["scenario_type"].to_pylist() if "scenario_type" in table.column_names else [None] * len(scores)
    scenarios = table["scenario"].to_pylist() if "scenario" in table.column_names else [None] * len(scores)

    for score, scenario_type, scenario in reversed(list(zip(scores, scenario_types, scenarios))):
        if score is not None and (scenario_type == "final_score" or scenario == "final_score"):
            return score_to_points(float(score))

    numeric_scores = [float(s) for s in scores if s is not None]
    if not numeric_scores:
        return None
    return score_to_points(float(np.mean(numeric_scores)))


def read_runner_runtime_ms(path: Path) -> float | None:
    import pyarrow.parquet as pq

    col = "compute_trajectory_runtimes_mean"
    if col not in set(pq.read_schema(path).names):
        return None
    values = [x for x in pq.read_table(path, columns=[col])[col].to_pylist() if x is not None]
    if not values:
        return None
    return float(np.mean(values) * 1000.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", type=Path, default=DIAG)
    parser.add_argument("--exp-root", type=Path, default=REPO / "exp" / "exp" / "simulation" / "closed_loop_nonreactive_agents")
    parser.add_argument("--out", type=Path, default=REPO / "0实验结果" / "DP-val14-gate-warmstart.md")
    parser.add_argument("--allow-any-aggregator", action="store_true")
    args = parser.parse_args()

    frames = load_frames(args.diag)
    lines = ["# val14 gate warm-start 评测汇总\n"]

    if frames:
        nfe = np.array([r["nfe"] for r in frames if r.get("nfe") is not None], dtype=float)
        ms = np.array([r["decoder_ms"] for r in frames if r.get("decoder_ms") is not None], dtype=float)
        forced = sum(1 for r in frames if r.get("forced_full"))
        passive = sum(1 for r in frames if r.get("passive_fallback"))
        avg_nfe = float(nfe.mean()) if nfe.size else float("nan")
        avg_ms = float(ms.mean()) if ms.size else float("nan")
        lines += [
            "## 诊断统计 (frames.jsonl)\n",
            f"- 帧数: {len(frames)}",
            f"- 平均 NFE: {avg_nfe:.2f} (基线 {BASELINE_NFE})",
            f"- NFE 加速比: {BASELINE_NFE / max(avg_nfe, 1e-6):.2f}x",
            f"- 平均 decoder 延迟: {avg_ms:.2f} ms (profile 基线 {BASELINE_DECODER_MS} ms)",
            f"- decoder 加速比: {BASELINE_DECODER_MS / max(avg_ms, 1e-6):.2f}x",
            f"- 主动判难率: {forced / len(frames):.3f}",
            f"- 被动回退率: {passive / len(frames):.3f}",
            "",
        ]
    else:
        lines += ["## 诊断统计\n", "- 未找到 frames.jsonl，请先运行 `sim_gate_warmstart_runner.sh`\n"]

    agg = find_latest_aggregator(args.exp_root / "diffusion_planner", allow_any=args.allow_any_aggregator)
    if agg:
        try:
            lines += [f"## 仿真分数\n", f"- aggregator: `{agg}`\n"]
            total = read_score_points(agg)
            if total is not None:
                lines.append(f"- CLS-NR 总分: **{total:.2f}** (基线 {BASELINE_SCORE:.2f}, Δ {total - BASELINE_SCORE:+.2f})")
            runner = agg.parent.parent / "runner_report.parquet"
            if runner.exists():
                runtime_ms = read_runner_runtime_ms(runner)
                if runtime_ms is not None:
                    lines.append(f"- runner 平均 compute_trajectory: {runtime_ms:.2f} ms")
        except Exception as e:
            lines += [f"## 仿真分数\n", f"- 读取 aggregator 失败: {e}\n"]
    else:
        lines += ["## 仿真分数\n", "- 未找到 aggregator parquet\n"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
