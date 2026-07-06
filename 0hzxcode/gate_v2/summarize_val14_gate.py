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


def find_latest_aggregator(exp_root: Path):
    candidates = sorted(exp_root.glob("**/aggregator_metric/*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", type=Path, default=DIAG)
    parser.add_argument("--exp-root", type=Path, default=REPO / "exp" / "exp" / "simulation" / "closed_loop_nonreactive_agents")
    parser.add_argument("--out", type=Path, default=REPO / "0实验结果" / "DP-val14-gate-warmstart.md")
    args = parser.parse_args()

    frames = load_frames(args.diag)
    lines = ["# val14 gate warm-start 评测汇总\n"]

    if frames:
        nfe = np.array([r["nfe"] for r in frames if r.get("nfe") is not None], dtype=float)
        ms = np.array([r["decoder_ms"] for r in frames if r.get("decoder_ms") is not None], dtype=float)
        forced = sum(1 for r in frames if r.get("forced_full"))
        passive = sum(1 for r in frames if r.get("passive_fallback"))
        lines += [
            "## 诊断统计 (frames.jsonl)\n",
            f"- 帧数: {len(frames)}",
            f"- 平均 NFE: {nfe.mean():.2f} (基线 {BASELINE_NFE})",
            f"- NFE 加速比: {BASELINE_NFE / max(nfe.mean(), 1e-6):.2f}x",
            f"- 平均 decoder 延迟: {ms.mean():.2f} ms (profile 基线 {BASELINE_DECODER_MS} ms)",
            f"- decoder 加速比: {BASELINE_DECODER_MS / max(ms.mean(), 1e-6):.2f}x",
            f"- 主动判难率: {forced / len(frames):.3f}",
            f"- 被动回退率: {passive / len(frames):.3f}",
            "",
        ]
    else:
        lines += ["## 诊断统计\n", "- 未找到 frames.jsonl，请先运行 `sim_gate_warmstart_runner.sh`\n"]

    agg = find_latest_aggregator(args.exp_root / "diffusion_planner")
    if agg:
        try:
            import pandas as pd

            df = pd.read_parquet(agg)
            score_col = [c for c in df.columns if "score" in c.lower() or "metric" in c.lower()]
            lines += [f"## 仿真分数\n", f"- aggregator: `{agg}`\n"]
            if "score" in df.columns:
                total = float(df["score"].mean())
                lines.append(f"- CLS-NR 总分: **{total:.2f}** (基线 {BASELINE_SCORE}, Δ {total - BASELINE_SCORE:+.2f})")
        except Exception as e:
            lines += [f"## 仿真分数\n", f"- 读取 aggregator 失败: {e}\n"]
    else:
        lines += ["## 仿真分数\n", "- 未找到 aggregator parquet\n"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
