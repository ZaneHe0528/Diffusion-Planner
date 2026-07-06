#!/usr/bin/env python3
"""汇总多 run 的 decision_metrics.json，产出 d 口径选型对比表。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="name=path/to/run_dir (含 metrics.json 或 decision_metrics.json)",
    )
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load_run_metrics(run_dir: Path) -> dict:
    for name in ("decision_metrics.json", "metrics.json"):
        p = run_dir / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no metrics in {run_dir}")


def main() -> None:
    args = parse_args()
    rows = []
    for spec in args.runs:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            label, path = Path(spec).name, spec
        m = load_run_metrics(Path(path))
        test = m["splits"]["test"]
        pri = test.get("primary", test)
        r95 = test.get("reuse_at_hard_recall_95", {})
        rel = test.get("reliability", {})
        rows.append(
            {
                "run": label,
                "d_column": m.get("d_column", m.get("data", {}).get("d_column", "?")),
                "spearman": pri.get("spearman", test["regression"]["spearman"]),
                "rmse_log1p": pri.get("rmse_log1p", test["regression"].get("rmse_log1p")),
                "adj_acc": pri.get("adjacent_level_accuracy", test["level"]["adjacent_accuracy"]),
                "auroc": test["hard"]["auroc"],
                "reuse_at_hr95": r95.get("reuse_rate", float("nan")),
                "score_threshold": r95.get("score_threshold", test["hard"].get("score_threshold")),
                "calibration_monotonic": rel.get("monotonic_median", False),
            }
        )

    lines = [
        "# gate d 口径 / 模型选型对比 (test)",
        "",
        "| run | d_column | Spearman | rmse_log1p | adj acc | AUROC | reuse@HR95 | 校准单调 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['d_column']} | {r['spearman']:.4f} | {r['rmse_log1p']:.4f} | "
            f"{r['adj_acc']:.3f} | {r['auroc']:.4f} | {r['reuse_at_hr95']:.3f} | {r['calibration_monotonic']} |"
        )
    lines += [
        "",
        "主指标: **Spearman** + **reuse@HR95**(hard_recall≥95% 下可复用率) + **校准单调**。",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output.with_suffix(".json")).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
