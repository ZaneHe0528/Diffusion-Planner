#!/usr/bin/env python3
"""Convert adjacent trajectory L2 analysis output into d_labels.csv.

Input is the per-pair CSV produced by 0hzxcode/analyze_adjacent_traj_l2.py.
The d label uses perstep_max_m, matching the M-probe target:

    d = max_t || current_full[t, :2] - shifted_previous_full[t, :2] ||_2

The generated sample_id identifies the current/new planner frame in the
closed-loop log. It is not an exp/cache/mini .npz filename.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "0hzxcode" / "adjacent_traj_l2_output" / "per_pair_overlap_l2.csv"
DEFAULT_OUTPUT = REPO_ROOT / "0hzxcode" / "m_probe_output" / "d_labels.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--d-column",
        choices=("perstep_max_m", "normalized_l2_xy", "overall_l2_xy"),
        default="perstep_max_m",
    )
    parser.add_argument("--require-time-aligned", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def make_sample_id(row: dict[str, str]) -> str:
    return f"{row['log_name']}__{row['scenario_name']}__iter_{int(row['new_iteration']):06d}"


def make_scenario_id(row: dict[str, str]) -> str:
    return f"{row['log_name']}__{row['scenario_name']}"


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_read = 0
    rows_written = 0
    skipped_unaligned = 0

    with args.input.open(newline="", encoding="utf-8") as src, args.output.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        required = {
            "log_file",
            "scenario_type",
            "log_name",
            "scenario_name",
            "old_iteration",
            "new_iteration",
            "gap_steps",
            "overlap_len",
            "time_aligned",
            args.d_column,
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"input CSV missing required columns: {sorted(missing)}")

        fieldnames = [
            "sample_id",
            "d",
            "scenario_id",
            "source",
            "d_metric",
            "scenario_type",
            "log_name",
            "scenario_name",
            "old_iteration",
            "new_iteration",
            "gap_steps",
            "overlap_len",
            "time_aligned",
            "log_file",
        ]
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            rows_read += 1
            if args.require_time_aligned and row["time_aligned"].lower() != "true":
                skipped_unaligned += 1
                continue
            writer.writerow(
                {
                    "sample_id": make_sample_id(row),
                    "d": row[args.d_column],
                    "scenario_id": make_scenario_id(row),
                    "source": "analyze_adjacent_traj_l2",
                    "d_metric": args.d_column,
                    "scenario_type": row["scenario_type"],
                    "log_name": row["log_name"],
                    "scenario_name": row["scenario_name"],
                    "old_iteration": row["old_iteration"],
                    "new_iteration": row["new_iteration"],
                    "gap_steps": row["gap_steps"],
                    "overlap_len": row["overlap_len"],
                    "time_aligned": row["time_aligned"],
                    "log_file": row["log_file"],
                }
            )
            rows_written += 1
            if args.max_rows is not None and rows_written >= args.max_rows:
                break

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "d_column": args.d_column,
        "rows_read": rows_read,
        "rows_written": rows_written,
        "skipped_unaligned": skipped_unaligned,
        "sample_id_format": "<log_name>__<scenario_name>__iter_<new_iteration:06d>",
        "note": "sample_id identifies closed-loop log frames, not exp/cache/mini .npz filenames.",
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote: {args.output}")
    print(f"wrote: {summary_path}")
    print(f"rows_written: {rows_written}")


if __name__ == "__main__":
    main()
