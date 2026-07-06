#!/usr/bin/env python3
"""对已训 gate checkpoint 做决策导向评测（无需重训）。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_features import GROUP_BY_NAME
from gate_dataset import load_chunk_dir, scenario_split
from gate_metrics import choose_threshold_for_recall, full_decision_metrics
from gate_model import load_gate

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "0hzxcode" / "gate_output" / "gate_dataset_chunks",
    )
    p.add_argument(
        "--adjacent-csv",
        type=Path,
        default=REPO_ROOT / "0hzxcode" / "adjacent_traj_l2_output" / "per_pair_overlap_l2.csv",
    )
    p.add_argument("--d-column", type=str, default=None, help="覆盖 checkpoint 中记录的 d 口径")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    return p.parse_args()


@torch.no_grad()
def predict(model, arrays, idx, device, tensor_keys_list):
    model.eval()
    d_hat, level_pred = [], []
    for s in range(0, idx.shape[0], 1024):
        rows = idx[s : s + 1024]
        batch = {
            k: torch.from_numpy(np.ascontiguousarray(arrays[k][rows])).to(device)
            for k in tensor_keys_list
            if k in arrays
        }
        out = model(batch)
        d_hat.append(out["d_hat"].cpu().numpy())
        level_pred.append(out["level"].cpu().numpy())
    return {"d_hat": np.concatenate(d_hat), "level_pred": np.concatenate(level_pred)}


def write_report(results: dict, out_dir: Path, ckpt_name: str) -> None:
    lines = [
        f"# Gate 决策导向评测 — `{ckpt_name}`",
        "",
        "## 主指标（面向分档/复用决策）",
        "",
        "| split | Spearman | rmse_log1p | adj level acc | reuse@HR95 | reuse@HR99 | 校准单调 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, m in results["splits"].items():
        pri = m["primary"]
        r95 = m["reuse_at_hard_recall_95"]
        r99 = m["reuse_at_hard_recall_99"]
        mono = m["reliability"]["monotonic_median"]
        lines.append(
            f"| {split} | {pri['spearman']:.4f} | {pri['rmse_log1p']:.4f} | "
            f"{pri['adjacent_level_accuracy']:.3f} | {r95['reuse_rate']:.3f} | "
            f"{r99['reuse_rate']:.3f} | {mono} |"
        )

    lines += [
        "",
        "## 说明",
        "",
        "- **Spearman / adj level acc / reuse@HR95** 是主指标；Pearson/原始 RMSE 对重尾 d 不适用，见 JSON。",
        "- **reuse@HR95**：hard_recall≥95% 约束下最大可复用率。",
        "- **校准单调**：按 d_hat 分 10 箱，真实 d 中位数是否单调递增。",
        "",
    ]
    if results.get("per_scenario_type"):
        lines += ["## 按场景类型 Spearman (test)", ""]
        for t, m in sorted(results["per_scenario_type"].items(), key=lambda kv: -kv[1]["spearman"]):
            lines.append(f"- {t}: {m['spearman']:.3f} (n={m['n']})")
        lines.append("")

    (out_dir / "decision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_curves(results: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split = "test" if "test" in results["splits"] else next(iter(results["splits"]))
    m = results["splits"][split]
    curve = m["reuse_risk"]["curve"]
    reuse = [p["reuse_rate"] for p in curve]
    miss = [p["hard_miss_rate"] for p in curve]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(reuse, miss, "b-", lw=1.5)
    axes[0].set_xlabel("reuse_rate")
    axes[0].set_ylabel("hard_miss_rate")
    axes[0].set_title(f"reuse-risk ({split})")
    axes[0].grid(True, alpha=0.3)

    bins = m["reliability"]["bins"]
    xh = [b["d_hat_median"] for b in bins]
    yd = [b["d_median"] for b in bins]
    axes[1].plot(xh, yd, "o-", lw=1.5)
    axes[1].plot([min(xh), max(xh)], [min(xh), max(xh)], "k--", alpha=0.4)
    axes[1].set_xlabel("d_hat median (bin)")
    axes[1].set_ylabel("d median (bin)")
    axes[1].set_title(f"reliability ({split})")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "decision_curves.png", dpi=130)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    model, state = load_gate(str(args.checkpoint), device=args.device)
    ckpt_args = state.get("args", {})
    d_column = args.d_column or ckpt_args.get("d_column", state.get("d_column", "perstep_max_m"))
    tensor_keys_list = []
    for gname in model.group_order:
        tensor_keys_list.extend(GROUP_BY_NAME[gname].keys.keys())

    arrays = load_chunk_dir(
        args.dataset_dir,
        max_chunks=None,
        adjacent_csv=args.adjacent_csv,
        d_column=d_column,
    )

    hard_threshold = float(state.get("hard_threshold_m", model.hard_threshold_m.item()))
    level_edges = [float(x) for x in state.get("level_edges_m", model.level_edges_m.tolist())]
    level_all = np.digitize(arrays["d"], level_edges)

    masks = scenario_split(arrays["scenario_id"], args.val_ratio, args.test_ratio, args.seed)
    idx = {
        "train": np.nonzero(masks["train"])[0],
        "val": np.nonzero(masks["val"])[0],
        "test": np.nonzero(masks["test"])[0],
    }

    pred_val = predict(model, arrays, idx["val"], args.device, tensor_keys_list)
    score_threshold = float(
        state.get(
            "score_threshold_m",
            choose_threshold_for_recall(
                arrays["d"][idx["val"]] > hard_threshold,
                pred_val["d_hat"],
                0.95,
            ),
        )
    )

    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    results = {
        "checkpoint": str(args.checkpoint),
        "d_column": d_column,
        "hard_threshold_m": hard_threshold,
        "score_threshold_m": score_threshold,
        "level_edges_m": level_edges,
        "splits": {},
    }

    for split in splits:
        pred = predict(model, arrays, idx[split], args.device, tensor_keys_list)
        d = arrays["d"][idx[split]]
        st = arrays["scenario_type"][idx[split]] if split == "test" else None
        results["splits"][split] = full_decision_metrics(
            d,
            pred["d_hat"],
            level_all[idx[split]],
            pred["level_pred"],
            hard_threshold,
            score_threshold,
            scenario_type=st,
        )

    if "test" in results["splits"] and "per_scenario_type" in results["splits"]["test"]:
        results["per_scenario_type"] = results["splits"]["test"]["per_scenario_type"]

    (out_dir / "decision_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(results, out_dir, args.checkpoint.name)
    plot_curves(results, out_dir)
    print(f"wrote {out_dir}/decision_metrics.json, decision_report.md, decision_curves.png")


if __name__ == "__main__":
    main()
