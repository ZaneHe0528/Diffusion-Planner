#!/usr/bin/env python3
"""训练 gate_v2：ego + 邻车 (+ 可选 prev_d)，决策导向评测。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_dataset import load_chunk_dir, redundancy_downsample_mask, scenario_split
from gate_features import D_COLUMN_CHOICES, FEATURE_GROUPS, GROUP_BY_NAME, resolve_enabled_groups
from gate_metrics import (
    auroc,
    choose_threshold_for_recall,
    compute_level_edges,
    full_decision_metrics,
    spearman,
)
from gate_model import GateConfig, LiteGate

REPO_ROOT = Path(__file__).resolve().parents[2]


def tensor_keys(enabled_groups: list[str]) -> list[str]:
    keys = []
    for g in FEATURE_GROUPS:
        if g.name in enabled_groups:
            keys.extend(g.keys)
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "0hzxcode" / "gate_output" / "gate_dataset_chunks",
    )
    parser.add_argument(
        "--adjacent-csv",
        type=Path,
        default=REPO_ROOT / "0hzxcode" / "adjacent_traj_l2_output" / "per_pair_overlap_l2.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--enable-groups", nargs="+", default=None)
    parser.add_argument("--disable-groups", nargs="+", default=None)
    parser.add_argument("--d-column", choices=D_COLUMN_CHOICES, default="perstep_max_m")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--group-dropout", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--hard-quantile", type=float, default=0.90)
    parser.add_argument(
        "--level-quantiles",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 0.9],
        help="档位边界 = train 集 d 的这些分位数（与 d-column 同口径）",
    )
    parser.add_argument("--level-edges-m", type=float, nargs="+", default=None, help="手动指定边界，覆盖分位数")
    parser.add_argument("--no-redundancy-downsample", action="store_true")
    parser.add_argument("--skip-importance", action="store_true")
    return parser.parse_args()


def iter_batches(
    arrays: dict[str, np.ndarray],
    idx: np.ndarray,
    batch_size: int,
    device: str,
    tensor_keys_list: list[str],
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
    permute_keys: dict[str, np.ndarray] | None = None,
):
    order = idx.copy()
    if shuffle:
        assert rng is not None
        rng.shuffle(order)
    for s in range(0, order.shape[0], batch_size):
        rows = order[s : s + batch_size]
        batch: dict[str, torch.Tensor] = {}
        for k in tensor_keys_list:
            if k not in arrays:
                continue
            src_rows = rows
            if permute_keys is not None and k in permute_keys:
                pos = np.searchsorted(idx, rows)
                src_rows = permute_keys[k][pos]
            batch[k] = torch.from_numpy(np.ascontiguousarray(arrays[k][src_rows])).to(device)
        batch["d"] = torch.from_numpy(arrays["d"][rows]).to(device)
        yield batch


@torch.no_grad()
def predict_split(
    model: LiteGate,
    arrays: dict[str, np.ndarray],
    idx: np.ndarray,
    device: str,
    tensor_keys_list: list[str],
    batch_size: int = 1024,
    disabled_groups: set[str] | None = None,
    permute_keys: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    d_hat, levels_pred = [], []
    for batch in iter_batches(
        arrays, idx, batch_size, device, tensor_keys_list, permute_keys=permute_keys
    ):
        out = model(batch, disabled_groups=disabled_groups)
        d_hat.append(out["d_hat"].cpu().numpy())
        levels_pred.append(out["level"].cpu().numpy())
    return {"d_hat": np.concatenate(d_hat), "level_pred": np.concatenate(levels_pred)}


def save_checkpoint(path, model, cfg, level_edges, hard_threshold, score_threshold, args) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "level_edges_m": level_edges,
            "hard_threshold_m": hard_threshold,
            "score_threshold_m": score_threshold,
            "d_column": args.d_column,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        path,
    )


def main() -> None:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    arrays = load_chunk_dir(
        args.dataset_dir,
        args.max_chunks,
        adjacent_csv=args.adjacent_csv,
        d_column=args.d_column,
    )
    masks = scenario_split(arrays["scenario_id"], args.val_ratio, args.test_ratio, args.seed)
    train_mask = masks["train"]
    if not args.no_redundancy_downsample:
        before = int(train_mask.sum())
        train_mask = redundancy_downsample_mask(arrays, train_mask)
        print(f"redundancy downsample: train {before} -> {int(train_mask.sum())}")
    idx = {
        "train": np.nonzero(train_mask)[0],
        "val": np.nonzero(masks["val"])[0],
        "test": np.nonzero(masks["test"])[0],
    }
    d_train = arrays["d"][idx["train"]]

    hard_threshold = float(np.quantile(d_train, args.hard_quantile))
    if args.level_edges_m is not None:
        level_edges = [float(x) for x in args.level_edges_m]
    else:
        level_edges = compute_level_edges(d_train, tuple(args.level_quantiles))
    num_levels = len(level_edges) + 1
    level_all = np.digitize(arrays["d"], level_edges)
    print(
        f"d_column={args.d_column}, hard_threshold(P{args.hard_quantile*100:.0f})={hard_threshold:.4f}, "
        f"level_edges={[round(e, 4) for e in level_edges]}"
    )

    enabled = resolve_enabled_groups(args.enable_groups, args.disable_groups)
    keys = tensor_keys(enabled)
    cfg = GateConfig(
        enabled_groups=enabled,
        num_levels=num_levels,
        dropout=args.dropout,
        group_dropout=args.group_dropout,
    )
    device = args.device
    model = LiteGate(cfg).to(device)
    print(f"enabled groups: {model.group_order}, params: {model.count_parameters()/1e6:.3f} M")

    model.fit_normalization(
        iter_batches(arrays, idx["train"], 2048, device, keys),
        level_edges,
        d_train,
        hard_threshold,
        hard_threshold,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = int(np.ceil(idx["train"].shape[0] / args.batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * steps_per_epoch, eta_min=1e-5
    )

    history = []
    best = {"val_spearman": -2.0}
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in iter_batches(
            arrays, idx["train"], args.batch_size, device, keys, shuffle=True, rng=rng
        ):
            out = model(batch)
            z_true = model.d_to_z(batch["d"])
            loss = F.huber_loss(out["z"], z_true, delta=1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))

        pred_val = predict_split(model, arrays, idx["val"], device, keys)
        val_sp = spearman(arrays["d"][idx["val"]], pred_val["d_hat"])
        val_auroc = auroc(arrays["d"][idx["val"]] > hard_threshold, pred_val["d_hat"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_spearman": val_sp,
            "val_auroc": val_auroc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"epoch {epoch:3d} loss={row['train_loss']:.4f} val_spearman={val_sp:.4f} "
            f"val_auroc={val_auroc:.4f} ({(time.time()-t0)/60:.1f} min)",
            flush=True,
        )

        if val_sp > best["val_spearman"]:
            best = {"val_spearman": val_sp, "epoch": epoch}
            save_checkpoint(out_dir / "best.pt", model, cfg, level_edges, hard_threshold, hard_threshold, args)

    state = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    print(f"best epoch = {best['epoch']}, val_spearman = {best['val_spearman']:.4f}")

    pred_val = predict_split(model, arrays, idx["val"], device, keys)
    score_threshold = choose_threshold_for_recall(
        arrays["d"][idx["val"]] > hard_threshold, pred_val["d_hat"], target_recall=0.95
    )
    model.score_threshold_m.fill_(score_threshold)
    save_checkpoint(out_dir / "best.pt", model, cfg, level_edges, hard_threshold, score_threshold, args)

    results = {
        "data": {
            "rows": int(arrays["d"].shape[0]),
            "d_column": args.d_column,
            "enabled_groups": model.group_order,
            "parameters": model.count_parameters(),
            "hard_threshold_m": hard_threshold,
            "score_threshold_m": score_threshold,
            "level_edges_m": level_edges,
        },
        "splits": {},
    }
    for split in ("train", "val", "test"):
        pred = predict_split(model, arrays, idx[split], device, keys)
        st = arrays["scenario_type"][idx[split]] if split == "test" else None
        results["splits"][split] = full_decision_metrics(
            arrays["d"][idx[split]],
            pred["d_hat"],
            level_all[idx[split]],
            pred["level_pred"],
            hard_threshold,
            score_threshold,
            scenario_type=st,
        )

    if not args.skip_importance:
        base_sp = results["splits"]["test"]["primary"]["spearman"]
        importance = []
        test_idx_sorted = np.sort(idx["test"])
        for name in model.group_order:
            pred_loo = predict_split(model, arrays, idx["test"], device, keys, disabled_groups={name})
            loo_sp = spearman(arrays["d"][idx["test"]], pred_loo["d_hat"])
            others = set(model.group_order) - {name}
            pred_solo = predict_split(model, arrays, idx["test"], device, keys, disabled_groups=others)
            solo_sp = spearman(arrays["d"][idx["test"]], pred_solo["d_hat"])
            gkeys = list(GROUP_BY_NAME[name].keys)
            perm = rng.permutation(test_idx_sorted)
            pred_perm = predict_split(
                model,
                arrays,
                test_idx_sorted,
                device,
                keys,
                permute_keys={k: perm for k in gkeys if k in arrays},
            )
            perm_sp = spearman(arrays["d"][test_idx_sorted], pred_perm["d_hat"])
            importance.append(
                {
                    "group": name,
                    "loo_delta_spearman": base_sp - loo_sp,
                    "solo_spearman": solo_sp,
                    "perm_delta_spearman": base_sp - perm_sp,
                }
            )
        importance.sort(key=lambda r: r["loo_delta_spearman"], reverse=True)
        results["importance"] = importance

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    # 决策报告
    from evaluate_gate import plot_curves, write_report

    eval_results = {
        "checkpoint": str(out_dir / "best.pt"),
        "d_column": args.d_column,
        "hard_threshold_m": hard_threshold,
        "score_threshold_m": score_threshold,
        "level_edges_m": level_edges,
        "splits": results["splits"],
    }
    if "test" in results["splits"] and "per_scenario_type" in results["splits"]["test"]:
        eval_results["per_scenario_type"] = results["splits"]["test"]["per_scenario_type"]
    (out_dir / "decision_metrics.json").write_text(
        json.dumps(eval_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(eval_results, out_dir, "best.pt")
    plot_curves(eval_results, out_dir)
    print(f"wrote {out_dir}/metrics.json, decision_metrics.json, decision_report.md")


if __name__ == "__main__":
    main()
