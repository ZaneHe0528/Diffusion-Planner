#!/usr/bin/env python3
"""训练 gate_v2：ego + 邻车，仅回归 d_hat。"""

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
from gate_features import FEATURE_GROUPS, GROUP_BY_NAME, resolve_enabled_groups
from gate_model import GateConfig, LiteGate

REPO_ROOT = Path(__file__).resolve().parents[2]
TENSOR_KEYS = [key for g in FEATURE_GROUPS for key in g.keys]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "0hzxcode" / "gate_output" / "gate_dataset_chunks",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--enable-groups", nargs="+", default=None)
    parser.add_argument("--disable-groups", nargs="+", default=None)
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
        "--level-edges-m",
        type=float,
        nargs="+",
        default=[0.275, 0.696, 1.385],
        help="4 档边界（米），默认全量 normalized-L2 的 P50/P75/P90",
    )
    parser.add_argument("--no-redundancy-downsample", action="store_true")
    parser.add_argument("--skip-importance", action="store_true")
    return parser.parse_args()


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.shape[0], dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < a.shape[0]:
        j = i + 1
        while j < a.shape[0] and sorted_a[j] == sorted_a[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def pearson(y: np.ndarray, pred: np.ndarray) -> float:
    if np.std(y) < 1e-12 or np.std(pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(y, pred)[0, 1])


def spearman(y: np.ndarray, pred: np.ndarray) -> float:
    return pearson(rankdata(y), rankdata(pred))


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def choose_threshold_for_recall(labels: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    y = labels.astype(bool)
    if y.sum() == 0:
        return 0.5
    pos_scores = np.sort(scores[y])
    k = int(np.floor((1.0 - target_recall) * pos_scores.shape[0]))
    return float(pos_scores[min(k, pos_scores.shape[0] - 1)])


def iter_batches(
    arrays: dict[str, np.ndarray],
    idx: np.ndarray,
    batch_size: int,
    device: str,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
    permute_keys: dict[str, np.ndarray] | None = None,
):
    order = idx.copy()
    if shuffle:
        assert rng is not None
        rng.shuffle(order)
    keys = [k for k in TENSOR_KEYS if k in arrays]
    for s in range(0, order.shape[0], batch_size):
        rows = order[s : s + batch_size]
        batch: dict[str, torch.Tensor] = {}
        for k in keys:
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
    batch_size: int = 1024,
    disabled_groups: set[str] | None = None,
    permute_keys: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    d_hat, levels_pred, z = [], [], []
    for batch in iter_batches(arrays, idx, batch_size, device, permute_keys=permute_keys):
        out = model(batch, disabled_groups=disabled_groups)
        d_hat.append(out["d_hat"].cpu().numpy())
        levels_pred.append(out["level"].cpu().numpy())
        z.append(out["z"].cpu().numpy())
    return {
        "d_hat": np.concatenate(d_hat),
        "level_pred": np.concatenate(levels_pred),
        "z": np.concatenate(z),
    }


def regression_metrics(d: np.ndarray, d_hat: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean((d - d_hat) ** 2))),
        "mae": float(np.mean(np.abs(d - d_hat))),
        "pearson": pearson(d, d_hat),
        "spearman": spearman(d, d_hat),
    }


def level_metrics(level: np.ndarray, level_pred: np.ndarray, num_levels: int) -> dict:
    acc = float((level == level_pred).mean())
    adj = float((np.abs(level - level_pred) <= 1).mean())
    return {"accuracy": acc, "adjacent_accuracy": adj}


def hard_metrics(d: np.ndarray, scores: np.ndarray, hard_threshold: float, score_threshold: float) -> dict[str, float]:
    y = d > hard_threshold
    pred = scores >= score_threshold
    tp = int((pred & y).sum())
    fn = int((~pred & y).sum())
    fp = int((pred & ~y).sum())
    return {
        "auroc": auroc(y, scores),
        "hard_recall": float(tp / max(tp + fn, 1)),
        "hard_precision": float(tp / max(tp + fp, 1)),
        "false_easy_rate": float(fn / max(tp + fn, 1)),
        "score_threshold": float(score_threshold),
    }


def full_metrics(
    arrays: dict[str, np.ndarray],
    idx: np.ndarray,
    pred: dict[str, np.ndarray],
    level: np.ndarray,
    num_levels: int,
    hard_threshold: float,
    score_threshold: float,
) -> dict:
    d = arrays["d"][idx]
    return {
        "regression": regression_metrics(d, pred["d_hat"]),
        "level": level_metrics(level[idx], pred["level_pred"], num_levels),
        "hard": hard_metrics(d, pred["d_hat"], hard_threshold, score_threshold),
    }


def main() -> None:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    arrays = load_chunk_dir(args.dataset_dir, args.max_chunks)
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
    level_edges = [float(x) for x in args.level_edges_m]
    num_levels = len(level_edges) + 1
    level_all = np.digitize(arrays["d"], level_edges)
    print(f"hard_threshold={hard_threshold:.3f} m, level_edges={level_edges}")

    enabled = resolve_enabled_groups(args.enable_groups, args.disable_groups)
    cfg = GateConfig(enabled_groups=enabled, num_levels=num_levels, dropout=args.dropout, group_dropout=args.group_dropout)
    device = args.device
    model = LiteGate(cfg).to(device)
    print(f"enabled groups: {model.group_order}, params: {model.count_parameters()/1e6:.3f} M")

    model.fit_normalization(
        iter_batches(arrays, idx["train"], 2048, device),
        level_edges,
        d_train,
        hard_threshold,
        hard_threshold,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = int(np.ceil(idx["train"].shape[0] / args.batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch, eta_min=1e-5)

    history = []
    best = {"val_spearman": -2.0}
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in iter_batches(arrays, idx["train"], args.batch_size, device, shuffle=True, rng=rng):
            out = model(batch)
            z_true = model.d_to_z(batch["d"])
            loss = F.huber_loss(out["z"], z_true, delta=1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))

        pred_val = predict_split(model, arrays, idx["val"], device)
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
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "level_edges_m": level_edges,
                    "hard_threshold_m": hard_threshold,
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                },
                out_dir / "best.pt",
            )

    state = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    print(f"best epoch = {best['epoch']}, val_spearman = {best['val_spearman']:.4f}")

    pred_val = predict_split(model, arrays, idx["val"], device)
    score_threshold = choose_threshold_for_recall(
        arrays["d"][idx["val"]] > hard_threshold, pred_val["d_hat"], target_recall=0.95
    )
    model.score_threshold_m.fill_(score_threshold)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "level_edges_m": level_edges,
            "hard_threshold_m": hard_threshold,
            "score_threshold_m": score_threshold,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        out_dir / "best.pt",
    )

    results = {
        "data": {
            "rows": int(arrays["d"].shape[0]),
            "enabled_groups": model.group_order,
            "parameters": model.count_parameters(),
            "hard_threshold_m": hard_threshold,
            "score_threshold_m": score_threshold,
            "level_edges_m": level_edges,
        },
        "splits": {},
    }
    for split in ("train", "val", "test"):
        pred = predict_split(model, arrays, idx[split], device)
        results["splits"][split] = full_metrics(
            arrays, idx[split], pred, level_all, num_levels, hard_threshold, score_threshold
        )

    if not args.skip_importance:
        base_sp = results["splits"]["test"]["regression"]["spearman"]
        importance = []
        test_idx_sorted = np.sort(idx["test"])
        for name in model.group_order:
            pred_loo = predict_split(model, arrays, idx["test"], device, disabled_groups={name})
            loo_sp = spearman(arrays["d"][idx["test"]], pred_loo["d_hat"])
            others = set(model.group_order) - {name}
            pred_solo = predict_split(model, arrays, idx["test"], device, disabled_groups=others)
            solo_sp = spearman(arrays["d"][idx["test"]], pred_solo["d_hat"])
            keys = list(GROUP_BY_NAME[name].keys)
            perm = rng.permutation(test_idx_sorted)
            pred_perm = predict_split(
                model, arrays, test_idx_sorted, device, permute_keys={k: perm for k in keys}
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
    print(f"wrote {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
