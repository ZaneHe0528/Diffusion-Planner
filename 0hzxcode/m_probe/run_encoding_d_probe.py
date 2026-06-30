#!/usr/bin/env python3
"""Run the M-probe for the adaptive start-level route.

This script tests whether frozen planner encoding predicts the frame-change
distance d better than simple ego-motion features. It is intentionally small
and dependency-light: numpy for Ridge probes, torch for logistic probes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "0hzxcode" / "m_probe_output"


@dataclass(frozen=True)
class DatasetBundle:
    encoding: np.ndarray
    d: np.ndarray
    scenario_id: np.ndarray
    ego_features: np.ndarray | None
    sample_id: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Probe dataset .npz or .csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--ridge-alpha", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--target-transform", choices=("none", "log1p"), default="log1p")
    parser.add_argument("--hard-threshold", type=float, default=None, help="Absolute d threshold for hard/easy classification.")
    parser.add_argument("--hard-quantile", type=float, default=0.90, help="Train d quantile used when --hard-threshold is omitted.")
    parser.add_argument("--min-rmse-improvement", type=float, default=0.15)
    parser.add_argument("--min-spearman", type=float, default=0.30)
    parser.add_argument("--min-auroc", type=float, default=0.70)
    parser.add_argument("--min-hard-recall", type=float, default=0.85)
    parser.add_argument("--self-test", action="store_true", help="Run a synthetic sanity check.")
    return parser.parse_args()


def require_input(args: argparse.Namespace) -> Path:
    if args.input is None:
        raise SystemExit("--input is required unless --self-test is used")
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")
    return args.input


def as_1d_string_array(values: Any, n: int, default_prefix: str) -> np.ndarray:
    if values is None:
        return np.asarray([f"{default_prefix}_{i}" for i in range(n)], dtype=str)
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"expected 1D string-like array of length {n}, got {arr.shape}")
    return arr.astype(str)


def load_npz(path: Path) -> DatasetBundle:
    data = np.load(path, allow_pickle=True)
    encoding_key = first_existing_key(data, ["encoding", "x_encoding", "enc"])
    d_key = first_existing_key(data, ["d", "target", "y"])

    encoding = np.asarray(data[encoding_key], dtype=np.float64)
    if encoding.ndim == 3:
        encoding = encoding.mean(axis=1)
    if encoding.ndim != 2:
        raise ValueError(f"encoding must be [N, D] or [N, T, D], got {encoding.shape}")

    d = np.asarray(data[d_key], dtype=np.float64).reshape(-1)
    if d.shape[0] != encoding.shape[0]:
        raise ValueError(f"d length {d.shape[0]} does not match encoding rows {encoding.shape[0]}")

    n = d.shape[0]
    scenario_id = as_1d_string_array(data["scenario_id"] if "scenario_id" in data else None, n, "row")
    sample_id = as_1d_string_array(data["sample_id"] if "sample_id" in data else None, n, "sample")

    ego_features = None
    if "ego_features" in data:
        ego_features = np.asarray(data["ego_features"], dtype=np.float64)
        if ego_features.ndim != 2 or ego_features.shape[0] != n:
            raise ValueError(f"ego_features must be [N, K], got {ego_features.shape}")

    return DatasetBundle(encoding=encoding, d=d, scenario_id=scenario_id, ego_features=ego_features, sample_id=sample_id)


def first_existing_key(data: Any, candidates: list[str]) -> str:
    for key in candidates:
        if key in data:
            return key
    raise ValueError(f"missing required key; tried {candidates}")


def load_csv(path: Path) -> DatasetBundle:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty csv: {path}")

    headers = rows[0].keys()
    enc_cols = sorted([h for h in headers if h.startswith("enc_")])
    ego_cols = sorted([h for h in headers if h.startswith("ego_")])
    if "d" not in headers:
        raise ValueError("CSV must contain column d")
    if not enc_cols:
        raise ValueError("CSV must contain encoding columns with prefix enc_")

    encoding = np.asarray([[float(r[c]) for c in enc_cols] for r in rows], dtype=np.float64)
    d = np.asarray([float(r["d"]) for r in rows], dtype=np.float64)
    scenario_id = np.asarray([r.get("scenario_id") or f"row_{i}" for i, r in enumerate(rows)], dtype=str)
    sample_id = np.asarray([r.get("sample_id") or f"sample_{i}" for i, r in enumerate(rows)], dtype=str)
    ego_features = None
    if ego_cols:
        ego_features = np.asarray([[float(r[c]) for c in ego_cols] for r in rows], dtype=np.float64)

    return DatasetBundle(encoding=encoding, d=d, scenario_id=scenario_id, ego_features=ego_features, sample_id=sample_id)


def load_dataset(path: Path) -> DatasetBundle:
    if path.suffix == ".npz":
        ds = load_npz(path)
    elif path.suffix == ".csv":
        ds = load_csv(path)
    else:
        raise ValueError(f"unsupported input suffix: {path.suffix}")
    keep = np.isfinite(ds.d) & np.all(np.isfinite(ds.encoding), axis=1)
    if ds.ego_features is not None:
        keep &= np.all(np.isfinite(ds.ego_features), axis=1)
    if keep.sum() != ds.d.shape[0]:
        print(f"[warn] dropped {ds.d.shape[0] - keep.sum()} rows with non-finite values", file=sys.stderr)
    if keep.sum() == 0:
        raise SystemExit(
            "no valid labeled rows remain after filtering non-finite values. "
            "Check that the probe dataset contains a finite d label for each sample."
        )
    ego = ds.ego_features[keep] if ds.ego_features is not None else None
    return DatasetBundle(ds.encoding[keep], ds.d[keep], ds.scenario_id[keep], ego, ds.sample_id[keep])


def scenario_split(scenario_id: np.ndarray, val_ratio: float, test_ratio: float, seed: int) -> dict[str, np.ndarray]:
    scenarios = sorted(set(scenario_id.tolist()))
    rng = random.Random(seed)
    rng.shuffle(scenarios)
    n = len(scenarios)
    n_test = max(1, int(round(n * test_ratio))) if n >= 3 else 1
    n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 1
    test_set = set(scenarios[:n_test])
    val_set = set(scenarios[n_test : n_test + n_val])
    train_set = set(scenarios[n_test + n_val :])
    if not train_set:
        train_set = set(scenarios) - test_set
    return {
        "train": np.asarray([s in train_set for s in scenario_id]),
        "val": np.asarray([s in val_set for s in scenario_id]),
        "test": np.asarray([s in test_set for s in scenario_id]),
    }


def standardize_from_train(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_mask].mean(axis=0, keepdims=True)
    std = x[train_mask].std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (x - mean) / std, mean.reshape(-1), std.reshape(-1)


def transform_target(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y
    return np.log1p(np.maximum(y, 0.0))


def inverse_target(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y
    return np.expm1(y)


def fit_ridge(x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray, alphas: list[float]) -> dict[str, Any]:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    best: dict[str, Any] | None = None
    for alpha in alphas:
        xtx = x_aug[train].T @ x_aug[train]
        reg = np.eye(xtx.shape[0], dtype=x.dtype) * alpha
        reg[-1, -1] = 0.0
        beta = np.linalg.solve(xtx + reg, x_aug[train].T @ y[train])
        pred = x_aug @ beta
        val_rmse = rmse(y[val], pred[val])
        if best is None or val_rmse < best["val_rmse"]:
            best = {"alpha": alpha, "beta": beta, "pred": pred, "val_rmse": val_rmse}
    assert best is not None
    return best


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def pearson(y: np.ndarray, pred: np.ndarray) -> float:
    if np.std(y) < 1e-12 or np.std(pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(y, pred)[0, 1])


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


def spearman(y: np.ndarray, pred: np.ndarray) -> float:
    return pearson(rankdata(y), rankdata(pred))


def regression_metrics(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y[mask], pred[mask]),
        "mae": mae(y[mask], pred[mask]),
        "pearson": pearson(y[mask], pred[mask]),
        "spearman": spearman(y[mask], pred[mask]),
    }


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(labels.shape[0]) + 1)
    return float((precision * sorted_labels).sum() / n_pos)


def fit_logistic(x: np.ndarray, labels: np.ndarray, train: np.ndarray, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    x_train = torch.as_tensor(x[train], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(labels[train].astype(np.float32), dtype=torch.float32, device=device)
    model = torch.nn.Linear(x.shape[1], 1)
    pos = float(y_train.sum().item())
    neg = float(y_train.numel() - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-2, weight_decay=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for _ in range(600):
        opt.zero_grad()
        logits = model(x_train).squeeze(-1)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits_all = model(torch.as_tensor(x, dtype=torch.float32, device=device)).squeeze(-1)
        return torch.sigmoid(logits_all).cpu().numpy()


def choose_threshold_for_recall(labels: np.ndarray, scores: np.ndarray, mask: np.ndarray, target_recall: float) -> float:
    y = labels[mask].astype(bool)
    s = scores[mask]
    if y.sum() == 0:
        return 0.5
    thresholds = np.unique(s)
    best = float(thresholds.min())
    for thr in sorted(thresholds, reverse=True):
        pred = s >= thr
        recall = float((pred & y).sum() / max(y.sum(), 1))
        if recall >= target_recall:
            best = float(thr)
            break
    return best


def classification_metrics(labels: np.ndarray, scores: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, float]:
    y = labels[mask].astype(bool)
    s = scores[mask]
    pred = s >= threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())
    return {
        "auroc": auroc(y, s),
        "average_precision": average_precision(y, s),
        "threshold": float(threshold),
        "hard_recall": float(tp / max(tp + fn, 1)),
        "hard_precision": float(tp / max(tp + fp, 1)),
        "false_easy_rate": float(fn / max(tp + fn, 1)),
        "easy_recall": float(tn / max(tn + fp, 1)),
    }


def evaluate_feature_set(
    name: str,
    x: np.ndarray,
    y_raw: np.ndarray,
    hard_labels: np.ndarray,
    masks: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    x_std, _, _ = standardize_from_train(x, masks["train"])
    y_model = transform_target(y_raw, args.target_transform)
    y_std, y_mean, y_scale = standardize_target(y_model, masks["train"])
    ridge = fit_ridge(x_std, y_std, masks["train"], masks["val"], args.ridge_alpha)
    pred_model = ridge["pred"] * y_scale + y_mean
    pred_raw = inverse_target(pred_model, args.target_transform)
    pred_raw = np.maximum(pred_raw, 0.0)

    log_scores = fit_logistic(x_std, hard_labels, masks["train"], args.seed)
    calibration_recall = max(args.min_hard_recall, 0.95)
    threshold = choose_threshold_for_recall(hard_labels, log_scores, masks["val"], calibration_recall)

    return {
        "name": name,
        "ridge_alpha": ridge["alpha"],
        "regression": {split: regression_metrics(y_raw, pred_raw, mask) for split, mask in masks.items()},
        "classification": {split: classification_metrics(hard_labels, log_scores, mask, threshold) for split, mask in masks.items()},
    }


def standardize_target(y: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(y[train_mask].mean())
    std = float(y[train_mask].std())
    if std < 1e-8:
        std = 1.0
    return (y - mean) / std, mean, std


def build_synthetic_dataset(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    n_scenarios = 120
    frames = 20
    n = n_scenarios * frames
    scenario_id = np.repeat([f"s{i:04d}" for i in range(n_scenarios)], frames)
    latent = rng.normal(size=(n, 3))
    d = np.exp(1.0 * latent[:, 0] - 0.35 * latent[:, 1] + 0.03 * rng.normal(size=n))
    encoding = rng.normal(scale=0.25, size=(n, 64))
    encoding[:, 0:3] += 3.0 * latent
    ego = rng.normal(size=(n, 5))
    ego[:, 0] += 0.20 * latent[:, 0]
    sample_id = np.asarray([f"sample_{i}" for i in range(n)], dtype=str)
    return DatasetBundle(encoding=encoding, d=d, scenario_id=scenario_id, ego_features=ego, sample_id=sample_id)


def run(ds: DatasetBundle, args: argparse.Namespace) -> dict[str, Any]:
    if ds.d.shape[0] < 30:
        raise ValueError("need at least 30 rows for a meaningful probe")
    masks = scenario_split(ds.scenario_id, args.val_ratio, args.test_ratio, args.seed)
    if masks["train"].sum() == 0 or masks["val"].sum() == 0 or masks["test"].sum() == 0:
        raise ValueError("empty train/val/test split; check scenario_id")

    hard_threshold = args.hard_threshold
    if hard_threshold is None:
        hard_threshold = float(np.quantile(ds.d[masks["train"]], args.hard_quantile))
    hard_labels = ds.d > hard_threshold

    results: dict[str, Any] = {
        "data": {
            "num_rows": int(ds.d.shape[0]),
            "encoding_dim": int(ds.encoding.shape[1]),
            "ego_feature_dim": int(ds.ego_features.shape[1]) if ds.ego_features is not None else None,
            "num_scenarios": int(len(set(ds.scenario_id.tolist()))),
            "split_rows": {k: int(v.sum()) for k, v in masks.items()},
            "target_d": summarize(ds.d),
            "hard_threshold": hard_threshold,
            "hard_positive_rate": float(hard_labels.mean()),
        },
        "feature_sets": {},
    }

    results["feature_sets"]["encoding"] = evaluate_feature_set("encoding", ds.encoding, ds.d, hard_labels, masks, args)
    if ds.ego_features is not None:
        results["feature_sets"]["ego_features"] = evaluate_feature_set("ego_features", ds.ego_features, ds.d, hard_labels, masks, args)

    results["decision"] = make_decision(results, args)
    return results


def summarize(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


def make_decision(results: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    enc = results["feature_sets"]["encoding"]
    enc_reg = enc["regression"]["test"]
    enc_cls = enc["classification"]["test"]

    checks: dict[str, Any] = {
        "spearman_ok": enc_reg["spearman"] >= args.min_spearman,
        "auroc_ok": enc_cls["auroc"] >= args.min_auroc,
        "hard_recall_ok": enc_cls["hard_recall"] >= args.min_hard_recall,
    }
    if "ego_features" in results["feature_sets"]:
        ego_reg = results["feature_sets"]["ego_features"]["regression"]["test"]
        improvement = 1.0 - enc_reg["rmse"] / max(ego_reg["rmse"], 1e-12)
        checks["rmse_improvement_vs_ego"] = float(improvement)
        checks["rmse_improvement_ok"] = improvement >= args.min_rmse_improvement
    else:
        checks["rmse_improvement_vs_ego"] = None
        checks["rmse_improvement_ok"] = False

    pass_probe = bool(
        checks["spearman_ok"]
        and checks["auroc_ok"]
        and checks["hard_recall_ok"]
        and checks["rmse_improvement_ok"]
    )
    return {
        "pass": pass_probe,
        "recommendation": "GO: train the learned gate" if pass_probe else "STOP: keep fixed t_s warm-start; do not train learned gate yet",
        "checks": checks,
        "thresholds": {
            "min_rmse_improvement": args.min_rmse_improvement,
            "min_spearman": args.min_spearman,
            "min_auroc": args.min_auroc,
            "min_hard_recall": args.min_hard_recall,
        },
    }


def write_report(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "probe_results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    lines = [
        "# M-probe report",
        "",
        f"Decision: **{'PASS' if results['decision']['pass'] else 'FAIL'}**",
        "",
        results["decision"]["recommendation"],
        "",
        "## Data",
        "",
        f"- rows: {results['data']['num_rows']}",
        f"- scenarios: {results['data']['num_scenarios']}",
        f"- split rows: {results['data']['split_rows']}",
        f"- hard threshold d: {results['data']['hard_threshold']:.6g}",
        f"- hard positive rate: {results['data']['hard_positive_rate']:.4f}",
        "",
        "## Test metrics",
        "",
        "| feature | RMSE | MAE | Spearman | AUROC | AP | hard recall | false easy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in results["feature_sets"].items():
        reg = item["regression"]["test"]
        cls = item["classification"]["test"]
        lines.append(
            f"| {key} | {reg['rmse']:.6g} | {reg['mae']:.6g} | {reg['spearman']:.4f} | "
            f"{cls['auroc']:.4f} | {cls['average_precision']:.4f} | {cls['hard_recall']:.4f} | {cls['false_easy_rate']:.4f} |"
        )
    lines.extend(["", "## Checks", ""])
    for key, value in results["decision"]["checks"].items():
        lines.append(f"- {key}: {value}")
    (output_dir / "probe_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.self_test:
        ds = build_synthetic_dataset(args.seed)
    else:
        ds = load_dataset(require_input(args))
    results = run(ds, args)
    write_report(results, args.output_dir)
    print(json.dumps(results["decision"], indent=2, ensure_ascii=False))
    print(f"wrote: {args.output_dir / 'probe_results.json'}")
    print(f"wrote: {args.output_dir / 'probe_report.md'}")


if __name__ == "__main__":
    main()
