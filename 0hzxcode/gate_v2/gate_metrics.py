#!/usr/bin/env python3
"""gate 决策导向评测指标：排序、log 回归、分箱校准、复用收益-风险曲线。"""

from __future__ import annotations

import numpy as np


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


def regression_metrics(d: np.ndarray, d_hat: np.ndarray) -> dict[str, float]:
    d_hat_c = np.clip(d_hat, 0.0, None)
    return {
        "spearman": spearman(d, d_hat_c),
        "pearson": pearson(d, d_hat_c),
        "rmse": float(np.sqrt(np.mean((d - d_hat_c) ** 2))),
        "mae": float(np.mean(np.abs(d - d_hat_c))),
        "rmse_log1p": float(np.sqrt(np.mean((np.log1p(d) - np.log1p(d_hat_c)) ** 2))),
        "mae_log1p": float(np.mean(np.abs(np.log1p(d) - np.log1p(d_hat_c)))),
    }


def level_metrics(level: np.ndarray, level_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float((level == level_pred).mean()),
        "adjacent_accuracy": float((np.abs(level - level_pred) <= 1).mean()),
    }


def hard_metrics(
    d: np.ndarray,
    scores: np.ndarray,
    hard_threshold: float,
    score_threshold: float,
) -> dict[str, float]:
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


def reliability_curve(
    d: np.ndarray,
    d_hat: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """按 d_hat 分箱，输出每箱真实 d 统计，用于检验单调性。"""
    order = np.argsort(d_hat)
    d_hat_s = d_hat[order]
    d_s = d[order]
    n = d.shape[0]
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    bins = []
    medians = []
    for i in range(n_bins):
        sl = slice(edges[i], edges[i + 1])
        if sl.stop <= sl.start:
            continue
        chunk_d = d_s[sl]
        chunk_hat = d_hat_s[sl]
        bins.append(
            {
                "bin": i,
                "count": int(chunk_d.shape[0]),
                "d_hat_median": float(np.median(chunk_hat)),
                "d_median": float(np.median(chunk_d)),
                "d_mean": float(np.mean(chunk_d)),
                "d_p90": float(np.quantile(chunk_d, 0.9)),
            }
        )
        medians.append(float(np.median(chunk_d)))

    monotonic = True
    for i in range(1, len(medians)):
        if medians[i] < medians[i - 1] - 1e-9:
            monotonic = False
            break

    return {"bins": bins, "monotonic_median": monotonic, "n_bins": len(bins)}


def reuse_risk_curve(
    d: np.ndarray,
    d_hat: np.ndarray,
    hard_threshold: float,
    n_thresholds: int = 50,
) -> dict:
    """扫描 score_threshold：复用率 vs hard 漏复用率(false_easy_rate)。"""
    y = d > hard_threshold
    lo = float(np.quantile(d_hat, 0.01))
    hi = float(np.quantile(d_hat, 0.99))
    thresholds = np.linspace(lo, hi, n_thresholds)
    points = []
    for thr in thresholds:
        reuse = d_hat < thr
        fn = int((y & reuse).sum())
        hard_miss = fn
        recall = 1.0 - fn / max(int(y.sum()), 1)
        points.append(
            {
                "score_threshold": float(thr),
                "reuse_rate": float(reuse.mean()),
                "hard_miss_rate": float(hard_miss / max(y.sum(), 1)),
                "hard_recall": float(recall),
            }
        )
    return {"curve": points, "hard_threshold": float(hard_threshold)}


def reuse_rate_at_hard_recall(
    d: np.ndarray,
    d_hat: np.ndarray,
    hard_threshold: float,
    target_recall: float = 0.95,
) -> dict:
    """在 hard_recall >= target 约束下，最大可复用率及对应阈值。"""
    y = d > hard_threshold
    if y.sum() == 0:
        return {"reuse_rate": 1.0, "score_threshold": float("inf"), "hard_recall": 1.0}

    order = np.argsort(d_hat)
    best = {"reuse_rate": 0.0, "score_threshold": float(np.max(d_hat)), "hard_recall": 0.0}
    for i in range(len(order)):
        thr = d_hat[order[i]]
        reuse = d_hat < thr
        tp = int((y & (~reuse)).sum())
        fn = int((y & reuse).sum())
        recall = 1.0 - fn / max(int(y.sum()), 1)
        if recall >= target_recall:
            rr = float(reuse.mean())
            if rr > best["reuse_rate"]:
                best = {
                    "reuse_rate": rr,
                    "score_threshold": float(thr),
                    "hard_recall": float(recall),
                }
    return best


def per_scenario_type_spearman(
    d: np.ndarray,
    d_hat: np.ndarray,
    scenario_type: np.ndarray,
    min_n: int = 50,
) -> dict[str, dict]:
    out = {}
    for t in sorted(set(scenario_type.tolist())):
        m = scenario_type == t
        if int(m.sum()) >= min_n:
            out[t] = {"n": int(m.sum()), "spearman": spearman(d[m], d_hat[m])}
    return out


def full_decision_metrics(
    d: np.ndarray,
    d_hat: np.ndarray,
    level: np.ndarray,
    level_pred: np.ndarray,
    hard_threshold: float,
    score_threshold: float,
    scenario_type: np.ndarray | None = None,
) -> dict:
    reg = regression_metrics(d, d_hat)
    result = {
        "primary": {
            "spearman": reg["spearman"],
            "rmse_log1p": reg["rmse_log1p"],
            "mae_log1p": reg["mae_log1p"],
            "adjacent_level_accuracy": level_metrics(level, level_pred)["adjacent_accuracy"],
        },
        "regression": reg,
        "level": level_metrics(level, level_pred),
        "hard": hard_metrics(d, d_hat, hard_threshold, score_threshold),
        "reliability": reliability_curve(d, d_hat),
        "reuse_risk": reuse_risk_curve(d, d_hat, hard_threshold),
        "reuse_at_hard_recall_95": reuse_rate_at_hard_recall(d, d_hat, hard_threshold, 0.95),
        "reuse_at_hard_recall_99": reuse_rate_at_hard_recall(d, d_hat, hard_threshold, 0.99),
    }
    if scenario_type is not None:
        result["per_scenario_type"] = per_scenario_type_spearman(d, d_hat, scenario_type)
    return result


def compute_level_edges(d_train: np.ndarray, quantiles: tuple[float, ...] = (0.5, 0.75, 0.9)) -> list[float]:
    return [float(np.quantile(d_train, q)) for q in quantiles]
