#!/usr/bin/env python3
"""训练原始观测 gate：预测帧间变化距离 d + 变化档位 level，并输出特征组重要性报告。

用法示例（全部特征组）：
  python 0hzxcode/gate/train_gate.py \
    --dataset-dir 0hzxcode/gate_output/gate_dataset_chunks \
    --output-dir 0hzxcode/gate_output/runs/all_groups

关闭某些特征组（例：只留 ego 运动学，作 rule-based 对照）：
  ... --enable-groups ego_history --output-dir .../ego_only

评测协议与 M-probe 对齐：按 scenario 划分 train/val/test（seed 3407），
hard = d > train P90，回归目标 log1p(d) 标准化。
"""

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
from gate_model import GateConfig, RawObsGate

REPO_ROOT = Path(__file__).resolve().parents[2]
TENSOR_KEYS = [key for g in FEATURE_GROUPS for key in g.keys]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "0hzxcode" / "gate_output" / "gate_dataset_chunks")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-chunks", type=int, default=None, help="冒烟用：只读前 N 个 chunk")
    parser.add_argument("--enable-groups", nargs="+", default=None)
    parser.add_argument("--disable-groups", nargs="+", default=None)
    parser.add_argument("--encoding-npz", type=Path, default=None, help="可选：外挂 encoding 数据集按 sample_id 对齐，加入对照组")
    # 训练
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--group-dropout", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--w-cls", type=float, default=1.0, help="level 分类损失权重")
    parser.add_argument("--w-reg", type=float, default=1.0, help="d 回归损失权重")
    # 数据协议
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--hard-quantile", type=float, default=0.90)
    parser.add_argument("--level-quantiles", type=float, nargs="+", default=[0.5, 0.8, 0.95])
    parser.add_argument("--level-edges-m", type=float, nargs="+", default=None, help="绝对档位边界（米），覆盖分位数")
    parser.add_argument("--no-redundancy-downsample", action="store_true")
    parser.add_argument("--skip-importance", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 指标（自包含 numpy 实现，与 M-probe 口径一致）
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# batch 迭代
# ---------------------------------------------------------------------------

def iter_batches(
    arrays: dict[str, np.ndarray],
    idx: np.ndarray,
    batch_size: int,
    device: str,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
    permute_keys: dict[str, np.ndarray] | None = None,
    extra_keys: tuple[str, ...] = (),
):
    """按 idx 切 batch，输出 dict[str, torch.Tensor]（含 d）。

    permute_keys: {npz键: 打乱后的行号数组(与 idx 等长)}，用于 permutation importance。
    """
    order = idx.copy()
    if shuffle:
        assert rng is not None
        rng.shuffle(order)
    keys = [k for k in TENSOR_KEYS if k in arrays] + [k for k in extra_keys if k in arrays]
    for s in range(0, order.shape[0], batch_size):
        rows = order[s : s + batch_size]
        batch: dict[str, torch.Tensor] = {}
        for k in keys:
            src_rows = rows
            if permute_keys is not None and k in permute_keys:
                # permute_keys[k] 与 idx 对齐：先找 rows 在 idx 中的位置
                pos = np.searchsorted(idx, rows)
                src_rows = permute_keys[k][pos]
            batch[k] = torch.from_numpy(np.ascontiguousarray(arrays[k][src_rows])).to(device)
        batch["d"] = torch.from_numpy(arrays["d"][rows]).to(device)
        yield batch


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_split(
    model: RawObsGate,
    arrays: dict[str, np.ndarray],
    idx: np.ndarray,
    device: str,
    batch_size: int = 1024,
    disabled_groups: set[str] | None = None,
    permute_keys: dict[str, np.ndarray] | None = None,
    extra_keys: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    model.eval()
    d_hat, levels_pred, z = [], [], []
    for batch in iter_batches(arrays, idx, batch_size, device, permute_keys=permute_keys, extra_keys=extra_keys):
        out = model(batch, disabled_groups=disabled_groups)
        d_hat.append(out["d_hat"].cpu().numpy())
        levels_pred.append(out["logits"].argmax(dim=1).cpu().numpy())
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
        "rmse_log1p": float(np.sqrt(np.mean((np.log1p(d) - np.log1p(np.clip(d_hat, 0, None))) ** 2))),
    }


def classification_metrics(level: np.ndarray, level_pred: np.ndarray, num_levels: int) -> dict:
    acc = float((level == level_pred).mean())
    adj = float((np.abs(level - level_pred) <= 1).mean())
    confusion = np.zeros((num_levels, num_levels), dtype=int)
    for t, p in zip(level, level_pred):
        confusion[t, p] += 1
    recalls, precisions = [], []
    for k in range(num_levels):
        tp = confusion[k, k]
        recalls.append(float(tp / max(confusion[k].sum(), 1)))
        precisions.append(float(tp / max(confusion[:, k].sum(), 1)))
    f1 = [2 * p * r / max(p + r, 1e-9) for p, r in zip(precisions, recalls)]
    return {
        "accuracy": acc,
        "adjacent_accuracy": adj,
        "macro_f1": float(np.mean(f1)),
        "per_level_recall": recalls,
        "per_level_precision": precisions,
        "top_level_recall": recalls[-1],
        "confusion": confusion.tolist(),
    }


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
        "positive_rate": float(y.mean()),
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
        "classification": classification_metrics(level[idx], pred["level_pred"], num_levels),
        "hard": hard_metrics(d, pred["d_hat"], hard_threshold, score_threshold),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    arrays = load_chunk_dir(args.dataset_dir, args.max_chunks)

    # 可选外挂 encoding（对照组）
    use_encoding = False
    if args.encoding_npz is not None:
        enc_data = np.load(args.encoding_npz, allow_pickle=False)
        enc_map = {sid: i for i, sid in enumerate(enc_data["sample_id"].tolist())}
        pos = np.array([enc_map.get(s, -1) for s in arrays["sample_id"].tolist()])
        keep = pos >= 0
        arrays = {k: v[keep] for k, v in arrays.items()}
        arrays["encoding"] = np.asarray(enc_data["encoding"], dtype=np.float32)[pos[keep]]
        use_encoding = True
        print(f"joined encoding: kept {int(keep.sum())} rows")

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

    # hard 阈值 & level 档位边界
    hard_threshold = float(np.quantile(d_train, args.hard_quantile))
    if args.level_edges_m is not None:
        level_edges = [float(x) for x in args.level_edges_m]
    else:
        level_edges = [float(np.quantile(d_train, q)) for q in args.level_quantiles]
    num_levels = len(level_edges) + 1
    level_all = np.digitize(arrays["d"], level_edges)
    print(f"hard_threshold(P{args.hard_quantile*100:.0f})={hard_threshold:.3f} m, level_edges={[round(e,3) for e in level_edges]} m")
    print(f"train level 分布: {[int((level_all[idx['train']]==k).sum()) for k in range(num_levels)]}")

    enabled = resolve_enabled_groups(args.enable_groups, args.disable_groups)
    cfg = GateConfig(
        enabled_groups=enabled,
        num_levels=num_levels,
        dropout=args.dropout,
        group_dropout=args.group_dropout,
        use_encoding=use_encoding,
    )
    device = args.device
    model = RawObsGate(cfg).to(device)
    print(f"enabled groups: {model.group_order}")
    print(f"gate parameters: {model.count_parameters()/1e6:.3f} M")

    extra_keys = ("encoding",) if use_encoding else ()
    model.fit_normalization(
        iter_batches(arrays, idx["train"], 2048, device, extra_keys=extra_keys),
        level_edges,
        d_train,
    )

    # 类别权重（inverse freq，均值归一，封顶 10）
    counts = np.bincount(level_all[idx["train"]], minlength=num_levels).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1.0)
    w = w / w.mean()
    class_weights = torch.tensor(np.minimum(w, 10.0), dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = int(np.ceil(idx["train"].shape[0] / args.batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch, eta_min=1e-5)

    level_t = torch.from_numpy(level_all).to(device)
    history = []
    best = {"val_spearman": -2.0}
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in iter_batches(arrays, idx["train"], args.batch_size, device, shuffle=True, rng=rng, extra_keys=extra_keys):
            out = model(batch)
            z_true = model.d_to_z(batch["d"])
            # batch 内行号与 level_all 对齐：直接用 d 重新分档（等价且免传行号）
            lvl_true = model.d_to_level(batch["d"])
            loss_reg = F.huber_loss(out["z"], z_true, delta=1.0)
            loss_cls = F.cross_entropy(out["logits"], lvl_true, weight=class_weights)
            loss = args.w_reg * loss_reg + args.w_cls * loss_cls
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))

        pred_val = predict_split(model, arrays, idx["val"], device, disabled_groups=None, extra_keys=extra_keys)
        val_sp = spearman(arrays["d"][idx["val"]], pred_val["d_hat"])
        val_auroc = auroc(arrays["d"][idx["val"]] > hard_threshold, pred_val["d_hat"])
        val_acc = float((level_all[idx["val"]] == pred_val["level_pred"]).mean())
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_spearman": val_sp,
            "val_auroc": val_auroc,
            "val_level_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(f"epoch {epoch:3d} loss={row['train_loss']:.4f} val_spearman={val_sp:.4f} "
              f"val_auroc={val_auroc:.4f} val_acc={val_acc:.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)

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

    # ---------------- 最终评测（best checkpoint） ----------------
    state = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    print(f"best epoch = {best['epoch']}, val_spearman = {best['val_spearman']:.4f}")

    pred_val = predict_split(model, arrays, idx["val"], device, extra_keys=extra_keys)
    score_threshold = choose_threshold_for_recall(
        arrays["d"][idx["val"]] > hard_threshold, pred_val["d_hat"], target_recall=0.95
    )

    results = {"data": {
        "rows": int(arrays["d"].shape[0]),
        "scenarios": len(set(arrays["scenario_id"].tolist())),
        "split_rows": {k: int(v.shape[0]) for k, v in idx.items()},
        "hard_threshold_m": hard_threshold,
        "level_edges_m": level_edges,
        "enabled_groups": model.group_order,
        "parameters": model.count_parameters(),
    }, "splits": {}}
    for split in ("train", "val", "test"):
        pred = predict_split(model, arrays, idx[split], device, extra_keys=extra_keys)
        results["splits"][split] = full_metrics(arrays, idx[split], pred, level_all, num_levels, hard_threshold, score_threshold)

    # 按场景类型分解（test）
    pred_test = predict_split(model, arrays, idx["test"], device, extra_keys=extra_keys)
    types = arrays["scenario_type"][idx["test"]]
    per_type = {}
    for t in sorted(set(types.tolist())):
        m = types == t
        if m.sum() >= 50:
            per_type[t] = {
                "n": int(m.sum()),
                "spearman": spearman(arrays["d"][idx["test"]][m], pred_test["d_hat"][m]),
            }
    results["per_scenario_type_test"] = per_type

    # ---------------- 特征组重要性 ----------------
    if not args.skip_importance:
        base_sp = results["splits"]["test"]["regression"]["spearman"]
        base_auroc = results["splits"]["test"]["hard"]["auroc"]
        importance = []
        test_idx_sorted = np.sort(idx["test"])
        for name in model.group_order:
            # 1) leave-one-out：置零该组 embedding
            pred_loo = predict_split(model, arrays, idx["test"], device, disabled_groups={name}, extra_keys=extra_keys)
            loo_sp = spearman(arrays["d"][idx["test"]], pred_loo["d_hat"])
            loo_auroc = auroc(arrays["d"][idx["test"]] > hard_threshold, pred_loo["d_hat"])
            # 2) solo：只保留该组
            others = set(model.group_order) - {name}
            pred_solo = predict_split(model, arrays, idx["test"], device, disabled_groups=others, extra_keys=extra_keys)
            solo_sp = spearman(arrays["d"][idx["test"]], pred_solo["d_hat"])
            # 3) permutation：打乱该组原始输入
            keys = list(GROUP_BY_NAME[name].keys) if name in GROUP_BY_NAME else ["encoding"]
            perm = rng.permutation(test_idx_sorted)
            pred_perm = predict_split(
                model, arrays, test_idx_sorted, device,
                permute_keys={k: perm for k in keys}, extra_keys=extra_keys,
            )
            perm_sp = spearman(arrays["d"][test_idx_sorted], pred_perm["d_hat"])
            importance.append({
                "group": name,
                "prior_rank": GROUP_BY_NAME[name].rank if name in GROUP_BY_NAME else None,
                "loo_delta_spearman": base_sp - loo_sp,
                "loo_delta_auroc": base_auroc - loo_auroc,
                "solo_spearman": solo_sp,
                "perm_delta_spearman": base_sp - perm_sp,
            })
        importance.sort(key=lambda r: r["loo_delta_spearman"], reverse=True)
        for measured_rank, row in enumerate(importance, start=1):
            row["measured_rank"] = measured_rank
        results["importance"] = importance

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    write_report(results, out_dir)
    make_plots(arrays, idx, model, device, history, out_dir, extra_keys)
    print(f"wrote: {out_dir}/metrics.json, importance_report.md, plots")


def write_report(results: dict, out_dir: Path) -> None:
    lines = ["# Gate 训练报告", ""]
    data = results["data"]
    lines += [
        f"- 样本数: {data['rows']}，场景数: {data['scenarios']}，划分: {data['split_rows']}",
        f"- 启用特征组: {data['enabled_groups']}，参数量: {data['parameters']/1e6:.3f} M",
        f"- hard 阈值: {data['hard_threshold_m']:.3f} m，level 边界: {[round(e,3) for e in data['level_edges_m']]} m",
        "",
        "## 各划分指标",
        "",
        "| split | RMSE | MAE | Pearson | Spearman | AUROC | hard recall | false easy | level acc | adj acc | macro F1 | top recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, m in results["splits"].items():
        r, c, h = m["regression"], m["classification"], m["hard"]
        lines.append(
            f"| {split} | {r['rmse']:.3f} | {r['mae']:.3f} | {r['pearson']:.3f} | {r['spearman']:.4f} | "
            f"{h['auroc']:.4f} | {h['hard_recall']:.3f} | {h['false_easy_rate']:.3f} | "
            f"{c['accuracy']:.3f} | {c['adjacent_accuracy']:.3f} | {c['macro_f1']:.3f} | {c['top_level_recall']:.3f} |"
        )

    if "importance" in results:
        lines += [
            "",
            "## 特征组重要性（test 集，按 leave-one-out ΔSpearman 排序）",
            "",
            "| 实测排名 | 特征组 | 先验排名 | LOO ΔSpearman | LOO ΔAUROC | solo Spearman | perm ΔSpearman |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for row in results["importance"]:
            lines.append(
                f"| {row['measured_rank']} | {row['group']} | {row['prior_rank']} | "
                f"{row['loo_delta_spearman']:.4f} | {row['loo_delta_auroc']:.4f} | "
                f"{row['solo_spearman']:.4f} | {row['perm_delta_spearman']:.4f} |"
            )

    if results.get("per_scenario_type_test"):
        lines += ["", "## test 按场景类型 Spearman", ""]
        for t, m in sorted(results["per_scenario_type_test"].items(), key=lambda kv: -kv[1]["spearman"]):
            lines.append(f"- {t}: {m['spearman']:.3f} (n={m['n']})")

    (out_dir / "importance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(arrays, idx, model, device, history, out_dir: Path, extra_keys) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ep = [h["epoch"] for h in history]
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train loss")
    axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].set_title("loss")
    axes[1].plot(ep, [h["val_spearman"] for h in history], label="val spearman")
    axes[1].plot(ep, [h["val_auroc"] for h in history], label="val auroc")
    axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].set_title("val metrics")

    pred = predict_split(model, arrays, idx["test"], device, extra_keys=extra_keys)
    d = arrays["d"][idx["test"]]
    n = min(6000, d.shape[0])
    sel = np.random.default_rng(0).choice(d.shape[0], n, replace=False)
    axes[2].scatter(np.log1p(d[sel]), np.log1p(np.clip(pred["d_hat"][sel], 0, None)), s=3, alpha=0.25)
    lim = [0, max(np.log1p(d[sel]).max(), 0.1)]
    axes[2].plot(lim, lim, "r--", lw=1)
    axes[2].set_xlabel("log1p(d)"); axes[2].set_ylabel("log1p(d_hat)"); axes[2].set_title("test: pred vs true")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
