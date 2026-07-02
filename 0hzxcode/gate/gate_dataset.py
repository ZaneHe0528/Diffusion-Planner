#!/usr/bin/env python3
"""gate 数据集加载：合并 chunk npz、按 scenario 划分、BLUE 式冗余下采样。"""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np

from gate_features import FEATURE_GROUPS, LABEL_KEYS

STORED_KEYS = [key for g in FEATURE_GROUPS if not g.derived for key in g.keys] + list(LABEL_KEYS)


def _iteration_from_sample_id(sample_id: np.ndarray) -> np.ndarray:
    return np.asarray([int(s.rsplit("iter_", 1)[1]) for s in sample_id.tolist()], dtype=np.int64)


def _derive_prev_d(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """上一帧实测 d（[d_prev, log1p(d_prev), valid]）；场景首帧或迭代不连续 valid=0。

    依赖行已按 (scenario_id, iteration) 排序。
    """
    d = arrays["d"]
    sid = arrays["scenario_id"]
    it = _iteration_from_sample_id(arrays["sample_id"])
    prev = np.zeros((d.shape[0], 3), dtype=np.float32)
    same = (sid[1:] == sid[:-1]) & (it[1:] == it[:-1] + 1)
    prev[1:, 0] = np.where(same, d[:-1], 0.0)
    prev[1:, 1] = np.where(same, np.log1p(np.clip(d[:-1], 0.0, None)), 0.0)
    prev[1:, 2] = same.astype(np.float32)
    return prev


def load_chunk_dir(path: Path, max_chunks: int | None = None) -> dict[str, np.ndarray]:
    """读取目录下全部 chunk_*.npz 并按行拼接；丢弃 d 非有限的行；派生 prev_d。"""
    files = sorted(path.glob("chunk_*.npz"))
    if max_chunks is not None:
        files = files[:max_chunks]
    if not files:
        raise SystemExit(f"no chunk_*.npz under {path}")

    columns: dict[str, list[np.ndarray]] = {k: [] for k in STORED_KEYS}
    for f in files:
        with np.load(f, allow_pickle=False) as data:
            missing = set(STORED_KEYS) - set(data.files)
            if missing:
                raise SystemExit(f"{f} missing keys: {sorted(missing)}")
            for k in STORED_KEYS:
                columns[k].append(np.asarray(data[k]))

    arrays = {k: np.concatenate(v, axis=0) for k, v in columns.items()}
    keep = np.isfinite(arrays["d"]) & (arrays["d"] >= 0)
    if keep.sum() != keep.shape[0]:
        print(f"[warn] dropped {int((~keep).sum())} rows with invalid d")
        arrays = {k: v[keep] for k, v in arrays.items()}

    # 统一按 (scenario_id, sample_id) 排序，保证同场景帧连续且按迭代升序
    # （冗余下采样与 prev_d 派生都依赖这一顺序）。
    order = np.lexsort((arrays["sample_id"], arrays["scenario_id"]))
    arrays = {k: v[order] for k, v in arrays.items()}
    arrays["prev_d"] = _derive_prev_d(arrays)
    print(f"loaded {arrays['d'].shape[0]} rows from {len(files)} chunks, "
          f"{len(set(arrays['scenario_id'].tolist()))} scenarios")
    return arrays


def scenario_split(scenario_id: np.ndarray, val_ratio: float, test_ratio: float, seed: int) -> dict[str, np.ndarray]:
    """按 scenario 划分 train/val/test（防相邻帧泄漏），与 M-probe 协议一致。"""
    scenarios = sorted(set(scenario_id.tolist()))
    rng = random.Random(seed)
    rng.shuffle(scenarios)
    n = len(scenarios)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    test_set = set(scenarios[:n_test])
    val_set = set(scenarios[n_test : n_test + n_val])
    return {
        "train": np.asarray([s not in test_set and s not in val_set for s in scenario_id]),
        "val": np.asarray([s in val_set for s in scenario_id]),
        "test": np.asarray([s in test_set for s in scenario_id]),
    }


def _frame_signature(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """每帧签名向量（ego 历史 + 邻车当前帧状态），用于 BLUE 式冗余检测。"""
    ego = arrays["ego_history"].reshape(arrays["ego_history"].shape[0], -1).astype(np.float32)
    nbr = arrays["neighbor_agents_past"][:, :, -1, :].reshape(ego.shape[0], -1).astype(np.float32)
    sig = np.concatenate([ego, nbr], axis=1)
    norm = np.linalg.norm(sig, axis=1, keepdims=True)
    norm[norm < 1e-6] = 1.0
    return sig / norm


def redundancy_downsample_mask(
    arrays: dict[str, np.ndarray],
    base_mask: np.ndarray,
    cos_threshold: float = 0.99,
    seed: int = 0,
) -> np.ndarray:
    """BLUE 式时间冗余下采样（仅作用于 base_mask 内的行，通常是 train）。

    连续帧签名余弦相似度 > 阈值的段视为冗余段，长度 L 的段保留
    max(2, ceil(sqrt(L))) 个等间隔代表帧。
    """
    sig = _frame_signature(arrays)
    scenario_id = arrays["scenario_id"]
    keep = np.zeros_like(base_mask)
    keep[~base_mask] = False

    idx_all = np.nonzero(base_mask)[0]
    # 行已按 (scenario, iteration) 排序，直接按 scenario 分组。
    start = 0
    while start < idx_all.shape[0]:
        end = start
        sid = scenario_id[idx_all[start]]
        while end < idx_all.shape[0] and scenario_id[idx_all[end]] == sid:
            end += 1
        rows = idx_all[start:end]
        cos = np.sum(sig[rows[1:]] * sig[rows[:-1]], axis=1) if rows.shape[0] > 1 else np.zeros(0)

        seg_start = 0
        for i in range(rows.shape[0]):
            is_last = i == rows.shape[0] - 1
            # 段边界：与下一帧相似度跌破阈值，或到场景末尾
            if is_last or cos[i] <= cos_threshold:
                seg = rows[seg_start : i + 1]
                length = seg.shape[0]
                n_keep = length if length <= 2 else max(2, int(np.ceil(np.sqrt(length))))
                sel = np.unique(np.linspace(0, length - 1, n_keep).round().astype(int))
                keep[seg[sel]] = True
                seg_start = i + 1
        start = end

    return keep
