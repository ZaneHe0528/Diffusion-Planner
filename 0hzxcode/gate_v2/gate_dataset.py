#!/usr/bin/env python3
"""gate_v2 数据集加载（复用 gate v1 导出的 chunk npz）。"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from gate_features import FEATURE_GROUPS, LABEL_KEYS

STORED_KEYS = [key for g in FEATURE_GROUPS for key in g.keys] + list(LABEL_KEYS)


def load_chunk_dir(path: Path, max_chunks: int | None = None) -> dict[str, np.ndarray]:
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

    order = np.lexsort((arrays["sample_id"], arrays["scenario_id"]))
    arrays = {k: v[order] for k, v in arrays.items()}
    print(
        f"loaded {arrays['d'].shape[0]} rows from {len(files)} chunks, "
        f"{len(set(arrays['scenario_id'].tolist()))} scenarios"
    )
    return arrays


def scenario_split(scenario_id: np.ndarray, val_ratio: float, test_ratio: float, seed: int) -> dict[str, np.ndarray]:
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
) -> np.ndarray:
    sig = _frame_signature(arrays)
    scenario_id = arrays["scenario_id"]
    keep = np.zeros_like(base_mask)
    keep[~base_mask] = False

    idx_all = np.nonzero(base_mask)[0]
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
            if is_last or cos[i] <= cos_threshold:
                seg = rows[seg_start : i + 1]
                length = seg.shape[0]
                n_keep = length if length <= 2 else max(2, int(np.ceil(np.sqrt(length))))
                sel = np.unique(np.linspace(0, length - 1, n_keep).round().astype(int))
                keep[seg[sel]] = True
                seg_start = i + 1
        start = end

    return keep
