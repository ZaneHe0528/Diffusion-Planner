#!/usr/bin/env python3
"""gate_v2 数据集加载（复用 gate v1 chunk + adjacent CSV join d 口径）。"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

from gate_features import D_COLUMN_CHOICES, FEATURE_GROUPS, LABEL_KEYS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADJACENT_CSV = REPO_ROOT / "0hzxcode" / "adjacent_traj_l2_output" / "per_pair_overlap_l2.csv"

STORED_KEYS = [key for g in FEATURE_GROUPS if not g.derived for key in g.keys] + list(LABEL_KEYS)


def _iteration_from_sample_id(sample_id: np.ndarray) -> np.ndarray:
    return np.asarray([int(s.rsplit("iter_", 1)[1]) for s in sample_id.tolist()], dtype=np.int64)


def _derive_prev_d(arrays: dict[str, np.ndarray]) -> np.ndarray:
    d = arrays["d"]
    sid = arrays["scenario_id"]
    it = _iteration_from_sample_id(arrays["sample_id"])
    prev = np.zeros((d.shape[0], 3), dtype=np.float32)
    same = (sid[1:] == sid[:-1]) & (it[1:] == it[:-1] + 1)
    prev[1:, 0] = np.where(same, d[:-1], 0.0)
    prev[1:, 1] = np.where(same, np.log1p(np.clip(d[:-1], 0.0, None)), 0.0)
    prev[1:, 2] = same.astype(np.float32)
    return prev


def _make_sample_id(log_name: str, scenario_name: str, new_iteration: int) -> str:
    return f"{log_name}__{scenario_name}__iter_{int(new_iteration):06d}"


def load_d_column_from_adjacent_csv(
    adjacent_csv: Path,
    d_column: str = "perstep_max_m",
) -> dict[str, float]:
    if d_column not in D_COLUMN_CHOICES:
        raise ValueError(f"d_column must be one of {D_COLUMN_CHOICES}")
    if not adjacent_csv.exists():
        raise SystemExit(f"adjacent CSV not found: {adjacent_csv}")

    mapping: dict[str, float] = {}
    with adjacent_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if d_column not in (reader.fieldnames or []):
            raise SystemExit(f"{adjacent_csv} missing column {d_column}")
        for row in reader:
            if row.get("time_aligned", "true").lower() != "true":
                continue
            sid = _make_sample_id(row["log_name"], row["scenario_name"], int(row["new_iteration"]))
            mapping[sid] = float(row[d_column])
    print(f"loaded {len(mapping)} d labels ({d_column}) from {adjacent_csv.name}")
    return mapping


def load_chunk_dir(
    path: Path,
    max_chunks: int | None = None,
    *,
    adjacent_csv: Path | None = DEFAULT_ADJACENT_CSV,
    d_column: str = "perstep_max_m",
    use_chunk_d: bool = False,
) -> dict[str, np.ndarray]:
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

    if not use_chunk_d and adjacent_csv is not None:
        d_map = load_d_column_from_adjacent_csv(adjacent_csv, d_column)
        sample_ids = arrays["sample_id"].tolist()
        new_d = np.array([d_map.get(s, np.nan) for s in sample_ids], dtype=np.float64)
        missing = int(np.isnan(new_d).sum())
        if missing:
            print(f"[warn] {missing} rows missing in adjacent CSV for d_column={d_column}, keeping chunk d")
            keep_old = np.isnan(new_d)
            new_d[keep_old] = arrays["d"][keep_old]
        arrays["d"] = new_d.astype(np.float32)
        arrays["d_column"] = np.full(arrays["d"].shape[0], d_column, dtype=object)

    keep = np.isfinite(arrays["d"]) & (arrays["d"] >= 0)
    if keep.sum() != keep.shape[0]:
        print(f"[warn] dropped {int((~keep).sum())} rows with invalid d")
        arrays = {k: v[keep] for k, v in arrays.items()}

    order = np.lexsort((arrays["sample_id"], arrays["scenario_id"]))
    arrays = {k: v[order] for k, v in arrays.items()}
    arrays["prev_d"] = _derive_prev_d(arrays)
    print(
        f"loaded {arrays['d'].shape[0]} rows from {len(files)} chunks, "
        f"d_column={d_column}, {len(set(arrays['scenario_id'].tolist()))} scenarios"
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
