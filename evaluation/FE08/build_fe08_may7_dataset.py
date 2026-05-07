"""Build the FE-08 May7 dataset.

FE-08 follows the Claude May 7 plan exactly:

- Drop high-missing ``user_int`` and ``item_int`` fids with threshold 0.80.
- Add safe item dense fids 86/91/92 and normalize them from train row groups.
- Add item int fids 89/90/91 for target-history match.
- Add user int fids 130/131 for sample hour/day-of-week.
- Sort every sequence domain by event timestamp descending before writing.

The script writes an enhanced parquet dataset plus schema/ns_groups/FE08
sidecars. It does not introduce FE-07 P2-Domain buckets or FE-07 user_dense
110/120/121 by default.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_ITEM_DENSE_FIDS = (86, 91, 92)
RISKY_ITEM_DENSE_FIDS = {
    87: "FE-01 full eval AUC collapse risk: purchase-frequency train/eval mismatch",
    88: "label_time-dependent delay feature; excluded from FE-08 inputs",
}
USER_INT_ADDS = [(130, 25, 1), (131, 8, 1)]
ITEM_INT_ADDS_BASE = [(89, 3, 1)]
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1


@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update_many(self, values: Iterable[float]) -> None:
        for raw in values:
            value = float(raw)
            if not math.isfinite(value):
                continue
            self.n += 1
            delta = value - self.mean
            self.mean += delta / self.n
            delta2 = value - self.mean
            self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        var = self.m2 / (self.n - 1)
        return max(math.sqrt(max(var, 0.0)), 1e-6)

    def to_dict(self) -> Dict[str, float]:
        return {"n": self.n, "mean": self.mean, "std": self.std}


@dataclass
class IntAudit:
    total_rows: int = 0
    missing_rows: int = 0
    max_positive: int = 0
    value_stats: RunningStats = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.value_stats is None:
            self.value_stats = RunningStats()

    @property
    def missing_ratio(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return self.missing_rows / self.total_rows

    @property
    def fill_value(self) -> int:
        if self.value_stats.n == 0:
            return 0
        return max(1, int(round(self.value_stats.mean)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "missing_rows": self.missing_rows,
            "missing_ratio": self.missing_ratio,
            "max_positive": self.max_positive,
            "fill_value": self.fill_value,
            "value_mean": self.value_stats.mean,
            "value_count": self.value_stats.n,
        }


class PrefixState:
    def __init__(self) -> None:
        self.user_total: DefaultDict[int, int] = defaultdict(int)
        self.item_total: DefaultDict[int, int] = defaultdict(int)

    def before_update(self, user_id: int, item_id: int) -> Tuple[int, int]:
        return self.user_total[user_id], self.item_total[item_id]

    def update(self, user_id: int, item_id: int) -> None:
        self.user_total[user_id] += 1
        self.item_total[item_id] += 1


def parquet_files(input_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    return files


def parquet_row_groups(files: Sequence[str]) -> List[Tuple[str, int]]:
    row_groups: List[Tuple[str, int]] = []
    for path in files:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.metadata.num_row_groups):
            row_groups.append((path, rg_idx))
    if not row_groups:
        raise ValueError("Input parquet files contain no row groups")
    return row_groups


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_input_schema_for_fe08(
    schema: Dict[str, Any],
    allow_risky_item_dense: bool,
) -> None:
    user_dense_fids = {int(row[0]) for row in schema.get("user_dense", [])}
    item_dense_fids = {int(row[0]) for row in schema.get("item_dense", [])}

    forbidden_user_dense = {110, 120, 121} & user_dense_fids
    if forbidden_user_dense:
        raise ValueError(
            "FE-08 mainline expects raw/baseline input schema and must not "
            "inherit FE-07 user_dense generated fids: "
            f"{sorted(forbidden_user_dense)}."
        )

    risky_existing = set(RISKY_ITEM_DENSE_FIDS) & item_dense_fids
    if risky_existing and not allow_risky_item_dense:
        details = ", ".join(
            f"{fid}: {RISKY_ITEM_DENSE_FIDS[fid]}" for fid in sorted(risky_existing)
        )
        raise ValueError(
            "FE-08 input schema already contains risky item_dense fids. "
            f"{details}. Pass --enable_risky_item_dense_fids only if this is "
            "an explicit non-mainline override."
        )


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def iter_batches(
    row_groups: Sequence[Tuple[str, int]],
    batch_size: int,
    columns: Optional[Sequence[str]] = None,
    progress_label: Optional[str] = None,
    progress_every: int = 50,
) -> Iterable[Tuple[str, pa.RecordBatch]]:
    read_columns = None if columns is None else list(dict.fromkeys(columns))
    total = len(row_groups)
    for pos, (path, rg_idx) in enumerate(row_groups, start=1):
        if progress_label and (
            pos == 1
            or pos == total
            or (progress_every > 0 and pos % progress_every == 0)
        ):
            print(f"[FE-08] {progress_label}: {pos}/{total} row groups", flush=True)
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(
            batch_size=batch_size,
            row_groups=[rg_idx],
            columns=read_columns,
        ):
            yield path, batch


def _existing_columns(
    row_groups: Sequence[Tuple[str, int]],
    columns: Sequence[str],
) -> List[str]:
    if not row_groups:
        return []
    available = set(pq.ParquetFile(row_groups[0][0]).schema_arrow.names)
    return [name for name in dict.fromkeys(columns) if name in available]


def column_name(kind: str, fid: int) -> str:
    return f"{kind}_feats_{fid}"


def _strip_prefix(prefix: str) -> str:
    return prefix[:-1] if prefix.endswith("_") else prefix


def _append_feature_specs(
    specs: Sequence[Sequence[int]],
    adds: Sequence[Sequence[int]],
) -> List[List[int]]:
    by_fid = {int(row[0]): [int(x) for x in row] for row in specs}
    for row in adds:
        by_fid[int(row[0])] = [int(x) for x in row]
    return [by_fid[fid] for fid in sorted(by_fid)]


def _to_int_array(col: pa.Array) -> np.ndarray:
    return col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=True)


def _to_float_array(col: pa.Array) -> np.ndarray:
    return col.fill_null(np.nan).to_numpy(zero_copy_only=False).astype(np.float32, copy=True)


def _positive_values(col: pa.Array) -> List[int]:
    values: List[int] = []
    if pa.types.is_list(col.type):
        for row in col.to_pylist():
            if row is None:
                continue
            for x in row:
                if x is not None and int(x) > 0:
                    values.append(int(x))
    else:
        arr = _to_int_array(col)
        values.extend(int(x) for x in arr if int(x) > 0)
    return values


def _missing_rows(col: pa.Array) -> int:
    if pa.types.is_list(col.type):
        missing = 0
        for row in col.to_pylist():
            if row is None or not any(x is not None and int(x) > 0 for x in row):
                missing += 1
        return missing
    arr = _to_int_array(col)
    return int((arr <= 0).sum())


def _update_int_audit(audit: IntAudit, col: pa.Array, num_rows: int) -> None:
    audit.total_rows += num_rows
    positives: List[int] = []
    if pa.types.is_list(col.type):
        for row in col.to_pylist():
            if row is None:
                audit.missing_rows += 1
                continue
            row_has_positive = False
            for x in row:
                if x is not None and int(x) > 0:
                    value = int(x)
                    positives.append(value)
                    row_has_positive = True
                    if value > audit.max_positive:
                        audit.max_positive = value
            if not row_has_positive:
                audit.missing_rows += 1
    else:
        arr = _to_int_array(col)
        positive_arr = arr[arr > 0]
        audit.missing_rows += int((arr <= 0).sum())
        if positive_arr.size:
            positives.extend(int(x) for x in positive_arr)
            audit.max_positive = max(audit.max_positive, int(positive_arr.max()))
    audit.value_stats.update_many(positives)


def _dense_values(col: pa.Array) -> List[float]:
    out: List[float] = []
    if pa.types.is_list(col.type):
        for row in col.to_pylist():
            if row is None:
                continue
            for x in row:
                if x is not None and math.isfinite(float(x)):
                    out.append(float(x))
    else:
        arr = _to_float_array(col)
        out.extend(float(x) for x in arr if math.isfinite(float(x)))
    return out


def _first_scalar_from_maybe_list(values: Iterable[Any]) -> np.ndarray:
    out: List[int] = []
    for value in values:
        if value is None:
            out.append(0)
        elif isinstance(value, list):
            first = 0
            for x in value:
                if x is not None and int(x) > 0:
                    first = int(x)
                    break
            out.append(first)
        else:
            out.append(int(value))
    return np.asarray(out, dtype=np.int64)


def _list_values(col: pa.Array) -> List[List[int]]:
    values = col.to_pylist()
    return [[] if v is None else [int(x) for x in v if x is not None] for v in values]


def _bucketize_counts(counts: np.ndarray, edges: Sequence[int]) -> np.ndarray:
    upper_edges = np.asarray(list(edges)[1:], dtype=np.int64)
    return (np.searchsorted(upper_edges, counts, side="right") + 1).astype(np.int64)


def _normalize(name: str, values: np.ndarray, stats: Dict[str, RunningStats]) -> np.ndarray:
    tracker = stats[name]
    return ((values.astype(np.float32) - tracker.mean) / tracker.std).astype(np.float32)


def _append_or_replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx == -1:
        return table.append_column(name, values)
    return table.set_column(idx, name, values)


def _resolve_domain_d_columns(
    schema: Dict[str, Any],
    parquet_names: Sequence[str],
) -> Tuple[str, str]:
    names = set(parquet_names)
    candidates: List[Tuple[str, Optional[int]]] = []
    for domain, cfg in schema.get("seq", {}).items():
        prefix = _strip_prefix(str(cfg.get("prefix", "")))
        if domain in {"domain_d", "seq_d"} or prefix == "domain_d_seq":
            candidates.append((prefix, cfg.get("ts_fid")))
    candidates.append(("domain_d_seq", 26))
    for prefix, ts_fid in candidates:
        match_col = f"{prefix}_19"
        ts_col = f"{prefix}_{ts_fid}" if ts_fid is not None else f"{prefix}_26"
        if match_col in names and ts_col in names:
            return match_col, ts_col
    raise KeyError(
        "Could not resolve domain_d match columns. Expected domain_d_seq_19 "
        "and a timestamp column such as domain_d_seq_26."
    )


def _dense_feature_names(schema: Dict[str, Any]) -> List[str]:
    names = []
    for fid, _ in schema.get("user_dense", []):
        names.append(column_name("user_dense", int(fid)))
    for fid, _ in schema.get("item_dense", []):
        names.append(column_name("item_dense", int(fid)))
    return names


def _seq_timestamp_columns(schema: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for domain, cfg in schema.get("seq", {}).items():
        prefix = _strip_prefix(str(cfg.get("prefix", "")))
        ts_fid = cfg.get("ts_fid")
        if ts_fid is not None:
            out[domain] = f"{prefix}_{ts_fid}"
    return out


def _collect_audit_and_min_ts(
    row_groups: Sequence[Tuple[str, int]],
    schema: Dict[str, Any],
    batch_size: int,
    collect_feature_audits: bool = True,
    include_seq_audits: bool = False,
    progress_label: Optional[str] = None,
) -> Tuple[Dict[str, IntAudit], Dict[str, Any]]:
    audits: Dict[str, IntAudit] = {}
    if collect_feature_audits:
        for fid, _, _ in schema.get("user_int", []):
            audits[column_name("user_int", int(fid))] = IntAudit()
        for fid, _, _ in schema.get("item_int", []):
            audits[column_name("item_int", int(fid))] = IntAudit()
        if include_seq_audits:
            for domain, cfg in schema.get("seq", {}).items():
                prefix = _strip_prefix(str(cfg.get("prefix", "")))
                ts_fid = cfg.get("ts_fid")
                for fid, _ in cfg.get("features", []):
                    if int(fid) == int(ts_fid):
                        continue
                    audits[f"{prefix}_{int(fid)}"] = IntAudit()

    label_counts: Counter[int] = Counter()
    total_rows = 0
    pos = 0
    neg = 0
    ts_min: Optional[int] = None
    ts_max: Optional[int] = None
    lt_min: Optional[int] = None
    lt_max: Optional[int] = None
    lt_observed = 0

    read_columns: List[str] = ["label_type", "timestamp", "label_time"]
    if collect_feature_audits:
        read_columns.extend(audits.keys())
    read_columns = _existing_columns(row_groups, read_columns)

    for _, batch in iter_batches(
        row_groups,
        batch_size,
        columns=read_columns,
        progress_label=progress_label,
    ):
        names = batch.schema.names
        idx = {name: i for i, name in enumerate(names)}
        B = batch.num_rows
        total_rows += B

        if "label_type" in idx:
            labels = _to_int_array(batch.column(idx["label_type"]))
            label_counts.update(int(x) for x in labels)
            pos += int((labels == 2).sum())
            neg += int((labels != 2).sum())

        if "timestamp" in idx:
            ts = _to_int_array(batch.column(idx["timestamp"]))
            if ts.size:
                cur_min = int(ts.min())
                cur_max = int(ts.max())
                ts_min = cur_min if ts_min is None else min(ts_min, cur_min)
                ts_max = cur_max if ts_max is None else max(ts_max, cur_max)

        if "label_time" in idx:
            lt = _to_int_array(batch.column(idx["label_time"]))
            observed = lt[lt > 0]
            lt_observed += int(observed.size)
            if observed.size:
                cur_min = int(observed.min())
                cur_max = int(observed.max())
                lt_min = cur_min if lt_min is None else min(lt_min, cur_min)
                lt_max = cur_max if lt_max is None else max(lt_max, cur_max)

        if collect_feature_audits:
            for name, audit in audits.items():
                if name not in idx:
                    continue
                _update_int_audit(audit, batch.column(idx[name]), B)

    diagnostics = {
        "rows": total_rows,
        "positive_label_type_2": pos,
        "negative_not_label_type_2": neg,
        "label_type_counts": dict(sorted(label_counts.items())),
        "timestamp_min": ts_min,
        "timestamp_max": ts_max,
        "label_time_min": lt_min,
        "label_time_max": lt_max,
        "label_time_observed_rows": lt_observed,
    }
    return audits, diagnostics


def _fill_int_column(col: pa.Array, fill_value: int, fill_empty_lists: bool) -> pa.Array:
    if fill_value <= 0:
        return col
    if pa.types.is_list(col.type):
        rows: List[List[int]] = []
        for row in col.to_pylist():
            if row is None or len(row) == 0:
                rows.append([fill_value] if fill_empty_lists else ([] if row is None else row))
                continue
            rows.append([
                int(x) if x is not None and int(x) > 0 else fill_value
                for x in row
            ])
        return pa.array(rows, type=pa.list_(pa.int64()))
    arr = _to_int_array(col)
    arr[arr <= 0] = fill_value
    return pa.array(arr, type=pa.int64())


def _normalize_dense_column(col: pa.Array, name: str, stats: Dict[str, RunningStats]) -> pa.Array:
    tracker = stats[name]
    if pa.types.is_list(col.type):
        rows: List[List[float]] = []
        for row in col.to_pylist():
            if row is None:
                rows.append([])
                continue
            vals = []
            for x in row:
                value = tracker.mean if x is None or not math.isfinite(float(x)) else float(x)
                vals.append((value - tracker.mean) / tracker.std)
            rows.append(vals)
        return pa.array(rows, type=pa.list_(pa.float32()))
    arr = col.fill_null(tracker.mean).to_numpy(zero_copy_only=False).astype(np.float32, copy=True)
    arr = np.nan_to_num(arr, nan=tracker.mean, posinf=tracker.mean, neginf=tracker.mean)
    return pa.array(((arr - tracker.mean) / tracker.std).astype(np.float32), type=pa.float32())


def _compute_generated_features(
    batch: pa.RecordBatch,
    state: PrefixState,
    match_col: str,
    match_ts_col: str,
    seq_ts_cols: Dict[str, str],
    min_timestamp: int,
    match_window_seconds: int,
    count_edges: Sequence[int],
) -> Dict[str, np.ndarray]:
    names = batch.schema.names
    idx = {name: i for i, name in enumerate(names)}
    B = batch.num_rows

    user_ids = _to_int_array(batch.column(idx["user_id"]))
    item_ids = _to_int_array(batch.column(idx["item_id"]))
    timestamps = _to_int_array(batch.column(idx["timestamp"]))

    if "item_int_feats_9" not in idx:
        raise KeyError("FE-08 requires item_int_feats_9 for target-history match")
    target_item_attr = _first_scalar_from_maybe_list(batch.column(idx["item_int_feats_9"]).to_pylist())
    seq_values = _list_values(batch.column(idx[match_col]))
    seq_times = _list_values(batch.column(idx[match_ts_col]))

    item_total = np.zeros(B, dtype=np.float32)
    has_match = np.zeros(B, dtype=np.int64)
    match_count = np.zeros(B, dtype=np.int64)
    min_match_delta = np.zeros(B, dtype=np.float32)
    match_count_7d = np.zeros(B, dtype=np.float32)
    latest_match_bucket = np.zeros(B, dtype=np.int64)
    hour_id = ((timestamps // 3600) % 24 + 1).astype(np.int64)
    dow_id = (((timestamps // 86400) + 4) % 7 + 1).astype(np.int64)

    row_order = np.argsort(timestamps, kind="stable")
    for i in row_order:
        uid = int(user_ids[i])
        iid = int(item_ids[i])
        ut, it = state.before_update(uid, iid)
        item_total[i] = it

        target = int(target_item_attr[i])
        if target > 0:
            deltas: List[int] = []
            count_7d = 0
            now = int(timestamps[i])
            for value, event_time in zip(seq_values[i], seq_times[i]):
                if int(value) != target or int(event_time) <= 0 or int(event_time) > now:
                    continue
                match_count[i] += 1
                delta = now - int(event_time)
                deltas.append(delta)
                if delta <= match_window_seconds:
                    count_7d += 1
            has_match[i] = 2 if match_count[i] > 0 else 1
            if deltas:
                delta_min = min(deltas)
                min_match_delta[i] = float(delta_min)
                raw_bucket = int(np.searchsorted(BUCKET_BOUNDARIES, delta_min))
                raw_bucket = min(raw_bucket, len(BUCKET_BOUNDARIES) - 1)
                latest_match_bucket[i] = raw_bucket + 1
            match_count_7d[i] = count_7d
        else:
            has_match[i] = 0

        state.update(uid, iid)

    return {
        "item_dense_feats_86": np.log1p(item_total).astype(np.float32),
        "item_int_feats_89": has_match,
        "item_int_feats_90": _bucketize_counts(match_count, count_edges),
        "item_int_feats_91": latest_match_bucket,
        "item_dense_feats_91": np.log1p(min_match_delta).astype(np.float32),
        "item_dense_feats_92": np.log1p(match_count_7d).astype(np.float32),
        "user_int_feats_130": hour_id,
        "user_int_feats_131": dow_id,
    }


def _generated_dense_names() -> List[str]:
    return [
        "item_dense_feats_86",
        "item_dense_feats_91",
        "item_dense_feats_92",
    ]


def _fit_dense_stats(
    row_groups: Sequence[Tuple[str, int]],
    schema: Dict[str, Any],
    batch_size: int,
    match_col: str,
    match_ts_col: str,
    seq_ts_cols: Dict[str, str],
    min_timestamp: int,
    match_window_seconds: int,
    count_edges: Sequence[int],
) -> Dict[str, RunningStats]:
    dense_names = _dense_feature_names(schema)
    stats = {name: RunningStats() for name in dense_names}
    for name in _generated_dense_names():
        stats[name] = RunningStats()

    state = PrefixState()
    required_columns = [
        *dense_names,
        "user_id",
        "item_id",
        "timestamp",
        "item_int_feats_9",
        match_col,
        match_ts_col,
        *seq_ts_cols.values(),
    ]
    read_columns = _existing_columns(row_groups, required_columns)
    for _, batch in iter_batches(
        row_groups,
        batch_size,
        columns=read_columns,
        progress_label="fit dense stats",
    ):
        names = batch.schema.names
        idx = {name: i for i, name in enumerate(names)}
        for name in dense_names:
            if name in idx:
                stats[name].update_many(_dense_values(batch.column(idx[name])))
        feats = _compute_generated_features(
            batch,
            state,
            match_col,
            match_ts_col,
            seq_ts_cols,
            min_timestamp,
            match_window_seconds,
            count_edges,
        )
        for name in _generated_dense_names():
            stats[name].update_many(feats[name].ravel())
    return stats


def _build_augmented_schema(
    schema: Dict[str, Any],
    audits: Dict[str, IntAudit],
    missing_threshold: float,
    item_dense_fids: Sequence[int],
    match_count_vocab_size: int,
) -> Tuple[Dict[str, Any], List[int], List[int], Dict[str, Any]]:
    out = dict(schema)
    vocab_audit: Dict[str, Any] = {}
    dropped_user_fids: List[int] = []
    dropped_item_fids: List[int] = []

    user_int: List[List[int]] = []
    for fid, vs, dim in schema.get("user_int", []):
        name = column_name("user_int", int(fid))
        audit = audits.get(name, IntAudit())
        if audit.missing_ratio > missing_threshold:
            dropped_user_fids.append(int(fid))
            continue
        user_int.append([int(fid), int(vs), int(dim)])
        vocab_audit[name] = {"vocab_size": int(vs), **audit.to_dict()}
    user_int = _append_feature_specs(user_int, USER_INT_ADDS)

    item_int: List[List[int]] = []
    for fid, vs, dim in schema.get("item_int", []):
        name = column_name("item_int", int(fid))
        audit = audits.get(name, IntAudit())
        if audit.missing_ratio > missing_threshold:
            dropped_item_fids.append(int(fid))
            continue
        item_int.append([int(fid), int(vs), int(dim)])
        vocab_audit[name] = {"vocab_size": int(vs), **audit.to_dict()}
    item_int = _append_feature_specs(
        item_int,
        [*ITEM_INT_ADDS_BASE, (90, match_count_vocab_size, 1), (91, NUM_TIME_BUCKETS, 1)],
    )

    seq_out: Dict[str, Any] = {}
    for domain, cfg in schema.get("seq", {}).items():
        prefix = _strip_prefix(str(cfg.get("prefix", "")))
        ts_fid = int(cfg.get("ts_fid")) if cfg.get("ts_fid") is not None else None
        features = []
        for fid, vs in cfg.get("features", []):
            fid_i = int(fid)
            if ts_fid is not None and fid_i == ts_fid:
                features.append([fid_i, int(vs)])
                continue
            name = f"{prefix}_{fid_i}"
            audit = audits.get(name, IntAudit())
            features.append([fid_i, int(vs)])
            vocab_audit[name] = {"vocab_size": int(vs), **audit.to_dict()}
        new_cfg = dict(cfg)
        new_cfg["features"] = features
        seq_out[domain] = new_cfg

    out["user_int"] = user_int
    out["item_int"] = item_int
    out["seq"] = seq_out
    out["user_dense"] = out.get("user_dense", [])
    out["item_dense"] = _append_feature_specs(
        out.get("item_dense", []),
        [(int(fid), 1) for fid in item_dense_fids],
    )
    return out, dropped_user_fids, dropped_item_fids, vocab_audit


def _build_ns_groups(schema: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "_purpose": "FE-08 May7 NS groups. Dense feature ids enter through dense tokens.",
        "_note": "Use with rankmixer user_ns_tokens=5,item_ns_tokens=2,num_queries=2 and TokenGNN4.",
        "user_ns_groups": {
            "U1_user_profile": [1, 15, 48, 49],
            "U2_user_temporal_behavior": [50, 60, 130, 131],
            "U3_user_context": [51, 52, 53, 54, 55, 56, 57, 58, 59],
            "U4_user_dense_aligned": [62, 63, 64, 65, 66],
            "U5_user_interest": [80, 82, 86],
            "U6_user_long_tail": [89, 90, 91, 92, 93],
            "U7_user_high_cardinality": [
                94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
                104, 105, 106, 107, 108, 109,
            ],
        },
        "item_ns_groups": {
            "I1_item_identity": [5, 6, 7, 8],
            "I2_item_category": [9, 10, 11, 12, 13],
            "I3_item_semantic": [16, 81, 83, 84, 85],
            "I4_target_matching_fields": [89, 90, 91],
        },
    }
    allowed_user = {int(row[0]) for row in schema.get("user_int", [])}
    allowed_item = {int(row[0]) for row in schema.get("item_int", [])}
    out: Dict[str, Any] = {k: v for k, v in base.items() if not k.endswith("_ns_groups")}
    out["user_ns_groups"] = {
        name: [fid for fid in fids if fid in allowed_user]
        for name, fids in base["user_ns_groups"].items()
        if any(fid in allowed_user for fid in fids)
    }
    out["item_ns_groups"] = {
        name: [fid for fid in fids if fid in allowed_item]
        for name, fids in base["item_ns_groups"].items()
        if any(fid in allowed_item for fid in fids)
    }
    return out


def _sequence_columns(schema: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for domain, cfg in schema.get("seq", {}).items():
        prefix = _strip_prefix(str(cfg.get("prefix", "")))
        cols: List[str] = []
        for fid, _ in cfg.get("features", []):
            name = f"{prefix}_{int(fid)}"
            if name not in cols:
                cols.append(name)
        ts_fid = cfg.get("ts_fid")
        if ts_fid is not None:
            ts_name = f"{prefix}_{int(ts_fid)}"
            if ts_name not in cols:
                cols.append(ts_name)
        out[str(domain)] = cols
    return out


def _sort_sequence_columns_by_recency(table: pa.Table, schema: Dict[str, Any]) -> pa.Table:
    names = set(table.schema.names)
    idx = {name: i for i, name in enumerate(table.schema.names)}
    for domain, cfg in schema.get("seq", {}).items():
        prefix = _strip_prefix(str(cfg.get("prefix", "")))
        ts_fid = cfg.get("ts_fid")
        if ts_fid is None:
            continue
        ts_col = f"{prefix}_{int(ts_fid)}"
        if ts_col not in names:
            continue
        cols = [name for name in _sequence_columns({"seq": {domain: cfg}})[str(domain)] if name in names]
        if not cols:
            continue
        ts_arr = table.column(idx[ts_col]).combine_chunks()
        if not pa.types.is_list(ts_arr.type):
            raise TypeError(f"FE-08 sequence timestamp column must be list type: {ts_col}")
        ts_rows = ts_arr.to_pylist()
        permutations: List[Optional[np.ndarray]] = []
        for row in ts_rows:
            if row is None or len(row) <= 1:
                permutations.append(None)
                continue
            ts_values = np.asarray([int(x) if x is not None else 0 for x in row], dtype=np.int64)
            permutations.append(np.argsort(-ts_values, kind="stable"))

        for col_name in cols:
            arr = table.column(idx[col_name]).combine_chunks()
            if not pa.types.is_list(arr.type):
                raise TypeError(f"FE-08 sequence side column must be list type: {col_name}")
            rows = arr.to_pylist()
            sorted_rows: List[Any] = []
            for row_i, (row, perm) in enumerate(zip(rows, permutations)):
                if row is None:
                    sorted_rows.append(None)
                elif perm is None:
                    sorted_rows.append(row)
                else:
                    if len(row) != len(perm):
                        raise ValueError(
                            f"FE-08 sequence sort length mismatch in {col_name} "
                            f"row={row_i}: side_len={len(row)}, ts_len={len(perm)}"
                        )
                    sorted_rows.append([row[int(j)] for j in perm])
            table = _append_or_replace_column(table, col_name, pa.array(sorted_rows, type=arr.type))
            idx = {name: i for i, name in enumerate(table.schema.names)}
            names = set(table.schema.names)
    return table


def _write_augmented_parquet(
    row_groups: Sequence[Tuple[str, int]],
    output_dir: str,
    schema: Dict[str, Any],
    batch_size: int,
    int_stats: Dict[str, IntAudit],
    dense_stats: Dict[str, RunningStats],
    match_col: str,
    match_ts_col: str,
    seq_ts_cols: Dict[str, str],
    min_timestamp: int,
    match_window_seconds: int,
    count_edges: Sequence[int],
    fill_empty_int_lists: bool,
    sort_sequence_by_recency: bool,
) -> None:
    original_dense = set(_dense_feature_names(schema))
    state = PrefixState()
    writers: Dict[str, pq.ParquetWriter] = {}
    try:
        for input_path, batch in iter_batches(row_groups, batch_size, progress_label="write enhanced parquet"):
            names = batch.schema.names
            idx = {name: i for i, name in enumerate(names)}
            table = pa.Table.from_batches([batch])

            for fid, _, _ in schema.get("user_int", []):
                name = column_name("user_int", int(fid))
                if name in idx:
                    fill_value = int_stats.get(name, IntAudit()).fill_value
                    table = _append_or_replace_column(
                        table, name, _fill_int_column(batch.column(idx[name]), fill_value, fill_empty_int_lists))
            for fid, _, _ in schema.get("item_int", []):
                name = column_name("item_int", int(fid))
                if name in idx:
                    fill_value = int_stats.get(name, IntAudit()).fill_value
                    table = _append_or_replace_column(
                        table, name, _fill_int_column(batch.column(idx[name]), fill_value, fill_empty_int_lists))

            for name in original_dense:
                if name in idx and name in dense_stats:
                    table = _append_or_replace_column(
                        table, name, _normalize_dense_column(batch.column(idx[name]), name, dense_stats))

            feats = _compute_generated_features(
                batch,
                state,
                match_col,
                match_ts_col,
                seq_ts_cols,
                min_timestamp,
                match_window_seconds,
                count_edges,
            )
            for name in _generated_dense_names():
                values = _normalize(name, feats[name], dense_stats)
                if values.ndim == 1:
                    arr = pa.array(values, type=pa.float32())
                else:
                    arr = pa.array(values.tolist(), type=pa.list_(pa.float32()))
                table = _append_or_replace_column(table, name, arr)

            for name in ("item_int_feats_89", "item_int_feats_90", "item_int_feats_91"):
                table = _append_or_replace_column(
                    table, name, pa.array(feats[name], type=pa.int64()))
            for name in ("user_int_feats_130", "user_int_feats_131"):
                table = _append_or_replace_column(
                    table, name, pa.array(feats[name], type=pa.int64()))

            if sort_sequence_by_recency:
                table = _sort_sequence_columns_by_recency(table, schema)

            out_path = os.path.join(output_dir, os.path.basename(input_path))
            writer = writers.get(out_path)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
                writers[out_path] = writer
            writer.write_table(table)
    finally:
        for writer in writers.values():
            writer.close()


def parse_edges(raw: str) -> List[int]:
    edges = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(edges) < 2 or edges[0] != 0:
        raise ValueError("--match_count_buckets must start with 0 and include at least one upper edge")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("--match_count_buckets must be strictly increasing")
    return edges


def parse_item_dense_fids(raw: str, allow_risky: bool) -> List[int]:
    fids = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not fids:
        raise ValueError("--item_dense_fids must contain at least one fid")
    for fid in fids:
        if fid in RISKY_ITEM_DENSE_FIDS and not allow_risky:
            raise ValueError(
                f"item_dense_feats_{fid} is RISKY: {RISKY_ITEM_DENSE_FIDS[fid]}. "
                "Pass --enable_risky_item_dense_fids to override."
            )
    return fids


def bucket_boundaries_md5() -> str:
    return hashlib.md5(BUCKET_BOUNDARIES.astype(np.int64).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FE-08 May7 dataset")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--input_schema", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--missing_threshold", type=float, default=0.80)
    ap.add_argument("--fit_stats_row_group_ratio", type=float, default=0.9)
    ap.add_argument("--match_window_days", type=int, default=7)
    ap.add_argument("--match_count_buckets", default="0,1,2,4,8")
    ap.add_argument("--fill_empty_int_lists", action="store_true")
    ap.add_argument("--item_dense_fids", default="86,91,92")
    ap.add_argument("--enable_risky_item_dense_fids", action="store_true")
    ap.add_argument("--sort_sequence_by_recency", action="store_true", default=True)
    ap.add_argument("--no_sort_sequence_by_recency", dest="sort_sequence_by_recency", action="store_false")
    ap.add_argument(
        "--diagnostic_row_group_limit",
        type=int,
        default=100,
        help="Number of row groups used only for label/time diagnostics; 0 means all.",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("--output_dir must be different from --input_dir")

    files = parquet_files(args.input_dir)
    row_groups = parquet_row_groups(files)
    schema = load_json(args.input_schema)
    first_names = pq.ParquetFile(files[0]).schema_arrow.names
    match_col, match_ts_col = _resolve_domain_d_columns(schema, first_names)
    seq_ts_cols = _seq_timestamp_columns(schema)
    count_edges = parse_edges(args.match_count_buckets)
    item_dense_fids = parse_item_dense_fids(args.item_dense_fids, args.enable_risky_item_dense_fids)
    _validate_input_schema_for_fe08(schema, args.enable_risky_item_dense_fids)
    match_window_seconds = args.match_window_days * 86400

    n_fit = max(1, int(len(row_groups) * args.fit_stats_row_group_ratio))
    fit_row_groups = row_groups[:n_fit]

    print(f"[FE-08] input parquet files: {len(files)}")
    print(f"[FE-08] input row groups: {len(row_groups)}; fitting stats on first {len(fit_row_groups)}")
    print(f"[FE-08] match columns: {match_col}, {match_ts_col}")
    print(f"[FE-08] item_dense_fids: {item_dense_fids}")
    print(f"[FE-08] sort_sequence_by_recency: {args.sort_sequence_by_recency}")
    print("[FE-08] collecting int fill/drop audit and label/time diagnostics...")
    if args.diagnostic_row_group_limit and args.diagnostic_row_group_limit > 0:
        diagnostic_row_groups = row_groups[:min(len(row_groups), args.diagnostic_row_group_limit)]
    else:
        diagnostic_row_groups = row_groups
    print(f"[FE-08] diagnostics row groups: {len(diagnostic_row_groups)}")
    _, diagnostics = _collect_audit_and_min_ts(
        diagnostic_row_groups,
        schema,
        args.batch_size,
        collect_feature_audits=False,
        progress_label="collect diagnostics",
    )
    fit_audits, fit_diagnostics = _collect_audit_and_min_ts(
        fit_row_groups,
        schema,
        args.batch_size,
        collect_feature_audits=True,
        include_seq_audits=False,
        progress_label="fit user/item int audit",
    )
    min_timestamp = int(fit_diagnostics["timestamp_min"] or diagnostics["timestamp_min"] or 0)

    print(
        "[FE-08] labels: "
        f"positive(label_type==2)={diagnostics['positive_label_type_2']}, "
        f"negative={diagnostics['negative_not_label_type_2']}, "
        f"label_type_counts={diagnostics['label_type_counts']}"
    )
    print(
        "[FE-08] timestamp: "
        f"min={diagnostics['timestamp_min']}, max={diagnostics['timestamp_max']}"
    )
    if diagnostics["label_time_observed_rows"] > 0:
        print(
            "[FE-08] label_time: "
            f"min={diagnostics['label_time_min']}, max={diagnostics['label_time_max']}, "
            f"observed_rows={diagnostics['label_time_observed_rows']}"
        )
    else:
        print("[FE-08] label_time: not present or no positive label_time values")

    print("[FE-08] fitting dense normalization stats...")
    dense_stats = _fit_dense_stats(
        fit_row_groups,
        schema,
        args.batch_size,
        match_col,
        match_ts_col,
        seq_ts_cols,
        min_timestamp,
        match_window_seconds,
        count_edges,
    )

    augmented_schema, dropped_user_fids, dropped_item_fids, vocab_audit = _build_augmented_schema(
        schema,
        fit_audits,
        args.missing_threshold,
        item_dense_fids,
        match_count_vocab_size=len(count_edges) + 1,
    )
    output_ns_groups = _build_ns_groups(augmented_schema)

    print(f"[FE-08] dropped high-missing user_int fids: {dropped_user_fids}")
    print(f"[FE-08] dropped high-missing item_int fids: {dropped_item_fids}")
    print("[FE-08] writing enhanced parquet...")
    _write_augmented_parquet(
        row_groups,
        args.output_dir,
        schema,
        args.batch_size,
        fit_audits,
        dense_stats,
        match_col,
        match_ts_col,
        seq_ts_cols,
        min_timestamp,
        match_window_seconds,
        count_edges,
        args.fill_empty_int_lists,
        args.sort_sequence_by_recency,
    )

    write_json(os.path.join(args.output_dir, "schema.json"), augmented_schema)
    write_json(os.path.join(args.output_dir, "ns_groups.may7.json"), output_ns_groups)
    write_json(
        os.path.join(args.output_dir, "dropped_feats.may7.json"),
        {
            "user_int": dropped_user_fids,
            "item_int": dropped_item_fids,
            "threshold": args.missing_threshold,
            "row_groups_used_for_audit": len(fit_row_groups),
            "interpretation": "Empty list means no fid hit the missing>0.80 threshold.",
        },
    )
    dense_stats_payload = {name: stat.to_dict() for name, stat in dense_stats.items()}
    write_json(
        os.path.join(args.output_dir, "fe08_dense_normalization_stats.json"),
        dense_stats_payload,
    )
    write_json(
        os.path.join(args.output_dir, "fe08_transform_stats.json"),
        {
            "experiment": "FE-08",
            "feature_set": "fe08_may7_gnn",
            "int_missing_fill": "FE00-literal average fill for user/item int features",
            "fill_empty_int_lists": args.fill_empty_int_lists,
            "dense_stats": dense_stats_payload,
            "int_fill_audit": vocab_audit,
            "dropped_user_int_fids": dropped_user_fids,
            "dropped_item_int_fids": dropped_item_fids,
            "user_int_drops": dropped_user_fids,
            "item_int_drops": dropped_item_fids,
            "missing_threshold": args.missing_threshold,
            "min_timestamp_for_day_since": min_timestamp,
            "match_col": match_col,
            "match_ts_col": match_ts_col,
            "domain_match_columns": {"match_col": match_col, "match_ts_col": match_ts_col},
            "match_window_days": args.match_window_days,
            "match_count_buckets": count_edges,
            "match_count_vocab_size": len(count_edges) + 1,
            "seq_time_columns": seq_ts_cols,
            "num_time_buckets": NUM_TIME_BUCKETS,
            "bucket_boundaries_md5": bucket_boundaries_md5(),
            "item_dense_fids": item_dense_fids,
            "enable_risky_item_dense_fids": bool(args.enable_risky_item_dense_fids),
            "sequence_sort_by_recency": bool(args.sort_sequence_by_recency),
            "sort_order": "descending_by_event_timestamp" if args.sort_sequence_by_recency else "original",
            "fit_stats_row_group_ratio": args.fit_stats_row_group_ratio,
            "fit_stats_row_groups": len(fit_row_groups),
            "fit_row_groups": len(fit_row_groups),
            "diagnostic_row_groups": len(diagnostic_row_groups),
            "transform_batch_size": args.batch_size,
            "total_row_groups": len(row_groups),
            "input_diagnostics": diagnostics,
            "fit_diagnostics": fit_diagnostics,
            "excluded_features": [
                "user_dense_feats_111 purchase frequency",
                "item_dense_feats_87 purchase frequency",
                "item_dense_feats_88 avg delay / label_time input",
                "FE-07 P2-Domain domain_time_bucket_boundaries",
                "FE-07 user_dense_feats_110/120/121 mainline defaults",
                "user/item avg delay FE-02 features",
                "direct final NS head",
            ],
        },
    )
    write_json(
        os.path.join(args.output_dir, "feature_engineering_stats.json"),
        {
            "experiment": "FE-08",
            "dense_stats": dense_stats_payload,
            "match_col": match_col,
            "match_ts_col": match_ts_col,
            "match_window_days": args.match_window_days,
            "match_count_buckets": count_edges,
            "seq_time_columns": seq_ts_cols,
            "min_timestamp_for_day_since": min_timestamp,
        },
    )
    write_json(
        os.path.join(args.output_dir, "docx_alignment.fe08.json"),
        {
            "experiment": "FE-08",
            "source_of_truth": [
                "experiment_plans/Claude/5月7日_GNN结合验证特征_保持NS影响域_方案.md",
                "experiment_plans/Claude/5月7日_FE08代码结构_供AI_Agent搭建.md",
            ],
            "item_dense_fids": item_dense_fids,
            "kept_from_fe01b_safe_set": [
                "item_int_feats_89",
                "item_int_feats_90",
                "item_dense_feats_91",
                "item_dense_feats_92",
            ],
            "may7_added": [
                "item_int_feats_91 latest match time bucket",
                "user_int_feats_130 hour_of_day",
                "user_int_feats_131 day_of_week",
                "sequence sort by recency",
            ],
            "explicitly_excluded": [
                "item_dense_feats_87 purchase frequency",
                "item_dense_feats_88 avg delay",
                "FE-07 P2-Domain",
                "output_include_ns",
                "current label_time input",
            ],
        },
    )
    print(f"[FE-08] wrote dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()
