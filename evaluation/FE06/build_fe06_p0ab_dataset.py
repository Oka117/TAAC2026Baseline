"""Build FE-06 P0AB dataset.

FE-06 combines:

- FE-00 dense normalization and high-missing user-int schema drop.
- P0-L1 sparse-id vocab shift: 0=padding, 1=missing, k+1=original id k.
- FE01AB-safe features: total frequency 110/86 plus target-history match
  89/90/91/92, excluding purchase-frequency 111/87.
- Claude P0 dense blocks:
  user_dense_feats_120 = current timestamp context, dim=3.
  user_dense_feats_121 = domain seq_len/window-count summary, dim=20.

The script writes an enhanced parquet dataset, schema.json,
ns_groups.feature_engineering.json, fe06_transform_stats.json and audit files.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


USER_DENSE_ADDS = [(110, 1), (120, 3), (121, 20)]
ITEM_DENSE_ADDS = [(86, 1), (91, 1), (92, 1)]
ITEM_INT_ADDS_BASE = [(89, 3, 1)]
SEQ_WINDOWS = (3600, 86400, 604800, 2592000)


@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update_many(self, values: Iterable[float]) -> None:
        if isinstance(values, np.ndarray):
            self.update_array(values)
            return
        for raw in values:
            value = float(raw)
            if not math.isfinite(value):
                continue
            self.n += 1
            delta = value - self.mean
            self.mean += delta / self.n
            delta2 = value - self.mean
            self.m2 += delta * delta2

    def update_array(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        batch_n = int(arr.size)
        if batch_n == 0:
            return
        batch_mean = float(arr.mean())
        batch_m2 = float(np.square(arr - batch_mean).sum())
        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        total = self.n + batch_n
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + delta * delta * self.n * batch_n / total
        self.mean += delta * batch_n / total
        self.n = total

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

    @property
    def missing_ratio(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return self.missing_rows / self.total_rows

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "missing_rows": self.missing_rows,
            "missing_ratio": self.missing_ratio,
            "max_positive": self.max_positive,
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


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def iter_batches(
    row_groups: Sequence[Tuple[str, int]],
    batch_size: int,
    columns: Optional[Sequence[str]] = None,
) -> Iterable[Tuple[str, pa.RecordBatch]]:
    read_columns = None if columns is None else list(dict.fromkeys(columns))
    for path, rg_idx in row_groups:
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
        if len(col.values) == 0:
            return values
        arr = _to_int_array(col.values)
        positives = arr[arr > 0]
        values.extend(int(x) for x in positives)
    else:
        arr = _to_int_array(col)
        values.extend(int(x) for x in arr if int(x) > 0)
    return values


def _audit_sparse_column(col: pa.Array) -> IntAudit:
    audit = IntAudit(total_rows=len(col))
    if pa.types.is_list(col.type):
        if len(col) == 0:
            return audit
        offsets = col.offsets.to_numpy(zero_copy_only=False)
        values = _to_int_array(col.values) if len(col.values) else np.asarray([], dtype=np.int64)
        positives = values > 0
        if positives.any():
            audit.max_positive = int(values[positives].max())
            positive_idx = np.flatnonzero(positives)
            left = np.searchsorted(positive_idx, offsets[:-1], side="left")
            right = np.searchsorted(positive_idx, offsets[1:], side="left")
            positive_counts = right - left
        else:
            positive_counts = np.zeros(len(col), dtype=np.int64)
        valid = col.is_valid().to_numpy(zero_copy_only=False)
        audit.missing_rows = int((~valid | (positive_counts == 0)).sum())
        return audit

    arr = _to_int_array(col)
    audit.missing_rows = int((arr <= 0).sum())
    positives = arr[arr > 0]
    if positives.size:
        audit.max_positive = int(positives.max())
    return audit


def _missing_rows(col: pa.Array) -> int:
    if pa.types.is_list(col.type):
        if len(col) == 0:
            return 0
        offsets = col.offsets.to_numpy(zero_copy_only=False)
        values = _to_int_array(col.values) if len(col.values) else np.asarray([], dtype=np.int64)
        positives = values > 0
        valid = col.is_valid().to_numpy(zero_copy_only=False)
        missing = 0
        for i in range(len(col)):
            if not bool(valid[i]) or not positives[offsets[i]:offsets[i + 1]].any():
                missing += 1
        return missing
    arr = _to_int_array(col)
    return int((arr <= 0).sum())


def _dense_values(col: pa.Array) -> np.ndarray:
    if pa.types.is_list(col.type):
        if len(col.values) == 0:
            return np.asarray([], dtype=np.float32)
        arr = _to_float_array(col.values)
    else:
        arr = _to_float_array(col)
    return arr[np.isfinite(arr)]


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


def _collect_label_time_diagnostics(
    row_groups: Sequence[Tuple[str, int]],
    batch_size: int,
) -> Dict[str, Any]:
    label_counts: Counter[int] = Counter()
    total_rows = 0
    pos = 0
    neg = 0
    ts_min: Optional[int] = None
    ts_max: Optional[int] = None
    lt_min: Optional[int] = None
    lt_max: Optional[int] = None
    lt_observed = 0

    columns = _existing_columns(row_groups, ["label_type", "timestamp", "label_time"])
    for _, batch in iter_batches(row_groups, batch_size, columns=columns):
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

    return {
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


def _collect_user_int_audit(
    row_groups: Sequence[Tuple[str, int]],
    schema: Dict[str, Any],
    batch_size: int,
) -> Dict[str, IntAudit]:
    user_names = [column_name("user_int", int(fid)) for fid, _, _ in schema.get("user_int", [])]
    read_columns = _existing_columns(row_groups, user_names)
    audits = {name: IntAudit() for name in user_names}
    if not read_columns:
        return audits

    for processed, (_, batch) in enumerate(iter_batches(row_groups, batch_size, columns=read_columns), start=1):
        names = batch.schema.names
        idx = {name: i for i, name in enumerate(names)}
        for name in user_names:
            if name not in idx:
                continue
            cur = _audit_sparse_column(batch.column(idx[name]))
            audit = audits[name]
            audit.total_rows += cur.total_rows
            audit.missing_rows += cur.missing_rows
            audit.max_positive = max(audit.max_positive, cur.max_positive)
        if processed % 100 == 0:
            print(f"[FE-06] user_int audit progress: {processed}/{len(row_groups)} row groups", flush=True)
    return audits


def _shift_int_values_array(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.int64, copy=True)
    missing = out <= 0
    out[missing] = 1
    out[~missing] += 1
    return out


def _shift_sparse_column(col: pa.Array, is_sequence: bool = False) -> pa.Array:
    if pa.types.is_list(col.type):
        if is_sequence and col.null_count == 0:
            values = _to_int_array(col.values) if len(col.values) else np.asarray([], dtype=np.int64)
            shifted = _shift_int_values_array(values)
            return pa.ListArray.from_arrays(
                col.offsets,
                pa.array(shifted, type=pa.int64()),
            )
        shifted_rows: List[List[int]] = []
        for row in col.to_pylist():
            if row is None:
                shifted_rows.append([] if is_sequence else [1])
                continue
            vals = []
            for x in row:
                if x is None or int(x) <= 0:
                    vals.append(1)
                else:
                    vals.append(int(x) + 1)
            if not vals and not is_sequence:
                vals = [1]
            shifted_rows.append(vals)
        return pa.array(shifted_rows, type=pa.list_(pa.int64()))
    arr = _to_int_array(col)
    return pa.array(_shift_int_values_array(arr), type=pa.int64())


def _normalize_dense_column(col: pa.Array, name: str, stats: Dict[str, RunningStats]) -> pa.Array:
    tracker = stats[name]
    if pa.types.is_list(col.type):
        if col.null_count == 0:
            values = _to_float_array(col.values) if len(col.values) else np.asarray([], dtype=np.float32)
            values = np.nan_to_num(values, nan=tracker.mean, posinf=tracker.mean, neginf=tracker.mean)
            normalized = ((values - tracker.mean) / tracker.std).astype(np.float32)
            return pa.ListArray.from_arrays(
                col.offsets,
                pa.array(normalized, type=pa.float32()),
            )
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
        raise KeyError("FE-06 requires item_int_feats_9 for target-history match")
    target_item_attr = _first_scalar_from_maybe_list(batch.column(idx["item_int_feats_9"]).to_pylist())
    seq_values = _list_values(batch.column(idx[match_col]))
    seq_times = _list_values(batch.column(idx[match_ts_col]))

    user_total = np.zeros(B, dtype=np.float32)
    item_total = np.zeros(B, dtype=np.float32)
    has_match = np.zeros(B, dtype=np.int64)
    match_count = np.zeros(B, dtype=np.int64)
    min_match_delta = np.zeros(B, dtype=np.float32)
    match_count_7d = np.zeros(B, dtype=np.float32)

    hour = ((timestamps // 3600) % 24).astype(np.float32) / 23.0
    dow = (((timestamps // 86400) + 4) % 7).astype(np.float32) / 6.0
    day_since_min = np.log1p(np.maximum(timestamps - int(min_timestamp), 0) // 86400).astype(np.float32)
    time_context = np.stack([hour, dow, day_since_min], axis=1).astype(np.float32)

    seq_summary_parts: List[np.ndarray] = []
    for domain in sorted(seq_ts_cols):
        ts_col = seq_ts_cols[domain]
        if ts_col not in idx:
            seq_summary_parts.append(np.zeros((B, 5), dtype=np.float32))
            continue
        rows = _list_values(batch.column(idx[ts_col]))
        summary = np.zeros((B, 5), dtype=np.float32)
        for i, row in enumerate(rows):
            counts = [0, 0, 0, 0]
            valid = 0
            now = int(timestamps[i])
            for event_time in row:
                if event_time <= 0 or event_time > now:
                    continue
                valid += 1
                delta = now - int(event_time)
                for wi, window in enumerate(SEQ_WINDOWS):
                    if delta <= window:
                        counts[wi] += 1
            summary[i, 0] = math.log1p(valid)
            for wi, count in enumerate(counts):
                summary[i, wi + 1] = math.log1p(count)
        seq_summary_parts.append(summary)
    domain_summary = np.concatenate(seq_summary_parts, axis=1).astype(np.float32)

    row_order = np.argsort(timestamps, kind="stable")
    for i in row_order:
        uid = int(user_ids[i])
        iid = int(item_ids[i])
        ut, it = state.before_update(uid, iid)
        user_total[i] = ut
        item_total[i] = it

        target = int(target_item_attr[i])
        if target > 0:
            deltas: List[int] = []
            count_7d = 0
            for value, event_time in zip(seq_values[i], seq_times[i]):
                if int(value) != target:
                    continue
                match_count[i] += 1
                if event_time > 0:
                    delta = max(int(timestamps[i]) - int(event_time), 0)
                    deltas.append(delta)
                    if delta <= match_window_seconds:
                        count_7d += 1
            has_match[i] = 2 if match_count[i] > 0 else 1
            min_match_delta[i] = min(deltas) if deltas else 0.0
            match_count_7d[i] = count_7d
        else:
            has_match[i] = 0

        state.update(uid, iid)

    return {
        "user_dense_feats_110": np.log1p(user_total).astype(np.float32),
        "item_dense_feats_86": np.log1p(item_total).astype(np.float32),
        "item_int_feats_89": has_match,
        "item_int_feats_90": _bucketize_counts(match_count, count_edges),
        "item_dense_feats_91": np.log1p(min_match_delta).astype(np.float32),
        "item_dense_feats_92": np.log1p(match_count_7d).astype(np.float32),
        "user_dense_feats_120": time_context,
        "user_dense_feats_121": domain_summary,
    }


def _generated_dense_names() -> List[str]:
    return [
        "user_dense_feats_110",
        "user_dense_feats_120",
        "user_dense_feats_121",
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
    stats = {name: RunningStats() for name in _dense_feature_names(schema)}
    for name in _generated_dense_names():
        stats[name] = RunningStats()

    needed_columns = [
        "user_id",
        "item_id",
        "timestamp",
        "item_int_feats_9",
        match_col,
        match_ts_col,
        *_dense_feature_names(schema),
        *seq_ts_cols.values(),
    ]
    read_columns = _existing_columns(row_groups, needed_columns)
    state = PrefixState()
    for processed, (_, batch) in enumerate(
        iter_batches(row_groups, batch_size, columns=read_columns),
        start=1,
    ):
        names = batch.schema.names
        idx = {name: i for i, name in enumerate(names)}
        for name in _dense_feature_names(schema):
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
        if processed % 100 == 0:
            print(f"[FE-06] dense stats progress: {processed}/{len(row_groups)} row groups", flush=True)
    return stats


def _new_vocab_size(old_vocab_size: int, max_positive: int) -> int:
    return max(int(old_vocab_size) + 1, int(max_positive) + 2, 2)


def _build_augmented_schema(
    schema: Dict[str, Any],
    audits: Dict[str, IntAudit],
    missing_threshold: float,
    match_count_vocab_size: int,
) -> Tuple[Dict[str, Any], List[int], Dict[str, Any]]:
    out = dict(schema)
    vocab_audit: Dict[str, Any] = {}
    dropped_user_fids: List[int] = []

    user_int: List[List[int]] = []
    for fid, vs, dim in schema.get("user_int", []):
        name = column_name("user_int", int(fid))
        audit = audits.get(name, IntAudit())
        if audit.missing_ratio > missing_threshold:
            dropped_user_fids.append(int(fid))
            continue
        new_vs = _new_vocab_size(int(vs), audit.max_positive)
        user_int.append([int(fid), new_vs, int(dim)])
        vocab_audit[name] = {"old_vocab_size": int(vs), "new_vocab_size": new_vs, **audit.to_dict()}

    item_int: List[List[int]] = []
    for fid, vs, dim in schema.get("item_int", []):
        name = column_name("item_int", int(fid))
        audit = audits.get(name, IntAudit())
        new_vs = _new_vocab_size(int(vs), audit.max_positive)
        item_int.append([int(fid), new_vs, int(dim)])
        vocab_audit[name] = {"old_vocab_size": int(vs), "new_vocab_size": new_vs, **audit.to_dict()}
    item_int = _append_feature_specs(item_int, [*ITEM_INT_ADDS_BASE, (90, match_count_vocab_size, 1)])

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
            new_vs = _new_vocab_size(int(vs), audit.max_positive)
            features.append([fid_i, new_vs])
            vocab_audit[name] = {"old_vocab_size": int(vs), "new_vocab_size": new_vs, **audit.to_dict()}
        new_cfg = dict(cfg)
        new_cfg["features"] = features
        seq_out[domain] = new_cfg

    out["user_int"] = user_int
    out["item_int"] = item_int
    out["seq"] = seq_out
    out["user_dense"] = _append_feature_specs(out.get("user_dense", []), USER_DENSE_ADDS)
    out["item_dense"] = _append_feature_specs(out.get("item_dense", []), ITEM_DENSE_ADDS)
    return out, dropped_user_fids, vocab_audit


def _build_ns_groups(schema: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "_purpose": "FE-06 NS groups. Dense feature ids are omitted and enter through dense tokens.",
        "_note": "Use with rankmixer user_ns_tokens=6,item_ns_tokens=4,num_queries=1 for FE06 full.",
        "user_ns_groups": {
            "U1_user_profile": [1, 15, 48, 49],
            "U2_user_behavior_stats": [50, 60],
            "U3_user_context": [51, 52, 53, 54, 55, 56, 57, 58, 59],
            "U4_user_temporal_behavior": [62, 63, 64, 65, 66],
            "U5_user_interest_ids": [80, 82, 86],
            "U6_user_long_tail_sparse": [89, 90, 91, 92, 93],
            "U7_user_high_cardinality": [
                94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
                104, 105, 106, 107, 108, 109,
            ],
        },
        "item_ns_groups": {
            "I1_item_identity": [5, 6, 7, 8],
            "I2_item_category_brand": [9, 10, 11, 12, 13],
            "I3_item_semantic_sparse": [16, 81, 83, 84, 85],
            "I4_target_matching_fields": [89, 90],
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


def _write_augmented_parquet(
    row_groups: Sequence[Tuple[str, int]],
    output_dir: str,
    schema: Dict[str, Any],
    batch_size: int,
    dense_stats: Dict[str, RunningStats],
    match_col: str,
    match_ts_col: str,
    seq_ts_cols: Dict[str, str],
    min_timestamp: int,
    match_window_seconds: int,
    count_edges: Sequence[int],
) -> None:
    original_dense = set(_dense_feature_names(schema))
    seq_ts_fids = {
        _strip_prefix(str(cfg.get("prefix", ""))): cfg.get("ts_fid")
        for cfg in schema.get("seq", {}).values()
    }
    state = PrefixState()
    writers: Dict[str, pq.ParquetWriter] = {}
    try:
        for processed, (input_path, batch) in enumerate(iter_batches(row_groups, batch_size), start=1):
            names = batch.schema.names
            idx = {name: i for i, name in enumerate(names)}
            table = pa.Table.from_batches([batch])

            for fid, _, _ in schema.get("user_int", []):
                name = column_name("user_int", int(fid))
                if name in idx:
                    table = _append_or_replace_column(
                        table, name, _shift_sparse_column(batch.column(idx[name]), is_sequence=False))
            for fid, _, _ in schema.get("item_int", []):
                name = column_name("item_int", int(fid))
                if name in idx:
                    table = _append_or_replace_column(
                        table, name, _shift_sparse_column(batch.column(idx[name]), is_sequence=False))

            for cfg in schema.get("seq", {}).values():
                prefix = _strip_prefix(str(cfg.get("prefix", "")))
                ts_fid = cfg.get("ts_fid")
                for fid, _ in cfg.get("features", []):
                    if ts_fid is not None and int(fid) == int(ts_fid):
                        continue
                    name = f"{prefix}_{int(fid)}"
                    if name in idx:
                        table = _append_or_replace_column(
                            table, name, _shift_sparse_column(batch.column(idx[name]), is_sequence=True))

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

            for name in ("item_int_feats_89", "item_int_feats_90"):
                table = _append_or_replace_column(
                    table, name, pa.array(feats[name], type=pa.int64()))

            out_path = os.path.join(output_dir, os.path.basename(input_path))
            writer = writers.get(out_path)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
                writers[out_path] = writer
            writer.write_table(table)
            if processed % 100 == 0:
                print(f"[FE-06] write progress: {processed}/{len(row_groups)} row groups", flush=True)
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FE-06 P0AB dataset")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--input_schema", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--missing_threshold", type=float, default=0.75)
    ap.add_argument("--fit_stats_row_group_ratio", type=float, default=0.9)
    ap.add_argument("--match_window_days", type=int, default=7)
    ap.add_argument("--match_count_buckets", default="0,1,2,4,8")
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
    match_window_seconds = args.match_window_days * 86400

    n_fit = max(1, int(len(row_groups) * args.fit_stats_row_group_ratio))
    fit_row_groups = row_groups[:n_fit]

    print(f"[FE-06] input parquet files: {len(files)}")
    print(f"[FE-06] input row groups: {len(row_groups)}; fitting stats on first {len(fit_row_groups)}")
    print(f"[FE-06] match columns: {match_col}, {match_ts_col}")
    print("[FE-06] collecting label/time diagnostics...")
    diagnostics = _collect_label_time_diagnostics(row_groups, args.batch_size)
    fit_diagnostics = _collect_label_time_diagnostics(fit_row_groups, args.batch_size)
    min_timestamp = int(fit_diagnostics["timestamp_min"] or diagnostics["timestamp_min"] or 0)

    print(
        "[FE-06] labels: "
        f"positive(label_type==2)={diagnostics['positive_label_type_2']}, "
        f"negative={diagnostics['negative_not_label_type_2']}, "
        f"label_type_counts={diagnostics['label_type_counts']}"
    )
    print(
        "[FE-06] timestamp: "
        f"min={diagnostics['timestamp_min']}, max={diagnostics['timestamp_max']}"
    )
    if diagnostics["label_time_observed_rows"] > 0:
        print(
            "[FE-06] label_time: "
            f"min={diagnostics['label_time_min']}, max={diagnostics['label_time_max']}, "
            f"observed_rows={diagnostics['label_time_observed_rows']}"
        )
    else:
        print("[FE-06] label_time: not present or no positive label_time values")

    print("[FE-06] collecting high-missing user_int audit...")
    fit_audits = _collect_user_int_audit(fit_row_groups, schema, args.batch_size)

    print("[FE-06] fitting dense normalization stats...")
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

    augmented_schema, dropped_user_fids, vocab_audit = _build_augmented_schema(
        schema,
        fit_audits,
        args.missing_threshold,
        match_count_vocab_size=len(count_edges) + 1,
    )
    output_ns_groups = _build_ns_groups(augmented_schema)

    print(f"[FE-06] dropped high-missing user_int fids: {dropped_user_fids}")
    print("[FE-06] writing enhanced parquet...")
    _write_augmented_parquet(
        row_groups,
        args.output_dir,
        schema,
        args.batch_size,
        dense_stats,
        match_col,
        match_ts_col,
        seq_ts_cols,
        min_timestamp,
        match_window_seconds,
        count_edges,
    )

    write_json(os.path.join(args.output_dir, "schema.json"), augmented_schema)
    write_json(os.path.join(args.output_dir, "ns_groups.feature_engineering.json"), output_ns_groups)
    write_json(os.path.join(args.output_dir, "dropped_user_int_fids.json"), dropped_user_fids)
    write_json(
        os.path.join(args.output_dir, "fe06_transform_stats.json"),
        {
            "experiment": "FE-06",
            "feature_set": "fe06_p0ab_gnn4",
            "p0_l1": "vocab_shift: 0 padding, 1 missing, k+1 original id k",
            "dense_stats": {name: stat.to_dict() for name, stat in dense_stats.items()},
            "vocab_shift_audit": vocab_audit,
            "dropped_user_int_fids": dropped_user_fids,
            "min_timestamp_for_day_since": min_timestamp,
            "match_col": match_col,
            "match_ts_col": match_ts_col,
            "match_window_days": args.match_window_days,
            "match_count_buckets": count_edges,
            "seq_time_columns": seq_ts_cols,
            "fit_stats_row_group_ratio": args.fit_stats_row_group_ratio,
            "fit_stats_row_groups": len(fit_row_groups),
            "total_row_groups": len(row_groups),
            "input_diagnostics": diagnostics,
            "fit_diagnostics": fit_diagnostics,
            "excluded_features": [
                "user_dense_feats_111 purchase frequency",
                "item_dense_feats_87 purchase frequency",
                "user/item avg delay FE-02 features",
                "direct final NS head",
            ],
        },
    )
    write_json(
        os.path.join(args.output_dir, "docx_alignment.fe06.json"),
        {
            "experiment": "FE-06",
            "kept_from_fe01a": ["user_dense_feats_110", "item_dense_feats_86"],
            "kept_from_fe01b": [
                "item_int_feats_89",
                "item_int_feats_90",
                "item_dense_feats_91",
                "item_dense_feats_92",
            ],
            "p0_added": [
                "sparse vocab shift 0/1/k+1",
                "user_dense_feats_120 timestamp context",
                "user_dense_feats_121 domain seq_len/window count summary",
            ],
            "gnn_training_flags": {
                "use_token_gnn": True,
                "token_gnn_layers": 4,
                "token_gnn_graph": "full",
                "token_gnn_layer_scale": 0.15,
            },
        },
    )
    print(f"[FE-06] wrote dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()
