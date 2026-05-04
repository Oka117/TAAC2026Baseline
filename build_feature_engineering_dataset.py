"""Build FE-01 / FE-01A / FE-01B / FE-02 feature-engineering parquet datasets.

This script implements the FE-01 plan in
``experiment_plans/FE01/experiment_01_feature_engineering_plan.zh.md`` and,
when ``--enable_delay_history`` is set, the incremental FE-02 plan in
``experiment_plans/FE02/experiment_02_delay_history_features_plan.zh.md``.

It streams parquet files in sorted file / row-group order and creates:

- user_dense_feats_110: log1p(prefix user total frequency), z-scored
- user_dense_feats_111: log1p(prefix user purchase frequency), z-scored
- user_dense_feats_112: log1p(prefix user avg delay), z-scored (FE-02 only)
- item_dense_feats_86: log1p(prefix item total frequency), z-scored
- item_dense_feats_87: log1p(prefix item purchase frequency), z-scored
- item_dense_feats_88: log1p(prefix item avg delay), z-scored (FE-02 only)
- item_int_feats_89: has_match(item_int_feats_9, domain_d_seq_19)
- item_int_feats_90: bucketized match_count(item_int_feats_9, domain_d_seq_19)
- item_dense_feats_91: log1p(min_match_delta), z-scored
- item_dense_feats_92: log1p(match_count_7d), z-scored

The prefix frequency features are computed from rows already seen by the
streaming pass. For leakage safety, feed this script data ordered by timestamp
or by the same historical order used by training.

FE-02 delay history also uses only previously seen conversion rows. It follows
the uploaded DOCX definition ``delay = timestamp - label_time``; negative delay
values are counted in the audit stats and clipped to zero before averaging so
that ``log1p`` stays well-defined.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


USER_DENSE_ADDS = [(110, 1), (111, 1)]
USER_DELAY_DENSE_ADDS = [(112, 1)]
ITEM_DENSE_ADDS = [(86, 1), (87, 1), (91, 1), (92, 1)]
ITEM_DELAY_DENSE_ADDS = [(88, 1)]
ITEM_INT_BASE_ADDS = [(89, 3, 1)]

BASE_DENSE_FEATURE_NAMES = [
    "user_dense_feats_110",
    "user_dense_feats_111",
    "item_dense_feats_86",
    "item_dense_feats_87",
    "item_dense_feats_91",
    "item_dense_feats_92",
]
DELAY_DENSE_FEATURE_NAMES = [
    "user_dense_feats_112",
    "item_dense_feats_88",
]

FEATURE_SET_TO_EXPERIMENT = {
    "fe01": "FE-01",
    "fe01a": "FE-01A",
    "fe01b": "FE-01B",
}


@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update_many(self, values: np.ndarray) -> None:
        for value in values.astype(np.float64, copy=False):
            self.n += 1
            delta = float(value) - self.mean
            self.mean += delta / self.n
            delta2 = float(value) - self.mean
            self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 1.0
        variance = self.m2 / (self.n - 1)
        return max(math.sqrt(max(variance, 0.0)), 1e-6)

    def to_dict(self) -> Dict[str, float]:
        return {"n": self.n, "mean": self.mean, "std": self.std}


class PrefixState:
    def __init__(self) -> None:
        self.user_total: DefaultDict[int, int] = defaultdict(int)
        self.user_purchase: DefaultDict[int, int] = defaultdict(int)
        self.item_total: DefaultDict[int, int] = defaultdict(int)
        self.item_purchase: DefaultDict[int, int] = defaultdict(int)
        self.user_delay_sum: DefaultDict[int, float] = defaultdict(float)
        self.user_delay_count: DefaultDict[int, int] = defaultdict(int)
        self.item_delay_sum: DefaultDict[int, float] = defaultdict(float)
        self.item_delay_count: DefaultDict[int, int] = defaultdict(int)
        self.delay_observed_count: int = 0
        self.delay_negative_count: int = 0

    def before_update(
        self,
        user_id: int,
        item_id: int,
    ) -> Tuple[int, int, int, int, float, float]:
        user_delay_count = self.user_delay_count[user_id]
        item_delay_count = self.item_delay_count[item_id]
        user_avg_delay = (
            self.user_delay_sum[user_id] / user_delay_count
            if user_delay_count > 0 else 0.0
        )
        item_avg_delay = (
            self.item_delay_sum[item_id] / item_delay_count
            if item_delay_count > 0 else 0.0
        )
        return (
            self.user_total[user_id],
            self.user_purchase[user_id],
            self.item_total[item_id],
            self.item_purchase[item_id],
            user_avg_delay,
            item_avg_delay,
        )

    def update(
        self,
        user_id: int,
        item_id: int,
        is_purchase: bool,
        timestamp: Optional[int] = None,
        label_time: Optional[int] = None,
        collect_delay: bool = False,
    ) -> None:
        self.user_total[user_id] += 1
        self.item_total[item_id] += 1
        if is_purchase:
            self.user_purchase[user_id] += 1
            self.item_purchase[item_id] += 1
            if collect_delay and timestamp is not None and label_time is not None and label_time > 0:
                raw_delay = int(timestamp) - int(label_time)
                self.delay_observed_count += 1
                if raw_delay < 0:
                    self.delay_negative_count += 1
                    raw_delay = 0
                self.user_delay_sum[user_id] += float(raw_delay)
                self.user_delay_count[user_id] += 1
                self.item_delay_sum[item_id] += float(raw_delay)
                self.item_delay_count[item_id] += 1

    def delay_quality(self) -> Dict[str, Any]:
        ratio = (
            self.delay_negative_count / self.delay_observed_count
            if self.delay_observed_count > 0 else 0.0
        )
        return {
            "observed_positive_label_with_label_time": self.delay_observed_count,
            "negative_delay_count": self.delay_negative_count,
            "negative_delay_ratio": ratio,
            "delay_formula": "timestamp - label_time",
            "negative_delay_policy": "clip_to_zero_before_log1p_avg",
        }


def _parquet_files(input_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    return files


def _parquet_row_groups(files: Sequence[str]) -> List[Tuple[str, int]]:
    row_groups: List[Tuple[str, int]] = []
    for path in files:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.metadata.num_row_groups):
            row_groups.append((path, rg_idx))
    if not row_groups:
        raise ValueError("Input parquet files contain no row groups")
    return row_groups


def _load_schema(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _append_feature_specs(
    specs: Sequence[Sequence[int]],
    additions: Sequence[Sequence[int]],
) -> List[List[int]]:
    by_fid = {int(row[0]): list(map(int, row)) for row in specs}
    for row in additions:
        by_fid[int(row[0])] = list(map(int, row))
    return [by_fid[fid] for fid in sorted(by_fid)]


def normalize_feature_set(feature_set: str) -> str:
    normalized = feature_set.strip().lower()
    if normalized not in FEATURE_SET_TO_EXPERIMENT:
        raise ValueError(
            "--feature_set must be one of: "
            + ", ".join(sorted(FEATURE_SET_TO_EXPERIMENT))
        )
    return normalized


def selected_user_dense_adds(feature_set: str, enable_delay_history: bool) -> List[Tuple[int, int]]:
    if feature_set == "fe01a":
        adds: List[Tuple[int, int]] = [(110, 1)]
    elif feature_set == "fe01b":
        adds = []
    else:
        adds = list(USER_DENSE_ADDS)
    if enable_delay_history:
        adds.extend(USER_DELAY_DENSE_ADDS)
    return adds


def selected_item_dense_adds(feature_set: str, enable_delay_history: bool) -> List[Tuple[int, int]]:
    if feature_set == "fe01a":
        adds: List[Tuple[int, int]] = [(86, 1)]
    elif feature_set == "fe01b":
        adds = [(91, 1), (92, 1)]
    else:
        adds = list(ITEM_DENSE_ADDS)
    if enable_delay_history:
        adds.extend(ITEM_DELAY_DENSE_ADDS)
    return adds


def selected_item_int_adds(feature_set: str, match_count_vocab_size: int) -> List[Tuple[int, int, int]]:
    if feature_set == "fe01a":
        return []
    return [*ITEM_INT_BASE_ADDS, (90, match_count_vocab_size, 1)]


def dense_feature_names(feature_set: str, enable_delay_history: bool) -> List[str]:
    user_names = [f"user_dense_feats_{fid}" for fid, _ in selected_user_dense_adds(feature_set, False)]
    item_names = [f"item_dense_feats_{fid}" for fid, _ in selected_item_dense_adds(feature_set, False)]
    names = user_names + item_names
    if enable_delay_history:
        names.extend(DELAY_DENSE_FEATURE_NAMES)
    return names


def item_int_feature_names(feature_set: str) -> List[str]:
    if feature_set == "fe01a":
        return []
    return ["item_int_feats_89", "item_int_feats_90"]


def build_augmented_schema(
    schema: Dict[str, Any],
    match_count_vocab_size: int,
    feature_set: str = "fe01",
    enable_delay_history: bool = False,
) -> Dict[str, Any]:
    feature_set = normalize_feature_set(feature_set)
    out = dict(schema)
    user_dense_adds = selected_user_dense_adds(feature_set, enable_delay_history)
    item_dense_adds = selected_item_dense_adds(feature_set, enable_delay_history)
    item_int_adds = selected_item_int_adds(feature_set, match_count_vocab_size)
    out["user_dense"] = _append_feature_specs(out.get("user_dense", []), user_dense_adds)
    out["item_dense"] = _append_feature_specs(out.get("item_dense", []), item_dense_adds)
    out["item_int"] = _append_feature_specs(out.get("item_int", []), item_int_adds)
    return out


def build_ns_groups() -> Dict[str, Any]:
    return {
        "_purpose": "FE-01/FE-02 NS groups. Dense feature ids are intentionally omitted; they enter through user/item dense tokens.",
        "_note": "Use with --ns_tokenizer_type rankmixer --user_ns_tokens 6 --item_ns_tokens 4 --num_queries 1 when item_dense is enabled.",
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


def filter_ns_groups(ns_groups: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    allowed_user_fids = {int(row[0]) for row in schema.get("user_int", [])}
    allowed_item_fids = {int(row[0]) for row in schema.get("item_int", [])}
    out: Dict[str, Any] = {
        k: v for k, v in ns_groups.items()
        if not k.startswith("user_ns_groups") and not k.startswith("item_ns_groups")
    }
    user_groups = {}
    for name, fids in ns_groups.get("user_ns_groups", {}).items():
        kept = [int(fid) for fid in fids if int(fid) in allowed_user_fids]
        if kept:
            user_groups[name] = kept
    item_groups = {}
    for name, fids in ns_groups.get("item_ns_groups", {}).items():
        kept = [int(fid) for fid in fids if int(fid) in allowed_item_fids]
        if kept:
            item_groups[name] = kept
    out["user_ns_groups"] = user_groups
    out["item_ns_groups"] = item_groups
    return out


def _strip_prefix(prefix: str) -> str:
    return prefix[:-1] if prefix.endswith("_") else prefix


def resolve_domain_d_columns(schema: Dict[str, Any], parquet_names: Sequence[str]) -> Tuple[str, str]:
    """Return ``(domain_d_seq_19_col, domain_d_timestamp_col)``."""
    names = set(parquet_names)
    seq_cfg = schema.get("seq", {})

    candidates: List[Tuple[str, Optional[int]]] = []
    for domain, cfg in seq_cfg.items():
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
        "Could not resolve domain_d sequence columns. Expected "
        "domain_d_seq_19 and a timestamp column such as domain_d_seq_26."
    )


def _to_int_array(arr: pa.Array) -> np.ndarray:
    return arr.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=True)


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


def _list_values(arr: pa.Array) -> List[List[int]]:
    values = arr.to_pylist()
    return [[] if v is None else [int(x) for x in v if x is not None] for v in values]


def _bucketize_counts(counts: np.ndarray, edges: Sequence[int]) -> np.ndarray:
    # 0 is reserved for missing/padding. Non-missing rows get ids 1..len(edges).
    upper_edges = np.asarray(list(edges)[1:], dtype=np.int64)
    return (np.searchsorted(upper_edges, counts, side="right") + 1).astype(np.int64)


def _compute_raw_features(
    batch: pa.RecordBatch,
    state: PrefixState,
    match_col: str,
    match_ts_col: str,
    match_window_seconds: int,
    count_edges: Sequence[int],
    enable_delay_history: bool = False,
) -> Dict[str, np.ndarray]:
    names = batch.schema.names
    idx = {name: i for i, name in enumerate(names)}
    B = batch.num_rows

    user_ids = _to_int_array(batch.column(idx["user_id"]))
    item_ids = _to_int_array(batch.column(idx["item_id"]))
    labels = _to_int_array(batch.column(idx["label_type"]))
    timestamps = _to_int_array(batch.column(idx["timestamp"]))
    if enable_delay_history:
        if "label_time" not in idx:
            raise KeyError("FE-02 requires label_time in the input parquet")
        label_times = _to_int_array(batch.column(idx["label_time"]))
    else:
        label_times = np.zeros(B, dtype=np.int64)

    if "item_int_feats_9" not in idx:
        raise KeyError("FE-01 requires item_int_feats_9 in the input parquet")
    target_item_attr = _first_scalar_from_maybe_list(batch.column(idx["item_int_feats_9"]).to_pylist())

    seq_values = _list_values(batch.column(idx[match_col]))
    seq_times = _list_values(batch.column(idx[match_ts_col]))

    user_total = np.zeros(B, dtype=np.float32)
    user_purchase = np.zeros(B, dtype=np.float32)
    item_total = np.zeros(B, dtype=np.float32)
    item_purchase = np.zeros(B, dtype=np.float32)
    user_avg_delay = np.zeros(B, dtype=np.float32)
    item_avg_delay = np.zeros(B, dtype=np.float32)
    has_match = np.zeros(B, dtype=np.int64)
    match_count = np.zeros(B, dtype=np.int64)
    min_match_delta = np.zeros(B, dtype=np.float32)
    match_count_7d = np.zeros(B, dtype=np.float32)

    # Compute prefix features in timestamp order inside each batch, while
    # writing results back to the original row positions.
    row_order = np.argsort(timestamps, kind="stable")
    for i in row_order:
        uid = int(user_ids[i])
        iid = int(item_ids[i])
        label_is_purchase = int(labels[i]) == 2
        ut, up, it, ip, u_delay, i_delay = state.before_update(uid, iid)
        user_total[i] = ut
        user_purchase[i] = up
        item_total[i] = it
        item_purchase[i] = ip
        user_avg_delay[i] = u_delay
        item_avg_delay[i] = i_delay

        target = int(target_item_attr[i])
        if target > 0:
            deltas: List[int] = []
            count_7d = 0
            for value, event_time in zip(seq_values[i], seq_times[i]):
                if value != target:
                    continue
                match_count[i] += 1
                if event_time > 0:
                    delta = max(int(timestamps[i]) - int(event_time), 0)
                    deltas.append(delta)
                    if delta <= match_window_seconds:
                        count_7d += 1
            has_match[i] = 2 if match_count[i] > 0 else 1
            match_count_7d[i] = count_7d
            min_match_delta[i] = min(deltas) if deltas else 0.0
        else:
            has_match[i] = 0

        state.update(
            uid,
            iid,
            label_is_purchase,
            timestamp=int(timestamps[i]),
            label_time=int(label_times[i]),
            collect_delay=enable_delay_history,
        )

    features = {
        "user_dense_feats_110": np.log1p(user_total),
        "user_dense_feats_111": np.log1p(user_purchase),
        "item_dense_feats_86": np.log1p(item_total),
        "item_dense_feats_87": np.log1p(item_purchase),
        "item_int_feats_89": has_match,
        "item_int_feats_90": _bucketize_counts(match_count, count_edges),
        "item_dense_feats_91": np.log1p(min_match_delta),
        "item_dense_feats_92": np.log1p(match_count_7d),
    }
    if enable_delay_history:
        features["user_dense_feats_112"] = np.log1p(user_avg_delay)
        features["item_dense_feats_88"] = np.log1p(item_avg_delay)
    return features


def iter_batches(
    row_groups: Sequence[Tuple[str, int]],
    batch_size: int,
) -> Iterable[Tuple[str, pa.RecordBatch]]:
    for path, rg_idx in row_groups:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, row_groups=[rg_idx]):
            yield path, batch


def fit_stats(
    row_groups: Sequence[Tuple[str, int]],
    batch_size: int,
    match_col: str,
    match_ts_col: str,
    match_window_seconds: int,
    count_edges: Sequence[int],
    feature_set: str = "fe01",
    enable_delay_history: bool = False,
) -> Dict[str, RunningStats]:
    feature_set = normalize_feature_set(feature_set)
    stats = {
        name: RunningStats()
        for name in dense_feature_names(feature_set, enable_delay_history)
    }
    state = PrefixState()
    for _, batch in iter_batches(row_groups, batch_size):
        feats = _compute_raw_features(
            batch,
            state,
            match_col,
            match_ts_col,
            match_window_seconds,
            count_edges,
            enable_delay_history,
        )
        for name, tracker in stats.items():
            tracker.update_many(feats[name])
    return stats


def _normalize(name: str, values: np.ndarray, stats: Dict[str, RunningStats]) -> np.ndarray:
    tracker = stats[name]
    return ((values.astype(np.float32) - tracker.mean) / tracker.std).astype(np.float32)


def _append_or_replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx == -1:
        return table.append_column(name, values)
    return table.set_column(idx, name, values)


def write_augmented_parquet(
    row_groups: Sequence[Tuple[str, int]],
    output_dir: str,
    batch_size: int,
    stats: Dict[str, RunningStats],
    match_col: str,
    match_ts_col: str,
    match_window_seconds: int,
    count_edges: Sequence[int],
    feature_set: str = "fe01",
    enable_delay_history: bool = False,
) -> Dict[str, Any]:
    feature_set = normalize_feature_set(feature_set)
    state = PrefixState()
    writers: Dict[str, pq.ParquetWriter] = {}
    try:
        for input_path, batch in iter_batches(row_groups, batch_size):
            feats = _compute_raw_features(
                batch,
                state,
                match_col,
                match_ts_col,
                match_window_seconds,
                count_edges,
                enable_delay_history,
            )
            table = pa.Table.from_batches([batch])
            for name in dense_feature_names(feature_set, enable_delay_history):
                table = _append_or_replace_column(
                    table,
                    name,
                    pa.array(_normalize(name, feats[name], stats), type=pa.float32()),
                )
            for name in item_int_feature_names(feature_set):
                table = _append_or_replace_column(
                    table, name, pa.array(feats[name], type=pa.int64()))

            out_path = os.path.join(output_dir, os.path.basename(input_path))
            writer = writers.get(out_path)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
                writers[out_path] = writer
            writer.write_table(table)
    finally:
        for writer in writers.values():
            writer.close()
    return state.delay_quality() if enable_delay_history else {}


def parse_edges(raw: str) -> List[int]:
    edges = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(edges) < 2 or edges[0] != 0:
        raise ValueError("--match_count_buckets must start with 0 and include at least one upper edge")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("--match_count_buckets must be strictly increasing")
    return edges


def build_alignment(
    match_col: str,
    match_ts_col: str,
    count_edges: Sequence[int],
    match_window_days: int,
    feature_set: str = "fe01",
    enable_delay_history: bool = False,
) -> Dict[str, Any]:
    feature_set = normalize_feature_set(feature_set)
    mappings: List[Dict[str, Any]] = [
        {
            "docx_ref": "P005",
            "docx_text": "user_dense_feats_110 = log(1 + user_total_frequency)",
            "implementation": "Prefix user total count before current timestamp, log1p then train-row-group z-score.",
            "output_column": "user_dense_feats_110",
        },
        {
            "docx_ref": "P009",
            "docx_text": "user_dense_feats_111 = log(1 + user_purchase_frequency)",
            "implementation": "Prefix user purchase count where historical label_type == 2, log1p then train-row-group z-score.",
            "output_column": "user_dense_feats_111",
        },
        {
            "docx_ref": "P007",
            "docx_text": "item_dense_feats_86 = log(1 + item_total_frequency)",
            "implementation": "Prefix item total count before current timestamp, log1p then train-row-group z-score.",
            "output_column": "item_dense_feats_86",
        },
        {
            "docx_ref": "P011",
            "docx_text": "item_dense_feats_87 = log(1 + item_purchase_frequency)",
            "implementation": "Prefix item purchase count where historical label_type == 2, log1p then train-row-group z-score.",
            "output_column": "item_dense_feats_87",
        },
        {
            "docx_ref": "P017",
            "docx_text": "Item_int_feats_89 = has_match(item_int_feats_9, domain_d_seq_19)",
            "implementation": "0=missing target, 1=no match, 2=has match.",
            "output_column": "item_int_feats_89",
            "match_column": match_col,
        },
        {
            "docx_ref": "P018",
            "docx_text": "Item_int_feats_90 = match_count(item_int_feats_9, domain_d_seq_19)",
            "implementation": "Bucketized match_count for Embedding compatibility; raw continuous count is not used as an id.",
            "output_column": "item_int_feats_90",
            "bucket_edges": list(map(int, count_edges)),
        },
        {
            "docx_ref": "P019",
            "docx_text": "item_dense_feats_91 = log(1 + min_match_delta)",
            "implementation": "Minimum timestamp - matched domain_d event timestamp, log1p then train-row-group z-score.",
            "output_column": "item_dense_feats_91",
            "timestamp_column": match_ts_col,
        },
        {
            "docx_ref": "P020",
            "docx_text": "item_dense_feats_92 = log(1 + match_count_7d)",
            "implementation": "Count matches whose timestamp delta is within the configured day window, log1p then train-row-group z-score.",
            "output_column": "item_dense_feats_92",
            "match_window_days": int(match_window_days),
        },
    ]

    if enable_delay_history:
        mappings.extend([
            {
                "docx_ref": "P013-P014",
                "docx_text": "delay= timestamp-label_time; user_dense_feats_112 = log(1 + user_avg_delay)",
                "implementation": "Prefix user average delay from previously seen conversion rows only. Uses timestamp - label_time, clips negative delays to zero, then log1p and train-row-group z-score.",
                "output_column": "user_dense_feats_112",
            },
            {
                "docx_ref": "P013-P015",
                "docx_text": "delay= timestamp-label_time; item_dense_feats_88 = log(1 + item_avg_delay)",
                "implementation": "Prefix item average delay from previously seen conversion rows only. Uses timestamp - label_time, clips negative delays to zero, then log1p and train-row-group z-score.",
                "output_column": "item_dense_feats_88",
            },
        ])

    selected_outputs = set(dense_feature_names(feature_set, enable_delay_history))
    selected_outputs.update(item_int_feature_names(feature_set))
    mappings = [m for m in mappings if m.get("output_column") in selected_outputs]

    not_included = [
        "delay-aware weighted loss",
        "multi-task learning",
    ]
    if feature_set == "fe01a":
        not_included.extend([
            "user_dense_feats_111 / item_dense_feats_87 purchase frequency",
            "item_int_feats_89/90 and item_dense_feats_91/92 target-history match features",
        ])
    elif feature_set == "fe01b":
        not_included.extend([
            "user_dense_feats_110 / item_dense_feats_86 total frequency",
            "user_dense_feats_111 / item_dense_feats_87 purchase frequency",
        ])
    if not enable_delay_history:
        not_included.insert(0, "user_dense_feats_112 / item_dense_feats_88 avg delay")

    experiment_name = "FE-02" if enable_delay_history else FEATURE_SET_TO_EXPERIMENT[feature_set]
    return {
        "source_docx": "/Users/gaogang/Downloads/feature-engineering.docx",
        "experiment": experiment_name,
        "feature_set": feature_set,
        "base_experiment": "FE-01" if enable_delay_history else None,
        "not_included": not_included,
        "mappings": mappings,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FE-01/FE-01A/FE-01B/FE-02 augmented parquet dataset")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--input_schema", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument(
        "--feature_set",
        choices=["fe01", "fe01a", "fe01b"],
        default="fe01",
        help="Ablation feature set: fe01=full safe features, "
             "fe01a=total frequency only, fe01b=target-history match only.",
    )
    ap.add_argument("--match_window_days", type=int, default=7)
    ap.add_argument("--match_count_buckets", default="0,1,2,4,8")
    ap.add_argument(
        "--fit_stats_row_group_ratio",
        type=float,
        default=0.9,
        help="Fraction of leading row groups used to fit dense normalization stats. "
             "Default 0.9 matches train.py's default tail-10%% validation split.",
    )
    ap.add_argument(
        "--enable_delay_history",
        action="store_true",
        help="Enable FE-02 historical avg-delay dense features 112 and 88.",
    )
    args = ap.parse_args()
    args.feature_set = normalize_feature_set(args.feature_set)
    if args.enable_delay_history and args.feature_set != "fe01":
        raise ValueError("--enable_delay_history is FE-02 and must be used with --feature_set fe01")

    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("--output_dir must be different from --input_dir")
    files = _parquet_files(args.input_dir)
    row_groups = _parquet_row_groups(files)
    schema = _load_schema(args.input_schema)
    first_names = pq.ParquetFile(files[0]).schema_arrow.names
    match_col, match_ts_col = resolve_domain_d_columns(schema, first_names)
    count_edges = parse_edges(args.match_count_buckets)
    match_window_seconds = int(args.match_window_days * 86400)
    n_fit_row_groups = max(1, int(len(row_groups) * args.fit_stats_row_group_ratio))
    fit_row_groups = row_groups[:n_fit_row_groups]

    print(f"Input parquet files: {len(files)}")
    print(f"Input row groups: {len(row_groups)}; fitting stats on first {len(fit_row_groups)}")
    print(f"Resolved FE-01 match columns: {match_col}, {match_ts_col}")
    print(f"Feature set: {args.feature_set}")
    print(f"Delay history enabled: {args.enable_delay_history}")
    print("Fitting feature-engineering dense normalization stats...")
    stats = fit_stats(
        fit_row_groups,
        args.batch_size,
        match_col,
        match_ts_col,
        match_window_seconds,
        count_edges,
        args.feature_set,
        args.enable_delay_history,
    )

    print("Writing augmented parquet files...")
    delay_quality = write_augmented_parquet(
        row_groups, args.output_dir, args.batch_size, stats,
        match_col, match_ts_col, match_window_seconds, count_edges,
        args.feature_set,
        args.enable_delay_history,
    )

    augmented_schema = build_augmented_schema(
        schema,
        match_count_vocab_size=len(count_edges) + 1,
        feature_set=args.feature_set,
        enable_delay_history=args.enable_delay_history,
    )
    output_ns_groups = filter_ns_groups(build_ns_groups(), augmented_schema)
    _write_json(os.path.join(args.output_dir, "schema.json"), augmented_schema)
    _write_json(os.path.join(args.output_dir, "ns_groups.feature_engineering.json"), output_ns_groups)
    _write_json(
        os.path.join(args.output_dir, "feature_engineering_stats.json"),
        {
            "experiment": "FE-02" if args.enable_delay_history else FEATURE_SET_TO_EXPERIMENT[args.feature_set],
            "feature_set": args.feature_set,
            "dense_feature_names": dense_feature_names(args.feature_set, args.enable_delay_history),
            "item_int_feature_names": item_int_feature_names(args.feature_set),
            "enable_delay_history": args.enable_delay_history,
            "dense_stats": {k: v.to_dict() for k, v in stats.items()},
            "match_col": match_col,
            "match_ts_col": match_ts_col,
            "match_window_days": args.match_window_days,
            "match_count_buckets": count_edges,
            "delay_quality": delay_quality,
            "fit_stats_row_group_ratio": args.fit_stats_row_group_ratio,
            "fit_stats_row_groups": len(fit_row_groups),
            "total_row_groups": len(row_groups),
            "leakage_note": "Prefix features are computed in timestamp order within each batch and use only previously seen stream state. For a global guarantee, feed timestamp-sorted parquet/row groups.",
        },
    )
    _write_json(
        os.path.join(args.output_dir, "docx_alignment.fe01.json"),
        build_alignment(match_col, match_ts_col, count_edges, args.match_window_days, args.feature_set, False),
    )
    if args.feature_set in {"fe01a", "fe01b"}:
        _write_json(
            os.path.join(args.output_dir, f"docx_alignment.{args.feature_set}.json"),
            build_alignment(
                match_col,
                match_ts_col,
                count_edges,
                args.match_window_days,
                args.feature_set,
                False,
            ),
        )
    if args.enable_delay_history:
        _write_json(
            os.path.join(args.output_dir, "docx_alignment.fe02.json"),
            build_alignment(match_col, match_ts_col, count_edges, args.match_window_days, "fe01", True),
        )
    experiment_name = "FE-02" if args.enable_delay_history else FEATURE_SET_TO_EXPERIMENT[args.feature_set]
    print(f"Wrote {experiment_name} dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()
