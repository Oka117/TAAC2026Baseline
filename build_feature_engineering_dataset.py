"""Build the FE-01 feature-engineering parquet dataset.

This script implements the first experiment plan in
``experiment_plans/FE01/experiment_01_feature_engineering_plan.zh.md``.

It streams parquet files in sorted file / row-group order and creates:

- user_dense_feats_110: log1p(prefix user total frequency), z-scored
- user_dense_feats_111: log1p(prefix user purchase frequency), z-scored
- item_dense_feats_86: log1p(prefix item total frequency), z-scored
- item_dense_feats_87: log1p(prefix item purchase frequency), z-scored
- item_int_feats_89: has_match(item_int_feats_9, domain_d_seq_19)
- item_int_feats_90: bucketized match_count(item_int_feats_9, domain_d_seq_19)
- item_dense_feats_91: log1p(min_match_delta), z-scored
- item_dense_feats_92: log1p(match_count_7d), z-scored

The prefix frequency features are computed from rows already seen by the
streaming pass. For leakage safety, feed this script data ordered by timestamp
or by the same historical order used by training.
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
ITEM_DENSE_ADDS = [(86, 1), (87, 1), (91, 1), (92, 1)]
ITEM_INT_BASE_ADDS = [(89, 3, 1)]


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

    def before_update(
        self,
        user_id: int,
        item_id: int,
    ) -> Tuple[int, int, int, int]:
        return (
            self.user_total[user_id],
            self.user_purchase[user_id],
            self.item_total[item_id],
            self.item_purchase[item_id],
        )

    def update(self, user_id: int, item_id: int, is_purchase: bool) -> None:
        self.user_total[user_id] += 1
        self.item_total[item_id] += 1
        if is_purchase:
            self.user_purchase[user_id] += 1
            self.item_purchase[item_id] += 1


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


def build_augmented_schema(schema: Dict[str, Any], match_count_vocab_size: int) -> Dict[str, Any]:
    out = dict(schema)
    out["user_dense"] = _append_feature_specs(out.get("user_dense", []), USER_DENSE_ADDS)
    out["item_dense"] = _append_feature_specs(out.get("item_dense", []), ITEM_DENSE_ADDS)
    out["item_int"] = _append_feature_specs(
        out.get("item_int", []),
        [*ITEM_INT_BASE_ADDS, (90, match_count_vocab_size, 1)],
    )
    return out


def build_ns_groups() -> Dict[str, Any]:
    return {
        "_purpose": "FE-01 NS groups. Dense feature ids are intentionally omitted; they enter through user/item dense tokens.",
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
) -> Dict[str, np.ndarray]:
    names = batch.schema.names
    idx = {name: i for i, name in enumerate(names)}
    B = batch.num_rows

    user_ids = _to_int_array(batch.column(idx["user_id"]))
    item_ids = _to_int_array(batch.column(idx["item_id"]))
    labels = _to_int_array(batch.column(idx["label_type"]))
    timestamps = _to_int_array(batch.column(idx["timestamp"]))

    if "item_int_feats_9" not in idx:
        raise KeyError("FE-01 requires item_int_feats_9 in the input parquet")
    target_item_attr = _first_scalar_from_maybe_list(batch.column(idx["item_int_feats_9"]).to_pylist())

    seq_values = _list_values(batch.column(idx[match_col]))
    seq_times = _list_values(batch.column(idx[match_ts_col]))

    user_total = np.zeros(B, dtype=np.float32)
    user_purchase = np.zeros(B, dtype=np.float32)
    item_total = np.zeros(B, dtype=np.float32)
    item_purchase = np.zeros(B, dtype=np.float32)
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
        ut, up, it, ip = state.before_update(uid, iid)
        user_total[i] = ut
        user_purchase[i] = up
        item_total[i] = it
        item_purchase[i] = ip

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

        state.update(uid, iid, label_is_purchase)

    return {
        "user_dense_feats_110": np.log1p(user_total),
        "user_dense_feats_111": np.log1p(user_purchase),
        "item_dense_feats_86": np.log1p(item_total),
        "item_dense_feats_87": np.log1p(item_purchase),
        "item_int_feats_89": has_match,
        "item_int_feats_90": _bucketize_counts(match_count, count_edges),
        "item_dense_feats_91": np.log1p(min_match_delta),
        "item_dense_feats_92": np.log1p(match_count_7d),
    }


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
) -> Dict[str, RunningStats]:
    stats = {name: RunningStats() for name in [
        "user_dense_feats_110",
        "user_dense_feats_111",
        "item_dense_feats_86",
        "item_dense_feats_87",
        "item_dense_feats_91",
        "item_dense_feats_92",
    ]}
    state = PrefixState()
    for _, batch in iter_batches(row_groups, batch_size):
        feats = _compute_raw_features(
            batch, state, match_col, match_ts_col, match_window_seconds, count_edges)
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
) -> None:
    state = PrefixState()
    writers: Dict[str, pq.ParquetWriter] = {}
    try:
        for input_path, batch in iter_batches(row_groups, batch_size):
            feats = _compute_raw_features(
                batch, state, match_col, match_ts_col, match_window_seconds, count_edges)
            table = pa.Table.from_batches([batch])
            for name in [
                "user_dense_feats_110",
                "user_dense_feats_111",
                "item_dense_feats_86",
                "item_dense_feats_87",
                "item_dense_feats_91",
                "item_dense_feats_92",
            ]:
                table = _append_or_replace_column(
                    table,
                    name,
                    pa.array(_normalize(name, feats[name], stats), type=pa.float32()),
                )
            for name in ["item_int_feats_89", "item_int_feats_90"]:
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
) -> Dict[str, Any]:
    return {
        "source_docx": "/Users/gaogang/Downloads/feature-engineering.docx",
        "experiment": "FE-01",
        "not_included": [
            "user_dense_feats_112 / item_dense_feats_88 avg delay",
            "delay-aware weighted loss",
            "multi-task learning",
        ],
        "mappings": [
            {
                "docx_ref": "P008",
                "docx_text": "user_dense_feats_110 = log(1+user_total_frequency)",
                "implementation": "Prefix user total count before current timestamp, log1p then train-row-group z-score.",
                "output_column": "user_dense_feats_110",
            },
            {
                "docx_ref": "P012",
                "docx_text": "user_dense_feats_111 = log(1+user_purchase_frequency)",
                "implementation": "Prefix user purchase count where historical label_type == 2, log1p then train-row-group z-score.",
                "output_column": "user_dense_feats_111",
            },
            {
                "docx_ref": "P010",
                "docx_text": "item_dense_feats_86 = log(1+item_total_frequency)",
                "implementation": "Prefix item total count before current timestamp, log1p then train-row-group z-score.",
                "output_column": "item_dense_feats_86",
            },
            {
                "docx_ref": "P014",
                "docx_text": "item_dense_feats_87 = log(1+item_purchase_frequency)",
                "implementation": "Prefix item purchase count where historical label_type == 2, log1p then train-row-group z-score.",
                "output_column": "item_dense_feats_87",
            },
            {
                "docx_ref": "P020",
                "docx_text": "Item_int_feats_89 = has_match(item_int_feats_9, domain_d_seq_19)",
                "implementation": "0=missing target, 1=no match, 2=has match.",
                "output_column": "item_int_feats_89",
                "match_column": match_col,
            },
            {
                "docx_ref": "P021",
                "docx_text": "Item_int_feats_90 = match_count(item_int_feats_9, domain_d_seq_19)",
                "implementation": "Bucketized match_count for Embedding compatibility; raw continuous count is not used as an id.",
                "output_column": "item_int_feats_90",
                "bucket_edges": list(map(int, count_edges)),
            },
            {
                "docx_ref": "P022",
                "docx_text": "item_dense_feats_91 = log(1+min_match_delta)",
                "implementation": "Minimum timestamp - matched domain_d event timestamp, log1p then train-row-group z-score.",
                "output_column": "item_dense_feats_91",
                "timestamp_column": match_ts_col,
            },
            {
                "docx_ref": "P023",
                "docx_text": "item_dense_feats_92 = log(1+match_count_7d)",
                "implementation": "Count matches whose timestamp delta is within the configured day window, log1p then train-row-group z-score.",
                "output_column": "item_dense_feats_92",
                "match_window_days": int(match_window_days),
            },
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FE-01 augmented parquet dataset")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--input_schema", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--match_window_days", type=int, default=7)
    ap.add_argument("--match_count_buckets", default="0,1,2,4,8")
    ap.add_argument(
        "--fit_stats_row_group_ratio",
        type=float,
        default=0.9,
        help="Fraction of leading row groups used to fit dense normalization stats. "
             "Default 0.9 matches train.py's default tail-10%% validation split.",
    )
    args = ap.parse_args()

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
    print("Fitting FE-01 dense normalization stats...")
    stats = fit_stats(
        fit_row_groups, args.batch_size, match_col, match_ts_col, match_window_seconds, count_edges)

    print("Writing augmented parquet files...")
    write_augmented_parquet(
        row_groups, args.output_dir, args.batch_size, stats,
        match_col, match_ts_col, match_window_seconds, count_edges)

    augmented_schema = build_augmented_schema(schema, match_count_vocab_size=len(count_edges) + 1)
    output_ns_groups = filter_ns_groups(build_ns_groups(), augmented_schema)
    _write_json(os.path.join(args.output_dir, "schema.json"), augmented_schema)
    _write_json(os.path.join(args.output_dir, "ns_groups.feature_engineering.json"), output_ns_groups)
    _write_json(
        os.path.join(args.output_dir, "feature_engineering_stats.json"),
        {
            "dense_stats": {k: v.to_dict() for k, v in stats.items()},
            "match_col": match_col,
            "match_ts_col": match_ts_col,
            "match_window_days": args.match_window_days,
            "match_count_buckets": count_edges,
            "fit_stats_row_group_ratio": args.fit_stats_row_group_ratio,
            "fit_stats_row_groups": len(fit_row_groups),
            "total_row_groups": len(row_groups),
            "leakage_note": "Prefix features are computed in timestamp order within each batch and use only previously seen stream state. For a global guarantee, feed timestamp-sorted parquet/row groups.",
        },
    )
    _write_json(
        os.path.join(args.output_dir, "docx_alignment.fe01.json"),
        build_alignment(match_col, match_ts_col, count_edges, args.match_window_days),
    )
    print(f"Wrote FE-01 dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()
