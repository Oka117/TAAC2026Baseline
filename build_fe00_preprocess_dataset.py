"""Build the FE-00 preprocessing dataset aligned with feature-engineering.docx.

DOCX one-to-one mapping:

- P000: delete user int features whose missing value proportion >75%.
- P001: replace missing value in int_feats using average value.
- P002: normalize all dense numerical features.

Baseline compatibility:

- Dropping user int features is implemented by removing their fids from the
  emitted schema and ns_groups. The physical parquet columns may remain; the
  baseline dataset ignores columns not listed in schema.json.
- Average fill is applied to user/item int feature columns by default. For list
  int features, missing elements are filled; empty lists are kept empty unless
  --fill_empty_int_lists is passed.
- Sequence side features in this baseline are int sequence ids consumed by
  Embedding layers, so they are not z-scored by default. The script records this
  compatibility decision in docx_alignment.fe00.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


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
        variance = self.m2 / (self.n - 1)
        return max(math.sqrt(max(variance, 0.0)), 1e-6)

    def to_dict(self) -> Dict[str, float]:
        return {"n": self.n, "mean": self.mean, "std": self.std}


@dataclass
class IntFeatureStats:
    total_rows: int = 0
    missing_rows: int = 0
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
            "fill_value": self.fill_value,
            "value_mean": self.value_stats.mean,
            "value_count": self.value_stats.n,
        }


def parquet_files(input_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    return files


def parquet_row_groups(files: Sequence[str]) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for path in files:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.metadata.num_row_groups):
            out.append((path, rg_idx))
    if not out:
        raise ValueError("Input parquet files contain no row groups")
    return out


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
) -> Iterable[Tuple[str, pa.RecordBatch]]:
    for path, rg_idx in row_groups:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, row_groups=[rg_idx]):
            yield path, batch


def column_name(kind: str, fid: int) -> str:
    return f"{kind}_feats_{fid}"


def scalar_int_values(col: pa.Array) -> np.ndarray:
    return col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def positive_values_from_int_column(col: pa.Array) -> List[int]:
    values: List[int] = []
    if pa.types.is_list(col.type):
        for row in col.to_pylist():
            if row is None:
                continue
            for x in row:
                if x is not None and int(x) > 0:
                    values.append(int(x))
    else:
        arr = scalar_int_values(col)
        values.extend(int(x) for x in arr if int(x) > 0)
    return values


def missing_rows_for_int_column(col: pa.Array) -> int:
    if pa.types.is_list(col.type):
        missing = 0
        for row in col.to_pylist():
            if row is None or not any(x is not None and int(x) > 0 for x in row):
                missing += 1
        return missing
    arr = scalar_int_values(col)
    return int((arr <= 0).sum())


def dense_values(col: pa.Array) -> List[float]:
    out: List[float] = []
    if pa.types.is_list(col.type):
        for row in col.to_pylist():
            if row is None:
                continue
            for x in row:
                if x is not None and math.isfinite(float(x)):
                    out.append(float(x))
    else:
        arr = col.fill_null(np.nan).to_numpy(zero_copy_only=False)
        out.extend(float(x) for x in arr if math.isfinite(float(x)))
    return out


def collect_stats(
    row_groups: Sequence[Tuple[str, int]],
    schema: Dict[str, Any],
    batch_size: int,
) -> Tuple[Dict[str, IntFeatureStats], Dict[str, RunningStats]]:
    int_stats: Dict[str, IntFeatureStats] = {}
    dense_stats: Dict[str, RunningStats] = {}

    int_specs: List[Tuple[str, int]] = []
    for fid, _, _ in schema.get("user_int", []):
        int_specs.append(("user_int", int(fid)))
    for fid, _, _ in schema.get("item_int", []):
        int_specs.append(("item_int", int(fid)))

    dense_specs: List[Tuple[str, int]] = []
    for fid, _ in schema.get("user_dense", []):
        dense_specs.append(("user_dense", int(fid)))
    for fid, _ in schema.get("item_dense", []):
        dense_specs.append(("item_dense", int(fid)))

    for kind, fid in int_specs:
        int_stats[column_name(kind, fid)] = IntFeatureStats()
    for kind, fid in dense_specs:
        dense_stats[column_name(kind, fid)] = RunningStats()

    for _, batch in iter_batches(row_groups, batch_size):
        names = set(batch.schema.names)
        B = batch.num_rows
        for kind, fid in int_specs:
            name = column_name(kind, fid)
            if name not in names:
                continue
            col = batch.column(batch.schema.get_field_index(name))
            tracker = int_stats[name]
            tracker.total_rows += B
            tracker.missing_rows += missing_rows_for_int_column(col)
            tracker.value_stats.update_many(positive_values_from_int_column(col))

        for kind, fid in dense_specs:
            name = column_name(kind, fid)
            if name not in names:
                continue
            col = batch.column(batch.schema.get_field_index(name))
            dense_stats[name].update_many(dense_values(col))

    return int_stats, dense_stats


def fill_int_column(col: pa.Array, fill_value: int, fill_empty_lists: bool) -> pa.Array:
    if fill_value <= 0:
        return col
    if pa.types.is_list(col.type):
        rows = []
        for row in col.to_pylist():
            if row is None or len(row) == 0:
                rows.append([fill_value] if fill_empty_lists else ([] if row is None else row))
                continue
            rows.append([
                int(x) if x is not None and int(x) > 0 else fill_value
                for x in row
            ])
        return pa.array(rows, type=pa.list_(pa.int64()))
    arr = scalar_int_values(col)
    arr[arr <= 0] = fill_value
    return pa.array(arr, type=pa.int64())


def normalize_dense_column(col: pa.Array, stats: RunningStats) -> pa.Array:
    mean = stats.mean
    std = stats.std
    if pa.types.is_list(col.type):
        rows = []
        for row in col.to_pylist():
            if row is None:
                rows.append([])
                continue
            new_row = []
            for x in row:
                value = mean if x is None or not math.isfinite(float(x)) else float(x)
                new_row.append((value - mean) / std)
            rows.append(new_row)
        return pa.array(rows, type=pa.list_(pa.float32()))

    arr = col.fill_null(mean).to_numpy(zero_copy_only=False).astype(np.float32)
    arr = np.nan_to_num(arr, nan=mean, posinf=mean, neginf=mean)
    return pa.array(((arr - mean) / std).astype(np.float32), type=pa.float32())


def replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx == -1:
        return table
    return table.set_column(idx, name, values)


def filter_schema(
    schema: Dict[str, Any],
    dropped_user_fids: Sequence[int],
) -> Dict[str, Any]:
    dropped = set(int(x) for x in dropped_user_fids)
    out = dict(schema)
    out["user_int"] = [
        row for row in schema.get("user_int", [])
        if int(row[0]) not in dropped
    ]
    return out


def filter_ns_groups(
    ns_groups: Dict[str, Any],
    output_schema: Dict[str, Any],
) -> Dict[str, Any]:
    allowed_user_fids = {int(row[0]) for row in output_schema.get("user_int", [])}
    allowed_item_fids = {int(row[0]) for row in output_schema.get("item_int", [])}
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


def write_preprocessed_parquet(
    row_groups: Sequence[Tuple[str, int]],
    output_dir: str,
    batch_size: int,
    schema: Dict[str, Any],
    int_stats: Dict[str, IntFeatureStats],
    dense_stats: Dict[str, RunningStats],
    fill_empty_int_lists: bool,
) -> None:
    writers: Dict[str, pq.ParquetWriter] = {}
    try:
        for input_path, batch in iter_batches(row_groups, batch_size):
            table = pa.Table.from_batches([batch])
            names = set(table.schema.names)

            for fid, _, _ in schema.get("user_int", []):
                name = column_name("user_int", int(fid))
                if name in names:
                    fill_value = int_stats.get(name, IntFeatureStats()).fill_value
                    table = replace_column(
                        table, name,
                        fill_int_column(table[name].combine_chunks(), fill_value, fill_empty_int_lists),
                    )
            for fid, _, _ in schema.get("item_int", []):
                name = column_name("item_int", int(fid))
                if name in names:
                    fill_value = int_stats.get(name, IntFeatureStats()).fill_value
                    table = replace_column(
                        table, name,
                        fill_int_column(table[name].combine_chunks(), fill_value, fill_empty_int_lists),
                    )

            for fid, _ in schema.get("user_dense", []):
                name = column_name("user_dense", int(fid))
                if name in names:
                    table = replace_column(
                        table, name,
                        normalize_dense_column(table[name].combine_chunks(), dense_stats.get(name, RunningStats())),
                    )
            for fid, _ in schema.get("item_dense", []):
                name = column_name("item_dense", int(fid))
                if name in names:
                    table = replace_column(
                        table, name,
                        normalize_dense_column(table[name].combine_chunks(), dense_stats.get(name, RunningStats())),
                    )

            out_path = os.path.join(output_dir, os.path.basename(input_path))
            writer = writers.get(out_path)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
                writers[out_path] = writer
            writer.write_table(table)
    finally:
        for writer in writers.values():
            writer.close()


def build_alignment(
    dropped_user_fids: Sequence[int],
    int_stats: Dict[str, IntFeatureStats],
    dense_stats: Dict[str, RunningStats],
    fill_empty_int_lists: bool,
) -> Dict[str, Any]:
    return {
        "source_docx": "/Users/gaogang/Downloads/feature-engineering.docx",
        "mappings": [
            {
                "docx_ref": "P000",
                "docx_text": "delete int feature from user whose missing value proportion >75%",
                "implementation": "Dropped user_int fids from emitted schema/ns_groups when missing_ratio > threshold.",
                "outputs": ["schema.json", "ns_groups.fe00.json", "dropped_user_int_fids.json"],
                "dropped_user_fids": list(map(int, dropped_user_fids)),
            },
            {
                "docx_ref": "P001",
                "docx_text": "Replace missing value in int_feats using average value",
                "implementation": "Filled user/item int missing values with rounded train-split positive-value mean.",
                "outputs": ["int_fill_values.json", "preprocessed parquet"],
                "fill_empty_int_lists": fill_empty_int_lists,
            },
            {
                "docx_ref": "P002",
                "docx_text": "all dense numerical feature(including sequence side feature) do normalization",
                "implementation": "Normalized user/item dense numerical columns with train-split z-score. Sequence side ids remain embedding inputs unless explicitly retyped as numerical.",
                "outputs": ["dense_normalization_stats.json", "preprocessed parquet"],
                "dense_columns": sorted(dense_stats.keys()),
            },
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FE-00 preprocessing dataset")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--input_schema", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ns_groups_json", default="ns_groups.json")
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--missing_threshold", type=float, default=0.75)
    ap.add_argument("--fit_stats_row_group_ratio", type=float, default=0.9)
    ap.add_argument("--fill_empty_int_lists", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("--output_dir must be different from --input_dir")

    schema = load_json(args.input_schema)
    files = parquet_files(args.input_dir)
    row_groups = parquet_row_groups(files)
    n_fit = max(1, int(len(row_groups) * args.fit_stats_row_group_ratio))
    fit_row_groups = row_groups[:n_fit]

    print(f"Input parquet files: {len(files)}")
    print(f"Input row groups: {len(row_groups)}; fitting stats on first {len(fit_row_groups)}")
    print("Collecting FE-00 missing/fill/normalization stats...")
    int_stats, dense_stats = collect_stats(fit_row_groups, schema, args.batch_size)

    dropped_user_fids = []
    for fid, _, _ in schema.get("user_int", []):
        name = column_name("user_int", int(fid))
        if int_stats.get(name, IntFeatureStats()).missing_ratio > args.missing_threshold:
            dropped_user_fids.append(int(fid))

    print(f"Dropping {len(dropped_user_fids)} user_int fids with missing_ratio>{args.missing_threshold}")
    output_schema = filter_schema(schema, dropped_user_fids)

    ns_groups = load_json(args.ns_groups_json) if args.ns_groups_json and os.path.exists(args.ns_groups_json) else {
        "user_ns_groups": {},
        "item_ns_groups": {},
    }
    output_ns_groups = filter_ns_groups(ns_groups, output_schema)

    print("Writing FE-00 preprocessed parquet...")
    write_preprocessed_parquet(
        row_groups=row_groups,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        schema=schema,
        int_stats=int_stats,
        dense_stats=dense_stats,
        fill_empty_int_lists=args.fill_empty_int_lists,
    )

    write_json(os.path.join(args.output_dir, "schema.json"), output_schema)
    write_json(os.path.join(args.output_dir, "ns_groups.fe00.json"), output_ns_groups)
    write_json(os.path.join(args.output_dir, "dropped_user_int_fids.json"), dropped_user_fids)
    write_json(
        os.path.join(args.output_dir, "feature_missing_report.json"),
        {
            name: stats.to_dict()
            for name, stats in sorted(int_stats.items())
            if name.startswith("user_int_feats_")
        },
    )
    write_json(
        os.path.join(args.output_dir, "int_fill_values.json"),
        {name: stats.fill_value for name, stats in sorted(int_stats.items())},
    )
    write_json(
        os.path.join(args.output_dir, "dense_normalization_stats.json"),
        {name: stats.to_dict() for name, stats in sorted(dense_stats.items())},
    )
    write_json(
        os.path.join(args.output_dir, "docx_alignment.fe00.json"),
        build_alignment(dropped_user_fids, int_stats, dense_stats, args.fill_empty_int_lists),
    )

    print(f"Wrote FE-00 dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()
