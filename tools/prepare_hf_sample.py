"""Prepare the official 1k-row HuggingFace sample for local smoke tests.

This script:
- downloads demo_1000.parquet from HF (TAAC2026/data_sample_1000)
- generates a *debug* schema.json based on the sample (NOT the official vocab)

The generated schema is only meant to make the baseline runnable on the sample.
Do NOT use it for official training.

Usage:
  python tools/prepare_hf_sample.py --out_dir ./data_sample_1000

Then:
  bash run.sh --data_dir ./data_sample_1000 --schema_path ./data_sample_1000/schema.json

Notes:
- Requires: pyarrow, numpy, huggingface_hub
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _max_in_list_array(arr: pa.ListArray) -> tuple[int, int]:
    """Return (max_value, max_len) for a ListArray.

    Works for integer ListArray. Nulls are ignored.
    """
    offsets = arr.offsets.to_numpy()
    lens = offsets[1:] - offsets[:-1]
    max_len = int(lens.max()) if len(lens) else 0

    values = arr.values
    if len(values) == 0:
        return 0, max_len

    v = values.to_numpy(zero_copy_only=False)
    # filter nulls
    if v.dtype.kind == "f":
        v = v[~np.isnan(v)]
    if len(v) == 0:
        return 0, max_len
    return int(np.max(v)), max_len


def _max_in_scalar(col: pa.Array) -> int:
    v = col.to_numpy(zero_copy_only=False)
    if v.dtype.kind == "f":
        v = v[~np.isnan(v)]
    if len(v) == 0:
        return 0
    return int(np.max(v))


def build_debug_schema(parquet_path: str) -> dict:
    table = pq.read_table(parquet_path)
    names = table.schema.names

    schema: dict = {"user_int": [], "item_int": [], "user_dense": [], "seq": {}}

    # Collect 4-domain sequence columns.
    # Columns look like: domain_a_seq_38
    seq_cols: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    m = re.compile(r"^(domain_[abcd]_seq)_(\d+)$")
    for n in names:
        mm = m.match(n)
        if not mm:
            continue
        prefix_base, fid = mm.group(1), int(mm.group(2))
        domain = prefix_base.replace("_seq", "")  # domain_a
        prefix = prefix_base + "_"              # domain_a_seq_
        seq_cols[domain].append((fid, n, prefix))

    for n in names:
        if n.startswith("user_int_feats_"):
            fid = int(n.split("_")[-1])
            col = table[n]
            if pa.types.is_list(col.type):
                mx, max_len = _max_in_list_array(col)
                schema["user_int"].append([fid, mx + 1, max_len])
            else:
                mx = _max_in_scalar(col)
                schema["user_int"].append([fid, mx + 1, 1])

        elif n.startswith("item_int_feats_"):
            fid = int(n.split("_")[-1])
            col = table[n]
            if pa.types.is_list(col.type):
                mx, max_len = _max_in_list_array(col)
                schema["item_int"].append([fid, mx + 1, max_len])
            else:
                mx = _max_in_scalar(col)
                schema["item_int"].append([fid, mx + 1, 1])

        elif n.startswith("user_dense_feats_"):
            fid = int(n.split("_")[-1])
            col = table[n]
            if pa.types.is_list(col.type):
                offsets = col.offsets.to_numpy()
                lens = offsets[1:] - offsets[:-1]
                max_dim = int(lens.max()) if len(lens) else 0
            else:
                max_dim = 1
            schema["user_dense"].append([fid, max_dim])

    # Fill seq domain configs.
    for domain, cols in seq_cols.items():
        cols = sorted(cols, key=lambda x: x[0])
        prefix = cols[0][2]
        feats = []
        for fid, colname, _ in cols:
            col = table[colname]
            if pa.types.is_list(col.type):
                mx, _ = _max_in_list_array(col)
            else:
                mx = _max_in_scalar(col)
            feats.append([fid, mx + 1])
        schema["seq"][domain] = {
            "prefix": prefix,
            "ts_fid": None,  # sample does not include a dedicated per-event timestamp feature
            "features": feats,
        }

    # Stable order for readability.
    schema["user_int"] = sorted(schema["user_int"], key=lambda x: x[0])
    schema["item_int"] = sorted(schema["item_int"], key=lambda x: x[0])
    schema["user_dense"] = sorted(schema["user_dense"], key=lambda x: x[0])
    schema["seq"] = {k: schema["seq"][k] for k in sorted(schema["seq"].keys())}

    return schema


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    parquet_out = os.path.join(args.out_dir, "demo_1000.parquet")
    schema_out = os.path.join(args.out_dir, "schema.json")

    # Download parquet.
    from huggingface_hub import hf_hub_download  # type: ignore

    path = hf_hub_download(
        repo_id="TAAC2026/data_sample_1000",
        filename="demo_1000.parquet",
        repo_type="dataset",
    )

    # Copy to out_dir.
    if os.path.abspath(path) != os.path.abspath(parquet_out):
        import shutil

        shutil.copyfile(path, parquet_out)

    schema = build_debug_schema(parquet_out)
    with open(schema_out, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {parquet_out}")
    print(f"Wrote: {schema_out}")
    print("NOTE: schema.json is derived from the 1k-row sample and is only for smoke tests.")


if __name__ == "__main__":
    main()
