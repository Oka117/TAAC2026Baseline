"""PCVRHyFormer inference script (uploaded by the contestant into the
evaluation container).

Model construction mirrors ``train.py``: we rebuild the model from
``schema.json`` + ``ns_groups.json`` + ``train_config.json``. All model
hyperparameters are resolved first from the ckpt directory's
``train_config.json`` (written by ``trainer.py`` when saving a checkpoint),
falling back to ``_FALLBACK_MODEL_CFG`` below (which must stay consistent
with the CLI defaults in ``train.py``).

Only the Parquet data format is supported.

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory (points at the ``global_step``
                       sub-directory containing ``model.pt`` / ``train_config.json``).
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH   Directory for the generated ``predictions.json``.
"""

import os
import json
import logging
import glob
import tempfile
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

from dataset import (
    FeatureSchema,
    PCVRParquetDataset,
    NUM_TIME_BUCKETS,
    load_domain_time_bucket_boundaries,
)
from model import PCVRHyFormer, ModelInput
from build_fe07_p012_domain_dataset import (
    PrefixState as FE07PrefixState,
    _append_or_replace_column as _fe07_append_or_replace_column,
    _compute_generated_features as _fe07_compute_generated_features,
    _fill_int_column as _fe07_fill_int_column,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# Fallback values used only when ``train_config.json`` is missing from the
# ckpt directory.
#
# These MUST match the argparse defaults in ``train.py``; otherwise once the
# fallback path is actually taken the built model will shape-mismatch the
# saved state_dict.
#
# Special note on ``num_time_buckets``: this value is strictly determined by
# ``dataset.BUCKET_BOUNDARIES`` and is NOT an independent hyperparameter.
# When the feature is enabled we therefore use the constant exposed by the
# dataset module; ``0`` means disabled.
_FALLBACK_MODEL_CFG = {
    'd_model': 64,
    'emb_dim': 64,
    'num_queries': 1,
    'num_hyformer_blocks': 2,
    'num_heads': 4,
    'seq_encoder_type': 'transformer',
    'hidden_mult': 4,
    'dropout_rate': 0.015,
    'seq_top_k': 50,
    'seq_causal': False,
    'action_num': 1,
    'num_time_buckets': NUM_TIME_BUCKETS,
    'rank_mixer_mode': 'full',
    'use_rope': False,
    'rope_base': 10000.0,
    'emb_skip_threshold': 0,
    'seq_id_threshold': 10000,
    'ns_tokenizer_type': 'rankmixer',
    'user_ns_tokens': 6,
    'item_ns_tokens': 4,
    'use_token_gnn': False,
    'token_gnn_layers': 4,
    'token_gnn_graph': 'full',
    'token_gnn_layer_scale': 0.15,
}

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:128,seq_d:768'
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 8


# Hyperparameter keys used to build the model. Everything else in
# ``train_config.json`` is ignored when constructing ``PCVRHyFormer``.
_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())

FE01_USER_DENSE_COLUMNS = [
    "user_dense_feats_110",
    "user_dense_feats_111",
]
FE01_ITEM_DENSE_COLUMNS = [
    "item_dense_feats_86",
    "item_dense_feats_87",
    "item_dense_feats_91",
    "item_dense_feats_92",
]
FE01_ITEM_INT_COLUMNS = [
    "item_int_feats_89",
    "item_int_feats_90",
]
FE01_ALL_COLUMNS = FE01_USER_DENSE_COLUMNS + FE01_ITEM_DENSE_COLUMNS + FE01_ITEM_INT_COLUMNS


class PrefixState:
    def __init__(self) -> None:
        self.user_total: DefaultDict[int, int] = defaultdict(int)
        self.user_purchase: DefaultDict[int, int] = defaultdict(int)
        self.item_total: DefaultDict[int, int] = defaultdict(int)
        self.item_purchase: DefaultDict[int, int] = defaultdict(int)

    def before_update(self, user_id: int, item_id: int) -> Tuple[int, int, int, int]:
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


def _load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _resolve_fe01_sidecar(model_dir: str, filename: str) -> Optional[str]:
    local_path = os.path.join(model_dir, filename)
    if os.path.exists(local_path):
        return local_path

    stats_dir = os.environ.get('FE01_STATS_DIR') or os.environ.get('FE01_STATS_PATH')
    if stats_dir:
        candidate = os.path.join(stats_dir, filename)
        if os.path.exists(candidate):
            return candidate

    return None


def _load_fe01_stats(model_dir: str) -> Dict[str, Any]:
    stats_path = _resolve_fe01_sidecar(model_dir, 'feature_engineering_stats.json')
    if stats_path is None:
        raise FileNotFoundError(
            "Missing FE-01 sidecar: feature_engineering_stats.json. "
            "Put it next to model.pt in MODEL_OUTPUT_PATH, or set FE01_STATS_DIR "
            "to the FE-01 preprocessing output directory. Evaluation must reuse "
            "training dense normalization stats and must not re-fit on eval data."
        )
    logging.info(f"Using FE-01 feature stats: {stats_path}")
    return _load_json(stats_path)


def _parquet_files(input_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(input_dir, '*.parquet')))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    return files


def _iter_batches(
    files: Sequence[str],
    batch_size: int,
    columns: Optional[Sequence[str]] = None,
) -> Iterable[Tuple[str, pa.RecordBatch]]:
    read_columns = None if columns is None else list(dict.fromkeys(columns))
    for path in files:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.metadata.num_row_groups):
            for batch in pf.iter_batches(
                batch_size=batch_size,
                row_groups=[rg_idx],
                columns=read_columns,
            ):
                yield path, batch


def _schema_required_parquet_columns(schema_path: str) -> List[str]:
    schema = _load_json(schema_path)
    columns: List[str] = ['user_id', 'timestamp']
    for fid, _, _ in schema.get('user_int', []):
        columns.append(f'user_int_feats_{int(fid)}')
    for fid, _, _ in schema.get('item_int', []):
        columns.append(f'item_int_feats_{int(fid)}')
    for fid, _ in schema.get('user_dense', []):
        columns.append(f'user_dense_feats_{int(fid)}')
    for fid, _ in schema.get('item_dense', []):
        columns.append(f'item_dense_feats_{int(fid)}')
    for cfg in schema.get('seq', {}).values():
        prefix = str(cfg.get('prefix', ''))
        for fid, _ in cfg.get('features', []):
            columns.append(f'{prefix}_{int(fid)}')
    return list(dict.fromkeys(columns))


def _fe07_expected_generated_columns() -> List[str]:
    return [
        'user_dense_feats_110',
        'user_dense_feats_120',
        'user_dense_feats_121',
        'item_dense_feats_86',
        'item_dense_feats_91',
        'item_dense_feats_92',
        'item_int_feats_89',
        'item_int_feats_90',
    ]


def _eval_has_fe07_generated_columns(data_dir: str) -> bool:
    files = _parquet_files(data_dir)
    first_names = pq.ParquetFile(files[0]).schema_arrow.names
    return all(name in first_names for name in _fe07_expected_generated_columns())


def _rebuild_dataset_column_plans(
    dataset: PCVRParquetDataset,
    schema_names: Sequence[str],
) -> None:
    dataset._col_idx = {name: i for i, name in enumerate(schema_names)}

    dataset._user_int_plan = []
    offset = 0
    for fid, vs, dim in dataset._user_int_cols:
        name = f'user_int_feats_{fid}'
        ci = dataset._col_idx.get(name)
        if ci is None:
            raise KeyError(f"FE-07 eval dataset missing required column: {name}")
        dataset._user_int_plan.append((ci, dim, offset, vs))
        offset += dim

    dataset._item_int_plan = []
    offset = 0
    for fid, vs, dim in dataset._item_int_cols:
        name = f'item_int_feats_{fid}'
        ci = dataset._col_idx.get(name)
        if ci is None:
            raise KeyError(f"FE-07 eval dataset missing required column: {name}")
        dataset._item_int_plan.append((ci, dim, offset, vs))
        offset += dim

    dataset._user_dense_plan = []
    offset = 0
    for fid, dim in dataset._user_dense_cols:
        name = f'user_dense_feats_{fid}'
        ci = dataset._col_idx.get(name)
        if ci is None:
            raise KeyError(f"FE-07 eval dataset missing required column: {name}")
        dataset._user_dense_plan.append((ci, dim, offset))
        offset += dim

    dataset._item_dense_plan = []
    offset = 0
    for fid, dim in dataset._item_dense_cols:
        name = f'item_dense_feats_{fid}'
        ci = dataset._col_idx.get(name)
        if ci is None:
            raise KeyError(f"FE-07 eval dataset missing required column: {name}")
        dataset._item_dense_plan.append((ci, dim, offset))
        offset += dim

    dataset._seq_plan = {}
    for domain in dataset.seq_domains:
        prefix = dataset._seq_prefix[domain]
        side_plan = []
        for slot, fid in enumerate(dataset.sideinfo_fids[domain]):
            name = f'{prefix}_{fid}'
            ci = dataset._col_idx.get(name)
            if ci is None:
                raise KeyError(f"FE-07 eval dataset missing required column: {name}")
            vs = dataset.seq_vocab_sizes[domain][fid]
            side_plan.append((ci, slot, vs))
        ts_fid = dataset.ts_fids[domain]
        ts_ci = None
        if ts_fid is not None:
            name = f'{prefix}_{ts_fid}'
            ts_ci = dataset._col_idx.get(name)
            if ts_ci is None:
                raise KeyError(f"FE-07 eval dataset missing required column: {name}")
        dataset._seq_plan[domain] = (side_plan, ts_ci)


def _slice_batch_dict(batch: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value[start:end]
        elif key == 'user_id':
            out[key] = value[start:end]
        else:
            out[key] = value
    return out


class FE07OnTheFlyEvalDataset(IterableDataset):
    """Generate FE-07 eval features in memory and feed tensors directly."""

    def __init__(
        self,
        data_dir: str,
        schema_path: str,
        model_dir: str,
        infer_batch_size: int,
        transform_batch_size: int,
        seq_max_lens: Dict[str, int],
        domain_time_buckets: Optional[Dict[str, Sequence[int]]],
    ) -> None:
        super().__init__()
        self.files = _parquet_files(data_dir)
        self.schema_path = schema_path
        self.infer_batch_size = infer_batch_size
        self.transform_batch_size = transform_batch_size
        logging.info(
            "Using FE-07 on-the-fly eval transform: "
            f"transform_batch_size={transform_batch_size}, "
            f"infer_batch_size={infer_batch_size}")

        stats = _load_fe07_stats(model_dir)
        if stats is None:
            raise FileNotFoundError(
                "Checkpoint schema requires FE-07 generated columns, but "
                "fe07_transform_stats.json was not found next to model.pt."
            )
        self.stats = stats
        self.dense_stats = stats.get('dense_stats', {})
        self.count_edges = [int(x) for x in stats.get('match_count_buckets', [0, 1, 2, 4, 8])]
        self.match_col = str(stats.get('match_col', 'domain_d_seq_19'))
        self.match_ts_col = str(stats.get('match_ts_col', 'domain_d_seq_26'))
        self.seq_ts_cols = {str(k): str(v) for k, v in stats.get('seq_time_columns', {}).items()}
        self.min_timestamp = int(stats.get('min_timestamp_for_day_since', 0))
        self.match_window_seconds = int(stats.get('match_window_days', 7)) * 86400
        self.int_fill_audit = stats.get('int_fill_audit', {})
        self.fill_empty_int_lists = bool(stats.get('fill_empty_int_lists', False))

        first_names = pq.ParquetFile(self.files[0]).schema_arrow.names
        missing_base = [
            name for name in ('user_id', 'item_id', 'timestamp', 'item_int_feats_9', self.match_col, self.match_ts_col)
            if name not in first_names
        ]
        if missing_base:
            raise KeyError("FE-07 eval transform missing required raw columns: " + ", ".join(missing_base))

        self.required_output = _schema_required_parquet_columns(schema_path)
        self.required_set = set(self.required_output)
        self.generated = set(_fe07_expected_generated_columns())
        required_raw_output = [
            name for name in self.required_output
            if name in first_names and name not in self.generated
        ]
        read_columns = set(required_raw_output)
        read_columns.update(['user_id', 'item_id', 'timestamp', 'item_int_feats_9', self.match_col, self.match_ts_col])
        read_columns.update(self.seq_ts_cols.values())
        read_columns.update(
            name for name in self.int_fill_audit
            if name in first_names and (name.startswith('user_int_feats_') or name.startswith('item_int_feats_'))
        )
        read_columns.update(
            name for name in self.dense_stats
            if name in first_names and name not in self.generated
        )
        self.read_columns = [name for name in first_names if name in read_columns]

        self.converter = PCVRParquetDataset(
            parquet_path=data_dir,
            schema_path=schema_path,
            batch_size=transform_batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=False,
            buffer_batches=0,
            is_training=False,
            domain_time_buckets=domain_time_buckets,
        )
        _rebuild_dataset_column_plans(self.converter, self.required_output)

        self.num_rows = self.converter.num_rows
        self.num_time_buckets = self.converter.num_time_buckets
        self.seq_domains = self.converter.seq_domains
        self.user_int_schema = self.converter.user_int_schema
        self.user_int_vocab_sizes = self.converter.user_int_vocab_sizes
        self.item_int_schema = self.converter.item_int_schema
        self.item_int_vocab_sizes = self.converter.item_int_vocab_sizes
        self.user_dense_schema = self.converter.user_dense_schema
        self.item_dense_schema = self.converter.item_dense_schema
        self.seq_domain_vocab_sizes = self.converter.seq_domain_vocab_sizes

    def _transform_batch(self, batch: pa.RecordBatch, state: FE07PrefixState) -> pa.RecordBatch:
        idx = {name: i for i, name in enumerate(batch.schema.names)}
        input_table = pa.Table.from_batches([batch])
        table = input_table.select([
            name for name in input_table.schema.names
            if name in self.required_set and name not in self.generated
        ])

        for name, audit in self.int_fill_audit.items():
            if name in idx and name in self.required_set and (
                name.startswith('user_int_feats_') or name.startswith('item_int_feats_')
            ):
                fill_value = int(audit.get('fill_value', 0))
                table = _fe07_append_or_replace_column(
                    table,
                    name,
                    _fe07_fill_int_column(batch.column(idx[name]), fill_value, self.fill_empty_int_lists),
                )

        for name in self.dense_stats:
            if name in idx and name in self.required_set and name not in self.generated:
                table = _fe07_append_or_replace_column(
                    table,
                    name,
                    _normalize_fe07_dense_column(batch.column(idx[name]), name, self.dense_stats),
                )

        feats = _fe07_compute_generated_features(
            batch,
            state,
            self.match_col,
            self.match_ts_col,
            self.seq_ts_cols,
            self.min_timestamp,
            self.match_window_seconds,
            self.count_edges,
        )

        for name in [
            'user_dense_feats_110',
            'user_dense_feats_120',
            'user_dense_feats_121',
            'item_dense_feats_86',
            'item_dense_feats_91',
            'item_dense_feats_92',
        ]:
            if name not in self.required_set:
                continue
            values = _normalize_fe07_array(name, feats[name], self.dense_stats)
            if values.ndim == 1:
                arr = pa.array(values, type=pa.float32())
            else:
                arr = pa.array(values.tolist(), type=pa.list_(pa.float32()))
            table = _fe07_append_or_replace_column(table, name, arr)

        for name in ['item_int_feats_89', 'item_int_feats_90']:
            if name in self.required_set:
                table = _fe07_append_or_replace_column(
                    table, name, pa.array(feats[name], type=pa.int64()))

        missing_output = [name for name in self.required_output if name not in table.schema.names]
        if missing_output:
            raise KeyError(
                "FE-07 on-the-fly eval could not materialize required columns: "
                + ", ".join(missing_output[:20])
            )
        return table.select(self.required_output).to_batches()[0]

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        state = FE07PrefixState()
        processed_rows = 0
        processed_batches = 0
        for _path, raw_batch in _iter_batches(
            self.files,
            self.transform_batch_size,
            columns=self.read_columns,
        ):
            transformed = self._transform_batch(raw_batch, state)
            tensor_batch = self.converter._convert_batch(transformed)
            B = transformed.num_rows
            processed_rows += B
            processed_batches += 1
            if processed_batches == 1 or processed_batches % 100 == 0:
                logging.info(
                    f"FE-07 on-the-fly eval transformed {processed_rows} rows "
                    f"({processed_batches} batches)")
            for start in range(0, B, self.infer_batch_size):
                yield _slice_batch_dict(tensor_batch, start, min(start + self.infer_batch_size, B))


def _strip_prefix(prefix: str) -> str:
    return prefix[:-1] if prefix.endswith('_') else prefix


def _resolve_domain_d_columns(
    schema: Dict[str, Any],
    parquet_names: Sequence[str],
    stats: Dict[str, Any],
) -> Tuple[str, str]:
    names = set(parquet_names)
    stats_match_col = stats.get('match_col')
    stats_ts_col = stats.get('match_ts_col')
    if stats_match_col in names and stats_ts_col in names:
        return str(stats_match_col), str(stats_ts_col)

    candidates: List[Tuple[str, Optional[int]]] = []
    for domain, cfg in schema.get('seq', {}).items():
        prefix = _strip_prefix(str(cfg.get('prefix', '')))
        if domain in {'domain_d', 'seq_d'} or prefix == 'domain_d_seq':
            candidates.append((prefix, cfg.get('ts_fid')))
    candidates.append(('domain_d_seq', 26))

    for prefix, ts_fid in candidates:
        match_col = f'{prefix}_19'
        ts_col = f'{prefix}_{ts_fid}' if ts_fid is not None else f'{prefix}_26'
        if match_col in names and ts_col in names:
            return match_col, ts_col

    raise KeyError(
        "Could not resolve FE-01 domain_d columns in eval parquet. Expected "
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
    upper_edges = np.asarray(list(edges)[1:], dtype=np.int64)
    return (np.searchsorted(upper_edges, counts, side='right') + 1).astype(np.int64)


def _compute_raw_fe01_features(
    batch: pa.RecordBatch,
    state: PrefixState,
    match_col: str,
    match_ts_col: str,
    match_window_seconds: int,
    count_edges: Sequence[int],
    compute_match_features: bool = True,
) -> Dict[str, np.ndarray]:
    names = batch.schema.names
    idx = {name: i for i, name in enumerate(names)}
    B = batch.num_rows

    user_ids = _to_int_array(batch.column(idx['user_id']))
    item_ids = _to_int_array(batch.column(idx['item_id']))
    timestamps = _to_int_array(batch.column(idx['timestamp']))
    # Never consume eval labels when constructing online features. Official
    # eval data may omit label_type, or may contain placeholder/private values;
    # purchase-prefix state is therefore kept at zero to avoid leakage.
    labels = np.zeros(B, dtype=np.int64)

    if compute_match_features:
        if 'item_int_feats_9' not in idx:
            raise KeyError("FE-01B/FE-01 evaluation requires item_int_feats_9 in eval parquet")
        target_item_attr = _first_scalar_from_maybe_list(batch.column(idx['item_int_feats_9']).to_pylist())
        seq_values = _list_values(batch.column(idx[match_col]))
        seq_times = _list_values(batch.column(idx[match_ts_col]))
    else:
        target_item_attr = np.zeros(B, dtype=np.int64)
        seq_values = [[] for _ in range(B)]
        seq_times = [[] for _ in range(B)]

    user_total = np.zeros(B, dtype=np.float32)
    user_purchase = np.zeros(B, dtype=np.float32)
    item_total = np.zeros(B, dtype=np.float32)
    item_purchase = np.zeros(B, dtype=np.float32)
    has_match = np.zeros(B, dtype=np.int64)
    match_count = np.zeros(B, dtype=np.int64)
    min_match_delta = np.zeros(B, dtype=np.float32)
    match_count_7d = np.zeros(B, dtype=np.float32)

    row_order = np.argsort(timestamps, kind='stable')
    for i in row_order:
        uid = int(user_ids[i])
        iid = int(item_ids[i])
        is_purchase = int(labels[i]) == 2
        ut, up, it, ip = state.before_update(uid, iid)
        user_total[i] = ut
        user_purchase[i] = up
        item_total[i] = it
        item_purchase[i] = ip

        target = int(target_item_attr[i])
        if compute_match_features and target > 0:
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
            min_match_delta[i] = min(deltas) if deltas else 0.0
            match_count_7d[i] = count_7d
        else:
            has_match[i] = 0

        state.update(uid, iid, is_purchase)

    return {
        'user_dense_feats_110': np.log1p(user_total),
        'user_dense_feats_111': np.log1p(user_purchase),
        'item_dense_feats_86': np.log1p(item_total),
        'item_dense_feats_87': np.log1p(item_purchase),
        'item_int_feats_89': has_match,
        'item_int_feats_90': _bucketize_counts(match_count, count_edges),
        'item_dense_feats_91': np.log1p(min_match_delta),
        'item_dense_feats_92': np.log1p(match_count_7d),
    }


def _normalize_with_training_stats(
    name: str,
    values: np.ndarray,
    dense_stats: Dict[str, Dict[str, float]],
) -> np.ndarray:
    if name not in dense_stats:
        raise KeyError(
            f"feature_engineering_stats.json missing dense_stats.{name}. "
            "Cannot normalize FE-01 eval features exactly."
        )
    stat = dense_stats[name]
    mean = float(stat.get('mean', 0.0))
    std = max(float(stat.get('std', 1.0)), 1e-6)
    return ((values.astype(np.float32) - mean) / std).astype(np.float32)


def _append_or_replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx == -1:
        return table.append_column(name, values)
    return table.set_column(idx, name, values)


def _expected_fe01_columns_from_schema(schema_path: str) -> List[str]:
    schema = _load_json(schema_path)
    expected: List[str] = []
    user_dense_fids = {int(row[0]) for row in schema.get('user_dense', [])}
    item_dense_fids = {int(row[0]) for row in schema.get('item_dense', [])}
    item_int_fids = {int(row[0]) for row in schema.get('item_int', [])}
    for name in FE01_USER_DENSE_COLUMNS:
        if int(name.rsplit('_', 1)[1]) in user_dense_fids:
            expected.append(name)
    for name in FE01_ITEM_DENSE_COLUMNS:
        if int(name.rsplit('_', 1)[1]) in item_dense_fids:
            expected.append(name)
    for name in FE01_ITEM_INT_COLUMNS:
        if int(name.rsplit('_', 1)[1]) in item_int_fids:
            expected.append(name)
    return expected


def maybe_build_fe01_eval_dataset(
    data_dir: str,
    schema_path: str,
    model_dir: str,
    batch_size: int,
    result_dir: str,
) -> str:
    """Materialize FE-01 columns for raw eval parquet when checkpoint schema expects them."""
    files = _parquet_files(data_dir)
    first_names = pq.ParquetFile(files[0]).schema_arrow.names
    expected = _expected_fe01_columns_from_schema(schema_path)
    missing = [name for name in expected if name not in first_names]
    if not missing:
        logging.info("Eval parquet already contains FE-01 columns; using raw eval directory.")
        return data_dir

    unsupported = [name for name in missing if name not in FE01_ALL_COLUMNS]
    if unsupported:
        raise KeyError(
            "Checkpoint schema expects columns that FE-01 infer.py cannot generate: "
            + ", ".join(unsupported)
        )
    expected_user_dense = [name for name in FE01_USER_DENSE_COLUMNS if name in expected]
    expected_item_dense = [name for name in FE01_ITEM_DENSE_COLUMNS if name in expected]
    expected_item_int = [name for name in FE01_ITEM_INT_COLUMNS if name in expected]
    needs_match_features = bool(expected_item_int or any(
        name in expected_item_dense for name in ('item_dense_feats_91', 'item_dense_feats_92')
    ))

    logging.info(f"Eval parquet missing FE-01 columns: {missing}")
    stats = _load_fe01_stats(model_dir)
    dense_stats = stats.get('dense_stats', {})
    count_edges = [int(x) for x in stats.get('match_count_buckets', [0, 1, 2, 4, 8])]
    match_window_days = int(stats.get('match_window_days', 7))
    match_window_seconds = match_window_days * 86400
    if needs_match_features:
        raw_schema = _load_json(schema_path)
        match_col, match_ts_col = _resolve_domain_d_columns(raw_schema, first_names, stats)
    else:
        match_col, match_ts_col = '', ''
    if 'label_type' in first_names:
        logging.info(
            "Eval parquet contains label_type, but FE-01 eval transform will "
            "ignore it to avoid label leakage.")
    else:
        logging.warning(
            "Eval parquet has no label_type column; FE-01 purchase-frequency "
            "features 111/87 will use zero online-observable purchase state.")

    output_dir = os.environ.get('FE01_EVAL_DATA_DIR')
    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix='taac_fe01_eval_')
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Writing FE-01 eval parquet to: {output_dir}")
    if needs_match_features:
        logging.info(f"Resolved FE-01 eval match columns: {match_col}, {match_ts_col}")
    else:
        logging.info("FE-01 eval transform does not need target-history match columns.")

    state = PrefixState()
    writers: Dict[str, pq.ParquetWriter] = {}
    try:
        for input_path, batch in _iter_batches(files, batch_size):
            feats = _compute_raw_fe01_features(
                batch,
                state,
                match_col,
                match_ts_col,
                match_window_seconds,
                count_edges,
                needs_match_features,
            )
            table = pa.Table.from_batches([batch])
            for name in expected_user_dense + expected_item_dense:
                table = _append_or_replace_column(
                    table,
                    name,
                    pa.array(
                        _normalize_with_training_stats(name, feats[name], dense_stats),
                        type=pa.float32(),
                    ),
                )
            for name in expected_item_int:
                table = _append_or_replace_column(
                    table, name, pa.array(feats[name], type=pa.int64()))

            out_path = os.path.join(output_dir, os.path.basename(input_path))
            writer = writers.get(out_path)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
                writers[out_path] = writer
            writer.write_table(table)
    finally:
        for writer in writers.values():
            writer.close()

    return output_dir


def _resolve_fe07_sidecar(model_dir: str, filename: str) -> Optional[str]:
    local_path = os.path.join(model_dir, filename)
    if os.path.exists(local_path):
        return local_path
    stats_dir = os.environ.get('FE07_STATS_DIR') or os.environ.get('FE07_STATS_PATH')
    if stats_dir:
        candidate = os.path.join(stats_dir, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def _load_fe07_stats(model_dir: str) -> Optional[Dict[str, Any]]:
    stats_path = _resolve_fe07_sidecar(model_dir, 'fe07_transform_stats.json')
    if stats_path is None:
        return None
    logging.info(f"Using FE-07 transform stats: {stats_path}")
    return _load_json(stats_path)


def _schema_has_fe07_columns(schema_path: str) -> bool:
    schema = _load_json(schema_path)
    user_dense = {int(row[0]) for row in schema.get('user_dense', [])}
    item_dense = {int(row[0]) for row in schema.get('item_dense', [])}
    item_int = {int(row[0]) for row in schema.get('item_int', [])}
    return bool(
        {110, 120, 121} & user_dense
        or {86, 91, 92} & item_dense
        or {89, 90} & item_int
    )


def _normalize_fe07_array(
    name: str,
    values: np.ndarray,
    dense_stats: Dict[str, Dict[str, float]],
) -> np.ndarray:
    if name not in dense_stats:
        raise KeyError(f"fe07_transform_stats.json missing dense_stats.{name}")
    stat = dense_stats[name]
    mean = float(stat.get('mean', 0.0))
    std = max(float(stat.get('std', 1.0)), 1e-6)
    return ((values.astype(np.float32) - mean) / std).astype(np.float32)


def _normalize_fe07_dense_column(
    col: pa.Array,
    name: str,
    dense_stats: Dict[str, Dict[str, float]],
) -> pa.Array:
    if name not in dense_stats:
        return col
    stat = dense_stats[name]
    mean = float(stat.get('mean', 0.0))
    std = max(float(stat.get('std', 1.0)), 1e-6)
    if pa.types.is_list(col.type):
        rows: List[List[float]] = []
        for row in col.to_pylist():
            if row is None:
                rows.append([])
                continue
            vals = []
            for x in row:
                value = mean if x is None else float(x)
                vals.append((value - mean) / std)
            rows.append(vals)
        return pa.array(rows, type=pa.list_(pa.float32()))
    arr = col.fill_null(mean).to_numpy(zero_copy_only=False).astype(np.float32, copy=True)
    arr = np.nan_to_num(arr, nan=mean, posinf=mean, neginf=mean)
    return pa.array(((arr - mean) / std).astype(np.float32), type=pa.float32())


def maybe_build_fe07_eval_dataset(
    data_dir: str,
    schema_path: str,
    model_dir: str,
    batch_size: int,
    result_dir: str,
) -> str:
    """Materialize FE-07 generated columns and FE00-literal transforms for raw eval parquet."""
    schema_has_fe07 = _schema_has_fe07_columns(schema_path)
    if not schema_has_fe07:
        return data_dir

    files = _parquet_files(data_dir)
    first_names = pq.ParquetFile(files[0]).schema_arrow.names
    expected_generated = [
        'user_dense_feats_110',
        'user_dense_feats_120',
        'user_dense_feats_121',
        'item_dense_feats_86',
        'item_dense_feats_91',
        'item_dense_feats_92',
        'item_int_feats_89',
        'item_int_feats_90',
    ]
    if all(name in first_names for name in expected_generated):
        logging.info("Eval parquet already contains FE-07 generated columns; using raw eval directory.")
        return data_dir

    stats = _load_fe07_stats(model_dir)
    if stats is None:
        raise FileNotFoundError(
            "Checkpoint schema requires FE-07 generated columns, but "
            "fe07_transform_stats.json was not found next to model.pt. "
            "Copy the FE-07 sidecars from the training feature directory into "
            "MODEL_OUTPUT_PATH, or set FE07_STATS_DIR to that directory."
        )

    dense_stats = stats.get('dense_stats', {})
    count_edges = [int(x) for x in stats.get('match_count_buckets', [0, 1, 2, 4, 8])]
    match_col = str(stats.get('match_col', 'domain_d_seq_19'))
    match_ts_col = str(stats.get('match_ts_col', 'domain_d_seq_26'))
    seq_ts_cols = {str(k): str(v) for k, v in stats.get('seq_time_columns', {}).items()}
    min_timestamp = int(stats.get('min_timestamp_for_day_since', 0))
    match_window_seconds = int(stats.get('match_window_days', 7)) * 86400
    int_fill_audit = stats.get('int_fill_audit', {})
    transform_batch_size = int(os.environ.get(
        'FE07_EVAL_TRANSFORM_BATCH_SIZE',
        stats.get('transform_batch_size', 8192),
    ))

    missing_base = [
        name for name in ('user_id', 'item_id', 'timestamp', 'item_int_feats_9', match_col, match_ts_col)
        if name not in first_names
    ]
    if missing_base:
        raise KeyError("FE-07 eval transform missing required raw columns: " + ", ".join(missing_base))

    output_base_dir = os.environ.get('FE07_EVAL_DATA_DIR')
    if output_base_dir:
        os.makedirs(output_base_dir, exist_ok=True)
        output_dir = tempfile.mkdtemp(prefix='materialized_', dir=output_base_dir)
    else:
        output_dir = tempfile.mkdtemp(prefix='taac_fe07_eval_')
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Writing FE-07 eval parquet to: {output_dir}")
    logging.info(f"FE-07 eval transform batch_size={transform_batch_size}")
    logging.info(
        "FE-07 eval transform ignores eval label_type; total-frequency state "
        "starts from zero and uses only eval stream order.")

    required_output = _schema_required_parquet_columns(schema_path)
    required_set = set(required_output)
    generated_set = set(expected_generated)
    required_raw_output = [
        name for name in required_output
        if name in first_names and name not in generated_set
    ]
    read_columns = set(required_raw_output)
    read_columns.update(['user_id', 'item_id', 'timestamp', 'item_int_feats_9', match_col, match_ts_col])
    read_columns.update(seq_ts_cols.values())
    read_columns.update(
        name for name in int_fill_audit
        if name in first_names and (name.startswith('user_int_feats_') or name.startswith('item_int_feats_'))
    )
    read_columns.update(
        name for name in dense_stats
        if name in first_names and name not in generated_set
    )
    read_columns = [name for name in first_names if name in read_columns]

    fill_empty_int_lists = bool(stats.get('fill_empty_int_lists', False))
    state = FE07PrefixState()
    output_path = os.path.join(output_dir, 'part-00000.parquet')
    writer: Optional[pq.ParquetWriter] = None
    processed_rows = 0
    processed_batches = 0
    try:
        for _input_path, batch in _iter_batches(files, transform_batch_size, columns=read_columns):
            processed_batches += 1
            processed_rows += batch.num_rows
            if processed_batches == 1 or processed_batches % 100 == 0:
                logging.info(
                    f"FE-07 eval transform processed {processed_rows} rows "
                    f"({processed_batches} batches)")
            idx = {name: i for i, name in enumerate(batch.schema.names)}
            input_table = pa.Table.from_batches([batch])
            table = input_table.select([
                name for name in input_table.schema.names
                if name in required_set and name not in generated_set
            ])

            for name, audit in int_fill_audit.items():
                if name in idx and name in required_set and (
                    name.startswith('user_int_feats_') or name.startswith('item_int_feats_')
                ):
                    fill_value = int(audit.get('fill_value', 0))
                    table = _fe07_append_or_replace_column(
                        table, name, _fe07_fill_int_column(batch.column(idx[name]), fill_value, fill_empty_int_lists))

            for name in dense_stats:
                if name in idx and name in required_set and name not in generated_set:
                    table = _fe07_append_or_replace_column(
                        table, name, _normalize_fe07_dense_column(batch.column(idx[name]), name, dense_stats))

            feats = _fe07_compute_generated_features(
                batch,
                state,
                match_col,
                match_ts_col,
                seq_ts_cols,
                min_timestamp,
                match_window_seconds,
                count_edges,
            )

            for name in [
                'user_dense_feats_110',
                'user_dense_feats_120',
                'user_dense_feats_121',
                'item_dense_feats_86',
                'item_dense_feats_91',
                'item_dense_feats_92',
            ]:
                values = _normalize_fe07_array(name, feats[name], dense_stats)
                if values.ndim == 1:
                    arr = pa.array(values, type=pa.float32())
                else:
                    arr = pa.array(values.tolist(), type=pa.list_(pa.float32()))
                table = _fe07_append_or_replace_column(table, name, arr)

            for name in ['item_int_feats_89', 'item_int_feats_90']:
                if name in required_set:
                    table = _fe07_append_or_replace_column(
                        table, name, pa.array(feats[name], type=pa.int64()))

            missing_output = [name for name in required_output if name not in table.schema.names]
            if missing_output:
                raise KeyError(
                    "FE-07 eval transform could not materialize required columns: "
                    + ", ".join(missing_output[:20])
                )
            table = table.select(required_output)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    return output_dir


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build ``feature_specs = [(vocab_size, offset, length), ...]`` in the
    order of ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    """Parse a string like ``'seq_a:256,seq_b:256,...'`` into a dict."""
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(','):
        k, v = pair.split(':')
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def load_train_config(model_dir: str) -> Dict[str, Any]:
    """Load ``train_config.json`` from the ckpt directory.

    Returns an empty dict (which triggers fallback resolution) if the file is
    not present.
    """
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    logging.warning(
        f"train_config.json not found in {model_dir}, "
        f"falling back to hardcoded defaults. "
        f"Shape mismatch may occur if training used non-default hyperparameters.")
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model hyperparameters from ``train_config``; missing keys fall
    back to ``_FALLBACK_MODEL_CFG``.

    Special handling for ``num_time_buckets``: it is not exposed on the CLI
    as an independent hyperparameter; the bucket count is uniquely determined
    by the length of ``dataset.BUCKET_BOUNDARIES``. Resolution order:

      1) ``train_config`` contains ``num_time_buckets`` directly (legacy ckpt)
         -> use that value;
      2) ``train_config`` contains ``use_time_buckets`` (new-style training)
         -> derive as ``NUM_TIME_BUCKETS`` or ``0``;
      3) neither is present -> fall back to ``_FALLBACK_MODEL_CFG[...]``.
    """
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key == 'num_time_buckets':
            if 'num_time_buckets' in train_config:
                cfg[key] = train_config['num_time_buckets']
            elif 'use_time_buckets' in train_config:
                cfg[key] = NUM_TIME_BUCKETS if train_config['use_time_buckets'] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
                logging.warning(
                    f"train_config missing both 'num_time_buckets' and 'use_time_buckets', "
                    f"using fallback = {cfg[key]}")
            continue

        if key in train_config:
            cfg[key] = train_config[key]
        else:
            cfg[key] = _FALLBACK_MODEL_CFG[key]
            logging.warning(
                f"train_config missing '{key}', using fallback = {cfg[key]}")
    return cfg


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    ns_groups_json: Optional[str] = None,
    device: str = 'cpu',
) -> PCVRHyFormer:
    """Construct a ``PCVRHyFormer`` from the dataset schema, an NS-groups JSON,
    and a resolved ``model_cfg`` dict.

    Args:
        dataset: a ``PCVRParquetDataset`` providing the feature schema.
        model_cfg: resolved model hyperparameters, typically the output of
            ``resolve_model_cfg``.
        ns_groups_json: path to the NS-groups JSON file, or ``None`` / empty
            string to disable it (each feature becomes its own singleton group).
        device: torch device.
    """
    # NS grouping. The JSON schema uses *fid* (feature id) values; convert
    # them to positional indices into ``user_int_schema.entries`` /
    # ``item_int_schema.entries`` so ``GroupNSTokenizer`` /
    # ``RankMixerNSTokenizer`` can index ``feature_specs`` directly. This is
    # the same conversion ``train.py`` performs when loading the JSON; doing
    # it here keeps infer.py symmetric with training.
    user_ns_groups: List[List[int]]
    item_ns_groups: List[List[int]]
    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info(f"Loading NS groups from {ns_groups_json}")
        with open(ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)
        }
        item_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)
        }
        try:
            user_ns_groups = [
                [user_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['user_ns_groups'].values()
            ]
            item_ns_groups = [
                [item_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg['item_ns_groups'].values()
            ]
        except KeyError as exc:
            raise KeyError(
                f"NS-groups JSON references fid {exc.args[0]} which is not "
                f"present in the checkpoint's schema.json. The ns_groups.json "
                f"and schema.json must come from the same training run."
            ) from exc
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(dataset.item_int_schema.entries))]

    # Feature specs.
    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes)

    logging.info(f"Building PCVRHyFormer with cfg: {model_cfg}")
    model = PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        **model_cfg,
    ).to(device)

    return model


def load_model_state_strict(
    model: nn.Module,
    ckpt_path: str,
    device: str,
) -> None:
    """Strictly load ``state_dict``; any missing/unexpected key fails fast
    with a diagnostic message.
    """
    state_dict = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logging.error(
            "Failed to load state_dict in strict mode. This usually means the "
            "model constructed by build_model does NOT match the checkpoint. "
            "Check that train_config.json in the ckpt dir is present and matches "
            "the training hyperparameters.")
        raise e


def get_ckpt_path() -> Optional[str]:
    """Locate the first ``*.pt`` file inside the directory pointed at by
    ``$MODEL_OUTPUT_PATH``. Returns ``None`` if no checkpoint is found.
    """
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if not ckpt_path:
        return None
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    return None


def _batch_to_model_input(
    batch: Dict[str, Any],
    device: str,
) -> ModelInput:
    """Convert a batch dict to ``ModelInput``, handling dynamic seq domains."""
    device_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            device_batch[k] = v.to(device, non_blocking=True)
        else:
            device_batch[k] = v

    seq_domains = device_batch['_seq_domains']
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f'{domain}_len']
        B, _, L = device_batch[domain].shape
        seq_time_buckets[domain] = device_batch.get(
            f'{domain}_time_bucket',
            torch.zeros(B, L, dtype=torch.long, device=device))

    return ModelInput(
        user_int_feats=device_batch['user_int_feats'],
        item_int_feats=device_batch['item_int_feats'],
        user_dense_feats=device_batch['user_dense_feats'],
        item_dense_feats=device_batch['item_dense_feats'],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
    )


def main() -> None:
    # ---- Read environment variables ----
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')

    os.makedirs(result_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Schema: prefer the one from model_dir (to exactly match training);
    #      fall back to the one in data_dir if missing. ----
    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    logging.info(f"Using schema: {schema_path}")

    # ---- Load train_config.json (single source of truth for all hyperparams) ----
    train_config = load_train_config(model_dir)

    # ---- Parse seq_max_lens ----
    sml_str = train_config.get('seq_max_lens', _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")

    # ---- Data loading: reuse batch_size / num_workers from training config ----
    batch_size = int(train_config.get('batch_size', _FALLBACK_BATCH_SIZE))
    num_workers = int(train_config.get('num_workers', _FALLBACK_NUM_WORKERS))

    domain_bucket_path = train_config.get('domain_bucket_path')
    if domain_bucket_path:
        local_candidate = os.path.join(model_dir, os.path.basename(domain_bucket_path))
        if os.path.exists(local_candidate):
            domain_bucket_path = local_candidate
    else:
        local_candidate = os.path.join(model_dir, 'domain_time_bucket_boundaries.json')
        domain_bucket_path = local_candidate if os.path.exists(local_candidate) else None
    domain_time_buckets = None
    if train_config.get('domain_time_buckets', False):
        if not domain_bucket_path:
            raise FileNotFoundError(
                "Checkpoint train_config enables domain_time_buckets but "
                "domain_time_bucket_boundaries.json was not found next to model.pt."
            )
        domain_time_buckets = load_domain_time_bucket_boundaries(domain_bucket_path)
        logging.info(f"Loaded FE-07 domain time buckets from {domain_bucket_path}")

    schema_has_fe07 = _schema_has_fe07_columns(schema_path)
    eval_has_fe07 = _eval_has_fe07_generated_columns(data_dir) if schema_has_fe07 else False
    force_materialize = os.environ.get('FE07_EVAL_MATERIALIZE', '').lower() in {'1', 'true', 'yes'}
    use_on_the_fly_fe07 = schema_has_fe07 and not eval_has_fe07 and not force_materialize

    if use_on_the_fly_fe07:
        transform_batch_size = int(os.environ.get('FE07_EVAL_TRANSFORM_BATCH_SIZE', 1024))
        test_dataset = FE07OnTheFlyEvalDataset(
            data_dir=data_dir,
            schema_path=schema_path,
            model_dir=model_dir,
            infer_batch_size=batch_size,
            transform_batch_size=transform_batch_size,
            seq_max_lens=seq_max_lens,
            domain_time_buckets=domain_time_buckets,
        )
        loader_num_workers = 0
    else:
        # Compatibility path: use pre-generated FE-07 parquet, or opt into
        # materialization with FE07_EVAL_MATERIALIZE=1 for debugging.
        data_dir = maybe_build_fe07_eval_dataset(
            data_dir=data_dir,
            schema_path=schema_path,
            model_dir=model_dir,
            batch_size=batch_size,
            result_dir=result_dir,
        )
        test_dataset = PCVRParquetDataset(
            parquet_path=data_dir,
            schema_path=schema_path,
            batch_size=batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=False,
            buffer_batches=0,
            is_training=False,
            domain_time_buckets=domain_time_buckets,
        )
        loader_num_workers = num_workers
    total_test_samples = test_dataset.num_rows
    logging.info(f"Total test samples: {total_test_samples}")

    # ---- Build model: every structural hyperparameter is resolved from train_config ----
    model_cfg = resolve_model_cfg(train_config)
    if train_config.get('use_time_buckets', True):
        model_cfg['num_time_buckets'] = test_dataset.num_time_buckets
    if model_cfg.get('use_token_gnn', False):
        raise ValueError(
            "FE-07 evaluation is no-GNN. The checkpoint train_config has "
            "use_token_gnn=true; use the matching evaluator for that checkpoint instead."
        )

    # ns_groups_json also comes from training config (e.g. run.sh may have
    # passed an empty string to disable it). When trainer.py has copied the
    # JSON into the ckpt dir, train_config records just the basename, so try
    # resolving against ``model_dir`` first before honoring the raw (possibly
    # absolute) path as a fallback.
    ns_groups_json = train_config.get('ns_groups_json', None)
    if ns_groups_json:
        local_candidate = os.path.join(model_dir, os.path.basename(ns_groups_json))
        if os.path.exists(local_candidate):
            ns_groups_json = local_candidate

    model = build_model(
        test_dataset,
        model_cfg=model_cfg,
        ns_groups_json=ns_groups_json,
        device=device,
    )

    # ---- Strictly load weights ----
    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"The directory contains: {os.listdir(model_dir) if model_dir and os.path.isdir(model_dir) else 'N/A'}. "
            "This typically means the training job wrote only the sidecar "
            "files (schema.json / train_config.json) for this step but did "
            "not persist model.pt — a symptom of a race between "
            "_remove_old_best_dirs and EarlyStopping.save_checkpoint."
        )
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    loader_kwargs: Dict[str, Any] = {
        'batch_size': None,
        'num_workers': loader_num_workers,
        'pin_memory': torch.cuda.is_available(),
    }
    if loader_num_workers > 0:
        loader_kwargs['prefetch_factor'] = 2
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    all_probs = []
    all_user_ids = []
    processed_count = 0
    logging.info("Starting inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get('user_id', [])

            logits, _ = model.predict(model_input)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_user_ids.extend(user_ids)
            processed_count += len(probs)

            if (batch_idx + 1) % 100 == 0:
                logging.info(f"  Processed {processed_count} samples")

    logging.info(f"Inference complete: {len(all_probs)} predictions")

    predictions = {
        "predictions": dict(zip(all_user_ids, all_probs)),
    }

    # ---- Save predictions.json ----
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()
