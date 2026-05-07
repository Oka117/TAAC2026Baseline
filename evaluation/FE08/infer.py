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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

from dataset import (
    FeatureSchema,
    PCVRParquetDataset,
    BUCKET_BOUNDARIES,
    NUM_TIME_BUCKETS,
)
from model import PCVRHyFormer, ModelInput
from build_fe08_may7_dataset import (
    PrefixState as FE08PrefixState,
    _append_or_replace_column as _fe08_append_or_replace_column,
    _compute_generated_features as _fe08_compute_generated_features,
    _fill_int_column as _fe08_fill_int_column,
    _sort_sequence_columns_by_recency as _fe08_sort_sequence_columns_by_recency,
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
    'd_model': 136,
    'emb_dim': 64,
    'num_queries': 2,
    'num_hyformer_blocks': 2,
    'num_heads': 4,
    'seq_encoder_type': 'transformer',
    'hidden_mult': 4,
    'dropout_rate': 0.05,
    'seq_top_k': 100,
    'seq_causal': False,
    'action_num': 1,
    'num_time_buckets': NUM_TIME_BUCKETS,
    'rank_mixer_mode': 'full',
    'use_rope': False,
    'rope_base': 10000.0,
    'emb_skip_threshold': 1000000,
    'seq_id_threshold': 10000,
    'ns_tokenizer_type': 'rankmixer',
    'user_ns_tokens': 5,
    'item_ns_tokens': 2,
    'use_token_gnn': True,
    'token_gnn_layers': 4,
    'token_gnn_graph': 'full',
    'token_gnn_layer_scale': 0.15,
}

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:128,seq_d:512'
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 8


# Hyperparameter keys used to build the model. Everything else in
# ``train_config.json`` is ignored when constructing ``PCVRHyFormer``.
_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())

def _load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


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
        ts_fid = cfg.get('ts_fid')
        if ts_fid is not None:
            columns.append(f'{prefix}_{int(ts_fid)}')
    return list(dict.fromkeys(columns))


def _fe08_expected_generated_columns() -> List[str]:
    return [
        'item_dense_feats_86',
        'item_dense_feats_91',
        'item_dense_feats_92',
        'item_int_feats_89',
        'item_int_feats_90',
        'item_int_feats_91',
        'user_int_feats_130',
        'user_int_feats_131',
    ]


def _eval_has_fe08_generated_columns(data_dir: str) -> bool:
    files = _parquet_files(data_dir)
    first_names = pq.ParquetFile(files[0]).schema_arrow.names
    return all(name in first_names for name in _fe08_expected_generated_columns())


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
            raise KeyError(f"FE-08 eval dataset missing required column: {name}")
        dataset._user_int_plan.append((ci, dim, offset, vs))
        offset += dim

    dataset._item_int_plan = []
    offset = 0
    for fid, vs, dim in dataset._item_int_cols:
        name = f'item_int_feats_{fid}'
        ci = dataset._col_idx.get(name)
        if ci is None:
            raise KeyError(f"FE-08 eval dataset missing required column: {name}")
        dataset._item_int_plan.append((ci, dim, offset, vs))
        offset += dim

    dataset._user_dense_plan = []
    offset = 0
    for fid, dim in dataset._user_dense_cols:
        name = f'user_dense_feats_{fid}'
        ci = dataset._col_idx.get(name)
        if ci is None:
            raise KeyError(f"FE-08 eval dataset missing required column: {name}")
        dataset._user_dense_plan.append((ci, dim, offset))
        offset += dim

    dataset._item_dense_plan = []
    offset = 0
    for fid, dim in dataset._item_dense_cols:
        name = f'item_dense_feats_{fid}'
        ci = dataset._col_idx.get(name)
        if ci is None:
            raise KeyError(f"FE-08 eval dataset missing required column: {name}")
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
                raise KeyError(f"FE-08 eval dataset missing required column: {name}")
            vs = dataset.seq_vocab_sizes[domain][fid]
            side_plan.append((ci, slot, vs))
        ts_fid = dataset.ts_fids[domain]
        ts_ci = None
        if ts_fid is not None:
            name = f'{prefix}_{ts_fid}'
            ts_ci = dataset._col_idx.get(name)
            if ts_ci is None:
                raise KeyError(f"FE-08 eval dataset missing required column: {name}")
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


class FE08OnTheFlyEvalDataset(IterableDataset):
    """Generate FE-08 eval features in memory and feed tensors directly."""

    def __init__(
        self,
        data_dir: str,
        schema_path: str,
        model_dir: str,
        infer_batch_size: int,
        transform_batch_size: int,
        seq_max_lens: Dict[str, int],
    ) -> None:
        super().__init__()
        self.files = _parquet_files(data_dir)
        self.schema_path = schema_path
        self.output_schema = _load_json(schema_path)
        self.infer_batch_size = infer_batch_size
        self.transform_batch_size = transform_batch_size

        stats = _load_fe08_stats(model_dir)
        if stats is None:
            raise FileNotFoundError(
                "Checkpoint schema requires FE-08 generated columns, but "
                "fe08_transform_stats.json was not found next to model.pt."
            )
        self.stats = stats
        expected_transform_batch_size = int(stats.get('transform_batch_size', transform_batch_size))
        allow_batch_override = os.environ.get(
            'FE08_EVAL_ALLOW_BATCH_OVERRIDE', '').lower() in {'1', 'true', 'yes'}
        if transform_batch_size != expected_transform_batch_size and not allow_batch_override:
            raise ValueError(
                "FE-08 eval transform_batch_size must match the training builder "
                f"sidecar ({expected_transform_batch_size}); got {transform_batch_size}. "
                "Prefix item-frequency features are batch-boundary sensitive."
            )
        logging.info(
            "Using FE-08 on-the-fly eval transform: "
            f"transform_batch_size={transform_batch_size}, "
            f"infer_batch_size={infer_batch_size}, "
            "exact_generated_features=True")
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
            raise KeyError("FE-08 eval transform missing required raw columns: " + ", ".join(missing_base))

        self.required_output = _schema_required_parquet_columns(schema_path)
        self.required_set = set(self.required_output)
        self.generated = set(_fe08_expected_generated_columns())
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

    def _transform_batch(self, batch: pa.RecordBatch, state: FE08PrefixState) -> pa.RecordBatch:
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
                table = _fe08_append_or_replace_column(
                    table,
                    name,
                    _fe08_fill_int_column(batch.column(idx[name]), fill_value, self.fill_empty_int_lists),
                )

        for name in self.dense_stats:
            if name in idx and name in self.required_set and name not in self.generated:
                table = _fe08_append_or_replace_column(
                    table,
                    name,
                    _normalize_fe08_dense_column(batch.column(idx[name]), name, self.dense_stats),
                )

        feats = _compute_fe08_generated_features_fast(
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
            'item_dense_feats_86',
            'item_dense_feats_91',
            'item_dense_feats_92',
        ]:
            if name not in self.required_set:
                continue
            values = _normalize_fe08_array(name, feats[name], self.dense_stats)
            table = _fe08_append_or_replace_column(
                table, name, pa.array(values, type=pa.float32()))

        for name in [
            'item_int_feats_89',
            'item_int_feats_90',
            'item_int_feats_91',
            'user_int_feats_130',
            'user_int_feats_131',
        ]:
            if name in self.required_set:
                table = _fe08_append_or_replace_column(
                    table, name, pa.array(feats[name], type=pa.int64()))

        table = _fe08_sort_sequence_columns_by_recency(table, self.output_schema)

        missing_output = [name for name in self.required_output if name not in table.schema.names]
        if missing_output:
            raise KeyError(
                "FE-08 on-the-fly eval could not materialize required columns: "
                + ", ".join(missing_output[:20])
            )
        return table.select(self.required_output).to_batches()[0]

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        state = FE08PrefixState()
        processed_rows = 0
        processed_batches = 0
        files = self.files
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            files = [
                path for idx, path in enumerate(files)
                if idx % worker_info.num_workers == worker_info.id
            ]
        for _path, raw_batch in _iter_batches(
            files,
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
                    f"FE-08 on-the-fly eval transformed {processed_rows} rows "
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
        "Could not resolve FE-08 domain_d columns in eval parquet. Expected "
        "domain_d_seq_19 and a timestamp column such as domain_d_seq_26."
    )


def _to_int_array(arr: pa.Array) -> np.ndarray:
    return arr.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=True)


def _bucketize_counts(counts: np.ndarray, edges: Sequence[int]) -> np.ndarray:
    upper_edges = np.asarray(list(edges)[1:], dtype=np.int64)
    return (np.searchsorted(upper_edges, counts, side='right') + 1).astype(np.int64)


def _is_arrow_list(arr: pa.Array) -> bool:
    return pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type)


def _list_offsets_values_int(arr: pa.Array) -> Tuple[np.ndarray, np.ndarray]:
    if not _is_arrow_list(arr):
        raise TypeError(f"Expected Arrow list array, got {arr.type}")
    offsets = arr.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    values = arr.values.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=True)
    return offsets, values


def _first_positive_scalar_array(arr: pa.Array) -> np.ndarray:
    if not _is_arrow_list(arr):
        return _to_int_array(arr)
    offsets, values = _list_offsets_values_int(arr)
    out = np.zeros(len(arr), dtype=np.int64)
    for i in range(len(arr)):
        row = values[offsets[i]:offsets[i + 1]]
        positive = row[row > 0]
        if positive.size:
            out[i] = int(positive[0])
    return out


def _row_ids_from_offsets(offsets: np.ndarray) -> np.ndarray:
    lengths = np.diff(offsets)
    if lengths.size == 0 or int(lengths.sum()) == 0:
        return np.empty(0, dtype=np.int64)
    return np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)


def _compute_prefix_totals_fast(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    timestamps: np.ndarray,
    state: FE08PrefixState,
) -> Tuple[np.ndarray, np.ndarray]:
    num_rows = len(timestamps)
    user_total = np.zeros(num_rows, dtype=np.float32)
    item_total = np.zeros(num_rows, dtype=np.float32)
    row_order = np.argsort(timestamps, kind='stable')
    for i in row_order:
        uid = int(user_ids[i])
        iid = int(item_ids[i])
        ut, it = state.before_update(uid, iid)
        user_total[i] = ut
        item_total[i] = it
        state.update(uid, iid)
    return user_total, item_total


def _compute_target_match_fast(
    batch: pa.RecordBatch,
    target_item_attr: np.ndarray,
    timestamps: np.ndarray,
    match_col: str,
    match_ts_col: str,
    match_window_seconds: int,
    count_edges: Sequence[int],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    idx = {name: i for i, name in enumerate(batch.schema.names)}
    values_arr = batch.column(idx[match_col])
    times_arr = batch.column(idx[match_ts_col])
    if (
        not _is_arrow_list(values_arr)
        or not _is_arrow_list(times_arr)
        or values_arr.values.null_count
        or times_arr.values.null_count
    ):
        return None

    value_offsets, seq_values = _list_offsets_values_int(values_arr)
    time_offsets, seq_times = _list_offsets_values_int(times_arr)
    if not np.array_equal(np.diff(value_offsets), np.diff(time_offsets)):
        return None

    num_rows = batch.num_rows
    row_ids = _row_ids_from_offsets(value_offsets)
    match_count = np.zeros(num_rows, dtype=np.int64)
    min_match_delta = np.zeros(num_rows, dtype=np.float32)
    match_count_7d = np.zeros(num_rows, dtype=np.float32)
    latest_match_bucket = np.zeros(num_rows, dtype=np.int64)

    if row_ids.size:
        target_for_event = target_item_attr[row_ids]
        valid_time = (seq_times > 0) & (seq_times <= timestamps[row_ids])
        matched = (target_for_event > 0) & (seq_values == target_for_event) & valid_time
        if bool(matched.any()):
            match_count = np.bincount(row_ids[matched], minlength=num_rows).astype(np.int64)
            matched_rows = row_ids[matched]
            deltas = (timestamps[matched_rows] - seq_times[matched]).astype(np.int64)
            min_delta = np.full(num_rows, np.iinfo(np.int64).max, dtype=np.int64)
            np.minimum.at(min_delta, matched_rows, deltas)
            observed = min_delta != np.iinfo(np.int64).max
            min_match_delta[observed] = min_delta[observed].astype(np.float32)
            match_count_7d = np.bincount(
                matched_rows[deltas <= match_window_seconds],
                minlength=num_rows,
            ).astype(np.float32)
            raw_bucket = np.searchsorted(BUCKET_BOUNDARIES, min_delta[observed])
            latest_match_bucket[observed] = (
                np.minimum(raw_bucket, len(BUCKET_BOUNDARIES) - 1) + 1
            ).astype(np.int64)

    has_match = np.where(
        target_item_attr > 0,
        np.where(match_count > 0, 2, 1),
        0,
    ).astype(np.int64)
    return (
        has_match,
        _bucketize_counts(match_count, count_edges),
        latest_match_bucket,
        min_match_delta,
        match_count_7d,
    )


def _compute_fe08_generated_features_fast(
    batch: pa.RecordBatch,
    state: FE08PrefixState,
    match_col: str,
    match_ts_col: str,
    seq_ts_cols: Dict[str, str],
    min_timestamp: int,
    match_window_seconds: int,
    count_edges: Sequence[int],
) -> Dict[str, np.ndarray]:
    names = batch.schema.names
    idx = {name: i for i, name in enumerate(names)}
    user_ids = _to_int_array(batch.column(idx['user_id']))
    item_ids = _to_int_array(batch.column(idx['item_id']))
    timestamps = _to_int_array(batch.column(idx['timestamp']))

    if 'item_int_feats_9' not in idx:
        raise KeyError("FE-08 requires item_int_feats_9 for target-history match")
    target_item_attr = _first_positive_scalar_array(batch.column(idx['item_int_feats_9']))

    match_features = _compute_target_match_fast(
        batch,
        target_item_attr,
        timestamps,
        match_col,
        match_ts_col,
        match_window_seconds,
        count_edges,
    )
    if match_features is None:
        return _fe08_compute_generated_features(
            batch,
            state,
            match_col,
            match_ts_col,
            seq_ts_cols,
            min_timestamp,
            match_window_seconds,
            count_edges,
        )
    (
        has_match,
        match_count_bucket,
        latest_match_bucket,
        min_match_delta,
        match_count_7d,
    ) = match_features

    _, item_total = _compute_prefix_totals_fast(user_ids, item_ids, timestamps, state)
    hour_id = ((timestamps // 3600) % 24 + 1).astype(np.int64)
    dow_id = (((timestamps // 86400) + 4) % 7 + 1).astype(np.int64)

    return {
        'item_dense_feats_86': np.log1p(item_total).astype(np.float32),
        'item_int_feats_89': has_match,
        'item_int_feats_90': match_count_bucket,
        'item_int_feats_91': latest_match_bucket,
        'item_dense_feats_91': np.log1p(min_match_delta).astype(np.float32),
        'item_dense_feats_92': np.log1p(match_count_7d).astype(np.float32),
        'user_int_feats_130': hour_id,
        'user_int_feats_131': dow_id,
    }


def _resolve_fe08_sidecar(model_dir: str, filename: str) -> Optional[str]:
    local_path = os.path.join(model_dir, filename)
    if os.path.exists(local_path):
        return local_path
    stats_dir = os.environ.get('FE08_STATS_DIR') or os.environ.get('FE08_STATS_PATH')
    if stats_dir:
        candidate = os.path.join(stats_dir, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def _load_fe08_stats(model_dir: str) -> Optional[Dict[str, Any]]:
    stats_path = _resolve_fe08_sidecar(model_dir, 'fe08_transform_stats.json')
    if stats_path is None:
        return None
    logging.info(f"Using FE-08 transform stats: {stats_path}")
    return _load_json(stats_path)


def _require_fe08_sidecars(model_dir: str, schema_path: str) -> None:
    required = [
        'fe08_transform_stats.json',
        'fe08_dense_normalization_stats.json',
        'dropped_feats.may7.json',
    ]
    resolved: Dict[str, str] = {}
    for filename in required:
        path = _resolve_fe08_sidecar(model_dir, filename)
        if path is None:
            raise FileNotFoundError(
                f"FE-08 checkpoint package is missing required sidecar: {filename}. "
                "Evaluation must reuse training transform stats and drop decisions."
            )
        resolved[filename] = path

    dropped = _load_json(resolved['dropped_feats.may7.json'])
    schema = _load_json(schema_path)
    schema_user_int = {int(row[0]) for row in schema.get('user_int', [])}
    schema_item_int = {int(row[0]) for row in schema.get('item_int', [])}
    dropped_user_int = {int(fid) for fid in dropped.get('user_int', [])}
    dropped_item_int = {int(fid) for fid in dropped.get('item_int', [])}
    leaked_user = dropped_user_int & schema_user_int
    leaked_item = dropped_item_int & schema_item_int
    if leaked_user or leaked_item:
        raise ValueError(
            "FE-08 schema conflicts with dropped_feats.may7.json: "
            f"user_int={sorted(leaked_user)}, item_int={sorted(leaked_item)}."
        )
    logging.info(
        "FE-08 dropped feature sidecar: "
        f"user_int={sorted(dropped_user_int)}, "
        f"item_int={sorted(dropped_item_int)}, "
        f"threshold={dropped.get('threshold')}")


def _schema_has_fe08_columns(schema_path: str) -> bool:
    schema = _load_json(schema_path)
    user_int = {int(row[0]) for row in schema.get('user_int', [])}
    item_dense = {int(row[0]) for row in schema.get('item_dense', [])}
    item_int = {int(row[0]) for row in schema.get('item_int', [])}
    return bool(
        {130, 131} & user_int
        or {86, 91, 92} & item_dense
        or {89, 90, 91} & item_int
    )


def _normalize_fe08_array(
    name: str,
    values: np.ndarray,
    dense_stats: Dict[str, Dict[str, float]],
) -> np.ndarray:
    if name not in dense_stats:
        raise KeyError(f"fe08_transform_stats.json missing dense_stats.{name}")
    stat = dense_stats[name]
    mean = float(stat.get('mean', 0.0))
    std = max(float(stat.get('std', 1.0)), 1e-6)
    return ((values.astype(np.float32) - mean) / std).astype(np.float32)


def _normalize_fe08_dense_column(
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


def maybe_build_fe08_eval_dataset(
    data_dir: str,
    schema_path: str,
    model_dir: str,
    batch_size: int,
    result_dir: str,
) -> str:
    """Materialize FE-08 generated columns and FE00-literal transforms for raw eval parquet."""
    schema_has_fe08 = _schema_has_fe08_columns(schema_path)
    if not schema_has_fe08:
        return data_dir

    files = _parquet_files(data_dir)
    first_names = pq.ParquetFile(files[0]).schema_arrow.names
    expected_generated = _fe08_expected_generated_columns()
    if all(name in first_names for name in expected_generated):
        logging.info("Eval parquet already contains FE-08 generated columns; using raw eval directory.")
        return data_dir

    stats = _load_fe08_stats(model_dir)
    if stats is None:
        raise FileNotFoundError(
            "Checkpoint schema requires FE-08 generated columns, but "
            "fe08_transform_stats.json was not found next to model.pt. "
            "Copy the FE-08 sidecars from the training feature directory into "
            "MODEL_OUTPUT_PATH, or set FE08_STATS_DIR to that directory."
        )

    dense_stats = stats.get('dense_stats', {})
    count_edges = [int(x) for x in stats.get('match_count_buckets', [0, 1, 2, 4, 8])]
    match_col = str(stats.get('match_col', 'domain_d_seq_19'))
    match_ts_col = str(stats.get('match_ts_col', 'domain_d_seq_26'))
    seq_ts_cols = {str(k): str(v) for k, v in stats.get('seq_time_columns', {}).items()}
    min_timestamp = int(stats.get('min_timestamp_for_day_since', 0))
    match_window_seconds = int(stats.get('match_window_days', 7)) * 86400
    int_fill_audit = stats.get('int_fill_audit', {})
    output_schema = _load_json(schema_path)
    sidecar_transform_batch_size = int(stats.get('transform_batch_size', 8192))
    transform_batch_size = int(os.environ.get(
        'FE08_EVAL_TRANSFORM_BATCH_SIZE',
        sidecar_transform_batch_size,
    ))
    allow_batch_override = os.environ.get(
        'FE08_EVAL_ALLOW_BATCH_OVERRIDE', '').lower() in {'1', 'true', 'yes'}
    if transform_batch_size != sidecar_transform_batch_size and not allow_batch_override:
        raise ValueError(
            "FE-08 eval materialization transform_batch_size must match the "
            f"training builder sidecar ({sidecar_transform_batch_size}); got "
            f"{transform_batch_size}."
        )

    missing_base = [
        name for name in ('user_id', 'item_id', 'timestamp', 'item_int_feats_9', match_col, match_ts_col)
        if name not in first_names
    ]
    if missing_base:
        raise KeyError("FE-08 eval transform missing required raw columns: " + ", ".join(missing_base))

    output_base_dir = os.environ.get('FE08_EVAL_DATA_DIR')
    if output_base_dir:
        os.makedirs(output_base_dir, exist_ok=True)
        output_dir = tempfile.mkdtemp(prefix='materialized_', dir=output_base_dir)
    else:
        output_dir = tempfile.mkdtemp(prefix='taac_fe08_eval_')
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Writing FE-08 eval parquet to: {output_dir}")
    logging.info(f"FE-08 eval transform batch_size={transform_batch_size}")
    logging.info(
        "FE-08 eval transform ignores eval label_type; total-frequency state "
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
    state = FE08PrefixState()
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
                    f"FE-08 eval transform processed {processed_rows} rows "
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
                    table = _fe08_append_or_replace_column(
                        table, name, _fe08_fill_int_column(batch.column(idx[name]), fill_value, fill_empty_int_lists))

            for name in dense_stats:
                if name in idx and name in required_set and name not in generated_set:
                    table = _fe08_append_or_replace_column(
                        table, name, _normalize_fe08_dense_column(batch.column(idx[name]), name, dense_stats))

            feats = _compute_fe08_generated_features_fast(
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
                'item_dense_feats_86',
                'item_dense_feats_91',
                'item_dense_feats_92',
            ]:
                if name not in required_set:
                    continue
                values = _normalize_fe08_array(name, feats[name], dense_stats)
                table = _fe08_append_or_replace_column(
                    table, name, pa.array(values, type=pa.float32()))

            for name in [
                'item_int_feats_89',
                'item_int_feats_90',
                'item_int_feats_91',
                'user_int_feats_130',
                'user_int_feats_131',
            ]:
                if name in required_set:
                    table = _fe08_append_or_replace_column(
                        table, name, pa.array(feats[name], type=pa.int64()))

            table = _fe08_sort_sequence_columns_by_recency(table, output_schema)

            missing_output = [name for name in required_output if name not in table.schema.names]
            if missing_output:
                raise KeyError(
                    "FE-08 eval transform could not materialize required columns: "
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

    FE-08 evaluation is tied to the saved training contract; missing
    ``train_config.json`` is treated as an invalid checkpoint package.
    """
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    raise FileNotFoundError(
        f"FE-08 checkpoint package is missing train_config.json under {model_dir}. "
        "Evaluation must rebuild the TokenGNN model from the saved training config."
    )


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


def _validate_fe08_contract(
    train_config: Dict[str, Any],
    model_cfg: Dict[str, Any],
    schema_path: str,
) -> None:
    expected_cfg = {
        'd_model': 136,
        'emb_dim': 64,
        'num_queries': 2,
        'seq_encoder_type': 'transformer',
        'dropout_rate': 0.05,
        'seq_top_k': 100,
        'rank_mixer_mode': 'full',
        'ns_tokenizer_type': 'rankmixer',
        'user_ns_tokens': 5,
        'item_ns_tokens': 2,
        'use_token_gnn': True,
        'token_gnn_layers': 4,
        'token_gnn_graph': 'full',
        'emb_skip_threshold': 1000000,
    }
    for key, expected in expected_cfg.items():
        actual = model_cfg.get(key)
        if actual != expected:
            raise ValueError(
                f"FE-08 config mismatch: {key}={actual!r}, expected {expected!r}."
            )
    if abs(float(model_cfg.get('token_gnn_layer_scale', 0.0)) - 0.15) > 1e-12:
        raise ValueError(
            "FE-08 config mismatch: token_gnn_layer_scale="
            f"{model_cfg.get('token_gnn_layer_scale')!r}, expected 0.15."
        )
    if train_config.get('domain_time_buckets', False):
        raise ValueError("FE-08 must not enable FE-07 P2-Domain domain_time_buckets.")

    schema = _load_json(schema_path)
    user_int = {int(row[0]) for row in schema.get('user_int', [])}
    item_int = {int(row[0]) for row in schema.get('item_int', [])}
    user_dense = {int(row[0]) for row in schema.get('user_dense', [])}
    item_dense = {int(row[0]) for row in schema.get('item_dense', [])}

    missing_user_int = {130, 131} - user_int
    missing_item_int = {89, 90, 91} - item_int
    missing_item_dense = {86, 91, 92} - item_dense
    if missing_user_int or missing_item_int or missing_item_dense:
        raise ValueError(
            "FE-08 schema missing required generated fids: "
            f"user_int={sorted(missing_user_int)}, "
            f"item_int={sorted(missing_item_int)}, "
            f"item_dense={sorted(missing_item_dense)}."
        )
    forbidden_user_dense = {110, 120, 121} & user_dense
    forbidden_item_dense = {87, 88} & item_dense
    if forbidden_user_dense or forbidden_item_dense:
        raise ValueError(
            "FE-08 schema contains non-mainline generated fids: "
            f"user_dense={sorted(forbidden_user_dense)}, "
            f"item_dense={sorted(forbidden_item_dense)}."
        )


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
    _require_fe08_sidecars(model_dir, schema_path)

    # ---- Parse seq_max_lens ----
    sml_str = train_config.get('seq_max_lens', _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")

    # ---- Data loading: reuse batch_size / num_workers from training config ----
    batch_size = int(train_config.get('batch_size', _FALLBACK_BATCH_SIZE))
    num_workers = int(train_config.get('num_workers', _FALLBACK_NUM_WORKERS))

    schema_has_fe08 = _schema_has_fe08_columns(schema_path)
    eval_has_fe08 = _eval_has_fe08_generated_columns(data_dir) if schema_has_fe08 else False
    force_materialize = os.environ.get('FE08_EVAL_MATERIALIZE', '').lower() in {'1', 'true', 'yes'}
    use_on_the_fly_fe08 = schema_has_fe08 and not eval_has_fe08 and not force_materialize

    if use_on_the_fly_fe08:
        fe08_stats = _load_fe08_stats(model_dir)
        default_transform_batch_size = 8192
        if fe08_stats is not None:
            default_transform_batch_size = int(
                fe08_stats.get('transform_batch_size', default_transform_batch_size)
            )
        transform_batch_size = int(os.environ.get(
            'FE08_EVAL_TRANSFORM_BATCH_SIZE',
            default_transform_batch_size,
        ))
        test_dataset = FE08OnTheFlyEvalDataset(
            data_dir=data_dir,
            schema_path=schema_path,
            model_dir=model_dir,
            infer_batch_size=batch_size,
            transform_batch_size=transform_batch_size,
            seq_max_lens=seq_max_lens,
        )
        loader_num_workers = 0
    else:
        # Compatibility path: use pre-generated FE-08 parquet, or opt into
        # materialization with FE08_EVAL_MATERIALIZE=1 for debugging.
        data_dir = maybe_build_fe08_eval_dataset(
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
        )
        loader_num_workers = num_workers
    total_test_samples = test_dataset.num_rows
    logging.info(f"Total test samples: {total_test_samples}")

    # ---- Build model: every structural hyperparameter is resolved from train_config ----
    model_cfg = resolve_model_cfg(train_config)
    if train_config.get('use_time_buckets', True):
        model_cfg['num_time_buckets'] = test_dataset.num_time_buckets
    _validate_fe08_contract(train_config, model_cfg, schema_path)

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
    logging.info(f"Eval DataLoader num_workers={loader_num_workers}")
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
