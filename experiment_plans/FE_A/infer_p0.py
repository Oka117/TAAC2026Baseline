"""P0 evaluation script — based on the baseline infer.py shipped with the
evaluation container, plus verbose dataset-attribute printing.

Why this file exists
--------------------
P0 is the calibration ablation that trains with ``--rank_mixer_mode ffn_only``
on top of the published baseline NS layout. Two reasons the baseline
``infer.py`` already works for P0 ckpts:

1. ``rank_mixer_mode`` is recorded in ``train_config.json`` (written by
   ``trainer.py`` at ckpt save time), so ``resolve_model_cfg`` will rebuild
   the model with the right mode. No code change needed.
2. P0 does not enable Plan A (``match_pairs_json=""``), so the saved
   ``state_dict`` has the same shape as a vanilla baseline ckpt — strict
   load succeeds.

What this file adds
-------------------
- Prints every test-set attribute name we can derive from the dataset:
  parquet column names, FeatureSchema entries (user_int / item_int /
  user_dense), seq domain / sideinfo / ts fids, vocab sizes, sequence
  truncation lengths, and the contents of MODEL_OUTPUT_PATH.
- Prints the first batch's tensor shapes before prediction so we have a
  sanity check that train/eval feature plumbing matches.
- Stays forward-compatible with the modified ``model.py`` that adds
  ``match_feats_dim`` / ``match_inject_mode`` (used by P2/P3/P4): unknown
  kwargs are filtered via ``inspect.signature`` and the optional
  ``match_feats`` field on ``ModelInput`` is passed only when the field
  exists.

Environment variables (same as baseline infer.py):
    MODEL_OUTPUT_PATH  Checkpoint directory (with model.pt + train_config.json).
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH   Directory for the generated ``predictions.json``.
"""

import os
import json
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pyarrow.parquet as pq

from dataset import FeatureSchema, PCVRParquetDataset, NUM_TIME_BUCKETS
from model import PCVRHyFormer, ModelInput


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# ---------------------------------------------------------------------------
# Fallback config (used only when train_config.json is absent). All keys must
# match either the baseline ``train.py`` argparse defaults OR the Plan-A
# extensions; unknown-to-PCVRHyFormer keys are dropped via inspect.signature.
# ---------------------------------------------------------------------------
_FALLBACK_MODEL_CFG = {
    'd_model': 64,
    'emb_dim': 64,
    'num_queries': 2,
    'num_hyformer_blocks': 2,
    'num_heads': 4,
    'seq_encoder_type': 'transformer',
    'hidden_mult': 4,
    'dropout_rate': 0.01,
    'seq_top_k': 50,
    'seq_causal': False,
    'action_num': 1,
    'num_time_buckets': NUM_TIME_BUCKETS,
    # P0 default; train_config.json overrides if present.
    'rank_mixer_mode': 'ffn_only',
    'use_rope': False,
    'rope_base': 10000.0,
    'emb_skip_threshold': 1000000,
    'seq_id_threshold': 10000,
    'ns_tokenizer_type': 'rankmixer',
    'user_ns_tokens': 5,
    'item_ns_tokens': 2,
    # Plan-A keys: harmless no-ops on P0; required for P2/P3/P4 reuse.
    'match_feats_dim': 0,
    'match_inject_mode': 'ns_token',
}

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 16

_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def parse_seq_max_lens(s: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for pair in s.split(','):
        k, v = pair.split(':')
        out[k.strip()] = int(v.strip())
    return out


def load_train_config(model_dir: str) -> Dict[str, Any]:
    p = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(p):
        with open(p, 'r') as f:
            return json.load(f)
    logging.warning(f"train_config.json not found in {model_dir}; using fallbacks.")
    return {}


def resolve_model_cfg(train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for k in _MODEL_CFG_KEYS:
        if k == 'num_time_buckets':
            if 'num_time_buckets' in train_cfg:
                cfg[k] = train_cfg[k]
            elif 'use_time_buckets' in train_cfg:
                cfg[k] = NUM_TIME_BUCKETS if train_cfg['use_time_buckets'] else 0
            else:
                cfg[k] = _FALLBACK_MODEL_CFG[k]
            continue
        cfg[k] = train_cfg.get(k, _FALLBACK_MODEL_CFG[k])

    # Drop keys the deployed PCVRHyFormer doesn't accept (so this script
    # works against both the baseline and the Plan-A modified model.py).
    accepted = set(inspect.signature(PCVRHyFormer.__init__).parameters)
    dropped = [k for k in list(cfg.keys()) if k not in accepted]
    if dropped:
        logging.info(f"[CFG] dropping kwargs not accepted by PCVRHyFormer: {dropped}")
    cfg = {k: v for k, v in cfg.items() if k in accepted}
    return cfg


# ---------------------------------------------------------------------------
# Dataset attribute printing
# ---------------------------------------------------------------------------
def print_test_dataset_attributes(
    dataset: PCVRParquetDataset,
    schema_path: str,
    model_dir: str,
) -> None:
    """Dump everything we can know about the test dataset structure to stdout.

    Uses ``print(...)`` (not ``logging.info``) so it shows up in any platform
    that captures stdout, regardless of logger handler configuration.
    """
    bar = "=" * 78
    print(bar, flush=True)
    print("[TEST DATASET ATTRIBUTES]", flush=True)
    print(bar, flush=True)

    # 1. Parquet files & row counts
    print(f"  parquet_files ({len(dataset._parquet_files)}):", flush=True)
    for f in dataset._parquet_files:
        try:
            pf = pq.ParquetFile(f)
            n_rg = pf.metadata.num_row_groups
            n_rows = sum(pf.metadata.row_group(i).num_rows for i in range(n_rg))
            print(f"    {f}  rows={n_rows}, row_groups={n_rg}", flush=True)
        except Exception as e:
            print(f"    {f}  (cannot inspect: {e})", flush=True)

    # 2. Raw parquet column names
    if dataset._parquet_files:
        try:
            pf = pq.ParquetFile(dataset._parquet_files[0])
            names = list(pf.schema_arrow.names)
            print(f"  parquet_columns ({len(names)}):", flush=True)
            # Print in groups of 6 columns per line for readability
            for i in range(0, len(names), 6):
                print(f"    {names[i:i+6]}", flush=True)
        except Exception as e:
            print(f"  (cannot read columns: {e})", flush=True)

    # 3. schema metadata
    print(f"  schema_path: {schema_path}", flush=True)
    print(f"  total_rows : {dataset.num_rows}", flush=True)
    print(f"  batch_size : {dataset.batch_size}", flush=True)

    # 4. user_int schema
    print(f"\n  USER_INT  total_dim={dataset.user_int_schema.total_dim}, "
          f"num_fids={len(dataset.user_int_schema.entries)}:", flush=True)
    for fid, offset, length in dataset.user_int_schema.entries:
        vs = dataset.user_int_vocab_sizes[offset]
        col_name = f"user_int_feats_{fid}"
        print(f"    {col_name:<28} fid={fid:<4} offset={offset:<5} "
              f"length={length:<3} vocab_size={vs}", flush=True)

    # 5. item_int schema
    print(f"\n  ITEM_INT  total_dim={dataset.item_int_schema.total_dim}, "
          f"num_fids={len(dataset.item_int_schema.entries)}:", flush=True)
    for fid, offset, length in dataset.item_int_schema.entries:
        vs = dataset.item_int_vocab_sizes[offset]
        col_name = f"item_int_feats_{fid}"
        print(f"    {col_name:<28} fid={fid:<4} offset={offset:<5} "
              f"length={length:<3} vocab_size={vs}", flush=True)

    # 6. user_dense schema
    print(f"\n  USER_DENSE total_dim={dataset.user_dense_schema.total_dim}, "
          f"num_fids={len(dataset.user_dense_schema.entries)}:", flush=True)
    for fid, offset, length in dataset.user_dense_schema.entries:
        col_name = f"user_dense_feats_{fid}"
        print(f"    {col_name:<28} fid={fid:<4} offset={offset:<5} length={length}",
              flush=True)

    # 7. seq domains
    print(f"\n  SEQ_DOMAINS: {dataset.seq_domains}", flush=True)
    for d in dataset.seq_domains:
        prefix = dataset._seq_prefix[d]
        ts_fid = dataset.ts_fids[d]
        sideinfo = dataset.sideinfo_fids[d]
        vs_list = dataset.seq_domain_vocab_sizes[d]
        max_len = dataset._seq_maxlen[d]
        print(f"    {d}:", flush=True)
        print(f"      prefix         = {prefix!r}", flush=True)
        print(f"      ts_fid         = {ts_fid}  ({prefix}_{ts_fid})"
              if ts_fid is not None else f"      ts_fid         = None",
              flush=True)
        print(f"      max_len        = {max_len}", flush=True)
        print(f"      sideinfo count = {len(sideinfo)}", flush=True)
        for slot, (sf, vs) in enumerate(zip(sideinfo, vs_list)):
            print(f"        slot={slot:<2} fid={sf:<4} col={prefix}_{sf:<4} "
                  f"vocab_size={vs}", flush=True)

    # 8. MODEL_OUTPUT_PATH contents
    print(f"\n  MODEL_DIR: {model_dir}", flush=True)
    if model_dir and os.path.isdir(model_dir):
        for f in sorted(os.listdir(model_dir)):
            full = os.path.join(model_dir, f)
            try:
                sz = os.path.getsize(full)
                print(f"    {f}  ({sz} bytes)", flush=True)
            except OSError:
                print(f"    {f}  (cannot stat)", flush=True)
    print(bar, flush=True)


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    ns_groups_json: Optional[str] = None,
    device: str = 'cpu',
) -> PCVRHyFormer:
    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info(f"Loading NS groups from {ns_groups_json}")
        with open(ns_groups_json, 'r') as f:
            ns = json.load(f)
        u_idx = {fid: i for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)}
        i_idx = {fid: i for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)}
        try:
            user_ns_groups = [
                [u_idx[f] for f in fids]
                for fids in ns['user_ns_groups'].values()
            ]
            item_ns_groups = [
                [i_idx[f] for f in fids]
                for fids in ns['item_ns_groups'].values()
            ]
        except KeyError as exc:
            raise KeyError(
                f"NS-groups JSON references fid {exc.args[0]} not in schema."
            ) from exc
    else:
        logging.info("No NS groups JSON: each feature is one group")
        user_ns_groups = [[i] for i in range(len(dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(dataset.item_int_schema.entries))]

    user_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes)

    logging.info(f"Building PCVRHyFormer with cfg: {model_cfg}")
    model = PCVRHyFormer(
        user_int_feature_specs=user_specs,
        item_int_feature_specs=item_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        **model_cfg,
    ).to(device)
    return model


def get_ckpt_path() -> Optional[str]:
    p = os.environ.get("MODEL_OUTPUT_PATH")
    if not p:
        return None
    for it in os.listdir(p):
        if it.endswith(".pt"):
            return os.path.join(p, it)
    return None


def batch_to_model_input(batch: Dict[str, Any], device: str) -> ModelInput:
    """Construct ``ModelInput`` from a batch dict, forward-compatible with
    both the baseline ``ModelInput`` (no match_feats field) and the Plan-A
    modified one (optional match_feats with empty default).
    """
    db: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            db[k] = v.to(device, non_blocking=True)
        else:
            db[k] = v

    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_tb: Dict[str, torch.Tensor] = {}
    for d in db['_seq_domains']:
        seq_data[d] = db[d]
        seq_lens[d] = db[f'{d}_len']
        B, _, L = db[d].shape
        seq_tb[d] = db.get(
            f'{d}_time_bucket',
            torch.zeros(B, L, dtype=torch.long, device=device),
        )

    base_kwargs = dict(
        user_int_feats=db['user_int_feats'],
        item_int_feats=db['item_int_feats'],
        user_dense_feats=db['user_dense_feats'],
        item_dense_feats=db['item_dense_feats'],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_tb,
    )
    # Plan A passthrough — only included if both dataset emitted it AND
    # the deployed ModelInput accepts it.
    if 'match_feats' in db and 'match_feats' in ModelInput._fields:
        return ModelInput(**base_kwargs, match_feats=db['match_feats'])
    return ModelInput(**base_kwargs)


def main() -> None:
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')
    if not (model_dir and data_dir and result_dir):
        raise SystemExit(
            "Missing one of MODEL_OUTPUT_PATH / EVAL_DATA_PATH / EVAL_RESULT_PATH"
        )
    os.makedirs(result_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[ENV] MODEL_OUTPUT_PATH={model_dir}", flush=True)
    print(f"[ENV] EVAL_DATA_PATH={data_dir}", flush=True)
    print(f"[ENV] EVAL_RESULT_PATH={result_dir}", flush=True)
    print(f"[ENV] device={device}", flush=True)

    # Schema preference: model_dir's schema.json (matches training) > data_dir.
    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    print(f"[CFG] schema={schema_path}", flush=True)

    train_config = load_train_config(model_dir)
    print(f"[CFG] train_config keys: {sorted(train_config.keys())}", flush=True)

    sml_str = train_config.get('seq_max_lens', _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = parse_seq_max_lens(sml_str)
    batch_size = int(train_config.get('batch_size', _FALLBACK_BATCH_SIZE))
    num_workers = int(train_config.get('num_workers', _FALLBACK_NUM_WORKERS))
    print(f"[CFG] seq_max_lens={seq_max_lens}", flush=True)
    print(f"[CFG] batch_size={batch_size}, num_workers={num_workers}", flush=True)

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
    )

    # ---- attribute dump ----
    print_test_dataset_attributes(test_dataset, schema_path, model_dir)

    # ---- build model ----
    model_cfg = resolve_model_cfg(train_config)
    print(f"[CFG] resolved rank_mixer_mode={model_cfg.get('rank_mixer_mode')}, "
          f"user_ns_tokens={model_cfg.get('user_ns_tokens')}, "
          f"item_ns_tokens={model_cfg.get('item_ns_tokens')}, "
          f"d_model={model_cfg.get('d_model')}", flush=True)

    ns_groups_json = train_config.get('ns_groups_json', None)
    if ns_groups_json:
        local = os.path.join(model_dir, os.path.basename(ns_groups_json))
        if os.path.exists(local):
            ns_groups_json = local

    model = build_model(test_dataset, model_cfg, ns_groups_json, device)
    n_params = sum(p.numel() for p in model.parameters())
    n_ns = getattr(model, 'num_ns', None)
    T = (model_cfg.get('num_queries', 0) * model.num_sequences + n_ns
         if n_ns is not None else None)
    print(f"[MODEL] total parameters: {n_params:,}, num_ns={n_ns}, T={T}",
          flush=True)

    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"Contents: {os.listdir(model_dir)}"
        )
    print(f"[MODEL] loading ckpt: {ckpt_path}", flush=True)
    state_dict = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
        print("[MODEL] strict load OK", flush=True)
    except RuntimeError as e:
        print(f"[MODEL] strict load FAILED: {e}", flush=True)
        raise

    model.eval()

    # ---- DataLoader for test ----
    dl_kwargs = dict(
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        dl_kwargs['prefetch_factor'] = 2
    test_loader = DataLoader(test_dataset, **dl_kwargs)

    # ---- Inspect first batch shape (sanity check) ----
    print("\n[FIRST BATCH SHAPE INSPECTION]", flush=True)
    first_batch_inspected = False
    all_probs: List[float] = []
    all_user_ids: List[Any] = []

    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if not first_batch_inspected:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k:<30} shape={tuple(v.shape)}  dtype={v.dtype}",
                              flush=True)
                    elif isinstance(v, list):
                        print(f"    {k:<30} list[{len(v)}]", flush=True)
                    else:
                        print(f"    {k:<30} type={type(v).__name__}", flush=True)
                print("[FIRST BATCH] done; starting prediction loop", flush=True)
                first_batch_inspected = True

            mi = batch_to_model_input(batch, device)
            uids = batch.get('user_id', [])
            logits, _ = model.predict(mi)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_user_ids.extend(uids)

            if (bi + 1) % 100 == 0:
                print(f"[INFER] processed ~{(bi + 1) * batch_size} samples",
                      flush=True)

    print(f"[INFER] complete: {len(all_probs)} predictions", flush=True)

    out = os.path.join(result_dir, 'predictions.json')
    with open(out, 'w') as f:
        json.dump({"predictions": dict(zip(all_user_ids, all_probs))}, f)
    print(f"[INFER] saved predictions to {out}", flush=True)


if __name__ == "__main__":
    main()
