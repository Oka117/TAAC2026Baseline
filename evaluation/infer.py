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
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FeatureSchema, PCVRParquetDataset, NUM_TIME_BUCKETS
from model import PCVRHyFormer, ModelInput


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# Fallback values used only when ``train_config.json`` is missing from the
# ckpt directory.
#
# These mirror the current ``run.sh`` main configuration so the fallback path
# can still rebuild the submitted model if ``train_config.json`` is missing.
#
# Special note on ``num_time_buckets``: this value is strictly determined by
# ``dataset.BUCKET_BOUNDARIES`` and is NOT an independent hyperparameter.
# When the feature is enabled we therefore use the constant exposed by the
# dataset module; ``0`` means disabled.
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
    'rank_mixer_mode': 'full',
    'use_rope': False,
    'rope_base': 10000.0,
    'emb_skip_threshold': 1000000,
    'seq_id_threshold': 10000,
    'ns_tokenizer_type': 'rankmixer',
    'user_ns_tokens': 5,
    'item_ns_tokens': 2,
    # GNN-NS defaults. These are used only when train_config.json is missing
    # the keys; normal evaluation should read the exact values saved during
    # training.
    'use_token_gnn': True,
    'token_gnn_layers': 4,
    'token_gnn_graph': 'full',
    'token_gnn_layer_scale': 0.1,
    'use_seq_graph': False,
    'seq_graph_layers': 2,
    'seq_graph_layer_scale': 0.1,
    'seq_graph_use_target': True,
    'graph_output_fusion': True,
    'output_include_ns': True,
    'use_aligned_dense_int_graph': False,
    'aligned_graph_fids': '62,63,64,65,66,89,90,91',
    'aligned_graph_layers': 1,
    'aligned_graph_tokens': 8,
    'aligned_graph_top_k': 64,
}

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 16


# Hyperparameter keys used to build the model. Everything else in
# ``train_config.json`` is ignored when constructing ``PCVRHyFormer``.
_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


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


def _step_number(path: str) -> int:
    """Best-effort parser for names like global_step2500.best_model."""
    base = os.path.basename(os.path.dirname(path))
    if base.startswith('global_step'):
        rest = base[len('global_step'):]
        digits = []
        for ch in rest:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if digits:
            return int(''.join(digits))
    return -1


def resolve_model_dir_and_ckpt_path(model_output_path: str) -> Tuple[str, str]:
    """Resolve a robust checkpoint directory and ``model.pt`` path.

    Evaluation platforms may pass either the exact ``global_step*.best_model``
    directory or its parent checkpoint root. This function supports both and
    chooses a best-model checkpoint when several candidates exist.
    """
    if not model_output_path:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    if not os.path.exists(model_output_path):
        raise FileNotFoundError(
            f"MODEL_OUTPUT_PATH does not exist: {model_output_path!r}")

    if os.path.isfile(model_output_path):
        if not model_output_path.endswith('.pt'):
            raise FileNotFoundError(
                f"MODEL_OUTPUT_PATH is a file but not a .pt checkpoint: "
                f"{model_output_path!r}")
        return os.path.dirname(model_output_path), model_output_path

    direct = sorted(glob.glob(os.path.join(model_output_path, '*.pt')))
    if direct:
        model_pt = [
            p for p in direct if os.path.basename(p) == 'model.pt'
        ]
        ckpt_path = model_pt[0] if model_pt else direct[0]
        return os.path.dirname(ckpt_path), ckpt_path

    patterns = [
        os.path.join(model_output_path, 'global_step*.best_model', 'model.pt'),
        os.path.join(model_output_path, 'global_step*', 'model.pt'),
    ]
    candidates: List[str] = []
    seen = set()
    for pattern in patterns:
        for path in glob.glob(pattern):
            if path in seen or not os.path.isfile(path):
                continue
            candidates.append(path)
            seen.add(path)

    if not candidates:
        listing = os.listdir(model_output_path)
        raise FileNotFoundError(
            f"No .pt checkpoint found under MODEL_OUTPUT_PATH={model_output_path!r}. "
            f"Top-level contents: {listing}")

    def _rank(path: str) -> Tuple[int, int, str]:
        parent = os.path.basename(os.path.dirname(path))
        is_best = 1 if parent.endswith('.best_model') else 0
        return (is_best, _step_number(path), path)

    ckpt_path = max(candidates, key=_rank)
    return os.path.dirname(ckpt_path), ckpt_path


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
        user_int_feature_ids=dataset.user_int_schema.feature_ids,
        user_dense_feature_specs=dataset.user_dense_schema.entries,
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
    model_output_path = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')

    missing_env = [
        name for name, value in [
            ('MODEL_OUTPUT_PATH', model_output_path),
            ('EVAL_DATA_PATH', data_dir),
            ('EVAL_RESULT_PATH', result_dir),
        ]
        if not value
    ]
    if missing_env:
        raise ValueError(f"Missing required environment variables: {missing_env}")

    os.makedirs(result_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_dir, ckpt_path = resolve_model_dir_and_ckpt_path(model_output_path)
    logging.info(f"Resolved MODEL_OUTPUT_PATH={model_output_path}")
    logging.info(f"Resolved checkpoint dir: {model_dir}")
    logging.info(f"Resolved checkpoint file: {ckpt_path}")

    model_output_dir = (
        model_output_path if os.path.isdir(model_output_path)
        else os.path.dirname(model_output_path)
    )

    # ---- Schema: prefer checkpoint sidecars (to exactly match training);
    #      fall back to the eval data schema if missing. ----
    schema_candidates = [
        os.path.join(model_dir, 'schema.json'),
        os.path.join(model_output_dir, 'schema.json'),
        os.path.join(data_dir, 'schema.json'),
    ]
    schema_path = next((p for p in schema_candidates if os.path.exists(p)), None)
    if schema_path is None:
        raise FileNotFoundError(
            f"schema.json not found. Checked: {schema_candidates}")
    logging.info(f"Using schema: {schema_path}")

    # ---- Load train_config.json (single source of truth for all hyperparams) ----
    train_config = load_train_config(model_dir)
    if not train_config and model_output_dir != model_dir:
        train_config = load_train_config(model_output_dir)

    # ---- Parse seq_max_lens ----
    sml_str = train_config.get('seq_max_lens', _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")

    # ---- Data loading: reuse training workers unless explicitly overridden.
    batch_size = int(train_config.get('batch_size', _FALLBACK_BATCH_SIZE))
    num_workers = int(os.environ.get(
        'EVAL_NUM_WORKERS',
        train_config.get('num_workers', _FALLBACK_NUM_WORKERS),
    ))

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
    )
    total_test_samples = test_dataset.num_rows
    logging.info(f"Total test samples: {total_test_samples}")

    # ---- Build model: every structural hyperparameter is resolved from train_config ----
    model_cfg = resolve_model_cfg(train_config)

    # ns_groups_json also comes from training config (e.g. run.sh may have
    # passed an empty string to disable it). When trainer.py has copied the
    # JSON into the ckpt dir, train_config records just the basename, so try
    # resolving against ``model_dir`` first before honoring the raw (possibly
    # absolute) path as a fallback.
    ns_groups_json = train_config.get('ns_groups_json', None)
    if ns_groups_json:
        for local_candidate in [
            os.path.join(model_dir, os.path.basename(ns_groups_json)),
            os.path.join(model_output_dir, os.path.basename(ns_groups_json)),
        ]:
            if os.path.exists(local_candidate):
                ns_groups_json = local_candidate
                break

    model = build_model(
        test_dataset,
        model_cfg=model_cfg,
        ns_groups_json=ns_groups_json,
        device=device,
    )

    # ---- Strictly load weights ----
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    loader_kwargs: Dict[str, Any] = {
        'batch_size': None,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = 2
    test_loader = DataLoader(test_dataset, **loader_kwargs)

    all_probs = []
    all_user_ids = []
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

            if (batch_idx + 1) % 100 == 0:
                logging.info(f"  Processed {(batch_idx + 1) * batch_size} samples")

    logging.info(f"Inference complete: {len(all_probs)} predictions")
    logging.info(
        f"Prediction accounting: total_test_samples={total_test_samples}, "
        f"len(all_probs)={len(all_probs)}, len(all_user_ids)={len(all_user_ids)}")
    if len(all_probs) != total_test_samples:
        logging.warning(
            f"Prediction count mismatch: expected {total_test_samples}, "
            f"got {len(all_probs)}")

    predictions = {
        "predictions": dict(zip(all_user_ids, all_probs)),
    }
    if len(predictions["predictions"]) != len(all_probs):
        logging.warning(
            "Duplicate user_id values detected; dict output has fewer entries "
            f"({len(predictions['predictions'])}) than predictions ({len(all_probs)}). "
            "Keeping baseline output format.")

    # ---- Save predictions.json ----
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()
