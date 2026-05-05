#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# FE-06 full entrypoint:
# FE00 dense cleanup + P0-L1 vocab shift + FE01AB-safe + Claude P0 dense
# blocks + 4-layer TokenGNN (scale=0.15).

ORIG_DATA_DIR="${TRAIN_DATA_PATH:-}"
ORIG_SCHEMA_PATH=""

ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    case "${ARGS[$i]}" in
        --data_dir)
            if (( i + 1 < ${#ARGS[@]} )); then
                ORIG_DATA_DIR="${ARGS[$((i + 1))]}"
            fi
            ;;
        --data_dir=*)
            ORIG_DATA_DIR="${ARGS[$i]#--data_dir=}"
            ;;
        --schema_path)
            if (( i + 1 < ${#ARGS[@]} )); then
                ORIG_SCHEMA_PATH="${ARGS[$((i + 1))]}"
            fi
            ;;
        --schema_path=*)
            ORIG_SCHEMA_PATH="${ARGS[$i]#--schema_path=}"
            ;;
    esac
done

if [[ -z "${ORIG_DATA_DIR}" ]]; then
    echo "Missing data path. Set TRAIN_DATA_PATH or pass --data_dir /path/to/dataset." >&2
    exit 1
fi

if [[ -z "${ORIG_SCHEMA_PATH}" ]]; then
    ORIG_SCHEMA_PATH="${ORIG_DATA_DIR}/schema.json"
fi

if [[ ! -d "${ORIG_DATA_DIR}" ]]; then
    echo "Data directory not found: ${ORIG_DATA_DIR}" >&2
    exit 1
fi

if [[ ! -f "${ORIG_SCHEMA_PATH}" ]]; then
    echo "Schema file not found: ${ORIG_SCHEMA_PATH}" >&2
    exit 1
fi

FE06_ROOT="${FE06_DATA_DIR:-${TMPDIR:-/tmp}/taac_fe06_p0ab_gnn4_$$}"
FE06_SCHEMA="${FE06_ROOT}/schema.json"
FE06_GROUPS="${FE06_ROOT}/ns_groups.feature_engineering.json"

echo "[FE-06] input data: ${ORIG_DATA_DIR}"
echo "[FE-06] input schema: ${ORIG_SCHEMA_PATH}"
echo "[FE-06] output data: ${FE06_ROOT}"

python3 -u "${SCRIPT_DIR}/build_fe06_p0ab_dataset.py" \
    --input_dir "${ORIG_DATA_DIR}" \
    --input_schema "${ORIG_SCHEMA_PATH}" \
    --output_dir "${FE06_ROOT}" \
    --missing_threshold 0.75 \
    --match_window_days 7 \
    --match_count_buckets 0,1,2,4,8 \
    --fit_stats_row_group_ratio 0.9

TRAIN_DATA_PATH="${FE06_ROOT}" python3 -u "${SCRIPT_DIR}/train.py" \
    "$@" \
    --schema_path "${FE06_SCHEMA}" \
    --ns_groups_json "${FE06_GROUPS}" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 6 \
    --item_ns_tokens 4 \
    --num_queries 1 \
    --rank_mixer_mode full \
    --d_model 64 \
    --emb_dim 64 \
    --num_hyformer_blocks 2 \
    --num_heads 4 \
    --seq_encoder_type transformer \
    --seq_max_lens seq_a:256,seq_b:256,seq_c:512,seq_d:512 \
    --use_time_buckets \
    --loss_type bce \
    --lr 1e-4 \
    --sparse_lr 0.05 \
    --dropout_rate 0.015 \
    --batch_size 256 \
    --num_workers 8 \
    --buffer_batches 20 \
    --valid_ratio 0.1 \
    --split_by_timestamp \
    --train_ratio 1.0 \
    --patience 3 \
    --num_epochs 6 \
    --emb_skip_threshold 1000000 \
    --seq_id_threshold 10000 \
    --use_token_gnn \
    --token_gnn_layers 4 \
    --token_gnn_graph full \
    --token_gnn_layer_scale 0.15
