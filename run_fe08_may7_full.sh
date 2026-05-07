#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# FE-08 May7 main entrypoint:
# 0.8159 GNN baseline + May7 locked feature set + sequence recency sort.

ORIG_DATA_DIR="${TRAIN_DATA_PATH:-}"
ORIG_SCHEMA_PATH=""

ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    case "${ARGS[$i]}" in
        --domain_time_buckets|--domain_bucket_path|--domain_bucket_path=*|--use_rope|--seq_causal)
            echo "FE-08 May7 mainline forbids ${ARGS[$i]}; keep domain_time_buckets/rope/seq_causal disabled." >&2
            exit 1
            ;;
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

FE08_ROOT="${FE08_DATA_DIR:-${TMPDIR:-/tmp}/taac_fe08_may7_$$}"
FE08_SCHEMA="${FE08_ROOT}/schema.json"
FE08_GROUPS="${FE08_ROOT}/ns_groups.may7.json"

echo "[FE-08] input data: ${ORIG_DATA_DIR}"
echo "[FE-08] input schema: ${ORIG_SCHEMA_PATH}"
echo "[FE-08] output data: ${FE08_ROOT}"

python3 -u "${SCRIPT_DIR}/build_fe08_may7_dataset.py" \
    --input_dir "${ORIG_DATA_DIR}" \
    --input_schema "${ORIG_SCHEMA_PATH}" \
    --output_dir "${FE08_ROOT}" \
    --missing_threshold 0.80 \
    --match_window_days 7 \
    --match_count_buckets "0,1,2,4,8" \
    --fit_stats_row_group_ratio 0.9 \
    --diagnostic_row_group_limit 100 \
    --item_dense_fids "86,91,92" \
    --sort_sequence_by_recency

TRAIN_DATA_PATH="${FE08_ROOT}" python3 -u "${SCRIPT_DIR}/train.py" \
    "$@" \
    --schema_path "${FE08_SCHEMA}" \
    --ns_groups_json "${FE08_GROUPS}" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --rank_mixer_mode full \
    --d_model 136 \
    --emb_dim 64 \
    --num_hyformer_blocks 2 \
    --num_heads 4 \
    --seq_encoder_type transformer \
    --seq_max_lens seq_a:256,seq_b:256,seq_c:128,seq_d:512 \
    --use_time_buckets \
    --loss_type bce \
    --lr 1e-4 \
    --sparse_lr 0.05 \
    --dropout_rate 0.05 \
    --seq_top_k 100 \
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
