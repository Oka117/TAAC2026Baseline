#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# FE-00 platform entrypoint.
#
# Path policy:
# - Original dataset is read from TRAIN_DATA_PATH or --data_dir.
# - Preprocessed parquet/schema are written outside the Training Code folder
#   by default, avoiding upload-code storage limits and read-only code paths.
# - train.py is then pointed at the generated FE-00 dataset.

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

FE00_ROOT="${FE00_DATA_DIR:-${TMPDIR:-/tmp}/taac_fe00_$$}"
FE00_SCHEMA="${FE00_ROOT}/schema.json"
FE00_GROUPS="${FE00_ROOT}/ns_groups.fe00.json"

echo "[FE-00] input data: ${ORIG_DATA_DIR}"
echo "[FE-00] input schema: ${ORIG_SCHEMA_PATH}"
echo "[FE-00] output data: ${FE00_ROOT}"

python3 -u "${SCRIPT_DIR}/build_fe00_preprocess_dataset.py" \
    --input_dir "${ORIG_DATA_DIR}" \
    --input_schema "${ORIG_SCHEMA_PATH}" \
    --output_dir "${FE00_ROOT}" \
    --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
    --missing_threshold 0.75 \
    --fit_stats_row_group_ratio 0.9

TRAIN_DATA_PATH="${FE00_ROOT}" python3 -u "${SCRIPT_DIR}/train.py" \
    "$@" \
    --schema_path "${FE00_SCHEMA}" \
    --ns_groups_json "${FE00_GROUPS}" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
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
    --dropout_rate 0.01 \
    --batch_size 256 \
    --num_workers 8 \
    --buffer_batches 20 \
    --valid_ratio 0.1 \
    --train_ratio 1.0 \
    --patience 3 \
    --num_epochs 6 \
    --emb_skip_threshold 1000000 \
    --seq_id_threshold 10000
