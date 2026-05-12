#!/bin/bash
# 11thMay-PlanB platform entrypoint.
# Pipeline: original parquet → FE-00 preprocess → 11thmay_a feature augmentation → train.
#
# Differences vs the baseline run.sh:
#   1. Chains FE-00 (drop high-missing user_int + fill + dense z-score) into pre-train.
#   2. Chains 11thmay_a (FE-01A frequency + 12 temporal buckets) onto FE-00 output.
#   3. Switches --rank_mixer_mode to ffn_only (P0 implemented +0.236% AUC).
#
# Required:
#   - dataset.py            (patched, item_dense path enabled)
#   - train.py / model.py   (unchanged from baseline)
#   - trainer.py / utils.py (unchanged from baseline)
#   - ns_groups.json        (unchanged; FE-00 reads it to sync drop NS groups)
#   - build_fe00_preprocess_dataset.py        (new)
#   - build_feature_engineering_dataset.py    (new)
#   - run.sh                (this file, replaces baseline run.sh)
#
# Platform invocation (example):
#   bash run.sh --data_dir $TRAIN_DATA_PATH --ckpt_dir $CKPT --log_dir $LOG

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# ---- Parse --data_dir / --schema_path from $@ ----
ORIG_DATA_DIR="${TRAIN_DATA_PATH:-}"
ORIG_SCHEMA_PATH=""
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    case "${ARGS[$i]}" in
        --data_dir)        ORIG_DATA_DIR="${ARGS[$((i + 1))]}";;
        --data_dir=*)      ORIG_DATA_DIR="${ARGS[$i]#--data_dir=}";;
        --schema_path)     ORIG_SCHEMA_PATH="${ARGS[$((i + 1))]}";;
        --schema_path=*)   ORIG_SCHEMA_PATH="${ARGS[$i]#--schema_path=}";;
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

# ---- Intermediate / final dataset paths (outside code dir to avoid storage limit) ----
PIPELINE_BASE="${PIPELINE_DATA_DIR:-${TMPDIR:-/tmp}/taac_11thmay_b_$$}"
FE00_ROOT="${PIPELINE_BASE}/fe00"
PLANB_ROOT="${PIPELINE_BASE}/11thmay_b"

echo "============================================"
echo "11thMay-PlanB Pipeline"
echo "============================================"
echo "[stage A0 FE-00] input data:   ${ORIG_DATA_DIR}"
echo "[stage A0 FE-00] input schema: ${ORIG_SCHEMA_PATH}"
echo "[stage A0 FE-00] output:       ${FE00_ROOT}"
echo "[stage A1 PlanB] output:       ${PLANB_ROOT}"
echo "============================================"

# ---- Stage A0: FE-00 preprocess ----
echo
echo "[stage A0] Running FE-00 preprocess (drop user_int missing>0.75 + fill + z-score)..."
python3 -u "${SCRIPT_DIR}/build_fe00_preprocess_dataset.py" \
    --input_dir       "${ORIG_DATA_DIR}" \
    --input_schema    "${ORIG_SCHEMA_PATH}" \
    --output_dir      "${FE00_ROOT}" \
    --ns_groups_json  "${SCRIPT_DIR}/ns_groups.json" \
    --missing_threshold 0.75 \
    --fit_stats_row_group_ratio 0.9

# ---- Stage A1: 11thmay_a feature augmentation ----
echo
echo "[stage A1] Running 11thmay_a feature augmentation (FE-01A frequency + 12 temporal buckets)..."
python3 -u "${SCRIPT_DIR}/build_feature_engineering_dataset.py" \
    --input_dir       "${FE00_ROOT}" \
    --input_schema    "${FE00_ROOT}/schema.json" \
    --output_dir      "${PLANB_ROOT}" \
    --feature_set     11thmay_a \
    --fit_stats_row_group_ratio 0.9 \
    --temporal_max_len 1024

# ---- Stage B: training ----
echo
echo "[stage B] Launching training (rankmixer + ffn_only)..."

# Strip user-supplied --data_dir / --schema_path from $@ so we can substitute ours.
TRAIN_ARGS=()
SKIP=0
for arg in "$@"; do
    if (( SKIP > 0 )); then SKIP=$((SKIP-1)); continue; fi
    case "$arg" in
        --data_dir|--schema_path) SKIP=1;;
        --data_dir=*|--schema_path=*) ;;
        *) TRAIN_ARGS+=("$arg");;
    esac
done

TRAIN_DATA_PATH="${PLANB_ROOT}" python3 -u "${SCRIPT_DIR}/train.py" \
    "${TRAIN_ARGS[@]}" \
    --data_dir       "${PLANB_ROOT}" \
    --schema_path    "${PLANB_ROOT}/schema.json" \
    --ns_groups_json "" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --rank_mixer_mode ffn_only \
    --emb_skip_threshold 1000000 \
    --num_workers 8
