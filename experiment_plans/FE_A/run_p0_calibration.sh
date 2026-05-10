#!/bin/bash
# P0 calibration — baseline NS layout (user_ns_tokens=5) but in ffn_only mode,
# so it can be paired with P1 to isolate the user_ns 5→4 effect from the
# full-vs-ffn_only mode-switch effect.
#
# This is NOT the published baseline (which is full mode). Use this only to
# anchor the comparison vector for run_p1_probe.sh.
#
# Usage:
#   bash experiment_plans/FE_A/run_p0_calibration.sh --data_dir /path/to/dataset

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${SCRIPT_DIR}/train.py" ]]; then
    # Taiji flattens the submitted script archive into one runtime/script dir.
    PROJ_DIR="${SCRIPT_DIR}"
else
    PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH}"

python3 -u "${PROJ_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --rank_mixer_mode ffn_only \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"
