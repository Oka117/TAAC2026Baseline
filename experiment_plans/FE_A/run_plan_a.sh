#!/bin/bash
# Plan A — Target-Item × History Matching feature family.
#
# Difference from baseline run.sh:
#   --user_ns_tokens 4   (was 5)  → leave one slot for the match-feats token,
#                                   keeping T = num_q*4 + num_ns = 8 + 8 = 16
#                                   so that 64 % T == 0 in rank_mixer 'full' mode.
#   --match_pairs_json   path to JSON listing target-item × seq matching pairs.
#
# Usage:
#   bash experiment_plans/FE_A/run_plan_a.sh --data_dir /path/to/dataset
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${SCRIPT_DIR}/train.py" ]]; then
    # Taiji flattens the submitted script archive into one runtime/script dir.
    PROJ_DIR="${SCRIPT_DIR}"
else
    PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH}"

MATCH_JSON="${MATCH_JSON:-${SCRIPT_DIR}/match_pairs.default.json}"

python3 -u "${PROJ_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --match_pairs_json "${MATCH_JSON}" \
    "$@"
