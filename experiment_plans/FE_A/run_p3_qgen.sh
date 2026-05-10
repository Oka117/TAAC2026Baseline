#!/bin/bash
# P3 (S4) — Match feats injected into MultiSeqQueryGenerator as a condition,
# instead of consuming an NS-token slot.
#
# Why this exists:
#   FE_A (P2) had to drop --user_ns_tokens 5→4 to make room for the match
#   token while keeping T=16. P3 keeps user_ns_tokens=5 (no NS compression)
#   AND feeds match feats into the Q generator's global_info concat. Match
#   signal ends up in every Q token directly, without any NS-pool trade-off.
#
# Key flags:
#   --match_inject_mode qgen_cond    activate the new code path
#   --user_ns_tokens 5               restore baseline NS layout
#   --rank_mixer_mode full           T = 8 + (5+1+2+0) = 16, 64%16=0 ✓
#                                    (no T regression vs published baseline)
#
# Usage:
#   bash experiment_plans/FE_A/run_p3_qgen.sh --data_dir /path/to/dataset

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
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --rank_mixer_mode full \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --match_pairs_json "${MATCH_JSON}" \
    --match_inject_mode qgen_cond \
    "$@"
