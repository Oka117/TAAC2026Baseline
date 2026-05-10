#!/bin/bash
# P4 (S5) — Capacity expansion + Plan A (qgen_cond mode by default).
#
# Long-term path: d_model 64 → 128 widens the backbone so that NS-pool
# pressure relaxes and Plan A can stack with future Plan B/C/G dense token
# families without further compression. This is the same direction as
# README.research_directions.zh.md §2 (model capacity).
#
# Geometry (d_model=128, T must divide 128 → T ∈ {1,2,4,8,16,32}):
#   num_queries=2, num_sequences=4, num_user_ns=5, num_item_ns=2,
#   has_user_dense=1, has_item_dense=0 → num_ns = 8, T = 16, 128 % 16 = 0 ✓
#
# We pair this with --match_inject_mode qgen_cond by default so that:
#   1) user_ns_tokens stays at 5 (no chunk-compression cost);
#   2) any future +1 NS token (e.g. Plan B/C/G) can still fit while keeping
#      T ∈ {16, 32}.
#
# Cost: ~4× dense parameters (FFN, projection layers); embedding tables also
# ~2× (emb_dim 64→128). Plan to run on GPU; do NOT try this on CPU.
#
# Usage:
#   bash experiment_plans/FE_A/run_p4_capacity.sh --data_dir /path/to/dataset

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
    --d_model 128 \
    --emb_dim 128 \
    --num_heads 8 \
    --hidden_mult 4 \
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
