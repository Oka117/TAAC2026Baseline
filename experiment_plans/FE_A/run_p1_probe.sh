#!/bin/bash
# P1 — Probe ablation: user_ns_tokens 5 → 4 with NO match feats.
#
# Purpose: isolate the cost of compressing user-NS chunks from 5 to 4 so we can
# attribute Plan A's net AUC gain correctly.
#
# Why ffn_only? 4 + 1 + 2 + 0 = 7 NS tokens → T = 8 + 7 = 15, and 64 % 15 ≠ 0,
# which would crash full-mode rank-mixer at model construction. Running BOTH
# this probe and its calibration baseline (run_p0_calibration.sh) in
# ffn_only mode keeps the user_ns 5→4 effect cleanly isolated from the full-vs-
# ffn_only mode-switch cost. We then sum the two costs to extrapolate to the
# Plan A (P2) full-mode setting.
#
# Direct comparison flow:
#   AUC(baseline full,  user_ns=5)  = 0.810  (given)
#   AUC(P0_calib  ffn_only, user_ns=5) = call A
#   AUC(P1_probe  ffn_only, user_ns=4) = call B
#   ΔAUC of full→ffn_only mode  = 0.810 - A
#   ΔAUC of user_ns 5→4 (in ffn_only) = A - B
#   Predicted P2 AUC ≈ 0.810 - (A - B) + match_gain
#
# Pass criterion (per FE_A/README.zh.md §8.5):
#   |A - B| ≤ 0.10 % AUC  → continue with P2 (Plan A) using user_ns=4
#   |A - B| >  0.10 %     → escalate to P3 (match into MultiSeqQueryGenerator)
#
# Usage:
#   bash experiment_plans/FE_A/run_p1_probe.sh --data_dir /path/to/dataset

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
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --rank_mixer_mode ffn_only \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"
