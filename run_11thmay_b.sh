#!/bin/bash
# 11thMay-PlanB: rankmixer + ffn_only + FE-01A frequency + 12 temporal buckets.
#
# Geometry MATCHES P0 exactly (user_ns=5, item_ns=2, num_q=2, T=16) so the
# +0.2364 % AUC gain from --rank_mixer_mode ffn_only (P0 vs A0) is the same
# configuration that's already been measured on the leaderboard.
#
# Differences vs run.sh (baseline):
#   --rank_mixer_mode ffn_only      ← P0 finding (+0.236 % AUC)
#
# Differences vs FE-01A (worktree group path):
#   --ns_tokenizer_type rankmixer   ← P0/P1/run.sh geometry
#   --user_ns_tokens 5              ← P0/P1 same, NOT FE-01A's 6
#   --item_ns_tokens 2              ← P0/P1 same, NOT FE-01A's 4
#   --num_queries 2                 ← P0/P1 same, NOT FE-01A's 1
#   --ns_groups_json ""             ← rankmixer mode skips group JSON
#
# Data assumption:
#   --data_dir points at the output of build_feature_engineering_dataset.py
#                with --feature_set 11thmay_a (schema contains the 14 added
#                dense fids: user 110/113..124, item 86).
#
# Usage:
#   bash run_11thmay_b.sh --data_dir /path/to/11thmay_a_dataset \
#                         --schema_path /path/to/11thmay_a_dataset/schema.json

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --rank_mixer_mode ffn_only \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"
