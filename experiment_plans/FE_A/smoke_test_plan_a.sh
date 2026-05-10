#!/usr/bin/env bash
# Plan A smoke test on the official 1k-row HuggingFace sample.
# Verifies the patched dataset / model / trainer can:
#   1. parse --match_pairs_json,
#   2. compute match_feats inside _convert_batch,
#   3. add the match NS token without breaking d_model % T == 0,
#   4. run forward + backward + evaluate without crashing.
#
# Notes:
#   - The smoke schema sets ts_fid=None, so windowed match counts and
#     min_match_delta degrade to 0 (has_match / match_count are still active).
#   - num_queries=1 is used here (instead of 2) so T = 4 + 8 = 12,
#     and 64 % 12 != 0 — we run rank_mixer_mode=ffn_only to bypass the
#     divisibility constraint without changing user_ns_tokens.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH}"

DATA_DIR="${1:-${PROJ_DIR}/data_sample_1000}"

python3 -u "${PROJ_DIR}/tools/prepare_hf_sample.py" --out_dir "${DATA_DIR}"

python3 -u "${PROJ_DIR}/train.py" \
    --data_dir "${DATA_DIR}" \
    --schema_path "${DATA_DIR}/schema.json" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 0 \
    --batch_size 64 \
    --num_epochs 1 \
    --eval_every_n_steps 5 \
    --match_pairs_json "${SCRIPT_DIR}/match_pairs.default.json"
