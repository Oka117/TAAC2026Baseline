#!/usr/bin/env bash
set -euo pipefail

# Smoke test using the official HuggingFace 1k-row sample.
# This is intended for local debugging only.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

DATA_DIR="${1:-${SCRIPT_DIR}/data_sample_1000}"

python3 -u "${SCRIPT_DIR}/tools/prepare_hf_sample.py" --out_dir "${DATA_DIR}"

bash "${SCRIPT_DIR}/run.sh" \
  --data_dir "${DATA_DIR}" \
  --schema_path "${DATA_DIR}/schema.json" \
  --num_workers 0 \
  --batch_size 64 \
  --num_epochs 1 \
  --eval_every_n_steps 50
