#!/bin/bash
# 11thMay-PlanB full data preprocessing pipeline.
#
# Chain: original → FE-00 (drop missing>threshold + fill + z-score)
#                 → 11thmay_a (FE-01A frequency + 12 temporal buckets)
#
# Usage:
#   bash build_11thmay_b_pipeline.sh \
#       --input_dir   /path/to/original_dataset \
#       --input_schema /path/to/original_dataset/schema.json \
#       --output_dir  /path/to/11thmay_b_dataset
#
# Optional:
#   --missing_threshold       FE-00 drop threshold for user_int features (default 0.75)
#   --temporal_max_len        seq scan length for 12 time-bucket counters (default 1024)
#   --fit_stats_row_group_ratio   default 0.9 (matches train.py valid_ratio=0.1)
#   --intermediate_dir        explicit path for the FE-00 intermediate dataset
#                            (default: ${output_dir}_fe00)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_DIR=""
INPUT_SCHEMA=""
OUTPUT_DIR=""
INTERMEDIATE_DIR=""
MISSING_THRESHOLD=0.75
TEMPORAL_MAX_LEN=1024
FIT_RATIO=0.9
NS_GROUPS_JSON="${SCRIPT_DIR}/ns_groups.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_dir)        INPUT_DIR="$2";        shift 2;;
        --input_schema)     INPUT_SCHEMA="$2";     shift 2;;
        --output_dir)       OUTPUT_DIR="$2";       shift 2;;
        --intermediate_dir) INTERMEDIATE_DIR="$2"; shift 2;;
        --missing_threshold) MISSING_THRESHOLD="$2"; shift 2;;
        --temporal_max_len) TEMPORAL_MAX_LEN="$2"; shift 2;;
        --fit_stats_row_group_ratio) FIT_RATIO="$2"; shift 2;;
        --ns_groups_json)   NS_GROUPS_JSON="$2";   shift 2;;
        *) echo "Unknown flag: $1"; exit 1;;
    esac
done

if [[ -z "$INPUT_DIR" || -z "$INPUT_SCHEMA" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 --input_dir <dir> --input_schema <schema.json> --output_dir <dir>"
    exit 1
fi
if [[ -z "$INTERMEDIATE_DIR" ]]; then
    INTERMEDIATE_DIR="${OUTPUT_DIR%/}_fe00"
fi

echo "=========================================="
echo "11thMay-PlanB data pipeline"
echo "=========================================="
echo "Stage A0 (FE-00 preprocess):"
echo "  input:        $INPUT_DIR"
echo "  intermediate: $INTERMEDIATE_DIR"
echo "Stage A1 (11thmay_a augment):"
echo "  output:       $OUTPUT_DIR"
echo "Parameters:"
echo "  missing_threshold       = $MISSING_THRESHOLD"
echo "  temporal_max_len        = $TEMPORAL_MAX_LEN"
echo "  fit_stats_row_group_ratio = $FIT_RATIO"
echo "=========================================="

# ---- Stage A0: FE-00 preprocessing ----
echo
echo "[Stage A0] Running FE-00 preprocess..."
python3 -u "${SCRIPT_DIR}/build_fe00_preprocess_dataset.py" \
    --input_dir       "$INPUT_DIR" \
    --input_schema    "$INPUT_SCHEMA" \
    --output_dir      "$INTERMEDIATE_DIR" \
    --ns_groups_json  "$NS_GROUPS_JSON" \
    --missing_threshold "$MISSING_THRESHOLD" \
    --fit_stats_row_group_ratio "$FIT_RATIO"

# ---- Stage A1: 11thmay_a feature augmentation ----
echo
echo "[Stage A1] Running 11thmay_a feature augmentation..."
python3 -u "${SCRIPT_DIR}/build_feature_engineering_dataset.py" \
    --input_dir       "$INTERMEDIATE_DIR" \
    --input_schema    "$INTERMEDIATE_DIR/schema.json" \
    --output_dir      "$OUTPUT_DIR" \
    --feature_set     11thmay_a \
    --fit_stats_row_group_ratio "$FIT_RATIO" \
    --temporal_max_len "$TEMPORAL_MAX_LEN"

echo
echo "=========================================="
echo "Pipeline finished. Final dataset: $OUTPUT_DIR"
echo "Now launch training:"
echo "  bash run_11thmay_b.sh \\"
echo "       --data_dir   $OUTPUT_DIR \\"
echo "       --schema_path $OUTPUT_DIR/schema.json \\"
echo "       --ckpt_dir   outputs/exp_11thmay_b/ckpt \\"
echo "       --log_dir    outputs/exp_11thmay_b/log"
echo "=========================================="
