#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: RankMixer NS tokenizer + low-risk GNN-NS + generalization ----
# GNN-NS places a 4-layer fully-connected TokenGNN after NS token construction.
# Token augmentation and EMA are enabled as conservative generalization
# regularizers: modestly slower training, with validation logloss expected to
# become more stable and AUC expected to improve slightly or stay close.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --use_token_gnn \
    --token_gnn_layers 4 \
    --token_gnn_graph full \
    --token_gnn_layer_scale 0.1 \
    --use_generalization_aug \
    --ns_token_dropout_rate 0.05 \
    --ns_token_noise_std 0.01 \
    --seq_token_dropout_rate 0.03 \
    --seq_token_noise_std 0.005 \
    --weight_averaging ema \
    --ema_decay 0.995 \
    --weight_avg_start_step 0 \
    --weight_avg_update_every 1 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12 (7 user_int + 1 user_dense + 4 item_int),
# only num_queries=1 satisfies d_model % T == 0 (T = num_queries*4 + num_ns).
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
