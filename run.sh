#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: RTG-HyFormer high-AUC graph plan ----
# Compared with the old 4-layer NS-only GNN, this run keeps a smaller NS GNN,
# adds target-aware temporal sequence graphs, directly exposes final NS tokens
# to the head, and adds element-level graph memory for aligned dense-int lists.
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --seq_encoder_type longer \
    --seq_top_k 96 \
    --use_rope \
    --use_token_gnn \
    --token_gnn_layers 2 \
    --token_gnn_layer_scale 0.05 \
    --use_seq_graph \
    --seq_graph_layers 2 \
    --seq_graph_layer_scale 0.08 \
    --graph_output_fusion \
    --output_include_ns \
    --use_aligned_dense_int_graph \
    --aligned_graph_fids 62,63,64,65,66,89,90,91 \
    --aligned_graph_layers 1 \
    --aligned_graph_tokens 8 \
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
