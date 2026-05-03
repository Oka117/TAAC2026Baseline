#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_groups_json "${SCRIPT_DIR}/ns_groups.feature_engineering.json" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 6 \
    --item_ns_tokens 4 \
    --num_queries 1 \
    --rank_mixer_mode full \
    --d_model 64 \
    --emb_dim 64 \
    --num_hyformer_blocks 2 \
    --num_heads 4 \
    --seq_encoder_type transformer \
    --seq_max_lens seq_a:256,seq_b:256,seq_c:512,seq_d:512 \
    --use_time_buckets \
    --loss_type bce \
    --lr 1e-4 \
    --sparse_lr 0.05 \
    --dropout_rate 0.01 \
    --batch_size 256 \
    --num_workers 8 \
    --buffer_batches 20 \
    --valid_ratio 0.1 \
    --train_ratio 1.0 \
    --patience 5 \
    --emb_skip_threshold 1000000 \
    --seq_id_threshold 10000 \
    "$@"
