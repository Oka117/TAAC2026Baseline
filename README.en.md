# TAAC2026Baseline Source Code Analysis

This document analyzes the official baseline code, including the data pipeline, model architecture, training process, strengths, limitations, and improvement directions. The English version also keeps an optional GNN-to-Transformer research note.

## 1. Repository Structure

```text
TAAC2026Baseline/
├── README.md
├── README.zh.md
├── README.en.md
├── train.py
├── trainer.py
├── dataset.py
├── model.py
├── utils.py
├── run.sh
├── ns_groups.json
└── demo_1000.parquet
```

| File | Role |
| --- | --- |
| `README.md` | Main competition introduction, dataset, and task description. |
| `README.zh.md` | Chinese source-code analysis. |
| `README.en.md` | English source-code analysis. |
| `train.py` | Training entry point: argument parsing, data loading, model construction, and trainer construction. |
| `trainer.py` | Training loop, validation, AUC computation, checkpoints, and early stopping. |
| `dataset.py` | Parquet loading, schema parsing, padding, and time-bucket construction. |
| `model.py` | Main PCVRHyFormer model. |
| `utils.py` | Logging, random seed, early stopping, focal loss, and helper utilities. |
| `run.sh` | Official default launch script. |
| `ns_groups.json` | Example non-sequential feature grouping file. |

## 2. Overall Assessment

This baseline is not a simple DNN. It is a hybrid token model designed around the competition theme. It already includes non-sequential tokenization, multi-domain sequence tokenization, query-based sequence reading, stackable blocks, RankMixer fusion, time buckets, and an AUC-driven training loop.

However, it is not a fully unified architecture. Sequence and non-sequence features still have dedicated branches: four sequence domains are encoded separately, query tokens read from their corresponding sequences, and non-sequential tokens are mainly fused through RankMixer. It is closer to a multi-sequence-tower plus non-sequence-tower architecture than a single shared token stream from the input stage.

## 3. Data Pipeline

The data pipeline is implemented in `dataset.py`, mainly through `PCVRParquetDataset`.

### 3.1 Schema-Driven Parsing

The code uses `schema.json` to describe feature layouts instead of hard-coding all columns. `FeatureSchema` records the offset and length of each feature id inside a flattened tensor.

Supported groups:

- `user_int`: user sparse features, including scalar integers and list integers.
- `item_int`: item/ad sparse features.
- `user_dense`: user dense/list-float features.
- `item_dense`: interface reserved by the code, but empty for the current data.
- `seq`: four behavior-domain sequence configurations.

### 3.2 Non-Sequential Feature Processing

Inside `_convert_batch`, user and item integer features are written into pre-allocated numpy buffers:

- scalar integers occupy one position;
- list integers are padded or truncated to the schema-defined length;
- values less than or equal to zero are treated as padding and mapped to 0;
- out-of-vocabulary ids are clipped to 0 by default and tracked in OOB statistics.

User dense features are padded into a fixed-length float tensor. One important limitation is that some `user_int_feats_x` and `user_dense_feats_x` columns are element-wise aligned according to the official data description, but the baseline does not explicitly preserve this relation. Integer lists and dense lists are processed through separate paths.

### 3.3 Sequence Feature Processing

Each sequence domain is converted into:

```text
[B, num_side_features, max_len]
```

where `B` is the batch size, `num_side_features` is the number of side-information fields excluding timestamp, and `max_len` is controlled by `--seq_max_lens`. The default setting is:

```text
seq_a:256,seq_b:256,seq_c:512,seq_d:512
```

Sequence lengths are stored in `{domain}_len`, which is later used by the model to build padding masks.

### 3.4 Time Buckets

The dataset computes the time difference between the sample timestamp and each historical event timestamp, then discretizes the difference using `BUCKET_BOUNDARIES`.

Time-bucket behavior:

- padding positions use bucket id 0;
- valid buckets start from 1;
- differences beyond the largest boundary are clipped to the last bucket;
- the model adds `nn.Embedding(num_time_buckets, d_model, padding_idx=0)` to sequence tokens.

Recency is important in advertising sequence modeling, so this is a useful built-in signal.

### 3.5 Train/Validation Split

`get_pcvr_data` splits data by Parquet Row Group:

- training uses earlier row groups;
- validation uses the tail `valid_ratio` fraction;
- default `valid_ratio` is 0.1.

This is efficient, but whether it behaves like temporal validation depends on the ordering of parquet files and row groups.

## 4. Model Architecture

The main model is `PCVRHyFormer` in `model.py`.

High-level flow:

```text
user/item sparse features -> NS tokens
user dense features       -> dense NS token
domain sequences          -> sequence tokens
NS tokens + sequence summaries -> query tokens
query tokens + sequence tokens + NS tokens -> MultiSeqHyFormerBlock stack
final query tokens -> CVR prediction head
```

Main components:

- non-sequential tokenizer;
- sequence tokenizer;
- query generator;
- stacked `MultiSeqHyFormerBlock`;
- CVR prediction head.

## 5. Non-Sequential Tokenizers

The code supports two non-sequential tokenizers.

### 5.1 GroupNSTokenizer

`GroupNSTokenizer` uses manual groups from `ns_groups.json`. Each group of feature ids becomes one NS token.

Steps:

1. Each sparse feature has an independent embedding table.
2. Multi-value features are mean-pooled.
3. Embeddings inside one group are concatenated.
4. A Linear + LayerNorm projection maps the group vector to `d_model`.
5. Each group outputs one NS token.

The advantage is semantic clarity. The downside is the need for manual grouping and a token count tied to the number of groups.

### 5.2 RankMixerNSTokenizer

`RankMixerNSTokenizer` is the default tokenizer used by `run.sh`. It concatenates all feature embeddings into a long vector, splits it into a specified number of chunks, and projects each chunk into one token.

Default launch configuration:

```bash
--ns_tokenizer_type rankmixer
--user_ns_tokens 5
--item_ns_tokens 2
--num_queries 2
--ns_groups_json ""
```

Because `--ns_groups_json ""` is passed, the provided `ns_groups.json` is not used by default. The code falls back to singleton feature groups and then chunks the concatenated embeddings.

This makes token count controllable and helps satisfy the `d_model % T == 0` constraint, but semantic grouping is weaker and multi-value features are still mean-pooled.

## 6. Sequence Tokenizer

Each sequence-domain input has shape:

```text
[B, S, L]
```

where `S` is the number of side-information fields and `L` is the sequence length.

`_embed_seq_domain` works as follows:

1. Each side-info feature uses an independent embedding lookup, producing `[B, L, emb_dim]`.
2. High-cardinality sequence features receive extra dropout during training if their vocab size exceeds `seq_id_threshold`.
3. Side-info embeddings at the same event position are concatenated.
4. A Linear + LayerNorm projection maps the event representation to `d_model`.
5. Time-bucket embeddings are added.

Each behavior event becomes one `d_model`-dimensional token. The design is clean and stable, but each domain still owns a separate tokenizer and projection.

## 7. Query Generator

`MultiSeqQueryGenerator` generates query tokens for each sequence domain. For each domain, query generation depends on:

- flattened NS tokens;
- the mean-pooled representation of the current domain sequence.

Independent FFNs generate `num_queries` query tokens per domain. This can be interpreted as a retrieval-style mechanism: the model uses current user/item/context information and a sequence summary to decide what should be read from each sequence.

## 8. MultiSeqHyFormerBlock

`MultiSeqHyFormerBlock` is the main stackable block. Each block has three stages.

### 8.1 Sequence Evolution

Each sequence domain is independently processed by a sequence encoder. Three encoder types are available:

- `swiglu`: attention-free SwiGLU feed-forward encoder;
- `transformer`: standard self-attention + FFN;
- `longer`: top-k compressed attention for longer sequences.

The default is `transformer`.

### 8.2 Query Decoding

Each domain's query tokens cross-attend to the encoded sequence of the same domain. This allows query tokens to read sequence information conditioned on the current sample.

### 8.3 Token Fusion

The block concatenates all decoded query tokens and NS tokens:

```text
[Q_a, Q_b, Q_c, Q_d, NS]
```

The concatenated tokens are passed to `RankMixerBlock`, which supports:

- `full`: token mixing + FFN; requires `d_model % T == 0`;
- `ffn_only`: per-token FFN only;
- `none`: identity passthrough.

The default mode is `full`.

## 9. Prediction Head

After all blocks, the model only uses final query tokens:

```python
all_q = torch.cat(curr_qs, dim=1)
output = all_q.view(B, -1)
output = self.output_proj(output)
logits = self.clsfier(output)
```

The final `curr_ns` tokens are not directly fed into the prediction head. NS tokens can only affect the prediction indirectly through RankMixer interactions with query tokens. For CVR prediction, strong non-sequential signals from user, item, and context features may deserve a direct path to the head.

## 10. Training Pipeline

Training is implemented in `trainer.py`.

### 10.1 Loss

Two losses are supported:

- `bce`: `binary_cross_entropy_with_logits`;
- `focal`: custom sigmoid focal loss.

The default is BCE.

### 10.2 Optimizer

The trainer separates parameters into:

- sparse parameters: all `nn.Embedding` weights, optimized by Adagrad;
- dense parameters: all non-embedding weights, optimized by AdamW.

This is common in recommendation and advertising systems because sparse embeddings are updated differently from dense neural-network layers.

### 10.3 High-Cardinality Embedding Reinitialization

The model supports reinitializing high-cardinality embeddings after configured epochs:

```text
--reinit_sparse_after_epoch
--reinit_cardinality_threshold
```

The goal is to reduce overfitting when reusing the same training data for multiple epochs.

### 10.4 Evaluation

Validation procedure:

1. Run forward pass on the validation set.
2. Apply sigmoid to logits.
3. Compute ROC-AUC with sklearn `roc_auc_score`.
4. Compute binary logloss.
5. Use AUC for early stopping.

NaN predictions are filtered before metric computation.

## 11. Strengths

- Complete data engineering: Parquet row-group loading, list padding, multi-domain sequences, time buckets, OOB clipping, and multi-worker DataLoader.
- Stronger than a simple DNN: tokenization, query-based sequence reading, stackable blocks, token mixer, and Transformer encoder are included.
- Good scaling controls: `d_model`, `emb_dim`, number of layers, heads, queries, sequence lengths, and token counts are configurable.
- Complete training loop: loss, AUC, logloss, early stopping, checkpointing, and TensorBoard logging are available.

## 12. Limitations

- The backbone is not fully unified: sequences and non-sequence features still follow separate early branches.
- Aligned dense/integer features are underused: aligned int lists and dense lists are not modeled element by element.
- Final NS tokens are not directly used by the prediction head.
- Multi-value features mostly rely on mean pooling.
- Full `RankMixerBlock` requires `d_model % T == 0`, limiting configuration flexibility.

## 13. Recommended Improvements

1. **Use Q + NS tokens in the final head**  
   Concatenate final query tokens and final NS tokens, or add a CLS/CVR token.

2. **Explicitly model aligned dense-integer features**  
   For `user_int_feats_62-66/89-91` and aligned dense features, build element-level tokens: `id_embedding + dense_value_projection + field_embedding`.

3. **Build a more unified token stream**  
   Feed target item tokens, user fields, context fields, sequence event tokens, and dense-aligned tokens into a homogeneous block.

4. **Run systematic scaling-law experiments**  
   Sweep `d_model`, number of layers, heads, sequence length, query count, and NS token count. Track AUC, logloss, throughput, memory, and latency.

## 14. Optional Direction: GNN to Transformer

One optional research direction is **GNN to Transformer**. This content is kept only in the English analysis file and intentionally not included in `README.md` or the Chinese analysis.

The idea is to use a graph encoder as a relation-induction module and then feed graph-enhanced tokens into a Transformer-style unified backbone. It may be useful because anonymized integer IDs often contain strong co-occurrence, transition, and high-order interaction patterns.

Possible graph structures:

- user-item interaction graph;
- item-item co-occurrence graph;
- feature-id co-occurrence graph;
- sequence event transition graph;
- in-sample heterogeneous graph over user, item, field, domain, and behavior tokens.

A low-risk implementation path would be:

1. Build an offline feature-id or item-item co-occurrence graph from training data only.
2. Train or compute graph embeddings.
3. Inject graph embeddings into existing user/item/sequence embedding paths.
4. Keep the HyFormer backbone unchanged for the first ablation.
5. Compare against the original baseline on AUC, logloss, memory, and throughput.

Key risks include graph construction cost, long-tail ID overfitting, temporal leakage, and extra inference latency.

## 15. Suggested Experiment Order

1. Reproduce the official baseline AUC locally.
2. Modify the final head to use both Q and NS tokens.
3. Test RoPE, longer encoder, and different sequence lengths.
4. Implement aligned dense-int tokens.
5. Implement a CLS/CVR-token unified backbone.
6. Build a scaling-law experiment table.
7. Optionally test graph-enhanced embeddings as an additional ablation.

## 16. Summary

This baseline is a complete and reasonably strong official starting point. It is close to the competition theme, but it is still a hybrid architecture rather than a fully unified model. The most valuable next steps are final-head improvement, aligned dense-int tokenization, a more unified token stream, and systematic scaling-law experiments.
