# GNN-NS Experiment Plan

## Goal

Add a low-risk graph module on top of the existing baseline:

```text
PCVRHyFormer + TokenGNN
```

The goal is to let non-sequential tokens exchange field-level information before they are used by the query generator and HyFormer blocks.

## Design

| Item | Setting |
| --- | --- |
| Base model | `PCVRHyFormer` |
| New module | `TokenGNN` |
| GNN position | after `ns_tokens` construction |
| GNN layers | 4 |
| Graph type | fully connected graph over NS tokens |
| Default run script | enabled in `run.sh` |
| Stability setting | residual layer scale initialized to `0.1` |

## Forward Path

The original baseline path is:

```text
user/item/dense features -> NS tokenizer -> ns_tokens
seq features             -> sequence tokens
ns_tokens + sequence tokens -> query generator -> HyFormer blocks -> CVR head
```

The GNN-NS path is:

```text
user/item/dense features -> NS tokenizer -> ns_tokens
                                           -> TokenGNN
                                           -> graph-enhanced ns_tokens
seq features             -> sequence tokens
graph-enhanced ns_tokens + sequence tokens -> query generator -> HyFormer blocks -> CVR head
```

## TokenGNN Layer

Each NS token is treated as a node. A fully connected graph is built inside each sample.

For each token:

1. normalize all NS tokens;
2. aggregate the mean representation of all other NS tokens;
3. combine self representation and neighbor representation;
4. apply a small nonlinear projection;
5. add a small scaled residual update.

This keeps the module cheap and stable. The residual scale starts at `0.1`, so
the four GNN layers begin close to the baseline representation instead of
rewriting the NS tokens aggressively.

## Expected Result

- AUC may improve slightly or stay close to baseline.
- Training speed will be slightly slower.
- Risk is low because sequence encoders, token counts, and HyFormer blocks are unchanged.
- On a 1k smoke-test split, metric variance can be large, so the main signal is
  successful end-to-end training plus no obvious degradation in validation loss.

## How to Run

`run.sh` enables GNN-NS by default:

```bash
bash run.sh
```

Equivalent explicit flags:

```bash
--use_token_gnn \
--token_gnn_layers 4 \
--token_gnn_graph full \
--token_gnn_layer_scale 0.1
```

## How to Disable

Use `train.py` directly without `--use_token_gnn`, or remove these flags from `run.sh`:

```bash
--use_token_gnn
--token_gnn_layers 4
--token_gnn_graph full
--token_gnn_layer_scale 0.1
```

## Recommended Ablation

Run two experiments with the same data split and seed:

| Experiment | Flags |
| --- | --- |
| Baseline | no `--use_token_gnn` |
| GNN-NS | `--use_token_gnn --token_gnn_layers 4 --token_gnn_graph full --token_gnn_layer_scale 0.1` |

Compare:

- validation AUC;
- validation logloss;
- training throughput;
- GPU memory usage.

## What to Compare Against Today's Baseline Report

Your current smoke-test baseline report is:

| Metric | Baseline value |
| --- | ---: |
| Train rows | 500 |
| Valid rows | 500 |
| Batch size | 16 |
| Train steps | 32 |
| Epochs | 1 |
| Average train loss | 0.3967 |
| Validation AUC | 0.7083 |
| Validation LogLoss | 0.3160 |

For the 4-layer GNN-NS run, inspect these differences:

| Result field | Expected GNN-NS behavior |
| --- | --- |
| Data split | Should remain `train 500 / valid 500`. Any change means data setup changed. |
| Train steps | Should remain about `32/32` for batch size 16. |
| Average train loss | Should be close to baseline; a large jump suggests the GNN update is too strong. |
| Validation AUC | May be slightly higher, similar, or noisy on 500 validation rows. Treat changes below about 0.01 cautiously. |
| Validation LogLoss | Should stay close to `0.3160`; a lower value is a good sign even if AUC is noisy. |
| Total parameters | Should increase more than the 2-layer version because TokenGNN now has four small message-passing layers. |
| Step time / epoch time | Should be slower than the 2-layer version because NS tokens now run through four message-passing layers. |
| Checkpoint path | Should still save a `.best_model/model.pt` checkpoint. |
| Training stability | No NaN warnings, no exploding loss, and normal early-stopping/checkpoint logs. |

Because this is a small demo setup with skipped high-cardinality features and
missing-column zero filling, use it as a smoke test rather than a final AUC
judgment. The first real check is whether GNN-NS trains cleanly and produces
validation LogLoss/AUC close to baseline.
