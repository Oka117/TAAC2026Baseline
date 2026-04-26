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
| GNN layers | 1 |
| Graph type | fully connected graph over NS tokens |
| Default run script | enabled in `run.sh` |

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
5. add a residual connection and LayerNorm.

This keeps the module cheap and stable.

## Expected Result

- AUC may improve slightly or stay close to baseline.
- Training speed will be slightly slower.
- Risk is low because sequence encoders, token counts, and HyFormer blocks are unchanged.

## How to Run

`run.sh` enables GNN-NS by default:

```bash
bash run.sh
```

Equivalent explicit flags:

```bash
--use_token_gnn \
--token_gnn_layers 1 \
--token_gnn_graph full
```

## How to Disable

Use `train.py` directly without `--use_token_gnn`, or remove these flags from `run.sh`:

```bash
--use_token_gnn
--token_gnn_layers 1
--token_gnn_graph full
```

## Recommended Ablation

Run two experiments with the same data split and seed:

| Experiment | Flags |
| --- | --- |
| Baseline | no `--use_token_gnn` |
| GNN-NS | `--use_token_gnn --token_gnn_layers 1 --token_gnn_graph full` |

Compare:

- validation AUC;
- validation logloss;
- training throughput;
- GPU memory usage.
