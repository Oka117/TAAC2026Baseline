# 4-Layer GNN-NS Generalization Experiment

## Goal

Build a conservative generalization variant on top of the current 4-layer
GNN-NS baseline:

```text
PCVRHyFormer + 4-layer TokenGNN + token augmentation + EMA/SWA weight averaging
```

The experiment combines two low-risk directions:

1. token-level stochastic augmentation, to reduce over-reliance on individual
   sparse fields or sequence events;
2. flat-minima training through EMA/SWA, to evaluate and save smoother model
   weights.

## Model Structure

The unchanged 4-layer GNN-NS path is:

```text
user/item/dense features -> NS tokenizer -> ns_tokens
                                           -> 4-layer TokenGNN
                                           -> graph-enhanced ns_tokens
seq features             -> sequence tokenizers -> seq_tokens
graph-enhanced ns_tokens + seq_tokens -> query generator
query tokens + ns/seq tokens          -> HyFormer blocks
final query representation            -> CVR classifier
```

The generalization path adds training-only augmentation:

```text
NS tokenizer output
  -> NS token dropout/noise
  -> 4-layer TokenGNN

Sequence tokenizer output
  -> sequence token dropout/noise
  -> query generator + HyFormer blocks
```

Inference is deterministic. The augmentation code is disabled automatically
when the model is in eval mode, and the same checkpoint can be loaded by
`evaluation/infer.py`.

For flat-minima training, the optimizer still updates the live model as before:

```text
Embedding params -> Adagrad
Dense params     -> AdamW
```

After each optimizer step, the trainer optionally updates a shadow average of
selected parameters:

```text
EMA: avg = decay * avg + (1 - decay) * current
SWA: avg = arithmetic mean of sampled weights
```

Validation and best-checkpoint saving temporarily swap the averaged dense
weights into the live model. Training then continues from the normal live
weights.

## Main Parameters

### Token Augmentation

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `--use_generalization_aug` | off | Enables training-time token augmentation. |
| `--ns_token_dropout_rate` | `0.0` | Drops whole non-sequential tokens before `TokenGNN`. |
| `--ns_token_noise_std` | `0.0` | Adds Gaussian noise to non-sequential tokens before `TokenGNN`. |
| `--seq_token_dropout_rate` | `0.0` | Drops whole sequence event tokens after sequence embedding. |
| `--seq_token_noise_std` | `0.0` | Adds Gaussian noise to sequence event tokens after sequence embedding. |

The active `run.sh` setting is intentionally mild:

```bash
--use_generalization_aug
--ns_token_dropout_rate 0.05
--ns_token_noise_std 0.01
--seq_token_dropout_rate 0.03
--seq_token_noise_std 0.005
```

### Weight Averaging

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `--weight_averaging` | `none` | One of `none`, `ema`, or `swa`. |
| `--ema_decay` | `0.995` | EMA decay when `--weight_averaging=ema`. |
| `--weight_avg_start_step` | `0` | First global step that updates averaged weights. |
| `--weight_avg_update_every` | `1` | Update averaged weights every N optimizer steps. |
| `--weight_avg_include_sparse` | off | Also average sparse embeddings. Disabled by default to save memory. |

The active `run.sh` setting uses:

```bash
--weight_averaging ema
--ema_decay 0.995
--weight_avg_start_step 0
--weight_avg_update_every 1
```

## Experiment Purpose

This experiment tests whether the 4-layer GNN-NS model can become less
sensitive to sparse feature noise and validation split variance without
changing the main architecture.

Specific questions:

1. Does token dropout/noise improve validation logloss or reduce overfitting?
2. Does EMA produce a smoother validation curve than raw weights?
3. Does the combination preserve the original 4-layer GNN-NS training
   stability?
4. Are gains large enough to justify the extra regularization cost?

## Recommended Ablation

Run all experiments with the same data split and seed:

| Experiment | Flags |
| --- | --- |
| GNN-NS baseline | `--use_token_gnn --token_gnn_layers 4` |
| Aug only | baseline + `--use_generalization_aug ...` |
| EMA only | baseline + `--weight_averaging ema --ema_decay 0.995` |
| Aug + EMA | active `run.sh` setting |
| Aug + SWA | replace EMA with `--weight_averaging swa` |

Suggested parameter sweep:

| Group | Values |
| --- | --- |
| `ns_token_dropout_rate` | `0.02`, `0.05`, `0.10` |
| `seq_token_dropout_rate` | `0.01`, `0.03`, `0.05` |
| `ns_token_noise_std` | `0.005`, `0.01`, `0.02` |
| `ema_decay` | `0.99`, `0.995`, `0.999` |

## Expected Results

On small smoke-test splits:

- AUC may move noisily and should not be over-interpreted.
- Logloss should stay close to baseline or improve slightly.
- Training should remain stable with no NaN predictions.
- EMA may lag on very short runs, so `0.99` or `0.995` can be better than
  `0.999` for quick tests.

On larger realistic splits:

- Augmentation should reduce overfitting to high-cardinality sparse fields.
- EMA/SWA should improve validation stability and often produce a better
  best checkpoint.
- The combined setting should be most useful when train/valid are split by
  time or long-tail item/user groups.

## Performance Analysis

### Compute Cost

Token dropout and Gaussian noise are elementwise operations. Their overhead is
small compared with sequence attention and the HyFormer blocks. The 4-layer
TokenGNN cost remains unchanged.

EMA/SWA adds a per-step update over averaged parameters. By default only dense
parameters are averaged, so the overhead is modest.

### Memory Cost

Token augmentation adds no persistent parameters.

EMA/SWA stores one shadow copy of averaged parameters. Sparse embeddings are
excluded by default because recommender embedding tables can dominate memory.
Use `--weight_avg_include_sparse` only for controlled experiments where memory
headroom is known.

### Latency Cost

Inference latency is unchanged:

- token augmentation is disabled in eval mode;
- checkpoints contain ordinary model weights;
- no extra averaging module is used during inference.

### Risk

| Risk | Mitigation |
| --- | --- |
| Excessive token dropout hurts signal | Start with `0.03` to `0.05`; sweep upward only after stable baseline. |
| Noise destabilizes low-dimensional tokens | Keep `noise_std <= 0.02` initially. |
| EMA too stale on short runs | Use `ema_decay=0.99` or `0.995` for smoke tests. |
| Sparse embedding averaging doubles memory | Keep `--weight_avg_include_sparse` off by default. |

## Current Active Command

`run.sh` now enables the combined experiment by default:

```bash
bash run.sh
```

To disable the new generalization additions while keeping 4-layer GNN-NS, run
`train.py` directly without the `--use_generalization_aug` and
`--weight_averaging` flags, or remove those flags from `run.sh`.

## References

- Mixup-style vicinal smoothing: https://arxiv.org/abs/1710.09412
- SAM and flat minima: https://arxiv.org/abs/2010.01412
- SWAD for domain generalization by flat minima: https://arxiv.org/abs/2102.08604
- Simple graph contrastive recommendation: https://arxiv.org/abs/2112.08679
- XSimGCL noise-based recommendation contrastive learning: https://arxiv.org/abs/2209.02544
