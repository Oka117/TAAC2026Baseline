# FE-00 Evaluation Files

本目录保存平台默认评估文件的 FE-00 版本：

```text
infer.py
dataset.py
model.py
```

## 评估流程

`infer.py` 会先把平台原始评估 parquet 转成 FE-00 评估 parquet，再加载 checkpoint 推理：

```text
EVAL_DATA_PATH raw parquet
-> FE-00 transform-only preprocessing
-> temporary FE-00 eval parquet
-> checkpoint schema.json / ns_groups / train_config
-> model inference
-> predictions.json
```

## 必需 sidecar

FE-00 评估必须复用训练阶段的统计文件，不能在评估集重新 fit：

```text
int_fill_values.json
dense_normalization_stats.json
```

推荐把这两个文件放在 `MODEL_OUTPUT_PATH` 指向的 checkpoint 目录，和下面文件同级：

```text
model.pt
schema.json
ns_groups.fe00.json
train_config.json
```

如果这两个文件不在 checkpoint 目录，也可以设置：

```text
FE00_STATS_DIR=/path/to/fe00_preprocess_output
```

## 可选环境变量

```text
FE00_EVAL_DATA_DIR=/tmp/taac_fe00_eval_xxx
FE00_EVAL_BUILD_BATCH_SIZE=8192
FE00_ALLOW_RAW_FALLBACK=1
```

如果缺少 `int_fill_values.json` 或 `dense_normalization_stats.json`，当前 `infer.py`
会退化为使用 checkpoint 的 `schema.json` 直接读取原始评估 parquet，避免在评估集
重新 fit 统计量。这个 fallback 能让已完成但未打包 sidecar 的 FE-00 checkpoint
跑完评估，但不是严格一致的 FE-00 评估。

如需强制缺 sidecar 即失败，可设置：

```text
FE00_REQUIRE_TRANSFORM_STATE=1
```

新的训练侧 `trainer.py` 已补充 checkpoint sidecar 打包逻辑；重新训练 FE-00 后，
best_model 目录会自动携带这两个文件，评估会使用严格 transform-only 路径。
