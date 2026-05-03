# FE-00 文档索引

## 实验定位

FE-00 对齐 `feature-engineering.docx` 中最基础的数据预处理设计：

```text
P000: 删除 user int 高缺失字段
P001: int_feats missing average fill
P002: dense numerical normalization
```

## 代码文件

| 文件 | 作用 |
| --- | --- |
| `run.sh` | 平台 FE-00 入口，先构建 FE-00 数据再训练 |
| `run_fe00.sh` | 本地别名，直接委托 `run.sh` |
| `build_fe00_preprocess_dataset.py` | 生成 FE-00 parquet/schema/ns_groups 与对齐报告 |
| `dataset.py` | 训练数据读取，兼容 FE-00 输出 schema |
| `trainer.py` | 保存 checkpoint 时打包 FE-00 stats sidecar，供严格评估复用 |

## 文档文件

| 文件 | 作用 |
| --- | --- |
| `experiment_00_preprocess_alignment_plan.zh.md` | FE-00 实验方案、参数、输出和 DOCX 对齐说明 |

## 平台上传清单

FE-00 最小需要更新：

```text
run.sh
build_fe00_preprocess_dataset.py
dataset.py
trainer.py
ns_groups.json
```

其余 baseline 训练文件保留：

```text
model.py
train.py
utils.py
```

## 一致性检查

```text
run.sh -> build_fe00_preprocess_dataset.py -> FE00_ROOT/schema.json
run.sh -> FE00_ROOT/ns_groups.fe00.json -> train.py
trainer.py -> checkpoint/int_fill_values.json
trainer.py -> checkpoint/dense_normalization_stats.json
```

重新训练后的 best checkpoint 应包含：

```text
model.pt
schema.json
ns_groups.fe00.json
train_config.json
int_fill_values.json
dense_normalization_stats.json
```

默认训练参数：

```text
user_ns_tokens=5
item_ns_tokens=2
num_queries=2
patience=3
num_epochs=6
```
