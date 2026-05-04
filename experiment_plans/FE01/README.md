# FE-01 文档索引

## 实验定位

FE-01 只验证 DOCX 中第一组低泄漏风险特征，不加入 delay loss 或 multi-task：

```text
P005/P009: user prefix frequency dense
P007/P011: item prefix frequency dense
P017/P018: item_int_feats_9 与 domain_d_seq_19 的 match 特征
P019/P020: min_match_delta 与 match_count_7d dense
```

## 代码文件

| 文件 | 作用 |
| --- | --- |
| `run_fe01.sh` | FE-01 一键入口，先构建增强数据再训练 |
| `build_feature_engineering_dataset.py` | 生成 FE-01 parquet/schema/ns_groups/stats/DOCX 对齐报告 |
| `ns_groups.feature_engineering.json` | FE-01 推荐 int-only NS groups |
| `dataset.py` | 读取 `item_dense`，让新增 item dense token 进入模型 |
| `trainer.py` | 保存 checkpoint 时打包 FE-01 stats sidecar，供严格评估复用 |
| `evaluation/FE01/infer.py` | 评估时把 raw eval parquet 转成 FE-01 增强 parquet，再按 checkpoint schema 推理 |

## 文档文件

| 文件 | 作用 |
| --- | --- |
| `experiment_01_feature_engineering_plan.zh.md` | FE-01 实验方案、参数、输出和 DOCX 对齐说明 |

## 平台上传清单

FE-01 最小需要更新：

```text
dataset.py
build_feature_engineering_dataset.py
ns_groups.feature_engineering.json
trainer.py
```

如果平台只能执行固定的 `run.sh`，需要把 `run_fe01.sh` 的内容上传/覆盖为平台 `run.sh`。当前仓库根目录 `run.sh` 保留为 FE-00 入口，避免两个实验入口混淆。

其余 baseline 训练文件保留：

```text
model.py
train.py
utils.py
```

## 一致性检查

```text
run_fe01.sh -> build_feature_engineering_dataset.py -> FE01_ROOT/schema.json
run_fe01.sh -> FE01_ROOT/ns_groups.feature_engineering.json -> train.py
dataset.py -> item_dense_feats -> model.py item_dense_proj
trainer.py -> checkpoint/feature_engineering_stats.json
evaluation/FE01/infer.py -> feature_engineering_stats.json -> FE01 eval parquet
```

重新训练后的 best checkpoint 应包含：

```text
model.pt
schema.json
ns_groups.feature_engineering.json
train_config.json
feature_engineering_stats.json
```

默认训练参数：

```text
user_ns_tokens=6
item_ns_tokens=4
num_queries=1
patience=3
num_epochs=6
```

token 整除关系：

```text
num_ns = 6 + 1(user_dense) + 4 + 1(item_dense) = 12
T = 1 * 4 + 12 = 16
64 % 16 == 0
```
