# 第零次实验方案：FE-00 缺失处理与归一化对齐

## 1. 实验目标

对齐上传 DOCX 的第 1、2 条设计：

```text
1. delete int feature from user whose missing value proportion >75%
2. replace missing value in int_feats using average value
3. all dense numerical feature(including sequence side feature) do normalization
```

FE-00 是预处理和统计实验，不改变 HyFormer block。

## 2. 实验内容

### 2.1 User Int 高缺失字段处理

统计每个 `user_int_feats_*` 的 missing ratio：

```text
missing = null or value <= 0 or empty list
drop_user_int_fid if missing_ratio > 0.75
```

输出：

```text
feature_missing_report.json
dropped_user_int_fids.json
schema.fe00.json
ns_groups.fe00.json
```

### 2.2 Int Missing Fill

DOCX 原文要求：

```text
Replace missing value in int_feats using average value
```

baseline 适配规则：

```text
若字段被确认是 ordinal/numeric int：使用 train split mean/round 后填充
若字段是匿名 categorical id：不使用 raw average，保留 0 missing bucket
```

原因：当前 baseline 的 int feature 会进入 Embedding，未确认语义的平均 id 会制造不存在的类别。FE-00 需要先输出字段角色判断报告，再决定哪些 fid 可用 average fill。

### 2.3 Dense Numerical Normalization

对所有 dense numerical feature 拟合 train-only normalization stats：

```text
user_dense_feats_*
item_dense_feats_*（增强后）
明确为 numerical 的 sequence side feature
```

默认变换：

```text
clip at q=0.999
zscore = (x - mean_train) / std_train
```

注意：匿名 categorical sequence side feature 不做 z-score，继续走 embedding。

## 3. 参数

```bash
python3 tools/build_feature_engineering_dataset.py \
  --input_dir /path/to/original_dataset \
  --input_schema /path/to/original_dataset/schema.json \
  --output_dir /path/to/fe01_dataset \
  --match_window_days 7 \
  --match_count_buckets 0,1,2,4,8 \
  --fit_stats_row_group_ratio 0.9
```

当前仓库的 FE-01 脚本已经实现新增 dense 特征的 `log1p + zscore`。完整 FE-00 的“删除高缺失 user int”和“可确认 numerical int 的 average fill”需要后续继续扩展脚本。

## 4. 预期结果

| 方向 | 预测 |
| --- | --- |
| 删除高缺失 user int | 可能降低过拟合，AUC `-0.0002 ~ +0.0006` |
| int average fill | 仅对 ordinal/numeric 字段可能有效；categorical id 上可能负收益 |
| dense normalization | 通常稳定训练，AUC 小幅正向或持平 |

验收标准：

```text
1. 删除字段后 schema、ns_groups、parquet 完全一致
2. normalization stats 只在 train row groups 拟合
3. 不把 categorical id 当 continuous value 做 z-score
```
