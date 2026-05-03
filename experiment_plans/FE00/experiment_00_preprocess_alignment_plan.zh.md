# 第零次实验方案：FE-00 缺失处理与归一化对齐

## 1. 实验目标

对齐上传 DOCX 的第 1、2 条设计：

```text
1. delete int feature from user whose missing value proportion >75%
2. replace missing value in int_feats using average value
3. all dense numerical feature(including sequence side feature) do normalization
```

FE-00 是预处理和统计实验，不改变 HyFormer block。

当前已生成代码：

```text
build_fe00_preprocess_dataset.py
run.sh
run_fe00.sh
```

实际训练关系说明：

```text
平台实际入口是 run.sh，当前 run.sh 已更新为 FE-00 入口。
run_fe00.sh 只是本地别名，会直接委托 run.sh。
build_fe00_preprocess_dataset.py 不替换 dataset.py，它先生成 FE-00 parquet/schema。
训练时仍然由 train.py + dataset.py 读取 FE-00 输出目录下的 schema.json 和 parquet。
```

也就是说，实际流程是：

```text
原始 parquet/schema -> run.sh -> build_fe00_preprocess_dataset.py -> FE-00 输出目录 -> train.py/dataset.py
```

## 1.1 与原 DOCX 的一一关联

| DOCX 位置 | 原文设计 | FE-00 实现 |
| --- | --- | --- |
| P000 | `delete int feature from user whose missing value proportion >75%` | `build_fe00_preprocess_dataset.py` 统计 `user_int_feats_*` missing ratio，超过阈值的 fid 从输出 `schema.json` 和 `ns_groups.fe00.json` 删除 |
| P001 | `Replace missing value in int_feats using average value` | 脚本对 user/item int 特征用 train row groups 的正值均值四舍五入后填补 missing；list int 默认只填补已有元素中的 missing，`--fill_empty_int_lists` 可填补空 list |
| P002 | `all dense numerical feature(including sequence side feature) do normalization` | 脚本对 user/item dense 数值列拟合 train row groups z-score 并重写 parquet；sequence side 在当前 baseline 是 embedding id，默认不 z-score，并在 `docx_alignment.fe00.json` 中记录适配说明 |

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
schema.json
ns_groups.fe00.json
```

### 2.2 Int Missing Fill

DOCX 原文要求：

```text
Replace missing value in int_feats using average value
```

baseline 适配规则：

```text
使用 train row groups 中的正值均值，round 后填充 user/item int missing
list int 默认只填已有元素中的 missing；空 list 默认保持为空
如需填充空 list，显式开启 --fill_empty_int_lists
```

说明：当前 baseline 的 int feature 会进入 Embedding；这里按 DOCX 原文先做 FE-00 对齐实验，并把实际 fill value 输出到 `int_fill_values.json`，便于复核是否存在负收益字段。

### 2.3 Dense Numerical Normalization

对所有 dense numerical feature 拟合 train-only normalization stats：

```text
user_dense_feats_*
item_dense_feats_*（增强后）
明确为 numerical 的 sequence side feature
```

默认变换：

```text
zscore = (x - mean_train) / std_train
```

注意：匿名 categorical sequence side feature 不做 z-score，继续走 embedding。

## 3. 参数

```bash
python3 build_fe00_preprocess_dataset.py \
  --input_dir /path/to/original_dataset \
  --input_schema /path/to/original_dataset/schema.json \
  --output_dir /path/to/fe00_dataset \
  --ns_groups_json ns_groups.json \
  --missing_threshold 0.75 \
  --fit_stats_row_group_ratio 0.9
```

输出：

```text
/path/to/fe00_dataset/schema.json
/path/to/fe00_dataset/ns_groups.fe00.json
/path/to/fe00_dataset/dropped_user_int_fids.json
/path/to/fe00_dataset/feature_missing_report.json
/path/to/fe00_dataset/int_fill_values.json
/path/to/fe00_dataset/dense_normalization_stats.json
/path/to/fe00_dataset/docx_alignment.fe00.json
```

训练：

```bash
bash run.sh \
  --data_dir /path/to/original_dataset \
  --schema_path /path/to/original_dataset/schema.json \
  --ckpt_dir outputs/exp_fe00_preprocess/ckpt \
  --log_dir outputs/exp_fe00_preprocess/log \
  --seed 42
```

`run.sh` 会自动把训练阶段的 `TRAIN_DATA_PATH`、`--schema_path`、`--ns_groups_json` 指向 FE-00 输出目录。若平台 `/tmp` 空间不足，可设置 `FE00_DATA_DIR=/path/to/writable_scratch`。

FE-00 当前默认训练参数：

```text
--user_ns_tokens 5
--item_ns_tokens 2
--num_queries 2
--patience 3
--num_epochs 6
```

该组合在原始 schema 没有 item dense token 时满足：

```text
num_ns = 5 + 1(user_dense) + 2 = 8
T = 2 * 4 + 8 = 16
64 % 16 == 0
```

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
