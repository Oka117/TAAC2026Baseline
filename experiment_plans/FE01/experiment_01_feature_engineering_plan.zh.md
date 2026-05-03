# 第一次实验方案：FE-01 安全特征增强

## 1. 实验目标

验证 DOCX 中最容易落地、泄漏风险最低的一组特征工程是否能提升当前 HyFormer baseline 的 AUC。FE-01 严格围绕 DOCX 明确写出的 `item_int_feats_9 <-> domain_d_seq_19` 匹配设计展开，不额外引入其他 item/sequence pair。

本次实验只增强输入特征和 NS 分组，不修改 HyFormer block，不引入 delay-aware loss，不做 multi-task。这样实验变量比较干净，方便判断“新增特征本身”是否有效。

当前已生成代码：

```text
build_feature_engineering_dataset.py
run_fe01.sh
ns_groups.feature_engineering.json
dataset.py
```

实际训练关系说明：

```text
run_fe01.sh 是 FE-01 的一键实验入口。
build_feature_engineering_dataset.py 不替换 dataset.py，它先生成 FE-01 parquet/schema。
训练时仍然由 train.py + dataset.py 读取 FE-01 输出目录下的 schema.json 和 parquet。
```

也就是说，实际流程是：

```text
原始 parquet/schema -> run_fe01.sh -> build_feature_engineering_dataset.py -> FE-01 输出目录 -> train.py/dataset.py
```

## 2. 实验假设

用户/物品历史频次、购买频次，以及目标 item 属性与历史行为序列的匹配信号，能补充当前序列 token 和 NS token 没有显式表达的统计信息，从而提升 CVR 预测效果。

## 3. 实验分组

### B0：当前 Baseline 对照组

使用当前仓库默认训练方式，不使用新增离线特征，不使用新的 NS 分组。

说明：当前平台 `run.sh` 已更新为 FE-00 入口；若要跑“未做 FE-00 预处理、未做 FE-01 特征”的纯 B0 对照，应直接调用 `train.py`，并显式传入 baseline active config。B0 不额外传新的 NS groups。

推荐参数：

```bash
python3 -u train.py \
  --data_dir /path/to/original_dataset \
  --schema_path /path/to/original_dataset/schema.json \
  --ns_groups_json "" \
  --ns_tokenizer_type rankmixer \
  --user_ns_tokens 5 \
  --item_ns_tokens 2 \
  --num_queries 2 \
  --rank_mixer_mode full \
  --ckpt_dir outputs/exp_b0_baseline/ckpt \
  --log_dir outputs/exp_b0_baseline/log \
  --batch_size 256 \
  --d_model 64 \
  --emb_dim 64 \
  --num_hyformer_blocks 2 \
  --num_heads 4 \
  --seq_encoder_type transformer \
  --seq_max_lens seq_a:256,seq_b:256,seq_c:512,seq_d:512 \
  --loss_type bce \
  --lr 1e-4 \
  --sparse_lr 0.05 \
  --dropout_rate 0.01 \
  --seed 42
```

### FE-01：安全特征增强实验组

使用增强后的 parquet、增强后的 `schema.json`，并使用新的 NS 分组。

本组只测试以下特征：

```text
user_dense_feats_110 = log1p(user_total_frequency_before_timestamp)
user_dense_feats_111 = log1p(user_purchase_frequency_before_timestamp)

item_dense_feats_86 = log1p(item_total_frequency_before_timestamp)
item_dense_feats_87 = log1p(item_purchase_frequency_before_timestamp)

item_int_feats_89 = has_match(item_int_feats_9, domain_d_seq_19)
item_int_feats_90 = bucketize(match_count(item_int_feats_9, domain_d_seq_19))

item_dense_feats_91 = log1p(min_match_delta(item_int_feats_9, domain_d_seq_19))
item_dense_feats_92 = log1p(match_count_7d(item_int_feats_9, domain_d_seq_19))
```

说明：DOCX 原文写的是 `Item_int_feats_90 = match_count(item_int_feats_9, domain_d_seq_19)`。结合当前 baseline 的 item int 特征会进入 Embedding，FE-01 将它设计为分桶后的 categorical count；未分桶的连续 count 不直接作为 item int id。

本次不加入：

```text
user_dense_feats_112 = user_avg_delay
item_dense_feats_88 = item_avg_delay
delay-aware weighted loss
multi-task loss
```

原因是 delay 类特征更容易引入 label_time 泄漏，建议第二次实验单独验证。

## 4. 数据预处理参数

离线预处理脚本建议命名为：

```text
build_feature_engineering_dataset.py
```

当前仓库已提供该脚本。它会输出增强 parquet、增强 `schema.json`、`ns_groups.feature_engineering.json`、`feature_engineering_stats.json` 和 `docx_alignment.fe01.json`。

输入：

```bash
python3 build_feature_engineering_dataset.py \
  --input_dir /path/to/original_dataset \
  --input_schema /path/to/original_dataset/schema.json \
  --output_dir /path/to/fe01_dataset \
  --match_window_days 7 \
  --match_count_buckets 0,1,2,4,8 \
  --fit_stats_row_group_ratio 0.9
```

当前脚本参数与固定设置：

| 项 | 值 | 说明 |
| --- | ---: | --- |
| `timestamp` | 固定列 | 用于 prefix 统计和防泄漏 |
| `label_type` | 固定列 | `label_type == 2` 作为 purchase/conversion |
| prefix 统计 | 固定为不包含当前样本 | batch 内按 timestamp 稳定排序，先产出特征，再更新当前样本状态 |
| `--match_window_days` | `7` | 计算 `match_count_7d` |
| `--match_count_buckets` | `0,1,2,4,8` | `item_int_feats_90` 的 count 分桶边界，生成 0-5 编码 |
| dense transform | 固定为 `log1p,zscore` | 脚本两遍扫描：先拟合均值方差，再写增强 parquet |
| `--fit_stats_row_group_ratio` | `0.9` | 用前 90% row groups 拟合 normalization，对齐默认尾部 10% 验证 |
| int missing policy | 固定为 `zero_bucket` | int 缺失继续映射到 0，不做平均填充 |
| `--batch_size` | `8192` | 离线脚本的 parquet batch 大小，可按内存调整 |

防泄漏要求：

- 所有 user/item frequency 必须只统计当前样本 `timestamp` 之前的数据。
- 全局严格防泄漏要求输入 parquet/row group 按时间或训练可见顺序排列；脚本能保证 batch 内 timestamp 顺序，但不能重排跨 row group 的物理顺序。
- `purchase_frequency` 只能使用历史中已经观测到的正样本。
- validation/test 的统计只能来自训练历史或当前行之前的可见历史，不能使用未来行。
- 当前样本的 `label_time` 不进入模型输入。
- `min_match_delta` 使用样本 `timestamp` 与历史行为事件时间的差值，不使用 `label_time`。

## 5. Schema 修改

增强后的 `schema.json` 需要新增：

```json
{
  "user_dense": [
    [110, 1],
    [111, 1]
  ],
  "item_dense": [
    [86, 1],
    [87, 1],
    [91, 1],
    [92, 1]
  ],
  "item_int": [
    [89, 3, 1],
    [90, 6, 1]
  ]
}
```

注意：上面是“增量说明”，实际生成的 `schema.json` 必须保留原始所有字段，并把新增字段合并进去。

`item_int_feats_89/90` 编码建议：

```text
item_int_feats_89:
  0 = missing / padding
  1 = no match
  2 = has match

item_int_feats_90:
  0 = missing / padding
  1 = count == 0
  2 = count == 1
  3 = 2 <= count < 4
  4 = 4 <= count < 8
  5 = count >= 8
```

这里 `[90, 6, 1]` 与上述 0-5 编码对应。若后续想把 `match_count` 当连续值使用，应新增 dense feature，而不是继续用 item int embedding。

## 6. NS 分组参数

新增文件：

```text
ns_groups.feature_engineering.json
```

建议内容：

```json
{
  "user_ns_groups": {
    "U1_user_profile": [1, 15, 48, 49],
    "U2_user_behavior_stats": [50, 60],
    "U3_user_context": [51, 52, 53, 54, 55, 56, 57, 58, 59],
    "U4_user_temporal_behavior": [62, 63, 64, 65, 66],
    "U5_user_interest_ids": [80, 82, 86],
    "U6_user_long_tail_sparse": [89, 90, 91, 92, 93],
    "U7_user_high_cardinality": [94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
  },
  "item_ns_groups": {
    "I1_item_identity": [5, 6, 7, 8],
    "I2_item_category_brand": [9, 10, 11, 12, 13],
    "I3_item_semantic_sparse": [16, 81, 83, 84, 85],
    "I4_target_matching_fields": [89, 90]
  }
}
```

注意：DOCX 的 NS group 示例中把 `user_dense_feats_110/111/112` 和 `item_dense_feats_86/87/88/91/92` 混入了 int groups。当前 baseline 的 `ns_groups_json` 只接受 int fid，所以 FE-01 不把这些 dense fid 写进 NS groups；它们通过 user/item dense token 进入模型。

## 7. 训练参数

FE-01 推荐训练命令：

```bash
bash run_fe01.sh \
  --data_dir /path/to/original_dataset \
  --schema_path /path/to/original_dataset/schema.json \
  --ckpt_dir outputs/exp_fe01_safe_features/ckpt \
  --log_dir outputs/exp_fe01_safe_features/log \
  --seed 42
```

`run_fe01.sh` 会先生成 FE-01 输出目录，再自动把训练阶段的 `TRAIN_DATA_PATH`、`--schema_path`、`--ns_groups_json` 指向 FE-01 输出目录。若平台 `/tmp` 空间不足，可设置 `FE01_DATA_DIR=/path/to/writable_scratch`。

平台如果只能执行固定的 `run.sh`，则 FE-01 提交时需要把 `run_fe01.sh` 的内容上传/覆盖为平台的 `run.sh`；当前根目录 `run.sh` 仍保留为 FE-00 入口。

关键参数解释：

| 参数 | 值 | 原因 |
| --- | ---: | --- |
| `--num_queries` | `1` | 保持 token 总数可被 `d_model=64` 整除 |
| `--user_ns_tokens` | `6` | 压缩 user int groups，控制 `T` |
| `--item_ns_tokens` | `4` | 保留 item matching fields 独立语义 |
| `--rank_mixer_mode` | `full` | 继续使用完整 RankMixer |
| `--d_model` | `64` | 与 baseline 保持一致 |
| `--loss_type` | `bce` | 第一次实验不混入 loss 变量 |
| `--seq_max_lens` | `a/b=256,c/d=512` | 沿用当前默认配置 |
| `--patience` | `3` | 与 FE-00 当前实验轮数设置保持一致 |
| `--num_epochs` | `6` | 与 FE-00 当前实验轮数设置保持一致 |

加入 item dense token 后：

```text
num_ns = user_ns_tokens + user_dense_token + item_ns_tokens + item_dense_token
       = 6 + 1 + 4 + 1
       = 12

T = num_queries * 4 + num_ns
  = 1 * 4 + 12
  = 16

d_model = 64, 64 % 16 == 0
```

因此该配置满足 `rank_mixer_mode=full` 的整除约束。

## 8. 需要配套修改的代码

FE-01 至少需要两处代码修改：

1. 新增 `build_feature_engineering_dataset.py`。
2. 修改 `dataset.py`，支持从 schema 读取 `item_dense` 并返回非空 `item_dense_feats`。
3. 新增/更新 `run_fe01.sh`，负责原始数据到 FE-01 数据再到训练的一键链路。

`model.py` 第一轮不需要改，因为当前已经有：

```text
item_dense_proj
item_dense token
```

只要 `dataset.py` 的 `item_dense_schema.total_dim > 0`，模型就会自动启用 item dense token。

训练前必须做一次 preflight 检查：

```text
1. 增强 parquet 中存在 user_dense_feats_110/111、item_dense_feats_86/87/91/92、item_int_feats_89/90。
2. 增强 schema 中 `item_dense` 不为空。
3. `train.py` 日志中 `num_ns=12, T=16, d_model=64`。
4. `dataset.py` 返回的 `item_dense_feats` shape 不是 `[B, 0]`。
```

## 9. 验收指标

主指标：

```text
valid AUC
```

辅助指标：

```text
valid logloss
训练速度 samples/sec
显存峰值
best checkpoint step
```

判定标准：

- FE-01 AUC 高于 B0，且至少连续 2 个 seed 方向一致，才认为有效。
- 如果 AUC 提升但 logloss 明显变差，需要检查是否 frequency 特征过拟合。
- 如果训练显存或速度显著恶化，优先减少 `user_ns_tokens` 或增大 dense 特征压缩强度，而不是先改 backbone。

推荐补跑 seed：

```text
seed = 42, 2026, 3407
```

## 10. 实验记录模板

| 实验 | Seed | Best Step | Valid AUC | Valid Logloss | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| B0 | 42 |  |  |  | 当前 baseline |
| FE-01 | 42 |  |  |  | 安全特征增强 |
| FE-01 | 2026 |  |  |  | 复验 |
| FE-01 | 3407 |  |  |  | 复验 |

## 11. 下一步

如果 FE-01 有稳定收益，第二次实验再加入：

```text
user_dense_feats_112 = historical user_avg_delay
item_dense_feats_88 = historical item_avg_delay
delay-aware weighted BCE
```

如果 FE-01 没收益，优先检查：

```text
1. prefix frequency 是否泄漏或统计窗口错误
2. item_dense 是否真的被 dataset.py 读入
3. 新增 schema 与 parquet 列是否一致
4. match feature 的正负样本 lift 是否在完整训练集上成立
```
