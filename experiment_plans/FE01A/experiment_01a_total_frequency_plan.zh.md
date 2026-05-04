# FE-01A 实验方案：Total Frequency Only

## 1. 实验定位

FE-01A 是 FE-01 的消融实验，只保留你这次指定的两个 total frequency 特征：

```text
user_dense_feats_110 = log1p(user_total_frequency_before_timestamp)
item_dense_feats_86 = log1p(item_total_frequency_before_timestamp)
```

它不引入 purchase frequency，不引入 target-history match，不改 HyFormer block，不改 loss。目的很明确：验证 `total_frequency_before_timestamp` 这类线上可复现的 prefix 统计是否能解释 FE-01 的有效部分。

## 2. 与上传特征和 DOCX 的一致性

| 来源 | 设计内容 | FE-01A 实现 |
| --- | --- | --- |
| 用户本轮指定 | `user_dense_feats_110 = log1p(user_total_frequency_before_timestamp)` | `build_feature_engineering_dataset.py --feature_set fe01a` 输出 `user_dense_feats_110` |
| 用户本轮指定 | `item_dense_feats_86 = log1p(item_total_frequency_before_timestamp)` | `build_feature_engineering_dataset.py --feature_set fe01a` 输出 `item_dense_feats_86` |
| DOCX P005 | `user_dense_feats_110 = log(1 + user_total_frequency)` | 用当前样本 timestamp 之前的 prefix user 出现次数，`log1p` 后用训练 row groups 拟合 z-score |
| DOCX P007 | `item_dense_feats_86 = log(1 + item_total_frequency)` | 用当前样本 timestamp 之前的 prefix item 出现次数，`log1p` 后用训练 row groups 拟合 z-score |

明确不包含：

```text
user_dense_feats_111
item_dense_feats_87
item_int_feats_89
item_int_feats_90
item_dense_feats_91
item_dense_feats_92
user_dense_feats_112
item_dense_feats_88
```

因此 FE-01A 与你上传的新特征清单完全一致，不混入 FE-01B 的 match 模块，也不混入完整 FE-01 中造成线上不一致风险的 purchase frequency。

## 3. 测试模块

测试模块：

```text
prefix total frequency dense features
```

它回答的问题是：

```text
只使用 user/item 在当前 timestamp 之前出现过多少次，是否能带来稳定的 valid/eval 收益？
```

该模块线上评估更安全，因为不需要当前 eval label，也不需要历史 purchase label。

## 4. 代码结构

当前已生成/更新：

```text
build_feature_engineering_dataset.py
tools/build_feature_engineering_dataset.py
run_fe01a.sh
evaluation/FE01/infer.py
trainer.py
experiment_plans/FE01A/README.md
experiment_plans/FE01A/experiment_01a_total_frequency_plan.zh.md
```

核心代码开关：

```bash
python3 build_feature_engineering_dataset.py \
  --input_dir /path/to/original_dataset \
  --input_schema /path/to/original_dataset/schema.json \
  --output_dir /path/to/fe01a_dataset \
  --feature_set fe01a \
  --fit_stats_row_group_ratio 0.9
```

`--feature_set fe01a` 会让 schema 只新增：

```json
{
  "user_dense": [[110, 1]],
  "item_dense": [[86, 1]]
}
```

## 5. 特征生成规则

统计规则：

```text
1. 按 parquet file / row group / batch 的 streaming 顺序读取。
2. batch 内按 timestamp stable sort 计算 prefix。
3. 当前样本先读取历史 state，再更新当前样本，所以当前样本不参与自己的 frequency。
4. user_total_frequency = 当前样本之前同 user 出现次数。
5. item_total_frequency = 当前样本之前同 item 出现次数。
6. dense 原始值先 log1p，再使用前 90% row groups 拟合 mean/std 做 z-score。
```

输出审计文件：

```text
feature_engineering_stats.json
docx_alignment.fe01.json
docx_alignment.fe01a.json
```

其中 `feature_engineering_stats.json` 应包含：

```text
feature_set = fe01a
dense_feature_names = [user_dense_feats_110, item_dense_feats_86]
dense_stats.user_dense_feats_110
dense_stats.item_dense_feats_86
```

## 6. NS 分组

FE-01A 不新增 int fid，因此 `ns_groups.feature_engineering.json` 会经过 schema filter 后自动去掉不存在的 `item_int_feats_89/90`。dense fid 不写入 NS groups，而是通过 user/item dense token 进入模型。

训练 token 结构仍保持 FE-01 的安全配置：

```text
user_ns_tokens = 6
item_ns_tokens = 4
num_queries = 1
```

由于 item_dense 增加了 `item_dense_feats_86`，仍有 item dense token：

```text
num_ns = 6 + 1(user_dense) + 4 + 1(item_dense) = 12
T = 1 * 4 + 12 = 16
64 % 16 == 0
```

## 7. 训练命令

```bash
bash run_fe01a.sh \
  --data_dir /path/to/original_dataset \
  --schema_path /path/to/original_dataset/schema.json \
  --ckpt_dir outputs/exp_fe01a_total_frequency/ckpt \
  --log_dir outputs/exp_fe01a_total_frequency/log \
  --seed 42
```

关键参数与 FE-01 保持一致：

| 参数 | 值 |
| --- | ---: |
| `--feature_set` | `fe01a` |
| `--user_ns_tokens` | `6` |
| `--item_ns_tokens` | `4` |
| `--num_queries` | `1` |
| `--loss_type` | `bce` |
| `--patience` | `3` |
| `--num_epochs` | `6` |

## 8. 评估设计

评估继续使用：

```text
evaluation/FE01/infer.py
```

该 infer 会读取 checkpoint 的 `schema.json`，发现只需要 `user_dense_feats_110` 和 `item_dense_feats_86`，因此只生成 FE-01A 所需列。它会复用 checkpoint 中的 `feature_engineering_stats.json`，不会在 eval 上重新拟合 normalization。

## 9. 结果预测

基于当前结果：

```text
B0 eval AUC = 0.810525
FE-01 full eval AUC = 0.775054
```

FE-01A 去掉了 purchase frequency 与 match 模块，预计 valid 收益会小于完整 FE-01，但 eval 会比完整 FE-01 稳定。

| 指标 | 预测 |
| --- | --- |
| Valid AUC | 相比 B0 `+0.0000 ~ +0.0006` |
| Valid Logloss | 持平或小幅改善 |
| Eval AUC | 预计接近 B0，范围 `0.807 ~ 0.813` |
| Infer time | 高于 B0，但应明显低于完整 FE-01 |

判断标准：

```text
1. 若 FE-01A eval 接近或高于 B0，说明 total frequency 可保留。
2. 若 FE-01A valid 涨但 eval 掉，说明 prefix streaming 历史在 valid/eval 分布仍不一致。
3. 若 FE-01A eval 正常而 FE-01 full 掉，主要风险来自 purchase frequency 或 match 模块，需要看 FE-01B。
```
