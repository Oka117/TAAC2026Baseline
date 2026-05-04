# FE-01B 实验方案：Target Item History Match Only

## 1. 实验定位

FE-01B 是 FE-01 的消融实验，只保留你这次指定的目标 item 属性与历史序列匹配特征：

```text
item_int_feats_89 = has_match(item_int_feats_9, domain_d_seq_19)
item_int_feats_90 = bucketize(match_count(item_int_feats_9, domain_d_seq_19))
item_dense_feats_91 = log1p(min_match_delta(item_int_feats_9, domain_d_seq_19))
item_dense_feats_92 = log1p(match_count_7d(item_int_feats_9, domain_d_seq_19))
```

它不引入 user/item frequency，不引入 purchase frequency，不引入 delay，不改 loss。目的：单独验证 DOCX 中 “目标 item 属性与历史序列匹配特征” 是否能稳定提升。

## 2. 与上传特征和 DOCX 的一致性

| 来源 | 设计内容 | FE-01B 实现 |
| --- | --- | --- |
| 用户本轮指定 | `item_int_feats_89 = has_match(item_int_feats_9, domain_d_seq_19)` | `0=missing target, 1=no match, 2=has match` |
| 用户本轮指定 | `item_int_feats_90 = bucketize(match_count(...))` | 使用 `0,1,2,4,8` 分桶，输出 0-5 类别 id |
| 用户本轮指定 | `item_dense_feats_91 = log1p(min_match_delta(...))` | 最近匹配事件的 `timestamp - event_time`，`log1p` 后 z-score |
| 用户本轮指定 | `item_dense_feats_92 = log1p(match_count_7d(...))` | 7 天窗口内匹配次数，`log1p` 后 z-score |
| DOCX P017 | `Item_int_feats_89 = has_match(item_int_feats_9, domain_d_seq_19)` | 完全对应，baseline 适配为 embedding id |
| DOCX P018 | `Item_int_feats_90 = match_count(item_int_feats_9, domain_d_seq_19)` | 因 item int 进入 Embedding，按你本轮指定做 bucketize |
| DOCX P019 | `item_dense_feats_91 = log(1 + min_match_delta)` | 完全对应，使用 domain_d timestamp side column |
| DOCX P020 | `item_dense_feats_92 = log(1 + match_count_7d)` | 完全对应，窗口固定 7 天 |

明确不包含：

```text
user_dense_feats_110
item_dense_feats_86
user_dense_feats_111
item_dense_feats_87
user_dense_feats_112
item_dense_feats_88
```

因此 FE-01B 与你上传的新特征清单完全一致，是一个纯 match 模块实验。

## 3. 测试模块

测试模块：

```text
target item attribute <-> domain_d historical sequence match
```

它回答的问题是：

```text
目标 item 的匿名属性 item_int_feats_9 是否在用户 domain_d 历史 domain_d_seq_19 中出现过，
以及出现次数、最近匹配距离、7 天匹配次数是否能提升 CVR 预测？
```

这个模块不使用当前样本 label，也不依赖 eval 侧 purchase history，因此理论上比 purchase frequency 更适合线上评估。

## 4. 代码结构

当前已生成/更新：

```text
build_feature_engineering_dataset.py
tools/build_feature_engineering_dataset.py
run_fe01b.sh
evaluation/FE01/infer.py
trainer.py
experiment_plans/FE01B/README.md
experiment_plans/FE01B/experiment_01b_target_match_plan.zh.md
```

核心代码开关：

```bash
python3 build_feature_engineering_dataset.py \
  --input_dir /path/to/original_dataset \
  --input_schema /path/to/original_dataset/schema.json \
  --output_dir /path/to/fe01b_dataset \
  --feature_set fe01b \
  --match_window_days 7 \
  --match_count_buckets 0,1,2,4,8 \
  --fit_stats_row_group_ratio 0.9
```

`--feature_set fe01b` 会让 schema 只新增：

```json
{
  "item_int": [
    [89, 3, 1],
    [90, 6, 1]
  ],
  "item_dense": [
    [91, 1],
    [92, 1]
  ]
}
```

## 5. 特征生成规则

```text
target = first positive value from item_int_feats_9
history = domain_d_seq_19
history_time = domain_d timestamp side column，默认 domain_d_seq_26
```

编码规则：

```text
item_int_feats_89:
  0 = missing target
  1 = no match
  2 = has match

item_int_feats_90:
  0 = padding / missing
  1 = count == 0
  2 = count == 1
  3 = 2 <= count < 4
  4 = 4 <= count < 8
  5 = count >= 8
```

dense 规则：

```text
item_dense_feats_91 = zscore(log1p(min(timestamp - matched_event_time)))
item_dense_feats_92 = zscore(log1p(count_matched_event_with_delta <= 7 days))
```

所有 dense normalization stats 只用前 90% row groups 拟合。

## 6. NS 分组

FE-01B 新增 `item_int_feats_89/90`，因此 `ns_groups.feature_engineering.json` 会保留：

```json
"I4_target_matching_fields": [89, 90]
```

新增 dense `91/92` 不写入 NS groups，而是通过 item dense token 进入模型。

训练 token 结构沿用 FE-01：

```text
user_ns_tokens = 6
item_ns_tokens = 4
num_queries = 1
```

加入 item dense token 后：

```text
num_ns = 6 + 1(user_dense) + 4 + 1(item_dense) = 12
T = 1 * 4 + 12 = 16
64 % 16 == 0
```

## 7. 训练命令

```bash
bash run_fe01b.sh \
  --data_dir /path/to/original_dataset \
  --schema_path /path/to/original_dataset/schema.json \
  --ckpt_dir outputs/exp_fe01b_target_match/ckpt \
  --log_dir outputs/exp_fe01b_target_match/log \
  --seed 42
```

关键参数：

| 参数 | 值 |
| --- | ---: |
| `--feature_set` | `fe01b` |
| `--match_window_days` | `7` |
| `--match_count_buckets` | `0,1,2,4,8` |
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

该 infer 会读取 checkpoint 的 `schema.json`，发现只需要 `item_int_feats_89/90` 和 `item_dense_feats_91/92`，因此只生成 FE-01B 所需列。它会复用 checkpoint 中的 `feature_engineering_stats.json`，不会在 eval 上重新拟合 normalization。

## 9. 结果预测

基于当前完整 FE-01 的现象：

```text
valid 小涨，但 eval 大掉
```

FE-01B 去掉了线上不可复现风险最大的 purchase frequency，也去掉了 total frequency 的 streaming state 偏移，只保留序列内匹配信号，因此 eval 应显著好于完整 FE-01。如果 DOCX 中观察到的 match lift 在全量数据成立，FE-01B 是更可能保留的模块。

| 指标 | 预测 |
| --- | --- |
| Valid AUC | 相比 B0 `+0.0003 ~ +0.0012` |
| Valid Logloss | 小幅改善或持平 |
| Eval AUC | 预计高于完整 FE-01，范围 `0.805 ~ 0.814` |
| Infer time | 高于 FE-01A，因为需要扫描 domain_d sequence；可能接近完整 FE-01 |

判断标准：

```text
1. 若 FE-01B eval >= B0，说明 target match 模块可保留。
2. 若 FE-01B valid 涨但 eval 仍掉，说明 match lift 可能只在训练切分内成立，或 domain_d_seq_19 与 item_int_feats_9 在 eval 分布变了。
3. 若 FE-01B eval 正常而 FE-01 full 掉，完整 FE-01 的主要负因子基本锁定为 purchase frequency 或 total frequency streaming mismatch。
```
