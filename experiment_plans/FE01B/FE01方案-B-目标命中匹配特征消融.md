# FE01方案-B-目标命中匹配特征消融

> 本文是 `experiment_plans/FE01B/experiment_01b_target_match_plan.zh.md` 的“强对齐版本”，专门用于回答本轮 PR 中“上传图片所分析的问题”：FE-01 完整集 valid 微涨、eval 大掉，需要把 FE-01 拆成两个最小可证伪的消融实验，并且方案必须与本轮上传的特征清单、`README.feature_engineering.zh.md` 整体设计、以及 `build_feature_engineering_dataset.py` 现有代码逐字段、逐分支保持一致。

## 0. 上传图片所分析问题的复述与本方案定位

| 上传图片中标记的问题 | FE-01B 的回答 |
| --- | --- |
| 完整 FE-01 把 frequency 与 match 同时打开，eval 不可解释 | FE-01B 只保留 `item_int_feats_89/90` 与 `item_dense_feats_91/92`，把 match 模块单独切出来 |
| `match_count` 原文是连续值，但 baseline 的 `item_int` 走 Embedding 不能直接喂浮点 | 按本轮上传指定，分桶为 `0,1,2,4,8` 边界、共 `0..5` 共 6 个 id，由 schema `[90, 6, 1]` 落地 |
| `min_match_delta` / `match_count_7d` 需要 domain_d 时间字段 | `resolve_domain_d_columns` 自动从 schema 找 `domain_d_seq_19` 与 `domain_d_seq_<ts_fid>`（默认 26）|
| 是否会因为 `target = 0` 误判为有匹配 | `_first_scalar_from_maybe_list` 返回首个 `>0` 值；`target<=0` 时 `has_match=0`（missing）|
| `match_count_7d` 是否使用未来事件 | 7 天窗口取自 `timestamp - event_time`，仅当 `event_time<=timestamp` 进入计数 |
| dense 是否走 normalization 且不跨切片 | `_normalize` 使用前 90% row group 拟合的 z-score，与 valid_ratio=0.1 对齐 |

## 1. 实验定位

FE-01B 是 FE-01 的另一个**严格消融**：只测试本轮上传方案中“目标 item 属性 vs domain_d 历史”的 4 条 match 特征：

```text
item_int_feats_89  = has_match(item_int_feats_9, domain_d_seq_19)
item_int_feats_90  = bucketize(match_count(item_int_feats_9, domain_d_seq_19))
item_dense_feats_91 = log1p(min_match_delta(item_int_feats_9, domain_d_seq_19))
item_dense_feats_92 = log1p(match_count_7d(item_int_feats_9, domain_d_seq_19))
```

不引入 `frequency`、`purchase`、`delay`，不修改 loss、不动 backbone。它要回答：

```text
仅依靠目标 item 的匿名属性 item_int_feats_9 在 user 的 domain_d 历史 domain_d_seq_19
中是否出现 / 出现次数 / 最近匹配距离 / 7 天匹配次数，能否稳定提升 CVR？
```

## 2. 与本轮上传特征 + 设计文档的逐项对齐

| 来源 | 设计字段 | FE-01B 实现位置 | 一致性说明 |
| --- | --- | --- | --- |
| 用户本轮上传 | `item_int_feats_89 = has_match(item_int_feats_9, domain_d_seq_19)` | `_compute_raw_features` 中 `has_match[i] = 2 if match_count>0 else 1`；`target<=0 → 0` | 编码 `0=missing, 1=no match, 2=has match` |
| 用户本轮上传 | `item_int_feats_90 = bucketize(match_count(...))`，桶边界 `0,1,2,4,8` | `_bucketize_counts(match_count, count_edges)`；schema `[90, 6, 1]` | 6 个类别 id：0=padding/missing, 1=count==0, 2=1, 3=2-3, 4=4-7, 5=≥8 |
| 用户本轮上传 | `item_dense_feats_91 = log1p(min_match_delta(...))` | `min_match_delta[i] = min(deltas)`；`np.log1p` + z-score | `delta = max(timestamp - event_time, 0)`，与 README §2.6 防泄漏一致 |
| 用户本轮上传 | `item_dense_feats_92 = log1p(match_count_7d(...))` | `count_7d` 在循环内累计；`np.log1p` + z-score | 窗口由 `--match_window_days 7` → `match_window_seconds=604800` |
| `README.feature_engineering.zh.md §2.3` 目标 item 与历史行为关系 | `has_match / match_count / min_match_delta / match_count_7d` | 完整 4 条全部覆盖 | 字面映射 |
| `README.feature_engineering.zh.md §6.3` lift 实证 | `item_int_feats_9 ↔ domain_d_seq_19` 26.8% vs 11.8% | 仅实现 README 实证显著的 pair，不引入 §6.3 中其他候选 | 与“DOCX 第一组”严格对齐 |
| DOCX P017–P020 | `Item_int_feats_89/90`, `item_dense_feats_91/92` | `selected_item_int_adds("fe01b")=[(89,3,1),(90,6,1)]`、`selected_item_dense_adds("fe01b")=[(91,1),(92,1)]` | schema diff 与计划完全一致 |

明确**不包含**（保持 FE-01B 的最小性）：

```text
user_dense_feats_110/111   # FE-01A frequency
item_dense_feats_86/87     # FE-01A frequency
user_dense_feats_112       # FE-02 delay
item_dense_feats_88        # FE-02 delay
```

## 3. 测试模块

```text
target item attribute  ↔  domain_d historical sequence  match
```

模块语义：

```text
不依赖当前样本 label，也不依赖历史 purchase label；
唯一外部依赖是 domain_d 序列的 timestamp 字段，由 resolve_domain_d_columns 自动解析。
```

## 4. 代码与设计的对应表（强相关）

```text
FE-01B 字段                  代码入口                                                       设计依据
item_int_feats_89    _compute_raw_features:has_match (0/1/2)                          README §2.3 / DOCX P017
item_int_feats_90    _bucketize_counts(match_count, [0,1,2,4,8])                      README §6.3 / DOCX P018 (bucket adapter)
item_dense_feats_91  np.log1p(min(timestamp - event_time))  + z-score                 README §2.6 / DOCX P019
item_dense_feats_92  np.log1p(count(delta <= 7d))           + z-score                 README §6.3 / DOCX P020
domain_d 列解析       resolve_domain_d_columns(schema, parquet_names)                  README §2.2 时间字段独立处理
target<=0 处理        _first_scalar_from_maybe_list → has_match=0                     README §6.4 稀疏字段显式建模
NS 分组保留 89/90    filter_ns_groups → I4_target_matching_fields=[89,90]            README §6.5 NS 分组建议
```

## 5. 数据生成命令（与 `run_fe01b.sh` 对齐）

```bash
python3 -u build_feature_engineering_dataset.py \
  --input_dir   /path/to/original_dataset \
  --input_schema /path/to/original_dataset/schema.json \
  --output_dir  /path/to/fe01b_dataset \
  --feature_set fe01b \
  --match_window_days 7 \
  --match_count_buckets 0,1,2,4,8 \
  --fit_stats_row_group_ratio 0.9
```

`--feature_set fe01b` 在代码中触发的分支：

```python
selected_user_dense_adds("fe01b", False) == []
selected_item_dense_adds("fe01b", False) == [(91, 1), (92, 1)]
selected_item_int_adds("fe01b", 6)       == [(89, 3, 1), (90, 6, 1)]
```

输出审计文件：

```text
schema.json
ns_groups.feature_engineering.json     # 仅保留 I4_target_matching_fields=[89,90]
feature_engineering_stats.json         # dense 仅含 91, 92
docx_alignment.fe01.json
docx_alignment.fe01b.json              # not_included 含 frequency / delay
```

## 6. NS 分组与 token 计数

FE-01B 新增 int fid `89/90`，`filter_ns_groups` 会保留：

```json
"item_ns_groups": {
  "I1_item_identity": [5, 6, 7, 8],
  "I2_item_category_brand": [9, 10, 11, 12, 13],
  "I3_item_semantic_sparse": [16, 81, 83, 84, 85],
  "I4_target_matching_fields": [89, 90]
}
```

dense `91/92` 不写入 NS groups，通过 `item_dense_token` 进入模型。token 计数：

```text
user_ns_tokens = 6
item_ns_tokens = 4
num_queries    = 1
num_ns         = 6 + 1(user_dense) + 4 + 1(item_dense) = 12
T              = 1*4 + 12 = 16
d_model        = 64,  64 % 16 == 0   # rank_mixer_mode=full 整除约束成立
```

注意：FE-01B 没有新增 user dense fid，但只要原 schema 中 `user_dense` 非空，user dense token 仍存在；token 数无需重新调整。

## 7. 训练命令（沿用 `run_fe01b.sh`）

```bash
bash run_fe01b.sh \
  --data_dir   /path/to/original_dataset \
  --schema_path /path/to/original_dataset/schema.json \
  --ckpt_dir   outputs/exp_fe01b_target_match/ckpt \
  --log_dir    outputs/exp_fe01b_target_match/log \
  --seed 42
```

固定参数（来源：`run_fe01b.sh` 实际命令行）：

| 参数 | 值 | 来源/原因 |
| ---: | ---: | --- |
| `--feature_set` | `fe01b` | `build_feature_engineering_dataset.py` 分支 |
| `--match_window_days` | `7` | DOCX P020 7 天窗口 |
| `--match_count_buckets` | `0,1,2,4,8` | 本轮上传指定，对应 `[90, 6, 1]` |
| `--ns_tokenizer_type` | `rankmixer` | 与 FE-01 一致 |
| `--user_ns_tokens` | `6` | 维持 `T=16` |
| `--item_ns_tokens` | `4` | 保留 `I4_target_matching_fields` 独立语义 |
| `--num_queries` | `1` | 满足 `d_model=64` 整除 |
| `--loss_type` | `bce` | 与 FE-01 一致，不混入 loss 变量 |
| `--patience` | `3` | 与 FE-00 / FE-01 对齐 |
| `--num_epochs` | `6` | 与 FE-00 / FE-01 对齐 |
| `--seq_max_lens` | `seq_a:256,seq_b:256,seq_c:512,seq_d:512` | 与 FE-01 一致 |
| `--fit_stats_row_group_ratio` | `0.9` | 与 `valid_ratio=0.1` 严格对齐 |

## 8. 评估设计

继续复用：

```text
evaluation/FE01/infer.py
```

按 checkpoint 内的 `schema.json` 自动只生成 FE-01B 需要的：

```text
item_int_feats_89
item_int_feats_90
item_dense_feats_91
item_dense_feats_92
```

并复用 checkpoint 内的 `feature_engineering_stats.json`，**不在 eval 上重新拟合 normalization**，避免 valid/eval 统计漂移。

`resolve_domain_d_columns` 在 eval 端会再跑一次，自动找到与训练相同的 `domain_d_seq_19 / domain_d_seq_26`，不依赖训练侧硬编码。

## 9. 结果预测

| 指标 | 预测区间 | 推理依据 |
| --- | --- | --- |
| Valid AUC | `+0.0003 ~ +0.0012` over B0 | README §6.3 实证 26.8% vs 11.8% lift，预计 valid 端可保留 |
| Valid Logloss | 小幅改善或持平 | embedding + dense 两路都对 CVR 单调相关 |
| Eval AUC | `0.805 ~ 0.814`，预期高于完整 FE-01 | 已剔除 purchase / streaming-prefix 不一致风险 |
| Infer time | 高于 FE-01A、可能接近完整 FE-01 | 需要扫 `domain_d_seq_19` + 时间列，扫描成本与 domain_d 长度相关 |

判定规则：

```text
1. FE-01B eval ≥ B0       → match 模块可单独保留进入下一轮
2. FE-01B valid 涨但 eval 仍掉 → match lift 可能只在训练切片内成立，需要换 pair（README §6.3 其他候选）
3. FE-01B eval 正常而完整 FE-01 掉 → 主负因子锁定为 frequency 模块（与 FE-01A 联合验证）
```

## 10. 一致性 checklist（提交前必看）

- [ ] `feature_engineering_stats.json.dense_feature_names == ["item_dense_feats_91", "item_dense_feats_92"]`
- [ ] `feature_engineering_stats.json.item_int_feature_names == ["item_int_feats_89", "item_int_feats_90"]`
- [ ] `schema.json["item_int"]` 比原始多且仅多 `[89, 3, 1]` 与 `[90, 6, 1]`
- [ ] `schema.json["item_dense"]` 比原始多且仅多 `[91, 1]` 与 `[92, 1]`
- [ ] `schema.json["user_dense"]` 与原始一致（不含 110/111/112）
- [ ] `ns_groups.feature_engineering.json` 包含 `I4_target_matching_fields=[89, 90]`
- [ ] `feature_engineering_stats.json.match_col` 等于 `domain_d_seq_19`
- [ ] `feature_engineering_stats.json.match_ts_col` 形如 `domain_d_seq_<ts_fid>`（默认 `26`）
- [ ] `feature_engineering_stats.json.match_count_buckets == [0, 1, 2, 4, 8]`
- [ ] `feature_engineering_stats.json.match_window_days == 7`
- [ ] `train.py` 日志：`num_ns=12, T=16, d_model=64`
- [ ] eval 输出未生成 `user_dense_feats_110/111` 列
