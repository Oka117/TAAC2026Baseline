# FE01方案-A-总频次特征消融

> 本文是 `experiment_plans/FE01A/experiment_01a_total_frequency_plan.zh.md` 的“强对齐版本”，专门用于回答本轮 PR 中“上传图片所分析的问题”：FE-01 完整集 valid 微涨、eval 大掉，需要把 FE-01 拆成两个最小可证伪的消融实验，并且方案必须与本轮上传的特征清单、`README.feature_engineering.zh.md` 整体设计、以及 `build_feature_engineering_dataset.py` 现有代码逐字段、逐分支保持一致。

## 0. 上传图片所分析问题的复述与本方案定位

| 上传图片中标记的问题 | FE-01A 的回答 |
| --- | --- |
| 完整 FE-01 同时混入了 total / purchase frequency 与 target-history match，valid 涨但 eval 掉，无法判断真凶 | FE-01A 只保留 `user_dense_feats_110` 和 `item_dense_feats_86`，把 frequency 模块单独切出来 |
| `purchase_frequency` 来自历史 `label_type==2`，eval 端是否复现存疑 | FE-01A 明确剔除 `user_dense_feats_111 / item_dense_feats_87`，避免 purchase signal 污染 |
| match 模块依赖 `domain_d_seq_19`，与 frequency 不在一条对齐链路上 | FE-01A 不引入 `item_int_feats_89/90` 与 `item_dense_feats_91/92` |
| dense 字段是否真的进入模型 | 通过 `dataset.py` 的 `item_dense` schema 判断，FE-01A 仍保留 user/item dense token，token 数仍为 12 |
| Normalization 是否会跨越验证窗口造成泄漏 | 固定 `--fit_stats_row_group_ratio 0.9`，与 `train.py` 默认尾 10% 验证集对齐 |

## 1. 实验定位

FE-01A 是 FE-01 的一个**严格消融**：只测试本轮上传方案中的两条 `total frequency` dense 特征，不引入任何 purchase、delay、match 信号。

```text
user_dense_feats_110 = log1p(user_total_frequency_before_timestamp)
item_dense_feats_86  = log1p(item_total_frequency_before_timestamp)
```

它要回答的问题是：

```text
仅使用 (user, item) 在当前样本 timestamp 之前的累计出现次数，
能否在不掉 eval 的前提下解释 FE-01 完整版的有效部分？
```

## 2. 与本轮上传特征 + 设计文档的逐项对齐

| 来源 | 设计字段 | FE-01A 实现位置 | 一致性说明 |
| --- | --- | --- | --- |
| 用户本轮上传 | `user_dense_feats_110 = log1p(user_total_frequency_before_timestamp)` | `build_feature_engineering_dataset.py` 中 `_compute_raw_features` → `np.log1p(user_total)` | fid 与命名严格匹配，`prefix_state.before_update` 保证不含当前样本 |
| 用户本轮上传 | `item_dense_feats_86 = log1p(item_total_frequency_before_timestamp)` | 同函数 → `np.log1p(item_total)` | fid 与命名严格匹配，prefix 仅累加“此前已扫描行” |
| `README.feature_engineering.zh.md §6.2` 推荐的 “Domain-level dense summary token” | 用户/物品历史活跃度作为额外 dense NS token | `build_ns_groups()` 不把 dense fid 写入 int-only NS groups；改为通过 `user_dense_token / item_dense_token` 直接进入 HyFormer | 与“dense 不混入 int NS groups”的代码事实一致 |
| `README.feature_engineering.zh.md §2.6` 时间泄漏检查 | `event_time <= timestamp` | `_compute_raw_features` 内 `np.argsort(timestamps, kind='stable')` 后 `before_update → update` 顺序 | 当前样本不会读到自身贡献 |
| DOCX P005 / P007 | `log(1 + user_total_frequency)`, `log(1 + item_total_frequency)` | `np.log1p(...)` + `_normalize` z-score | `--fit_stats_row_group_ratio 0.9` 保证统计仅来自训练 row group |
| `experiment_01_feature_engineering_plan.zh.md §5` | 增量 schema | `selected_user_dense_adds("fe01a")=[(110,1)]`, `selected_item_dense_adds("fe01a")=[(86,1)]` | schema diff 与计划完全一致 |

明确**不包含**（否则就破坏 FE-01A 的最小性）：

```text
user_dense_feats_111  # purchase frequency
item_dense_feats_87   # purchase frequency
user_dense_feats_112  # FE-02 delay
item_dense_feats_88   # FE-02 delay
item_int_feats_89/90  # FE-01B match
item_dense_feats_91/92 # FE-01B match
```

## 3. 测试模块

```text
prefix total frequency dense features
```

模块语义：

```text
仅依赖 (user_id, item_id, timestamp)，不依赖 label，因此线上 / eval 端的可复现性最强，
是当前 FE-01 拆解中“最难产生 eval drop 的”候选。
```

## 4. 代码与设计的对应表（强相关）

```text
FE-01A 字段                     代码入口                                                     设计依据
user_dense_feats_110     build_feature_engineering_dataset.py:_compute_raw_features  README §6.2 / DOCX P005
item_dense_feats_86      同上                                                          README §6.2 / DOCX P007
prefix 不含当前样本       PrefixState.before_update + update                            README §2.6 时间泄漏
log1p + zscore            _normalize + RunningStats                                    DOCX P002 dense normalization
fit on first 90% rg       fit_stats(... fit_row_groups ...)                            train.py 默认尾 10% valid
schema 增量               selected_user_dense_adds("fe01a"), selected_item_dense_adds 实验计划 §5
NS groups 过滤            filter_ns_groups → 删除 89/90/91/92 等不存在 fid              README §6.5
item dense token 启用     dataset.py item_dense_schema.total_dim>0                      实验计划 §8
```

## 5. 数据生成命令（与 `run_fe01a.sh` 对齐）

```bash
python3 -u build_feature_engineering_dataset.py \
  --input_dir   /path/to/original_dataset \
  --input_schema /path/to/original_dataset/schema.json \
  --output_dir  /path/to/fe01a_dataset \
  --feature_set fe01a \
  --match_window_days 7 \
  --match_count_buckets 0,1,2,4,8 \
  --fit_stats_row_group_ratio 0.9
```

`--feature_set fe01a` 在代码中触发的分支：

```python
selected_user_dense_adds("fe01a", False) == [(110, 1)]
selected_item_dense_adds("fe01a", False) == [(86, 1)]
selected_item_int_adds("fe01a", _)       == []        # FE-01A 不增 int fid
```

输出审计文件：

```text
schema.json
ns_groups.feature_engineering.json
feature_engineering_stats.json     # 仅含 user_dense_feats_110, item_dense_feats_86
docx_alignment.fe01.json
docx_alignment.fe01a.json
```

## 6. NS 分组与 token 计数

FE-01A 不新增 int fid，因此 `filter_ns_groups` 会删掉 `I4_target_matching_fields=[89, 90]`，但 `U1~U7 / I1~I3` 全部保留：

```text
user_ns_tokens = 6
item_ns_tokens = 4
num_queries    = 1
num_ns         = 6 + 1(user_dense) + 4 + 1(item_dense) = 12
T              = 1*4 + 12 = 16
d_model        = 64,  64 % 16 == 0   # rank_mixer_mode=full 整除约束成立
```

## 7. 训练命令（沿用 `run_fe01a.sh`）

```bash
bash run_fe01a.sh \
  --data_dir   /path/to/original_dataset \
  --schema_path /path/to/original_dataset/schema.json \
  --ckpt_dir   outputs/exp_fe01a_total_frequency/ckpt \
  --log_dir    outputs/exp_fe01a_total_frequency/log \
  --seed 42
```

固定参数（来源：`run_fe01a.sh` 实际命令行）：

| 参数 | 值 | 来源/原因 |
| ---: | ---: | --- |
| `--feature_set` | `fe01a` | `build_feature_engineering_dataset.py` 分支 |
| `--ns_tokenizer_type` | `rankmixer` | 与 FE-01 一致，方便消融对照 |
| `--user_ns_tokens` | `6` | 维持 `T=16` |
| `--item_ns_tokens` | `4` | 维持 `T=16` |
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

按 checkpoint 内的 `schema.json` 自动只生成 FE-01A 需要的：

```text
user_dense_feats_110
item_dense_feats_86
```

并复用 checkpoint 内的 `feature_engineering_stats.json`，**不在 eval 上重新拟合 normalization**，避免 valid/eval 统计漂移。

## 9. 结果预测

| 指标 | 预测区间 | 推理依据 |
| --- | --- | --- |
| Valid AUC | `+0.0000 ~ +0.0006` over B0 | 仅保留 frequency 信号，不再触发 purchase / match 复合增益 |
| Valid Logloss | 持平或小幅改善 | dense z-score 后量纲与现有 dense 一致 |
| Eval AUC | `0.807 ~ 0.813`（围绕 B0=0.810525） | 该信号最不依赖 label，理论上 eval 可复现 |
| Infer time | 高于 B0、低于完整 FE-01 | 仅多扫两列 prefix，不需要 domain_d 序列扫描 |

判定规则：

```text
1. FE-01A eval ≥ B0       → total frequency 模块可单独保留进入下一轮
2. FE-01A valid 涨但 eval 掉 → 即便 frequency 模块也存在 streaming-prefix 与 eval 切片不一致
3. FE-01A eval 正常但 FE-01 完整版掉 → 主负因子集中在 purchase frequency 或 match 模块（去 FE-01B 验证）
```

## 10. 一致性 checklist（提交前必看）

- [ ] `feature_engineering_stats.json.dense_feature_names == ["user_dense_feats_110", "item_dense_feats_86"]`
- [ ] `schema.json["user_dense"]` 比原始多且仅多 `[110, 1]`
- [ ] `schema.json["item_dense"]` 比原始多且仅多 `[86, 1]`
- [ ] `schema.json["item_int"]` 不出现 `[89,3,1]`、`[90,6,1]`
- [ ] `train.py` 日志：`num_ns=12, T=16, d_model=64`
- [ ] `dataset.py` 返回的 `item_dense_feats.shape != [B, 0]`
- [ ] `docx_alignment.fe01a.json.not_included` 含 `purchase frequency` 与 `target-history match`
- [ ] eval 输出未生成 `item_int_feats_89/90` 列
