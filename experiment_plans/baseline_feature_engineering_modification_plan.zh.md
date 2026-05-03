# Baseline 特征工程分析与修改方向

本文基于 `/Users/gaogang/Downloads/feature-engineering.docx` 的内容，结合当前 `TAAC2026Baseline` 代码结构，整理 baseline 的可落地修改方案。

## 1. DOCX 内容要点

文档提出的核心方向可以归纳为 6 类：

1. 缺失处理：删除缺失比例超过 75% 的 user int feature；对 int feature 缺失值做填充。
2. 数值归一化：所有 dense numerical feature，包括 sequence side feature，做 normalization。
3. 新增频次类 dense feature：
   - `user_dense_feats_110 = log(1 + user_total_frequency)`
   - `item_dense_feats_86 = log(1 + item_total_frequency)`
   - `user_dense_feats_111 = log(1 + user_purchase_frequency)`
   - `item_dense_feats_87 = log(1 + item_purchase_frequency)`
   - `user_dense_feats_112 = log(1 + user_avg_delay)`
   - `item_dense_feats_88 = log(1 + item_avg_delay)`
4. 新增目标 item 与历史序列匹配特征：
   - `has_match(item_int_feats_9, domain_d_seq_19)`
   - `match_count(item_int_feats_9, domain_d_seq_19)`
   - `min_match_delta`
   - `match_count_7d`
   - 其他 item field 与 sequence field 组合按完整训练集 label lift 排序筛选。
5. 重写 NS 分组：
   - user 分为 profile、behavior stats、context、temporal behavior、interest ids、long-tail sparse、high cardinality 等组。
   - item 分为 identity、category/brand、semantic sparse、target matching fields 等组。
6. 过拟合缓解：
   - Delay-aware weighted loss。
   - Multi-task loss，同时预测 conversion、delay bucket、engagement 等辅助目标。

## 2. 当前 Baseline 状态

当前代码已经具备以下基础能力：

- `dataset.py` 从 parquet 读取 user int、item int、user dense、四个 domain sequence，并生成 time bucket。
- `model.py` 已支持 user dense token 和 item dense token，但 `dataset.py` 目前把 `item_dense_feats` 固定返回为空 tensor。
- `train.py` 通过 `ns_groups.json` 读取 user/item int feature 分组，再传入 NS tokenizer。
- `trainer.py` 支持 BCE 和 Focal Loss，但不支持 delay-aware sample weight，也没有辅助任务 head。
- 当前 `label_time` 没有作为模型输入使用，这是合理的；它很容易成为泄漏源。

因此，DOCX 中的大部分方案不需要立刻改 HyFormer backbone，第一阶段应集中在数据预处理、schema、NS 分组和 loss。

### 2.1 DOCX 设计与 Baseline 适配结论

| DOCX 设计 | Baseline 现状 | 可执行适配 |
| --- | --- | --- |
| 删除缺失率 `>75%` 的 user int feature | `schema.json` 决定读取哪些 fid，`ns_groups.json` 决定哪些 fid 进入 tokenizer | 先做统计和配置化 ablation；删除时必须同步 parquet、schema 和 NS groups |
| int feature 缺失用平均值填充 | baseline 把 `<=0` 映射到 0，作为 missing/padding | 不建议对匿名 id 取平均；保留 0 bucket 或新增 missing bucket |
| dense numerical feature normalization | baseline 只直接读取 `user_dense`；`item_dense` 未读取；sequence side 当前按 int embedding 处理 | 对 user/item dense 和派生 dense 做离线 normalization；不要对 categorical sequence id 做 z-score |
| `user_dense_feats_110/111/112` | user dense token 已支持 | 只需增强 parquet/schema 即可进入 user dense token |
| `item_dense_feats_86/87/88/91/92` | `model.py` 支持 item dense token，但 `dataset.py` 固定返回空 item dense | 必须先补齐 `dataset.py` 的 `item_dense` schema/读取/缓冲逻辑 |
| `item_int_feats_89 = has_match(...)` | item int tokenizer 可直接支持 binary/categorical fid | 适合作为 item int feature，建议 vocab 为 3 |
| `item_int_feats_90 = match_count(...)` | raw count 直接做 embedding 不稳定 | 若保留 item int fid 90，建议先 bucketize；若保留连续值，则更适合改成 dense feature |
| delay-aware weighted loss | trainer 当前 BCE/Focal 都是无 sample weight | 第二阶段改 `trainer.py`，用 `reduction='none'` 后按 delay 加权 |
| multi-task loss | `action_num>1` 只改变输出维度，没有多任务 label/loss | 第三阶段加独立 heads 和多任务 label |

## 3. 需要先修正的风险点

### 3.1 Int feature 不建议用平均值填充

匿名 int feature 大概率是 categorical id。对 id 取平均值没有语义，可能制造不存在的类别。更稳妥的做法是：

- 缺失、`-1`、非法值继续映射到 `0`，作为 padding/missing bucket。
- 如需区分 missing 和 padding，可新增专门 missing bucket，但要同步 schema vocab。
- 只有确认为连续数值的 dense feature 才做均值/中位数填充与标准化。

### 3.2 `label_time` 不能作为线上输入特征

DOCX 中的 `delay = timestamp - label_time` 和 `avg_delay` 需要谨慎：

- 不建议把当前样本的 `label_time` 或由它直接生成的 delay 作为模型输入。
- 可以在训练阶段用 `label_time` 构造 loss weight 或辅助监督标签。
- 如果要构造 user/item 历史平均 delay，必须只使用当前样本 `timestamp` 之前的历史记录，不能包含当前样本或未来样本。

### 3.3 新增 feature id 必须同步 schema

当前 baseline 的 feature 列来自 parquet 和 `schema.json`。如果新增：

- `user_dense_feats_110/111/112`
- `item_dense_feats_86/87/88/91/92`
- `item_int_feats_89/90`

则必须在增强后的 parquet 中真实写入这些列，并同步更新 `schema.json`。否则 `train.py` 或 `dataset.py` 会找不到列或无法建立 feature schema。

### 3.4 Dense feature 不应写进 `ns_groups.json`

当前 `ns_groups.json` 只消费 user/item int feature 的 fid。DOCX 中 item group 把 `86/87/91/92` 混入 item 分组，但这些在文档前文被定义成 dense feature。建议规则如下：

- `item_int_feats_89` 和 bucketized `item_int_feats_90` 可进入 `item_ns_groups`。
- `item_dense_feats_86/87/88/91/92` 由 item dense token 统一投影，不放进 `ns_groups.json`。
- DOCX 中 user group 写入了 `110/111/112`，但这些是 `user_dense_feats_*`，也不应放进当前 baseline 的 `user_ns_groups`。

## 4. 推荐修改方案

### 4.1 新增离线预处理脚本

新增 `build_feature_engineering_dataset.py`，输入原始 parquet 目录和原始 `schema.json`，输出增强后的 parquet 目录、增强后的 `schema.json` 和一份新的 `ns_groups.feature_engineering.json`。

预处理脚本建议完成：

- 统计训练集内 feature 缺失率、覆盖率、unique、label lift。
- 对 dense feature 拟合 train-only normalization stats。
- 按时间排序后计算 user/item prefix frequency。
- 计算 target item 属性与历史序列匹配特征。
- 生成增强 schema，避免手工改 JSON 出错。

推荐使用离线预处理，而不是在 `dataset.py` 的 batch 转换阶段临时计算，原因是频次类特征需要全局按时间前缀统计，在线 batch 内计算容易泄漏或不一致。

### 4.2 缺失处理和归一化

低风险实现：

- user/item int 和 sequence categorical side feature：保留当前 `<=0 -> 0` 逻辑。
- dense feature：使用训练集均值/标准差或分位数裁剪后 z-score。
- 频次、count、delta 类新特征：先 `log1p`，再 z-score。
- DOCX 提到的 sequence side feature normalization 只适用于数值型 side feature；当前 baseline 的 sequence side 多数以 categorical id embedding 进入模型，不应直接 z-score。
- 缺失率超过 75% 的 user int feature：不要直接物理删除，先通过配置做 ablation。

如果要删除高缺失 user int feature，需要同步改：

- 增强 `schema.json` 的 `user_int` 列表。
- `ns_groups.feature_engineering.json`，删除对应 fid。
- 训练日志记录删除列表，保证复现实验。

### 4.3 频次与 delay 历史统计特征

按 DOCX 建议新增以下 dense 特征：

```text
user_dense_feats_110 = log1p(user_total_frequency_before_timestamp)
user_dense_feats_111 = log1p(user_purchase_frequency_before_timestamp)
user_dense_feats_112 = log1p(user_avg_delay_before_timestamp)

item_dense_feats_86 = log1p(item_total_frequency_before_timestamp)
item_dense_feats_87 = log1p(item_purchase_frequency_before_timestamp)
item_dense_feats_88 = log1p(item_avg_delay_before_timestamp)
```

实现要求：

- 所有统计必须是 `event_time < current timestamp` 的 prefix 统计。
- `purchase_frequency` 只在训练数据中可由 label 推导；验证和测试必须使用训练历史或历史窗口内可见统计。
- `avg_delay` 只能用历史已完成转化样本，缺失时填 0 并加一个可选 `has_delay_history` binary 特征。

代码改动：

- `schema.json` 增加 user dense 和 item dense schema。
- `dataset.py` 需要补齐 item dense 读取逻辑；目前 `item_dense_schema` 固定为空，`item_dense_feats` 固定是 `[B, 0]`。
- `model.py` 已有 `item_dense_proj`，只要 `item_dense_dim > 0` 就会自动生成 item dense token。

### 4.4 目标 item 与历史序列匹配特征

DOCX 中给出的第一组组合可以先实现为 MVP：

```text
item_int_feats_89 = has_match(item_int_feats_9, domain_d_seq_19)
item_int_feats_90 = bucketize(match_count(item_int_feats_9, domain_d_seq_19))

item_dense_feats_91 = log1p(min_match_delta(item_int_feats_9, domain_d_seq_19))
item_dense_feats_92 = log1p(match_count_7d(item_int_feats_9, domain_d_seq_19))
```

其中 `item_int_feats_90` 在 DOCX 中写成 `match_count(...)`。结合 baseline 的 Embedding tokenizer，推荐把 count 离散分桶后保留为 item int；如果希望保留连续 count，则应改成 dense feature，而不是直接把未分桶 raw count 当成 categorical id。

实现细节：

- `has_match` 是 binary categorical，可作为 item int feature，vocab size = 3，`0` missing/padding，`1` no match，`2` match。
- `match_count` 若进入 item int，应先分桶；`match_count_7d`、`min_match_delta` 更适合作为 dense feature。
- `min_match_delta` 必须使用 sequence timestamp fid 计算；若某个 domain 的 timestamp 缺失，则该组合只能做 count/has_match，不能做 delta。
- 匹配统计应按完整训练集 label lift 排序，保留 Top-K 组合，避免把所有组合都塞进模型造成噪声。

### 4.5 NS 分组重写

新增 `ns_groups.feature_engineering.json`，不要直接覆盖当前示例文件。建议初版：

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

注意：

- `user_dense_feats_110/111/112` 和 `item_dense_feats_86/87/88/91/92` 不写入该 JSON。
- 如果新增 item dense token，则默认 `num_ns` 会增加 1。当前 `rank_mixer_mode=full` 要求 `d_model % T == 0`，其中 `T = num_queries * 4 + num_ns`。
- 使用默认 `d_model=64` 时，要么调低 `--user_ns_tokens/--item_ns_tokens` 让 `T=16`，要么改 `--rank_mixer_mode ffn_only`，要么把 `d_model` 改成可被新 `T` 整除的值。

推荐第一轮配置：

```bash
--ns_groups_json ns_groups.feature_engineering.json \
--ns_tokenizer_type rankmixer \
--user_ns_tokens 6 \
--item_ns_tokens 4 \
--num_queries 1 \
--rank_mixer_mode full \
--d_model 64
```

这样在存在 user dense token 和 item dense token 时：

```text
num_ns = 6 + 1 + 4 + 1 = 12
T = 1 * 4 + 12 = 16
64 % 16 == 0
```

### 4.6 Delay-aware weighted loss

建议作为第二阶段实验，不要与新特征同时上线。实现方向：

- `dataset.py` 在 training mode 下返回 `delay_seconds` 或 `delay_bucket`，但不传入 `ModelInput`。
- `trainer.py` 将 BCE 改为 `reduction='none'`，按 delay weight 加权后求均值。
- 新增参数：

```text
--delay_weight_mode none|fast_boost|long_discount|bucket
--delay_weight_clip 3.0
```

一个保守公式：

```text
loss = BCEWithLogits(logit, label, reduction='none')
loss = loss * clamp(w(delay), min=0.5, max=3.0)
loss = loss.mean()
```

是否提升需要验证，因为 delay weighting 可能改变 AUC 与 logloss 的取舍。

### 4.7 Multi-task loss

这是第三阶段方案，改动比 weighted loss 更大。当前 `action_num > 1` 只是改变输出维度，但 trainer 没有构造多任务 label，也没有按任务分 loss。

推荐改法：

- `model.py` 保留 shared HyFormer output。
- 新增独立 heads：
  - `conversion_head`
  - `delay_bucket_head`
  - `engagement_head`
- `dataset.py` 返回 `delay_bucket_label`，仅训练正样本或有 label_time 的样本参与 delay loss。
- `trainer.py` 计算：

```text
total_loss = cvr_loss + lambda_delay * delay_loss + lambda_engagement * engagement_loss
```

engagement label 需要先确认官方字段语义；若只能使用 proxy label，必须在实验记录中固定映射规则。

## 5. 文件级修改清单

建议按以下文件推进：

| 文件 | 修改方向 | 优先级 |
| --- | --- | --- |
| `build_feature_engineering_dataset.py` | 新增离线特征生成、normalization stats、增强 schema 和 NS groups | P0 |
| `dataset.py` | 支持 `item_dense` schema 与读取；可选返回 train-only delay label/weight | P0 |
| `ns_groups.feature_engineering.json` | 新增一份实验分组，不覆盖示例 `ns_groups.json` | P0 |
| `train.py` | 增加 feature-engineering 配置、delay loss 参数、校验 NS token 数与 `d_model` | P1 |
| `trainer.py` | 支持 delay-aware weighted BCE；后续支持 multi-task loss | P1 |
| `model.py` | 第一阶段基本不用改；multi-task 阶段再加多个 head | P2 |

## 6. 推荐实验顺序

建议按 ablation 单独推进，避免多个变量混在一起：

```text
B0: 当前 baseline
B1: dense normalization + log1p count/delta 特征
B2: + user/item prefix frequency 和 purchase frequency
B3: + target item 与历史序列匹配特征
B4: + 新 NS 分组与 token 数配置
B5: + delay-aware weighted loss
B6: + conversion/delay/engagement multi-task loss
```

优先级最高的是 B2/B3，因为它们最贴近 DOCX 的核心想法，且不需要改 HyFormer block。

## 7. 验证与防泄漏检查

每个实验都应固定以下检查：

- 所有 prefix frequency 只使用当前样本 timestamp 之前的数据。
- 不把当前样本 `label_time` 或由它直接生成的 delay 放进模型输入。
- normalization stats 只在训练集拟合，验证和测试复用训练 stats。
- 新增 feature id 在 parquet、schema、`ns_groups` 三处一致；dense fid 不写入当前 baseline 的 int NS groups。
- `rank_mixer_mode=full` 下确认 `d_model % T == 0`。
- 同时报 AUC 和 logloss，delay-aware loss 可能提升一个指标但伤害另一个指标。

## 8. 结论

基于 DOCX 内容，最合理的 baseline 修改路线是：

1. 先做离线增强数据集和 schema，不先动 backbone。
2. 补齐 item dense 支持，让频次、delay 历史统计、match count/delta 能以 dense token 进入模型。
3. 把 `has_match` 这类离散匹配信号放入 item int feature，并重写 NS 分组。
4. 在特征收益稳定后，再加入 delay-aware weighted loss。
5. Multi-task 作为后续增强，不建议第一轮与所有特征一起改。

这条路线改动可控，也方便逐步做 AUC ablation。
