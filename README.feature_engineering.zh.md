# GNN + HyFormer 数据分析与特征工程建议

本文基于 `demo_1000.parquet` 的 1000 条样本，对数据结构、序列字段、非序列特征和 GNN + HyFormer 特征工程方向做初步分析。

注意：本文中的统计值来自 demo 数据，适合用于确定分析方向和实验优先级。正式建模前，应在完整训练集上重新计算覆盖率、基数、label lift 和时间分布。

## 1. 分析目标

当前模型结构可以理解为：

```text
非序列特征 -> NS tokens
四个行为域序列 -> sequence tokens
NS tokens + sequence summary -> domain query tokens
query tokens + sequence tokens + NS tokens -> HyFormer blocks
```

如果引入 GNN，建议不要把它作为完全独立模型，而是作为关系增强模块：

```text
高基数字段 / item-like 字段 -> 图节点
低基数字段 / 行为类型字段 -> 边类型或边属性
图 embedding -> 注入 HyFormer 的 token embedding
```

第一阶段建议保持 HyFormer backbone 不变，只增强输入 token，便于做清晰的 ablation。

## 2. 推荐数据分析方向

### 2.1 序列健康度分析

对 `domain_a/b/c/d` 分别统计：

- 序列长度分布；
- 空序列比例；
- 事件时间是否早于样本 `timestamp`；
- 事件时间是否按新到旧排序；
- 最近 1 小时、1 天、7 天、30 天事件数量；
- 每个 side feature 的缺失率、基数和覆盖率。

### 2.2 字段角色识别

对每个序列字段区分：

- timestamp 字段；
- 低基数行为类型字段；
- 中基数类别字段；
- 高基数实体 / item-like 字段；
- 稀疏可选字段。

当前 demo 中可反推出的时间字段是：

```text
domain_a_seq_39
domain_b_seq_67
domain_c_seq_27
domain_d_seq_26
```

这些时间字段应单独用于构造 time bucket，不应作为普通 side feature embedding。

### 2.3 目标 item 与历史行为关系

需要检查当前样本的 `item_id` 或 `item_int_feats_*` 是否出现在历史行为字段里：

```text
has_match(target_item_feature, sequence_field)
match_count
min_match_delta
match_count_1d / 7d / 30d
```

这类特征可以作为额外 dense NS token，也可以作为 query generator 的条件输入。

### 2.4 图构建候选字段分析

重点关注高基数字段：

- 如果字段 coverage 高、unique 多，适合做主要图节点；
- 如果字段 coverage 低但 unique 多，适合做稀疏辅助图节点；
- 如果字段低基数且覆盖率高，适合作为边类型或 relation attribute。

### 2.5 NS 特征分组分析

当前 `ns_groups.json` 是示例分组，不是强制最佳分组。建议按以下信息重新设计：

- 字段类型：user / item / dense / list；
- 基数：低基数 profile、高基数 entity；
- 缺失率；
- 与 label 的 lift；
- 是否与 dense list 对齐。

### 2.6 时间泄漏检查

建模和建图时必须保证：

```text
event_time <= timestamp
```

不要使用 `label_time` 作为输入特征。`label_time` 是标签发生时间或观测时间相关字段，直接使用会引入泄漏。

## 3. Demo 数据主要发现

### 3.1 Label 分布

demo 样本正例率：

```text
12.4%
```

### 3.2 四个 domain 的长度与新近度

```text
domain_a: 平均长度 701，历史事件中位年龄 73.5 天
domain_b: 平均长度 571，历史事件中位年龄 94.2 天
domain_c: 平均长度 449，历史事件中位年龄 275.3 天
domain_d: 平均长度 1100，历史事件中位年龄 12.5 天
```

结论：

- `domain_d` 是最新、最高频的行为域；
- `domain_c` 明显更老，更适合用时间衰减、压缩注意力或图摘要；
- `domain_a/b` 介于二者之间，可保留较短近期窗口并配合图增强。

### 3.3 近期行为比例

7 天内事件占比：

```text
domain_a: 5.4%
domain_b: 7.8%
domain_c: 1.5%
domain_d: 29.5%
```

因此，序列长度配置不应一刀切。`domain_d` 值得保留更长的最近窗口，`domain_c` 则更适合压缩。

## 4. 高基数字段：GNN 节点候选

以下字段适合优先作为图节点：

```text
domain_c_seq_47: 287k unique，覆盖率 100%
domain_b_seq_69: 192k unique，覆盖率 99.8%
domain_c_seq_29: 172k unique，覆盖率 100%
domain_d_seq_23: 123k unique，覆盖率 95.6%
domain_c_seq_36: 55k unique，覆盖率 70.7%
domain_d_seq_22: 17.9k unique，覆盖率 6.7%
domain_a_seq_38: 17.8k unique，覆盖率 5.5%
domain_c_seq_34: 16.9k unique，覆盖率 100%
domain_b_seq_74: 14.3k unique，覆盖率 10.6%
domain_b_seq_88: 12.3k unique，覆盖率 40.6%
```

建议采用 field-aware node，而不是直接合并不同字段中的相同整数 ID：

```text
field=domain_c_seq_47, value=x
field=domain_b_seq_69, value=y
```

原因是匿名 ID 不一定共享同一个 ID 空间。不同字段中相同的整数值可能代表完全不同的实体。

## 5. 低基数字段：边类型候选

以下字段适合作为边类型或边属性：

```text
domain_a: seq_40, seq_41, seq_46
domain_b: seq_68, seq_75
domain_c: seq_28, seq_32, seq_33
domain_d: seq_17, seq_24, seq_25
```

示例图边：

```text
user_id -> domain_c_seq_47:value
edge_attr = {
  domain: c,
  time_bucket,
  seq_28,
  seq_32,
  seq_33
}
```

这样可以把行为域、时间和行为类型都编码进图结构。

## 6. 具体特征工程建议

### 6.1 高基数字段图 embedding

优先增强这些字段：

```text
domain_c_seq_47
domain_b_seq_69
domain_c_seq_29
domain_d_seq_23
domain_c_seq_36
domain_b_seq_88
domain_a_seq_38
```

注入 HyFormer tokenizer 时可以采用：

```text
normal_id_embedding + graph_embedding
```

或：

```text
concat(normal_id_embedding, graph_embedding) -> linear -> d_model
```

第一轮建议保持 HyFormer block 不变，只改 embedding 层。

### 6.2 Domain-level dense summary token

新增一个 dense NS token，包含每个 domain 的聚合统计：

```text
len_a / len_b / len_c / len_d
last_delta_a / last_delta_b / last_delta_c / last_delta_d
count_1h / count_1d / count_7d / count_30d per domain
unique high-card ids per domain
nonzero coverage ratio for sparse fields
```

demo 中部分 activity 特征有正向 lift：

```text
domain_a count_7d 高分位: 15.4% 正例率 vs 低分位 10.4%
domain_c len 高分位: 15.1% vs 11.6%
domain_d len 高分位: 14.0% vs 10.0%
```

### 6.3 目标 item 属性与历史序列匹配特征

demo 中直接 `item_id` 匹配不强，但 item 属性与历史字段匹配有明显 lift：

```text
item_int_feats_9 匹配 domain_d_seq_19:
26.8% 正例率 vs 11.8%

item_int_feats_10 匹配 domain_d_seq_24:
23.2% 正例率 vs 11.4%

item_int_feats_13 匹配 domain_c_seq_31:
35.0% 正例率 vs 11.9%
```

建议构造：

```text
has_match(item_int_feats_9, domain_d_seq_19)
match_count(item_int_feats_9, domain_d_seq_19)
min_match_delta(item_int_feats_9, domain_d_seq_19)
match_count_7d(item_int_feats_9, domain_d_seq_19)
```

其他 item field 与 sequence field 组合也可以按完整训练集 lift 排序筛选。

### 6.4 稀疏高基数字段显式建模

例如 `domain_a_seq_38`：

```text
非 0 覆盖率只有 5.5%
非 0 unique 有 17.8k
```

建议增加：

```text
domain_a_seq_38_present_count
domain_a_seq_38_present_ratio
domain_a_seq_38_recent_present_count
```

GNN 建边时只对 `value > 0` 建节点和边，不要把 `0` 当作图节点。

### 6.5 重新设计 NS 分组

当前 `ns_groups.json` 只是示例。建议改成更贴合 GNN + HyFormer 的分组：

```text
User profile low-card token:
user_int_feats_1,49,50,55,58,59,82,92,94-109

User interest/list token:
user_int_feats_15,60,62-66,80,89-91

User dense embedding token:
user_dense_feats_61,87

User aligned dense-list token:
user_dense_feats_62-66,89-91

Item category/type token:
item_int_feats_5,6,7,8,9,10,12,13

Item high-card/entity token:
item_int_feats_11,16,83,84,85

Item sparse flags token:
item_int_feats_81,83,84,85
```

如果继续使用 `RankMixerNSTokenizer`，这些分组仍然有价值，因为它们影响 embedding 拼接顺序和 chunk 语义。

### 6.6 序列窗口策略

建议不要四个 domain 使用完全相同策略：

```text
domain_d: 保留更长近期窗口
domain_c: 更强时间衰减或图摘要
domain_a/b: 中等窗口 + 高基数字段图增强
```

可尝试：

```text
seq_a: 256
seq_b: 256
seq_c: 256 或图摘要后 128
seq_d: 512 或 768
```

最终配置应以验证集 AUC 和训练成本共同决定。

## 7. 推荐实验顺序

建议按以下 ablation 推进：

```text
A0: 当前 HyFormer baseline
A1: + domain summary dense token
A2: + target-item-attribute history match features
A3: + 高基数字段 graph embeddings
A4: + relation-aware GNN，低基数字段作为边类型
A5: 调整 seq_max_lens，domain_d 更长，domain_c 更压缩
```

优先级最高的是：

```text
A2: 目标 item 属性与历史序列匹配特征
A3: 高基数字段 graph embedding
```

这两个方向最贴合 GNN + HyFormer，而且不需要一开始就大改 backbone。

## 8. 下一步建议

建议在完整训练集上补充以下离线统计脚本：

```text
1. 每个字段 coverage / unique / top values / label lift
2. 每个 domain 的 recency count 和长度分布
3. item feature 与 sequence field 的 row-level match lift
4. 高基数字段共现图的 degree 分布
5. 图 embedding 加入前后的 AUC ablation
```

完成完整训练集统计后，再决定最终：

- 哪些字段进图；
- 哪些字段只做普通 embedding；
- 哪些字段做 dense summary；
- `ns_groups.json` 如何重写；
- 四个 domain 的 `seq_max_lens` 如何配置。
