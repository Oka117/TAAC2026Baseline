# suggestion.pdf 正确性与提升空间分析报告

## 1. 分析结论

这份 suggestion 的总体方向有一定价值：它关注了稀疏样本、时间信息、数值特征尺度、缺失值和模型复杂度，这些都是推荐/CVR 任务中真实会影响效果的问题。

但如果放到当前 TAAC2026Baseline 和比赛任务语境下，文档中有几处需要谨慎修正：

- `delay_feat = timestamp - label_time` 作为输入特征或图边特征存在较高标签泄漏风险，不建议直接使用。
- `int_feats` 缺失值用均值填充不适合当前数据，因为这些 int 特征主要是匿名类别 ID 或多值类别 ID。
- 简单删除低频 user/item 可能会损害冷启动、长尾覆盖和验证/测试分布一致性。
- 用 LightGCN 完全替代 HyFormer 不一定符合比赛“统一序列建模与特征交互”的目标，更稳妥的方式是把 GNN/LightGCN 当作关系增强模块，而不是直接替换主干。

更合理的改造路线是：保留 HyFormer baseline 的序列和非序列 token 建模能力，在输入侧加入经过严格防泄漏处理的时间统计、图 embedding、目标 item 与历史行为匹配特征，以及更细粒度的 dense/list 特征处理。

## 2. 依据与上下文

本报告基于以下本地文件与当前 baseline 设计进行分析：

- `suggestion_summary.md`：对 PDF 内容的整理。
- `README.md`：比赛任务、字段说明和建模目标。
- `README.zh.md`：当前 baseline 源码分析。
- `README.feature_engineering.zh.md`：demo 数据上的特征工程分析。
- `dataset.py`：数据读取、padding、时间桶和缺失处理逻辑。
- `model.py`：PCVRHyFormer、NS tokenizer、sequence tokenizer 和 HyFormer block 实现。

当前 baseline 的核心事实：

- 任务是 pCVR 二分类，评估指标是 AUC。
- 输入包含 user/item/context/cross 非序列多字段特征，以及 4 个 domain 的用户行为序列。
- 代码已经使用 `timestamp - sequence_event_timestamp` 构造序列时间桶。
- `label_time` 当前没有进入模型。
- 缺失或 padding 的离散 ID 统一映射为 `0`。
- 多值离散特征目前通过 embedding 后 mean pooling 处理。
- user dense 特征被拼接后投影为一个 dense NS token。

## 3. 逐条正确性分析

| 原建议 | 正确性判断 | 主要问题 | 建议修正 |
| --- | --- | --- | --- |
| 删除 `frequency(user_id) < 5` 和 `frequency(item_id) < 5` | 部分正确 | 能降噪，但可能删除大量长尾样本，破坏线上/测试分布，并削弱冷启动能力 | 不建议直接硬删；先做频次分桶、低频标记、UNK/冷启动 bucket、样本权重或分组验证 |
| 构造 `delay_feat = timestamp - label_time` | 高风险/不建议直接使用 | `label_time` 与标签发生或观测窗口强相关，作为特征会泄漏；若 conversion 发生在点击后，`timestamp - label_time` 可能为负 | 用 `timestamp - sequence_event_time`、最近行为间隔、统计窗口 count 等替代；不要把 `label_time` 输入模型或图 |
| 删除 `delay_feat < 3` | 需要重新定义后才可评估 | 如果 delay 来自 `label_time`，该过滤可能按标签结果筛样本，改变正负样本分布 | 只对合法的历史事件时间差做异常过滤，如 `event_time > timestamp` 或时间差异常 |
| dense numerical feature 归一化 | 基本正确 | 当前代码没有显式归一化 dense 特征；但必须区分真实数值、padding 和 aligned dense list | 用训练集统计均值/方差或分位数；保留 padding mask；对重尾特征做 clip/log/robust scale |
| sequence side feature 归一化 | 在当前数据上不完全适用 | 当前 4 个 sequence domain 的 side features 主要是 `list<int64>` 类别 ID，timestamp 已单独做 time bucket | 对类别 ID 不做数值归一化；只对真实连续值或时间差统计做归一化 |
| multi-value/list 特征 average pooling | 可作为 baseline，但信息损失较大 | 对多值类别 ID，mean pooling 会丢失元素级别关系；对 aligned int-dense list，直接平均会破坏对齐关系 | 保留当前 embedding mean pooling baseline，同时尝试 attention pooling、target-aware pooling、element-level token |
| 删除缺失比例超过 70% 的 user/item 特征 | 部分正确 | 高缺失不一定无用，稀疏但高 lift 的特征可能很强 | 结合 coverage、unique、label lift、稳定性、训练成本一起判断，不只看缺失率 |
| `int_feats` 缺失值用平均值填充 | 不正确 | 当前 int 特征多数是匿名类别 ID，均值没有语义，会制造不存在的类别 | 缺失保持 `0` padding/UNK，或增加 missing indicator；只对真实连续 dense 特征做均值/中位数填充 |
| 简化为 LightGCN | 部分正确 | LightGCN 擅长协同过滤，但难以完整利用多字段特征、序列时序、上下文和冷启动 | 优先做 GNN/LightGCN embedding 注入 HyFormer，而不是直接替换主干 |
| `delay_feat` 作为 edge contribution | 高风险/需替换 | 如果 edge 用 `label_time` 构造，会泄漏；如果 graph 使用未来交互，也会泄漏 | 用历史序列事件的 recency、行为类型、domain、时间桶作为边属性或边权 |

## 4. 关键风险

### 4.1 标签泄漏风险

`label_time` 是最需要警惕的字段。它与标签生成过程直接相关，通常不是线上预测时可用的因果输入。

因此：

- 不应把 `label_time` 直接作为模型输入。
- 不应构造 `timestamp - label_time` 后作为模型特征。
- 不应把基于 `label_time` 的差值作为图边权重。
- 不应用 `delay_feat < 3` 这类规则在不了解标签语义时过滤训练样本。

更安全的时间信息来源是历史行为序列中的 event timestamp。当前 `dataset.py` 已经用样本 `timestamp` 减去序列事件时间来做 time bucket，这是合理方向。

### 4.2 长尾样本硬删除风险

推荐广告任务天然有长尾用户和长尾物品。直接删除低频 user/item 会带来几个问题：

- 训练分布与验证/测试分布不一致。
- 冷启动能力下降。
- AUC 可能在局部验证集上升，但线上泛化下降。
- 如果频次统计使用了验证集或全量数据，还会产生数据穿越。

更稳妥的方式是把频次作为特征或建模条件，而不是一开始硬删样本。

### 4.3 类别特征均值填充风险

当前 `user_int_feats_*` 和 `item_int_feats_*` 主要是匿名整数类别 ID。类别 ID 的大小没有连续数值意义，用平均值填充会引入伪类别。

当前 baseline 将缺失、非法和 padding 映射到 `0`，并通过 `padding_idx=0` 避免它产生普通类别 embedding。这比均值填充更合理。

### 4.4 过度简化模型结构风险

LightGCN 的优势是用户-物品协同过滤，但 TAAC2026 的任务重点是统一序列建模和非序列多字段特征交互。完全替换为 LightGCN 可能会丢掉：

- 4 个 domain 的异构行为序列信息；
- target item 多字段属性；
- user dense/cross 特征；
- 序列新近度和行为类型；
- 统一 token/block 的创新空间。

因此 LightGCN 更适合作为辅助关系表征，而不是唯一主干。

## 5. 可提升空间

### 5.1 时间特征改造

把原建议中的 `delay_feat` 改为不泄漏的历史时间统计：

```text
last_delta_per_domain = timestamp - last_event_time
count_1h / count_1d / count_7d / count_30d per domain
mean_delta / min_delta / max_delta per domain
recent_unique_item_like_ids
event_time_bucket distribution
```

可落地方式：

- 继续使用当前 sequence time bucket。
- 新增 domain-level dense summary token。
- 对异常样本只检查 `event_time > timestamp`、负时间差、过大时间差等明确无效情况。

### 5.2 低频用户/物品处理

替代硬删除的方案：

```text
user_freq_bucket
item_freq_bucket
is_rare_user
is_rare_item
rare_user/item shared embedding
frequency-aware dropout
sample weighting
```

实验时要注意：

- 频次只能在训练集内部统计。
- 验证集和测试集不能参与频次统计。
- 如果做 K-fold 或时间切分，频次统计也要跟随 fold 或时间边界。

### 5.3 缺失值处理

建议改为按特征类型处理：

- 类别 ID：缺失保持 `0` 或映射到专门的 missing/UNK bucket。
- 多值类别 list：保留空 list mask、list 长度、非零比例等统计。
- dense scalar/list：用训练集均值/中位数填充，并增加 missing indicator。
- 高缺失但高 lift 特征：保留或单独建 sparse token，不要仅按 70% 阈值删除。

### 5.4 多值与 aligned int-dense 特征

当前 baseline 已对多值离散特征做 embedding mean pooling，但还没有充分利用 aligned int-dense 关系。

更好的方向：

```text
id_embedding(element_id) + dense_value_projection(element_value) + field_embedding
-> element-level token
-> attention/target-aware pooling
```

这比简单 average pooling 更能保留“哪个 ID 对应哪个数值”的信息。

### 5.5 GNN/LightGCN 的合理接入方式

建议采用“图增强 HyFormer”，而不是“LightGCN 替代 HyFormer”：

```text
高基数字段 / item-like 字段 -> 图节点
用户、目标 item、历史行为实体 -> 图边
domain / 行为类型 / recency bucket -> 边类型或边属性
图 embedding -> 注入 NS token 或 sequence token
HyFormer -> 最终 pCVR 预测
```

第一阶段可以先做静态图 embedding：

```text
normal_id_embedding + graph_embedding
```

或：

```text
concat(normal_id_embedding, graph_embedding) -> linear -> d_model
```

这样改动小，方便做 ablation。

### 5.6 模型侧优先改进点

结合当前 baseline，优先级较高的模型改进包括：

1. 让 final head 同时使用最终 query tokens 和 NS tokens，而不是只用 query tokens。
2. 增加 domain summary dense token。
3. 增加 target item 属性与历史序列字段的匹配特征。
4. 对 aligned int-dense list 做 element-level 建模。
5. 试验 `longer` encoder、domain-specific `seq_max_lens` 和时间衰减。
6. 在稳定 baseline 之后再接入 GNN/LightGCN embedding。

## 6. 推荐实验路线

建议按风险从低到高推进：

| 阶段 | 实验 | 预期收益 | 风险 |
| --- | --- | --- | --- |
| A0 | 复现当前 baseline | 建立可比较基线 | 低 |
| A1 | dense 特征训练集归一化 + missing indicator | 改善数值尺度 | 低 |
| A2 | domain summary dense token | 强化时间/活跃度统计 | 低 |
| A3 | target item 与历史序列匹配特征 | 捕捉强相关交互 | 中 |
| A4 | final head 使用 Q + NS tokens | 强化非序列强特征直连 | 中 |
| A5 | aligned int-dense element-level token | 保留对齐关系 | 中 |
| A6 | 低频 user/item 分桶或 shared rare embedding | 提升长尾鲁棒性 | 中 |
| A7 | graph embedding 注入 HyFormer | 加强高基数字段关系建模 | 中高 |
| A8 | LightGCN/异构 GNN 与 HyFormer 融合 | 捕捉协同过滤结构 | 高 |

不建议优先做：

- 直接使用 `label_time` 构造输入特征。
- 直接删除所有低频 user/item。
- 对类别 int 特征均值填充。
- 完全用 LightGCN 替换 HyFormer 后再从零调参。

## 7. 建议的验证方式

每个改动都应至少记录：

```text
AUC
logloss
正负样本比例变化
训练样本保留比例
冷启动/低频用户 AUC
低频 item AUC
高频 item AUC
训练吞吐
显存占用
```

涉及过滤或图构建时，需要额外检查：

- 是否使用了验证/测试信息。
- 是否使用了 `label_time` 或未来行为。
- 是否改变了验证集分布。
- 是否在低频/冷启动样本上显著退化。

## 8. 最终判断

这份 suggestion 可以作为“数据清洗和简化建模”的粗粒度提醒，但不能原样落地。它最有价值的部分是提醒关注：

- 低频用户/物品；
- 时间间隔；
- dense 特征尺度；
- 多值特征聚合；
- 缺失率；
- 图结构关系。

最需要修正的部分是：

- 把 `label_time` 相关的 `delay_feat` 改成不泄漏的历史行为时间特征。
- 把 int 类别特征均值填充改成 padding/UNK/missing indicator。
- 把“直接删除低频样本”改成频次分桶、低频共享表示或样本权重。
- 把“LightGCN 替代 HyFormer”改成“图 embedding 增强 HyFormer”。

建议下一步先实现低风险的 A1-A4，再根据验证结果决定是否进入 GNN/LightGCN 融合实验。

