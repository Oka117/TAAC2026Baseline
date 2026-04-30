# suggestion.pdf 正确性与提升空间分析报告（修正版）

## 0. 核心修正

本报告按以下架构前提重新分析：

```text
前端：
  NS raw feature / user features / item features
  + Delay_feat edge contribution
  + LightGCN 或 item interaction 表征增强
  -> NS Token

序列侧：
  Sequence feature
  -> Longer
  -> Sequence token

后端：
  NS Token + Sequence token
  -> Query Generation Module
  -> Global token
  -> HyFormer
```

也就是说，`Light GCN` 不是替换 HyFormer，而是给前面的非序列特征、user/item embedding 或 item interaction 增加关系增强信号。后面的 `Longer`、`Query Generation Module`、`Global token`、`HyFormer` 仍然保留。

这是和上一版分析最重要的差异。

## 1. 强制核对清单

为避免漏读或误读，本次按 PDF 的全部文本逐项核对：

| PDF 内容 | 是否纳入本报告 | 本报告中的处理 |
| --- | --- | --- |
| `filtering rare item and user` | 已纳入 | 分析低频 user/item 过滤的正确性、风险和替代方案 |
| `Delete frequency(user_id)<5 frequency(item_id)<5` | 已纳入 | 分析阈值 5、硬删除、训练集内统计和冷启动风险 |
| `get new feature` | 已纳入 | 分析新增 `Delay_feat` 的用途 |
| `Delay_feat=timestamp - label_time` | 已纳入 | 重点分析 `label_time` 语义和泄漏风险 |
| `Filter noise data` | 已纳入 | 分析基于 delay 的样本过滤是否合理 |
| `Delete delay_feat<3` | 已纳入 | 分析阈值、单位、负值和分布风险 |
| `all dense numerical feature ... do normalization` | 已纳入 | 分析 dense 归一化的正确性和落地方法 |
| `including sequence side feature` | 已纳入 | 区分真实 dense side feature 与类别 ID side feature |
| `Multi value features(list) ... do average pooling` | 已纳入 | 分析 list pooling 的正确性、信息损失和更优方案 |
| `handle missing value` | 已纳入 | 分析缺失处理策略 |
| `Delete missing proportion (item_features)>70%` | 已纳入 | 分析 item 特征删除策略 |
| `Delete missing proportion (user_features)>70%` | 已纳入 | 分析 user 特征删除策略 |
| `Replace Missing value in int_feats using average value` | 已纳入 | 判断在当前匿名类别 ID 场景下不合适 |
| `Simplify the structure` | 已纳入 | 按“前端结构简化/增强，HyFormer 保留”理解 |
| `NS raw feature` | 已纳入 | 归入非序列 raw feature 输入 |
| `Light GCN` | 已纳入 | 归入前端 item interaction / graph enhancement |
| `Sequence feature` | 已纳入 | 归入序列输入路径 |
| `Every row` | 已纳入 | 分析每行样本作为 edge/interaction 的可能含义 |
| `User features` | 已纳入 | 归入 user side node/feature |
| `Item features` | 已纳入 | 归入 item side node/feature |
| `User Embedding` | 已纳入 | 分析 user embedding 与 graph embedding 融合 |
| `ItemEmbedding` | 已纳入 | 分析 item embedding 与 graph embedding 融合 |
| `Delay_feat as edge contributes` | 已纳入 | 分析 delay 作为边权/边属性/贡献项 |
| `Longer` | 已纳入 | 明确 sequence feature 先过 Longer |
| `NS Token` | 已纳入 | 明确前端增强后仍产出 NS token |
| `Query Genration Module` | 已纳入 | 按 Query Generation Module 分析 |
| `Sequence token` | 已纳入 | 明确 Longer 输出 sequence token |
| `Global token` | 已纳入 | 保留在后端 HyFormer 前 |
| `Hyformer` | 已纳入 | 明确仍然是后端主干 |

结论：PDF 中出现的全部实质内容都已纳入分析，没有再把 `Light GCN` 解释为替代 HyFormer。

## 2. 总体正确性判断

这份 suggestion 的方向是合理的，尤其是放在“HyFormer 前端特征增强”的语境下：

- 低频 user/item 处理有助于稳定 embedding 和图结构。
- `Delay_feat` 如果语义安全，可以作为 user-item edge 的时间贡献或边权。
- dense 特征归一化是必要的基础工程。
- 多值 list 特征 pooling 是常见 baseline。
- 高缺失特征筛选可以降低噪声和模型复杂度。
- LightGCN/item interaction 作为前端增强模块，与 HyFormer 后端是兼容的。
- Sequence feature 先经过 `Longer`，再作为 `Sequence token` 进入 HyFormer，是当前 baseline 可支持的改造方向。

但它仍然有几个必须强检查的点：

- `label_time` 是否能用于特征或图边，必须先确认语义和预测时可用性。
- `int_feats` 是否真的是连续整数数值，还是匿名类别 ID；在当前 TAAC 数据说明中，更接近类别 ID。
- 删除低频样本和删除高缺失特征会改变训练分布，不能只看局部训练稳定性。
- average pooling 可以作为 baseline，但对 aligned int-dense list 和 target-aware interaction 来说信息损失明显。
- LightGCN 的边、节点、权重、时间边界和冷启动 fallback 需要设计清楚，否则容易泄漏或训练/验证不一致。

## 3. 逐条正确性分析

### 3.1 低频 user/item 过滤

原建议：

```text
Delete frequency(user_id)<5
Delete frequency(item_id)<5
```

正确的部分：

- 对 LightGCN 或 item interaction 特征来说，低频 user/item 的邻居太少，图传播信号弱，容易产生不稳定 embedding。
- 对 embedding 学习来说，极低频 ID 的参数更新次数少，容易过拟合或随机性较强。
- 用频次阈值清理图节点，确实可以让图结构更干净。

风险：

- 推荐广告数据天然长尾，硬删除低频 user/item 可能删掉大量真实线上分布。
- AUC 评估通常包含长尾样本，训练时删掉长尾可能导致泛化下降。
- 如果频次统计用到了验证集、测试集或全量数据，会产生数据穿越。
- 如果删除的是样本而不是图节点，正负样本比例也可能被改变。

更推荐的做法：

```text
user_freq
item_freq
user_freq_bucket
item_freq_bucket
is_rare_user
is_rare_item
rare_user_shared_embedding
rare_item_shared_embedding
```

对 LightGCN 图：

- 训练图中可以过滤极低频边或节点，但模型输入仍要保留这些样本。
- 低频或未见 user/item 应提供 fallback embedding。
- 频次统计必须只在训练集内部完成。

判断：方向正确，但“直接删除”过于激进。更适合改成低频分桶、低频共享表示或仅在图构建阶段过滤。

### 3.2 `Delay_feat = timestamp - label_time`

原建议：

```text
Delay_feat=timestamp - label_time
```

第 2 页进一步写到：

```text
Delay_feat as edge contributes
```

正确的部分：

- 把时间差作为 edge contribution 是有意义的。推荐系统中的边不应该只表示“是否交互”，还可以包含时间新近度、行为间隔、边强度或可靠性。
- 如果 `Delay_feat` 是可在预测时合法获得的时间差，它可以帮助 LightGCN/item interaction 模块区分不同时间关系的 user-item 边。

最大风险：

- `label_time` 通常与标签生成、转化发生时间或观测窗口有关。
- 如果 `label_time` 反映 conversion 或 label 观测结果，使用它构造特征会造成标签泄漏。
- 即使 `Delay_feat` 不直接输入 HyFormer，只要它影响 LightGCN 边权或 graph embedding，泄漏信号仍会通过 graph embedding 传播进 NS token。

必须检查：

```text
1. label_time 在测试/线上预测时是否可用？
2. label_time 是否发生在 timestamp 之后？
3. label_time 是否与 label_type 直接相关？
4. 正负样本的 timestamp - label_time 分布是否显著不同？
5. 用 label_time 构图时，是否把验证/测试标签信息传播到了训练图？
```

更安全的替代：

```text
safe_delay = timestamp - historical_event_timestamp
last_delta_per_domain = timestamp - last_event_time
count_1h / count_1d / count_7d / count_30d
time_bucket(edge_event_time)
edge_recency_weight = f(timestamp - event_time)
```

判断：作为“时间边贡献”的思想正确，但用 `label_time` 计算存在高风险。只有在确认 `label_time` 是预测时合法可用、且不携带标签结果信息时，才能保留。

### 3.3 删除 `delay_feat < 3`

原建议：

```text
Filter noise data
Delete delay_feat<3
```

正确的部分：

- 过滤异常时间差是合理的。
- 如果时间差过小代表脏数据、非法边或无效曝光，删除可以减少噪声。

风险：

- 原文没有说明时间单位，`3` 可能是秒、分钟、小时或天。
- 如果 `timestamp - label_time` 对正负样本有不同分布，过滤会改变 label 分布。
- 如果 `timestamp < label_time` 是正常业务现象，那么 `delay_feat` 为负并不一定是噪声，而可能说明公式方向写反了。
- 如果这一步在构图前执行，会改变 graph degree 分布。

更推荐的做法：

- 先画出 `delay_feat` 的整体分布、按 label 分布、按 user/item 频次分布。
- 用分位数、业务规则和验证集 ablation 决定阈值。
- 对边权可以用 clip 或 bucket，而不是直接删除样本：

```text
delay_bucket = bucketize(delay_feat)
edge_weight = log1p(max(delay_feat, 0))
edge_weight = exp(-delta / tau)
```

判断：过滤噪声的意图正确，但阈值 3 和 `label_time` 公式都需要强验证。

### 3.4 dense numerical feature normalization

原建议：

```text
all dense numerical feature (including sequence side feature) do normalization
```

正确的部分：

- dense 数值特征进入 Linear、LayerNorm、token projection 或图边属性前，归一化通常必要。
- 当前 baseline 中 user dense 特征被拼接后投影为一个 dense NS token，若原始尺度差异较大，会影响训练稳定性。

当前数据/代码中的注意点：

- `README.md` 说明 user dense features 是 `list<float>`。
- 当前 baseline 主要处理 `user_dense_feats_*`，item dense 在代码中为空接口。
- 当前 4 个 sequence domain 的 side features 在文档中是 `list<int64>`，更像类别 ID；timestamp 单独用于 time bucket。

因此：

- 对真实 float dense feature 应做归一化。
- 对匿名类别 ID 不应做数值归一化。
- 对 sequence side feature，只有它确实是连续值时才归一化。

更推荐的落地方式：

```text
训练集统计 mean/std 或 median/IQR
缺失值单独 mask
重尾特征先 clip 或 log1p
归一化参数保存到 checkpoint/config
验证/测试只使用训练集统计量
```

判断：对 dense 数值特征正确；对 sequence side feature 需要先确认字段类型。

### 3.5 multi-value/list average pooling

原建议：

```text
Multi value features(list) ( including sequence side feature) do average pooling
```

正确的部分：

- list 特征必须转成固定维度表示，average pooling 是简单稳定的 baseline。
- 当前 baseline 对多值离散特征已经采用 embedding 后 mean pooling。
- 对图增强后的多邻居 item interaction，也可以先用 mean pooling 做第一版。

问题：

- 直接平均类别 ID 没有语义；应该先 embedding 再 pooling。
- 平均池化丢失元素权重、顺序和 target-aware 关系。
- 对 aligned int-dense list，平均会破坏 ID 与 dense value 的一一对应。
- 对 sequence token，过早 pooling 可能丢掉时序模式。

更强的替代：

```text
embedding mean pooling
attention pooling
target-aware attention pooling
DIN-style activation unit
weighted pooling by recency / delay / frequency
element-level token + HyFormer
aligned id_embedding + dense_value_projection
```

判断：作为 baseline 正确，但不是最佳。尤其当前任务强调统一 token 和序列/特征交互，保留元素级信息会更有潜力。

### 3.6 缺失值处理

原建议：

```text
Delete missing proportion (item_features)>70%
Delete missing proportion (user_features)>70%
Replace Missing value in int_feats using average value
```

正确的部分：

- 删除极高缺失率特征可以降低噪声和计算开销。
- 对真实连续数值特征，用均值或中位数填充是常见 baseline。

问题：

- 高缺失不等于无用。有些稀疏特征覆盖率低，但在出现时 label lift 很高。
- 当前 TAAC 的 `user_int_feats_*` 和 `item_int_feats_*` 多数是匿名类别 ID，均值没有类别语义。
- 对类别 ID 均值填充会制造不存在的 ID，导致 embedding 表示混乱。
- 删除字段会影响 schema、embedding table、NS grouping 和模型输入维度。

更推荐的做法：

```text
类别 ID 缺失 -> 0 / padding / UNK / missing bucket
dense 缺失 -> train mean / median + missing indicator
list 缺失 -> 空 list + length feature + present ratio
高缺失特征 -> coverage + unique + label lift + 稳定性综合判断
```

判断：删除高缺失特征“可以作为候选实验”；`int_feats` 用平均值填补在当前数据语义下不建议。

### 3.7 LightGCN / item interaction 前端增强

第 2 页结构：

```text
NS raw feature -> Every row -> User features / Item features
User Embedding / Item Embedding
Delay_feat as edge contributes
Light GCN
NS Token
```

修正后的理解：

- 每一行样本可以形成 user-item interaction。
- `User features` 与 `Item features` 生成 user/item embedding。
- `Delay_feat` 作为边贡献项进入 LightGCN 或 interaction 计算。
- LightGCN 输出增强后的 user/item/item-interaction 表征。
- 这些表征再转成或注入 `NS Token`。
- 后端 HyFormer 不变。

正确的部分：

- 当前 HyFormer 的 NS token 主要来自原始 user/item sparse/dense 特征，加入 graph/item interaction embedding 可以补充协同过滤信号。
- LightGCN 能捕捉 user-item 共现结构，尤其适合高基数 ID 和交互图。
- 把 graph embedding 注入 NS token，是比替换后端主干更稳的方案。

需要设计清楚：

```text
节点是什么？
  user_id
  item_id
  item feature value
  user feature value
  field-aware feature node

边是什么？
  every row 的 user-item 边
  user-feature 边
  item-feature 边
  item-item co-occurrence 边

边属性是什么？
  delay bucket
  edge weight
  domain
  frequency
  recency
  label-free interaction count

图 embedding 如何注入？
  user_embedding + graph_user_embedding
  item_embedding + graph_item_embedding
  concat -> projection -> NS token
  new graph NS token
```

判断：在“前端增强，HyFormer 保留”的前提下，这条建议是合理且值得做的，但必须严格控制泄漏和冷启动 fallback。

### 3.8 Sequence feature -> Longer -> Sequence token -> HyFormer

PDF 第 2 页明确包含：

```text
Sequence feature
Longer
Sequence token
Hyformer
```

正确的部分：

- 当前 `model.py` 已支持 `seq_encoder_type='longer'`。
- Longer 适合处理长序列，能用 top-k 压缩注意力降低长序列成本。
- 先把 sequence feature 通过 Longer 变成 sequence token，再进入后续 HyFormer，是与 baseline 兼容的路径。

需要注意：

- 不同 domain 的最佳 `seq_max_lens` 可能不同。
- Longer 的 `top_k` 会影响保留信息量和计算成本。
- 如果 sequence side feature 已经经过 pooling，可能削弱 Longer 的作用。
- 需要检查 padding mask、time bucket、RoPE/position 信息是否仍然一致。

判断：这部分与当前 baseline 方向一致，属于可落地的前端/序列侧增强，不是后端替换。

## 4. 与当前 baseline 的兼容性

当前 baseline 已支持：

- NS tokenization。
- sequence tokenization。
- `seq_encoder_type=longer`。
- Query Generation Module。
- MultiSeqHyFormerBlock。
- time bucket。
- 多值离散特征 embedding 后 mean pooling。

PDF 建议可以按最小侵入方式接入：

```text
1. 数据预处理阶段：
   频次统计、缺失率统计、dense 归一化参数、delay/recency 统计。

2. 图/interaction 阶段：
   用训练集构造 user-item/item-feature/user-feature 图。
   训练 LightGCN 或生成 graph embedding。

3. 模型输入阶段：
   将 graph embedding 或 interaction feature 注入 user/item NS token。

4. 序列阶段：
   保留 Longer 生成 sequence token。

5. 后端阶段：
   Query Generation Module、Global token、HyFormer 不改或尽量少改。
```

这样可以保持实验可控，也方便做 ablation。

## 5. 最重要的提升空间

### 5.1 明确 `Delay_feat` 的安全语义

优先级最高。

需要先做数据检查：

```text
delay_feat = timestamp - label_time
min / p1 / p50 / p99 / max
delay_feat < 0 比例
delay_feat < 3 比例
按 label_type 分组的 delay 分布
按 train/valid 分组的 delay 分布
delay 与 label 的 AUC / IV / lift
```

如果 `delay_feat` 对 label 几乎是直接区分信号，应视作泄漏风险。

更安全方案：

- 将 `Delay_feat` 替换为历史行为 `timestamp - event_timestamp`。
- 或只用 `Delay_feat` 做训练图去噪，不把结果传播到验证/测试。
- 或把 delay 分桶后作为弱边属性，并做严格 temporal split 检查。

### 5.2 把低频删除改成低频建模

不建议第一版就硬删所有低频样本。

推荐：

```text
freq_bucket -> dense/embedding feature
low_freq shared embedding
LightGCN graph 中过滤极低频节点
HyFormer 输入样本仍保留
验证分 low/high frequency 看 AUC
```

### 5.3 Graph embedding 注入方式

三种可选方案：

```text
方案 A：加法注入
normal_embedding + graph_embedding

方案 B：拼接投影
concat(normal_embedding, graph_embedding) -> Linear -> d_model

方案 C：新增 token
graph_user_item_token 作为额外 NS token
```

建议顺序：

1. 先用方案 A，改动最小。
2. 再试方案 B，表达力更强。
3. 最后试方案 C，但要注意 RankMixer 的 `d_model % T == 0` 约束。

### 5.4 从 LightGCN 升级到异构图时要 field-aware

如果节点不只是 `user_id` 和 `item_id`，还包括匿名特征值，必须 field-aware：

```text
node = (field_name, value)
```

不要把不同字段中相同整数 ID 当成同一个节点。匿名 ID 的空间通常不共享。

### 5.5 dense/list 特征处理增强

对 dense numerical feature：

- 用训练集统计量归一化。
- 对重尾值 clip 或 log transform。
- 加 missing indicator。

对 list feature：

- 类别 list 先 embedding 再 pooling。
- 对 aligned int-dense list 保留元素级对应关系。
- 对 item interaction list 使用 target-aware pooling。

### 5.6 高缺失特征不要只按 70% 删除

建议每个特征计算：

```text
missing_rate
coverage
unique_count
top value ratio
positive rate by presence
label lift
train/valid stability
embedding memory cost
```

删除条件应至少结合：

```text
missing_rate 高
label lift 低
train/valid 分布不稳定
计算成本高
```

而不是只看 `>70%`。

## 6. 推荐实验路线

### A0：复现 baseline

目的：

- 得到稳定 AUC、logloss、训练吞吐和显存。
- 后续所有改动都和同一个 baseline 比较。

### A1：只做安全的数据预处理

包括：

- dense feature normalization。
- missing indicator。
- list length / present ratio。
- 不使用 `label_time`。
- 不硬删低频样本。

目的：

- 验证基础数据工程收益。

### A2：启用 Longer 路径

配置：

```text
--seq_encoder_type longer
--seq_top_k <候选值>
```

同时扫描不同 domain 的 `seq_max_lens`。

目的：

- 验证 `Sequence feature -> Longer -> Sequence token` 的收益。

### A3：加入低频分桶特征

包括：

```text
user_freq_bucket
item_freq_bucket
is_rare_user
is_rare_item
```

目的：

- 替代硬删除，提升低频样本鲁棒性。

### A4：加入安全的 item interaction 特征

先不用 `label_time`，只用历史交互、频次、共现、recency bucket。

目的：

- 验证 item interaction 本身是否提升 NS token 表征。

### A5：LightGCN graph embedding 注入 NS token

方案：

```text
user_embedding + graph_user_embedding
item_embedding + graph_item_embedding
```

或：

```text
concat -> projection -> NS token
```

目的：

- 验证前端 graph enhancement 对 HyFormer 后端的增益。

### A6：受控测试 `Delay_feat as edge contributes`

前提：

- 已确认 `label_time` 不泄漏，或改用安全 delay。

测试：

```text
edge_weight = delay bucket
edge_weight = log1p(delay)
edge_weight = recency decay
```

目的：

- 验证 delay edge contribution 是否优于普通无权图。

### A7：aligned int-dense list element-level token

目的：

- 替代简单 average pooling。
- 保留 ID 与 dense value 对齐关系。

## 7. 必做验证指标

每个实验都需要记录：

```text
AUC
logloss
训练样本数
验证样本数
正负样本比例
低频 user AUC
高频 user AUC
低频 item AUC
高频 item AUC
graph 节点数
graph 边数
graph 覆盖率
unseen user/item fallback 比例
训练吞吐
显存占用
```

涉及 `Delay_feat` 时额外记录：

```text
delay min / p1 / p50 / p99 / max
delay < 0 比例
delay < 3 比例
delay 按 label 分布
delay 按 train/valid 分布
使用 delay 前后各分组 AUC
```

涉及特征删除时额外记录：

```text
删除字段列表
每个字段 missing_rate
每个字段 label lift
删除前后样本覆盖率
删除前后 AUC
```

## 8. 最终建议

这份 suggestion 在你补充的架构前提下是更合理的：

- 它不是要移除 HyFormer。
- 它是在 HyFormer 前面做更强的特征清洗、特征变换和 user-item/item interaction 增强。
- `Light GCN` 应理解为增强 NS token 或 user/item embedding 的前端模块。
- `Longer` 应理解为 sequence feature 到 sequence token 的路径。
- `Query Generation Module`、`Global token` 和 `HyFormer` 仍然是后端主干。

最优先需要修正和确认的是：

1. `Delay_feat = timestamp - label_time` 是否安全，是否存在标签泄漏。
2. 低频 user/item 不要先硬删，优先做分桶和图侧过滤。
3. `int_feats` 不要默认均值填充，除非确认它们是连续数值而不是类别 ID。
4. 高缺失特征删除要结合 label lift 和稳定性。
5. LightGCN/item interaction 的结果建议注入 NS token，而不是改 HyFormer 后端。

推荐的第一版落地路径：

```text
安全 preprocessing
-> Longer sequence token
-> item interaction / LightGCN graph embedding 注入 NS token
-> 原 Query Generation Module + Global token + HyFormer
-> 分组 ablation 验证
```

这样既保留了 PDF 的全部建议，也符合你实际设计的“前端增强、后端 HyFormer 保留”的架构意图。

