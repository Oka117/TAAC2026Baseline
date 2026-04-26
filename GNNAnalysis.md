## 14. GNN to Transformer 方向建议

你的思路是 **GNN to Transformer**，这个方向非常适合当前比赛。原因是：数据虽然匿名，但本质上是高度关系型的广告日志。用户、物料、稀疏字段、行为事件、行为域、时间桶之间存在大量共现、转移和高阶交互，而这些关系不一定能通过单纯的 embedding lookup + Transformer 充分表达。

可以把 GNN 看作 **关系归纳与 token 增强模块**，把 Transformer 看作 **全局交互与预测模块**。

整体范式可以设计为：

```text
Raw sparse/dense/sequence features
        │
        ▼
Heterogeneous graph construction
        │
        ▼
GNN / Graph Encoder
        │
        ▼
Graph-enhanced tokens
        │
        ▼
Unified Transformer / HyFormer backbone
        │
        ▼
CVR prediction
```

### 14.1 为什么 GNN 适合这份数据

这份数据有几个非常适合图建模的特点：

- 大量匿名 ID 特征没有文本语义，但有强共现关系。
- 用户行为序列天然形成事件转移图。
- 用户、物料、字段、行为域之间是异构关系。
- 多值特征本质上是一组实体集合，可以建成局部子图。
- 目标 item 与历史 item/行为 token 的关系非常关键。
- aligned int-dense 特征可以看作带边权或节点属性的局部图。

因此，GNN 可以补足 baseline 中“每个特征独立 embedding、list 特征 mean pooling”的缺陷。

### 14.2 可构建的图类型

建议从轻到重设计多种图，不要一开始就做全量巨大图。

#### 14.2.1 样本内异构图

每个样本构建一个小图：

```text
user token
item token
field tokens
sequence event tokens
domain tokens
time bucket tokens
```

边可以包括：

- user -> user field
- item -> item field
- event -> event side-info
- event -> domain
- event -> time bucket
- target item -> historical event
- adjacent event -> adjacent event

这种做法不需要离线全局图，容易集成到当前 dataloader 和模型里。

#### 14.2.2 行为序列转移图

对每个用户的历史事件建图：

- 节点：历史行为事件或事件中的 item/id。
- 边：相邻行为转移。
- 边属性：时间间隔、domain、action type。

然后用 GNN 得到每个事件的 graph-enhanced event embedding，再输入 Transformer。

这可以增强序列 token，替代当前 `_embed_seq_domain` 中简单 concat + projection 的方式。

#### 14.2.3 特征共现图

基于训练集统计构造 feature-id 共现图：

- 节点：高频 user/item/field id。
- 边：同样本共现、同序列共现、同 item 共现。
- 边权：PMI、co-count、时间衰减共现、转化率差异等。

训练时可以用离线图 embedding 初始化 ID embedding，或在线使用 GraphSAGE/LightGCN 风格的邻居聚合。

这种方法能利用匿名 ID 之间的统计关系，通常对广告推荐很有价值。

#### 14.2.4 User-Item 交互图

如果数据允许从日志中恢复 user-item 交互，可以构造二部图：

```text
User ID <-> Item ID
```

边权可由曝光、点击、转化、时间衰减等构成。可以用 LightGCN、GraphSAGE 或 PinSAGE 思路得到 user/item graph embedding，再作为额外 token 输入 baseline。

注意：如果 user_id 在测试集冷启动严重，需要做好 OOV 和时间切分，避免过拟合。

### 14.3 推荐架构一：Graph-Enhanced Tokenizer

这是最容易接入当前 baseline 的路线。

当前 baseline 的 tokenization 是：

```text
id embedding -> pooling/concat -> projection -> token
```

可以改成：

```text
id embedding
  + graph embedding
  + field/domain/time embedding
  + dense value projection
  -> token
```

对代码的改动点：

- 在 `model.py` 的 NS tokenizer 中加入 graph embedding lookup。
- 在 `_embed_seq_domain` 中为 sequence side-info 加入 graph-enhanced embedding。
- 对 aligned int-dense 特征，把 `(id_i, dense_i)` 构造成 element token。
- 保持后面的 `MultiSeqHyFormerBlock` 不变，先验证 AUC 增益。

优点：

- 改动小。
- 风险低。
- 可以复用当前训练框架。
- 适合作为第一阶段实验。

### 14.4 推荐架构二：Sample Graph Encoder + Transformer

第二种路线是每个样本构造一个局部异构图，用轻量 GNN 编码后，再把节点表示作为 Transformer tokens。

示意：

```text
Sample graph nodes:
  [target item, user fields, item fields, dense-int elements, behavior events]

Sample graph edges:
  [field ownership, event order, event-domain, target-event, dense-int alignment]

GNN:
  R-GCN / GraphSAGE / GAT

Transformer input:
  graph-updated node embeddings + type/position/time embeddings
```

这种结构更符合“GNN to Transformer”的名字，也更有创新性。它可以把局部结构先编码出来，再交给 Transformer 做全局交互。

缺点是工程复杂度更高，需要处理 batch 内变长图、mask、边类型和效率。

### 14.5 推荐架构三：Graph Tokens as Memory

可以不把所有图节点都输入 Transformer，而是把 GNN 输出压缩成少量 graph memory tokens：

```text
sequence tokens + NS tokens + graph memory tokens -> Transformer
```

graph memory tokens 可以包括：

- user graph token
- item graph token
- sequence transition graph token
- aligned dense-int graph token
- domain graph tokens

这条路线的优点是 token 数可控，适合大规模训练和推理延迟约束。

### 14.6 与当前 baseline 的结合方式

建议按三个阶段改造，不要一次重写。

第一阶段：Graph Embedding Injection

- 离线训练或统计 ID graph embedding。
- 在 user/item/seq embedding 中拼接或相加 graph embedding。
- 保持 HyFormer 主体不变。
- 观察 AUC 是否提升。

第二阶段：Aligned Feature Graph

- 针对 `user_int_feats_62-66/89-91` 与 dense 对齐字段构造局部 element graph。
- 每个 element 节点包含 id embedding 和 dense value projection。
- 用 attention/GNN 聚合成 field token。
- 替换当前 mean pooling。

第三阶段：Unified Graph-to-Transformer Block

- 引入 CLS/CVR token。
- 所有 NS、sequence、graph memory tokens 进入统一 Transformer block。
- 逐步减少当前 per-domain 独立 encoder 和 query cross-attention。
- 对齐比赛的 Unified Block 创新方向。

### 14.7 推荐优先实现的最小版本

如果想快速验证 GNN-to-Transformer 是否有收益，建议先做最小版本：

1. 用训练集构造 item-item 共现图或 feature-id 共现图。
2. 用 LightGCN/GraphSAGE 离线得到节点 embedding。
3. 在 `GroupNSTokenizer` / `RankMixerNSTokenizer` 和 sequence embedding 中加入 graph embedding。
4. 将原始 ID embedding 与 graph embedding concat 后 projection。
5. 其他模型结构不变。

这样可以快速回答一个关键问题：图关系是否给 AUC 带来增益。

如果有增益，再继续做更复杂的 sample graph 和 unified backbone。

### 14.8 实验设计建议

GNN-to-Transformer 方向需要清楚证明图模块有价值。建议做以下 ablation：

| 实验 | 说明 |
| --- | --- |
| Baseline | 原始 PCVRHyFormer。 |
| + Graph Embedding | 只加入离线图 embedding。 |
| + Aligned Dense-Int Graph | 加入 aligned 字段图建模。 |
| + Graph Memory Tokens | 加入图压缩 token。 |
| + Unified Transformer | 用统一 Transformer 替换部分分支结构。 |

记录：

- AUC
- logloss
- 参数量
- 训练吞吐
- 显存占用
- 推理延迟

如果你要冲创新奖，还要记录不同图规模、embedding 维度、GNN 层数、Transformer 层数之间的 scaling law。

### 14.9 风险点

这个方向也有几个风险：

- 全局图过大，构图和采样成本高。
- user_id/item_id 可能有严重长尾，graph embedding 容易过拟合热门节点。
- 如果图构造使用了未来信息，会造成数据泄漏。
- batch 内动态图会增加训练复杂度。
- GNN 层数过深会过平滑，未必适合广告稀疏 ID。

建议先用时间安全的训练集统计图，避免用验证/测试信息构图。

## 15. 建议实验顺序

建议按以下顺序推进：
3. 开启或测试 RoPE、longer encoder、不同 seq length。
4. 实现最小版 Graph Embedding Injection。
5. 实现 aligned int-dense graph/token。
6. 实现 CLS/CVR token 统一 backbone。
7. 系统做 GNN-to-Transformer scaling law 表格。


## 16. 总结

- 更系统的 scaling law 实验。
- 将 GNN 作为关系归纳模块，把图增强 token 输入统一 Transformer。
