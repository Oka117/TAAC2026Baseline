# TAAC2026Baseline 源码分析

本文档分析官方 baseline 的数据读取、模型结构、训练流程、优点、局限和可改进方向。本文只讨论当前源码和比赛统一建模目标，不包含额外研究方向扩展。

## 1. 项目结构

```text
TAAC2026Baseline/
├── README.md
├── README.zh.md
├── README.en.md
├── train.py
├── trainer.py
├── dataset.py
├── model.py
├── utils.py
├── run.sh
├── ns_groups.json
└── demo_1000.parquet
```

| 文件 | 作用 |
| --- | --- |
| `README.md` | 大赛 Introduction、Dataset、Task 说明主页。 |
| `README.zh.md` | 中文源码分析。 |
| `README.en.md` | 英文源码分析。 |
| `train.py` | 训练入口，负责参数解析、数据加载、模型构造和 trainer 构造。 |
| `trainer.py` | 训练循环、验证、AUC 计算、checkpoint 和 early stopping。 |
| `dataset.py` | Parquet 数据读取、schema 解析、padding、时间桶构造。 |
| `model.py` | PCVRHyFormer 模型主体。 |
| `utils.py` | 日志、随机种子、early stopping、focal loss 等工具。 |
| `run.sh` | 官方默认启动脚本。 |
| `ns_groups.json` | 非序列特征分组示例。 |

## 2. Baseline 总体判断

这份 baseline 不是简单 DNN，而是一个面向比赛主题设计的混合 token 模型。它已经包含非序列 tokenization、多域序列 tokenization、query-based sequence reading、stackable block、RankMixer 融合、时间桶和 AUC 训练闭环。

但它还不是完全统一的架构。当前实现中，序列和非序列特征仍然有明显的专用处理分支：4 个序列域分别编码，query 分别从对应序列读取信息，非序列 token 主要在 RankMixer 阶段与 query token 融合。因此它更接近“多序列塔 + 非序列塔 + 后融合”，不是所有 token 从输入阶段就进入同一个同构 backbone。

## 3. 数据管道

数据读取逻辑位于 `dataset.py`，核心类是 `PCVRParquetDataset`。

### 3.1 Schema 驱动

源码通过 `schema.json` 描述特征布局，不把所有字段硬编码进模型。`FeatureSchema` 记录每个 feature id 在扁平 tensor 中的 offset 和 length。

支持的特征组包括：

- `user_int`：用户离散特征，包含 scalar int 和 list int。
- `item_int`：广告/物料离散特征。
- `user_dense`：用户 dense/list-float 特征。
- `item_dense`：代码保留接口，当前数据为空。
- `seq`：四个 domain 的序列特征配置。

### 3.2 非序列特征处理

在 `_convert_batch` 中，用户和物料离散特征会被写入预分配 numpy buffer：

- scalar int 直接填入一个位置；
- list int padding/truncation 到 schema 指定长度；
- 小于等于 0 的值统一视作 padding，置为 0；
- 超出词表范围的 ID 默认裁剪为 0，并记录 OOB 统计。

用户 dense 特征被 padding 成定长 float tensor。需要注意的是，官方数据说明中有部分 `user_int_feats_x` 和 `user_dense_feats_x` 是逐元素对齐的，但 baseline 没有显式保留这种 element-wise 关系，而是把 int list 和 dense list 分别放入不同处理路径。

### 3.3 序列特征处理

每个 domain 的序列被处理成：

```text
[B, num_side_features, max_len]
```

其中 `B` 是 batch size，`num_side_features` 是该 domain 中除 timestamp 外的 side-info 字段数量，`max_len` 由 `--seq_max_lens` 控制。默认配置为：

```text
seq_a:256,seq_b:256,seq_c:512,seq_d:512
```

序列长度记录在 `{domain}_len` 中，模型后续用它构造 padding mask。

### 3.4 时间桶

代码使用样本当前 `timestamp` 减去序列事件 timestamp，得到相对时间差，再用 `BUCKET_BOUNDARIES` 离散化为时间桶 ID。

时间桶设计如下：

- padding 位置为 0；
- 有效 bucket 从 1 开始；
- 超过最大边界的时间差被压到最后一个 bucket；
- 模型端通过 `nn.Embedding(num_time_buckets, d_model, padding_idx=0)` 加到序列 token 上。

这是广告序列建模中的重要信号，因为行为新近程度通常强影响转化概率。

### 3.5 训练/验证切分

`get_pcvr_data` 按 Parquet Row Group 切分数据：

- 训练集使用前面的 row groups；
- 验证集使用尾部 `valid_ratio` 比例；
- 默认 `valid_ratio=0.1`。

这种切分方式高效，但是否接近时间外推取决于 parquet 文件和 row group 的排序方式。

## 4. 模型结构

模型主体是 `PCVRHyFormer`，位于 `model.py`。

整体流程如下：

```text
user/item sparse features -> NS tokens
user dense features       -> dense NS token
domain sequences          -> sequence tokens
NS tokens + sequence summaries -> query tokens
query tokens + sequence tokens + NS tokens -> MultiSeqHyFormerBlock stack
final query tokens -> CVR prediction head
```

模型主要包含：

- 非序列 tokenizer；
- 序列 tokenizer；
- query generator；
- 多层 `MultiSeqHyFormerBlock`；
- CVR prediction head。

## 5. 非序列 Tokenizer

源码支持两种非序列 tokenizer。

### 5.1 GroupNSTokenizer

`GroupNSTokenizer` 根据 `ns_groups.json` 的人工分组，把若干 feature id 组合为一个 token。

处理方式：

1. 每个离散 feature 使用独立 embedding table。
2. 多值 feature 使用 mean pooling。
3. 同一 group 内多个 feature embedding 拼接。
4. 通过 Linear + LayerNorm 投影到 `d_model`。
5. 每个 group 输出一个 NS token。

优点是语义清晰，缺点是需要人工分组，且 token 数受 group 数限制。

### 5.2 RankMixerNSTokenizer

`RankMixerNSTokenizer` 是 `run.sh` 默认使用的模式。它把所有 feature embedding 按顺序拼接成一个长向量，再等分成指定数量的 chunk，每个 chunk 投影成一个 token。

默认启动脚本中：

```bash
--ns_tokenizer_type rankmixer
--user_ns_tokens 5
--item_ns_tokens 2
--num_queries 2
--ns_groups_json ""
```

由于 `--ns_groups_json ""`，默认不会使用 `ns_groups.json` 中的人工分组，而是退化为每个特征一个 singleton group，再由 RankMixer tokenizer 切成固定数量 token。

优点是 token 数可控，便于满足 `d_model % T == 0` 的约束；缺点是语义分组较弱，多值特征仍然会先 mean pooling。

## 6. 序列 Tokenizer

每个序列 domain 输入形状为：

```text
[B, S, L]
```

其中 `S` 是 side-info 特征数量，`L` 是序列长度。

`_embed_seq_domain` 的处理流程：

1. 每个 side-info feature 独立 embedding lookup，得到 `[B, L, emb_dim]`。
2. 如果 feature 词表较大且超过 `seq_id_threshold`，训练时额外加 dropout。
3. 同一事件位置上的多个 side-info embedding concat。
4. 通过 Linear + LayerNorm 投影到 `d_model`。
5. 加上时间桶 embedding。

因此，一个行为事件最终变成一个 `d_model` 维 token。这个设计直接、稳定，但不同 domain 仍然有各自的 tokenizer 和 projection，没有在输入阶段统一成一个 token stream。

## 7. Query Generator

`MultiSeqQueryGenerator` 为每个序列 domain 生成若干 query tokens。每个 domain 的 query 依赖：

- 所有 NS tokens 的 flatten 表示；
- 当前 domain 序列 token 的 mean pooling 表示。

然后通过独立 FFN 生成 `num_queries` 个 query token。这个机制可以理解为：模型根据当前用户、物料、上下文和序列摘要，决定应该从该序列中读取哪些信息。

## 8. MultiSeqHyFormerBlock

`MultiSeqHyFormerBlock` 是模型的主要可堆叠模块。每层包含三步。

### 8.1 Sequence Evolution

每个序列 domain 独立经过一个 sequence encoder。代码支持：

- `swiglu`：无 attention，只做 SwiGLU FFN；
- `transformer`：标准 self-attention + FFN；
- `longer`：保留最近 top-k token 的压缩注意力，适合长序列。

默认是 `transformer`。

### 8.2 Query Decoding

每个 domain 的 query tokens 对该 domain 的 encoded sequence 做 cross-attention，从序列中读取与当前样本相关的信息。

### 8.3 Token Fusion

所有 domain 的 decoded query tokens 与 NS tokens 拼接：

```text
[Q_a, Q_b, Q_c, Q_d, NS]
```

然后进入 `RankMixerBlock`。`RankMixerBlock` 支持三种模式：

- `full`：token mixing + FFN，需要 `d_model % T == 0`；
- `ffn_only`：只做 per-token FFN；
- `none`：跳过 mixer。

默认是 `full`，通过 reshape/transpose 做参数无关的 token mixing，再接 FFN 和残差。

## 9. 输出 Head

所有 block 结束后，代码只取最终 query tokens：

```python
all_q = torch.cat(curr_qs, dim=1)
output = all_q.view(B, -1)
output = self.output_proj(output)
logits = self.clsfier(output)
```

注意：最终 `curr_ns` 没有直接进入输出 head。NS tokens 只能通过中间 RankMixer 对 query tokens 的影响间接参与预测。对于广告 CVR，用户、物料、上下文等非序列强特征往往非常重要，因此这是一个可优先尝试的改进点。

## 10. 训练流程

训练逻辑在 `trainer.py`。

### 10.1 Loss

支持两种 loss：

- `bce`：`binary_cross_entropy_with_logits`；
- `focal`：自定义 sigmoid focal loss。

默认是 BCE。

### 10.2 Optimizer

代码将参数分成两类：

- sparse params：所有 `nn.Embedding` 参数，使用 Adagrad；
- dense params：其他参数，使用 AdamW。

这是推荐/广告模型常见做法。Embedding 多且稀疏更新，用 Adagrad 通常更合适。

### 10.3 高基数 Embedding 重置

模型支持按 epoch 对高基数 embedding 重新初始化：

```text
--reinit_sparse_after_epoch
--reinit_cardinality_threshold
```

目的是缓解多 epoch 复用训练数据时高基数 ID embedding 的过拟合问题。

### 10.4 Evaluation

验证阶段：

1. 对验证集跑 forward。
2. sigmoid 得到概率。
3. 用 sklearn `roc_auc_score` 计算 AUC。
4. 同时计算 logloss。
5. early stopping 监控 AUC。

如果预测中出现 NaN，会先过滤再计算指标。

## 11. 优点

- 数据工程完整：Parquet row group 读取、list padding、多域序列、时间桶、OOB 裁剪、多 worker dataloader 都已覆盖。
- 模型结构不弱：已经包含 tokenization、query-based sequence reading、stackable block、token mixer 和 Transformer encoder。
- 参数可扩展：`d_model`、`emb_dim`、层数、head 数、query 数、序列长度、token 数等都可以通过命令行调节。
- 训练闭环完整：loss、AUC、logloss、early stopping、checkpoint 和 TensorBoard 都已经具备。

## 12. 局限

- 统一性不彻底：序列与非序列仍然先走不同分支，后续再融合。
- aligned int-dense 信息利用不足：对应的 int list 和 dense list 没有按元素组合建模。
- 最终 head 没有直接使用 NS tokens。
- 多值特征多用 mean pooling，粒度偏粗。
- `RankMixerBlock` 的 full 模式要求 `d_model % T == 0`，限制了部分配置空间。

## 13. 推荐改进方向

1. **让 final head 使用 Q + NS tokens**  
   当前最终只用 query tokens，可以尝试拼接 final query tokens 和 final NS tokens，或加入 CLS/CVR token。

2. **显式建模 aligned int-dense 特征**  
   对 `user_int_feats_62-66/89-91` 和对应 dense 特征，构造 element-level token：`id_embedding + dense_value_projection + field_embedding`。

3. **构建更统一的 token stream**  
   将 target item、user fields、context fields、sequence event tokens、dense-aligned tokens 统一输入同构 block。

4. **系统做 scaling law 实验**  
   扫描 `d_model`、层数、head 数、序列长度、query 数、NS token 数，记录 AUC、logloss、吞吐、显存和延迟。

## 14. 建议实验顺序

1. 跑通官方 baseline，得到本地可复现 AUC。
2. 修改 final head，让 Q + NS 一起预测。
3. 测试 RoPE、longer encoder 和不同序列长度。
4. 实现 aligned int-dense token。
5. 实现 CLS/CVR token 统一 backbone。
6. 系统整理 scaling law 表格。

## 15. 总结

这份 baseline 是一个工程完整、结构不弱的官方起点。它已经接近比赛“统一序列建模与特征交互”的方向，但仍然属于混合式架构。最值得优先改进的是 final head、aligned int-dense token、统一 token stream 和系统 scaling law 实验。
