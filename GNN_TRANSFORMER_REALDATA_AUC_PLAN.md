# 真实数据导向的 GNN + Transformer AUC 提升方案

## 1. 当前结果判断

你现在的 4LayerGNN 方案把 2-layer 版本的 AUC 从 `0.806` 提升到 `0.815`，这是一个很关键的信号：**图关系建模是有效的**。但这个提升也说明当前 GNN 的收益已经接近 NS-only 图的上限。

当前 4LayerGNN 的核心是对非序列 NS tokens 做样本内全连接 message passing。它能学习用户字段、物料字段、dense token 之间的静态共现关系，所以有 `+0.009` AUC；但它还没有直接解决 CVR 数据里更强的三类信号：

- 用户历史行为的时间转移和最近行为趋势；
- target item 与历史行为事件之间的匹配关系；
- `user_int_feats_62-66/89-91` 与 `user_dense_feats_62-66/89-91` 的逐元素对齐关系。

因此下一阶段不要继续简单加深 TokenGNN。更切实的方向是：**保留成功的 NS 图逻辑，但把 GNN 从静态 token 平滑升级为“序列事件图 + 对齐特征图 + Transformer 全局交互”。**

## 2. 可见真实数据结构分析

根据 `README.md`、`dataset.py` 和 `demo_1000.parquet` 暴露的字段结构，比赛数据不是普通 tabular，而是典型广告推荐日志：

| 数据块 | 结构 | 对 AUC 的含义 |
| --- | --- | --- |
| meta | `user_id`、`item_id`、`label_type`、`label_time`、`timestamp` | 可以用于时间安全切分、构图防泄漏和近期行为建模 |
| user int | 46 列，其中 11 列是 list<int64> | 用户画像、兴趣、多值实体集合，适合 field graph |
| user dense | 10 列 list<float> | 其中 8 列和 user int 逐元素对齐，是当前 baseline 未充分利用的强信号 |
| item int | 14 列，其中 `item_int_feats_11` 是 list<int64> | target item 多字段属性，是 target-aware graph 的核心 |
| sequence | 4 个行为域，共 45 列 list<int64> | 每个样本自带多域历史行为，天然是 temporal graph |

最重要的一点是：

```text
user_int_feats_62:   [id_1, id_2, id_3]
user_dense_feats_62: [v_1,  v_2,  v_3]
```

这类字段不是两个独立特征，而是 element-wise pair：`(id_i, dense_i)`。当前 baseline 把 int list mean pooling、dense list flatten/projection 分开处理，会丢掉“哪个 dense 值属于哪个 id”的关系。这个信息在广告 CVR 中通常非常强，因为 dense 可能表示统计强度、兴趣分、频次、时长、置信度或历史反馈强度。

## 3. 为什么 4LayerGNN 增益有限

4LayerGNN 成功的逻辑是：

```text
NS tokens -> complete graph TokenGNN -> Transformer/HyFormer
```

它解决的是“静态字段之间需要交互”的问题。但 AUC 从 `0.806` 到 `0.815` 后继续靠加层数大概率会遇到三个瓶颈：

- **过平滑**：NS token 数很少，全连接图 4 层以后不同 token 容易同质化；
- **图节点过粗**：RankMixer NS token 是多个 feature embedding 的压缩 chunk，不是原始 feature/entity 节点；
- **信号路径不够直接**：行为序列、target-event 匹配、aligned dense-int element 没有在图里显式出现。

所以后续应该让 GNN 做更真实的关系归纳，让 Transformer 做全局重排：

```text
局部真实关系 -> GNN
跨域全局交互 -> Transformer / HyFormer
最终排序特征 -> Q + NS + graph memory
```

## 4. 推荐主方案：RTG-HyFormer

建议把下一版命名为 **RTG-HyFormer**：Real-data Typed Graph HyFormer。

整体结构：

```text
user/item sparse + dense
        |
        +--> NS tokenizer -> 2-layer TokenGNN
        |
        +--> aligned dense-int element graph
        |
sequence domains
        |
        +--> target-aware temporal SequenceGraphEncoder
        |
        v
graph-enhanced sequence tokens
        |
Longer/Transformer HyFormer blocks
        |
        v
final head: Q tokens + final NS tokens + seq graph memory + aligned graph memory
        |
        v
CVR logit
```

### 4.1 模块一：保留但收缩 TokenGNN

保留现有 TokenGNN，但建议从 4 层降为 2 层：

```bash
--use_token_gnn
--token_gnn_layers 2
--token_gnn_layer_scale 0.05
```

原因：

- 4LayerGNN 已证明 NS 图有效；
- 继续加深 NS-only 图容易过平滑；
- 让 TokenGNN 做静态字段关系增强，把主要容量留给序列图和 aligned graph。

### 4.2 模块二：Target-aware SequenceGraphEncoder

每个行为域构建一个轻量 temporal graph：

```text
节点：一个历史行为事件 token
边：
  event_i <-> event_{i-1}
  event_i <-> event_{i+1}
  event_i <-> event_{i-2/i+2}   可选 skip edge
  target_item -> event_i        target-aware gated edge
边权：
  time bucket decay
  recent-position bias
```

GNN 输出两部分：

- 更新后的 event tokens，继续输入 Transformer/LongerEncoder；
- 每个 domain 一个 graph-memory summary，进入 final head。

这比 NS-only GNN 更可能带来 AUC 增益，因为 CVR 的强信号经常来自“最近看过什么、行为链如何变化、当前广告和历史行为是否匹配”。

### 4.3 模块三：Aligned Dense-Int Element Graph

对这些字段重点建模：

```text
62, 63, 64, 65, 66, 89, 90, 91
```

每个字段构造 element nodes：

```text
node_i = id_embedding(user_int_feats_fid[i])
       + dense_value_projection(user_dense_feats_fid[i])
       + field_embedding(fid)
       + position_embedding(i)
```

然后用 1 到 2 层轻量 GNN 或 attention pooling 得到 field token：

```text
element nodes -> aligned field graph -> aligned graph token
```

最终得到 4 到 8 个 aligned graph memory tokens，加入 NS graph 和 final head。

这是最值得做的新增模块，因为它直接修复 baseline 的信息损失：当前代码把 int list 和 dense list 拆开处理，而真实数据里它们是逐元素绑定的。

### 4.4 模块四：Final Head 使用 Q + NS + Graph Memory

当前模型主要用最终 query tokens 预测。建议最终 head 改为：

```text
head_input = [
  final_q_tokens,
  final_ns_tokens,
  sequence_graph_summaries,
  aligned_dense_int_graph_summaries
]
```

原因：

- NS token 是强 tabular signal，不能只靠中间 RankMixer 间接传递；
- graph memory 是 GNN 的压缩判别信息，应该直接影响 logit；
- final head 增量小、风险低，通常比继续堆 GNN 层更稳定。

## 5. 需要修改的文件

### 5.1 `dataset.py`

当前 `dataset.py` 已经把 `user_int_feats` 和 `user_dense_feats` 分别 flatten 到 tensor，但模型不知道每个 dense fid 的 offset。

建议新增或暴露：

```python
pcvr_dataset.user_dense_schema.entries
```

训练时传入模型：

```python
user_dense_feature_specs = [(fid, offset, length), ...]
```

不一定要改变 batch 数据格式；只要模型拿到 dense offset，就能从现有 `inputs.user_dense_feats` 中切出对应 dense list。

### 5.2 `train.py`

新增 `model_args`：

```python
"user_dense_feature_specs": pcvr_dataset.user_dense_schema.entries,
"use_aligned_dense_int_graph": args.use_aligned_dense_int_graph,
"aligned_graph_fids": args.aligned_graph_fids,
"aligned_graph_tokens": args.aligned_graph_tokens,
"output_include_ns": args.output_include_ns,
```

新增 CLI：

```bash
--use_aligned_dense_int_graph
--aligned_graph_fids 62,63,64,65,66,89,90,91
--aligned_graph_layers 1
--aligned_graph_tokens 8
--output_include_ns
```

已有 sequence graph 参数继续保留：

```bash
--use_seq_graph
--seq_graph_layers
--seq_graph_layer_scale
--graph_output_fusion
```

### 5.3 `model.py`

建议新增 3 个能力。

第一，`AlignedDenseIntGraphEncoder`：

```python
class AlignedDenseIntGraphEncoder(nn.Module):
    """
    For each aligned fid, build element-level nodes from:
    id embedding + dense value projection + field embedding + position embedding.
    Then pool them into graph memory tokens.
    """
```

输入：

```text
inputs.user_int_feats
inputs.user_dense_feats
```

输出：

```text
aligned_graph_tokens: (B, N_aligned_tokens, d_model)
```

第二，升级 `SequenceGraphLayer`：

- 当前已有 prev/next temporal edge；
- 下一版增加 skip edge：`i±2` 或 recent-window mean；
- target message 加入 dot-product matching gate；
- time bucket 进入 gate，而不只是加到 event token 上。

第三，修改 `_run_multi_seq_blocks` 的输出拼接：

```python
output_tokens = [all_q]
if output_include_ns:
    output_tokens.append(curr_ns)
if graph_summaries is not None:
    output_tokens.append(seq_graph_tokens)
if aligned_graph_tokens is not None:
    output_tokens.append(aligned_graph_tokens)
```

然后按 token 数动态设置 `output_proj` 输入维度。

### 5.4 `evaluation/model.py`

必须和 `model.py` 完全同步，否则提交评测时 checkpoint 无法加载。

同步内容：

- `AlignedDenseIntGraphEncoder`
- `PCVRHyFormer.__init__` 新参数
- final head 输入维度变化
- forward/predict 中 aligned graph tokens 的生成和拼接

### 5.5 `evaluation/infer.py`

在 fallback config 加入默认值：

```python
'use_aligned_dense_int_graph': False,
'aligned_graph_fids': '62,63,64,65,66,89,90,91',
'aligned_graph_layers': 1,
'aligned_graph_tokens': 8,
'output_include_ns': False,
```

正常提交时以 `train_config.json` 为准，fallback 只保证旧 checkpoint 可运行。

### 5.6 `run.sh`

推荐主跑配置：

```bash
python3 -u train.py \
  --ns_tokenizer_type rankmixer \
  --user_ns_tokens 5 \
  --item_ns_tokens 2 \
  --num_queries 2 \
  --seq_encoder_type longer \
  --seq_top_k 96 \
  --use_rope \
  --use_token_gnn \
  --token_gnn_layers 2 \
  --token_gnn_layer_scale 0.05 \
  --use_seq_graph \
  --seq_graph_layers 2 \
  --seq_graph_layer_scale 0.08 \
  --graph_output_fusion \
  --output_include_ns \
  --use_aligned_dense_int_graph \
  --aligned_graph_fids 62,63,64,65,66,89,90,91 \
  --aligned_graph_layers 1 \
  --aligned_graph_tokens 8 \
  --emb_skip_threshold 1000000
```

如果显存或速度吃紧，先保留：

```bash
--output_include_ns
--use_seq_graph
```

再开启 aligned graph。

## 6. 推荐实验顺序

不要一次把所有模块全开，否则无法判断增益来源。

| 实验 | 配置 | 目标 |
| --- | --- | --- |
| A | 当前 4LayerGNN | 固定复现 `0.815` |
| B | 2Layer TokenGNN + `output_include_ns` | 检查 final NS 直接入 head 的收益 |
| C | B + SequenceGraphEncoder | 验证行为序列图是否带来主增益 |
| D | C + graph memory summary | 检查 sequence graph summary 对 logit 的贡献 |
| E | D + AlignedDenseIntGraphEncoder | 验证逐元素 int-dense 对齐强信号 |
| F | E + `seq_top_k=128` / `d_model=96` | 容量扩展 |

建议每个实验至少记录：

- validation AUC；
- validation logloss；
- best step；
- 单 epoch 时间；
- 显存；
- 是否出现 NaN 或 OOB 激增。

## 7. 预期收益

较现实的目标：

```text
2LayerGNN:        0.806
4LayerGNN:        0.815
RTG-HyFormer-B/C: 0.820 ~ 0.830
RTG-HyFormer-E:   0.830+，取决于 aligned dense-int 字段真实覆盖率
```

这里不能保证单次训练一定大幅提升，因为 AUC 还受数据切分、seed、验证集大小影响。但从信号强度看，**aligned dense-int graph + target-aware sequence graph** 比继续加深 NS-only GNN 更有机会产生实质增益。

## 8. 风险和防泄漏

- 如果要做全局 item/user/feature 共现图，只能用训练切分构图，不能用验证或测试数据统计 label 后验；
- 如果 parquet row group 是按时间排序，验证集尾部更接近线上，构图时必须只用训练 row groups；
- aligned graph 不要保留过长 list，建议每个 fid top-k 截断到 32 或 64；
- GNN 层数不宜深，`seq_graph_layers=1/2`、`aligned_graph_layers=1` 更稳；
- 如果 AUC 升但 logloss 降不明显，需要看校准；如果 logloss 降但 AUC 不动，继续调 `seq_top_k` 和 head fusion。

## 9. 最小可落地版本

如果只做一版，建议按这个顺序落地：

1. `output_include_ns`：final head 直接拼接 final NS tokens。
2. `SequenceGraphEncoder`：每个行为域 temporal GNN + target-aware message。
3. `AlignedDenseIntGraphEncoder`：只做 `62,63,64,65,66,89,90,91` 这 8 个 fid。
4. `run.sh` 默认启用 2-layer TokenGNN + sequence graph + aligned graph。
5. `evaluation/model.py` 和 `evaluation/infer.py` 同步。

这套方案是在当前 `0.815` 的成功逻辑上继续放大图建模收益：不是盲目加深 GNN，而是把图节点放到真实强关系上，再交给 Transformer 做跨域融合。
