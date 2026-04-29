# RTG-HyFormer 模型结构

## 1. 一句话概括

RTG-HyFormer 是在 PCVRHyFormer 上加入真实数据关系图的版本：用 GNN 先编码局部强关系，再让 Transformer/HyFormer 做跨域全局交互，最后用 query、NS 和 graph memory 一起预测 CVR。

## 2. 总体流程

```text
user/item sparse + dense
        |
        v
NS tokenizer
        |
        v
2-layer TokenGNN
        |
        +----------------------+
                               |
sequence tokens                |
        |                      |
        v                      |
SequenceGraphEncoder           |
        |                      |
        v                      |
graph-enhanced sequence tokens |
        |                      |
        v                      |
HyFormer / Longer blocks <-----+
        |
        v
Q tokens + final NS tokens + sequence graph memory + aligned graph memory
        |
        v
CVR head
```

## 3. 输入模块

### NS Tokens

来源：

- `user_int_feats`
- `item_int_feats`
- `user_dense_feats`
- `item_dense_feats`

默认使用 RankMixer tokenizer：

```text
user_int embeddings -> 5 user NS tokens
item_int embeddings -> 2 item NS tokens
user_dense projection -> 1 dense NS token
```

随后进入 `TokenGNN`：

```text
NS tokens -> 2-layer full graph message passing
```

作用：保留 4LayerGNN 已验证有效的静态字段交互，但减少过平滑。

### Sequence Tokens

四个行为域：

```text
seq_a, seq_b, seq_c, seq_d
```

每个事件位置把多个 side-info embedding 拼接后投影成一个 token，并加 time bucket embedding。

## 4. 图模块

### TokenGNN

位置：

```text
NS tokenizer 后，query generator 前
```

图结构：

```text
NS token complete graph
```

默认配置：

```bash
--token_gnn_layers 2
--token_gnn_layer_scale 0.05
```

### SequenceGraphEncoder

位置：

```text
sequence tokenization 后，HyFormer block 前
```

图节点：

```text
一个历史行为事件 = 一个节点
```

边：

```text
event_i <-> event_{i-1}
event_i <-> event_{i+1}
target item -> event_i
```

输出：

- graph-enhanced sequence tokens；
- 每个行为域一个 sequence graph memory token。

### AlignedDenseIntGraphEncoder

位置：

```text
并行于 NS/sequence 分支，输出 graph memory
```

建模字段：

```text
62, 63, 64, 65, 66, 89, 90, 91
```

节点构造：

```text
node_i = id_embedding(user_int_feats_fid[i])
       + dense_value_projection(user_dense_feats_fid[i])
       + field_embedding(fid)
       + position_embedding(i)
```

然后：

```text
element nodes -> field token -> field-token TokenGNN -> aligned graph memory tokens
```

作用：保留 int list 和 dense list 的逐元素绑定关系，避免 baseline 中 int mean pooling 与 dense projection 分离造成的信息损失。

## 5. HyFormer 主干

每层 `MultiSeqHyFormerBlock` 包含：

```text
1. sequence evolution: Transformer / LongerEncoder
2. query decoding: query cross-attention reads each sequence
3. token fusion: decoded Q tokens + NS tokens -> RankMixer
```

当前主方案使用：

```bash
--seq_encoder_type longer
--seq_top_k 96
--use_rope
```

## 6. 输出 Head

旧 head 主要使用：

```text
final Q tokens
```

RTG-HyFormer 使用：

```text
final Q tokens
+ final NS tokens
+ sequence graph memory tokens
+ aligned dense-int graph memory tokens
```

对应开关：

```bash
--output_include_ns
--graph_output_fusion
--use_aligned_dense_int_graph
```

## 7. 当前默认主配置

```bash
bash run.sh
```

等价核心参数：

```bash
--use_token_gnn
--token_gnn_layers 2
--use_seq_graph
--seq_graph_layers 2
--graph_output_fusion
--output_include_ns
--use_aligned_dense_int_graph
--aligned_graph_tokens 8
--seq_encoder_type longer
--seq_top_k 96
--use_rope
```

## 8. 主要改动文件

| 文件 | 改动 |
| --- | --- |
| `model.py` | 新增 aligned graph、sequence graph、NS head fusion |
| `train.py` | 新增图模块 CLI 和 model args |
| `evaluation/model.py` | 同步训练模型结构 |
| `evaluation/infer.py` | 同步 fallback config 和 schema 参数 |
| `run.sh` | 默认启用 RTG-HyFormer 主方案 |

