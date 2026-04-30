# 实验结论

## 1. 实验结果

本轮对比了两个方案：

| 方案 | AUC | Inference Time |
| --- | ---: | ---: |
| 当前方案：4LayerGNN + `output_include_ns` | 0.811474 | 170.22s |
| GNN_NS_4Layer | 0.815064 | 311.65s |

结论：

```text
当前方案 AUC 低于 GNN_NS_4Layer，未带来正向收益。
```

虽然当前方案推理时间更短，但 AUC 从 `0.815064` 降到 `0.811474`，说明 `output_include_ns` 不是稳定增益点，至少在当前训练配置和 checkpoint 选择下不应作为主方案。

## 2. 当前方案配置

当前主跑配置核心为：

```bash
--ns_tokenizer_type rankmixer
--user_ns_tokens 5
--item_ns_tokens 2
--num_queries 2
--seq_encoder_type transformer
--use_token_gnn
--token_gnn_layers 4
--token_gnn_layer_scale 0.1
--output_include_ns
--emb_skip_threshold 1000000
```

相对 GNN_NS_4Layer，主要新增变量是：

```text
output_include_ns=True
```

也就是 final head 除了 final Q tokens，还直接拼接 final NS tokens。

## 3. 推理时间减少的原因

推理时间从 `311.65s` 降到 `170.22s`，更可能来自 evaluation 运行路径和平台状态，而不是模型结构本身显著变轻。

主要原因：

1. 当前 evaluation 已经修复了 timeout 问题，DataLoader 使用多 worker 高吞吐模式。
2. checkpoint 搜索去掉了深递归，只查当前层和一层 `global_step*`，启动开销更小。
3. `output_include_ns` 只增加 final head 输入，不会显著增加模型主体推理计算量。
4. 平台推理时间受 parquet IO、worker 启动、缓存、机器负载影响较大，170s 和 311s 不应完全归因于模型结构。

因此：

```text
推理时间减少是好现象，但不能证明当前模型结构更优。
```

## 4. AUC 下降的原因分析

### 4.1 `output_include_ns` 可能是负增益

GNN_NS_4Layer 的 final head 主要使用 final Q tokens。NS tokens 已经通过 RankMixer 与 Q tokens 多层交互，间接影响预测。

当前方案直接把 final NS tokens 拼进输出 head：

```text
final Q tokens + final NS tokens -> output head
```

这可能导致 head 过度依赖静态 user/item/tabular 信息，削弱 query 从序列中读出的排序信号。

### 4.2 head 输入维度增大，训练配置未调整

原 head 输入约为：

```text
num_queries * num_sequences * d_model = 2 * 4 * 64 = 512
```

加入 NS tokens 后，输入约变成：

```text
(8 + num_ns) * 64
```

head 参数增加，但训练学习率、dropout、early stopping 等没有专门调整，容易造成验证 AUC 波动或过拟合。

### 4.3 final NS tokens 不一定是干净强特征

final NS tokens 不是原始 user/item 特征，而是经过 TokenGNN 和 RankMixer 多层混合后的 token。

直接拼接这些 token 可能带来噪声：

```text
NS token 原始强信号
  -> TokenGNN 平滑
  -> RankMixer 融合
  -> final NS token
  -> 直接进 head
```

这个路径不一定比只用 final Q tokens 更稳定。

### 4.4 Early stopping 更激进

当前 `patience=3`，如果 GNN_NS_4Layer 结果来自更长 patience 或不同 checkpoint 选择，AUC 差距可能部分来自早停时机。

当前差距：

```text
0.815064 - 0.811474 = 0.003590
```

这个量级可能受 checkpoint step、seed、验证波动影响。

## 5. 结论判断

本轮实验可以得出：

```text
output_include_ns 不能作为当前主方案保留。
```

它没有在 GNN_NS_4Layer 基础上提升 AUC，反而降低到 `0.811474`。

更稳的主线应恢复：

```bash
--use_token_gnn
--token_gnn_layers 4
--token_gnn_layer_scale 0.1
# 不启用 output_include_ns
```

也就是回到 GNN_NS_4Layer。

## 6. 下一步建议

### 6.1 立即回退主方案

主跑配置建议改回：

```bash
--ns_tokenizer_type rankmixer
--user_ns_tokens 5
--item_ns_tokens 2
--num_queries 2
--seq_encoder_type transformer
--use_token_gnn
--token_gnn_layers 4
--token_gnn_layer_scale 0.1
--ns_groups_json ""
--emb_skip_threshold 1000000
```

关闭：

```text
output_include_ns
use_seq_graph
use_aligned_dense_int_graph
seq_encoder_type=longer
use_rope
```

### 6.2 如果继续尝试 NS head，改成 gated fusion

不要直接拼接 NS tokens。建议改成：

```text
q_repr = output_proj(Q tokens)
ns_repr = ns_proj(final NS tokens)
gate = sigmoid(gate_proj([q_repr, ns_repr]))
final = q_repr + small_init_gate * ns_repr
```

关键点：

- gate 初始化接近 0；
- 保证模型初始行为接近 GNN_NS_4Layer；
- 让 NS direct path 只在训练确认有用时逐步加入。

### 6.3 保持 evaluation 高吞吐修改

当前 inference time 降低，说明 evaluation 侧高吞吐修复有价值。建议保留：

- 多 worker DataLoader；
- 非递归 checkpoint 搜索；
- 不做大规模 user_id 去重统计；
- checkpoint 目录兼容逻辑。

## 7. 最终结论

当前最优稳定方案仍是：

```text
GNN_NS_4Layer
```

当前 `4LayerGNN + output_include_ns` 的实验结果证明：

```text
final NS tokens 直接拼 head 会降低 AUC，不建议保留为主方案。
```

后续优化应以 GNN_NS_4Layer 为基线，采用小步 ablation，而不是一次性加入多个新模块。
