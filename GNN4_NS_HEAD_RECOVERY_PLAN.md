# 4LayerGNN + NS Head 恢复实验

## 1. 背景

上一个 RTG-HyFormer 全开方案 AUC 只有 `0.806`，低于已验证的 `4LayerGNN=0.815`。主要原因是一次性改动太多，并且把已经有效的 `token_gnn_layers=4` 降成了 2。

本轮实验目标是回到已验证强基线，只测试一个低风险新增点：

```text
4LayerGNN + final NS tokens directly into output head
```

## 2. 主跑配置

`run.sh` 当前使用：

```bash
--ns_tokenizer_type rankmixer \
--user_ns_tokens 5 \
--item_ns_tokens 2 \
--num_queries 2 \
--seq_encoder_type transformer \
--use_token_gnn \
--token_gnn_layers 4 \
--token_gnn_layer_scale 0.1 \
--output_include_ns \
--ns_groups_json "" \
--emb_skip_threshold 1000000 \
--num_workers 8
```

## 3. 相对失败方案的变化

关闭这些未验证变量：

```text
seq_encoder_type=longer
seq_top_k=96
use_rope
use_seq_graph
graph_output_fusion
use_aligned_dense_int_graph
```

恢复这些已验证变量：

```text
seq_encoder_type=transformer
token_gnn_layers=4
token_gnn_layer_scale=0.1
```

只保留一个新增变量：

```text
output_include_ns=True
```

## 4. 结果预测

已知：

```text
2LayerGNN: 0.806
4LayerGNN: 0.815
RTG 全开方案: 0.806
```

本实验预期：

| 情况 | AUC 预测 | 解释 |
| --- | --- | --- |
| 不生效 | 0.812 - 0.816 | NS head 没有贡献，但 4LayerGNN 主干应恢复大部分收益 |
| 正常 | 0.816 - 0.820 | final NS tokens 直接入 head 有小幅正收益 |
| 理想 | 0.820+ | NS tokens 是强 tabular signal，直接进 head 明显改善排序 |

判断标准：

- 如果 AUC 回到 `0.815` 附近，说明上一个方案主要是新增模块/longer 路径拖累；
- 如果 AUC 超过 `0.815`，说明 `output_include_ns` 是可保留增益；
- 如果 AUC 仍然只有 `0.806` 左右，需要检查代码改动是否影响了 4LayerGNN 默认路径。

## 5. 下一步

如果本实验 `>0.815`，下一步建议只加一个变量：

```bash
bash run.sh --use_seq_graph --seq_graph_layers 1 --seq_graph_layer_scale 0.03 --graph_output_fusion
```

不要同时打开 `longer` 和 `aligned graph`。先确认 sequence graph 单独是否有正收益。
