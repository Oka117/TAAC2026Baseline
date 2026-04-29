# GNN + Transformer AUC 提升方案

## 1. 背景与目标

当前 4LayerGNN 方案的 AUC 约为 0.806。这个方案的核心是对非序列 NS tokens 做 4 层全连接 TokenGNN，它只能建模 user/item/dense 这些静态 token 之间的关系，对广告转化里最关键的行为序列转移、目标 item 与历史行为匹配关系建模不足。

本次新增方案的目标是：

- 保留现有 `PCVRHyFormer` 主干和 parquet 数据管线，降低工程风险。
- 把 GNN 从“只作用于 NS token”扩展到“行为序列事件图”。
- 让 GNN 输出继续进入 Transformer/HyFormer，并额外把 graph-memory 摘要融合到最终预测头。
- 用可开关的 CLI 参数保证可以做 ablation，而不是一次性不可回退地重写模型。

整体结构：

```text
user/item sparse + dense
        |
        v
NS tokenizer -> optional TokenGNN -> target item context
                                      |
sequence side-info + time bucket      |
        |                             |
        v                             |
sequence tokens -> SequenceGraphEncoder
        |          (temporal GNN + target-aware message passing)
        v
graph-enhanced sequence tokens
        |
        v
Longer/Transformer HyFormer blocks
        |
        + graph memory summaries
        v
CVR head
```

## 2. 为什么这个方案比 4LayerGNN 更有机会提升 AUC

原 4LayerGNN 的图节点是 NS tokens，边是样本内全连接，属于静态 field-level 平滑。它的优点是稳定，但对下面这些强信号利用较弱：

- 历史行为事件之间的相邻转移关系。
- 最近行为与较早行为的变化趋势。
- 目标 item 与历史事件 token 的匹配关系。
- 每个行为域 `seq_a/seq_b/seq_c/seq_d` 的图级摘要。

新增 `SequenceGraphEncoder` 后，每个行为域都会先经过轻量 temporal GNN：

- 节点：该行为域内的每个序列事件 token。
- 边：相邻事件的双向 temporal edges。
- 目标条件：target item NS context 对每个事件发送 gated message。
- 输出：更新后的事件 token 继续给 Transformer；同时产生一个 graph-memory summary 进入最终 head。

这样 GNN 负责局部关系归纳，Transformer/HyFormer 负责长程交互和跨域融合，分工更清晰。

## 3. 已修改文件

### 3.1 `model.py`

新增模块：

- `SequenceGraphLayer`
  - 对单个行为序列做一层消息传递。
  - 每个事件 token 聚合前一个和后一个有效事件 token。
  - 可使用 target item context 做 gated target-aware message。
  - 使用 residual + layer scale，默认 `0.1`，避免训练初期破坏 baseline 表示。

- `SequenceGraphEncoder`
  - 堆叠多层 `SequenceGraphLayer`。
  - 输出更新后的 sequence tokens。
  - 额外生成 graph-memory summary：`mean_pool + last_valid_token -> MLP -> summary token`。

修改 `PCVRHyFormer.__init__`：

- 新增超参：
  - `use_seq_graph`
  - `seq_graph_layers`
  - `seq_graph_layer_scale`
  - `seq_graph_use_target`
  - `graph_output_fusion`
- 记录 item token 在 NS tokens 里的位置：
  - `item_ns_start`
  - `item_ns_end`
- 当 `use_seq_graph=True` 时，为每个 sequence domain 建一个 `SequenceGraphEncoder`。
- 当 `graph_output_fusion=True` 时，最终 `output_proj` 的输入从：

```text
num_queries * num_sequences * d_model
```

扩展为：

```text
(num_queries * num_sequences + num_sequences) * d_model
```

也就是把每个行为域的 graph-memory summary 加入预测头。

新增/修改内部函数：

- `_target_context_from_ns`
  - 从 NS tokens 中提取 target item tokens，并做 mean pooling。
- `_build_sequence_tokens_and_masks`
  - 统一构建 sequence tokens/masks。
  - 如果启用 sequence graph，则在 token 进入 HyFormer 之前完成 temporal GNN 增强。
- `_run_multi_seq_blocks`
  - 增加 `graph_summaries` 参数。
  - 在最终 head 前拼接 `all_q` 和 graph-memory summaries。
- `forward` / `predict`
  - 改为复用 `_build_sequence_tokens_and_masks`，保证训练和推理一致。

### 3.2 `train.py`

新增 CLI 参数：

```bash
--use_seq_graph
--seq_graph_layers
--seq_graph_layer_scale
--seq_graph_use_target
--no_seq_graph_target
--graph_output_fusion
--no_graph_output_fusion
```

并把这些参数加入 `model_args` 和训练日志，checkpoint 的 `train_config.json` 会记录完整结构超参。

### 3.3 `evaluation/model.py`

已同步 `model.py`，确保评测容器构建模型时包含同样的 `SequenceGraphEncoder` 结构。

### 3.4 `evaluation/infer.py`

在 `_FALLBACK_MODEL_CFG` 中新增 sequence graph 相关默认值：

```python
'use_seq_graph': False,
'seq_graph_layers': 2,
'seq_graph_layer_scale': 0.1,
'seq_graph_use_target': True,
'graph_output_fusion': True,
```

正常情况下评测会优先读取 checkpoint 里的 `train_config.json`；fallback 只用于缺少配置文件的旧 checkpoint。

### 3.5 `run.sh`

默认运行配置已经从旧的 4LayerGNN-NS 调整为 GNN + Transformer：

```bash
bash run.sh
```

核心新增/调整参数：

```bash
--seq_encoder_type longer
--seq_top_k 96
--use_rope
--use_token_gnn
--token_gnn_layers 2
--token_gnn_layer_scale 0.05
--use_seq_graph
--seq_graph_layers 2
--seq_graph_layer_scale 0.1
--graph_output_fusion
```

说明：

- `TokenGNN` 从 4 层降到 2 层：让它作为静态 NS token 辅助增强，减少过平滑。
- `SequenceGraphEncoder` 负责主要增益来源：行为事件转移图和 target-aware sequence graph。
- `longer + seq_top_k=96`：保留较多最近行为，同时降低长序列 Transformer 成本。
- `use_rope`：增强序列位置建模。

## 4. 推荐实验顺序

### 实验 A：旧方案复现

用于确认当前 0.806 AUC 的训练环境、数据切分、seed、提交流程没有变化。

```bash
python3 -u train.py \
  --ns_tokenizer_type rankmixer \
  --user_ns_tokens 5 \
  --item_ns_tokens 2 \
  --num_queries 2 \
  --use_token_gnn \
  --token_gnn_layers 4 \
  --token_gnn_layer_scale 0.1 \
  --ns_groups_json "" \
  --emb_skip_threshold 1000000
```

### 实验 B：新增 GNN + Transformer 主方案

```bash
bash run.sh
```

这是当前建议主跑版本。

### 实验 C：只看 SequenceGraphEncoder 增益

关闭 NS TokenGNN，只保留序列图：

```bash
bash run.sh --token_gnn_layers 0
```

如果这个版本优于旧 4LayerGNN，说明 AUC 增益主要来自行为序列图。

### 实验 D：检查 graph-memory summary 是否有效

```bash
bash run.sh --no_graph_output_fusion
```

如果 AUC 明显下降，说明 graph summary 对最终排序有帮助。

### 实验 E：调大容量

在显存允许时尝试：

```bash
bash run.sh --d_model 96 --num_heads 4 --seq_top_k 128 --dropout_rate 0.02
```

如果 `rank_mixer_mode=full` 因 `d_model % T` 不满足而报错，可以改：

```bash
bash run.sh --d_model 96 --rank_mixer_mode ffn_only
```

## 5. 预期收益与风险

预期收益来源：

- sequence graph 显式建模行为转移，比 NS-only GNN 更贴近 CVR 信号。
- target-aware message 让历史行为 token 在进入 Transformer 前已经感知目标 item。
- graph-memory summary 让最终 head 直接看到每个行为域的图级信息。
- RoPE + LongerEncoder 对长序列近期行为更友好。

主要风险：

- 训练耗时会比旧 4LayerGNN 略高。
- 如果验证集很小，AUC 波动可能掩盖真实增益，需要至少固定 seed 跑 2 到 3 次。
- 如果过拟合，优先调小：
  - `seq_graph_layers=1`
  - `seq_graph_layer_scale=0.05`
  - `dropout_rate=0.02~0.05`

## 6. 判断是否真的提升

每次实验至少记录：

- validation AUC
- validation logloss
- best checkpoint step
- 训练吞吐
- 显存占用
- 是否出现 NaN warning

建议优先比较：

```text
旧 4LayerGNN AUC: 0.806
新 GNN+Transformer AUC: 目标至少 +0.01，理想 +0.02 以上
```

如果新方案 AUC 只小幅提升但 logloss 明显下降，可以继续调 `seq_top_k`、`dropout_rate`、`seq_graph_layer_scale`，不要过早放弃。
