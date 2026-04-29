# RTG-HyFormer 主跑方案与结果预测

## 1. 主跑模型参数

当前主跑模型使用以下参数：

```bash
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

`run.sh` 额外保留两个工程参数：

```bash
--ns_groups_json ""
--num_workers 8
```

其中 `--ns_groups_json ""` 保持当前 RankMixer 的 singleton feature 输入顺序，不启用示例分组文件；`--num_workers 8` 只影响 dataloader 吞吐。

## 2. 一句话方案

本方案把当前已经有效的 GNN 思路从 **NS token 静态关系** 扩展到 **行为序列事件图** 和 **aligned dense-int element 图**，再把 graph memory 与 final NS tokens 一起送入预测头，目标是在 `4LayerGNN=0.815` 的基础上继续提升 AUC。

## 3. 模型结构

```text
user/item sparse + user dense
        |
        v
RankMixer NS tokenizer
        |
        v
2-layer TokenGNN
        |
        +------------------------------+
                                       |
sequence event tokens                  |
        |                              |
        v                              |
target-aware SequenceGraphEncoder      |
        |                              |
        v                              |
graph-enhanced sequence tokens         |
        |                              |
        v                              |
LongerEncoder + RoPE HyFormer blocks <-+
        |
        v
final Q tokens
+ final NS tokens
+ sequence graph memory
+ aligned dense-int graph memory
        |
        v
CVR logit
```

## 4. 参数设计逻辑

### 4.1 RankMixer NS tokenization

```bash
--ns_tokenizer_type rankmixer
--user_ns_tokens 5
--item_ns_tokens 2
--num_queries 2
```

使用较少 NS tokens 控制 token 数，保证 `d_model=64` 下 RankMixer full mode 仍然可用。`num_queries=2` 让每个行为域有两个 query，从序列里读取不同角度的信息。

### 4.2 Longer + RoPE

```bash
--seq_encoder_type longer
--seq_top_k 96
--use_rope
```

`longer` 保留更长的近期行为窗口，`seq_top_k=96` 比默认 50 更适合广告行为序列，RoPE 用于增强事件位置建模。

### 4.3 TokenGNN 从 4 层降到 2 层

```bash
--use_token_gnn
--token_gnn_layers 2
--token_gnn_layer_scale 0.05
```

4LayerGNN 已经证明 NS 图有效，但继续加深容易让少量 NS tokens 过平滑。这里保留 2 层作为静态字段交互模块，把主要增益空间留给序列图和 aligned graph。

### 4.4 Target-aware sequence graph

```bash
--use_seq_graph
--seq_graph_layers 2
--seq_graph_layer_scale 0.08
--graph_output_fusion
```

每个行为域先做 temporal GNN：事件节点从相邻事件和 target item context 接收消息。GNN 后的 sequence tokens 继续进入 HyFormer，同时每个行为域输出一个 graph memory token 进入 final head。

### 4.5 Final NS 直接入 head

```bash
--output_include_ns
```

旧模型主要用 final Q tokens 预测，NS tokens 只通过中间 RankMixer 间接影响输出。CVR 里用户、物料、上下文是强 tabular signal，所以 final head 直接拼接 final NS tokens，通常是低风险增益点。

### 4.6 Aligned dense-int graph

```bash
--use_aligned_dense_int_graph
--aligned_graph_fids 62,63,64,65,66,89,90,91
--aligned_graph_layers 1
--aligned_graph_tokens 8
```

这些字段的 `user_int_feats_fid[i]` 与 `user_dense_feats_fid[i]` 是逐元素对齐关系。当前 baseline 把 int list pooling 和 dense projection 分开，会损失 `(id_i, value_i)` 的绑定信息。Aligned graph 把每个 pair 构造成 element node，再聚合为 graph memory tokens。

## 5. 当前代码改动位置

| 文件 | 关键改动 |
| --- | --- |
| `model.py` | `SequenceGraphEncoder`、`AlignedDenseIntGraphEncoder`、final head fusion |
| `train.py` | 新增 sequence graph、aligned graph、NS head fusion 参数 |
| `evaluation/model.py` | 与 `model.py` 同步，保证评测加载 checkpoint |
| `evaluation/infer.py` | fallback config 增加新结构参数，构建模型时传入 dense/int schema |
| `run.sh` | 默认使用本主跑配置 |

## 6. 结果预测

已知基线：

```text
2LayerGNN AUC: 0.806
4LayerGNN AUC: 0.815
```

对本主跑方案的预测：

| 情况 | 预期 AUC | 判断 |
| --- | --- | --- |
| 保守 | 0.817 - 0.821 | NS head 或 sequence graph 有轻微收益，但 aligned graph 覆盖率/稳定性一般 |
| 正常 | 0.822 - 0.830 | sequence graph 与 final NS fusion 生效，aligned graph 有正贡献 |
| 理想 | 0.830+ | aligned dense-int 字段覆盖率高，且验证切分能稳定反映该信号 |

本方案最可能带来增益的顺序：

```text
output_include_ns
> SequenceGraphEncoder + graph_output_fusion
> AlignedDenseIntGraphEncoder
> Longer top_k=96 + RoPE
```

## 7. 风险预测

如果 AUC 没有超过 `0.815`，优先看这几个点：

- `seq_graph_layer_scale=0.08` 可能偏强，可降到 `0.05`；
- aligned graph 可能带来过拟合，可尝试 `--aligned_graph_tokens 4`；
- `seq_top_k=96` 可能增加噪声，可对比 `64` 和 `128`；
- 验证集波动较大时，需要固定 seed 跑 2 到 3 次。

## 8. 推荐记录模板

每次训练后记录：

```text
exp_name=rtg-main
seed=42
best_auc=
best_logloss=
best_step=
epoch_time=
gpu_memory=
checkpoint=
notes=
```

如果本主跑达到 `0.822+`，下一步优先尝试：

```bash
bash run.sh --seq_top_k 128 --dropout_rate 0.02
```

如果达到 `0.830` 附近，再考虑：

```bash
bash run.sh --d_model 96 --rank_mixer_mode ffn_only --seq_top_k 128 --dropout_rate 0.02
```
