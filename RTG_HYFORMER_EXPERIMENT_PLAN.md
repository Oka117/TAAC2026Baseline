# RTG-HyFormer 实验计划

## 1. 实验目标

当前已知结果：

```text
2LayerGNN AUC: 0.806
4LayerGNN AUC: 0.815
```

本轮实验目标是验证新版 **RTG-HyFormer** 是否能继续提升 AUC。核心判断不是单纯看 GNN 层数，而是验证三类新增信号：

- final head 直接使用最终 NS tokens；
- target-aware sequence graph 是否增强行为序列；
- aligned dense-int graph 是否利用了 `62,63,64,65,66,89,90,91` 的逐元素强信号。

## 2. 当前主跑配置

直接运行：

```bash
bash run.sh
```

当前 `run.sh` 默认启用：

```bash
--seq_encoder_type longer
--seq_top_k 96
--use_rope
--use_token_gnn
--token_gnn_layers 2
--token_gnn_layer_scale 0.05
--use_seq_graph
--seq_graph_layers 2
--seq_graph_layer_scale 0.08
--graph_output_fusion
--output_include_ns
--use_aligned_dense_int_graph
--aligned_graph_fids 62,63,64,65,66,89,90,91
--aligned_graph_layers 1
--aligned_graph_tokens 8
--aligned_graph_top_k 64
```

## 3. 实验顺序

建议按下面顺序跑，不要一次只看最终主方案。这样能知道 AUC 提升来自哪里。

| 编号 | 实验 | 命令差异 | 目的 |
| --- | --- | --- | --- |
| A | 4LayerGNN 复现 | 关闭 seq graph、aligned graph、NS head fusion，TokenGNN=4 | 确认 `0.815` 可复现 |
| B | 2LayerGNN + NS head | `--token_gnn_layers 2 --output_include_ns` | 验证 final NS tokens 是否直接提分 |
| C | B + sequence graph | 加 `--use_seq_graph --graph_output_fusion` | 验证行为序列图 |
| D | C + aligned graph | 加 `--use_aligned_dense_int_graph` | 验证逐元素 dense-int 图 |
| E | 主方案 | `bash run.sh` | 冲最高 AUC |
| F | 容量扩展 | `bash run.sh --seq_top_k 128 --dropout_rate 0.02` | 显存允许时继续冲分 |

## 4. 推荐命令

### A. 4LayerGNN 复现

这个实验不要用 `bash run.sh`，因为 `run.sh` 已经默认启用了 sequence graph。要完全复现旧 4LayerGNN，直接运行：

```bash
python3 -u train.py \
  --ns_tokenizer_type rankmixer \
  --user_ns_tokens 5 \
  --item_ns_tokens 2 \
  --num_queries 2 \
  --use_token_gnn \
  --token_gnn_layers 4 \
  --token_gnn_graph full \
  --token_gnn_layer_scale 0.1 \
  --ns_groups_json "" \
  --emb_skip_threshold 1000000 \
  --num_workers 8
```

### B. 2LayerGNN + NS Head

```bash
python3 -u train.py \
  --ns_tokenizer_type rankmixer \
  --user_ns_tokens 5 \
  --item_ns_tokens 2 \
  --num_queries 2 \
  --use_token_gnn \
  --token_gnn_layers 2 \
  --token_gnn_layer_scale 0.05 \
  --output_include_ns \
  --ns_groups_json "" \
  --emb_skip_threshold 1000000 \
  --num_workers 8
```

### C. 加 Sequence Graph

```bash
bash run.sh \
  --no_output_include_ns \
  --aligned_graph_tokens 0
```

用于单独观察 sequence graph 的收益。

### D. 加 Aligned Graph

```bash
bash run.sh
```

这是当前主方案。

### F. 容量扩展

```bash
bash run.sh \
  --seq_top_k 128 \
  --dropout_rate 0.02
```

如果显存足够，再尝试：

```bash
bash run.sh \
  --d_model 96 \
  --rank_mixer_mode ffn_only \
  --seq_top_k 128 \
  --dropout_rate 0.02
```

## 5. 每次实验必须记录

| 字段 | 说明 |
| --- | --- |
| config name | 例如 `E-main-rtg` |
| seed | 默认 `42`，最好固定 |
| best validation AUC | 主指标 |
| validation logloss | 判断校准和稳定性 |
| best step / epoch | 判断是否过早过拟合 |
| train time / epoch | 判断额外图模块成本 |
| GPU memory | 判断是否能扩容 |
| checkpoint path | 方便提交和复查 |

建议记录格式：

```text
exp=E-main-rtg
seed=42
best_auc=
best_logloss=
best_step=
notes=
```

## 6. 预期判断

合理预期：

```text
A: 0.815 附近
B: 如果 NS 强，AUC 应该小幅提升或更稳
C: 如果行为序列有效，AUC 应该继续提升
D/E: aligned dense-int 覆盖率高时最可能冲高
```

如果主方案 AUC 不升，优先排查：

- aligned graph 是否真的启用，日志里不应出现 aligned specs missing；
- `seq_graph_layer_scale=0.08` 是否过强，可降到 `0.05`；
- `aligned_graph_top_k=64` 是否过慢或过拟合，可降到 `32`；
- 验证集是否太小，建议固定 seed 跑 2 到 3 次。

## 7. 下一步调参优先级

优先级从高到低：

1. `seq_top_k`: `96 -> 128`
2. `seq_graph_layer_scale`: `0.08 -> 0.05 / 0.1`
3. `aligned_graph_top_k`: `64 -> 32 / 96`
4. `aligned_graph_tokens`: `8 -> 4 / 12`
5. `dropout_rate`: `0.01 -> 0.02 / 0.05`
6. `d_model`: `64 -> 96`，必要时 `rank_mixer_mode=ffn_only`
