# FE-08 May7 GNN 强对齐方案索引

FE-08 是严格按照 Claude `5 月 7 日` 两份文档整理出的下一版主线实验方案：

```text
FE08-May7-main =
  0.8159 GNN baseline
  + FE07/FE01B 已验证安全数据层信号
  + item_dense {86,91,92} token + normalization
  + sequence recency sort
  + item_int 89/90/91 + user_int 130/131
  + rank_mixer_mode=full + d_model=136
  + 4-layer TokenGNN, full graph, layer_scale=0.15
```

本目录只保存实验方案、强检查、预测和代码搭建约束；代码实现应另按本文档执行。

## Source Of Truth

| 文件 | 作用 |
| --- | --- |
| `../Claude/5月7日_GNN结合验证特征_保持NS影响域_方案.md` | FE-08 主方案，定义最终结构、偏差落点、AUC 目标和强检查原则 |
| `../Claude/5月7日_FE08代码结构_供AI_Agent搭建.md` | FE-08 代码结构搭建指南，定义新增/修改文件、builder/eval/run 入口和 sidecar 合约 |
| `../Claude/Baseline与数据联合分析-提分路线.md` | FE-08 的数据层动机来源，解释 baseline 丢失哪些信号 |

执行时若文档内部存在旧 checklist 残留，以 `5月7日_FE08代码结构_供AI_Agent搭建.md` 中的代码清单和主方案结论为准：

```text
rank_mixer_mode = full
d_model = 136
seq_encoder_type = transformer
seq_top_k = 100 marker only
item_dense_fids = 86,91,92
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `experiment_08_may7_gnn_feature_alignment_plan.zh.md` | FE-08 主实验方案、代码搭建结构、训练参数、eval 设计和消融顺序 |
| `experiment_08_strong_check_and_prediction.zh.md` | FE-08 强力检查、代码就绪度、模块级预测、失败场景和回滚规则 |
| `experiment_08_claude_alignment_audit.zh.md` | FE-08 方案与 Claude FE08 文档逐项对齐审计，标注对应联系点 |

## 代码入口规划

| 文件 | 作用 |
| --- | --- |
| `build_fe08_may7_dataset.py` | 训练侧 FE08 builder，从 FE07 builder fork |
| `tools/build_fe08_may7_dataset.py` | 平台上传副本 |
| `run_fe08_may7_full.sh` | 一键构建 FE08 数据并训练 |
| `evaluation/FE08/build_fe08_may7_dataset.py` | eval-side transform builder，只 transform，不 fit |
| `evaluation/FE08/dataset.py` | FE08 eval dataset wrapper，从 FE07/dataset.py fork |
| `evaluation/FE08/model.py` | FE08 eval model，必须支持 TokenGNN |
| `evaluation/FE08/infer.py` | FE08 strict checkpoint inference |

## 一句话定义

```text
FE-08 = GNN4 validated backbone + FE07 safe feature builder
        + May7 locked feature additions + strict train/eval sidecar parity.
```

## 推荐执行顺序

```text
0. GNN-baseline replay
1. + drop >80% missing
2. + item_dense token + norm, full + d_model=136
3. + sequence recency sort
4. + item_int 89/90/91 + user_int 130/131
5. + seq_max_lens 256/256/128/512
6. + dropout=0.05 + seq_top_k=100 marker
6.B optional: switch seq_encoder_type=longer so seq_top_k becomes active
```

第一轮主线直接跑 step 6；若结果低于 `0.8159`，按上述顺序回拆定位。
