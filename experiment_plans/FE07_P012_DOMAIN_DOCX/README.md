# FE-07 P0/P1/P2-Domain 强对齐方案索引

FE-07 是在当前 `FE_00_01AB_P0_1_2domain` 分支语义上整理出的下一版实验方案。

它以已经落账的 `FE-00`、`FE-01A`、`FE-01B` 为基础，把 Claude 文档中的 P0、P1 和 P2-Domain 路线合并成一条可消融主线，并显式展示每个提升方法与上传 `feature-engineering.docx` 的关联。

| 文件 | 作用 |
| --- | --- |
| `experiment_07_fe00_01ab_claude_p012_domain_docx_alignment_plan.zh.md` | FE-07 主实验方案、git history 回顾、DOCX 对齐矩阵、ablation 和验收标准 |
| `experiment_07_strong_check_and_prediction.zh.md` | FE-07 强力检查、当前代码就绪度、模块级提分点和实验结果预测 |

## 代码入口

| 文件 | 作用 |
| --- | --- |
| `build_fe07_p012_domain_dataset.py` | 生成 FE07 增强 parquet/schema/ns_groups/stats/domain bucket sidecar |
| `tools/build_fe07_p012_domain_dataset.py` | 平台工具入口 |
| `run_fe07_p012_domain.sh` | 一键构建 FE07 数据并训练，无 GNN |
| `evaluation/FE07/infer.py` | FE07 raw eval transform + strict checkpoint inference |

一句话定义：

```text
FE-07 = FE00 + FE01AB-safe + Claude P0 + Claude P1 + P2-Domain
```

本实验不加入 GNN / TokenGNN 结构；GNN 相关内容只作为历史结果背景，不进入 FE-07 ablation。
