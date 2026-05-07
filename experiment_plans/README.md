# TAAC2026 Baseline 实验计划目录

本目录集中保存基于 `feature-engineering.docx` 的实验设计、参数和预期结果。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `FE00/README.md` | FE-00 子目录索引、代码文件和上传清单 |
| `FE00/experiment_00_preprocess_alignment_plan.zh.md` | FE-00 缺失处理与归一化对齐 |
| `FE01/README.md` | FE-01 子目录索引、代码文件和上传清单 |
| `FE01/experiment_01_feature_engineering_plan.zh.md` | FE-01 安全特征增强方案 |
| `FE01A/README.md` | FE-01A total frequency 消融实验索引 |
| `FE01A/experiment_01a_total_frequency_plan.zh.md` | FE-01A total frequency only 方案 |
| `FE01B/README.md` | FE-01B target match 消融实验索引 |
| `FE01B/experiment_01b_target_match_plan.zh.md` | FE-01B target-history match only 方案 |
| `FE02/README.md` | FE-02 子目录索引、代码文件和上传清单 |
| `FE02/experiment_02_delay_history_features_plan.zh.md` | FE-02 历史 delay 特征方案 |
| `FE07_P012_DOMAIN_DOCX/README.md` | FE-07 P0/P1/P2-Domain 强对齐方案索引 |
| `FE07_P012_DOMAIN_DOCX/experiment_07_fe00_01ab_claude_p012_domain_docx_alignment_plan.zh.md` | FE00 + 01AB + Claude P0/P1/P2-Domain 与上传 DOCX 的强关联实验方案 |
| `FE07_P012_DOMAIN_DOCX/experiment_07_strong_check_and_prediction.zh.md` | FE-07 强力检查、模块级提分点和实验结果预测 |
| `FE08_MAY7_GNN/README.md` | FE-08 May7 GNN 强对齐方案索引 |
| `FE08_MAY7_GNN/experiment_08_may7_gnn_feature_alignment_plan.zh.md` | 严格按 Claude 5 月 7 日 FE08 方案整理的 GNN + 验证特征主实验方案 |
| `FE08_MAY7_GNN/experiment_08_strong_check_and_prediction.zh.md` | FE-08 强力检查、代码就绪度、模块级预测和回滚规则 |
| `FE08_MAY7_GNN/experiment_08_claude_alignment_audit.zh.md` | FE-08 与 Claude 5 月 7 日方案逐项对齐审计和对应联系点 |
| `feature_engineering_design_alignment_audit.zh.md` | 上传 DOCX 与实验计划逐条对齐审计 |
| `baseline_feature_engineering_modification_plan.zh.md` | 总体 baseline 修改方向 |
| `experiment_03_delay_weighted_loss_plan.zh.md` | FE-03 delay-aware weighted loss 方案 |
| `experiment_04_multitask_plan.zh.md` | FE-04 conversion + delay bucket + engagement multi-task 方案 |
| `experiment_result_predictions.zh.md` | 实验结构、参数和结果预测总表 |

## 推荐执行顺序

```text
B0 baseline
FE-00 missing handling + dense normalization alignment
FE-01 safe frequency + match features
FE-01A total frequency only ablation
FE-01B target-history match only ablation
FE-02 historical delay features
FE-07 FE00 + 01AB + Claude P0/P1/P2-Domain strong DOCX alignment
FE-08 May7 GNN + verified feature return path
FE-03 delay-aware weighted loss
FE-04 multi-task conversion + delay bucket + engagement
```

原则：每轮只引入一个主要变量，保留上一轮最优配置作为新对照。
