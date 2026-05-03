# Feature Engineering 设计强对齐审计

本文以 `/Users/gaogang/Downloads/feature-engineering.docx` 为唯一 source of truth，对当前实验计划和 FE-01 代码做逐条核对。

## 1. Source Of Truth 摘要

上传 DOCX 明确包含以下设计点：

| DOCX 行号 | 设计点 |
| ---: | --- |
| P000 | 删除 missing value proportion `>75%` 的 user int feature |
| P001 | int feature missing value 使用 average value 替换 |
| P002 | 所有 dense numerical feature，包括 sequence side feature，做 normalization |
| P008 | `user_dense_feats_110 = log(1 + user_total_frequency)` |
| P010 | `item_dense_feats_86 = log(1 + item_total_frequency)` |
| P012 | `user_dense_feats_111 = log(1 + user_purchase_frequency)` |
| P014 | `item_dense_feats_87 = log(1 + item_purchase_frequency)` |
| P017 | `delay = timestamp - label_time` |
| Table 0 | `user_dense_feats_112 = log(1 + user_avg_delay)` |
| Table 0 | `item_dense_feats_88 = log(1 + item_avg_delay)` |
| P020 | `Item_int_feats_89 = has_match(item_int_feats_9, domain_d_seq_19)` |
| P021 | `Item_int_feats_90 = match_count(item_int_feats_9, domain_d_seq_19)` |
| P022 | `item_dense_feats_91 = log(1 + min_match_delta)` |
| P023 | `item_dense_feats_92 = log(1 + match_count_7d)` |
| P024 | 其他 item field 与 sequence field 组合按完整训练集 lift 排序筛选 |
| P025-P057 | user/item NS groups |
| P064-P078 | Delay-Aware Weighted Loss |
| P080-P089 | Multi-Task Loss：conversion、delay bucket、engagement |

## 2. 当前实验覆盖矩阵

| DOCX 设计点 | 当前对应文件 | 状态 | 说明 |
| --- | --- | --- | --- |
| 删除缺失率 `>75%` user int | `baseline_feature_engineering_modification_plan.zh.md`，`experiment_00_preprocess_alignment_plan.zh.md` | 已覆盖 | FE-00 作为预处理/统计实验 |
| int missing average fill | `experiment_00_preprocess_alignment_plan.zh.md` | 已覆盖但需 baseline 适配 | DOCX 原文保留；baseline 中匿名 id 不直接 raw average，采用可复现兼容策略 |
| dense numerical normalization | `experiment_00_preprocess_alignment_plan.zh.md`，`tools/build_feature_engineering_dataset.py` | 已覆盖 | FE-01 已对新增 dense 做 `log1p + zscore` |
| `user_dense_feats_110/111` | `tools/build_feature_engineering_dataset.py`，FE-01 | 已实现 | prefix frequency，不包含当前样本 |
| `item_dense_feats_86/87` | `tools/build_feature_engineering_dataset.py`，FE-01 | 已实现 | 依赖 `dataset.py` item dense 支持 |
| `user_dense_feats_112` | FE-02 | 已规划 | 历史 prefix avg delay |
| `item_dense_feats_88` | FE-02 | 已规划 | 历史 prefix avg delay |
| `item_int_feats_89` | `tools/build_feature_engineering_dataset.py`，FE-01 | 已实现 | 编码为 0/1/2 |
| `item_int_feats_90` | `tools/build_feature_engineering_dataset.py`，FE-01 | 已实现但 baseline 适配 | DOCX 为 raw `match_count`；baseline 用 Embedding，故分桶为 categorical count |
| `item_dense_feats_91/92` | `tools/build_feature_engineering_dataset.py`，FE-01 | 已实现 | 使用 domain_d sequence timestamp 计算 |
| lift 筛选其他 pair | `experiment_00_preprocess_alignment_plan.zh.md` | 已规划 | 不在 FE-01 直接扩展 pair，避免偏离 DOCX 第一组 |
| NS groups | `ns_groups.feature_engineering.json`，FE-01 | 已实现但 baseline 适配 | dense fid 不写入 int-only NS groups，通过 dense token 进入 |
| Delay-aware weighted loss | FE-03 | 已规划 | 使用 `delay = timestamp - label_time` |
| Multi-task conversion/delay/engagement | FE-04 | 已规划 | 已补齐 engagement head/label 设计 |

## 3. 已修正偏差

| 偏差 | 修正 |
| --- | --- |
| FE-03 序列窗口实验不属于上传 DOCX | 已从核心实验计划删除，不再作为 DOCX 对齐实验 |
| FE-04 只写 conversion + delay，漏掉 engagement | 已改为 conversion + delay bucket + engagement 三任务 |
| FE-02 delay 符号曾与 DOCX 相反 | 已修正为 DOCX 的 `timestamp - label_time` |
| `item_int_feats_90` 曾被误写为其他 pair 的 has_match | 已修正为 `match_count(item_int_feats_9, domain_d_seq_19)` 的分桶适配 |

## 4. 仍需人工确认

以下不是设计偏差，而是需要在完整数据上确认的实现细节：

1. `label_time` 和 `timestamp` 的业务含义是否保证 `timestamp - label_time` 为非负。若有负值，训练端应记录比例并裁剪或单独建桶。
2. `label_type` 是否足以构造 engagement label。若官方没有 engagement 定义，FE-04 的 engagement head 只能使用明确验证过的 proxy label。
3. sequence side feature 中哪些是数值型。当前 baseline 多数 sequence side 以匿名 int embedding 处理，不能把 categorical id 直接 z-score。
4. DOCX 的 int average fill 对匿名 categorical id 不安全。若要字面实现，必须先确认相关 int feature 是 ordinal/numeric，而非 id。

结论：核心实验已按上传 DOCX 对齐；所有非 DOCX 扩展已移出核心链路。
