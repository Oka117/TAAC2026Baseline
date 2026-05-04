# 实验结构、参数和结果预测总表

以下预测是基于上传 DOCX 设计和当前 baseline 结构的先验判断，不是实际 leaderboard 结果。真实结论以相同切分、相同 seed 的 ablation 为准。

## 总体实验链路

| 实验 | 主变量 | 代码改动 | 推荐参数 | AUC 预测 |
| --- | --- | --- | --- | --- |
| B0 | 当前 baseline | 无 | 直接 `python3 -u train.py` 跑原始数据与 baseline active config | 基准 |
| FE-00 | 缺失处理 + dense normalization | `build_fe00_preprocess_dataset.py` | `missing_threshold=0.75,fit_stats_row_group_ratio=0.9` | `-0.0002 ~ +0.0006` |
| FE-01 | prefix frequency + target match + item dense token | `build_feature_engineering_dataset.py`, `dataset.py item_dense` | `user_ns_tokens=6,item_ns_tokens=4,num_queries=1,d_model=64` | `+0.0010 ~ +0.0030` |
| FE-01A | total frequency only | `build_feature_engineering_dataset.py --feature_set fe01a` | `user_dense_feats_110,item_dense_feats_86` | valid `+0.0000 ~ +0.0006`; eval 预计接近 B0 |
| FE-01B | target-history match only | `build_feature_engineering_dataset.py --feature_set fe01b` | `item_int_feats_89/90,item_dense_feats_91/92` | valid `+0.0003 ~ +0.0012`; eval 预计高于完整 FE-01 |
| FE-02 | 历史 avg delay dense | 扩展 FE 脚本 | 沿用 FE-01 | `+0.0002 ~ +0.0010` |
| FE-03 | delay-aware weighted BCE | `trainer.py`, `train.py`, `dataset.py` | `fast_boost alpha=0.3 clip=3.0` | `-0.0003 ~ +0.0008` |
| FE-04 | conversion + delay bucket + engagement multi-task | `model.py`, `trainer.py`, `train.py`, `dataset.py` | `delay_loss_weight=0.05, engagement_loss_weight=0.02` | `+0.0003 ~ +0.0015` |

## 推荐保留策略

```text
1. FE-00 对齐 DOCX 的缺失处理和 normalization，是数据审计基础。
2. 完整 FE-01 valid 小涨但 eval 大掉，因此先跑 FE-01A/FE-01B 拆因。
3. FE-01A 用来判断 total frequency 是否可保留。
4. FE-01B 用来判断 target-history match 是否可保留。
5. FE-02 只有在 FE-01A/FE-01B 找到稳定模块后再接。
6. FE-03 的收益不确定，必须同时看 AUC 和 logloss。
7. FE-04 是论文/创新方向更强的实验，但工程风险最高。
```

## 结果记录模板

| 实验 | Seed | Best Epoch/Step | Valid AUC | Delta vs Parent | Logloss | Eval AUC | Eval Δ vs B0 | 推理时间(s) | 结论 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0 | 42 | epoch 6 | 0.862268 |  | 0.2245 | 0.810525 | +0.000000 | 164.8 | Baseline 已完成训练与平台评估 |
| FE-00 | 42 | epoch 5 | 0.86338 | +0.001112 | 0.22355 | 0.811646 | +0.001121 | 854.09 | valid/eval 均高于 B0 |
| FE-01 | 42 | epoch 6 | 0.863048 | +0.000780 | 0.2228 | 0.775054 | -0.035471 | 1093.64 | valid 高于 B0，eval 低于 B0，需复查评估特征一致性 |
| FE-01A | 42 | epoch 5 | 0.86238 | +0.000112 | 0.22464 | 0.81078 | +0.000255 | 1071.39 | total frequency only；valid/eval 基本回到 B0 附近，未复现完整 FE-01 的 eval 大掉 |
| FE-01B | 42 | epoch 5 | 0.86136 | -0.000908 | 0.22623 | 0.812102 | +0.001577 | 611.65 | target-history match only；valid 低于 B0，但 eval 高于 B0/FE-01A，目标匹配特征更可能具备泛化价值 |
| FE-01 | 2026 |  |  |  |  |  |  |  |  |
| FE-02 | 42 |  |  |  |  |  |  |  |  |
| FE-03A | 42 |  |  |  |  |  |  |  |  |
| FE-04 | 42 |  |  |  |  |  |  |  |  |
