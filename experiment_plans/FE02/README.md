# FE-02 历史 Delay 特征

FE-02 是 FE-01 之上的增量实验。它可以从原始数据一键独立运行，但实验变量不是独立于 FE-01 的：FE-02 会生成 FE-01 的全部安全特征，再额外加入 DOCX 中的历史平均 delay 特征。

## 文件

```text
run_fe02.sh
build_feature_engineering_dataset.py
experiment_plans/FE02/experiment_02_delay_history_features_plan.zh.md
```

训练时平台仍然读取根目录代码。若平台只能执行固定 `run.sh`，提交 FE-02 时需要把 `run_fe02.sh` 的内容覆盖为平台的 `run.sh`。

## 新增实验变量

FE-02 相比 FE-01 只新增两个 dense 特征：

```text
user_dense_feats_112 = log1p(user_avg_delay_before_timestamp)
item_dense_feats_88 = log1p(item_avg_delay_before_timestamp)
```

其余保持 FE-01 一致：

```text
FE-01 frequency features
FE-01 target item attribute matching features
ns_groups.feature_engineering.json
--user_ns_tokens 6
--item_ns_tokens 4
--num_queries 1
--loss_type bce
--patience 3
--num_epochs 6
```

## 设计文档关联

| DOCX 段落 | 设计内容 | FE-02 实现 |
| --- | --- | --- |
| P013 | `delay= timestamp-label_time` | 仅对历史转化样本计算 `timestamp - label_time`；负 delay 记录比例并裁剪为 0 |
| P014 | `user_dense_feats_112 = log(1 + user_avg_delay)` | 按 user 维护 prefix 历史平均 delay，当前样本不参与自己的统计 |
| P015 | `item_dense_feats_88 = log(1 + item_avg_delay)` | 按 item 维护 prefix 历史平均 delay，当前样本不参与自己的统计 |

## 测试模块

FE-02 测试的是 DOCX 的 `user and item historical data` 模块，也就是历史转化延迟统计是否能给 CVR 预测提供额外信号。

本实验不测试：

```text
delay-aware weighted loss
multi-task loss
GNN embedding
HyFormer block 改造
```

这些分别留给 FE-03、FE-04 或后续图特征实验。
