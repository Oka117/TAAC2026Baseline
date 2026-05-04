# FE-01A Total Frequency Ablation

FE-01A 是从 FE-01 拆出来的第一个消融实验，只保留你这次指定的 total frequency 特征：

```text
user_dense_feats_110 = log1p(user_total_frequency_before_timestamp)
item_dense_feats_86 = log1p(item_total_frequency_before_timestamp)
```

它用于验证 FE-01 掉评估分是否来自 prefix frequency 模块，尤其是去掉 purchase frequency 后，total frequency 是否仍能稳定泛化。

## 文件

```text
run_fe01a.sh
build_feature_engineering_dataset.py --feature_set fe01a
experiment_plans/FE01A/experiment_01a_total_frequency_plan.zh.md
```

## 上传/训练关系

若平台只能执行固定 `run.sh`，提交 FE-01A 时把 `run_fe01a.sh` 的内容覆盖为平台 `run.sh`。

训练仍使用：

```text
dataset.py
model.py
train.py
trainer.py
utils.py
ns_groups.feature_engineering.json
```

评估可继续使用 `evaluation/FE01/infer.py`。它会按 checkpoint schema 只生成 FE-01A 需要的 110/86，不会生成 FE-01B 的 match 特征。
