# 第二次实验方案：FE-02 历史 Delay 特征

## 1. 实验目标

在 FE-01 的基础上，验证 DOCX 中的历史平均 delay 特征是否带来增益。

本实验仍然不使用当前样本的 `label_time` 作为模型输入，只用当前样本 timestamp 之前的历史转化样本构造 prefix delay 统计。

## 2. 新增特征

```text
user_dense_feats_112 = log1p(user_avg_delay_before_timestamp)
item_dense_feats_88 = log1p(item_avg_delay_before_timestamp)
```

建议同时生成两个可选 binary int 特征，但第一轮默认不开：

```text
user_int_feats_112_has_delay_history
item_int_feats_88_has_delay_history
```

原因：当前 schema 的 user/item int fid 空间已有固定编号，贸然添加新 int fid 会扩大 NS 分组变量。FE-02 第一版只加 dense。

## 3. 数据生成参数

基于 FE-01 数据增强脚本扩展：

```bash
python3 tools/build_feature_engineering_dataset.py \
  --input_dir /path/to/original_dataset \
  --input_schema /path/to/original_dataset/schema.json \
  --output_dir /path/to/fe02_dataset \
  --match_window_days 7 \
  --match_count_buckets 0,1,2,4,8 \
  --enable_delay_history
```

delay 统计规则：

```text
delay = timestamp - label_time
仅历史 label_type == 2 的样本参与 avg_delay
当前样本不参与自己的 avg_delay
无历史转化时 avg_delay = 0
```

注意：DOCX 明确写的是 `delay = timestamp - label_time`。如果完整训练集出现负 delay，需要记录比例，并在构造 bucket/weight 时单独裁剪或建桶。训练输入只允许使用历史聚合后的 avg_delay，不允许使用当前样本 delay。

## 4. 训练参数

沿用 FE-01：

```bash
bash run_fe01.sh \
  --data_dir /path/to/fe02_dataset \
  --schema_path /path/to/fe02_dataset/schema.json \
  --ckpt_dir outputs/exp_fe02_delay_history/ckpt \
  --log_dir outputs/exp_fe02_delay_history/log \
  --seed 42
```

## 5. 预期结果

| 指标 | 预测 |
| --- | --- |
| AUC | 相比 FE-01 小幅提升或持平，预计 `+0.0002 ~ +0.0010` |
| Logloss | 可能轻微改善 |
| 风险 | delay 历史稀疏时收益不稳定；若统计泄漏会出现异常大涨 |

验收标准：

```text
FE-02 AUC > FE-01，且至少 2/3 seed 同向
无异常大的 train-valid gap
```
