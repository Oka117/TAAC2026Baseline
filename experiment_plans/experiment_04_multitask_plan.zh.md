# 第四次实验方案：FE-04 Conversion + Delay Bucket + Engagement Multi-Task

## 1. 实验目标

验证 DOCX 中的 multi-task loss 是否能通过辅助 delay bucket 和 engagement 预测提高 shared HyFormer 表征的泛化能力。

这是改动最大的实验，建议在 FE-01/FE-02/FE-03 有稳定收益后再做。

## 2. 模型结构

保留 HyFormer backbone，替换单一 classifier：

```text
shared_output = HyFormer(...)
conversion_logit = conversion_head(shared_output)
delay_bucket_logits = delay_head(shared_output)
engagement_logit = engagement_head(shared_output)
```

## 3. Label 设计

```text
conversion_label = (label_type == 2)
engagement_label = verified_engagement_proxy(label_type or behavior signal)
delay_bucket_label:
  0: no positive conversion / ignore
  1: delay <= 1h
  2: 1h < delay <= 1d
  3: 1d < delay <= 7d
  4: delay > 7d
```

delay loss 只对正样本计算：

```text
delay_mask = conversion_label == 1
```

engagement label 必须先确认官方字段语义。若只能从 `label_type` 构造 proxy，需在实验记录里明确写出映射规则，避免把 conversion label 简单复制成 engagement label。

## 4. Loss

```text
total_loss = cvr_bce + lambda_delay * delay_ce + lambda_engagement * engagement_bce
```

推荐参数：

```text
--multi_task_delay true
--multi_task_engagement true
--delay_loss_weight 0.05
--engagement_loss_weight 0.02
--delay_num_buckets 5
```

如果训练稳定，再尝试：

```text
--delay_loss_weight 0.1
```

## 5. 训练参数

```bash
bash run_fe01.sh \
  --data_dir /path/to/best_fe_dataset \
  --schema_path /path/to/best_fe_dataset/schema.json \
  --ckpt_dir outputs/exp_fe04_multitask/ckpt \
  --log_dir outputs/exp_fe04_multitask/log \
  --multi_task_delay \
  --multi_task_engagement \
  --delay_loss_weight 0.05 \
  --engagement_loss_weight 0.02 \
  --seed 42
```

## 6. 预期结果

| 指标 | 预测 |
| --- | --- |
| AUC | 可能 `+0.0003 ~ +0.0015`，但依赖 delay 和 engagement label 质量 |
| Logloss | 可能持平或轻微变差 |
| 训练稳定性 | 比 FE-03 更敏感，需要调 loss weight |

验收标准：

```text
conversion AUC 稳定提升
delay auxiliary accuracy 不是随机水平
engagement auxiliary AUC 高于随机水平
CVR 主任务 loss 不被 delay/engagement loss 主导
```
