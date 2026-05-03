# 第三次实验方案：FE-03 Delay-Aware Weighted Loss

## 1. 实验目标

验证 DOCX 中的 delay-aware weighted BCE 是否能稳定训练、改善 AUC 或 logloss。

本实验只改 loss，不新增输入特征。使用 FE-01/FE-02 中最优的数据配置。

## 2. 需要代码改动

`dataset.py`：

```text
返回 delay_seconds 或 delay_bucket，仅 trainer 使用，不传给 ModelInput
delay = timestamp - label_time
```

`trainer.py`：

```text
BCEWithLogitsLoss(reduction='none')
loss = weighted_loss.mean()
```

`train.py` 新增参数：

```text
--delay_weight_mode none|fast_boost|long_discount|bucket
--delay_weight_clip 3.0
--delay_weight_alpha 0.3
```

## 3. 推荐权重函数

先跑两个方向：

```text
FE-03A fast_boost:
  w = 1 + alpha * exp(-delay_days / 1.0)

FE-03B long_discount:
  w = 1 / (1 + alpha * log1p(delay_days))
```

统一裁剪：

```text
w = clamp(w, 0.5, 3.0)
```

只对正样本使用 delay 权重，负样本权重保持 1。

## 4. 训练参数

```bash
bash run_fe01.sh \
  --data_dir /path/to/best_fe_dataset \
  --schema_path /path/to/best_fe_dataset/schema.json \
  --ckpt_dir outputs/exp_fe03_delay_weight/ckpt \
  --log_dir outputs/exp_fe03_delay_weight/log \
  --delay_weight_mode fast_boost \
  --delay_weight_alpha 0.3 \
  --delay_weight_clip 3.0 \
  --seed 42
```

## 5. 预期结果

| 指标 | 预测 |
| --- | --- |
| AUC | 不确定，预计 `-0.0003 ~ +0.0008` |
| Logloss | 可能变差，因为训练目标偏离无权 BCE |
| 泛化 | 若 delayed conversion 噪声大，可能更稳定 |

验收标准：

```text
不能只看 AUC；若 logloss 明显变差，需要谨慎保留
权重后正负样本 loss 曲线不能发散
```
