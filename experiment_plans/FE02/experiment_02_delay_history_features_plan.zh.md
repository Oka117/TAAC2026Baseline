# 第二次实验方案：FE-02 历史 Delay 特征

## 1. 实验定位

FE-02 是 FE-01 的增量实验，不是从零开始的独立变量实验。

它可以独立运行，因为 `run_fe02.sh` 会从原始 parquet/schema 重新生成完整 FE-02 数据集；但它的实验对照必须是 FE-01。原因是 FE-02 保留 FE-01 的频次特征、目标 item 属性匹配特征、NS 分组和训练参数，只额外打开历史 delay 聚合。

实验目标：

```text
验证 DOCX 中 user/item historical avg delay 特征是否能在 FE-01 的基础上继续提升 AUC。
```

## 2. 与 DOCX 的一一关联

| DOCX 位置 | 原文设计 | FE-02 实现 |
| --- | --- | --- |
| P013 | `delay= timestamp-label_time` | 对历史转化样本计算 `timestamp - label_time`；若结果为负，记录到 `delay_quality` 并裁剪为 0 |
| P014 | `user_dense_feats_112 = log(1 + user_avg_delay)` | `build_feature_engineering_dataset.py --enable_delay_history` 输出 `user_dense_feats_112`，来源是当前样本之前同 user 的历史转化平均 delay |
| P015 | `item_dense_feats_88 = log(1 + item_avg_delay)` | `build_feature_engineering_dataset.py --enable_delay_history` 输出 `item_dense_feats_88`，来源是当前样本之前同 item 的历史转化平均 delay |

FE-02 仍然继承 FE-01 已对齐的 DOCX 内容：

```text
P005/P009: user frequency / purchase frequency
P007/P011: item frequency / purchase frequency
P017-P020: item_int_feats_9 与 domain_d_seq_19 的匹配特征
```

## 3. 测试模块

FE-02 测试模块：

```text
DOCX: user and item historical data
模块变量: historical avg delay dense features
模型入口: user_dense token + item_dense token
```

不测试以下模块：

```text
Option D: Delay-Aware Weighted Loss
Option E: Multi-Task Loss
GNN / graph embedding
HyFormer block 结构修改
```

因此，FE-02 的收益或负收益应主要归因于“历史平均 delay 作为 dense 输入特征”。

## 4. 新增特征

FE-02 在 FE-01 的全部特征基础上新增：

```text
user_dense_feats_112 = zscore(log1p(user_avg_delay_before_timestamp))
item_dense_feats_88 = zscore(log1p(item_avg_delay_before_timestamp))
```

统计规则：

```text
1. 只使用已经在 streaming pass 中出现过的历史行。
2. 只使用历史 `label_type == 2` 的转化样本更新 delay 状态。
3. 当前样本先读取 prefix avg delay，再更新自身状态，所以当前样本不参与自己的 delay 特征。
4. delay 公式严格按 DOCX：timestamp - label_time。
5. delay < 0 时裁剪为 0，并在 `feature_engineering_stats.json.delay_quality` 记录数量和比例。
6. 没有历史转化时 avg_delay = 0。
7. dense 特征先 `log1p`，再用前 90% row groups 拟合 z-score 统计。
```

## 5. 代码与文件

当前已生成/更新：

```text
build_feature_engineering_dataset.py
run_fe02.sh
trainer.py
experiment_plans/FE02/README.md
experiment_plans/FE02/experiment_02_delay_history_features_plan.zh.md
```

`build_feature_engineering_dataset.py` 新增参数：

```text
--enable_delay_history
```

默认不传该参数时仍是 FE-01 行为，不会生成 `user_dense_feats_112` 和 `item_dense_feats_88`。

## 6. Schema 增量

FE-02 相比 FE-01 的 schema 增量：

```json
{
  "user_dense": [
    [112, 1]
  ],
  "item_dense": [
    [88, 1]
  ]
}
```

实际生成的 `schema.json` 会保留原始全部字段、FE-01 新增字段，并合并上述 FE-02 字段。`item_dense` 会按 fid 排序，实际顺序包含：

```text
item_dense_feats_86
item_dense_feats_87
item_dense_feats_88
item_dense_feats_91
item_dense_feats_92
```

## 7. NS 分组

FE-02 不新增 int 特征，因此 `ns_groups.feature_engineering.json` 沿用 FE-01。

注意：DOCX 的 NS 示例把 dense fid 放进了 int group，例如 `110/111/112` 和 `86/87/88/91/92`。当前 baseline 的 `ns_groups_json` 只接收 int fid，所以这些 dense 特征不写入 NS group，而是通过 user/item dense token 进入模型。

## 8. 训练命令

推荐命令：

```bash
bash run_fe02.sh \
  --data_dir /path/to/original_dataset \
  --schema_path /path/to/original_dataset/schema.json \
  --ckpt_dir outputs/exp_fe02_delay_history/ckpt \
  --log_dir outputs/exp_fe02_delay_history/log \
  --seed 42
```

`run_fe02.sh` 会自动执行：

```text
原始 parquet/schema
-> build_feature_engineering_dataset.py --enable_delay_history
-> /tmp/taac_fe02_xxx/schema.json + parquet + ns_groups.feature_engineering.json
-> train.py
```

如果平台只能执行固定 `run.sh`，提交 FE-02 时把 `run_fe02.sh` 的内容覆盖成平台的 `run.sh`。

## 9. 训练参数

FE-02 沿用 FE-01 参数，保证只测试 delay history：

| 参数 | 值 |
| --- | ---: |
| `--user_ns_tokens` | `6` |
| `--item_ns_tokens` | `4` |
| `--num_queries` | `1` |
| `--rank_mixer_mode` | `full` |
| `--d_model` | `64` |
| `--emb_dim` | `64` |
| `--num_hyformer_blocks` | `2` |
| `--num_heads` | `4` |
| `--seq_encoder_type` | `transformer` |
| `--seq_max_lens` | `seq_a:256,seq_b:256,seq_c:512,seq_d:512` |
| `--loss_type` | `bce` |
| `--patience` | `3` |
| `--num_epochs` | `6` |

加入两个 dense feature 后，dense token 数不变：

```text
num_ns = user_ns_tokens + user_dense_token + item_ns_tokens + item_dense_token
       = 6 + 1 + 4 + 1
       = 12

T = num_queries * 4 + num_ns
  = 1 * 4 + 12
  = 16

d_model = 64, 64 % 16 == 0
```

## 10. 输出文件

FE-02 预处理输出目录包含：

```text
schema.json
ns_groups.feature_engineering.json
feature_engineering_stats.json
docx_alignment.fe01.json
docx_alignment.fe02.json
*.parquet
```

其中 `feature_engineering_stats.json` 需要重点检查：

```text
enable_delay_history = true
dense_stats.user_dense_feats_112
dense_stats.item_dense_feats_88
delay_quality.observed_positive_label_with_label_time
delay_quality.negative_delay_count
delay_quality.negative_delay_ratio
```

`trainer.py` 已更新，会把 `docx_alignment.fe02.json` 和 `feature_engineering_stats.json` 随 checkpoint 一起保存，便于后续评估和审计。

## 11. 预期结果

| 指标 | 预测 |
| --- | --- |
| AUC | 相比 FE-01 `+0.0002 ~ +0.0010` 或持平 |
| Logloss | 可能轻微改善 |
| 风险 | 如果 `timestamp - label_time` 大量为负，delay 特征会被大量裁剪为 0，收益可能不稳定 |

判定标准：

```text
1. FE-02 AUC > FE-01。
2. 至少 2/3 seed 同向。
3. delay_quality.negative_delay_ratio 不应异常高；若异常高，需要重新确认 DOCX delay 符号是否与数据语义一致。
4. train-valid gap 不出现异常扩大。
```

推荐 seed：

```text
42, 2026, 3407
```
