# FE-01B Target Match Ablation

FE-01B 是从 FE-01 拆出来的第二个消融实验，只保留你这次指定的 target item 属性与 domain_d 历史序列匹配特征：

```text
item_int_feats_89 = has_match(item_int_feats_9, domain_d_seq_19)
item_int_feats_90 = bucketize(match_count(item_int_feats_9, domain_d_seq_19))
item_dense_feats_91 = log1p(min_match_delta(item_int_feats_9, domain_d_seq_19))
item_dense_feats_92 = log1p(match_count_7d(item_int_feats_9, domain_d_seq_19))
```

它用于验证 FE-01 中真正和 DOCX “目标 item 属性与历史序列匹配特征”对应的模块是否有效。

## 文件

```text
run_fe01b.sh
build_feature_engineering_dataset.py --feature_set fe01b
experiment_plans/FE01B/experiment_01b_target_match_plan.zh.md
```

## 上传/训练关系

若平台只能执行固定 `run.sh`，提交 FE-01B 时把 `run_fe01b.sh` 的内容覆盖为平台 `run.sh`。

评估可继续使用 `evaluation/FE01/infer.py`。它会按 checkpoint schema 只生成 FE-01B 需要的 89/90/91/92。
