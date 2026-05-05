# FE-06 方案：FE00 + FE01AB-safe + Claude P0 + 4-layer TokenGNN

## 0. 一句话结论

FE-06 是一版组合型候选方案：以已经落账安全的 `FE-00` 数据清洗、`FE-01A` 的 total frequency、`FE-01B` 的 target-history match 为基础，补上 Claude 提分路线里的 P0 数据层信号，再接入历史 git 分支中验证过的 `4-layer TokenGNN` 结构。

核心原则：

```text
保留已验证能涨 eval 的模块；
排除完整 FE-01 中导致 eval AUC 掉到 0.775 的 label-dependent purchase frequency；
P0 数据特征和 TokenGNN 结构分开做 ablation，避免一次性混入不可解释变量。
```

## 1. Git 历史回顾

以下只列和本方案直接相关的提交链路。

| Commit | 分支/位置 | 内容 | 对 FE-06 的意义 |
| --- | --- | --- | --- |
| `7c9f32e` | baseline | 腾讯 Angel baseline 原始代码 | B0 起点 |
| `e16f25c` | main | `README.feature_engineering.zh.md`，GNN + HyFormer 特征工程建议 | 提供 target-history match、domain summary、GNN token 思路 |
| `61fba26` | `GNN_NS(0.815)` | 初版 `TokenGNN`，放在 NS token 后 | 确定 GNN 插入位置：NS tokenizer 后、query generator 前 |
| `9ba9e4c` | `GNN_NS(0.815)` | GNN 4 layers update | 形成 `GNN_NS_4Layer` 主线 |
| `bb918ea` | `GNN+NS_head(0.811)` | 实验结论报告 | 记录 `GNN_NS_4Layer` eval AUC `0.815064`，直接拼 final NS head 掉到 `0.811474` |
| `813f0f9` | `gnn4layerAdjustPara` | `token_gnn_layer_scale=0.15` 调参 | 给 FE-06 的 TokenGNN 推荐 scale |
| `19a6384` | `gnn4layerAdjustPara` | `use_token_gnn=true` | 确认 GNN4 可作为 active config |
| `4a8820c` | FE plans | 基于 feature engineering 文档生成 FE00/FE01/FE02/FE03/FE04 计划 | FE 系列文档起点 |
| `a40a77d` | FE00 | FE-00 代码与文档 | 缺失处理 + dense normalization 起点 |
| `0e0ba5f` | FE01 eval | 修复 FE-01 evaluation 缺列问题 | 形成 checkpoint schema 驱动的 eval 逻辑 |
| `f521482` | FE01A/B | FE01A、FE01B 文档和脚本 | 拆出 total frequency 与 target-history match 两个安全消融 |
| `f7da4d5` / `4c08102` | result table | 写入 FE01A/B 实验结果 | 提供本方案的真实 eval 基线 |
| `da1ecf9` | Claude | `Baseline与数据联合分析-提分路线.md` | 提供 P0 数据层修补路线 |
| `591babe` | FE04 | delay bucket / multitask 初版 | 不纳入 FE-06，原因是 label_time/aux 变量风险更高 |

## 2. 已有结果基线

比赛看 `Eval AUC`，因此本方案以 `Eval Δ vs B0` 为主排序。

| 实验 | Eval AUC | Eval Δ vs B0 | 结论 |
| --- | ---: | ---: | --- |
| B0 | 0.810525 | +0.000000 | baseline |
| FE-00 | 0.811646 | +0.001121 | 清洗和 normalization 有正收益 |
| FE-01 full | 0.775054 | -0.035471 | 不可用，强烈怀疑 purchase-frequency train/eval 不一致 |
| FE-01A | 0.810780 | +0.000255 | total frequency 安全但收益小 |
| FE-01B | 0.812102 | +0.001577 | 当前 FE 特征中最稳，target-history match 有泛化价值 |
| GNN_NS_4Layer | 0.815064 | +0.004539 | 历史 GNN 分支最优稳定结构 |
| 4LayerGNN + direct NS head | 0.811474 | +0.000949 | 直接拼 final NS head 是负向，不保留 |

## 3. FE-06 实验目标

FE-06 要回答三个问题：

1. `FE-00 + FE-01AB-safe` 是否能稳定超过单独 FE-01B？
2. Claude P0 数据层修补是否能和 FE-01B、GNN4 叠加，而不是互相冲突？
3. 4-layer TokenGNN 是否能在更干净、更丰富的 NS token 输入上继续保持历史收益？

预期主实验不是一次性替代所有方案，而是生成一个可拆分的组合路线：

```text
B0
  -> FE00
  -> FE01AB-safe
  -> P0 data features
  -> TokenGNN4
```

## 4. 实验定义

### 4.1 FE01AB-safe 特征集合

保留 FE01A：

```text
user_dense_feats_110 = log1p(user_total_frequency_before_timestamp)
item_dense_feats_86  = log1p(item_total_frequency_before_timestamp)
```

保留 FE01B：

```text
item_int_feats_89   = has_match(item_int_feats_9, domain_d_seq_19)
item_int_feats_90   = bucketize(match_count(item_int_feats_9, domain_d_seq_19))
item_dense_feats_91 = log1p(min_match_delta(item_int_feats_9, domain_d_seq_19))
item_dense_feats_92 = log1p(match_count_7d(item_int_feats_9, domain_d_seq_19))
```

明确排除：

```text
user_dense_feats_111 = log1p(user_purchase_frequency_before_timestamp)
item_dense_feats_87  = log1p(item_purchase_frequency_before_timestamp)
```

排除原因：完整 FE-01 的 eval AUC `0.775054`，而 FE01A/FE01B 都没有复现这个大掉点。最可疑源头是 purchase-frequency 在训练中可由 `label_type==2` 更新，但评估时不能读取真实 eval label，导致 train/eval 分布不一致。

### 4.2 Claude P0 数据层修补

纳入 Claude P0 的 5 个模块。

| P0 模块 | FE-06 实现方式 | 是否新增功能 |
| --- | --- | :---: |
| P0-T1 timestamp valid split | 训练切分按 row group `max(timestamp)` 排序后取最后 10% valid | 是，训练评估方式新增 |
| P0-L1 missing/padding 拆开 | 采用方案 A：vocab shift，`0=padding,1=missing,k+1=原 id k` | 是，新增 sparse id 编码规则 |
| P0-L3 dense normalization | 继承 FE-00 normalization；本轮不额外加入 dense missing mask，避免和 P0-L1 变量混在一起 | 半新增，FE-00 已有 normalization |
| P0-L4 当前样本 timestamp 衍生 | 生成 `hour_of_day/day_of_week/day_since_min` dense block | 是，新增时间上下文特征 |
| P0-L5/L6 seq_len + 多窗口计数 | 对 4 个 domain 生成 `len/count_1h/count_1d/count_7d/count_30d` | 是，新增 domain summary dense 特征 |

P0-L1 采用 Claude 文档中的最低成本方案 A，而不是方案 B 的 dense missing mask：

```text
id = 0      -> padding
id = 1      -> missing，即原始 -1
id = k + 1  -> 原始合法 id k
```

schema 适配规则：

```text
new_vocab_size = old_vocab_size + 1
```

前提是原 schema 已经把 `0` 作为 padding，合法原始 id 落在 `[1, old_vocab_size-1]`。实现时必须在预处理阶段做 max-id audit；如果发现合法原始 id 可以等于 `old_vocab_size`，则用 `max_observed_id + 2` 回写 schema，避免 embedding 越界。

新增 dense fid 规划：

```text
user_dense_feats_120: sample_time_context, dim=3
  [hour_of_day / 23, day_of_week / 6, log1p(day_since_train_min)]

user_dense_feats_121: domain_sequence_summary, dim=20
  for domain in a,b,c,d:
    [log1p(seq_len), log1p(count_1h), log1p(count_1d), log1p(count_7d), log1p(count_30d)]
```

说明：

- P0-L1 不再生成 `user_int_missing_mask` / `item_int_missing_mask` dense block。
- P0-L3 不额外生成 dense missing mask，先保持低成本、低变量。
- 所有新增 dense block 都走 user dense token，不写进 `ns_groups.json` 的 int groups。
- vocab shift 必须在 FE-00 int fill 之前完成；FE-06 中不再使用 FE-00 的 int average fill 路径处理 sparse id missing。

### 4.3 4-layer TokenGNN 结构

从 GNN 历史分支继承：

```bash
--use_token_gnn
--token_gnn_layers 4
--token_gnn_graph full
--token_gnn_layer_scale 0.15
```

结构位置：

```text
user/item/dense features
  -> NS tokenizer
  -> ns_tokens
  -> 4-layer TokenGNN
  -> graph-enhanced ns_tokens
  -> query generator
  -> HyFormer blocks
  -> final Q tokens
  -> CVR head
```

明确不纳入：

```text
--output_include_ns
```

原因：历史结论中 direct NS head 从 `0.815064` 降到 `0.811474`，说明 final NS token 直接拼 head 容易让 head 过度依赖静态 token 或引入噪声。

## 5. 组合结构强检查

### 5.1 与 FE-00 的关系

| FE-00 内容 | FE-06 处理 |
| --- | --- |
| 高缺失 user int 删除 | 保留，作为 schema 清洗 |
| int missing average fill | 不用于 FE-06 主配置；替换为 P0-L1 方案 A vocab shift |
| dense normalization | 保留，避免 P0 dense block 与原 dense 量级冲突 |
| sidecar: `dense_normalization_stats.json` | 需要进入 FE-06 checkpoint 或 FE-06 eval sidecar |

关键检查：

```text
FE-06 不能直接复用完整 FE-00 sparse-id 填补逻辑；
P0-L1 的 vocab shift 必须从 raw parquet 读取 `-1`，先映射为 missing id，再写入增强 parquet 和 schema。
```

### 5.2 与 FE-01A 的关系

| FE-01A 内容 | FE-06 处理 |
| --- | --- |
| `user_dense_feats_110` | 保留 |
| `item_dense_feats_86` | 保留 |
| prefix total frequency | 只用当前样本之前的 streaming state |
| purchase frequency | 不保留 |

检查：

```text
FE-01A eval 只 +0.000255，收益小；
因此它在 FE-06 中是低权重安全特征，不作为主收益来源。
```

### 5.3 与 FE-01B 的关系

| FE-01B 内容 | FE-06 处理 |
| --- | --- |
| `item_int_feats_89/90` | 保留 |
| `item_dense_feats_91/92` | 保留 |
| `domain_d_seq_19` 目标匹配 | 保留 |
| 7d match count | 保留 |

检查：

```text
FE-01B 是当前 FE 特征中 eval 最强模块；
FE-06 必须保证这 4 个字段的定义和 FE-01B 完全一致，不能在 P0 或 GNN 接入时改变编码。
```

### 5.4 与 Claude P0 的关系

| Claude P0 | FE-06 关系 | 风险 |
| --- | --- | --- |
| P0-T1 timestamp split | 训练 valid 更接近 leaderboard | valid AUC 可能下降，但更可信 |
| P0-L1 vocab shift | 新增 sparse missing token，替代 FE-00 int average fill | 必须 raw-before-fill，并同步 vocab size |
| P0-L3 normalization | FE-00 已做一部分 | 避免重复 z-score，不额外加入 mask 变量 |
| P0-L4 current timestamp | 新增线上可见时间上下文 | 需固定时区/UTC 规则 |
| P0-L5/L6 seq count windows | 新增 activity / recency 信号 | 只统计 event_time <= sample timestamp |

### 5.5 与 GNN4 历史方案的关系

| GNN 历史内容 | FE-06 处理 |
| --- | --- |
| `TokenGNN` after NS tokenizer | 保留 |
| 4 layers | 保留 |
| fully connected graph over NS tokens | 保留 |
| layer scale `0.15` | 主配置 |
| `user_ns_tokens=5,item_ns_tokens=2,num_queries=2` | GNN4-only 消融完全继承；FE06 full 因 item dense token 需改成合法 token 数 |
| direct final NS head | 明确排除 |

历史 GNN4 最佳 AUC 参数包作为 `GNN4-only` 消融的复刻配置：

```bash
--ns_tokenizer_type rankmixer
--user_ns_tokens 5
--item_ns_tokens 2
--num_queries 2
--use_token_gnn
--token_gnn_layers 4
--token_gnn_graph full
--token_gnn_layer_scale 0.15
--dropout_rate 0.015
--patience 3
--ns_groups_json ""
--emb_skip_threshold 1000000
```

为什么 FE06 full 不完全照搬这组 token 数？

GNN 历史最佳配置来自 baseline/GNN-only 路线，主要 token 数是：

```text
user_ns_tokens=5
item_ns_tokens=2
num_queries=2
```

但 FE01AB-safe 会启用 item dense token。若继续用 `num_queries=2`，token 总数容易破坏 `rank_mixer_mode=full` 的 `d_model % T == 0` 约束。

FE-06 full 主配置采用“GNN 最佳结构参数 + FE01B 合法 token 数”：

```text
user_ns_tokens = 6
item_ns_tokens = 4
num_queries    = 1
num_sequences  = 4
num_ns          = 6 user int + 1 user dense + 4 item int + 1 item dense = 12
T               = 1 * 4 + 12 = 16
d_model         = 64
64 % 16         = 0
```

这保证结构合法，并且把变量集中在 P0 + TokenGNN。若后续一定要在 FE06 full 中完全复刻 `5/2/2`，需要额外引入一个新变量，例如 `rank_mixer_mode=ffn_only` 或调整 `d_model`，本方案第一版不建议这样做。

### 5.6 与 FE-02 / FE-04 的关系

| 方案 | 是否纳入 FE-06 | 原因 |
| --- | :---: | --- |
| FE-02 avg delay | 否 | 依赖 `label_time` 与历史 conversion，和 FE-01 purchase-frequency 有相似风险 |
| FE-03 delay weighted loss | 否 | loss 变量，先不混入数据+结构组合 |
| FE-04 multitask | 否 | label_type/time 多任务变量较大，应单独实验 |

## 6. 推荐实现结构

### 6.1 新增或修改文件

| 文件 | 类型 | 功能 |
| --- | --- | --- |
| `build_fe06_p0ab_dataset.py` | 新增 | 统一完成 FE-00 dense 清洗、P0-L1 vocab shift、FE01AB-safe、P0 dense block，避免多脚本串联覆盖 raw missing |
| `tools/build_fe06_p0ab_dataset.py` | 新增 | 平台上传版本，同步根目录脚本 |
| `run_fe06_p0ab_gnn4.sh` | 新增 | 数据生成 + 训练入口 |
| `dataset.py` | 修改 | 支持 timestamp row group split；若所有 P0 特征离线物化，则无需在线算特征 |
| `train.py` | 修改 | 加入 `--use_token_gnn`、`--token_gnn_layers`、`--token_gnn_graph`、`--token_gnn_layer_scale`，以及 timestamp split 开关 |
| `model.py` | 修改 | 从 GNN 分支合入 `TokenGNNLayer` / `TokenGNN` |
| `trainer.py` | 修改 | 保存 FE-06 sidecars 和 train config |
| `evaluation/FE06/infer.py` | 新增 | 原始 eval parquet -> FE-06 eval parquet；复用训练 sidecar，不在 eval 重新 fit |
| `evaluation/FE06/model.py` | 新增 | 与训练侧 TokenGNN 结构一致 |
| `evaluation/FE06/dataset.py` | 新增 | 与训练侧 schema/dense 读取一致 |
| `experiment_plans/FE06_P0AB_GNN4/README.md` | 新增 | 实验索引 |

### 6.2 为什么建议新建 `build_fe06_p0ab_dataset.py`

不建议简单串联：

```text
FE00 output -> FE01AB script -> P0 script
```

原因：

1. FE-00 原逻辑会填补或替换 sparse 缺失值，之后再做 P0-L1 vocab shift 已经晚了。
2. FE01AB prefix frequency 和 P0 seq-window 特征都依赖 timestamp 顺序，应共享同一套 row group 顺序和 leakage 检查。
3. evaluation 需要同一套 sidecar，分散在多个脚本里容易遗漏。

推荐统一 builder：

```text
raw parquet
  -> sparse id vocab shift: 0 padding, 1 missing, k+1 original id
  -> FE-00 dense fill/drop/norm
  -> FE01AB prefix + target match
  -> P0 timestamp + seq window dense block
  -> output parquet + schema.json + ns_groups + sidecars
```

## 7. 训练参数建议

主配置：

```bash
python3 -u train.py \
  --schema_path "${FE06_SCHEMA}" \
  --ns_groups_json "${FE06_GROUPS}" \
  --ns_tokenizer_type rankmixer \
  --user_ns_tokens 6 \
  --item_ns_tokens 4 \
  --num_queries 1 \
  --rank_mixer_mode full \
  --d_model 64 \
  --emb_dim 64 \
  --num_hyformer_blocks 2 \
  --num_heads 4 \
  --seq_encoder_type transformer \
  --seq_max_lens seq_a:256,seq_b:256,seq_c:512,seq_d:512 \
  --use_time_buckets \
  --loss_type bce \
  --lr 1e-4 \
  --sparse_lr 0.05 \
  --dropout_rate 0.015 \
  --batch_size 256 \
  --num_workers 8 \
  --buffer_batches 20 \
  --valid_ratio 0.1 \
  --train_ratio 1.0 \
  --patience 3 \
  --num_epochs 6 \
  --emb_skip_threshold 1000000 \
  --seq_id_threshold 10000 \
  --use_token_gnn \
  --token_gnn_layers 4 \
  --token_gnn_graph full \
  --token_gnn_layer_scale 0.15
```

参数说明：

| 参数 | 原因 |
| --- | --- |
| `user_ns_tokens=6,item_ns_tokens=4,num_queries=1` | FE06 full 的合法 token 数，保持 `T=16`，满足 `d_model=64` 整除约束 |
| `dropout_rate=0.015` | 来自 GNN 分支轻微 dropout bump，比 FE01B 的 `0.01` 稍强 |
| `token_gnn_layer_scale=0.15` | 历史 GNN4 最佳 AUC 参数包中的主推值 |
| `patience=3,num_epochs=6` | 沿用当前平台稳定训练设置 |
| `loss_type=bce` | 不混入 loss 变量 |

## 8. Ablation 顺序

不要一口气只跑最终组合。推荐按以下顺序跑，保证可解释。

| 顺序 | 实验名 | 内容 | 目的 |
| ---: | --- | --- | --- |
| 0 | B0 replay | 当前 baseline | 确认平台/seed 没漂 |
| 1 | FE00 replay | 现有 FE-00 | 复核清洗收益 |
| 2 | FE01AB-safe | FE01A + FE01B，排除 purchase | 检查 A/B 合并是否互相干扰 |
| 3 | P0-only | FE00 + P0，不加 FE01AB/GNN | 验证 Claude P0 数据层修补 |
| 4 | P0AB | FE00 + P0 + FE01AB-safe | 验证数据特征可叠加 |
| 5 | GNN4-only | baseline + TokenGNN4，完全复刻 `5/2/2 + layer4 + scale0.15` | 对齐历史 GNN4 最佳 AUC 参数 |
| 6 | FE06 full | FE00 + P0 + FE01AB-safe + TokenGNN4 | 主候选 |

若平台资源有限，最低限度先跑：

```text
P0AB
FE06 full
```

## 9. Evaluation 方案

FE-06 不能直接复用当前 `evaluation/FE01/infer.py`。

原因：

1. 当前 FE01 infer 只知道固定 FE01-family 列：

```text
user_dense_feats_110/111
item_dense_feats_86/87/91/92
item_int_feats_89/90
```

2. FE-06 新增 P0 dense block `user_dense_feats_120/121`，并对 sparse id 做 vocab shift。
3. FE-06 model 需要 TokenGNN 结构参数，否则 checkpoint strict load 会失败。

推荐新增：

```text
evaluation/FE06/infer.py
evaluation/FE06/model.py
evaluation/FE06/dataset.py
```

eval 必须从 checkpoint sidecar 读取：

```text
schema.json
train_config.json
ns_groups.feature_engineering.json
fe06_transform_stats.json
feature_engineering_stats.json
dense_normalization_stats.json
docx_alignment.fe06.json
```

严禁：

```text
在 eval 数据上重新拟合 dense mean/std
在 eval 数据上重新选择 match pair
读取 eval label_type 来更新 purchase/history conversion state
```

## 10. 预期结果

保守预测：

| 实验 | Eval AUC 预期 | 依据 |
| --- | ---: | --- |
| FE01AB-safe | 0.8115 ~ 0.8130 | FE01A 收益小，FE01B 已到 0.812102 |
| P0AB | 0.8140 ~ 0.8170 | Claude P0 数据层信号 + FE01B |
| GNN4-only | 0.8140 ~ 0.8160 | 历史 GNN_NS_4Layer 为 0.815064 |
| FE06 full | 0.8165 ~ 0.8205 | P0 数据层与 GNN4 结构大体正交，但不假设完全相加 |

更保守的验收标准：

```text
FE06 eval AUC >= FE01B + 0.0020  即 >= 0.8141：可继续优化
FE06 eval AUC >= 0.8160：可作为新主线
FE06 eval AUC <  FE01B：立即拆分 P0AB/GNN4 找冲突
```

推理时间预期：

```text
900s ~ 1400s
```

原因：FE-06 评估要先物化 P0/FE01AB 特征，且模型增加 4-layer TokenGNN。若推理时间超过 FE01B 两倍且 AUC 收益小于 `+0.0015`，需要回退到 P0AB 或 FE01B。

## 11. 强力检查清单

### 11.1 数据与泄漏

```text
[ ] P0-L1 使用 vocab shift：0 padding，1 missing，k+1 原 id k。
[ ] sparse id vocab size 已按 remap 后最大 id 回写 schema，不能 embedding 越界。
[ ] FE-06 主配置不使用 FE-00 int average fill 覆盖 sparse missing。
[ ] dense normalization stats 只用训练 row groups 拟合。
[ ] FE01AB prefix frequency 不包含当前样本。
[ ] 不生成 user_dense_feats_111 / item_dense_feats_87。
[ ] 不使用当前样本 label_time 作为输入特征。
[ ] seq window count 只统计 event_time <= sample timestamp。
[ ] timestamp 衍生特征使用固定 UTC 规则，不依赖本地时区。
[ ] eval transform 不读取 eval label_type。
```

### 11.2 Schema 与 token 结构

```text
[ ] user_dense 新 fid 不与 FE01/FE02/FE05 规划冲突。
[ ] dense block dim 写入 schema.json，dataset.py 按 schema 读取。
[ ] item_int_feats_89 vocab=3，item_int_feats_90 vocab=len(match_count_buckets)+1。
[ ] ns_groups 只包含 int fid；dense fid 不写入 int NS groups。
[ ] num_ns = 12，T = 16，d_model=64 满足 full rank mixer 约束。
[ ] GNN4-only 消融使用历史最佳 AUC 参数：user_ns_tokens=5,item_ns_tokens=2,num_queries=2,scale=0.15。
[ ] TokenGNN 只作用在 NS tokens，不改 sequence tokenizers。
[ ] 不启用 output_include_ns。
```

### 11.3 Checkpoint 与 evaluation

```text
[ ] trainer.py 把 FE-06 sidecar 复制到 best_model 目录。
[ ] train_config.json 包含 use_token_gnn / token_gnn_layers / token_gnn_layer_scale。
[ ] evaluation/FE06/model.py 与训练 model.py 的 TokenGNN 参数名完全一致。
[ ] infer.py 根据 checkpoint schema 生成 FE-06 所需列。
[ ] strict load checkpoint 成功；不允许 missing/unexpected key 被静默忽略。
```

### 11.4 实验解释性

```text
[ ] 至少保留 P0AB 和 FE06 full 两个实验点。
[ ] 若 FE06 full 不涨，先看 P0AB 是否涨，再判断冲突来自 GNN 还是 P0。
[ ] 若 valid 涨但 eval 不涨，以 eval 为准。
[ ] 若 eval 涨但 logloss 变差，可保留；比赛排序指标是 AUC。
```

## 12. 新增功能/结构标注

| 类别 | 新增内容 | 是否已有方案覆盖 |
| --- | --- | --- |
| 数据清洗 | FE-00 raw fill/norm | 已有 FE-00 |
| 特征 | FE01AB-safe | FE01A/B 已有，但组合版新增 |
| 特征/编码 | sparse id vocab shift missing token | Claude P0 方案 A，当前 FE 代码未实现 |
| 特征 | current timestamp dense block | Claude P0 提出，当前 FE 代码未实现 |
| 特征 | domain seq len/window count block | Claude P0 提出，当前 FE 代码未实现 |
| 训练切分 | timestamp row group valid split | Claude P0 提出，当前 train split 未实现 |
| 模型结构 | 4-layer TokenGNN after NS tokenizer | GNN 历史分支已有，当前 FE 分支需合入 |
| 禁用结构 | direct final NS head | 历史证明负向，明确不采用 |
| 评估 | FE06 transform-only infer | 新增，当前 FE01 infer 不够 |

## 13. 结论

FE-06 是当前最合理的组合主线，但必须按 ablation 推进。

推荐优先级：

```text
1. 先实现统一 FE06 builder，保证 FE00/P0/FE01AB 特征在同一原始流上生成。
2. 先跑 P0AB，不加 GNN，确认数据层是否已经超过 FE01B。
3. 再合入 4-layer TokenGNN，保持 user_ns_tokens=6,item_ns_tokens=4,num_queries=1。
4. evaluation 单独建 FE06，不要复用当前 FE01 infer。
```

最终主候选：

```text
FE06 full = FE00 + FE01AB-safe + Claude P0 + TokenGNN4
```

目标：

```text
Eval AUC >= 0.8160
```

若达到该线，FE-06 可替代 FE-01B 成为新的实验基线；若未达到，则保留 P0AB 或 FE01B，TokenGNN4 单独回到 GNN 分支继续调参。
