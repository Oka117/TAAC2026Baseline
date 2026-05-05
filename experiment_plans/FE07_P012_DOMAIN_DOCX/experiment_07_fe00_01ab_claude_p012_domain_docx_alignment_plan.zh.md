# FE-07 方案：FE00 + 01AB + Claude P0/P1/P2-Domain 强对齐版

## 0. 一句话结论

FE-07 不是重新发明一组特征，而是把三份已有证据合并成一条更干净的实验主线：

```text
上传 feature-engineering.docx
  -> FE00: 缺失处理 + dense normalization
  -> FE01A/B: total frequency + target-history match
  -> Claude P0/P1: 修补 baseline 丢掉的数据层和 NS 语义层信号
  -> Claude P2-Domain: 四个 domain 使用差异化时间桶与序列策略
```

主目标：

```text
在不引入 purchase-frequency / 当前样本 label_time 泄漏风险的前提下，
把 FE01B 已验证的 target-history match 收益，和 P0/P1/P2-Domain 的低泄漏信号叠加起来。
```

推荐第一轮主线：

```text
FE07-Domain-main = FE00-literal + FE01AB-safe + P0-safe + P1-NS + P2-Domain
```

本实验明确不加入 GNN / TokenGNN 结构。原因是当前目标是验证 `FE00 + 01AB + Claude P0/P1/P2-Domain` 这条纯特征工程和 domain-aware 路线；GNN 历史结果只作为对照背景，不作为 FE-07 的二阶段分支。

## 1. Git History 回顾

以下按历史脉络归纳全部本地分支和关键提交。重点不是逐行复述 diff，而是说明每条历史对 FE-07 的约束。

| Commit / 分支 | 内容 | 对 FE-07 的结论 |
| --- | --- | --- |
| `303c6c8` / `7c9f32e` baseline | 腾讯 Angel baseline，原始 `dataset.py/model.py/trainer.py/run.sh` | B0 起点，Eval AUC `0.810525` |
| `e16f25c` main | `README.feature_engineering.zh.md`，GNN + HyFormer 数据分析与特征建议 | 四个 domain 长度、新近度、高基数字段、target match lift 是 P2-Domain 的数据证据 |
| `61fba26`、`9ba9e4c` / `GNN_NS(0.815)` | `TokenGNN` 放在 NS tokenizer 后，4 layers | 仅作为历史对照；FE-07 不引入 GNN |
| `bb918ea` / `GNN+NS_head(0.811)` | 记录 `output_include_ns` 直接拼 head 后 AUC 从 `0.815064` 掉到 `0.811474` | 证明直接加结构路径有风险；FE-07 不加入该类结构变量 |
| `813f0f9`、`19a6384` / `gnn4layerAdjustPara` | `token_gnn_layer_scale=0.15`、`use_token_gnn=true` | 仅记录历史，不进入本实验 |
| `4a8820c` | 基于上传 feature engineering 文档生成 FE00-FE04 总体方案 | 上传 DOCX 是核心 source of truth，FE-07 必须保留逐项映射 |
| `a40a77d` | FE-00 代码和文档 | FE00 清洗与 normalization 已落账，Eval AUC `0.811646` |
| `7262b5b`、`0e0ba5f` | FE01 实验与 evaluation 修复 | 证明增强数据必须有训练/eval 一致的 sidecar 和 schema |
| `f521482`、`f7da4d5`、`4c08102` | 拆出 FE01A、FE01B 并写入结果 | FE01B 是当前最强 DOCX 对齐特征，Eval AUC `0.812102` |
| `da1ecf9` / `feature_FE_01_optimise` | Claude `Baseline与数据联合分析-提分路线.md` | 提供 P0/P1/P2-Domain 路线，本方案主要吸收这里 |
| `591babe` / `FE_04` | delay bucket / multitask 初版 | P2-Aux/FE04 有价值，但变量较大，不进 FE-07 第一轮主线 |
| `e4411b7` / 当前 `FE_00_01AB_P0_1_2domain` | FE06 builder、FE06 evaluation、TokenGNN 合入 | FE-07 只借鉴 FE06 的 FE00+01AB+P0 数据处理经验，不继承 TokenGNN 结构 |

已有结果基线：

| 实验 | Eval AUC | Eval Δ vs B0 | 结论 |
| --- | ---: | ---: | --- |
| B0 | 0.810525 | +0.000000 | 原始 baseline |
| FE-00 | 0.811646 | +0.001121 | 缺失处理与 normalization 有正收益 |
| FE-01 full | 0.775054 | -0.035471 | 不可直接保留，疑似 purchase-frequency train/eval 不一致 |
| FE-01A | 0.810780 | +0.000255 | total frequency 安全但收益小 |
| FE-01B | 0.812102 | +0.001577 | target-history match 泛化价值最好 |
| GNN_NS_4Layer | 0.815064 | +0.004539 | 历史结构增强对照；不纳入 FE-07 |
| 4LayerGNN + direct NS head | 0.811474 | +0.000949 | 历史负例；FE-07 不加入 direct NS head 或 GNN |

## 2. Source Of Truth 与强关联矩阵

上传 DOCX 已在 `feature_engineering_design_alignment_audit.zh.md` 中拆成 P000-P067。FE-07 只选择和当前历史证据最强、泄漏风险最低的部分进入主线。

| DOCX 设计点 | 已有 FE 基础 | Claude 路线 | FE-07 处理 | 关联强度 |
| --- | --- | --- | --- | :---: |
| P000 删除 missing ratio `>75%` 的 user int | FE-00 已实现 | P0-L1 missing/padding 拆开 | 保留 FE-00 删除逻辑；另做 missing bucket 消融 | 高 |
| P001 int missing average fill | FE-00 已实现 | P0-L1 提出 `0=padding,1=missing` | 主线保留 FE00-literal；P0-L1-vocab-shift 作为替代消融 | 高 |
| P002 dense numerical normalization | FE-00 已实现 | P0-L3 dense norm + missing indicator | 继承 FE-00 normalization；新增 P0 dense block 统一 log1p+z-score | 高 |
| P005 `user_dense_feats_110` total frequency | FE-01A 已实现 | P0-L5/L6 活跃度 summary | 保留，但不作为主收益来源 | 高 |
| P007 `item_dense_feats_86` total frequency | FE-01A 已实现 | P1-L2 item dense token | 保留，必须确保 item dense 训练/eval 都开启 | 高 |
| P009/P011 purchase frequency | FE-01 full 大幅掉 eval | 无低风险替代 | 第一轮排除；只留回滚说明 | 高，负例 |
| P017-P020 target-history match | FE-01B 已验证 | P1-Match | 主线保留，作为 DOCX 最强正向特征 | 最高 |
| P021 其他 item-field × sequence-field lift 筛选 | 已规划，未系统跑 | README §6.3，P2-HighCard | 第一轮只跑 full-train lift 审计，不直接扩 Top-K | 中 |
| P022-P042 user/item NS groups | `ns_groups.feature_engineering.json` 已有 | P1-NS | 主线加入语义 NS groups，保证 dense fid 不写进 int groups | 高 |
| P043-P057 Delay-aware weighted loss | FE03 规划 | P2-Aux 不完全等价 | 不进 FE-07 主线，避免和 domain 效果混在一起 | 中 |
| P058-P067 Multi-task loss | FE04 初版 | P2-Aux | 第二阶段单独跑，不进第一轮主线 | 中 |
| sequence side normalization | FE-00 有适配说明 | P2-Domain 各 domain bucket | 用 per-domain time bucket 与 seq_max_len 策略实现 domain-aware side feature 处理 | 高 |

关键处理原则：

```text
1. DOCX 里已验证正向或安全的内容，进入 FE07 主线。
2. DOCX 里和历史结果冲突的内容，保留为消融或排除项，并写清楚原因。
3. Claude P0/P1/P2-Domain 不是独立花活，而是对 DOCX 设计的工程化补全。
```

## 3. FE-07 主实验定义

### 3.1 FE00-literal 基础层

主线从 FE-00 已落账实现出发：

```text
delete user int feature if missing ratio > 75%
int missing average fill
dense numerical normalization
```

注意：Claude P0-L1 的 `missing/padding split` 与 DOCX P001 的 `average fill` 有冲突。FE-07 不把它们混在一个主实验里。

主线：

```text
FE00-literal = 保留 FE-00 当前实现
```

消融：

```text
FE00-P0L1 = 用 vocab shift 替代 int average fill
  0 -> padding
  1 -> missing
  k+1 -> 原始 id k
```

判断标准：

```text
若 FE00-P0L1 > FE00-literal，则后续切到 missing bucket 路线；
若 FE00-literal >= FE00-P0L1，则优先尊重上传 DOCX 的 average fill 设计。
```

### 3.2 FE01AB-safe 特征层

保留 FE01A 的安全 total frequency：

```text
user_dense_feats_110 = log1p(user_total_frequency_before_timestamp)
item_dense_feats_86  = log1p(item_total_frequency_before_timestamp)
```

保留 FE01B 的 target-history match：

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

排除原因：

```text
FE-01 full eval AUC = 0.775054，远低于 B0。
FE-01A/FE-01B 拆分后没有复现大掉点，因此 purchase-frequency 或其 eval 侧状态最可疑。
```

### 3.3 Claude P0-safe 数据补全层

FE-07 第一轮纳入 P0 中与 DOCX 高相关、且不引入 label 依赖的模块。

| P0 模块 | 与 DOCX 的关系 | FE-07 处理 |
| --- | --- | --- |
| P0-T1 timestamp valid split | 保证 prefix frequency、normalization、lift 验证不被时间错位污染 | 加入训练切分，按 row group `max(timestamp)` 取最后 10% valid |
| P0-L1 missing/padding split | 对应 P000/P001，但与 average fill 冲突 | 不进主线，单独做 FE00-P0L1 消融 |
| P0-L3 dense norm + missing indicator | 对应 P002 | 继承 FE-00 norm；新增 dense block 使用同一 train-only stats |
| P0-L4 current timestamp features | 与时间上下文、delay 设计同源，但不使用 label_time | 新增 `user_dense_feats_120`，dim=3 |
| P0-L5/L6 seq_len + window counts | 与 sequence side feature 和 domain summary 强相关 | 新增 `user_dense_feats_121`，dim=20 |

新增 dense 规划：

```text
user_dense_feats_120: sample_time_context, dim=3
  [hour_of_day / 23, day_of_week / 6, log1p(day_since_train_min)]

user_dense_feats_121: domain_sequence_summary, dim=20
  for domain in a,b,c,d:
    [log1p(seq_len), log1p(count_1h), log1p(count_1d), log1p(count_7d), log1p(count_30d)]
```

这些特征与 `README.feature_engineering.zh.md` 的实证直接相关：

```text
domain_a count_7d 高分位: 15.4% 正例率 vs 低分位 10.4%
domain_c len 高分位: 15.1% vs 11.6%
domain_d len 高分位: 14.0% vs 10.0%
```

### 3.4 Claude P1 语义 NS 层

P1 的主线只纳入两个低风险部分：

```text
P1-L2: 启用 item_dense token
P1-NS: 使用 ns_groups.feature_engineering.json 做语义 NS 分组
```

原因：

```text
1. DOCX P007/P019/P020 依赖 item_dense token，否则 item dense 特征进不了模型。
2. DOCX P022-P042 本质是 NS groups 设计，必须用语义分组承接。
```

推荐 token 配置：

```text
user_ns_tokens = 6
item_ns_tokens = 4
num_queries    = 1
num_ns          = 6 user int + 1 user dense + 4 item int + 1 item dense = 12
T               = 1 * 4 + 12 = 16
d_model         = 64
64 % 16         = 0
```

暂不纳入 P1-Output direct NS head。

若后续要测试 P1-Output，只允许 gated fusion：

```text
final = q_repr + gate * ns_repr
gate bias 初始化为负值，训练初期接近关闭
```

直接 `output_include_ns` 已在历史中掉到 `0.811474`，不能作为 FE-07 主线。

### 3.5 Claude P2-Domain 层

P2-Domain 是 FE-07 的核心新增点：四个行为域不能共用同一套时间桶与序列窗口策略。

数据证据：

```text
domain_a: 平均长度 701，历史事件中位年龄 73.5 天，7 天内事件占比 5.4%
domain_b: 平均长度 571，历史事件中位年龄 94.2 天，7 天内事件占比 7.8%
domain_c: 平均长度 449，历史事件中位年龄 275.3 天，7 天内事件占比 1.5%
domain_d: 平均长度 1100，历史事件中位年龄 12.5 天，7 天内事件占比 29.5%
```

主配置：

```text
domain_a: 中等近期窗口，time bucket 分辨率偏 1d/7d/30d
domain_b: 中等近期窗口，保留稍长历史
domain_c: 更老、更稀疏，压缩序列长度并加强长周期 bucket
domain_d: 最新、最高频，保留更长近期窗口并细化 1h/1d/7d bucket
```

实现建议：

```text
BUCKET_BOUNDARIES_A = train domain_a delta quantiles + fixed anchors
BUCKET_BOUNDARIES_B = train domain_b delta quantiles + fixed anchors
BUCKET_BOUNDARIES_C = train domain_c delta quantiles + fixed anchors
BUCKET_BOUNDARIES_D = train domain_d delta quantiles + fixed anchors
```

固定 anchors：

```text
1h, 1d, 7d, 30d, 90d, 180d, 365d
```

推荐 `seq_max_lens` 消融：

| 实验 | seq_a | seq_b | seq_c | seq_d | 目的 |
| --- | ---: | ---: | ---: | ---: | --- |
| uniform-256 | 256 | 256 | 256 | 256 | 对照 |
| domain-balanced | 256 | 256 | 256 | 512 | 保留 domain_d 高频近期 |
| c-compressed-d-long | 256 | 256 | 128 | 768 | 测试 domain_c 压缩、domain_d 拉长 |

P2-HighCard 与 P2-Aux 暂不进 FE-07 第一轮主线：

```text
P2-HighCard hashing/count 会改 sequence id 表达，和 P2-Domain bucket 同时上会难解释。
P2-Aux / FE04 会改 loss 和 label 结构，和 DOCX delay/multitask 相关，但变量过大。
```

## 4. 实验序列

推荐按以下顺序跑。每一步只新增一个主要变量。

| 顺序 | 实验名 | 内容 | 目的 | 预期 Eval AUC |
| ---: | --- | --- | --- | ---: |
| 0 | B0 replay | 原始 baseline | 确认平台和 seed 未漂 | 0.8105 |
| 1 | FE00-literal replay | FE-00 当前实现 | 复核 DOCX P000-P002 收益 | 0.8110 ~ 0.8120 |
| 2 | FE01AB-safe | FE00 + FE01A + FE01B，排除 purchase | 验证 01A/01B 合并是否稳定 | 0.8120 ~ 0.8135 |
| 3 | P0-safe | FE01AB-safe + P0-T1/L3/L4/L5/L6 | 验证数据层补全 | 0.8140 ~ 0.8170 |
| 4 | P1-NS | P0-safe + item dense token + semantic NS groups | 验证 DOCX NS group 设计 | 0.8145 ~ 0.8180 |
| 5 | P2-Domain | P1-NS + domain-specific buckets + seq lens | 验证四域差异化建模 | 0.8150 ~ 0.8190 |
| 6 | FE00-P0L1 消融 | 用 missing bucket 替换 FE00 average fill | 判定 P001 与 P0-L1 哪条更好 | 视结果决定 |
| 7 | FE00-P0L1 + P2-Domain | 若 FE00-P0L1 消融优于 FE00-literal，则替换后复跑 P2-Domain | 判定 missing bucket 是否成为最终主线 | 0.8150 ~ 0.8195 |

最低资源版本：

```text
P0-safe
P2-Domain
FE00-P0L1 消融
```

若 `P0-safe` 不涨，不要继续叠 P2-Domain，先拆 `user_dense_feats_120/121`。

## 5. 训练与参数建议

主配置建议：

```bash
python3 -u train.py \
  --schema_path "${FE07_SCHEMA}" \
  --ns_groups_json "${FE07_GROUPS}" \
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
  --seq_max_lens seq_a:256,seq_b:256,seq_c:128,seq_d:768 \
  --use_time_buckets \
  --domain_time_buckets \
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
  --seq_id_threshold 10000
```

本实验不追加 `--use_token_gnn`、`--token_gnn_layers`、`--token_gnn_graph` 或 `--token_gnn_layer_scale`。

## 6. 数据与 sidecar 设计

建议新建统一 builder，而不是串联 FE00 -> FE01AB -> P0/P2 多个脚本：

```text
raw parquet
  -> FE00 missing/drop/norm
  -> FE01AB prefix total frequency + target match
  -> P0 timestamp/context/domain summary dense
  -> P2 domain time bucket stats
  -> output parquet + schema + ns_groups + sidecars
```

推荐新增 sidecar：

```text
fe07_docx_alignment.json
fe07_transform_stats.json
domain_time_bucket_boundaries.json
dense_normalization_stats.json
feature_engineering_stats.json
ns_groups.feature_engineering.json
train_config.json
```

`domain_time_bucket_boundaries.json` 示例：

```json
{
  "domain_a": [3600, 86400, 604800, 2592000],
  "domain_b": [3600, 86400, 604800, 2592000],
  "domain_c": [86400, 604800, 2592000, 7776000, 15552000, 31536000],
  "domain_d": [300, 1800, 3600, 21600, 86400, 604800, 2592000]
}
```

真实边界应由 train row groups 拟合，固定 anchors 只作为下限。

## 7. Evaluation 设计

FE-07 不能直接复用 FE01 infer，也不应直接复用 FE06 infer 后硬补 P2-Domain。

原因：

```text
1. FE01 infer 不知道 P0 dense block 与 domain-specific buckets。
2. FE06 infer 主要服务 FE00+01AB+P0+TokenGNN，未显式拆 P1-NS/P2-Domain，且包含本实验不需要的 GNN 结构假设。
3. P2-Domain 的 bucket boundaries 必须从 checkpoint sidecar 读取，不能 eval 重新 fit。
```

推荐新增：

```text
evaluation/FE07/infer.py
evaluation/FE07/dataset.py
evaluation/FE07/model.py
evaluation/FE07/build_fe07_p012_domain_dataset.py
```

Eval 严禁：

```text
在 eval 上重新拟合 dense mean/std
在 eval 上重新拟合 domain bucket quantiles
在 eval 上重新选择 item-sequence match pair
读取 eval label_type 或 label_time 更新 purchase/delay 历史
```

## 8. 强检查清单

### 8.1 DOCX 对齐

```text
[ ] FE00-literal 输出 `docx_alignment.fe00.json`，记录 P000/P001/P002。
[ ] FE01AB 输出 `docx_alignment.fe01ab.json`，记录 P005/P007/P017-P020。
[ ] purchase frequency P009/P011 在 schema 中不存在。
[ ] dense fid 110/86/91/92 真实写入 parquet，并通过 item/user dense token 进入模型。
[ ] item_int_feats_89/90 真实写入 parquet，并进入 item NS group。
[ ] dense fid 不写进 int-only `ns_groups.feature_engineering.json`。
```

### 8.2 时间与泄漏

```text
[ ] timestamp split 只用 sample timestamp，不用 label_time。
[ ] prefix total frequency 不包含当前样本。
[ ] target-history match 只统计 event_time <= sample timestamp。
[ ] match_count_7d 使用固定 7 天窗口。
[ ] domain bucket boundaries 只由 train row groups 拟合。
[ ] current timestamp dense 使用固定 UTC 规则。
```

### 8.3 P2-Domain

```text
[ ] 四个 domain 的 timestamp fid 正确：a=39, b=67, c=27, d=26。
[ ] 每个 domain 使用自己的 time embedding 或 offset 后的 bucket id。
[ ] seq_max_lens 与 bucket 配置写入 train_config。
[ ] uniform-256 对照保留，避免把 seq length 收益误归因于 bucket。
```

### 8.4 模型结构

```text
[ ] P1-NS 主线 token 数满足 T=16, d_model=64 可整除。
[ ] 不启用 direct `output_include_ns`。
[ ] 不启用 `--use_token_gnn`。
[ ] train_config 中不包含 active GNN 结构参数。
```

## 9. 决策规则

保留规则：

```text
P0-safe >= FE01AB-safe + 0.0010：继续上 P1-NS。
P1-NS >= P0-safe - 0.0003 且 logloss 不恶化：保留语义 groups。
P2-Domain >= P1-NS + 0.0005：进入 FE07-Domain-main。
FE00-P0L1 >= FE00-literal + 0.0005：用 missing bucket 替代 average fill 后复跑 P2-Domain。
```

回退规则：

```text
P0-safe 不涨：拆 user_dense_feats_120 和 user_dense_feats_121。
P1-NS 掉分：保留 item dense token，回退 ns_groups_json="" 或减少 user_ns_tokens。
P2-Domain 掉分：先回退 seq_max_lens，再回退 per-domain bucket。
若任何结构参数被误启用，回退到无 GNN 的 FE07-Domain-main。
```

## 10. 最终推荐

FE-07 的核心不是“更多模块”，而是把上传特征工程文档中已经写清楚的信号，用更贴合 baseline 的方式补回去：

```text
FE00 负责 DOCX 的清洗和 normalization；
FE01AB 负责 DOCX 的 total frequency 与 target-history match；
P0 负责把 timestamp、domain activity、train-time stats 这些 baseline 丢掉的上下文补回 NS token；
P1 负责让 DOCX 的 NS groups 和 item dense 真正进入模型；
P2-Domain 负责尊重四个 domain 完全不同的时间分布。
```

第一轮主候选：

```text
FE07-Domain-main = FE00-literal + FE01AB-safe + P0-safe + P1-NS + P2-Domain
```

目标：

```text
Eval AUC >= 0.8150：说明 P0/P1/P2-Domain 至少超过 FE01B，可继续做非 GNN 特征消融。
Eval AUC >= 0.8170：可替代 FE01B/FE06-P0AB 成为新的数据特征主线。
Eval AUC >= 0.8190：进入多 seed、EMA/SWA 或非 GNN 的特征消融阶段。
```
