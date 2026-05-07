# FE-08 5 月 7 日方案：GNN 结合验证特征，保持 NS 影响域

## 0. 一句话结论

FE-08 严格落地 Claude `5 月 7 日` 方案：

```text
5 月 7 日方案 = 0.8159 GNN baseline
              + FE01B/FE07 已验证的 item_dense/match 安全集合
              + item_dense token
              + sequence 时间排序
              + 3 个 item int 新特征
              + 2 个 user int 时间特征
              + d_model=136, full mode
              + dropout=0.05
```

目标不是重新设计一条新模型路线，而是把已经有证据的低风险数据层信号回流进 `4-layer TokenGNN` 主干，同时保持 GNN message-passing 的 token 拓扑稳定。

预期 Eval AUC：

```text
可接受替代线: >= 0.8185
强主线门槛  : >= 0.8200
预测区间    : 0.8185 ~ 0.8215
```

## 1. Source Of Truth

本实验只按以下 Claude 文档执行：

| 文档 | 本方案使用方式 |
| --- | --- |
| `experiment_plans/Claude/5月7日_GNN结合验证特征_保持NS影响域_方案.md` | 主方案，定义最终结构、偏差落点、参数和 AUC 目标 |
| `experiment_plans/Claude/5月7日_FE08代码结构_供AI_Agent搭建.md` | 代码搭建指南，定义新增文件、builder/eval/run 脚本和 sidecar |
| `experiment_plans/Claude/Baseline与数据联合分析-提分路线.md` | 数据层动机，解释为什么 item_dense、timestamp、target match 可叠加到 GNN；FE07 P0 dense 110/120/121 只作为可选继承，不是本主线必选项 |

若文档内存在早期残留，以最终锁定项为准：

```text
采用: rank_mixer_mode=full + d_model=136
不采用: ffn_only + d_model=128
采用: transformer encoder + seq_top_k=100 marker
不采用: 第一轮切 longer encoder
采用: item_dense_fids={86,91,92}
不采用: risky item_dense_fids={87,88}
```

### 1.1 Claude 对应联系点矩阵（摘要）

完整逐项审计见 `experiment_08_claude_alignment_audit.zh.md`。本方案的关键联系点如下：

| 本方案模块 | Claude 主方案联系点 | Claude 代码指南联系点 | 对齐结论 |
| --- | --- | --- | --- |
| 总目标与 AUC 区间 | 主方案 L52-L69：FE08 = GNN baseline + FE07 稳态收益 + d_model=136/full | 代码指南 L8-L9：完成产物与 0.8185~0.8215 目标 | 对齐 |
| 新增/修改文件 | 主方案 L774-L810：builder/eval/run 文件结构 | 代码指南 L23-L42：新增文件与修改文件清单 | 对齐 |
| 不改 dataset/model | 主方案 L298-L301：model.py 已支持 item_dense；L366-L369：sort 后训练端无需改 | 代码指南 L44-L51：dataset.py/model.py/ns_groups.json 不动 | 对齐 |
| missing drop | 主方案 L78、L150-L208：user/item missing >80% drop | 代码指南 L80-L82、L114-L116、L193：阈值与 CLI | 对齐 |
| item_dense fid | 主方案 L216-L301：86/91/92 锁定，87/88 排除 | 代码指南 L87-L92、L199-L203：白名单与 risky flag | 对齐 |
| sequence sort | 主方案 L337-L374：row/domain 级 event_time desc argsort | 代码指南 L145-L149、L204-L207：sort_sequence_by_recency | 对齐 |
| item/user int 新特征 | 主方案 L376-L492：89/90/91、130/131 规则 | 代码指南 L130-L140、L279-L307：生成规则 | 对齐 |
| Token 结构 | 主方案 L61-L69、L497-L527：T=17，d_model=136/full | 代码指南 L56-L73、L165-L168：num_ns=9, TokenGNN 4层 | 对齐 |
| seq_top_k | 主方案 L567-L613：transformer 下 marker only | 代码指南 L70-L71、L582-L590：加 warning | 对齐 |
| eval parity | 主方案 L478-L492：eval 自动生成且 strict | 代码指南 L424-L472：FE08 eval 读取 sidecar transform | 对齐 |
| sidecar | 主方案 L482-L487、L895-L906 | 代码指南 L364-L421、L612-L626 | 对齐 |

已处理的内部冲突：

```text
Claude 主方案 L883-L888 / L919 存在旧 checklist 残留，写了 ffn_only / d_model=128。
本方案按同一 Claude 文档前文最终锁定项与代码指南执行：
rank_mixer_mode=full, d_model=136。
```

## 2. 历史证据与父实验

| 实验 | Eval AUC | 与 FE-08 的关系 |
| --- | ---: | --- |
| B0 | 0.810525 | 原始下界 |
| FE-00 | 0.811646 | 缺失处理与 dense normalization 有正收益 |
| FE-01 full | 0.775054 | 负例，purchase/delay 类特征高度可疑 |
| FE-01A | 0.810780 | total frequency 安全但收益小 |
| FE-01B | 0.812102 | target-history match 是当前最稳数据层特征 |
| GNN_NS_4Layer | 0.815064 ~ 0.8159 | FE-08 的结构起点 |
| 4LayerGNN + direct NS head | 0.811474 | 负例，FE-08 禁止 `output_include_ns` |
| FE-07 | 待全量确认 | FE-08 的 builder/eval fork 起点，不继承 no-GNN 设定 |

FE-08 与 FE-07 的关系：

```text
FE-07 = FE00 + FE01AB-safe + Claude P0/P1 + P2-Domain, no GNN
FE-08 = FE07 safe builder 思路 + GNN4 主干 + May7 新特征, 不启用 P2-Domain
```

## 3. FE-08 主实验定义

### 3.1 数据清洗：drop missing > 80%

实现位置：

```text
build_fe08_may7_dataset.py
evaluation/FE08/build_fe08_may7_dataset.py
```

规则：

```text
missing_threshold = 0.80
扫描范围 = user_int + item_int
输出 = dropped_feats.may7.json
schema 和 ns_groups.may7.json 同步删除命中 fid
```

sidecar 示例：

```json
{
  "user_int": [],
  "item_int": [],
  "threshold": 0.80,
  "row_groups_used_for_audit": 0,
  "interpretation": "Empty list means no fid hit the missing>0.80 threshold."
}
```

空列表是合法输出，eval 端必须显式打印，不允许把空集当作 sidecar 缺失。

### 3.2 item_dense token + normalization

FE-08 启用 item_dense token，fid 严格锁定：

```text
item_dense_feats_86 = log1p(item_total_frequency_before_timestamp)
item_dense_feats_91 = log1p(min_match_delta(item_int_9, domain_d_seq_19))
item_dense_feats_92 = log1p(match_count_7d(item_int_9, domain_d_seq_19))
```

schema：

```json
"item_dense": [[86, 1], [91, 1], [92, 1]]
```

明确排除：

```text
item_dense_feats_87 = purchase frequency
item_dense_feats_88 = avg delay / label_time dependent
```

builder 必须提供双保险：

```text
--item_dense_fids "86,91,92"
--enable_risky_item_dense_fids  # 默认关闭；只有显式传入才允许 87/88
```

normalization：

```text
只用 train row groups 拟合 mean/std
输出 fe08_dense_normalization_stats.json
eval 端只读取 sidecar，不重新 fit
```

### 3.3 sequence recency sort

对应 Claude 原文：

```text
确保每个 sequence 是按由距离现在最近到最远的时间顺序排序
```

实现规则：

```text
对每行、每个 domain，按该 domain timestamp 列降序 argsort
同一 permutation 重排该 domain 的所有 side feature 列和 timestamp 列
list 长度必须保持不变
length=0 或 length=1 直接跳过
```

timestamp fid：

```text
domain_a: 39
domain_b: 67
domain_c: 27
domain_d: 26
```

排序只依赖 event timestamp，不依赖 `label_time`。

### 3.4 新增 item_int 特征

| fid | 含义 | vocab_size | dim | 来源 |
| --- | --- | ---: | ---: | --- |
| 89 | `has_match(item_int_feats_9, domain_d_seq_19)`，0=missing/padding, 1=no, 2=yes | 3 | 1 | FE-01B 复用 |
| 90 | `bucketize(match_count(...))`，bucket=`0,1,2,4,8` | 7 | 1 | FE-01B 复用 |
| 91 | 最近匹配事件所处全局 time bucket，非匹配为 0 | 64 | 1 | FE-08 新增 |

`item_int_feats_91` 与 `item_dense_feats_91` 允许共存：

```text
item_int_feats_91   -> int schema, NS tokenizer
item_dense_feats_91 -> dense schema, item_dense_proj
```

它们列名前缀不同，schema list 不同，模型入口不同，不构成物理冲突。

### 3.5 新增 user_int 时间特征

| fid | 含义 | vocab_size | dim |
| --- | --- | ---: | ---: |
| 130 | `hour_of_day = (timestamp // 3600) % 24 + 1` | 25 | 1 |
| 131 | `day_of_week = ((timestamp // 86400) + 4) % 7 + 1` | 8 | 1 |

必须 `+1`，因为 sparse id 体系中 `0` 是 padding。合法 hour 落到 `1..24`，合法 dow 落到 `1..7`。

### 3.6 ns_groups.may7.json

dense fid 不写入 ns_groups。只加入新增 int fid：

```json
{
  "user_ns_groups": {
    "U2_user_temporal_behavior": [50, 60, 130, 131]
  },
  "item_ns_groups": {
    "I4_target_matching_fields": [89, 90, 91]
  }
}
```

实际文件应由 builder 根据最终 schema 过滤生成，确保 missing drop 后的 fid 不会残留在 group 中。

## 4. 模型结构锁定

### 4.1 token 数

FE-08 必须保持 Claude 锁定的 NS 影响域：

```text
user_ns_tokens = 5
item_ns_tokens = 2
num_queries = 2
has_user_dense = True
has_item_dense = True
```

因此：

```text
num_ns = 5 + 1 + 2 + 1 = 9
T = num_queries * num_sequences + num_ns
  = 2 * 4 + 9
  = 17
```

### 4.2 d_model 与 rank_mixer_mode

最终锁定：

```text
rank_mixer_mode = full
d_model = 136
136 % 17 = 0
```

不采用早期残留的 `ffn_only + d_model=128`。原因：

```text
1. full 与 0.8159 GNN baseline 行为一致
2. item_dense token 真实启用后 T=17，128 不可整除
3. 136 是最接近 128 且保持 capacity 上调方向的可行值
4. 保留 RankMixer token mixing 与 TokenGNN message passing 的双路径
```

### 4.3 TokenGNN

```text
--use_token_gnn
--token_gnn_layers 4
--token_gnn_graph full
--token_gnn_layer_scale 0.15
```

禁止：

```text
--output_include_ns
```

## 5. 代码文件清单

### 5.1 新增文件

| 文件 | 来源 | 责任 |
| --- | --- | --- |
| `build_fe08_may7_dataset.py` | fork `build_fe07_p012_domain_dataset.py` | 训练侧 builder |
| `tools/build_fe08_may7_dataset.py` | 同步副本 | 平台上传入口 |
| `run_fe08_may7_full.sh` | 新建 | 端到端 run 入口 |
| `evaluation/FE08/build_fe08_may7_dataset.py` | 与训练侧 builder 同逻辑 | eval transform |
| `evaluation/FE08/dataset.py` | fork `evaluation/FE07/dataset.py` | eval dataset |
| `evaluation/FE08/model.py` | 以根 `model.py` / FE07 model 为基础 | 必须支持 TokenGNN |
| `evaluation/FE08/infer.py` | fork `evaluation/FE07/infer.py` | strict infer |

### 5.2 修改文件

| 文件 | 修改 |
| --- | --- |
| `train.py` | 加 `seq_top_k` no-op warning；加 FE08 `d_model/T` sanity check |
| `trainer.py` | checkpoint sidecar copy 列表加入 FE08 sidecar |

### 5.3 不动文件

| 文件 | 原因 |
| --- | --- |
| `dataset.py` | 已支持 schema-driven item_dense、split_by_timestamp、time buckets |
| `model.py` | 已支持 item_dense token、TokenGNN、layer_scale、full-mode divisibility check |
| `ns_groups.json` | 仅作示例；FE08 生成 `ns_groups.may7.json` |

## 6. Builder 数据流

```text
raw parquet
  -> audit user_int/item_int missing ratio
  -> drop missing_ratio > 0.80
  -> fit dense stats on train row groups
  -> compute generated features:
       item_dense 86/91/92
       item_int 89/90/91
       user_int 130/131
  -> sort sequence by event timestamp desc
  -> write augmented parquet
  -> write schema.json / ns_groups.may7.json / FE08 sidecars
```

必须输出：

```text
schema.json
ns_groups.may7.json
dropped_feats.may7.json
fe08_transform_stats.json
fe08_dense_normalization_stats.json
feature_engineering_stats.json
docx_alignment.fe08.json
```

`fe08_transform_stats.json` 至少包含：

```json
{
  "missing_threshold": 0.8,
  "match_window_days": 7,
  "match_count_buckets": [0, 1, 2, 4, 8],
  "match_count_vocab_size": 7,
  "item_dense_fids": [86, 91, 92],
  "enable_risky_item_dense_fids": false,
  "domain_match_columns": {
    "match_col": "domain_d_seq_19",
    "match_ts_col": "domain_d_seq_26"
  },
  "num_time_buckets": 64,
  "sequence_sort_by_recency": true,
  "sort_order": "descending_by_event_timestamp"
}
```

## 7. 训练配置

主 run 配置：

```bash
python3 -u train.py \
  --schema_path "${FE08_SCHEMA}" \
  --ns_groups_json "${FE08_GROUPS}" \
  --ns_tokenizer_type rankmixer \
  --user_ns_tokens 5 \
  --item_ns_tokens 2 \
  --num_queries 2 \
  --rank_mixer_mode full \
  --d_model 136 \
  --emb_dim 64 \
  --num_hyformer_blocks 2 \
  --num_heads 4 \
  --seq_encoder_type transformer \
  --seq_max_lens seq_a:256,seq_b:256,seq_c:128,seq_d:512 \
  --use_time_buckets \
  --loss_type bce \
  --lr 1e-4 \
  --sparse_lr 0.05 \
  --dropout_rate 0.05 \
  --seq_top_k 100 \
  --batch_size 256 \
  --num_workers 8 \
  --buffer_batches 20 \
  --valid_ratio 0.1 \
  --split_by_timestamp \
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

`seq_top_k=100` 在 `seq_encoder_type=transformer` 下不生效，只作为 marker 写入 `train_config.json`。让它真实生效需要另开 `6.B longer encoder` 消融。

## 8. Evaluation 设计

`evaluation/FE08/infer.py` 必须：

```text
1. 优先读取 checkpoint 目录下的 schema.json
2. 读取 train_config.json 作为模型结构唯一来源
3. 读取 ns_groups.may7.json
4. 读取 dropped_feats.may7.json
5. 读取 fe08_transform_stats.json
6. 读取 fe08_dense_normalization_stats.json
7. 对 raw eval parquet 运行同一份 transform 逻辑
8. 禁止 re-fit / re-select / 读取 eval label 更新统计
9. strict load checkpoint
10. 输出 raw input rows / transformed rows / final dataset samples 三类日志
```

必须强校验：

```text
d_model == 136
rank_mixer_mode == full
user_ns_tokens == 5
item_ns_tokens == 2
num_queries == 2
use_token_gnn == true
token_gnn_layers == 4
token_gnn_graph == full
token_gnn_layer_scale == 0.15
```

## 9. 消融顺序

为了诊断失败原因，记录如下消融链路：

| 顺序 | 实验名 | 内容 | 预期 Eval AUC |
| ---: | --- | --- | ---: |
| 0 | GNN-baseline replay | 0.8159 GNN 配置复跑 | ≈ 0.8159 |
| 1 | + drop >80% missing | user_int/item_int schema 缩窄 | 0.8155 ~ 0.8165 |
| 2 | + item_dense token + norm | `{86,91,92}` + full + d_model=136 | 0.8165 ~ 0.8180 |
| 3 | + sequence sort | row/domain/event_time desc | 0.8170 ~ 0.8185 |
| 4 | + new int features | item_int 89/90/91 + user_int 130/131 | 0.8175 ~ 0.8195 |
| 5 | + seq lens | 256/256/128/512 | 0.8180 ~ 0.8200 |
| 6 | May7 main | dropout=0.05 + seq_top_k marker | 0.8185 ~ 0.8215 |
| 6.B | optional longer | step 6 + `seq_encoder_type=longer` | 0.8130 ~ 0.8210 |

第一轮可以直接跑 step 6；若 step 6 失败，再按链路回拆。

## 10. 验收标准

```text
Eval AUC >= 0.8185:
  可替代 0.8159 baseline，成为新主线候选。

Eval AUC >= 0.8200:
  进入多 seed / SWA / EMA 阶段。

Eval AUC < 0.8159:
  不接受，按 step 2/3/4/5 回拆定位。
```

主线通过后，下一轮才考虑：

```text
1. seq_encoder_type=longer + seq_top_k=100
2. d_model=119/153 capacity 消融
3. dropout=0.03/0.025 调参
4. 多 seed / EMA / SWA
```
