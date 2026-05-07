# FE-08 强力检查与实验结果预测

## 0. 结论先行

FE-08 的强检查重点不是“能不能训练启动”，而是确认以下 4 件事同时成立：

```text
1. 数据侧新增特征与 Claude FE08 完全一致。
2. Token 拓扑满足 full RankMixer 的 T=17 整除约束。
3. TokenGNN 评估侧与训练侧模型完全同构。
4. eval transform 只复用 checkpoint sidecar，不在 eval 上重新 fit。
```

最关键的风险排序：

| 排名 | 风险 | 严重性 | 处理原则 |
| ---: | --- | :---: | --- |
| 1 | eval 侧模型仍继承 FE07 no-GNN guard | 高 | `evaluation/FE08/model.py` 必须支持 TokenGNN |
| 2 | FE08 sidecar 未复制进 checkpoint | 高 | `trainer.py` copy 列表必须补全 |
| 3 | sequence sort 改变 list 对齐 | 高 | 同一 permutation 重排 timestamp 与所有 side 列 |
| 4 | item_dense 87/88 被误启用 | 高 | builder 白名单 + risky flag 双保险 |
| 5 | d_model/T 配置不合法 | 高 | `full + d_model=136 + T=17` 强校验 |
| 6 | eval transform 偷偷 re-fit | 高 | eval 只读取 sidecar |
| 7 | `seq_top_k=100` 被误解为 active | 中 | transformer 主线下 marker only |

## 1. 当前代码就绪度

| 模块 | 当前状态 | FE-08 处理 |
| --- | --- | --- |
| `dataset.py` item_dense | 已支持 schema-driven item_dense 读取 | 不改 |
| 根 `model.py` TokenGNN | 已支持 TokenGNN / layer_scale / item_dense token | 不改 |
| `train.py` split_by_timestamp | 已支持 | 复用 |
| `train.py` seq_top_k | 已有参数，仅 longer active | 加 no-op warning |
| `trainer.py` sidecar copy | 已有 FE00/FE06/FE07 sidecar | 增加 FE08 sidecar |
| `build_fe07_p012_domain_dataset.py` | 已有 FE07 builder，可 fork | 改成 FE08 builder |
| `evaluation/FE07/infer.py` | 已有 raw eval transform + strict infer 框架 | fork 到 FE08，去掉 no-GNN 限制 |
| `evaluation/FE07/model.py` | 明确拒绝 `use_token_gnn=true` | FE08 不能原样照搬 |

结论：

```text
FE-08 不是从零写。
最合理路径是 fork FE07 builder/eval 框架，保留根 dataset/model 的能力，
补上 FE08 sidecar、TokenGNN eval 支持和 May7 特征生成。
```

## 2. 数据正确性强检查

| 检查 | 必须满足 | 失败后果 |
| --- | --- | --- |
| missing threshold | `0.80`，不是 FE07 的 `0.75` | 与 Claude FE08 不一致 |
| drop 范围 | `user_int + item_int` | item 高缺失字段残留 |
| dropped sidecar | `dropped_feats.may7.json` 含 user/item 两组 | eval 无法复现 schema |
| item_dense fid | 默认 `[86,91,92]` | item_dense token 语义错误 |
| risky fid | 87/88 默认 raise | 重演 FE01 full eval 崩盘风险 |
| item_int 89 | vocab=3, value in `{0,1,2}` | embedding 越界或语义错 |
| item_int 90 | vocab=7, bucket=`0,1,2,4,8` | count 分桶不一致 |
| item_int 91 | vocab=64, non-match=0 | 最近匹配 time bucket 错 |
| user_int 130 | vocab=25, hour in `1..24` | 0 点被 padding 吃掉 |
| user_int 131 | vocab=8, dow in `1..7` | 周期特征被 padding 吃掉 |
| sequence sort | row/domain 级按 event_time desc | truncation 仍可能丢最近事件 |
| dense stats | train row groups only | eval 泄漏或不可复现 |

## 3. 防泄漏检查

| 检查 | 必须满足 |
| --- | --- |
| match 统计 | 只统计 `event_time <= sample_timestamp` 的历史事件 |
| min_match_delta | 使用 sample timestamp 与历史 sequence timestamp 差值 |
| item_int_91 | 最近匹配 time bucket 只来自历史匹配事件 |
| hour/dow | 只来自 `timestamp`，不来自 `label_time` |
| item_dense 86 | prefix item_total_frequency，不包含当前样本 |
| item_dense 91/92 | 来自历史 match，不使用 eval label |
| dense normalization | eval 不重新 fit |
| missing drop | eval 不重新 audit/drop，只按 checkpoint sidecar |
| label_time | 不作为输入特征 |

## 4. Schema 与 NS groups 检查

### 4.1 schema

必须出现：

```json
"item_dense": [[86, 1], [91, 1], [92, 1]]
```

必须包含：

```text
item_int: 89 vocab=3
item_int: 90 vocab=7
item_int: 91 vocab=64
user_int: 130 vocab=25
user_int: 131 vocab=8
```

必须不包含：

```text
item_dense_feats_87
item_dense_feats_88
user/item purchase_frequency
avg_delay as input feature
```

### 4.2 ns_groups.may7.json

必须满足：

```text
dense fid 不写入 ns_groups
user_int 130/131 进入 user temporal/behavior group
item_int 89/90/91 进入 item target matching group
missing drop 后不存在的 fid 不残留
```

### 4.3 item_int 91 与 item_dense 91

必须在日志里用 prefix 打印，避免阅读歧义：

```text
item_int_feats_91 vocab=64, meaning=latest_match_time_bucket
item_dense_feats_91 stats={mean,std}, meaning=log1p(min_match_delta)
```

## 5. 模型 / token 结构检查

必须打印并满足：

```text
user_ns_tokens = 5
item_ns_tokens = 2
has_user_dense = true
has_item_dense = true
num_ns = 9
num_queries = 2
num_sequences = 4
T = 2*4 + 9 = 17
d_model = 136
136 % 17 = 0
rank_mixer_mode = full
```

TokenGNN：

```text
use_token_gnn = true
token_gnn_layers = 4
token_gnn_graph = full
token_gnn_layer_scale = 0.15
```

禁止：

```text
rank_mixer_mode = ffn_only
d_model = 128
output_include_ns = true
seq_encoder_type = longer in first main run
```

## 6. Evaluation 检查

| 检查 | 必须满足 | 失败处理 |
| --- | --- | --- |
| schema 来源 | 优先 checkpoint `schema.json` | 不允许用 raw eval schema 代替 |
| train_config | checkpoint `train_config.json` 存在 | 缺失则 fail-fast |
| ns groups | checkpoint `ns_groups.may7.json` 存在 | 缺失则 fail-fast |
| transform stats | `fe08_transform_stats.json` 存在 | 缺失则 fail-fast |
| dense stats | `fe08_dense_normalization_stats.json` 存在 | 缺失则 fail-fast |
| dropped feats | `dropped_feats.may7.json` 存在 | 缺失则 fail-fast |
| model load | strict=True | missing/unexpected key 立即失败 |
| GNN support | eval model 支持 TokenGNN | 不允许沿用 FE07 no-GNN guard |
| row logs | raw rows / transformed rows / dataset samples 分开打印 | 用于定位 transform 与 dataset 问题 |

## 7. Sidecar 复制检查

`trainer.py` 保存 checkpoint 时必须复制：

```text
schema.json
train_config.json
ns_groups.may7.json
fe08_transform_stats.json
fe08_dense_normalization_stats.json
dropped_feats.may7.json
feature_engineering_stats.json
docx_alignment.fe08.json
```

若 sidecar 只存在于 builder 输出目录但没进 checkpoint，eval 不能继续，应直接报错。

## 8. 模块级预测

| 模块 | 预计 Eval Δ | 置信度 | 证据 |
| --- | ---: | :---: | --- |
| GNN baseline replay | 0 | 高 | 历史 0.815064 ~ 0.8159 |
| drop >80% missing | `0 ~ +0.0005` | 中 | FE00 cleanup 有收益，但 0.80 更保守 |
| item_dense 86/91/92 + norm | `+0.0005 ~ +0.0015` | 中高 | FE01B match 与 item_dense token 已有证据 |
| sequence recency sort | `+0.0005 ~ +0.0010` | 中 | 修复 truncation 可能丢最近事件的问题 |
| item_int 91 | `+0.0002 ~ +0.0008` | 中 | 在 FE01B match 基础上补最近匹配时间桶 |
| user_int 130/131 | `+0.0003 ~ +0.0010` | 中 | 当前样本时间上下文低成本 |
| seq lens 256/256/128/512 | `+0.0003 ~ +0.0008` | 中 | 压缩 domain_c，保留 domain_d |
| d_model=136 + dropout=0.05 | `+0.0003 ~ +0.0010` | 中 | capacity 上调与正则配套 |
| seq_top_k=100 marker | 0 | 高 | transformer 主线不消费该参数 |

主预测：

```text
FE08-May7-main = 0.8185 ~ 0.8215
base case ≈ 0.8200
```

## 9. 实验结果预测

| 实验 | 组成 | Eval AUC 保守 | Eval AUC Base | Eval AUC 乐观 | 结论规则 |
| --- | --- | ---: | ---: | ---: | --- |
| step 0 | GNN baseline replay | 0.8150 | 0.8159 | 0.8162 | 复刻结构起点 |
| step 1 | + drop >80% missing | 0.8155 | 0.8160 | 0.8165 | 主要看无退化 |
| step 2 | + item_dense token + norm | 0.8165 | 0.8172 | 0.8180 | item_dense 生效 |
| step 3 | + sequence sort | 0.8170 | 0.8177 | 0.8185 | 检查排序正确 |
| step 4 | + 89/90/91/130/131 | 0.8175 | 0.8185 | 0.8195 | 新 int 特征主增益 |
| step 5 | + seq lens | 0.8180 | 0.8190 | 0.8200 | 序列策略微调 |
| step 6 | May7 main | 0.8185 | 0.8200 | 0.8215 | 主线验收 |

## 10. 失败场景与回滚

| 失败现象 | 最可能原因 | 第一回滚动作 |
| --- | --- | --- |
| `d_model=128 must be divisible by T=17` | run 脚本没改成 136 | 改 `--d_model 136` |
| eval 报 `use_token_gnn=true` 不支持 | FE08 model 原样继承 FE07 no-GNN guard | 用根 `model.py` 的 TokenGNN 实现修 FE08 model |
| checkpoint missing FE08 sidecar | `trainer.py` copy 列表没补 | 补 sidecar 并重训/重存 ckpt |
| AUC 大幅掉到 0.78 左右 | risky purchase/delay fid 误入 | 检查 schema 中 87/88，立即移除 |
| step 3 后掉分 | sequence sort 错位 | 对 5k rows diff：所有 side list length 与 timestamp 对齐 |
| item_int 130 OOB | hour 未 +1 或 vocab 写错 | `hour=(ts//3600)%24+1`, vocab=25 |
| item_int 91 OOB | vocab 写成 63 或 bucket clip 错 | vocab=64，非匹配=0，匹配=1..63 |
| eval 速度极慢 | eval transform per-row Python loop 过重 | 借鉴 FE06/FE07 hot path，做 projected reads + NumPy/Arrow vectorization |
| valid 涨 eval 不涨 | split 或新增特征过拟合 | 回拆 step 4/5/6，优先保留 item_dense 与 match |
| OOM | d_model=136 + batch=256 资源不足 | 先降 batch 到 224/192，不先改结构 |

## 11. 代码实现拆分建议

### M0: builder scaffold

```text
copy build_fe07_p012_domain_dataset.py -> build_fe08_may7_dataset.py
copy to tools/build_fe08_may7_dataset.py
改 FE07 命名和 sidecar 名称
先保留 FE07 89/90/86/91/92 基础生成
```

验收：

```text
demo 数据能写出 schema.json / ns_groups.may7.json / fe08_transform_stats.json
```

### M1: May7 features

```text
missing drop user_int + item_int
item_int_feats_91
user_int_feats_130/131
sequence sort
risky item_dense guard
```

验收：

```text
schema 包含 89/90/91/130/131 和 item_dense 86/91/92
所有新增 int 最大值 < vocab_size
```

### M2: run script and trainer sidecar

```text
run_fe08_may7_full.sh
train.py warning/check
trainer.py FE08 sidecar copy
```

验收：

```text
训练日志打印 num_ns=9, T=17, d_model=136
best_model 目录包含 FE08 sidecars
```

### M3: evaluation/FE08

```text
copy evaluation/FE07 -> FE08
remove no-GNN rejection
load FE08 sidecars
eval transform only
strict checkpoint load
```

验收：

```text
raw eval parquet 自动 materialize FE08 columns
strict load 通过
输出 Total test samples
```

## 12. 最终提交前 checklist

```text
[ ] FE08 文件夹和文档已建
[ ] build_fe08_may7_dataset.py 与 tools 副本同步
[ ] evaluation/FE08 builder 与训练 builder 逻辑同步
[ ] missing_threshold=0.80
[ ] item_dense_fids=[86,91,92]
[ ] risky 87/88 默认报错
[ ] sequence_sort_by_recency=true
[ ] item_int 89/90/91 schema 正确
[ ] user_int 130/131 schema 正确
[ ] ns_groups.may7.json 不含 dense fid
[ ] rank_mixer_mode=full
[ ] d_model=136
[ ] TokenGNN 4/full/0.15
[ ] output_include_ns 未启用
[ ] eval 不 re-fit
[ ] ckpt sidecar 完整
[ ] demo smoke test 训练启动
[ ] demo eval smoke test 完成
```

通过以上检查后，FE-08 才能进入全量训练和平台 eval。
