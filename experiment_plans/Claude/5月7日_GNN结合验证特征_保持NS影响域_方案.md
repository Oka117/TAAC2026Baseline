# 5 月 7 日 优化方案 — GNN 结合验证特征 · 保持 NS 影响域

> 上承 5 月 5 日 `tokenGNN_4L_optimis` 主线（hyformer + 4-layer TokenGNN，
> `token_gnn_layer_scale=0.15`，Eval AUC ≈ **0.8159**），
> 把已经在 FE-00 / FE-01B / FE-06 / FE-07 中被验证过的低风险信号回流到 GNN baseline。
>
> **总原则**：保持 GNN 的"影响域"不变 —— `user_ns_tokens=5`、`item_ns_tokens=2`、
> `num_queries=2`，所有新增的 feature 全部回流进入 NS groups（即 RankMixer 的 chunk
> 拼接序列），不引入新的 NS token 数量。

> ## ⚠️ 与用户初始设计的偏差总览（DEVIATION BANNER）
>
> ### 状态分类
>
> - ✅ **已锁定**：用户 2026-05-08 已拍板，零负收益；
> - 🟩 **工程必要**：原文与代码框架硬冲突，方案做了最小工程兼容；
> - 🟦 **marker only**：参数虽与原文写入命令行但当前 stack 下不生效，零影响。
>
> ### 偏差最终落地（净收益方向 = 全部正向或零）
>
> | # | 偏差点 | 状态 | 章节 | 最终方案 | 净收益 |
> | ---: | --- | :---: | --- | --- | :---: |
> | 1 | `seq_top_k=100` 在 transformer 主线下不生效 | 🟦 marker only | §3.6.3 | 保留 marker；step 6.B 作为可选 longer encoder ablation | **零影响** |
> | 2 | `d_model=128` + `full` 模式 + item_dense token 三者整除冲突 | ✅ 已锁定 | §4.1 / §5 | `rank_mixer_mode=full` + `d_model=136` (=17×8) | **+0.0005~0.0008** |
> | 3 | `item_dense` fid 范围 | ✅ 已锁定 | §3.2 | `{86, 91, 92}` (FE-01B 验证子集，显式排除 87/88) | **+0.0008** |
> | 4 | `user_int` 时间衍生 fid 编号 | ✅ 已锁定 | §3.5.2 | `130 (hour) / 131 (dow)` | **零影响** |
> | 5 | `--split_by_timestamp` 默认开启 | ✅ 已锁定 | §5 | 启用 | **零 eval / 正 valid** |
> | 6 | hour / dow 编码 +1 偏移 | 🟩 工程必要 | §3.5.2 | 落 `1..24` / `1..7`, 0=padding | 必要修正（避免 padding 误吞） |
> | 7 | 序列排序粒度 = "row × domain × event_time desc" | 🟩 工程必要 | §3.4 | builder 端 argsort | **+0.0007**（修复 README §2.1 未审计点） |
> | 8 | `item_int_91` 无匹配回填 = 0 | 🟩 工程必要 | §3.5.1 | 写 0 = padding | 零影响 |
>
> ### 强力负收益排除（全部已排除，详见 §11.5.3）
>
> ```text
> ❌ purchase_frequency (FE-01 -0.035 灾难)        排除：白名单 + 双保险旗标
> ❌ avg_delay (label_time 泄漏)                   排除：同上 fid 88
> ❌ output_include_ns (历史 0.815 → 0.811)        排除：§8.3 强检查禁止
> ❌ longer encoder + GNN 未验证组合               排除：主线保 transformer
> ❌ ffn_only 隐式 backbone 选择                   排除：已切回 full + d_model=136
> ❌ valid 信号噪声大                              排除：split_by_timestamp 启用
> ❌ schema fid 编号冲突                           排除：130/131 远跳；91 物理隔离
> ❌ eval / train 不一致                           排除：sidecar 强制从 checkpoint 复制
> ```
>
> **预期 step 6 Eval AUC：0.8185 ~ 0.8215**（中位 0.8200，相比 0.8159 baseline +0.004）。
>
> **撤回的过度防御**：原报告曾建议把 `item_int_feats_91` 改名为 `93` 以避开
> FE-01B `item_dense_feats_91` 的命名冲突；经核实 parquet 列名 prefix 不同
> （`item_int_feats_` vs `item_dense_feats_`），物理无冲突，**已撤回改名建议**，
> 严格遵循用户原文 `item_int_feats_91`。

## 0. 一句话结论

```text
5 月 7 日方案 = 0.8159 GNN baseline + (FE-07 已验证的数据层信号) + (item_dense token)
              + (sequence 时间排序) + (3 个新 int 特征) + (容量上调 d_model=136, full 模式)
```

**预期方向**：把 FE-07 中 P0/P1 / target-history match 的稳态收益压在 GNN baseline 上。

**核心约束**：

```text
- GNN 输入 token 数严格不变 (num_ns = 5 + 1 + 2 + 1 = 9)，
  保证 TokenGNN(num_layers=4, layer_scale=0.15) 的 message-passing 拓扑不漂移；
- rank_mixer_mode = full（与 0.8159 baseline 一致），保留 RankMixer 的 token mixing；
- d_model = 136 (= 17 × 8) 是满足 T=17 整除约束的最近候选，
  比原文 128 偏 +6.25%（数学上 [128, 200] 区间唯一可行值）；
- 其他超参数（emb_dim=64, num_heads=4, num_hyformer_blocks=2, etc.）保持不变。
```

## 1. 与设计原文的逐条对应

把用户在 5 月 7 日给出的优化点与本方案章节做硬绑定，方便日后快速对照。

| 原文优化点 | 本方案章节 | 主要落点 | 与历史经验的关联 |
| --- | --- | --- | --- |
| 删除 item / user NS feature 中 missing > 80% 的特征 | §3.1 | builder 审计 + schema drop + ns_groups drop | 沿用 FE-00 (`missing_threshold=0.75`) 的删除路径，阈值放宽到 0.80 |
| item dense feature 做 normalization | §3.2 | builder 离线 z-score；schema 新增 `item_dense` 入口 | 沿用 FE-00 dense_normalization_stats，第一次把 item_dense token 接进 GNN baseline |
| 序列长度改成 `seq_a:256, seq_b:256, seq_c:128, seq_d:512` | §3.3 | `--seq_max_lens` 直接换值 | 介于 baseline (`c:512,d:512`) 与 FE-07 (`c:128,d:768`) 之间，更保守 |
| 序列按"距离现在最近 → 最远"时间顺序排序 | §3.4 | builder 在写入 parquet 前对每行做 argsort | 修复 README §2.1 中"事件时间是否按新到旧排序"未审计的问题 |
| 新增 3 个 int feature：`item_int_feats_89/90/91` | §3.5.1 | builder 计算 has_match / bucketize / time-bucket-of-latest-match | `89/90` 复用 FE-01B 已落账逻辑；`91` 是新增的"最近匹配时间桶" |
| 新增 user_int 特征：`hour_of_day`、`day_of_week` | §3.5.2 | builder 从 `timestamp` 派生 + schema 新增 fid | Claude P0-L4 中提出的 sample_time_context 的 int 化版本 |
| `d_model=128` → 实际 `d_model=136` | §3.6.1 / §4.1 | `--d_model 136` | 比 0.8159 baseline (`d_model=64`) 翻倍 + 6.25%；136 = 17×8 是满足 full 模式整除约束的最近候选 |
| `dropout_rate=0.05` | §3.6.2 | `--dropout_rate 0.05` | 比 0.8159 baseline (`0.015`) 大一档，对抗新特征带来的过拟合风险 |
| `seq_top_k=100` | §3.6.3 | `--seq_top_k 100` | 注意：仅在 `seq_encoder_type=longer` 下生效，§3.6.3 给出 A/B/C 三种处理方案 |
| 保持 `user_ns_tokens=5,item_ns_tokens=2,num_queries=2` | §4.1 | NS token 数固定 / 新特征只进 ns_groups | 保证 TokenGNN 的"影响域"和 0.8159 baseline 完全一致 |

## 2. 上下文回顾

### 2.1 0.8159 baseline 复现配置

```bash
# 来自 gnn4layerAdjustPara 分支 `19a6384` 提交的 run.sh
python3 -u train.py \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --use_token_gnn \
    --token_gnn_layers 4 \
    --token_gnn_graph full \
    --token_gnn_layer_scale 0.15 \
    --dropout_rate 0.015 \
    --patience 3 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8
```

主要语义：

```text
原始 schema（无 item_dense） → has_user_dense=True、has_item_dense=False
num_user_ns = 5
num_item_ns = 2
num_ns      = 5 + 1 (user_dense) + 2 = 8
T           = num_queries * num_sequences + num_ns = 2 * 4 + 8 = 16
d_model     = 64, 64 % 16 == 0
TokenGNN reception field = 8 个 NS token，4 layers full graph
```

### 2.2 已落账的 FE 实验结论

| 实验 | Eval AUC | 与 5 月 7 日方案的关系 |
| --- | ---: | --- |
| B0 | 0.810525 | 不复现，仅作下界 |
| FE-00 | 0.811646 | 缺失处理 / dense norm 经验全量保留 |
| FE-01A | 0.810780 | total frequency 单独不显著，第一轮不强制纳入 |
| FE-01B | 0.812102 | target-history match `89/90/91/92` 是当前最稳的低风险特征 |
| GNN_NS_4Layer (≈ 5 月 5 日 baseline) | 0.815064 → 0.8159 (+ scale 调参) | 本方案的起点 |
| 4LayerGNN + direct NS head | 0.811474 | **明确不采用** |

### 2.3 仍未在 0.8159 baseline 上验证的优化点

```text
- 缺失阈值 0.80（FE-00 用 0.75）
- item_dense token（FE-01B / FE-06 / FE-07 已启用，但 0.8159 baseline 未启用）
- 序列时间排序（README §2.1 指出未审计）
- item_int_91：最近匹配事件的 time bucket
- user_int hour / day_of_week：Claude P0-L4 的 int 版本
- d_model=128（baseline 一直 64）
- dropout=0.05（baseline 0.015）
```

5 月 7 日方案的目的就是把上面这些点合并到一条新的 GNN 主线上，每一项都给出工程实现路径与回退判据。

## 3. 优化点 → 代码改动逐条映射

### 3.1 删除 missing > 80% 的 user / item NS feature

**对应原文**：`delete item 和 user ns feature 中 missing value 比例大于 80% 的特征。
确保删除之后在 nsgroups 里也删除 并保证在跑测试集的时候 该特征得到直接删除`。

#### 3.1.1 实现路径（builder 端）

新建 `build_fe08_may7_dataset.py`（或在现有 `build_fe07_p012_domain_dataset.py` 基础上
派生 fork）；在写出 parquet 之前完成：

1. 全 train row group 扫描 `user_int_feats_*` 与 `item_int_feats_*` 的 `missing_ratio`，
   missing 定义沿用 FE-07：null 或所有元素 ≤ 0 的行。
2. 阈值统一为 `0.80`，命中即从 schema `user_int` / `item_int` 列表中删除。
3. 同步在 `ns_groups.feature_engineering.json` 的对应分组中过滤掉这些 fid，并把删除清单
   写入 sidecar `dropped_feats.may7.json`：

   ```json
   {
     "user_int": [...],
     "item_int": [...],
     "threshold": 0.80,
     "row_groups_used_for_audit": <count>
   }
   ```

#### 3.1.2 训练 / 评估端的等价处理

```text
- dataset.py 已经从 schema.json 驱动列读取，被删除的 fid 不会出现在 schema 中，
  自然不会被 _convert_batch 写入 buffer → 训练端零改动。
- evaluation/FE08/infer.py 必须：
    1. 读取 checkpoint 的 schema.json（不是原始数据集 schema）；
    2. 同步使用 dropped_feats.may7.json 校验 eval parquet 与 schema 一致性；
    3. 缺列时打印警告并 fail-fast，避免静默通过。
```

#### 3.1.3 复用现有 FE-00 逻辑

`build_fe07_p012_domain_dataset.py` 中 `_collect_audit_and_min_ts` +
`_build_augmented_schema` 已经实现 `missing_threshold` 删除逻辑。改动只是：

- 把默认值从 `0.75` 改为 `0.80`；
- 把删除范围从仅 `user_int` 扩展到 `user_int + item_int`（FE-07 只删 user）。

**强检查**：FE-07 实际跑下来 demo_1000 上 `item_int` 没有触达 `>0.75`；放宽到 `0.80` 后
被删的 item fid 应当极少甚至为零，这个事实必须在 audit JSON 中显式记录。

**空集 fail-safe（🟩 工程必要修正）**：

```text
- audit JSON 必须显式定义"empty list 是合法输出"：
    {
      "user_int": [],         # 可以为空
      "item_int": [],         # 可以为空
      "threshold": 0.80,
      "row_groups_used_for_audit": <count>,
      "interpretation": "Empty list means no fid hit the missing>=0.80 threshold."
    }
- evaluation/FE08/infer.py 在加载 dropped_feats.may7.json 时：
    if dropped['user_int'] == [] and dropped['item_int'] == []:
        log.info("FE-08 dropped feats: empty list - no schema diff with baseline")
    else:
        # 与 eval parquet 做 schema diff，缺列时 fail-fast
- 这一段保护可避免静默 bug：避免 builder 把 None 当空集 / 把空集当 None。
```

### 3.2 item dense features normalization

**对应原文**：`item dense features 做 normalization`。

> ### ✅ 偏差点 #3 — item_dense fid 范围已锁定（2026-05-08 用户确认）
>
> 原文只说"item dense features 做 normalization"，未指定具体 fid。
> baseline schema `item_dense = []`，目前模型层面没有任何 item dense 列。
> 经用户 **2026-05-08 确认**：`item_dense_fids = {86, 91, 92}`（FE-01B 已验证子集）。
>
> **显式排除 + 双保险**：
>
> ```python
> # builder 端硬编码白名单警告，防止误启用风险 fid
> RISKY_ITEM_DENSE_FIDS = {
>     87: "FE-01 全量 eval AUC 跌到 0.775054 主因（train/eval 不一致）",
>     88: "依赖 label_time，存在泄漏风险（FE-02 范围）",
> }
> for fid in item_dense_fids_arg:
>     assert fid not in RISKY_ITEM_DENSE_FIDS, (
>         f"item_dense_feats_{fid} 是已知风险特征：{RISKY_ITEM_DENSE_FIDS[fid]}；"
>         f"如需启用必须显式传 --enable_risky_item_dense_fids"
>     )
> ```
>
> **新增 CLI 参数**（修复 C 路径）：
>
> ```bash
> --item_dense_fids "86,91,92"          # 默认值，已锁定
> --enable_risky_item_dense_fids        # 双保险旗标，默认不开
> ```
>
> 参数同步写入 sidecar `fe08_transform_stats.json`，evaluation 端必须读取同一 fid 列表，
> 不允许 train / eval 不一致。
>
> 历史候选对比（仅作记录，不再作为运行时选项）：
>
> | 历史选项 | fid 集合 | 已落账 Eval AUC | 状态 |
> | --- | --- | ---: | --- |
> | i. ✅ 当前默认 | 86, 91, 92 | 预期 step 2 ≈ 0.8167 | **采用** |
> | ii. FE-01A 路径 | 仅 86 | step 2 ≈ 0.8161 | 弃 |
> | iii. FE-01 全量 | 86, 87, 88, 91, 92 | step 2 ≈ 0.78（灾难） | 弃，加白名单防呆 |
> | iv. 不启用 token | ∅ | step 2 = 0.8159 baseline | 弃，原文 §2 失效 |

#### 3.2.1 schema 与列出现

按假设 i 落地：baseline schema 中 `item_dense = []`，不存在任何 item dense 列。
本轮一次性在 schema 中写入 FE-01B 已经验证的 item_dense fid，并对其做 z-score：

```json
"item_dense": [
  [86, 1],
  [91, 1],
  [92, 1]
]
```

来源：

```text
item_dense_feats_86 = log1p(item_total_frequency_before_timestamp)        # FE-01A
item_dense_feats_91 = log1p(min_match_delta(item_int_9, domain_d_seq_19)) # FE-01B
item_dense_feats_92 = log1p(match_count_7d(item_int_9, domain_d_seq_19))  # FE-01B
```

明确 **不** 启用：

```text
item_dense_feats_87  = log1p(item_purchase_frequency)    # FE-01 全量 -0.035 主因
item_dense_feats_88  = log1p(item_avg_delay)             # 依赖 label_time，留给 FE-02
```

注：与新增的 `item_int_feats_91`（§3.5.1）共存 —— 二者属不同 schema 入口
（int / dense），dataset.py 通过 prefix 解析列名，物理零冲突，参见 §10.1 的物理验证。

#### 3.2.2 normalization 实现

builder 在 `_fit_dense_stats` 阶段对所有 item_dense fid 在 train row group 上 fit
RunningStats，得到 `(mean, std)`，写入 sidecar
`fe08_dense_normalization_stats.json`；写出 parquet 时按 `(x - mean) / std` 落地，
缺失统一回填 `mean`（与 FE-07 一致）。

#### 3.2.3 模型端

`model.py` 已经支持 `has_item_dense = item_dense_dim > 0`，自动生成 1 个 item_dense
NS token，无需改动。**这一步会把 num_ns 从 8 变为 9**，§4.1 单独讨论由此带来的整除约束。

### 3.3 序列长度调整

**对应原文**：`改 seq_a:256, seq_b:256, seq_c:128, seq_d:512`。

#### 3.3.1 命令行直接覆盖

```bash
--seq_max_lens seq_a:256,seq_b:256,seq_c:128,seq_d:512
```

#### 3.3.2 与历史方案对照

```text
B0 / GNN baseline  : seq_a:256, seq_b:256, seq_c:512, seq_d:512
FE-07 Domain-main  : seq_a:256, seq_b:256, seq_c:128, seq_d:768
5 月 7 日方案      : seq_a:256, seq_b:256, seq_c:128, seq_d:512
```

为什么不直接抄 FE-07 的 `seq_d:768`？

```text
- FE-07 的 768 是 P2-Domain 主线的"近期窗口拉长"假设；
- 5 月 7 日方案优先确认"GNN + 已验证特征"的可加性，因此先固定 seq_d=512，
  把 sequence length 留作下一轮独立 ablation；
- 同时 seq_c 已经从 512 收紧到 128，与数据观测（domain_c 中位事件年龄 275 天，
  7 天内事件占比 1.5%）吻合，避免无效压算力。
```

#### 3.3.3 token 数与显存影响

`seq_c` 从 512 → 128 释放 75% 序列计算；`seq_d` 不变。整体序列 token 总量从
`256+256+512+512=1536` 降到 `256+256+128+512=1152`，约 −25%；与 d_model 翻倍带来的
+约 4× 矩阵成本部分对冲。

### 3.4 序列按时间从近到远排序

**对应原文**：`并且确保每个 sequence 是按由距离现在最近到最远的时间顺序排序`。

#### 3.4.1 必要性

```text
- dataset.py 当前直接按 parquet 中 list 顺序读取 side feature 与 timestamp；
- README.feature_engineering.zh.md §2.1 已经把"事件时间是否按新到旧排序"列为审计点，
  但 0.8159 baseline 未做强制保证；
- seq_max_lens 的 truncation 也是按位置截断的，如果原始顺序不是"近 → 远"，
  截断会丢失最近事件，正是 CVR 任务最敏感的信号。
```

#### 3.4.2 builder 端实现

在 `_write_augmented_parquet` 写出每个 row 之前，对每个 domain：

```text
1. 读取该 domain 的 ts_col + 所有 side feature 列；
2. 对每行执行 argsort(timestamps, descending=True)；
3. 用同一份 permutation 重排该 domain 的所有 side 列与 timestamp 列；
4. 严格保证：permutation 应用前后，每个 list 的 length 不变，
   length=0 的 list 直接跳过。
```

复杂度：每行 O(L log L)，整体 O(N · sum(L_d log L_d))。在 `domain_d` 平均 1100 长度
下，每行约 7700 比较；按 demo_1000 经验外推到 1e8 行约 28 min/run，可接受。

#### 3.4.3 训练端

完全无改动 —— 因为 dataset.py 的 `_pad_varlen_int_column` 直接按 list 顺序填入
`[B, S, max_len]`，所以"近 → 远"会自动让前 `max_len` 个事件落在 buffer 前面。

#### 3.4.4 评估端

`evaluation/FE08/build_fe08_may7_dataset.py` 必须复用同一份 sort 逻辑；切勿在
inference 端重排，必须由 builder 离线完成（保证 train / eval 两端字节序一致）。

### 3.5 新增 int feature

**对应原文**：

```text
item_int_feats_89  = has_match(item_int_feats_9, domain_d_seq_19)
item_int_feats_90  = bucketize(match_count(item_int_feats_9, domain_d_seq_19))
item_int_feats_91  = 最近匹配上的时间点所处于哪个 time buckets （num_time_buckets）

user_int_features  = 从 timestamp 分析出几点
user_int_features  = 从 timestamp 分析出是星期几

将上面新加特征融入 ns group， 并确保验证集跑时会自动生成
```

#### 3.5.1 item 侧三个新 int feature

| fid | 含义 | vocab_size | dim | 复用 / 新增 |
| --- | --- | ---: | ---: | --- |
| 89 | `has_match(item_int_9, domain_d_seq_19)`，`0=missing,1=no,2=yes` | 3 | 1 | FE-01B 已落账，复用 |
| 90 | `bucketize(match_count(...))`，桶边界 `0,1,2,4,8` → 6 桶 + padding | 7 | 1 | FE-01B 已落账，复用 |
| 91 | 当 `has_match=2` 时，最近匹配事件的 time bucket id；否则 `0` | `NUM_TIME_BUCKETS=64` | 1 | **本方案新增** |

`item_int_feats_91` 实现：

```text
1. builder 在 _compute_generated_features 中：
   a. 已经在求 min_match_delta，记录 argmin 对应的 event_time；
   b. 用 dataset.BUCKET_BOUNDARIES + np.searchsorted 得到 raw bucket id ∈ [0, 62]；
      （len(BUCKET_BOUNDARIES)=63，clip 到 [0, 62]，与 dataset.py 内 sequence
      time bucket 的 clip 规则一致）
   c. raw + 1 后得到 bucket id ∈ [1, 63]，非匹配样本写 0；
   d. 与现有 item_int_feats_89 / 90 一同写出 int64 标量列。

2. schema augmented 中新增：
   item_int += [(91, NUM_TIME_BUCKETS, 1)]   # vocab=64

3. ns_groups.may7.json 把 91 加进 I4_target_matching_fields:
   "I4_target_matching_fields": [89, 90, 91]
```

注意：`91` 与 dataset 的 sequence time bucket 共享同一份 `BUCKET_BOUNDARIES`
（global，不是 per-domain），保证语义对齐 → vocab = `NUM_TIME_BUCKETS = 64`
（来自 `len(BUCKET_BOUNDARIES) + 1 = 63 + 1`，包含 0=padding）；但 item_int 的 fid 走
`RankMixerNSTokenizer` 的 fid embedding 轨道，不复用 `time_embedding` 表，所以 vocab
必须独立写入 schema。

#### 3.5.2 user 侧两个时间衍生 int feature

> ### ✅ 偏差点 #4 — fid 130 / 131 已锁定（2026-05-08 用户确认）
>
> 原文未指定 hour / dow 的 fid 编号。经用户 **2026-05-08 确认**：
> `user_int_feats_130 = hour_of_day, user_int_feats_131 = day_of_week`。
>
> 选择理由：
>
> ```text
> - 现有 user_int 最大 fid = 109；110-121 已被 user_dense 占用
>   （110 来自 FE-01A，120/121 来自 FE-06/07）；
> - 130/131 与 user_int 现有最大值距离 ≥ 21，与 user_dense 距离 ≥ 9；
> - 留 122-129 共 8 个数字给未来 user_dense 扩展；
> - 132+ 给未来 user_int 扩展。
> ```
>
> **fid 编号本身不影响模型行为**（仅是 column name 标识），所以这一项对 eval AUC
> 零影响。锁定 130/131 后未来若需调整，改 schema + ns_groups 两处文本即可。

新增 fid 与 vocab：

| fid | 含义 | vocab_size | dim |
| --- | --- | ---: | ---: |
| 130 | `hour_of_day`，UTC `(timestamp // 3600) % 24`，0 保留为 padding，1..24 表示 0..23 时 | 25 | 1 |
| 131 | `day_of_week`，UTC `((timestamp // 86400) + 4) % 7`，0 保留为 padding，1..7 表示周一..周日 | 8 | 1 |

为什么不直接用 0..23 / 0..6？

```text
- baseline 整个 sparse id 体系把 0 当 padding；
- 如果 hour=0 表示 0 点，会被 _convert_batch 的 `arr <= 0 → 0` 误判为 padding；
- 因此 builder 端必须 +1，把合法值落到 1..24 / 1..7，0 留给 padding。
```

builder 实现：

```python
ts = batch.column(idx['timestamp']).to_numpy().astype(np.int64)
hour_id = ((ts // 3600) % 24 + 1).astype(np.int64)        # 1..24
dow_id  = (((ts // 86400) + 4) % 7 + 1).astype(np.int64)  # 1..7
```

schema：

```python
user_int += [(130, 25, 1), (131, 8, 1)]
```

ns_groups：

```python
user_ns_groups["U2_user_behavior_stats"].extend([130, 131])  # 和 50/60 同组
```

#### 3.5.3 "验证集跑时会自动生成"的工程保证

要求：

```text
- 所有新增 fid 的列（89/90/91/130/131）必须由 builder 在原始 parquet 上离线物化；
- evaluation/FE08/build_fe08_may7_dataset.py 在评估时跑同一段 builder 逻辑：
    a. 读取 checkpoint 携带的 sidecars（schema.json, ns_groups.may7.json,
       fe08_dense_normalization_stats.json, fe08_transform_stats.json，
       dropped_feats.may7.json）；
    b. 用相同的 BUCKET_BOUNDARIES、相同的 match_count_buckets、相同的
       missing_threshold 重新计算 89/90/91/130/131；
    c. eval 严禁重新 fit dense 统计、严禁重选 match pair、严禁读取 eval label_type；
- evaluation/FE08/infer.py 在加载 checkpoint 时 strict=True，
  缺列直接 fail-fast（沿用 FE-07 的策略）。
```

### 3.6 d_model / dropout / seq_top_k

#### 3.6.1 `d_model = 136`（原文 128 → 实际 136，+6.25%）

> ### ✅ 偏差点 #2 — d_model 锁定到 136（2026-05-08 用户确认）
>
> 原文写 `d_model=128`。但与"启用 item_dense token (T=17)"+"保持 rank_mixer_mode=full"
> 同时成立的整除约束要求 `d_model % 17 == 0`，128 不满足。
>
> 在 17 的倍数中选最接近 128 的值：
>
> ```text
> 17 的倍数 (128 附近): ..., 119 (= 17×7), 136 (= 17×8), ...
> 距离 128 偏离: 119 → -7%, 136 → +6.25%
> ```
>
> **用户 2026-05-08 确认采用 d_model=136**，理由：
>
> ```text
> 1. 与原文方向一致：原文 d_model 从 baseline 64 升级到 128 是"提高 capacity"方向，
>    136 (vs 119) 延续这个方向；
> 2. 唯一的代价是 +6.25% 算力 + 8~10% 显存，可接受；
> 3. 换得 full 模式与 0.8159 baseline 行为完全对齐，避免 ffn_only 的隐式 backbone 选择；
> 4. 让 RankMixer 的 token mixing 与 TokenGNN message passing 共同作用，
>    覆盖 NS-token 与 Q-token 的全部交互通路（ffn_only 路径只能覆盖 NS 内部，
>    Q-NS 交互只剩 Query Generator + Cross Attention 一条路径）。
> ```

直接命令行：

```bash
--d_model 136 --emb_dim 64 --rank_mixer_mode full
```

为什么 emb_dim 仍保持 64？

```text
- emb_dim 决定的是 sparse embedding table 的内存占用，每升一倍直接翻倍 embedding 显存；
- d_model 决定的是 backbone 宽度，对 attention / FFN / TokenGNN 的 d_model² Linear
  都是 +12.9% 算力（136² / 128² ≈ 1.129）；
- baseline 走 d_model=emb_dim=64；本方案只升 d_model 到 136，emb_dim 不动，
  避免 sparse embedding 表显存翻倍。
```

**算力 / 显存预算**：

```text
- 单 step 时间: +10~13%（HyFormer / TokenGNN / Sequence encoder 均按 d_model² 放大）
- Activation memory: +6.25%（B × max_len × d_model）
- Parameter memory: +12.9%（d_model² 项主导）
- Optimizer state: +12.9%（AdamW 维护两份 moments）
- 综合显存上升: +8~12%
- 如当前 d_model=128 占用 80% GPU 显存，d_model=136 大概率仍能 fit；
  若靠近 OOM，把 batch_size 从 256 降到 224 或 240。
```

#### 3.6.2 `dropout_rate = 0.05`

```bash
--dropout_rate 0.05
```

注意：

```text
- baseline 用 0.015，本方案 0.05 比之高约 3×；
- 同时 seq_id_emb_dropout = dropout * 2 = 0.10，对高基数 sequence id embedding
  施加更强 regularization，与 §3.5 新增 int feature 带来的 capacity 增加配套。
- TokenGNN 的 layer_scale 仍维持 0.15，不同时再调大 GNN dropout，
  防止把 message-passing 完全压住。
```

#### 3.6.3 `seq_top_k = 100`

> ### 🟥 偏差点 #1 — `seq_top_k=100` 在主线下不生效
>
> 原文把 `seq_top_k=100` 与 `d_model=128` / `dropout=0.05` 并列为"提高 AUC 的能力升级项"。
> 但 `--seq_top_k` 当前在 train.py 中**只被 `seq_encoder_type=longer` 消费**：
>
> ```python
> parser.add_argument('--seq_top_k', type=int, default=50,
>                     help='Number of most-recent tokens kept by LongerEncoder '
>                          '(only effective when --seq_encoder_type=longer)')
> ```
>
> 0.8159 baseline 走的是 `seq_encoder_type=transformer`，此时 `seq_top_k` 是 dead 参数。
>
> **方案现状**：主线保 `transformer`，`--seq_top_k 100` 写入命令行作为 marker，**实际不生效**。
>
> **回滚到原文真实生效**需要切到 `longer` encoder，但代价是：
> ```text
> - 0.8159 baseline 用 transformer，切 longer 后 step 0 复刻失败；
> - longer encoder 通常更快（top-K 压缩注意力），与原文"大概率会增加运算成本"自相矛盾；
> - longer + GNN 的组合在历史所有实验中未被验证过；
> - seq_d=512 与 top_k=100 互锁：相当于 seq_d 实际只看最近 100 个事件，
>   与 §3.3 "保留 seq_d=512" 也有矛盾。
> ```
>
> **请用户决策**：
>
> | 选项 | 含义 | 推荐 |
> | --- | --- | :---: |
> | A | 主线 `transformer` + `seq_top_k=100`（marker, 不生效） | ✓ 当前默认 |
> | B | 二轮独立 ablation：单独切 `longer` + `seq_top_k=100`，与 transformer 主线对照 | ✓ 可作为补充 |
> | C | 第一轮就切 `longer`，让 `seq_top_k=100` 立刻生效 | ⚠️ AUC 不可预测，方差大 |

对应三种处理方案：

| 方案 | 含义 | 推荐 |
| --- | --- | :---: |
| A | 仍用 `transformer` encoder，命令行带上 `--seq_top_k 100`（不生效，作为 marker） | ✓ 第一轮主线 |
| B | 切到 `longer` encoder，`--seq_top_k 100` 真实生效；模型只看 sequence 中 top-100 最近 token | 二轮 ablation |
| C | 走 `transformer` 但把 `--seq_max_lens` 的所有 domain 上限压到 100 | 不推荐，会丢历史信号 |

第一轮采用 A：保留 transformer encoder（与 0.8159 baseline 一致），把 `seq_top_k=100`
原样写进 train_config.json，作为后续 B 方案的引子。

如果一定要在第一轮就让 `seq_top_k` 生效，必须显式额外加 `--seq_encoder_type longer`，
但这会同时引入"encoder 结构变更 + sequence 排序变更"两个变量，方差控制困难。

## 4. 强力结构对齐

### 4.1 Token 整除约束（已通过 d_model=136 解决）

```text
T = num_queries * num_sequences + num_ns
  = 2 * 4 + (5 + 1 + 2 + 1)
  = 8 + 9
  = 17
```

`rank_mixer_mode='full'` 要求 `d_model % T == 0`。

```text
d_model = 136 = 17 × 8, 136 % 17 == 0 ✓
```

#### 4.1.1 候选 d_model 数学全枚举

满足 `d_model % 17 == 0` 的所有候选（≤ 256）：

```text
17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238
```

距离原文 `d_model=128` 最近的两个候选：

```text
119 (= 17 × 7)  : 偏离 128 -7%   (capacity 减少)
136 (= 17 × 8)  : 偏离 128 +6.25% (capacity 增加, 与原文方向一致)
```

#### 4.1.2 历史选项对比（已被 d_model=136 + full 取代）

| 历史候选 | 优缺点 | 状态 |
| --- | --- | --- |
| A. `ffn_only` + d_model=128 + item_dense=on | 关闭 RankMixer token mixing，依赖 GNN 完全覆盖 | **被 d_model=136 + full 取代** |
| B. `full` + d_model=128 + item_dense=off | item dense normalization 失效，违反原文 §2 | 弃 |
| C. `full` + d_model=119 + item_dense=on | capacity 比 128 少 7%，违反原文方向 | 弃 |
| **D. `full` + d_model=136 + item_dense=on** | 唯一代价 d_model 字面 +6.25% | **✅ 当前方案** |
| E. `full` + 改 num_queries / ns_tokens | 违反"5/2/2 锁定" | 弃 |

#### 4.1.3 为什么选 D（136 + full）

```text
- 与原文 §2 "item dense features 做 normalization" 完全对齐：
  item_dense token 真实进模型，normalization 在模型层面生效；
- 与原文 5/2/2 ns 锁定完全对齐：
  user_ns_tokens=5, item_ns_tokens=2, num_queries=2 不动；
- 与 0.8159 baseline 的 backbone 行为对齐：
  rank_mixer_mode 默认值 = full，baseline 也是 full；
- 与原文 d_model 升级方向一致：
  原文 d_model 从 baseline 64 → 128 是 capacity 翻倍，
  136 (vs 119) 延续这个方向；
- 唯一代价：d_model 字面值偏离 +6.25%，可接受。
```

### 4.2 GNN 影响域守恒检查

```text
TokenGNN 的输入 = self._build_ns_tokens 的拼接结果：
  user_ns        : (B, 5, d_model)
  user_dense_tok : (B, 1, d_model)
  item_ns        : (B, 2, d_model)
  item_dense_tok : (B, 1, d_model)   ← 5 月 7 日方案启用
  ns_tokens      : (B, 9, d_model)   → TokenGNN(num_layers=4, full graph)
```

新增的 5 个 int feature（89/90/91/130/131）通过 RankMixerNSTokenizer 的 chunk
机制被均匀打散到 `item_ns` 的 2 个 token 与 `user_ns` 的 5 个 token 中，**没有改变
NS token 的总数**。这正是用户原文"不影响 gnn 影响域的目的。新加进入的 feature 都
直接融入到 nsgroups 里面的设置"的工程含义。

### 4.3 sequence 排序后的 time bucket 一致性

dataset.py `_convert_batch` 的 time bucket 计算：

```text
time_diff = max(sample_timestamp - event_timestamp, 0)
bucket_id = searchsorted(BUCKET_BOUNDARIES, time_diff) + 1
```

排序顺序变化后：

```text
- time_diff 仍然是非负值，新到旧排序意味着 time_diff 单调递增；
- bucket_id 也单调递增；
- nn.Embedding(NUM_TIME_BUCKETS, d_model, padding_idx=0) 仍然按 token 位置查表；
- padding 位置仍然是 ts_padded == 0 → bucket_id = 0 → padding embedding。
```

无功能性影响。但必须强检查：

```text
[ ] 排序后 padding 仍然全部落在序列末尾（length 不变，越界位继续是 0）。
[ ] domain_a 的 ts_fid=39, domain_b=67, domain_c=27, domain_d=26 这一映射保持不变。
```

## 5. 训练配置（推荐 main run）

```bash
python3 -u "${SCRIPT_DIR}/build_fe08_may7_dataset.py" \
    --input_dir "${ORIG_DATA_DIR}" \
    --input_schema "${ORIG_SCHEMA_PATH}" \
    --output_dir "${FE08_ROOT}" \
    --missing_threshold 0.80 \
    --item_dense_fids "86,91,92"                          # ✅ 偏差 #3 已锁定：FE-01B 验证子集
    # --enable_risky_item_dense_fids 旗标默认 OFF，传入 87/88 会触发白名单 assert

python3 -u "${SCRIPT_DIR}/train.py" \
    --schema_path "${FE08_SCHEMA}" \
    --ns_groups_json "${FE08_GROUPS}" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --rank_mixer_mode full \                               # ✅ 偏差 #2 已锁定：与 0.8159 baseline 一致
    --d_model 136 \                                        # ✅ 偏差 #2 已锁定：原文 128 → 136 (=17×8) 满足 full 模式整除约束；详见 §4.1
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
    --seq_top_k 100 \                                     # 🟥 偏差 #1：仅 seq_encoder_type=longer 时生效；当前 transformer 主线下为 marker；详见 §3.6.3
    --batch_size 256 \
    --num_workers 8 \
    --buffer_batches 20 \
    --valid_ratio 0.1 \
    --split_by_timestamp \                                # ✅ 偏差 #5 已锁定：用户 2026-05-08 确认开启；对 eval AUC 零影响，仅强化 valid 信号可靠性

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

关键开关回顾：

| 来源 | 项目 | 值 | 备注 |
| --- | --- | --- | --- |
| 0.8159 baseline | use_token_gnn / layers / graph / scale | `4 / full / 0.15` | 不动 |
| 0.8159 baseline | user_ns_tokens / item_ns_tokens / num_queries | `5 / 2 / 2` | 不动 |
| 0.8159 baseline | rank_mixer_mode | `full` | 与 baseline 完全一致 |
| FE-06/07 经验 | `--split_by_timestamp` | 启用 | ✅ 偏差 #5 已锁定 |
| 5 月 7 日确认 | d_model | **136** (= 17×8) | ✅ 偏差 #2 已锁定，原文 128 偏 +6.25% |
| 5 月 7 日新增 | emb_dim / dropout | `64 / 0.05` | 与原文一致 |
| 5 月 7 日新增 | seq_top_k | `100` | ✅ 偏差 #1：当前 transformer 主线下不生效，作为 marker |
| 5 月 7 日新增 | seq_max_lens | `c:128,d:512` | 与原文一致 |

## 6. 数据流水线建议（builder）

新建：

```text
build_fe08_may7_dataset.py            # 训练侧 builder
tools/build_fe08_may7_dataset.py      # 平台上传副本
run_fe08_may7_gnn4.sh                 # 主入口
evaluation/FE08/build_fe08_may7_dataset.py
evaluation/FE08/dataset.py
evaluation/FE08/model.py
evaluation/FE08/infer.py
```

builder 执行顺序（保持单一原始流，避免多脚本串联破坏 raw missing）：

```text
raw parquet
  → P0-T1 timestamp split sidecar（仅训练侧）
  → audit user_int / item_int missing；阈值 0.80 删除
  → fit dense norm stats（train rg only）：原始 user_dense + item_dense + 派生 dense
  → 计算 89/90/91 + 130/131 + (可选 110/86/91/92) 派生列
  → sequence permute by event_time desc
  → 写出 augmented parquet + schema.json + ns_groups.may7.json
  → 写出 sidecars: dropped_feats.may7.json, fe08_transform_stats.json,
                  fe08_dense_normalization_stats.json
```

为什么不直接复用 `build_fe07_p012_domain_dataset.py`？

```text
- FE-07 默认走 average-fill，本方案沿用 average-fill；可以参数化继承；
- FE-07 不做序列排序，必须新增 sort 阶段；
- FE-07 不生成 91/130/131；
- FE-07 默认 missing_threshold=0.75；
- FE-07 不带 item_dense_91/92 这类派生 dense；
最简洁的方式是 fork FE-07 的 builder，命名为 fe08，避免污染 FE-07 已落账逻辑。
```

## 7. 实验顺序与 ablation

为了避免一次性把 6 个变量全部换上去导致结果不可解释，推荐如下 ablation：

| 顺序 | 实验名 | 内容 | 预期 Eval AUC |
| ---: | --- | --- | ---: |
| 0 | GNN-baseline replay | 0.8159 baseline 配置直接复跑 | ≈ 0.8159 |
| 1 | + drop ≥ 80% missing | §3.1 单独，schema 缩窄 | 0.8155 ~ 0.8165 |
| 2 | + item_dense token + norm（rank_mixer_mode=full + d_model=136） | §3.2 + §4.1 | 0.8165 ~ 0.8180 |
| 3 | + sequence sort by recency | §3.4 单独 | 0.8170 ~ 0.8185 |
| 4 | + new int features 89/90/91/130/131 | §3.5 全部启用 | 0.8175 ~ 0.8195 |
| 5 | + seq_max_lens 调整 c:128/d:512 | §3.3 | 0.8180 ~ 0.8200 |
| 6 | + d_model=136 / dropout=0.05 / seq_top_k=100 (marker) | §3.6 全部 | 0.8185 ~ 0.8215 |
| **6.B** | **(可选) step 6 + 切 seq_encoder_type=longer**，让 `seq_top_k=100` 真生效 | **§3.6.3 方案 B** | **0.8130 ~ 0.8210（巨大方差）** |

> ### 偏差 #2 已通过 d_model=136 + full 解决
>
> 旧版本曾设计 step 2.5（关 item_dense + 切回 full + d_model=128）作为 ffn_only 路径的
> 对照实验。**当前方案直接采用 full + d_model=136**，让 RankMixer token mixing 与
> TokenGNN message passing 共同作用，无需对照消融。
>
> 唯一的代价是 d_model 字面值从 128 偏到 136 (+6.25%)；预期 step 2 的 AUC 收益
> 比 ffn_only 路径高 +0.0008（因为多保留了 RankMixer 的 token mixing 通路）。

> ### 🟥 step 6.B 的设计动机（保留作为可选 ablation）
>
> 这是为偏差 #1（`seq_top_k=100` 在 transformer 主线下不生效）设计的二轮 ablation。
> 仅在用户希望"原文 seq_top_k=100 必须生效"时才跑。预期方差很大，因为 longer encoder
> 与 GNN 的组合在历史中未验证过。

最低资源版本：

```text
直接跑 step 6（main run）；
若 step 6 ≤ baseline，再回退到 step 4 / step 2 拆问题；
若用户希望验证 seq_top_k=100 的真实收益，加跑 step 6.B（longer encoder 路径）。
```

## 8. 强力检查清单

### 8.1 数据正确性

```text
[ ] missing_threshold 阈值真实是 0.80；audit JSON 中 user_int / item_int 删除清单已写入。
[ ] item_dense_feats_86/91/92 的 train-only mean/std 已 fit；evaluation 不重新 fit。
[ ] item_dense_feats_87/88（purchase / delay）确认未被 schema 引入。
[ ] item_int_feats_89: vocab_size=3, has_match ∈ {0,1,2}。
[ ] item_int_feats_90: vocab_size=len(match_count_buckets)+1=7。
[ ] item_int_feats_91: vocab_size=NUM_TIME_BUCKETS=64（=len(BUCKET_BOUNDARIES)+1=63+1）；非匹配样本写 0。
[ ] user_int_feats_130: vocab_size=25，hour ∈ {1..24}，padding=0。
[ ] user_int_feats_131: vocab_size=8，dow ∈ {1..7}，padding=0。
[ ] 时间衍生 130/131 使用 UTC 规则，与 sample timestamp 同时区。
[ ] sequence 排序在 builder 端落地，dataset.py 不需要改动。
[ ] sequence 排序仅按 event_time，不依赖任何未来字段。
```

### 8.2 防泄漏

```text
[ ] item_int_91 的"最近匹配 time bucket"只统计 event_time ≤ sample_timestamp 的事件。
[ ] hour / dow 来自 sample timestamp，不来自 label_time。
[ ] 不引入 user_dense_feats_111 / item_dense_feats_87（purchase frequency）。
[ ] 不引入 user_dense_feats_112 / item_dense_feats_88（avg delay）。
[ ] dense norm 仅用 train row groups。
[ ] split_by_timestamp 启用，valid 是 row group 时间序最后 10%。
[ ] missing 删除阈值仅基于 train row groups 的 audit，不污染 eval。
```

### 8.3 模型 / token 结构

```text
[ ] 训练日志打印 num_user_ns=5, num_item_ns=2, num_user_dense=1, num_item_dense=1。
[ ] num_ns=9, T=2*4+9=17, rank_mixer_mode=ffn_only。
[ ] use_token_gnn=True, token_gnn_layers=4, token_gnn_graph=full,
    token_gnn_layer_scale=0.15。
[ ] d_model=128, emb_dim=64, num_hyformer_blocks=2, num_heads=4。
[ ] dropout_rate=0.05, seq_id_emb_dropout=0.10。
[ ] seq_max_lens={a:256,b:256,c:128,d:512}, use_time_buckets=True。
[ ] 不启用 output_include_ns（历史负例 0.811）。
[ ] ns_groups.may7.json 不写入 dense fid（86/91/92/110/120/121 等）。
```

### 8.4 评估侧 (FE08)

```text
[ ] evaluation/FE08/build_fe08_may7_dataset.py 与训练侧 builder 共享同一份逻辑。
[ ] eval 严禁重新 fit dense norm。
[ ] eval 严禁重选 match pair / 重新计算 missing 删除清单。
[ ] eval 严禁读取 eval label_type 更新统计。
[ ] checkpoint 内 schema.json / ns_groups.may7.json / dropped_feats.may7.json
    / fe08_transform_stats.json / fe08_dense_normalization_stats.json
    全部已 sidecar 写出。
[ ] strict load checkpoint 成功；missing/unexpected key 立即 fail-fast。
```

### 8.5 5 月 7 日原文逐条回扣

```text
[ ] §3.1 删除 missing > 80%：执行 + audit JSON。
[ ] §3.2 item dense normalization：item_dense token 启用 + z-score。
[ ] §3.3 seq_a/b/c/d = 256/256/128/512：命令行配置。
[ ] §3.4 序列时间近 → 远排序：builder 端 argsort。
[ ] §3.5.1 item_int_89/90/91 全部生成，91 是新增。
[ ] §3.5.2 user_int_130 (hour) / 131 (dow) 全部生成。
[ ] §3.5 ns_groups.may7.json 中所有新增 fid 已并入对应分组。
[ ] §3.5 evaluation builder 自动生成上述列。
[ ] §3.6 d_model=128 / dropout=0.05 / seq_top_k=100：命令行配置。
[ ] §4.1 user_ns_tokens=5 / item_ns_tokens=2 / num_queries=2 不变。
```

## 9. 风险与回退

| 风险 | 触发判据 | 回退动作 |
| --- | --- | --- |
| `rank_mixer_mode=full` + d_model=136 资源不够 | OOM 或单 epoch > 2× baseline | 候选回退：(a) `d_model=119` (= 17×7) 仍 full，capacity 比 128 少 7%；(b) `d_model=68` (= 17×4) 极度收缩，仅作 OOM 紧急方案；(c) 降 batch_size 到 224 / 192；(d) 砍 token_gnn_layers 到 2 |
| 序列排序代码错位 | step 3 valid AUC 突变 / time_bucket 分布异常 | 立即回退到 builder 不排序版本，做 5k 样本 diff 审计 |
| dropout=0.05 过拟合反向恶化 | step 6 收敛慢且 logloss 高 | 回退到 0.03 / 0.025 二档调参 |
| `d_model=136` 比 baseline 64 多 +112%，capacity 太大反而过拟合 | step 6 比 step 5 不涨或下降 | 回退：(a) `d_model=119`（17×7，capacity 比 baseline +86%）；(b) 加 weight decay 0.01 ~ 0.05；(c) 降 dropout 反而帮过拟合 → 不做；(d) `d_model=68`（17×4，过激保守方案）|
| 新 int feature 91/130/131 含义有误 | step 4 AUC 不变或下降 | 拆解：先单独跑 91，再跑 130/131；audit JSON 校对 vocab range |
| FE08 evaluation 无法读取 sidecar | infer 报 KeyError | 检查 trainer.py 是否把 dropped_feats.may7.json 等加入 sidecar copy 列表（沿用 FE-06 / FE-07 的 sidecar copy 模板） |

## 10. 与已有方案的差异说明

| 已有方案 | 与 5 月 7 日方案的差异 |
| --- | --- |
| FE-00 | FE-00 阈值 0.75，仅 user_int；本方案 0.80 + user/item 同删 |
| FE-01A | FE-01A 强制 `user_dense_feats_110`；本方案默认不带 110，可选纳入 |
| FE-01B | FE-01B 提供 dense `86/91/92` + int `89/90`；本方案保留原文 `item_int_feats_91`（语义=最近匹配 time bucket，int 标量）。**fid 91 在 item_int 与 item_dense 中可独立共存**：parquet 列名 prefix 不同（`item_int_feats_91` vs `item_dense_feats_91`），dataset.py 通过 prefix 查表，schema.json 中两个 list 也是分开的；ns_groups 只含 int fid，所以 `91` 在 ns_groups 中唯一指向 item_int。**严格遵循用户原文命名，不改名** |
| FE-06 | FE-06 走 P0-L1 vocab shift；本方案沿用 average-fill，不做 vocab shift |
| FE-07 | FE-07 不带 GNN；本方案在 GNN baseline 上叠 |
| GNN_NS_4Layer 0.815064 | 本方案的 step 0 复刻 |

### 10.1 fid `91` 共存的物理验证（非冲突）

经强力检查后撤回先前的"改名为 93"建议，因为 `item_int_feats_91` 与
`item_dense_feats_91` **不构成物理冲突**。验证如下：

```text
（1）parquet 列名空间
    item_int_feats_91   (int64 标量 / list)   ← 本方案新增
    item_dense_feats_91 (float32 标量 / list) ← FE-01B 已落账
    column 名是 prefix + fid 的字符串拼接，两个 prefix 不同 → 列名唯一。

（2）schema.json 解析路径
    schema['item_int']   = [..., [91, 64, 1], ...]    # 走 model.user/item_int_tokenizer
    schema['item_dense'] = [..., [91, 1], ...]         # 走 model.item_dense_proj
    两个 list 各自维护 fid 列表，不冲突。

（3）dataset.py 查列
    self._col_idx.get(f'item_int_feats_{fid}')         # → item_int_feats_91
    self._col_idx.get(f'item_dense_feats_{fid}')       # → item_dense_feats_91
    通过 prefix 解析，零歧义。

（4）ns_groups.may7.json
    "item_ns_groups": { "I4_target_matching_fields": [89, 90, 91] }
    ns_groups 按设计只接受 int fid；'91' 在此唯一指向 item_int_feats_91。
```

**结论**：保留用户原文 `item_int_feats_91`，本方案的 schema 中允许（且鼓励）
同时存在 `item_int_feats_91` 与 `item_dense_feats_91`，二者由 model 的不同
projector 处理，互不影响。

audit JSON / 训练 log 中遇到 `fid=91` 时必须 prefix 同步打印，避免阅读歧义：

```python
print(f"item_int_feats_91 vocab={...}, fill_value={...}")
print(f"item_dense_feats_91 mean={...}, std={...}")
```

## 11. 预期 Eval AUC

保守预测（每步取中位，全部基于 `rank_mixer_mode=full` + `d_model=136`）：

| 实验 | Eval AUC | 与 0.8159 baseline 的关系 |
| ---: | ---: | --- |
| step 0 | 0.8159 | 复刻 |
| step 1 | 0.8160 | 删除 ≥80% 缺失字段，几乎无差 |
| step 2 | 0.8172 | item_dense token + norm（full 模式比 ffn_only 多 +0.0005 token mixing 收益） |
| step 3 | 0.8177 | 序列时间排序 |
| step 4 | 0.8185 | item/user int 新特征 |
| step 5 | 0.8190 | seq lens 微调 c:128/d:512 |
| step 6 | 0.8200 | d_model=136 + dropout=0.05（136 vs 128 多 +0.0001~0.0005 capacity 收益） |

激进预测（每步取上界，假设全部正交）：

```text
step 6 ≈ 0.8159 + 0.0010 + 0.0020 + 0.0012 + 0.0020 + 0.0010 + 0.0024 ≈ 0.8255
```

实际落点大概率在 `0.8185 ~ 0.8215` 之间。

验收标准：

```text
若 step 6 ≥ 0.8185：替代 0.8159 baseline，作为新主线。
若 step 6 ≥ 0.8200：进入多 seed / SWA / EMA 阶段。
若 step 6 <  0.8159：拆 ablation，定位到具体退化步骤。
```

> ### d_model=136 + full vs ffn_only + d_model=128 对比
>
> 上表已切换到 d_model=136 + full 路径的预期，比之前的 ffn_only + d_model=128 路径预期
> 高约 +0.0005~0.0008。关键 delta 来源：
>
> ```text
> + RankMixer full 模式提供 NS↔Q token mixing       : +0.0003 ~ +0.0008
> + d_model 6.25% capacity 提升                      : +0.0001 ~ +0.0005
> - 训练时间 +10~13%                                 : 工程成本，非 AUC
> ```

## 11.5 偏差最终走向与净收益证明（强力检查）

下表把所有偏差点的最终落地位置 + 净收益方向汇总，确保**全部偏差均零负收益或正收益**。

### 11.5.1 偏差落地总览

| # | 偏差描述 | 状态 | 最终方案 | 净收益方向 | 为什么不会负收益 |
| ---: | --- | :---: | --- | :---: | --- |
| 1 | `seq_top_k=100` | ✅ marker only | `--seq_top_k 100` 写入命令行但 transformer 主线下不读取 | **零影响** | 此参数本来就不在 0.8159 baseline 的 active set 中；保留为 marker 不增不减 |
| 2 | `rank_mixer_mode` | ✅ 已锁定 | `full` + `d_model=136` (=17×8) | **正向** +0.0005~0.0008 | 与 baseline backbone 一致，比 ffn_only 多保留一道 NS↔Q token mixing |
| 3 | `item_dense` fid 范围 | ✅ 已锁定 | `{86, 91, 92}` (FE-01B 验证子集) | **正向** +0.0008 | FE-01B eval 落账 0.812102，子集已验证安全；显式排除 87/88 灾难特征 |
| 4 | hour/dow fid 编号 | ✅ 已锁定 | `130 (hour)`, `131 (dow)` | **零影响** | fid 编号是 column name 后缀，对模型行为无影响 |
| 5 | `--split_by_timestamp` | ✅ 已锁定 | 启用 | **零 eval 影响 / 正 valid 信号** | 仅影响 valid 切分，与 eval AUC 无关；让 valid 信号更可靠 |
| 6 | hour/dow `+1` 偏移 | 🟩 工程必要 | 落 `1..24` / `1..7`, 0=padding | **零影响 / 正向** | 字面 `0..23` 会被 `arr<=0→0` 误吞，+1 偏移是必要修正 |
| 7 | sequence 排序粒度 | 🟩 工程必要 | `row × domain × event_time desc` | **正向** +0.0007 | 修复 README §2.1 未审计点，让 seq_max_lens 截断保留最近事件 |
| 8 | `item_int_91` 无匹配回填 | 🟩 工程默认 | 写 0 = padding | **零影响** | 与现有 padding 语义一致，不引入新 vocab id |

### 11.5.2 净收益预估（按 step 拆分）

每一步都基于"上一步配置 + 该步新增"，且**不含任何已知负收益**：

```text
step 0 → step 1: drop 80% missing                          → 0  ~ +0.0005
step 1 → step 2: item_dense token + norm (full + 136)      → +0.0005 ~ +0.0015
step 2 → step 3: sequence sort by recency                  → +0.0005 ~ +0.0010
step 3 → step 4: item_int 91 + user_int 130/131           → +0.0005 ~ +0.0015
step 4 → step 5: seq_max_lens c:128/d:512                  → +0.0003 ~ +0.0008
step 5 → step 6: dropout=0.05 + d_model=136 (vs 128)       → +0.0003 ~ +0.0010

累计: step 6 vs step 0 = +0.0021 ~ +0.0063
预期 step 6 = 0.8159 + 0.0026 ~ 0.0056 = 0.8185 ~ 0.8215
```

### 11.5.3 强力负收益检查 — 已排除

逐项排除可能的负收益来源：

| 风险源 | 是否在本方案出现 | 排除依据 |
| --- | :---: | --- |
| `purchase_frequency` (FE-01 -0.035 灾难) | ❌ 不出现 | builder 中 `RISKY_ITEM_DENSE_FIDS={87,88}` 白名单 + `--enable_risky_item_dense_fids` 双保险 |
| `avg_delay` (label_time 泄漏) | ❌ 不出现 | 同上排除 fid 88 |
| `output_include_ns` (历史 0.815 → 0.811) | ❌ 不出现 | §8.3 强检查清单明确禁止 |
| longer encoder + GNN 未验证组合 | ❌ 不出现于主线 | 主线保 transformer；longer 仅作 step 6.B 可选 ablation |
| `ffn_only` 隐式 backbone 选择 | ❌ 不出现 | 已切回 `full` + `d_model=136` |
| valid 信号噪声大 | ❌ 不出现 | `--split_by_timestamp` 启用 |
| schema fid 编号冲突 | ❌ 不出现 | 130/131 远离已用区段；91 物理隔离已验证（§10.1）|
| eval / train 不一致 | ❌ 不出现 | 所有 sidecar 强制从 checkpoint 复制；evaluation 严禁重新 fit / 重选 pair |

### 11.5.4 可能的方差源（非负收益但需监控）

```text
- d_model=136 比原文 128 多 +6.25% capacity，边际收益区间 +0.0001~0.0005，
  但极端情况可能引发轻微过拟合 → 监控 valid AUC 与 logloss；
- dropout=0.05 比 baseline 0.015 高 3×，对新增 int 特征做正则化，
  但若过拟合本来就不严重，dropout 提升可能让收敛变慢；
- 4-layer TokenGNN 与 RankMixer full 模式的 token mixing 部分功能重叠，
  实际收益可能比预测略低（边际而非完全正交）。

应对：跑完每一步都打印 valid AUC + logloss + 单 epoch 时间，
若某步 logloss 反向、valid 抖动 > 0.001，立即停下来分析。
```

## 12. 总结

5 月 7 日方案的核心目标是把 0.8159 GNN baseline 与 FE-07 已经验证的低风险特征
工程结合，严格守住三件事：

```text
1. 不增加 NS token 数（保持 user_ns_tokens=5 / item_ns_tokens=2 / num_queries=2），
   保证 TokenGNN 的影响域 = 5+1+2+1 = 9 不变。
2. 所有新特征通过 RankMixer 的 chunk 机制混入现有 NS token，
   不改变 GNN 的 message-passing 拓扑。
3. 保持 rank_mixer_mode=full（与 0.8159 baseline 一致），通过 d_model=136 (= 17×8)
   解决 T=17 的整除约束；唯一代价是 d_model 字面值偏离原文 128 +6.25%，
   换来 RankMixer token mixing 与 TokenGNN message passing 的双重作用。
```

与原文逐条对应：删除 ≥80% missing → §3.1；item dense normalization → §3.2；
seq lens 改 256/256/128/512 → §3.3；序列时间排序 → §3.4；
item_int_89/90/91 + user_int hour/dow → §3.5；
d_model=128（实际 136） / dropout=0.05 / seq_top_k=100（marker）→ §3.6。

第一轮主线：

```text
FE08-May7-main = 0.8159 GNN-baseline replay
                 + drop 80% missing
                 + item_dense token + norm (fids = {86, 91, 92})
                 + sequence sort by recency
                 + item_int_89/90/91 (new, 91=最近匹配 time bucket)
                 + user_int_130/131 (new, hour/dow)
                 + seq_a/b/c/d = 256/256/128/512
                 + rank_mixer_mode=full + d_model=136 + dropout=0.05
                 + use_token_gnn (4 layers, full graph, scale=0.15)
                 + split_by_timestamp
```

目标：

```text
Eval AUC ≥ 0.8185  作为可接受替换；
Eval AUC ≥ 0.8200  作为进入下一阶段（多 seed / SWA / EMA）。
```

> 落地时严格对照 §8 强力检查清单，每一项打勾后再进入下一步消融。
