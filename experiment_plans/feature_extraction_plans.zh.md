# 数据特征提取逐步方案（Feature Extraction Roadmap）

本文基于 [README.feature_engineering.zh.md](../README.feature_engineering.zh.md)（demo 数据已观察的 lift 信号）与 [README.research_directions.zh.md](../README.research_directions.zh.md)（工程改造点）整理。目标：在**不改 HyFormer backbone** 的前提下，给 NS tokenizer 与序列编码层注入新的离线/在线统计特征，逐步提点。

---

## 0. 前置说明

### 0.1 当前 baseline 的特征边界

- NS 特征只有 user_int (46)、item_int (14)、user_dense (10)、item_dense (0)。**没有任何 cross / interaction / aggregation 特征**。
- 序列特征仅做 ID embedding + time bucket，**没有抽取序列内部的统计量、也没有把目标 item 与历史序列的关系喂给模型**。
- Final head 只用 Q tokens（[model.py:1626-1632](../model.py:1626)）。即使加了再多 NS token，也要先确认 Q token 路径能感知它们。

### 0.2 优先级原则

1. **demo 已观察到强 lift 的方向优先**：item-属性 × 历史序列匹配（最高 35.0 % vs 11.9 % ≈ 3× lift），其次域级活跃度（最高 15.4 % vs 10.4 % ≈ +50 % 相对 lift）。
2. **离线可预计算的优先于在线**：减少训练 IO/CPU 压力。
3. **改动不破坏 `d_model % T == 0`**（[model.py:1320](../model.py:1320)）：每次 NS token 数变化都要重新核对 T。
4. **从最少代码侵入开始**：先做 dataset.py 内部聚合 → 再做模型端 token 注入 → 最后才考虑改 query generator。

### 0.3 demo 已验证的强信号一览（按相对 lift 排序）

| 信号 | 高分位正例率 | 低分位正例率 | 相对 lift |
|---|---:|---:|---:|
| `item_int_feats_13` × `domain_c_seq_31` 匹配 | 35.0 % | 11.9 % | **+194 %** |
| `item_int_feats_9` × `domain_d_seq_19` 匹配 | 26.8 % | 11.8 % | **+127 %** |
| `item_int_feats_10` × `domain_d_seq_24` 匹配 | 23.2 % | 11.4 % | **+104 %** |
| `domain_a count_7d` 高分位 | 15.4 % | 10.4 % | +48 % |
| `domain_c len` 高分位 | 15.1 % | 11.6 % | +30 % |
| `domain_d len` 高分位 | 14.0 % | 10.0 % | +40 % |

→ Tier-1 信号（匹配特征）的 lift 比 Tier-2（活跃度）大一个数量级，**应优先提取**。

---

## 1. 方案 A：Target-Item 属性 × 历史序列匹配特征族 🟢 最高优先级

### 1.1 设计动机

当前模型只把 `item_int_feats_*` 单独 embedding，把 `domain_*_seq_*` 单独 embedding，**完全没有让模型显式知道"当前 target item 的某属性是否、何时、几次出现在用户某个域的历史里"**。这是 RecSys 中最经典也最强的 cross 信号。

demo 数据三对组合的 lift 均在 +100 % 以上，说明这一类特征在完整数据集上很可能撑起单一 ablation 中 **最大那一档**的提点。

### 1.2 候选 (item_field, seq_field) 对

**Item 端候选字段**（共 13 个 scalar + 1 个 list）：

```text
低基数（适合做 has_match / count_recent_*）:
  item_int_feats_5  (82 类，三级品类)
  item_int_feats_9  (24 类，一级行业)        ✓ demo 验证
  item_int_feats_10 (110 类，创意标签)       ✓ demo 验证
  item_int_feats_13 (8 类，创意类型)         ✓ demo 验证
  item_int_feats_81 (3 档，投放渠道)
  item_int_feats_83 (22 类，二级行业, 缺失 83 %)
  item_int_feats_84 (66 类，行业子类, 缺失 83 %)

中基数（has_match 仍有信号，count 较稀疏）:
  item_int_feats_6  (216 类，行业细类)
  item_int_feats_7  (349 类，count 分桶)
  item_int_feats_12 (352 类，count 分桶)

高基数（适合做精确 ID 命中）:
  item_int_feats_16 (662 类，物料子 ID)
  item_int_feats_11 (list，多标签)

注意：item_id 本身是 837 unique，可作为最严格的命中 (item_id, domain_*_seq_*)。
```

**Seq 端候选字段**：每域排除 timestamp 后剩约 8~13 个，全量配对会产生 13×40 ≈ 520 对，**不可全量上线**。需先离线扫描所有对的 lift，按 lift 排序保留 Top-K（建议 K=15~25）。

### 1.3 一次性提取的特征列表（约 60~90 个）

对每个保留下来的 (item_field, seq_field) 对，生成下面 6~8 个标量特征：

```text
A1. has_match                       int8  ∈{0,1}      target 值是否曾出现在序列里
A2. match_count                     int32              总命中次数
A3. match_count_1d                  int32              过去 1 天命中次数
A4. match_count_7d                  int32              过去 7 天
A5. match_count_30d                 int32              过去 30 天
A6. match_count_90d                 int32              过去 90 天
A7. min_match_delta_seconds         float32 (log1p)    最近一次命中距 timestamp 的秒数
A8. last_match_position             int32              最近一次命中在序列中的索引（new-to-old）
```

可选附加（命中率高的对再做）：

```text
A9.  match_ratio = match_count / max(seq_len, 1)
A10. mean_match_delta_seconds (log1p)   命中事件的平均时间差
A11. match_count_in_recent_top64        最近 64 个 token 内的命中数（与 longer encoder 协同）
```

按 K=15 对、每对 6 特征算 → **约 90 个新特征**，全部为 dense scalar。

### 1.4 实现路径

#### 阶段 1 — 离线 lift 扫描脚本（一次性）

新建 `tools/scan_match_lift.py`：

- 输入：完整训练集 parquet
- 输出：`tools/match_lift_top.json`，含每对 (item_field, seq_field) 的 `n_pos / n_neg / has_match_lift`
- 只挑 lift ≥ 1.5× 且支持度 ≥ 0.5 % 样本的对

#### 阶段 2 — dataset.py 内嵌提取

[dataset.py:_convert_batch](../dataset.py:505) 末尾增加：

```python
# 新增：target-match aggregation
match_feats = np.zeros((B, len(MATCH_PAIRS) * 6), dtype=np.float32)
for k, (item_fid, dom, seq_fid) in enumerate(MATCH_PAIRS):
    target_vals = item_int[:, item_int_offset[item_fid]:item_int_offset[item_fid]+item_int_dim[item_fid]]
    seq_vals = result[dom][:, seq_slot[dom][seq_fid], :]   # (B, max_len)
    seq_ts   = ts_padded_per_domain[dom]                   # (B, max_len)
    dt = np.maximum(timestamps.reshape(-1,1) - seq_ts, 0)
    # 命中位置 mask
    match_mask = (seq_vals == target_vals[:, :1])  # 假设 scalar item_field
    match_count   = match_mask.sum(axis=1)
    match_1d   = (match_mask & (dt <= 86400)).sum(axis=1)
    match_7d   = (match_mask & (dt <= 604800)).sum(axis=1)
    match_30d  = (match_mask & (dt <= 2592000)).sum(axis=1)
    has_match = (match_count > 0).astype(np.float32)
    # 最小 dt
    masked_dt = np.where(match_mask, dt, 1<<30).astype(np.int64)
    min_dt = np.log1p(masked_dt.min(axis=1).astype(np.float32))
    match_feats[:, k*6:(k+1)*6] = np.stack([
        has_match, match_count, match_1d, match_7d, match_30d, min_dt
    ], axis=1)

result['match_feats'] = torch.from_numpy(match_feats.copy())
```

#### 阶段 3 — 模型端注入

最低侵入方案：把 `match_feats` 作为额外 dense token 加进 NS：

```python
# model.py PCVRHyFormer.__init__
self.has_match_feats = match_feats_dim > 0
if self.has_match_feats:
    self.match_proj = nn.Sequential(
        nn.Linear(match_feats_dim, d_model),
        nn.LayerNorm(d_model),
    )
self.num_ns += (1 if self.has_match_feats else 0)
# 重新计算 T，确认 d_model % T == 0
```

forward 中把 `match_proj(match_feats)` 拼到 `ns_tokens`。

#### 阶段 4（可选）— 进 query generator 条件输入

更高收益但更深的改造：把 `match_feats` 投影后作为 `MultiSeqQueryGenerator` 的额外 condition，使 query 在生成时就能感知"目标 item 与各域的关系"。这一步要改 [model.py](../model.py:1382) 的 `query_generator` 接口。

### 1.5 风险

- 🔴 **特征膨胀**：90 个新 dense 特征 + 1 个 NS token，T 会变化，必须重算 `d_model % T`，否则 [model.py:1320](../model.py:1320) 报错。
- 🟠 **list 类 item_field（如 `item_int_feats_11`）匹配实现复杂**：要做 set-membership，CPU 上 numpy `np.isin` 性能差；建议先排除 list-item，迭代再补。
- 🟠 **离线扫描成本**：完整训练集扫 520 对需要 1~2 小时单机；可只扫 1 % 子集近似排序。
- 🟢 **泄漏低风险**：`event_ts ≤ timestamp` 是序列字段已具备的属性，只要严格过滤 `dt ≥ 0` 就不会泄漏。
- 🟠 **未做归一化**：match_count 范围可能 0~100+，需做 `log1p` 或除以 seq_len，否则 Linear 层易被极值主导。

### 1.6 预期收益

- 🟢 **+0.30 % ~ +0.60 % AUC**（最大那一档）。demo 三对已显示 +100 % 以上 lift，扩到 15 对后边际收益会下降但仍是单方向最强。
- 训练耗时增加 5~10 %（CPU 端 numpy 运算占主导）。
- 显存几乎不变（只多 1 个 NS token）。

---

## 2. 方案 B：Domain 活跃度与时间聚合特征族 🟢 高优先级

### 2.1 设计动机

[README.feature_engineering.zh.md](../README.feature_engineering.zh.md) §3.2~3.3 已经显示 4 个 domain 的活跃度差异极大（domain_d 30 % 7d 内、domain_c 仅 1.5 %）。但 baseline **从未把"用户在每个 domain 的活跃度"作为显式 NS 特征**。模型必须从原始序列里反推这一信号，效率低。

把 4 个域的长度、计数、最近时间等聚合统计直接作为 dense NS token，是最便宜的提点路径之一。

### 2.2 一次性提取的特征列表（约 50~60 个）

对每个 domain ∈ {a, b, c, d} 提取以下 13 个 scalar：

```text
B1.  seq_len_raw                    int32   未截断的原始长度
B2.  seq_len_used                   int32   max_len 截断后的有效长度
B3.  count_1h                       int32   1 小时内事件数
B4.  count_1d                       int32   1 天
B5.  count_7d                       int32   7 天
B6.  count_30d                      int32   30 天
B7.  count_90d                      int32   90 天
B8.  count_365d                     int32   1 年
B9.  last_event_delta_seconds       float32 (log1p)   最近一次事件距 timestamp
B10. first_event_delta_seconds      float32 (log1p)   最早一次事件距 timestamp
B11. mean_event_delta_seconds       float32 (log1p)   平均时间差
B12. max_inter_event_delta          float32 (log1p)   最大相邻事件间隔（疲劳度）
B13. median_event_delta_seconds     float32 (log1p)   中位时间差
```

跨域比例（再增加 8 个 cross-domain 特征）：

```text
B14~B17. count_7d_share_a/b/c/d  = count_7d_x / sum_x(count_7d_x)
B18~B21. seq_len_share_a/b/c/d   = seq_len_x / sum_x(seq_len_x)
```

合计 4×13 + 8 = **60 个新 dense 特征**。

### 2.3 实现路径

完全在 [dataset.py:_convert_batch](../dataset.py:505) 序列处理循环里完成，**不需要新文件**。在已有的 `time_diff` 计算后增加：

```python
domain_aggs = []
for domain in self.seq_domains:
    ts_padded = ...      # 复用已有
    time_diff = ...      # (B, max_len)
    valid = (ts_padded > 0)
    seq_len = valid.sum(axis=1)
    cnt_1h  = ((time_diff <= 3600)    & valid).sum(axis=1)
    cnt_1d  = ((time_diff <= 86400)   & valid).sum(axis=1)
    cnt_7d  = ((time_diff <= 604800)  & valid).sum(axis=1)
    cnt_30d = ((time_diff <= 2592000) & valid).sum(axis=1)
    cnt_90d = ((time_diff <= 7776000) & valid).sum(axis=1)
    cnt_365 = ((time_diff <= 31536000)& valid).sum(axis=1)
    last_delta  = np.log1p(np.where(valid, time_diff, 1<<30).min(axis=1))
    first_delta = np.log1p(np.where(valid, time_diff, 0).max(axis=1))
    # 平均、中位、相邻间隔等同理
    domain_aggs.append(np.stack([...], axis=1))   # (B, 13)

agg = np.concatenate(domain_aggs, axis=1)        # (B, 52)
# 再补 8 个 cross-domain share
result['domain_aggs'] = torch.from_numpy(agg.copy())
```

模型端：与方案 A 相同的 `dense_proj → NS token` 注入路径。

### 2.4 风险

- 🟢 **几乎零泄漏**：所有统计仅依赖 `event_ts < timestamp`。
- 🟢 **零新词表**：无 vocab 膨胀。
- 🟡 **尺度差异**：`count_*` 量级 0~10000，`*_delta_seconds` 在 log1p 后量级 0~25，需在 dense_proj 前加 BatchNorm/LayerNorm 或预先 z-score。
- 🟠 **与方案 A 部分冗余**：方案 A 的 `match_count_*d` 已经是"匹配次数随时间衰减"信号；与方案 B 的总活跃度叠加时，模型可能把匹配信号"分解"成 (匹配率 × 活跃度) 两路。**不冲突，但要在做 ablation 时分别测**。

### 2.5 预期收益

- 🟢 **+0.10 % ~ +0.25 % AUC**。demo 已显示 `count_7d` 高分位 +48 % lift，但绝对差只有 5 %，提点空间不如方案 A。
- 训练耗时增加 1~3 %。
- 显存增加 < 1 %。

---

## 3. 方案 C：稀疏 / 高基数字段元统计特征族 🟡 中等优先级

### 3.1 设计动机

[README.feature_engineering.zh.md](../README.feature_engineering.zh.md) §6.4 提到 `domain_a_seq_38`（5.5 % 覆盖率，17.8k unique）、`domain_b_seq_88`（40.6 % 覆盖率，12.3k unique）这一批"稀疏但高基数"字段。当前 baseline 的 embedding 对它们既贵（17.8k × 64 ≈ 1.1 M 参数）又稀（绝大多数样本里全是 0），训练梯度信号严重不均。

**显式建模"该字段是否出现 / 出现了几次 / 出现了几个不同 ID"** 比用 embedding 强行拟合更稳。

### 3.2 候选字段

```text
极稀疏（覆盖 < 15 %）:
  domain_a_seq_38 (5.5 %, 17.8k unique)
  domain_d_seq_22 (6.7 %, 17.9k unique)
  domain_b_seq_74 (10.6 %, 14.3k unique)

中稀疏（覆盖 15~50 %）:
  domain_b_seq_88 (40.6 %, 12.3k unique)

高覆盖高基数（看作 meta 信号亦有用）:
  domain_c_seq_47 (100 %, 287k unique)
  domain_b_seq_69 (99.8 %, 192k unique)
  domain_c_seq_29 (100 %, 172k unique)
  domain_d_seq_23 (95.6 %, 123k unique)
  domain_c_seq_36 (70.7 %,  55k unique)
  domain_c_seq_34 (100 %,  16.9k unique)
```

10 个字段为候选。

### 3.3 一次性提取的特征列表（约 40~50 个）

对每个候选字段提取以下 5 个标量：

```text
C1. is_present                  int8     有 ≥1 个非零值
C2. present_count               int32    非零事件数
C3. present_unique_count        int32    非零去重数（diversity）
C4. present_count_7d            int32    7 天内非零数
C5. present_count_30d           int32    30 天内非零数
```

10 字段 × 5 特征 = **50 个新 dense 特征**。

可选附加（前 3 个极稀疏字段再加 2 个）：

```text
C6. last_present_delta_seconds  float32 (log1p)
C7. present_value_top1_count    int32     最高频值的命中次数
```

### 3.4 实现路径

复用方案 B 同一 `_convert_batch` 改动点。`present_unique_count` 用 `np.unique` 较慢，建议改用 `np.bincount` 后非零计数（`max_vocab` 已知）。如果 max_vocab > 1e5，用 `set()`：

```python
unique_count = np.array([len(set(row[row > 0])) for row in seq_vals], dtype=np.int32)
```

CPU 端循环可接受（B=256，每行长度 ≤ 1100）。

### 3.5 风险

- 🟢 **几乎零泄漏**：与方案 B 同。
- 🟡 **`np.unique` per-row 性能差**：B=1024 时单 batch +50 ms。可接受，但 batch 翻倍后要监控。
- 🟢 **可逐步上线**：与已有 embedding **完全独立**，可单独 ablation 验证 lift。
- 🟠 **C7 `value_top1_count` 需要全局统计 top-1 ID**，要先离线扫描；否则就是局部 top-1，意义减弱。

### 3.6 预期收益

- 🟡 **+0.05 % ~ +0.15 % AUC**。demo 没直接给这一批的 lift，但 `domain_a_seq_38` 这种 5.5 % 覆盖的字段，"出现 / 不出现"本身极可能与活跃高价值用户相关。
- 与方案 D（item-popularity）、方案 G（dense per-field）正交，可叠加。

---

## 4. 方案 D：Item-User 双向人气特征族 🟡 中等优先级

### 4.1 设计动机

CTR/CVR 模型中"item 自身的历史曝光/点击数"和"user 历史触达过多少同类 item"是经典强信号。当前 baseline 完全没有这一类特征。

注意：**这一族大部分要离线全局统计**，与方案 A/B/C 在线提取不同。

### 4.2 一次性提取的特征列表（约 30~40 个）

#### Item 端全局统计（离线预计算，按 item_id 索引）

```text
D1.  item_global_exposure_count            log1p(int32)   全样本中曝光次数
D2.  item_global_pos_count                 log1p(int32)   正例次数
D3.  item_global_pos_rate                  float32        D2/max(D1,1)
D4.  item_first_seen_delta_days            float32 (log1p)
D5.  item_last_seen_delta_days             float32 (log1p)
```

#### Item-attribute 端（按 item_int_feats_5/6/9/10/13/16 各做一份）

```text
D6.  attr_global_exposure_count            log1p
D7.  attr_global_pos_rate                  float32
```

7 个 attribute × 2 特征 = 14 个。

#### User 端全局统计（离线预计算，按 user_id 索引）

```text
D8.  user_total_exposure                   log1p
D9.  user_total_pos                        log1p
D10. user_pos_rate                         float32
D11. user_unique_items_seen_30d            log1p
D12. user_unique_industries_seen_30d       log1p
```

#### User-Item 交叉

```text
D13. user_target_item_exposure_30d         int8  (0 / 1+)   该 user 对该 item 30 天是否见过
D14. user_target_item_attr9_match_30d      int32           user 30 天内是否点过同一 attr9 行业
D15. user_target_item_attr13_match_30d     int32
```

合计 5 + 14 + 5 + 3 ≈ **30 个 dense 特征**。

### 4.3 实现路径

#### 阶段 1 — 离线全局统计

新建 `tools/build_global_stats.py`：

- 扫描完整训练集，按 `user_id` / `item_id` / `(item, attr)` 三种 key 聚合 exposure / pos
- 写到 `tools/global_stats/{users.parquet, items.parquet, item_attr.parquet}`，每张表 ≤ 50 M 行，可全量装内存

#### 阶段 2 — dataset 端 lookup

[dataset.py:_convert_batch](../dataset.py:505) 起手处加：

```python
# 在 __init__ 里：self._user_stats = pd.read_parquet(...).set_index('user_id')
user_ids = batch.column(self._col_idx['user_id']).to_pylist()
item_ids = batch.column(self._col_idx['item_id']).to_pylist()
user_lookup = self._user_stats.reindex(user_ids).fillna(0).values    # (B, n_user_feats)
item_lookup = self._item_stats.reindex(item_ids).fillna(0).values    # (B, n_item_feats)
result['user_global_feats'] = torch.from_numpy(user_lookup.astype(np.float32))
result['item_global_feats'] = torch.from_numpy(item_lookup.astype(np.float32))
```

模型端：与方案 A 相同的 `dense_proj → NS token`。

### 4.4 风险

- 🔴 **泄漏风险（最高）**：D1~D14 的全局统计**必须**只用训练集（不含验证 / 测试），否则会把 label 信息泄回训练。务必在 `tools/build_global_stats.py` 中显式过滤验证集 row group。
- 🔴 **未来 / 未见 user 的 cold-start**：测试集里出现训练集没见过的 user_id → lookup 全部 0。需要补"是否新用户"flag (D8 = 0)。
- 🟠 **存储 + IO**：完整训练集 200 M 用户，user_stats parquet 约 5 GB。在 dataset 启动时全量加载到内存可行（约 10 GB RAM），但小机器有压力；可改成 `pyarrow.dataset` 按需 lookup（慢 5~10×）。
- 🟠 **统计偏差**：对训练集做 leave-one-out 统计才严格无泄漏，工程量大；常见 workaround 是按 timestamp 切，"只用样本 timestamp 之前发生的事件" 累加，但需在离线脚本里实现 time-aware aggregation。

### 4.5 预期收益

- 🟡 **+0.10 % ~ +0.25 % AUC**。但前提是泄漏控制到位；泄漏失误反而会让线下 AUC 虚高 0.5 % +、线上掉点。
- 离线统计成本：单机 1~3 小时；之后训练几乎零开销（hash lookup）。
- 与方案 A 互补：方案 A 是 user 端的"我历史命中过没"，方案 D 是 item 端的"它/它的属性整体表现如何"。

---

## 5. 方案 E：序列多样性 / 熵特征族 🔵 探索性

### 5.1 设计动机

用户行为多样性（看过几个不同行业、几个不同创意类型）已被多个 RecSys 工作证实与转化倾向相关。当前 baseline 完全没有这类二阶统计。

### 5.2 一次性提取的特征列表（约 25~30 个）

对 4 个 domain 各选 2~3 个低基数语义字段（从 [README.feature_engineering.zh.md](../README.feature_engineering.zh.md) §5 的边类型候选里挑）：

```text
domain_a: seq_40, seq_41, seq_46     (低基数行为类型)
domain_b: seq_68, seq_75
domain_c: seq_28, seq_32, seq_33
domain_d: seq_17, seq_24, seq_25
```

每字段提取：

```text
E1. unique_count            int32        去重数（多样性近似）
E2. shannon_entropy         float32      H = -Σ p_i log p_i
E3. top1_share              float32      最高频值占比（集中度）
E4. unique_count_7d         int32        近 7 天去重数
```

11 字段 × 4 特征 = **44 个特征**（可裁掉 entropy 留 33 个）。

### 5.3 实现路径

E1 用 `np.unique`，E2 需循环计算（CPU 慢）。建议：

- 对低基数字段（vocab ≤ 200），用 `np.bincount(row, minlength=vocab+1)` → 计算 entropy 完全向量化
- 写一个 `_per_row_low_card_stats(seq_arr, vocab_size, time_diff_arr, max_dt)` 工具函数

### 5.4 风险

- 🟡 **CPU 开销**：entropy 是单方案中最贵的统计。建议先只上 unique_count + top1_share，验证有 lift 后再加 entropy。
- 🟢 **零泄漏 / 零词表膨胀**。
- 🟠 **与高基数字段无关**：domain_c_seq_47 这种 287k unique 的字段做 entropy 没意义（每个值都几乎独立），不要纳入。

### 5.5 预期收益

- 🔵 **+0 % ~ +0.10 % AUC**。文献支持但本数据集 demo 没直接验证；可能与方案 B 部分冗余（活跃用户行为多样性也高）。
- 应在方案 A/B/C 之后做。

---

## 6. 方案 F：行为类型 × 时间衰减交叉特征族 🔵 探索性

### 6.1 设计动机

baseline 的低基数字段（如 domain_a_seq_40 是行为类型）目前只参与 ID embedding，**没有显式按"过去 N 天发生了几次行为类型 = k"展开**。这是经典的 multi-hot recency tensor。

### 6.2 一次性提取的特征列表（约 30~50 个）

挑 4 个 domain 中各 1 个低基数行为类型字段（vocab ≤ 20 假设）：

```text
domain_a_seq_40   假设 vocab=10
domain_b_seq_68   假设 vocab=10
domain_c_seq_28   假设 vocab=10
domain_d_seq_17   假设 vocab=10
```

对每字段，每个值 v ∈ {1..vocab}，计算：

```text
F1. count_value_v_1d
F2. count_value_v_7d
F3. count_value_v_30d
```

4 域 × 10 值 × 3 时窗 = **120 个特征**（量很大）。

更紧凑的版本：每字段只保留 top-3 高频值：

```text
4 域 × 3 高频值 × 3 时窗 = 36 个特征
```

### 6.3 实现路径

每域增加：

```python
seq_vals = result[domain][:, slot_for_field, :]          # (B, max_len)
mask_in_1d  = ((time_diff <= 86400)   & (ts_padded > 0))
mask_in_7d  = ((time_diff <= 604800)  & (ts_padded > 0))
mask_in_30d = ((time_diff <= 2592000) & (ts_padded > 0))
# 用 bincount 在 (B, vocab) 上向量化
counts_1d  = np.array([np.bincount(seq_vals[i][mask_in_1d[i]],  minlength=vocab+1) for i in range(B)])
# 取 top-3 列
```

### 6.4 风险

- 🟡 **特征维度膨胀风险高**：完整版 120 维 vs 紧凑版 36 维。建议先紧凑版。
- 🟠 **与方案 B 冗余度高**：B 的 count_7d 已经是总数；F 是按值拆分。如果 B 已上线，F 的边际收益有限。
- 🟢 **零泄漏**。

### 6.5 预期收益

- 🔵 **+0 % ~ +0.10 % AUC**。**强烈建议先做 A/B/C，再决定是否做 F**。

---

## 7. 方案 G：Dense 特征精细化拆分族（与 A-F 正交）

### 7.1 设计动机

[README.research_directions.zh.md](../README.research_directions.zh.md) §4 已分析：当前 918 → 64 一步压缩破坏了 `user_dense_feats_61/87` 的预训练 embedding 语义。这是一个**结构性 fix**，不是新增特征，但对所有方案都有放大作用。

### 7.2 一次性提取的特征列表（约 20~30 个）

不增加新统计，而是把已有 dense 字段按 fid 拆开：

```text
G1.  dense_61_proj_token            d_model 维（原始 SUM embedding）
G2.  dense_87_proj_token            d_model 维（原始 LMF4Ads embedding）
G3.  dense_62~66_concat_token       5 个 list 字段拼接 → 1 个 token
G4.  dense_89~91_concat_token       3 个 list 字段拼接 → 1 个 token
```

合计 4 个 NS token 替换原来的 1 个 user_dense token。

附加：每个 dense 字段的元信息（**这一部分才是真正的新特征**）：

```text
G5.  dense_xx_l2_norm              float32   向量 L2 范数（强度）
G6.  dense_xx_max_abs              float32   最大绝对值
G7.  dense_xx_nonzero_count        int32     非零元素数
G8.  dense_xx_sparsity             float32   稀疏度
G9.  dense_xx_is_missing           int8      整个 list 缺失
```

10 个 dense 字段 × 5 元信息 = **50 个特征**。

### 7.3 实现路径

参考 [README.research_directions.zh.md §4.3 方案 A](../README.research_directions.zh.md)。需要修改 `num_user_ns`，必然改变 T，**严格校验** `d_model % T`。

### 7.4 风险

- 🔴 **改 T 风险**：从 1 个 dense token 变成 4 个，T = num_q*4 + num_ns 会涨 3，需要重算 d_model 整除性。可能要把 num_user_ns 减少（合并几个低基数 user_int 字段）来留余量。
- 🟡 **与方案 A/D 叠加时可能 OOM**：每多 1 个 NS token 在 RankMixer full 模式下会乘上 T 因子。
- 🟢 **无泄漏 / 无词表变化**。

### 7.5 预期收益

- 🟢 **+0.10 % ~ +0.30 % AUC**（[README.research_directions.zh.md §4.5](../README.research_directions.zh.md) 估计）。
- 这是方案中**唯一不依赖外部统计**就能直接改的提点路径。

---

## 8. 综合实施顺序建议

每一档都"改动小、可独立 ablation、容易回滚"。**方案 A 是最强候选，强烈建议第一档就做**。

```text
P0 (锚点):    现状 baseline，记录 AUC + logloss + 训练时长

P1 (高 ROI):  + 方案 B（域级活跃度 60 个 dense 特征）
              理由：纯在线 + 零泄漏 + 实现量最小，可作为新 NS token 流程的"试金石"

P2 (最强提点):+ 方案 A（target × history matching，约 90 个 dense 特征）
              先做离线 lift 扫描；上 Top-15 对的 6 个特征
              ⚠️ 必须重算 d_model % T

P3:           + 方案 C（稀疏字段元统计 50 个特征）
              与 A/B 完全独立，可叠加

P4:           + 方案 G（dense per-field 拆分 + 元信息 50 个）
              结构性改造，T 会变化；要慎重
              ⚠️ 必须重算 d_model % T

P5:           + 方案 D（Item-User 双向人气 30 个）
              最高泄漏风险，需要时间感知聚合脚本
              ⚠️ 一定要 train-only 统计

P6 (探索):    + 方案 E（序列多样性 30~44 个）
P7 (探索):    + 方案 F（行为类型 × 时间衰减 36 个）
              与 B/C 冗余度较高，按需上线
```

**P0 → P3 累计预期 +0.45 % ~ +1.00 % AUC**，这是该 baseline 上"加特征不改 backbone"能取得的主要收益。
**P4 → P7 累计再 +0.20 % ~ +0.55 %**，但工程复杂度和泄漏风险显著上升。

---

## 9. 落地工程清单（每条改造必答）

1. **dataset.py 改了哪些行 / 函数？**（精确到 [dataset.py:_convert_batch](../dataset.py:505)）
2. **新增了多少 NS token？T 是否仍满足 `d_model % T == 0`？**（[model.py:1320](../model.py:1320)）
3. **schema.json 是否需要更新？**（[dataset.py:208-330](../dataset.py:208)）
4. **inference 路径是否同步？**（[model.py:1634-1714](../model.py:1634)）
5. **离线统计依赖是否只用训练集？**（方案 A 的 lift 扫描、方案 D 的全局统计）
6. **dense 特征是否做了 z-score / log1p / clip？**（每个方案的 §4 风险段）
7. **checkpoint 兼容性如何处理？**（建议 `strict=False` + 手动迁移 NS token 相关层）

---

## 10. 备查表：每个方案的"是 / 不是"

| 方案 | 在线计算 | 离线依赖 | 泄漏风险 | 改 T | 最强 demo lift | 推荐档位 |
|---|:---:|:---:|:---:|:---:|---:|:---:|
| A target × history match | ✓ | lift 扫描 | 🟢 低 | ✓ | +194 % | **P2** |
| B domain 活跃度 | ✓ |  | 🟢 低 | ✓ | +48 % | **P1** |
| C 稀疏字段元统计 | ✓ |  | 🟢 低 | ✓ |  | P3 |
| D item/user 全局人气 | lookup | ✓ 全局 | 🔴 高 | ✓ |  | P5 |
| E 多样性 / 熵 | ✓ |  | 🟢 低 | ✓ |  | P6 |
| F 行为类型 × 时窗 | ✓ |  | 🟢 低 | ✓ |  | P7 |
| G dense per-field | ✓ |  | 🟢 低 | ✓ ⚠️ |  | P4 |

→ **从 P1（B 方案）开始铺路，P2（A 方案）做最大爆点**。
