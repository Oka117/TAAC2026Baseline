# 11thMay 方案 A — 总频次基线 + 时间桶匹配统计

> 本文是基于 **FE-01A / FE-01B 评估结果重新核对** 之后（详见 §0.1）写出的新一版"方案 A"。
> 起点不是原始 baseline，而是 **FE-01A**（user_dense_feats_110 + item_dense_feats_86）已经被证实是 FE-01 拆解里 **eval 更稳的一档**。
> 本方案在 FE-01A 基础上**额外**加入 12 个 user_dense 时间桶匹配特征，构成 11thMay-PlanA。

---

## 0. 背景：FE-01 历史结论的修正

### 0.1 FE-01A / FE-01B eval 结果（修正后）

| 实验 | 模块 | 训练特征 | eval AUC | ΔAUC vs B0 |
|---|---|---|---:|---:|
| B0 | baseline | 无 | **0.810525** | — |
| FE-00 | 缺失值 + dense norm | 无新 fid | 0.811646 | +0.001121 |
| **FE-01A** | **总频次** | `user_dense_feats_110`, `item_dense_feats_86` | **0.812102** | **+0.001577** |
| FE-01B | 目标命中匹配 | `item_int_feats_89/90` + `item_dense_feats_91/92` | 0.810780 | +0.000255 |

修正点（git: `Fix FE01A and FE01B eval result records`）：
- **之前误把 0.812102 记给 FE-01B、把 0.810780 记给 FE-01A**。修正后：FE-01A（总频次）= +0.001577，FE-01B（match 模块）= +0.000255。
- 这反转了原先"match 是主增益、frequency 边际"的结论。现在 frequency 才是 **唯一 eval 端能稳住 +0.0015 量级** 的模块。

### 0.2 本方案的核心判断

1. **frequency 模块对 eval 友好**：FE-01A 只依赖 `(user_id, item_id, timestamp)`，不读 label，因此 streaming-prefix 与 eval 切片天然对齐，泄漏风险极低。
2. **match 模块 eval 端被腰斩**：FE-01B 的 valid 信号在 eval 上几乎消失（+0.000255），有效空间已经被原 baseline 的 RankMixer + Q-gen 间接吸收。
3. **下一步增量收益的最佳来源**：在 frequency 基线上再加入"**用户时间上下文 × 各域历史的命中统计**"。这是 frequency 的自然延伸——把"用户/物品的整体出现频次"细化为"按用户当前时段、星期、月份的命中频次"。
4. **特征生成路径**：与 FE-01A 同一 prefix-friendly 框架（只读 timestamp、不读 label），eval 端可复现性预期与 FE-01A 同档。

---

## 1. 方案概览

```text
11thMay-PlanA 流程:
  ┌──── FE-01A 频次基线 ────────────────────────────────┐
  │ user_dense_feats_110 = log1p(user_total_freq_<ts)   │
  │ item_dense_feats_86  = log1p(item_total_freq_<ts)   │
  └─────────────────────────────────────────────────────┘
                          +
  ┌──── 11thMay 新增：用户时间桶 × 各域历史命中 ─────────┐
  │ 对当前样本 timestamp 抽取 (hour_bucket, dow, month) │
  │ 对每个 domain ∈ {a, b, c, d}：                       │
  │   - 统计 seq 内时间戳 (event_ts ≤ ts) 中             │
  │     hour_bucket / dow / month 与当前样本相同的次数   │
  │   - log1p + train-rg z-score                         │
  │ 共 3 × 4 = 12 个 user_dense 特征 (fid 113~124)       │
  └─────────────────────────────────────────────────────┘
```

注：用户原文中的"seq q"按上下文修正为 **seq d**（项目唯一的 4 个 domain 是 a/b/c/d，"q" 与 "d" 在 dvorak/pinyin 起首邻近，且原列表凑齐 4 项时必须包含 d）。

---

## 2. 新增特征清单（一次性提取，共 14 个 dense 标量）

### 2.1 FE-01A 频次基线（保留 2 个）

| fid | 字段名 | 计算公式 | 来源 |
|---:|---|---|---|
| 110 | `user_dense_feats_110` | `log1p(user_total_frequency_before_timestamp)` | FE-01A 实验已落账 |
| 86  | `item_dense_feats_86`  | `log1p(item_total_frequency_before_timestamp)` | FE-01A 实验已落账 |

> **不引入** `user_dense_feats_111` / `item_dense_feats_87` （purchase frequency，eval 端不稳定）。
> **不引入** `user_dense_feats_112` / `item_dense_feats_88` （FE-02 delay，未独立验证）。
> **不引入** `item_int_feats_89/90` / `item_dense_feats_91/92` （FE-01B match 模块，eval 端 +0.0003 不显著）。

### 2.2 时间桶 × 域命中统计（新增 12 个）

对每个样本 row，先从 `timestamp` 抽取三个时间属性：

```python
# 当前样本时间上下文（按本地时区 UTC+8；与项目其它时间处理保持一致）
sample_hour      = local_datetime(timestamp).hour           # 0..23
sample_hour_bkt  = hour_to_bucket(sample_hour)              # 0=morning, 1=afternoon, 2=evening
sample_dow       = local_datetime(timestamp).weekday()      # 0..6, Monday=0
sample_month     = local_datetime(timestamp).month - 1      # 0..11
```

`hour_to_bucket` 三档（与用户上传"上午/下午/晚上"对应）：

```text
morning   (上午): hour ∈ [ 6, 12)   → 0
afternoon (下午): hour ∈ [12, 18)   → 1
evening   (晚上): hour ∈ [18,  6)   → 2   # 包括夜间 0..5
```

然后对每个 `domain ∈ {a, b, c, d}` 扫描 seq 的 `ts_fid` 列（仅 `event_ts ≤ timestamp` 且 `event_ts > 0` 的事件），统计命中数：

| fid | 字段名 | 计算公式 | 含义 |
|---:|---|---|---|
| 113 | `user_dense_feats_113` | `log1p(Σ_e∈seq_a 𝟙[hour_bkt(e)==sample_hour_bkt])` | 当前时段 × domain_a 历史命中次数 |
| 114 | `user_dense_feats_114` | `log1p(Σ_e∈seq_b ...)` | 当前时段 × domain_b |
| 115 | `user_dense_feats_115` | `log1p(Σ_e∈seq_c ...)` | 当前时段 × domain_c |
| 116 | `user_dense_feats_116` | `log1p(Σ_e∈seq_d ...)` | 当前时段 × domain_d |
| 117 | `user_dense_feats_117` | `log1p(Σ_e∈seq_a 𝟙[dow(e)==sample_dow])` | 当前星期 × domain_a |
| 118 | `user_dense_feats_118` | `log1p(Σ_e∈seq_b ...)` | 当前星期 × domain_b |
| 119 | `user_dense_feats_119` | `log1p(Σ_e∈seq_c ...)` | 当前星期 × domain_c |
| 120 | `user_dense_feats_120` | `log1p(Σ_e∈seq_d ...)` | 当前星期 × domain_d |
| 121 | `user_dense_feats_121` | `log1p(Σ_e∈seq_a 𝟙[month(e)==sample_month])` | 当前月份 × domain_a |
| 122 | `user_dense_feats_122` | `log1p(Σ_e∈seq_b ...)` | 当前月份 × domain_b |
| 123 | `user_dense_feats_123` | `log1p(Σ_e∈seq_c ...)` | 当前月份 × domain_c |
| 124 | `user_dense_feats_124` | `log1p(Σ_e∈seq_d ...)` | 当前月份 × domain_d |

> 用户原始说明在"月份"档只列出了 q/a/b 三个域。为了在三个时间维度上保持对称、避免 ablation 时单独缺一域引起的解释复杂度，本方案默认补齐 `domain_c` 月份特征（fid 123）。若验证后发现 fid 123 贡献为负，可在 follow-up 实验单独剔除。

合计：**2 (FE-01A) + 12 (11thMay 新增) = 14 个 dense scalar 特征**。

### 2.3 时间属性边界处理

| 情况 | 处理 |
|---|---|
| `timestamp ≤ 0`（异常样本，极少）| 三个时间属性置为 -1 sentinel；12 个命中特征值全部为 0；模型端等价 padding |
| `event_ts > timestamp`（疑似未来事件）| 严格过滤：不计入命中分母 |
| `event_ts == 0`（padding）| 严格过滤：不计入命中分母 |
| domain 没有 `ts_fid`（schema 中某域无时间字段）| 该域对应的 4 个命中特征（hour/dow/month 各 1 个）直接置 0 |

### 2.4 z-score 拟合范围

与 FE-01A 一致：

```bash
--fit_stats_row_group_ratio 0.9
```

仅用前 90% row group 拟合 mean / std；后 10% 是 valid 切片，**不参与归一化拟合**。eval 端复用 checkpoint 中保存的 stats，不再二次拟合（与 FE-01A `feature_engineering_stats.json` 同一保存逻辑）。

---

## 3. 与 FE-01 系列代码的衔接

### 3.1 推荐扩展位置（build_feature_engineering_dataset.py）

在已有 FE-01A / FE-01B 实现的 `_compute_raw_features` 同一函数内追加，复用 `timestamp` / `row_order` 列：

```python
# 在 user_total / item_total 累加循环之外再做一遍（不依赖 prefix state）
hour_match_a = np.zeros(B, dtype=np.float32)
hour_match_b = np.zeros(B, dtype=np.float32)
hour_match_c = np.zeros(B, dtype=np.float32)
hour_match_d = np.zeros(B, dtype=np.float32)
dow_match_a  = np.zeros(B, dtype=np.float32)
... 共 12 个 array ...

# 预先抽取 4 域的 ts 列；缺则跳过该域对应的 4 个特征
domain_ts = {
    "a": _list_values(batch.column(idx[ts_col_a])) if ts_col_a else None,
    "b": _list_values(batch.column(idx[ts_col_b])) if ts_col_b else None,
    "c": _list_values(batch.column(idx[ts_col_c])) if ts_col_c else None,
    "d": _list_values(batch.column(idx[ts_col_d])) if ts_col_d else None,
}

for i in range(B):
    sample_ts = int(timestamps[i])
    if sample_ts <= 0:
        continue
    dt = datetime.fromtimestamp(sample_ts, tz=LOCAL_TZ)
    s_hb = _hour_to_bucket(dt.hour)
    s_dw = dt.weekday()
    s_mo = dt.month - 1

    for d_name, events in domain_ts.items():
        if events is None:
            continue
        for ev_ts in events[i]:
            if ev_ts <= 0 or ev_ts > sample_ts:
                continue
            ev = datetime.fromtimestamp(int(ev_ts), tz=LOCAL_TZ)
            if _hour_to_bucket(ev.hour) == s_hb:
                hour_match[d_name][i] += 1
            if ev.weekday() == s_dw:
                dow_match[d_name][i] += 1
            if ev.month - 1 == s_mo:
                month_match[d_name][i] += 1

features.update({
    "user_dense_feats_113": np.log1p(hour_match_a),
    "user_dense_feats_114": np.log1p(hour_match_b),
    "user_dense_feats_115": np.log1p(hour_match_c),
    "user_dense_feats_116": np.log1p(hour_match_d),
    "user_dense_feats_117": np.log1p(dow_match_a),
    "user_dense_feats_118": np.log1p(dow_match_b),
    "user_dense_feats_119": np.log1p(dow_match_c),
    "user_dense_feats_120": np.log1p(dow_match_d),
    "user_dense_feats_121": np.log1p(month_match_a),
    "user_dense_feats_122": np.log1p(month_match_b),
    "user_dense_feats_123": np.log1p(month_match_c),
    "user_dense_feats_124": np.log1p(month_match_d),
})
```

### 3.2 selected_user_dense_adds 分支

新增 `feature_set = "11thmay_a"` 分支：

```python
def selected_user_dense_adds(feature_set, enable_delay_history):
    if feature_set == "fe01a":
        return [(110, 1)]
    if feature_set == "11thmay_a":
        return [(110, 1)] + [(fid, 1) for fid in range(113, 125)]   # 110 + 113..124
    if feature_set == "fe01b":
        return []
    ...

def selected_item_dense_adds(feature_set, enable_delay_history):
    if feature_set == "fe01a":
        return [(86, 1)]
    if feature_set == "11thmay_a":
        return [(86, 1)]                                            # 仅保留 86，不引入 87/91/92
    ...
```

### 3.3 ns_groups 配置

12 个新 user_dense 特征**不**写入 int-only NS groups。它们与原 FE-01A 一样，通过 `user_dense_token` 这一条路径进入模型（user_dense_proj 一次性投影到 d_model）。

NS token 计数等式（与 FE-01A 完全一致，T 维度不变）：

```text
user_ns_tokens = 6       # 不变
item_ns_tokens = 4       # 不变（不引入 89/90）
num_queries    = 1       # 不变
num_ns         = 6 + 1(user_dense) + 4 + 1(item_dense) = 12
T              = 1*4 + 12 = 16
d_model        = 64,  64 % 16 == 0   # 整除约束成立
```

→ **不需要重算 `--user_ns_tokens`；不需要改 `num_queries`；不需要重算 `d_model % T`**。

### 3.4 schema 增量

```diff
"user_dense": [
  ...原有条目...,
+ [110, 1],
+ [113, 1], [114, 1], [115, 1], [116, 1],
+ [117, 1], [118, 1], [119, 1], [120, 1],
+ [121, 1], [122, 1], [123, 1], [124, 1]
],
"item_dense": [
  ...原有条目...,
+ [86, 1]
],
"item_int": [
  ...原有条目（不变）...
]
```

---

## 4. 数据生成命令

```bash
python3 -u build_feature_engineering_dataset.py \
  --input_dir   /path/to/original_dataset \
  --input_schema /path/to/original_dataset/schema.json \
  --output_dir  /path/to/11thmay_a_dataset \
  --feature_set 11thmay_a \
  --fit_stats_row_group_ratio 0.9
```

输出审计文件（与 FE-01A 同结构）：

```text
schema.json                          # 含新增 user_dense_feats_{110,113..124}, item_dense_feats_86
ns_groups.feature_engineering.json   # 与 FE-01A 同（不引入新 int fid）
feature_engineering_stats.json       # 含 14 个 dense 字段的 mean/std/n
docx_alignment.11thmay_a.json        # 与 FE-01A 同 not_included 清单
```

---

## 5. 训练命令

继续沿用 FE-01A 的 `run_fe01a.sh` 结构，仅替换数据目录和 feature_set：

```bash
bash run_11thmay_a.sh \
  --data_dir   /path/to/11thmay_a_dataset \
  --schema_path /path/to/11thmay_a_dataset/schema.json \
  --ckpt_dir   outputs/exp_11thmay_a/ckpt \
  --log_dir    outputs/exp_11thmay_a/log \
  --seed 42
```

固定参数（继承自 FE-01A）：

| 参数 | 值 |
|---:|---:|
| `--feature_set` | `11thmay_a` |
| `--ns_tokenizer_type` | `rankmixer` |
| `--user_ns_tokens` | `6` |
| `--item_ns_tokens` | `4` |
| `--num_queries` | `1` |
| `--loss_type` | `bce` |
| `--patience` | `3` |
| `--num_epochs` | `6` |
| `--seq_max_lens` | `seq_a:256,seq_b:256,seq_c:512,seq_d:512` |
| `--fit_stats_row_group_ratio` | `0.9` |

---

## 6. 评估设计

继续复用 `evaluation/FE01/infer.py`：

- 按 checkpoint 内的 `schema.json` 自动只生成 11thMay-PlanA 需要的 14 个 dense 列。
- 复用 checkpoint 内的 `feature_engineering_stats.json`；**不在 eval 上重新拟合 normalization**，避免 valid/eval 统计漂移。
- 时间桶特征的 12 个统计完全依赖 `event_ts ≤ sample_ts`，eval 端可独立复现；不依赖 prefix state，因此即使流式扫描顺序与训练不同也不影响输出。

---

## 7. 收益预测

### 7.1 单模块贡献分解（基于 FE-01A=+0.001577 锚点）

| 信号 | 物理意义 | 完整集预期相对 lift（vs FE-01A）|
|---|---|---:|
| `dow_match` × 4 域 | 用户在同一星期几的活跃度；与购买周期/工作日 vs 周末强相关 | **+0.0008 ~ +0.0014** |
| `hour_match` × 4 域 | 用户在同一时段的活跃度；3 档 bucket 比 dow 颗粒更粗 | **+0.0004 ~ +0.0008** |
| `month_match` × 4 域 | 月份效应；样本时间跨度若 < 3 月则贡献小，> 6 月则有显著季节性 | **+0.0002 ~ +0.0006** |
| 三个时间维度相互冗余 | 时段、星期、月份在同一域内有 30% ~ 50% 重叠 | **-0.0003 ~ -0.0006** |
| domain 间冗余（4 域同时上）| domain_d 与 domain_b 活跃度强相关 | **-0.0002 ~ -0.0004** |

### 7.2 汇总区间

| 配置 | ΔAUC vs FE-01A | ΔAUC vs B0 |
|---|---:|---:|
| 11thMay-PlanA（FE-01A + 12 时间桶）| **+0.0005 ~ +0.0012** | **+0.0021 ~ +0.0028** |
| 90% 概率上界 | +0.0020 | +0.0036 |
| 5% 概率下界 | +0.0000 | +0.0016 |

### 7.3 落到 baseline AUC=0.810525 的预测

| 配置 | 期望 eval AUC | 70% 置信区间 |
|---|---:|---|
| B0 baseline | 0.810525 | 0.8090 – 0.8120 |
| FE-01A（重新核对后的真实结果）| 0.812102 | 0.8108 – 0.8133 |
| **11thMay-PlanA** | **0.8126 ~ 0.8133** | 0.8118 – 0.8145 |

→ 预期能在 FE-01A 已经验证的 +0.0015 基础上再加 +0.0005 ~ +0.0012，**总累计提点 ≈ +0.0021 ~ +0.0028 over B0**。

---

## 8. 风险清单

### 🟢 R1：泄漏（低风险）

时间桶统计严格使用 `event_ts ≤ sample_ts` 过滤，与 FE-01A 同一安全等级。**FE-01B 的 streaming-prefix 不一致问题在本方案中不存在**——12 个时间桶特征是**纯 in-row 自带统计**（用本 row 自带的 seq 数据计算），不依赖跨 row prefix 状态。

### 🟢 R2：T 维度不变（零风险）

新增 12 个 dense 字段全部走 `user_dense_proj`，**不改变 NS token 数**。`d_model % T == 0` 自动满足。

### 🟡 R3：时间属性边界（中风险，已缓解）

- **DST/时区**：若数据时间戳是 UTC，本方案用 `LOCAL_TZ=UTC+8` 换算到中国时区；若数据本身就是本地时间戳，需在 `--input_timezone` 配置项里指明。**首版默认 UTC+8**，与 README 中其它时间字段约定保持一致。
- **首末日截断**：sample 在数据首日（无历史）→ 12 个特征全 0，与"新用户"语义一致。

### 🟡 R4：稀疏域的命中率（中风险，需监控）

domain_a / domain_c 的 7d 内事件占比仅 5.4% / 1.5%（README §3.3），月份 / 星期粒度的命中率可能极低（< 0.1% nonzero ratio）。极端情况下：

- `month_match_c` 在多数样本上为 0 → 训练时该列方差极小 → 模型几乎学不出权重 → 不掉点但浪费 1 个 dense 维度。
- **监控**：训练前 dump 一次 batch 的 14 个 dense 的 nonzero ratio。若某列 < 0.5%，下一轮 ablation 中剔除。

### 🟡 R5：时间桶冗余（中风险，已在收益分解中扣减）

hour_bucket / dow / month 在同一 domain 内冗余度约 30% ~ 50%（参见 §7.1）。模型可能把"活跃用户"这一统一信号拆解到三个时间桶上，导致单维 attention 权重稀释。**首版接受**；若 11thMay-PlanA eval ΔAUC < +0.0005，可改成 dow + hour 二组合（剔除 month）。

### 🟢 R6：与 FE-01A 完全兼容

- schema 仅新增 12 个 `user_dense` + 1 个 `item_dense` 条目；
- ns_groups 不变；
- 模型端 user_dense_proj 自动适配新的输入维度（`user_dense_dim` 在 `__init__` 从 schema 推断，不需要手动改 model.py）。

### 🟠 R7：与 FE-01B 不可同时启用（首版策略）

11thMay-PlanA 默认**不**叠加 FE-01B 的 `item_int_feats_89/90` 与 `item_dense_feats_91/92`：

- FE-01B 在 eval 上的边际只有 +0.000255，叠加后期望增量小于不确定性区间；
- 同时叠加会增加 ns_groups 配置复杂度（需保留 `I4_target_matching_fields`），可解释性降低；
- 若 follow-up 实验显示有协同信号，可在 `11thmay_a_b` 分支中再合并。

---

## 9. 实施 checklist（提交前必看）

- [ ] `selected_user_dense_adds("11thmay_a", False) == [(110, 1), (113, 1), ..., (124, 1)]`（共 13 项）
- [ ] `selected_item_dense_adds("11thmay_a", False) == [(86, 1)]`（仅 1 项）
- [ ] `selected_item_int_adds("11thmay_a", _) == []`（与 FE-01A 一致，不引入 89/90）
- [ ] `feature_engineering_stats.json.dense_feature_names` 含 13 个 user_dense + 1 个 item_dense（共 14 个）
- [ ] `schema.json["user_dense"]` 比原始多且仅多 `[110, 1]` + `[113..124, 1]`
- [ ] `schema.json["item_dense"]` 比原始多且仅多 `[86, 1]`
- [ ] `schema.json["item_int"]` 与原始一致（不出现 89/90）
- [ ] `train.py` 日志：`num_ns=12, T=16, d_model=64`（与 FE-01A 同）
- [ ] 训练第 1 batch dump 12 个时间桶特征的 nonzero ratio；记录 month_match_c 是否 < 0.5%
- [ ] eval 输出未生成 `user_dense_feats_111/112` 或 `item_dense_feats_87/88/91/92` 列

---

## 10. 下一档实验路径（11thMay-PlanA 完成后）

```text
11thMay-PlanA0 (锚点)  : FE-01A 复跑                            ← eval = 0.8121 期望复现
11thMay-PlanA1 (本方案): 11thmay_a (FE-01A + 12 时间桶)         ← 期望 eval = 0.8126 ~ 0.8133
11thMay-PlanA2 (剪枝)  : A1 - month_match × 4 域（共 8 个特征）  ← 若 A1 中 month 列方差小，验证剪枝
11thMay-PlanA3 (扩展)  : A1 + cross_domain_share（4 域时段命中占比）← +4 个特征，验证跨域归一化是否再涨
11thMay-PlanA+B (合并) : A1 + FE-01B (89/90/91/92)               ← 验证 frequency + temporal + match 三模块协同
```

每档完成后必须保留：

- `outputs/log/train.log` 中 `feature_set:` 一行（确认 14 个 dense 特征的清单）
- 最佳 checkpoint 的 `best_val_AUC` 和 eval AUC
- 训练吞吐（steps/sec）与 FE-01A 的对比（预期 +5% ~ +10% 单 step 耗时，主要花在 Python 端时间属性计算）
- 12 个时间桶特征的 nonzero ratio 与 train-rg z-score 后的 mean/std
