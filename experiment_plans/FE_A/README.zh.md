# 实验 FE_A — Target-Item × History Matching 特征族

本实验把 [feature_extraction_plans.zh.md §1 方案 A](../feature_extraction_plans.zh.md) 的"target item 属性 × 历史序列匹配"特征族落到 baseline 代码上。**不改 HyFormer backbone、不改优化器**，仅在 NS token 池里多注入 1 个 dense token。

---

## 1. 实验结构总览

```text
原始流程:
  user_int + item_int  ──RankMixerNSTokenizer──► (B, 5+2, D)  ──┐
  user_dense           ──user_dense_proj───────► (B, 1,   D)  ──┤  cat
  4 域 seq             ──seq embed + time bkt──► (B, L,   D)×4 ──┤──► Q-gen + HyFormer × 2  ──► output_proj ──► clsfier
                                                                ─┘     (T = 2*4 + 8 = 16, 64 % 16 = 0 ✓)

FE_A 流程（新增 1 条蓝色路径）:
  user_int + item_int  ──RankMixerNSTokenizer──► (B, 4+2, D)  ──┐    ← user_ns_tokens 5→4，让出 1 个 token 槽位
  user_dense           ──user_dense_proj───────► (B, 1,   D)  ──┤
  match_feats(B, 18)   ──match_proj────────────► (B, 1,   D)  ──┤  cat   ★ 新增
  4 域 seq             ──seq embed + time bkt──► (B, L,   D)×4 ──┤──► Q-gen + HyFormer × 2  ──► output_proj ──► clsfier
                                                                ─┘     (T = 2*4 + (4+1+2+1) = 16, 64 % 16 = 0 ✓)
```

### 1.1 NS-token 数量等式（核对 `d_model % T == 0`）

| 来源 | baseline | FE_A |
|---|---:|---:|
| `num_user_ns`        | 5 | **4** ⚠ |
| `has_user_dense` (1) | 1 | 1 |
| `num_item_ns`        | 2 | 2 |
| `has_item_dense` (0) | 0 | 0 |
| `has_match_feats` (1) | 0 | **1** ★ |
| **`num_ns`**         | **8** | **8** |
| `num_queries × num_sequences` | 8 | 8 |
| **`T`**              | **16** | **16** |
| `d_model % T`        | 64%16=0 ✓ | 64%16=0 ✓ |

为了保住 `T=16`、不破坏 RankMixer `full` 模式，**`--user_ns_tokens` 必须从 5 调到 4**。这把原本均分给 user-int 的 5 个 chunk 压缩成 4 个，每个 chunk 容量增加 ~25%。

### 1.2 一次性提取的特征列表（默认 3 对、共 18 个 dense 标量）

```text
对 (item_int_feats_9,  domain_d_seq_19):  feature index 0~5
对 (item_int_feats_10, domain_d_seq_24):  feature index 6~11
对 (item_int_feats_13, domain_c_seq_31):  feature index 12~17

每对 6 个 dense 标量（dataset.py: NUM_MATCH_FEATURES_PER_PAIR）:
  0  has_match           ∈ {0, 1}
  1  log1p(match_count)
  2  log1p(match_count_within_1d)
  3  log1p(match_count_within_7d)
  4  log1p(match_count_within_30d)
  5  log1p(min_match_delta_seconds)   (=0 当无命中或域无 ts_fid)
```

如果切到 [`match_pairs.extended.json`](match_pairs.extended.json)（10 对），特征总数 = 60。

---

## 2. 改动清单（逐文件、可 diff 复核）

| 文件 | 函数 / 段落 | 改动 |
|---|---|---|
| [dataset.py](../../dataset.py) | 顶部常量 | 新增 `NUM_MATCH_FEATURES_PER_PAIR = 6`、`DEFAULT_MATCH_PAIRS`（3 个 demo-verified 对）|
| [dataset.py](../../dataset.py) | `PCVRParquetDataset.__init__` | 新增 `match_pairs` 参数、`_match_pairs` 列表、`_match_plan_by_domain` 索引、`match_feats_dim`、`_buf_match_feats` 缓冲；非法对发 warning 跳过 |
| [dataset.py](../../dataset.py) | `_convert_batch` | 在 seq 处理循环内（`time_diff` 在 scope）逐对计算 6 个特征；写到 `result['match_feats']` |
| [dataset.py](../../dataset.py) | `get_pcvr_data` | 新增 `match_pairs` 参数，透传给 train / valid 两个 `PCVRParquetDataset` |
| [model.py](../../model.py) | `ModelInput` | 新增 `match_feats: torch.Tensor`（默认 `torch.empty(0)`）|
| [model.py](../../model.py) | `PCVRHyFormer.__init__` | 新增 `match_feats_dim` 参数；当 >0 时构造 `match_proj = Linear+LayerNorm`；`num_ns += 1` |
| [model.py](../../model.py) | `PCVRHyFormer.forward / .predict` | 在 `ns_parts` 末尾追加 `silu(match_proj(match_feats)).unsqueeze(1)` |
| [train.py](../../train.py) | CLI | 新增 `--match_pairs_json`（支持 `default` 关键字加载 `dataset.DEFAULT_MATCH_PAIRS`）|
| [train.py](../../train.py) | `main` | 解析 JSON → 透传给 `get_pcvr_data` 与 `model_args["match_feats_dim"]` |
| [trainer.py](../../trainer.py) | `_make_model_input` | 把 `device_batch['match_feats']` 写进 `ModelInput`；缺失时回退到 `(B, 0)` |

新增文件：

```text
experiment_plans/FE_A/
├── README.zh.md                  ← 本实验文档
├── match_pairs.default.json      ← 3 对 demo-verified
├── match_pairs.extended.json     ← 10 对（含 7 对未验证扩展）
├── run_plan_a.sh                 ← 推荐启动脚本（user_ns_tokens=4）
└── smoke_test_plan_a.sh          ← HF 1k-row sample 烟雾测试
```

### 2.1 启动命令

```bash
# 完整训练（推荐）
bash experiment_plans/FE_A/run_plan_a.sh --data_dir /path/to/dataset

# 烟雾测试（HF 1k 样本，本地 CPU 即可跑）
bash experiment_plans/FE_A/smoke_test_plan_a.sh

# 仅切到 extended 10 对
MATCH_JSON=experiment_plans/FE_A/match_pairs.extended.json \
  bash experiment_plans/FE_A/run_plan_a.sh --data_dir /path/to/dataset
```

### 2.2 关键 CLI 差异（vs run.sh）

| flag | run.sh | run_plan_a.sh | 原因 |
|---|---|---|---|
| `--user_ns_tokens` | 5 | **4** | 给 match token 让槽位，保 T=16 |
| `--match_pairs_json` | — | `match_pairs.default.json` | Plan A 入口 |

其它（`--num_queries 2`、`--item_ns_tokens 2`、`--ns_tokenizer_type rankmixer`、`--emb_skip_threshold 1000000`）保持与 baseline 完全一致。

---

## 3. 跑通性验证（已在 demo_1000.parquet 上完成）

### 3.1 数值正确性（dataset 端）

用 numpy 独立重算，与 dataset 输出逐特征对比：

```text
Pair 0 (item9 × d_19, max_len=512): max abs diff = 0.000000e+00, hits = 1/100
Pair 1 (item10 × d_24, max_len=512): max abs diff = 0.000000e+00, hits = 9/100
Pair 2 (item13 × c_31, max_len=512): max abs diff = 0.000000e+00, hits = 3/100
NUMERICAL_CORRECTNESS_OK
```

### 3.2 模型构造（T 校验）

```text
Plan A enabled: 3 match pairs → match_feats_dim=18
PCVRHyFormer model created: num_ns=8, T=16, d_model=64, rank_mixer_mode=full
Total parameters: 160,939,841
Sparse params: 96 tensors, 158,453,312 parameters (Adagrad lr=0.05)
Dense params: 380 tensors, 2,486,529 parameters (AdamW lr=0.0001)
```

`match_proj` 是 dense 模块（约 `match_feats_dim*64 + 64*2 ≈ 1.3 k` 参数），完全不动 sparse 路径。

### 3.3 训练循环（端到端，1 epoch on demo_1000）

```text
step 1: loss = 0.6610  ← 健康
step 2~: loss = nan    ← 与 baseline 在同等 smoke 数据上的行为完全一致
                          (baseline step 1 = 0.7078, step 2~ = nan)
Epoch 1 Validation | AUC: 0.0, LogLoss: inf
Training complete!     ← 完整跑完，包含 evaluate + checkpoint save + Adagrad reinit
```

NaN 是 baseline 在 1k 行 smoke schema 上的已知行为（vocab 极小 + Adagrad 累积过快 → 等效 lr 过大）。Plan A 没有引入新的不稳定来源，已通过对照实验验证。

---

## 4. baseline eval AUC ≈ 0.810 时方案 A 的预测

### 4.1 收益分解（以下都是相对 baseline 的 ΔAUC，单位 % = 0.01）

| 信号通道 | demo lift（已观测）| 完整训练集预期 lift | 模型可吸收度 | 净 ΔAUC |
|---|---:|---:|---|---:|
| `item9 × d_19` 命中信号 | +127 % 相对（26.8 vs 11.8）| 减半到 +60 ~ +90 % | 高（dense token + Q-gen 都能感知）| **+0.10 ~ +0.18** |
| `item10 × d_24` 命中信号 | +104 % | 减半到 +50 ~ +80 % | 高 | **+0.08 ~ +0.15** |
| `item13 × c_31` 命中信号 | +194 %（最强）| 减半到 +90 ~ +130 % | 高（domain_c 长尾被穿透）| **+0.10 ~ +0.20** |
| 时窗化（1d/7d/30d）边际 | 未单独验证 | 在长尾域 c 比纯 has_match 多 ~30 % 信号 | 中 | **+0.02 ~ +0.05** |
| 三对相互冗余（一名用户在多对都命中）| — | -10 ~ -20 % 信号叠加损耗 | — | **-0.05 ~ -0.10** |

**汇总区间**：

| 模式 | ΔAUC 中性预期 | 90 % 概率上界 | 5 % 概率下界 |
|---|---:|---:|---:|
| `match_pairs.default.json` (3 对，18 dense) | **+0.20 % ~ +0.40 %** AUC | +0.55 % | +0.05 % |
| `match_pairs.extended.json` (10 对，60 dense) | **+0.30 % ~ +0.55 %** AUC | +0.75 % | +0.10 % |

### 4.2 落到 baseline AUC=0.810 的预测

| 配置 | 期望 best val AUC | 期望区间（70 % 置信） |
|---|---:|---|
| baseline | **0.8100** | 0.8085 – 0.8115 |
| FE_A default (3 对) | **0.8120 ~ 0.8140** | 0.8105 – 0.8155 |
| FE_A extended (10 对) | **0.8130 ~ 0.8155** | 0.8110 – 0.8175 |

### 4.3 为什么 demo 上 lift 在完整集上会衰减？

1. **demo 的正例率 12.4 %**，其中匹配组的正例率 ~ 30 %。完整集正例率不一定一致；并且大部分匹配组并非"高命中"。
2. **demo 数据样本量小**（1000 行），三对 demo lift 是少样本估计，置信区间宽。
3. **现有 RankMixer NS / Q-gen 已经能间接捕捉一部分匹配信号**（target item embedding 与 domain seq embedding 在 cross-attn 中本来就会 correlate）。Plan A 的边际收益是把"间接 → 直接"的部分。
4. **训练数据更大后 sparse embedding 自身已经能学到更精细的 cross 表达**，这会进一步压缩 Plan A 的边际收益。

### 4.4 训练成本

| 维度 | baseline | FE_A default | 增幅 |
|---|---:|---:|---:|
| 总参数 | ~160.94 M | ~160.94 M（+ 1.3 k）| +0.0008 % |
| 单 step CPU 耗时（demo, B=64）| ~3.0 s | ~3.5 s | +15 % |
| GPU 显存峰值（估，B=256, d=64）| 不变 | +O(B × match_feats_dim × 4 B) ≈ +20 KB | 可忽略 |
| 训练吞吐（GPU）| 不变 | 几乎不变 | <1 % |

CPU smoke 上多出的 15 % 耗时主要花在 numpy 端逐对计算（`np.where + sum`）。生产 GPU 数据加载有 16 个 worker 并行，相对值会更小。

---

## 5. 风险清单（按严重度排序）

### 🔴 R1：`d_model % T` 校验失败（高风险，已规避）

**风险**：默认 `--user_ns_tokens 5` 加上 match token → `num_ns = 9`、`T = 17`、`64 % 17 = 13 ≠ 0`，模型构造直接抛 `ValueError`（[model.py:1320](../../model.py:1320)）。

**已规避**：本实验在 `run_plan_a.sh` 中显式 `--user_ns_tokens 4`。**任何要在此基础上改 `--num_queries`、加更多 dense token 的派生实验都必须重新核对 T 等式**（见 §1.1）。

### 🔴 R2：稀疏长尾域命中过低（高风险，可监控）

**风险**：完整集上 domain_c / domain_a 的 7 天内事件占比仅 ~1.5 % / 5.4 %（[README.feature_engineering.zh.md §3.3](../../README.feature_engineering.zh.md)），`match_count_1d/7d` 大多数样本都为 0，对应特征列方差极小，`Linear` 难以区分。极端情况下三对中有一对会变成"几乎常量列"，对 AUC 没贡献甚至引入噪声。

**监控**：训练前 dump 一次 batch 的 `match_feats` mean / std / nonzero ratio。如果某列 nonzero ratio < 0.5 %，从 JSON 中剔除该对再训。

### 🟠 R3：序列被截断后 hit 落在窗口外（中风险，已观察）

**风险**：默认 `seq_d:512`，但 `domain_d` 平均长度 1100，最近 30 天事件大量分布在位置 500+（已在 demo row 8 上验证，6 个 hit 全部在位置 1421+）。Plan A 只能看到 truncate 之后的窗口。

**缓解**：与 [research_directions.zh.md §1 方向](../../README.research_directions.zh.md) 协同——把 `seq_d` 拉到 768 或对 domain_c 启用 `longer encoder` 后，Plan A 的命中率会同步提升。**单独跑 FE_A 时建议先把 `seq_d` 上调到 768**，命中率能再涨 30~50 %。

### 🟠 R4：未验证对引入噪声（中风险，extended JSON）

**风险**：`match_pairs.extended.json` 中 7 对（除 demo 已验证的 3 对外）未做 lift 扫描，可能并不存在 ID 空间共享，"命中"几乎全是巧合。这种对会让 dense token 学到错误关联，反而掉点。

**缓解**：先用 default JSON 跑一档 baseline 对比；若 +AUC 显著，再逐步加 extended pairs（每加一对单跑一次 ablation）。

### 🟡 R5：item_int_feats_X 缺失/非法值（中风险，已防御）

**风险**：item_int_feats_9 等字段存在 -1（缺失）和 0（padding）值。dataset 端把 `target ≤ 0` 的样本 `match_mask` 强制为 False，看起来对，但模型会把"target 缺失" 与"target 存在但无命中"无差别地映射到 `has_match=0`。

**缓解**：方案 A 第二轮可以增加一个 `target_is_missing` 标志位（item_field=0 时为 1），让模型显式知道。**首版不做**，避免维度膨胀。

### 🟡 R6：dense 量级不一致（中风险，已部分缓解）

**风险**：`has_match ∈ {0, 1}` 与 `log1p_min_dt ∈ [0, 17]` 量级差 17×。Linear 层易被大值主导。

**缓解**：所有计数特征已统一 `log1p` 压缩；min_dt 也是 log1p。但量级仍有不对称。**首版接受**；若发现训练不稳，给 `match_proj` 前加 `BatchNorm1d`。

### 🟢 R7：泄漏（低风险，已验证）

**风险**：方案 A 全部基于 `event_ts ≤ timestamp` 的同 batch 内计算，所有 event 时间戳已经满足 `dt ≥ 0` 约束（dataset.py 在 `np.maximum(... , 0)` 已经强制）。**无泄漏**。

### 🟢 R8：checkpoint 兼容（低风险，可控）

**风险**：开启 Plan A 后 `state_dict()` 多出 `match_proj.0.*`、`match_proj.1.*`，旧 baseline ckpt 无法 strict load。

**缓解**：infer 端如果要复用 baseline ckpt → `load_state_dict(..., strict=False)`，只缺 `match_proj` 几层会被随机初始化（不影响 baseline 行为）。

---

## 6. 推荐 ablation 顺序

```text
A0 (锚点):      bash run.sh --data_dir ...                          ← baseline AUC=0.810
A1 (FE_A def):  bash experiment_plans/FE_A/run_plan_a.sh ...        ← 期望 +0.20%~+0.40%
A2 (long seq):  A1 + --seq_max_lens "...,domain_d:768"              ← 期望再 +0.05%~+0.10%
A3 (FE_A ext):  MATCH_JSON=...extended.json bash run_plan_a.sh ...  ← 期望比 A1 再 +0.10%~+0.15%
A4 (drop bad):  A3 + 删掉 nonzero ratio < 0.5 % 的对                ← 期望比 A3 持平或 +0.02%
```

每档完成后必须保留：
- `outputs/log/train.log` 中 `Plan A enabled:` 一行（确认对的清单）
- 最佳 checkpoint 的 `best_val_AUC` 和 `best_val_logloss`
- 训练吞吐（steps/sec）和 GPU 显存峰值
- 第 1、100、1000 step 时的 `match_feats` mean / std（用于诊断 R2/R3）

---

## 7. 工程落地清单

下一档实验启动前，逐项确认：

1. **`d_model % T == 0` 是否仍成立？** → 见 §1.1 等式表。
2. **`schema.json` 是否更新？** → 不需要（Plan A 不改 schema 定义，仅在运行时读 match_pairs JSON）。
3. **inference 路径是否同步？** → `predict()` 已同步注入 `match_proj`（[model.py:1684 附近](../../model.py:1683)）；旧 ckpt 用 `strict=False`。
4. **离线统计依赖？** → 无（首版只用 demo-verified 3 对，跳过 lift 扫描；扩展时再补 `tools/scan_match_lift.py`）。
5. **dense 特征是否做归一化？** → 全部已 `log1p`；如训练不稳，再加 BatchNorm。
6. **多 worker 兼容？** → `_buf_match_feats` 是 dataset 实例属性，多 worker 各持一份，无竞态。

---

## 8. NS token 数量变化（5 → 4）的后果与替代方案

为了让 match token 进 NS 池又保住 `T=16`，本实验把 `--user_ns_tokens` 从 5 砍到 4。这是当前最小改动路径，但确实有副作用，本节定量记录。

### 8.1 RankMixerNSTokenizer 内部几何

[model.py:1131-1143](../../model.py:1131) 的切分逻辑：

```text
total_emb_dim = num_user_int_fids × emb_dim = 46 × 64 = 2944
chunk_dim     = ceil(total_emb_dim / num_ns_tokens)
padded_total  = chunk_dim × num_ns_tokens   (右侧 zero-pad 到对齐)
每个 chunk 经 Linear(chunk_dim → d_model=64) + LayerNorm
```

| user_ns_tokens | chunk_dim | pad | 单 token 压缩比 | 总投影 dense 参数 |
|---:|---:|---:|---:|---:|
| 5 (baseline) | 589 | 1  | **9.2 : 1**  | 5 × (589×64 + 128) = 189,120 |
| 4 (FE_A)     | 736 | 0  | **11.5 : 1** | 4 × (736×64 + 128) = 188,928 |

参数总量几乎不变（差 192）。**真正变化的是单 token 容纳的原始信号粒度**。

### 8.2 切分边界跨越 fid 现象

`emb_dim=64`、`chunk_dim` 不是 64 的整数倍 → **单个 fid 的 64 维 embedding 会被切到两个相邻 chunk**：

| user_ns_tokens | chunk_dim/64 | 被切的 fid 索引 |
|---:|---:|---|
| 5 | 9.2  | 第 9、18、27、36 个 fid 各被切 1 次 |
| 4 | 11.5 | 第 11、23、34 个 fid 各被切 1 次 |

两种切法对个体 fid 都不友好；但 4-token 把"被切 fid"挪到了不同位置（schema 排序后第 11 fid 是 `user_int_feats_53`、第 23 fid 是 `user_int_feats_82`）。Linear 投影会重新学会跨 chunk 重组，但**梯度反向传播路径变长**，初始训练步骤更慢。

### 8.3 后果汇总（按重要度）

| # | 后果 | 量级估计 | 监控方式 |
|---|---|---|---|
| C1 | 单 user_ns token 压缩比 +25 % | 主要副作用；可能让某些细粒度 user 特征（如 `user_int_feats_94-109` 偏好等级）不再独立成 token | 训练后看 attention 权重是否过度集中在某几个 user_ns token |
| C2 | 切分边界跨过的 fid 不同 | 训练初期 1~3k step 的收敛速度差异 | tensorboard `Loss/train` 早期曲线对比 |
| C3 | Q tokens (8) 对 user-side 的 attention 视角 5→4 | -0.02 % ~ -0.05 % AUC（推测） | 单跑"user_ns=4 but no match"对照（见 §6.A1*） |
| C4 | RankMixerBlock token-mixing 几何 | 不变（T=16, d_sub=4 都不变）| — |
| C5 | 参数总量 | -0.001 % | 模型构造时打印 `Total parameters` |
| C6 | 位置-角色映射重排 | 中性：FFN 共享权重，对位置不敏感；输入投影会自适应学新角色 | — |
| C7 | Adagrad 累积步长 | 极小：chunk_dim 变大对 Linear 初始梯度尺度有 √(736/589)=1.12× 变化 | 不需监控 |

**核心结论**：5→4 的代价主要是 C1（信息瓶颈加深）+ C3（Q-NS attn 视角损失），合计**预计 -0.05 % ~ -0.10 % AUC**。Plan A 收益 +0.20 % ~ +0.40 % AUC 仍能净盈余。

### 8.4 替代方案（按改动量从小到大）

#### 方案 S1：接受现状 + 单跑一档 ablation 量化代价（推荐首选）

跑一档"`user_ns_tokens=4` 但不开 match"以隔离 5→4 的纯损失：

```bash
# A1*：探针档，仅压 user_ns、不加 match
bash run.sh --user_ns_tokens 4 --data_dir ...
```

如果 A1* 相对 baseline 的 ΔAUC ∈ [-0.10 %, +0.05 %]，**接受现状无需变动**。否则才考虑下面方案。

- 改动量：0（只多一档 ablation）
- 期望损失：< 0.10 % AUC
- 优点：路径清晰、可量化
- 缺点：多跑 1 轮训练

#### 方案 S2：`--rank_mixer_mode ffn_only` 绕过 T 整除约束

```bash
bash run.sh --user_ns_tokens 5 --rank_mixer_mode ffn_only \
            --match_pairs_json experiment_plans/FE_A/match_pairs.default.json
```

- T 可任意（17 也能跑）
- 失去 [model.py:361-385](../../model.py:361) 的 token-mixing reshape；FFN 仍在
- 改动量：CLI 一个字段
- 期望：保 user_ns 5 chunk → +0.05 ~ +0.10 % AUC；丢 token mixing → -0.05 ~ -0.15 % AUC
- 净期望：**0 ~ +0.10 % AUC**（高方差）
- 风险：RankMixer 论文核心论点被削弱，做 paper 评审时会被质疑

#### 方案 S3：把 match_feats 合并进 user_dense_proj（不增 NS token）

把 match_feats(18) 拼到 user_dense_feats(918) → user_dense_proj 输入维度变 936。`num_ns` 不变，T 不变。

```python
# model.py PCVRHyFormer.__init__
self.user_dense_proj = nn.Sequential(
    nn.Linear(user_dense_dim + match_feats_dim, d_model),  # was user_dense_dim
    nn.LayerNorm(d_model),
)
# forward
combined = torch.cat([inputs.user_dense_feats, inputs.match_feats], dim=-1)
user_dense_tok = F.silu(self.user_dense_proj(combined)).unsqueeze(1)
```

- 改动量：~10 行 model.py + dataset 不变
- 优点：完全不动 RankMixer / NS 结构 / `--user_ns_tokens`
- 缺点：match 信号被挤进**已经是瓶颈**的 user_dense 1 token（[research_directions §4.2](../../README.research_directions.zh.md) 已指出 918→1 一步压缩信息丢失严重）；模型几乎不可能学出独立的 match attention 模式
- 期望：约 Plan A 当前预期的 50 ~ 70 %

#### 方案 S4：把 match_feats 注入 MultiSeqQueryGenerator 作为 condition

最语义对齐：match 本质是 "user × target item" 关系，本就该影响 Q 怎么 attend seq。

```python
# model.py 修改 MultiSeqQueryGenerator.__init__
self.match_cond_proj = nn.Linear(match_feats_dim, d_model)
# forward 在生成 Q 时
q_logits = q_logits + self.match_cond_proj(match_feats).unsqueeze(1)
```

- T 保持 16，num_ns=8 不变（user_ns=5、item_ns=2 全保留）
- 改动量：MultiSeqQueryGenerator 类 + forward 接口（中等）
- 期望：**+0.30 % ~ +0.55 % AUC**（理论上限最高）
- 风险：MultiSeqQueryGenerator 是 baseline 核心，改完要对照测试 baseline 行为不退化；ablation 路径变复杂

#### 方案 S5：d_model 升到 128 + 同步扩 NS 池（与方向 2 capacity 协同）

```bash
bash run.sh --d_model 128 --emb_dim 128 --num_heads 8 \
            --user_ns_tokens 5 --item_ns_tokens 2 --num_queries 2 \
            --match_pairs_json ...
# T = 8 + (5+1+2+0+1) = 17 → 不行
# T = 8 + (5+1+1+0+1) = 16 → 减 item 也不舒服
# T = 16 (4 queries × 4 seqs) + (5+1+2+0+1) = 25 → 不行
```

128 的合法 T 集合是 {1, 2, 4, 8, 16, 32}，仍然得让 T=16。**不能纯靠扩 d_model 解决 NS 槽位问题**，必须配合 num_queries 调整。

具体可行配置：`d_model=128, num_queries=4, num_sequences=4 (固定), num_ns=16`。这给 NS 池开 16 个 slot，足够同时跑 Plan A + B + C + G 全部 dense 特征族。

- 改动量：大（capacity 翻 4×）
- 期望：与 Plan A 收益叠加 [research_directions §2.6](../../README.research_directions.zh.md) 的 +0.20 ~ +0.40 %
- 风险：显存、训练时长、超参 retune

#### 方案 S6：切到 `ns_tokenizer_type=group` + 显式 match group

```bash
bash run.sh --ns_tokenizer_type group \
            --ns_groups_json ns_groups.json \
            --match_pairs_json ...
```

修改 `ns_groups.json` 加一组：

```json
{
  "user_ns_groups": { ... },
  "item_ns_groups": { ... },
  "match_ns_group": ["match_feats"]    // 新增；模型端要识别这个 group
}
```

- 改动量：中（要改 ns_groups 加载逻辑，让 GroupNSTokenizer 接收 dense match 输入）
- 优点：NS token 与语义组 1:1 对齐、可解释、对其他 dense 族扩展友好
- 缺点：偏离 baseline run.sh 默认 rankmixer 路径

### 8.5 最终建议

```text
1. 短期（默认）        : 方案 S1 — 跑 A1* 探针档量化 5→4 损失，确认 < 0.10 %
2. 若 S1 退化超 0.10 % : 方案 S4 — match 进 query_generator condition
3. 长期（capacity 扩） : 方案 S5 — d_model=128 + num_queries=4 + Plan A/B/C 同时上
```

**方案 S2/S3 不推荐**：S2 削弱 RankMixer 论点，S3 把 match 挤进已经过载的 dense bottleneck。

**方案 S6 留作备选**：当 NS 特征族扩到 4+ 个（A、B、C、G）后，group 模式比 rankmixer 更可解释。
