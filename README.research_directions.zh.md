# PCVRHyFormer 提点研究方向（细粒度工程版）

本文基于对 [model.py](model.py) / [dataset.py](dataset.py) / [trainer.py](trainer.py) / [train.py](train.py) / [run.sh](run.sh) 的源码审计，针对七个提点方向给出**可工程落地**的改造方案。每条方向都包含：

1. **当前状态**（精确到文件行号的代码定位）
2. **问题诊断**（why it can be improved）
3. **改造方案**（含可直接 patch 的代码片段）
4. **实验配置**（命令行 / 超参建议）
5. **预期收益** + **主要风险**（带颜色标识）

---

## 颜色标识图例

> 收益和风险等级仅用于辅助判断优先级。所有数值预测都来自对类似 RecSys / CVR 工作的迁移经验，必须以本数据集的 ablation 为准。

### 预期收益 (Expected gain)

| 标识 | 含义 | 预期 AUC 提升 |
|---|---|---|
| <span style="color:#16a34a; font-weight:bold;">🟢 高收益</span> | 大概率提点 | **+0.3 % 及以上** |
| <span style="color:#ca8a04; font-weight:bold;">🟡 中收益</span> | 多数情况能提点 | **+0.1 % ~ +0.3 %** |
| <span style="color:#2563eb; font-weight:bold;">🔵 探索性</span> | 收益不稳定，需多次实验 | **0 ~ +0.1 % 或不确定** |

### 风险等级 (Risk)

| 标识 | 含义 |
|---|---|
| <span style="color:#dc2626; font-weight:bold;">🔴 高风险</span> | 容易掉点 / OOM / 训练不收敛 |
| <span style="color:#ea580c; font-weight:bold;">🟠 中风险</span> | 需要小心调参，否则可能负向 |
| <span style="color:#65a30d; font-weight:bold;">🟢 低风险</span> | 几乎无副作用，可以直接尝试 |

---

## 0. Baseline 关键事实速查

阅读后续方向之前请先确认以下 baseline 行为，避免混淆：

| 项目 | 当前实现 | 代码位置 |
|---|---|---|
| 序列编码器（默认） | `transformer`（带 RoPE 可选） | [model.py:544](model.py:544) `TransformerEncoder` |
| 4 个序列域共用同一 encoder | 是 | [model.py:878](model.py:878) `nn.ModuleList` 但 `seq_encoder_type` 全局 1 个 |
| `d_model` / `emb_dim` | 64 / 64 | [train.py:89-92](train.py:89) |
| `num_hyformer_blocks` / `num_heads` | 2 / 4 | [train.py:95-98](train.py:95) |
| `num_queries` × `num_sequences` + `num_ns` (T) | 2×4 + 8 = 16（默认 run.sh） | [model.py:1319](model.py:1319) |
| `RankMixerBlock` 模式 | `full`，要求 `d_model % T == 0` | [model.py:344](model.py:344) |
| 时间桶边界（64 条） | 5s ~ 1 年共 64 个边界 → 65 个桶 | [dataset.py:110-121](dataset.py:110) |
| Optimizer（dense / sparse） | AdamW (lr=1e-4) / Adagrad (lr=0.05) | [trainer.py:84-89](trainer.py:84) |
| **LR schedule** | **无**（常数学习率） | [trainer.py:289-376](trainer.py:289) |
| 缺失值处理（int） | `null / -1 / ≤ 0 → 0`（与 padding 不可分） | [dataset.py:529](dataset.py:529)、[dataset.py:475](dataset.py:475) |
| 缺失值处理（float） | `null → 0`（与有效 0 不可分） | [dataset.py:493](dataset.py:493) |
| `seq_max_lens` 默认 | `seq_a:256, seq_b:256, seq_c:512, seq_d:512` | [train.py:84-86](train.py:84) |
| `batch_size` 默认 | 256 | [train.py:55](train.py:55) |
| 高基数 ID dropout | 训练时 `Dropout(2 × dropout_rate)` | [model.py:1332](model.py:1332)、[model.py:1564](model.py:1564) |
| Final head 输入 | 仅 `Q tokens`，**不包含 NS tokens** | [model.py:1626-1632](model.py:1626) |

**单条样本含 4 个序列域，每域可能 256~1100 个事件；非序列含 60+ 个 int/dense 字段。**

---

## 1. 序列编码器选型（Sequence Encoder Selection）

### 1.1 当前状态

[model.py:811-842](model.py:811) 提供三种 encoder：

```python
def create_sequence_encoder(encoder_type, d_model, num_heads=4, ...):
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(...)            # 无 attention，只做 SwiGLU FFN
    elif encoder_type == 'transformer':
        return TransformerEncoder(...)       # 标准 Self-Attn + FFN（默认）
    elif encoder_type == 'longer':
        return LongerEncoder(...)            # top-k 压缩注意力
```

**关键约束**：[model.py:878-889](model.py:878) 中所有 4 个 domain 共用同一个 `seq_encoder_type`。这忽视了 4 个域行为差异极大的事实（参考 [README.feature_engineering.zh.md](README.feature_engineering.zh.md) §3.2~3.3）：

```text
domain_d: 平均长度 1100，72% 7 天内事件      → 长且新近
domain_c: 平均长度 449，仅 1.5% 7 天内事件   → 短且陈旧
domain_a/b: 中等长度，中等时间分布
```

### 1.2 问题诊断

- **统一 encoder 不公平**：domain_c 即使序列长 449，绝大多数信号都是月级以上的旧行为；domain_d 短期行为占比 30%，应保留更多最近 token。
- **`transformer` 在 L=512 下计算开销高**：单 block 单域的 self-attention 复杂度 O(L²·d_model) ≈ 0.5 M·64 = 32 M FLOPs/sample/block，4 个域 × 2 个 block = 8x，是当前训练吞吐瓶颈之一。
- **`longer` 仅在第一层做 cross-attention 压缩**：之后变成 self-attn over top-k，丢失 K-V 全序列信息。

### 1.3 改造方案

#### 方案 A：每个 domain 独立选择 encoder（最低成本，强烈推荐）

修改 [train.py:99-105](train.py:99) 的 `--seq_encoder_type` 接受逗号分隔的 per-domain 配置：

```python
parser.add_argument(
    '--seq_encoder_type', type=str, default='transformer',
    help='单一类型 OR per-domain: "seq_a:transformer,seq_b:transformer,seq_c:longer,seq_d:transformer"'
)
```

修改 [model.py:878-889](model.py:878) 的 `MultiSeqHyFormerBlock.__init__`，把 `seq_encoder_type` 从 `str` 扩展为 `Union[str, List[str]]`，对每个 sequence index 用对应类型创建 encoder。

推荐组合（基于 [README.feature_engineering.zh.md](README.feature_engineering.zh.md) §3.3 数据分析）：

```bash
--seq_encoder_type "seq_a:transformer,seq_b:transformer,seq_c:longer,seq_d:transformer"
--seq_top_k 64                # longer encoder 只为 c 用
--seq_max_lens "seq_a:256,seq_b:256,seq_c:256,seq_d:768"
```

#### 方案 B：引入 HSTU / Mamba 风格 encoder（探索向）

参考 Meta HSTU（Hierarchical Sequential Transduction Units）：用 SiLU(Q⊙K) 替代 softmax，复杂度仍是 O(L²)，但常数小、对长序列更稳定。在 [model.py](model.py) 加一个 `HSTUEncoder` 类。

#### 方案 C：层间异构（Inter-layer heterogeneous）

`num_hyformer_blocks=2` 时：第 1 层用 `longer`（强压缩），第 2 层用 `transformer`（精细融合）。需要在 [model.py:1391-1406](model.py:1391) 把 `seq_encoder_type` 也按层 index 传入。

### 1.4 实验配置

| 实验 | `seq_encoder_type` | `seq_max_lens` | 备注 |
|---|---|---|---|
| 1.A1 | 全部 `transformer`（baseline） | a:256,b:256,c:512,d:512 | 锚点 |
| 1.A2 | per-domain: a/b/d=transformer, c=longer | a:256,b:256,c:256,d:768 | 推荐 |
| 1.A3 | 全部 `longer`(top_k=64) | a:256,b:256,c:512,d:512 | 速度对照 |
| 1.A4 | a/b=swiglu, c/d=transformer | a:256,b:256,c:256,d:768 | 参数减半 |

### 1.5 预期收益与风险

- <span style="color:#16a34a; font-weight:bold;">🟢 高收益</span>：方案 A1 → A2 单 ablation，预计 **+0.10% ~ +0.25% AUC**，主要来自给 `domain_d` 更长上下文 + 释放 `domain_c` 算力预算。
- <span style="color:#ca8a04; font-weight:bold;">🟡 中收益</span>：方案 C 层间异构，**+0.05% ~ +0.15% AUC**，但调参成本明显增加。
- <span style="color:#2563eb; font-weight:bold;">🔵 探索性</span>：方案 B HSTU/Mamba，可能持平 transformer，胜在吞吐。
- <span style="color:#ea580c; font-weight:bold;">🟠 中风险</span>：把 `seq_d` 拉到 768 → 单 batch 显存涨 ~50%（在 d_model=64 时仍可控，但和方向 2 叠加要算总账）。
- <span style="color:#65a30d; font-weight:bold;">🟢 低风险</span>：per-domain encoder 改造完全向后兼容（旧的 `--seq_encoder_type transformer` 仍工作）。

---

## 2. 模型容量调整（Model Capacity）

### 2.1 当前状态

默认值集中在 [train.py:89-110](train.py:89)：

```python
d_model = 64       # backbone 维度
emb_dim = 64       # 单 embedding table 维度（投影前）
num_queries = 1    # CLI 默认；run.sh 覆盖为 2
num_hyformer_blocks = 2
num_heads = 4
hidden_mult = 4    # FFN inner dim = d_model * 4 = 256
```

非常多的参数集中在 embedding tables（高基数字段如 `domain_c_seq_47` 词表 287k → 287k×64 ≈ 18M 参数/字段），dense 部分相对较小。

### 2.2 问题诊断

- **d_model=64 偏小**：T=16 时每 token 仅 4 维 sub-channel，token-mixing 有效维度极低。
- **2 层 HyFormer block 信息融合次数不足**：每层只做一次 NS↔Q fusion（[model.py:967](model.py:967) `concat`），堆叠层数线性提升融合次数。
- **`d_model % T == 0` 限制**：[model.py:1320](model.py:1320) 严格校验。当 `d_model=128, T=16` ✓；当 `d_model=128, T=12` ✗（128 % 12 = 8）。规划容量时必须先算 T。
- **`num_heads` 与 `d_model` 强耦合**：head_dim = d_model / num_heads。d_model=64, heads=4 → head_dim=16，对 RoPE 不太友好（建议 ≥ 32）。

### 2.3 改造方案

#### 方案 A：Width 优先（最稳定提点路径）

```bash
# 64 → 128 → 192 → 256（每一档先跑收敛）
--d_model 128 --emb_dim 128 --num_heads 8 --hidden_mult 4
# T 必须能整除 d_model：当 num_queries=2, num_sequences=4, num_user_ns=5, num_item_ns=2, +1 user_dense → T=16
# 128 % 16 == 0 ✓
```

*显存预算*：emb_dim 翻倍 → embedding tables 体积 × 2，d_model 翻倍 → FFN 参数 × 4。**先用 [train.py:330-331](train.py:330) 的 `total_params` 日志校验**。

#### 方案 B：Depth 优先

```bash
--num_hyformer_blocks 3      # 默认 2 → 3
--num_hyformer_blocks 4      # 进一步
```

代价：每多 1 层多一次 4 域的 sequence encoder + 4 域 cross-attn + 1 个 mixer。建议在 width 已经升到 128 后再加 depth。

#### 方案 C：解耦 emb_dim 和 d_model

```bash
--d_model 128 --emb_dim 32   # 减小 embedding 内存，靠 projection 升维
```

权衡：embedding tables 节省 4×，但 4 个 domain 拼接后 (4 × num_sideinfo × emb_dim) 投影到 d_model 的 Linear 层会膨胀。

#### 方案 D：FFN 深度（hidden_mult）

```bash
--hidden_mult 6   # FFN inner dim 4× → 6×；与 SwiGLU 配合时常 ≥ 4×
```

[model.py:103-114](model.py:103) 的 `SwiGLU` 已支持。

### 2.4 容量与 T 关系备查表

```text
d_model | 可整除的 T 集合（T ≤ 32）
   64   | 1, 2, 4, 8, 16, 32
   96   | 1, 2, 3, 4, 6, 8, 12, 16, 24, 32
  128   | 1, 2, 4, 8, 16, 32
  192   | 1, 2, 3, 4, 6, 8, 12, 16, 24, 32
  256   | 1, 2, 4, 8, 16, 32
```

如果遇到 `--rank_mixer_mode full` 报错，可临时切到 `ffn_only` 跳过整除约束（见 [model.py:339-356](model.py:339)），但会损失 token mixing 收益。

### 2.5 实验配置

| 实验 | d_model | emb_dim | blocks | heads | hidden_mult | 备注 |
|---|---|---|---|---|---|---|
| 2.B0 | 64 | 64 | 2 | 4 | 4 | baseline |
| 2.B1 | 96 | 96 | 2 | 4 | 4 | mid-width，T=16 不可用，需 T=12 |
| 2.B2 | 128 | 128 | 2 | 8 | 4 | width 翻倍，主推荐 |
| 2.B3 | 128 | 128 | 3 | 8 | 4 | + depth |
| 2.B4 | 128 | 32 | 2 | 8 | 4 | 解耦 emb_dim |

### 2.6 预期收益与风险

- <span style="color:#16a34a; font-weight:bold;">🟢 高收益</span>：B0 → B2，预计 **+0.20% ~ +0.40% AUC**（CVR 任务普遍会从 64 → 128 受益）。
- <span style="color:#ca8a04; font-weight:bold;">🟡 中收益</span>：B2 → B3 加深，**+0.05% ~ +0.15% AUC**，边际下降明显。
- <span style="color:#2563eb; font-weight:bold;">🔵 探索性</span>：B4 解耦 emb_dim，可能持平 B2 但显存友好。
- <span style="color:#dc2626; font-weight:bold;">🔴 高风险</span>：直接 d_model=256 + emb_dim=256 + blocks=4 容易 OOM；emb_dim 翻倍意味着每个高基数 embedding 翻倍，最大几个 table 单独可达 30M+ 参数。
- <span style="color:#ea580c; font-weight:bold;">🟠 中风险</span>：扩容后 lr=1e-4 可能偏大，需要配合方向 5 的 warmup（深模型对学习率更敏感）。
- <span style="color:#65a30d; font-weight:bold;">🟢 低风险</span>：B1（width 适度增）几乎一定不掉点，最坏持平。

---

## 3. 时间特征（Time Features）

### 3.1 当前状态

[dataset.py:110-132](dataset.py:110) 定义了**全局共用**的 64 条边界：

```python
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, ..., 60,                         # 秒级（12 条）
    120, 180, ..., 3600,                        # 分钟级
    5400, 7200, ..., 86400,                     # 小时级
    172800, 259200, ..., 604800,                # 天级
    1123200, 1641600, 2160000, 2592000,         # 月级
    4320000, 6048000, 7776000,                  # 月级（粗粒度）
    11664000, 15552000,                         # 半年级
    31536000,                                   # 1 年
])
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1   # = 65
```

[dataset.py:632-665](dataset.py:632) 把 `current_ts - event_ts` 用 `searchsorted` 离散化为 [1, 65] 的 bucket id（0 留给 padding）。模型端 [model.py:1378](model.py:1378) 用 `nn.Embedding(65, d_model, padding_idx=0)` 加到序列 token。

### 3.2 问题诊断

- **bucket 在不同 domain 上分布严重不均**。结合 [README.feature_engineering.zh.md](README.feature_engineering.zh.md) §3.2：
  - `domain_c` 中位 275 天 → 大部分事件落在最后 5~6 个 bucket，前 50 个 bucket 几乎全是 padding。
  - `domain_d` 中位 12.5 天，30% 在 7 天内 → 集中在 bucket 40~55，最后几个 bucket 几乎不用。
- **64 个 bucket 全部参与 embedding（~64×d_model）**，但 effective rank 远小于 64 → embedding 容量浪费。
- **bucket 是离散的，丢失同 bucket 内细节**：例如 1 小时和 2 小时同属一个 bucket（`120` 和 `180` 这两个边界），但行为意义可能差很多。
- **没有把 timestamp 本身作为特征传给 query generator**，只传给序列 token。

### 3.3 改造方案

#### 方案 A：Per-domain 时间桶边界

把 `BUCKET_BOUNDARIES` 改成 `dict`：

```python
# dataset.py
BUCKET_BOUNDARIES = {
    'seq_a': np.array([300, 1800, 3600, 14400, 86400, 604800, ...], dtype=np.int64),
    'seq_b': np.array([300, 1800, 3600, 14400, 86400, 604800, ...], dtype=np.int64),
    'seq_c': np.array([86400, 604800, 2592000, 7776000, 15552000, ...], dtype=np.int64),  # 粗粒度
    'seq_d': np.array([60, 300, 1800, 3600, 14400, 86400, 604800, ...], dtype=np.int64),  # 细粒度
}
NUM_TIME_BUCKETS = max(len(v) for v in BUCKET_BOUNDARIES.values()) + 1
```

[dataset.py:659-665](dataset.py:659) 处的 `searchsorted` 改成 `BUCKET_BOUNDARIES[domain]`。

#### 方案 B：连续时间 + 离散桶联合编码

在 [model.py:1571-1574](model.py:1571) 加 `log(time_diff + 1)` 作为额外 dense 通道：

```python
# model.py: _embed_seq_domain
log_dt = torch.log1p(time_diff_seconds.float()).unsqueeze(-1)  # (B, L, 1)
log_dt_proj = self.log_dt_proj(log_dt)                          # (B, L, d_model)
token_emb = token_emb + self.time_embedding(time_bucket_ids) + log_dt_proj
```

#### 方案 C：Sinusoidal Time Encoding

参考 Time2Vec / SASRec：

```python
# 将 dt 编码到多个频率
freqs = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
sin_emb = torch.sin(dt[..., None] * freqs)
cos_emb = torch.cos(dt[..., None] * freqs)
time_emb = torch.cat([sin_emb, cos_emb], dim=-1)  # (B, L, d_model)
```

放到 [model.py:117](model.py:117) 附近作为新 `TimeEmbedding` 模块，和 `RotaryEmbedding` 并列。

#### 方案 D：把 last-k recency 统计作为 NS token

为每个 domain 计算 `count_1h, count_1d, count_7d, count_30d, last_delta` 共 5 个标量，concat 成 dense token 加入 NS（参考 [README.feature_engineering.zh.md](README.feature_engineering.zh.md) §6.2 已建议）。改 [dataset.py:_convert_batch](dataset.py:505) 增加这些统计字段。

**这一方向与方向 4（dense 特征筛选）协同明显**。

#### 方案 E：Time-decayed cross-attention bias

在 [model.py:262-312](model.py:262) `CrossAttention.forward` 中给 attention scores 加 `-α·log(dt+1)`，让 query 偏好关注近期 token：

```python
# 在 attn 之前
recency_bias = -self.alpha * torch.log1p(dt.float())  # (B, L)
recency_bias = recency_bias.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
# 通过 attn_mask 通道传入 SDPA
```

### 3.4 实验配置

| 实验 | 改动 |
|---|---|
| 3.T0 | 全局 64 桶（baseline） |
| 3.T1 | Per-domain 桶（方案 A） |
| 3.T2 | T0 + log(dt) 连续通道（方案 B） |
| 3.T3 | T1 + recency NS token（方案 D） |
| 3.T4 | T1 + time-decay attention bias（方案 E） |

### 3.5 预期收益与风险

- <span style="color:#16a34a; font-weight:bold;">🟢 高收益</span>：方向 D（recency 统计 NS token）几乎是免费午餐，**+0.10% ~ +0.20% AUC**（demo 数据已经显示 `count_7d` 高分位有 5% 正例率 lift）。
- <span style="color:#ca8a04; font-weight:bold;">🟡 中收益</span>：方向 A per-domain bucket，**+0.05% ~ +0.15%**。需要在完整训练集做时间分布分析后再定边界。
- <span style="color:#ca8a04; font-weight:bold;">🟡 中收益</span>：方向 B/C 连续时间编码，**+0.05% ~ +0.10%**，与 RoPE 有重叠，提升空间有限。
- <span style="color:#2563eb; font-weight:bold;">🔵 探索性</span>：方向 E（time-decay attention bias），**0 ~ +0.10%**，调 α 比较敏感。
- <span style="color:#65a30d; font-weight:bold;">🟢 低风险</span>：A/D 完全是新增信息，几乎不会掉点。
- <span style="color:#ea580c; font-weight:bold;">🟠 中风险</span>：E 错误的 α 会导致 attention 完全偏向最近 1~5 个 token，丢失长期偏好；建议 α∈[0.01, 0.1] grid。

---

## 4. Dense 特征筛选（Dense Feature Filtering）

### 4.1 当前状态

[dataset.py:294-298](dataset.py:294) 加载 `user_dense`：约 10 个 fid，包括 `user_dense_feats_61/87`（embedding-like 大向量，比赛文档中提到的 SUM/LMF4Ads）和 `user_dense_feats_62-66/89-91`（与同号 int feature **逐元素对齐**）。`item_dense` 为空。

[model.py:1298-1316](model.py:1298) 的处理方式：

```python
self.user_dense_proj = nn.Sequential(
    nn.Linear(user_dense_dim, d_model),  # user_dense_dim 总和 ≈ 918
    nn.LayerNorm(d_model),
)
# forward
user_dense_tok = F.silu(self.user_dense_proj(user_dense_feats)).unsqueeze(1)  # (B, 1, D)
ns_parts.append(user_dense_tok)  # 1 个 dense token 进 NS
```

918 维 → 64 维一步压缩，并丢失 field 边界。

### 4.2 问题诊断

- **918 → 64 信息瓶颈过严**。`user_dense_feats_61` 和 `user_dense_feats_87` 本身就是高质量 embedding（SUM / LMF4Ads），它们的语义被强制混合后再压缩成 1 个 token。
- **没有利用 int-dense aligned 关系**。`user_int_feats_62-66/89-91` 的每个 ID 与 `user_dense_feats_62-66/89-91` 对应位置的 float 是同一对实体的 (id, weight)，当前完全分两路处理。
- **缺失 0 与有效 0 不可分**。[dataset.py:493-503](dataset.py:493) 直接 zero-pad，无 missing indicator，模型无法区分"没采集到"和"值确实是 0"。
- **没有归一化 / 分布对齐**。不同 dense 特征的尺度可能差几个数量级，未做 z-score / log / quantile 处理就直接 Linear，容易让数值大的字段主导 projection。

### 4.3 改造方案

#### 方案 A：Per-field projection（最容易上手）

[model.py:1298-1316](model.py:1298) 改成多 token：

```python
# 在 PCVRHyFormer.__init__
self.dense_field_protos = []  # [(name, slice, dim)]
offset = 0
for fid, dim in user_dense_cols:  # 来自 dataset.user_dense_schema
    self.dense_field_protos.append((fid, slice(offset, offset+dim), dim))
    offset += dim
self.dense_field_projs = nn.ModuleList([
    nn.Sequential(nn.Linear(dim, d_model), nn.LayerNorm(d_model))
    for (_, _, dim) in self.dense_field_protos
])

# forward
dense_tokens = []
for (_, sl, _), proj in zip(self.dense_field_protos, self.dense_field_projs):
    dense_tokens.append(F.silu(proj(user_dense_feats[:, sl])).unsqueeze(1))
user_dense_tok = torch.cat(dense_tokens, dim=1)  # (B, num_dense_fields, D)
```

注意 T 会变化，需要重新校验 `d_model % T`。

#### 方案 B：z-score 归一化 + 缺失 indicator

[dataset.py:483-503](dataset.py:483) 的 `_pad_varlen_float_column` 增加：

```python
# 同时返回 missing_mask
def _pad_varlen_float_column(self, arrow_col, max_dim, B):
    ...
    missing_mask = (raw_len == 0)  # (B,)
    # 或者 per-element: arrow_col.is_null
    return padded, missing_mask
```

模型端把 `missing_mask` 作为额外 channel 拼到 dense feature 中。z-score 用预训练阶段离线计算的 `mean / std` JSON（写到 schema 旁）。

#### 方案 C：低收益 dense 字段裁剪

通过 `permutation importance` 或在小子集 ablation 测每个 dense 字段单独的 lift；对 lift < 0.001 的字段直接 skip projection（节省 dense_proj 参数）。可以加一个 `--dense_field_keep_fids "61,87,62,..."` flag。

#### 方案 D：与 int feature 的 element-wise fusion（与 [README.zh.md](README.zh.md) §13.2 同方向）

对 `user_int_feats_62/dense_62`（list 长度对齐）：

```python
# 单条样本：list of (id, weight)
# id_emb = embedding(id)            # (L, emb_dim)
# weight_proj = Linear(1, emb_dim)(weight.unsqueeze(-1))  # (L, emb_dim)
# token = id_emb * weight_proj  + bias
# 然后 attention pool 或 sum 到 1 个 user-side token
```

这是改造最深的方案，但最贴合数据语义。

### 4.4 实验配置

| 实验 | 改动 |
|---|---|
| 4.D0 | Baseline（918 → 1 token） |
| 4.D1 | 方案 A：per-field token（约 8~10 个 dense token） |
| 4.D2 | D1 + z-score 归一化 |
| 4.D3 | D2 + missing indicator |
| 4.D4 | D3 + element-wise int-dense fusion（仅 62-66/89-91） |
| 4.D5 | D4 + 字段重要性裁剪（去掉 lift < 0.001 的 dense） |

### 4.5 预期收益与风险

- <span style="color:#16a34a; font-weight:bold;">🟢 高收益</span>：D0 → D1 单步，**+0.10% ~ +0.25% AUC**。`user_dense_feats_61/87` 是已经预训练的 embedding（强信号），不该被压缩进 1 个 token。
- <span style="color:#16a34a; font-weight:bold;">🟢 高收益</span>：D4 element-wise fusion，**+0.15% ~ +0.30%**，但工程改动量大。
- <span style="color:#ca8a04; font-weight:bold;">🟡 中收益</span>：D2 z-score，**+0.05% ~ +0.15%**。前提是先做离线统计。
- <span style="color:#2563eb; font-weight:bold;">🔵 探索性</span>：D5 裁剪，**0 ~ +0.05%**，主要节省训练时间。
- <span style="color:#dc2626; font-weight:bold;">🔴 高风险</span>：D1 改完后 num_user_ns 会变，T 也会变，可能触发 [model.py:1320-1326](model.py:1320) 的 `d_model % T != 0` 报错；上线前必须验证 T。
- <span style="color:#ea580c; font-weight:bold;">🟠 中风险</span>：D4 设计不当（如 missing 不区分），可能反而降低性能。
- <span style="color:#65a30d; font-weight:bold;">🟢 低风险</span>：D2 z-score 几乎无副作用。

---

## 5. LR Schedule

### 5.1 当前状态

**当前 baseline 没有任何 LR schedule**。[trainer.py:84-94](trainer.py:84) 创建优化器后直接进入训练循环，[trainer.py:289-376](trainer.py:289) 的 `train()` 中没有 `scheduler.step()`。

```python
self.sparse_optimizer = torch.optim.Adagrad(sparse_params, lr=0.05)
self.dense_optimizer = torch.optim.AdamW(dense_params, lr=1e-4, betas=(0.9, 0.98))
```

[trainer.py:355-376](trainer.py:355) 的 epoch 末"高基数 embedding 重置 + 重建 Adagrad" 是一种 cold-restart 形式，但只针对 embedding。

### 5.2 问题诊断

- **常数 LR 在 CTR/CVR 任务上次优**。这类任务 loss landscape 早期陡峭、后期平坦，warmup + decay 几乎是标配。
- **dense 与 sparse 不同步**：Adagrad 已自适应地缩小步长，AdamW 没有自适应衰减，可能在后期还在以原始 lr 摆动。
- **重建 Adagrad** 会丢掉历史 squared-gradient 累积，对 reinit 后的高基数 embedding 是合理的，但对未 reinit 的 embedding 通过 `old_state` 部分恢复（见 [trainer.py:357-376](trainer.py:357)）。这是一种有效 workaround，但与"全局 schedule"逻辑没有协同。

### 5.3 改造方案

#### 方案 A：Linear warmup + cosine decay（最常见 dense schedule）

```python
# trainer.py 新增
from torch.optim.lr_scheduler import LambdaLR

def make_warmup_cosine(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

# 在 __init__ 之后
self.dense_scheduler = make_warmup_cosine(
    self.dense_optimizer, warmup_steps=2000, total_steps=200000)

# 在 _train_step() 末尾
self.dense_scheduler.step()
```

不要给 Adagrad 加 cosine：Adagrad 自身的累积平方梯度已构成隐式衰减，再叠加 cosine 容易过早降到 0。

#### 方案 B：仅 warmup，不 decay

适合训练数据非常大但 epoch 数少（≤ 2 个 epoch）的情况：

```python
def make_warmup(optimizer, warmup_steps):
    return LambdaLR(optimizer, lambda step: min(1.0, step / warmup_steps))
```

#### 方案 C：Step decay on validation plateau

```python
self.dense_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    self.dense_optimizer, mode='max', factor=0.5, patience=2)

# 在每次 evaluate 之后
self.dense_scheduler.step(val_auc)
```

与 EarlyStopping 的 patience（默认 5）配合：scheduler patience=2，先 LR 减半再 early stop。

#### 方案 D：分离的 sparse / dense schedule

```python
--lr 3e-4              # 起始 dense lr 略大（配合 warmup）
--sparse_lr 0.05       # 保持
# warmup 2k 步 + cosine 到 0.1× → final dense lr ≈ 3e-5
```

#### 方案 E：Sparse Adagrad → SparseAdam

```python
# 替换：
self.sparse_optimizer = torch.optim.SparseAdam(sparse_params, lr=1e-3)
```

需要把所有 `nn.Embedding` 设为 `sparse=True`（[model.py:1013](model.py:1013) 等处）。SparseAdam 对长尾 embedding 更稳，但会改变现有 reinit 策略的语义。

### 5.4 落地代码骨架（patch 化）

```python
# train.py 新增 CLI
parser.add_argument('--lr_schedule', type=str, default='none',
                    choices=['none', 'warmup_cosine', 'warmup_only', 'plateau'])
parser.add_argument('--warmup_steps', type=int, default=2000)
parser.add_argument('--total_steps', type=int, default=0)  # 0 = 自动估算
parser.add_argument('--min_lr_ratio', type=float, default=0.1)

# trainer.py __init__ 末尾
self.dense_scheduler = self._build_scheduler(...)

# trainer.py _train_step 末尾
if self.dense_scheduler is not None and not isinstance(
        self.dense_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    self.dense_scheduler.step()

# trainer.py evaluate 后
if isinstance(self.dense_scheduler, ReduceLROnPlateau):
    self.dense_scheduler.step(val_auc)
```

### 5.5 实验配置

| 实验 | dense lr | schedule | warmup | 最终比例 |
|---|---|---|---|---|
| 5.S0 | 1e-4 | none | – | 1.0× |
| 5.S1 | 1e-4 | warmup_only | 2000 | 1.0× |
| 5.S2 | 3e-4 | warmup_cosine | 2000 | 0.1× |
| 5.S3 | 3e-4 | warmup_cosine | 5000 | 0.1× |
| 5.S4 | 1e-4 | plateau (factor 0.5) | – | 见 ReduceLROnPlateau |

### 5.6 预期收益与风险

- <span style="color:#16a34a; font-weight:bold;">🟢 高收益</span>：S0 → S2 单步，**+0.10% ~ +0.25% AUC**。warmup 在 dense 模型扩容（方向 2）后收益会更明显。
- <span style="color:#ca8a04; font-weight:bold;">🟡 中收益</span>：S1 仅 warmup，**+0.05% ~ +0.15%**。
- <span style="color:#2563eb; font-weight:bold;">🔵 探索性</span>：S4 plateau，依赖 patience，可能与 EarlyStopping 冲突。
- <span style="color:#dc2626; font-weight:bold;">🔴 高风险</span>：cosine 终值过低（< 0.05× 初始 LR）+ 数据未跑完 → 后期几乎不更新，掉点。建议 `min_lr_ratio ≥ 0.1`。
- <span style="color:#ea580c; font-weight:bold;">🟠 中风险</span>：训练数据是 IterableDataset，准确的 `total_steps` 不易估计，`num_rows / batch_size * epochs` 通常偏大；建议在 [train.py:330](train.py:330) 之后从 dataset 拿 `train_rows`。
- <span style="color:#65a30d; font-weight:bold;">🟢 低风险</span>：S1（仅 warmup）无副作用，强烈建议先开。

---

## 6. 缺失值处理（Missing Value Handling）

### 6.1 当前状态

**Int 系列**（[dataset.py:475](dataset.py:475)、[dataset.py:529-530](dataset.py:529)、[dataset.py:550-551](dataset.py:550)、[dataset.py:614-615](dataset.py:614)）：

```python
# null → 0 (via fill_null), -1 → 0 (via arr<=0)
arr = col.fill_null(0).to_numpy(...).astype(np.int64)
arr[arr <= 0] = 0
# OOB → 0 (via _record_oob with clip_vocab=True)
```

**Float 系列**（[dataset.py:493-503](dataset.py:493)）：

```python
padded = np.zeros((B, max_dim), dtype=np.float32)
# null arrays → 全 0 padding；空数组同样为 0
```

**Embedding**（[model.py:1013](model.py:1013)、[model.py:1112](model.py:1112)）：`padding_idx=0` → ID 0 永远输出 0 向量。

### 6.2 问题诊断

- **Padding / Missing / Real-zero 混淆**。`user_int_feats_15` 这类 list feature，0 既可能是"真实 ID 0"（如果存在）、也可能是"空槽 padding"、也可能是"原始 -1 缺失"。三类语义同等对待 → 模型需要从其他维度反推差异。
- **Dense 缺失 = 0**。比如 `user_dense_feats_87`（LMF4Ads），值确实可能是 -0.001、0、+0.002，但 missing 也被填为 0，无法区分。
- **OOB 静默裁剪**。[dataset.py:415-416](dataset.py:415) `clip_vocab=True` 时把 OOB → 0，错误的 schema 会被悄悄吞掉，仅在 `_oob_stats` 记录数量。
- **没有 missing 指示位**。模型无法显式利用"该字段缺失"这一信号本身（在 RecSys 里"缺失"经常对应低活用户、新设备等强信号）。

### 6.3 改造方案

#### 方案 A：保留 padding=0，把 missing 映射到 vocab+1

```python
# dataset.py: _pad_varlen_int_column
# 不再把 -1 → 0，而是把 -1 → vocab_size + 1 作为专用 missing slot
padded[padded == -1] = vocab_size + 1  # 需要传 vocab_size 进来
# scalar 同理：null → vocab_size + 1
```

[model.py](model.py) 创建 embedding 时多分配 1 个 slot：

```python
embs.append(nn.Embedding(int(vs) + 2, emb_dim, padding_idx=0))
# slot 0: padding
# slot 1..vs: 真实 id
# slot vs+1: missing
```

#### 方案 B：Dense 增加 missing-mask channel

```python
# dataset.py: _pad_varlen_float_column 改造
def _pad_varlen_float_column(self, arrow_col, max_dim, B):
    padded = np.zeros((B, max_dim), dtype=np.float32)
    missing = np.ones((B, max_dim), dtype=np.float32)  # 1 = missing
    for i in range(B):
        s, e = ...
        if e > s:
            padded[i, :e-s] = ...
            missing[i, :e-s] = 0
    return padded, missing
```

模型端：

```python
# user_dense_feats: (B, total_dim)，对应 missing_mask: (B, total_dim)
input = torch.cat([user_dense_feats, missing_mask], dim=-1)  # (B, 2*total_dim)
# user_dense_proj.in_features 翻倍
```

#### 方案 C：均值 / 中位数填充（offline 统计）

在 [README.feature_engineering.zh.md](README.feature_engineering.zh.md) §8 推荐的"完整训练集统计"基础上，dump 每个 dense 字段的 median，写入 schema：

```python
# schema.json 增加
{ "user_dense": [[fid, dim, median]], ... }

# dataset.py: _pad_varlen_float_column
padded[i, :use_len] = values[start:start+use_len]
# 缺失位置填 median
padded[missing_mask == 1] = median[fid]
```

#### 方案 D：Learnable missing token

针对 list-int 特征，对整个 list 都缺失（`raw_len=0`）的样本，生成一个特殊 NS token：

```python
# model.py 新增
self.missing_token = nn.Parameter(torch.zeros(1, 1, d_model))
# 在 forward 中检测 list_len==0 的样本，把对应 token 替换成 missing_token
```

#### 方案 E：把 OOB 单独建模

[dataset.py:388-421](dataset.py:388) 的 OOB 当前也被映射到 0，可以改成映射到 vocab+2：

```text
slot 0:    padding
slot 1..v: 正常
slot v+1:  missing (raw -1 / null)
slot v+2:  oob
```

### 6.4 实验配置

| 实验 | 改动 |
|---|---|
| 6.M0 | Baseline |
| 6.M1 | int 区分 missing slot（方案 A） |
| 6.M2 | dense missing-mask channel（方案 B） |
| 6.M3 | M1 + M2 |
| 6.M4 | M3 + 中位数填充（方案 C） |
| 6.M5 | M4 + OOB 独立 slot（方案 E） |

### 6.5 预期收益与风险

- <span style="color:#ca8a04; font-weight:bold;">🟡 中收益</span>：M0 → M1，**+0.05% ~ +0.15% AUC**。当某些字段 missing 比例高（参考 `domain_a_seq_38` 仅 5.5% 覆盖），missing 语义本身就是强信号。
- <span style="color:#ca8a04; font-weight:bold;">🟡 中收益</span>：M1 → M2，**+0.05% ~ +0.10%**。对 dense 大向量（user_dense_feats_61/87）missing 不常见，主要在 aligned 系列。
- <span style="color:#2563eb; font-weight:bold;">🔵 探索性</span>：M4 中位数填充，**0 ~ +0.05%**。需要离线统计且和 z-score 协同。
- <span style="color:#dc2626; font-weight:bold;">🔴 高风险</span>：方案 A 修改 vocab_size 后，所有 [model.py](model.py) 的 `nn.Embedding(int(vs) + 1, ...)` 需同步改为 `+2` 或 `+3`。**若漏改一处会触发 IndexError**，必须全局 grep `+ 1, emb_dim` 检查。
- <span style="color:#ea580c; font-weight:bold;">🟠 中风险</span>：dense_proj 维度翻倍（方案 B），与方向 4 改造叠加时要重新校验显存。
- <span style="color:#65a30d; font-weight:bold;">🟢 低风险</span>：方案 E 单独 slot 给 OOB，可观察 OOB 率是否高（[dataset.py:423-443](dataset.py:423) `dump_oob_stats`）后再决定。

---

## 7. Batch Size 调优

### 7.1 当前状态

[train.py:55](train.py:55)、[run.sh:6-14](run.sh:6) 的默认 `batch_size=256`。[dataset.py:216-229](dataset.py:216) 预分配 buffer 是 `[B, ...]`，B 即 batch_size。**buffer 是单 worker 内的，每个 DataLoader worker 都有一份**，[dataset.py:108](dataset.py:108) 用 `file_system` sharing strategy 防止 `/dev/shm` 爆。

[trainer.py:402-428](trainer.py:402) 的 `_train_step` 内：

```python
torch.nn.utils.clip_grad_norm_(..., max_norm=1.0, foreach=False)  # 注意 foreach=False
```

注释提到这是规避 PyTorch `_foreach_norm` CUDA bug。在大 batch 下 `foreach=True` 会更快但有风险。

### 7.2 问题诊断

- **batch_size=256 偏小**。高基数 embedding 占主导 → sparse 梯度信号噪声大；小 batch 让 Adagrad 累积变粗糙。
- **每个 worker 一份 256-buffer**：16 worker → 16×256 = 4096 行常驻内存，纯 numpy buffer 小（≈ 50 MB），不构成瓶颈。但 GPU 显存与 batch 强相关：
  - 当前最大序列 512（domain_d）+ d_model=64 → 每样本主要内存 = 4 × 512 × 64 × 4 bytes ≈ 0.5 MB seq tokens
  - batch=256 → 128 MB seq tokens（forward），加上 attention 中间 4 × 512² × 4 / batch ≈ ~1 MB/sample 共 256 MB
  - **batch=512 不会爆**，bs=1024 在 d_model=64 仍可
- **clip_grad_norm `foreach=False`**：如果模型扩容到 d_model=128，需要重新评估这个 workaround。
- **sparse Adagrad 的有效 lr 与 batch 解耦**：Adagrad lr=0.05 是基于 batch=256 调的，bs 翻倍后梯度尺度变化不大（per-sample 平均梯度），但累积速度变化 → 实际更新步长会变。

### 7.3 改造方案

#### 方案 A：直接 scaling

```bash
--batch_size 512 --lr 1.5e-4 --sparse_lr 0.05
--batch_size 1024 --lr 2e-4 --sparse_lr 0.05
--batch_size 2048 --lr 3e-4 --sparse_lr 0.06
```

经验法则（square-root scaling）：dense lr 按 √(BS_new/BS_base) 缩放，sparse Adagrad 不需要严格按比例（自适应）。

#### 方案 B：Gradient accumulation

不增加显存的等效大 batch。在 [trainer.py:402-428](trainer.py:402) 加 `accum_steps`：

```python
def _train_step(self, batch):
    ...
    loss = ... / self.accum_steps
    loss.backward()
    if (self.step % self.accum_steps == 0):
        torch.nn.utils.clip_grad_norm_(..., 1.0, foreach=False)
        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()
        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()
```

**但**：当前 [trainer.py:407-409](trainer.py:407) 在 step 开头 `zero_grad()`，必须改成只在累积完才清零。

#### 方案 C：动态 batch（按 domain 长度自适应）

如果 batch 内某些样本序列短得多，可以打包更大的 batch。但当前数据集 [dataset.py:226-228](dataset.py:226) 已经 padding 到固定 `max_len`，工程改动量大，不优先。

#### 方案 D：Mixed precision（与 batch size 解耦）

在 [trainer.py:289-376](trainer.py:289) 的 `train()` 包装 autocast：

```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    logits = self.model(model_input)
    loss = ...
loss.backward()
# 可不用 GradScaler（bf16）
```

bf16 减半显存 → 直接允许 batch=512 / 1024。但需要确认所有 LayerNorm / softmax 在 bf16 下数值稳定（[model.py:226-233](model.py:226) `nan_to_num` 已有保护，但仍建议监控 NaN 率）。

### 7.4 实验配置

| 实验 | batch_size | dense lr | accum_steps | precision |
|---|---|---|---|---|
| 7.B0 | 256 | 1e-4 | 1 | fp32 |
| 7.B1 | 512 | 1.5e-4 | 1 | fp32 |
| 7.B2 | 1024 | 2e-4 | 1 | fp32 |
| 7.B3 | 1024 | 2e-4 | 1 | bf16 |
| 7.B4 | 256 | 1e-4 | 4（等效 1024） | fp32 |
| 7.B5 | 4096 | 4e-4 | 4（等效 16384） | bf16 |

### 7.5 与其它方向的耦合

- **方向 5 LR schedule** 是必要前置：大 batch 必须配 warmup，否则前几百 step 梯度噪声小但 lr 已经满负载，容易直接发散。
- **方向 2 模型扩容**：d_model=128 + bs=1024 + fp32 大概率 OOM（中型 GPU）。先 bf16 + bs=512 起步。
- **方向 6 missing-mask** 会让 dense_proj 输入翻倍，进一步压榨显存。

### 7.6 预期收益与风险

- <span style="color:#ca8a04; font-weight:bold;">🟡 中收益</span>：B0 → B2（bs 256 → 1024），**+0.05% ~ +0.15% AUC**，但只有在 **同时配 warmup** 时才大概率提点；不配 warmup 多数情况下持平。
- <span style="color:#16a34a; font-weight:bold;">🟢 高收益</span>：B3 bf16 + 大 batch，主要收益是**训练吞吐 × 2~3**，让你能在同样时间内跑更多 ablation。
- <span style="color:#2563eb; font-weight:bold;">🔵 探索性</span>：B5 超大 batch（16k），多数 CTR 任务在这个量级会饱和甚至掉点。
- <span style="color:#dc2626; font-weight:bold;">🔴 高风险</span>：bs 提到 1024+ 但不配 LR warmup → 前 100~500 step 梯度爆炸 / NaN。日志里关注 [trainer.py:459-466](trainer.py:459) 的 `predictions are NaN` warning。
- <span style="color:#dc2626; font-weight:bold;">🔴 高风险</span>：bf16 下 LayerNorm 可能下溢；务必 keep `RoPEMultiheadAttention.W_g` 在 fp32 输出（[model.py:147-148](model.py:147) 初始化为 1.0，bf16 下会有截断）。
- <span style="color:#ea580c; font-weight:bold;">🟠 中风险</span>：sparse Adagrad 在大 batch 下累积平方梯度过快 → 后期实际 LR 严重衰减；可以同步把 `sparse_lr` 从 0.05 提到 0.08 ~ 0.1 验证。
- <span style="color:#65a30d; font-weight:bold;">🟢 低风险</span>：B1（bs=512）几乎是免费的，可以直接尝试。

---

## 8. 推荐实验顺序与组合

下面给出建议的实验顺序，每一档都是"改动小、可独立 ablation、容易回滚"的状态：

```text
P0 (锚点):   现状 baseline，记录 AUC + logloss + 训练时长
P1 (低风险): + S1 warmup-only LR schedule          [方向5]
P2:          P1 + B1 batch_size=512                [方向7]
P3:          P2 + T3 per-domain time bucket + recency NS token  [方向3]
P4:          P3 + D1 dense per-field projection    [方向4]
P5:          P4 + 1.A2 per-domain encoder + seq_d=768  [方向1]
P6:          P5 + B2 width 96 或 128 + S2 cosine   [方向2 + 方向5]
P7:          P6 + M1+M3 missing slot               [方向6]
P8:          P7 + D4 element-wise int-dense fusion  [方向4 深改]
```

**P0 → P5** 都是低工程风险的改造，预计累计 **+0.30% ~ +0.80% AUC**。
**P6 → P8** 是较大改造，需要更多算力 / 显存预算，但天花板更高，预计再 **+0.20% ~ +0.40%**。

每个阶段完成后需保留：
- `train.log` + tensorboard events
- `train_config.json`（[trainer.py:165-168](trainer.py:165) 已自动写入）
- 最佳 checkpoint 的 `best_val_AUC` / `best_val_logloss`
- 训练吞吐（steps/sec）和 GPU 显存峰值

---

## 9. 工程落地清单（速查）

每条改造都要回答这 5 个问题：

1. **改了哪些文件 / 函数？**（精确到行号）
2. **`d_model % T == 0` 是否仍成立？**（[model.py:1320](model.py:1320)）
3. **`schema.json` 是否需要更新？**（[dataset.py:208-330](dataset.py:208)）
4. **inference 路径是否同步改了？**（model 端的 forward / predict 双入口，[model.py:1634-1714](model.py:1634)）
5. **checkpoint 是否仍能 load？**（增加新参数会让旧 ckpt 的 `state_dict()` load 失败，建议 `strict=False` + 手动迁移）

---

## 10. 额外提醒

- **[ns_groups.json](ns_groups.json) 默认仅作示例**。在 run.sh 默认 `--ns_groups_json ""` 路径下，每个 feature 是 singleton group。如果切到 `group` 模式或重新设计 grouping（参考 [README.feature_engineering.zh.md](README.feature_engineering.zh.md) §6.5），整个改动是独立维度，与本文 7 个方向并行。
- **`emb_skip_threshold` 在 [run.sh:12](run.sh:12) 设为 1_000_000**，意味着默认所有 vocab ≤ 1e6 的字段都会建 embedding。结合 demo 数据 [README.feature_engineering.zh.md](README.feature_engineering.zh.md) §4 的高基数分析（最大 287k），所有字段都建了 embedding。如果在完整训练集上某些字段实际词表 > 1e6，则会被静默 skip，建议训练前 dump `_oob_stats` 检查。
- **Final head 没用 NS tokens** 是 [README.zh.md](README.zh.md) §13.1 已经强调的可优先改造点，与本文 7 个方向独立但收益叠加。建议在 P3 阶段一起做。
- **梯度爆炸保护**：[model.py:233](model.py:233) 的 `nan_to_num`、[trainer.py:422](trainer.py:422) 的 `clip_grad_norm_(max_norm=1.0)`、[trainer.py:459-466](trainer.py:459) 的 NaN filter 三层兜底已经存在。但当模型扩容 + 大 batch + 未配 schedule 时仍可能 NaN，建议把 [trainer.py:312](trainer.py:312) 的 `loss` 同步打到 tensorboard 关注异常 spike。
