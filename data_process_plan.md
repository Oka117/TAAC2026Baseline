# 数据预处理方案与实验设计（修订版）

> 修订时间：2026-05-10
> 设计意图：以「user list 的真实信号是 variance（不是 mean）」这一观察为中心，把 docx 的 7 步预处理落到 baseline 的流式 Parquet 通路上，**保留 domain 序列**让 HyFormer 仍能消费 sequence；同时规划 GNN ↔ HyFormer 融合路径。

---

## 0. 今日进展

### 0.1 完成项

- [x] **数据特征分析**（demo 1k）：标签分布、正负样本对比、列分类。
- [x] **GNN baseline 跑通**（[gnn.ipynb](gnn.ipynb)）：USER INT 29 + USER DENSE 21 / ITEM SCALAR 10 + ITEM SEQ 1 → AUC **0.7899**（demo 1k，20 epoch，明显过拟合：train AUC 1.0、test AUC 0.79）。
- [x] **核心数据洞察**：`user_int_feats_60/89/90/91` 在 demo 上每用户 list 的 mean **是常数** → 推论：信号在 variance 不在 mean → 与 docx §5.1 「这 4 列用 variance」的选择**方向一致**。

### 0.2 关键洞察的实证验证（demo 1k 上）

| 列 | per-user mean 分布 | per-user variance 分布 | 实际 list 内容 |
| --- | --- | --- | --- |
| `user_int_feats_15` | mean=531, std=205 | mean=66326, std=57575 | **mean/var 都变化** |
| `user_int_feats_60` | **mean=2, std=0** | **mean=0, std=0** | 全是 `[2]` / `[2,2]` 等常数序列 |
| `user_int_feats_62` | mean=5.9, std=1.5 | mean=4.05, std=3.3 | mean/var 都变化 |
| `user_int_feats_63` | mean=28.9, std=8.5 | mean=128, std=116 | mean/var 都变化 |
| `user_int_feats_64` | mean=22.7, std=9.0 | mean=150, std=121 | mean/var 都变化 |
| `user_int_feats_65` | mean=213, std=61 | mean=7989, std=6124 | mean/var 都变化 |
| `user_int_feats_66` | mean=776, std=226 | mean=121727, std=84405 | mean/var 都变化 |
| `user_int_feats_80` | mean=7.5, std=3.5 | mean=2.3, std=4.1 | mean/var 都变化 |
| `user_int_feats_89` | **mean=4.6, std=0** | **mean=10.84, std=0** | **所有用户共享同一 10 元素 list** |
| `user_int_feats_90` | **mean=4.1, std=0** | **mean=7.69, std=0** | 同上，共享 list |
| `user_int_feats_91` | **mean=4.7, std=0** | **mean=7.21, std=0** | 同上，共享 list |

**结论**：

1. demo 1k 上，**60/89/90/91 不只是 mean 常数，连 variance 都常数**——这是 demo 数据的简化（共享 list 或纯标量退化），**不能用来证伪 variance 假设**。
2. 你的洞察「mean 在用户间不变 → 用 variance」本质上是**对真实数据生成过程的猜想**：这些列可能是「定均值的归一化分布」，比如频率向量 `[p_i]` 总和守恒 → mean = 总和/长度 = 常数；variance = 分布形状的差异。
3. **必须在 ≥1M 行真实数据上重新审计**才能确认 variance 是否真的有信号；demo 不够。
4. 其余 7 列（15/62-66/80）demo 上 mean 和 var 都变化——**两者都可能有信号**，docx 选 mean 是经验默认，可消融。

### 0.3 GNN AUC 0.79 的解读

- demo 上 train AUC 1.0、test AUC 0.79 → **严重过拟合**（1k 行训练 GNN 必然如此）。
- 4 个 variance 列在 demo 上是常数 → embedding 层只学到一个固定向量，对 GNN 的 0.79 AUC **没有贡献**；信号来自其他 25+ 个列。
- 全量上能否到 0.79：完全不可外推。但作为 pipeline 跑通的证明已经达到目标。

---

## 1. 修订后的预处理流程

```
RAW DATA (200M 行 / 120 列)
    │
    ├─ ID & Label & timestamp（5 列） ─────────────────────┐
    ├─ User Int  Feats（46 列：35 标量 + 11 list）         │
    ├─ User Dense Feats（10 列：全 list）                   │
    ├─ Item Int  Feats（14 列：13 标量 + 1 list）          │  ※ 与原 docx 区别
    ├─ Item Dense Feats（0 列）                             │  domain 列不再删除
    └─ Domain Sequence Feats（45 列：seq_a/b/c/d）        ─┤
                                                            │
    ↓                                                       │
① 删除 >80% missing 列  ←— 离线 audit 决定具体列            │
    ↓                                                       │
② 标量数值列 fillna(0) + 非空 +1                            │
    ↓                                                       │
③ User Int list 列折叠                                      │
    ├── 60/89/90/91 → variance → user_new_feats           │
    └── 其他 7 列   → mean    → user_new_feats           │
    ↓                                                       │
④ Item Int list 列保留为 token 序列                        │
    ├── padding 到 max_len                                  │
    └── 准备 token-level embedding（baseline 已支持）      │
    ↓                                                       │
⑤ Dense / new 列归一化                                      │
    ├── scalar → StandardScaler                            │
    └── list   → padding → 位置归一化 → restore 原长       │
    ↓                                                       │
Domain seq 序列原样保留 ──────────────────────────────────┘
    ↓
PREPROCESSED BATCH（DataLoader 输出）
    ├── user_int_feats:  (B, total_user_int_dim)
    ├── user_dense_feats:(B, total_user_dense_dim)  ← 含归一化后的 user_new_feats
    ├── item_int_feats:  (B, total_item_int_dim)    ← 含 item_int_feats_11 序列
    ├── seq_data:   {seq_a/b/c/d: (B, S, L)}        ← 原样保留，HyFormer 消费
    ├── seq_lens:   {...}
    ├── seq_time_buckets: {...}
    └── label:      (B,)  = (label_type == 2)
```

**与原 docx 流程的关键差异**：

| | 原 docx | 修订流程 |
| --- | --- | --- |
| 删 `domain_*` 列 | ✅ 全部删除（45 列） | ❌ **保留**——HyFormer 需要 sequence；GNN 可单独忽略 |
| user int list 折叠 | mean/var | mean/var（与 docx 一致） |
| item int list | flatten | **保留为序列**（baseline `_buf_seq` 已实现，不需要 flatten） |
| dense 归一化 | StandardScaler | StandardScaler（与 docx 一致） |

---

## 2. 各步骤详解（新增「实证依据」「与 baseline 的差距」「落地代码点」三栏）

### 2.1 ① 删除 > 80% missing 列

**docx §3 设计意图**：高缺失率列噪声大、信号少。

**实证依据**：demo 上识别出 9 列（`user_int_feats_{99-103,109}` + `item_int_feats_{83-85}`）；**全量上必须重审**——demo 1k 不能代表 200M 分布。

**与 baseline 的差距**：baseline 现在不删列；缺失率高的列若 schema 标 `vs=0` 会被强制写 0（[dataset.py:531-533](dataset.py:531)），等价于退化为常量列，信息为零但仍占用 buffer 空间。

**落地代码点**：
- 新增 [tools/audit_schema.py](tools/audit_schema.py)：在 ≥1M 行子集上扫每列 null 率、unique 数、max_id、（list 列的）per-user mean/var 是否常数。输出 `schema_audit.json`。
- [dataset.py:_load_schema](dataset.py:272) 接受 `--audit_path`，把命中 high-missing 的 fid 从 `_user_int_cols / _user_dense_cols / _item_int_cols` 中物理移除。

---

### 2.2 ② 标量列 fillna(0) + 非空 +1

**docx §4.1 设计意图**：把 0 留作 padding/unknown，真实值从 1 开始，避免与 padding 冲突。

**与 baseline 的差距**：
- baseline 现在 [dataset.py:529](dataset.py:529) 把 null/-1/≤0 全部映射到 0，valid 0 与 padding 0 **不可区分**；
- baseline 在 [model.py:1013](model.py:1013) 用 `nn.Embedding(vs+1, padding_idx=0)` 在 embedding 层保护 padding 槽，但**数据侧的 valid 0 仍然落到 padding 槽**——这是潜在 bug。

**落地代码点**：
```python
# dataset.py._convert_batch  user_int / item_int 标量分支
if dim == 1:
    raw = col.to_numpy(zero_copy_only=False)
    null_mask = pd.isnull(raw) | (raw < 0)  # 注意：0 不再被当作 missing
    arr = np.where(null_mask, 0, raw + 1).astype(np.int64)
    # vs 在 schema 中已经是 max_id + 2（audit_schema.py 输出时 +1）
    if vs > 0:
        self._record_oob('user_int', ci, arr, vs)
```

`schema.json` 中所有 fid 的 `vs` 同步 +1（在 audit_schema.py 里做）。

**预测**：valid 0 占比小的列上几乎无差别；valid 0 占比大的列（如 0/1 binary 特征）上预期 +0.05% AUC。

---

### 2.3 ③ user list 折叠 ⭐核心⭐

**docx §5 设计意图**：
- {60, 89, 90, 91} → variance（behavioral diversity）
- 其他 7 列 → mean（summarized stats）

**实证依据（今日发现）**：
- demo 上 60/89/90/91 的 mean 是常数 → 与 docx 选 variance 的方向一致；
- demo 上 89/90/91 的 list 本身在所有用户间**完全相同** → 这是 demo simplification，**全量上需重新验证**；
- 其余 7 列 demo 上 mean 和 var 都有变化 → 用 mean 还是 var 是经验默认，可消融。

**与 baseline 的差距**：baseline 把这 11 列当 multi-hot：list embedding → mean-pool 到 `(B, emb_dim)`（[model.py:1059-1063](model.py:1059)）。**等价于「mean of embedded IDs」**——和 docx 的 mean(values) 不是同一回事：
- baseline mean-pool：每个 ID 经过 `nn.Embedding` 学到一个 64 维向量，再平均，**保留 ID 的语义**；
- docx mean：把 list `[1, 5, 9]` 当成数值序列，直接 `(1+5+9)/3 = 5`，**把 ID 当成有序数值用**。

如果这些列是 categorical（如类别 ID），docx 的 mean 是类型错误；如果是「频率向量 / 概率分布 / 计数」，docx 的 mean/var 才有数值意义。

**今日洞察支持「这是分布特征」**：60/89/90/91 mean 常数恰好是「定均值分布」的特征——比如归一化频率向量 `[p_i]` 满足 `sum=1` → mean=1/L 是常数 → variance 才捕捉差异。

**落地代码点**：

新增 schema 字段 `user_list_fold`：
```json
{
  "user_list_fold": {
    "mean": [15, 62, 63, 64, 65, 66, 80],
    "var":  [60, 89, 90, 91]
  }
}
```

[dataset.py:_convert_batch](dataset.py:505) 增加分支：
```python
# 折叠后的列从 user_int 转移到 user_dense（标量 float）
for fid, mode in user_list_fold.items():
    col = batch.column(self._col_idx[f'user_int_feats_{fid}'])
    offsets = col.offsets.to_numpy()
    values = col.values.to_numpy().astype(np.float32)
    out = np.zeros(B, dtype=np.float32)
    for i in range(B):
        s, e = int(offsets[i]), int(offsets[i+1])
        if e <= s:
            continue
        seg = values[s:e]
        if mode == 'mean':
            out[i] = seg.mean()
        else:  # var
            out[i] = seg.var()
    user_new[:, new_offset[fid]] = out
```

`schema.json` 中：
- `user_int` 删除 11 个 list 列；
- `user_dense` 新增 11 个 `[fid, 1]` 标量列（命名上保留 `user_int_feats_<fid>` 不变，避免与 `user_dense_feats_<fid>` 冲突）。

**预测**：
- variance 列（60/89/90/91）在全量上**如果**确实 mean 常数 var 变化，相对 baseline mean-pool 会有提升（因为 mean-pool 等价于学一个常数嵌入，信号被压平）；预期 +0.05% ~ +0.20% AUC。
- mean 列（15/62-66/80）：mean(values) 比 mean-pool(embed) 信息量少，多数情况下会**轻微掉点**（-0.05% ~ 0%）。
- **建议消融**：`--user_list_fold mean_var | mean_only | var_only | none`（baseline 路径）四种配置。

---

### 2.4 ④ item list 保留为序列

**docx §6 设计意图**：item 序列保留 token 级语义，做 padding + token embedding + flatten。

**与 baseline 的差距**：baseline 已经做了 padding（[dataset.py:_pad_varlen_int_column](dataset.py:445)）和 token embedding。区别在于 pooling：baseline mean-pool，docx flatten。

**修订决策**：**不照搬 docx 的 flatten**。`item_int_feats_11` 在 README 没说语义，从 demo 看像是无序标签集合，flatten 会引入「位置 0 vs 位置 1 是不同语义」的假设——不正确的归纳偏置。**保留 baseline 的 mean-pool**。

**这一步落地**：什么都不改，与 baseline 一致。文档里明确写「与 docx §6.3 不一致」。

**预测**：±0.05%（影响小）。

---

### 2.5 ⑤ Dense / new 列归一化

**docx §7 设计意图**：StandardScaler 让数值列稳定；list dense 在 padding 后位置归一化、再 restore 原长。

**与 baseline 的差距**：baseline 不归一化（[dataset.py:566-571](dataset.py:566)）；模型层靠 LayerNorm 兜底（[model.py:1029](model.py:1029)）。LayerNorm 是 per-sample 跨特征，StandardScaler 是 per-feature 跨样本——**互补不冗余**。

**落地代码点**：

1. 新增 [tools/fit_dense_stats.py](tools/fit_dense_stats.py)：在训练子集（≥1M 行）上 fit per-fid mean/std。**list dense 只在有效长度内统计**，不要把 padding 0 计入。输出 `dense_stats.json`：
```json
{
  "user_dense_feats_61": {"mean": [...32维...], "std": [...]},
  "user_dense_feats_62": {"mean": [...变长...], "std": [...]},
  ...
  "user_int_feats_60_var": {"mean": 5.32, "std": 1.81},   // user_new_feats 也归一化
  ...
}
```

2. [dataset.py._convert_batch](dataset.py:565) dense 路径：
```python
for ci, dim, offset in self._user_dense_plan:
    padded = self._pad_varlen_float_column(batch.column(ci), dim, B)
    if dense_stats and fid in dense_stats:
        m = dense_stats[fid]['mean']  # (dim,)
        s = np.maximum(dense_stats[fid]['std'], 1e-6)
        valid = padded != 0  # padding mask
        padded[valid] = (padded[valid] - m_broadcast[valid]) / s_broadcast[valid]
    user_dense[:, offset:offset+dim] = padded
```

**预测**：+0.10% ~ +0.30% AUC（**全方案中把握最高的一步**）。

---

## 3. baseline 落地动作清单

| 文件 | 修改点 | 复杂度 | 依赖 |
| --- | --- | --- | --- |
| [tools/audit_schema.py](tools/audit_schema.py) | **新增** | 中 | 无 |
| [tools/fit_dense_stats.py](tools/fit_dense_stats.py) | **新增** | 低 | audit_schema |
| [dataset.py:_load_schema](dataset.py:272) | 加 `--audit_path` 入参；`user_list_fold` 字段解析 | 低 | audit_schema |
| [dataset.py:_convert_batch](dataset.py:505) | 加 +1 偏移；user list 折叠；dense z-score | 中 | audit + stats |
| [schema.json](schema.json) | 重新生成（vs+1，user_list_fold 字段） | 低 | audit_schema |
| [model.py:GroupNSTokenizer](model.py:988) | `nn.Embedding(vs+2, ..., padding_idx=0)` 配合 +1 | 低 | – |
| [train.py](train.py) | 新增 `--audit_path / --dense_stats / --user_list_fold` 等 CLI | 低 | – |
| [run.sh](run.sh) | 默认启用 dense_stats（最稳的一步） | 低 | – |

**所有改造通过 CLI flag 控制**：不传任何 flag 时 dataset 行为与 main 分支字节一致。

---

## 4. GNN ↔ HyFormer 融合策略

### 4.1 共享前提

`gnn.ipynb` 和 baseline HyFormer 都在 §2 的预处理流程下游消费数据：
- 共享：`user_int_feats / user_dense_feats / item_int_feats / item_dense_feats / label`
- 独占：`seq_data / seq_lens / seq_time_buckets`（HyFormer 用，GNN 忽略）
- GNN 特有：`edge_index`（user-item 二部图，从 batch 的 user_id、item_id 实时建图）

这意味着两个模型可以**用同一个 DataLoader**，数据预处理只做一遍。

### 4.2 四种融合方案（按工程成本排序）

#### 方案 A：Logit Ensemble（最低成本，先做）

```
HyFormer.forward(batch) → logit_h
GNN.forward(batch)      → logit_g
final_prob = sigmoid(α * logit_h + (1-α) * logit_g)
```

- **训练**：两个模型独立训练；α 在 validation 集上网格搜索。
- **优点**：完全解耦；任何一边出错不影响另一边；可以 lazy 实现（保存两个模型的 prediction，离线 ensemble）。
- **缺点**：不是端到端；两个模型可能学到高度相关的特征，ensemble 收益打折。
- **预期**：+0.05% ~ +0.20% AUC（取决于两个模型的相关性，越解耦收益越大）。

#### 方案 B：GNN embedding 作为 NS Token 注入 HyFormer

```
# GNN 先产出 user/item 节点 embedding
gnn_user_emb, gnn_item_emb = GNN(edge_index, raw_features)  # (B, 128) each

# 拼到 HyFormer 的 NS token 序列
ns_tokens = HyFormer.user_ns_tokenizer(...)  # (B, num_user_groups, D)
ns_tokens = torch.cat([
    ns_tokens,
    gnn_user_emb.unsqueeze(1),  # 1 个新 token
    gnn_item_emb.unsqueeze(1),
], dim=1)
# 进入 HyFormer 主干
```

- **训练**：联合反传，GNN 的梯度从 HyFormer 的 NS token 通道流回。
- **优点**：HyFormer 的 attention 自动学习 GNN embedding 的权重；端到端。
- **缺点**：每个 batch 都要建图、跑 GNN forward，训练吞吐显著下降；需要保证 batch 内 user-item pairs 能形成有意义的子图。
- **预期**：+0.10% ~ +0.30% AUC，但工程成本高（图采样、子图缓存、显存）。

#### 方案 C：GNN 作为辅助损失

```
loss_main = BCE(HyFormer(batch), label)               # CVR 主任务
loss_aux  = BCE(GNN(edge_index, batch), label)        # 链接预测辅助
loss = loss_main + λ * loss_aux
```

- **训练**：共享 embedding 表，loss 加权和。
- **优点**：辅助任务正则化；无 inference 成本（推断只用 HyFormer）。
- **缺点**：λ 难调；如果两个 loss 梯度方向冲突会拖累主任务。
- **预期**：±0.10% AUC（中等不确定性）。

#### 方案 D：两阶段（HyFormer 先训 → GNN 用 HyFormer embedding 初始化）

- 阶段 1：HyFormer 端到端训练，导出每个 user_id / item_id 的最终 embedding。
- 阶段 2：GNN 用这些 embedding 作为节点特征初始化，再做 link prediction fine-tune。
- inference：方案 A 的 ensemble。

- **优点**：HyFormer 的丰富语义注入到 GNN；两阶段易于调试。
- **缺点**：GNN 阶段无法影响 HyFormer 的训练；embedding 表 200M 用户全量很大。
- **预期**：方案 A 之上再 +0.05%。

### 4.3 推荐路径

**先做 A（验证两个模型互补性）→ 看 ensemble 收益是否值得 → 再决定要不要做 B 或 C**。

A 几乎 0 工程成本（只需要两个模型分别能跑通 + 一个 logit 平均脚本）。如果 A 没收益（说明 GNN 和 HyFormer 学到的信号高度重合），B/C 也没必要做。

---

## 5. 后续计划（10 天压缩版）

### 5.1 范围裁剪原则

10 天能做完的极限是「**高把握项全做 + 一次 ensemble**」，所以原计划砍掉：

- ❌ **方案 B/C/D 融合**（GNN-as-NS-token、辅助损失、两阶段）：每个都需要 1 周以上调通，10 天内做不出可用版本。**只保留方案 A logit ensemble**。
- ❌ **每个 docx 改动单独消融**：没时间做完整 ablation matrix，只挑最关键的 1 次对照（z-score on/off）。
- ❌ **多轮超参调优**：HyFormer 用 baseline 默认配置，不改 lr/batch/schedule。
- ⚠️ **user_list_fold 设为条件项**：D1 audit 结果决定要不要做，不强行。

### 5.2 关键路径（必须按顺序）

`audit (D1) → schema.json + stats (D2) → dataset.py 改造 (D3) → HyFormer 训练 (D4-D6) ‖ GNN 训练 (D5-D7) → ensemble (D8-D9) → 提交 (D10)`

### 5.3 每日任务表

| Day | 主任务 | 副任务（并行） | 交付物 | 退出条件 / 决策点 |
| --- | --- | --- | --- | --- |
| **D1** | `tools/audit_schema.py`：在全量子集（≥1M 行）上扫每列 null 率、unique、max_id；**重点**输出 60/89/90/91 的 per-user mean/var 是否常数 | 评估全量数据访问/吞吐 | `schema_audit.json` | **决策点 A**：60/89/90/91 在全量上是否仍退化？<br>• 若 mean/var 都常数 → 删除这 4 列<br>• 若 mean 常数 var 变化 → 用 var<br>• 若都变化 → 同时输出 mean+var |
| **D2** | `tools/fit_dense_stats.py`（小活，半天） | 重新生成 `schema.json`（vs+1） | `dense_stats.json` + 新 `schema.json` | stats 与 numpy 直算误差 < 1e-6 |
| **D3** | `dataset.py` 改造：<br>• Dense z-score（**必做**）<br>• +1 偏移（按 D1 决策）<br>• user_list_fold（按 D1 决策） | `model.py:GroupNSTokenizer` 同步 vs+2 | smoke test 通过 | DataLoader 输出与 main 分支 diff 仅在三个改造点；OOB 计数为 0 |
| **D4** | HyFormer 训练 run #1：**baseline 默认配置 + 全部数据预处理改动** | 占用 GPU；同时准备 GNN 改造 | 第一组 valid AUC | 跑通即过；目标 AUC ≥ baseline |
| **D5** | HyFormer 训练 run #2：**仅 z-score 关闭**（消融对照） | GNN：把 `gnn.ipynb` 改成消费 baseline DataLoader 的脚本 `gnn_baseline.py` | 消融 AUC 差值；GNN 训练脚本 | **决策点 B**：z-score 提点是否 ≥ +0.10%？<br>• 是 → 保留<br>• 否 → 关闭 z-score 重训 |
| **D6** | HyFormer 训练继续 / 收敛 | GNN 训练 run #1（同一数据子集） | HyFormer best ckpt + valid AUC；GNN best ckpt | 两个模型都收敛 |
| **D7** | HyFormer 推理：dump valid + test 的 logit | GNN 训练继续 / 推理 dump logit | 两个 logit 文件 | 两边都有 logit 输出 |
| **D8** | 方案 A logit ensemble：网格搜索 α ∈ {0.0, 0.1, ..., 1.0} 在 valid 集上 | 准备提交脚本 | 最佳 α + ensemble valid AUC | **决策点 C**：ensemble 是否高于单模型最高？<br>• 是 → 用 ensemble 提交<br>• 否 → 提交 HyFormer 单模型 |
| **D9** | 最终训练（用 best 配置）：HyFormer 全量子集 / 适当扩样本 | GNN 同上 | 最终 ckpt | 时间允许就跑；不允许就用 D6/D7 的 ckpt |
| **D10** | 生成最终 submission；double-check 流程；写实验报告 | — | submission 文件 | 提交 |

### 5.4 决策点 / 应急预案

| 决策点 | 触发条件 | Plan A | Plan B（fallback） |
| --- | --- | --- | --- |
| **A**（D1） | 60/89/90/91 全量上仍退化 | 直接删 4 列 | 保留为 multi-hot mean-pool（即 baseline 现状） |
| **B**（D5） | z-score 反而掉点 | 关闭 z-score | 检查 padding mask 实现是否正确 |
| **C**（D8） | ensemble 没收益 | 提交 HyFormer 单模型 | — |
| **GPU 时间不够**（任何时刻） | 训练吞吐慢于预期 | 减小训练样本规模（10% 子集） | 减少 epochs |
| **GNN 跑不通**（D6） | OOM 或采样卡死 | 退回到 demo 1k 上的 GNN，用相同 user_id 做 ensemble | 放弃 GNN，只交 HyFormer |
| **dataset.py 改造引入 bug**（D3） | smoke test 失败 | git revert，只保留 dense z-score（最低风险） | 完全 revert，只交 baseline + 微调 |

### 5.5 三个里程碑

- **MVP（D6 必须达成）**：HyFormer 跑通带新预处理的训练，AUC ≥ baseline。**这一条是 10 天内的保底产出**。
- **目标（D8）**：HyFormer + GNN ensemble，AUC > HyFormer 单模型。
- **stretch（D10）**：在更大数据子集 / 更多 epochs 上重训，挤出最后 0.1% AUC。

---

## 6. 预测汇总（10 天工期下的现实预期）

> 因为没时间多轮调超参 / 多次 ablation / 大样本充分训练，以下预期都比 7 周方案下保守 30%~50%。

| 配置 | 预期相对 baseline 的 AUC 变化 | 把握 | 10 天内可达 |
| --- | --- | --- | --- |
| 仅 ⑤ dense z-score | **+0.10% ~ +0.20%** | 高 | ✅ D3-D6 |
| ② +1 偏移 | +0.00% ~ +0.05% | 低 | ✅ 顺手做 |
| ① 删 high-missing 列 | -0.05% ~ +0.05% | 低 | ✅ 顺手做 |
| ③ user list 折叠（D1 audit 确认 var 有信号） | +0.05% ~ +0.15% | 中 | ⚠️ 依赖 audit |
| ③ user list 折叠（var 也无信号） | -0.10% ~ 0% | 中 | ❌ 跳过 |
| **数据预处理完整开（①②③⑤）** | **+0.05% ~ +0.30%** | 中 | ✅ MVP（D6） |
| 方案 A logit ensemble（HyFormer + GNN） | 在上述基础上 **+0.05% ~ +0.15%** | 中 | ✅ 目标（D8） |
| ~~方案 B GNN-as-NS-token~~ | ~~+0.05% ~ +0.15%~~ | – | ❌ 10 天做不完 |
| **10 天内最佳预期** | **+0.10% ~ +0.40%** | 中 | D8-D10 |
| **10 天内保底预期**（仅 z-score） | **+0.10% ~ +0.20%** | 高 | D6 |

---

## 7. 风险

🔴 **高风险**

1. **demo 1k 上 60/89/90/91 退化**——既不能证伪「variance 是信号」也不能证实。所有关于这 4 列的设计决策**必须在 ≥1M 行真实数据上重新验证**。如果全量上它们也是常数，**直接删掉这 4 列**比折叠更省事。
2. **user list mean(values) 在 categorical 列上类型错误**——如果 15/62-66/80 是 categorical hash，docx 的 mean 没有数值意义；baseline 的 mean-pool(embedding) 反而是更合理的处理。需在全量上看每列的「value 是否单调有序」（audit_schema.py 加 `is_ordinal` 检查）。
3. **schema.json +1 不同步 → 大量 OOB**：`vs` 必须改成 `max_id + 2`，否则数据 +1 后所有最大值都越界被 clip 到 0。

🟠 **中风险**

4. **dense z-score 的 padding mask** 必须严格区分「padding 0」与「有效 0」——baseline 的 `_pad_varlen_float_column` 不返回 valid_len，需要修改签名同时返回。
5. **方案 B / C 的联合训练梯度**：两个 loss 可能冲突；λ 调参成本高；显存压力大。
6. **GNN 在全量上能否跑通**：1k 上 0.79 AUC 不能外推；200M 用户的二部图采样、消息传递、显存都是工程挑战。`gnn.ipynb` 当前是 in-memory 全图计算，需要改成 batch sampling（PyG `NeighborLoader` 或 DGL）。

🟢 **低风险**

7. **CLI flag 默认关闭**：不传 flag 时与 main 分支字节一致，不会影响其他实验。
8. **baseline 已有的 OOB clip**：兜底 schema 不匹配的越界。

---

## 8. 立即开始：D1 单日任务

### 8.1 D1 上午：audit_schema.py 工具开发

```bash
python tools/audit_schema.py \
    --data_dir /path/to/full_data \
    --n_sample_rows 1000000 \
    --threshold_missing 0.80 \
    --output schema_audit.json
```

输出：
- `high_missing`: 缺失率 > 阈值的 fid
- `constant`: unique = 1 的 fid
- `list_stats`: `{fid: {mean_std, var_std, len_p95, value_unique}}` ——**重点确认 60/89/90/91 在全量上的 mean/var 是否真的常数 / 变化**
- `vocab`: `{fid: max_id}` ——用于 schema.json 的 vs+1 重生

### 8.2 D1 下午：在全量子集上跑 audit + 做决策点 A

根据 `list_stats` 输出对 60/89/90/91 做决策：

```python
# 决策树
for fid in [60, 89, 90, 91]:
    s = list_stats[f'user_int_feats_{fid}']
    if s['mean_std'] < 0.01 * abs(s['mean_mean']) and s['var_std'] < 0.01 * abs(s['var_mean']):
        decision[fid] = 'drop'           # 全退化，删
    elif s['mean_std'] < 0.01 * abs(s['mean_mean']):
        decision[fid] = 'var_only'       # mean 常数，用 var（docx 设计）
    elif s['var_std'] < 0.01 * abs(s['var_mean']):
        decision[fid] = 'mean_only'      # var 常数，用 mean
    else:
        decision[fid] = 'mean_and_var'   # 都有信息，输出两个标量
```

写出 `user_list_fold` 配置，进 `schema.json`。**这一步决定后面 9 天的 user list 处理路径**。

### 8.3 D1 收尾：更新 schema.json 草稿 + 估算 D4 训练时长

- 重新生成 schema.json（vs+1、user_list_fold、删 high_missing 列）
- 用一个 batch 跑 baseline 现状的训练，估算单 epoch 时间，反推 D4-D6 能跑几个 epoch / 多大子集
- **如果发现单 epoch > 8 小时**：D4 改用 10% 子集，记入 D5 决策点 B

---

## 附录 A：今日实证数据（demo 1k）

```
=== Per-user list mean/variance distribution (demo 1k) ===
col                     mean_of_means   std_of_means   mean_of_vars    std_of_vars  verdict
--------------------------------------------------------------------------------------------------------------
user_int_feats_15            531.1318       204.5595     66326.3930     57575.4838  both vary
user_int_feats_60              2.0000         0.0000         0.0000         0.0000  ALL DEGENERATE on demo
user_int_feats_62              5.9380         1.4624         4.0503         3.2790  both vary
user_int_feats_63             28.8538         8.5164       127.5202       115.7576  both vary
user_int_feats_64             22.7069         8.9558       150.3051       120.6819  both vary
user_int_feats_65            212.8353        60.9188      7989.4367      6123.7181  both vary
user_int_feats_66            776.1925       225.9465    121727.1464     84405.0582  both vary
user_int_feats_80              7.4566         3.5174         2.3001         4.1385  both vary
user_int_feats_89              4.6000         0.0000        10.8400         0.0000  shared list across users
user_int_feats_90              4.1000         0.0000         7.6900         0.0000  shared list across users
user_int_feats_91              4.7000         0.0000         7.2100         0.0000  shared list across users
```

`std_of_means` = 0 表示所有用户的 mean 完全相同（不只是「近似相同」）。`std_of_vars` = 0 表示所有用户的 variance 也完全相同。这种极端退化是 demo 数据的 simplification artifact，不能代表全量分布。
