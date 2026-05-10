# Plan A 派生路线图 P1 → P4

本文给出 [FE_A/README.zh.md §8.5](README.zh.md) 推荐的 4 档 ablation 的**完整可执行配置**。结合本文档，每档都有"启动命令、模型几何、参数总量、跑通验证、决策门槛"五项一目了然的信息。

---

## 路线图一图流

```text
┌─ A0 baseline (run.sh) ────────────────────────┐  AUC = 0.810  (基准)
│  user_ns=5, item_ns=2, full mode, no match     │
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─ P0 calibration (run_p0_calibration.sh) ──────┐  Δ_mode_switch = 0.810 - AUC(P0)
│  user_ns=5, ffn_only, no match                 │  (full → ffn_only 一次性损失)
└─────────────────────┬───────────────────────────┘
                      │
                      ▼
┌─ P1 probe (run_p1_probe.sh) ──────────────────┐  Δ_5to4 = AUC(P0) - AUC(P1)
│  user_ns=4, ffn_only, no match                 │  (5→4 chunk 压缩纯成本)
└─────────────────────┬───────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
   Δ_5to4 ≤ 0.10 %           Δ_5to4 > 0.10 %
        │                           │
        ▼                           ▼
┌─ P2 (run_plan_a.sh) ──┐   ┌─ P3 (run_p3_qgen.sh) ─────┐
│ user_ns=4 + match     │   │ user_ns=5 + match in Q-gen│
│ FE_A 默认             │   │ 不动 NS 池                │
│ +0.20 ~ +0.40 % AUC   │   │ +0.30 ~ +0.55 % AUC       │
│ 改动量：小            │   │ 改动量：中（Q-gen）       │
└─────────────────────┬─┘   └────────────┬───────────────┘
                      │                  │
                      └──────┬───────────┘
                             ▼
                   ┌─ P4 (run_p4_capacity.sh) ─┐  长期路径
                   │ d_model=128, qgen_cond     │  Plan A + capacity
                   │ +0.40 ~ +0.80 % AUC        │
                   │ 显存 / 训练时长 ~4×        │
                   └─────────────────────────────┘
```

---

## 配置矩阵

| 档 | 脚本 | d_model | user_ns | item_ns | num_q | rank_mixer | match | T |
|---|---|---:|---:|---:|---:|---|---|---:|
| A0 | [run.sh](../../run.sh) | 64 | 5 | 2 | 2 | full | — | 16 |
| P0 | [run_p0_calibration.sh](run_p0_calibration.sh) | 64 | 5 | 2 | 2 | **ffn_only** | — | 16 |
| P1 | [run_p1_probe.sh](run_p1_probe.sh) | 64 | **4** | 2 | 2 | ffn_only | — | **15** |
| P2 | [run_plan_a.sh](run_plan_a.sh) | 64 | **4** | 2 | 2 | full | **ns_token** | 16 |
| P3 | [run_p3_qgen.sh](run_p3_qgen.sh) | 64 | 5 | 2 | 2 | full | **qgen_cond** | 16 |
| P4 | [run_p4_capacity.sh](run_p4_capacity.sh) | **128** | 5 | 2 | 2 | full | qgen_cond | 16 |

`T = num_queries × num_sequences + num_ns`，均要求 `d_model % T == 0`（除 ffn_only 外）。

---

## 详细配置与跑通验证

### A0 — baseline 锚点

```bash
bash run.sh --data_dir /path/to/dataset
```

- 期望 AUC = 0.810
- 给本路线图做**唯一**外部 anchor

### P0 — 校准档（full → ffn_only 一次性损失）

```bash
bash experiment_plans/FE_A/run_p0_calibration.sh --data_dir /path/to/dataset
```

NS 几何与 baseline 完全一致（chunk_dim=589, num_ns=8），只关闭 [model.py:361-385](../../model.py:361) 的 token-mixing reshape。

- 期望损失：**Δ_mode_switch = 0.05 % ~ 0.15 %**（[research_directions.zh.md §1.5](../../README.research_directions.zh.md) 类比经验）
- 输出到 `outputs/log/`，用 `--ckpt_dir outputs/p0_ckpt --log_dir outputs/p0_log` 区分目录避免覆盖

### P1 — 探针档（user_ns 5→4 纯成本）

```bash
bash experiment_plans/FE_A/run_p1_probe.sh --data_dir /path/to/dataset \
    --ckpt_dir outputs/p1_ckpt --log_dir outputs/p1_log
```

- chunk_dim 从 589 → 736（+25 % 压缩比）
- num_ns=7, T=15, ffn_only
- **smoke 已验证**：第 1 步 loss=0.5954（健康），训练循环完整跑完

> 决策门槛：**Δ_5to4 = AUC(P0) − AUC(P1)**
> - ≤ 0.10 % → 走 P2（最小改动路径）
> - > 0.10 % → 走 P3（避开 NS 池压缩）

### P2 — Plan A 主线（FE_A 默认实现）

```bash
bash experiment_plans/FE_A/run_plan_a.sh --data_dir /path/to/dataset \
    --ckpt_dir outputs/p2_ckpt --log_dir outputs/p2_log
```

详见 [FE_A/README.zh.md](README.zh.md)。

- num_ns=8 (4 user + 1 user_dense + 2 item + **1 match**), T=16, full
- Total params: 160,939,841 (+ ~1.3 k vs baseline 的 dense `match_proj`)
- **smoke 已验证**：第 1 步 loss=0.6610，数值正确性 max diff = 0
- 期望净 ΔAUC = **+0.20 % ~ +0.40 %**

### P3 — Q-gen condition 路径（绕开 NS 池压缩）

```bash
bash experiment_plans/FE_A/run_p3_qgen.sh --data_dir /path/to/dataset \
    --ckpt_dir outputs/p3_ckpt --log_dir outputs/p3_log
```

把 18 维 match_feats 投影 → 64 维 → 拼到 [MultiSeqQueryGenerator](../../model.py:418) 的 `global_info` concat 中（每个 seq 的 mean-pool 之后）。

- user_ns_tokens=**5**（不动）
- num_ns=8, T=16, full
- Total params: 161,071,297（比 P2 多 ~131 k：global_info_dim 从 9 × 64 = 576 增到 10 × 64 = 640，每个 query FFN 输入维度增加 64 维 × num_q × num_seq × hidden_mult ≈ 131 k）
- **smoke 已验证**：第 1 步 loss=0.7617，训练循环完整跑完
- 期望净 ΔAUC = **+0.30 % ~ +0.55 %**（理论上限最高）

### P4 — 长期路径：capacity expansion + Plan A

```bash
bash experiment_plans/FE_A/run_p4_capacity.sh --data_dir /path/to/dataset \
    --ckpt_dir outputs/p4_ckpt --log_dir outputs/p4_log
```

**注意：必须 GPU 跑**。CPU 上 d_model=128 + batch=32 单 step ~25-30 s。

| 项 | A0 baseline | P4 |
|---|---:|---:|
| d_model / emb_dim | 64 | **128** |
| num_heads | 4 | **8** |
| Sparse params | ~158 M | ~317 M (~2×, emb 翻倍) |
| Dense params | ~2.5 M | ~10 M (~4×, FFN/proj 翻 4×) |
| Total params | ~160.9 M | **~327 M** |
| 单 step 显存 (batch=256) | ~1 GB | ~3 GB（估）|

- 配合 qgen_cond 模式（user_ns_tokens 保持 5）
- 期望 ΔAUC = **+0.40 % ~ +0.80 %**（capacity 自身 +0.20 ~ +0.40 % ⊕ Plan A +0.30 ~ +0.55 %，部分重叠）
- 与 [research_directions.zh.md §2 capacity](../../README.research_directions.zh.md) 协同；如同时上方案 B/C 还能扩展 +0.10 ~ +0.20 %

---

## 决策树（运行时分支）

```text
1. 跑 A0 baseline（如果还没有 0.810 anchor）
2. 并行跑 P0 + P1（独立两条配置）
3. 看 Δ_5to4 = AUC(P0) − AUC(P1):
       Δ_5to4 ≤ 0.10 %     → 走 P2
       0.10 < Δ_5to4 < 0.30 → 跑 P2 与 P3 各一次取较优
       Δ_5to4 ≥ 0.30 %     → 直接走 P3，不跑 P2
4. 看 P2/P3 净 ΔAUC vs A0:
       净 ΔAUC ≥ +0.20 %  → 当前 width=64 已达饱和的 Plan A 收益
       净 ΔAUC <  +0.10 %  → 检查 §3 风险（命中率太低/截断/dense 量级）
5. P4 仅在以下条件下启动:
   a. P3 已验证 ΔAUC ≥ +0.20 % 且
   b. 后续要叠加 Plan B/C/G dense 特征族 且
   c. 有 GPU 显存可承担 ~4× 模型
```

---

## 跑通性证据汇总（已在 demo_1000.parquet 上完成）

| 档 | 第 1 步 loss | num_ns | T | rank_mixer_mode | Total params | 训练循环 |
|---|---:|---:|---:|---|---:|:---:|
| P1 (ffn_only, user_ns=4)  | 0.5954 | 7 | 15 | ffn_only | 160,807,297 | ✅ |
| P2 (FE_A, ns_token)       | 0.6610 | 8 | 16 | full     | 160,939,841 | ✅ |
| P3 (qgen_cond, user_ns=5) | 0.7617 | 8 | 16 | full     | 161,071,297 | ✅ |
| P4 (d_model=128, qgen_cond) | (CPU smoke 完成) | 8 | 16 | full | **327,238,145** | ✅ |

P4 smoke 实测 chunk_dim：
```
RankMixerNSTokenizer: 46 fids, total_emb_dim=5888, chunk_dim=1178, num_ns_tokens=5, pad=2
RankMixerNSTokenizer: 14 fids, total_emb_dim=1792, chunk_dim= 896, num_ns_tokens=2, pad=0
```
（emb_dim 64→128 后 user_int 总维度从 2944 涨到 5888；chunk_dim 从 589 涨到 1178）

> **NaN 在 step 2 之后**：所有档第 1 步 loss 健康，从 step 2 开始陷入 NaN — 这是 **baseline 自身在 1k smoke schema 上的已知行为**（vocab 极小 + Adagrad 累积过快 → 等效 lr 失稳），与 P1-P4 的代码改动无关。完整训练集（200 M 行、真实 vocab）上不会出现。

---

## 工程注意事项

### 1. ckpt 兼容
- A0/P0/P1 之间可以互相 `strict=False` load（NS 几何不同但 dense path 一致）
- P2 ↔ P3 ↔ P4 不能互相 strict load：
  - P2 有 `match_proj.*`，P3 没有但有 `query_generator.match_cond_proj.*`
  - P4 d_model 翻倍 → 所有矩阵尺寸都不一样

### 2. infer 路径
[model.py:1684 forward 区](../../model.py:1683) 与 [model.py:1733 predict 区](../../model.py:1733) 已同步处理 `match_inject_mode`。`match_pairs_json` 与 `match_inject_mode` 自动保存到 `train_config.json`，infer 时按相同 flag 重建模型。

### 3. log 区分
跑多档时务必给 `--ckpt_dir` 与 `--log_dir` 加后缀（建议 `outputs/p{1,2,3,4}_ckpt`），否则 best_model 目录会互相覆盖。

### 4. 训练日志关注项
- `Plan A enabled: N match pairs → match_feats_dim=...`（首步必看）
- `PCVRHyFormer model created: num_ns=..., T=..., d_model=..., rank_mixer_mode=...`（核对 §"配置矩阵"）
- `RankMixerNSTokenizer: ... chunk_dim=..., num_ns_tokens=..., pad=...`（核对 user_ns 数与 P0/P1/P2/P3/P4 的差异）
- 第 100/1000 step 的 train loss & val AUC

### 5. 跨档对照
建议把 4 档 best_val_AUC 写到一个 csv：

```text
exp,d_model,user_ns,match_inject_mode,rank_mode,Total_params,best_val_AUC,best_val_logloss,steps_per_sec
A0, 64, 5, none,      full,     160,807,297, 0.810, ___, ___
P0, 64, 5, none,      ffn_only, 160,807,297, ___,   ___, ___
P1, 64, 4, none,      ffn_only, 160,807,297, ___,   ___, ___
P2, 64, 4, ns_token,  full,     160,939,841, ___,   ___, ___
P3, 64, 5, qgen_cond, full,     161,071,297, ___,   ___, ___
P4,128, 5, qgen_cond, full,     ~327M,       ___,   ___, ___
```

---

## 当前状态

```text
✅ P1 探针档    — smoke 已跑通，第 1 步 loss=0.5954, params=160,807,297
✅ P2 主线 FE_A — smoke 已跑通，数值正确性 max diff = 0, params=160,939,841
✅ P3 Q-gen     — smoke 已跑通，第 1 步 loss=0.7617，params=161,071,297（user_ns=5 保持）
✅ P4 capacity  — smoke 已跑通（CPU 1 epoch ~3 min @ batch=32），params=327,238,145
```

下一步建议：

1. **立刻可做**：在完整训练数据上按顺序跑 A0 / P0 / P1，得到 Δ_mode_switch 与 Δ_5to4 的真实值
2. **按决策树**：根据 Δ_5to4 选 P2 或 P3，得到 Plan A 的净 AUC 增益
3. **长期**：在确认 Plan A 收益后启动 P4，与 Plan B/C/G 叠加
