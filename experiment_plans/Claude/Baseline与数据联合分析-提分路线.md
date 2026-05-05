# Baseline × 数据联合分析 — 提分路线

> 灵感来自 kkl 的提示："数据和 baseline 对不上的地方，就是提分点；优化的核心是把模型丢失的梯度和信息补回去。"
>
> 本文不预设方法，先把最早 baseline (`commit 7c9f32e`，文件: `dataset.py / model.py / trainer.py / ns_groups.json / run.sh`) 中**数据流的每一处信息丢失**列出来，再为每一处给出一条最小可证伪的提分路线（按 ROI 排序）。

## 0. 总览：信息丢失地图

按数据从 parquet 进入模型的顺序，每一层都列出"baseline 处理 vs 数据本身存在"的不一致：

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ Parquet                                                                     │
│   ├─ user_int / user_dense / item_int / item_dense                          │
│   ├─ domain_a/b/c/d × (categorical sideinfo + timestamp)                    │
│   ├─ timestamp / label_type / label_time / user_id                          │
│   └─ NULL/-1 missing 值                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ baseline dataset.py                                                         │
│   ✗ L1: missing(-1) 与 padding(0) 被映射成同一个 id (arr<=0 → 0)           │
│   ✗ L2: item_dense_feats 固定返回 [B, 0]（即使 schema 给出也丢弃）         │
│   ✗ L3: user_dense 缺失填 0，无 z-score normalization                       │
│   ✗ L4: 当前样本 timestamp 只用于 time-diff，不进 NS token（没 hour/dow）   │
│   ✗ L5: seq_lens 只做 padding mask，不进 NS token（活跃度信号丢）           │
│   ✗ L6: 1h/1d/7d/30d 计数没生成（README §6.2 实证强）                       │
│   ✗ L7: 4 个 domain 共用同一套 BUCKET_BOUNDARIES（domain 时间分布差异巨大） │
│   ✗ L8: 当前样本特征 vs 历史序列的"是否出现"完全不计算（README §6.3 lift） │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ baseline model.py                                                           │
│   ✗ M1: has_item_dense = (item_dense_dim > 0) = False（dataset L2 决定）   │
│   ✗ M2: ns_tokenizer_type='rankmixer' + ns_groups_json="" → 字段语义全丢   │
│           （把所有 fid embedding 拍平 chunk，完全无视 group）                │
│   ✗ M3: emb_skip_threshold=1M 等于不过滤，高基数字段稀疏更新不稳定         │
│   ✗ M4: seq_id_emb_dropout = dropout*2，砍了一半 high-card id 信号但不补   │
│   ✗ M5: output_proj 只用 final Q tokens，丢弃 final NS tokens 的丰富表征  │
│   ✗ M6: 没有 user × item 显式交叉项（无 FM/cross/dual-tower 风格）          │
│   ✗ M7: padding 全空序列与短序列共用同一种 mask 处理                        │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ baseline trainer.py                                                         │
│   ✗ T1: row_group split = 文件序最后 10%，可能不是时间序最后 10%           │
│           ⇒ valid AUC 与 leaderboard eval AUC 的相关性弱                    │
│           （kkl 截图直接点到该问题：按时间划分 val 与线上更接近）            │
│   ✗ T2: BCE 全样本同权重，无 sample weight                                  │
│   ✗ T3: 单任务（label_type==2），label_type ∈ {1,3,...} 的辅助监督信号丢   │
│   ✗ T4: reinit_sparse_after_epoch=1 默认值过早（小训练集容易丢前期表示）   │
│   ✗ T5: EarlyStopping delta 默认 0，AUC 抖动 0.0001 就保 ckpt              │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
       Logits → Eval AUC
```

每个 ✗ 都是一个"baseline 没用到的信号"或"baseline 处理把信息抹平的地方"。下文按 **ROI（不依赖训练即可估计）× 工程量（小 → 中 → 大）** 排序。

## 1. P0 路线（不动模型结构、低风险、累计 +0.003 ~ +0.008）

### 1.1 P0-T1：把 valid split 改成"按 timestamp 切最后 10%"

**问题（trainer.py L1）：** 当前 `row_group_range = (n_train_rgs, total_rgs)`，按文件 glob 顺序取最后 10% row group 作为 valid。如果 parquet 文件不是按时间严格排序写入，valid 与 train 的时间分布会重叠 → valid AUC 高估，与 leaderboard eval 切分不一致。

**数据 vs baseline 对不上：** 数据的 `timestamp` 列是真实事件时间，但 baseline 只用文件序近似时间序。

**改法：**

```python
# get_pcvr_data() 中替换 row_group split：
1. 第一遍 streaming 扫所有 row groups，记录每个 (file_idx, rg_idx) 的 max(timestamp)。
2. 按 max(timestamp) 升序排序得到 rg_time_order。
3. 取 rg_time_order 后 valid_ratio 比例作为 valid。
4. 把 PCVRParquetDataset 的 row_group_range 参数升级为 row_group_indices: List[int]。
```

**预期：** valid AUC 可能略下降（更"冷"的样本进入 valid），但 valid 涨跌方向与 eval AUC 相关性显著上升 → 后续所有实验调参都更可信。**这一步本身不直接涨 eval，但是让所有其他实验"有效"**。

**工程量：** dataset.py 改 ~30 行；不动 model 与 trainer。

---

### 1.2 P0-L1：missing(-1) 与 padding(0) 拆开

**问题（dataset.py L1）：** `arr <= 0` 全部映射到 0，与 padding 用同一个 token id。模型无法区分"这个特征用户没有值"与"该位置没填满"。

**数据 vs baseline 对不上：** 数据中 -1 是显式 missing，0 是合法 padding（来自 `pa.ListArray` 的实际 padding），两者语义不同。

**改法：**

```python
# 选项 A（最低成本）：vocab_size + 2，约定
#   id=0  → padding
#   id=1  → missing（即原来的 -1）
#   id=k+1 → 原 id=k

# 选项 B（schema 不变）：单独建 missing_mask 进 dense token
#   for fid in user_int/item_int:
#       missing_mask[fid] = (raw_value == -1).astype(float32)
#   把 missing_mask 拼到 user_dense / item_dense
```

推荐 **选项 B**，schema/embedding 不动，纯 dense 增量。

**预期：** valid+eval AUC 同向上涨 +0.0005 ~ +0.0015（多数 anonymous 类别字段缺失率 > 30%，missing 本身就是 strong signal）。

**工程量：** dataset.py 加 ~50 行 missing_mask 计算 + user_dense/item_dense 拼接；model.py 不动。

---

### 1.3 P0-L3：user_dense 归一化 + missing indicator

**问题（dataset.py L3）：** `_pad_varlen_float_column` 把缺失填 0，原始量级直接进 `user_dense_proj`。如果 dense 列里某些是 [0, 1e6] 量级、某些是 [0, 1] 量级，投影会被大数主导，小信号被湮没。

**数据 vs baseline 对不上：** 数据中 dense 列量级差异巨大，但 baseline 用同一个 `nn.Linear` 投影。

**改法：**

```python
# 离线脚本 build_normalization_stats.py：
#   用前 90% row groups 拟合 z-score (mean, std) per (fid, dim_index)
#   写到 dense_norm_stats.json
# dataset.py：
#   加载 dense_norm_stats.json
#   _pad_varlen_float_column 后做 z = (x - mean) / max(std, 1e-6)
#   missing 位置标记 mask=1，z=0
```

注：FE-00 已经做了一部分（`build_fe00_preprocess_dataset.py`），实测 +0.0011。**这里的关键是要把 missing mask 做出来，并和 z-score 一起进 dense token**。

**预期：** valid+eval +0.0008 ~ +0.0020。

**工程量：** 一个离线脚本 + dataset.py ~30 行。

---

### 1.4 P0-L4：当前样本 timestamp 衍生特征进 NS token

**问题（dataset.py L4）：** `timestamp` 作为 ndarray 取出，但只参与 `ts_expanded - ts_padded` 的 time-diff 计算。当前样本本身的 hour-of-day / day-of-week / day-since-min-train-ts 完全没进入模型。

**数据 vs baseline 对不上：** CVR 任务的曝光时间在一天/一周内有强周期性（晚上转化率高、周末下单多等），baseline 完全没建模。

**改法：**

```python
# dataset.py _convert_batch():
#   ts = timestamps   # int64, unix seconds
#   hour_of_day      = (ts // 3600) % 24            # [0, 24)
#   day_of_week      = ((ts // 86400) + 4) % 7      # [0, 7)，UTC 1970-01-01 是周四
#   day_since_min    = (ts - GLOBAL_MIN_TS) // 86400  # 训练集最小 timestamp
#   把这 3 个 int 加进 user_int_feats（vocab=24/7/N）
#   或者拼成 (B, 3) 的 dense 进 user_dense
# schema.json：增加 user_int_feats_TIME_HOUR / TIME_DOW / TIME_DSM
```

推荐 **dense 路径**：直接把 `[hour/24, day_of_week/7, log1p(day_since_min)]` 加到 user_dense 三列，配合 P0-L3 的 z-score。这样不动 vocab。

**预期：** valid+eval +0.0005 ~ +0.0015。最便宜的"添加新信息"。

**工程量：** dataset.py 加 ~20 行；user_dense schema dim 加 3。

---

### 1.5 P0-L5/L6：seq_len 与多窗口计数进 NS token

**问题（dataset.py L5/L6）：** `seq_lens[domain]` 只用于 padding mask；time bucket 只用于 sequence position embedding。**user 的活跃度（每个 domain 的序列长度）和近期事件密度（1h/1d/7d/30d 计数）这两个 README §6.2 已实证有 lift 的信号没有作为 user 级 dense 特征进入 NS token。**

**README §6.2 实证：**

```text
domain_a count_7d 高分位: 15.4% 正例率 vs 低分位 10.4%   (lift ~ 1.48)
domain_c len 高分位: 15.1% vs 11.6%
domain_d len 高分位: 14.0% vs 10.0%
```

**改法：**

```python
# dataset.py _convert_batch() 末尾：
#   for domain in seq_domains:
#       len_d = lengths[i].astype(np.float32)
#       count_1h, count_1d, count_7d, count_30d = ...
#         （从 time_bucket 反推：boundaries 包含 3600/86400/604800/2592000）
#         在已有 buckets 上做 cumcount 即可，几乎零额外成本
#   把这 5*4=20 个 dense 拼成 user_dense 增量
```

**预期：** valid+eval +0.0010 ~ +0.0025。这是 P0 中**单点 ROI 最高**的一项。

**工程量：** dataset.py 加 ~40 行；user_dense schema dim 加 20。

---

### 1.6 P0 阶段总计

| 实验 | 单步增益（中位） | 累计 (从 0.8105) |
| --- | ---: | ---: |
| P0-T1 时间序 split | 0（间接：让后续实验可信） | 0.8105 |
| P0-L1 missing/padding 拆开 | +0.0010 | 0.8115 |
| P0-L3 user_dense 归一化 + mask | +0.0014 | 0.8129 |
| P0-L4 当前 ts 衍生特征 | +0.0010 | 0.8139 |
| P0-L5/L6 seq_len + 多窗口计数 | +0.0017 | **0.8156** |

**P0 全部完成预期：约 0.8155 ~ 0.8170**（已超过当前 FE-01B 的 0.8121，逼近 GNN4 的 0.8159）。

## 2. P1 路线（中等改动、累计 +0.003 ~ +0.005）

### 2.1 P1-L2/M1：启用 item_dense token

**问题（dataset.py L2 + model.py M1）：** baseline `item_dense_feats = torch.zeros(B, 0)` 强制使 item dense token 关闭。即使后续 schema 升级给出 `item_dense`，也要先在 dataset 端读取 + 在 model 端 `has_item_dense=True`。

**改法：** dataset.py 接入 `_item_dense_plan`（与 user_dense 同结构）；模型不需要改（`has_item_dense=item_dense_dim>0` 已存在）。**P0-L4 中"当前样本时间"也可以放到 item_dense（item 维度可以有 freshness 等特征）。**

**预期：** 单独启用而不加新特征几乎没用；与下面的 frequency/match 联合启用才有意义。

---

### 2.2 P1-Match：当前 item 属性 × 历史序列匹配（FE-01B 已验证）

**问题（dataset.py L8）：** README §6.3 指出 `item_int_feats_9` 是否在 `domain_d_seq_19` 中出现是 26.8% vs 11.8% 的强 lift。baseline 完全不计算这种"target × history"的交互。

**已落账实测：** FE-01B 单独跑 eval = 0.812102 (+0.0016 vs B0)。

**P1 推荐：** 直接复用现有 `build_feature_engineering_dataset.py --feature_set fe01b` 输出，叠在 P0 完成后的 dataset 之上。在 P0 已经把 split / missing / norm / temporal token 修好的前提下，FE-01B 的 +0.0016 应该可以稳定加在 P0 的累计上。

**预期：** 与 P0-L1/L3/L4/L5 一起累计 ≈ 0.817 ~ 0.819。

---

### 2.3 P1-NS：NS group 按字段语义重写（model.py M2）

**问题（model.py M2 + run.sh）：** baseline 默认 `--ns_groups_json ""`，让 RankMixerNSTokenizer 把所有 fid 的 embedding cat 后等分 chunk。这是**完全无视字段语义**的暴力切分，profile 字段 / 高基数 entity 字段 / 行为统计字段会被混进同一 chunk。

**数据 vs baseline 对不上：** 数据中字段类别（profile / behavior / interest / temporal）天然分组，baseline 把它们物理打散后再随机投影。

**改法：**

```bash
--ns_tokenizer_type rankmixer
--ns_groups_json ns_groups.feature_engineering.json   # 按语义分 7+4 组
--user_ns_tokens 7      # 与 user group 数对齐
--item_ns_tokens 4
--num_queries 1
```

注意 token 整除约束：`T = 1*4 + (7+1+4+1) = 17`，`d_model=64` 不整除。两种解决：

```text
方案 a: --rank_mixer_mode ffn_only   去掉整除约束
方案 b: user_ns_tokens=6, T=16, 64 % 16 == 0
```

推荐方案 a（保留全部 7 组语义），观察是否对训练稳定性有损。

**预期：** +0.0005 ~ +0.0015。

---

### 2.4 P1-Output：final NS token 的 gated fusion 进 head（model.py M5）

**问题（model.py M5）：** `output = output_proj(all_q.view(B, -1))`，HyFormer 块迭代后 NS tokens 学到的 user/item 表征**完全被丢弃**。EXPERIMENT_CONCLUSION.md 中尝试过 `output_include_ns` 直接拼接，结果反而下降 (0.815 → 0.811)，说明粗暴拼接会让 head 过度依赖静态信号。

**改法（gated 而不是直接拼）：**

```python
# 在 PCVRHyFormer.forward 末尾：
q_repr  = self.output_proj(all_q.view(B, -1))      # (B, D)
ns_repr = self.ns_pool(curr_ns)                     # (B, D)，对 NS token 做 attention pooling
gate    = torch.sigmoid(self.gate_proj(torch.cat([q_repr, ns_repr], dim=-1)))  # (B, D)
nn.init.zeros_(self.gate_proj.weight)
nn.init.constant_(self.gate_proj.bias, -2.0)        # gate ≈ 0.12，几乎不开
final = q_repr + gate * ns_repr
logits = self.clsfier(final)
```

关键是 **gate bias 初始化偏负**，让训练初期模型行为接近原 baseline，gate 只在确认有用时被学起来。

**预期：** +0.0005 ~ +0.0015；同时降低 0.811 那种 catastrophic 风险。

---

### 2.5 P1 累计

| 实验 | 增益（中位） | 累计 |
| --- | ---: | ---: |
| P0 完成 | — | 0.8156 |
| P1-Match (FE-01B) | +0.0016 | 0.8172 |
| P1-NS (语义分组) | +0.0008 | 0.8180 |
| P1-Output (gated NS) | +0.0008 | **0.8188** |

## 3. P2 路线（结构性、累计 +0.003 ~ +0.006）

### 3.1 P2-Domain：4 个 domain 各用自己的 BUCKET_BOUNDARIES（dataset.py L7）

**问题：** README §3.2 数据：

```text
domain_a 中位事件年龄 73.5d
domain_b 中位事件年龄 94.2d
domain_c 中位事件年龄 275.3d
domain_d 中位事件年龄 12.5d
```

baseline 用同一套 `BUCKET_BOUNDARIES`（覆盖 5s ~ 1y），对 domain_d（高频近期）和 domain_c（低频久远）分辨率都不够。

**改法：** 给每个 domain 一套 `BUCKET_BOUNDARIES_<domain>`，按该 domain 的事件年龄分布做分位数边界。`time_embedding` 改成 `nn.ModuleDict({domain: nn.Embedding(...)})`。

**预期：** +0.0005 ~ +0.0012。

### 3.2 P2-HighCard：高基数字段 hashing + count（model.py M3）

**问题：** README §4 列出的高基数字段（`domain_c_seq_47` 287k unique 等）当前都建独立 Embedding。`emb_skip_threshold=1M` 实际未过滤。但 287k 容量的 Embedding 在 valid 时大半 lookup 表稀疏，泛化弱。

**改法：**

```text
1. 对 unique > 50k 的字段，hash 到 50k 桶（确定性 hash + 数据驱动 collision check）。
2. 同时构造 count features：每个 user 在该字段上的历史 unique 数 / 频次。
3. count 进 user_dense；hash id 仍走 Embedding 但容量减一个数量级。
```

**预期：** +0.0008 ~ +0.0020；显著降低 ckpt 大小与显存。

### 3.3 P2-Aux：辅助任务 head（trainer.py T3）

**问题：** baseline 只用 `label_type == 2` 作为正样本，但 `label_type ∈ {1, 3, ...}` 通常包含其他行为类型（点击、收藏、加购等），是 user-level engagement 的强信号，丢掉可惜。

**改法：**

```python
# 在 model 输出加一个 aux head：
engagement_head: nn.Linear(d_model, NUM_LABEL_TYPES)  # 多分类
total_loss = cvr_bce + 0.05 * engagement_ce
```

注：engagement 任务**不**必须比 conversion 难，做正则化用即可。

**预期：** +0.0005 ~ +0.0015。

### 3.4 P2 累计

| 实验 | 增益（中位） | 累计 |
| --- | ---: | ---: |
| P1 完成 | — | 0.8188 |
| P2-Domain 各自 bucket | +0.0008 | 0.8196 |
| P2-HighCard hashing+count | +0.0014 | 0.8210 |
| P2-Aux engagement head | +0.0010 | **0.8220** |

到这里已经接近 0.825 目标。继续往上需要 ensemble / multi-seed / SWA 等"工程性"通道。

## 4. 完整路线累计预测

| 阶段 | 实验数 | 累计 eval AUC（中位） | 增益来源 |
| --- | ---: | ---: | --- |
| baseline B0 | — | 0.8105 | — |
| P0 数据层修补 | 5 | 0.8156 | 把 baseline 丢的"显然信息"补回去 |
| P1 模块级 + match | 4 | 0.8188 | 已验证模块（FE-01B）+ 语义 NS |
| P2 结构性优化 | 3 | 0.8220 | 多 domain bucket + 高基数处理 + aux loss |
| Ensemble / 3 seed / SWA | — | 0.8240 ~ 0.8260 | 工程稳态 |

**乐观路径**（每步取上界）可达 **0.829 ~ 0.831**；悲观（中位下沿）约 0.819 ~ 0.821。

## 5. 与现有 FE 系列实验的关系

```text
FE-00       ⊂ P0-L3                    （已落账 +0.0011，未做 missing mask 与多窗口计数）
FE-01B      ⊂ P1-Match                 （已落账 +0.0016）
FE-01A      ⊂ P0-L5/L6 的子集           （仅 total frequency，未做 1d/7d 拆窗口与 seq_len）
FE-01 fail  ↔ purchase_frequency 是 label-leak，本路线明确拒绝
GNN4 0.8159 ↔ 等价于 P1-NS + P1-Output 的"以 GNN 替代 group + token 间关系学习"
```

**重要观察：** GNN4 之所以涨 +0.0054，本质上是把 baseline 的 M2（NS group 缺失）+ M5（output 丢 NS）一次性补回来 —— GNN 给所有 NS token 之间补了边，相当于让模型自己学 group 关系。但它不能补 dataset 层的 L1/L3/L4/L5/L6（数据本身没给的信号）。**所以 P0 数据层和 GNN4 几乎正交，可叠加**。

## 6. 推荐执行顺序

```text
Week 1:
  Day 1-2: P0-T1 时间序 split          → 让后续实验可信
  Day 2-3: P0-L1 missing mask           → eval 应涨 +0.001
  Day 3-4: P0-L3 dense norm             → 与 FE-00 部分重合，覆盖完整
  Day 4-5: P0-L4 当前 ts 衍生           → 最便宜
  Day 5-6: P0-L5/L6 seq_len + 多窗口    → P0 单点 ROI 最高

Week 2:
  Day 7:   P1-Match (FE-01B 复用)       → 直接套现成
  Day 8-9: P1-NS (语义 group 重写)
  Day 9:   P1-Output (gated NS)

Week 3:
  Day 10-11: P2-Domain 各自 bucket
  Day 11-12: P2-HighCard
  Day 12-13: P2-Aux

Week 3 末: 3-seed 复跑 + SWA + 集成
```

每步只动一个变量，valid 涨跌方向与 eval 一致后再走下一步。

## 7. 反检查清单（防止误判）

```text
[ ] P0-T1 时间序 split 改完后，重跑 baseline 应得到与原 baseline 不同的 valid AUC（说明 split 真换了）
[ ] missing mask 必须 train-only 拟合（mask 比例统计），eval 用同一份
[ ] dense norm stats 必须从训练 row groups 拟合，不从全量
[ ] seq_len 与多窗口 count 必须用 event_time <= sample_timestamp 的过滤
[ ] FE-01B 复用时，注意 P0 改完 dataset 后 schema 与原 FE-01B 是否兼容（fid 编号要避开 user_dense_feats 已被 P0-L4 占用的位置）
[ ] gated NS bias 必须显式初始化为负值，否则训练初期就被推走
[ ] aux engagement 的 label 必须实际取自 label_type ∈ {1,3,...}，不能误用 label_type==2 复制
[ ] valid AUC 涨但 eval 不涨 ≥ 2 个 seed 同向：立即回退该步
```

## 8. 一句话总结

```text
Baseline 是"最朴素的 RankMixer + HyFormer"，它把数据里 6 类显然存在的信号压扁了：
  1. missing 与 padding 不分；
  2. dense 不归一化；
  3. 当前样本时间不进 NS；
  4. 序列长度与窗口计数不进 NS；
  5. NS group 按物理 chunk 而非语义；
  6. final NS token 表征被 head 丢弃。

把这 6 件事一件一件补回来 —— 不发明新模型、不改 backbone、不引入 label 依赖特征 ——
预期就能从 0.8105 推到 0.819~0.822。
再叠 GNN backbone (0.815905 已落账起点) 与多 seed/SWA 集成，
冲 0.825 是有路径而非靠运气的。
```
