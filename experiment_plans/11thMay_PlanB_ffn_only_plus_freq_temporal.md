# 11thMay 方案 B — FE-00 + FE-01A 频次 + 12 时间桶 + ffn_only（最终可执行版）

> **状态：可上传平台训练**。FE-00 预处理、FE-01A 频次基线、12 时间桶、ffn_only 模式四个增益已全部集成。
> **改动锁定**：相对 `run.sh` 主分支 baseline (A0) 引入 1 个 CLI flag (`--rank_mixer_mode ffn_only`) + 14 个 dense 特征列 + FE-00 预处理 (drop 高缺失 user_int + 填补 missing + dense z-score)。

---

## 0. 一图流：本方案做了什么

```text
原 baseline (A0):
  parquet → train.py (rank_mixer_mode=full, user_ns=5, item_ns=2, num_q=2)
  → eval AUC = 0.810 *(假定 / leaderboard anchor)*

11thMay-PlanB (含 FE-00 预处理):
  parquet → build_fe00_preprocess_dataset.py
              --missing_threshold 0.75
              (P000: drop user_int with missing>75%, P001: fill missing with mean,
               P002: z-score user/item dense numerical)
          → fe00_dataset/  (含 schema.json + ns_groups.fe00.json
                            + dense_normalization_stats.json + 增广 parquet)
          → build_feature_engineering_dataset.py
              --feature_set 11thmay_a
              (新增 14 列：user_dense 110/113..124 + item_dense 86)
          → 11thmay_b_dataset/  (含最终 schema.json + feature_engineering_stats.json
                                + 增广 parquet)
          → train.py (rank_mixer_mode=ffn_only, user_ns=5, item_ns=2, num_q=2)
  → 预期 eval AUC ≈ 0.8145 ~ 0.8165
```

---

## 1. 相对 baseline 的修改（精确清单）

### 1.1 修改了哪些文件

| 文件 | 修改性质 | 行数 | 用途 |
|---|---|---|---|
| `dataset.py` | **代码改动** | +18 行 / -3 行 | 启用 item_dense 通路（原硬编码空）|
| `build_fe00_preprocess_dataset.py` | **新增文件** | +492 行（移植 worktree）| FE-00 预处理：drop 高缺失 user_int + 填补 missing + dense z-score |
| `build_feature_engineering_dataset.py` | **新增文件** | +870 行（移植 worktree 并扩展）| 生成 11thmay_a 数据集（含 FE-01A 频次 + 12 时间桶）|
| `build_11thmay_b_pipeline.sh` | **新增文件** | +90 行 | 一键运行 FE-00 → 11thmay_a 数据流水线 |
| `run_11thmay_b.sh` | **新增文件** | +40 行 | 启动训练（rankmixer + ffn_only）|
| `train.py` | **不动** | 0 | `--rank_mixer_mode ffn_only` 已支持，仅 CLI 切换 |
| `model.py` | **不动** | 0 | item_dense_dim>0 路径已实现 |
| `ns_groups.json` | **不动** | 0 | rankmixer 路径默认 `--ns_groups_json ""` 跳过 |

### 1.2 `dataset.py` 4 处补丁（item_dense 通路启用）

| 位置 | baseline | 11thMay-PlanB |
|---|---|---|
| `__init__` buffer 预分配 | 缺 `_buf_item_dense` | 新增（与 user_dense 对称）|
| `__init__` plan 构造 | 缺 `_item_dense_plan` | 新增（与 user_dense 对称）|
| `_load_schema` | `item_dense_schema = FeatureSchema()`（强制空）| `raw.get('item_dense', [])` → 填入 schema |
| `_convert_batch` | `'item_dense_feats': torch.zeros(B, 0)` | 真正读取并填充 |

**向后兼容**：schema.json 中没有 `item_dense` 字段时（如 baseline 数据），`raw.get('item_dense', [])` 返回空 list，`_item_dense_plan` 为空，输出 tensor shape = (B, 0)；模型端 `has_item_dense = item_dense_dim > 0` 自动跳过 item_dense_proj。**完全等价于改前行为**。

### 1.3 训练 CLI 改动

唯一相对 `run.sh` 的差异：

```diff
  python3 -u train.py \
      --ns_tokenizer_type rankmixer \
      --user_ns_tokens 5 \
      --item_ns_tokens 2 \
      --num_queries 2 \
      --ns_groups_json "" \
+     --rank_mixer_mode ffn_only \
      --emb_skip_threshold 1000000 \
      --num_workers 8 \
      "$@"
```

NS / Q / T 几何完全保持 A0 不变：

```text
user_ns_tokens = 5
item_ns_tokens = 2
num_queries    = 2
num_ns         = 5 + 1(user_dense) + 2 + 1(item_dense) = 9   ← 因 item_dense=1，比 A0 多 1
T              = 2*4 + 9 = 17

⚠️ T=17 在 full 模式下会触发 d_model % T 检查失败（[model.py:1385](../model.py:1385) 抛 ValueError）。
   在 ffn_only 模式下 [model.py:402](../model.py:402)，token-mixing reshape 被跳过，T 不需要整除。
```

→ **这是为什么 ffn_only 模式是 PlanB 的必要条件**：开启 item_dense_feats_86 后 num_ns 从 8 涨到 9，full 模式会直接报错。

---

## 2. 修改了哪些特征（FE-00 预处理 + 14 个 dense 标量）

### 2.0 FE-00 预处理（无新 fid，但修改 schema 与列值；eval 已验证 +0.001121 vs B0）

FE-00 不增加 fid，但在所有下游计算之前**根据 [sample_data_feature_summary.csv](../sample_data_feature_summary.csv) 中实测的 missing 率，自动删除高缺失字段 + 填补 missing + 对 dense 数值列做 z-score**：

| DOCX 位置 | 实现 | 影响范围 |
|---|---|---|
| **P000** drop **user_int** with missing > threshold | 统计每个 `user_int_feats_*` 的 missing ratio (`missing = null OR value<=0 OR empty list`)，超过 threshold 的 fid 从 schema + ns_groups 中删除 | 默认 0.75 阈值下 drop **6 个 user_int fid**（DOCX P000 原文限定 "from user"，item_int 不在此步删除范围）|
| **P001** fill int missing with mean | 用 train row groups 的正值均值四舍五入填补 user/item int missing；list int 默认只填已有元素中的 missing | **包含 item_int**：即使高缺失的 `item_int_feats_83/84/85` 也只是被 fill，不被 drop |
| **P002** z-score user/item dense numerical | 用 train row groups 拟合 mean / std，所有数值 dense 列做 z-score；sequence side embedding ID 不做 z-score | 训练时 user_dense / item_dense 量级一致 |

> ⚠️ **重要边界**：P000 严格只对 `user_int` 起作用（来源：DOCX 原文 "delete int feature **from user** whose missing value proportion >75%"；[build_fe00_preprocess_dataset.py:438-444](../build_fe00_preprocess_dataset.py:438) 实现验证）。item_int 即使 missing 高也不删除，只通过 P001 填补。

#### 2.0.1 实际被 drop 的字段（基于 [sample_data_feature_summary.csv](../sample_data_feature_summary.csv) demo 1000 行统计）

**threshold = 0.75（默认，DOCX P000 原文）→ 删除 6 个 user_int 字段：**

| fid | missing | 语义猜测（[CSV](../sample_data_feature_summary.csv) `guess` 列）| 所属 NS 组 |
|---|---:|---|---|
| `user_int_feats_101` | **0.910** | 兴趣偏好二值 | U6 |
| `user_int_feats_102` | **0.877** | 兴趣偏好二值 | U6 |
| `user_int_feats_103` | **0.862** | 兴趣偏好等级 (3 档) | U6 |
| `user_int_feats_109` | **0.854** | 兴趣偏好等级 (7 档) | U6 |
| `user_int_feats_100` | **0.845** | 兴趣偏好二值 | U6 |
| `user_int_feats_99`  | **0.812** | 兴趣偏好二值 | U6 |

→ 6 个被删字段**全部来自 baseline `ns_groups.json` 的 U6 组**（"user_high_cardinality"，原 18 个 fid 中含 99~109）。drop 后：

```text
U6 (user_high_cardinality):    18 fids → 12 fids   (-33 %)
U1, U2, U3, U4, U5, U7:        不受影响（这些组的 fid missing 均 ≤ 0.6）
```

#### 2.0.2 高缺失 item_int 字段（保留但被 fill）

虽然 P000 不删 item_int，但数据分析显示 item 端有 **3 个字段 missing = 0.832**（[CSV](../sample_data_feature_summary.csv) 第 74-76 行）：

| fid | missing | 语义 | FE-00 处理 |
|---|---:|---|---|
| `item_int_feats_83` | 0.832 | 二级行业 (22 类) | P001 用均值填（保留，但 83 % 的样本是同一个填充值）|
| `item_int_feats_84` | 0.832 | 行业子类 (66 类) | P001 用均值填 |
| `item_int_feats_85` | 0.832 | 广告主分组 (103 类) | P001 用均值填 |

这 3 个 fid 集体 missing=0.832 强烈暗示它们来自**同一上游字段链路**（同时缺失或同时存在）。**首版 PlanB 不去主动 drop 它们**——保留 DOCX P000 原文边界；如果 ablation 显示这 3 个 fid 是负贡献，可在 follow-up 实验中手动从 schema 移除（不是 FE-00 的工作）。

#### 2.0.3 阈值敏感度（PlanB 可选参数）

| threshold | drop user_int 数 | drop 集合 | 推荐场景 |
|---:|---:|---|---|
| 0.90 | 1 | 101 | 极保守 |
| 0.85 | 4 | 101, 102, 103, 109 | 仅删极端高缺失 |
| **0.75（默认）** | **6** | **99, 100, 101, 102, 103, 109** | **DOCX P000 原文阈值；PlanB 选用** |
| 0.50 | 11 | + 86, 96, 60, 94, 108 | 太激进——会误删消费/活跃等级 6 档（user_int_feats_94，missing=0.521，可能是有用信号）|

→ **PlanB 默认 `--missing_threshold 0.75`**，与 DOCX 原文一致。

#### 2.0.4 输出审计文件（`fe00_dataset/`）

```text
dropped_user_int_fids.json           ← ★ 训练前必看：列出实际被 drop 的 6 个 user_int fid
                                       预期内容（demo 验证）: [99, 100, 101, 102, 103, 109]
feature_missing_report.json          ← 全部 user_int_feats_* 的 missing_ratio（与 CSV 一致性核查）
int_fill_values.json                 ← 每个 int 字段填补用的均值（含 item_int_feats_83/84/85）
dense_normalization_stats.json       ← 每个 dense 字段的 mean/std/n
schema.json                          ← 移除 6 个 user_int fid 后的 schema（喂给 11thmay_a build）
ns_groups.fe00.json                  ← U6 从 18 → 12 fids，其它组不变
docx_alignment.fe00.json             ← P000/P001/P002 对齐说明
```

#### 2.0.5 与 baseline NS 几何的连锁影响

drop 6 个 user_int 字段后，**user_int 总维度变小**（46 → 40），但 NS token 数 `--user_ns_tokens 5` **不变**。RankMixerNSTokenizer 的 chunk 分布会自动调整：

```text
A0 baseline:           total_emb_dim = 46×64 = 2944, chunk_dim = ⌈2944/5⌉ = 589
PlanB (FE-00 + 14 dense): total_emb_dim = 40×64 = 2560, chunk_dim = ⌈2560/5⌉ = 512
```

→ chunk 容量从 9.2 fid/chunk 降到 8 fid/chunk，**每个 chunk 更专注**，理论上还有微小信号粒度提升（这部分增益已隐含在 G0 = +0.001121 的实测里）。`d_sub = 64/16 = 4` 依然不变；ffn_only 模式下不读 d_sub 也无所谓。

### 2.1 FE-01A 频次基线（2 个，eval 已验证 +0.001577 vs B0）

| fid | 字段名 | 计算公式 |
|---:|---|---|
| 110 | `user_dense_feats_110` | `log1p(user_total_frequency_before_timestamp)` |
| 86 | `item_dense_feats_86` | `log1p(item_total_frequency_before_timestamp)` |

frequency 是 prefix-friendly 流式累加（FE-01A 已验证 eval 端可复现），只读 `(user_id, item_id, timestamp)`，不读 label。

### 2.2 11thMay 时间桶命中（12 个，新增）

每个样本先按 **UTC** 从 `timestamp` 抽取 3 个时间属性（向量化 numpy，无 Python `datetime.fromtimestamp` 循环）：

```python
sample_hour_bkt ∈ {0=morning [6,12), 1=afternoon [12,18), 2=evening [18,6)}
sample_dow      ∈ {0..6}  Monday=0, derived from (epoch_day + 3) % 7
sample_month    ∈ {0..11} January=0
```

对每个 `domain ∈ {a, b, c, d}` 扫描 seq 时间戳列（`event_ts ≤ sample_ts` 且 `event_ts > 0`），统计命中数后 `log1p`：

| fid | 字段名 | 含义 |
|---:|---|---|
| 113 | `user_dense_feats_113` | `log1p(Σ_e∈seq_a 𝟙[hour_bkt(e)==sample_hour_bkt])` |
| 114 | `user_dense_feats_114` | 同上，domain_b |
| 115 | `user_dense_feats_115` | 同上，domain_c |
| 116 | `user_dense_feats_116` | 同上，domain_d |
| 117 | `user_dense_feats_117` | `log1p(Σ_e∈seq_a 𝟙[dow(e)==sample_dow])` |
| 118 | `user_dense_feats_118` | 同上，domain_b |
| 119 | `user_dense_feats_119` | 同上，domain_c |
| 120 | `user_dense_feats_120` | 同上，domain_d |
| 121 | `user_dense_feats_121` | `log1p(Σ_e∈seq_a 𝟙[month(e)==sample_month])` |
| 122 | `user_dense_feats_122` | 同上，domain_b |
| 123 | `user_dense_feats_123` | 同上，domain_c |
| 124 | `user_dense_feats_124` | 同上，domain_d |

### 2.3 timestamp 与 ts_fid 边界处理

| 情况 | 处理 |
|---|---|
| `sample_ts ≤ 0`（脏数据）| 三个时间属性置 -1 sentinel；12 命中特征值全 0 |
| `event_ts > sample_ts`（防御性）| 严格过滤，不计入 |
| `event_ts == 0`（padding）| 严格过滤，不计入 |
| domain 没有解析到 ts_fid（[build:_DOMAIN_TS_FID_FALLBACK](../build_feature_engineering_dataset.py)）| 该域 3 个特征全 0，PlanB 退化为 9 个有效时间桶 |

默认 ts_fid fallback（在 schema 没指定时使用）：

```python
_DOMAIN_TS_FID_FALLBACK = {
    "domain_a": 46,   # 未验证
    "domain_b": 88,   # 未验证
    "domain_c": 47,   # 未验证（demo 看到 seq_27 / seq_30 也是候选）
    "domain_d": 26,   # ✓ worktree FE-01B 已验证
}
```

→ **正式跑时由平台 schema.json 中的 `seq[<domain>].ts_fid` 覆盖**；fallback 只是当 ts_fid 为 None 时的保底。

### 2.4 归一化

全部 `log1p + z-score`，z-score 用前 90% row group 拟合（与 train.py 默认 valid_ratio=0.1 严格对齐）：

```bash
--fit_stats_row_group_ratio 0.9
```

eval 端复用 checkpoint 内的 `feature_engineering_stats.json`，**不重新拟合**。

---

## 3. 执行步骤（生产环境）

### Step A：一键准备数据（FE-00 + 11thmay_a 链式预处理）

```bash
bash build_11thmay_b_pipeline.sh \
    --input_dir   /path/to/original_dataset \
    --input_schema /path/to/original_dataset/schema.json \
    --output_dir  /path/to/11thmay_b_dataset
```

可选参数：

```bash
    --missing_threshold       0.75    # FE-00 user_int drop 阈值（默认 0.75，等同 P000）
    --temporal_max_len        1024    # 11thmay_a 时间桶 seq 扫描长度（默认 1024）
    --fit_stats_row_group_ratio 0.9   # train 切片比例（与 train.py valid_ratio=0.1 对齐）
    --intermediate_dir        /path   # FE-00 中间产物目录（默认 <output>_fe00）
```

或者手动分两步（便于调试 + 复用中间产物）：

```bash
# Step A0: FE-00 preprocess
python3 -u build_fe00_preprocess_dataset.py \
    --input_dir   /path/to/original_dataset \
    --input_schema /path/to/original_dataset/schema.json \
    --output_dir  /path/to/fe00_dataset \
    --ns_groups_json ns_groups.json \
    --missing_threshold 0.75 \
    --fit_stats_row_group_ratio 0.9

# Step A1: 11thmay_a feature augmentation (input 必须用 FE-00 输出)
python3 -u build_feature_engineering_dataset.py \
    --input_dir   /path/to/fe00_dataset \
    --input_schema /path/to/fe00_dataset/schema.json \
    --output_dir  /path/to/11thmay_b_dataset \
    --feature_set 11thmay_a \
    --fit_stats_row_group_ratio 0.9 \
    --temporal_max_len 1024
```

输出文件（含 FE-00 与 11thmay_a 两层审计产物）：

```text
fe00_dataset/                             ← Step A0 中间产物
├── *.parquet                            # 已 drop / fill / z-score 的 parquet
├── schema.json                          # 已删除 6 个 user_int 高缺失 fid（默认: 99/100/101/102/103/109）
├── ns_groups.fe00.json                  # U6 组 18→12 fids，其它组不变
├── dropped_user_int_fids.json           # ★ 被删除的 6 个 fid 清单（与数据分析 CSV 交叉验证）
├── feature_missing_report.json          # 全部 user_int_feats_* 的 missing_ratio
├── int_fill_values.json                 # 含 item_int_feats_83/84/85 的填补均值
├── dense_normalization_stats.json
└── docx_alignment.fe00.json

11thmay_b_dataset/                        ← Step A1 最终产物（喂训练）
├── *.parquet                            # 增广后的训练数据（FE-00 + 14 个 dense 列）
├── schema.json                          # 增量 user_dense=[110,113..124], item_dense=[86]
├── ns_groups.feature_engineering.json
├── feature_engineering_stats.json       # 14 个 dense 的 mean/std + ts_cols + tz=UTC
├── docx_alignment.fe01.json
└── docx_alignment.11thmay_a.json        # 12 个时间桶的映射文档
```

### Step B: 训练

```bash
bash run_11thmay_b.sh \
    --data_dir   /path/to/11thmay_b_dataset \
    --schema_path /path/to/11thmay_b_dataset/schema.json \
    --ckpt_dir   outputs/exp_11thmay_b/ckpt \
    --log_dir    outputs/exp_11thmay_b/log \
    --seed 42
```

### Step C: 推理（用 FE_A 现有 infer_p0.py，无需新写）

```bash
python3 -u experiment_plans/FE_A/infer_p0.py \
    --ckpt_dir outputs/exp_11thmay_b/ckpt/best_model \
    --eval_dir /path/to/eval_data \
    --output /path/to/predictions.parquet
```

`infer_p0.py` 自动从 `train_config.json` 重建模型（含 `rank_mixer_mode=ffn_only` 与 `item_dense_dim>0`）。

---

## 4. 预测结果与影响

### 4.1 已实测的三个增益锚点

| 实验 | 配置 | leaderboard eval AUC | Δ vs A0 |
|---|---|---:|---:|
| A0 (baseline) | full mode | 0.810000 *(假定)* | — |
| **FE-00** | **drop missing + fill + z-score** | **0.811646** | **+0.1121 %** |
| **P0** | **ffn_only** + 同 A0 几何 | **0.812364** | **+0.2364 %** |
| FE-01A | full mode + (110, 86) | 0.812102 | +0.157 % |

### 4.2 11thMay-PlanB 收益分解（基于实测锚点 + PlanA 时间桶估算）

| 增益来源 | 来源依据 | 量级 |
|---|---|---:|
| **G0** FE-00 预处理 (drop missing>0.75 + fill + z-score) | FE-00 实测 +0.1121 %（独立 ablation）| **+0.0008 ~ +0.0012** |
| **G1** ffn_only | P0 实测 +0.2364 %（同 A0 几何，PlanB 现也是同几何）| **+0.0018 ~ +0.0024** |
| **G2** FE-01A 频次（110 + 86）| FE-01A 实测 +0.0016（full 模式） | **+0.0014 ~ +0.0017** |
| **G3** 12 时间桶（113..124）| 11thMay-PlanA 估算 | **+0.0005 ~ +0.0012** |
| G0×G2 协同损耗 | G0 已 z-score 部分 dense 字段，G2 又是 dense 字段叠加 | **-0.0002 ~ -0.0004** |
| G1×G2 协同损耗 | 两者都改进 dense 通路 | **-0.0003 ~ -0.0005** |
| G1×G3 弱正协同 | ffn_only 让 user_dense_token 更易被 Q-gen attend | **+0.0001 ~ +0.0003** |

```text
合计 ΔAUC vs A0 = +0.0041 ~ +0.0059
预期 eval AUC   = 0.8146 ~ 0.8164
70 % 置信区间   = 0.8134 ~ 0.8175
```

### 4.3 与已实测档的距离

```text
PlanB 期望中值 ≈ 0.8155
├── 比 FE-00   (已实测 0.8116) 高 ~+0.0039
├── 比 FE-01A  (已实测 0.8121) 高 ~+0.0034
└── 比 P0      (已实测 0.8124) 高 ~+0.0031
```

### 4.4 各模块对训练吞吐 / 显存的影响

| 维度 | A0 baseline | 11thMay-PlanB | Δ |
|---|---:|---:|---:|
| 总参数 | ~160.94 M | ~160.94 M + 1.5 k | +0.001 % |
| 单 step CPU 端耗时（B=256, 数据加载）| ~5 ms | ~5 ms + 8~12 ms（12 时间桶向量化）| +160 ~ +240 % per-batch dataset 端 |
| 但 dataset 端在多 worker (8) 下并行 | — | 实际 GPU steps/sec | ≤ -2 % |
| GPU 单 step 耗时（B=256, d=64, 6 epoch full）| 不变 | 不变（仅 user_dense_proj 输入维度多 12 列） | < +0.1 % |
| GPU 显存峰值 | 不变 | + (B × 14 × 4 B) ≈ +14 KB | 忽略 |
| 训练总时长（6 epoch）| ~6 h | ~6 h | ≈ 0 |

### 4.5 失败兜底（如果实测 ΔAUC < 期望）

| 实测情况 | 解读 | 下一步 |
|---|---|---|
| ΔAUC ∈ [+0.0030, +0.0040] | G0+G1+G2+G3 部分协同符合预期 | 进 PlanB2 (时间桶 → qgen_cond 通道，补回 NS↔Q mixing) |
| ΔAUC ∈ [+0.0015, +0.0030] | G3 估算偏乐观；或 G0 与 G2 重叠多 | 进 PlanB1 (剪 fid 121..124 月份 4 列) |
| ΔAUC ∈ [+0.0005, +0.0015] | 多个增益叠加损耗大 | 检查 `dropped_user_int_fids.json` + dense stats；考虑跳过 FE-00 单跑 P0+11thmay_a |
| ΔAUC < +0.0005 | ffn_only 跨 num_ns（8→9）迁移失败 | 退回 PlanA: full mode + 只加 FE-01A 频次 |
| ΔAUC < 0（掉点）| F11 风险触发：domain ts_fid 解析错误 → 命中率全为 0 噪声；或 FE-00 误删高价值 user_int | 检查 `feature_engineering_stats.json.temporal_domain_ts_cols` + `dropped_user_int_fids.json` |

---

## 5. 风险清单与已采取的修复

| ID | 风险 | 严重度 | 修复状态 |
|---|---|---|---|
| **F1** | dataset.py 不支持 item_dense | 🔴→🟢 已修复 | 4 处 patch 已落码 |
| **F2** | 主仓没有 build 脚本 | 🔴→🟢 已修复 | 移植 + 扩展 11thmay_a |
| F3 | schema.json | 🟢 平台提供 | 不处理 |
| **F4** | train.py 没有 `--feature_set` | 🟢 已规避 | feature_set 仅在 build 阶段使用 |
| **F5** | group vs rankmixer 路径混淆 | 🟢 已规避 | PlanB 锁定 rankmixer，与 P0 同 |
| **F6** | run_11thmay_b.sh 不存在 | 🟢 已修复 | 已创建 |
| **F7** | 时区 UTC+8 vs UTC | 🟢 已修复 | 改为 UTC 与 demo 数据一致 |
| F8 | infer 脚本 | 🟢 已规避 | 复用 FE_A/infer_p0.py |
| F9 | 跨配置 ffn_only 迁移 | 🟢 已规避 | 用 rankmixer 路径完全锁定 P0 几何 |
| **F10** | datetime 循环耗时 | 🟢 已修复 | 纯 numpy 向量化（< 10 ms / batch）|
| F11 | domain_a/b ts_fid 未公开 | 🟡 接受 | fallback 表 + 缺失自动 0；正式跑由平台 schema 覆盖 |
| F12 | ns_groups.json 与 rankmixer 无关 | 🟢 文档说明 | run script 已注释 |
| **F13** | FE-00 缺失（drop 高缺失 user_int + fill + dense z-score）| 🟠→🟢 已修复 | 移植 `build_fe00_preprocess_dataset.py` + 流水线脚本 |

### 5.3 FE-00 链式预处理的额外风险

| ID | 风险 | 严重度 | 修复 / 监控 |
|---|---|---|---|
| **F13a** | 误删高价值 user_int fid（如某 fid missing=0.76 但单独贡献高）| 🟡 中 | 训练前打印 `dropped_user_int_fids.json`；若 drop 超过 6 个，先用 `--missing_threshold 0.80` 收紧 |
| **F13b** | int fill 用 mean round 后等于某真实 ID → embedding 冲撞 | 🟡 中 | 训练后看 oob 报告；监控 dropped fid + filled fid 上 train loss 曲线 |
| **F13c** | dense z-score 把 user_dense_feats_61/87 等已预训练 embedding 的语义打乱 | 🟢 低 | FE-00 实测 +0.001121 表明净效应是正向的；如果掉点，从 `dense_normalization_stats.json` 看是哪些列被改 |
| **F13d** | 11thmay_a 在 FE-00 输出上再做 z-score 拟合，与 FE-00 z-score 链式叠加 | 🟢 低 | 互不冲突：FE-00 z-score 原 dense 列；11thmay_a z-score 新增的 14 列。两次 z-score 独立 |
| **F13e** | FE-00 + 11thmay_a 两次扫数据，总预处理耗时 ~2 h（单机 200M 行）| 🟢 接受 | 平台一次性预处理，后续训练 / ablation 共用 |

### 5.1 剩余风险：T=17 / 模式锁定的连锁影响

**风险**：本方案启用 item_dense 后 num_ns = 9（多了 item_dense_token），T = 17。`d_model=64` 不整除 17 → **必须**用 `ffn_only` 模式。

**反过来说**：如果后续想切回 `full` 模式做对照实验（比如纯 G2 + G3 不带 G1），有两条路径：

1. 关掉 item_dense_feats_86（用空 list 覆盖 schema item_dense），num_ns 回到 8，T=16，`full` 可用。但这等于丢掉了 G2 的一半。
2. 把 num_queries 改为 1 或 3：T = 1*4+9 = 13 或 3*4+9 = 21；64 % 13 = 12, 64 % 21 = 1。**都不整除**。所以 full 模式下根本无法启用 item_dense=1 与 num_q=2 的组合。

→ **结论**：PlanB 的 `ffn_only` 选择不是可选项而是必选项。这进一步固化了 P0 路径的合理性。

### 5.2 剩余风险：稀疏域命中率

domain_a (5.4%) / domain_c (1.5%) 的 7d 内事件占比极低，月份桶可能命中率 < 0.1 %。极端样本下 12 个特征中可能有 2~4 个 nonzero ratio < 0.5%。

**监控指标**（启动后第 1 batch dump）：

```text
对每个 fid ∈ [113..124]：
  nonzero_ratio = (count of rows where feature > 0) / B
  mean / std    = （后处理 z-score 前）
```

判定门槛：

- nonzero_ratio ≥ 5 % → 保留
- nonzero_ratio ∈ [0.5%, 5%] → 接受首版，下一档 ablation 中评估剪枝
- nonzero_ratio < 0.5 % → 该列退化为常量列，对 AUC 贡献 ≈ 0；下一档剔除

---

## 6. 完整影响矩阵

| 改动 | 对训练正确性 | 对训练速度 | 对显存 | 对 ckpt 兼容 |
|---|---|---|---|---|
| dataset.py 启用 item_dense | 零（向后兼容）| 零（仅 schema 不空时多一次 pad）| < 1 % | A0 ckpt 可用 `strict=False` load |
| build script feature_set=11thmay_a | 离线一次性 | 离线一次性（~1 h on 完整集，14 列 × 全行）| — | — |
| 12 时间桶 numpy 向量化 | 零 | dataset 端 +8~12 ms / batch；GPU 端不变 | 忽略 | — |
| `--rank_mixer_mode ffn_only` | 零（已在 train.py / model.py） | GPU 端 ≈ 0；理论上 ffn_only 比 full 略快 1~2 % | 略降（去掉 reshape 中间张量）| 必须存进 train_config.json 供 infer 重建 |
| 14 个 dense 字段输入 user/item_dense_proj | 零 | GPU 端 < +0.1 % | +14 KB | — |

---

## 7. 上传平台前的最后核对

- [ ] `dataset.py` 4 处 patch 都在（搜 `11thMay-PlanB`，应有 4 处注释）
- [ ] `build_fe00_preprocess_dataset.py` 存在于项目根；`--help` 列出 `--missing_threshold`
- [ ] `build_feature_engineering_dataset.py` 存在于项目根；`--help` 列出 `11thmay_a` 选项
- [ ] `build_11thmay_b_pipeline.sh` 可执行（`ls -la` 看到 `x` 权限位）
- [ ] `run_11thmay_b.sh` 可执行（`ls -la` 看到 `x` 权限位）
- [ ] `train.py` 不需要任何改动；`--rank_mixer_mode ffn_only` 已支持
- [ ] `model.py` 不需要任何改动
- [ ] 平台 schema.json 含 `user_dense` 与 `item_dense` 顶层 key（脚本会按 `raw.get('item_dense', [])` 健壮处理）
- [ ] 平台 `ns_groups.json` 存在（FE-00 会读它来同步 drop 高缺失 fid 对应的 NS 分组）

### 7.1 平台运行命令模板（一键流水线）

```bash
# Step A: FE-00 + 11thmay_a 链式数据预处理（一条命令）
bash build_11thmay_b_pipeline.sh \
    --input_dir   $INPUT_DATA_DIR \
    --input_schema $INPUT_DATA_DIR/schema.json \
    --output_dir  $OUTPUT_DATA_DIR \
    --missing_threshold 0.75 \
    --temporal_max_len 1024 \
    --fit_stats_row_group_ratio 0.9

# Step B: train (在 GPU 机器上跑)
bash run_11thmay_b.sh \
    --data_dir   $OUTPUT_DATA_DIR \
    --schema_path $OUTPUT_DATA_DIR/schema.json \
    --ckpt_dir   $CKPT_DIR \
    --log_dir    $LOG_DIR \
    --num_epochs 6 \
    --patience 3 \
    --seed 42

# Step C: infer (推理脚本无需任何改动)
python3 -u experiment_plans/FE_A/infer_p0.py \
    --ckpt_dir $CKPT_DIR/best_model \
    --eval_dir $EVAL_DATA_DIR \
    --output predictions.parquet
```

### 7.2 手动分步运行（便于调试）

```bash
# Step A0: FE-00 preprocess
python3 -u build_fe00_preprocess_dataset.py \
    --input_dir   $INPUT_DATA_DIR \
    --input_schema $INPUT_DATA_DIR/schema.json \
    --output_dir  ${OUTPUT_DATA_DIR}_fe00 \
    --ns_groups_json ns_groups.json \
    --missing_threshold 0.75 \
    --fit_stats_row_group_ratio 0.9

# Step A1: 11thmay_a feature augmentation (input 必须用 FE-00 输出)
python3 -u build_feature_engineering_dataset.py \
    --input_dir   ${OUTPUT_DATA_DIR}_fe00 \
    --input_schema ${OUTPUT_DATA_DIR}_fe00/schema.json \
    --output_dir  $OUTPUT_DATA_DIR \
    --feature_set 11thmay_a \
    --fit_stats_row_group_ratio 0.9 \
    --temporal_max_len 1024
```

---

## 8. 后续 ablation 路径（PlanB 结果出来后）

```text
PlanB → eval ≥ 0.8140?
  ├─ 是 → PlanB+A1 (叠加 FE-01B 89/90/91/92)
  │       预期再 +0.0003 ~ +0.0008
  │
  ├─ 是 → PlanB2 (时间桶进 query_generator condition)
  │       预期再 +0.0010 ~ +0.0025（补回 ffn_only 下丢失的 NS↔Q mixing）
  │
  └─ 否 → PlanB1 (剪 121..124 月份 4 列；保留 hour + dow 共 8 桶)
          预期与 PlanB 持平或 +0.0002
```

**长期方向**：与 FE-07 Domain-main、FE_A 系列 P3' (qgen_cond + ffn_only) 并跑取 max。

---

## 9. 总结

```text
方案核心:    FE-00 预处理 + 1 行 CLI flag + 14 列 dense 特征
代码改动:    dataset.py +18/-3 行 + 3 个新文件（FE-00 + 11thmay_a + 流水线）
模型改动:    零（train.py / model.py 都不动）
风险等级:    低（所有红色阻塞已修复，F11 / F13a-e 已设监控）
预期增益:    +0.41 ~ +0.59 % AUC over A0 (= eval 0.8146 ~ 0.8164)
失败兜底:    5 档 ablation 路径已规划（§4.5）

四个增益来源（按贡献排序）:
  G1 ffn_only        +0.236 % (P0 实测)        ← 最强
  G2 FE-01A 频次     +0.157 % (FE-01A 实测)    ← 次强
  G0 FE-00 预处理    +0.112 % (FE-00 实测)     ← 第三
  G3 12 时间桶       +0.05~+0.12 % (估算)      ← 唯一估算项
```
