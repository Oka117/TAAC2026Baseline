# FE08 代码结构搭建指南（供 AI Agent 使用）

> 本文档面向**未读过 5 月 7 日方案讨论历史**的 AI Agent。
> 主方案：[5月7日_GNN结合验证特征_保持NS影响域_方案.md](5月7日_GNN结合验证特征_保持NS影响域_方案.md)
>
> 你的任务：在 `TAAC2026Baseline` 仓库（worktree 根目录）下，按以下结构建立 FE08 全套代码。
>
> 完成后产物：`build_fe08_may7_dataset.py` 训练 builder + `evaluation/FE08/` 评估套件 +
> `run_fe08_may7_full.sh` 入口脚本，可在平台上一键复现 5 月 7 日方案的 Eval AUC 0.8185~0.8215。

## 0. 建造前 sanity check

在动手前确认以下事实成立：

```text
[ ] 当前目录 = /path/to/TAAC2026Baseline 或其 worktree
[ ] 存在 dataset.py / model.py / train.py / trainer.py 主代码
[ ] 存在 build_fe07_p012_domain_dataset.py（FE-07 builder，本方案 fork 起点）
[ ] 存在 evaluation/FE07/ 目录（FE-07 评估套件，本方案 fork 起点）
[ ] git 当前在 claude/suspicious-dijkstra-dc9c87 或 tokenGNN_4L_optimis 分支
```

## 1. 文件清单（共需新增 6 个 + 修改 2 个）

### 1.1 新增文件

| 文件 | 来源 | 用途 |
| --- | --- | --- |
| `build_fe08_may7_dataset.py` | fork from `build_fe07_p012_domain_dataset.py` | 训练侧 builder |
| `tools/build_fe08_may7_dataset.py` | 同上副本 | 平台上传副本 |
| `run_fe08_may7_full.sh` | 新建 | 端到端 entry point |
| `evaluation/FE08/build_fe08_may7_dataset.py` | 同 builder | eval-side builder |
| `evaluation/FE08/dataset.py` | fork from `evaluation/FE07/dataset.py` | eval dataset wrapper |
| `evaluation/FE08/model.py` | fork from `evaluation/FE07/model.py` | eval-side model |
| `evaluation/FE08/infer.py` | fork from `evaluation/FE07/infer.py` | eval entry point |

### 1.2 修改文件

| 文件 | 改动 |
| --- | --- |
| `train.py` | 加 `--item_dense_fids` / `--enable_risky_item_dense_fids` CLI flag；加 `--seq_top_k` 与 `--seq_encoder_type` 不一致告警 |
| `trainer.py` | 在 sidecar 复制列表中加入 `fe08_*` 系列 |

### 1.3 不动的文件

```text
dataset.py    : 现有 PCVRParquetDataset 已支持 item_dense schema、split_by_timestamp、
                domain_time_buckets 全部接口；FE08 不需要改
model.py      : 现有 PCVRHyFormer + TokenGNN 已支持 use_token_gnn / token_gnn_layers /
                token_gnn_layer_scale / has_item_dense；FE08 不需要改
ns_groups.json: 仅作示例参考；本方案在 builder 中生成 ns_groups.may7.json，不覆盖
```

## 2. 关键参数（必须严格一致）

```python
# 这些值在主方案 §0/§5 已锁定，AI Agent 不得擅自修改
USER_NS_TOKENS = 5
ITEM_NS_TOKENS = 2
NUM_QUERIES = 2
NUM_SEQUENCES = 4              # domain a/b/c/d
HAS_USER_DENSE = True          # baseline schema 已有 user_dense
HAS_ITEM_DENSE = True          # 本方案启用，添加 fid {86, 91, 92}
NUM_NS = USER_NS_TOKENS + 1 + ITEM_NS_TOKENS + 1   # = 9
T = NUM_QUERIES * NUM_SEQUENCES + NUM_NS           # = 17
D_MODEL = 136                                       # = 17 × 8
EMB_DIM = 64
NUM_HEADS = 4
NUM_HYFORMER_BLOCKS = 2
DROPOUT_RATE = 0.05
SEQ_TOP_K = 100                                     # marker only, transformer 主线下不生效

assert D_MODEL % T == 0, f"d_model={D_MODEL} must be divisible by T={T}"
```

```bash
# 序列长度
SEQ_MAX_LENS = "seq_a:256,seq_b:256,seq_c:128,seq_d:512"

# 缺失阈值
MISSING_THRESHOLD = 0.80                # 同时作用于 user_int 与 item_int

# Match feature 桶边界
MATCH_COUNT_BUCKETS = "0,1,2,4,8"       # 6 桶 + padding → vocab=7
MATCH_WINDOW_DAYS = 7                    # match_count_7d 时间窗口

# item_dense 安全白名单
ITEM_DENSE_FIDS = [86, 91, 92]
RISKY_ITEM_DENSE_FIDS = {
    87: "FE-01 全量 eval AUC 跌到 0.775054 主因（train/eval 不一致）",
    88: "依赖 label_time，存在泄漏风险（FE-02 范围）",
}

# TokenGNN 配置
USE_TOKEN_GNN = True
TOKEN_GNN_LAYERS = 4
TOKEN_GNN_GRAPH = "full"
TOKEN_GNN_LAYER_SCALE = 0.15
```

## 3. 数据流图

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ 原始 parquet (TRAIN_DATA_PATH)                                          │
│   ├─ user_int_feats_*  / user_dense_feats_*                             │
│   ├─ item_int_feats_*  / item_dense_feats_*  (baseline 没有这一组)      │
│   ├─ domain_a_seq_* / domain_b_seq_* / domain_c_seq_* / domain_d_seq_*  │
│   └─ user_id, item_id, label_type, label_time, timestamp                │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ build_fe08_may7_dataset.py
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 1 — Audit                                                         │
│   - 扫描 user_int_feats_* / item_int_feats_* 的 missing_ratio           │
│   - 删除 missing_ratio > 0.80 的 fid → dropped_feats.may7.json          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 2 — Fit dense norm stats (train row groups only)                  │
│   - 对所有 user_dense_feats_* fit (mean, std)                           │
│   - 对新增 item_dense_feats_{86, 91, 92} fit (mean, std)                │
│   - 对新增 user_dense_feats_120/121 (如继承 FE-07 P0 dense block) fit   │
│   - 输出 → fe08_dense_normalization_stats.json                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 3 — Compute generated features (per batch)                        │
│   - user_total_freq → user_dense_feats_110 (FE-01A，可选)               │
│   - item_total_freq → item_dense_feats_86                               │
│   - has_match(item_int_9, domain_d_seq_19)        → item_int_feats_89   │
│   - bucketize match_count                          → item_int_feats_90  │
│   - log1p(min_match_delta)                         → item_dense_feats_91│
│   - log1p(match_count_7d)                          → item_dense_feats_92│
│   - 最近匹配 time bucket id (BUCKET_BOUNDARIES)    → item_int_feats_91  │
│   - hour_of_day = (ts // 3600) % 24 + 1            → user_int_feats_130 │
│   - day_of_week = ((ts//86400)+4) % 7 + 1          → user_int_feats_131 │
│   - 注意：item_int_feats_91 与 item_dense_feats_91 共存（不同列）       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 4 — Sequence sort by recency                                      │
│   - 对每个 domain (a/b/c/d) 的每行：                                    │
│     argsort = np.argsort(-event_timestamp)  # descending                │
│     用同一份 permutation 重排所有 side feature 列 + ts 列               │
│   - 严格保证：list 长度不变；length=0 list 跳过                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 5 — Write augmented parquet + sidecars                            │
│   - 增强 parquet 写到 FE08_DATA_DIR/                                    │
│   - schema.json (含新增 fid)                                            │
│   - ns_groups.may7.json                                                 │
│   - dropped_feats.may7.json                                             │
│   - fe08_dense_normalization_stats.json                                 │
│   - fe08_transform_stats.json                                           │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ train.py 训练
┌─────────────────────────────────────────────────────────────────────────┐
│ PCVRHyFormer (d_model=136, full mode, 4-layer TokenGNN)                 │
│   - NS tokens : (B, 9, 136)                                             │
│   - TokenGNN  : message passing on (B, 9, 136), 4 layers, scale=0.15    │
│   - HyFormer  : 2 blocks, 4 heads, num_queries=2, full mixing           │
│   - Output    : final Q tokens → CVR head                               │
│   - 训练完成 → ckpt + sidecar 全套 → /path/to/ckpt/                     │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼ evaluation/FE08/infer.py 评估
┌─────────────────────────────────────────────────────────────────────────┐
│ Eval pipeline                                                           │
│   - 读 ckpt + 全部 sidecar                                              │
│   - 用同一份 builder 逻辑 transform eval parquet                        │
│     (严禁 re-fit / re-select / 读 eval label)                           │
│   - 跑 forward → AUC                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

## 4. `build_fe08_may7_dataset.py` 详细规范

### 4.1 CLI 接口

```python
# 必须支持的所有参数（与 FE-07 保持一致 + FE08 新增）
parser.add_argument("--input_dir", required=True)
parser.add_argument("--input_schema", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--missing_threshold", type=float, default=0.80)        # FE08 改 0.80
parser.add_argument("--fit_stats_row_group_ratio", type=float, default=0.9)
parser.add_argument("--match_window_days", type=int, default=7)
parser.add_argument("--match_count_buckets", default="0,1,2,4,8")
parser.add_argument("--fill_empty_int_lists", action="store_true")

# FE08 新增 ↓
parser.add_argument("--item_dense_fids", default="86,91,92",
                    help="Comma-separated item_dense fids; default = FE-01B safe subset")
parser.add_argument("--enable_risky_item_dense_fids", action="store_true",
                    help="Allow risky fids 87/88; OFF by default for safety")
parser.add_argument("--sort_sequence_by_recency", action="store_true", default=True,
                    help="Sort sequences by event_timestamp descending; required by FE08 spec")
parser.add_argument("--no_sort_sequence_by_recency", dest="sort_sequence_by_recency",
                    action="store_false")
parser.add_argument("--diagnostic_row_group_limit", type=int, default=100)
```

### 4.2 关键函数签名

```python
# 1. fid 校验（白名单 + 双保险）
def parse_item_dense_fids(arg_str: str, allow_risky: bool) -> List[int]:
    fids = [int(x.strip()) for x in arg_str.split(",") if x.strip()]
    for fid in fids:
        if fid in RISKY_ITEM_DENSE_FIDS:
            if not allow_risky:
                raise ValueError(
                    f"item_dense_feats_{fid} is RISKY: {RISKY_ITEM_DENSE_FIDS[fid]}. "
                    f"Pass --enable_risky_item_dense_fids to override."
                )
            log.warning(f"RISK ACCEPTED: enabling item_dense_feats_{fid}")
    return fids


# 2. 序列排序（核心新增）
def sort_sequence_by_recency(
    batch: pa.RecordBatch,
    domain: str,
    ts_col: str,
    side_cols: List[str],
) -> Dict[str, pa.Array]:
    """对一个 domain 的所有 side feature + ts 按 event_time 降序重排。

    Args:
        batch: 原始 record batch
        domain: 'domain_a' / 'domain_b' / 'domain_c' / 'domain_d'
        ts_col: 该 domain 的 timestamp 列名 (e.g. 'domain_d_seq_26')
        side_cols: 该 domain 的所有 side feature 列名

    Returns:
        Dict[col_name -> sorted pa.ListArray]
    """
    ts_lists = batch.column(batch.schema.get_field_index(ts_col)).to_pylist()
    out = {}

    # 一次性读出所有 side 列（避免每行重复读列）
    side_lists = {col: batch.column(batch.schema.get_field_index(col)).to_pylist()
                  for col in side_cols}
    side_lists[ts_col] = ts_lists

    # 对每行求 permutation
    permutations = []
    for row_ts in ts_lists:
        if row_ts is None or len(row_ts) <= 1:
            permutations.append(None)  # 不需要排序
            continue
        ts_arr = np.array([t if t is not None else 0 for t in row_ts], dtype=np.int64)
        perm = np.argsort(-ts_arr)  # descending
        permutations.append(perm)

    # 应用 permutation
    for col_name, lists in side_lists.items():
        sorted_rows = []
        for row, perm in zip(lists, permutations):
            if row is None:
                sorted_rows.append(None)
            elif perm is None:
                sorted_rows.append(row)
            else:
                sorted_rows.append([row[i] for i in perm])
        out[col_name] = pa.array(sorted_rows, type=pa.list_(pa.int64()))

    return out


# 3. 最近匹配 time bucket id (item_int_feats_91)
def compute_latest_match_time_bucket(
    target_attr: int,
    seq_values: List[int],
    seq_times: List[int],
    sample_ts: int,
    bucket_boundaries: np.ndarray,
) -> int:
    """非匹配样本返回 0；否则返回 1..(len(boundaries)+1)。"""
    if target_attr <= 0:
        return 0
    deltas = []
    for v, t in zip(seq_values, seq_times):
        if int(v) != target_attr or t <= 0 or t > sample_ts:
            continue
        deltas.append(sample_ts - int(t))
    if not deltas:
        return 0
    delta_min = min(deltas)
    raw = int(np.searchsorted(bucket_boundaries, delta_min))
    raw = min(raw, len(bucket_boundaries) - 1)  # clip
    return raw + 1  # 1..(len+1)


# 4. hour / dow with +1 offset
def compute_user_temporal(timestamps: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "user_int_feats_130": ((timestamps // 3600) % 24 + 1).astype(np.int64),    # 1..24
        "user_int_feats_131": (((timestamps // 86400) + 4) % 7 + 1).astype(np.int64),  # 1..7
    }


# 5. 增强 schema 构造
def build_augmented_schema(
    base_schema: Dict,
    audits: Dict[str, IntAudit],
    missing_threshold: float,
    item_dense_fids: List[int],
    match_count_vocab_size: int,
    num_time_buckets: int,
) -> Tuple[Dict, List[int], List[int]]:
    """返回 (augmented_schema, dropped_user_fids, dropped_item_fids)。

    NEW: dropped_item_fids 现在也参与 missing_threshold 删除（FE-08 比 FE-07 多）。

    schema 新增：
      user_int += [(130, 25, 1), (131, 8, 1)]                           # hour, dow
      item_int += [(89, 3, 1), (90, match_count_vocab_size, 1),
                   (91, num_time_buckets, 1)]                            # match suite
      item_dense += [(fid, 1) for fid in item_dense_fids]                # {86, 91, 92}
    """
    ...


# 6. ns_groups.may7.json 生成
def build_ns_groups_may7(schema: Dict) -> Dict:
    """注意：dense fid 不写入 ns_groups（dense 走 user/item_dense_proj）。

    user_ns_groups 中加入 130 / 131 → U2_user_temporal_behavior
    item_ns_groups 中加入 89 / 90 / 91 → I4_target_matching_fields
    自动剔除已被 missing>80% 删除的 fid（保持一致）。
    """
    base = {
        "_purpose": "FE-08 NS groups (May 7 plan).",
        "_note": "Dense fid not in this file. user_ns_tokens=5, item_ns_tokens=2, num_queries=2.",
        "user_ns_groups": {
            "U1_user_profile": [1, 15, 48, 49],
            "U2_user_temporal_behavior": [50, 60, 130, 131],     # ← 新增 130/131
            "U3_user_context": [51, 52, 53, 54, 55, 56, 57, 58, 59],
            "U4_user_dense_aligned": [62, 63, 64, 65, 66],
            "U5_user_interest": [80, 82, 86],
            "U6_user_long_tail": [89, 90, 91, 92, 93],
            "U7_user_high_card": [94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        },
        "item_ns_groups": {
            "I1_item_identity": [5, 6, 7, 8],
            "I2_item_category": [9, 10, 11, 12, 13],
            "I3_item_semantic": [16, 81, 83, 84, 85],
            "I4_target_matching_fields": [89, 90, 91],            # ← 新增 91
        },
    }
    # 自动剔除已被删除的 fid
    return _filter_to_existing_fids(base, schema)
```

### 4.3 sidecar 输出契约

builder 必须生成以下 sidecar，evaluation 严格依赖它们：

```json
// dropped_feats.may7.json
{
  "user_int": [...],
  "item_int": [...],
  "threshold": 0.80,
  "row_groups_used_for_audit": <int>,
  "interpretation": "Empty list means no fid hit the missing>=0.80 threshold."
}

// fe08_dense_normalization_stats.json
{
  "user_dense_feats_61":  {"mean": <float>, "std": <float>, "n": <int>},
  "user_dense_feats_87":  {"mean": <float>, "std": <float>, "n": <int>},
  ...
  "item_dense_feats_86":  {"mean": <float>, "std": <float>, "n": <int>},
  "item_dense_feats_91":  {"mean": <float>, "std": <float>, "n": <int>},
  "item_dense_feats_92":  {"mean": <float>, "std": <float>, "n": <int>}
}

// fe08_transform_stats.json
{
  "missing_threshold": 0.80,
  "match_window_days": 7,
  "match_count_buckets": [0, 1, 2, 4, 8],
  "match_count_vocab_size": 7,
  "item_dense_fids": [86, 91, 92],
  "enable_risky_item_dense_fids": false,
  "user_int_drops": [...],
  "item_int_drops": [...],
  "domain_match_columns": {
    "match_col": "domain_d_seq_19",
    "match_ts_col": "domain_d_seq_26"
  },
  "num_time_buckets": 64,
  "bucket_boundaries_md5": "<sha>",
  "sequence_sort_by_recency": true,
  "sort_order": "descending_by_event_timestamp",
  "min_train_timestamp": <int>,
  "fit_row_groups": <int>,
  "total_row_groups": <int>
}

// ns_groups.may7.json
// (见上 build_ns_groups_may7)

// schema.json (augmented)
{
  "user_int":  [[fid, vocab, dim], ...],   // 含新增 130/131
  "item_int":  [[fid, vocab, dim], ...],   // 含新增 89/90/91
  "user_dense": [[fid, dim], ...],
  "item_dense": [[86, 1], [91, 1], [92, 1]],  // 仅含 ITEM_DENSE_FIDS
  "seq": {...}                              // 不动
}
```

## 5. `evaluation/FE08/` 详细规范

### 5.1 `evaluation/FE08/build_fe08_may7_dataset.py`

```text
- 完全复用训练侧 build_fe08_may7_dataset.py 的所有 transform 函数；
- 但禁止 re-fit dense stats / re-select match pair / 读 eval label_type；
- 必须从 checkpoint 读 sidecar 后 transform eval parquet：
    1. dense norm 用 fe08_dense_normalization_stats.json 的 mean/std；
    2. drop 列用 dropped_feats.may7.json 的 user_int / item_int 列表；
    3. match column / window / buckets 用 fe08_transform_stats.json 的固定值；
    4. 序列排序用同一份 argsort 逻辑；
    5. user_int_130/131 / item_int_89/90/91 / item_dense_86/91/92 全部 deterministically 计算。
```

### 5.2 `evaluation/FE08/dataset.py`

```text
fork from evaluation/FE07/dataset.py，关键改动：
- 不需要 domain_time_buckets（FE08 不启用 P2-Domain）；
- num_time_buckets 仍走 dataset.NUM_TIME_BUCKETS=64；
- 其他逻辑保持。
```

### 5.3 `evaluation/FE08/model.py`

```text
fork from evaluation/FE07/model.py，关键改动：
- 不动；FE-07 的 model.py 已经包含 TokenGNN 完整实现。
- 检查点：use_token_gnn / token_gnn_layers=4 / token_gnn_layer_scale=0.15 / token_gnn_graph='full'
```

### 5.4 `evaluation/FE08/infer.py`

```text
fork from evaluation/FE07/infer.py，关键改动：
- model 构造时传 d_model=136、rank_mixer_mode='full'、user_ns_tokens=5、
  item_ns_tokens=2、num_queries=2；
- strict load checkpoint，禁止 missing/unexpected key 静默通过；
- 校验 checkpoint 的 train_config.json：
    assert train_config['d_model'] == 136
    assert train_config['rank_mixer_mode'] == 'full'
    assert train_config['user_ns_tokens'] == 5
    assert train_config['item_ns_tokens'] == 2
    assert train_config['num_queries'] == 2
    assert train_config['use_token_gnn'] == True
    assert train_config['token_gnn_layers'] == 4
- 加载 fe08 sidecars 全套（见 §4.3）
```

## 6. `run_fe08_may7_full.sh` 完整入口

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# FE-08 main entrypoint:
# FE08-May7-main = 0.8159 GNN-baseline replay
#                  + drop 80% missing
#                  + item_dense {86, 91, 92} token + norm
#                  + sequence sort by recency
#                  + item_int_89/90/91 (new)
#                  + user_int_130/131 (new)
#                  + seq_a/b/c/d = 256/256/128/512
#                  + rank_mixer_mode=full + d_model=136 + dropout=0.05
#                  + use_token_gnn (4 layers, full graph, scale=0.15)
#                  + split_by_timestamp

ORIG_DATA_DIR="${TRAIN_DATA_PATH:-}"
ORIG_SCHEMA_PATH=""

ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    case "${ARGS[$i]}" in
        --data_dir)
            if (( i + 1 < ${#ARGS[@]} )); then ORIG_DATA_DIR="${ARGS[$((i + 1))]}"; fi
            ;;
        --data_dir=*) ORIG_DATA_DIR="${ARGS[$i]#--data_dir=}" ;;
        --schema_path)
            if (( i + 1 < ${#ARGS[@]} )); then ORIG_SCHEMA_PATH="${ARGS[$((i + 1))]}"; fi
            ;;
        --schema_path=*) ORIG_SCHEMA_PATH="${ARGS[$i]#--schema_path=}" ;;
    esac
done

if [[ -z "${ORIG_DATA_DIR}" ]]; then
    echo "Missing data path. Set TRAIN_DATA_PATH or pass --data_dir." >&2
    exit 1
fi

if [[ -z "${ORIG_SCHEMA_PATH}" ]]; then
    ORIG_SCHEMA_PATH="${ORIG_DATA_DIR}/schema.json"
fi

FE08_ROOT="${FE08_DATA_DIR:-${TMPDIR:-/tmp}/taac_fe08_may7_$$}"
FE08_SCHEMA="${FE08_ROOT}/schema.json"
FE08_GROUPS="${FE08_ROOT}/ns_groups.may7.json"

echo "[FE-08] input data: ${ORIG_DATA_DIR}"
echo "[FE-08] input schema: ${ORIG_SCHEMA_PATH}"
echo "[FE-08] output data: ${FE08_ROOT}"

# Stage 1: build augmented dataset
python3 -u "${SCRIPT_DIR}/build_fe08_may7_dataset.py" \
    --input_dir "${ORIG_DATA_DIR}" \
    --input_schema "${ORIG_SCHEMA_PATH}" \
    --output_dir "${FE08_ROOT}" \
    --missing_threshold 0.80 \
    --match_window_days 7 \
    --match_count_buckets "0,1,2,4,8" \
    --fit_stats_row_group_ratio 0.9 \
    --diagnostic_row_group_limit 100 \
    --item_dense_fids "86,91,92" \
    --sort_sequence_by_recency

# Stage 2: train
TRAIN_DATA_PATH="${FE08_ROOT}" python3 -u "${SCRIPT_DIR}/train.py" \
    "$@" \
    --schema_path "${FE08_SCHEMA}" \
    --ns_groups_json "${FE08_GROUPS}" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --rank_mixer_mode full \
    --d_model 136 \
    --emb_dim 64 \
    --num_hyformer_blocks 2 \
    --num_heads 4 \
    --seq_encoder_type transformer \
    --seq_max_lens "seq_a:256,seq_b:256,seq_c:128,seq_d:512" \
    --use_time_buckets \
    --loss_type bce \
    --lr 1e-4 \
    --sparse_lr 0.05 \
    --dropout_rate 0.05 \
    --seq_top_k 100 \
    --batch_size 256 \
    --num_workers 8 \
    --buffer_batches 20 \
    --valid_ratio 0.1 \
    --split_by_timestamp \
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

## 7. `train.py` 修改清单（最小改动）

### 7.1 加 seq_top_k sanity check

```python
# 在 main() 起始处，args 解析后
if args.seq_top_k > 0 and args.seq_encoder_type != 'longer':
    logging.warning(
        f"--seq_top_k={args.seq_top_k} is set but seq_encoder_type="
        f"{args.seq_encoder_type} → this flag is a NO-OP. "
        f"Use --seq_encoder_type=longer to activate it."
    )
```

### 7.2 整除约束 sanity check

```python
# 模型构造前
T = args.num_queries * len(pcvr_dataset.seq_domains) + model.num_ns
if args.rank_mixer_mode == 'full' and args.d_model % T != 0:
    raise ValueError(
        f"FATAL: d_model={args.d_model} must be divisible by T={T} in full mode. "
        f"Current FE08 spec uses d_model=136, T=17, 136%17=0."
    )
```

### 7.3 不需要新增 CLI（item_dense fids 由 builder 处理）

builder 已经把 item_dense fid 写进 schema.json，train.py 只读 schema 不需要单独 CLI。

## 8. `trainer.py` 修改清单

### 8.1 sidecar 复制列表

```python
# 在保存 best_model 时，复制以下文件到 ckpt 目录
SIDECAR_FILES_FE08 = [
    "fe08_transform_stats.json",
    "fe08_dense_normalization_stats.json",
    "dropped_feats.may7.json",
    "ns_groups.may7.json",
    "schema.json",
    "train_config.json",
    "feature_engineering_stats.json",  # 沿用 FE-01 的
    "docx_alignment.fe08.json",        # 可选，DOCX 对齐 sidecar
]
# 现有 trainer.py 里有 sidecar 复制循环，把这个列表加上去
```

## 9. 验证步骤（端到端）

按以下顺序跑完一遍，每步打勾：

```text
[ ] 1. 切换到 worktree 主目录，确认 git 分支正确
[ ] 2. 复制 build_fe07_p012_domain_dataset.py → build_fe08_may7_dataset.py
[ ] 3. 按 §4 规范修改 build_fe08：
        - missing_threshold 默认 0.80
        - 新增 --item_dense_fids / --enable_risky_item_dense_fids
        - 新增 sort_sequence_by_recency
        - 新增 item_int_91 (latest match time bucket)
        - 新增 user_int_130/131 (hour/dow with +1 offset)
        - schema 写入新 fid
        - ns_groups.may7.json 自动生成
        - 全套 sidecar 输出
[ ] 4. 复制 evaluation/FE07/* → evaluation/FE08/*
[ ] 5. 按 §5 规范修改 evaluation/FE08/*
[ ] 6. 按 §7 修改 train.py（加 sanity check）
[ ] 7. 按 §8 修改 trainer.py（sidecar 列表）
[ ] 8. 创建 run_fe08_may7_full.sh
[ ] 9. 在 demo_1000.parquet 上跑 smoke test：
        bash run_fe08_may7_full.sh --data_dir ./demo_1000_dir --num_epochs 1 --batch_size 64
        预期：训练能启动，前几个 step 不崩
[ ] 10. 在 demo_1000 上跑 evaluation：
         python3 evaluation/FE08/infer.py --ckpt_dir <ckpt_path>
         预期：完成 inference，输出 AUC
[ ] 11. 平台上传：tools/build_fe08_may7_dataset.py 与根目录版本同步
[ ] 12. 平台跑全量训练 → eval → 看 AUC 落点
```

## 10. 强力检查清单（必须全部通过才提交）

```text
== 数据正确性 ==
[ ] missing_threshold = 0.80（不是 0.75）
[ ] item_dense_fids 默认 = [86, 91, 92]
[ ] 87 / 88 在白名单中，不传 --enable_risky_item_dense_fids 时会 raise
[ ] item_int_feats_89: vocab=3
[ ] item_int_feats_90: vocab=7 (= len(buckets)+1)
[ ] item_int_feats_91: vocab=64 (= NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES)+1 = 63+1)
[ ] user_int_feats_130: vocab=25
[ ] user_int_feats_131: vocab=8
[ ] item_dense_feats_91 与 item_int_feats_91 共存于不同 schema list

== 模型 / token 结构 ==
[ ] num_user_ns=5 / num_item_ns=2 / num_user_dense=1 / num_item_dense=1 → num_ns=9
[ ] T = 2*4 + 9 = 17
[ ] d_model = 136, 136 % 17 = 0 ✓
[ ] rank_mixer_mode = full
[ ] use_token_gnn=True / token_gnn_layers=4 / scale=0.15
[ ] 不启用 output_include_ns

== 防泄漏 ==
[ ] sequence sort 仅按 event_time，不依赖 label_time
[ ] match feature 仅统计 event_time ≤ sample_timestamp 的事件
[ ] hour/dow 来自 sample timestamp 不来自 label_time
[ ] item_dense 不含 fid 87 / 88
[ ] dense norm 仅在 train row groups fit
[ ] split_by_timestamp 启用

== Evaluation ==
[ ] evaluation/FE08/build_fe08_may7_dataset.py 只 transform，不 fit
[ ] evaluation/FE08/infer.py strict load
[ ] eval 端读取 ckpt 内 train_config.json 校验 d_model=136 等关键参数
```

## 11. 常见错误与排查

### 11.1 "d_model=128 must be divisible by T=17"

```text
原因：忘记把 d_model 从 128 改成 136
修复：检查 run_fe08_may7_full.sh 的 --d_model 参数
```

### 11.2 "item_dense_feats_87 is RISKY"

```text
原因：command line 传了 --item_dense_fids "86,87,91,92"
修复：保持默认或显式去掉 87；如确需启用必须加 --enable_risky_item_dense_fids
```

### 11.3 "Sequence sorted but lengths changed"

```text
原因：argsort 应用时把 padding (=0) 也排了进去
修复：argsort 只对非空 list 操作；length=0 直接跳过
```

### 11.4 "user_int_feats_130 OOB error: max id=24, vocab=24"

```text
原因：忘记 +1 偏移，hour=23 落到 vocab[23] 但 vocab=24 → 0..23
修复：hour_id = ((ts // 3600) % 24 + 1) → 1..24, vocab=25
```

### 11.5 "checkpoint missing key TokenGNN.layers.0.layer_scale"

```text
原因：evaluation/FE08/model.py 没把 TokenGNN 的 layer_scale 参数加进来
修复：checkpoint 是从 model.py 训练侧保存的，evaluation 侧 model 必须完全一致
```

## 12. 与已有 FE-07 的差异速查

| 项 | FE-07 | FE-08 |
| --- | --- | --- |
| `missing_threshold` | 0.75，仅 user_int | **0.80，user_int + item_int** |
| `item_dense` | 86 / 91 / 92（FE-01B 路径） | **同上 + 显式白名单防呆** |
| `user_dense` 新增 | 110 (FE-01A) + 120/121 (P0) | **可选继承；本方案 fid 130/131 走 user_int** |
| user_int 新增 fid | 无 | **130 (hour) + 131 (dow)** |
| item_int 新增 fid | 89, 90 | **89, 90, 91 (latest match time bucket)** |
| sequence 排序 | 不排 | **按 event_time descending 重排** |
| domain_time_buckets | 启用 | **不启用**（用全局 BUCKET_BOUNDARIES） |
| seq_max_lens | seq_c:128, seq_d:768 | **seq_c:128, seq_d:512** |
| user_ns_tokens / item_ns_tokens / num_queries | 6 / 4 / 1 | **5 / 2 / 2** |
| rank_mixer_mode | full | **full** |
| d_model | 64 | **136** |
| dropout_rate | 0.015 | **0.05** |
| use_token_gnn | False | **True (4 layers, full, scale=0.15)** |

## 13. AI Agent 完成本任务的最小工作量估计

```text
- 复制 + 改名 6 个文件:           20 min
- 修改 build_fe08 (§4):             90 min
- 修改 evaluation/FE08/* (§5):      40 min
- 修改 train.py (§7):               10 min
- 修改 trainer.py (§8):             10 min
- 创建 run_fe08 (§6):               10 min
- 跑 smoke test (§9):               20 min
- 修复 smoke test 报错:             30 min (估计)
- 强力检查 (§10):                   20 min
- 总计:                            约 4 ~ 5 小时
```

## 14. 提交规范

```text
commit message: 
  "FE08 5月7日方案: full + d_model=136 + GNN + 新特征 + 序列排序"

PR description 必须包含:
  - 主方案文档链接（5月7日_GNN结合验证特征_保持NS影响域_方案.md）
  - 偏差点逐条对照（§11.5.1 偏差落地总览）
  - 强力检查清单全部 ✓
  - smoke test 输出截图
```

---

## 附录 A：关键代码段速查

### A.1 排序后 time_diff 仍然非负的证明

```text
sample_timestamp ≥ event_timestamp（builder 已过滤未来事件）
排序后 event_timestamp 单调递减
→ time_diff = sample_timestamp - event_timestamp 单调递增 ≥ 0
→ searchsorted(BUCKET_BOUNDARIES, time_diff) 单调递增
→ bucket_id 单调递增
→ time_embedding 查表无 OOB 风险
```

### A.2 NUM_TIME_BUCKETS 来源

```python
# dataset.py L110-132
BUCKET_BOUNDARIES = np.array([5, 10, ..., 31536000], dtype=np.int64)
# len(BUCKET_BOUNDARIES) = 63
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1   # = 64
# bucket_id ∈ {0=padding, 1, 2, ..., 63}
# 所以 item_int_feats_91 vocab = 64
```

### A.3 item_int_feats_91 与 item_dense_feats_91 共存证明

```text
parquet 列名:
    item_int_feats_91     ← int64 标量
    item_dense_feats_91   ← float32 标量
两个列名独立。

dataset.py 列查找:
    f'item_int_feats_{fid}'    → 'item_int_feats_91'
    f'item_dense_feats_{fid}'  → 'item_dense_feats_91'
通过 prefix 区分，零歧义。

schema.json:
    item_int:   [..., [91, 64, 1], ...]
    item_dense: [..., [91, 1], ...]
两个 list 各自维护，不冲突。

ns_groups.may7.json 仅含 int fid:
    item_ns_groups.I4_target_matching_fields: [89, 90, 91]
'91' 在此 = item_int_feats_91，唯一指向。
```

完成本指南后即可着手搭建。如遇歧义，**永远以主方案文档为准**：
[5月7日_GNN结合验证特征_保持NS影响域_方案.md](5月7日_GNN结合验证特征_保持NS影响域_方案.md)
