# FE-07 强力检查与实验结果预测

## 0. 结论先行

这次强力检查把 FE-07 拆成三类模块：

```text
已落账可复用:
  FE00 结果、FE01A/B 结果、item_dense 读取、split_by_timestamp

需要新增代码:
  FE07 builder、P2-Domain per-domain time bucket、evaluation/FE07

必须单独消融:
  FE00 average fill vs P0-L1 missing bucket、P1-NS group、P2-Domain bucket vs seq length
```

最可能的提分来源排序：

| 排名 | 模块 | 预计 Eval 增量 | 证据强度 | 备注 |
| ---: | --- | ---: | :---: | --- |
| 1 | FE01B target-history match | `+0.0010 ~ +0.0020` | 高 | 已落账 `+0.001577` |
| 2 | P0 domain summary dense | `+0.0008 ~ +0.0020` | 中高 | README demo lift 强，需全量验证 |
| 3 | FE00 dense normalization / cleanup | `+0.0008 ~ +0.0015` | 高 | 已落账 `+0.001121` |
| 4 | P2-Domain bucket + seq strategy | `+0.0003 ~ +0.0012` | 中 | 四域时间分布差异明显，但代码未就绪 |
| 5 | P1 semantic NS groups | `-0.0005 ~ +0.0012` | 中 | 语义正确，但 token 重排可能扰动已训练模式 |
| 6 | P0 current timestamp context | `+0.0003 ~ +0.0010` | 中 | 低成本时间上下文 |
| 7 | FE01A total frequency | `+0.0000 ~ +0.0005` | 高 | 已落账 `+0.000255`，保留但不押注 |

主预测：

```text
FE07-Domain-main: 0.8150 ~ 0.8190，base case 约 0.8165
```

## 1. 当前代码就绪度强检查

| 检查项 | 当前状态 | 风险等级 | 处理建议 |
| --- | --- | --- | --- |
| `--split_by_timestamp` | `train.py` 已有 | 低 | FE07 直接复用 |
| item dense 读取 | `dataset.py` / `model.py` 已支持 `item_dense_feats` | 低 | FE01A/B、P1-L2 可复用 |
| FE06 P0AB builder | `build_fe06_p0ab_dataset.py` 已有 | 中 | 可借鉴，但 FE06 使用 P0-L1 vocab shift，不等于 FE07 主线的 FE00-literal |
| FE07 builder | 尚未新增 | 高 | 必须新增，避免串联脚本造成 stats/sidecar 不一致 |
| P2-Domain per-domain bucket | 当前只有全局 `BUCKET_BOUNDARIES` | 高 | 需要新增 domain-specific boundaries 和 time embedding |
| `--domain_time_buckets` | 文档中建议，代码未有该参数 | 高 | 实现前不能直接跑 FE07 主命令 |
| evaluation/FE07 | 尚未新增 | 高 | 必须从 checkpoint sidecar 读取 schema、stats、bucket boundaries |
| direct `output_include_ns` | 当前主代码未启用 | 低 | 继续禁止；历史结果证明负向 |
| purchase frequency | FE07 文档排除 | 中 | builder/eval 必须强断言 schema 不含 111/87 |

结论：

```text
FE07 当前是实验方案，不是可直接运行分支。
可以借鉴 FE06 builder 的 P0AB 逻辑，但只要进入 P2-Domain，就必须补代码和 FE07 evaluation；本实验不跑 GNN 分支。
```

## 2. 强力检查清单

### 2.1 数据与泄漏

| 检查 | 必须满足 | 失败后果 |
| --- | --- | --- |
| prefix 统计 | `event_time < current timestamp`，当前样本不参与自己的统计 | valid/eval 相关性失真 |
| match 特征 | `domain_d_seq_26 <= timestamp` 后再统计 match/count/delta | target-history 变成未来信息 |
| dense normalization | 只用 train row groups 拟合，eval 复用 sidecar | eval 轻微泄漏且不可复现 |
| P2 bucket boundaries | 只用 train row groups 拟合 | eval 分布被偷看 |
| current timestamp context | 固定 UTC 规则，不用本地时区 | 训练/eval 时区不一致 |
| purchase frequency | 第一轮 schema 中不存在 111/87 | 重演 FE01 full 掉点风险 |
| label_time | 只允许用于 loss/aux，不允许进输入特征 | 当前样本标签泄漏 |

### 2.2 Schema 与 token

| 检查 | 必须满足 | 失败后果 |
| --- | --- | --- |
| dense fid | 110/120/121 进 user dense，86/91/92 进 item dense | 特征生成了但模型吃不到 |
| int fid | 89/90 进 item int schema 和 item NS groups | match 的离散信号丢失 |
| dense fid 不写 NS groups | `ns_groups.feature_engineering.json` 只含 int fid | tokenizer 误读 dense id |
| token 数 | `user_ns=6,item_ns=4,num_queries=1,T=16,d_model=64` | RankMixer full 不可用 |
| FE00/P0L1 二选一 | average fill 与 missing bucket 不混用 | 实验变量不可解释 |
| sidecar | schema/stats/groups/train_config 全部复制进 checkpoint | eval strict load 或 transform 失败 |

### 2.3 P2-Domain

| 检查 | 必须满足 | 失败后果 |
| --- | --- | --- |
| timestamp fid | a=39, b=67, c=27, d=26 | bucket 全错 |
| bucket id 空间 | 每个 domain 独立 embedding，或使用 domain offset 后共享 embedding | 不同 domain bucket 语义混淆 |
| 对照实验 | 必须保留 `uniform-256` 与全局 bucket 对照 | seq length 收益和 bucket 收益混在一起 |
| domain_d | 细化 1h/1d/7d，保留更长近期窗口 | 高频近期行为被截断 |
| domain_c | 压缩长度，强化 30d/90d/180d/365d | 老行为噪声拖慢训练 |

### 2.4 Evaluation

| 检查 | 必须满足 | 失败后果 |
| --- | --- | --- |
| infer 严格读 checkpoint schema | 不以 hard-coded FE01/FE06 schema 为准 | missing/unexpected key |
| infer 不 refit | 不重新拟合 normalization、bucket、match pair | leaderboard 结果不可复现 |
| strict model load | 不允许静默忽略 key | AUC 虚假或模型结构错 |
| 推理时间 | FE07-Domain-main 应接近 P1/P2 非 GNN 路线 | 若 seq_d 拉长导致收益小且耗时大，回退 seq length |

## 3. 模块级预测

下面的预测以 B0 eval `0.810525` 为锚点，以已有落账结果优先，Claude/README 先验作为增量修正。

| 模块 | 归属 | 增益来源 | 预计单模块 Eval Δ | 置信度 | 主要风险 |
| --- | --- | --- | ---: | :---: | --- |
| FE00 cleanup/norm | DOCX P000-P002 | 删除高缺失 user int、dense normalization | `+0.0008 ~ +0.0015` | 高 | average fill 对 categorical id 可能负向 |
| FE01A total frequency | DOCX P005/P007 | user/item 曝光频次先验 | `+0.0000 ~ +0.0005` | 高 | streaming state 与 eval 分布偏移 |
| FE01B target match | DOCX P017-P020 | 目标 item 属性与 domain_d 历史匹配 | `+0.0010 ~ +0.0020` | 高 | valid 可能低但 eval 高，需以 eval 为准 |
| P0-T1 timestamp split | Claude P0-T1 | 让 valid 更接近 leaderboard | 直接 `0`，间接增信 | 中高 | valid AUC 下降不是坏事 |
| P0-L4 timestamp context | Claude P0-L4 | hour/dow/day_since_min | `+0.0003 ~ +0.0010` | 中 | 时间周期性若弱则收益小 |
| P0-L5/L6 domain summary | Claude P0-L5/L6 + README §6.2 | seq_len、1h/1d/7d/30d 活跃度 | `+0.0008 ~ +0.0020` | 中高 | 与 FE01A total frequency 有重叠 |
| P1-L2 item dense token | Claude P1-L2 | 让 item_dense 86/91/92 真进入模型 | 单独 `0`，联合必要 | 高 | token 数不合法会直接失败 |
| P1-NS semantic groups | DOCX P022-P042 + Claude P1-NS | 按语义组织 user/item int token | `-0.0005 ~ +0.0012` | 中 | RankMixer chunk 重排可能扰动 |
| P2-Domain bucket | Claude P2-Domain + README §3 | 每个 domain 独立时间分辨率 | `+0.0003 ~ +0.0012` | 中 | 当前代码未实现，需严谨对照 |
| P2 seq lens strategy | README §6.6 | domain_d 拉长、domain_c 压缩 | `-0.0003 ~ +0.0008` | 中 | 推理时间上升或截断有用长历史 |
| GNN/TokenGNN | GNN history | NS token 间关系补边 | 不纳入本实验 | 高 | 作为历史对照，不进入 FE-07 |
| Gated NS output | Claude P1-Output | final NS 表征小门控进入 head | `+0.0000 ~ +0.0010` | 低中 | 历史 direct NS head 已负向，第一轮不做 |
| P2-HighCard | Claude P2-HighCard | 高基数字段 hashing/count | `+0.0008 ~ +0.0020` | 中 | 改 id 表达，不能和 P2-Domain 同时首跑 |
| P2-Aux/FE04 | Claude P2-Aux + DOCX P058-P067 | engagement/delay 辅助监督 | `+0.0003 ~ +0.0015` | 低中 | label 语义和 loss 权重风险高 |

## 4. 实验结果预测

### 4.1 主线预测

| 实验 | 组成 | Eval AUC 保守 | Eval AUC Base | Eval AUC 乐观 | Parent Δ Base | 关键提分点 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| B0 | baseline | 0.8105 | 0.8105 | 0.8105 | - | 起点 |
| FE00-literal | B0 + FE00 | 0.8110 | 0.8116 | 0.8120 | +0.0011 | cleanup/norm |
| FE01AB-safe | FE00 + 01A + 01B | 0.8120 | 0.8128 | 0.8135 | +0.0012 | target match 为主，total freq 为辅 |
| P0-safe | FE01AB + T1/L4/L5/L6 | 0.8140 | 0.8152 | 0.8170 | +0.0024 | domain summary dense |
| P1-NS | P0-safe + item_dense + semantic groups | 0.8145 | 0.8158 | 0.8180 | +0.0006 | item_dense token、语义 groups |
| P2-Domain | P1-NS + per-domain bucket/seq lens | 0.8150 | 0.8166 | 0.8190 | +0.0008 | domain-specific time |

解释：

```text
P0-safe 的 base case 比 Claude 原始 P0 预测更保守，
因为 FE07 主线不把 P0-L1 missing bucket 放进主实验，且 FE00 已吃掉一部分 normalization 收益。

FE07 的预测不包含 GNN 结构收益；若后续另开 GNN 实验，应单独命名并重新建对照。
```

### 4.2 消融预测

| 消融 | 对照 | 预计结果 | 结论规则 |
| --- | --- | --- | --- |
| FE00-P0L1 missing bucket | FE00-literal | `-0.0005 ~ +0.0012` | 若赢 `>= +0.0005`，后续替换 average fill |
| P0 timestamp only | FE01AB-safe | `+0.0003 ~ +0.0010` | 若不涨，仍可保留作低成本上下文 |
| P0 domain summary only | FE01AB-safe | `+0.0008 ~ +0.0020` | P0 最大押注点 |
| P1-NS only | P0-safe | `-0.0005 ~ +0.0012` | 掉分则保留 item_dense、回退 groups |
| P2 bucket only | P1-NS uniform seq | `+0.0003 ~ +0.0012` | 若 seq lens 同时改，必须另跑对照 |
| P2 seq lens only | P1-NS global bucket | `-0.0003 ~ +0.0008` | 若推理时间涨太多且 AUC 小，回退 |

### 4.3 失败场景预测

| 失败现象 | 最可能原因 | 第一回滚动作 |
| --- | --- | --- |
| FE01AB-safe 低于 FE01B | FE01A total frequency 与 FE00/P0 统计冲突 | 去掉 110/86，只保留 FE01B |
| P0-safe valid 涨、eval 不涨 | timestamp split 仍不能代表 leaderboard 或 P0 特征过拟合 | 拆 120/121；先保留 121 |
| P1-NS 掉分 | RankMixer chunk 重排破坏原 token 统计 | `ns_groups_json=""`，保留 item dense |
| P2-Domain 掉分 | bucket id 空间/embedding 实现错，或 seq length 变量混入 | 回到全局 bucket；单独跑 seq lens |
| 误启用 GNN 参数 | run/config 继承 FE06 GNN 设置 | 关闭 `--use_token_gnn`，重跑 FE07-Domain-main |
| eval infer 报 missing key | sidecar/schema/evaluation 不一致 | 停止评估，修 FE07 infer strict schema |

## 5. 推荐执行模块拆分

### M0: FE07 Builder

必须新增：

```text
build_fe07_p012_domain_dataset.py
tools/build_fe07_p012_domain_dataset.py
```

职责：

```text
FE00-literal cleanup/norm
FE01AB-safe
P0 dense block 120/121
P2 domain bucket stats sidecar
docx alignment audit
```

### M1: P2-Domain Dataset/Model

必须新增或修改：

```text
dataset.py: 支持 domain-specific bucket ids
model.py: 支持 per-domain time_embedding 或 domain-offset bucket embedding
train.py: 增加 --domain_time_buckets / --domain_bucket_path
```

实现优先级：

```text
先做 domain-offset shared embedding，代码更小；
稳定后再切 ModuleDict per-domain embedding。
```

### M2: FE07 Evaluation

必须新增：

```text
evaluation/FE07/infer.py
evaluation/FE07/dataset.py
evaluation/FE07/model.py
evaluation/FE07/build_fe07_p012_domain_dataset.py
```

要求：

```text
只读 checkpoint sidecar，不在 eval fit 任何统计。
strict load checkpoint。
schema 里没有 111/87 时，绝不生成 purchase frequency。
```

### M3: P1-NS / Item Dense

可复用：

```text
ns_groups.feature_engineering.json
dataset.py item_dense
model.py item_dense token
```

检查：

```text
user_ns_tokens=6
item_ns_tokens=4
num_queries=1
rank_mixer_mode=full
```

### M4: 明确排除的结构变量

本实验不允许新增：

```text
GNN / TokenGNN
direct output_include_ns
P2-HighCard
P2-Aux
```

## 6. 最终建议

第一批只跑三个点：

```text
1. FE01AB-safe
2. P0-safe
3. P2-Domain
```

如果 `P2-Domain < 0.8150`，优先拆：

```text
P0 domain summary only
P1-NS off
P2 bucket only
P2 seq lens only
```

最应该押注的提分点：

```text
domain summary dense token + target-history match + P2-Domain time bucket
```

最应该谨慎的模块：

```text
purchase frequency、direct NS head、P2-Domain 与 seq length 同时改、eval 侧重新 fit 统计
```
