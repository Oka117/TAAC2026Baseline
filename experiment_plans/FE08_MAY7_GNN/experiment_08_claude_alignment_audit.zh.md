# FE-08 与 Claude 方案强对齐审计

## 0. 审计结论

本目录下 FE-08 方案与 Claude FE08 最终方案 **对齐**。本轮审计做了两项收紧：

```text
1. 把“FE07 已验证数据层信号”收紧为“FE01B/FE07 已验证的 item_dense/match 安全集合”，
   避免误解为强制纳入 FE07 P2-Domain 或 user_dense 110/120/121。
2. 明确 Claude 主方案旧 checklist 中的 ffn_only / d_model=128 是历史残留，
   本 FE08 方案以 Claude 前文最终锁定项和代码结构指南为准：
   rank_mixer_mode=full, d_model=136。
```

审计状态：

```text
核心参数: 通过
数据特征: 通过
Token 结构: 通过
评估 sidecar: 通过
防泄漏约束: 通过
代码文件清单: 通过
需要人工注意: evaluation/FE08/model.py 不能原样继承当前 FE07 no-GNN guard
```

## 1. Source Of Truth 与优先级

| 优先级 | Claude 文件 | 使用规则 |
| ---: | --- | --- |
| 1 | `experiment_plans/Claude/5月7日_FE08代码结构_供AI_Agent搭建.md` | 代码搭建、文件清单、sidecar、eval 契约的直接依据 |
| 2 | `experiment_plans/Claude/5月7日_GNN结合验证特征_保持NS影响域_方案.md` | 实验目标、最终结构、偏差落地、AUC 预测和强检查依据 |
| 3 | `experiment_plans/Claude/Baseline与数据联合分析-提分路线.md` | 数据层动机来源；不直接扩大 FE08 主线特征集合 |

冲突处理规则：

```text
如果同一 Claude 主方案内前文最终锁定项与后文 checklist 残留冲突，
以后文 checklist 不作为主配置依据，按代码结构指南和主方案“偏差最终落地”执行。
```

## 2. 总体目标对齐

| 审计项 | Claude 联系点 | FE08 方案联系点 | 状态 |
| --- | --- | --- | :---: |
| 完成产物 | 代码指南 L8-L9：`build_fe08_may7_dataset.py` + `evaluation/FE08/` + `run_fe08_may7_full.sh`，目标 0.8185~0.8215 | `README.md` 代码入口规划；主方案 §5 | 通过 |
| 一句话定义 | 主方案 L52-L59：0.8159 GNN baseline + FE07 稳态收益 + item_dense + sequence sort + d_model=136/full | 主方案 §0 | 通过 |
| 核心约束 | 主方案 L61-L69：num_ns=9、full、d_model=136、其它超参保持 | 主方案 §4、强检查 §5 | 通过 |
| 第一轮目标 | 主方案 L1007-L1009、L1120-L1124：>=0.8185 替代，>=0.8200 下一阶段 | 主方案 §10、强检查 §9 | 通过 |

## 3. 文件结构对齐

| 文件/模块 | Claude 联系点 | FE08 方案联系点 | 状态 |
| --- | --- | --- | :---: |
| `build_fe08_may7_dataset.py` | 代码指南 L29、L188-L208 | 主方案 §5.1、强检查 §11 M0/M1 | 通过 |
| `tools/build_fe08_may7_dataset.py` | 代码指南 L30 | 主方案 §5.1 | 通过 |
| `run_fe08_may7_full.sh` | 代码指南 L31、L474-L578 | 主方案 §5.1、§7 | 通过 |
| `evaluation/FE08/build_fe08_may7_dataset.py` | 代码指南 L32、L426-L437 | 主方案 §5.1、§8 | 通过 |
| `evaluation/FE08/dataset.py` | 代码指南 L33、L439-L446 | 主方案 §5.1、强检查 §1 | 通过 |
| `evaluation/FE08/model.py` | 代码指南 L34、L448-L454 | 主方案 §5.1、强检查 §0/§1/§6 | 通过但需注意 |
| `evaluation/FE08/infer.py` | 代码指南 L35、L456-L472 | 主方案 §5.1、§8 | 通过 |
| `train.py` | 代码指南 L580-L608 | 主方案 §5.2、强检查 §1 | 通过 |
| `trainer.py` | 代码指南 L610-L626 | 主方案 §5.2、强检查 §7 | 通过 |

注意：

```text
代码指南 L451-L452 写“FE-07 的 model.py 已包含 TokenGNN 完整实现”。
当前仓库实际 evaluation/FE07/model.py 有 no-GNN guard。
因此 FE08 方案特别标注：evaluation/FE08/model.py 必须支持 TokenGNN，不能原样继承 no-GNN guard。
这是对本仓库当前代码事实的必要补充，不改变 Claude FE08 目标。
```

## 4. 数据特征对齐

### 4.1 missing drop

| 规则 | Claude 联系点 | FE08 方案联系点 | 状态 |
| --- | --- | --- | :---: |
| 阈值 0.80 | 代码指南 L80-L82、L193 | 主方案 §3.1、强检查 §2 | 通过 |
| user_int + item_int | 代码指南 L114-L116、L322 | 主方案 §3.1、强检查 §2 | 通过 |
| sidecar `dropped_feats.may7.json` | 代码指南 L368-L376 | 主方案 §3.1、§6 | 通过 |
| 空列表合法 | 主方案 L198-L212、代码指南 L375 | 主方案 §3.1 | 通过 |

### 4.2 item_dense

| 规则 | Claude 联系点 | FE08 方案联系点 | 状态 |
| --- | --- | --- | :---: |
| fid = 86/91/92 | 主方案 L220-L279、代码指南 L87-L92 | 主方案 §3.2、强检查 §4.1 | 通过 |
| 87/88 risky 排除 | 主方案 L226-L258、L281-L286 | 主方案 §3.2、强检查 §2/§4.1 | 通过 |
| CLI guard | 代码指南 L199-L203、L215-L224 | 主方案 §3.2 | 通过 |
| train-only normalization | 主方案 L291-L296、代码指南 L121-L125 | 主方案 §3.2、强检查 §3 | 通过 |
| stats sidecar | 代码指南 L378-L386 | 主方案 §6、强检查 §7 | 通过 |

### 4.3 sequence sort

| 规则 | Claude 联系点 | FE08 方案联系点 | 状态 |
| --- | --- | --- | :---: |
| 按近到远排序 | 主方案 L337-L340、代码指南 L145-L149 | 主方案 §3.3、强检查 §2 | 通过 |
| row x domain 粒度 | 主方案 L351-L360、代码指南 L229-L276 | 主方案 §3.3 | 通过 |
| train/eval 同逻辑 | 主方案 L371-L374、代码指南 L428-L436 | 主方案 §8、强检查 §6 | 通过 |
| 不依赖 label_time | 主方案 L865-L866、L872-L878 | 强检查 §3 | 通过 |

### 4.4 item_int 89/90/91

| fid | Claude 联系点 | FE08 方案联系点 | 状态 |
| --- | --- | --- | :---: |
| 89 has_match | 主方案 L380-L396、代码指南 L133 | 主方案 §3.4、强检查 §2 | 通过 |
| 90 bucketized match_count | 主方案 L380-L397、代码指南 L134 | 主方案 §3.4、强检查 §2 | 通过 |
| 91 latest match time bucket | 主方案 L383、L397-L421、代码指南 L137、L279-L302 | 主方案 §3.4、强检查 §2/§4.3 | 通过 |
| `item_int_feats_91` 与 `item_dense_feats_91` 共存 | 主方案 L288-L289、代码指南 L140、L805-L825 | 主方案 §3.4、强检查 §4.3 | 通过 |

### 4.5 user_int 130/131

| fid | Claude 联系点 | FE08 方案联系点 | 状态 |
| --- | --- | --- | :---: |
| 130 hour_of_day | 主方案 L423-L470、代码指南 L138、L306 | 主方案 §3.5、强检查 §2 | 通过 |
| 131 day_of_week | 主方案 L423-L470、代码指南 L139、L307 | 主方案 §3.5、强检查 §2 | 通过 |
| +1 offset | 主方案 L450-L463 | 主方案 §3.5、强检查 §10 | 通过 |

## 5. 模型结构对齐

| 审计项 | Claude 联系点 | FE08 方案联系点 | 状态 |
| --- | --- | --- | :---: |
| user_ns_tokens=5 | 主方案 L63-L65、L87；代码指南 L58 | 主方案 §4.1、强检查 §5 | 通过 |
| item_ns_tokens=2 | 主方案 L63-L65、L87；代码指南 L59 | 主方案 §4.1、强检查 §5 | 通过 |
| num_queries=2 | 主方案 L63-L65、L87；代码指南 L60 | 主方案 §4.1、强检查 §5 | 通过 |
| has_item_dense=True | 代码指南 L62-L64；主方案 L298-L301 | 主方案 §4.1 | 通过 |
| num_ns=9 | 代码指南 L64-L66；主方案 L63-L65 | 主方案 §4.1、强检查 §5 | 通过 |
| T=17 | 代码指南 L65-L73；主方案 L617-L629 | 主方案 §4.1 | 通过 |
| rank_mixer_mode=full | 主方案 L66-L68、L497-L527 | 主方案 §4.2、强检查 §5 | 通过 |
| d_model=136 | 主方案 L497-L527；代码指南 L66-L73 | 主方案 §4.2、强检查 §5 | 通过 |
| emb_dim=64 | 代码指南 L67；主方案 L523-L536 | 主方案 §7 | 通过 |
| dropout=0.05 | 主方案 L551-L565；代码指南 L70 | 主方案 §7、强检查 §8 | 通过 |
| TokenGNN 4/full/0.15 | 代码指南 L94-L99、L165-L168 | 主方案 §4.3、强检查 §5 | 通过 |
| 禁止 output_include_ns | 主方案 L1066、代码指南 L680 | 主方案 §4.3、强检查 §5 | 通过 |

## 6. seq_top_k 对齐

| 审计项 | Claude 联系点 | FE08 方案联系点 | 状态 |
| --- | --- | --- | :---: |
| `seq_top_k=100` 保留 | 主方案 L567-L583；代码指南 L70-L71 | 主方案 §7、强检查 §8 | 通过 |
| transformer 主线下 marker only | 主方案 L580-L610 | 主方案 §7、强检查 §0/§5 | 通过 |
| longer 只作为 6.B 消融 | 主方案 L593-L613 | README 推荐执行顺序、主方案 §9 | 通过 |
| `train.py` warning | 代码指南 L582-L590 | 主方案 §5.2、强检查 §1 | 通过 |

## 7. Evaluation 与 sidecar 对齐

| 审计项 | Claude 联系点 | FE08 方案联系点 | 状态 |
| --- | --- | --- | :---: |
| eval 读取 checkpoint sidecar | 主方案 L482-L492；代码指南 L428-L436 | 主方案 §8、强检查 §6 | 通过 |
| 禁止 eval re-fit | 主方案 L490-L492；代码指南 L429-L436 | 主方案 §8、强检查 §3/§6 | 通过 |
| strict load | 主方案 L491-L492；代码指南 L462 | 主方案 §8、强检查 §6 | 通过 |
| `fe08_transform_stats.json` | 代码指南 L388-L409 | 主方案 §6、强检查 §7 | 通过 |
| `fe08_dense_normalization_stats.json` | 代码指南 L378-L386 | 主方案 §6、强检查 §7 | 通过 |
| `dropped_feats.may7.json` | 代码指南 L368-L376 | 主方案 §3.1、强检查 §7 | 通过 |
| `ns_groups.may7.json` | 代码指南 L411-L419 | 主方案 §3.6、强检查 §4.2/§7 | 通过 |
| train_config 关键参数校验 | 代码指南 L463-L471 | 主方案 §8、强检查 §6 | 通过 |

## 8. 与 Claude 方案内部残留的处理

Claude 主方案中有一处旧 checklist 残留：

```text
主方案 L883-L888:
  写了 rank_mixer_mode=ffn_only 和 d_model=128。
主方案 L919:
  写了 d_model=128 / dropout=0.05 / seq_top_k=100。
```

这些内容与同一文档的最终锁定项冲突：

```text
主方案 L24 / L39 / L56-L68 / L497-L527:
  明确最终采用 rank_mixer_mode=full + d_model=136。
代码指南 L56-L73 / L165-L168 / L551-L552 / L677-L678:
  明确 d_model=136，rank_mixer_mode=full，T=17 可整除。
```

本 FE08 方案处理：

```text
采用 full + d_model=136。
把 ffn_only + d_model=128 标为禁止项和历史残留。
```

结论：此处理符合 Claude FE08 最终方案，不属于偏离。

## 9. 与 FE07 的边界检查

| 项 | Claude FE08 要求 | FE08 方案处理 | 状态 |
| --- | --- | --- | :---: |
| builder 起点 | fork FE07 builder | 采用 | 通过 |
| FE07 P2-Domain | FE08 不启用 domain_time_buckets | 主方案明确“不启用 P2-Domain” | 通过 |
| FE07 user_dense 110/120/121 | Claude 代码指南写可选继承 | 本主线不强制纳入，只保留为可选继承 | 通过 |
| FE07 no-GNN | FE08 必须启用 TokenGNN | 明确不能继承 no-GNN guard | 通过 |
| seq_d | FE07 768，FE08 512 | 主方案采用 512 | 通过 |

## 10. 最终检查清单

```text
[x] 方案文件夹存在: experiment_plans/FE08_MAY7_GNN
[x] README 包含 FE08 定义、source of truth、代码入口
[x] 主方案包含 Claude 对应联系点摘要
[x] 本审计文件标注逐项 Claude 联系点
[x] missing_threshold=0.80
[x] item_dense_fids={86,91,92}
[x] risky 87/88 默认排除
[x] sequence sort by recency
[x] item_int 89/90/91
[x] user_int 130/131
[x] ns_groups 不含 dense fid
[x] rank_mixer_mode=full
[x] d_model=136
[x] TokenGNN 4/full/0.15
[x] seq_top_k=100 marker only
[x] eval sidecar parity
[x] ffn_only/d_model=128 残留已标注为禁止项
```

## 11. 审计结论

FE-08 方案可以作为 Claude 5 月 7 日方案的执行版文档。后续进入代码实现时，必须按本审计文件的联系点逐项打勾，尤其是：

```text
1. evaluation/FE08/model.py 支持 TokenGNN。
2. sidecar 全部进入 checkpoint。
3. eval transform 不 re-fit。
4. sequence sort 不破坏 list 对齐。
5. full + d_model=136 不被旧 checklist 改回 128/ffn_only。
```
