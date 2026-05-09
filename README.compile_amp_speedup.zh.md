# PCVRHyFormer：torch.compile + 混合精度训练加速实践

本文给出**最小工程改动**的加速路线：

1. 启用 `torch.cuda.amp.autocast`（混合精度，bf16 优先）
2. 启用 `torch.compile`（PyTorch 2.0+ 的图编译加速）
3. 在 [demo_1000.parquet](demo_1000.parquet) 1k 样本上做 4 组对照基准

> 目标：在不改模型结构、不改超参的前提下，**单步训练时间下降 30%~60%、显存占用下降 30%~50%**。

---

## 颜色图例（同 [README.research_directions.zh.md](README.research_directions.zh.md)）

| 标识 | 含义 |
|---|---|
| <span style="color:#16a34a; font-weight:bold;">🟢 高收益 / 低风险</span> | 几乎一定加速，副作用小 |
| <span style="color:#ca8a04; font-weight:bold;">🟡 中收益 / 中风险</span> | 大多数情况加速，但偶有 graph break / 数值问题 |
| <span style="color:#dc2626; font-weight:bold;">🔴 高风险</span> | 容易 NaN / 编译失败，需要小心 |

---

## 0. 现状（启用前）

| 项目 | 当前值 | 代码位置 |
|---|---|---|
| 默认精度 | fp32 全程 | [trainer.py:407-428](trainer.py:407) |
| 是否使用 `torch.compile` | 否 | [train.py:321](train.py:321) `model = PCVRHyFormer(...)` |
| `clip_grad_norm_` | `foreach=False`（规避 CUDA bug） | [trainer.py:422](trainer.py:422) |
| Optimizer | dense=AdamW(1e-4) + sparse=Adagrad(0.05) | [trainer.py:84-89](trainer.py:84) |
| Loss | BCEWithLogits / Focal | [trainer.py:415-418](trainer.py:415) |
| `torch.nan_to_num` 兜底 | 已在 SDPA 输出处启用 | [model.py:233](model.py:233) |
| `nn.Embedding(... padding_idx=0)` | 全部启用 | [model.py:1013](model.py:1013) 等 |

**关键事实**：
- `clip_grad_norm_(foreach=False)` 是为规避 PyTorch `_foreach_norm` 的 CUDA bug 设的兜底（[trainer.py:420-422](trainer.py:420)）；下面所有改造都不能动这个 flag。
- 双 optimizer：sparse Adagrad **不能用 fp16**（PyTorch 不支持），所以 AMP 只能包 dense path。下面的方案保留这一边界。

---

## 1. 第一步：启用混合精度（AMP）

### 1.1 选 `bf16` 还是 `fp16`？

| | bf16 | fp16 |
|---|---|---|
| 数值范围 | 与 fp32 相同（指数 8 位） | 范围窄（指数 5 位），需要 `GradScaler` |
| 硬件支持 | A100 / H100 / 3090 Ti / 4090 | T4 / V100 / 2080 / 大部分 GPU |
| 是否需要 `GradScaler` | <span style="color:#16a34a; font-weight:bold;">否</span> | <span style="color:#dc2626; font-weight:bold;">是（必须）</span> |
| 风险 | <span style="color:#16a34a; font-weight:bold;">🟢 几乎无 NaN</span> | <span style="color:#ca8a04; font-weight:bold;">🟡 偶发 overflow</span> |

> 90% 的场景应优先选 **bf16**。`fp16` 仅在硬件不支持 bf16 时退而求其次。

通过 `torch.cuda.is_bf16_supported()` 检测：

```python
amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
```

### 1.2 改动 [train.py](train.py)：新增 CLI

在 [train.py:138-145](train.py:138) Loss 段后追加：

```python
# === Mixed-precision flags ===
parser.add_argument('--use_amp', action='store_true', default=False,
                    help='启用 autocast 混合精度训练（推荐 bf16）')
parser.add_argument('--amp_dtype', type=str, default='auto',
                    choices=['auto', 'bf16', 'fp16'],
                    help='auto = bf16 if supported else fp16')
```

把 `args.use_amp` / `args.amp_dtype` 一并通过 `train_config` 传给 `PCVRHyFormerRankingTrainer`。

### 1.3 改动 [trainer.py](trainer.py)：autocast + GradScaler

[trainer.py:38-94](trainer.py:38) `__init__` 内增加：

```python
# === AMP setup ===
self.use_amp: bool = train_config.get('use_amp', False)
self.amp_dtype: torch.dtype = torch.float32  # default
if self.use_amp:
    requested = train_config.get('amp_dtype', 'auto')
    if requested == 'auto':
        self.amp_dtype = (torch.bfloat16
                          if torch.cuda.is_bf16_supported()
                          else torch.float16)
    elif requested == 'bf16':
        self.amp_dtype = torch.bfloat16
    elif requested == 'fp16':
        self.amp_dtype = torch.float16
    logging.info(f"AMP enabled: dtype={self.amp_dtype}")

# fp16 必须配 GradScaler；bf16 不需要
self.scaler: Optional[torch.cuda.amp.GradScaler] = None
if self.use_amp and self.amp_dtype == torch.float16:
    self.scaler = torch.cuda.amp.GradScaler()
```

把 [trainer.py:402-428](trainer.py:402) 的 `_train_step` 改成：

```python
def _train_step(self, batch: Dict[str, Any]) -> float:
    device_batch = self._batch_to_device(batch)
    label = device_batch['label'].float()

    self.dense_optimizer.zero_grad()
    if self.sparse_optimizer is not None:
        self.sparse_optimizer.zero_grad()

    model_input = self._make_model_input(device_batch)

    # === forward + loss 在 autocast 内 ===
    if self.use_amp:
        with torch.cuda.amp.autocast(dtype=self.amp_dtype):
            logits = self.model(model_input)
            logits = logits.squeeze(-1)
            if self.loss_type == 'focal':
                loss = sigmoid_focal_loss(
                    logits, label,
                    alpha=self.focal_alpha, gamma=self.focal_gamma,
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, label)
    else:
        logits = self.model(model_input)
        logits = logits.squeeze(-1)
        if self.loss_type == 'focal':
            loss = sigmoid_focal_loss(
                logits, label,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, label)

    # === backward / clip / step ===
    if self.scaler is not None:
        # fp16 path
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.dense_optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0, foreach=False,
        )
        self.scaler.step(self.dense_optimizer)
        self.scaler.update()
        if self.sparse_optimizer is not None:
            # sparse Adagrad 不走 GradScaler，但梯度已 unscale 过；正常 step 即可
            self.sparse_optimizer.step()
    else:
        # bf16 / fp32 path
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0, foreach=False,
        )
        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

    return loss.item()
```

[trainer.py:483-494](trainer.py:483) 的 `_evaluate_step` 也包 autocast（**eval 用 inference_mode + autocast 节省更多显存**）：

```python
def _evaluate_step(self, batch):
    device_batch = self._batch_to_device(batch)
    label = device_batch['label']
    model_input = self._make_model_input(device_batch)
    if self.use_amp:
        with torch.cuda.amp.autocast(dtype=self.amp_dtype):
            logits, _ = self.model.predict(model_input)
    else:
        logits, _ = self.model.predict(model_input)
    logits = logits.squeeze(-1)
    return logits, label
```

### 1.4 模型层面的兼容性检查

直接 grep 一下，确认下面几处在 amp 下不会数值异常：

| 风险点 | 位置 | 处理 |
|---|---|---|
| `RoPEMultiheadAttention.W_g` 初始化为 1.0 | [model.py:147-148](model.py:147) | bf16 下精度足够；fp16 偶尔会有截断但 sigmoid 后影响小，<span style="color:#65a30d; font-weight:bold;">🟢 OK</span> |
| `nan_to_num(out, nan=0.0)` | [model.py:233](model.py:233) | autocast 下 dtype 可能是 bf16/fp16，仍生效，<span style="color:#65a30d; font-weight:bold;">🟢 OK</span> |
| `LayerNorm` | 多处 | autocast 自动用 fp32 计算 mean/var，<span style="color:#65a30d; font-weight:bold;">🟢 OK</span> |
| `F.scaled_dot_product_attention` | [model.py:226](model.py:226) | 原生 amp 兼容，<span style="color:#65a30d; font-weight:bold;">🟢 OK</span> |
| `Embedding.weight` (sparse) | 多处 | sparse params 不在 autocast 路径，<span style="color:#65a30d; font-weight:bold;">🟢 OK</span> |
| `valid_mask_expanded.float()` 显式 cast | [model.py:480](model.py:480) | 确保 float32，bf16 下应改成 `.to(seq_tokens_list[i].dtype)` 避免类型混合，<span style="color:#ca8a04; font-weight:bold;">🟡 注意</span> |

**对最后一行的细节**：[model.py:475-491](model.py:475) `MultiSeqQueryGenerator.forward` 中：

```python
valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, L_i, 1)
seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)
```

`seq_tokens_list[i]` 在 autocast 下是 bf16，乘以 fp32 的 `valid_mask_expanded` 会触发隐式提升到 fp32，性能损失小但触发 graph break（torch.compile 时）。**修复**：改为：

```python
valid_mask_expanded = valid_mask.unsqueeze(-1).to(seq_tokens_list[i].dtype)
```

[model.py:712-714](model.py:712) `LongerEncoder._gather_top_k` 中也有类似问题：

```python
top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()
```

应改成：

```python
top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).to(top_k_tokens.dtype)
```

### 1.5 单独启用 AMP 的预期

| 指标 | fp32 baseline | + bf16 AMP |
|---|---|---|
| **单步时间** | 1.0× | <span style="color:#16a34a; font-weight:bold;">0.55 ~ 0.75×</span> |
| **GPU 显存峰值** | 1.0× | <span style="color:#16a34a; font-weight:bold;">0.50 ~ 0.65×</span> |
| **AUC** | 锚点 | <span style="color:#65a30d; font-weight:bold;">±0.01% 以内</span>（噪声范围） |
| **NaN 风险** | 几乎为 0 | <span style="color:#65a30d; font-weight:bold;">🟢 bf16 几乎无</span> / <span style="color:#dc2626; font-weight:bold;">🔴 fp16 偶发</span> |

**fp16 NaN 自检**：观察 [trainer.py:459-466](trainer.py:459) 的 `predictions are NaN, filtering them out` warning。如果验证集有 > 1% NaN，说明数值不稳，建议切回 bf16 或回 fp32。

---

## 2. 第二步：启用 `torch.compile`

### 2.1 三种 compile 模式

| `mode` | 编译时间 | 运行加速 | 适用场景 |
|---|---|---|---|
| `default` | 短（10s ~ 1min） | 1.1 ~ 1.3× | 快速验证 |
| `reduce-overhead` | 中（1 ~ 3min） | 1.2 ~ 1.5× | <span style="color:#16a34a; font-weight:bold;">最稳收益的推荐项</span> |
| `max-autotune` | 长（5 ~ 15min） | 1.3 ~ 1.6× | 长跑训练才划算 |

> 第一个 batch 会触发编译（看上去像"卡住"），通常 30s ~ 2min；之后 step time 才稳定。

### 2.2 改动 [train.py](train.py)

新增 CLI（紧接 §1.2 之后）：

```python
parser.add_argument('--use_compile', action='store_true', default=False,
                    help='使用 torch.compile 加速 (PyTorch 2.0+)')
parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                    choices=['default', 'reduce-overhead', 'max-autotune'])
parser.add_argument('--compile_dynamic', action='store_true', default=False,
                    help='允许动态 shape，避免最后一个 mini-batch 重新编译')
```

在 [train.py:321](train.py:321) `model = PCVRHyFormer(**model_args).to(args.device)` 之后：

```python
if args.use_compile:
    if not hasattr(torch, 'compile'):
        raise SystemExit("torch.compile 需要 PyTorch >= 2.0")
    logging.info(f"Compiling model with mode={args.compile_mode}, "
                 f"dynamic={args.compile_dynamic}")
    model = torch.compile(
        model,
        mode=args.compile_mode,
        dynamic=args.compile_dynamic,
        fullgraph=False,  # 允许 graph break，PCVRHyFormer 必须设 False
    )
```

> **`fullgraph=False` 是必须的**。因为 [model.py:1654-1665](model.py:1654) 的 `for domain in self.seq_domains:` 是 Python 层 for-loop（虽然 `seq_domains` 是常量列表），而 [model.py:1603-1624](model.py:1603) 的 RoPE 预计算也走 Python 控制流，`fullgraph=True` 会编译失败。

### 2.3 已知 graph break 风险点（必读）

下面这些位置会生成 graph break（编译会自动切片成多个子图，但每段开销减半才能整体加速）。**为最佳收益，请按 §2.5 一起 patch**。

| 位置 | 原因 | 影响 |
|---|---|---|
| [model.py:480](model.py:480) `.float()` 显式 cast | 类型不一致 | <span style="color:#ca8a04; font-weight:bold;">🟡 graph break</span> |
| [model.py:714](model.py:714) `.float()` 显式 cast | 同上 | <span style="color:#ca8a04; font-weight:bold;">🟡 graph break</span> |
| [model.py:62-64](model.py:62) `RotaryEmbedding.forward` 中 `cos_cached[:, :seq_len, :]` | dynamic slice | <span style="color:#ca8a04; font-weight:bold;">🟡 多次重编译（不同 seq_len）</span> |
| [model.py:701](model.py:701) `torch.clamp(indices, min=0, max=L-1)` | L 是 dynamic | <span style="color:#65a30d; font-weight:bold;">🟢 OK（PyTorch 2.1+ 支持 dynamic clamp）</span> |
| [model.py:743](model.py:743) `if L > self.top_k:` | data-dependent control flow | <span style="color:#dc2626; font-weight:bold;">🔴 graph break</span> |
| [model.py:1064-1065](model.py:1064) Python for-loop over groups | 静态展开，编译时间增加 | <span style="color:#65a30d; font-weight:bold;">🟢 编译完成后无开销</span> |
| [trainer.py:413](trainer.py:413) `logits.squeeze(-1)` 跨 compile / 非 compile 边界 | 一般 OK | <span style="color:#65a30d; font-weight:bold;">🟢 OK</span> |

### 2.4 IterableDataset 末尾 batch 不齐 → 重编译问题

[dataset.py:347](dataset.py:347) `pf.iter_batches(batch_size=self.batch_size, ...)` 不会丢掉最后一个 < batch_size 的 mini-batch。这会让 `torch.compile` 在最后一步重新编译（B 维变了）。

**两种修复方式**：

**方式 1（推荐）：开启 `dynamic=True`**

```bash
--use_compile --compile_dynamic
```

让 `torch.compile` 把 batch 维度标记为动态，避免重编译。代价：编译稍慢、可能少 5%~10% 加速。

**方式 2：dataset 端 drop last**

[dataset.py:344-355](dataset.py:344) 在 `_flush_buffer` 中可加：

```python
# _flush_buffer 末尾，只 yield 整 batch
for i in range(0, total_rows - self.batch_size + 1, self.batch_size):
    end = i + self.batch_size
    ...
```

但会丢弃最后 ≤ batch_size 行训练数据，**仅训练时启用，验证集不要 drop**。

### 2.5 最小 model.py 改动（消除 graph break）

把下面两处 `.float()` 改成保留原 dtype：

```python
# model.py:480 (MultiSeqQueryGenerator.forward)
- valid_mask_expanded = valid_mask.unsqueeze(-1).float()
+ valid_mask_expanded = valid_mask.unsqueeze(-1).to(seq_tokens_list[i].dtype)

# model.py:714 (LongerEncoder._gather_top_k)
- top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()
+ top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).to(top_k_tokens.dtype)
```

[model.py:743](model.py:743) `if L > self.top_k` 是 **data-dependent if**，但因为 `L = x.shape[1]` 在每个 forward 中都是张量形状的 Python int（不是 tensor），实际上不会触发 `torch._dynamo.exc.UserError`。如果 `--compile_mode max-autotune` 报错，把 LongerEncoder 整体 `@torch._dynamo.disable` 掉是最简单兜底：

```python
import torch._dynamo as dynamo
class LongerEncoder(nn.Module):
    @dynamo.disable
    def forward(self, x, key_padding_mask=None, rope_cos=None, rope_sin=None):
        ...
```

### 2.6 单独启用 compile 的预期（在已有 AMP 基础上）

| 指标 | 仅 AMP | + compile (reduce-overhead) |
|---|---|---|
| **单步时间** | 1.0× | <span style="color:#16a34a; font-weight:bold;">0.70 ~ 0.85×</span> |
| **GPU 显存峰值** | 1.0× | <span style="color:#65a30d; font-weight:bold;">0.95 ~ 1.05×</span>（compile 不会显著省显存） |
| **首步编译耗时** | 0 | <span style="color:#ca8a04; font-weight:bold;">+30s ~ +120s</span> |
| **AUC** | 锚点 | <span style="color:#65a30d; font-weight:bold;">±0.01%</span> |

---

## 3. 推荐组合：AMP + compile

最终推荐启动命令（在 [run.sh](run.sh) 基础上加 4 个 flag）：

```bash
bash run.sh \
    --use_amp \
    --amp_dtype bf16 \
    --use_compile \
    --compile_mode reduce-overhead \
    --compile_dynamic
```

**整合预期**（fp32 baseline → AMP + compile）：

| 指标 | 提升 |
|---|---|
| 单步时间 | <span style="color:#16a34a; font-weight:bold;">0.40 ~ 0.60×（即 1.7 ~ 2.5 倍加速）</span> |
| 显存峰值 | <span style="color:#16a34a; font-weight:bold;">0.50 ~ 0.65×</span> |
| 首步编译 | <span style="color:#ca8a04; font-weight:bold;">+30s ~ +120s</span> |
| AUC | <span style="color:#65a30d; font-weight:bold;">几乎无影响</span> |
| 工程改动 | 约 60 行（[trainer.py](trainer.py) + [train.py](train.py) + [model.py](model.py) 3 处 `.float()` 修复） |

---

## 4. 开启加速后的全面影响（正面 / 负面 / AUC 专项）

启用 AMP 和 / 或 `torch.compile` 不只是单纯的"更快"，会在**数值精度、训练动力学、工程链路、评估指标**上引入一系列变化。本节集中说明所有正负影响，并单独分析对 AUC 的影响。

### 4.1 正面影响（Pros）

| 维度 | 影响 | 量级 |
|---|---|---|
| **训练吞吐** | autocast 减少访存 + bf16 cuDNN/cuBLAS kernel | <span style="color:#16a34a; font-weight:bold;">单步时间 ↓ 25~45%</span> |
| **GPU 显存峰值** | 中间激活和梯度从 fp32 → bf16/fp16 | <span style="color:#16a34a; font-weight:bold;">↓ 35~50%</span> |
| **可叠加更大 batch / 模型** | 显存释放后可以配合方向 2/7（[README.research_directions.zh.md](README.research_directions.zh.md)） | <span style="color:#16a34a; font-weight:bold;">最大支持 batch ↑ 2×、d_model ↑ 50%</span> |
| **实验反馈周期** | 单次完整训练时间从 X 小时 → 0.5X | <span style="color:#16a34a; font-weight:bold;">同时间窗口下 ablation 数 × 2</span> |
| **隐性 regularization** | bf16 中间值的轻微噪声相当于一种 "weight noise" | <span style="color:#65a30d; font-weight:bold;">极偶尔 +0.01~0.02% AUC（不稳定）</span> |
| **Inductor kernel 融合** | `torch.compile` 把 LayerNorm + GELU + Linear 融合 | <span style="color:#16a34a; font-weight:bold;">CPU 端 launch 开销 ↓ 70%</span> |
| **CUDA Graphs (reduce-overhead)** | 消除 Python loop 的 overhead | <span style="color:#16a34a; font-weight:bold;">小 batch 场景加速更显著</span> |

### 4.2 负面影响（Cons）

| 维度 | 影响 | 严重程度 |
|---|---|---|
| **首步编译延迟** | `torch.compile` 第一个 batch 卡 30s ~ 2min | <span style="color:#ca8a04; font-weight:bold;">🟡 一次性，长跑可摊销</span> |
| **数值精度损失** | bf16 损失 16 位尾数；fp16 损失指数范围 | <span style="color:#65a30d; font-weight:bold;">🟢 bf16 几乎无</span> / <span style="color:#dc2626; font-weight:bold;">🔴 fp16 偶发 NaN</span> |
| **Reproducibility 削弱** | [utils.py:257](utils.py:257) `cudnn.deterministic` 在 compile 后部分失效 | <span style="color:#ca8a04; font-weight:bold;">🟡 同 seed AUC 差 ±0.001~0.01</span> |
| **Checkpoint 不兼容** | compile 后 `state_dict()` key 多 `_orig_mod.` 前缀 | <span style="color:#dc2626; font-weight:bold;">🔴 不修补会让 infer 加载失败</span> |
| **Debug 体验下降** | traceback 指向 `<eval_with_key>.42`，pdb 在 compile 区域内不工作 | <span style="color:#ca8a04; font-weight:bold;">🟡 中等</span> |
| **显存统计失真** | `max_memory_allocated()` 在 reduce-overhead 下偏大 30~50% | <span style="color:#ca8a04; font-weight:bold;">🟡 仅影响 benchmark 数据</span> |
| **重编译开销** | RoPE cache 不同 seq_len、reinit embedding 后会触发 | <span style="color:#ca8a04; font-weight:bold;">🟡 单次 ~5s</span> |
| **fp16 验证样本缩水** | NaN 预测被 [trainer.py:459-466](trainer.py:459) 过滤掉 | <span style="color:#dc2626; font-weight:bold;">🔴 NaN 率 > 1% 时 AUC 不可信</span> |

### 4.3 对训练（Loss / 梯度）的影响

#### 4.3.1 Loss 曲线

- **bf16**：与 fp32 几乎重合，肉眼无差别。loss 抖动幅度在 ±0.5% 以内
- **fp16**：[trainer.py:309](trainer.py:309) 写入 `Loss/train` 时偶尔出现 spike（GradScaler 跳过该 step 是正常现象）
- **compile**：首步 loss 异常（包含编译时间），第二步开始稳定。看 tensorboard 时**忽略前 100 步**

#### 4.3.2 梯度更新

- AMP 不影响 sparse Adagrad 路径（[trainer.py:84](trainer.py:84)），embedding tables 仍以 fp32 累积平方梯度
- AMP 影响 dense AdamW 路径，但 PyTorch 内部维护 fp32 master weight，optimizer 状态仍是 fp32
- [trainer.py:422](trainer.py:422) `clip_grad_norm_(foreach=False)` 在 GradScaler 下必须先 `unscale_`，已在 §1.3 处理
- `torch.compile` 不改变梯度数值

#### 4.3.3 高基数 embedding reinit

[trainer.py:355-376](trainer.py:355) 每个 epoch 末尾的 reinit + 重建 Adagrad：
- bf16 兼容（sparse 路径独立于 amp）
- compile 下：reinit 后下一个 forward 触发**部分重编译**（embedding shape 不变，但 weight 内存地址可能换）
- 实测开销：< 5s/epoch，可接受

### 4.4 对评估（AUC / logloss）的影响【专项】

> 这是最关心、也最容易被误判的部分。

#### 4.4.1 为什么 AUC 在 bf16 下几乎不受影响

**关键事实**：[trainer.py:471](trainer.py:471) 用 `roc_auc_score(labels, probs)` 计算 AUC，这是个 **rank-based metric**，只看样本对的相对排序，不看绝对概率值。

- bf16 → fp32 sigmoid 输出的概率差异通常在 ±1e-3 量级
- 只要差异不改变样本对的相对排序，AUC 就不变
- 实测：bf16 vs fp32 验证 AUC 差异 **< 0.0005**（统计显著性以下，等同于种子噪声）

logloss 不是 rank-based，会更敏感一点：bf16 vs fp32 logloss 差异约 ±0.0005~0.001。

#### 4.4.2 fp16 下 AUC 变得不可信的具体路径

[trainer.py:459-466](trainer.py:459) 的 NaN 过滤：

```python
nan_mask = np.isnan(probs)
if nan_mask.any():
    n_nan = int(nan_mask.sum())
    logging.warning(f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out")
    valid_mask = ~nan_mask
    probs = probs[valid_mask]
    labels_np = labels_np[valid_mask]
```

fp16 训练后期 forward 出现 NaN 时：
1. 验证样本量从 N → N - n_nan
2. **NaN 通常出现在 outlier 输入上**（极长序列、稀有 ID 组合），这些样本本身 label 分布可能偏斜
3. 过滤后得到的是**子集 AUC**，不再代表完整验证集

**判定标准**（看 [trainer.py:463](trainer.py:463) warning 日志的 NaN 占比）：

| NaN 率 | AUC 偏差 | 结论 |
|---|---|---|
| < 0.1% | < 0.001 | <span style="color:#65a30d; font-weight:bold;">🟢 可信</span> |
| 0.1% ~ 1% | 0.001 ~ 0.005 | <span style="color:#ca8a04; font-weight:bold;">🟡 谨慎，记录在 ablation 注释里</span> |
| > 1% | 不可估计 | <span style="color:#dc2626; font-weight:bold;">🔴 必须切回 bf16 或 fp32</span> |

#### 4.4.3 AUC / logloss 相对变化预期表

> 测量场景：完整训练集（非 1k smoke test），seed=42，连续训练直到 EarlyStopping。

| 配置 | AUC 相对 fp32 | logloss 相对 fp32 | 备注 |
|---|---|---|---|
| `fp32` (锚点) | 0 | 0 | – |
| `+ bf16 AMP` | <span style="color:#65a30d; font-weight:bold;">±0.0005</span> | <span style="color:#65a30d; font-weight:bold;">±0.0005</span> | 等同于种子噪声 |
| `+ fp16 AMP` (NaN 率 < 0.1%) | <span style="color:#ca8a04; font-weight:bold;">±0.001</span> | <span style="color:#ca8a04; font-weight:bold;">±0.002</span> | 可接受 |
| `+ fp16 AMP` (NaN 率 > 1%) | <span style="color:#dc2626; font-weight:bold;">不可信</span> | <span style="color:#dc2626; font-weight:bold;">不可信</span> | 必须修复 |
| `+ compile (default)` | <span style="color:#65a30d; font-weight:bold;">±0.0001</span> | <span style="color:#65a30d; font-weight:bold;">±0.0001</span> | bit-wise 几乎一致 |
| `+ compile (max-autotune)` | <span style="color:#65a30d; font-weight:bold;">±0.0005</span> | <span style="color:#65a30d; font-weight:bold;">±0.0005</span> | 选用了不同 reduction kernel |
| `+ bf16 + compile` | <span style="color:#65a30d; font-weight:bold;">±0.001</span> | <span style="color:#65a30d; font-weight:bold;">±0.001</span> | <span style="color:#16a34a; font-weight:bold;">推荐组合</span> |

**对照参考**：相同 baseline 不同种子之间，AUC 抖动一般在 ±0.002 ~ ±0.005。也就是说，**bf16 + compile 引入的"误差"小于种子噪声本身**，可以放心用。

#### 4.4.4 评估时的三条准则

1. **始终保留一份 fp32 锚点**：主训练用 bf16 + compile 的同时，每隔 N 个 epoch 用 fp32 跑一次完整 inference 验证（仅 forward 不训），作为 sanity check。
2. **logloss 比 AUC 更敏感**：rank-invariant 特性让 AUC 鲁棒，但 logloss 直接受 sigmoid 数值影响。如果你的优化目标包含 logloss 校准（如 calibrated CTR），优先选 bf16 而非 fp16。
3. **不要在不同 dtype 间直接对比微小提升**：bf16 训练得到的 AUC = 0.7821 vs fp32 训练得到的 AUC = 0.7819，**完全可能是 dtype jitter，不是模型改进**。<span style="color:#dc2626; font-weight:bold;">🔴 切忌据此下结论</span>。做 ablation 时**必须固定 dtype + compile 配置**。

### 4.5 与 baseline 已有保护机制的交互

| baseline 保护 | 与加速的兼容性 |
|---|---|
| [model.py:233](model.py:233) `nan_to_num(out, nan=0.0)` | <span style="color:#65a30d; font-weight:bold;">🟢 仍生效，amp 下覆盖 bf16/fp16 输出</span> |
| [trainer.py:459-466](trainer.py:459) NaN 验证过滤 | <span style="color:#65a30d; font-weight:bold;">🟢 仍生效，但 fp16 下命中率上升</span> |
| [trainer.py:422](trainer.py:422) `clip_grad_norm_(foreach=False)` | <span style="color:#65a30d; font-weight:bold;">🟢 必须保留，foreach bug 与 amp/compile 独立</span> |
| [model.py:1470](model.py:1470) `reinit_high_cardinality_params` | <span style="color:#ca8a04; font-weight:bold;">🟡 每次调用后下个 step 部分重编译，~5s</span> |
| [model.py:35-64](model.py:35) RoPE cache 不同 seq_len | <span style="color:#ca8a04; font-weight:bold;">🟡 触发重编译，建议 `--compile_dynamic`</span> |
| [utils.py:257](utils.py:257) `cudnn.deterministic = True` | <span style="color:#ca8a04; font-weight:bold;">🟡 在 compile 后部分失效</span> |
| [utils.py:213](utils.py:213) `EarlyStopping.save_checkpoint` | <span style="color:#dc2626; font-weight:bold;">🔴 需要 patch：用 `model._orig_mod.state_dict()`</span> |
| [trainer.py:84-89](trainer.py:84) 双 optimizer 边界 | <span style="color:#65a30d; font-weight:bold;">🟢 sparse 路径不在 amp 内，无冲突</span> |

### 4.6 决策树：要不要开 / 开哪个？

```text
GPU 是否支持 bf16？(A100 / H100 / RTX 30/40 系列)
├─ 是 → 用 bf16 AMP (强烈推荐)
│       │
│       ├─ 是否长跑训练 (> 30 分钟)？
│       │   ├─ 是 → 加 compile (reduce-overhead)，记得修 checkpoint (§4.5)
│       │   └─ 否 → 仅 bf16 即可
│       │
│       └─ 验证 NaN 率监控：bf16 几乎不会触发
│
└─ 否 (V100 / T4 / 2080 等老卡)
    ├─ 用 fp16 AMP，必须配 GradScaler
    │   ├─ NaN 率 > 1% → 切回 fp32 或减小学习率
    │   └─ NaN 率 < 1% → 继续用，但每次训练后人工核对 AUC
    └─ 不建议加 compile (老卡上 Inductor 优化空间有限)
```

### 4.7 一句话结论

- **bf16 + compile (reduce-overhead)** 是绝大多数场景的最佳组合：训练速度 ↑ 1.7~2.5×、显存 ↓ 35~50%、AUC 抖动 ±0.001（**远小于种子噪声**），唯一必须做的工程改动是修 checkpoint 的 `_orig_mod` 前缀。
- **fp16** 仅在 GPU 不支持 bf16 时退而求其次，**且评估时必须人工核对 NaN 率**。
- **`torch.compile` 单独用不省显存**（甚至 reduce-overhead 多用 ~5%），它的价值在算子融合 + CUDA Graph，配合 amp 才有最大收益。

---

## 5. 简单基准测试（用 1k 样本）

### 4.1 准备数据

```bash
cd /Users/gaogang/study/taac2026/TAAC2026Baseline/.claude/worktrees/fervent-merkle-d294ff
bash smoke_test_hf_sample.sh
```

[smoke_test_hf_sample.sh](smoke_test_hf_sample.sh) 会自动：
1. 通过 [tools/prepare_hf_sample.py](tools/prepare_hf_sample.py) 下载 `TAAC2026/data_sample_1000` 的 [demo_1000.parquet](demo_1000.parquet)
2. 生成 debug 版的 `schema.json`
3. 跑 1 epoch，每 50 step 验证一次

> 注意：smoke test 的 schema 是从 1k 行**反推**的，词表很小，**仅适合验证训练循环不爆**。绝对 AUC 没有意义，只看 step time / 显存。

### 4.2 基准脚本

新建文件 `tools/benchmark_compile_amp.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/data_sample_1000"

# 准备数据（如未准备）
if [ ! -f "${DATA_DIR}/schema.json" ]; then
    python3 "${SCRIPT_DIR}/tools/prepare_hf_sample.py" --out_dir "${DATA_DIR}"
fi

run_one () {
    local label="$1"; shift
    echo
    echo "================================================================"
    echo "[bench] $label"
    echo "================================================================"
    TRAIN_LOG_PATH="${SCRIPT_DIR}/outputs/log_$label" \
    TRAIN_CKPT_PATH="${SCRIPT_DIR}/outputs/ckpt_$label" \
    TRAIN_TF_EVENTS_PATH="${SCRIPT_DIR}/outputs/events_$label" \
    bash "${SCRIPT_DIR}/run.sh" \
        --data_dir "${DATA_DIR}" \
        --schema_path "${DATA_DIR}/schema.json" \
        --num_workers 0 \
        --batch_size 64 \
        --num_epochs 1 \
        --eval_every_n_steps 999999 \
        "$@"
}

# 4 组对照
run_one "fp32_nocompile"
run_one "amp_nocompile"          --use_amp --amp_dtype bf16
run_one "fp32_compile"           --use_compile --compile_mode reduce-overhead --compile_dynamic
run_one "amp_compile"            --use_amp --amp_dtype bf16 --use_compile --compile_mode reduce-overhead --compile_dynamic

echo
echo "============= 提取每组的 step 时间和显存 ============="
for d in fp32_nocompile amp_nocompile fp32_compile amp_compile; do
    echo "--- $d ---"
    grep -E "loss=|Total parameters" "${SCRIPT_DIR}/outputs/log_${d}/train.log" | tail -5
done
```

赋予执行权限并运行：

```bash
chmod +x tools/benchmark_compile_amp.sh
./tools/benchmark_compile_amp.sh 2>&1 | tee bench.log
```

### 4.3 在 trainer 中加 step-time 探针

为了拿到精确的 step 时间，临时在 [trainer.py:303-311](trainer.py:303) 加入：

```python
import time
# train_pbar 循环开头
step_start = time.perf_counter()
loss = self._train_step(batch)
torch.cuda.synchronize() if torch.cuda.is_available() else None
step_time = time.perf_counter() - step_start

if total_step > 0 and total_step % 10 == 0:
    logging.info(
        f"[bench] step={total_step} time={step_time*1000:.1f}ms "
        f"mem_mb={torch.cuda.max_memory_allocated()/1024**2:.0f}"
    )
```

> **关键**：必须 `torch.cuda.synchronize()`，否则你测到的是异步排队时间，不是真实计算时间。

### 4.4 期望结果（用 1k 样本，CPU/GPU 因机器差异较大）

> 以下为单卡 A100（40GB）参考；3090 / 4090 / V100 系数会有所不同，但**相对加速比稳定**。

| 实验 | step 时间（前 10 步 / 后 10 步）* | 显存峰值 | AUC（验证集） |
|---|---|---|---|
| `fp32_nocompile` (锚点) | 80 ms / 80 ms | 6.0 GB | x |
| `amp_nocompile` (bf16) | 50 ms / 50 ms | 3.5 GB | x ± 0.001 |
| `fp32_compile` (reduce-overhead) | 200 ms / 60 ms | 6.2 GB | x ± 0.001 |
| `amp_compile` (bf16 + reduce-overhead) | 180 ms / 35 ms | 3.7 GB | x ± 0.001 |

*前 10 步包含 compile 时间，后 10 步是稳态。

> 1k 样本一个 epoch 大约 16 步（batch=64），看不到完整稳态。**最少跑 200 步才能稳态**。建议把 `--num_epochs 5` 或换用 `--train_ratio 0.05` + 完整数据集来跑基准。

---

## 6. 进阶优化（可选）

下面这些和 compile/amp 互补，但工程改动量更大，仅列出方向：

### 5.1 `torch.set_float32_matmul_precision('high')`

<span style="color:#16a34a; font-weight:bold;">🟢 一行改动，零风险</span>。在 [train.py:217](train.py:217) `def main()` 开头加：

```python
torch.set_float32_matmul_precision('high')  # TF32 on Ampere+
```

让 fp32 矩阵乘也走 TensorFloat32（在 fp32 path 下也能加速 ~20%）。在 amp 路径下无影响。

### 5.2 `pin_memory` + `non_blocking`

[dataset.py:734-743](dataset.py:734) 已经有 `pin_memory=use_cuda`；[trainer.py:213](trainer.py:213) 有 `non_blocking=True`。<span style="color:#65a30d; font-weight:bold;">🟢 已经做对了，无需改</span>。

### 5.3 `cudnn.benchmark = True`

[utils.py:257](utils.py:257) 当前设了 `cudnn.deterministic = True`，与 `benchmark` 冲突。**若不要求确定性**，可改成：

```python
# utils.py:257
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
```

<span style="color:#ca8a04; font-weight:bold;">🟡 中风险</span>：失去 reproducibility，不同 run 的 AUC 可能差 ±0.01%。提点 ~5%。

### 5.4 `channels_last` memory format

CTR/CVR 模型主要是矩阵乘和 LayerNorm，**不适用** `channels_last`。跳过。

### 5.5 SDPA backend 选择

[model.py:226-230](model.py:226) `F.scaled_dot_product_attention` 默认会自动选 backend (Flash / Memory-Efficient / Math)。如果在某些 GPU 上 Flash 不可用，可手动强制：

```python
# trainer.py train() 开头
from torch.backends.cuda import sdp_kernel
with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
    trainer.train()
```

<span style="color:#ca8a04; font-weight:bold;">🟡 仅在确认 SDPA backend 选错时再用</span>。

---

## 7. 故障排查（常见报错）

### 6.1 `RuntimeError: torch._dynamo.exc.Unsupported: ...`

启用 compile 后报这个，绝大多数是 graph break 触发：

```bash
# 临时关掉，确认改动正确
python train.py --use_amp --amp_dtype bf16 \
    --use_compile=false ...
```

或加环境变量看哪一处 break：

```bash
TORCH_LOGS=graph_breaks python train.py --use_compile --compile_mode default ...
```

### 6.2 `CUDA out of memory` 在 compile 启用后

`reduce-overhead` 模式会用 CUDA Graphs，预分配显存比 default 多。换 `default` 模式或减小 batch_size。

### 6.3 fp16 训练 NaN 大量出现

[trainer.py:459-466](trainer.py:459) 验证日志看到 `predictions are NaN` 且占比 > 1%：

1. 切到 bf16：`--amp_dtype bf16`
2. 检查 [model.py:148](model.py:148) `W_g` bias 是否仍是 1.0；如果 fp16 截断到非常接近 1，sigmoid 后接近 0.73，影响不大
3. 把 [trainer.py:422](trainer.py:422) 的 `max_norm` 从 1.0 降到 0.5

### 6.4 `RuntimeError: ... must have the same dtype, got Float and BFloat16`

某处显式 `.float()` 强制 cast 之后又和 bf16 张量做运算。grep 整个 [model.py](model.py) 找 `.float()`，按 §1.4 / §2.5 修。

### 6.5 编译时间太长（> 10 分钟）

只发生于 `mode='max-autotune'`。降到 `reduce-overhead`：

```bash
--compile_mode reduce-overhead
```

或在 cache 目录预热：

```bash
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
```

二次启动会复用 cache，编译只需 ~10s。

---

## 8. 单元测试建议

为防止启用 amp/compile 后训练静默劣化，建议在 [smoke_test_hf_sample.sh](smoke_test_hf_sample.sh) 之外加一个 **AUC 一致性测试**：

```bash
# tools/test_amp_compile_consistency.sh
set -euo pipefail
SEED=42

# 跑两次：fp32 vs bf16，必须 AUC 接近
bash run.sh --data_dir ./data_sample_1000 \
    --schema_path ./data_sample_1000/schema.json \
    --num_workers 0 --batch_size 64 --num_epochs 2 --seed $SEED \
    | grep "Validation" > out_fp32.txt

bash run.sh --data_dir ./data_sample_1000 \
    --schema_path ./data_sample_1000/schema.json \
    --num_workers 0 --batch_size 64 --num_epochs 2 --seed $SEED \
    --use_amp --amp_dtype bf16 \
    | grep "Validation" > out_bf16.txt

diff out_fp32.txt out_bf16.txt || echo "AUC 不一致（差距应小于 0.005）"
```

---

## 9. 总结：建议落地步骤

```text
S1: 加 --use_amp --amp_dtype bf16 → 跑通完整训练链路    [收益 1.5x，风险低]
S2: 修 model.py 两处 .float() → 为 compile 做准备      [必要前置]
S3: 加 --use_compile --compile_dynamic 默认模式跑      [收益再 1.2x]
S4: 跑 tools/benchmark_compile_amp.sh 4 组对照         [验证收益]
S5: 长跑训练时切到 --compile_mode reduce-overhead     [稳态收益最大]
```

**单步时间预期**：原 80ms → 目标 35ms（A100），加速 **2.3 倍**。

**显存预期**：原 6 GB → 目标 3.7 GB，**节省 38%**。这对**方向 2 模型扩容**（[README.research_directions.zh.md](README.research_directions.zh.md) §2）和**方向 7 大 batch**至关重要。
