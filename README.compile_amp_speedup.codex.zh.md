# Codex 执行版：torch.compile + AMP 混合精度训练加速

本文是 [README.compile_amp_speedup.zh.md](README.compile_amp_speedup.zh.md) 的 Codex 实施版。目标不是重新解释原理，而是给出一份可以交给 Codex 逐步执行的工程清单：改哪些文件、保留哪些边界、怎样验证、怎样回滚。

适用分支：`Baseline_compile_speedup`

当前代码状态：

- `train.py` 还没有 `--use_amp` / `--use_compile` 相关 CLI。
- `trainer.py` 训练和验证仍是 fp32 路径。
- `model.py` 仍有两处 `.float()` 掩码转换，会降低 AMP + compile 的收益。
- `run.sh` 已经把所有额外参数透传给 `train.py`，所以新增 CLI 后可以直接用 `bash run.sh ...` 启动。

## 0. Codex 执行目标

实现完成后，下面命令必须可运行：

```bash
bash run.sh \
  --use_amp \
  --amp_dtype bf16 \
  --use_compile \
  --compile_mode reduce-overhead \
  --compile_dynamic
```

新增能力必须满足：

| 项目 | 要求 |
|---|---|
| 默认行为 | 不带新 flag 时仍保持 fp32、不开 compile |
| AMP 默认推荐 | `bf16`，只有不支持 bf16 时才考虑 `fp16` |
| sparse optimizer | 继续使用 Adagrad，不要合并进 dense AdamW |
| gradient clipping | 必须保留 `foreach=False` |
| checkpoint | `model.pt` 中不能出现 `_orig_mod.` 前缀 |
| evaluation/infer | 不要求 inference 也 compile；训练 checkpoint 必须能被原 eval 路径加载 |

## 1. 直接给 Codex 的执行提示词

如果要让 Codex 真正修改代码，可以直接使用这段提示词：

```text
请按 README.compile_amp_speedup.codex.zh.md 执行 P0-P6。
目标是在当前 TAAC2026Baseline 分支中启用可选的 AMP 混合精度和 torch.compile 加速。
默认训练行为必须不变；只在显式传入 --use_amp / --use_compile 时启用。
必须保护 sparse Adagrad、clip_grad_norm_(foreach=False)、checkpoint 无 _orig_mod. 前缀。
完成后运行 py_compile、train.py --help 检查，并给出本机无法运行 GPU smoke test 时的原因。
不要提交 .claude/。
```

## 2. P0：前置检查

先确认工作区和入口文件：

```bash
git status --short --branch
python3 train.py --help | grep -E "loss_type|sparse_lr|ns_tokenizer_type"
nl -ba train.py | sed -n '137,180p'
nl -ba train.py | sed -n '315,350p'
nl -ba trainer.py | sed -n '70,120p'
nl -ba trainer.py | sed -n '402,494p'
nl -ba model.py | sed -n '476,482p'
nl -ba model.py | sed -n '713,715p'
```

当前应看到：

- `train.py` 的 Loss CLI 在 `--focal_gamma` 后结束，后面是 sparse optimizer 参数。
- `train.py` 在构造 `PCVRHyFormer(**model_args).to(args.device)` 后直接进入日志和 trainer。
- `trainer.py` 的 optimizer 从 `model.get_sparse_params()` / `model.get_dense_params()` 拆分。
- `trainer.py` 的 `_train_step` 直接 fp32 forward、loss、backward。
- `model.py` 的 `valid_mask.unsqueeze(-1).float()` 和 `(~new_padding_mask).unsqueeze(-1).float()` 仍存在。

## 3. P1：修改 `train.py`

### 3.1 新增 CLI

在 `--focal_gamma` 后、`# Sparse optimizer.` 前加入：

```python
    # Mixed precision / compile acceleration.
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Enable CUDA autocast mixed precision training')
    parser.add_argument('--amp_dtype', type=str, default='auto',
                        choices=['auto', 'bf16', 'fp16'],
                        help='AMP dtype: auto = bf16 if supported else fp16')
    parser.add_argument('--use_compile', action='store_true', default=False,
                        help='Enable torch.compile for the model forward path')
    parser.add_argument('--compile_mode', type=str, default='reduce-overhead',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')
    parser.add_argument('--compile_dynamic', action='store_true', default=False,
                        help='Enable dynamic shapes for torch.compile')
```

### 3.2 包装 `torch.compile`

在 `model = PCVRHyFormer(**model_args).to(args.device)` 后，先保留原始模型引用，再可选 compile：

```python
    model = PCVRHyFormer(**model_args).to(args.device)
    raw_model = model
```

日志统计仍使用 `raw_model`：

```python
    num_ns = raw_model.num_ns
    total_params = sum(p.numel() for p in raw_model.parameters())
```

在参数统计日志之后、创建 `EarlyStopping` 之前加入：

```python
    if args.use_compile:
        if not hasattr(torch, 'compile'):
            raise SystemExit("torch.compile requires PyTorch >= 2.0")
        if not str(args.device).startswith('cuda'):
            logging.warning("torch.compile is enabled on a non-CUDA device; "
                            "this is allowed but usually not useful for this workload")
        logging.info(
            f"Compiling model with mode={args.compile_mode}, "
            f"dynamic={args.compile_dynamic}"
        )
        model = torch.compile(
            raw_model,
            mode=args.compile_mode,
            dynamic=args.compile_dynamic,
            fullgraph=False,
        )
```

注意：不要把 `raw_model` 传给 trainer；trainer 的 forward 需要使用 compile 后的 `model`。但 trainer 内部必须能 unwrap 到 `raw_model` 来取 sparse 参数和保存 checkpoint，见 P2。

## 4. P2：修改 `trainer.py`

### 4.1 import `nullcontext`

文件顶部加入：

```python
from contextlib import nullcontext
```

### 4.2 保存 compile 前的原始模型引用

在 `__init__` 里，`self.model = model` 后加入：

```python
        self.base_model: nn.Module = getattr(model, '_orig_mod', model)
```

之后涉及参数分组、保存、reinit 的地方都应使用 `self.base_model`。

### 4.3 optimizer 参数分组改用 `base_model`

把 optimizer 初始化区域从：

```python
        if hasattr(model, 'get_sparse_params'):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
```

改成：

```python
        if hasattr(self.base_model, 'get_sparse_params'):
            sparse_params = self.base_model.get_sparse_params()
            dense_params = self.base_model.get_dense_params()
```

fallback 的 AdamW 也建议使用 `self.base_model.parameters()`，避免 compile wrapper 参数命名污染：

```python
            self.dense_optimizer = torch.optim.AdamW(
                self.base_model.parameters(), lr=lr, betas=(0.9, 0.98)
            )
```

### 4.4 AMP 初始化

在 `self.train_config = train_config` 后加入：

```python
        cfg = train_config or {}
        self.use_amp: bool = bool(cfg.get('use_amp', False))
        self.amp_dtype: torch.dtype = torch.float32
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None

        if self.use_amp and not (str(device).startswith('cuda') and torch.cuda.is_available()):
            logging.warning("AMP requested but CUDA is not available; disabling AMP")
            self.use_amp = False

        if self.use_amp:
            requested_dtype = cfg.get('amp_dtype', 'auto')
            if requested_dtype == 'auto':
                self.amp_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            elif requested_dtype == 'bf16':
                self.amp_dtype = torch.bfloat16
            elif requested_dtype == 'fp16':
                self.amp_dtype = torch.float16
            else:
                raise ValueError(f"Unsupported amp_dtype: {requested_dtype}")

            if self.amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                raise RuntimeError("amp_dtype=bf16 requested but this CUDA device does not support bf16")

            logging.info(f"AMP enabled: dtype={self.amp_dtype}")
            if self.amp_dtype == torch.float16:
                self.scaler = torch.cuda.amp.GradScaler()
```

### 4.5 新增 autocast helper

在 `_make_model_input` 前加入：

```python
    def _amp_context(self):
        """Return the active autocast context or a no-op context."""
        if self.use_amp:
            return torch.cuda.amp.autocast(dtype=self.amp_dtype)
        return nullcontext()
```

### 4.6 替换 `_train_step`

将 `_train_step` 替换为：

```python
    def _train_step(self, batch: Dict[str, Any]) -> float:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch['label'].float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        model_input = self._make_model_input(device_batch)

        with self._amp_context():
            logits = self.model(model_input)  # (B, 1)
            logits = logits.squeeze(-1)  # (B,)

            if self.loss_type == 'focal':
                loss = sigmoid_focal_loss(
                    logits, label,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, label)

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.scaler.unscale_(self.sparse_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.base_model.parameters(), max_norm=1.0, foreach=False)
            self.scaler.step(self.dense_optimizer)
            if self.sparse_optimizer is not None:
                self.scaler.step(self.sparse_optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.base_model.parameters(), max_norm=1.0, foreach=False)
            self.dense_optimizer.step()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.step()

        return loss.item()
```

关键点：

- `clip_grad_norm_` 必须继续使用 `foreach=False`。
- fp16 使用 `GradScaler` 时，dense 和 sparse optimizer 都要 unscale；否则 sparse Adagrad 会收到被 scale 过的梯度。
- bf16 不需要 scaler。

### 4.7 修改 `_evaluate_step`

只包 forward，不改变 AUC / logloss 计算：

```python
        model_input = self._make_model_input(device_batch)
        with self._amp_context():
            logits, _ = self.model.predict(model_input)  # (B, 1), (B, D)
        logits = logits.squeeze(-1)  # (B,)
```

### 4.8 checkpoint 和 reinit 都使用 `base_model`

把 `_save_step_checkpoint` 中的保存改成：

```python
            torch.save(self.base_model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
```

把 `_handle_validation_result` 中两次 `self.early_stopping(..., self.model, ...)` 改成：

```python
            self.early_stopping(val_auc, self.base_model, {
```

和：

```python
        self.early_stopping(val_auc, self.base_model, {
```

把 epoch 末尾 reinit 区域改成：

```python
                reinit_ptrs = self.base_model.reinit_high_cardinality_params(
                    self.reinit_cardinality_threshold)
                sparse_params = self.base_model.get_sparse_params()
```

## 5. P3：修改 `utils.py`，兜底处理 compiled model

虽然 P2 已经让 trainer 传 `base_model`，仍建议在 `EarlyStopping.save_checkpoint` 中做兜底。把：

```python
        torch.save(model.state_dict(), self.checkpoint_path)
```

改成：

```python
        state_model = getattr(model, '_orig_mod', model)
        torch.save(state_model.state_dict(), self.checkpoint_path)
```

这样即使未来其他调用方传入 compiled model，checkpoint 也不会带 `_orig_mod.` 前缀。

## 6. P4：修改 `model.py` 两处 mask dtype

这一步是为 AMP + compile 减少 dtype promotion 和 graph break。改动很小，但如果目标是绝对 fp32 数值可复现，可以先跳过这一步，只跑 P1-P3 验证训练链路。

### 6.1 `MultiSeqQueryGenerator.forward`

把：

```python
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, L_i, 1)
```

改成：

```python
            valid_mask_expanded = valid_mask.unsqueeze(-1).to(seq_tokens_list[i].dtype)  # (B, L_i, 1)
```

### 6.2 `LongerEncoder._gather_top_k`

把：

```python
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()
```

改成：

```python
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).to(top_k_tokens.dtype)
```

## 7. P5：验证命令

### 7.1 语法检查

在 macOS 沙箱中，优先把 pycache 写到 `/private/tmp`：

```bash
PYTHONPYCACHEPREFIX=/private/tmp/taac_pycache \
python3 -m py_compile train.py trainer.py model.py utils.py
```

如果本机默认 Python 没装依赖，只做语法检查仍应通过；真正训练需要装 `torch`、`pyarrow`、`pandas`、`scikit-learn`。

### 7.2 CLI 检查

```bash
python3 train.py --help | grep -E "use_amp|amp_dtype|use_compile|compile_mode|compile_dynamic"
```

必须看到 5 个新参数。

### 7.3 1k smoke test

先准备样本：

```bash
python3 tools/prepare_hf_sample.py --out_dir data_sample_1000
```

跑 baseline：

```bash
bash run.sh \
  --data_dir data_sample_1000 \
  --schema_path data_sample_1000/schema.json \
  --num_workers 0 \
  --batch_size 64 \
  --num_epochs 1 \
  --eval_every_n_steps 999999
```

只跑 AMP：

```bash
bash run.sh \
  --data_dir data_sample_1000 \
  --schema_path data_sample_1000/schema.json \
  --num_workers 0 \
  --batch_size 64 \
  --num_epochs 1 \
  --eval_every_n_steps 999999 \
  --use_amp \
  --amp_dtype bf16
```

跑 AMP + compile：

```bash
bash run.sh \
  --data_dir data_sample_1000 \
  --schema_path data_sample_1000/schema.json \
  --num_workers 0 \
  --batch_size 64 \
  --num_epochs 1 \
  --eval_every_n_steps 999999 \
  --use_amp \
  --amp_dtype bf16 \
  --use_compile \
  --compile_mode default \
  --compile_dynamic
```

smoke test 只验证训练链路是否爆炸，不判断 AUC 高低。1k 样本的 AUC 没有统计意义。

### 7.4 checkpoint key 检查

训练后找一个 `model.pt`，检查 key：

```bash
python3 - <<'PY'
import sys
import torch

path = sys.argv[1]
state = torch.load(path, map_location='cpu')
bad = [k for k in state.keys() if k.startswith('_orig_mod.')]
print(f"num_keys={len(state)}")
print(f"num_orig_mod_keys={len(bad)}")
if bad:
    print(bad[:5])
    raise SystemExit(1)
PY outputs/ckpt/<checkpoint-dir>/model.pt
```

`num_orig_mod_keys` 必须是 `0`。

## 8. P6：可选 benchmark 脚本

需要对比 4 组配置时，新增 `tools/benchmark_compile_amp.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${SCRIPT_DIR}/data_sample_1000"

python3 "${SCRIPT_DIR}/tools/prepare_hf_sample.py" --out_dir "${DATA_DIR}"

run_one() {
  local label="$1"
  shift
  echo "===== ${label} ====="
  TRAIN_LOG_PATH="${SCRIPT_DIR}/outputs/log_${label}" \
  TRAIN_CKPT_PATH="${SCRIPT_DIR}/outputs/ckpt_${label}" \
  TRAIN_TF_EVENTS_PATH="${SCRIPT_DIR}/outputs/events_${label}" \
  bash "${SCRIPT_DIR}/run.sh" \
    --data_dir "${DATA_DIR}" \
    --schema_path "${DATA_DIR}/schema.json" \
    --num_workers 0 \
    --batch_size 64 \
    --num_epochs 1 \
    --eval_every_n_steps 999999 \
    "$@"
}

run_one "fp32_nocompile"
run_one "amp_nocompile" --use_amp --amp_dtype bf16
run_one "fp32_compile" --use_compile --compile_mode default --compile_dynamic
run_one "amp_compile" --use_amp --amp_dtype bf16 --use_compile --compile_mode default --compile_dynamic
```

运行：

```bash
chmod +x tools/benchmark_compile_amp.sh
./tools/benchmark_compile_amp.sh 2>&1 | tee outputs/benchmark_compile_amp.log
```

注意：`default` 模式适合 smoke benchmark；完整训练再切到 `reduce-overhead`。

## 9. 失败判定和回滚

| 现象 | 处理 |
|---|---|
| `torch.compile requires PyTorch >= 2.0` | 当前环境不能开 compile，只保留 AMP |
| `bf16 requested but device does not support bf16` | 改用 `--amp_dtype fp16`，并重点看 NaN 率 |
| `predictions are NaN` 超过 1% | 不接受该实验结果；先切回 `bf16` 或禁用 AMP |
| compile 首 batch 卡住 | 正常，等待 1-3 分钟；smoke 用 `--compile_mode default` |
| checkpoint 有 `_orig_mod.` 前缀 | 立即修 P2/P3，不要继续训练 |
| AUC 与 fp32 差超过 0.005 | 固定 seed，重跑 fp32 与 bf16；若仍复现，先禁用 P4 的 dtype mask 改动 |

完全回滚运行时行为只需要去掉新 flag：

```bash
bash run.sh
```

代码回滚时优先回滚 `train.py` / `trainer.py` / `utils.py` / `model.py` 中本文件列出的块，不要动数据、schema、checkpoint 目录。

## 10. 推荐落地顺序

```text
P0  读取当前代码并确认没有冲突改动
P1  train.py 加 CLI 和 torch.compile 包装
P2  trainer.py 加 AMP、base_model unwrap、checkpoint/reinit 修正
P3  utils.py 对 compiled model state_dict 做兜底
P4  model.py 两处 mask dtype 优化
P5  py_compile + train.py --help + 1k smoke test
P6  需要量化收益时再加 benchmark 脚本
```

第一轮建议只跑：

```bash
bash run.sh --use_amp --amp_dtype bf16
```

确认 AMP 不影响训练后，再跑：

```bash
bash run.sh --use_amp --amp_dtype bf16 --use_compile --compile_mode reduce-overhead --compile_dynamic
```
