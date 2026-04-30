# Evaluation Failed 分析与修复方案

## 1. 当前检查结论

`evaluation/` 目录当前包含：

```text
evaluation/dataset.py
evaluation/infer.py
evaluation/model.py
```

已确认：

- `evaluation/model.py` 与根目录 `model.py` 完全一致；
- `evaluation/infer.py` 可以编译通过；
- `evaluation/dataset.py` 与根目录 `dataset.py` 不完全一致，但差异主要是评测侧缺列补零逻辑，属于可接受的评测容错；
- 当前主跑模型结构包含 `output_include_ns=True`，这会改变 `output_proj` 输入维度，因此 evaluation 必须用正确的 `train_config.json` 重建模型。

因此，evaluation failed 更可能不是语法问题，而是 **checkpoint/配置/评测运行环境路径** 问题。

## 2. 最可能失败原因

### 原因一：`MODEL_OUTPUT_PATH` 指向了错误目录

`evaluation/infer.py` 当前假设：

```text
MODEL_OUTPUT_PATH = 某个 global_step*.best_model 子目录
```

这个目录里必须有：

```text
model.pt
schema.json
train_config.json
```

如果评测平台传入的是 checkpoint 根目录，例如：

```text
ckpt/
  global_step1000.layer=2.head=4.hidden=64.best_model/
    model.pt
    schema.json
    train_config.json
```

而不是具体 `.best_model` 子目录，那么当前 `get_ckpt_path()` 只会在 `MODEL_OUTPUT_PATH` 当前层找 `*.pt`，找不到就失败。

失败表现通常类似：

```text
No *.pt file found under MODEL_OUTPUT_PATH=...
```

### 原因二：`train_config.json` 缺失或不是当前 checkpoint 的配置

现在模型结构依赖这些字段：

```text
use_token_gnn
token_gnn_layers
output_include_ns
use_seq_graph
use_aligned_dense_int_graph
user_ns_tokens
item_ns_tokens
emb_skip_threshold
```

特别是：

```text
output_include_ns=True
```

会改变 head shape。训练时如果打开了它，但 evaluation 重建模型时没有读到这个配置，就会出现 strict load shape mismatch。

失败表现通常类似：

```text
size mismatch for output_proj.0.weight
missing keys / unexpected keys
```

### 原因三：fallback 配置不应该承担正式评测

`evaluation/infer.py` 有 `_FALLBACK_MODEL_CFG`，但 fallback 只能用于旧 checkpoint 或 smoke test。正式评测应始终读取 checkpoint 里的 `train_config.json`。

如果 fallback 被触发，说明 checkpoint 包不完整。即使 fallback 当前被手动同步到 `run.sh`，也仍然有风险：

- 如果评测的是旧 4LayerGNN checkpoint，fallback 的 `output_include_ns=True` 会错；
- 如果评测的是新 checkpoint 但 fallback 缺某个结构字段，也会错；
- 如果 `emb_skip_threshold` 不一致，embedding 表数量可能错。

### 原因四：评测环境的 DataLoader worker 限制

`infer.py` 使用：

```python
num_workers = int(train_config.get('num_workers', _FALLBACK_NUM_WORKERS))
prefetch_factor=2
```

如果训练配置记录了 `num_workers=8`，评测沙箱不允许多进程或共享内存不足，可能在 DataLoader 阶段失败。

如果某些环境把 `num_workers` 改为 0，而代码仍然传 `prefetch_factor=2`，PyTorch 版本可能报错。

失败表现可能是 worker exit、shared memory、multiprocessing 或 prefetch_factor 相关错误。

### 原因五：输出 JSON 格式或预测数量不匹配

当前输出：

```python
predictions = {
    "predictions": dict(zip(all_user_ids, all_probs)),
}
```

这沿用了原 baseline，但有一个潜在风险：如果测试集中 `user_id` 不唯一，`dict` 会去重，最终 JSON 中预测条数小于样本数。

如果评测端要求每条样本一个预测，而不是按 user_id 去重，这会导致 evaluation failed 或样本数不匹配。

需要根据官方评测要求确认：

- 是否要求 `{user_id: score}`；
- 是否要求按输入顺序输出 score list；
- 是否存在单用户多样本。

## 3. 推荐修复顺序

### Step 1：让 `infer.py` 明确打印关键路径

在 `main()` 里打印：

```text
MODEL_OUTPUT_PATH
EVAL_DATA_PATH
EVAL_RESULT_PATH
schema_path
train_config exists?
ckpt_path
model_cfg
```

目的：先确认评测平台到底给了哪个 checkpoint 目录。

### Step 2：增强 checkpoint 查找

把 `get_ckpt_path()` 从“只查当前目录”升级为：

1. 如果 `MODEL_OUTPUT_PATH/*.pt` 存在，直接用；
2. 否则查找 `MODEL_OUTPUT_PATH/global_step*.best_model/model.pt`；
3. 优先选择包含 `.best_model` 的目录；
4. 找到 checkpoint 后，把 `model_dir` 切换到该 checkpoint 所在目录，以便读取同目录的 `schema.json` 和 `train_config.json`。

这能解决平台传入 checkpoint 根目录的问题。

### Step 3：正式评测要求必须有 `train_config.json`

建议改成：

```text
如果 checkpoint 目录没有 train_config.json，直接报错并打印目录内容。
```

不要在正式评测时静默 fallback。因为当前模型结构变化较多，fallback 很容易构建出错误 shape。

保留 fallback 只用于本地 smoke test。

### Step 4：DataLoader 推理配置不要继承训练 worker

评测推理建议强制更稳：

```python
num_workers = int(os.environ.get("EVAL_NUM_WORKERS", 0))
```

并且：

```python
loader_kwargs = {
    "batch_size": None,
    "num_workers": num_workers,
    "pin_memory": torch.cuda.is_available(),
}
if num_workers > 0:
    loader_kwargs["prefetch_factor"] = 2
```

这样可以避免评测沙箱多进程失败。

### Step 5：验证输出格式

如果官方要求每条样本一个预测，建议输出 list：

```json
{
  "predictions": [0.1, 0.2, 0.3]
}
```

如果官方要求按 `user_id` 映射，则保留当前 dict，但需要确认 `user_id` 唯一。

为了排查，可以在 `infer.py` 输出前打印：

```text
total_test_samples
len(all_probs)
len(all_user_ids)
len(set(all_user_ids))
```

如果：

```text
len(set(all_user_ids)) < len(all_probs)
```

说明 dict 输出会丢预测。

## 4. 推荐最终 evaluation 结构

正式提交时，`evaluation/` 应包含：

```text
evaluation/
  infer.py
  model.py
  dataset.py
```

同时 checkpoint 目录必须包含：

```text
model.pt
schema.json
train_config.json
```

如果使用 `ns_groups_json ""`，不需要额外提交 `ns_groups.json`。

## 5. 最小修复补丁方向

优先改 `evaluation/infer.py`：

1. 新增 `resolve_model_dir_and_ckpt_path()`，支持 checkpoint 根目录和 `.best_model` 子目录；
2. 强制从实际 checkpoint 目录读取 `schema.json`、`train_config.json`；
3. 如果 `train_config.json` 缺失，正式模式直接报错；
4. DataLoader 默认 `num_workers=0`，只在 worker > 0 时传 `prefetch_factor`；
5. 输出前记录预测数量和 user_id 去重数量。

## 6. 当前最可能的真实修复

如果你这次 evaluation failed 没有详细栈，我最优先怀疑：

```text
MODEL_OUTPUT_PATH 指向 checkpoint 根目录，而 infer.py 只在当前层找 model.pt
```

第二怀疑：

```text
train_config.json 没被评测读取，output_include_ns 导致 output_proj shape mismatch
```

第三怀疑：

```text
评测 DataLoader 使用训练时 num_workers=8，沙箱 worker 失败
```

建议下一步先把错误栈贴出来；如果没有错误栈，就直接按 Step 2、Step 3、Step 4 做稳健化修改。

## 7. 已实施的稳健化修改

当前已经修改 `evaluation/infer.py`：

- 支持 `MODEL_OUTPUT_PATH` 指向：
  - 具体 `model.pt` 文件；
  - 具体 `global_step*.best_model/` 目录；
  - checkpoint 根目录；
  - checkpoint 根目录下更深层的 `global_step*.best_model/model.pt`。
- 自动优先选择 `.best_model` checkpoint；如果有多个，优先较大的 `global_step`。
- 找到 checkpoint 后，将 `model_dir` 切换到真实 `model.pt` 所在目录。
- `schema.json` 查找顺序改为：
  1. 真实 checkpoint 目录；
  2. 原始 `MODEL_OUTPUT_PATH` 根目录；
  3. `EVAL_DATA_PATH`。
- `train_config.json` 查找顺序改为：
  1. 真实 checkpoint 目录；
  2. 原始 `MODEL_OUTPUT_PATH` 根目录。
- `ns_groups.json` 同样支持从真实 checkpoint 目录或原始根目录解析。
- Evaluation DataLoader 默认 `num_workers=0`，避免评测沙箱多进程失败；如需开启可设置：

```bash
EVAL_NUM_WORKERS=2
```

- 只有 `num_workers > 0` 时才传 `prefetch_factor=2`，避免 PyTorch 在单进程 DataLoader 下报错。
- 输出前检查 `len(all_probs) == total_test_samples`，避免静默少预测。
- 输出前记录 `user_id` 去重数量；如果存在重复 user_id，会保留 baseline 的 dict 输出格式并给出 warning。
