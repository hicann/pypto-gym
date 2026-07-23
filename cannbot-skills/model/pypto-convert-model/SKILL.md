---
name: pypto-convert-model
description: Bidirectional model-format conversion between PyTorch / ONNX / safetensors with round-trip numerical verification. 在 PyTorch / ONNX / safetensors 之间双向自动转换并做 round-trip 数值校验。Triggers on "/pypto-convert-model", "port the model", "convert to onnx", "this .pt to safetensors", "模型转换", "转 onnx", "把 .pt 转成 safetensors".
---

# pypto-convert-model

Bidirectional any-to-any porter for `onnx ↔ pt (TorchScript) ↔ safetensors`. After every conversion, the converted artifact is reloaded and a dummy-input forward is compared against the source forward; `max_abs` diff is the headline number.

`onnx ↔ pt (TorchScript) ↔ safetensors` 三格式双向互转。每次转换后，会把产物重新加载，用同一个 dummy input 跑一次 forward，与源模型输出比较 `max_abs` 差异。

The skill runs on the bundled code under [scripts/](scripts/) (executable tools the agent invokes) and [references/](references/) (frozen test evidence + run notes). **Never re-implement conversion logic — always call [scripts/port.py](scripts/port.py).** To add a new model category or format, register it in [scripts/registry.py](scripts/registry.py) and refresh the matrix via [scripts/run.py](scripts/run.py).

本 skill 基于打包代码运行 -- [scripts/](scripts/) 放 agent 调用的工具，[references/](references/) 放冻结的测试证据与运行说明。**不要重写转换逻辑 —— 始终调用 [scripts/port.py](scripts/port.py)。** 若要新增模型类别或格式，在 [scripts/registry.py](scripts/registry.py) 注册后用 [scripts/run.py](scripts/run.py) 刷新矩阵。

## 1. Input gathering / 输入收集

If the user hasn't specified, ask via `AskUserQuestion`:

用户未明确给出时，用 `AskUserQuestion` 收集：

1. **Model path / repo id** — local file or `org/repo` (HF). When `safetensors` is on either end, the architecture is required, so an HF repo id is mandatory.
   **模型路径或 repo id** — 本地文件或 HF 的 `org/repo`。`safetensors` 出现在输入或输出端时必须提供 HF repo id（架构需要）。
2. **Output format** — one of `onnx` / `pt` / `safetensors`.
   **输出格式** — `onnx` / `pt` / `safetensors` 之一。
3. **Input shape** — comma-separated, e.g. `1,3,224,224`. Skip if the model card already implies it.
   **输入 shape** — 逗号分隔，例如 `1,3,224,224`。若可从模型卡推断则不必问。

Format is inferred from extension: `.onnx`, `.pt` / `.pth`, `.safetensors`. Override with `--input-format` if the extension lies.

文件扩展名可推断格式：`.onnx`, `.pt`/`.pth`, `.safetensors`。扩展名不对就用 `--input-format` 覆盖。

## 2. Environment check / 环境检查

```bash
python -c "import torch, onnx, onnxruntime, safetensors, onnx2torch; print('ok')"
```

Install only what's missing, picked from [requirements.txt](requirements.txt). Confirm with the user once.

只装缺失的包，从 [requirements.txt](requirements.txt) 挑。征求用户一次确认。

| Always required / 必装 | torch>=2.5, transformers>=4.46, onnx, onnxruntime, safetensors |
|---|---|
| onnx -> pt / safetensors | onnx2torch |
| safetensors I/O | timm or transformers (to instantiate the HF repo) |

## 3. Single-model conversion / 单模型转换 — `port.py`

```bash
# pt -> onnx (TorchScript self-contained, only --input-shape needed)
python scripts/port.py model.pt out.onnx --input-shape 1,3,224,224

# safetensors -> onnx (weights only; provide architecture via HF repo)
python scripts/port.py model.safetensors out.onnx \
    --hf-repo google/mobilenet_v2_1.0_224

# onnx -> pt (graph rebuilt via onnx2torch)
python scripts/port.py model.onnx out.pt --input-shape 1,3,224,224
```

Options / 选项:

- `--input-format` / `--output-format` — override extension inference / 覆盖扩展名推断
- `--hf-repo` — provide architecture for safetensors endpoints / 为 safetensors 端提供架构
- `--skip-verify` — disable round-trip diff (for huge models) / 关掉 round-trip 校验（大模型用）
- `-v` / `--verbose` — stage timings, forward output stats (min/max/mean/std/first5), diff distribution (p50/p95/p99), top-5 mismatch indices in raw logs / 各阶段计时、两端 forward 输出统计、diff 分布与 top-5 mismatch 索引
- env `CONVERT_EXP_DEVICE=cpu|cuda|xpu|mps|npu` — override auto-detection / 覆盖自动检测

Exit codes / 退出码: `0` PASS, `1` DIFF (converted but output diff exceeds tolerance / 转出但输出差异超出容差), `2` conversion failed / 转换失败.

## 4. Batch experiment / 批量实验 — `run.py`

To produce a PASS/FAIL matrix across multiple models:

跑多个模型生成 PASS/FAIL 矩阵：

```bash
# all 11 registry models / registry 中全部 11 个模型
python scripts/run.py

# subset / 部分
python scripts/run.py mobilenet_v2 toy_moe
```

Outputs / 产出:

Committed to the repo as frozen test evidence / 作为冻结测试证据 commit 到仓库：

- `references/results/<model>.json` — per-model detail / 每模型明细
- `references/matrix.md` — aggregate table / 汇总表

Runtime artifacts (gitignored; under `~/.cache/pypto-convert-model/` by default, override with `$CONVERT_MODEL_WORKDIR`) / 运行时产物（已 gitignore，默认在 `~/.cache/pypto-convert-model/` 下，可用 `$CONVERT_MODEL_WORKDIR` 覆盖）：

- `$WORKDIR/models/` — HF / timm download cache / 下载缓存
- `$WORKDIR/outputs/<model>/<fmt>/` — converted artifacts / 转换产物
- `$WORKDIR/logs/master.log` + `$WORKDIR/logs/<model>__<format>.log` — run logs / 运行日志

To add a new model, edit [scripts/registry.py](scripts/registry.py). For CNNs use `loader: "image-classification"` or `"timm"`; for transformer LMs use `"causal-lm"` / `"seq2seq-lm"`; for custom architectures use `"toy"` and define the module as in [scripts/toy_moe.py](scripts/toy_moe.py).

新增模型在 [scripts/registry.py](scripts/registry.py) 添加。CNN 用 `loader: "image-classification"` 或 `"timm"`；transformer LM 用 `"causal-lm"` / `"seq2seq-lm"`；自定义模型用 `"toy"` 类别，写法参考 [scripts/toy_moe.py](scripts/toy_moe.py)。

## 5. Device auto-detection / 设备自动选择

`port.py` / `run.py` call `pick_device()` at startup. Priority: NPU > CUDA/ROCm > XPU > MPS > CPU. ORT picks an execution provider from what's installed (TensorRT, CUDA, ROCm, CANN, QNN, ...). See [scripts/device_util.py](scripts/device_util.py).

`port.py` / `run.py` 启动时调用 `pick_device()`。优先级 NPU > CUDA/ROCm > XPU > MPS > CPU。ORT 从可用 provider 中按优先级挑（TensorRT、CUDA、ROCm、CANN、QNN 等）。详见 [scripts/device_util.py](scripts/device_util.py)。

Confirm at the first line of run output / 启动第一行可以确认：

```
[INFO] test device: cuda (cuda); ORT providers: ['CPUExecutionProvider']
```

## 6. Result reporting / 结果汇报

`port.py` per-run output / 单次输出:

```
pt -> onnx: wrote /tmp/out.onnx (46.75 MB)
device=cuda  status=PASS  max_abs=5.25e-06  mean_abs=1.08e-06
```

After a batch run, show [references/matrix.md](references/matrix.md) and, when a cell fails, name the cause in one line (most common: a `transformers>=5` regression breaks trace-based export — recommend `pip install "transformers<5"`).

批量执行后展示 [references/matrix.md](references/matrix.md)，失败格用一行点出原因（最常见：`transformers>=5` 回归导致 trace 导出失败，建议 `pip install "transformers<5"`）。

## Known limitations / 已知限制

- **transformer-MoE (Switch-T, Qwen-MoE)** — On `transformers>=5`, an internal attention regression breaks both `onnx` and `pt` export. `safetensors` always passes. Workaround: downgrade to `transformers<5`, or take weights-only via `safetensors`.
  `transformers>=5` 上 attention 路径回归，`onnx` 和 `pt` 都会失败；`safetensors` 始终通过。绕过：降级到 `transformers<5`，或只走 `safetensors`。
- **CUDA fp32 determinism** — cudnn introduces ~5e-4 noise between repeated forwards even with identical weights/inputs. `port.py` relaxes to `atol=5e-3` on CUDA (CPU check uses `atol=1e-4`).
  cudnn 即便权重和输入一致，重复 forward 也有 ~5e-4 量级噪声。`port.py` 在 CUDA 上自动放宽 `atol=5e-3`，CPU 校验则 `atol=1e-4`。
- **No `onnxruntime-gpu` wheel on aarch64** — ORT falls back to CPU even with a GPU; only the PyTorch reference uses the accelerator.
  aarch64 没有 `onnxruntime-gpu` 轮子；即使有 GPU，ORT 也是 CPU 执行，PyTorch reference 部分还是用加速器。
- **Unified memory (e.g. NVIDIA Grace)** — 14B+ fp32 models exceed RAM when moved to GPU. The `_hf_causal` loader defaults to fp16 + `low_cpu_mem_usage=True`.
  统一内存机器（如 NVIDIA Grace）上 14B+ 模型 fp32 在搬到 GPU 时会爆 RAM。`_hf_causal` loader 默认 fp16 + `low_cpu_mem_usage=True`。

## Caveats / 注意事项

- **Confirm with user before large downloads** — A 7B+ LLM is 14 GB+ on disk. Announce the cost.
  大量下载前先确认 —— 7B+ LLM 下载量 14GB+，磁盘/时间成本提前告知。
- **Gated models** — Set `HF_TOKEN` or run `huggingface-cli login`. Abort with a clear error if missing.
  Gated 模型需要 `HF_TOKEN` 或 `huggingface-cli login`，缺失时明确报错并中断。
- **Conversion can be lossy** — opset differences, dynamic axes, dtype: 1e-4 ~ 1e-3 drift is normal. Use the matrix's `max_abs` as the first trust signal.
  转换可能 lossy —— opset/动态轴/dtype 差异下 1e-4 ~ 1e-3 漂移属正常，`max_abs` 作首要判据。
- **Do not invent new conversion logic** — Use only the functions in [scripts/converters.py](scripts/converters.py). If a new format is truly required, add it there and register in the `CONVERTERS` dict.
  不要随手写新的转换逻辑 —— 只用 [scripts/converters.py](scripts/converters.py) 的函数；确需新格式时在那里加并注册到 `CONVERTERS` 字典。
