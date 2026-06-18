# Phi-3-mini-4k-instruct NPU 迁移说明

## 模型信息

| 字段 | 值 |
|------|-----|
| HuggingFace | microsoft/Phi-3-mini-4k-instruct |
| 权重目录 | {model_dir}/{model_name} |
| 代码来源 | trust_remote_code (HuggingFace仓库自带) |
| 参数量 | ~3.8B (BF16) |
| 精度 | FP16 (运行时转) |

## 环境信息

| 字段 | 版本 |
|------|------|
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| torchvision | 0.24.0 |
| transformers | 4.41.2 (HF 要求: 4.41.2) |
| CANN | 8.3.RC1 |
| NPU | Ascend 910B2 |

## 运行命令

所有脚本通过环境变量 `PHI3_MODEL_PATH` 指定模型权重目录：

```bash
export PHI3_MODEL_PATH=/path/to/Phi-3-mini-4k-instruct

# 基础推理
python3 scripts/ask_Phi-3-mini-4k-instruct.py --prompt "你好" --device <NPU卡号>

# PyPTO 融合算子模式
python3 scripts/ask_Phi-3-mini-4k-instruct.py --prompt "你好" --device <NPU卡号> --use_pypto

# 基准测试
bash scripts/bench_Phi-3-mini-4k-instruct.sh
```

> 也可通过 `--model-path` 参数直接指定路径，优先级高于环境变量。

## 性能对比

| 模式 | 命令 | 模型加载 | 推理耗时 | 吞吐 | 峰值显存 |
|------|------|---------|---------|------|---------|
| baseline | `python3 scripts/ask_Phi-3-mini-4k-instruct.py --prompt "你好" --device 4 --output_length 50` | 7.218s | 4.632s | 10.8 tok/s | 7327.8 MB |
| pto | `python3 scripts/ask_Phi-3-mini-4k-instruct.py --prompt "你好" --device 4 --output_length 50 --use_pypto` | 6.833s | 12.209s | 4.1 tok/s | 7327.8 MB |

> 单算子替换时 PTO 比基线慢 2-3x 属正常（JIT 首编 + kernel launch 开销），收益来自多算子融合。

## 归档映射

| 来源 (`{model_dir}/`) | 目标 (`{pypto_gym_repo}/`) | 操作 |
|---|---|---|
| `scripts/*` | `modeling/transformers/phi_3_mini_4k_instruct/` | 全量覆盖 |
| `config.json` | `src/pypto_gym/transformers/phi_3_mini_4k_instruct/` | 覆盖 |
| `core/*.py` | `src/pypto_gym/transformers/phi_3_mini_4k_instruct/` | 全量覆盖 |
| `pto_kernels/` (除去 golden + test) | `src/pypto_gym/ops/pypto_tile/phi_3_mini_4k_instruct/` | 顶层覆盖 |
| `pto_kernels/rms_norm/test/` | `tests/ops/phi_3_mini_4k_instruct/` | 新建
