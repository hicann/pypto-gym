# Qwen3.5-9B NPU 迁移说明

| 字段 | 说明 |
|------|------|
| HuggingFace | Qwen/Qwen3.5-9B |
| 权重目录 | /mnt/workspace/gitCode/cann/models/pure/Qwen3.5-9B |
| 代码来源 | transformers 包 (built-in，使用 qwen3_5 架构) |
| 运行命令 | `python3 scripts/ask_Qwen3.5-9B.py` |
| transformers 版本 | 5.6.1 |
| 代码位置 | core/modeling_qwen3_5.py, core/configuration_qwen3_5.py |
| 修改内容 | 修复导入（transformers绝对导入）、添加auto_map到config.json |

## 已融合算子 (Fused operators)

| Operator | Toggle flag | Path |
|----------|-------------|------|
| `gated_delta_rule` | `USE_PTO_GATED_DELTA_RULE` | [`src/pypto_gym/ops/pypto_tile/qwen3_5_9b/gated_delta_rule/`](../../../src/pypto_gym/ops/pypto_tile/qwen3_5_9b/gated_delta_rule/) |

The fused operator replaces the chunk-prefill path in
`Qwen3_5GatedDeltaNet.forward` (linear-attention layers). The decode and
full-attention paths are unaffected.

`--use_pypto` 模式依赖运行时 `qwen3_5_9b_pto_kernels/` 适配层包：脚本在导入
`transformers` 之前 `sys.path.insert(0, model_path)` 并 `import
qwen3_5_9b_pto_kernels`。该适配层包预期部署在权重目录下
（`{weights_dir}/qwen3_5_9b_pto_kernels/`），由模型侧维护。

## 运行

```bash
# Baseline
python3 ask_Qwen3.5-9B.py --model-path <weights_dir>

# PyPTO fused
python3 ask_Qwen3.5-9B.py --model-path <weights_dir> --use_pypto
```

Benchmark:

```bash
bash bench_Qwen3.5-9B.sh
```
