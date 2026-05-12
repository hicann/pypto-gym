# Qwen3-1.7B PyPTO 算子

本目录提供 Qwen/Qwen3-1.7B 的自定义 PyPTO 融合算子，用于在 Ascend NPU 上替代 PyTorch 原生算子链路、提升推理性能。

## 算子清单

| 文件 | 说明 | 输入 → 输出 | 替代 |
|---|---|---|---|
| `qwen3_pre_attn_fused.py` | **核心融合 #1** | hidden + cos/sin + 4 weights → q/k/v 3D | RMSNorm + QKV proj + Q/K-norm + RoPE |
| `qwen3_k3_post_attn.py` | **核心融合 #2** | attn_out + residual + 5 weights → y | O-proj + 残差 + RMSNorm + SwiGLU MLP + 残差 |
| `qwen3_decode_attn.py` | 可选 | q + K/V cache + mask → out | Sq=1 decode attention（GQA-native, online softmax）|
| `qwen3_iter1a_kernel.py` | 备选拆分版 | — | RMSNorm + QKV proj |
| `qwen3_iter1b_kernel.py` | 历史版本 | — | 已被 `qwen3_pre_attn_fused.py` 取代 |
| `qwen3_k2_qk_rope.py` | 备选拆分版 | — | Q/K per-head RMSNorm + RoPE |

## 单元测试

对应测试位于 `tests/ops/qwen3_1_7b/`：

```bash
cd tests/ops/qwen3_1_7b
python3 test_pre_attn_fused.py
python3 test_k3.py
python3 test_decode_attn.py
```

## 集成与端到端

- `tests/ops/qwen3_1_7b/`：算子 golden 与网络真实形状的整层测试
- 端到端部署请参考模型侧 `qwen3_pto_kernels/` 适配层

## 性能（稳态，Ascend 910）

| 配置 | Decode 稳态 | vs Baseline |
|---|---|---|
| torch_npu Baseline | 31 ms/token | 1.0× |
| **PyPTO Fused (默认)** | **48 ms/token** | **1.55×** |
| PyPTO + decode-attn fused | 59 ms/token | 1.9× |

## 单算子 Kernel 性能

> 测试环境: Ascend 910B, CANN 8.5.0
>
> 测试方法: Swimlane (泳道图) — `debug_options={"runtime_debug_mode": 1}`, 解析 `merged_swimlane.json` X 事件 span

| 算子 | 输入 shape | 输入 dtype | kernel 耗时 (μs) | 数据来源 |
|--------|-----------|-----------|-----------------|---------|
| decode_attn | q=[16,128], kv=[8,64,128] (S=1) | bfloat16 | 58.4 | Swimlane |
| pre_attn_fused | x=[1,2048], QKV/rope (S=1) | bfloat16 | 78.4 | Swimlane |
| k2_Q RoPE | x=[32,16,128] | bfloat16 | 32.0 | Swimlane |
| k2_K RoPE | x=[32,8,128] | bfloat16 | 27.3 | Swimlane |
| k3_post_attn | attn=[32,2048], MLP | bfloat16 | 174.7 | Swimlane |

> RMSNorm 性能数据见 [rms_norm/README.md](rms_norm/README.md)
