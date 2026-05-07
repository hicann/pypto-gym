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
TILE_FWK_DEVICE_ID=7 python3 test_pre_attn_fused.py
TILE_FWK_DEVICE_ID=7 python3 test_k3.py
TILE_FWK_DEVICE_ID=7 python3 test_decode_attn.py
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
