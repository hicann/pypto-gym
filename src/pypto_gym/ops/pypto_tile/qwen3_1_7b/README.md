# Qwen3-1.7B PyPto 算子

本目录提供 Qwen/Qwen3-1.7B 的自定义 PyPto 融合算子，用于在 Ascend NPU 上替代 PyTorch 原生算子链路、提升推理性能。

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

每个 `qwen3_*.py` 算子都有对应的 `test_*.py` 同目录测试用例（pypto-gym 惯例）：

```bash
cd src/pypto_gym/ops/qwen3_1_7b
TILE_FWK_DEVICE_ID=7 python3 test_pre_attn_fused.py
TILE_FWK_DEVICE_ID=7 python3 test_k3.py
TILE_FWK_DEVICE_ID=7 python3 test_decode_attn.py
```

## 集成与端到端

模型执行脚本与端到端推理见：
- `modeling/qwen3_1_7b/`：模型 golden 与网络真实形状的整层测试
- `models/qwen3_1_7b/`：方案文档 (SPEC/DESIGN/FINAL_REPORT)
- 端到端 ask 脚本部署在模型 repo 下：`/data/z00885570/models/Qwen3-1.7B/qwen3_pto_kernels/` (运行时适配层) + `scripts/ask_Qwen3-1.7B_pto.py`

## 性能（稳态，Ascend 910）

| 配置 | Decode 稳态 | vs Baseline |
|---|---|---|
| torch_npu Baseline | 31 ms/token | 1.0× |
| **PyPto Fused (默认)** | **48 ms/token** | **1.55×** |
| PyPto + decode-attn fused | 59 ms/token | 1.9× |

详细见 `models/qwen3_1_7b/FINAL_REPORT.md`。
