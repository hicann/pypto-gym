# Spatial-SSRL-3B 算子测试

## 目录结构

```
tests/ops/spatial_ssrl_3b/
├── rms_norm_golden.py           # RMSNorm Golden 参考实现
├── test_rms_norm.py             # RMSNorm 测试脚本
├── test_cases.json              # RMSNorm 测试用例
├── rope_golden.py               # RoPE Golden 参考实现（新增）
├── test_rope.py                 # RoPE 测试脚本（新增）
├── test_cases_rope.json         # RoPE 测试用例（新增）
└── README.md                    # 本文档
```

## 测试算子

### 1. RMSNorm

**Golden 实现**: `rms_norm_golden.py`
- 场景A：直接复制原始代码（纯 PyTorch）
- 来源：Qwen3-1.7B/core/modeling_qwen3.py Qwen3RMSNorm.forward

**测试脚本**: `test_rms_norm.py`

**测试用例**: `test_cases.json`
- case_001: prefill阶段 input_layernorm/post_layernorm/final_norm
- case_002: prefill阶段 q_norm (num_heads=16)
- case_003: prefill阶段 k_norm (num_kv_heads=8)
- case_004: decode阶段 input_layernorm/post_layernorm/final_norm
- case_005: decode阶段 q_norm
- case_006: decode阶段 k_norm

### 2. RoPE（新增）

**Golden 实现**: `rope_golden.py`
- 场景A：直接复制原始代码（纯 PyTorch）
- 来源：Qwen2.5-VL/core/modeling_qwen2_5_vl.py
- 包含两种实现：
  - `apply_rotary_pos_emb_vision_golden`: Vision RoPE (2D)
  - `apply_multimodal_rotary_pos_emb_golden`: Multimodal RoPE (3D)

**测试脚本**: `test_rope.py`

**测试用例**: `test_cases_rope.json`
- case_001: multimodal RoPE - prefill阶段 (batch=1, seq_len=31)
- case_002: multimodal RoPE - decode阶段 (batch=1, seq_len=1)
- case_003: multimodal RoPE - 大batch场景 (batch=4, seq_len=128)
- case_004: vision RoPE - 图像编码器 (seq_len=100, num_heads=16)
- case_005: vision RoPE - 小序列 (seq_len=32)
- case_006: multimodal RoPE - float32精度测试

## 使用方法

### 前置条件

```bash
# 设置设备ID（NPU）
export TILE_FWK_DEVICE_ID=4

# 或使用 CPU
unset TILE_FWK_DEVICE_ID
```

### RMSNorm 测试

```bash
# 运行所有 RMSNorm 测试
cd tests/ops/spatial_ssrl_3b
python3 test_rms_norm.py

# 列出所有测试用例
python3 test_rms_norm.py --list

# 运行单个用例
python3 test_rms_norm.py case_001
```

### RoPE 测试

```bash
# 运行所有 RoPE 测试
cd tests/ops/spatial_ssrl_3b
python3 test_rope.py

# 列出所有测试用例
python3 test_rope.py --list

# 运行单个用例
python3 test_rope.py case_001
```

## 测试流程

每个测试脚本执行以下步骤：

1. **加载测试用例**: 从 JSON 文件读取 shape/dtype/参数
2. **生成随机输入**: 根据 seed 固定随机性
3. **计算 Golden**: 使用纯 PyTorch 参考实现
4. **计算 PyPTO**: 调用算子实现
5. **精度对比**: numpy.testing.assert_allclose
6. **Shape/Dtype 验证**: 确保输出符合预期

## 精度标准

| dtype | rtol | atol |
|-------|------|------|
| float16 | 1e-3 | 1e-3 |
| float32 | 1e-5 | 1e-5 |
| bfloat16 | 1e-2 | 1e-2 |

## 测试用例来源

所有测试用例均来自模型打点采集的真实场景：

- **RMSNorm**: 从 Spatial-SSRL-3B 模型 prefill/decode 阶段采集
- **RoPE**: 从 Qwen2.5-VL 模型 Vision/Language Attention 采集

## 实现对应关系

| 算子 | Golden 文件 | PyPTO 实现文件 |
|------|-------------|---------------|
| RMSNorm | `rms_norm_golden.py` | `src/pypto_gym/ops/pypto_tile/spatial_ssrl_3b/rms_norm/rms_norm_impl.py` |
| RoPE (Vision) | `rope_golden.py` | `src/pypto_gym/ops/pypto_tile/spatial_ssrl_3b/rope/rope_impl.py` |
| RoPE (Multimodal) | `rope_golden.py` | `src/pypto_gym/ops/pypto_tile/spatial_ssrl_3b/rope/rope_impl.py` |

## 状态追踪

### RMSNorm
✅ Golden 编写
✅ 测试用例编写
✅ 测试脚本编写
✅ 单算子精度验证
✅ 模型集成验证

### RoPE
✅ Golden 编写
✅ 测试用例编写
✅ 测试脚本编写
⏳ 单算子精度验证
⏳ 模型集成验证