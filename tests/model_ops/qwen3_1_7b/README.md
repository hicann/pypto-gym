# Qwen3-1.7B PyPTO 算子测试

## 测试文件结构

```
tests/ops/qwen3_1_7b/
├── rope_golden.py       # RoPE Golden 参考实现（部分融合算子）
├── test_rope.json       # RoPE 测试用例配置
├── test_rope.py         # RoPE 精度测试脚本
├── rms_norm_golden.py   # RMSNorm Golden 参考实现
├── test_cases.json      # RMSNorm 测试用例配置
└── test_rms_norm.py     # RMSNorm 精度测试脚本
```

## 实际集成算子

| 算子 | 融合范围 | 测试文件 |
|------|---------|---------|
| **RoPE (部分融合)** | Q/K per-head RMSNorm + RoPE | `test_rope.py` |
| RMSNorm | 单独算子（未集成到模型） | `test_rms_norm.py` |

## 运行测试

### 环境配置

```bash
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
export TILE_FWK_DEVICE_ID=0  # NPU device ID
```

### RoPE 测试

```bash
# 列出所有测试用例
python test_rope.py --list

# 运行所有 RoPE 测试
python test_rope.py

# 只测试 Q kernel
python test_rope.py --kernel q

# 只测试 K kernel
python test_rope.py --kernel k

# 运行单个用例
python test_rope.py rope_q_001
```

### RMSNorm 测试

```bash
# 列出所有测试用例
python test_rms_norm.py --list

# 运行所有测试
python test_rms_norm.py

# 运行单个用例
python test_rms_norm.py rms_norm_001
```

## 测试用例说明

### RoPE 测试用例 (test_rope.json)

| ID | Description | Shape | Kernel |
|----|-------------|-------|--------|
| `rope_q_001` | Q single token | [1, 16, 128] | `qwen3_qk_rope_q` |
| `rope_q_002` | Q multi token | [32, 16, 128] | `qwen3_qk_rope_q` |
| `rope_q_003` | Q long sequence | [128, 16, 128] | `qwen3_qk_rope_q` |
| `rope_k_001` | K single token | [1, 8, 128] | `qwen3_qk_rope_k` |
| `rope_k_002` | K multi token | [32, 8, 128] | `qwen3_qk_rope_k` |
| `rope_k_003` | K long sequence | [128, 8, 128] | `qwen3_qk_rope_k` |
| `rope_edge_001` | Odd sequence | [7, 16, 128] | `qwen3_qk_rope_q` |
| `rope_edge_002` | Short sequence | [4, 8, 128] | `qwen3_qk_rope_k` |

### 精度要求

- **RoPE**: rtol=1e-2, atol=1e-2 (BF16精度)
- **RMSNorm**: rtol=1e-2, atol=1e-2 (FP16精度)

## Golden 实现说明

### RoPE Golden (rope_golden.py)

部分融合算子，包含：

1. Per-head RMSNorm: `x [S,N,D] -> norm(x)`
2. RoPE: `norm(x) * cos + rotate_half(norm(x)) * sin`

```python
def rope_golden_3d(x, cos, sin, norm_weight, eps):
    # Step 1: RMSNorm
    normed = rms_norm_per_head(x, norm_weight, eps)
    
    # Step 2: RoPE
    cos_expanded = cos.unsqueeze(1)  # [S, 1, D]
    sin_expanded = sin.unsqueeze(1)
    output = normed * cos_expanded + rotate_half(normed) * sin_expanded
    
    return output
```

### RMSNorm Golden (rms_norm_golden.py)

单独算子，标准 RMSNorm 实现：

```python
def rms_norm_golden(hidden_states, weight, eps):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states
```
