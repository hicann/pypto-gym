# GutenOCR-3B RMSNorm 算子集成

## 概述

将 RMSNorm 算子从原始 torch 实现替换为 PyPTO 融合算子，提升 NPU 上的推理性能。

## 文件结构

```
pypto_gym/ops/pypto_tile/gutenocr_3b/
└── rms_norm/
    ├── __init__.py                # 模块导出
    ├── rms_norm_impl.py           # PyPTO kernel实现
    └── README.md                  # 本文档
```

## 技术说明

### 实现方式

使用 PyPTO 内置 `pypto.rms_norm` 融合算子：

```python
@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128},
    pass_options={"cube_l1_reuse_setting": {-1: 4}},
)
def rms_norm_impl_kernel(hidden_states, weight, output, epsilon):
    y = pypto.rms_norm(hidden_states, weight, epsilon)
    output[:] = y
```

### 融合优化

PyPTO `rms_norm` 融合以下算子为单个 kernel：
- `torch.mean(x**2, dim=-1, keepdim=True)` → 平方均值
- `torch.rsqrt(mean + eps)` → 平方根倒数
- `x * rsqrt * weight` → 缩放加权

**单算子性能**: +60% (相比torch baseline)

### 数据类型支持

- 输入: `torch.bfloat16` / `torch.float16`
- 输出: `torch.bfloat16` / `torch.float16`
- 权重: `torch.bfloat16` / `torch.float16`

## 测试验证

### 单算子测试

```bash
cd tests/ops/gutenocr_3b
python3 test_rms_norm.py
```

### 集成测试

```bash
# 标准prompt真实性能测试
python3 scripts/test_batch16_standard_prompt.py

# 多batch测试
python3 scripts/ask_gutenocr_3b_compile.py --batch 16 --use_dynamic_config --device 1
```

### 测试用例来源

从 GutenOCR-3B 模型打点采集的真实 shape/dtype：
- prefill 主 norm: `[batch, seq_len, 2048]`
- prefill q_norm: `[batch, seq_len, 16, 128]`
- prefill k_norm: `[batch, seq_len, 8, 128]`
- decode 主 norm: `[batch, 1, 2048]`

## 性能数据

### 单算子性能（Swimlane泳道图）

> 测试环境: Ascend 910B, CANN 9.0.0
> 测试方法: `debug_options={"runtime_debug_mode": 1}`，解析 `merged_swimlane.json`

| Shape | dtype | kernel耗时(μs) | 提升 |
|-------|-------|--------------|------|
| [1, 11, 2048] | BF16 | 53.2 → 32.1 | **+60%** |
| [1, 1, 2048] | BF16 | 28.1 → 16.8 | **+60%** |
| [1, 11, 16, 128] | BF16 | 31.9 → 19.1 | **+59%** |
| [1, 1, 16, 128] | BF16 | 6.6 → 4.0 | **+60%** |

### 端到端性能（标准prompt）

| Batch | Baseline吞吐 | RMSNorm启用 | RMSNorm禁用 | 最优 |
|-------|-------------|-----------|-----------|------|
| **1** | 25.14 | 25.43 (+1.4%) | **27.08** (+7.8%) | 禁用 ⭐⭐⭐ |
| **4** | 22.55 | 23.28 (+3.2%) | **25.66** (+13.7%) | 禁用 ⭐⭐⭐ |
| **8** | 20.91 | 19.36 (-7.4%) | **24.27** (+15.7%) | 禁用 ⭐⭐⭐ |
| **16** | 16.60 | 16.80 (+1.2%) | **17.12** (+3.1%) | 禁用 ⭐⭐ |

**结论**: ❌ **不推荐启用RMSNorm**（固化开销抵消优化）

## 状态

✅ 环境验证
✅ 网络基线验证
✅ 打点采集
✅ Golden编写
✅ 单算子精度验证
✅ 单算子性能采集（+60%）
✅ 模型集成（已集成到modeling_gutenocr_3b.py）
✅ 端到端验证（真实性能测试）
❌ **推荐禁用**（固化开销抵消优化）

## 集成说明

### 模型集成位置

文件: `modeling_gutenocr_3b.py`
类: `Qwen2RMSNorm`
行号: Line 47

集成方式：
```python
class Qwen2RMSNorm(nn.Module):
    def forward(self, hidden_states):
        # sys.modules固化机制
        pto_kernels = sys.modules.get("gutenocr_3b_pto_kernels")
        
        if pto_kernels and pto_kernels.USE_PTO_RMS_NORM:
            return pto_kernels.rms_norm_wrapper(hidden_states, self.weight, self.variance_epsilon)
        
        # torch fallback
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states
```
