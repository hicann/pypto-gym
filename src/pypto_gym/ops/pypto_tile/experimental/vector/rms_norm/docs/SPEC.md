---
schema_version: 1
op_name: RMSNorm
supported_dtypes: ["float32"]
p0_shapes: [[16, 64, 256, 256]]
tolerance: {"rtol": 0.001, "atol": 0.001}
dynamic_axes: ["B"]
dynamic_axes_ranges: {"B": [1, 1024]}
shape_constraints: {"input_rank": 4, "feature_dim": 1}
default_params: {"eps": 1e-5, "num_features": 64}
perf_target: null
---

## 算子需求规范

### 1. 基础信息
- **算子名称**: RMSNorm
- **算子分类**: normalization

### 1.1 功能描述

RMS Normalization (Root Mean Square Normalization)。对输入张量沿特征维度计算均方根(RMS)，并用 RMS 归一化输入。与 LayerNorm 不同，RMSNorm 不需要减去均值，仅除以 RMS 值。

### 1.2 算法参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| num_features | int | 64 | 特征维度大小（dim=1 的大小） |
| eps | float | 1e-5 | 防止除零的小常数 |

### 1.3 数学公式

$$
\text{out}[b, c, h, w] = \frac{x[b, c, h, w]}{\sqrt{\frac{1}{C}\sum_{j=0}^{C-1} x[b, j, h, w]^2 + \varepsilon}}
$$

其中 C 为 num_features（特征维度大小），ε 为 eps。

### 2. 关键特性

本算子为简单归一化算子，无复杂特性需求。

### 4. 数据流图

```
    输入 x                          输出 y
┌────────────────────┐         ┌────────────────────┐
│  [B, 64, 256, 256] │         │  [B, 64, 256, 256] │
│     float32        │ ──────▶ │     float32        │
└────────────────────┘  RMSNorm└────────────────────┘

公式: y = x / sqrt(mean(x^2, dim=1, keepdim=True) + eps)
动态轴: B (batch 维度)
```

### 5. 输入输出规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------|-------|--------|--------|------|
| x | [B, num_features, H, W] | float32 | B | ✓ 高 | 输入张量, B=batch_size, num_features=64, H=W=256 |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------|-------|--------|--------|------|
| y | [B, num_features, H, W] | float32 | B | ✓ 高 | RMS归一化后的输出, shape与输入一致 |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 | 是 | 0.001 | 0.001 | 默认 |

### 7. 精度要求
- **atol**: 0.001
- **rtol**: 0.001

### 8. 动态轴说明
- **动态轴**: ["B"]
- **轴含义**: B = batch_size (第0维)
- **取值范围**: B ∈ [1, 1024]

### 9. 边界条件处理
- **零值**: 正常计算 (输入全零时输出也为零)
- **极值**: 正常计算
- **NaN/Inf**: 正常计算

### 10. 性能要求
- **性能目标**: 无特殊要求

### 11. 参考信息
- **参考实现**: PyTorch: `y = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)`
- **论文**: Root Mean Square Normalization (Zhang & Sennrich, 2019)
- **类似算子**: LayerNorm, InstanceNorm

### 12. 应用场景
- **目标模型**: 通用 (LLM、Transformer 等)
- **使用位置**: 归一化层

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | eps=1e-5, num_features=64 | [16, 64, 256, 256] | [16, 64, 256, 256] | 核心性能场景 |
| 功能_P0 | 功能 | P0 | eps=1e-5, num_features=64 | [16, 64, 256, 256] | [16, 64, 256, 256] | 核心功能验证 |

---
*生成时间: 2026-05-16*
*确认状态: 已确认 (基于 REQUIRE.md 直接规格)*
*置信度说明: ✓ 高（REQUIRE.md 包含完整规格，含数学公式、shape、dtype、精度要求）*
