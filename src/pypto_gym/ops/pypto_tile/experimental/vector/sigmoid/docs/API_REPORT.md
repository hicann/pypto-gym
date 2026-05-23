---
schema_version: 1
op_name: Sigmoid
supported_dtypes: [DT_FP32]
dynamic_axes: ['B']
shape_constraints: "输出 shape 与输入 shape 完全一致"
tiling_required: true
feasibility: feasible
---

# API 探索报告

> **生成时间**: 2026-05-16

---

## 1. 概述

### 1.1 输入摘要

Sigmoid 激活函数：逐元素计算 `σ(x) = 1 / (1 + exp(-x))`。
- 输入: x [B, 16384], float32
- 输出: y [B, 16384], float32
- 动态轴: B (batch 维度)

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: Sigmoid 是逐元素（element-wise）一元激活函数，无矩阵乘法或归约操作，属于 Vector 计算类型

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | elementwise (activation) | y = σ(x) = 1/(1+exp(-x)) | 逐元素 Sigmoid 激活 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | y = σ(x) | `pypto.sigmoid(x)` | direct | ✓ |

### 3.2 Substitute 配方

无需 substitute，`pypto.sigmoid` 直接可用。

备选方案（如需降级）：
```
sigmoid(x): neg = pypto.mul(x, -1.0); exp_neg = pypto.exp(neg); ones = pypto.full(exp_neg.shape, 1.0, exp_neg.dtype, valid_shape=exp_neg.shape); result = pypto.div(ones, pypto.add(exp_neg, 1.0))
```

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | torch.float32 | float32 | ✓ |
| contiguous | tensor.is_contiguous() == True | 需确保输入连续 | 需确保 |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| pypto.sigmoid | dtype | 仅 DT_FP32 | ✓ |
| pypto.sigmoid | 空 Tensor | 不支持空 Tensor | ✓ (输入 [B, 16384]) |
| pypto.sigmoid | shape size | ≤ INT32_MAX (2,147,483,647) | ✓ |
| pypto.from_torch | contiguous | tensor.is_contiguous() == True | 需确保 |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

推荐 Tiling 配置（参考 activation.py）：
```python
# 根据输入维度动态配置
pypto.set_vec_tile_shapes(32, ...)  # 根据实际 shape 调整
```

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/02_intermediate/operators/activation/activation.py` | examples | 高 | 高 | `pypto.sigmoid()` 直接调用、configure_tiling() 动态 tiling、golden 验证模式 |
| `examples/03_advanced/patterns/function/function.py` | examples | 高 | 高 | 动态 tiling `[32 for _ in range(len(x.shape))]`、JIT 函数组合 |
| `models/arctic/sum_lstm.py` | models | 高 | 高 | LSTM 门控中多次调用 `pypto.sigmoid()`、vec tile 配置、loop_unroll 模式 |

### 6.2 可复用模式

- **API 调用模式**：`out[:] = pypto.sigmoid(x)` — 直接赋值到输出 tensor
- **Tiling 策略**：`pypto.set_vec_tile_shapes(32, ...)` — 根据输入维度动态配置
- **Loop 结构**：纯 element-wise 无需显式 loop，PyPTO 自动分块
- **边界处理**：无需特殊边界处理，依赖 `pypto.sigmoid` 内部实现

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| 输出方式 | 示例中多为 `out[:] = x * pypto.sigmoid(x)` (SiLU) | 本算子仅需 `out[:] = pypto.sigmoid(x)` | 去掉乘法部分，直接赋值 |
| 动态轴 | 示例多为静态 shape | 本算子有动态轴 B | 使用 `pypto.DYNAMIC` 声明动态维度，配合 loop 切 tile |

---

## 7. 风险评估

### 7.1 阻断问题

无阻断问题。`pypto.sigmoid` 直接 API 可用，FP32 精度匹配需求。

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| dtype 限制 | `pypto.sigmoid` 仅支持 DT_FP32，不支持 FP16/BF16；本算子仅要求 FP32，无影响 |
| contiguous | 需确保输入 tensor 为 contiguous，wrapper 中应调用 `.contiguous()` |
| 动态 shape | 需使用 `pypto.DYNAMIC` 声明 B 维度，并在 kernel 中用 loop 遍历 |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` (line 97) |
| sigmoid API 文档 | `docs/api/operation/pypto-sigmoid.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Vec Tiling 配置 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| 最佳参考实现 | `examples/02_intermediate/operators/activation/activation.py` |
| 模型级参考 | `models/arctic/sum_lstm.py` |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**: 无
- **推荐方案**: 直接使用 `pypto.sigmoid()` API，参考 `activation.py` 的 tiling 配置和 golden 验证模式
