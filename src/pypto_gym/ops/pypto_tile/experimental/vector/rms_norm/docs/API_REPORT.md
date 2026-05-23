---
schema_version: 1
op_name: RMSNorm
supported_dtypes: ["float32"]
dynamic_axes: ["B"]
shape_constraints: {"input_rank": 4, "feature_dim": 1, "feature_size": 64}
tiling_required: true
feasibility: "可行"
---

# API 探索报告

> **生成时间**: 2026-05-16

---

## 1. 概述

### 1.1 输入摘要

RMSNorm (Root Mean Square Normalization) 算子，对输入张量沿特征维度(dim=1)计算 RMS 并归一化。

公式: `out[b, c, h, w] = x[b, c, h, w] / sqrt(mean(x[b, j, h, w]^2) + eps)`

输入: x [B, 64, 256, 256] float32, 输出: y [B, 64, 256, 256] float32

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: 仅涉及逐元素运算和归约运算，无矩阵乘法

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | elementwise mul | `sq = x * x` | 逐元素平方 |
| 2 | reduction sum | `s = sum(sq, dim=1, keepdim=True)` | 沿特征维度求和 |
| 3 | elementwise div | `mean_sq = s / C` | 除以特征维度大小(C=64) |
| 4 | elementwise add | `mean_sq_eps = mean_sq + eps` | 加 epsilon |
| 5 | elementwise sqrt | `rms = sqrt(mean_sq_eps)` | 计算均方根 |
| 6 | elementwise div | `out = x / rms` | 归一化 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | `sq = x * x` | `pypto.mul(x, x)` 或 `x * x` | direct | ✓ |
| 2 | `s = sum(sq, dim=1, keepdim=True)` | `pypto.sum(sq, dim=1, keepdim=True)` | direct | ✓ |
| 3 | `mean_sq = s / C` | `pypto.div(s, C)` 或 `s / C` | direct | ✓ |
| 4 | `mean_sq_eps = mean_sq + eps` | `pypto.add(mean_sq, eps)` 或 `mean_sq + eps` | direct | ✓ |
| 5 | `rms = sqrt(mean_sq_eps)` | `pypto.sqrt(mean_sq_eps)` | direct | ✓ |
| 6 | `out = x / rms` | `pypto.div(x, rms)` 或 `x / rms` | direct | ✓ |

**注意**: 原生 `pypto.rms_norm` API 存在，但其 reduction 维度固定为最后一维(dim=-1)，而本算子需要沿 dim=1 归一化，因此需要手动分解实现。

### 3.2 Substitute 配置

无需 substitute，所有原子操作均有直接 API 支持。

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP32 支持 | float32 | ✓ |
| contiguous | 必须 | torch.randn 生成默认连续 | ✓ |
| shape | 非空, 1-4维 | [16, 64, 256, 256] (4维) | ✓ |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| pypto.mul | dtype | DT_FP32 | ✓ |
| pypto.mul | shape | 1-4维, Size ≤ INT32_MAX | ✓ (16*64*256*256=67108864) |
| pypto.sum | dtype | DT_FP32 | ✓ |
| pypto.sum | TileShape | TileShape ≤ 64KB | ✓ 需合理设置 |
| pypto.sum | dim | 支持任意轴 | ✓ dim=1 |
| pypto.add | dtype | DT_FP32 | ✓ |
| pypto.sqrt | dtype | DT_FP32 | ✓ |
| pypto.div | dtype | DT_FP32 | ✓ |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

**注意事项**:
- `pypto.sum` 后 keepdim=True 时，TileShape 维度与输入一致
- sum(dim=1, keepdim=True) 输出 shape 为 [tile_b, 1, tile_h, tile_w]，但 TileShape 仍按 4 维设置
- 各步骤间可能需要重新设置 TileShape（特别是 sum 后维度变化时）
- 输出通过 `pypto.assemble()` 或 `out[:] = result` 写回

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` | examples | **HIGH** | **HIGH** | 完整 RMSNorm kernel: sum→div→sqrt→div, `rms_norm_core()`, `NormConfig` dataclass, `assemble` 输出 |
| `models/glm_v4_5/glm_attention_fusion.py` (rms_norm_bias, L63-92) | models | **HIGH** | **HIGH** | 生产级: cast→mul→mul(mean_coff)→sum(-1)→add(eps)→sqrt→div→mul(gamma)→add(bias)→cast, 动态 batch loop |
| `models/deepseek_v4/hc_pre_impl.py` (rms_norm_denom, L22-28) | models | **HIGH** | **HIGH** | 精简版: `x*x → sum(-1) → /N → sqrt(...+eps)`, Python 运算符重载 |
| `models/deepseek_v4/mla_prolog_v4_impl.py` (rms_norm, L153-182) | models | **HIGH** | **HIGH** | 完整: reciprocal 模式(full+div), FP32 cast, loop_unroll 集成 |
| `examples/02_intermediate/operators/softmax/softmax.py` | examples | MEDIUM | **HIGH** | reduce+normalize 模式, 动态 shape, pypto.loop + view |

### 6.2 可复用模式

- **API 调用模式**: `x * x` (运算符重载) → `pypto.sum(sq, dim, keepdim=True)` → `result / scalar` → `pypto.sqrt(x + eps)` → `x / rms`
- **Tiling 策略**: `pypto.set_vec_tile_shapes(...)` 在 JIT kernel 开头设置；参考 layer_norm.py 用 (64, 128) for 2D input
- **Loop 结构**: 动态 batch 维度使用 `pypto.loop()` + `pypto.view()` 切片；静态 batch 可直接全量计算
- **边界处理**: epsilon 加到 sum 结果上防止除零；无需特殊 NaN/Inf 处理

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| Reduction 维度 | dim=-1 (最后一维) | dim=1 (第二维) | 需确认 pypto.sum 支持 dim=1；参考示例均用 dim=-1 |
| 输入维度 | 2D [batch, hidden] | 4D [B, C, H, W] | TileShape 需设置为 4 维 |
| Gamma 参数 | 有 gamma 缩放 | 无 gamma（仅除以 rms） | 不需要 gamma 相关逻辑 |
| 数据类型 | BF16 输入 + FP32 计算 | FP32 输入和计算 | 无需 cast 转换 |
| Batch 动态 | 动态 batch + loop | 动态 B 轴 | 需使用 pypto.DYNAMIC + loop 处理 |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| pypto.sum(dim=1) 兼容性 | 所有参考实现均用 dim=-1 | 需验证 dim=1 是否正常工作；若不支持，需考虑 transpose 后再 sum(dim=-1) |
| TileShape 维度变化 | sum(dim=1, keepdim=True) 后 TileShape 可能需重新设置 | 参考 pypto.sum 文档: keepdim=True 时 TileShape 维度不变 |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| pypto.sum TileShape ≤ 64KB | sum 操作的 TileShape 大小受 64KB 限制，需合理切分 |
| 输出写入方式 | pypto 不存在 save API，需用 `pypto.assemble()` 或 `out[:] = result` |
| 动态轴 B | 需在 from_torch 时设置 `dynamic_axis=[0]`，并用 pypto.loop 处理动态 batch |
| FP32 精度 | 默认使用 HIGH_PRECISION div（除法）和 INTRINSIC sqrt（开方），精度应满足 rtol=0.001, atol=0.001 |
| eps 作为标量参数 | 需通过 config/dataclass 传入 JIT kernel，不能直接作为 tensor 参数 |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` |
| pypto.mul 文档 | `docs/api/operation/pypto-mul.md` |
| pypto.sum 文档 | `docs/api/operation/pypto-sum.md` |
| pypto.add 文档 | `docs/api/operation/pypto-add.md` |
| pypto.sqrt 文档 | `docs/api/operation/pypto-sqrt.md` |
| pypto.div 文档 | `docs/api/operation/pypto-div.md` |
| pypto.rms_norm 文档 | `docs/api/operation/pypto-rms_norm.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Tiling 配置 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| DataType 枚举 | `docs/api/datatype/DataType.md` |
| JIT 装饰器 | `docs/api/config/pypto-frontend-jit.md` |
| 参考实现(示例) | `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` |
| 参考实现(生产) | `models/glm_v4_5/glm_attention_fusion.py` |
| 参考实现(生产) | `models/deepseek_v4/hc_pre_impl.py` |
| 动态 shape | `docs/api/pypto-DYNAMIC.md` |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**: Reduction 维度为 dim=1 而非 dim=-1，需验证 pypto.sum(dim=1) 的兼容性；若不兼容需转置后处理
- **推荐方案**: 基于手动分解实现 (mul → sum(dim=1) → div → add → sqrt → div)，参考 `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` 的 `rms_norm_core` 模式
- **首选参考**: `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` (最接近的完整 RMSNorm 示例)
