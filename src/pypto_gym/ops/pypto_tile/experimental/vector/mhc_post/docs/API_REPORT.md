# mhc_post 算子 API 映射报告

## 1. 概述

本报告记录 `mhc_post` 算子在 PyPTO 框架中的 API 映射分析结果，包括 PyTorch API 到 PyPTO API 的映射关系、约束条件和可行性评估。

---

## 2. PyTorch 操作分解

### 2.1 核心操作序列

根据 golden 实现，`mhc_post` 算子包含以下核心操作：

| 序号 | PyTorch 操作 | 输入 Shape | 输出 Shape | 说明 |
|------|-------------|-----------|-----------|------|
| 1 | `float()` | `[B*S, D]` BF16 | `[B*S, D]` FP32 | 类型转换 BF16 → FP32 |
| 2 | `float()` | `[B*S, N, D]` BF16 | `[B*S, N, D]` FP32 | 类型转换 BF16 → FP32 |
| 3 | `unsqueeze(-1)` | `[B*S, N]` | `[B*S, N, 1]` | 维度扩展 |
| 4 | `unsqueeze(-2)` | `[B*S, D]` | `[B*S, 1, D]` | 维度扩展 |
| 5 | `mul()` | `[B*S, N, 1]` × `[B*S, 1, D]` | `[B*S, N, D]` | 广播乘法 |
| 6 | `unsqueeze(-1)` | `[B*S, N, N]` | `[B*S, N, N, 1]` | 维度扩展 |
| 7 | `unsqueeze(-2)` | `[B*S, N, D]` | `[B*S, N, 1, D]` | 维度扩展 |
| 8 | `mul()` | `[B*S, N, N, 1]` × `[B*S, N, 1, D]` | `[B*S, N, N, D]` | 广播乘法 |
| 9 | `sum(dim=-3)` | `[B*S, N, N, D]` | `[B*S, N, D]` | 沿轴求和 |
| 10 | `add()` | `[B*S, N, D]` + `[B*S, N, D]` | `[B*S, N, D]` | 逐元素加法 |
| 11 | `to(bfloat16)` | `[B*S, N, D]` FP32 | `[B*S, N, D]` BF16 | 类型转换 FP32 → BF16 |

### 2.2 操作分类

| 类型 | 操作数量 | PyTorch API |
|------|---------|------------|
| 类型转换 | 3 | `float()`, `to(bfloat16)` |
| 维度操作 | 4 | `unsqueeze()` |
| 逐元素运算 | 3 | `mul()`, `add()` |
| 归约运算 | 1 | `sum()` |

---

## 3. PyPTO API 映射表

### 3.1 类型转换 API

| PyTorch API | PyPTO API | 支持状态 | 约束条件 |
|------------|-----------|---------|---------|
| `tensor.float()` | `pypto.cast(tensor, pypto.DT_FP32)` | ✅ 完全支持 | 无 |
| `tensor.to(torch.bfloat16)` | `pypto.cast(tensor, pypto.DT_BF16)` | ✅ 完全支持 | 无 |

**映射说明**：
- PyPTO 使用显式的 `pypto.cast()` 函数进行类型转换
- 支持所有常见数据类型：`DT_FP32`, `DT_FP16`, `DT_BF16`, `DT_INT32` 等
- 类型转换是向量操作，性能可预测

### 3.2 维度操作 API

| PyTorch API | PyPTO API | 支持状态 | 约束条件 |
|------------|-----------|---------|---------|
| `tensor.unsqueeze(dim)` | `pypto.reshape(tensor, new_shape, inplace=True)` | ✅ 完全支持 | 需显式指定新 shape |

**映射说明**：
- PyPTO 无独立的 `unsqueeze` API，使用 `reshape` 实现
- 推荐使用 `inplace=True` 减少内存拷贝
- shape 推导：将目标维度设为 1

**示例映射**：
```python
# PyTorch
h_post.unsqueeze(-1)  # [BS, N] -> [BS, N, 1]

# PyPTO
h_post_reshaped = pypto.reshape(h_post, [BS, N, 1], inplace=True)
```

### 3.3 逐元素运算 API

| PyTorch API | PyPTO API | 支持状态 | 约束条件 |
|------------|-----------|---------|---------|
| `torch.mul(a, b)` | `pypto.mul(a, b)` | ✅ 完全支持 | 支持广播 |
| `torch.add(a, b)` | `pypto.add(a, b)` | ✅ 完全支持 | 支持广播 |

**映射说明**：
- PyPTO 的逐元素运算与 PyTorch 语义一致
- 自动支持广播机制
- 性能优化通过 `set_vec_tile_shapes` 控制

### 3.4 归约运算 API

| PyTorch API | PyPTO API | 支持状态 | 约束条件 |
|------------|-----------|---------|---------|
| `torch.sum(tensor, dim)` | `pypto.sum(tensor, dim, keepdim=False)` | ✅ 完全支持 | 仅支持 FP32/FP16 |

**映射说明**：
- **重要约束**：`pypto.sum()` 仅支持 FP32 和 FP16，不支持 BF16
- 这是实现中所有计算需在 FP32 下进行的根本原因
- `keepdim` 参数默认为 `False`，与 PyTorch 行为一致

---

## 4. 约束条件清单

### 4.1 数据类型约束

| 约束 ID | 约束描述 | 影响 | 解决方案 |
|---------|---------|------|---------|
| C-DTYPE-001 | `pypto.sum()` 不支持 BF16 | 归约操作前需类型转换 | 所有计算在 FP32 下进行 |
| C-DTYPE-002 | BF16 精度较低 | 累加误差可能放大 | 中间计算使用 FP32 |
| C-DTYPE-003 | 输入包含 BF16 和 FP32 | 需统一计算类型 | 转换为 FP32 计算，最后转回 BF16 |

### 4.2 Shape 约束

| 约束 ID | 约束描述 | 影响 | 解决方案 |
|---------|---------|------|---------|
| C-SHAPE-001 | N 固定为 4 | 硬编码在 kernel 中 | 使用 Python 常量 `N = 4` |
| C-SHAPE-002 | D 使用 STATIC 标记 | D 变化触发重编译 | 使用 `pypto.STATIC` 标记 |
| C-SHAPE-003 | B*S 使用 DYNAMIC 标记 | 支持动态 shape | 使用 `pypto.DYNAMIC` 标记 |

### 4.3 内存约束

| 约束 ID | 约束描述 | 影响 | 解决方案 |
|---------|---------|------|---------|
| C-MEM-001 | 输入必须 contiguous | wrapper 需验证 | 使用 `tensor.is_contiguous()` 检查 |
| C-MEM-002 | 中间 FP32 tensor 占用内存 | BS×N×D×4 bytes | 使用 tile 分块控制内存峰值 |

---

## 5. 可行性评估

### 5.1 API 完备性

| 评估项 | 结果 | 说明 |
|--------|------|------|
| 所需 API 是否存在 | ✅ 是 | 所有操作均有对应 PyPTO API |
| API 功能是否完整 | ✅ 是 | 语义与 PyTorch 一致 |
| 性能是否可接受 | ✅ 是 | 纯向量操作，性能可预测 |
| 约束是否可满足 | ✅ 是 | 约束条件明确且可实现 |

### 5.2 实现路径

| 步骤 | PyTorch 操作 | PyPTO 实现 | 可行性 |
|------|-------------|-----------|--------|
| 1 | 类型转换 BF16→FP32 | `pypto.cast(x, pypto.DT_FP32)` | ✅ |
| 2 | 维度扩展 unsqueeze | `pypto.reshape(..., inplace=True)` | ✅ |
| 3 | 广播乘法 mul | `pypto.mul(a, b)` | ✅ |
| 4 | 归约求和 sum | `pypto.sum(..., dim=..., keepdim=False)` | ✅ |
| 5 | 逐元素加法 add | `pypto.add(a, b)` | ✅ |
| 6 | 类型转换 FP32→BF16 | `pypto.cast(..., pypto.DT_BF16)` | ✅ |

### 5.3 最终判定

**结论**：✅ **API 映射完全可行**

**理由**：
1. 所有 PyTorch 操作均有对应 PyPTO API
2. 约束条件明确，可通过类型转换和 shape 标记满足
3. 无矩阵乘法操作，纯向量计算，性能可预测
4. 无特殊算子需求，实现路径清晰

---

## 6. 性能优化建议

### 6.1 向量化优化

| 优化项 | API | 参数建议 | 说明 |
|--------|-----|---------|------|
| 向量分块 | `pypto.set_vec_tile_shapes()` | `(1, N, 1280)` | 控制向量化粒度 |
| 双缓冲 | `vec_nbuffer_setting` | `{-2: 1, -1: 8}` | 控制内存缓冲策略 |

### 6.2 循环展开

| 优化项 | API | 参数建议 | 说明 |
|--------|-----|---------|------|
| BS 轴展开 | `pypto.loop_unroll()` | `unroll_list=[128]` | 循环展开优化 |

### 6.3 内存优化

| 优化项 | 方法 | 说明 |
|--------|------|------|
| 原地操作 | `inplace=True` | 减少 reshape 内存拷贝 |
| 分块计算 | tile shapes | 控制中间 tensor 内存峰值 |

---

## 7. API 使用示例

### 7.1 核心计算流程

```python
import pypto

# 类型转换 BF16 → FP32
x_fp32 = pypto.cast(x, pypto.DT_FP32)
h_out_fp32 = pypto.cast(h_out, pypto.DT_FP32)

# 维度扩展（使用 reshape 实现 unsqueeze）
h_post_1 = pypto.reshape(h_post, [BS, N, 1], inplace=True)
h_out_1 = pypto.reshape(h_out_fp32, [BS, 1, D], inplace=True)

# 广播乘法
h_post_term = pypto.mul(h_post_1, h_out_1)

# 维度扩展
h_res_1 = pypto.reshape(h_res, [BS, N, N, 1], inplace=True)
x_1 = pypto.reshape(x_fp32, [BS, N, 1, D], inplace=True)

# 广播乘法
weighted = pypto.mul(h_res_1, x_1)

# 沿轴求和（必须在 FP32 下）
h_comb_term = pypto.sum(weighted, dim=1, keepdim=False)

# 逐元素加法
result_fp32 = pypto.add(h_post_term, h_comb_term)

# 类型转换 FP32 → BF16
result_bf16 = pypto.cast(result_fp32, pypto.DT_BF16)
```

### 7.2 完整 Kernel 签名

```python
@pypto.frontend.jit(
    pass_options={"vec_nbuffer_setting": {-2: 1, -1: 8}},
    debug_options={"runtime_debug_mode": 1}
)
def mhc_post_kernel_bf16(
    x: pypto.Tensor([pypto.DYNAMIC, 4, pypto.STATIC], pypto.DT_BF16),
    h_res: pypto.Tensor([pypto.DYNAMIC, 4, 4], pypto.DT_FP32),
    h_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    h_post: pypto.Tensor([pypto.DYNAMIC, 4], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 4, pypto.STATIC], pypto.DT_BF16),
):
    # 实现见 mhc_post_impl.py
    pass
```

---

## 8. 风险与限制

### 8.1 已知限制

| 限制 ID | 描述 | 影响等级 | 缓解措施 |
|---------|------|---------|---------|
| L-001 | N 固定为 4 | 低 | 硬编码在 kernel 中，无泛化需求 |
| L-002 | D 变化触发重编译 | 中 | 文档说明，用户知晓 |
| L-003 | sum 不支持 BF16 | 中 | 类型转换，性能影响可控 |

### 8.2 潜在风险

| 风险 ID | 描述 | 概率 | 应对方案 |
|---------|------|------|---------|
| R-001 | 大 BS 值内存峰值 | 中 | 使用 tile 分块控制 |
| R-002 | 类型转换开销 | 低 | 必要开销，无可优化空间 |
| R-003 | 广播性能 | 低 | PyPTO 自动优化 |

---

## 9. 参考文档

### 9.1 PyPTO API 文档

- `pypto.cast()`: 类型转换 API
- `pypto.reshape()`: 维度变换 API
- `pypto.mul()`: 逐元素乘法 API
- `pypto.add()`: 逐元素加法 API
- `pypto.sum()`: 归约求和 API
- `pypto.set_vec_tile_shapes()`: 向量化分块 API
- `pypto.loop_unroll()`: 循环展开 API

### 9.2 相关资源

- [PyPTO 编程指南](../../docs/pypto_programming_guide.md)
- [PyPTO API 参考](../../docs/api_reference.md)
- [MHC 论文](https://arxiv.org/abs/2406.07828)