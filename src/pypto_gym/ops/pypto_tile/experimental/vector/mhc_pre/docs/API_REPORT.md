# mhc_pre 算子 API 映射报告

## 1. 概述

本报告记录 `mhc_pre` 算子在 PyPTO 框架中的 API 映射分析结果，包括 PyTorch API 到 PyPTO API 的映射关系、约束条件和可行性评估。

---

## 2. PyTorch 操作分解

### 2.1 核心操作序列（7 步）

| 序号 | PyTorch 操作 | 输入 Shape | 输出 Shape | 说明 |
|------|-------------|-----------|-----------|------|
| 1 | `reshape()` | `[B*S, N, D]` | `[B*S, N*D]` | 维度变换 |
| 2 | `float()` | `[B*S, N*D]` BF16 | `[B*S, N*D]` FP32 | 类型转换 |
| 3 | `square()` | `[B*S, N*D]` | `[B*S, N*D]` | 逐元素平方 |
| 4 | `mean(-1)` | `[B*S, N*D]` | `[B*S, 1]` | 沿尾轴均值 |
| 5 | `rsqrt()` | `[B*S, 1]` | `[B*S, 1]` | 平方根倒数 |
| 6 | `F.linear()` | `[B*S, N*D] × [N²+2N, N*D]` | `[B*S, N²+2N]` | MatMul |
| 7 | `mul()` | `[B*S, N²+2N] × [B*S, 1]` | `[B*S, N²+2N]` | 归一化 |
| 8 | `split()` | `[B*S, N²+2N]` | 3 × `[B*S, N/N/N²]` | 三分流 |
| 9 | `sigmoid()` | `[B*S, N]` | `[B*S, N]` | Branch Pre/Post |
| 10 | `mul()` | `[B*S, N] × scalar` | `[B*S, N]` | 缩放 |
| 11 | `add()` | `[B*S, N] + [N]` | `[B*S, N]` | 加偏置 |
| 12 | `unsqueeze(-1)` | `[B*S, N]` | `[B*S, N, 1]` | 扩展维度 |
| 13 | `unflatten()` | `[B*S, N*D]` | `[B*S, N, D]` | Branch Pre |
| 14 | `mul()` | `[B*S, N, 1] × [B*S, N, D]` | `[B*S, N, D]` | 加权 |
| 15 | `sum(1)` | `[B*S, N, D]` | `[B*S, D]` | Branch Pre 输出 |
| 16 | `view()` | `[B*S, N²]` | `[B*S, N, N]` | Branch Res reshape |

### 2.2 操作分类

| 类型 | 操作数量 | PyTorch API |
|------|---------|------------|
| 维度操作 | 4 | `reshape`, `split`, `unsqueeze`, `unflatten`, `view` |
| 类型转换 | 2 | `float()`, `to(bfloat16)` |
| MatMul | 1 | `F.linear()` |
| 逐元素运算 | 7 | `square`, `mul`, `add` |
| 归约运算 | 2 | `mean`, `sum` |
| 数学函数 | 2 | `rsqrt`, `sigmoid` |

---

## 3. PyPTO API 映射表

### 3.1 维度操作 API

| PyTorch API | PyPTO API | 支持状态 | 约束条件 |
|------------|-----------|---------|---------|
| `tensor.reshape(shape)` | `pypto.reshape(tensor, shape, inplace=True)` | ✅ 完全支持 | 需显式指定新 shape |
| `tensor.unsqueeze(dim)` | `pypto.reshape(tensor, new_shape, inplace=True)` | ✅ 完全支持 | 使用 reshape 实现 |
| `tensor.view(shape)` | `pypto.reshape(tensor, shape, inplace=False)` | ✅ 完全支持 | 非 inplace 避免 内存问题 |
| `tensor.split(sizes, dim)` | 切片语法 `tensor[:, start:end]` | ✅ 完全支持 | 使用切片替代 split |
| `tensor.unflatten(dim, sizes)` | `pypto.reshape(tensor, new_shape)` | ✅ 完全支持 | reshape 实现 |

**映射说明**：
- PyPTO 无独立的 `unsqueeze`、`view`、`unflatten` API，统一使用 `reshape` 实现
- `split` 使用切片语法替代，更高效
- 推荐使用 `inplace=True` 减少内存拷贝（view 操作除外）

**示例映射**：
```python
# PyTorch
x_flat = x.reshape([BS, N*D])
h_pre, h_post, h_comb = weight.split([N, N, N*N], dim=-1)
h_pre_expanded = h_pre.unsqueeze(-1)

# PyPTO
x_flat = pypto.reshape(x, [BS, N_D], inplace=True)
X_pre = X_hat_norm[:, 0:N]
X_post = X_hat_norm[:, N:2*N]
X_comb = X_hat_norm[:, 2*N:2*N+N*N]
h_pre_expanded = pypto.reshape(h_pre, [unroll_length, N, 1], inplace=True)
```

### 3.2 类型转换 API

| PyTorch API | PyPTO API | 支持状态 | 约束条件 |
|------------|-----------|---------|---------|
| `tensor.float()` | `pypto.cast(tensor, pypto.DT_FP32)` | ✅ 完全支持 | 无 |
| `tensor.to(torch.bfloat16)` | `pypto.cast(tensor, pypto.DT_BF16)` | ✅ 完全支持 | 无 |

**映射说明**：
- PyPTO 使用显式的 `pypto.cast()` 函数进行类型转换
- 支持所有常见数据类型：`DT_FP32`, `DT_FP16`, `DT_BF16`, `DT_INT32` 等

### 3.3 MatMul API

| PyTorch API | PyPTO API | 支持状态 | 约束条件 |
|------------|-----------|---------|---------|
| `F.linear(input, weight)` | `pypto.matmul(A, B, dtype)` | ✅ 完全支持 | 需设置 cube tile shapes |

**映射说明**：
- `F.linear(x, weight)` 等价于 `matmul(x, weight.T)`
- PyPTO 使用 `pypto.matmul` 显式进行矩阵乘法
- **重要**：需要设置 `set_cube_tile_shapes` 优化性能
- MatMul 两侧 dtype 必须一致（本算子使用 FP32）

**示例映射**：
```python
# PyTorch
h_mix = F.linear(x_flat, phi)  # [B*S, N*D] @ [N²+2N, N*D].T

# PyPTO（需转置 phi）
phi_T = phi.T.contiguous()  # [N*D, N²+2N]
pypto.set_cube_tile_shapes([16, 16], [512, 1024], [128, 128], enable_split_k=True)
h_mix = pypto.matmul(x_flat, phi_T, pypto.DT_FP32)  # [B*S, N*D] @ [N*D, N²+2N]
```

### 3.4 逐元素运算 API

| PyTorch API | PyPTO API | 支持状态 | 约束条件 |
|------------|-----------|---------|---------|
| `torch.square(tensor)` | `pypto.mul(tensor, tensor)` | ✅ 完全支持 | 使用 mul 实现平方 |
| `torch.mul(a, b)` | `pypto.mul(a, b)` | ✅ 完全支持 | 支持 Tensor × Tensor 和 Tensor × scalar |
| `torch.add(a, b)` | `pypto.add(a, b)` | ✅ 完全支持 | 支持广播 |

**映射说明**：
- PyPTO 无独立的 `square` API，使用 `mul(a, a)` 实现
- `mul` 支持广播机制
- `mul(a, scalar)` 需注意顺序：`pypto.mul(tensor, float)`，不支持 `pypto.mul(float, tensor)`

**示例映射**：
```python
# PyTorch
X_sq = torch.square(X_flat)
scaled_X_pre = alpha[0] * X_pre
X_pre_bias = X_pre + bias[:N]

# PyPTO
X_sq = pypto.mul(X_flat, X_flat)
scaled_X_pre = pypto.mul(X_pre, alpha_0)  # 注意顺序
X_pre_bias = pypto.add(scaled_X_pre, bias_pre_2d)  # bias 需 reshape 为 2D 广播
```

### 3.5 归约运算 API

| PyTorch API | PyPTO API | 支持状态 | 约束条件 |
|------------|-----------|---------|---------|
| `torch.mean(tensor, dim)` | `sum / scalar` 组合 | ✅ 完全支持 | 使用 sum + div 替代 mean |
| `torch.sum(tensor, dim)` | `pypto.sum(tensor, dim, keepdim=False)` | ✅ 完全支持 | 仅支持 FP32/FP16 |

**映射说明**：
- **重要约束**：`pypto.sum()` 仅支持 FP32 和 FP16，不支持 BF16
- PyPTO 无独立的 `mean` API，使用 `sum(tensor, dim) / size` 实现
- `keepdim` 参数默认为 `False`

**示例映射**：
```python
# PyTorch
mean_val = torch.mean(X_sq, -1, keepdim=True)

# PyPTO（使用 sum + div 替代）
mean_val = pypto.sum(X_sq, -1, keepdim=True)  # [unroll_length, 1]
mean_coeff = 1.0 / N_D  # Python float
variance = pypto.mul(mean_val, mean_coeff)  # 均值
```

### 3.6 数学函数 API

| PyTorch API | PyPTO API | 支持状态 | 约束条件 |
|------------|-----------|---------|---------|
| `torch.rsqrt(tensor)` | `pypto.rsqrt(tensor)` | ✅ 完全支持 | 仅支持 FP32/FP16 |
| `torch.sigmoid(tensor)` | `pypto.sigmoid(tensor)` | ✅ 完全支持 | 仅支持 FP32/FP16 |

**映射说明**：
- **重要约束**：`rsqrt` 和 `sigmoid` 仅支持 FP32 和 FP16，不支持 BF16
- 这是实现中所有计算需在 FP32 下进行的根本原因

---

## 4. 约束条件清单

### 4.1 数据类型约束

| 约束 ID | 约束描述 | 影响 | 解决方案 |
|---------|---------|------|---------|
| C-DTYPE-001 | `pypto.sum()` 不支持 BF16 | 归约操作前需类型转换 | 所有计算在 FP32 下进行 |
| C-DTYPE-002 | `pypto.rsqrt()` 不支持 BF16 | RMSNorm 需 FP32 | Step 1 转 FP32 |
| C-DTYPE-003 | `pypto.sigmoid()` 不支持 BF16 | Branch Pre/Post 需 FP32 | 所有 sigmoid 操作在 FP32 下 |
| C-DTYPE-004 | MatMul 两侧 dtype 必须一致 | X_flat 和 phi_T 需统一类型 | 两者都使用 FP32 |
| C-DTYPE-005 | BF16 精度较低 | 累加误差可能放大 | 中间计算使用 FP32 |

### 4.2 Shape 约束

| 约束 ID | 约束描述 | 影响 | 解决方案 |
|---------|---------|------|---------|
| C-SHAPE-001 | N 固定为 8 | 硬编码在 kernel 中 | 使用 Python 常量 `N = 8` |
| C-SHAPE-002 | D 使用 STATIC 标记 | D 变化触发重编译 | 使用 `pypto.STATIC` 标记 |
| C-SHAPE-003 | B*S 使用 DYNAMIC 标记 | 支持动态 shape | 使用 `pypto.DYNAMIC` 标记 |
| C-SHAPE-004 | Bias 需预切片 | kernel 内切片复杂 | 在 wrapper 中切片 bias |

### 4.3 内存约束

| 约束 ID | 约束描述 | 影响 | 解决方案 |
|---------|---------|------|---------|
| C-MEM-001 | 输入必须 contiguous | wrapper 需验证 | 使用 `tensor.is_contiguous()` 检查 |
| C-MEM-002 | Phi 需转置并 contiguous | MatMul 输入格式要求 | wrapper 中 `phi.T.contiguous()` |
| C-MEM-003 | 中间 FP32 tensor 占用内存 | BS×N×D×4 bytes | 使用 tile 分块控制内存峰值 |

### 4.4 MatMul 约束

| 约束 ID | 约束描述 | 影响 | 解决方案 |
|---------|---------|------|---------|
| C-MAT-001 | Phi 需转置 `[N*D, N²+2N]` | F.linear → matmul 映射 | wrapper 中转置 phi |
| C-MAT-002 | 需设置 cube tile shapes | 性能优化必需 | `set_cube_tile_shapes([16,16],[512,1024],[128,128])` |
| C-MAT-003 | MatMul dtype 必须一致 | 类型转换必需 | 两侧都使用 FP32 |

---

## 5. 可行性评估

### 5.1 API 完备性

| 评估项 | 结果 | 说明 |
|--------|------|------|
| 所需 API 是否存在 | ✅ 是 | 所有操作均有对应 PyPTO API |
| API 功能是否完整 | ✅ 是 | 语义与 PyTorch 一致 |
| 性能是否可接受 | ✅ 是 | MatMul + Vector 混合，性能可预测 |
| 约束是否可满足 | ✅ 是 | 约束条件明确且可实现 |

### 5.2 实现路径

| 步骤 | PyTorch 操作 | PyPTO 实现 | 可行性 |
|------|-------------|-----------|--------|
| 1 | Reshape BF16 | `pypto.reshape(x, [BS, N_D], inplace=True)` | ✅ |
| 2 | 类型转换 BF16→FP32 | `pypto.cast(x_flat, pypto.DT_FP32)` | ✅ |
| 3 | RMSNorm (square, sum, rsqrt) | `mul + sum + rsqrt` 组合 | ✅ |
| 4 | MatMul | `pypto.matmul(X_flat, phi_T, pypto.DT_FP32)` | ✅ |
| 5 | Split | 切片语法 `tensor[:, start:end]` | ✅ |
| 6 | sigmoid + add + mul | `sigmoid + add + mul` 组合 | ✅ |
| 7 | Weighted sum | `reshape + mul + sum` 组合 | ✅ |

### 5.3 最终判定

**结论**：✅ **API 映射完全可行**

**理由**：
1. 所有 PyTorch 操作均有对应 PyPTO API
2. 约束条件明确，可通过类型转换和 shape 标记满足
3. MatMul + Vector 混合操作，性能可预测
4. 无特殊算子需求，实现路径清晰

---

## 6. 性能优化建议

### 6.1 MatMul 优化

| 优化项 | API | 参数建议 | 说明 |
|--------|-----|---------|------|
| Cube tile shapes | `pypto.set_cube_tile_shapes()` | `[16, 16], [512, 1024], [128, 128]` | MatMul 性能优化 |
| Split K | `enable_split_k=True` | - | 大矩阵乘法优化 |

### 6.2 向量化优化

| 优化项 | API | 参数建议 | 说明 |
|--------|-----|---------|------|
| Vector tile shapes | `pypto.set_vec_tile_shapes()` | 自适应调整 | Vector 算子分块 |
| 自适应策略 | 根据 D 大小动态调整 | D < 2560: bs_tile=8, D_tile=128<br>否则: bs_tile=1, D_tile=2560 | 内存访问优化 |

### 6.3 循环展开

| 优化项 | API | 参数建议 | 说明 |
|--------|-----|---------|------|
| BS 轴展开 | `pypto.loop_unroll()` | `unroll_list=[128]` | 循环展开优化 |

### 6.4 内存优化

| 优化项 | 方法 | 说明 |
|--------|------|------|
| 原地操作 | `inplace=True` | 减少 reshape 内存拷贝 |
| Bias 预切片 | wrapper 中切片 | 减少 kernel 内操作 |
| Phi 预转置 | wrapper 中转置 | 避免 kernel 内转置开销 |

---

## 7. API 使用示例

### 7.1 RMSNorm 实现

```python
# Step 2: RMSNorm
X_sq = pypto.mul(X_flat, X_flat)  # [BS, N_D] FP32
mean_val = pypto.sum(X_sq, -1, keepdim=True)  # [BS, 1] FP32
mean_coeff = 1.0 / N_D  # Python float
variance = pypto.mul(mean_val, mean_coeff)  # [BS, 1] FP32
variance_eps = pypto.add(variance, norm_eps)  # [BS, 1] FP32
rsqrt_val = pypto.rsqrt(variance_eps)  # [BS, 1] FP32
```

### 7.2 MatMul 实现

```python
# Step 3: MatMul with Normalization
pypto.set_cube_tile_shapes([16, 16], [512, 1024], [128, 128], enable_split_k=True)
X_hat = pypto.matmul(X_flat, phi_T, pypto.DT_FP32)  # [BS, N_SQUARED_PLUS_2N] FP32
X_hat_norm = pypto.mul(X_hat, rsqrt_val)  # [BS, N_SQUARED_PLUS_2N] FP32
```

### 7.3 Split 实现

```python
# Step 4: Split（使用切片）
X_pre = X_hat_norm[:, 0:N]  # [BS, N] FP32
X_post = X_hat_norm[:, N:2*N]  # [BS, N] FP32
X_comb = X_hat_norm[:, 2*N:2*N+N*N]  # [BS, N²] FP32
```

### 7.4 Branch Pre 实现

```python
# Step 5: Branch Pre
scaled_X_pre = pypto.mul(X_pre, alpha_0)  # [BS, N] FP32
X_pre_bias = pypto.add(scaled_X_pre, bias_pre_2d)  # [BS, N] FP32
H_pre = pypto.sigmoid(X_pre_bias)  # [BS, N] FP32
H_pre_eps = pypto.add(H_pre, hc_eps)  # [BS, N] FP32
H_pre_expanded = pypto.reshape(H_pre_eps, [BS, N, 1], inplace=True)  # [BS, N, 1] FP32
weighted_X = pypto.mul(H_pre_expanded, x_fp32_3d)  # [BS, N, D] FP32
h_in_fp32 = pypto.sum(weighted_X, 1)  # [BS, D] FP32
h_in = pypto.cast(h_in_fp32, pypto.DT_BF16)  # [BS, D] BF16
```

### 7.5 Branch Post 实现

```python
# Step 6: Branch Post
scaled_X_post = pypto.mul(X_post, alpha_1)  # [BS, N] FP32
X_post_bias = pypto.add(scaled_X_post, bias_post_2d)  # [BS, N] FP32
H_post = pypto.sigmoid(X_post_bias)  # [BS, N] FP32
h_post = pypto.mul(H_post, 2.0)  # [BS, N] FP32
```

### 7.6 Branch Res 实现

```python
# Step 7: Branch Res
scaled_X_comb = pypto.mul(X_comb, alpha_2)  # [BS, N²] FP32
h_res_2d = pypto.add(scaled_X_comb, bias_comb_2d)  # [BS, N²] FP32
h_res = pypto.reshape(h_res_2d, [BS, N, N])  # [BS, N, N] FP32
```

### 7.7 完整 Kernel 签名

```python
@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128},
    pass_options={"vec_nbuffer_setting": {-2: 1, -1: 4}},
    debug_options={"runtime_debug_mode": 1}
)
def mhc_pre_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    phi_T: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    bias_pre: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    bias_post: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    bias_comb: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    h_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    h_post: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    h_res: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    alpha_0: float = 1.0,
    alpha_1: float = 1.0,
    alpha_2: float = 1.0,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
):
    # 实现见 mhc_pre_impl.py
```

---

## 8. 风险与限制

### 8.1 已知限制

| 限制 ID | 描述 | 影响等级 | 缓解措施 |
|---------|------|---------|---------|
| L-001 | N 固定为 8 | 低 | 硬编码在 kernel 中，无泛化需求 |
| L-002 | D 变化触发重编译 | 中 | 文档说明，用户知晓 |
| L-003 | sum/rsqrt/sigmoid 不支持 BF16 | 中 | 类型转换，性能影响可控 |
| L-004 | Phi 需预转置 | 低 | wrapper 中自动处理 |

### 8.2 潜在风险

| 风险 ID | 描述 | 概率 | 应对方案 |
|---------|------|------|---------|
| R-001 | MatMul 内存峰值 | 中 | 使用 tile 分块 + loop_unroll 控制 |
| R-002 | 类型转换开销 | 低 | 必要开销，无可优化空间 |
| R-003 | Bias 切片索引错误 | 低 | wrapper 验证切片范围 |

---

## 9. 参考文档

### 9.1 PyPTO API 文档

- `pypto.cast()`: 类型转换 API
- `pypto.reshape()`: 维度变换 API
- `pypto.matmul()`: 矩阵乘法 API
- `pypto.mul()`: 逐元素乘法 API
- `pypto.add()`: 逐元素加法 API
- `pypto.sum()`: 归约求和 API
- `pypto.rsqrt()`: 平方根倒数 API
- `pypto.sigmoid()`: sigmoid 函数 API
- `pypto.set_vec_tile_shapes()`: 向量化分块 API
- `pypto.set_cube_tile_shapes()`: MatMul 分块 API
- `pypto.loop_unroll()`: 循环展开 API

### 9.2 相关资源

- [PyPTO 编程指南](../../docs/pypto_programming_guide.md)
- [PyPTO API 参考](../../docs/api_reference.md)
- [MHC 论文](https://arxiv.org/abs/2406.07828)