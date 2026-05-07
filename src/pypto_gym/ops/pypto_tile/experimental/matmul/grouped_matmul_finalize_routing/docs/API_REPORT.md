# grouped_matmul_finalize_routing 算子 API 映射报告

## 1. 概述

本报告记录 `grouped_matmul_finalize_routing` 算子在 PyPTO 框架中的 API 映射分析结果，包括 PyTorch golden 到 PyPTO API 的映射关系、约束条件和可行性评估。

---

## 2. PyTorch 操作分解

### 2.1 核心操作序列

根据 golden 实现，算子包含以下核心操作：

| 序号 | PyTorch 操作 | 输入 Shape | 输出 Shape | 说明 |
|------|-------------|-----------|-----------|------|
| 1 | expert 范围解析 | `[E]` | 标量 start/end | 根据 `group_list` 计算 expert token 范围 |
| 2 | slice | `[M, K]` | `[M_i, K]` | 获取当前 expert 的 token 输入 |
| 3 | slice | `[E, *, *]` | `[K, N]` 或 `[N, K]` | 获取当前 expert 权重 |
| 4 | scale 展开 | scale tensor | FP32 scale | MXFP8 scale 展开 |
| 5 | `torch.matmul()` | `[M_i, K] × [K, N]` | `[M_i, N]` | expert matmul |
| 6 | `unsqueeze(-1)` | `[M_i]` | `[M_i, 1]` | 扩展 logit 维度 |
| 7 | `mul()` | `[M_i, N] × [M_i, 1]` | `[M_i, N]` | logit 加权 |
| 8 | `index_add_()` | `[batch, N]` + `[M_i, N]` | `[batch, N]` | 按 `row_index` 累加输出 |
| 9 | `add()` | `[batch, N]` | `[batch, N]` | shared input 叠加 |

### 2.2 操作分类

| 类型 | 操作数量 | PyTorch API |
|------|---------|-------------|
| 分组/切片 | 3 | tensor slice、Python range |
| 矩阵乘 | 1 | `torch.matmul()` |
| 维度操作 | 1 | `unsqueeze()` |
| 逐元素运算 | 2 | `mul()`、`add()` |
| 索引累加 | 1 | `index_add_()` |
| 类型转换 | 多处 | `to(torch.float32)` |

---

## 3. PyPTO API 映射表

### 3.1 MXFP8 scaled matmul API

| PyTorch/Golden 操作 | PyPTO API | 支持状态 | 约束条件 |
|--------------------|-----------|---------|---------|
| `torch.matmul(x * x_scale, w * w_scale)` | `pypto.scaled_mm(...)` | 支持 | 输入/scale 布局需匹配 MXFP8 规则 |

**映射说明**：

- Golden 中显式展开 scale 并用 FP32 matmul 模拟。
- PyPTO 中直接使用 `pypto.scaled_mm` 表达 MXFP8 scaled matmul。
- 输出类型指定为 `pypto.DT_FP32`。

### 3.2 维度操作 API

| PyTorch API | PyPTO API | 支持状态 | 约束条件 |
|------------|-----------|---------|---------|
| `tensor.unsqueeze(-1)` | `pypto.unsqueeze(tensor, -1)` | 支持 | 目标维度需用于广播乘法 |

**映射说明**：

`logit` 从 `[M_i]` 扩展为 `[M_i, 1]`，用于和 `[M_i, N]` 的 matmul 结果广播相乘。

### 3.3 逐元素运算 API

| PyTorch API | PyPTO API | 支持状态 | 约束条件 |
|------------|-----------|---------|---------|
| `torch.mul(a, b)` | `pypto.mul(a, b)` | 支持 | 支持广播 |
| `tensor + shared_input` | host 侧 torch add | 支持 | 当前未融合进 kernel |

**映射说明**：

- logit 加权在 PyPTO kernel 中完成。
- shared input 叠加当前在 `gen_pypto` 的 host 侧提前写入 `out_host`。

### 3.4 索引累加 API

| PyTorch API | PyPTO API | 支持状态 | 约束条件 |
|------------|-----------|---------|---------|
| `index_add_(0, row_index, value)` | `pypto.index_put_(out, (index,), value, accumulate=True)` | 支持 | `row_index` 必须在输出行范围内 |

**映射说明**：

`pypto.index_put_` 使用 `accumulate=True` 对齐 scatter-add 语义：

```python
pypto.index_put_(out, (index_tile,), mm_result, accumulate=True)
```

---

## 4. 约束条件清单

### 4.1 数据类型约束

| 约束 ID | 约束描述 | 影响 | 解决方案 |
|---------|---------|------|---------|
| C-DTYPE-001 | `x1`/`x2` 为 FP8 E4M3/E5M2 | 需使用 MXFP8 matmul | 使用 `pypto.scaled_mm` |
| C-DTYPE-002 | scale 为 E8M0FNU | scale 布局需匹配 | 按 `[ceil(K/64), N, 2]` 或 `[N, ceil(K/64), 2]` 构造 |
| C-DTYPE-003 | 输出需要 FP32 | 避免累加精度损失 | `scaled_mm` 输出指定 FP32 |
| C-DTYPE-004 | shared input 为 BF16 | 叠加前需转 FP32 | host 侧 `to(torch.float32)` |

### 4.2 Shape 约束

| 约束 ID | 约束描述 | 影响 | 解决方案 |
|---------|---------|------|---------|
| C-SHAPE-001 | `transpose_x1=True` 未支持 | 不支持该路径 | 显式限制为 `False` |
| C-SHAPE-002 | kernel 当前按均匀 expert 分组 | 非均匀 group_list 未覆盖 | 后续扩展 kernel 内 offset 解析 |
| C-SHAPE-003 | `row_index` 必须合法 | 越界会导致写回错误 | 测试数据中使用 `arange(M) % batch` |
| C-SHAPE-004 | `shared_input_offset` 不能越界 | shared 叠加越界风险 | host 侧配置保证边界 |

### 4.3 内存约束

| 约束 ID | 约束描述 | 影响 | 解决方案 |
|---------|---------|------|---------|
| C-MEM-001 | row_index scatter-add 非连续写 | 可能存在写冲突 | 使用 accumulate 语义 |
| C-MEM-002 | 大 K/N matmul 中间结果大 | 占用 cube/vector 资源 | 使用 tile shape 控制 |
| C-MEM-003 | shared input 叠加增加 kernel 分支 | kernel 复杂度上升 | 当前放在 host 侧处理 |

---

## 5. 可行性评估

### 5.1 API 完备性

| 评估项 | 结果 | 说明 |
|--------|------|------|
| 所需 API 是否存在 | 是 | `scaled_mm`、`mul`、`unsqueeze`、`index_put_` 均可用 |
| API 功能是否完整 | 是 | 可覆盖 matmul、logit、scatter-add 语义 |
| 性能是否可接受 | 是 | 主计算使用 cube，后处理使用 vector |
| 约束是否可满足 | 是 | 当前测试规格均满足实现约束 |

### 5.2 实现路径

| 步骤 | PyTorch/Golden 操作 | PyPTO 实现 | 可行性 |
|------|---------------------|-----------|--------|
| 1 | expert 切片 | Python slice + `pypto.loop` | 可行 |
| 2 | MXFP8 matmul | `pypto.scaled_mm` | 可行 |
| 3 | logit unsqueeze | `pypto.unsqueeze` | 可行 |
| 4 | logit 乘法 | `pypto.mul` | 可行 |
| 5 | row_index 累加 | `pypto.index_put_(accumulate=True)` | 可行 |
| 6 | shared input 叠加 | host 侧 torch add | 可行 |

### 5.3 最终判定

**结论**：API 映射可行。

**理由**：

1. MXFP8 matmul 可由 `pypto.scaled_mm` 表达。
2. finalize routing 的 logit 加权可由 vector API 表达。
3. `row_index` scatter-add 可由 `pypto.index_put_` 的 accumulate 模式表达。
4. shared input 可在 host 侧预处理，不影响 kernel 主路径。

---

## 6. 性能优化建议

### 6.1 Cube 优化

| 优化项 | API | 参数建议 | 说明 |
|--------|-----|---------|------|
| cube 分块 | `pypto.set_cube_tile_shapes()` | 由 config 提供 | 控制 M/K/N 分块 |
| cube buffer | `cube_nbuffer_setting` | `{-1: 4}` | 控制 cube 缓冲 |

### 6.2 Vector 优化

| 优化项 | API | 参数建议 | 说明 |
|--------|-----|---------|------|
| vector 分块 | `pypto.set_vec_tile_shapes()` | `[1, 32, 512, 2]` 或 `[M_tile, N_tile]` | 控制 scale/logit/scatter 后处理 |
| vector buffer | `vec_nbuffer_setting` | `{-2: 1, -1: 4}` | 控制 vector 缓冲 |

### 6.3 内存优化

| 优化项 | 方法 | 说明 |
|--------|------|------|
| shared input 预处理 | host 侧加到 `out_host` | 降低 kernel 分支复杂度 |
| expert 并行 | `pypto.loop(..., parallel=True)` | 提高 expert 维并行度 |
| row_index 累加 | `accumulate=True` | 保持 scatter-add 语义 |

---

## 7. API 使用示例

### 7.1 核心计算流程

```python
pypto.set_cube_tile_shapes(config.m_tile_shape, config.k_tile_shape, config.n_tile_shape)
pypto.set_vec_tile_shapes(*config.vector_tile_shape)

for expert_idx in pypto.loop(config.num_experts, parallel=True):
    x_tile = x1[start:end, :]
    weight_tile = x2[expert_idx, :, :]
    pertoken_scale_tile = pertoken_scale[start:end, :, :]

    mm_result = pypto.scaled_mm(
        x_tile,
        weight_tile,
        pypto.DT_FP32,
        pertoken_scale_tile,
        scale[:, :, :],
        a_trans=False,
        scale_a_trans=False,
        b_trans=config.transpose_x2,
        scale_b_trans=config.transpose_x2,
    )

    logit_2d = pypto.unsqueeze(logit[start:end], -1)
    mm_result = pypto.mul(mm_result, logit_2d)
    pypto.index_put_(out, (row_index[start:end],), mm_result, accumulate=True)
```

### 7.2 完整 Kernel 签名

```python
@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1},
    pass_options={
        "cube_nbuffer_setting": {-1: 4},
        "vec_nbuffer_setting": {-2: 1, -1: 4},
    },
    runtime_options={"stitch_function_max_num": 8},
)
def gmm_finalize_routing_kernel(
    x1: pypto.Tensor(),
    x2: pypto.Tensor(),
    scale: pypto.Tensor(),
    pertoken_scale: pypto.Tensor(),
    logit: pypto.Tensor(),
    row_index: pypto.Tensor(),
    out: pypto.Tensor(),
    group_list,
    config: FinalizeRoutingConfig,
):
    pass
```

---

## 8. 风险与限制

### 8.1 已知限制

| 限制 ID | 描述 | 影响等级 | 缓解措施 |
|---------|------|---------|---------|
| L-001 | `transpose_x1=True` 未支持 | 中 | 文档显式说明 |
| L-002 | kernel 当前按均匀分组 | 中 | 后续支持动态 offset |
| L-003 | shared input 未融合进 kernel | 低 | host 侧预处理 |
| L-004 | 当前内置测试主要覆盖 `transpose_x2=True` | 低 | 后续补充 False 用例 |

### 8.2 潜在风险

| 风险 ID | 描述 | 概率 | 应对方案 |
|---------|------|------|---------|
| R-001 | row_index 写回冲突影响性能 | 中 | 评估实际路由分布 |
| R-002 | 大 N/K 场景资源占用高 | 中 | 调整 cube/vector tile |
| R-003 | 非均匀 group_list 与 kernel 切分不一致 | 中 | 扩展 kernel offset 解析 |

---

## 9. 参考文档

### 9.1 PyPTO API

- `pypto.scaled_mm()`: MXFP8 scaled matmul API
- `pypto.unsqueeze()`: 维度扩展 API
- `pypto.mul()`: 逐元素乘法 API
- `pypto.index_put_()`: 索引写回/累加 API
- `pypto.set_cube_tile_shapes()`: cube 分块 API
- `pypto.set_vec_tile_shapes()`: vector 分块 API

### 9.2 相关资源

- `gmm_finalize_routing_impl.py`: PyPTO kernel 与 host 封装
- `tests/.../gmm_finalize_routing_golden.py`: Golden 参考实现
- `aclnnGroupedMatmulFinalizeRoutingV3`: 对齐的 CANN 算子语义
