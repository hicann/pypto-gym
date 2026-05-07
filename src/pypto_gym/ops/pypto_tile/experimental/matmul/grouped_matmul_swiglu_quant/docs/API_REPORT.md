# grouped_matmul_swiglu_quant 算子 API 映射报告

## 1. 概述

本报告记录 `grouped_matmul_swiglu_quant` 算子在 PyPTO 框架中的 API 映射分析结果，包括 PyTorch golden 到 PyPTO 新前端 API 的映射关系、约束条件和可行性评估。

---

## 2. PyTorch 操作分解

### 2.1 核心操作序列

| 序号 | PyTorch 操作 | 输入 Shape | 输出 Shape | 说明 |
|------|-------------|-----------|-----------|------|
| 1 | slice | `[M, K]` | `[M_i, K]` | 按 expert token 范围切分 `a` |
| 2 | slice | `[E, *, *]` | `[K, N]` 或 `[N, K]` | 读取当前 expert 权重 |
| 3 | scale 展开 | scale tensor | FP32 scale | MXFP8 scale 展开 |
| 4 | `torch.matmul()` | `[M_i, K] × [K, N]` | `[M_i, N]` | grouped matmul |
| 5 | `chunk(2, dim=-1)` | `[M_i, N]` | `[M_i, N/2]` × 2 | SwiGLU value/gate 切分 |
| 6 | `sigmoid()` | `[M_i, N/2]` | `[M_i, N/2]` | SiLU 计算 |
| 7 | `mul()` | `[M_i, N/2]` | `[M_i, N/2]` | SwiGLU 乘法 |
| 8 | `to(bfloat16)` / `to(float32)` | `[M_i, N/2]` | `[M_i, N/2]` | 精度对齐 |
| 9 | `abs()` | `[M_i, N/2]` | `[M_i, N/2]` | 量化前取绝对值 |
| 10 | `amax(dim=-1)` | `[M_i, N/2]` | `[M_i, 1]` | per-token 最大值 |
| 11 | `div()` | `[M_i, N/2]` | `[M_i, N/2]` | scale 归一化 |
| 12 | `round()` / `clamp()` / `to(int8)` | `[M_i, N/2]` | `[M_i, N/2]` | 生成 INT8 输出 |
| 13 | concat | 分 expert 输出 | `[M, N/2]` | 拼接 golden 输出 |

### 2.2 操作分类

| 类型 | 操作数量 | PyTorch API |
|------|---------|-------------|
| 分组/切片 | 多处 | tensor slice |
| 矩阵乘 | 1 | `torch.matmul()` |
| 维度切分 | 1 | `chunk()` |
| 激活函数 | 1 | `sigmoid()` |
| 逐元素运算 | 多处 | `mul()`、`div()`、`abs()` |
| 归约运算 | 1 | `amax()` |
| 类型转换 | 多处 | `to()` |
| 组装输出 | 1 | `cat()` |

---

## 3. PyPTO API 映射表

### 3.1 MXFP8 scaled matmul API

| PyTorch/Golden 操作 | PyPTO API | 支持状态 | 约束条件 |
|--------------------|-----------|---------|---------|
| `torch.matmul(x * x_scale, w * w_scale)` | `pypto.scaled_mm(...)` | 支持 | 输入和 scale 布局需匹配 MXFP8 规则 |

### 3.2 SwiGLU API

| PyTorch API | PyPTO API | 支持状态 | 约束条件 |
|------------|-----------|---------|---------|
| `tensor[:, :N/2]` | tensor slice | 支持 | N 必须为偶数 |
| `tensor[:, N/2:]` | tensor slice | 支持 | N 必须为偶数 |
| `torch.sigmoid(value)` | `pypto.sigmoid(value)` | 支持 | 输入为 FP32 |
| `value * sigmoid(value)` | `value * pypto.sigmoid(value)` | 支持 | 逐元素乘法 |
| `mul(silu_value, gate)` | `pypto.mul(silu_value, gate)` | 支持 | shape 一致 |

### 3.3 量化 API

| PyTorch API | PyPTO API | 支持状态 | 约束条件 |
|------------|-----------|---------|---------|
| `to(torch.bfloat16)` | `pypto.cast(..., pypto.DT_BF16)` | 支持 | 指定 cast mode |
| `to(torch.float32)` | `pypto.cast(..., pypto.DT_FP32)` | 支持 | 无 |
| `torch.abs()` | `pypto.abs()` | 支持 | 无 |
| `torch.amax(dim=-1, keepdim=True)` | `pypto.amax(..., -1, True)` | 支持 | keepdim=True |
| `torch.div()` | `pypto.div()` | 支持 | 注意除零风险 |
| `torch.round()` | `pypto.round()` | 支持 | 无 |
| `to(torch.int8)` | `pypto.cast(..., pypto.DT_INT8)` | 支持 | 需先完成 rounding |

### 3.4 输出组装 API

| PyTorch API | PyPTO API | 支持状态 | 约束条件 |
|------------|-----------|---------|---------|
| `torch.cat()` | `pypto.assemble()` | 支持 | 需提供全局输出起始 offset |

---

## 4. 约束条件清单

### 4.1 数据类型约束

| 约束 ID | 约束描述 | 影响 | 解决方案 |
|---------|---------|------|---------|
| C-DTYPE-001 | `a`/`b` 为 FP8 E4M3 | 需使用 MXFP8 matmul | 使用 `pypto.scaled_mm` |
| C-DTYPE-002 | scale 为 E8M0FNU | scale 布局需匹配 | 按 K/64 分块构造 |
| C-DTYPE-003 | SwiGLU 后需 BF16 往返 | 影响数值对齐 | kernel 中显式 cast |
| C-DTYPE-004 | 输出为 INT8 | 需 round 后 cast | 使用 `pypto.round` + `pypto.cast` |

### 4.2 Shape 约束

| 约束 ID | 约束描述 | 影响 | 解决方案 |
|---------|---------|------|---------|
| C-SHAPE-001 | N 必须为偶数 | SwiGLU chunk 依赖 | 测试配置保证 |
| C-SHAPE-002 | K 需能被 64 整除 | scale shape 使用 K/64 | 测试配置保证 |
| C-SHAPE-003 | `sum(group_list) == M` | 输出组装依赖 | 构造参数时保证 |
| C-SHAPE-004 | `b_trans=True` 需匹配 scale 布局 | scaled_mm 布局敏感 | 后续补充测试 |

### 4.3 内存约束

| 约束 ID | 约束描述 | 影响 | 解决方案 |
|---------|---------|------|---------|
| C-MEM-001 | matmul 输出 `[M_i, N]` 较大 | 占用中间内存 | tile 分块 |
| C-MEM-002 | quant 输出为 INT8 | 输出写回类型转换 | 预分配 int8 `out` |
| C-MEM-003 | scale 输出 `[M,1]` | 返回前需 squeeze | host 侧 squeeze |

---

## 5. 可行性评估

### 5.1 API 完备性

| 评估项 | 结果 | 说明 |
|--------|------|------|
| 所需 API 是否存在 | 是 | `scaled_mm`、`sigmoid`、`amax`、`round`、`assemble` 均可用 |
| API 功能是否完整 | 是 | 可覆盖 GMM、SwiGLU 和 quant 语义 |
| 新前端是否支持 | 是 | kernel 已迁移为 `pypto.frontend.jit` |
| 约束是否可满足 | 是 | 当前 testcase6 满足约束 |

### 5.2 实现路径

| 步骤 | PyTorch/Golden 操作 | PyPTO 实现 | 可行性 |
|------|---------------------|-----------|--------|
| 1 | expert 切片 | Python slice | 可行 |
| 2 | MXFP8 matmul | `pypto.scaled_mm` | 可行 |
| 3 | SwiGLU 切分 | tensor slice | 可行 |
| 4 | sigmoid + mul | `pypto.sigmoid` + `pypto.mul` | 可行 |
| 5 | abs + amax | `pypto.abs` + `pypto.amax` | 可行 |
| 6 | round + cast | `pypto.round` + `pypto.cast` | 可行 |
| 7 | 输出组装 | `pypto.assemble` | 可行 |

### 5.3 最终判定

**结论**：API 映射可行。

**理由**：

1. 主计算可由 `pypto.scaled_mm` 覆盖。
2. SwiGLU 的激活和乘法均可用 PyPTO vector API 表达。
3. per-token INT8 量化所需的 abs、amax、div、round、cast 均已映射。
4. 新前端调用方式可直接接收 NPU tensor，host 侧不再需要 `pypto.from_torch`。

---

## 6. 性能优化建议

### 6.1 Cube 优化

| 优化项 | API | 参数建议 | 说明 |
|--------|-----|---------|------|
| cube 分块 | `pypto.set_cube_tile_shapes()` | 由 `ShapeConfig` 提供 | 控制 M/K/N 分块 |
| split-k | `enable_split_k=True` | 已启用 | 改善 K 维并行 |
| multi data load | `enable_multi_data_load=True` | 已启用 | 优化数据加载 |

### 6.2 Vector 优化

| 优化项 | API | 参数建议 | 说明 |
|--------|-----|---------|------|
| vector 分块 | `pypto.set_vec_tile_shapes()` | `[1, 8, 256, 32]` | 控制 scale 和量化路径 |
| SwiGLU 分块 | `pypto.set_vec_tile_shapes(64, 256)` | 已配置 | 覆盖 `[M_i, N/2]` 后处理 |

### 6.3 内存优化

| 优化项 | 方法 | 说明 |
|--------|------|------|
| 分 expert 组装 | `pypto.assemble` | 避免 kernel 内 cat |
| INT8 输出 | 预分配 int8 output | 降低输出带宽 |
| per-token scale | `[M,1]` 输出 | 方便 kernel 写回 |

---

## 7. API 使用示例

### 7.1 核心计算流程

```python
current_mm_out = pypto.scaled_mm(x, weight, pypto.DT_FP32, scaled_x, scaled_weight)

value = current_mm_out[:, : n_size // 2]
gate = current_mm_out[:, n_size // 2:]
silu_value = value * pypto.sigmoid(value)
swiglu_out = pypto.mul(silu_value, gate)

x_bf16 = pypto.cast(swiglu_out, pypto.DT_BF16, pypto.CastMode.CAST_RINT)
x_fp32 = pypto.cast(x_bf16, pypto.DT_FP32)
x_abs = pypto.abs(x_fp32)
x_max = pypto.amax(x_abs, -1, True)
x_scale = pypto.div(pypto.full([shape_0, shape_1], 127.0, pypto.DT_FP32), x_max)
x_mul = pypto.mul(x_fp32, x_scale)
x_int8 = pypto.cast(pypto.cast(pypto.round(x_mul), pypto.DT_FP16, pypto.CastMode.CAST_RINT), pypto.DT_INT8)
```

### 7.2 完整 Kernel 签名

```python
@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1},
    runtime_options={"device_sched_mode": 3},
)
def scaled_matmul_kernel(
    a: pypto.Tensor(),
    b: pypto.Tensor(),
    scaled_a: pypto.Tensor(),
    scaled_b: pypto.Tensor(),
    out: pypto.Tensor(),
    out_quant: pypto.Tensor(),
    group_list,
    tile_config,
) -> None:
    pass
```

---

## 8. 风险与限制

### 8.1 已知限制

| 限制 ID | 描述 | 影响等级 | 缓解措施 |
|---------|------|---------|---------|
| L-001 | 当前仅内置 testcase6 | 低 | 后续补充更多 shape |
| L-002 | scale 为 0 时存在除零风险 | 中 | 后续增加 eps 保护 |
| L-003 | b_trans=True 未在内置测试覆盖 | 中 | 补充转置路径测试 |
| L-004 | expert 循环当前未并行 | 低 | 后续评估 parallel loop |

### 8.2 潜在风险

| 风险 ID | 描述 | 概率 | 应对方案 |
|---------|------|------|---------|
| R-001 | 大 N 场景 vector 后处理耗时高 | 中 | 调整 vector tile |
| R-002 | INT8 cast 与 golden rounding 差异 | 低 | 保留 BF16 往返和容差 |
| R-003 | 非均匀 group_list 下 tile 不均衡 | 中 | 评估分组调度策略 |

---

## 9. 参考文档

### 9.1 PyPTO API

- `pypto.frontend.jit()`: 新前端 JIT API
- `pypto.scaled_mm()`: MXFP8 scaled matmul API
- `pypto.sigmoid()`: sigmoid API
- `pypto.amax()`: 最大值归约 API
- `pypto.round()`: round API
- `pypto.cast()`: 类型转换 API
- `pypto.assemble()`: 输出组装 API

### 9.2 相关资源

- `gmm_swiglu_quant_impl.py`: PyPTO kernel 与 host 封装
- `tests/.../gmm_swiglu_quant_golden.py`: Golden 参考实现
