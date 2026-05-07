# grouped_matmul_swiglu_quant 算子设计文档

## 1. 设计概述

### 1.1 算子定位

`grouped_matmul_swiglu_quant` 是 MoE grouped matmul 后处理融合算子，用于将 MXFP8 expert matmul 输出转换为下游 INT8 matmul 可消费的量化输入。

### 1.2 核心特性

- **新前端 JIT**：使用 `@pypto.frontend.jit`，host 侧直接传入 NPU tensor。
- **MXFP8 grouped matmul**：使用 `pypto.scaled_mm` 完成 FP8 输入和 E8M0FNU scale 的矩阵乘。
- **SwiGLU 激活融合**：在 matmul 后直接执行 value/gate 切分和 SiLU 门控。
- **Per-token INT8 量化**：每个 token 独立计算量化 scale 并输出 INT8。

---

## 2. 计算图设计

### 2.1 整体计算流程

```
输入: a[M, K] FP8
     b[E, K, N] 或 b[E, N, K] FP8
     scaled_a[M, K/64, 2] E8M0FNU
     scaled_b[E, K/64, N, 2] 或 scaled_b[E, N, K/64, 2] E8M0FNU
     group_list[E]

Step 1: expert 顺序循环
  └─ for i in range(E)

Step 2: expert 切片
  ├─ x = a[begin:end, :]              [M_i, K]
  ├─ weight = b[i]                    [K, N] 或 [N, K]
  ├─ scaled_x = scaled_a[begin:end]   [M_i, K/64, 2]
  └─ scaled_weight = scaled_b[i]

Step 3: MXFP8 scaled matmul
  └─ current_mm_out = scaled_mm(...)  [M_i, N] FP32

Step 4: SwiGLU
  ├─ value = current_mm_out[:, :N/2]
  ├─ gate = current_mm_out[:, N/2:]
  └─ swiglu_out = value * sigmoid(value) * gate

Step 5: Per-token INT8 量化
  ├─ x_max = amax(abs(swiglu_out), -1, keepdim=True)
  ├─ x_scale = 127 / x_max
  ├─ x_int8 = cast(round(swiglu_out * x_scale), INT8)
  └─ x_scale_quant = 1 / x_scale

Step 6: 输出组装
  ├─ assemble(x_int8, [begin, 0], out)
  └─ assemble(x_scale_quant, [begin, 0], out_quant)

输出: out[M, N/2] INT8
     out_quant[M] FP32
```

### 2.2 计算图可视化

```
 a(FP8)       b(FP8)       scaled_a       scaled_b
   │            │             │              │
   └── slice ───┴── slice ────┴── slice ────┘
                         │
                         ↓
                  pypto.scaled_mm
                         │
                         ↓
                 current_mm_out(FP32)
                         │
               ┌─────────┴─────────┐
               ↓                   ↓
            value                gate
               │                   │
               ↓                   │
          sigmoid(value)           │
               │                   │
               ↓                   │
          value * sigmoid          │
               └─────────┬─────────┘
                         ↓
                    swiglu_out
                         │
                         ↓
          BF16 cast → FP32 cast
                         │
                         ↓
             abs → amax → scale
                         │
                         ↓
             div/mul → round → int8
                         │
                         ↓
                    assemble
```

**数据流说明**：

| 分支 | 输入 | 计算 | 输出 | Shape 变化 |
|------|------|------|------|-----------|
| Matmul | `a`、`b`、scale | `scaled_mm` | `current_mm_out` | `[M_i,K] × [K,N] -> [M_i,N]` |
| SwiGLU | `current_mm_out` | slice + sigmoid + mul | `swiglu_out` | `[M_i,N] -> [M_i,N/2]` |
| Quant | `swiglu_out` | abs + amax + div + round + cast | `x_int8`、`x_scale_quant` | `[M_i,N/2] -> [M_i,N/2]`、`[M_i,1]` |
| Assemble | expert 输出 | `pypto.assemble` | global output | `[M_i,*] -> [M,*]` |

---

## 3. Tiling 策略

### 3.1 循环结构

**expert 轴循环**：

```python
begin = 0
end = 0
for i in range(num_groups):
    begin = end
    end = end + group_list[i]
```

**分组策略**：

- `group_list` 记录每个 expert 的 token 数。
- 支持非均匀分组，例如 testcase6 的 `[7, 9]`。
- 每个 expert 独立完成 matmul、SwiGLU 和量化。

### 3.2 Cube 分块

**分块配置**：

| 维度 | 配置来源 | testcase6 |
|------|---------|-----------|
| M tile | `tile_config.m_tile_shape` | `[9, 9]` |
| K tile | `tile_config.k_tile_shape` | `[256, 256]` |
| N tile | `tile_config.n_tile_shape` | `[256, 256]` |

**分块原理**：

- M tile 覆盖单 expert 的 token 数。
- K tile 与 MXFP8 scale block 对齐。
- N tile 覆盖大输出维度 7168。

### 3.3 Vector 分块

**分块配置**：

| 操作 | Tile Shape | 说明 |
|------|-----------|------|
| scale 读取/前处理 | `tile_config.vector_tile_shape` | testcase6 为 `[1, 8, 256, 32]` |
| SwiGLU/quant | `(64, 256)` | 对 `[M_i, N/2]` 做 vector 后处理 |

### 3.4 内存访问模式

| Tensor | Shape | 访问模式 | Tile 优化 |
|--------|-------|---------|----------|
| `a` | `[M, K]` | expert 范围连续切片 | M/K 分块 |
| `b` | `[E, K, N]` | 按 expert 读取 | K/N 分块 |
| `scaled_a` | `[M, K/64, 2]` | expert 范围连续切片 | M/K 分块 |
| `scaled_b` | `[E, K/64, N, 2]` | 按 expert 读取 | K/N 分块 |
| `out` | `[M, N/2]` | 按 begin 偏移连续写回 | assemble |
| `out_quant` | `[M, 1]` | 按 begin 偏移连续写回 | assemble |

---

## 4. Loop 结构设计

### 4.1 外层循环（expert 轴）

```python
num_groups = b.shape[0]
begin = 0
end = 0

for i in range(num_groups):
    begin = end
    end = end + group_list[i]
    x = a[begin:end, :]
    weight = b[i]
    scaled_x = scaled_a[begin:end, :, :]
    scaled_weight = scaled_b[i]
```

**循环参数**：

- `num_groups`：expert 数量。
- `group_list[i]`：第 i 个 expert 的 token 数。
- `begin/end`：当前 expert 在全局 token 维度上的范围。

### 4.2 内层计算（单个 expert）

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
x_mul_round = pypto.round(x_mul)
x_int8 = pypto.cast(pypto.cast(x_mul_round, pypto.DT_FP16, pypto.CastMode.CAST_RINT), pypto.DT_INT8)
x_scale_quant = pypto.div(pypto.full([shape_0, shape_1], 1.0, pypto.DT_FP32), x_scale)
```

**计算优化**：

- matmul 与后处理在同一个 kernel 中完成，减少 host 往返。
- INT8 输出减少下游带宽。
- per-token scale 保持 FP32，便于后续反量化或 INT8 matmul 使用。

---

## 5. 数据流设计

### 5.1 输入预处理

**Host wrapper 函数职责**：

```python
def gen_mxfp8(inputs: GroupedMatmulInputs, tile_config: ShapeConfig):
    a = inputs.a.npu()
    b = inputs.b.npu()
    scaled_a = inputs.scaled_a.npu()
    scaled_b = inputs.scaled_b.npu()

    out = torch.zeros((a.shape[0], b.shape[-1] // 2), dtype=torch.int8).npu()
    out_quant = torch.zeros((a.shape[0], 1), dtype=torch.float32).npu()

    scaled_matmul_kernel(a, b, scaled_a, scaled_b, out, out_quant, inputs.group_list, tile_config)
    return out.to(torch.float32), out_quant.squeeze(dim=1)
```

新前端迁移后，host 侧直接传入 NPU tensor，不再使用 `pypto.from_torch`。

### 5.2 中间数据流

```
expert 切片 [M_i, ...]
    ↓
MXFP8 scaled_mm
    ↓
FP32 matmul 输出 [M_i, N]
    ↓
SwiGLU [M_i, N/2]
    ↓
BF16 → FP32 对齐
    ↓
per-token INT8 quant
    ↓
assemble 到全局输出
```

### 5.3 输出组装

```python
pypto.assemble(x_int8, [begin, 0], out)
pypto.assemble(x_scale_quant, [begin, 0], out_quant)
```

**组装参数**：

- `x_int8`：当前 expert 的量化输出。
- `x_scale_quant`：当前 expert 的 per-token scale。
- `[begin, 0]`：当前 expert 在全局输出中的起始位置。

---

## 6. 精度设计

### 6.1 精度转换路径

```
输入精度: FP8 E4M3 (a, b) + E8M0FNU (scaled_a, scaled_b)
    ↓
matmul 计算精度: FP32
    ↓
SwiGLU 计算精度: FP32
    ↓
对齐路径: BF16 → FP32
    ↓
输出精度: INT8 + FP32 scale
```

### 6.2 精度保证措施

| 措施 | 说明 |
|------|------|
| `scaled_mm` 输出 FP32 | 降低 matmul 误差 |
| SwiGLU 使用 FP32 | 避免激活阶段误差扩大 |
| BF16 往返 | 对齐 golden 的数值路径 |
| per-token scale | 每个 token 独立量化，控制 INT8 误差 |

### 6.3 精度验证标准

```python
assert_allclose(golden.cpu().numpy(), result.cpu().numpy(), rtol=1e-3, atol=1)
assert_allclose(golden_quant.cpu().numpy(), result_quant.cpu().numpy(), rtol=1e-4, atol=1e-4)
```

---

## 7. 性能优化设计

### 7.1 新前端 JIT

```python
@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1},
    runtime_options={"device_sched_mode": 3},
)
```

新前端方式使用 `pypto.Tensor()` 参数标注，host 侧直接传入 NPU tensor。

### 7.2 向量化策略

| 操作 | Tile Shape | 向量化效率 | 说明 |
|------|-----------|------------|------|
| scale 前处理 | `[1, 8, 256, 32]` | 中 | MXFP8 scale 相关 |
| SwiGLU | `(64, 256)` | 高 | 连续尾轴处理 |
| quant | `(64, 256)` | 高 | abs/amax/div/round/cast 链路 |

### 7.3 内存优化

| 优化项 | 方法 | 效果 |
|--------|------|------|
| 输出 int8 | `out` 使用 INT8 | 降低输出带宽 |
| scale 单列 | `out_quant` 使用 `[M,1]` | 便于 assemble |
| 分 expert 写回 | `begin` 偏移 assemble | 避免中间 cat |

---

## 8. 边界情况处理

### 8.1 Shape 验证

```python
assert params.n % 2 == 0
assert sum(params.group_list) == params.m
assert params.k % 64 == 0
```

### 8.2 DType 验证

```python
assert a.dtype == torch.float8_e4m3fn
assert b.dtype == torch.float8_e4m3fn
assert scaled_a.dtype == torch.float8_e8m0fnu
assert scaled_b.dtype == torch.float8_e8m0fnu
```

### 8.3 Quant 验证

```python
assert out.shape == (params.m, params.n // 2)
assert out_quant.shape == (params.m,)
```

---

## 9. Kernel 签名设计

### 9.1 Tensor 标记

| 参数 | Shape | 标记 | 说明 |
|------|-------|------|------|
| `a` | `[M, K]` | 运行时 tensor | FP8 token 输入 |
| `b` | `[E, K, N]` 或 `[E, N, K]` | 运行时 tensor | FP8 expert 权重 |
| `scaled_a` | `[M, K/64, 2]` | 运行时 tensor | token scale |
| `scaled_b` | `[E, K/64, N, 2]` 或 `[E, N, K/64, 2]` | 运行时 tensor | expert scale |
| `out` | `[M, N/2]` | 运行时 tensor | INT8 输出 |
| `out_quant` | `[M, 1]` | 运行时 tensor | FP32 scale 输出 |

### 9.2 完整签名

```python
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
    """Run grouped scaled matmul, then apply SwiGLU and per-token quantization."""
```

---

## 10. 测试设计

### 10.1 测试矩阵

| 测试名称 | M | K | N | group_list | 验证点 |
|---------|---|---|---|------------|--------|
| testcase6 | 16 | 512 | 7168 | `[7, 9]` | 非均匀 expert 分组 |

### 10.2 精度验证方法

```python
golden, golden_quant = gen_golden(grouped_inputs, params.transpose)
result, result_quant = gen_mxfp8(grouped_inputs, tile_config)
assert_allclose(golden.cpu().numpy(), result.cpu().numpy(), rtol=1e-3, atol=1)
assert_allclose(golden_quant.cpu().numpy(), result_quant.cpu().numpy(), rtol=1e-4, atol=1e-4)
```

### 10.3 通过标记

- 无显式打印标记，`assert_allclose` 不抛异常即通过。

---

## 11. 实现约束

### 11.1 硬编码参数

| 参数 | 值 | 原因 |
|------|---|------|
| SwiGLU split | `N // 2` | value/gate 二等分 |
| quant clamp | INT8 有效范围 | golden 使用 `[-127, 127]` |
| testcase | `testcase6` | 当前唯一内置 case |

### 11.2 编译时确定

| 参数 | 来源 | 说明 |
|------|------|------|
| tile shape | `ShapeConfig` | 控制 cube/vector 分块 |
| transpose flags | `ShapeConfig` | 控制 matmul 布局 |

### 11.3 运行时确定

| 参数 | 来源 |
|------|------|
| expert token 数 | `group_list` |
| 输出 shape | `a.shape` 和 `b.shape` |
| scale 输出 | kernel 内 amax 计算 |

---

## 12. 性能预期

### 12.1 计算复杂度

| 操作 | FLOPS | 说明 |
|------|-------|------|
| grouped matmul | `2 × M × K × N` | 主计算 |
| SwiGLU | `约 3 × M × N/2 + sigmoid` | value/gate 激活 |
| quant | `约 4 × M × N/2` | abs、amax、div、round、cast |

**总 FLOPS**：

```
约 2 × M × K × N + O(M × N)
```

### 12.2 内存访问量

| Tensor | 访问量 | 说明 |
|--------|-------|------|
| `a` | `M × K × 1 byte` | FP8 |
| `b` | `E × K × N × 1 byte` | FP8 |
| `scaled_a` | `M × K/64 × 2 × 1 byte` | E8M0FNU |
| `scaled_b` | `E × K/64 × N × 2 × 1 byte` | E8M0FNU |
| `out` | `M × N/2 × 1 byte` | INT8 |
| `out_quant` | `M × 4 bytes` | FP32 |

### 12.3 性能特征

- **计算密集**：主耗时来自 large-N grouped matmul。
- **后处理带宽友好**：最终输出为 INT8。
- **分组不均衡敏感**：group_list 非均匀会影响 expert 间负载。

---

## 13. 文件清单

| 文件 | 职责 |
|------|------|
| `SPEC.md` | 需求规格 |
| `API_REPORT.md` | API 映射报告 |
| `DESIGN.md` | 本设计文档 |
| `README.md` | 使用说明 |
| `gmm_swiglu_quant_impl.py` | PyPTO kernel 与 `gen_mxfp8` |
| `tests/.../gmm_swiglu_quant_golden.py` | Golden 参考实现 |
| `tests/.../test_gmm_swiglu_quant.py` | 精度验证入口 |

---

## 14. 参考文档

- PyPTO `frontend.jit` API
- PyPTO `scaled_mm` API
- PyPTO vector API
