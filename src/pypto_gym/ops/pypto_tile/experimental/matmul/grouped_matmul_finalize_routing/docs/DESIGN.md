# grouped_matmul_finalize_routing 算子设计文档

## 1. 设计概述

### 1.1 算子定位

`grouped_matmul_finalize_routing` 是 MoE grouped matmul 的 finalize routing 融合算子，用于将按 expert 计算得到的 MXFP8 matmul 输出回写到 token 输出空间，并支持 logit 加权和 shared expert 叠加。

### 1.2 核心特性

- **MXFP8 grouped matmul**：使用 `pypto.scaled_mm` 完成 FP8 E4M3/E5M2 输入与 E8M0FNU scale 的矩阵乘。
- **expert 并行**：expert 维度通过 `pypto.loop(..., parallel=True)` 并行。
- **scatter-add finalize**：使用 `pypto.index_put_(..., accumulate=True)` 实现 `row_index` 累加。
- **FP32 输出累加**：matmul 输出和最终结果均使用 FP32，降低累加误差。

---

## 2. 计算图设计

### 2.1 整体计算流程

```
输入: x1[M, K] FP8
     x2[E, K, N] 或 x2[E, N, K] FP8
     scale[ceil(K/64), N, 2] 或 scale[N, ceil(K/64), 2] E8M0FNU
     pertoken_scale[M, ceil(K/64), 2] E8M0FNU
     logit[M] FP32
     row_index[M] INT64
     out[batch, N] FP32

Step 1: expert 并行循环
  └─ for expert_idx in range(E)

Step 2: expert 切片
  ├─ x_tile = x1[start:end, :]                    [M_i, K]
  ├─ weight_tile = x2[expert_idx, :, :]           [K, N] 或 [N, K]
  └─ pertoken_scale_tile = pertoken_scale[start:end, :, :]

Step 3: MXFP8 scaled matmul
  └─ mm_result = scaled_mm(...)                   [M_i, N] FP32

Step 4: logit 加权
  ├─ logit_2d = unsqueeze(logit[start:end], -1)   [M_i, 1]
  └─ mm_result = mul(mm_result, logit_2d)         [M_i, N]

Step 5: row_index 累加
  └─ index_put_(out, row_index[start:end], mm_result, accumulate=True)

输出: out[batch, N] FP32
```

### 2.2 计算图可视化

```
 x1(FP8)       x2(FP8)        pertoken_scale      scale
    │             │                │               │
    └──── slice ──┴──── slice ─────┴──── slice ────┘
                         │
                         ↓
                  pypto.scaled_mm
                         │
                         ↓
                  mm_result(FP32)
                         │
 logit(FP32) ─ unsqueeze ┘
                         ↓
                    pypto.mul
                         │
                         ↓
                 weighted_result
                         │
 row_index(INT64) ───────┘
                         ↓
        pypto.index_put_(accumulate=True)
                         │
                         ↓
                    out(FP32)
```

**数据流说明**：

| 分支 | 输入 | 计算 | 输出 | Shape 变化 |
|------|------|------|------|-----------|
| Matmul | `x1`、`x2`、scale | `scaled_mm` | `mm_result` | `[M_i,K] × [K,N] -> [M_i,N]` |
| Logit | `logit` | `unsqueeze` + `mul` | `weighted_result` | `[M_i] -> [M_i,1] -> [M_i,N]` |
| Routing | `row_index`、`weighted_result` | `index_put_` | `out` | `[M_i,N] -> [batch,N]` |
| Shared | `shared_input` | host 侧 add | `out_host` | `[batch,N] -> [batch,N]` |

---

## 3. Tiling 策略

### 3.1 循环结构

**expert 轴并行**：

```python
for expert_idx in pypto.loop(config.num_experts, parallel=True):
    start = expert_idx * token_num
    end = (expert_idx + 1) * token_num
```

**并行策略**：

- 每个 expert 独立处理自己的 token 切片。
- expert 间 matmul 互不依赖。
- 输出写回可能因 `row_index` 重复而存在累加冲突，使用 accumulate 语义保证正确性。

### 3.2 Cube 分块

**分块配置**：

| 维度 | 配置来源 | 示例 |
|------|---------|------|
| M tile | `config.m_tile_shape` | `[128, 128]` |
| K tile | `config.k_tile_shape` | `[64, 192]` |
| N tile | `config.n_tile_shape` | `[256, 1024]` |

**分块原理**：

- K 维与 MXFP8 scale block 对齐。
- N 维按大块处理，提高 cube 计算效率。
- M 维按 expert token 范围切分。

### 3.3 Vector 分块

**分块配置**：

| 操作 | Tile Shape | 说明 |
|------|-----------|------|
| scale/logit 前置配置 | `config.vector_tile_shape` | 用于 vector 路径初始配置 |
| logit 加权 | `(m_tile_shape[-1], n_tile_shape[-1])` | 对 `[M_i, N]` 结果做广播乘法 |
| row_index 写回 | `(m_tile_shape[-1], n_tile_shape[-1])` | scatter-add 输出 |

### 3.4 内存访问模式

| Tensor | Shape | 访问模式 | Tile 优化 |
|--------|-------|---------|----------|
| `x1` | `[M, K]` | expert 范围连续切片 | M/K 分块 |
| `x2` | `[E, *, *]` | 按 expert 读取 | K/N 分块 |
| `scale` | `[ceil(K/64), N, 2]` 或 `[N, ceil(K/64), 2]` | 广播给各 expert | K/N 分块 |
| `pertoken_scale` | `[M, ceil(K/64), 2]` | expert 范围连续切片 | M/K 分块 |
| `out` | `[batch, N]` | `row_index` 非连续写 | accumulate 写回 |

---

## 4. Loop 结构设计

### 4.1 外层循环（expert 轴）

```python
token_num = config.m // config.num_experts
for expert_idx in pypto.loop(config.num_experts, parallel=True):
    start = expert_idx * token_num
    end = (expert_idx + 1) * token_num
    x_tile = x1[start:end, :]
    pertoken_scale_tile = pertoken_scale[start:end, :, :]
    weight_tile = x2[expert_idx, :, :]
```

**循环参数**：

- `config.num_experts`：expert 数量。
- `parallel=True`：expert 维度并行。
- `token_num`：当前实现中每个 expert 的 token 数。

### 4.2 内层计算（单个 expert）

```python
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

if config.has_logit:
    logit_tile = logit[start:end]
    logit_2d = pypto.unsqueeze(logit_tile, -1)
    mm_result = pypto.mul(mm_result, logit_2d)

index_tile = row_index[start:end]
pypto.index_put_(out, (index_tile,), mm_result, accumulate=True)
```

**计算优化**：

- `scaled_mm` 完成主要 cube 计算。
- logit 加权通过 vector 计算完成。
- `index_put_` 直接作用于输出，减少中间拷贝。

---

## 5. 数据流设计

### 5.1 输入预处理

**Host wrapper 函数职责**：

```python
def gen_pypto(inputs: FinalizeRoutingInputs) -> torch.Tensor:
    x1 = inputs.x1.npu()
    x2 = inputs.x2.npu()
    scale = inputs.scale.npu()
    pertoken_scale = inputs.pertoken_scale.npu()
    logit = inputs.logit.npu()
    row_index = inputs.row_index.npu()

    out_host = inputs.out.clone()
    if inputs.config.has_shared_input:
        out_host[shared_start:shared_end, :] += (
            inputs.shared_input.to(torch.float32) * inputs.config.shared_input_weight
        )
    out = out_host.npu()

    gmm_finalize_routing_kernel(...)
    return out.to(torch.float32)
```

### 5.2 中间数据流

```
expert 切片 [M_i, ...]
    ↓
MXFP8 scaled_mm
    ↓
FP32 matmul 输出 [M_i, N]
    ↓
logit 广播乘法
    ↓
row_index scatter-add
    ↓
out[batch, N]
```

### 5.3 输出组装

```python
pypto.index_put_(out, (index_tile,), mm_result, accumulate=True)
```

**组装参数**：

- `out`：最终输出 tensor。
- `index_tile`：当前 expert token 对应的输出行。
- `mm_result`：当前 expert 计算后的 `[M_i, N]` 结果。
- `accumulate=True`：允许多个 token 累加到同一输出行。

---

## 6. 精度设计

### 6.1 精度转换路径

```
输入精度: FP8 E4M3/E5M2 (x1, x2) + E8M0FNU (scale)
    ↓
计算精度: FP32 (scaled_mm 输出、logit 加权、scatter-add)
    ↓
输出精度: FP32 (out)
```

### 6.2 精度保证措施

| 措施 | 说明 |
|------|------|
| `scaled_mm` 输出 FP32 | 降低 matmul 结果误差 |
| logit 使用 FP32 | 避免加权阶段引入额外量化误差 |
| out 使用 FP32 | 支持 scatter-add 累加 |
| golden 展开 scale 对比 | 对齐 MXFP8 缩放语义 |

### 6.3 精度验证标准

```python
RTOL = 1e-3
ATOL = 1e-3
numpy.testing.assert_allclose(result, golden, rtol=RTOL, atol=ATOL)
```

---

## 7. 性能优化设计

### 7.1 双缓冲策略

```python
@pypto.frontend.jit(
    pass_options={
        "cube_nbuffer_setting": {-1: 4},
        "vec_nbuffer_setting": {-2: 1, -1: 4},
    },
    runtime_options={"stitch_function_max_num": 8},
)
```

**参数说明**：

- `cube_nbuffer_setting`：控制 cube 路径缓冲。
- `vec_nbuffer_setting`：控制 vector 路径缓冲。
- `stitch_function_max_num`：控制 stitch function 数量上限。

### 7.2 向量化策略

| 操作 | Tile Shape | 向量化效率 | 说明 |
|------|-----------|------------|------|
| logit 加权 | `[M_tile, N_tile]` | 高 | 连续处理 matmul 输出 |
| row_index 写回 | `[M_tile, N_tile]` | 中 | 写回地址由 `row_index` 决定 |
| shared input 叠加 | host 侧 | 可控 | 避免 kernel 内分支 |

### 7.3 内存优化

| 优化项 | 方法 | 效果 |
|--------|------|------|
| expert 切片 | 按 expert 范围读取 `x1`/`pertoken_scale` | 降低无效访问 |
| out 原地累加 | `index_put_(accumulate=True)` | 减少额外输出 buffer |
| shared 预处理 | host 侧写入 `out_host` | 简化 kernel |

---

## 8. 边界情况处理

### 8.1 Shape 验证

```python
assert config.transpose_x1 is False
assert x1.shape == (config.m, config.k)
assert row_index.shape == (config.m,)
assert out.shape == (config.batch, config.n)
```

### 8.2 DType 验证

```python
assert x1.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
assert x2.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
assert scale.dtype == torch.float8_e8m0fnu
assert pertoken_scale.dtype == torch.float8_e8m0fnu
```

### 8.3 Index 验证

```python
assert torch.all(row_index >= 0)
assert torch.all(row_index < config.batch)
```

---

## 9. Kernel 签名设计

### 9.1 Tensor 标记

| 参数 | Shape | 标记 | 说明 |
|------|-------|------|------|
| `x1` | `[M, K]` | 运行时 tensor | FP8 token 输入 |
| `x2` | `[E, K, N]` 或 `[E, N, K]` | 运行时 tensor | FP8 expert 权重 |
| `scale` | `[ceil(K/64), N, 2]` 或 `[N, ceil(K/64), 2]` | 运行时 tensor | 权重 scale |
| `pertoken_scale` | `[M, ceil(K/64), 2]` | 运行时 tensor | token scale |
| `logit` | `[M]` | 运行时 tensor | token 权重 |
| `row_index` | `[M]` | 运行时 tensor | 输出行索引 |
| `out` | `[batch, N]` | 运行时 tensor | 输出 tensor |

### 9.2 完整签名

```python
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
    """Fused grouped matmul finalize routing kernel."""
```

---

## 10. 测试设计

### 10.1 测试矩阵

| 测试名称 | batch | M | K | N | E | 验证点 |
|---------|-------|---|---|---|---|--------|
| case1 | 128 | 768 | 6144 | 4096 | 32 | 标准 FP8 E4M3 路径 |
| case2 | 256 | 768 | 8192 | 4096 | 32 | 大 K |
| case3 | 64 | 128 | 5120 | 4096 | 8 | 小 M |
| case4 | 64 | 256 | 7168 | 4096 | 16 | FP8 E5M2 |

### 10.2 精度验证方法

```python
result_np = result.cpu().numpy()
golden_np = golden.cpu().numpy()
assert_allclose(golden_np, result_np, rtol=1e-3, atol=1e-3)
```

### 10.3 通过标记

- `PASSED`：当前配置精度验证通过。

---

## 11. 实现约束

### 11.1 硬编码参数

| 参数 | 值 | 原因 |
|------|---|------|
| `transpose_x1` | `False` | 目标算子当前路径限制 |
| expert token 数 | `M // E` | 当前 kernel 简化为均匀分组 |

### 11.2 编译时确定

| 参数 | 来源 | 说明 |
|------|------|------|
| tile shape | `FinalizeRoutingConfig` | 控制 cube/vector 分块 |
| `transpose_x2` | `FinalizeRoutingConfig` | 控制 B 矩阵和 scale 布局 |

### 11.3 运行时确定

| 参数 | 来源 |
|------|------|
| `row_index` | 输入 tensor |
| `logit` | 输入 tensor |
| `out` 初值 | host 侧构造 |

---

## 12. 性能预期

### 12.1 计算复杂度

| 操作 | FLOPS | 说明 |
|------|-------|------|
| grouped matmul | `2 × M × K × N` | 主计算 |
| logit 加权 | `M × N` | vector 乘法 |
| row_index 累加 | `M × N` | scatter-add |
| shared input 叠加 | `batch × N` | host 侧加法 |

**总 FLOPS**：

```
约 2 × M × K × N + 2 × M × N + batch × N
```

### 12.2 内存访问量

| Tensor | 访问量 | 说明 |
|--------|-------|------|
| `x1` | `M × K × 1 byte` | FP8 |
| `x2` | `E × K × N × 1 byte` | FP8 |
| `scale` | `ceil(K/64) × N × 2 × 1 byte` | E8M0FNU |
| `pertoken_scale` | `M × ceil(K/64) × 2 × 1 byte` | E8M0FNU |
| `out` | `batch × N × 4 bytes` | FP32 |

### 12.3 性能特征

- **计算密集**：主耗时来自大规模 matmul。
- **scatter 写回敏感**：`row_index` 分布会影响写回效率。
- **tile 参数敏感**：M/K/N tile 对性能影响较大。

---

## 13. 文件清单

| 文件 | 职责 |
|------|------|
| `SPEC.md` | 需求规格 |
| `API_REPORT.md` | API 映射报告 |
| `DESIGN.md` | 本设计文档 |
| `README.md` | 使用说明 |
| `gmm_finalize_routing_impl.py` | PyPTO kernel 与 `gen_pypto` |
| `tests/.../gmm_finalize_routing_golden.py` | Golden 参考实现 |
| `tests/.../test_gmm_finalize_routing.py` | 精度验证入口 |

---

## 14. 参考文档

- PyPTO `scaled_mm` API
- PyPTO `index_put_` API
- `aclnnGroupedMatmulFinalizeRoutingV3` 算子语义
