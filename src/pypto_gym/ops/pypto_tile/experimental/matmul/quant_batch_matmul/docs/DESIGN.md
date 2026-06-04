# quant_batch_matmul 算子设计文档

## 1. 设计概述

### 1.1 算子定位

`quant_batch_matmul` 实现 **F-T 布局 + pertensor 标量 scale** 的 batch 量化矩阵乘，输出 INT8。对齐 `aclnnQuantMatmulV5` F-T 路径，在 PyPTO 中通过 cube matmul fixpipe 完成 scale 融合与量化。

### 1.2 核心特性

- **F-T 物理布局**：`x1 [batch,m,k]`，`x2 [batch,n,k]`，`b_trans=True`。
- **pertensor scale**：标量 `x1Scale`/`x2Scale`，经 fixpipe 掩码后传入 matmul。
- **F-4 展平**：batch 与 m/n 合并，kernel 内仅 2D `view`。
- **N 轴分块**：`LOOP_N` + `n_block_size` 控制并行粒度与 L1 占用。

---

## 2. 计算图设计

### 2.1 整体计算流程

```
输入: x1[batch, m, k] FP8
     x2[batch, n, k] FP8
     combined_scale (标量, host 写入 config)

Step 0: Host
  └─ compute_combined_scale → mask_fixpipe → launch_config.combined_scale

Step 1: 展平
  ├─ x1_flat = reshape(x1, [batch*m, k])
  └─ x2_flat = reshape(x2, [batch*n, k])

Step 2: LOOP_BATCH (b_idx)
  ├─ x1_block = view(x1_flat, [m, k], [b_idx*m, 0])
  └─ LOOP_N (n_idx)
        ├─ x2_block = view(x2_flat, [n_block, k], [b_idx*n + n_off, 0], valid_shape=[valid_n, k])
        ├─ result = matmul(x1_block, x2_block, INT8, b_trans=True, scale=combined_scale)
        ├─ result_out = reshape(result, [1, m, n_block])
        └─ assemble(view(..., valid_shape=[1,m,valid_n]), [b_idx,0,n_off], out)

输出: out[batch, m, n] INT8
```

### 2.2 计算图可视化

```
 x1[batch,m,k]                    x2[batch,n,k]
       │                                │
       └──────── reshape ───────────────┘
                 │                │
            x1_flat            x2_flat
                 │                │
         LOOP_BATCH b_idx         │
                 │                │
            view [m,k]            │
                 │                │
                 └──── LOOP_N ────┘
                         │
                         ↓
              pypto.matmul (cube, b_trans=True)
              extend_params: {scale}
                         │
                         ↓
                   reshape + assemble
                         │
                         ↓
                 out[batch,m,n] INT8
```

**数据流说明**：

| 阶段 | 输入 | 计算 | 输出 | Shape 变化 |
|------|------|------|------|-----------|
| 展平 | `x1`, `x2` | `reshape(inplace)` | `x1_flat`, `x2_flat` | 3D → 2D |
| Batch 切片 | `x1_flat` | `view` | `x1_block` | `[batch*m,k]` → `[m,k]` |
| N 切片 | `x2_flat` | `view` + `valid_shape` | `x2_block` | `[batch*n,k]` → `[n_block,k]` |
| Matmul | `x1_block`, `x2_block` | `matmul` + fixpipe | `result` | `[m,k]×[k,n]` → `[m,n]` INT8 |
| 写回 | `result` | `reshape` + `assemble` | `out` | 分块写入 `[batch,m,n]` |

---

## 3. Tiling 策略

### 3.1 循环结构

**Batch 轴顺序循环**：

```python
for b_idx in pypto.loop(batch, name="LOOP_BATCH", idx_name="b_idx"):
    x1_off = b_idx * m
    x1_block = pypto.view(x1_flat, [m, k], [x1_off, 0])
```

**N 轴分块循环**：

```python
n_loop = (n + n_block - 1) // n_block
for n_idx in pypto.loop(n_loop, name="LOOP_N", idx_name="n_idx"):
    n_off = n_idx * n_block
    valid_n = (n - n_off).min(n_block)
```

**并行策略**：

- 当前实现为顺序 `pypto.loop`（未对 batch/N 显式 `parallel=True`）。
- 编译器可将 loop 展开为多个 stitch 任务；`stitch_function_max_num=128`。

### 3.2 Cube 分块

**分块配置**：

| 维度 | 配置来源 | 示例（default case） |
|------|---------|---------------------|
| M tile | `config.m_tile_shape` | `[4, 4]` |
| K tile | `config.k_tile_shape` | `[128, 512]` |
| N tile | `config.n_tile_shape` | `[128, 512]` |

**调用**：

```python
pypto.set_cube_tile_shapes(
    config.m_tile_shape,
    config.k_tile_shape,
    config.n_tile_shape,
)
```

### 3.3 Vector 分块

```python
pypto.set_vec_tile_shapes(*config.vector_tile_shape)
```

典型配置：`vector_tile_shape = [m, n_block]`，与 assemble 块形状一致。

### 3.4 内存访问模式

| Tensor | Shape | 访问模式 | 说明 |
|--------|-------|---------|------|
| `x1_flat` | `[batch*m, k]` | 按 batch 连续 view `[m,k]` | F-4 展平 |
| `x2_flat` | `[batch*n, k]` | batch 偏移 + N 块 view | 尾块 valid_n |
| `out` | `[batch, m, n]` | `assemble` 分块写 | 索引 `[b_idx, 0, n_off]` |

---

## 4. Loop 结构设计

### 4.1 外层循环（batch）

```python
for b_idx in pypto.loop(batch, name="LOOP_BATCH", idx_name="b_idx"):
    x1_off = b_idx * m
    x1_block = pypto.view(x1_flat, [m, k], [x1_off, 0])
    x2_batch_off = b_idx * n
```

每个 batch 共享同一块 `x1_block`，在 N 循环中复用。

### 4.2 内层循环（N 分块）

```python
result = pypto.matmul(
    x1_block,
    x2_block,
    pypto.DT_INT8,
    a_trans=False,
    b_trans=True,
    extend_params={"scale": config.combined_scale},
)
result_out = pypto.reshape(result, [1, m, n_block])
pypto.assemble(
    pypto.view(result_out, [1, m, n_block], [0, 0, 0], valid_shape=[1, m, valid_n]),
    [b_idx, 0, n_off],
    out,
)
```

**设计要点**：

- `b_trans=True` 对应 F-T 中 `x2[n,k]` 参与 `x2^T`。
- scale 在 matmul 内完成，避免额外 vector 乘法。
- `valid_shape` 处理 `n % n_block != 0` 的尾块。

---

## 5. 数据流设计

### 5.1 Host wrapper 职责

```python
def quant_batch_matmul(inputs, config):
    x1 = inputs.x1.npu().contiguous()
    x2 = inputs.x2.npu().contiguous()
    kernel_scale, _ = compute_combined_scale(inputs.x1_scale, inputs.x2_scale)
    launch_config = replace(config, combined_scale=kernel_scale)
    out = torch.zeros(get_output_shape(launch_config), dtype=torch.int8).npu()
    quant_batch_matmul_kernel(x1, x2, out, launch_config)
    return out
```

### 5.2 Scale 处理

```python
_FIXPIPE_SCALE_MASK = 0xFFFFE000

def mask_fixpipe_scale(scale: float) -> tuple[float, float]:
    packed = struct.pack("f", float(scale))
    as_int = struct.unpack("I", packed)[0]
    masked_int = as_int & _FIXPIPE_SCALE_MASK
    golden_scale = struct.unpack("f", struct.pack("I", masked_int))[0]
    return golden_scale, golden_scale
```

kernel 与 golden 共用 `compute_combined_scale`，保证 fixpipe 与参考一致。

### 5.3 中间数据流

```
FP8 x1/x2 (NPU)
    ↓ reshape flat
LOOP_BATCH → x1_block [m,k]
    ↓ LOOP_N
x2_block [n_block,k] + matmul(scale) → INT8 [m, n_block]
    ↓ assemble
out [batch, m, n]
```

---

## 6. 精度设计

### 6.1 精度转换路径

```
输入: FP8 E4M3 (x1, x2)
    ↓
计算: cube matmul + fixpipe scale (硬件语义)
    ↓
输出: INT8 (out)

Golden: FP32 matmul → × masked scale → round → clamp[-128,127] → INT8
```

### 6.2 精度保证措施

| 措施 | 说明 |
|------|------|
| fixpipe 掩码对齐 | kernel/golden 同一 `0xFFFFE000` |
| 无后处理 mul | scale 仅通过 `extend_params` |
| INT8 输出 | 与 golden clamp 范围一致 |
| 逐点 rtol | `1e-3`，适应 fixpipe 与 round 差异 |

### 6.3 精度验证标准

```python
_PER_POINT_REL_RTOL = 1e-3
assert_allclose(result_f, golden_f, rtol=1e-3, atol=0)
```

---

## 7. 性能优化设计

### 7.1 JIT 编译选项

```python
@pypto.frontend.jit(
    pass_options={
        "cube_nbuffer_setting": {-1: 2},
        "vec_nbuffer_setting": {-2: 1, -1: 2},
    },
    runtime_options={"stitch_function_max_num": 128},
    debug_options={
        "runtime_debug_mode": 1,
        "compile_debug_mode": 1,
    },
    host_options={
        "compile_monitor_enable": 1,
        "compile_monitor_print_interval": 2,
    },
)
```

### 7.2 已验证优化

| 优化项 | 方法 | 效果 |
|--------|------|------|
| F-4 展平 | inplace reshape 到 2D | 减少 AIV view/reshape |
| N 分块 | `n_block=512` 等 | 降低任务数、改善 cube 利用率 |
| scale 融合 | `extend_params` | 去掉 matmul 后 vector mul |

### 7.3 已知无效/回退项

| 尝试 | 结果 |
|------|------|
| `loop_unroll` 过大 | 精度错误，已回退 |
| Python `for batch` 替代 `pypto.loop` | 并行度下降 |
| matmul 后 vector `mul` scale | 多余 AIV，已改为 extend_params |

---

## 8. 边界情况处理

### 8.1 Shape 验证

```python
# QuantBatchMatmulConfig.__post_init__
assert len(ori_shape) == 4
assert ori_shape[0] >= 1
assert in_dtype == pypto.DT_FP8E4M3
assert out_dtype == pypto.DT_INT8
```

### 8.2 尾块 N

```python
valid_n = (n - n_off).min(n_block)
x2_block = pypto.view(..., valid_shape=[valid_n, k])
pypto.assemble(..., valid_shape=[1, m, valid_n], ...)
```

### 8.3 x1_scale 为空

```python
if x1_scale is None:
    raw_scale = x2_value
else:
    raw_scale = x1_value * x2_value
```

---

## 9. Kernel 签名设计

### 9.1 Tensor 参数

| 参数 | Shape | 说明 |
|------|-------|------|
| `x1` | `[batch, m, k]` | FP8 左矩阵 |
| `x2` | `[batch, n, k]` | FP8 右矩阵 |
| `out` | `[batch, m, n]` | INT8 输出（预分配） |
| `config` | dataclass | 含 `ori_shape`、tile、scale 等 |

### 9.2 完整签名

```python
def quant_batch_matmul_kernel(
    x1: pypto.Tensor(),
    x2: pypto.Tensor(),
    out: pypto.Tensor(),
    config: QuantBatchMatmulConfig,
):
    """Compute F-T pertensor quantized batch matmul into out."""
```

---

## 10. 测试设计

### 10.1 测试入口

- 模块：`tests/ops/experimental/matmul/quant_batch_matmul/test_quant_batch_matmul.py`
- 标记：`@pytest.mark.soc("950", "910")`
- CLI：`--golden-only`、`--case`、`--profile`

### 10.2 精度验证方法

```python
golden = gen_golden(inputs, config)
result = quant_batch_matmul(inputs, config)
assert_allclose(result.cpu().float(), golden.cpu().float(), rtol=1e-3, atol=0)
```

### 10.3 通过标记

- `[golden] ... PASSED` / `[kernel] ... PASSED`
- 脚本结束打印 `ALL PASSED`

---

## 11. 实现约束

### 11.1 硬编码参数

| 参数 | 值 | 原因 |
|------|---|------|
| `a_trans` | `False` | F-T 左矩阵不转置 |
| `b_trans` | `True` | F-T 右矩阵转置乘 |
| `in_dtype` / `out_dtype` | FP8E4M3 / INT8 | 当前目标路径 |

### 11.2 编译时确定

| 参数 | 来源 |
|------|------|
| tile shapes | `QuantBatchMatmulConfig` |
| `n_block_size` | config |
| `combined_scale` | host 写入 config |

### 11.3 运行时确定

| 参数 | 来源 |
|------|------|
| `x1`, `x2`, scale 张量 | 输入 |
| `valid_n` | `n`, `n_off`, `n_block` |

---

## 12. 性能预期

### 12.1 计算复杂度

| 操作 | FLOPS（量级） | 说明 |
|------|--------------|------|
| batch matmul | `2 × batch × m × k × n` | 主计算 |
| assemble | `batch × m × n` | 写回 |

### 12.2 内存访问量

| Tensor | 访问量 | 说明 |
|--------|-------|------|
| `x1` | `batch × m × k × 1B` | FP8 |
| `x2` | `batch × n × k × 1B` | FP8 |
| `out` | `batch × m × n × 1B` | INT8 |

### 12.3 性能特征

- **Cube 为主**：matmul 占主导；大 N 时 N 循环次数影响调度。
- **AIV 前置/间隙**：view/reshape/assemble 在 cube 前后产生 AIV 任务。
- **编译耗时**：大 `k`/`n` 时 `LOOP_N` 子图 pass 可能较慢（compile monitor 可见）。

---

## 13. 文件清单

| 文件 | 职责 |
|------|------|
| `SPEC.md` | 需求规格 |
| `API_REPORT.md` | API 映射报告 |
| `DESIGN.md` | 本设计文档 |
| `README.md` | 使用说明 |
| `../quant_batch_matmul_impl.py` | Kernel 与 host wrapper |
| `tests/.../quant_batch_matmul_golden.py` | Golden |
| `tests/.../test_quant_batch_matmul.py` | 测试入口 |

---

## 14. 参考文档

- CANN `aclnnQuantMatmulV5` F-T pertensor 语义
- PyPTO `pypto.matmul` + `extend_params`
- PyPTO `pypto.assemble` / `pypto.view` / `pypto.loop`
