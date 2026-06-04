# quant_batch_matmul 算子 API 映射报告

## 1. 概述

本报告记录 **F-T pertensor** 变体 `quant_batch_matmul` 在 PyPTO 框架中的 API 映射分析，包括 PyTorch golden 到 PyPTO API 的对应关系、约束条件与可行性评估。

---

## 2. PyTorch 操作分解

### 2.1 核心操作序列

根据 golden 实现，算子包含以下核心操作：

| 序号 | PyTorch 操作 | 输入 Shape | 输出 Shape | 说明 |
|------|-------------|-----------|-----------|------|
| 1 | `float()` 提升 | `[batch,m,k]` FP8 | FP32 | 反量化到 FP32 计算 |
| 2 | `transpose(-1,-2)` | `[batch,n,k]` | `[batch,k,n]` | F-T 右矩阵转置 |
| 3 | `torch.matmul()` | `[batch,m,k] × [batch,k,n]` | `[batch,m,n]` | batch matmul |
| 4 | 标量乘法 | `[batch,m,n] × scalar` | `[batch,m,n]` | `x1Scale * x2Scale`（经掩码） |
| 5 | `clamp(-128,127)` | FP32 | FP32 | INT8 范围 |
| 6 | `round()` | FP32 | FP32 | 四舍五入 |
| 7 | `to(int8)` | FP32 | INT8 | 输出类型 |

### 2.2 操作分类

| 类型 | 操作数量 | PyTorch API |
|------|---------|-------------|
| 矩阵乘 | 1 | `torch.matmul` + `transpose` |
| 标量缩放 | 1 | 广播乘法 |
| 量化 | 2 | `clamp`、`round` |
| 类型转换 | 2 | `float()`、`to(int8)` |

---

## 3. PyPTO API 映射表

### 3.1 量化 Matmul API

| PyTorch/Golden 操作 | PyPTO API | 支持状态 | 约束条件 |
|--------------------|-----------|---------|---------|
| `matmul + scale + quant` | `pypto.matmul(..., DT_INT8, extend_params={"scale": s})` | 支持 | F-T：`b_trans=True`；scale 为标量 |

**映射说明**：

- Golden 在 CPU 上用 FP32 matmul 后乘 scale 再 round/clamp。
- PyPTO 在 cube fixpipe 路径一次完成 matmul、scale、INT8 输出。
- `combined_scale` 必须在 host 侧经 `mask_fixpipe_scale` 与 golden 对齐。

**示例**：

```python
result = pypto.matmul(
    x1_block,           # [m, k]
    x2_block,           # [n_block, k]
    pypto.DT_INT8,
    a_trans=False,
    b_trans=True,
    extend_params={"scale": config.combined_scale},
)
```

### 3.2 维度与切片 API

| PyTorch API | PyPTO API | 支持状态 | 约束条件 |
|------------|-----------|---------|---------|
| `reshape` | `pypto.reshape(..., inplace=True)` | 支持 | F-4 展平 |
| 切片/view | `pypto.view(tensor, shape, offset, valid_shape=...)` | 支持 | 尾块需 `valid_shape` |
| 写回 | `pypto.assemble(src, index, dst)` | 支持 | 分块写入 `out` |

**映射说明**：

- batch/N 循环通过 `pypto.loop` 表达，而非 Python `for`。
- `x2` 物理为 `[n,k]`，matmul 时 `b_trans=True` 等价 golden 的 `transpose`。

### 3.3 循环 API

| 语义 | PyPTO API | 支持状态 | 约束条件 |
|------|-----------|---------|---------|
| batch 循环 | `pypto.loop(batch, name="LOOP_BATCH")` | 支持 | 顺序 loop |
| N 分块循环 | `pypto.loop(n_loop, name="LOOP_N")` | 支持 | `n_loop = ceil(n / n_block)` |

### 3.4 Tiling API

| 配置项 | PyPTO API | 支持状态 |
|--------|-----------|---------|
| Cube tile | `pypto.set_cube_tile_shapes(m, k, n)` | 支持 |
| Vector tile | `pypto.set_vec_tile_shapes(*vector_tile_shape)` | 支持 |

---

## 4. 约束条件清单

### 4.1 数据类型约束

| 约束 ID | 约束描述 | 影响 | 解决方案 |
|---------|---------|------|---------|
| C-DTYPE-001 | 输入必须为 FP8 E4M3 | kernel dtype 检查 | `DT_FP8E4M3` |
| C-DTYPE-002 | 输出必须为 INT8 | fixpipe 量化 | `DT_INT8` |
| C-DTYPE-003 | scale 为 FP32 标量 | host 读取后写 `combined_scale` | `compute_combined_scale` |
| C-DTYPE-004 | fixpipe scale 位宽 | 与硬件不一致导致精度失败 | `mask_fixpipe_scale` |

### 4.2 Shape 约束

| 约束 ID | 约束描述 | 影响 | 解决方案 |
|---------|---------|------|---------|
| C-SHAPE-001 | 仅 F-T 布局 | 不支持 B-B/其他 transpose 组合 | 固定 `b_trans=True` |
| C-SHAPE-002 | `x1`/`x2` batch 维一致 | 非法 shape | `ori_shape[0]` 约束 |
| C-SHAPE-003 | N 尾块 | 越界读写 | `valid_n` + `valid_shape` |
| C-SHAPE-004 | tile 不超过实际轴长 | 编译/运行失败 | m 小时 `m_tile=[4,4]` 等 |

### 4.3 内存约束

| 约束 ID | 约束描述 | 影响 | 解决方案 |
|---------|---------|------|---------|
| C-MEM-001 | 输入非连续 | NPU 性能/正确性 | `.contiguous()` |
| C-MEM-002 | `out` 需预分配 | kernel 为 inplace 写 | `torch.zeros(...).npu()` |
| C-MEM-003 | 大 shape 任务数多 | 编译时间长 | 调 `n_block_size` |

---

## 5. 可行性评估

### 5.1 API 完备性

| 评估项 | 结果 | 说明 |
|--------|------|------|
| 所需 API 是否存在 | 是 | matmul、reshape、view、assemble、loop |
| API 功能是否完整 | 是 | 覆盖 F-T batch matmul + pertensor scale |
| 性能是否可接受 | 是 | cube 主路径 + N 分块 |
| 约束是否可满足 | 是 | 已有多 shape 精度测试 |

### 5.2 实现路径

| 步骤 | PyTorch/Golden | PyPTO 实现 | 可行性 |
|------|----------------|-----------|--------|
| 1 | batch matmul | `LOOP_BATCH` + `view` | 可行 |
| 2 | N 分块 | `LOOP_N` + `view` | 可行 |
| 3 | scale + quant | `matmul(..., extend_params)` | 可行 |
| 4 | 写回 out | `assemble` | 可行 |
| 5 | fixpipe 掩码 | host `compute_combined_scale` | 可行 |

### 5.3 最终判定

**结论**：API 映射可行。

**理由**：

1. F-T matmul 可由 `pypto.matmul(b_trans=True)` 表达。
2. pertensor scale 可由 `extend_params["scale"]` 融合进 fixpipe。
3. batch/N 分块可由 `pypto.loop` + `view`/`assemble` 表达。
4. golden 与 kernel 通过 fixpipe 掩码对齐，满足 `rtol=1e-3`。

---

## 6. 性能优化建议

### 6.1 Cube 优化

| 优化项 | API | 参数建议 | 说明 |
|--------|-----|---------|------|
| cube 分块 | `set_cube_tile_shapes` | 按 m/k/n 自适应 | 小 m 勿用过大的 m_tile |
| cube buffer | `cube_nbuffer_setting` | `{-1: 2}` | 当前 JIT 配置 |
| N 分块 | `n_block_size` | 512 / 256 / 128 | 平衡任务数与 L1 |

### 6.2 Vector 优化

| 优化项 | API | 参数建议 | 说明 |
|--------|-----|---------|------|
| vector 分块 | `set_vec_tile_shapes` | `[m, n_block]` | 对齐 assemble |
| vector buffer | `vec_nbuffer_setting` | `{-2: 1, -1: 2}` | 当前 JIT 配置 |

### 6.3 内存优化

| 优化项 | 方法 | 说明 |
|--------|------|------|
| F-4 展平 | inplace reshape | 减少 3D 操作 |
| scale 融合 | `extend_params` | 避免 matmul 后 mul |
| 预分配 out | host `torch.zeros` | 单次分配 |

---

## 7. API 使用示例

### 7.1 核心 Kernel 片段

```python
pypto.set_cube_tile_shapes(config.m_tile_shape, config.k_tile_shape, config.n_tile_shape)
pypto.set_vec_tile_shapes(*config.vector_tile_shape)

x1_flat = pypto.reshape(x1, [batch * m, k], inplace=True)
x2_flat = pypto.reshape(x2, [batch * n, k], inplace=True)

for b_idx in pypto.loop(batch, name="LOOP_BATCH", idx_name="b_idx"):
    x1_block = pypto.view(x1_flat, [m, k], [b_idx * m, 0])
    for n_idx in pypto.loop(n_loop, name="LOOP_N", idx_name="n_idx"):
        n_off = n_idx * n_block
        valid_n = (n - n_off).min(n_block)
        x2_block = pypto.view(x2_flat, [n_block, k], [b_idx * n + n_off, 0], valid_shape=[valid_n, k])
        result = pypto.matmul(x1_block, x2_block, pypto.DT_INT8, a_trans=False, b_trans=True,
                              extend_params={"scale": config.combined_scale})
        # reshape + assemble ...
```

### 7.2 Host 调用

```python
from experimental.matmul.quant_batch_matmul.quant_batch_matmul_impl import (
    QuantBatchMatmulConfig,
    QuantBatchMatmulInputs,
    quant_batch_matmul,
)

config = QuantBatchMatmulConfig(ori_shape=[4, 4, 7168, 2048], ...)
inputs = QuantBatchMatmulInputs(x1=..., x2=..., x1_scale=..., x2_scale=...)
out = quant_batch_matmul(inputs, config)
```

### 7.3 完整 JIT 签名

```python
@pypto.frontend.jit(
    pass_options={"cube_nbuffer_setting": {-1: 2}, "vec_nbuffer_setting": {-2: 1, -1: 2}},
    runtime_options={"stitch_function_max_num": 128},
)
def quant_batch_matmul_kernel(
    x1: pypto.Tensor(),
    x2: pypto.Tensor(),
    out: pypto.Tensor(),
    config: QuantBatchMatmulConfig,
):
    ...
```

---

## 8. 风险与限制

### 8.1 已知限制

| 限制 ID | 描述 | 影响等级 | 缓解措施 |
|---------|------|---------|---------|
| L-001 | 仅 F-T pertensor | 中 | 其他布局需新模块 |
| L-002 | 仅 FP8→INT8 | 中 | 扩展需改 `__post_init__` |
| L-003 | 标量 scale | 低 | perblock 需另实现 |
| L-004 | debug/monitor 常开 | 低 | 生产可关闭 JIT 选项 |

### 8.2 潜在风险

| 风险 ID | 描述 | 概率 | 应对方案 |
|---------|------|------|---------|
| R-001 | fixpipe 与 golden round 差异 | 中 | fixpipe 掩码 + rtol 1e-3 |
| R-002 | 大 shape SYNC/设备偶发失败 | 低 | 单 case 重试、拆分进程 |
| R-003 | 大 K/N 编译 pass 超时警告 | 中 | 调整 tile / n_block |
| R-004 | m 过大时 tile 超界 | 低 | 自适应 m_tile |

---

## 9. 参考文档

### 9.1 PyPTO API

- `pypto.matmul()`：矩阵乘 + 输出 dtype + `extend_params`
- `pypto.reshape()` / `pypto.view()` / `pypto.assemble()`
- `pypto.loop()`：循环
- `pypto.set_cube_tile_shapes()` / `pypto.set_vec_tile_shapes()`

### 9.2 相关资源

- `quant_batch_matmul_impl.py`：实现源码
- `tests/.../quant_batch_matmul_golden.py`：Golden
- `aclnnQuantMatmulV5`：CANN 算子语义
- `docs/SPEC.md`、`docs/DESIGN.md`：规格与设计
