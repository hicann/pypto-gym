# transpose_quant_batch_matmul 算子 API 映射报告

## 1. 概述

本报告记录 `transpose_quant_batch_matmul` 算子在 PyPTO 框架中的核心 API 映射关系，包括关键 API 的功能、约束条件和可行性评估。

---

## 2. 核心API映射表

### 2.1 批量计算API

| PyPTO API | 功能 | 支持状态 | 关键参数 |
|-----------|------|---------|---------|
| `pypto.scaled_mm()` | MXFP8量化矩阵乘法 | ✅ 完全支持 | x1, x2, dtype, x1Scale, x2Scale, b_trans, scale_b_trans |
| `pypto.loop(parallel=True)` | 并行batch循环 | ✅ 完全支持 | B, parallel=True |
| `pypto.set_cache_policy()` | Cache策略设置 | ✅ 完全支持 | NONE_CACHEABLE |

**映射说明**：
- `scaled_mm`：核心计算API，实现MXFP8量化矩阵乘法，支持FP8E4M3/E5M2输入 + FP8E8M0 scale + FP16/BF16输出
- `pypto.loop(parallel=True)`：batch循环API，parallel=True启用并行执行，每个batch独立计算
- `set_cache_policy(NONE_CACHEABLE)`：输入tensor不cacheable，减少L1缓存占用

### 2.2 perm组合API映射

| permX2 | scaled_mm参数 | 说明 |
|--------|--------------|------|
| `[0, 1, 2]` | 无b_trans | x2为 `[B,K,N]` 布局，直接使用 |
| `[0, 2, 1]` | `b_trans=True, scale_b_trans=True` | x2为 `[B,N,K]` 布局，硬件内转置 |

**映射说明**：
- permX2决定scaled_mm的b_trans参数
- `[0,2,1]` 时需要b_trans=True和scale_b_trans=True，让scaled_mm在硬件内部完成转置
- 避免数据搬运，直接利用硬件transpose能力

### 2.3 配置API

| PyPTO API | 功能 | 参数说明 |
|-----------|------|---------|
| `pypto.set_cube_tile_shapes()` | Cube分块配置 | m_tile_shape, k_tile_shape, n_tile_shape |
| `pypto.set_vec_tile_shapes()` | Vector分块配置 | vector_tile_shape (4维) |

**映射说明**：
- `set_cube_tile_shapes`：配置Cube核的M/K/N轴分块，影响矩阵乘法性能
- `set_vec_tile_shapes`：配置Vector核的分块，影响向量操作性能

### 2.4 JIT配置API

| 参数 | 功能 | 配置示例 |
|------|------|---------|
| `auto_mix_partition` | 自动混合分区 | 1 |
| `cube_l1_reuse_setting` | Cube L1复用配置 | {-1: 2} |
| `cube_nbuffer_setting` | Cube NBuffer配置 | {-1: 2} |
| `vec_nbuffer_setting` | Vector NBuffer配置 | {-2: 1, -1: 16} |
| `stitch_function_max_num` | 函数拼接上限 | 512 |
| `device_sched_mode` | 设备调度模式 | 0 |

---

## 3. 约束条件清单

### 3.1 关键约束

| 约束ID | 约束描述 | 影响 | 解决方案 |
|--------|---------|------|---------|
| C-ALIGN-001 | K轴64对齐：K % 64 == 0 | MX量化要求 | 验证K值，确保为64倍数 |
| C-ALIGN-002 | Scale形状对齐 | MXFP8格式要求 | x1Scale `[M,B,K//64,2]` |
| C-PERM-001 | permX2决定b_trans参数 | 计算路径选择 | permX2=[0,2,1] 时传b_trans=True |
| C-DTYPE-001 | 输出dtype决定输出tensor类型 | 内存分配 | dtype=1→FP16, dtype=27→BF16 |
| C-BATCH-001 | B轴编译期固定 | 动态化限制 | batch_size作为ShapeConfig参数 |

### 3.2 约束详细说明

**C-ALIGN-001: K轴64对齐**

```
K 必须是 64 的倍数
```

**验证方法**：
```python
assert K % 64 == 0, f"K must be multiple of 64, got K={K}"
```

**C-PERM-001: perm组合约束**

```
permX2=[0,1,2] → scaled_mm(..., b_trans=False)
permX2=[0,2,1] → scaled_mm(..., b_trans=True, scale_b_trans=True)
```

---

## 4. 常见问题

### Q1: 为什么permX2=[0,2,1]需要b_trans=True？

**答**：
- permX2=[0,2,1]表示x2的布局是 `[B,N,K]`，即"反序"存储
- `b_trans=True` 让 `scaled_mm` 在硬件内部完成转置，将 `[B,N,K]` 转为 `[B,K,N]`
- `scale_b_trans=True` 同时处理scale tensor的转置
- 这样避免了额外的数据搬运操作

### Q2: M轴动态化如何实现？

**答**：
- M轴使用动态shape，运行时通过 `shape[0]` 获取实际大小
- 编译期不固定M值，同一kernel可处理不同M的输入
- B轴在编译期固定（batch_size参数），通过parallel loop切分

### Q3: 为什么输入tensor使用NONE_CACHEABLE？

**答**：
- MXFP8数据是一次性读取，不需要缓存复用
- NONE_CACHEABLE减少L1缓存占用，为其他tensor腾出空间
- 对性能无负面影响（数据只读取一次）

---

## 5. API使用示例

### 5.1 核心计算流程

```python
import pypto

# 配置参数
M = 8192
K = 128
N = 512
B = 128

# 设置Tile Shapes
pypto.set_cube_tile_shapes([256, 256], [128, 128], [256, 256])
pypto.set_vec_tile_shapes(1, 128, 256, 32)

# 设置Cache策略
x1.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
x2.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
x1Scale.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
x2Scale.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)

# 并行循环：每个batch独立计算
for b_idx in pypto.loop(B, name="LOOP_B", idx_name="b_idx", parallel=True):
    x1_slice = x1[:, b_idx, :]           # [M, K]
    x1_scale_slice = x1Scale[:, b_idx, :, :]  # [M, K//64, 2]

    if permX2 == [0, 1, 2]:
        # x2为 [B,K,N] 布局 — 不需要b_trans
        mm_result = pypto.scaled_mm(
            x1_slice, x2[b_idx, :, :], out_dtype,
            x1_scale_slice, x2Scale[b_idx, :, :, :]
        )
    else:
        # x2为 [B,N,K] 布局 — 需要b_trans
        mm_result = pypto.scaled_mm(
            x1_slice, x2[b_idx, :, :], out_dtype,
            x1_scale_slice, x2Scale[b_idx, :, :, :],
            b_trans=True, scale_b_trans=True
        )

    out[:, b_idx, :] = mm_result
```

### 5.2 perm组合示例

```python
# permX2=[0,1,2] (K,N顺序)
x2_shape = [B, K, N]           # [128, 128, 512]
x2Scale_shape = [B, K//64, N, 2]  # [128, 2, 512, 2]
# scaled_mm参数: 无b_trans

# permX2=[0,2,1] (N,K反序)
x2_shape = [B, N, K]           # [128, 512, 128]
x2Scale_shape = [B, N, K//64, 2]  # [128, 512, 2, 2]
# scaled_mm参数: b_trans=True, scale_b_trans=True
```

---

## 6. 可行性评估

### 6.1 API完备性

| 评估项 | 结果 | 说明 |
|--------|------|------|
| 所需 API 是否存在 | ✅ 是 | scaled_mm、loop、set_cache_policy等核心API均存在 |
| API 功能是否完整 | ✅ 是 | 支持MXFP8量化、并行循环、perm组合 |
| 约束是否可满足 | ✅ 是 | K轴对齐、Scale形状、perm组合约束明确 |
| 性能是否可优化 | ✅ 是 | 通过NBuffer、TileShape和Cache配置可优化 |

### 6.2 最终判定

**结论**：✅ **API 映射完全可行**

**理由**：
1. PyPTO提供完整的MXFP8量化矩阵乘法API（scaled_mm）
2. parallel loop支持batch并行计算
3. b_trans/scale_b_trans支持硬件内转置，避免数据搬运
4. M轴动态化减少重编译开销

---

## 7. 参考文档

### 7.1 PyPTO API 文档

- `pypto.scaled_mm()`：MXFP8量化矩阵乘法API（支持b_trans参数）
- `pypto.loop()`：并行循环API
- `pypto.set_cache_policy()`：Cache策略配置API
- `pypto.set_cube_tile_shapes()`：Cube分块配置API
- `pypto.set_vec_tile_shapes()`：Vector分块配置API

### 7.2 相关资源

- [MXFP8量化规范](../../../docs/tutorials/quantization/)
- [PyPTO编程指南](../../../docs/pypto_programming_guide.md)
- [性能优化指南](../../../docs/tutorials/debug/performance.md)