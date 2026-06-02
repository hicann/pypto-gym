# Incre Flash Attention GQA Anti-Quant API 使用报告

## 1. 报告概述

本报告记录 `incre_flash_attention_gqa_antiquant` 算子实现中使用的 PyPTO API，包括 API 使用方式、参数配置、约束条件和注意事项。

### 1.1 API 使用统计

| API 类别 | API 数量 | 使用次数 | 说明 |
|----------|----------|----------|------|
| 数据操作 | 5 | 20+ | reshape, view, assemble, cast, tensor |
| 数学运算 | 8 | 15+ | matmul, mul, sub, div, exp, amax, sum, maximum |
| 控制流 | 3 | 6+ | loop, cond, is_loop_begin/end |
| 配置 | 3 | 10+ | set_cube_tile_shapes, set_vec_tile_shapes, set_operation_options |
| 编译器配置 | 1 | 1 | pypto.frontend.jit |

## 2. 编译器配置 API

### 2.1 `pypto.frontend.jit`

**用途**：装饰器，用于配置 kernel 函数的编译和运行选项

**使用示例**：
```python
@pypto.frontend.jit(
    runtime_options={
        "stitch_function_max_num": 256,
        "device_sched_mode": 1
    },
    pass_options={
        "cube_l1_reuse_setting": {0: 16},
        "vec_nbuffer_setting": {0: 16},
    }
)
def incre_flash_attention_gqa_antiquant_kernel(...):
    ...
```

## 3. 数据操作 API

### 3.1 `pypto.reshape`

**用途**：改变张量形状，支持 inplace 操作

**使用示例**：
```python
# Inplace reshape，避免额外内存分配
q_2d = pypto.reshape(query, (b * n1 * s1, d), inplace=True)
k_2d = pypto.reshape(key, (block_num * block_size * n2, d), inplace=True)
v_2d = pypto.reshape(value, (block_num * block_size * n2, d), inplace=True)

# 普通 reshape（finalize_output）
oi_final_4d = pypto.reshape(oi_final, [1, g_tile, 1, d])
```

### 3.2 `pypto.view`

**用途**：创建张量的视图，支持非连续内存访问和 valid_shape 设置

**使用示例**：
```python
# 从 2D K/V 中提取一个 block
kj_view = pypto.view(k_2d, [block_size, d], [(block_idx_valid * n2 + n2_idx) * block_size, 0])
vj_view = pypto.view(v_2d, [block_size, d], [(block_idx_valid * n2 + n2_idx) * block_size, 0])

# 从 2D Q 中提取一行
qi_row = pypto.view(q_2d, [1, d], [qi_row_ofs, 0])

# 设置 valid_shape（动态序列长度）
sij = pypto.view(sij, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])
vj_assemble = pypto.view(vj_assemble, [s2_tile, d], [0, 0], valid_shape=[actual_s2_tile, d])
```

### 3.3 `pypto.assemble`

**用途**：将数据组装到目标张量的指定位置

**使用示例**：
```python
# 将 K/V block 组装到组装缓冲区
pypto.assemble(kj_view, [i * block_size, 0], kj_assemble)
pypto.assemble(vj_view, [i * block_size, 0], vj_assemble)

# 将 Query 行组装到 qi
pypto.assemble(qi_row, [g_i, 0], qi)

# 将最终输出组装到 atten_out
pypto.assemble(oi_final_4d, [b_idx, n2_idx_start, s1_idx, 0], atten_out)
```

### 3.4 `pypto.cast`

**用途**：数据类型转换，支持 CastMode 配置

**使用示例**：
```python
# FP8 → FP32（反量化中间步骤）
kj_fp32 = pypto.cast(kj_assemble, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
vj_fp32 = pypto.cast(vj_assemble, pypto.DT_FP32, pypto.CastMode.CAST_NONE)

# BF16_scale → FP32（反量化中间步骤）
kj_antiquant_scale_fp32 = pypto.cast(key_antiquant_scale, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
vj_antiquant_scale_fp32 = pypto.cast(value_antiquant_scale, pypto.DT_FP32, pypto.CastMode.CAST_NONE)

# FP32 → BF16（反量化结果）
kj_assemble_antiquanted = pypto.cast(out_data_fp32, pypto.DT_BF16, pypto.CastMode.CAST_NONE)

# FP32 → BF16（Softmax 概率降精度用于后续 matmul）
tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)

# FP32 → BF16（最终输出）
oi_final_4d = pypto.cast(pypto.reshape(oi_final, [1, g_tile, 1, d]), dtype)
```

### 3.5 `pypto.tensor`

**用途**：创建中间张量（Workspace 分配）

**使用示例**：
```python
# 创建 online softmax 中间状态
out_update = pypto.tensor([g_tile, d], pypto.DT_FP32, "out_update")
sum_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "sum_update")
max_update = pypto.tensor([g_tile, 1], pypto.DT_FP32, "max_update")

# 创建组装缓冲区
kj_assemble = pypto.tensor([s2_tile, d], k_2d.dtype, "kj_assemble")
vj_assemble = pypto.tensor([s2_tile, d], v_2d.dtype, "vj_assemble")

# 创建 query 组缓冲区
qi = pypto.tensor([g_tile, d], dtype, "qi")
```

## 4. 数学运算 API

### 4.1 `pypto.matmul`

**用途**：矩阵乘法，核心计算 API

**使用示例**：
```python
# C1: Q × K^T (反量化后的 K)
sij = pypto.matmul(qi, kj_assemble_antiquanted, pypto.DT_FP32, a_trans=False, b_trans=True)

# C2: Softmax_prob × V (反量化后的 V)
oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble_antiquanted, pypto.DT_FP32)
```

### 4.2 向量运算 API

#### 4.2.1 `pypto.mul`

**用途**：元素级乘法，用于反量化缩放和 Softmax 计算

**使用示例**：
```python
# 反量化: FP32_K × FP32_scale
out_data_fp32 = pypto.mul(kj_fp32, kj_antiquant_scale_fp32)

# Softmax 缩放
sij_scale = pypto.mul(sij, softmax_scale)

# Online Softmax 缩放因子计算
sum_update[:] = sum_update * update_mul + sum_local
out_update[:] = out_update * update_mul + oi_tmp
```

#### 4.2.2 `pypto.sub`

**用途**：元素级减法，用于 Softmax 数值稳定性

**使用示例**：
```python
# Softmax: score - max（数值稳定性）
tsub = pypto.sub(sij_scale, tilda_mij)

# Online Softmax: 旧 max - 新 max（缩放因子）
tsub2 = pypto.sub(max_update, max_new)
```

#### 4.2.3 `pypto.div`

**用途**：元素级除法，用于最终归一化

**使用示例**：
```python
# V2: 最终归一化 output / sum
oi_final = pypto.div(out_update, sum_update, precision_type=pypto.PrecisionType.INTRINSIC)
```

### 4.3 数学函数 API

#### 4.3.1 `pypto.exp`

**用途**：指数函数，用于 Softmax 计算

**使用示例**：
```python
# Softmax: exp(score - max)
tilda_pij = pypto.exp(tsub)

# Online Softmax: 缩放因子
update_mul = pypto.exp(tsub2)
```

#### 4.3.2 `pypto.amax`

**用途**：沿指定维度求最大值

**使用示例**：
```python
# Softmax: 局部最大值
tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)
```

#### 4.3.3 `pypto.sum`

**用途**：沿指定维度求和

**使用示例**：
```python
# Softmax: 局部指数和
sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
```

#### 4.3.4 `pypto.maximum`

**用途**：元素级最大值

**使用示例**：
```python
# Online Softmax: 新的最大值
max_new = pypto.maximum(max_update, tilda_mij)
```

## 5. 控制流 API

### 5.1 `pypto.loop`

**用途**：循环控制

**使用示例**：
```python
# 外层循环
for b_idx in pypto.loop(kernel_cfg.b, name="LOOP_b", idx_name="b_idx"):
    for s1_idx in pypto.loop(s1, name="LOOP_s1", idx_name="s1_idx"):
        for n2_idx in pypto.loop(n2, name="LOOP_n2", idx_name="n2_idx"):
            for group_idx in pypto.loop(group_loop, name="LOOP_group_idx", idx_name="group_idx"):
                ...

# 内层循环（支持展开）
for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=[8, 1]):
    ...
```

### 5.2 `pypto.cond`

**用途**：条件判断，用于区分循环首次和后续处理

**使用示例**：
```python
# 区分首 tile 和后续 tile
if pypto.cond(pypto.is_loop_begin(s2_idx)):
    compute_first_tile(...)
else:
    compute_other_tile(...)

# 最后一个 tile 归一化输出
if pypto.cond(pypto.is_loop_end(s2_idx)):
    finalize_output(...)
```

### 5.3 `pypto.is_loop_begin` / `pypto.is_loop_end`

**用途**：判断循环状态

**使用示例**：
```python
for s2_idx in pypto.loop(s2_loop, ...):
    if pypto.cond(pypto.is_loop_begin(s2_idx)):
        # 首 tile: 初始化 softmax 状态
        compute_first_tile(...)
    else:
        # 后续 tile: 更新 softmax 状态
        compute_other_tile(...)
    
    if pypto.cond(pypto.is_loop_end(s2_idx)):
        # 末 tile: 归一化并输出
        finalize_output(...)
```

## 6. 配置 API

### 6.1 `pypto.set_cube_tile_shapes`

**用途**：设置矩阵计算的 Cube Tiling 配置

**使用示例**：
```python
# C1 阶段: Q × K^T
pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
sij = pypto.matmul(qi, kj_assemble_antiquanted, pypto.DT_FP32, a_trans=False, b_trans=True)

# C2 阶段: Softmax × V
pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble_antiquanted, pypto.DT_FP32)
```

### 6.2 `pypto.set_vec_tile_shapes`

**用途**：设置向量计算的 Vec Tiling 配置

**使用示例**：
```python
# 反量化阶段
pypto.set_vec_tile_shapes(128, 128)
kj_fp32 = pypto.cast(kj_assemble, pypto.DT_FP32, pypto.CastMode.CAST_NONE)

# V1 阶段: Softmax
pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
sij = pypto.view(sij, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])

# V2 阶段: 输出更新
pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])

# 多级切分（finalize_output）
pypto.set_vec_tile_shapes(1, g_tile, 1, d)
```

## 7. 数据类型定义

### 7.1 `pypto.Tensor`

**用途**：定义张量类型和形状

**使用示例**：
```python
def incre_flash_attention_gqa_antiquant_kernel(
    query: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    key: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP8E4M3),
    value: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP8E4M3),
    key_antiquant_scale: pypto.Tensor([...], pypto.DT_BF16),
    value_antiquant_scale: pypto.Tensor([...], pypto.DT_BF16),
    kv_actual_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    block_table: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    atten_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
):
    ...
```

### 7.2 支持的数据类型

| 数据类型 | 常量 | 说明 | 使用场景 |
|----------|------|------|----------|
| Bfloat16 | `pypto.DT_BF16` | 16 位浮点数 | query 输入/输出, antiquant_scale |
| Float32 | `pypto.DT_FP32` | 32 位浮点数 | 中间计算 (Softmax, matmul 输出) |
| Float8 E4M3 | `pypto.DT_FP8E4M3` | 8 位浮点数 (E4M3 格式) | KV Cache 存储 |
| Int32 | `pypto.DT_INT32` | 32 位整数 | Block Table, KV Seqs |

## 8. API 使用最佳实践

### 8.1 性能优化最佳实践

1. **合理使用 Tiling 配置**：
   - Cube 操作前使用 `set_cube_tile_shapes`
   - Vec 操作前使用 `set_vec_tile_shapes`
   - Tiling 参数应与硬件特性匹配

2. **减少数据拷贝**：
   - 使用 `reshape(..., inplace=True)` 避免 extra memory
   - 使用 `view` 和 `assemble` 实现非连续访问
   - 避免不必要的 `cast` 操作

3. **优化循环结构**：
   - 使用 `unroll_list` 支持循环展开
   - 合理设置循环名称和索引名
   - 使用 `is_loop_begin/end` 和 `cond` 优化循环状态判断

4. **配置编译器选项**：
   - 使用多缓冲区配置 (`vec_nbuffer_setting`, `cube_l1_reuse_setting`)
   - 使用多核调度 (`device_sched_mode=1`)
   - 使用 `combine_axis=True` 优化轴合并

### 8.2 精度保证最佳实践

1. **反量化精度保证**：
   - FP8 → FP32 → mul(FP32_scale) → FP32 → BF16
   - 反量化在 FP32 下执行乘法，避免精度损失
   - 使用 `CastMode.CAST_NONE` 避免 cast 中的额外处理

2. **Softmax 数值稳定性**：
   - 减去最大值后再计算指数
   - 使用 `amax` 计算局部最大值
   - 使用 `sum` 计算局部指数和
   - 所有 Softmax 中间计算使用 FP32

3. **类型转换时机**：
   - 输入：BF16 (query), FP8 (K/V)
   - 反量化后：BF16 (用于 matmul 输入)
   - matmul 输出：FP32
   - Softmax：FP32
   - 最终输出：BF16

### 8.3 内存管理最佳实践

1. **Workspace 分配**：
   - 使用 `pypto.tensor` 分配中间张量
   - 形状应与 Tiling 配置匹配
   - online softmax 状态张量使用 FP32

2. **内存复用**：
   - 循环内中间张量复用 (out_update, sum_update, max_update)
   - 使用 inplace reshape

3. **非连续内存访问**：
   - 使用 `view` 和 `assemble` 实现非连续访问
   - 支持 Paged KV Cache
   - 合理设置 Block Table

## 9. API 使用注意事项

### 9.1 约束条件

1. **形状约束**：
   - reshape 前后总元素数量一致
   - view 形状不能超过原始张量范围
   - assemble 目标位置不能超出目标张量

2. **数据类型约束**：
   - matmul 输出类型必须显式指定
   - FP8 类型仅用于 KV Cache 存储，计算前必须反量化
   - `CastMode.CAST_NONE` 用于无特殊处理的 cast

3. **计算顺序约束**：
   - `set_cube_tile_shapes` 必须在 `matmul` 前调用
   - `set_vec_tile_shapes` 必须在 Vec 操作前调用
   - 反量化必须在 matmul 前完成

## 10. API 版本和兼容性

### 10.1 兼容性说明

| API | 稳定性 | 兼容性 | 说明 |
|-----|--------|--------|------|
| 核心计算 API (matmul, mul, etc.) | 稳定 | 高 | 基础 API，长期支持 |
| 数据操作 API (reshape, view, etc.) | 稳定 | 高 | 基础 API，长期支持 |
| 控制流 API (loop, cond, is_loop_begin/end) | 稳定 | 高 | 基础 API，长期支持 |
| 配置 API (set_tile_shapes, etc.) | 稳定 | 中 | 配置 API，参数可能有调整 |
| 编译器配置 (jit) | 稳定 | 中 | 配置 API，参数可能有调整 |
| 实验性 API (experimental) | 实验性 | 低 | 实验性 API，可能变更 |

## 11. 附录

### 11.1 API 快速参考表

| API | 类别 | 主要用途 | 关键参数 |
|-----|------|----------|----------|
| `pypto.frontend.jit` | 编译器配置 | Kernel 配置 | pass_options, runtime_options |
| `pypto.reshape` | 数据操作 | 改变形状 | shape, inplace |
| `pypto.view` | 数据操作 | 创建视图 | shape, offset, valid_shape |
| `pypto.assemble` | 数据操作 | 数据组装 | src, position, dst |
| `pypto.cast` | 数据操作 | 类型转换 | dtype, cast_mode |
| `pypto.tensor` | 数据操作 | 分配张量 | shape, dtype, name |
| `pypto.matmul` | 数学运算 | 矩阵乘法 | a, b, dtype, a_trans, b_trans |
| `pypto.mul` | 数学运算 | 元素乘法 | a, b |
| `pypto.sub` | 数学运算 | 元素减法 | a, b |
| `pypto.div` | 数学运算 | 元素除法 | a, b, precision_type |
| `pypto.exp` | 数学运算 | 指数函数 | tensor |
| `pypto.amax` | 数学运算 | 最大值 | tensor, dim, keepdim |
| `pypto.sum` | 数学运算 | 求和 | tensor, dim, keepdim |
| `pypto.maximum` | 数学运算 | 元素最大值 | a, b |
| `pypto.loop` | 控制流 | 循环控制 | count, name, idx_name, unroll_list |
| `pypto.cond` | 控制流 | 条件判断 | condition |
| `pypto.is_loop_begin` | 控制流 | 循环状态判断 | loop_idx |
| `pypto.is_loop_end` | 控制流 | 循环状态判断 | loop_idx |
| `pypto.set_cube_tile_shapes` | 配置 | Cube Tiling | tile_m, tile_k, tile_n |
| `pypto.set_vec_tile_shapes` | 配置 | Vec Tiling | tile_0, tile_1, tile_2, tile_3 |
| `pypto.experimental.set_operation_options` | 实验性 | 操作选项 | combine_axis |

### 11.2 反量化流程详解

```
FP8 KV Cache (key/value) ─── pypto.cast ───→ FP32
                                              │
BF16 antiquant_scale ─── pypto.cast ───→ FP32
                                              │
                              pypto.mul ───→ FP32 (dequantized)
                                              │
                              pypto.cast ───→ BF16 (for matmul input)
```