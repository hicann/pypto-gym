# Incre Flash Attention GQA Anti-Quant 设计文档

## 1. 设计概述

### 1.1 设计目标

本算子设计目标为实现高效的增量式 GQA (Grouped Query Attention) Incre Flash Attention，支持 KV Cache 的反量化计算，用于大语言模型自回归生成场景。核心设计要点：

- **增量式计算**：每次只计算新增 token，支持流式生成
- **Paged KV Cache**：支持非连续内存访问，提高内存利用率
- **在线 Softmax**：避免存储完整注意力矩阵，降低内存开销
- **分组查询注意力**：支持 GQA，减少计算和存储开销
- **反量化支持**：KV Cache 以 FP8 存储，通过反量化还原为 BF16 进行计算
- **高性能实现**：通过 Tiling、流水线和并行优化提升性能

## 2. 数据流设计

### 2.1 输入数据准备

#### 2.1.1 Query 数据

```
query: [b, n1, s1, d]  (BNSD 布局)

reshape 为 2D:
q_2d: [b * n1 * s1, d]
```

#### 2.1.2 Key/Value Cache 数据

```
key_cache:   [block_num, n2, block_size, d]  (PA_BnNBsD 格式, FP8)
value_cache: [block_num, n2, block_size, d]  (PA_BnNBsD 格式, FP8)

reshape 为 2D:
k_2d: [block_num * block_size * n2, d]
v_2d: [block_num * block_size * n2, d]
```

#### 2.1.3 反量化 Scale 数据

```
key_antiquant_scale:   [n2, d]  (BF16, 每个 KV head 独立缩放)
value_antiquant_scale: [n2, d]  (BF16, 每个 KV head 独立缩放)
```

#### 2.1.4 Block Table

```
block_table: [b, max_blocks_per_query]

映射逻辑块索引 -> 物理块索引
-1 表示无效块，计算时使用 .max(0) 处理
```

### 2.2 计算流程

```
┌───────────────────────────────────────────────────────────────┐
│  Input: query, key(FP8), value(FP8),                         │
│         key_antiquant_scale, value_antiquant_scale,           │
│         block_table, kv_actual_seqs                           │
└───────────────────────────────────────────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────────────┐
│  init_kernel_cfg + reshape_qkv_to_2d:                         │
│  - q_2d: [b*n1*s1, d]                                        │
│  - k_2d: [block_num*block_size*n2, d]                         │
│  - v_2d: [block_num*block_size*n2, d]                         │
└───────────────────────────────────────────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────────────┐
│  Loop Structure:                                              │
│  batch_size → s1 → n2 → group → s2_tiles                     │
└───────────────────────────────────────────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────────────┐
│  For each s2_tile:                                            │
│  [Assemble KV → Anti-Quant K → C1 → V1 → Anti-Quant V → C2 → V2] │
└───────────────────────────────────────────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────────────┐
│  Output: atten_out [b, n1, s1, d] (BNSD 布局)                 │
└───────────────────────────────────────────────────────────────┘
```

## 3. Tiling 设计

### 3.1 Tiling 策略

#### 3.1.1 Group Tiling

```python
group = n1 // n2  # 每个KV头对应的query头数
g_tile = group    # 当前实现中 g_tile = group
group_loop = group // g_tile  # 通常为 1
```

**设计理由**：
- 当前实现将整个 group 作为一次处理，简化循环结构
- g_tile = group，group_loop = 1（单一循环）
- 未来可优化为更小的 g_tile，降低 L1/L0 缓存压力

#### 3.1.2 Sequence Tiling

```python
s2_tile = 2048  # KV序列分块大小
s2_loop = ceildiv(cur_seq_len, s2_tile)  # 动态计算
```

**设计理由**：
- 支持长序列 (2048, 4096, 8192, 16384)
- 优化 Block 访问效率 (block_size=128, s2_tile=2048 -> 16 blocks/tile)
- 降低单次计算内存需求

#### 3.1.3 Vector/Cube Tiling

```python
c1_tile = [[128, 128], [128, 128], [128, 128]]  # C1: Q×K^T
v1_tile = [128, 2048]               # V1: Softmax
c2_tile = [[128, 128], [128, 128], [128, 128]]  # C2: Softmax×V
v2_tile = [128, 128]                # V2: 输出更新
```

**设计理由**：
- 匹配 AICore 向量和矩阵计算单元特性
- 优化 L1/L0 缓存利用
- 支持流水线并行

### 3.2 Tiling 参数调优建议

| 场景 | 建议配置 |
|------|---------|
| 小批次短序列 (b=1, s2=2048) | g_tile=group, s2_tile=2048 |
| 中等批次 (b=8-16, s2=2048-4096) | g_tile=group, s2_tile=2048 |
| 大批次长序列 (b=64, s2=16384) | g_tile=group, s2_tile=2048 |
| d=256 场景 | g_tile=group, s2_tile=2048, 调整 cube tile |

## 4. 核心计算单元设计

### 4.1 KV 组装阶段 (Assemble KV)

**功能**：从 Paged Cache 中组装 Key 和 Value，支持反量化

**实现细节**：
```python
# 创建组装缓冲区
kj_assemble = pypto.tensor([s2_tile, d], k_2d.dtype, "kj_assemble")  # FP8
vj_assemble = pypto.tensor([s2_tile, d], v_2d.dtype, "vj_assemble")  # FP8

block_num = s2_tile // block_size  # 16 blocks/tile

# 遍历块并组装
for i in range(block_num):
    block_idx = block_table[b_idx, idx + i]
    block_idx_valid = block_idx.max(0)  # 处理无效块 (-1)
    
    kj_view = pypto.view(k_2d, [block_size, d], [(block_idx_valid * n2 + n2_idx) * block_size, 0])
    vj_view = pypto.view(v_2d, [block_size, d], [(block_idx_valid * n2 + n2_idx) * block_size, 0])
    
    pypto.assemble(kj_view, [i * block_size, 0], kj_assemble)
    pypto.assemble(vj_view, [i * block_size, 0], vj_assemble)

# 设置 valid_shape（动态序列长度）
kj_assemble = pypto.view(kj_assemble, [s2_tile, d], [0, 0], valid_shape=[s2_tile, d])
vj_assemble = pypto.view(vj_assemble, [s2_tile, d], [0, 0], valid_shape=[actual_s2_tile, d])
```

**关键点**：
- 组装缓冲区 dtype 为 FP8（与 KV Cache 一致）
- 使用 `pypto.view` 和 `pypto.assemble` 实现非连续内存访问
- `block_idx_valid = block_idx.max(0)` 处理无效块 (-1)
- V 的 valid_shape 使用 `actual_s2_tile`（动态序列长度）
- K 的 valid_shape 使用完整 `s2_tile`（后续 view 会重新设置）

### 4.2 Key 反量化阶段

**功能**：将 FP8 Key 通过缩放因子还原为 BF16

**实现细节**：
```python
pypto.set_vec_tile_shapes(128, 128)

# 获取当前 KV head 的 scale
key_antiquant_scale = ctx_params.loop_tensors.key_antiquant_scale[n2_idx]

# FP8 → FP32
kj_fp32 = pypto.cast(kj_assemble, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
kj_antiquant_scale_fp32 = pypto.cast(key_antiquant_scale, pypto.DT_FP32, pypto.CastMode.CAST_NONE)

# FP32 × FP32_scale → FP32 (反量化)
out_data_fp32 = pypto.mul(kj_fp32, kj_antiquant_scale_fp32)

# FP32 → BF16 (用于 matmul 输入)
kj_assemble_antiquanted = pypto.cast(out_data_fp32, pypto.DT_BF16, pypto.CastMode.CAST_NONE)
```

**关键点**：
- 反量化流程：FP8 → FP32 → mul(FP32_scale) → FP32 → BF16
- 缩放因子按 KV head 索引取值：`key_antiquant_scale[n2_idx]`
- 所有中间乘法在 FP32 下执行，保证精度
- 最终输出为 BF16，作为 matmul 的输入

### 4.3 C1 阶段: Q × K^T

**功能**：计算 Query 与反量化后 Key 的注意力分数

**实现细节**：
```python
# 组装 Query (在 group loop 外一次性完成)
qi = pypto.tensor([g_tile, d], dtype, "qi")
for g_i in range(g_tile):
    qi_row_ofs = b_idx * n1 * s1 + (n1g_ofs + g_i) * s1 + s1_idx
    qi_row = pypto.view(q_2d, [1, d], [qi_row_ofs, 0])
    pypto.assemble(qi_row, [g_i, 0], qi)

# 计算 Q × K^T (反量化后的 K)
pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
sij = pypto.matmul(qi, kj_assemble_antiquanted, pypto.DT_FP32, a_trans=False, b_trans=True)
# 结果形状: [g_tile, s2_tile]

# 设置 valid_shape
pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
sij = pypto.view(sij, [g_tile, s2_tile], [0, 0], valid_shape=[g_tile, actual_s2_tile])
```

**关键点**：
- Query 组装在 s2 循环外完成（每个 s2_tile 复用同一个 qi）
- Query 按 BNSD 布局索引：`b_idx * n1 * s1 + (n1g_ofs + g_i) * s1 + s1_idx`
- K 需先反量化再 matmul
- matmul 使用 `b_trans=True`（K 转置）
- valid_shape 处理动态序列长度

### 4.4 V1 阶段: Online Softmax

**功能**：计算缩放、局部最大值和指数和

**首次循环 (compute_first_tile)**：

```python
# 缩放
sij_scale = pypto.mul(sij, softmax_scale)

# 局部最大值
tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

# 计算 exp(score - max)
tsub = pypto.sub(sij_scale, tilda_mij)
tilda_pij = pypto.exp(tsub)

# 转 BF16 用于后续 matmul
tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)

# 初始化 sum 和 max
sum_update[:] = pypto.sum(tilda_pij, dim=-1, keepdim=True)
max_update[:] = tilda_mij
```

**后续循环 (compute_other_tile)**：

```python
# 缩放
sij_scale = pypto.mul(sij, softmax_scale)

# 局部最大值
tilda_mij = pypto.amax(sij_scale, dim=-1, keepdim=True)

# 更新全局最大值
max_new = pypto.maximum(max_update, tilda_mij)

# 计算 exp(score - max_new)
tsub = pypto.sub(sij_scale, max_new)
tilda_pij = pypto.exp(tsub)
tilda_pij_fp16 = pypto.cast(tilda_pij, dtype)

# 计算局部 sum
sum_local = pypto.sum(tilda_pij, dim=-1, keepdim=True)

# 计算缩放因子
tsub2 = pypto.sub(max_update, max_new)
max_update[:] = max_new
update_mul = pypto.exp(tsub2)

# 更新 sum
sum_update[:] = sum_update * update_mul + sum_local
```

**关键点**：
- 在线 Softmax 算法，避免存储完整注意力矩阵
- 首次循环直接初始化 max 和 sum
- 后续循环使用 Welford 算法更新
- 所有中间计算使用 FP32 保证数值稳定性

### 4.5 Value 反量化阶段

**功能**：将 FP8 Value 通过缩放因子还原为 BF16

**实现细节**：
```python
pypto.set_vec_tile_shapes(128, 128)

# 获取当前 KV head 的 scale
value_antiquant_scale = ctx_params.loop_tensors.value_antiquant_scale[n2_idx]

# FP8 → FP32
vj_fp32 = pypto.cast(vj_assemble, pypto.DT_FP32, pypto.CastMode.CAST_NONE)
vj_antiquant_scale_fp32 = pypto.cast(value_antiquant_scale, pypto.DT_FP32, pypto.CastMode.CAST_NONE)

# FP32 × FP32_scale → FP32
vj_out_data_fp32 = pypto.mul(vj_fp32, vj_antiquant_scale_fp32)

# FP32 → BF16
vj_assemble_antiquanted = pypto.cast(vj_out_data_fp32, pypto.DT_BF16, pypto.CastMode.CAST_NONE)
```

**关键点**：
- Value 反量化流程与 Key 完全一致
- 在首 tile 和后续 tile 中都需要执行
- 缩放因子按 KV head 索引取值

### 4.6 C2 阶段: Softmax × V

**功能**：计算 Softmax 概率与反量化后 Value 的加权聚合

**实现细节**：
```python
pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
oi_tmp = pypto.matmul(tilda_pij_fp16, vj_assemble_antiquanted, pypto.DT_FP32)
# 结果形状: [g_tile, d]
```

**关键点**：
- 输入为 BF16（Softmax 概率和反量化后的 V）
- 输出为 FP32
- V 需先反量化再 matmul

### 4.7 V2 阶段: Output Update

**功能**：使用在线 Softmax 算法更新输出

**首次循环 (compute_first_tile)**：
```python
# V 反量化后 matmul 结果直接作为初始输出
pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
out_update[:] = oi_tmp
```

**后续循环 (compute_other_tile)**：
```python
# 更新输出: old_output × scale + new_output
pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
out_update[:] = out_update * update_mul + oi_tmp
```

**关键点**：
- 首次循环直接存储 matmul 结果
- 后续循环使用缩放因子更新：`out_update * update_mul + oi_tmp`
- `update_mul = exp(max_update_old - max_update_new)` 是在线 Softmax 的核心

### 4.8 Finalize Output

**功能**：归一化并写入最终输出

**实现细节**：
```python
# 归一化: output / sum
oi_final = pypto.div(out_update, sum_update, precision_type=pypto.PrecisionType.INTRINSIC)

# reshape 并 cast 到输出格式 (BNSD)
pypto.set_vec_tile_shapes(1, g_tile, 1, d)
oi_final_4d = pypto.cast(pypto.reshape(oi_final, [1, g_tile, 1, d]), dtype)

# 组装到 atten_out (BNSD 布局)
n2_idx_start = n2_idx * group
out_ofs = [b_idx, n2_idx_start, s1_idx, 0]
pypto.assemble(oi_final_4d, out_ofs, atten_out)
```

**关键点**：
- 最终归一化使用 `pypto.PrecisionType.INTRINSIC`
- 输出按 BNSD 布局组装
- offset 计算：`n2_idx_start = n2_idx * group`（GQA 中 query head 与 KV head 的映射）

## 5. Loop 结构设计

### 5.1 外层循环结构

```python
for b_idx in pypto.loop(kernel_cfg.b, name="LOOP_b", idx_name="b_idx"):
    for s1_idx in pypto.loop(s1, name="LOOP_s1", idx_name="s1_idx"):
        cur_seq_len = kv_act_seqs[b_idx] - (s1 - 1 - s1_idx)
        s2_loop = pypto.ceildiv(cur_seq_len, s2_tile)
        
        for n2_idx in pypto.loop(n2, name="LOOP_n2", idx_name="n2_idx"):
            for group_idx in pypto.loop(group_loop, name="LOOP_group_idx", idx_name="group_idx"):
                # 组装 qi (一次性)
                # 初始化 online softmax 状态
                for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=[8, 1]):
                    [Assemble KV → Anti-Quant K → C1 → V1 → Anti-Quant V → C2 → V2]
                finalize_output()
```

**循环设计理由**：
- **Batch Loop**：批次间无依赖，可并行优化
- **S1 Loop**：支持多 query tokens（增量生成通常 s1=1）
- **N2 Loop**：KV heads 间无依赖，可并行优化
- **Group Loop**：query head 分组，当前 group_loop=1
- **S2 Loop**：KV 序列分块，核心计算循环，支持展开

### 5.2 内层循环 (S2 Loop)

```python
for s2_idx in pypto.loop(s2_loop, name="LOOP_s2", idx_name="s2_idx", unroll_list=[8, 1]):
    if pypto.cond(pypto.is_loop_begin(s2_idx)):
        compute_first_tile(...)
    else:
        compute_other_tile(...)
    if pypto.cond(pypto.is_loop_end(s2_idx)):
        finalize_output(...)
```

**关键设计**：
- `unroll_list=[8, 1]`：支持 8 倍和 1 倍展开优化
- 首次循环初始化 softmax 状态
- 后续循环更新 softmax 状态
- 最后循环归一化并输出

### 5.3 循环优化建议

| 循环层级 | 优化策略 |
|---------|---------|
| Batch Loop | 多核并行（device_sched_mode=1） |
| S1 Loop | 通常 s1=1，无优化需求 |
| N2 Loop | 多核并行（device_sched_mode=1） |
| Group Loop | 当前 group_loop=1，无优化需求 |
| S2 Loop | 循环展开、流水线优化 |

## 6. 内存管理设计

### 6.1 Workspace 内存分配

```python
# Online softmax 中间状态 (FP32)
out_update: [g_tile, d]   # FP32
sum_update: [g_tile, 1]   # FP32
max_update: [g_tile, 1]   # FP32

# 组装缓冲区 (FP8)
kj_assemble: [s2_tile, d]  # FP8
vj_assemble: [s2_tile, d]  # FP8

# Query 组缓冲区 (BF16)
qi: [g_tile, d]  # BF16

# 中间计算结果 (FP32)
sij: [g_tile, s2_tile]               # FP32 (matmul 输出)
oi_tmp: [g_tile, d]                  # FP32 (matmul 输出)
```

### 6.2 内存优化策略

1. **In-place Reshape**：使用 `pypto.reshape(..., inplace=True)` 避免额外内存分配
2. **Buffer Reuse**：循环内中间缓冲区复用 (out_update, sum_update, max_update)
3. **Paged Cache**：非连续内存访问，提高利用率
4. **FP8 KV Cache**：KV Cache 以 FP8 存储，内存占用为 BF16 的一半
5. **反量化即时计算**：不存储反量化后的完整 K/V，按 tile 即时反量化

### 6.3 L1/L0 缓存优化

```python
pypto.frontend.jit(
    pass_options={
        "cube_l1_reuse_setting": {0: 16},
        "vec_nbuffer_setting": {0: 16},
    },
    runtime_options={
        "stitch_function_max_num": 256,
        "device_sched_mode": 1
    }
)
```

**配置说明**：
- `cube_l1_reuse_setting`: 矩阵计算 L1 缓存 16 级复用
- `vec_nbuffer_setting`: 向量操作 16 级缓冲
- `stitch_function_max_num`: 最大拼接函数数量 256
- `device_sched_mode=1`: 设备调度模式

## 7. API 映射

### 7.1 PyPTO API 使用

| PyPTO API | 使用场景 | 说明 |
|-----------|----------|------|
| `pypto.reshape` | 数据预处理 | inplace=True 避免额外内存 |
| `pypto.view` | 非连续内存访问 | 从 Paged Cache 提取数据，支持 valid_shape |
| `pypto.assemble` | 数据组装 | 将分散数据组装到连续缓冲区 |
| `pypto.cast` | 反量化类型转换 | FP8→FP32, FP32→BF16 |
| `pypto.matmul` | 矩阵乘法 | Cube 单元计算（C1, C2） |
| `pypto.mul` | 反量化缩放 + Softmax | Vec 单元计算 |
| `pypto.sub` | Softmax 稳定性 | score - max |
| `pypto.div` | 最终归一化 | output / sum |
| `pypto.exp` | Softmax 指数 | Vec 单元计算 |
| `pypto.amax/sum/maximum` | Online Softmax | 最大值、求和 |
| `pypto.loop/cond/is_loop_begin/end` | 循环控制 | 支持展开和状态判断 |
| `pypto.tensor` | 中间张量分配 | Workspace 管理 |
| `pypto.set_cube_tile_shapes` | Cube Tiling | 矩阵计算切分配置 |
| `pypto.set_vec_tile_shapes` | Vec Tiling | 向量计算切分配置 |
| `pypto.ceildiv` | 动态循环次数 | 计算序列分块数 |
| `pypto.experimental.set_operation_options` | 操作优化 | combine_axis |

### 7.2 数据类型路由

| 计算阶段 | 数据类型 | 说明 |
|----------|----------|------|
| KV Cache 存储 | FP8 (E4M3) | 内存节省 |
| 反量化中间计算 | FP32 | 精度保证 |
| 反量化输出 | BF16 | matmul 输入 |
| Query 输入/输出 | BF16 | BNSD 布局 |
| Q×K^T 输出 | FP32 | 精度保证 |
| Softmax 中间 | FP32 | 数值稳定性 |
| Softmax × V 输入 | BF16 | matmul 输入 |
| Softmax × V 输出 | FP32 | 精度保证 |
| 最终归一化 | FP32 → BF16 | 输出降精度 |

## 8. 性能优化设计

### 8.1 计算优化

1. **Cube 单元利用**：
   - C1 (Q×K_dequant^T) 和 C2 (Softmax×V_dequant) 使用 Cube 单元
   - Tiling 配置优化 Cube 利用率
   - L1 缓存 16 级复用

2. **Vec 单元利用**：
   - 反量化 (cast + mul)、V1 (softmax)、V2 (update) 使用 Vec 单元
   - 16 级缓冲减少等待

3. **流水线优化**：
   - Cube 和 Vec 单元并行执行
   - 多缓冲区配置支持流水线

### 8.2 内存访问优化

1. **Paged Cache**：
   - 非连续内存访问，减少内存碎片
   - Block Table 映射，提高利用率

2. **FP8 KV Cache**：
   - 内存占用为 BF16 的一半
   - 反量化按 tile 即时执行，不增加全局存储

3. **Tiling 策略**：
   - Sequence Tiling 优化 Block 访问 (s2_tile/block_size = 16 blocks/tile)
   - Query 组装在 s2 循环外完成（复用 qi）

### 8.3 并行优化

```python
# 设备调度
device_sched_mode = 1

# 循环展开
unroll_list = [8, 1]

# Combine Axis 优化
pypto.experimental.set_operation_options(combine_axis=True)
```

## 9. 调试与验证设计

### 9.1 精度验证

```python
compare(pypto_atten_out.cpu(), gqa_antiquant_golden.cpu(), "pypto_atten_out",
        atol=0.0001, rtol=0.0078125, max_error_ratio=0.005)
```

**验证标准**：
- 绝对容差 (atol): 0.0001
- 相对容差 (rtol): 0.0078125 (1/128)
- 最大误差比例: 0.5%

### 9.2 Golden 实现

```python
def ifa_gqa_antiquant_golden(query, key, value, key_antiquant_scale, value_antiquant_scale):
    # 反量化
    key_dequant = key * key_antiquant_scale.reshape(1, n2, 1, d)
    value_dequant = value * value_antiquant_scale.reshape(1, n2, 1, d)
    
    # GQA 扩展
    key_expanded = key_dequant.repeat_interleave(group, dim=1)
    value_expanded = value_dequant.repeat_interleave(group, dim=1)
    
    # 标准 Flash Attention
    scores = torch.matmul(query, key_expanded.transpose(-2, -1)) * softmax_scale
    attention_weights = F.softmax(scores, dim=-1)
    attention_out = torch.matmul(attention_weights, value_expanded)
    return attention_out
```

## 10. 扩展性设计

### 10.1 支持的扩展场景

1. **不同头数量配置**：
   - 支持任意 n1/n2 组合（需满足 n1 % n2 == 0）
   - 通过 Group Tiling 自适应

2. **不同序列长度**：
   - 支持动态 KV 序列长度 (kv_actual_seqs)
   - 自适应 Tiling 配置

3. **不同头维度**：
   - 支持 d=128 和 d=256

4. **不同量化格式**：
   - 当前支持 FP8 E4M3
   - 未来可扩展支持其他量化格式

### 10.2 可配置参数

- **IFAKernelCfg**：算子配置参数（从输入张量自动推导）
- **AttentionTileConfig**：Tiling 配置参数
- **Pass Options**：编译优化选项
- **Runtime Options**：运行时调度选项

## 11. 设计权衡与决策

### 11.1 性能 vs 精度

- **选择**：KV Cache 使用 FP8 存储，反量化后 FP32 计算，输出 BF16
- **权衡**：FP8 存储节省内存，但增加反量化计算开销
- **理由**：推理场景内存是瓶颈，FP8 KV Cache 内存减半收益大于反量化开销

### 11.2 灵活性 vs 优化

- **选择**：提供 Tiling 配置参数
- **权衡**：增加配置复杂度换取性能调优空间
- **理由**：不同场景需要不同 Tiling 策略

### 11.3 通用性 vs 专用性

- **选择**：专注于 GQA 增量推理场景，支持反量化
- **权衡**：牺牲通用性换取场景优化
- **理由**：自回归生成是高频场景，专用优化收益大