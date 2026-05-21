# Incre Flash Attention MLA 设计文档

## 1. 设计概述

### 1.1 设计目标

本算子设计目标为实现高效的增量式 MLA (Multi-Head Latent Attention) Incre Flash Attention，用于大语言模型自回归生成场景。核心设计要点：

- **增量式计算**：每次只计算新增 token，支持流式生成
- **Paged KV Cache**：支持非连续内存访问，提高内存利用率
- **在线 Softmax**：避免存储完整注意力矩阵，降低内存开销
- **分组查询注意力**：支持 GQA，减少计算和存储开销
- **高性能实现**：通过 Tiling、流水线和并行优化提升性能

## 2. 数据流设计

### 2.1 输入数据准备

#### 2.1.1 Query 数据

```
query: [batch_size, n1, s1, q_d]  (BNSD布局)
query_rope: [batch_size, n1, s1, q_rope_d]

拼接后: query_full = [query, query_rope]  维度: q_d + q_rope_d
```

#### 2.1.2 Key/Value Cache 数据

```
key_cache: [block_num, n2, block_size, kv_d]  (PA_BnNBsD格式)
value_cache: [block_num, n2, block_size, kv_d]
key_rope_cache: [block_num, n2, block_size, k_rope_d]

通过 block_table 重构:
key_full = [key_cache (via block_table), key_rope_cache (via block_table)]
维度: kv_d + k_rope_d
```

#### 2.1.3 Block Table

```
block_table: [batch_size, max_blocks_per_query]

映射逻辑块索引 -> 物理块索引
-1 表示无效块
```

### 2.2 计算流程

```
┌─────────────────────────────────────────────────────────┐
│  Input: query, query_rope, key_cache, key_rope_cache    │
│         value_cache, block_table, kv_actual_seqs        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  reshape_qkv_to_2d:                                      │
│  - query_2d: [b*n1*s1, q_d]                              │
│  - query_rope_2d: [b*n1*s1, q_rope_d]                    │
│  - key_2d: [block_num*block_size*n2, kv_d]               │
│  - key_rope_2d: [block_num*block_size*n2, k_rope_d]      │
│  - value_2d: [block_num*block_size*n2, kv_d]             │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Loop Structure:                                         │
│  batch_size -> s1 -> n2 -> group -> s2_tiles             │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  For each s2_tile:                                       │
│  [V0 -> C1 -> V1 -> C2 -> V2]                            │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Output: attention_output [batch_size, n1, s1, q_d]      │
└─────────────────────────────────────────────────────────┘
```

## 3. Tiling 设计

### 3.1 Tiling 策略

#### 3.1.1 Group Tiling

```python
group = n1 // n2  # 每个KV头对应的query头数
group_tile = 128  # 每次处理128个query头
group_loop = group // group_tile  # 组循环次数
```

**设计理由**：
- 降低 L1/L0 缓存压力
- 支持大规模 query heads (128)
- 平衡计算和内存访问

#### 3.1.2 Sequence Tiling

```python
s2_tile = 2048  # KV序列分块大小
s2_loop = (current_seq + s2_tile - 1) // s2_tile
```

**设计理由**：
- 支持长序列 (4096, 8192)
- 优化 Block 访问效率 (block_size=128, s2_tile=2048 -> 16 blocks/tile)
- 降低单次计算内存需求

#### 3.1.3 Vector/Cube Tiling

```python
v0_tile = [128, 576]    # V0阶段组装计算
c1_tile = [[128, 128], [128, 128], [128, 128]]  # C1阶段Q×K^T
v1_tile = [8, 2048]     # V1阶段Softmax
c2_tile = [[128, 128], [128, 128], [128, 128]]  # C2阶段Softmax×V
v2_tile = [64, 512]     # V2阶段输出更新
v2_update_tile = [32, 512]  # V2更新阶段
```

**设计理由**：
- 匹配 AICore 向量和矩阵计算单元特性
- 优化 L1/L0 缓存利用
- 支持流水线并行

### 3.2 Tiling 参数调优建议

| 场景 | 建议配置 |
|------|---------|
| 小批次长序列 (b=1, s2=4096) | group_tile=128, s2_tile=2048 |
| 中等批次 (b=8-16, s2=4096-8192) | group_tile=128, s2_tile=2048 |
| 大批次短序列 (b=32, s2=2048) | group_tile=128, s2_tile=2048 |
| 大批次长序列 (b=32, s2=4096) | group_tile=128, s2_tile=2048 |

## 4. 核心计算单元设计

### 4.1 V0 阶段 (MLA_V0): Key Assemble

**功能**：从 Paged Cache 中组装 Key 和 Key_Rope

**实现细节**：
```python
# 创建组装缓冲区
key_assemble: [s2_tile, query_dim + rope_dim]
key_nope_assemble: [s2_tile, query_dim]
key_rope_assemble: [s2_tile, rope_dim]

# 遍历块并组装
block_num_per_tile = s2_tile // block_size  # 16 blocks/tile
for block_i in range(block_num_per_tile):
    block_index = block_table[batch_idx, block_base_index + block_i]
    valid_block_index = block_index.max(0)
    block_start_offset = valid_block_index * block_size
    
    # 从 key_2d 和 key_rope_2d 中提取数据
    key_nope_view = pypto.view(key_2d, [block_size, query_dim], [block_start_offset, 0])
    key_rope_view = pypto.view(key_rope_2d, [block_size, rope_dim], [block_start_offset, 0])
    
    # 组装到缓冲区
    pypto.assemble(key_nope_view, [block_i*block_size, 0], key_nope_assemble)
    pypto.assemble(key_rope_view, [block_i*block_size, 0], key_rope_assemble)
    pypto.assemble(key_nope_view, [block_i*block_size, 0], key_assemble)
    pypto.assemble(key_rope_view, [block_i*block_size, query_dim], key_assemble)
```

**关键点**：
- 使用 `pypto.view` 和 `pypto.assemble` 实现非连续内存访问
- `valid_block_index.max(0)` 处理无效块（-1）
- 同时组装完整 key 和分部分 key（优化后续 C2 计算）

### 4.2 C1 阶段 (MLA_C1): Query-K Attention Score

**功能**：计算 Q × K^T

**实现细节**：
```python
# 组装 Query
query_assemble: [group_tile, query_dim + rope_dim]
query_no_rope_view = pypto.view(query_2d, [group_tile, query_dim], [batch_seq_offset*query_heads + query_head_group_offset, 0])
query_rope_view = pypto.view(query_rope_2d, [group_tile, rope_dim], [batch_seq_offset*query_heads + query_head_group_offset, 0])
pypto.assemble(query_no_rope_view, [0, 0], query_assemble)
pypto.assemble(query_rope_view, [0, query_dim], query_assemble)

# 计算 Q × K^T
attention_score = pypto.matmul(query_assemble, key_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)
# 结果形状: [group_tile, s2_tile]
```

**关键点**：
- Query 只需要组装一次（每个 s2_tile 循环复用）
- 使用 FP32 保证精度
- 矩阵乘法采用 Cube 单元

### 4.3 V1 阶段 (MLA_V1): Online Softmax

**功能**：计算缩放、局部最大值和指数和

**实现细节**：
```python
# 缩放
scaled_attention_score = pypto.mul(attention_score, softmax_scale)

# 局部最大值
tile_max_score_reduce = pypto.amax(scaled_attention_score, dim=-1, keepdim=True)
tile_max_score = pypto.reshape(tile_max_score_reduce, [1, group_tile])

# 计算指数
score_diff = pypto.sub(scaled_attention_score, tile_max_score_reduce)
tile_attention_prob = pypto.exp(score_diff)

# 指数和
tile_logsumexp_reduce = pypto.sum(tile_attention_prob, dim=-1, keepdim=True)
tile_logsumexp = pypto.reshape(tile_logsumexp_reduce, [1, group_tile])

# 转换回 BF16 用于后续矩阵乘法
tile_attention_prob_fp16 = pypto.cast(tile_attention_prob, dtype)
```

**关键点**：
- 在线 Softmax 算法，避免存储完整注意力矩阵
- 计算 tile 内的 max 和 sum，用于后续跨 tile 更新
- 数值稳定性：减去最大值后再计算指数

### 4.4 C2 阶段 (MLA_C2): Weighted Value Aggregation

**功能**：计算 Softmax × V

**实现细节**：
```python
# 设置矩阵尺寸
pypto.set_matrix_size([group_tile, s2_tile, query_dim])

# 计算 Softmax × V
weighted_value_intermediate = pypto.matmul(tile_attention_prob_fp16, key_nope_assemble, pypto.DT_FP32)
# 结果形状: [group_tile, query_dim]
```

**关键点**：
- 使用 `key_nope_assemble` 而非完整 key（优化点）
- 输出为 FP32 精度
- 矩阵尺寸设置优化 Cube 单元利用率

### 4.5 V2 阶段 (MLA_V2): Output Update

**功能**：使用在线 Softmax 算法更新输出

**实现细节**：

**首次循环 (loop_begin)**：
```python
if pypto.is_loop_begin(s2_idx):
    output_tmp = weighted_value_intermediate
    
    if pypto.is_loop_end(s2_idx):
        # 只有一个 tile，直接归一化输出
        out_update[:] = output_tmp / tile_logsumexp_reduce
        # 组装到最终输出
        output_4_dim = pypto.cast(pypto.reshape(out_update, [1, group_tile, 1, query_dim]), dtype)
        pypto.assemble(output_4_dim, output_offset, attention_output)
    else:
        # 多个 tile，存储中间结果
        out_update[:] = output_tmp
        sum_update[:] = tile_logsumexp
        max_update[:] = tile_max_score
```

**后续循环**：
```python
else:
    # 在线 Softmax 更新算法
    output_intermediate = out_update
    logsumexp_intermediate = sum_update
    max_intermediate = max_update
    
    # 计算新的最大值
    new_max_intermediate = pypto.maximum(max_intermediate, tile_max_score)
    
    # 计算缩放因子
    max_diff_old = pypto.sub(max_intermediate, new_max_intermediate)
    exp_max_diff_old = pypto.exp(max_diff_old)
    max_diff_new = pypto.sub(tile_max_score, new_max_intermediate)
    exp_max_diff_new = pypto.exp(max_diff_new)
    
    # 更新 sum
    scaled_logsumexp_new = pypto.mul(exp_max_diff_new, tile_logsumexp)
    scaled_logsumexp_old = pypto.mul(exp_max_diff_old, logsumexp_intermediate)
    new_logsumexp_intermediate = pypto.add(scaled_logsumexp_old, scaled_logsumexp_new)
    
    # 更新 output
    output_scaled_old = pypto.mul(output_intermediate, pypto.reshape(exp_max_diff_old, [group_tile, 1]))
    output_scaled_new = pypto.mul(weighted_value_intermediate, pypto.reshape(exp_max_diff_new, [group_tile, 1]))
    output_tmp = pypto.add(output_scaled_old, output_scaled_new)
    
    if pypto.is_loop_end(s2_idx):
        # 最后一个 tile，归一化并输出
        out_update[:] = pypto.div(output_tmp, pypto.reshape(new_logsumexp_intermediate, [group_tile, 1]), pypto.DivAlgorithm.INTRINSIC)
        # 组装到最终输出
        output_4_dim = pypto.cast(pypto.reshape(out_update, [1, group_tile, 1, query_dim]), dtype)
        pypto.assemble(output_4_dim, output_offset, attention_output)
    else:
        # 中间 tile，存储更新后的中间结果
        out_update[:] = output_tmp
        sum_update[:] = new_logsumexp_intermediate
        max_update[:] = new_max_intermediate
```

**关键点**：
- 在线 Softmax 算法的核心实现
- 维护三个中间状态：`out_update`, `sum_update`, `max_update`
- 使用 `pypto.is_loop_begin/end` 判断循环状态
- 最后一个 tile 时归一化并输出

## 5. Loop 结构设计

### 5.1 外层循环结构

```python
for batch_idx in pypto.loop(batch_size, name="LOOP_b", idx_name="bIdx"):
    current_actual_seq = kv_actual_seqs[batch_idx]
    
    for s1_idx in pypto.loop(s1, name="LOOP_s1", idx_name="s1Idx"):
        current_seq = (current_actual_seq - s1 + 1 + s1_idx)
        current_seq.as_variable()
        s2_loop = (current_seq + s2_tile - 1) // s2_tile
        
        for n2_idx in pypto.loop(kv_heads, name="LOOP_n2", idx_name="n2Idx"):
            for group_index in pypto.loop(group_loop, name="LOOP_group", idx_name="gIdx"):
                [V0 -> C1 -> V1 -> C2 -> V2 循环]
```

**循环设计理由**：
- **Batch Loop**：批次间无依赖，可并行优化
- **S1 Loop**：支持多 query tokens（增量生成通常 s1=1）
- **N2 Loop**：KV heads 间无依赖，可并行优化
- **Group Loop**：query head 分组，降低内存压力
- **S2 Loop**：KV 序列分块，核心计算循环

### 5.2 内层循环 (S2 Loop)

```python
for s2_idx in pypto.loop(
    s2_loop, name="FLASH_LOOP_L4_s2_SA", idx_name="s2_idx",
    unroll_list=[8, 2, 1]
):
    [V0 -> C1 -> V1 -> C2 -> V2]
```

**关键设计**：
- `unroll_list=[8, 2, 1]`：支持循环展开优化
- 使用 `pypto.is_loop_begin/end` 判断循环状态
- 在线 Softmax 状态更新逻辑

### 5.3 循环优化建议

| 循环层级 | 优化策略 |
|---------|---------|
| Batch Loop | 多核并行（device_sched_mode=3） |
| S1 Loop | 通常 s1=1，无优化需求 |
| N2 Loop | 多核并行（device_sched_mode=3） |
| Group Loop | Tiling 优化内存访问 |
| S2 Loop | 循环展开、流水线优化 |

## 6. 内存管理设计

### 6.1 Workspace 内存分配

```python
# 中间状态张量
out_update: [group_tile, query_dim]  # FP32
sum_update: [1, group_tile]          # FP32
max_update: [1, group_tile]          # FP32

# 组装缓冲区
key_assemble: [s2_tile, query_dim + rope_dim]         # BF16
key_nope_assemble: [s2_tile, query_dim]               # BF16
key_rope_assemble: [s2_tile, rope_dim]                # BF16
query_assemble: [group_tile, query_dim + rope_dim]    # BF16

# 计算结果
attention_score: [group_tile, s2_tile]               # FP32
tile_attention_prob: [group_tile, s2_tile]           # FP32
weighted_value_intermediate: [group_tile, query_dim] # FP32
```

### 6.2 内存优化策略

1. **In-place Reshape**：使用 `pypto.reshape(..., inplace=True)` 避免额外内存分配
2. **Buffer Reuse**：循环内中间缓冲区复用
3. **Paged Cache**：非连续内存访问，提高利用率
4. **Workspace 限制**：通过 Tiling 控制 workspace 大小

### 6.3 L1/L0 缓存优化

```python
# Pass Options 配置
pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 2, 0: 8},
        "cube_l1_reuse_setting": {-1: 2},
        "cube_nbuffer_setting": {-1: 2},
    },
    runtime_options={
        "stitch_function_max_num": 256,
        "device_sched_mode": 3
    }
)
```

**配置说明**：
- `vec_nbuffer_setting`：向量操作多缓冲区，减少等待
- `cube_l1_reuse_setting`：矩阵计算 L1 缓存复用
- `cube_nbuffer_setting`：矩阵计算多缓冲区
- `stitch_function_max_num`：最大拼接函数数量
- `device_sched_mode=3`：多核调度模式

## 7. API 映射

### 7.1 PyPTO API 使用

| PyPTO API | 使用场景 | 说明 |
|-----------|----------|------|
| `pypto.reshape` | 数据预处理 | inplace=True 避免额外内存 |
| `pypto.view` | 非连续内存访问 | 从 Paged Cache 提取数据 |
| `pypto.assemble` | 数据组装 | 将分散数据组装到连续缓冲区 |
| `pypto.matmul` | 矩阵乘法 | Cube 单元计算 |
| `pypto.mul/sub/add/div` | 向量运算 | Vec 单元计算 |
| `pypto.exp/amax/sum/maximum` | Softmax 相关 | Vec 单元计算 |
| `pypto.cast` | 类型转换 | BF16 ↔ FP32 |
| `pypto.loop` | 循环控制 | 支持展开和状态判断 |
| `pypto.tensor` | 中间张量分配 | Workspace 管理 |
| `pypto.set_cube_tile_shapes` | Cube Tiling | 矩阵计算切分配置 |
| `pypto.set_vec_tile_shapes` | Vec Tiling | 向量计算切分配置 |
| `pypto.set_semantic_label` | 语义标记 | 优化和调试辅助 |

### 7.2 数据类型路由

| 计算阶段 | 数据类型 | 说明 |
|----------|----------|------|
| 输入/输出 | BF16 | 存储和传输效率 |
| Q×K^T | FP32 | 精度保证 |
| Softmax | FP32 | 数值稳定性 |
| Softmax×V | FP32 | 精度保证 |
| 输出更新 | FP32 → BF16 | 最终输出降精度 |

## 8. 性能优化设计

### 8.1 计算优化

1. **Cube 单元利用**：
   - C1 (Q×K^T) 和 C2 (Softmax×V) 使用 Cube 单元
   - Tiling 配置优化 Cube 利用率

2. **Vec 单元利用**：
   - V0 (assemble)、V1 (softmax)、V2 (update) 使用 Vec 单元
   - 多缓冲区配置减少等待

3. **流水线优化**：
   - Cube 和 Vec 单元并行执行
   - 多缓冲区配置支持流水线

### 8.2 内存访问优化

1. **Paged Cache**：
   - 非连续内存访问，减少内存碎片
   - Block Table 映射，提高利用率

2. **Tiling 策略**：
   - Group Tiling 降低 L1/L0 压力
   - Sequence Tiling 优化 Block 访问

3. **数据组装优化**：
   - 一次性组装 Key 和 Key_Rope
   - Query 组装复用（每个 s2_tile）

### 8.3 并行优化

```python
# 多核并行调度
device_sched_mode = 3

# 循环展开
unroll_list = [8, 2, 1]

# Combine Axis 优化
pypto.experimental.set_operation_options(combine_axis=True)
```

## 9. 调试与验证设计

### 9.1 调试选项

```python
debug_options={"runtime_debug_mode": 1}
```

**功能**：
- 运行时调试信息输出
- 支持中间结果检查
- 性能分析辅助

### 9.2 精度验证

```python
compare(pypto_atten_out.cpu(), mla_golden.cpu(), "pypto_atten_out", 
        atol=0.0001, rtol=0.0078125, max_error_ratio=0.005)
```

**验证标准**：
- 绝对容差 (atol): 0.0001
- 相对容差 (rtol): 0.0078125 (1/128)
- 最大误差比例: 0.5%

### 9.3 Golden 实现

```python
def ifa_mla_golden(query, key, value, query_rope, key_rope):
    query_full = torch.cat([query, query_rope], dim=-1)
    key_full = torch.cat([key, key_rope], dim=-1)
    
    softmax_scale = query_full.shape[-1] ** -0.5
    
    qk_mm_res = torch.matmul(query_full, key_full.transpose(-2, -1))
    qk_ele_res = qk_mm_res * softmax_scale
    
    softmax_res = F.softmax(qk_ele_res, dim=-1)
    
    attention_out = torch.matmul(softmax_res, value)
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

3. **不同精度**：
   - 当前支持 BF16

### 10.2 可配置参数

- **MlaConfig**：算子配置参数
- **AttentionTileConfig**：Tiling 配置参数
- **Pass Options**：编译优化选项
- **Runtime Options**：运行时调度选项

## 11. 设计权衡与决策

### 11.1 性能 vs 精度

- **选择**：使用 FP32 中间计算，BF16 输入输出
- **权衡**：牺牲部分内存效率换取精度保证
- **理由**：Softmax 对精度敏感，FP32 保证数值稳定性

### 11.2 灵活性 vs 优化

- **选择**：提供 Tiling 配置参数
- **权衡**：增加配置复杂度换取性能调优空间
- **理由**：不同场景需要不同 Tiling 策略

### 11.3 通用性 vs 专用性

- **选择**：专注于 MLA 增量场景
- **权衡**：牺牲通用性换取场景优化
- **理由**：自回归生成是高频场景，专用优化收益大