# Incre Flash Attention MLA API 使用报告

## 1. 报告概述

本报告记录 `incre_flash_attention_mla` 算子实现中使用的 PyPTO API，包括 API 使用方式、参数配置、约束条件和注意事项。

### 1.1 API 使用统计

| API 类别 | API 数量 | 使用次数 | 说明 |
|----------|----------|----------|------|
| 数据操作 | 6 | 20+ | reshape, view, assemble, cast, tensor, set_matrix_size |
| 数学运算 | 8 | 15+ | matmul, mul, sub, add, div, exp, amax, sum |
| 控制流 | 2 | 5+ | loop, is_loop_begin/end |
| 配置 | 4 | 10+ | set_cube_tile_shapes, set_vec_tile_shapes, set_semantic_label, set_pass_options |
| 编译器配置 | 1 | 1 | pypto.frontend.jit |

## 2. 编译器配置 API

### 2.1 `pypto.frontend.jit`

**用途**：装饰器，用于配置 kernel 函数的编译和运行选项

**使用示例**：
```python
@pypto.frontend.jit(
    pass_options={
        "vec_nbuffer_setting": {-1: 2, 0: 8},
        "cube_l1_reuse_setting": {-1: 2},
        "cube_nbuffer_setting": {-1: 2},
    },
    runtime_options={
        "stitch_function_max_num": 256,
        "device_sched_mode": 3
    },
    debug_options={"runtime_debug_mode": 1}
)
def incre_flash_attention_mla_kernel(...):
    ...
```

**参数说明**：

#### pass_options

| 参数名 | 值 | 说明 |
|--------|---|------|
| `vec_nbuffer_setting` | {-1: 2, 0: 8} | 向量操作多缓冲区配置，-1 表示默认，0 表示特定优化 |
| `cube_l1_reuse_setting` | {-1: 2} | 矩阵计算 L1 缓存复用配置 |
| `cube_nbuffer_setting` | {-1: 2} | 矩阵计算多缓冲区配置 |

**优化效果**：
- 多缓冲区配置减少计算单元等待时间
- L1 缓存复用提高内存访问效率
- 支持流水线并行执行

#### runtime_options

| 参数名 | 值 | 说明 |
|--------|---|------|
| `stitch_function_max_num` | 256 | 最大拼接函数数量，用于优化 kernel 组合 |
| `device_sched_mode` | 3 | 设备调度模式，支持多核并行调度 |

**优化效果**：
- `device_sched_mode=3`: 多核并行调度，提高计算利用率
- `stitch_function_max_num`: 控制 kernel 组合粒度

#### debug_options

| 参数名 | 值 | 说明 |
|--------|---|------|
| `runtime_debug_mode` | 1 | 运行时调试模式，输出调试信息 |

**用途**：
- 输出运行时调试信息
- 支持性能分析和问题定位

## 3. 数据操作 API

### 3.1 `pypto.reshape`

**用途**：改变张量形状，支持 inplace 操作

**使用示例**：
```python
# Inplace reshape，避免额外内存分配
query_2d = pypto.reshape(query, (batch_size * query_heads * s1, query_dim), inplace=True)

# 普通 reshape
output_4_dim = pypto.reshape(out_update, [1, group_tile, 1, query_dim])
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `tensor` | pypto.Tensor | 输入张量 |
| `shape` | tuple/list | 目标形状 |
| `inplace` | bool | 是否 inplace 操作，默认 False |

**注意事项**：
- `inplace=True` 避免额外内存分配，推荐在数据预处理时使用
- reshape 前后总元素数量必须一致
- 支持动态形状（使用 pypto.DYNAMIC）

### 3.2 `pypto.view`

**用途**：创建张量的视图，支持非连续内存访问

**使用示例**：
```python
# 从 2D key_cache 中提取一个 block
key_nope_view = pypto.view(
    key_2d, 
    [block_size, query_dim],       # 视图形状
    [block_start_offset, 0]         # 起始偏移
)
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `tensor` | pypto.Tensor | 输入张量 |
| `shape` | list | 视图形状 |
| `offset` | list | 起始偏移（每个维度的偏移量） |

**注意事项**：
- 用于非连续内存访问，支持 Paged KV Cache
- 不实际拷贝数据，只是创建视图
- 常与 `pypto.assemble` 配合使用

### 3.3 `pypto.assemble`

**用途**：将数据组装到目标张量的指定位置

**使用示例**：
```python
# 将 key block 组装到组装缓冲区
pypto.assemble(
    key_nope_view,                  # 源张量
    [block_i * block_size, 0],      # 目标位置
    key_nope_assemble               # 目标张量
)

# 将 Query 组装到 query_assemble
pypto.assemble(query_no_rope_view, [0, 0], query_assemble)
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `src` | pypto.Tensor | 源张量（通常是 view 创建的视图） |
| `position` | list | 目标位置（每个维度的起始位置） |
| `dst` | pypto.Tensor | 目标张量 |

**注意事项**：
- 支持非连续到连续的数据组装
- 常用于从 Paged Cache 中提取数据
- `position` 指定目标张量中的插入位置

### 3.4 `pypto.cast`

**用途**：数据类型转换

**使用示例**：
```python
# FP32 转 BF16
tile_attention_prob_fp16 = pypto.cast(tile_attention_prob, dtype)

```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `tensor` | pypto.Tensor | 输入张量 |
| `dtype` | pypto.DataType | 目标数据类型 |

**注意事项**：
- 中间计算使用 FP32 保证精度
- 输入/输出使用 BF16 提高效率
- Softmax 相关计算必须使用 FP32（数值稳定性）

### 3.5 `pypto.tensor`

**用途**：创建中间张量（Workspace 分配）

**使用示例**：
```python
# 创建中间状态张量
out_update = pypto.tensor([group_tile, query_dim], pypto.DT_FP32, "out_update")
sum_update = pypto.tensor([1, group_tile], pypto.DT_FP32, "sum_update")
max_update = pypto.tensor([1, group_tile], pypto.DT_FP32, "max_update")

# 创建组装缓冲区
key_assemble = pypto.tensor([s2_tile, query_dim + rope_dim], key_2d.dtype, "key_assemble")
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `shape` | list | 张量形状 |
| `dtype` | pypto.DataType | 数据类型 |
| `name` | str | 张量名称（用于调试） |

**注意事项**：
- 用于分配 Workspace 内存
- 生命周期：kernel 函数内
- 建议添加名称方便调试
- 形状应与 Tiling 配置匹配

### 3.6 `pypto.set_matrix_size`

**用途**：设置矩阵乘法的矩阵尺寸

**使用示例**：
```python
# C2 阶段设置矩阵尺寸
pypto.set_matrix_size([
    tile_attention_prob_fp16.shape[0],  # M
    tile_attention_prob_fp16.shape[1],  # K
    key_nope_assemble.shape[1]          # N
])
weighted_value_intermediate = pypto.matmul(tile_attention_prob_fp16, key_nope_assemble, pypto.DT_FP32)
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `size` | list | 矩阵尺寸 [M, K, N] |

**注意事项**：
- 用于优化矩阵乘法的 Cube 单元利用率
- 应在 `pypto.matmul` 前调用
- 尺寸应与实际矩阵形状匹配

## 4. 数学运算 API

### 4.1 `pypto.matmul`

**用途**：矩阵乘法

**使用示例**：
```python
# C1: Q × K^T
attention_score = pypto.matmul(
    query_assemble,       # [group_tile, q_d + q_rope_d]
    key_assemble,         # [s2_tile, q_d + q_rope_d]
    pypto.DT_FP32,        # 输出类型
    a_trans=False,        # A 不转置
    b_trans=True          # B 转置
)

# C2: Softmax × V
weighted_value_intermediate = pypto.matmul(
    tile_attention_prob_fp16,  # [group_tile, s2_tile]
    key_nope_assemble,         # [s2_tile, q_d]
    pypto.DT_FP32              # 输出类型
)
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `a` | pypto.Tensor | 左矩阵 |
| `b` | pypto.Tensor | 右矩阵 |
| `dtype` | pypto.DataType | 输出数据类型 |
| `a_trans` | bool | A 是否转置，默认 False |
| `b_trans` | bool | B 是否转置，默认 False |

**注意事项**：
- 使用 Cube 单元计算
- 输出类型建议使用 FP32 保证精度
- 应配合 `pypto.set_cube_tile_shapes` 优化 Tiling
- 支持 transpose 参数，避免显式转置操作

### 4.2 向量运算 API

#### 4.2.1 `pypto.mul`

**用途**：元素级乘法

**使用示例**：
```python
# V1: 缩放注意力分数
scaled_attention_score = pypto.mul(attention_score, softmax_scale)

# V2: 缩放因子计算
exp_max_diff_old = pypto.exp(max_diff_old)
exp_max_diff_new = pypto.exp(max_diff_new)
scaled_logsumexp_new = pypto.mul(exp_max_diff_new, tile_logsumexp)
```

**注意事项**：
- 支持张量和常量的乘法
- 使用 Vec 单元计算
- 应配合 `pypto.set_vec_tile_shapes` 优化 Tiling

#### 4.2.2 `pypto.sub`

**用途**：元素级减法

**使用示例**：
```python
# V1: Softmax 稳定性优化
score_diff = pypto.sub(scaled_attention_score, tile_max_score_reduce)

# V2: 最大值差计算
max_diff_old = pypto.sub(max_intermediate, new_max_intermediate)
max_diff_new = pypto.sub(tile_max_score, new_max_intermediate)
```

#### 4.2.3 `pypto.add`

**用途**：元素级加法

**使用示例**：
```python
# V2: 更新 sum 和 output
new_logsumexp_intermediate = pypto.add(scaled_logsumexp_old, scaled_logsumexp_new)
output_tmp = pypto.add(output_scaled_old, output_scaled_new)
```

#### 4.2.4 `pypto.div`

**用途**：元素级除法

**使用示例**：
```python
# V2: 最终归一化
out_update[:] = pypto.div(
    output_tmp,
    pypto.reshape(new_logsumexp_intermediate, [group_tile, 1]),
    pypto.DivAlgorithm.INTRINSIC
)
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `a` | pypto.Tensor | 被除数 |
| `b` | pypto.Tensor | 除数 |
| `algorithm` | pypto.DivAlgorithm | 除法算法，默认标准除法 |

**注意事项**：
- `pypto.DivAlgorithm.INTRINSIC`: 使用硬件 intrinsic 指令
- 除法操作较慢，建议尽量减少使用

### 4.3 数学函数 API

#### 4.3.1 `pypto.exp`

**用途**：指数函数

**使用示例**：
```python
# V1: 计算 softmax 指数部分
tile_attention_prob = pypto.exp(score_diff)

# V2: 计算缩放因子
exp_max_diff_old = pypto.exp(max_diff_old)
exp_max_diff_new = pypto.exp(max_diff_new)
```

**注意事项**：
- 使用 Vec 单元计算
- 数值稳定性：输入应先减去最大值
- 应配合 `pypto.set_vec_tile_shapes` 优化

#### 4.3.2 `pypto.amax`

**用途**：沿指定维度求最大值

**使用示例**：
```python
# V1: 计算局部最大值
tile_max_score_reduce = pypto.amax(scaled_attention_score, dim=-1, keepdim=True)
# 结果形状: [group_tile, 1]
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `tensor` | pypto.Tensor | 输入张量 |
| `dim` | int | 指定维度（-1 表示最后一维） |
| `keepdim` | bool | 是否保持维度，默认 False |

**注意事项**：
- 用于在线 Softmax 算法
- `keepdim=True` 保持形状方便后续计算

#### 4.3.3 `pypto.sum`

**用途**：沿指定维度求和

**使用示例**：
```python
# V1: 计算局部指数和
tile_logsumexp_reduce = pypto.sum(tile_attention_prob, dim=-1, keepdim=True)
# 结果形状: [group_tile, 1]
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `tensor` | pypto.Tensor | 输入张量 |
| `dim` | int | 指定维度（-1 表示最后一维） |
| `keepdim` | bool | 是否保持维度，默认 False |

**注意事项**：
- 用于在线 Softmax 算法
- `keepdim=True` 保持形状方便后续计算

#### 4.3.4 `pypto.maximum`

**用途**：元素级最大值

**使用示例**：
```python
# V2: 计算新的最大值
new_max_intermediate = pypto.maximum(max_intermediate, tile_max_score)
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `a` | pypto.Tensor | 第一个张量 |
| `b` | pypto.Tensor | 第二个张量 |

**注意事项**：
- 用于在线 Softmax 算法
- 两个张量形状必须相同

## 5. 控制流 API

### 5.1 `pypto.loop`

**用途**：循环控制

**使用示例**：
```python
# 外层循环
for batch_idx in pypto.loop(batch_size, name="LOOP_b", idx_name="bIdx"):
    current_actual_seq = kv_actual_seqs[batch_idx]
    
    for s1_idx in pypto.loop(s1, name="LOOP_s1", idx_name="s1Idx"):
        current_seq = (current_actual_seq - s1 + 1 + s1_idx)
        
        for n2_idx in pypto.loop(kv_heads, name="LOOP_n2", idx_name="n2Idx"):
            for group_index in pypto.loop(group_loop, name="LOOP_group", idx_name="gIdx"):
                ...

# 内层循环（支持展开）
for s2_idx in pypto.loop(
    s2_loop, 
    name="FLASH_LOOP_L4_s2_SA", 
    idx_name="s2_idx",
    unroll_list=[8, 2, 1]
):
    ...
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `count` | int | 循环次数 |
| `name` | str | 循环名称（用于调试和优化） |
| `idx_name` | str | 循环索引变量名 |
| `unroll_list` | list | 循环展开配置（可选） |

**注意事项**：
- `name`: 用于调试和优化，建议使用有意义的名称
- `idx_name`: 循环索引变量名，方便引用
- `unroll_list`: 支持循环展开优化，[8, 2, 1] 表示支持 8、2、1 倍展开

### 5.2 `pypto.is_loop_begin` / `pypto.is_loop_end`

**用途**：判断循环状态

**使用示例**：
```python
for s2_idx in pypto.loop(s2_loop, ...):
    ...
    
    # V2: 输出更新
    if pypto.is_loop_begin(s2_idx):
        # 第一次循环的特殊处理
        output_tmp = weighted_value_intermediate
        
        if pypto.is_loop_end(s2_idx):
            # 只有一个 tile，直接输出
            out_update[:] = output_tmp / tile_logsumexp_reduce
            ...
        else:
            # 多个 tile，存储中间状态
            out_update[:] = output_tmp
            ...
    else:
        # 后续循环的在线 Softmax 更新
        ...
        
        if pypto.is_loop_end(s2_idx):
            # 最后一个 tile，归一化输出
            out_update[:] = pypto.div(output_tmp, ...)
            ...
        else:
            # 中间 tile，更新状态
            out_update[:] = output_tmp
            ...
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `loop_idx` | pypto.LoopIndex | 循环索引变量 |

**返回值**：
- `bool`: True 表示是循环开始/结束

**注意事项**：
- 用于在线 Softmax 算法，处理不同循环状态
- `is_loop_begin`: 第一次循环，初始化状态
- `is_loop_end`: 最后一次循环，输出结果
- 单次循环时 `begin` 和 `end` 同时为 True

### 5.3 `current_seq.as_variable()`

**用途**：将循环相关变量标记为中间变量

**使用示例**：
```python
current_seq = (current_actual_seq - s1 + 1 + s1_idx)
current_seq.as_variable()  # 标记为中间变量
s2_loop = (current_seq + s2_tile - 1) // s2_tile
```

**注意事项**：
- 用于动态序列长度场景
- 使编译器正确处理动态循环次数

## 6. 配置 API

### 6.1 `pypto.set_cube_tile_shapes`

**用途**：设置矩阵计算的 Cube Tiling 配置

**使用示例**：
```python
# C1 阶段: Q × K^T
pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
attention_score = pypto.matmul(query_assemble, key_assemble, pypto.DT_FP32, a_trans=False, b_trans=True)

# C2 阶段: Softmax × V
pypto.set_cube_tile_shapes(c2_tile[0], c2_tile[1], c2_tile[2])
weighted_value_intermediate = pypto.matmul(tile_attention_prob_fp16, key_nope_assemble, pypto.DT_FP32)
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `tile_m` | list/int | M 维度切分配置 |
| `tile_k` | list/int | K 维度切分配置 |
| `tile_n` | list/int | N 维度切分配置 |

**典型配置**：
```python
c1_tile = [[128, 128], [128, 128], [128, 128]]  # [tile_m, tile_k, tile_n]
c2_tile = [[128, 128], [128, 128], [128, 128]]
```

**注意事项**：
- 应在 `pypto.matmul` 前调用
- 影响 Cube 单元利用率和 L1/L0 缓存使用
- 参数可以是 int 或 list，list 表示多级切分

### 6.2 `pypto.set_vec_tile_shapes`

**用途**：设置向量计算的 Vec Tiling 配置

**使用示例**：
```python
# V0 阶段: Key 组装
pypto.set_vec_tile_shapes(v0_tile[0], v0_tile[1])
key_assemble = pypto.tensor([s2_tile, query_dim + rope_dim], key_2d.dtype, "key_assemble")

# V1 阶段: Softmax
pypto.set_vec_tile_shapes(v1_tile[0], v1_tile[1])
scaled_attention_score = pypto.mul(attention_score, softmax_scale)

# V2 阶段: 输出更新
pypto.set_vec_tile_shapes(v2_tile[0], v2_tile[1])
output_scaled_old = pypto.mul(output_intermediate, pypto.reshape(exp_max_diff_old, [group_tile, 1]))

# 多级切分
pypto.set_vec_tile_shapes(1, v2_tile[0], 1, v2_tile[1])  # 4 参数版本
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `tile_0` | int | 第一维度切分大小 |
| `tile_1` | int | 第二维度切分大小 |
| `tile_2` | int | 第三维度切分大小（可选） |
| `tile_3` | int | 第四维度切分大小（可选） |

**典型配置**：
```python
v0_tile = [128, 576]    # Key 组装: [128, 576]
v1_tile = [8, 2048]     # Softmax: [8, 2048]
v2_tile = [64, 512]     # 输出更新: [64, 512]
```

**注意事项**：
- 应在向量操作前调用
- 影响 Vec 单元利用率和 L0 缓存使用
- 支持 2 参数或 4 参数版本

### 6.3 `pypto.set_semantic_label`

**用途**：设置语义标签，用于优化和调试

**使用示例**：
```python
# V0 阶段: Key 组装
pypto.set_semantic_label("MLA_V0")
pypto.set_vec_tile_shapes(v0_tile[0], v0_tile[1])
...

# C1 阶段: Q × K^T
pypto.set_semantic_label("MLA_C1")
pypto.set_cube_tile_shapes(c1_tile[0], c1_tile[1], c1_tile[2])
...

# V1 阶段: Softmax
pypto.set_semantic_label("MLA_V1")
...

# C2 阶段: Softmax × V
pypto.set_semantic_label("MLA_C2")
...

# V2 阶段: 输出更新
pypto.set_semantic_label("MLA_V2")
...

# V2 更新阶段
pypto.set_semantic_label("MLA_UpdateVec2")
...
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `label` | str | 语义标签名称 |

**注意事项**：
- 用于优化器识别计算阶段
- 用于调试输出和性能分析
- 建议使用有意义的标签名

### 6.4 `pypto.set_pass_options`

**用途**：设置 Pass 选项，用于优化器配置

**使用示例**：
```python
# V2 更新阶段，设置作用域
pypto.set_pass_options(sg_set_scope=1)
new_max_intermediate = pypto.maximum(max_intermediate, tile_max_score)
...
pypto.set_pass_options(sg_set_scope=-1)  # 结束作用域
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `sg_set_scope` | int | 作用域设置，1 表示开始，-1 表示结束 |

**注意事项**：
- 用于控制优化器的作用域
- 影响某些优化的应用范围
- 应在特定计算阶段前后设置

### 6.5 `pypto.experimental.set_operation_options`

**用途**：设置操作选项

**使用示例**：
```python
pypto.experimental.set_operation_options(combine_axis=True)
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `combine_axis` | bool | 是否合并轴 |

**注意事项**：
- 实验性 API，用于高级优化
- `combine_axis=True` 优化轴合并

## 7. 数据类型定义

### 7.1 `pypto.Tensor`

**用途**：定义张量类型和形状

**使用示例**：
```python
def incre_flash_attention_mla_kernel(
    query: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    key: pypto.Tensor([...], pypto.DT_BF16),
    value: pypto.Tensor([...], pypto.DT_BF16),
    query_rope: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    key_rope: pypto.Tensor([...], pypto.DT_BF16),
    block_table: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_INT32),
    kv_actual_seqs: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    attention_output: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    ...
):
    ...
```

**参数说明**：

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `shape` | list | 张量形状，使用 `pypto.DYNAMIC` 表示动态维度 |
| `dtype` | pypto.DataType | 数据类型 |

**注意事项**：
- `pypto.DYNAMIC`: 表示动态维度
- `...`: 表示省略的维度
- 应与实际输入形状匹配

### 7.2 支持的数据类型

| 数据类型 | 常量 | 说明 | 使用场景 |
|----------|------|------|----------|
| Bfloat16 | `pypto.DT_BF16` | 16 位浮点数 | 输入/输出 |
| Float32 | `pypto.DT_FP32` | 32 位浮点数 | 中间计算 |
| Float16 | `pypto.DT_FP16` | 16 位浮点数 | 未使用 |
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
   - 使用 `is_loop_begin/end` 优化循环状态判断

4. **配置编译器选项**：
   - 使用多缓冲区配置 (`vec_nbuffer_setting`, `cube_nbuffer_setting`)
   - 使用 L1 缓存复用 (`cube_l1_reuse_setting`)
   - 使用多核调度 (`device_sched_mode=3`)

### 8.2 精度保证最佳实践

1. **中间计算使用 FP32**：
   - `matmul` 输出使用 `pypto.DT_FP32`
   - Softmax 相关计算使用 FP32
   - 避免精度损失累积

2. **Softmax 数值稳定性**：
   - 减去最大值后再计算指数
   - 使用 `amax` 计算局部最大值
   - 使用 `sum` 计算局部指数和

3. **类型转换时机**：
   - 输入：BF16
   - 计算：FP32
   - 输出：BF16
   - 避免过早或过晚转换

### 8.3 内存管理最佳实践

1. **Workspace 分配**：
   - 使用 `pypto.tensor` 分配中间张量
   - 形状应与 Tiling 配置匹配
   - 避免过大的 workspace

2. **内存复用**：
   - 循环内中间张量复用
   - 使用 inplace 操作
   - MLA 中 Key 和 Value 共享内存

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
   - cast 输入输出类型必须兼容
   - 某些 API 仅支持特定数据类型

3. **计算顺序约束**：
   - `set_cube_tile_shapes` 必须在 `matmul` 前调用
   - `set_vec_tile_shapes` 必须在 Vec 操作前调用
   - `set_matrix_size` 必须在 `matmul` 前调用

### 9.2 常见错误

1. **Tiling 参数不匹配**：
   ```python
   # 错误：Tiling 参数与实际形状不匹配
   pypto.set_vec_tile_shapes(64, 512)  # 但张量形状是 [32, 256]
   ```

2. **忘记设置 Tiling 配置**：
   ```python
   # 错误：matmul 前未设置 cube tiling
   attention_score = pypto.matmul(query, key, pypto.DT_FP32)
   ```

3. **数据类型不匹配**：
   ```python
   # 错误：Softmax 使用 BF16（数值不稳定）
   tile_attention_prob = pypto.exp(scaled_attention_score)  # scaled_attention_score 是 BF16
   ```

4. **循环状态判断错误**：
   ```python
   # 错误：在单次循环中分别处理 begin 和 end
   if pypto.is_loop_begin(s2_idx):
       # 处理第一次
   if pypto.is_loop_end(s2_idx):  # 应使用 elif
       # 处理最后一次
   ```

## 10. API 版本和兼容性

### 10.1 API 版本

本算子使用的 PyPTO API 版本：
- PyPTO: 当前版本
- CANN: 兼容版本

### 10.2 兼容性说明

| API | 稳定性 | 兼容性 | 说明 |
|-----|--------|--------|------|
| 核心计算 API (matmul, mul, etc.) | 稳定 | 高 | 基础 API，长期支持 |
| 数据操作 API (reshape, view, etc.) | 稳定 | 高 | 基础 API，长期支持 |
| 控制流 API (loop, is_loop_begin/end) | 稳定 | 高 | 基础 API，长期支持 |
| 配置 API (set_tile_shapes, etc.) | 稳定 | 中 | 配置 API，参数可能有调整 |
| 编译器配置 (jit) | 稳定 | 中 | 配置 API，参数可能有调整 |
| 实验性 API (experimental) | 实验性 | 低 | 实验性 API，可能变更 |

### 10.3 未来扩展

可能的 API 扩展：
- 支持更多数据类型 (FP16, INT8)
- 支持更多 Tiling 配置选项
- 支持更多编译器优化选项
- 支持更多数学运算 API

## 11. 参考资料

- PyPTO API 文档：官方 API 参考文档
- Ascend C 编程指南：底层实现参考
- Flash Attention 论文：算法原理参考
- MLA 论文：架构原理参考

## 12. 附录

### 12.1 API 快速参考表

| API | 类别 | 主要用途 | 关键参数 |
|-----|------|----------|----------|
| `pypto.frontend.jit` | 编译器配置 | Kernel 配置 | pass_options, runtime_options |
| `pypto.reshape` | 数据操作 | 改变形状 | shape, inplace |
| `pypto.view` | 数据操作 | 创建视图 | shape, offset |
| `pypto.assemble` | 数据操作 | 数据组装 | src, position, dst |
| `pypto.cast` | 数据操作 | 类型转换 | dtype |
| `pypto.tensor` | 数据操作 | 分配张量 | shape, dtype, name |
| `pypto.matmul` | 数学运算 | 矩阵乘法 | a, b, dtype, a_trans, b_trans |
| `pypto.mul` | 数学运算 | 元素乘法 | a, b |
| `pypto.sub` | 数学运算 | 元素减法 | a, b |
| `pypto.add` | 数学运算 | 元素加法 | a, b |
| `pypto.div` | 数学运算 | 元素除法 | a, b, algorithm |
| `pypto.exp` | 数学运算 | 指数函数 | tensor |
| `pypto.amax` | 数学运算 | 最大值 | tensor, dim, keepdim |
| `pypto.sum` | 数学运算 | 求和 | tensor, dim, keepdim |
| `pypto.maximum` | 数学运算 | 元素最大值 | a, b |
| `pypto.loop` | 控制流 | 循环控制 | count, name, idx_name, unroll_list |
| `pypto.is_loop_begin` | 控制流 | 循环状态判断 | loop_idx |
| `pypto.is_loop_end` | 控制流 | 循环状态判断 | loop_idx |
| `pypto.set_cube_tile_shapes` | 配置 | Cube Tiling | tile_m, tile_k, tile_n |
| `pypto.set_vec_tile_shapes` | 配置 | Vec Tiling | tile_0, tile_1, tile_2, tile_3 |
| `pypto.set_semantic_label` | 配置 | 语义标签 | label |
| `pypto.set_pass_options` | 配置 | Pass 选项 | sg_set_scope |
| `pypto.set_matrix_size` | 配置 | 矩阵尺寸 | size |
| `pypto.experimental.set_operation_options` | 实验性 | 操作选项 | combine_axis |