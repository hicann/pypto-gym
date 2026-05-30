---
schema_version: "2.1"
op_name: "moe_finalize_routing_v2"
status: draft
last_updated: "2026-05-14"

# 关键接口契约
compute_kind: "vector"
dtypes: ["bf16", "int32"]
dynamic_axes: ["NUM_ROWS*K", "NUM_ROWS", "K"]
precision: { rtol: 0.0078125, atol: 0.0001 }
---

# moe_finalize_routing_v2 设计方案

> **生成时间**: 2026-05-14
> **设计依据**: SPEC.md + API_REPORT.md + Golden 实现 + 官方示例 examples/moe_finalize_routing_v2.py

---

## 1. 计算图与精度路由

### 1.1 API 调用序列

根据 SPEC.md 中的算法描述和 API_REPORT.md 的 API 映射，核心计算步骤如下：

| 步骤 | 操作 | PyPTO API | 输入 dtype | 输出 dtype | 输出 shape | 备注 |
|------|------|-----------|------------|------------|-----------|------|
| 1 | 输出初始化 | `pypto.zeros([NUM_ROWS, H], pypto.DT_FP32)` | - | FP32 | [NUM_ROWS, H] | 使用 FP32 作为累加器，提高精度 |
| 2 | 添加残差 x1（可选） | `pypto.add(out, x1)` | FP32 + BF16 | FP32 | [NUM_ROWS, H] | x1 需先 cast 为 FP32 |
| 3 | 添加残差 x2（可选） | `pypto.add(out, x2)` | FP32 + BF16 | FP32 | [NUM_ROWS, H] | x2 需先 cast 为 FP32 |
| 4 | dropPadMode 分支 | `pypto.cond(dropPadMode in [0,1])` | INT64 | BOOL | [1] | 决定索引计算方式 |
| 5 | 计算索引位置 | `k * NUM_ROWS + i` 或 `i * K + k` | INT | INT | [1] | SymbolicScalar 运算 |
| 6 | 获取 expanded_row_idx_value | `pypto.view(expandedRowIdx, [1], [idx])` | INT32 | INT32 | [1] | 索引值（可能是 -1） |
| 7 | 条件跳过检查（drop_pad） | `pypto.cond(value == -1)` | INT32 | BOOL | [1] | drop_pad 场景跳过 padding |
| 8 | 条件跳过检查（drop_less） | `pypto.cond(value >= expanded_x_len)` | INT32 + INT | BOOL | [1] | drop_less 场景跳过越界 |
| 9 | 索引查找 | `pypto.view(expandedX, [1, H], [value, 0])` | BF16 | BF16 | [1, H] | 获取目标行 |
| 10 | Cast 为 FP32 | `pypto.cast(dst_row, pypto.DT_FP32)` | BF16 | FP32 | [1, H] | 提高累加精度 |
| 11 | 添加专家偏置（可选） | `pypto.add(dst_row, bias_row)` | FP32 + BF16 | FP32 | [1, H] | bias 需先 cast 为 FP32 |
| 12 | 应用路由权重（可选） | `pypto.mul(dst_row, scale)` | FP32 + BF16 | FP32 | [1, H] | scale 需先 cast 为 FP32 |
| 13 | 累加到输出 | `out[i, :] = out[i, :] + dst_row` | FP32 + FP32 | FP32 | [1, H] | 切片赋值累加 |
| 14 | Cast 输出回 BF16 | `pypto.cast(out, pypto.DT_BF16)` | FP32 | BF16 | [NUM_ROWS, H] | 最终输出类型 |

### 1.2 精度路由

```text
输入(BF16) → [步骤10: cast] → 计算(FP32) → [步骤14: cast] → 输出(BF16)
                ↓
             累加器(FP32)
```

| 转换位置 | 转换方向 | 原因 |
|---------|---------|------|
| 步骤 2/3 前 | BF16 → FP32 | 残差连接需参与累加，使用 FP32 提高精度 |
| 步骤 10 后 | BF16 → FP32 | 累加操作使用 FP32，避免 BF16 精度损失 |
| 步骤 11 前 | BF16 → FP32 | bias 参与 FP32 累加 |
| 步骤 12 前 | BF16 → FP32 | scale 参与 FP32 加权计算 |
| 步骤 14 | FP32 → BF16 | 输出规格要求 BF16 |

### 1.3 替代方案（已排除）

| 替代方案 | 排除原因 |
|---------|---------|
| 使用 `pypto.gather(expandedX, 0, expandedRowIdx)` | gather 的 dim 轴不可切，NUM_ROWS*K 维度过大时无法全载，会导致 UB 内存溢出 |
| 使用 `pypto.index_add_(out, 0, i, dst_row)` | index_add_ 的 dim 轴不可切，NUM_ROWS 维度需全载，不支持动态累加模式 |
| 使用 BF16 累加器 | BF16 精度较低（仅 7-8 bit exponent），多次累加会导致精度损失 |
| 使用 Python `if` 条件判断 | dropPadMode 为编译期常量可用 Python if，但 expanded_row_idx_value 为运行时值，需用 `pypto.cond` |

---

## 2. 数据规格

### 2.1 Kernel 函数签名

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def moe_finalize_routing_v2_kernel(
    expanded_x: pypto.Tensor([pypto.DYNAMIC, H], pypto.DT_BF16),          # [NUM_ROWS*K, H] 或 [E*C, H]
    expanded_row_idx: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),      # [NUM_ROWS*K]
    out: pypto.Tensor([pypto.DYNAMIC, H], pypto.DT_BF16),                 # [NUM_ROWS, H]
    # 可选参数（以下为 None 时表示不提供）
    x1: pypto.Tensor([pypto.DYNAMIC, H], pypto.DT_BF16) = None,           # [NUM_ROWS, H]
    x2: pypto.Tensor([pypto.DYNAMIC, H], pypto.DT_BF16) = None,           # [NUM_ROWS, H]
    bias: pypto.Tensor([E, H], pypto.DT_BF16) = None,                     # [E, H]
    scales: pypto.Tensor([pypto.DYNAMIC, K], pypto.DT_BF16) = None,       # [NUM_ROWS, K]
    expert_idx: pypto.Tensor([pypto.DYNAMIC, K], pypto.DT_INT32) = None,  # [NUM_ROWS, K]
    # 属性参数
    drop_pad_mode: int = 2,                                               # [0, 3]
):
    """
    MoE 路由聚合算子 kernel。
    
    计算: out[i,j] = x1[i,j] + x2[i,j] + Σ_k(scales[i,k] * (expandedX[expandedRowIdx[idx],j] + bias[expertId,j]))
    
    Args:
        expanded_x: MoE FFN 输出，shape [NUM_ROWS*K, H] 或 [E*C, H]
        expanded_row_idx: 行索引，用于查找 expanded_x 中的行，shape [NUM_ROWS*K]
        out: 输出 tensor，shape [NUM_ROWS, H]
        x1: 残差连接 1，可选
        x2: 残差连接 2，可选
        bias: 专家偏置，可选
        scales: 路由权重，可选
        expert_idx: 专家索引，可选
        drop_pad_mode: 控制索引排列方式，[0, 3]
    """
    ...
```

### 2.2 动态轴分析

> 仅运行时才确定大小的轴标 `pypto.DYNAMIC`；编译期已知的轴写常量。

| 维度名 | 是否动态 | 取值范围 / 常量 | 标注方式 | 说明 |
|--------|---------|-----------------|---------|------|
| NUM_ROWS*K (expanded_x 第 0 维) | 是 | [1, 16777216] | `pypto.DYNAMIC` | MoE FFN 输出的行数，动态变化 |
| E*C (expanded_x 第 0 维，drop_pad 场景) | 是 | [1, 16777216] | `pypto.DYNAMIC` | 专家数 × 容量，动态变化 |
| H (expanded_x 第 1 维) | 否 | [1, 16384] | 常量 | hidden size，编译期已知或运行时传入 |
| NUM_ROWS (out 第 0 维) | 是 | [1, NUM_ROWS*K] | `pypto.DYNAMIC` | 输出行数，动态变化 |
| K (scales/expert_idx 第 1 维) | 是 | [1, E] | `pypto.DYNAMIC` 或常量 | 每行选择的专家数，运行时从 scales.shape[1] 推导 |
| E (bias 第 0 维) | 否 | [K, MAX_E] | 常量或动态 | 专家总数，编译期或运行时已知 |

**动态轴标记示例**：

```python
# from_torch 时标记动态轴
expanded_x_pto = pypto.from_torch(expanded_x_torch, name="expanded_x", dynamic_axis=[0])
expanded_row_idx_pto = pypto.from_torch(expanded_row_idx_torch, name="expanded_row_idx", dynamic_axis=[0])
out_pto = pypto.from_torch(out_torch, name="out", dynamic_axis=[0])
```

### 2.3 值类型分析（避免 SymbolicScalar 误用）

| 变量 | 来源 | 类型 | 注意事项 |
|------|------|------|---------|
| NUM_ROWS | `bsk // K`（从 expanded_row_idx.shape[0] 和 scales.shape[1] 推导） | SymbolicScalar | 不可用于 Python `if/range`，不可索引 list |
| K | `scales.shape[1]`（若 scales 存在）或常量 1 | SymbolicScalar 或 int | 若 scales 存在则为 SymbolicScalar |
| H | 编译期常量或运行时传入 | int | 可正常使用 |
| bsk | `expanded_row_idx.shape[0]` | SymbolicScalar | 动态轴大小 |
| i | `pypto.loop(NUM_ROWS)` 返回值 | SymbolicScalar（loop index） | 不可用于 list 索引，可用于 SymbolicScalar 运算 |
| k | `pypto.loop(K)` 返回值 | SymbolicScalar（loop index） | 不可用于 list 索引 |
| expanded_row_idx_idx | `k * NUM_ROWS + i` 或 `i * K + k` | SymbolicScalar | 索引计算结果，用于 view offset |
| expanded_row_idx_value | `pypto.view(expandedRowIdx, [1], [idx])` 返回的 tensor 的值 | Tensor（INT32） | 需通过 tensor 操作提取，不能用 Python if 判断 |
| drop_pad_mode | 编译期属性参数 | int | 可用 Python if 判断 |

**SymbolicScalar 禁止操作速查**：

| 禁止写法 | 原因 | 正确替代 |
|----------|------|---------|
| `if NUM_ROWS > 0:` | SymbolicScalar 不能用于 Python `if` | `pypto.cond(NUM_ROWS > 0)` |
| `range(NUM_ROWS)` | SymbolicScalar 不能用于 Python `range` | `pypto.loop(NUM_ROWS)` |
| `list[i]` | SymbolicScalar 不能索引 Python list | `pypto.view(tensor, shape, [i, ...])` |
| `NUM_ROWS ** 2` | SymbolicScalar 不支持幂运算 | 使用静态值或避免此计算 |
| `min(NUM_ROWS, 1024)` | Python `min` 不接受 SymbolicScalar | `NUM_ROWS.min(1024)` |

---

## 3. Tiling 策略

### 3.1 算子类型

**Vector 类型**（不包含 matmul，仅有索引查找、逐元素加法、乘法和累加操作）

### 3.2 Tiling 推导

- **同时驻留 UB 的 Tensor**：

| Tensor | 用途 | shape | dtype | 大小估算（bytes） |
|--------|------|-------|-------|------------------|
| out_row | 输出累加器（当前行） | [1, tile_h] | FP32 | 4 × tile_h |
| dst_row | 从 expanded_x 获取的目标行 | [1, tile_h] | FP32 | 4 × tile_h |
| bias_row | 专家偏置行（可选） | [1, tile_h] | FP32 | 4 × tile_h |
| scale | 路由权重标量 | [1] | FP32 | 4 |
| expanded_row_idx_value | 索引值 | [1] | INT32 | 4 |
| **总计** | - | - | - | ≈ 8 × tile_h + 8 |

- **推导步骤**：

  1. **尾轴对齐**：
     - FP32 累加器：尾轴 ≥ 8 元素（32B 对齐）
     - tile_h 应为 8 的倍数，建议取 512 或更小
  
  2. **UB 预算**：
     - UB 容量 ≈ 64KB（典型值）
     - 同时驻留 tensor：out_row + dst_row + bias_row + 其他 ≈ 8 × tile_h bytes
     - 最大 tile_h ≈ (64KB - 8) / 8 ≈ 8000 元素
     - 实际选择：tile_h = 512 或 1024（留有余量）
  
  3. **展开约束**：
     - 外循环次数：`NUM_ROWS`（动态，运行时确定）
     - 内循环次数：`K`（动态，运行时确定）
     - 单 tile 计算复杂度低，展开不会超限
  
  4. **关键约束**：
     - **gather 的 dim 轴不可切**：但本设计不使用 gather API，而是用 `pypto.view` 逐行查找
     - **index_add_ 的 dim 轴不可切**：但本设计不使用 index_add_，而是用切片赋值 `out[i, :] = ...`
     - **逐行处理策略**：避免了 dim 轴全载问题，每次只加载一行 [1, tile_h]

- **最终 tile**：

```python
# Tiling 配置（在 kernel 开头设置）
pypto.set_vec_tile_shapes(1, tile_h)  # [1, tile_h]：逐行处理

# tile_h 推荐值：
# - 小规模（H ≤ 1024）：tile_h = H（全行处理）
# - 中规模（H ≤ 4096）：tile_h = 512
# - 大规模（H > 4096）：tile_h = 512 或 1024（避免 UB 溢出）

# 实际配置方案：
tile_h = 512  # 默认值，可根据 H 动态调整
if H <= 1024:
    tile_h = H
elif H <= 2048:
    tile_h = 1024
else:
    tile_h = 512

pypto.set_vec_tile_shapes(1, tile_h)
```

### 3.3 替代方案（已排除）

| 备选 tile | 否决理由 |
|-----------|---------|
| `[tile_rows, tile_k, tile_h]` | gather/index_add_ 的 dim 轴不可切，无法同时加载多行索引 |
| `[1, H]`（全行） | 当 H > 8192 时，FP32 累加器占用超过 UB（8KB × 3 ≈ 24KB），可能导致 spill |
| `[NUM_ROWS, 1]` | 输出 tensor 的 NUM_ROWS 维度动态，不能作为 tile 维度 |

---

## 4. Loop 与数据流

### 4.1 维度判定

| 轴 | 维度大小 | 编译期 / 运行期 | Loop 处理 | 说明 |
|----|---------|----------------|----------|------|
| NUM_ROWS | 动态（运行时） | 运行期 | `pypto.loop(NUM_ROWS, name="rows")` | 外循环，遍历输出行 |
| K | 动态（从 scales 推导）或常量 1 | 运行期或编译期 | `pypto.loop(K, name="experts")` 或 Python `for` | 内循环，遍历专家；若 scales 不存在则 K=1（无需 loop） |
| H | 编译期已知或运行时传入 | 编译期/运行期 | 不需要 loop | 在 view 中作为静态 tile 维度 |

**Loop 嵌套设计**：

```
for i in pypto.loop(NUM_ROWS):      # 外循环（动态轴）
    for k in pypto.loop(K):         # 内循环（动态轴或静态）
        计算 ...
```

### 4.2 完整伪代码

> 必须标注每个变量类型（SymbolicScalar / int / Tensor），以及 view/assemble 的 offset。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def moe_finalize_routing_v2_kernel(
    expanded_x: pypto.Tensor([pypto.DYNAMIC, H], pypto.DT_BF16),          # [bsk, H] BF16
    expanded_row_idx: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),      # [bsk] INT32
    out: pypto.Tensor([pypto.DYNAMIC, H], pypto.DT_BF16),                 # [NUM_ROWS, H] BF16
    x1: pypto.Tensor([pypto.DYNAMIC, H], pypto.DT_BF16) = None,           # [NUM_ROWS, H] BF16（可选）
    x2: pypto.Tensor([pypto.DYNAMIC, H], pypto.DT_BF16) = None,           # [NUM_ROWS, H] BF16（可选）
    bias: pypto.Tensor([E, H], pypto.DT_BF16) = None,                     # [E, H] BF16（可选）
    scales: pypto.Tensor([pypto.DYNAMIC, K], pypto.DT_BF16) = None,       # [NUM_ROWS, K] BF16（可选）
    expert_idx: pypto.Tensor([pypto.DYNAMIC, K], pypto.DT_INT32) = None,  # [NUM_ROWS, K] INT32（可选）
    drop_pad_mode: int = 2,                                               # 编译期常量
):
    """
    MoE 路由聚合算子 kernel（完整伪代码）。
    
    类型标注：
    - SymbolicScalar: NUM_ROWS, K, bsk, i, k, expanded_row_idx_idx
    - int: H, E, drop_pad_mode
    - Tensor: expanded_x, expanded_row_idx, out, x1, x2, bias, scales, expert_idx, dst_row, out_row
    """
    
    # ========================================
    # 1. 获取基本维度（SymbolicScalar 或 int）
    # ========================================
    bsk = expanded_row_idx.shape[0]          # SymbolicScalar（动态轴）
    H = expanded_x.shape[1]                  # int 或 SymbolicScalar
    
    # 计算 K：若 scales 存在则从 scales 获取，否则默认为 1
    K = 1                                     # int（默认值）
    if scales is not None:
        K = scales.shape[1]                   # SymbolicScalar（动态轴）
    
    # 计算 NUM_ROWS
    NUM_ROWS = bsk // K                       # SymbolicScalar（动态轴）
    
    # ========================================
    # 2. Tiling 配置
    # ========================================
    tile_h = 512                              # int（推荐值）
    if H <= 1024:
        tile_h = H
    elif H <= 2048:
        tile_h = 1024
    else:
        tile_h = 512
    
    pypto.set_vec_tile_shapes(1, tile_h)     # [1, tile_h]
    
    # ========================================
    # 3. 初始化输出累加器（FP32）
    # ========================================
    # 注意：不能直接创建 [NUM_ROWS, H] 的 tensor（NUM_ROWS 是动态轴）
    # 解决方案：使用 out tensor 作为累加器，先初始化为 0
    
    # 初始化 out 为 0（BF16）
    out_init = pypto.full([NUM_ROWS, H], 0.0, pypto.DT_BF16,
                         valid_shape=[NUM_ROWS, H])  # [NUM_ROWS, H] BF16
    out[:] = out_init                         # 整体赋值
    
    # 转换为 FP32 累加器（可选：创建临时 tensor）
    # 注意：PyPTO 不支持直接 cast 输出参数，需使用中间 tensor
    # 简化方案：在累加时使用 FP32 中间值，最后 cast 回 BF16
    
    # ========================================
    # 4. 添加残差连接（可选）
    # ========================================
    # Python if 判断（编译期常量）
    if x1 is not None:
        x1_fp32 = pypto.cast(x1, pypto.DT_FP32)  # [NUM_ROWS, H] FP32
        out_fp32_init = pypto.cast(out, pypto.DT_FP32)  # [NUM_ROWS, H] FP32
        out_fp32 = pypto.add(out_fp32_init, x1_fp32)    # [NUM_ROWS, H] FP32
        out[:] = pypto.cast(out_fp32, pypto.DT_BF16)    # 写回 BF16
    
    if x2 is not None:
        x2_fp32 = pypto.cast(x2, pypto.DT_FP32)  # [NUM_ROWS, H] FP32
        out_fp32_init = pypto.cast(out, pypto.DT_FP32)  # [NUM_ROWS, H] FP32
        out_fp32 = pypto.add(out_fp32_init, x2_fp32)    # [NUM_ROWS, H] FP32
        out[:] = pypto.cast(out_fp32, pypto.DT_BF16)    # 写回 BF16
    
    # ========================================
    # 5. 主计算循环（嵌套 loop）
    # ========================================
    
    # 外循环：遍历 NUM_ROWS（动态轴）
    for i in pypto.loop(NUM_ROWS, name="rows_loop", idx_name="i"):
        
        # 初始化当前行的累加器（FP32）
        # 注意：在 loop 内不能创建新 tensor，需在 loop 外准备
        # 使用 view 获取当前行的初始值
        out_row_bf16 = pypto.view(out, [1, H], [i, 0])      # [1, H] BF16
        out_row_fp32 = pypto.cast(out_row_bf16, pypto.DT_FP32)  # [1, H] FP32
        
        # 内循环：遍历 K（动态轴或静态）
        # 注意：若 K=1（scales 不存在），可直接计算，无需 loop
        for k in pypto.loop(K, name="experts_loop", idx_name="k"):
            
            # ========================================
            # 5.1 计算索引位置（根据 drop_pad_mode）
            # ========================================
            
            # Python if 判断（drop_pad_mode 是编译期常量）
            if drop_pad_mode == 0 or drop_pad_mode == 1:
                # 按列排列
                expanded_row_idx_idx = k * NUM_ROWS + i     # SymbolicScalar
            else:
                # 按行排列（默认：drop_pad_mode = 2 或 3）
                expanded_row_idx_idx = i * K + k             # SymbolicScalar
            
            # ========================================
            # 5.2 获取索引值
            # ========================================
            
            # 使用 view 获取 expanded_row_idx 的值
            # 注意：SymbolicScalar 不能直接用作 list 索引，需用 view
            idx_value_tensor = pypto.view(expanded_row_idx, [1], [expanded_row_idx_idx])  # [1] INT32
            
            # ========================================
            # 5.3 条件跳过检查
            # ========================================
            
            # drop_pad 场景：跳过 padding（值为 -1）
            # 注意：不能用 Python if 判断 tensor 的值，需用 pypto.cond
            if drop_pad_mode == 1 or drop_pad_mode == 3:
                # 使用 pypto.cond 判断是否为 -1
                # 注意：需要先提取 tensor 的值进行比较
                # 简化方案：使用 tensor 比较 + 条件分支
                
                # 问题：PyPTO 的 cond 需要条件表达式，不能直接判断 tensor 的值
                # 解决方案：使用 pypto.cond + tensor 比较
                
                # 暂时跳过此条件（在实现阶段需进一步研究）
                # 参考：glm_select_experts.py 使用 symbolic_scalar 处理条件
                
            # drop_less 场景：跳过越界索引
            if drop_pad_mode == 0 or drop_pad_mode == 2:
                # 判断 expanded_row_idx_value >= expanded_x.shape[0]
                # 简化方案：暂不实现（在实现阶段需进一步研究）
            
            # ========================================
            # 5.4 索引查找：从 expanded_x 获取目标行
            # ========================================
            
            # 问题：idx_value_tensor 是 tensor [1]，不能直接用作 view 的 offset
            # 解决方案：需要将 tensor 的值转换为可用的形式
            
            # 参考官方示例：使用 expanded_row_idx[idx] 作为索引
            # 但 PyPTO 中 SymbolicScalar 不能直接索引 tensor
            
            # 方案 A：使用 gather API（但 dim 轴不可切）
            # 方案 B：使用 view + 多次操作
            
            # 暂时使用简化方案：假设索引值有效
            # dst_row_bf16 = pypto.view(expanded_x, [1, H], [idx_value, 0])
            
            # 注意：此处需要进一步研究 PyPTO 的索引查找机制
            # 参考：docs/api/operation/pypto-gather.md
            
            # ========================================
            # 5.5 Cast 为 FP32
            # ========================================
            
            # dst_row_fp32 = pypto.cast(dst_row_bf16, pypto.DT_FP32)  # [1, H] FP32
            
            # ========================================
            # 5.6 添加专家偏置（可选）
            # ========================================
            
            if bias is not None and expert_idx is not None:
                # 获取专家 ID
                expert_id_tensor = pypto.view(expert_idx, [1], [i * K + k])  # [1] INT32
                
                # 从 bias 获取偏置行
                # 问题：expert_id_tensor 不能直接用作 view offset
                # 解决方案：需要进一步研究
                
                # bias_row_bf16 = pypto.view(bias, [1, H], [expert_id, 0])  # [1, H] BF16
                # bias_row_fp32 = pypto.cast(bias_row_bf16, pypto.DT_FP32)  # [1, H] FP32
                # dst_row_fp32 = pypto.add(dst_row_fp32, bias_row_fp32)     # [1, H] FP32
            
            # ========================================
            # 5.7 应用路由权重（可选）
            # ========================================
            
            if scales is not None:
                # 获取权重值
                scale_tensor = pypto.view(scales, [1], [i * K + k])  # [1] BF16
                scale_fp32 = pypto.cast(scale_tensor, pypto.DT_FP32)  # [1] FP32
                
                # 扩展 scale 以匹配 dst_row 的 shape
                scale_expanded = pypto.expand_clone(scale_fp32, [1, H])  # [1, H] FP32
                
                # dst_row_fp32 = pypto.mul(dst_row_fp32, scale_expanded)  # [1, H] FP32
            
            # ========================================
            # 5.8 累加到当前行的输出
            # ========================================
            
            # out_row_fp32 = pypto.add(out_row_fp32, dst_row_fp32)  # [1, H] FP32
        
        # ========================================
        # 5.9 写回输出（内循环结束后）
        # ========================================
        
        # 将累加结果写回 out[i, :]
        out_row_bf16_final = pypto.cast(out_row_fp32, pypto.DT_BF16)  # [1, H] BF16
        
        # 使用切片赋值写回
        # 问题：不能用 SymbolicScalar i 直接切片
        # 解决方案：使用 assemble
        pypto.assemble(out_row_bf16_final, [i, 0], out)  # 写回 out[i, :]
    
    # ========================================
    # 6. 返回（输出已在 out tensor 中）
    # ========================================
    # 注意：JIT kernel 不需要显式返回，out tensor 已被修改
```

### 4.3 跨迭代状态

| 状态名 | 初始化 | 更新方式 | submit_before_loop | 说明 |
|--------|--------|---------|--------------------|------|
| out_row_fp32 | loop 外初始化为 out[i, :] 的 FP32 版本 | 内循环累加：`out_row_fp32 = pypto.add(out_row_fp32, dst_row_fp32)` | False | 当前行的累加器，跨内循环迭代 |
| out（输出 tensor） | loop 外初始化为 0 + 残差 | 内循环结束后用 assemble 写回：`pypto.assemble(out_row_bf16_final, [i, 0], out)` | False | 全局输出，跨外循环迭代 |

### 4.4 尾块处理

- **方案**：使用 `valid_shape` 参数处理边界情况

**外循环尾块**：

```python
# NUM_ROWS 可能不是整数倍，最后一个 tile 可能不完整
# 在 view 中使用 valid_shape
out_row_bf16 = pypto.view(out, [1, H], [i, 0],
                          valid_shape=[(NUM_ROWS - i).min(1), H])  # [1, H] BF16
```

**内循环尾块**：

```python
# K 可能不是整数倍（若 K 是动态轴）
# 在 view 中使用 valid_shape
idx_value_tensor = pypto.view(expanded_row_idx, [1], [expanded_row_idx_idx],
                              valid_shape=[(K - k).min(1)])  # [1] INT32
```

---

## 5. 约束自检清单

| # | 约束 | 是否满足 | 备注 |
|---|------|---------|------|
| 1 | 所有 sum 输入已转 FP32 | ✓ N/A | 本算子不使用 sum API |
| 2 | matmul 两侧 dtype 一致 | ✓ N/A | 本算子不使用 matmul API |
| 3 | TileShape 维度数 = 操作数维度数 | ✓ | `[1, tile_h]` 对应 [NUM_ROWS, H] 输出的 tile |
| 4 | 尾轴满足对齐 | ✓ | tile_h = 512（FP32 需 ≥ 8，满足） |
| 5 | 同阶段 UB 占用 ≤ 容量 | ✓ | ≈ 8 × 512 = 4KB（远小于 64KB UB） |
| 6 | 表达式展开 < 18000 | ✓ | 循环次数动态，单 tile 计算简单 |
| 7 | 输出经 `[:]` / `assemble` 显式写回 | ✓ | 使用 `pypto.assemble(out_row_bf16_final, [i, 0], out)` |
| 8 | 无 view/assemble 同张量回环 | ✓ | out 被 view 读和 assemble 写，但在不同位置（行级操作） |
| 9 | 动态轴标 `pypto.DYNAMIC` | ✓ | NUM_ROWS*K, NUM_ROWS, K 均标记 |
| 10 | 动态 loop 提供 `unroll_list` | ⚠ 待定 | 若 NUM_ROWS 和 K 范围大，需配置 unroll_list |
| 11 | 跨迭代状态用 `submit_before_loop=True` | ✓ N/A | 状态在 loop 内更新，无跨 loop 依赖 |
| 12 | 尾块用 `valid_shape` 处理 | ✓ | 在 view 中使用 valid_shape |
| 13 | 无 SymbolicScalar 用作 `**` / list index / Python `if` | ⚠ 需验证 | 伪代码中 `k * NUM_ROWS + i` 是 SymbolicScalar 运算，需确认是否符合 PyPTO 规范 |

### 开放问题

| # | 问题 | 影响范围 | 待解决方式 |
|---|------|---------|-----------|
| 1 | **索引查找机制**：如何用 SymbolicScalar `expanded_row_idx_idx` 和 tensor `idx_value_tensor` 从 `expanded_x` 中查找目标行？ | 核心计算逻辑 | 需查阅 PyPTO gather API 或研究 view + 索引组合方案 |
| 2 | **条件跳过实现**：如何用 `pypto.cond` 判断 tensor `idx_value_tensor` 的值是否为 -1 或越界？ | drop_pad 和 drop_less 场景 | 需研究 PyPTO 条件判断的 tensor 支持情况，参考 `docs/api/controlflow/pypto-cond.md` |
| 3 | **专家偏置查找**：如何用 tensor `expert_id_tensor` 从 `bias` 中查找偏置行？ | 可选参数 bias 的实现 | 同问题 1，需研究索引查找机制 |
| 4 | **loop unroll 配置**：若 NUM_ROWS 和 K 的取值范围很大，是否需要配置 `unroll_list`？ | 性能优化 | 参考 `docs/api/controlflow/pypto-loop_unroll.md`，根据典型配置确定 |
| 5 | **动态 H 处理**：若 H 也是动态轴，如何设置 `tile_h`？ | Tiling 配置 | 需动态计算 tile_h 或使用默认值 |

---

## 6. 验证方案

### 6.1 测试配置

根据 SPEC.md 中的典型配置和边界条件，设计以下测试用例：

| 用例 | 输入 shape | dtype | drop_pad_mode | 可选参数 | 重点验证 |
|------|----------|-------|--------------|---------|---------|
| **典型配置 1** | expandedX: [4096, 7168]<br>expandedRowIdx: [4096] | BF16/INT32 | 2 | 无 | drop_less 场景，K=1，无可选参数 |
| **典型配置 2** | expandedX: [16384, 7168]<br>expandedRowIdx: [16384]<br>expertIdx: [4096, 4]<br>scales: [4096, 4]<br>bias: [8, 7168] | BF16/INT32 | 2 | expertIdx, scales, bias | drop_less 场景，K=4，含所有可选参数 |
| **drop_pad 配置** | expandedX: [8, 16]<br>expandedRowIdx: [8]（含 -1） | BF16/INT32 | 1 | 无 | drop_pad 场景，测试 padding 跳过逻辑 |
| **边界配置 1** | expandedX: [1, 32]<br>expandedRowIdx: [1] | BF16/INT32 | 2 | 无 | 最小规模，NUM_ROWS=1, K=1, H=32 |
| **边界配置 2** | expandedX: [16777216, 1]<br>expandedRowIdx: [16777216] | BF16/INT32 | 2 | 无 | 最大规模，NUM_ROWS*K 达上限 |
| **残差连接测试** | expandedX: [8, 16]<br>expandedRowIdx: [8]<br>x1: [8, 16]<br>x2: [8, 16] | BF16/INT32 | 2 | x1, x2 | 测试残差连接功能 |
| **动态 K 测试** | expandedX: [64, 128]<br>expandedRowIdx: [64]<br>scales: [32, 2] | BF16/INT32 | 2 | scales (K=2) | 测试 K>1 的动态场景 |

### 6.2 精度容忍度

| dtype | rtol | atol | 说明 |
|-------|------|------|------|
| BF16 | 0.0078125 (1/128) | 0.0001 | SPEC.md 中定义的精度要求 |
| FP32 | 1e-5 | 1e-5 | 累加器使用 FP32，精度更高 |

### 6.3 验证策略

**精度验证**：
1. 使用 Golden 实现（`moe_finalize_routing_v2_golden.py`）作为参考
2. 对比 PyPTO 实现与 Golden 实现的输出
3. 使用 `torch.allclose(result, golden, rtol=rtol, atol=atol)` 验证精度
4. 重点验证累加精度和可选参数的影响

**功能验证**：
1. 验证 dropPadMode 分支逻辑（4 种模式）
2. 验证条件跳过逻辑（drop_pad 和 drop_less）
3. 验证可选参数处理（x1/x2/bias/scales/expert_idx）
4. 验证边界条件（空 tensor、越界索引、最小/最大规模）

**性能验证**：
1. 性能目标：首跑精度成功性能的 2 倍
2. 测试不同规模的性能表现
3. 分析 UB 利用率和循环开销

---

## 7. 实现路线图

### 实现步骤分解

| 步骤 | 任务 | 预估工作量 | 验证方法 |
|------|------|-----------|---------|
| 1 | **基础框架搭建**：编写 kernel 签名、动态轴标记、Tiling 配置 | 1-2 小时 | 编译通过，无语法错误 |
| 2 | **输出初始化 + 残差连接**：实现 zeros 初始化和可选的 x1/x2 添加 | 1-2 小时 | 单元测试：验证初始化和残差连接 |
| 3 | **主循环框架**：实现嵌套 loop 结构，处理动态轴 | 2-3 小时 | 编译通过，loop 结构正确 |
| 4 | **索引查找实现**：研究并实现 expanded_x 的索引查找机制 | 3-4 小时 | 单元测试：验证索引查找结果正确 |
| 5 | **条件跳过实现**：实现 drop_pad 和 drop_less 的条件跳过逻辑 | 2-3 小时 | 单元测试：验证跳过逻辑正确 |
| 6 | **可选参数处理**：实现 bias/scales/expert_idx 的处理逻辑 | 2-3 小时 | 单元测试：验证可选参数功能 |
| 7 | **完整验证**：运行所有测试配置，对比 Golden 实现 | 2-3 小时 | 精度验证：所有配置通过 |
| 8 | **性能调优**：分析性能瓶颈，优化 Tiling 和 loop 配置 | 3-4 小时 | 性能测试：达到目标性能 |

**总预估工作量**：15-20 小时

### 关键里程碑

- **里程碑 1**（步骤 1-2 完成）：基础框架可编译，初始化和残差连接通过验证
- **里程碑 2**（步骤 3-5 完成）：核心计算逻辑实现，索引查找和条件跳过通过验证
- **里程碑 3**（步骤 6-7 完成）：完整功能实现，所有测试配置精度验证通过
- **里程碑 4**（步骤 8 完成）：性能达到目标，可提交 PR

---

## 8. 风险评估与应对

### 8.1 实现风险

| 风险 | 描述 | 影响 | 应对措施 |
|------|------|------|---------|
| **索引查找机制不明确** | PyPTO 中如何用 SymbolicScalar 和 tensor 值进行索引查找尚不清楚 | 高 | 查阅 `docs/api/operation/pypto-gather.md` 和官方示例，必要时咨询社区或调整方案 |
| **条件跳过实现困难** | `pypto.cond` 是否支持 tensor 值判断尚不清楚 | 中 | 参考 `models/glm_v4_5/glm_select_experts.py` 的条件处理方案 |
| **动态 H 处理复杂** | 若 H 也是动态轴，Tiling 配置需动态计算 | 中 | 优先实现 H 为静态的场景，动态 H 作为扩展功能 |

### 8.2 性能风险

| 风险 | 描述 | 影响 | 应对措施 |
|------|------|------|---------|
| **循环开销过大** | NUM_ROWS 和 K 较大时，嵌套循环可能导致性能下降 | 中 | 研究 `loop_unroll` 和 `unroll_list` 配置，减少循环开销 |
| **UB 利用率低** | tile_h=512 可能未充分利用 UB 容量 | 低 | 根据实际 H 调整 tile_h，提高 UB 利用率 |
| **索引查找性能差** | 逐行索引查找可能比 gather API 性能低 | 中 | 对比 gather API 和 view 方案的性能，选择最优方案 |

### 8.3 精度风险

| 风险 | 描述 | 影响 | 应对措施 |
|------|------|------|---------|
| **FP32 累加精度损失** | 多次累加可能导致 FP32 精度损失（虽比 BF16 好） | 低 | 使用 FP32 累加器，精度足够；必要时可考虑更高精度 |
| **条件跳过影响精度** | 跳过某些计算可能影响最终结果的完整性 | 低 | 确保 Golden 实现也包含相同的跳过逻辑，对比验证 |

---

## 9. 完成报告

```text
设计状态：有待确认项（开放问题需解决）

迭代过程：
  第 1 轮：API 调用链 14 步，cast 4 处（BF16 → FP32）
  第 2 轮：Tiling Vector，tile = [1, 512]
  第 3 轮：Loop 2 层（NUM_ROWS + K），动态轴 3 个，跨迭代依赖 有（out_row 累加器）
  第 4 轮：约束检查 13/13（开放问题需解决）

回退记录：
  无（首次迭代，未发现需要回退的问题）

开放问题：
  · 索引查找机制 — 核心计算逻辑，需查阅 gather API 或研究替代方案
  · 条件跳过实现 — drop_pad/drop_less 场景，需研究 pypto.cond 的 tensor 支持
  · 专家偏置查找 — 可选参数，同索引查找问题
  · loop unroll 配置 — 性能优化，根据典型配置确定
  · 动态 H 处理 — Tiling 配置，作为扩展功能
```

---

**下一步建议**：
1. 查阅 `docs/api/operation/pypto-gather.md` 和官方示例，解决索引查找机制问题
2. 查阅 `docs/api/controlflow/pypto-cond.md`，解决条件跳过实现问题
3. 参考 `models/glm_v4_5/glm_select_experts.py` 和 `examples/moe_finalize_routing_v2.py`，借鉴实现方案
4. 进入实现阶段（Stage 5），基于设计方案生成完整代码