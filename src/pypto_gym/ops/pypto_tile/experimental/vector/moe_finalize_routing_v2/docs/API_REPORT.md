---
schema_version: 1
op_name: moe_finalize_routing_v2
supported_dtypes: [bfloat16, int32, bool]
dynamic_axes: ['NUM_ROWS*K', 'E*C', 'NUM_ROWS']
shape_constraints: 'NUM_ROWS >= 1, K >= 1, E >= K, H >= 1, C >= 1 (for drop_pad mode)'
tiling_required: 'Vector Tiling (必需)'
feasibility: '可行'
---

# API 探索报告

> **生成时间**: 2026-05-14

---

## 1. 概述

### 1.1 输入摘要

**算子名称**: moe_finalize_routing_v2  
**数学公式**:
```
expertid = expertIdx[i,k]
out(i,j) = x1_{i,j} + x2_{i,j} + Σ_{k=0}^{K}(scales_{i,k} * (expandedX_{expandedRowIdx_{i+k*num_rows},j} + bias_{expertid,j}))
```

**核心计算逻辑**:
- 索引查找：通过 expandedRowIdx 查找 expandedX 中的行
- 加权求和：dst_row * scales + 累加到输出
- 条件跳过：根据 dropPadMode 和 expandedRowIdx 值跳过计算
- 可选参数处理：x1/x2/bias/scales/expertIdx 条件调用

### 1.2 算子分类

- **类型**: Vector 类型
- **判断依据**: 不涉及矩阵乘法，仅包含索引查找、逐元素加法、乘法和累加操作，属于 Vector 操作类型

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 | 优先级 |
|------|----------|----------|------|--------|
| 1 | index | `expandedX[expandedRowIdx[idx], :]` | 索引查找专家输出 | P0 |
| 2 | elementwise | `dst_row * scales[i,k]` | 加权计算 | P0 |
| 3 | elementwise | `out[i,:] + dst_row` | 累加到输出 | P0 |
| 4 | elementwise | `dst_row + bias[expertId,:]` | 可选：添加专家偏置 | P1 |
| 5 | elementwise | `out + x1 + x2` | 可选：添加残差连接 | P2 |
| 6 | condition | `if expandedRowIdx[idx] == -1: continue` | drop_pad 模式跳过 | P0 |
| 7 | condition | `if dropPadMode == 0/2: idx = k*num_rows + i else: idx = i*K + k` | dropPadMode 分支 | P0 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 | 置信度 |
|------|----------|-----------|----------|----------|--------|
| 索引查找 | `expandedX[expandedRowIdx[idx], :]` | `pypto.gather(expandedX, 0, expandedRowIdx)` | direct | ✓ 完全满足 | 高 |
| | | `expandedX[expandedRowIdx[idx], :]` (切片) | direct | ✓ 完全满足 | 高 |
| 加权计算 | `dst_row * scales[i,k]` | `pypto.mul(dst_row, scales)` | direct | ✓ 完全满足 | 高 |
| 累加 | `out[i,:] + dst_row` | `pypto.add(out, dst_row)` | direct | ✓ 完全满足 | 高 |
| | | `pypto.index_add_(out, 0, idx, dst_row, alpha=1.0)` | direct | ✓ 完全满足 | 高 |
| 专家偏置 | `dst_row + bias[expertId,:]` | `pypto.add(dst_row, bias_row)` | direct | ✓ 支持广播 | 高 |
| 残差连接 | `out + x1 + x2` | `pypto.add(pypto.add(out, x1), x2)` | direct | ✓ 完全满足 | 高 |
| 输出初始化 | `zeros([num_rows, H])` | `pypto.zeros([num_rows, H], DT_BF16)` | direct | ✓ 支持动态 shape | 高 |
| 条件跳过 | `if value == -1: continue` | `if pypto.cond(expandedRowIdx[idx] >= 0):` | direct | ✓ 完全满足 | 高 |
| dropPadMode 分支 | `if dropPadMode == 0: ...` | `if pypto.cond(dropPadMode == 0):` | direct | ✓ 完全满足 | 高 |
| 循环遍历 | `for i in range(num_rows):` | `for i in pypto.loop(num_rows):` | direct | ✓ 支持动态轴 | 高 |

### 3.2 Substitute 配方

无 substitute 操作，所有计算步骤均有直接 API 支持。

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 | 来源文档 |
|--------|------|--------|------|----------|
| dtype 支持 | BFLOAT16/INT32 | expandedX: BFLOAT16, expandedRowIdx: INT32 | ✓ | pypto-from_torch.md L41-53 |
| contiguous | 必须连续 | torch tensor.is_contiguous() == True | ⚠ 需确保 | pypto-from_torch.md L39 |
| 动态轴标记 | 需标记动态维度 | expandedX 第一维度动态 | ✓ 可标记 | pypto-from_torch.md L28 |
| shape 非空 | 不支持空 Tensor | shape >= [1, 1] | ✓ | pypto-from_torch.md L38 |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 | 来源文档 |
|-----|--------|------|------|----------|
| `gather` | dtype | 支持 BFLOAT16 输入，INT32 索引 | ✓ | pypto-gather.md |
| | 动态 shape | 支持动态 shape | ✓ | pypto-gather.md L44-50 |
| | dim 轴 | dim 轴不可切，需全载 | ⚠ 需注意 | pypto-gather.md L50-64 |
| `add`, `mul` | dtype | 支持 BFLOAT16 | ✓ | pypto-add.md, pypto-mul.md |
| | 广播 | 支持单轴广播 | ✓ | pypto-add.md L52-54 |
| | NaN/Inf | 不支持 nan、inf | ⚠ 需前置检查 | pypto-add.md L42 |
| `zeros`, `full` | dtype | 支持 BFLOAT16 | ✓ | pypto-zeros.md, pypto-full.md |
| | 动态 shape | 支持 SymbolicScalar 构建 | ✓ | pypto-zeros.md |
| `cond` | 条件表达式 | 支持动态条件 | ✓ | pypto-cond.md |
| `loop` | 动态轴 | 支持动态循环次数 | ✓ | pypto-loop.md |

### 4.3 Tiling 约束

| API | TileShape 约束 | 关键约束 | 来源文档 |
|-----|----------------|---------|----------|
| `gather` | TileShape 维度与 index 相同 | dim 轴不可切，总和不超过 UB | pypto-gather.md L50-64 |
| `index_add_` | TileShape 维度与 source 相同 | dim 轴不可切，保证全载 | pypto-index_add_.md L50-57 |
| `add`, `mul` | TileShape 维度与输出相同 | 支持广播，无特殊约束 | pypto-add.md L48-57 |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API | 说明 |
|----------|-----------|------|
| Vector | `pypto.set_vec_tile_shapes(tile_m, tile_n, tile_h)` | 必需，需根据动态轴范围配置 TileShape |

**关键约束**:
```python
# 索引操作的 dim 轴约束：
# - gather: dim 轴（行索引轴）不可切，必须全载
# - index_add_: dim 轴不可切，保证全载
# 建议：设置合理的 tile_h（H 维度），避免 UB 内存压力
pypto.set_vec_tile_shapes(tile_rows=1, tile_k=1, tile_h=512)  # 根据实际 shape 调整
```

---

## 6. 参考实现

### 6.1 匹配示例

**Top 1: 官方完全匹配示例**

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/moe_finalize_routing_v2.py` | examples | **高（完全匹配）** | 高 | 完整的核心计算逻辑、索引查找、条件跳过、加权求和、可选参数处理、dropPadMode 分支 |

**Top 2-3: 生产级 MoE 参考实现**

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `models/glm_v4_5/glm_select_experts.py` | models（生产） | 高 | 高 | Gather + 加权求和模式、条件处理、动态 Shape 处理、Loop 结构 |
| `models/glm_v4_5/glm_moe_fusion.py` | models（生产） | 高 | 高 | 完整 MoE 流程、Loop Unroll 策略、累加写入模式、高性能动态轴处理 |
| `models/deepseek_v32_exp/sparse_flash_attention_quant_impl.py` | models（生产） | 中 | 高 | gather_in_ub 高级 API、索引 + Block Table 模式、动态 Tiling 配置 |

**Top 4-5: 官方示例实现**

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/01_beginner/transform/transform_ops.py` | examples | 中 | 高 | gather API 调用方式、Tiling 配置、动态维度标记 |
| `examples/02_intermediate/controlflow/condition/condition.py` | examples | 中 | 高 | 动态条件判断、循环边界条件、嵌套循环 + 条件分支 |

### 6.2 可复用模式

- **API 调用模式**:
  - 索引查找：`expandedX[expandedRowIdx[idx], :]` 或 `pypto.gather(expandedX, 0, expandedRowIdx)`
  - 加权求和：`pypto.mul(dst_row, scales)` + `pypto.add(out, dst_row)`
  - 条件跳过：`if pypto.cond(expandedRowIdx[idx] >= 0):`
  
- **Tiling 策略**:
  - Vector Tiling：`pypto.set_vec_tile_shapes(tile_rows=1, tile_k=1, tile_h=512)`
  - 关键约束：dim 轴不可切，需全载
  
- **Loop 结构**:
  - 外循环：`for i in pypto.loop(num_rows)`（遍历行）
  - 内循环：`for k in pypto.loop(K)`（遍历 expert）
  - 动态 Loop：`bs_loop = (bs + view_shape[0] - 1) // view_shape[0]`
  
- **边界处理**:
  - dropPadMode 分支：`if pypto.cond(dropPadMode == 0): expanded_row_idx_idx = k * num_rows + i`
  - 条件跳过：`if expanded_row_idx_value == -1: continue`
  - 可选参数：`if bias is not None and expert_idx is not None:`

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| 索引查找 | `expanded_x[expanded_row_idx[idx], :]`（直接切片） | PyPTO 需支持动态索引 | 可直接使用切片语法，或使用 `pypto.gather` |
| 条件跳过 | `if expanded_row_idx_value == -1: continue`（Python 条件） | PyPTO 需支持运行时条件 | 使用 `if pypto.cond(condition):` 实现动态条件 |
| 动态 shape | NumPy 固定 shape | PyPTO 需处理动态轴 | 使用 `pypto.view` + `valid_shape` 处理边界 |
| Tiling | NumPy 无 Tiling | PyPTO 必需 Vector Tiling | 需设置合理 TileShape，注意 dim 轴不可切约束 |

---

## 7. 风险评估

### 7.1 阻断问题

无阻断问题，所有核心 API 都有直接支持。

### 7.2 注意事项

| 注意点 | 说明 | 建议 |
|--------|------|------|
| **动态轴标记不完整** | expandedRowIdx 整个张量动态，需确保所有轴正确标记 | 在 `from_torch` 中明确标记所有动态轴 |
| **Tiling dim 轴约束** | gather/index_add_ 的 dim 轴不可切，需全载，可能导致 UB 内存压力 | 设置合理的 tile_h，避免单次加载过多数据 |
| **NaN/Inf 不支持** | PyPTO add/mul 不支持 NaN/Inf 输入 | 在前置处理中过滤特殊值，或在测试中避免生成 NaN/Inf |
| **contiguous 要求** | from_torch 要求输入 tensor 必须连续 | 在测试中确保输入 tensor.is_contiguous() == True |
| **循环边界处理** | 动态轴可能导致最后一个 tile 不完整 | 使用 `valid_shape` 参数处理边界情况 |

### 7.3 高风险项详细说明

#### 风险 1：动态轴标记不完整

**问题描述**: expandedRowIdx 整个张量动态（shape 为 [NUM_ROWS*K]，整个张量都需要标记为动态），若标记不完整可能导致编译期 shape 推导失败。

**解决方案**:
```python
expandedRowIdx_pto = pypto.from_torch(
    expandedRowIdx_torch,
    name="expandedRowIdx",
    dynamic_axis=[0]  # 第一维度动态（整个张量）
)
```

#### 风险 2：Tiling dim 轴约束

**问题描述**: gather 和 index_add_ 的 dim 轴（行索引轴）不可切，必须全载。若 dim 轴过大（如 NUM_ROWS*K 很大），可能导致 UB 内存压力。

**解决方案**:
- 设置合理的 tile_h（H 维度），减少单次加载的数据量
- 若 NUM_ROWS*K 过大，可考虑分批处理（牺牲性能）

#### 风险 3：NaN/Inf 不支持

**问题描述**: PyPTO 的 add/mul 不支持 NaN/Inf 输入，若输入包含特殊值可能导致运行时错误。

**解决方案**:
- 在前置处理中检查并过滤 NaN/Inf
- 在测试中避免生成包含 NaN/Inf 的输入

---

## 8. 证据索引

| 信息 | 文档路径 | 关键内容 |
|------|----------|----------|
| API 存在性 | `docs/api/operation/index.md` | 所有支持的操作列表 |
| gather 文档 | `docs/api/operation/pypto-gather.md` | 索引查找，动态 shape，BFLOAT16 支持，dim 轴约束 |
| index_add_ 文档 | `docs/api/operation/pypto-index_add_.md` | 索引累加，alpha 缩放因子，dim 轴约束 |
| add 文档 | `docs/api/operation/pypto-add.md` | 逐元素加法，广播支持，NaN/Inf 约束 |
| mul 文档 | `docs/api/operation/pypto-mul.md` | 逐元素乘法，广播支持，NaN/Inf 约束 |
| zeros 文档 | `docs/api/operation/pypto-zeros.md` | 零张量初始化，动态 shape 支持 |
| full 文档 | `docs/api/operation/pypto-full.md` | 填充张量初始化，动态 shape 支持 |
| cond 文档 | `docs/api/controlflow/pypto-cond.md` | 条件分支控制，动态条件支持 |
| loop 文档 | `docs/api/controlflow/pypto-loop.md` | 循环操作定义，动态轴支持 |
| DataType 文档 | `docs/api/datatype/DataType.md` | DataType 枚举，DT_BF16, DT_INT32 定义 |
| 入口约束 | `docs/api/others/pypto-from_torch.md` | Torch Tensor 转换，动态轴标记，dtype 约束，contiguous 要求 |
| Vector Tiling | `docs/api/config/pypto-set_vec_tile_shapes.md` | Vector Tiling 设置，TileShape 约束 |
| 官方参考实现 | `examples/moe_finalize_routing_v2.py` | 完整的 moe_finalize_routing_v2 实现，核心计算逻辑 |
| 生产级参考1 | `models/glm_v4_5/glm_select_experts.py` | Gather + 加权求和模式，条件处理 |
| 生产级参考2 | `models/glm_v4_5/glm_moe_fusion.py` | 完整 MoE 流程，Loop Unroll 策略 |

---

## 9. 结论

- **可行性**: ✓ **可行**
- **主要发现**: 
  1. 所有核心计算步骤均有直接 API 支持（gather, add, mul, zeros, cond, loop）
  2. 找到**完全匹配**的官方示例 `examples/moe_finalize_routing_v2.py`
  3. 找到多个高相似度生产级参考实现（glm_select_experts.py, glm_moe_fusion.py）
  4. dtype 约束完全满足（BFLOAT16, INT32）
  5. 动态 shape 支持完整，但需注意 Tiling dim 轴约束
  6. 无阻断问题，可直接进入实现阶段

- **推荐实现路径**:
  1. 直接参考 `examples/moe_finalize_routing_v2.py` 的核心逻辑
  2. 参考 `models/glm_v4_5/glm_select_experts.py` 的 Gather + 加权求和模式
  3. 参考 `examples/02_intermediate/controlflow/dynamic.py` 的动态 shape 处理
  4. 注意 Tiling dim 轴约束和动态轴标记完整性

- **预期难点**:
  1. Tiling 配置：需合理设置 tile_h，避免 UB 内存压力
  2. 动态轴处理：需确保所有动态轴正确标记，使用 valid_shape 处理边界
  3. 条件分支：需正确使用 pypto.cond 实现运行时条件判断

---

**下一步建议**: 直接进入 Stage 4（Design 设计），基于官方参考实现和 API 映射结果生成完整的设计方案。