---
schema_version: 1
op_name: moe_finalize_routing_v2
supported_dtypes: [bfloat16]
p0_shapes: [[16384, 7168], [16384, 7168]]
tolerance: {atol: 0.0001, rtol: 0.0078125}
dynamic_axes: ['NUM_ROWS*K', 'E*C']
dynamic_axes_ranges: {NUM_ROWS*K: [1, 16777216], E*C: [1, 16777216], H: [1, 16384]}
shape_constraints: 'NUM_ROWS >= 1, K >= 1, E >= K, H >= 1, C >= 1 (for drop_pad mode)'
default_params: {drop_pad_mode: 2}
perf_target: '首跑精度成功性能的 2 倍'
---

## 算子需求规范

### 1. 基础信息
- **算子名称**: moe_finalize_routing_v2
- **算子分类**: custom  <!-- MoE 路由聚合算子 -->

### 1.1 功能描述

该算子用于 MoE（Mixture of Experts）模型的最终路由聚合阶段。它将多个专家网络的输出根据路由权重进行加权求和，并可选地加上残差连接和专家偏置。支持两种场景：
- **drop less 场景**：丢弃未选中的专家，只保留 K 个选中的专家输出
- **drop pad 场景**：使用 padding 填充，保留所有专家的输出（包括无效位置）

### 1.2 算法参数

- **dropPadMode**: INT64 属性，取值范围 [0, 3]
  - 0: drop less 场景，expandedRowIdx 按列排列
  - 1: drop pad 场景，expandedRowIdx 按列排列
  - 2: drop less 场景，expandedRowIdx 按行排列
  - 3: drop pad 场景，expandedRowIdx 按行排列

### 1.3 数学公式

$$
expertid = expertIdx[i,k]
$$

$$
out(i,j) = x1_{i,j} + x2_{i,j} + \sum_{k=0}^{K}(scales_{i,k} * (expandedX_{expandedRowIdx_{i+k*num\_rows},j} + bias_{expertid,j}))
$$

### 2. 关键特性

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| 动态 Shape | ✓ 需要 | ✓ 高 | expandedX 第一维度为动态轴 | P0 |
| 可选参数处理 | ✓ 需要 | ✓ 高 | 多个可选输入，需条件分支 | P0 |
| 条件计算 | ✓ 需要 | ✓ 高 | 根据 dropPadMode 和 expandedRowIdx 值跳过计算 | P0 |
| 循环结构 | ✓ 需要 | ✓ 高 | 遍历 num_rows 和 K | P0 |
| 索引查找 | ✓ 需要 | ✓ 高 | 通过 expandedRowIdx 查找 expandedX 中的行 | P1 |

### 3. 算法描述

```
Algorithm: MoE Finalize Routing V2
───────────────────────────────────
输入: expandedX, expandedRowIdx, x1(可选), x2(可选), bias(可选), scales(可选), expertIdx(可选)
输出: out [num_rows, H]

1. 初始化 out = zeros(num_rows, H)
2. if x1 存在: out = out + x1
3. if x2 存在: out = out + x2
4. for i in range(num_rows):
     for k in range(K):
       4.1 根据 dropPadMode 计算 expanded_row_idx_idx
           - dropPadMode = 0 或 1: expanded_row_idx_idx = k * num_rows + i (按列)
           - dropPadMode = 2 或 3: expanded_row_idx_idx = i * K + k (按行)
       4.2 获取 expanded_row_idx_value = expandedRowIdx[expanded_row_idx_idx]
       4.3 条件跳过检查:
           - drop_pad 场景(mode=1或3) 且值为 -1: continue (跳过 padding)
           - drop_less 场景(mode=0或2) 且值 >= expandedX.shape[0]: continue (跳过越界)
       4.4 dst_row = expandedX[expanded_row_idx_value, :].astype(float32)
       4.5 if bias 和 expertIdx 存在: dst_row += bias[expertIdx[i,k], :].astype(float32)
       4.6 if scales 存在: dst_row *= scales[i,k].astype(float32)
       4.7 out[i,:] += dst_row
5. return out.astype(bfloat16)
```

### 4. 数据流图

```
   输入 expandedX          输入 expandedRowIdx      可选输入 x1/x2/bias/scales/expertIdx
  ┌──────────────┐        ┌──────────────┐        ┌────────────────────────┐
  │ (NUM_ROWS*K, H)│       │ (NUM_ROWS*K) │        │ x1: (NUM_ROWS, H)       │
  │  或 (E, C, H) │        │    INT32     │        │ x2: (NUM_ROWS, H)       │
  │   BFLOAT16    │        └──────┬───────┘        │ bias: (E, H)            │
  └───────┬──────┘               │                │ scales: (NUM_ROWS, K)  │
          │                      │                │ expertIdx: (NUM_ROWS, K)│
          │                      │                └────────────┬───────────┘
          │                      │                             │
          │                      │                             │
          └──────────────────────┼─────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   主计算逻辑 (循环)      │
                    │  for i in num_rows:     │
                    │    for k in K:          │
                    │      - 索引查找         │
                    │      - 条件跳过         │
                    │      - 加权求和         │
                    └────────────┬───────────┘
                                 │
                                 ▼
                         ┌──────────────┐
                         │   输出 out    │
                         │ (NUM_ROWS, H) │
                         │   BFLOAT16    │
                         └──────────────┘
```

### 5. 输入输出规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 置信度 | 说明 | 优先级 |
|------|-------|-------|--------|--------|------|--------|
| expandedX | (NUM_ROWS\*K, H) 或 (E, C, H) | BFLOAT16 | 第一维度 | ✓ 高 | MoE的FFN输出 | P0 |
| expandedRowIdx | (NUM_ROWS\*K) | INT32 | 整个张量 | ✓ 高 | 行索引，用于查找 expandedX 中的行 | P0 |
| x1Optional | (NUM_ROWS, H) | BFLOAT16 | 第一维度 | ✓ 高 | 残差连接1，可选输入 | P2 |
| x2Optional | (NUM_ROWS, H) | BFLOAT16 | 第一维度 | ✓ 高 | 残差连接2，可选输入 | P2 |
| biasOptional | (E, H) | BFLOAT16 | 第一维度 | ✓ 高 | 专家偏置，可选输入 | P1 |
| scalesOptional | (NUM_ROWS, K) | BFLOAT16 | 第一维度 | ✓ 高 | 路由权重，可选输入 | P1 |
| expertIdxOptional | (NUM_ROWS, K) | INT32 | 第一维度 | ✓ 高 | 专家索引，可选输入 | P1 |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------|-------|--------|--------|------|
| out | (NUM_ROWS, H) | BFLOAT16 | 第一维度 | ✓ 高 | 聚合后的输出结果 |

**属性**:

| 属性 | Dtype | 取值范围 | 说明 |
|------|-------|----------|------|
| dropPadMode | INT64 | [0, 3] | 控制扩展行索引的排列方式 |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| bfloat16 | ✓ | 0.0001 | 0.0078125 | 主要支持类型 |

### 7. 精度要求
- **atol**: 0.0001
- **rtol**: 0.0078125

### 8. 动态轴说明
- **动态轴**: ['NUM_ROWS*K', 'E*C']  <!-- expandedX 的第一维度 -->
- **轴含义**: 
  - NUM_ROWS*K: 行数乘以每个样本选择的专家数（drop less 场景）
  - E*C: 专家数乘以容量（drop pad 场景）
  - H: hidden size，每个 token 的特征维度（静态轴）
- **取值范围**: 
  - NUM_ROWS*K: [1, 16777216]
  - E*C: [1, 16777216]
  - H: [1, 16384]

### 9. 边界条件处理
- **零值**: normal (正常计算)
- **极值**: normal (正常计算)
- **NaN/Inf**: normal (正常计算)
- **特殊索引**: 
  - drop_pad 场景下，expandedRowIdx 值为 -1 时跳过该位置
  - drop_less 场景下，expandedRowIdx 值越界时跳过该位置

### 10. 性能要求
- **性能目标**: 首跑精度成功性能的 2 倍

### 11. 参考信息
- **参考实现**: 已提供 NumPy Golden 实现（见原始需求文档）
- **论文**: MoE 相关论文
- **类似算子**: moe_init_routing, moe_init_routing_v2

### 12. 应用场景
- **目标模型**: GLM, LLaMA, Qwen 等 MoE 模型
- **使用位置**: MoE 层的路由聚合阶段

**典型配置**（建议至少提供一个，用于下游 golden 验证和设计方案生成）:

| 配置名称 | 类型 | 优先级 | drop_pad_mode | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|---------------|------------|------------|------|
| 配置1_drop_less_basic | 功能 | P0 | 2 | expandedX: [16384, 7168]<br>expandedRowIdx: [16384] | [4096, 7168] | drop_less场景，无可选参数，K=4 |
| 配置2_drop_less_expert | 功能 | P0 | 2 | expandedX: [16384, 7168]<br>expandedRowIdx: [16384]<br>expertIdx: [4096, 4] | [4096, 7168] | drop_less场景，含可选参数expertIdx，K=4 |

**约束说明**:
1. NUM_ROWS 表示行数；K 表示从总的专家 E 中选出 K 个专家；H 表示 hidden size；E 表示专家数，E ≥ K；C 表示专家容量
2. expandedRowIdx 取值范围：
   - dropPadMode = 0 或 2 时：[0, NUM_ROWS\*K - 1]
   - dropPadMode = 1 或 3 时：[-1, E\*C - 1]
3. x1Optional 未输入时，x2Optional 也不能输入
4. scalesOptional 不存在时，K = 1
5. biasOptional 存在时，expertIdxOptional 必须同时存在

---
*生成时间: 2026-05-14*
*确认状态: 已确认*
*置信度说明: ✓ 高（基于需求文档提取）*