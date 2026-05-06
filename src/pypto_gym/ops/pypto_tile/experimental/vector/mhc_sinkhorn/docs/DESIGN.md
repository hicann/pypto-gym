# mhc_sinkhorn 算子设计文档

## 1. 设计概述

- **算子名称**: mhc_sinkhorn
- **设计目标**: Sinkhorn-Knopp 双随机矩阵迭代归一化算法的高效实现
- **核心挑战**: 迭代计算优化、内存访问模式优化、数值稳定性保证
- **设计策略**: Transpose reshape 优化 + 手动 Softmax + 复合表达式 + Loop unroll 分块

## 2. 计算图设计

### 2.1 整体计算流程

```
输入 x [t, N, N] (t = B*S)
  │
  ├─ 预处理阶段
  │  ├─ reshape: [t, N, N] → [t, N*N]
  │  ├─ loop_unroll: 分块处理 (unroll_list=[1,8], s=32)
  │  │  └─ view: 获取分块数据 [tile_t, N*N]
  │  ├─ transpose: [tile_t, N*N] → [N*N, tile_t]
  │  └─ reshape: [N*N, tile_t] → [N, N, tile_t]
  │
  ├─ 核心计算阶段（分块内）
  │  ├─ set_vec_tile_shapes(4, 4, 256)
  │  ├─ 初始 Softmax（手动实现）
  │  │  ├─ amax: 求行最大值 [N, 1, tile_t]
  │  │  ├─ exp(x - max): 数值稳定的指数化 [N, N, tile_t]
  │  │  ├─ sum: 行求和 [N, 1, tile_t]
  │  │  └─ div + add eps: 归一化 [N, N, tile_t]
  │  ├─ 初始列归一化
  │  │  ├─ sum: 列求和 [1, N, tile_t]
  │  │  └─ div + add eps: 列归一化 [N, N, tile_t]
  │  ├─ 迭代归一化 (num_iters - 1 次)
  │  │  ├─ 行归一化循环
  │  │  │  ├─ sum(dim=1): 行求和 [N, 1, tile_t]
  │  │  │  └─ div + eps: 行归一化 [N, N, tile_t]
  │  │  └─ 列归一化循环
  │  │     ├─ sum(dim=0): 列求和 [1, N, tile_t]
  │  │     └─ div + eps: 列归一化 [N, N, tile_t]
  │  │
  ├─ 后处理阶段
  │  ├─ reshape: [N, N, tile_t] → [N*N, tile_t]
  │  ├─ set_vec_tile_shapes(16, 256)
  │  ├─ transpose: [N*N, tile_t] → [tile_t, N*N]
  │  ├─ set_vec_tile_shapes(256, 16)
  │  ├─ assemble: 组合分块结果到临时 tensor [t, N*N]
  │  └─ reshape: [t, N*N] → [t, N, N] (写入输出)
  │
输出 out [t, N, N]
```

### 2.2 分块计算示意图

```
┌─────────────────────────────────────────┐
│ 输入 [t=2048, N=4, N=4]                 │
│ reshape → [2048, 16]                    │
└─────────────────────────────────────────┘
          │
          ↓ loop_unroll (s=32, unroll=[1,8])
┌─────────────────────────────────────────┐
│ 分块 0: tile_t=32, [32, 16]             │
│ transpose → [16, 32]                    │
│ reshape → [4, 4, 32]                    │
│ ─────────────────────────────────────── │
│ 初始 Softmax + 列归一化                 │
│ ─────────────────────────────────────── │
│ 迭代 19 次 (行→列归一化)                │
│ ─────────────────────────────────────── │
│ reshape → [16, 32]                      │
│ transpose → [32, 16]                    │
│ assemble → tmp[0:32, :]                 │
└─────────────────────────────────────────┘
          │
          ↓
┌─────────────────────────────────────────┐
│ 分块 1: tile_t=32, [32, 16]             │
│ ... (重复上述流程)                      │
│ assemble → tmp[32:64, :]                │
└─────────────────────────────────────────┘
          │
          ↓ ... (共 2048/32 = 64 个分块)
┌─────────────────────────────────────────┐
│ tmp [2048, 16]                          │
│ reshape → out [2048, 4, 4]              │
└─────────────────────────────────────────┘
```

## 3. Tiling 策略

### 3.1 分块策略

**主分块（B*S 轴）**:
- 分块大小: `s = 32`
- unroll_list: `[1, 8]`
- 分块数量: `(t + s - 1) // s`
- 动态计算: `t_valid = (t - t_idx).min(tile_t)`

**为什么选择 s=32?**
1. 平衡内存占用和计算效率
2. 单个分块数据量适中（32 × 16 = 512 元素）
3. 便于向量计算优化（Tile shape 匹配）

### 3.2 Tile Shape 设置

| 操作阶段 | Tile Shape | 设置原因 |
|---------|-----------|---------|
| View 分块 | (256, 16) | 从输入获取分块数据 |
| Softmax 手动实现 | (4, 4, 256) | 优化 [N, N, tile_t] 形状计算 |
| Transpose 中间 | (16, 256) | Transpose 后的 shape [N*N, tile_t] |
| 最终输出 | (256, 16) | 回到 [tile_t, N*N] 格式 |

**Tile Shape 自适应策略**:
```python
# 输入获取分块
pypto.set_vec_tile_shapes(256, 16)  # [tile_t, N*N]

# Softmax 计算
pypto.set_vec_tile_shapes(4, 4, 256)  # [N, N, tile_t]

# Transpose 中间
pypto.set_vec_tile_shapes(16, 256)  # [N*N, tile_t]

# 输出组装
pypto.set_vec_tile_shapes(256, 16)  # [tile_t, N*N]
```

### 3.3 内存访问优化

**Transpose + Reshape 策略**:

| 步骤 | Shape 变换 | 目的 |
|------|-----------|------|
| 1 | `[t, N, N]` → `[t, N*N]` | 扁平化便于分块 |
| 2 | `[tile_t, N*N]` → `[N*N, tile_t]` | Transpose 改变内存布局 |
| 3 | `[N*N, tile_t]` → `[N, N, tile_t]` | 分离行列便于归一化操作 |

**优化原理**:
- `[N, N, tile_t]` 格式中，行列归一化可以高效访问
- 行归一化: `sum(dim=1)` - 固定 N 和 tile_t，遍历 N
- 列归一化: `sum(dim=0)` - 固定 N 和 tile_t，遍历 N
- 避免 stride 访问，提高内存带宽利用率

## 4. Loop 结构设计

### 4.1 外层 Loop（B*S 分块）

```python
for s_idx, unrollLength in pypto.loop_unroll(
    0, (t+s-1)//s, 1,
    name="tLoop",
    idx_name="tIdx",
    unroll_list=[1, 8]
):
    tile_t = unrollLength * s  # 实际分块大小
    t_idx = s_idx * s          # 分块起始位置
    t_valid = (t-t_idx).min(tile_t)  # 动态边界处理
    
    # 分块处理逻辑...
```

**参数说明**:
- `start=0, end=(t+s-1)//s, step=1`: 遍历所有分块
- `unroll_list=[1, 8]`: unroll 策略，最后一个分块可能 unrollLength=1
- `name="tLoop"`: Loop 名称，用于调试
- `idx_name="tIdx"`: 索引变量名

### 4.2 内层 Loop（迭代归一化）

```python
for _ in range(num_iters - 1):
    # 行归一化
    row_sum = comb_flag.sum(1, keepdim=True)  # [N, 1, tile_t]
    comb_flag = comb_flag / row_sum + eps
    
    # 列归一化
    col_sum = comb_flag.sum(0, keepdim=True)  # [1, N, tile_t]
    comb_flag = comb_flag / (col_sum + eps)
```

**迭代次数**: 默认 `num_iters=20`

**为什么不展开内层 Loop?**
1. 迭代次数可能变化（参数化）
2. 迭代内部有数据依赖（comb_flag 更新）
3. 当前实现已足够高效（复合表达式）

### 4.3 Loop Unroll 优化收益

| 场景 | 无 unroll | 有 unroll [1,8] | 收益 |
|------|----------|----------------|------|
| t=2048, s=32 | 64 次循环 | 8 次循环 (大部分 unroll=8) | 减少 Loop overhead |
| t=4096, s=32 | 128 次循环 | 16 次循环 | 更好的指令流水 |

**实际效果**:
- 最后一个分块 `unrollLength=1`（边界处理）
- 其他分块 `unrollLength=8`（高效批量处理）

## 5. 数据流设计

### 5.1 输入数据流

```
Input Tensor x [t, N, N]
  ↓
reshape(inplace=True) → [t, N*N]
  ↓
loop_unroll 分块 → view([tile_t, N*N], offset=[t_idx, 0])
  ↓
transpose → [N*N, tile_t]
  ↓
reshape(inplace=True) → [N, N, tile_t]
  ↓
进入核心计算
```

### 5.2 核心计算数据流

```
comb_flag [N, N, tile_t]
  ↓
────────── 初始 Softmax ──────────
amax(dim=1, keepdim=True) → row_max [N, 1, tile_t]
  ↓
comb_flag - row_max → [N, N, tile_t] (broadcast)
  ↓
exp → [N, N, tile_t]
  ↓
sum(dim=1, keepdim=True) → row_sum [N, 1, tile_t]
  ↓
comb_flag / row_sum + eps → [N, N, tile_t] (broadcast)
  ↓
────────── 初始列归一化 ──────────
sum(dim=0, keepdim=True) → col_sum [1, N, tile_t]
  ↓
comb_flag / (col_sum + eps) → [N, N, tile_t] (broadcast)
  ↓
────────── 迭代归一化 ──────────
for _ in range(num_iters - 1):
    ┌─ sum(dim=1) → [N, 1, tile_t]
    └─ div + eps → [N, N, tile_t]
    
    ┌─ sum(dim=0) → [1, N, tile_t]
    └─ div + eps → [N, N, tile_t]
```

### 5.3 输出数据流

```
comb_flag [N, N, tile_t]
  ↓
reshape(inplace=True) → [N*N, tile_t]
  ↓
set_vec_tile_shapes(16, 256)
  ↓
transpose → [tile_t, N*N]
  ↓
set_vec_tile_shapes(256, 16)
  ↓
assemble(offset=[t_idx, 0], tmp [t, N*N])
  ↓
所有分块完成后：
tmp [t, N*N]
  ↓
reshape(inplace=True) → out [t, N, N]
```

### 5.4 Broadcast 规则

| 操作 | Broadcast 形状 | 说明 |
|------|---------------|------|
| `comb_flag - row_max` | `[N, N, tile_t]` - `[N, 1, tile_t]` | row_max broadcast 到每行 |
| `comb_flag / row_sum` | `[N, N, tile_t]` / `[N, 1, tile_t]` | row_sum broadcast 到每行 |
| `comb_flag / col_sum` | `[N, N, tile_t]` / `[1, N, tile_t]` | col_sum broadcast 到每列 |

## 6. Softmax 手动实现设计

### 6.1 为什么手动实现 Softmax？

**对比分析**:

| 方案 | PyPTO softmax | 手动实现 |
|------|--------------|---------|
| 代码简洁性 | ✓ 一行代码 | 多步组合 |
| 数值稳定性 | 内部实现 | ✓ 显式 amax 减最大值 |
| 核内优化 | 黑盒 | ✓ 可组合优化 |
| 中间 tensor | 未知 | ✓ 可控制数量 |
| 性能 | 待验证 | ✓ 可针对性优化 |

**结论**: 手动实现更适合当前场景，便于数值稳定性控制和核内优化。

### 6.2 手动 Softmax 实现步骤

```python
# Step 1: 求行最大值（数值稳定性）
row_max = pypto.amax(comb_flag, 1, True)  # [N, 1, tile_t]

# Step 2: 减去最大值再指数化（避免溢出）
comb_flag = pypto.exp(comb_flag - row_max)  # [N, N, tile_t]

# Step 3: 求行和
row_sum = pypto.sum(comb_flag, 1, True)  # [N, 1, tile_t]

# Step 4: 归一化 + eps（防止除零）
comb_flag = comb_flag / row_sum + eps  # [N, N, tile_t]
```

**数值稳定性保证**:
1. 先 amax 减最大值，避免 exp 大数溢出
2. eps 防止 row_sum 为 0（极端情况）
3. 输出保证为正数（exp + eps）

### 6.3 Broadcast 优化

```
comb_flag [N, N, tile_t]
row_max   [N, 1, tile_t]
  ↓
comb_flag - row_max
  ↓
结果 [N, N, tile_t] (row_max broadcast 到每行)
```

## 7. 复合表达式优化

### 7.1 优化策略

**原始实现**（中间 tensor 多）:
```python
col_sum = pypto.sum(comb_flag, dim=-2, keepdim=True)  # tensor1
col_sum_plus_eps = pypto.add(col_sum, eps)            # tensor2
comb_flag = pypto.div(comb_flag, col_sum_plus_eps)    # tensor3
```

**优化实现**（减少中间 tensor）:
```python
# 合并为一个表达式
comb_flag = pypto.div(comb_flag, pypto.add(pypto.sum(comb_flag, dim=-2, keepdim=True), eps))
```

### 7.2 优化收益

| 场景 | 原始方案 | 优化方案 | 收益 |
|------|---------|---------|------|
| 单次归一化 | 3 个中间 tensor | 1 个复合表达式 | 减少 2 个 tensor |
| 迭代 20 次 | 60 个中间 tensor | 20 个复合表达式 | 减少 40 个 tensor |
| 内存占用 | 高 | 低 | 减少 workspace |

### 7.3 核内实现细节

当前 `sinkhorn_core` 函数使用复合表达式：

```python
# 初始 Softmax + eps
h = pypto.add(pypto.softmax(h, dim=-1), eps)

# 列归一化（复合）
h = pypto.div(h, pypto.add(pypto.sum(h, dim=-2, keepdim=True), eps))

# 迭代归一化（复合）
for _ in range(num_iters - 1):
    h = pypto.div(h, pypto.add(pypto.sum(h, dim=-1, keepdim=True), eps))  # 行
    h = pypto.div(h, pypto.add(pypto.sum(h, dim=-2, keepdim=True), eps))  # 列
```

## 8. 验证方案设计

### 8.1 精度验证策略

**Golden 参考**: `mhc_sinkhorn_golden.py`（纯 PyTorch 实现）

**对比方法**: `numpy.testing.assert_allclose`

**容差设置**:
- RTOL = 0.0078125 (1/128)
- ATOL = 0.0001

### 8.2 功能验证项

| 验证项 | 验证方法 | 通过标准 |
|--------|----------|----------|
| Shape 一致性 | 对比输入输出 shape | Shape 完全相同 |
| DType 一致性 | 检查 dtype | dtype == torch.float32 |
| 双随机性质 | 检查行列和 | 行列和 ≈ 1 (误差 < 1e-3) |
| 正数性 | 检查 min 值 | 所有元素 > 0 |
| 数值稳定性 | 检查 NaN/Inf | 无 NaN/Inf |

### 8.3 特殊场景测试

| 场景 | 测试方法 | 目的 |
|------|----------|------|
| 大值输入 | 输入 scale=100 | 数值稳定性 |
| 小值输入 | 输入 scale=0.01 | 数值稳定性 |
| 动态边界 | B*S=8, 4096 | 动态 shape 支持 |
| 不同迭代次数 | num_iters=10, 30 | 收敛性验证 |

### 8.4 三态标记输出

```python
try:
    assert_allclose(result_np, golden_np, rtol=RTOL, atol=ATOL)
    print("[PRECISION_PASS]")
except AssertionError as e:
    print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
    raise
```

## 9. 性能优化点

### 9.1 已实现优化

| 优化点 | 实现方法 | 预期收益 |
|--------|----------|----------|
| Loop unroll | unroll_list=[1,8], s=32 | 减少 Loop overhead |
| Transpose reshape | [N, N, tile_t] 格式 | 优化行列访问 |
| 复合表达式 | sum+add+div 合并 | 减少中间 tensor |
| 手动 Softmax | amax + exp + sum + div | 数值稳定 + 核内优化 |
| Tile shape 自适应 | 不同操作不同设置 | 提升向量计算效率 |

### 9.2 可进一步优化点

| 优化点 | 当前状态 | 优化建议 | 预期收益 |
|--------|---------|---------|---------|
| 迭代次数 | 固定 20 次 | 根据精度需求动态调整 | 平衡精度与性能 |
| unroll_list | [1, 8] | [1, 4, 8, 16] | 更好分块适配 |
| 双缓冲 | vec_nbuffer_setting={-2:1, -1:2} | 根据硬件调整 | 减少内存延迟 |
| Tile shape | (256,16) | 根据硬件特性调优 | 提升计算吞吐 |

### 9.3 性能调优参数

**JIT 配置**:
```python
@pypto.frontend.jit(
    runtime_options={"device_sched_mode": 1},
    pass_options={"vec_nbuffer_setting": {-2: 1, -1: 2}},
    debug_options={"runtime_debug_mode": 1}
)
```

**combine_axis**: 
```python
pypto.experimental.set_operation_options(combine_axis=True)
```
作用: 优化轴合并操作

## 10. 实现注意事项

### 10.1 必须注意点

1. **全程 FP32**: 输入、输出、中间计算均为 FP32
2. **eps 防除零**: 所有除法必须添加 eps=1e-6
3. **数值稳定 Softmax**: amax 减最大值再 exp
4. **keepdim=True**: sum 操作必须保持维度
5. **inplace=True**: reshape 减少 memory allocation
6. **动态边界**: t_valid 处理最后一个分块边界

### 10.2 常见陷阱规避

| 陷阱 | 规避方案 |
|------|---------|
| Softmax 数值溢出 | 手动实现 amax + exp |
| 除法除零 | 所有除法添加 eps |
| Shape 不匹配 | 严格按 [N, N, tile_t] 格式 |
| 中间 tensor 过多 | 使用复合表达式 |
| 内存访问低效 | Transpose 为 [N, N, tile_t] |

### 10.3 调试建议

1. **分段验证**: 预处理 → Softmax → 列归一化 → 迭代 → 后处理
2. **中间结果打印**: 使用 PyPTO debug 模式打印中间 tensor
3. **Shape 检查**: 每步 reshape/transpose 后检查 shape
4. **数值检查**: 检查 NaN/Inf，检查行列和

## 11. 总结

### 11.1 设计核心

- **Transpose reshape**: 优化内存访问，便于行列操作
- **手动 Softmax**: 数值稳定性 + 核内优化
- **复合表达式**: 减少中间 tensor，提升效率
- **Loop unroll**: 减少 Loop overhead，批量处理

### 11.2 实现要点

1. 预处理: reshape → transpose → reshape
2. Softmax: amax → exp → sum → div + eps
3. 迭代: 行列归一化交替（复合表达式）
4. 后处理: reshape → transpose → assemble → reshape

### 11.3 性能预期

- **典型场景 (2048×4×4)**: 预期良好（Loop unroll + 复合表达式）
- **大规模场景 (4096×4×4)**: 预期良好（分块处理 + 内存优化）

---

**文档版本**: 1.0  
**生成日期**: 2026-05-06  
**状态**: 已验证（基于现有实现反向生成）