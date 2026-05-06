# mhc_post 算子设计文档

## 1. 设计概述

### 1.1 算子定位
`mhc_post` 是 MHC (Manifold-Constrained Hyper-Connections) 系统的后处理融合算子，用于注意力机制中的流间混合计算。

### 1.2 核心特性
- **纯向量运算**：无矩阵乘法，纯逐元素操作
- **动态 Shape 支持**：B*S 维度支持动态变化
- **精度转换路径**：BF16 输入 → FP32 计算 → BF16 输出
- **循环展开优化**：BS 轴展开，提升并行度

---

## 2. 计算图设计

### 2.1 整体计算流程

```
输入: x[B*S, N, D] (BF16)
     h_res[B*S, N, N] (FP32)
     h_out[B*S, D] (BF16)
     h_post[B*S, N] (FP32)

Step 1: 类型转换 (BF16 → FP32)
  ├─ x_fp32 = cast(x, FP32)         [B*S, N, D] FP32
  └─ h_out_fp32 = cast(h_out, FP32) [B*S, D] FP32

Step 2: 维度扩展 (reshape 实现 unsqueeze)
  ├─ h_post_1 = reshape(h_post, [BS, N, 1])
  ├─ h_out_1 = reshape(h_out_fp32, [BS, 1, D])
  ├─ h_res_1 = reshape(h_res, [BS, N, N, 1])
  └─ x_1 = reshape(x_fp32, [BS, N, 1, D])

Step 3: 计算 h_post_term
  └─ h_post_term = mul(h_post_1, h_out_1)  [B*S, N, D]

Step 4: 计算 h_comb_term
  ├─ weighted = mul(h_res_1, x_1)         [B*S, N, N, D]
  └─ h_comb_term = sum(weighted, dim=1)  [B*S, N, D]

Step 5: 融合输出
  ├─ result_fp32 = add(h_post_term, h_comb_term)  [B*S, N, D]
  └─ output = cast(result_fp32, BF16)             [B*S, N, D]

输出: output[B*S, N, D] (BF16)
```

### 2.2 计算图可视化

```
    x(BF16)          h_res(FP32)       h_out(BF16)        h_post(FP32)
       │                  │                 │                  │
       ↓                  │                 ↓                  │
   cast(FP32)             │             cast(FP32)             │
       │                  │                 │                  │
       ↓                  ↓                 ↓                  ↓
  reshape            reshape           reshape            reshape
   [BS,N,1,D]        [BS,N,N,1]        [BS,1,D]           [BS,N,1]
       │                  │                 │                  │
       │                  │                 │                  │
       └─── x_1 ──────────┤                 └─── h_out_1 ──────┤
                          │                                    │
                          ↓                                    ↓
                     mul(x_1, h_res_1)                   mul(h_post_1, h_out_1)
                          │                                    │
                          ↓                                    ↓
                      weighted                              h_post_term
                     [BS,N,N,D]                             [BS,N,D]
                          │                                    │
                          ↓                                    │
                     sum(dim=1)                                │
                          │                                    │
                          ↓                                    │
                      h_comb_term                              │
                        [BS,N,D]                               │
                          │                                    │
                          └───────────┬────────────────────────┘
                                      │
                                      ↓
                                 add()
                                      │
                                      ↓
                                 result_fp32
                                      │
                                      ↓
                                 cast(BF16)
                                      │
                                      ↓
                                   output
                                 [BS,N,D] BF16
```

**数据流说明**：

| 分支 | 输入 | 计算 | 输出 | Shape 变化 |
|------|------|------|------|-----------|
| **分支1** | x(BF16) → cast → reshape | × h_res(FP32) → reshape | weighted [BS,N,N,D] | [BS,N,D] × [BS,N,N] → [BS,N,N,D] |
| **分支2** | h_out(BF16) → cast → reshape | × h_post(FP32) → reshape | h_post_term [BS,N,D] | [BS,D] × [BS,N] → [BS,N,D] |
| **归约** | weighted [BS,N,N,D] | sum(dim=1) | h_comb_term [BS,N,D] | [BS,N,N,D] → [BS,N,D] |
| **融合** | h_comb_term + h_post_term | add() → cast(BF16) | output [BS,N,D] | 两分支合并 |

---

## 3. Tiling 策略

### 3.1 循环结构

**BS 轴展开**：
```python
for bs_idx, unroll_length in pypto.loop_unroll(0, BS, 1, name="LOOP_BS", unroll_list=[128]):
    # 处理 bs_idx : bs_idx + unroll_length 切片
```

**展开策略**：
- `unroll_list=[128]`：每次处理 128 个样本
- BS 为动态轴，循环次数运行时确定
- 循环展开提升指令级并行

### 3.2 向量化分块

**分块配置**：

| 操作 | Tile Shape | 说明 |
|------|-----------|------|
| x 类型转换 | `(1, N, 1, 1280)` | 尾轴分块，优化内存访问 |
| h_out 类型转换 | `(1, N, 1280)` | 2D 分块 |
| h_post_term 计算 | `(1, N, 1280)` | 2D 分块 |
| weighted 计算 | `(1, N, N, 1280)` | 3D 分块，N=4 固定 |
| result 计算 | `(1, N, 1280)` | 2D 分块 |

**分块原理**：
- **尾轴分块**：避免跨 cache line 访问
- **固定 N 维**：N=4 硬编码，编译期优化
- **动态 D 维**：D 使用 STATIC 标记，变化触发重编译

### 3.3 内存访问模式

| Tensor | Shape | 访问模式 | Tile 优化 |
|--------|-------|---------|----------|
| x | [BS, N, D] | 顺序访问 | 尾轴分块 |
| h_res | [BS, N, N] | 顺序访问 | 固定 N=4 |
| h_out | [BS, D] | 顺序访问 | 2D 分块 |
| h_post | [BS, N] | 顺序访问 | 固定 N=4 |
| output | [BS, N, D] | 顺序访问 | 尾轴分块 |

**内存访问优化**：
- 所有 tensor 按 C-order 连续存储
- 顺序访问模式，利于 cache 预取
- tile 大小选择考虑 L1/L2 cache 容量

---

## 4. Loop 结构设计

### 4.1 外层循环（BS 轴）

```python
for bs_idx, unroll_length in pypto.loop_unroll(0, BS, 1, name="LOOP_BS", unroll_list=[128]):
    # 切片
    x_slice = x_reshaped[bs_idx: bs_idx + unroll_length, :, :, :]
    h_res_slice = h_res_reshaped[bs_idx: bs_idx + unroll_length, :, :, :]
    h_out_slice = h_out_reshaped[bs_idx: bs_idx + unroll_length, :, :]
    h_post_slice = h_post_reshaped[bs_idx: bs_idx + unroll_length, :, :]
    
    # 计算（见 4.2 内层计算）
    # ...
    
    # 组装结果
    pypto.assemble(result_bf16, [bs_idx, 0, 0], output)
```

**循环参数**：
- `start=0, stop=BS, step=1`：遍历 BS 轴
- `unroll_list=[128]`：展开长度为 128
- `idx_name="bs_idx"`：循环变量名

### 4.2 内层计算（单个 tile）

```python
# Step 1: 类型转换 BF16 → FP32
pypto.set_vec_tile_shapes(1, N, 1, 1280)
x_fp32 = pypto.cast(x_slice, pypto.DT_FP32)

pypto.set_vec_tile_shapes(1, N, 1280)
h_out_fp32 = pypto.cast(h_out_slice, pypto.DT_FP32)

# Step 2: 计算 h_post_term
h_post_term = pypto.mul(h_post_slice, h_out_fp32)

# Step 3: 计算 weighted 和 h_comb_term
pypto.set_vec_tile_shapes(1, N, N, 1280)
weighted = pypto.mul(h_res_slice, x_fp32)
h_comb_term = pypto.sum(weighted, dim=1, keepdim=False)

# Step 4: 融合并转换回 BF16
pypto.set_vec_tile_shapes(1, N, 1280)
result_fp32 = pypto.add(h_post_term, h_comb_term)
result_bf16 = pypto.cast(result_fp32, pypto.DT_BF16)
```

**计算优化**：
- 每个操作前设置对应的 tile shape
- FP32 中间结果保证精度
- BF16 输出减少内存占用

---

## 5. 数据流设计

### 5.1 输入预处理

**Wrapper 函数职责**：
```python
def mhc_post_wrapper(x, h_res, h_out, h_post, output=None):
    # 1. 验证输入
    assert x.is_contiguous()
    assert h_res.is_contiguous()
    assert h_out.is_contiguous()
    assert h_post.is_contiguous()
    
    # 2. 提取 shape
    B, S, N, D = x.shape
    assert N == 4
    
    # 3. Reshape [B, S, ...] → [B*S, ...]
    BS = B * S
    x_reshaped = x.view(BS, N, D).contiguous()
    h_res_reshaped = h_res.view(BS, N, N).contiguous()
    h_out_reshaped = h_out.view(BS, D).contiguous()
    h_post_reshaped = h_post.view(BS, N).contiguous()
    
    # 4. 调用 kernel
    mhc_post_kernel_bf16(x_reshaped, h_res_reshaped, h_out_reshaped, h_post_reshaped, output_reshaped)
    
    # 5. Reshape [B*S, ...] → [B, S, ...]
    return output_reshaped.view(B, S, N, D)
```

### 5.2 中间数据流

```
输入切片 [unroll_length, ...]
    ↓
类型转换 BF16 → FP32
    ↓
维度扩展 (reshape)
    ↓
广播乘法 (mul)
    ↓
归约求和 (sum)
    ↓
逐元素加法 (add)
    ↓
类型转换 FP32 → BF16
    ↓
组装到输出 tensor
```

### 5.3 输出组装

```python
pypto.assemble(result_bf16, [bs_idx, 0, 0], output)
```

**组装参数**：
- `result_bf16`：当前 tile 计算结果
- `[bs_idx, 0, 0]`：输出 tensor 起始位置
- `output`：输出 tensor

---

## 6. 精度设计

### 6.1 精度转换路径

```
输入精度: BF16 (x, h_out) + FP32 (h_res, h_post)
    ↓
计算精度: FP32 (所有中间计算)
    ↓
输出精度: BF16 (output)
```

### 6.2 精度保证措施

| 措施 | 说明 |
|------|------|
| 中间计算使用 FP32 | 避免累加误差 |
| sum 操作在 FP32 下 | `pypto.sum()` 不支持 BF16 |
| 输出转换回 BF16 | 减少内存占用 |

### 6.3 精度验证标准

```python
RTOL = 0.0078125  # 1/128
ATOL = 0.0001
numpy.testing.assert_allclose(result, golden, rtol=RTOL, atol=ATOL)
```

---

## 7. 性能优化设计

### 7.1 双缓冲策略

```python
@pypto.frontend.jit(
    pass_options={"vec_nbuffer_setting": {-2: 1, -1: 8}},
    debug_options={"runtime_debug_mode": 1}
)
```

**参数说明**：
- `-2: 1`：倒数第二维缓冲数
- `-1: 8`：最后一维缓冲数

### 7.2 向量化策略

| 操作 | Tile Shape | 向量化效率 | 说明 |
|------|-----------|----------|------|
| cast | `(1, N, 1280)` | 高 | 尾轴连续访问 |
| mul | `(1, N, N, 1280)` | 高 | 固定 N=4 |
| sum | `(1, N, 1280)` | 高 | 归约操作 |
| add | `(1, N, 1280)` | 高 | 逐元素操作 |

### 7.3 内存优化

| 优化项 | 方法 | 效果 |
|--------|------|------|
| inplace reshape | `inplace=True` | 减少内存拷贝 |
| tile 分块 | 控制中间 tensor 大小 | 降低内存峰值 |
| 顺序访问 | 连续内存布局 | 提升 cache 命中率 |

---

## 8. 边界情况处理

### 8.1 Shape 验证

```python
# Wrapper 中验证
assert N == 4, f"N must be 4, got {N}"
assert h_res.shape == (B, S, N, N)
assert h_out.shape == (B, S, D)
assert h_post.shape == (B, S, N)
```

### 8.2 DType 验证

```python
assert x.dtype == torch.bfloat16
assert h_res.dtype == torch.float32
assert h_out.dtype == torch.bfloat16
assert h_post.dtype == torch.float32
```

### 8.3 Contiguous 验证

```python
assert x.is_contiguous()
assert h_res.is_contiguous()
assert h_out.is_contiguous()
assert h_post.is_contiguous()
```

---

## 9. Kernel 签名设计

### 9.1 Tensor 标记

| 参数 | Shape | 标记 | 说明 |
|------|-------|------|------|
| x | `[B*S, N, D]` | `[DYNAMIC, 4, STATIC]` | N 固定，D 静态 |
| h_res | `[B*S, N, N]` | `[DYNAMIC, 4, 4]` | N 固定 |
| h_out | `[B*S, D]` | `[DYNAMIC, STATIC]` | D 静态 |
| h_post | `[B*S, N]` | `[DYNAMIC, 4]` | N 固定 |
| output | `[B*S, N, D]` | `[DYNAMIC, 4, STATIC]` | N 固定，D 静态 |

### 9.2 完整签名

```python
@pypto.frontend.jit(
    pass_options={"vec_nbuffer_setting": {-2: 1, -1: 8}},
    debug_options={"runtime_debug_mode": 1}
)
def mhc_post_kernel_bf16(
    x: pypto.Tensor([pypto.DYNAMIC, 4, pypto.STATIC], pypto.DT_BF16),
    h_res: pypto.Tensor([pypto.DYNAMIC, 4, 4], pypto.DT_FP32),
    h_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    h_post: pypto.Tensor([pypto.DYNAMIC, 4], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 4, pypto.STATIC], pypto.DT_BF16),
):
    """mhc_post kernel BF16 版本."""
    # 实现
```

---

## 10. 测试设计

### 10.1 测试矩阵

| 测试名称 | B*S | N | D | 验证点 |
|---------|-----|---|----|----|
| 极小规模 | 8 | 4 | 128 | 基本功能 |
| 小规模 | 256 | 4 | 128 | 功能验证 |
| 中等规模（大D） | 1024 | 4 | 5120 | D 维度切换 |
| 大规模 | 4096 | 4 | 2560 | 性能验证 |

### 10.2 精度验证方法

```python
# 对比方法
result_np = result.cpu().float().numpy()
golden_np = golden_output.float().numpy()
assert_allclose(result_np, golden_np, rtol=0.0078125, atol=0.0001)
```

### 10.3 三态标记

- `[PRECISION_PASS]`：精度验证通过
- `[PRECISION_FAIL]`：精度验证失败

---

## 11. 实现约束

### 11.1 硬编码参数

| 参数 | 值 | 原因 |
|------|---|------|
| N | 4 | 注意力流数量固定 |
| unroll_list | [128] | 循环展开长度 |

### 11.2 编译时确定

| 参数 | 标记 | 说明 |
|------|------|------|
| D | `pypto.STATIC` | D 变化触发重编译 |
| B*S | `pypto.DYNAMIC` | 运行时确定 |

### 11.3 运行时确定

| 参数 | 来源 |
|------|------|
| BS | `x.shape[0]` |
| N | Python 常量 `4` |
| D | `x.shape[2]` |

---

## 12. 性能预期

### 12.1 计算复杂度

| 操作 | FLOPS | 说明 |
|------|-------|------|
| 类型转换 (BF16→FP32) | 2 × B*S × N × D | 2 个 BF16 tensor |
| h_post_term 计算 | 2 × B*S × N × D | 乘法 |
| weighted 计算 | 2 × B*S × N × N × D | 乘法 |
| sum 归约 | B*S × N × (N-1) × D | 加法 |
| add 融合 | B*S × N × D | 加法 |
| 类型转换 (FP32→BF16) | B*S × N × D | 1 个 FP32 tensor |

**总 FLOPS**：
```
约 4 × B*S × N × D × (N + 2) ≈ 4 × B*S × 4 × D × 6 = 96 × B*S × D
```

### 12.2 内存访问量

| 操作 | 访问量 | 说明 |
|------|-------|------|
| 读 x | B*S × N × D × 2 bytes | BF16 |
| 读 h_res | B*S × N × N × 4 bytes | FP32 |
| 读 h_out | B*S × D × 2 bytes | BF16 |
| 读 h_post | B*S × N × 4 bytes | FP32 |
| 写 output | B*S × N × D × 2 bytes | BF16 |

**总访问量**：
```
≈ B*S × D × (2N + 4N + 2 + 4 + 2N) = B*S × D × (8N + 6) = 38 × B*S × D bytes
```

### 12.3 计算强度

```
计算强度 = FLOPS / 访存量 ≈ (96 × B*S × D) / (38 × B*S × D) ≈ 2.5 FLOP/byte
```

**性能特征**：
- **内存受限**：计算强度较低，内存带宽是瓶颈
- **向量化友好**：纯逐元素操作，向量化效率高
- **并行度高**：BS 轴完全并行

---

## 13. 文件清单

| 文件 | 职责 |
|------|------|
| `SPEC.md` | 需求规格 |
| `API_REPORT.md` | API 映射报告 |
| `DESIGN.md` | 本设计文档 |
| `mhc_post_golden.py` | Golden 参考实现 |
| `mhc_post_impl.py` | PyPTO kernel 实现 |
| `test_mhc_post.py` | 精度验证测试 |
| `README.md` | 使用说明 |

---

## 14. 参考文档

- [PyPTO 编程指南](../../docs/pypto_programming_guide.md)
- [PyPTO API 参考](../../docs/api_reference.md)
- [MHC 论文](https://arxiv.org/abs/2406.07828)