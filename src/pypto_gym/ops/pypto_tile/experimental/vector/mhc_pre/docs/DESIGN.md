# mhc_pre 算子设计文档

## 1. 设计概述

### 1.1 算子定位
`mhc_pre` 是 MHC (Multi-Head Context) 系统的前处理算子，用于多头注意力机制中的特征归一化、矩阵变换和三分支分流处理。

### 1.2 核心特性
- **混合算子**：MatMul (Cube) + Vector 操作组合
- **三分支输出**：h_in (BF16)、h_post (FP32)、h_res (FP32)
- **动态 Shape 支持**：B*S 维度支持动态变化
- **精度转换路径**：BF16 输入 → FP32 计算 → BF16/FP32 输出
- **循环展开优化**：BS 轴展开，提升并行度

---

## 2. 计算图设计

### 2.1 整体计算流程（7 步）

```
输入: x [B*S, N, D] (BF16)
      phi [N²+2N, N*D] (FP32)
      alpha [3] (FP32)
      bias [N²+2N] (FP32)

Step 1: Reshape & Cast
  x [B*S, N, D] BF16 → reshape → x_flat [B*S, N*D] BF16
  x_flat BF16 → cast → X_flat [B*S, N*D] FP32

Step 2: RMSNorm
  X_sq = X_flat²
  mean_val = mean(X_sq, dim=-1)
  variance = mean_val / N*D
  variance_eps = variance + norm_eps
  rsqrt_val = rsqrt(variance_eps)

Step 3: MatMul with Normalization
  phi [N²+2N, N*D] → transpose → phi_T [N*D, N²+2N]
  X_hat = matmul(X_flat, phi_T)
  X_hat_norm = X_hat * rsqrt_val

Step 4: Split
  X_hat_norm [B*S, N²+2N] → split →
    ├─ X_pre [B*S, N] FP32
    ├─ X_post [B*S, N] FP32
    └─ X_comb [B*S, N²] FP32

Step 5: Branch Pre (生成 h_in)
  scaled_X_pre = X_pre * alpha[0]
  X_pre_bias = scaled_X_pre + bias[:N]
  H_pre = sigmoid(X_pre_bias) + hc_eps
  H_pre_expanded = reshape(H_pre, [B*S, N, 1])
  weighted_X = H_pre_expanded * x_fp32_3d
  h_in_fp32 = sum(weighted_X, dim=1)
  h_in = cast(h_in_fp32, BF16)

Step 6: Branch Post (生成 h_post)
  scaled_X_post = X_post * alpha[1]
  X_post_bias = scaled_X_post + bias[N:2*N]
  H_post = sigmoid(X_post_bias)
  h_post = H_post * 2.0

Step 7: Branch Res (生成 h_res)
  scaled_X_comb = X_comb * alpha[2]
  h_res_2d = scaled_X_comb + bias[2*N:].view(N*N)
  h_res = reshape(h_res_2d, [B*S, N, N])

输出: h_in [B*S, D] (BF16)
      h_post [B*S, N] (FP32)
      h_res [B*S, N, N] (FP32)
```

### 2.2 计算图可视化

```
  x(BF16)             phi(FP32)          alpha[3](FP32)      bias[N²+2N](FP32)
     │                    │                    │                    │
     ↓                    ↓                    │                    │
  reshape              transpose               │                    │
   [BS,N*D]            [N*D,N²+2N]             │                    │
     │                    │                    │                    │
     ↓                    │                    │                    │
  cast(FP32)              │                    │                    │
     │                    │                    │                    │
     ├────────────────────┼────────────────────┤                    │
     │                    │                    │                    │
     │ X_flat             │ phi_T              │                    │
     │                    │                    │                    │
     │                    ↓                    │                    │
     │                 matmul                  │                    │
     │                 (X_flat, phi_T)         │                    │
     │                    │                    │                    │
     │                    ├────────────────────┤                    │
     │                    │                    │                    │
     │                    │ X_hat              │                    │
     │                    │                    │                    │
     ├────────────────────┐                    │                    │
     │                    │                    │                    │
     │ (用于 RMSNorm)     │                    │                    │
     ↓                    │                    │                    │
  square                  │                    │                    │
     │                    │                    │                    │
     ↓                    │                    │                    │
  sum(-1)                 │                    │                    │
     │                    │                    │                    │
     ↓                    │                    │                    │
  mul(1/N*D)              │                    │                    │
     │                    │                    │                    │
     ↓                    │                    │                    │
  add(norm_eps)           │                    │                    │
     │                    │                    │                    │
     ↓                    │                    │                    │
  rsqrt                   │                    │                    │
     │                    │                    │                    │
     ├────────────────────┤                    │                    │
     │                    │                    │                    │
     │ rsqrt_val          │                    │                    │
     │                    │                    │                    │
     │                    ↓                    │                    │
     │             mul(X_hat, rsqrt_val)       │                    │
     │                    │                    │                    │
     │                    ├────────────────────┤                    │
     │                    │                    │                    │
     │                    │ X_hat_norm         │                    │
     │                    │                    │                    │
     │                    │                    ↓                    │
     │                    │                 split                   │
     │                    │                    │                    │
     │                    │                    ├────────────────────┐
     │                    │                    │                    │
     │                    │                    │ X_pre              │
     │                    │                    ├────────────────────┤
     │                    │                    │ X_post             │
     │                    │                    ├────────────────────┤
     │                    │                    │ X_comb             │
     │                    │                    │                    │
     │                    │                    │                    │
  x(BF16) ── reshape ── x_3d(BF16) ── cast ── x_fp32_3d             │
                                          │                         │
                                          │                         │
                                          ├─────────────────────────┘
                                          │
                                          │
                                          ├─── Branch Pre ─────────────────────────────
                                          │    X_pre [BS,N]
                                          │       │
                                          │       ↓
                                          │    mul(alpha[0])
                                          │       │
                                          │       ↓
                                          │    add(bias[:N])
                                          │       │
                                          │       ↓
                                          │    sigmoid
                                          │       │
                                          │       ↓
                                          │    add(hc_eps)
                                          │       │
                                          │       ↓
                                          │    reshape [BS,N,1]
                                          │       │
                                          │       ├────────────────── mul ── weighted_X
                                          │       │                        │
                                          │       │ H_pre_expanded         ↓
                                          │       │                     sum(dim=1)
                                          │       │                        │
                                          │       │                        ↓
                                          │       │                    cast(BF16)
                                          │       │                        │
                                          │       │                        ├─────────────
                                          │       │                        │
                                          │       │                        │ h_in [BS,D] BF16
                                          │       │
                                          │       │
                                          ├─── Branch Post ────────────────────────────
                                          │    X_post [BS,N]
                                          │       │
                                          │       ↓
                                          │    mul(alpha[1])
                                          │       │
                                          │       ↓
                                          │    add(bias[N:2*N])
                                          │       │
                                          │       ↓
                                          │    sigmoid
                                          │       │
                                          │       ↓
                                          │    mul(2.0)
                                          │       │
                                          │       ├──────────────────
                                          │       │
                                          │       │ h_post [BS,N] FP32
                                          │       │
                                          │
                                          └─── Branch Res ─────────────────────────────
                                               X_comb [BS,N²]
                                                  │
                                                  ↓
                                               mul(alpha[2])
                                                  │
                                                  ↓
                                               add(bias[2*N:])
                                                  │
                                                  ↓
                                               reshape [BS,N,N]
                                                  │
                                                  ├─────────────────
                                                  │
                                                  │ h_res [BS,N,N] FP32
```

**数据流说明**：

| 分支 | 输入 | 计算 | 输出 | Shape 变化 |
|------|------|------|------|-----------|
| **RMSNorm** | X_flat [BS,N*D] | square → mean → rsqrt | rsqrt_val [BS,1] | 归一化系数 |
| **MatMul** | X_flat [BS,N*D] × phi_T [N*D,N²+2N] | matmul → mul(rsqrt) | X_hat_norm [BS,N²+2N] | 矩阵变换 + 归一化 |
| **Branch Pre** | X_pre [BS,N] | sigmoid + 加权求和 | h_in [BS,D] | [BS,N] → [BS,D] (聚合) |
| **Branch Post** | X_post [BS,N] | sigmoid × 2 | h_post [BS,N] | 保持 shape |
| **Branch Res** | X_comb [BS,N²] | linear + reshape | h_res [BS,N,N] | [BS,N²] → [BS,N,N] |

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

### 3.2 向量化分块（自适应策略）

**自适应 Tile Shape 设置**：
```python
# 根据 D 大小动态调整
if D < 2560:
    bs_tile = 8
    D_tile = 128
else:
    bs_tile = 1
    D_tile = 2560
```

**Vector 算子分块配置**：

| 操作 | Tile Shape | 自适应调整 | 说明 |
|------|-----------|----------|------|
| Cast (Step 1) | `(bs_tile*8, D_tile)` | 是 | 大 D: bs_tile=1, D_tile=2560<br>小 D: bs_tile=8, D_tile=128 |
| RMSNorm (Step 2) | `(bs_tile, D_tile)` | 是 | 自适应调整 |
| Weighted (Step 5) | `(bs_tile, N, D_tile)` | 是 | 3D tensor 分块 |
| Branch Post/Res | `(bs_tile, N)` | 是 | 2D tensor 分块 |

**Cube 算子分块配置**：

| 操作 | Tile Shape | 说明 |
|------|-----------|------|
| MatMul (Step 3) | `[16, 16], [512, 1024], [128, 128]` | Cube tile shapes |
| Split K | `enable_split_k=True` | 大矩阵优化 |

**分块原理**：
- **自适应策略**：根据 D 大小动态调整，平衡内存与性能
- **小 D 场景**：bs_tile=8 提升并行度
- **大 D 场景**：bs_tile=1 避免内存峰值
- **尾轴分块**：避免跨 cache line 访问

### 3.3 内存访问模式

| Tensor | Shape | 访问模式 | Tile 优化 |
|--------|-------|---------|----------|
| x | [BS, N, D] BF16 | 顺序访问 | 自适应分块 |
| x_flat | [BS, N*D] FP32 | 顺序访问 | 自适应分块 |
| phi_T | [N*D, N²+2N] FP32 | 固定权重 | Cube tile shapes |
| X_hat_norm | [BS, N²+2N] FP32 | 顺序访问 | 分块访问 |
| h_in | [BS, D] BF16 | 顺序写入 | 尾轴分块 |
| h_post | [BS, N] FP32 | 顺序写入 | 固定 N=8 |
| h_res | [BS, N, N] FP32 | 顺序写入 | 固定 N=8 |

**内存访问优化**：
- 所有 tensor 按 C-order 连续存储
- Phi_T 为固定权重，使用 Cube tile shapes 优化
- 顺序访问模式，利于 cache 预取
- 自适应 tile 大小，平衡内存与性能

---

## 4. Loop 结构设计

### 4.1 外层循环（BS 轴）

```python
for bs_idx, unroll_length in pypto.loop_unroll(0, BS, 1, name="LOOP_BS", unroll_list=[128]):
    # 提取切片
    x_slice_flat = x_flat[bs_idx: bs_idx + unroll_length, :]
    x_slice_3d = x[bs_idx: bs_idx + unroll_length, :, :]
    
    # Step 1: Cast to FP32
    X_flat = pypto.cast(x_slice_flat, pypto.DT_FP32)
    x_fp32_3d = pypto.cast(x_slice_3d, pypto.DT_FP32)
    
    # Step 2: RMSNorm
    rsqrt_val = compute_rmsnorm_rsqrt(X_flat, N_D, norm_eps)
    
    # Step 3: MatMul
    X_hat = pypto.matmul(X_flat, phi_T, pypto.DT_FP32)
    X_hat_norm = pypto.mul(X_hat, rsqrt_val)
    
    # Step 4: Split
    X_pre = X_hat_norm[:, 0:N]
    X_post = X_hat_norm[:, N:2*N]
    X_comb = X_hat_norm[:, 2*N:2*N+N*N]
    
    # Step 5-7: 三分支处理
    # ...（见下文）
    
    # 输出组装
    pypto.assemble(h_in_tile, [bs_idx, 0], h_in)
    pypto.assemble(h_post_tile, [bs_idx, 0], h_post)
    pypto.assemble(h_res_tile, [bs_idx, 0, 0], h_res)
```

**循环参数**：
- `start=0, stop=BS, step=1`：遍历 BS 轴
- `unroll_list=[128]`：展开长度为 128
- `idx_name="bs_idx"`：循环变量名

### 4.2 RMSNorm Helper Function

```python
def compute_rmsnorm_rsqrt(X_flat, N_D, norm_eps):
    """计算 RMSNorm 的平方根倒数归一化系数"""
    # Step 2.1: 计算平方
    X_sq = pypto.mul(X_flat, X_flat)  # [unroll_length, N_D] FP32
    
    # Step 2.2: 沿尾轴求和
    mean_val = pypto.sum(X_sq, -1, keepdim=True)  # [unroll_length, 1] FP32
    
    # Step 2.3: 计算均值（使用 sum + div 替代 mean API）
    mean_coeff = 1.0 / N_D  # Python float
    variance = pypto.mul(mean_val, mean_coeff)  # [unroll_length, 1] FP32
    
    # Step 2.4: 加 epsilon 防止除零
    variance_eps = pypto.add(variance, norm_eps)  # [unroll_length, 1] FP32
    
    # Step 2.5: 计算平方根倒数
    rsqrt_val = pypto.rsqrt(variance_eps)  # [unroll_length, 1] FP32
    
    return rsqrt_val
```

### 4.3 三分支处理（内层计算）

**Branch Pre (生成 h_in)**:
```python
# Step 5: Branch Pre
pypto.set_vec_tile_shapes(bs_tile, N)
scaled_X_pre = pypto.mul(X_pre, alpha_0)  # [unroll_length, N] FP32
X_pre_bias = pypto.add(scaled_X_pre, bias_pre_2d)  # [unroll_length, N] FP32
H_pre = pypto.sigmoid(X_pre_bias)  # [unroll_length, N] FP32
H_pre_eps = pypto.add(H_pre, hc_eps)  # [unroll_length, N] FP32
H_pre_expanded = pypto.reshape(H_pre_eps, [unroll_length, N, 1], inplace=True)

pypto.set_vec_tile_shapes(bs_tile, N, D_tile)
weighted_X = pypto.mul(H_pre_expanded, x_fp32_3d)  # [unroll_length, N, D] FP32
h_in_fp32 = pypto.sum(weighted_X, 1)  # [unroll_length, D] FP32

pypto.set_vec_tile_shapes(bs_tile, D_tile)
h_in_tile = pypto.cast(h_in_fp32, pypto.DT_BF16)  # [unroll_length, D] BF16
```

**Branch Post (生成 h_post)**:
```python
# Step 6: Branch Post
pypto.set_vec_tile_shapes(bs_tile_2, N)
scaled_X_post = pypto.mul(X_post, alpha_1)  # [unroll_length, N] FP32
X_post_bias = pypto.add(scaled_X_post, bias_post_2d)  # [unroll_length, N] FP32
H_post = pypto.sigmoid(X_post_bias)  # [unroll_length, N] FP32
h_post_tile = pypto.mul(H_post, 2.0)  # [unroll_length, N] FP32
```

**Branch Res (生成 h_res)**:
```python
# Step 7: Branch Res
pypto.set_vec_tile_shapes(bs_tile_2, N * N)
scaled_X_comb = pypto.mul(X_comb, alpha_2)  # [unroll_length, N²] FP32
h_res_2d = pypto.add(scaled_X_comb, bias_comb_2d)  # [unroll_length, N²] FP32

pypto.set_vec_tile_shapes(bs_tile_2, 16, 32)  # 固定对齐 tile shape
h_res_tile = pypto.reshape(h_res_2d, [unroll_length, N, N])  # [unroll_length, N, N] FP32
```

---

## 5. 数据流设计

### 5.1 输入预处理（Wrapper）

**Wrapper 函数职责**：
```python
def mhc_pre_wrapper(x, phi, alpha, bias, norm_eps=1e-6, hc_eps=1e-6):
    # 1. Phi 转置
    phi_T = phi.T.contiguous()  # [N²+2N, N*D] → [N*D, N²+2N]
    
    # 2. 提取 alpha 为 Python float
    alpha_0 = float(alpha[0].item())
    alpha_1 = float(alpha[1].item())
    alpha_2 = float(alpha[2].item())
    
    # 3. Bias 预切片
    N = x.shape[1]
    bias_pre = bias[:N].contiguous()  # [N]
    bias_post = bias[N:2*N].contiguous()  # [N]
    bias_comb = bias[2*N:].contiguous()  # [N*N]
    
    # 4. 创建输出 tensor
    bs, D = x.shape[0], x.shape[2]
    h_in = torch.empty(bs, D, dtype=torch.bfloat16, device=x.device)
    h_post = torch.empty(bs, N, dtype=torch.float32, device=x.device)
    h_res = torch.empty(bs, N, N, dtype=torch.float32, device=x.device)
    
    # 5. 调用 kernel
    mhc_pre_kernel(x, phi_T, bias_pre, bias_post, bias_comb,
                   h_in, h_post, h_res,
                   alpha_0, alpha_1, alpha_2, norm_eps, hc_eps)
    
    return h_in, h_post, h_res
```

### 5.2 中间数据流

```
输入切片 [unroll_length, ...]
    ↓
类型转换 BF16 → FP32
    ↓
RMSNorm (归一化系数)
    ↓
MatMul (矩阵变换)
    ↓
归一化 (mul rsqrt)
    ↓
Split (三分流)
    ↓
三分支并行处理:
    ├─ Branch Pre: sigmoid → 加权求和 → h_in
    ├─ Branch Post: sigmoid × 2 → h_post
    └─ Branch Res: linear → reshape → h_res
    ↓
组装到输出 tensor
```

### 5.3 输出组装

```python
# 3D 输出需要 3 个索引
pypto.assemble(h_in_tile, [bs_idx, 0], h_in)
pypto.assemble(h_post_tile, [bs_idx, 0], h_post)
pypto.assemble(h_res_tile, [bs_idx, 0, 0], h_res)
```

**组装参数**：
- `h_in_tile`：当前 tile 的 h_in 结果
- `[bs_idx, 0]`：输出 tensor 起始位置
- `h_in`：输出 tensor

---

## 6. 精度设计

### 6.1 精度转换路径

```
输入精度: BF16 (x)
    ↓
计算精度: FP32 (所有中间计算)
    ├─ RMSNorm: FP32
    ├─ MatMul: FP32
    ├─ sigmoid: FP32
    └─ 加权求和: FP32
    ↓
输出精度:
    ├─ h_in: BF16 (FP32 → BF16)
    ├─ h_post: FP32 (保持)
    └─ h_res: FP32 (保持)
```

### 6.2 精度保证措施

| 措施 | 说明 |
|------|------|
| 中间计算使用 FP32 | 避免累加误差 |
| RMSNorm 在 FP32 下 | `rsqrt` 仅支持 FP32 |
| sigmoid 在 FP32 下 | `sigmoid` 仅支持 FP32 |
| MatMul 在 FP32 下 | 两侧 dtype 必须一致 |
| h_in 输出转换回 BF16 | 减少内存占用 |
| h_post/h_res 保持 FP32 | 保持精度 |

### 6.3 精度验证标准

```python
RTOL = 0.0078125  # 1/128
ATOL = 0.0001
numpy.testing.assert_allclose(result, golden, rtol=RTOL, atol=ATOL)
```

---

## 7. 性能优化设计

### 7.1 MatMul 优化策略

| 优化项 | 方法 | 效果 |
|--------|------|------|
| Cube tile shapes | `[16, 16], [512, 1024], [128, 128]` | MatMul 性能优化 |
| Split K | `enable_split_k=True` | 大矩阵并行优化 |
| Phi 预转置 | wrapper 中处理 | 避免 kernel 内开销 |

### 7.2 向量化策略

| 优化项 | 方法 | 效果 |
|--------|------|------|
| 自适应 tile shapes | 根据 D 动态调整 | 平衡内存与性能 |
| 循环展开 | `unroll_list=[128]` | 提升指令级并行 |
| Bias 预切片 | wrapper 中切片 | 减少 kernel 内操作 |

### 7.3 内存优化

| 优化项 | 方法 | 效果 |
|--------|------|------|
| 原地 reshape | `inplace=True` | 减少内存拷贝 |
| 自适应分块 | 控制中间 tensor 大小 | 降低内存峰值 |
| 顺序访问 | 连续内存布局 | 提升 cache 命中率 |

---

## 8. 边界情况处理

### 8.1 Shape 验证

```python
# Wrapper 中验证
assert x.shape[1] == N, f"N mismatch"
assert phi.shape == (N_SQUARED_PLUS_2N, N * D), f"phi shape mismatch"
assert alpha.shape == (3,), f"alpha shape mismatch"
assert bias.shape == (N_SQUARED_PLUS_2N,), f"bias shape mismatch"
```

### 8.2 DType 验证

```python
assert x.dtype == torch.bfloat16
assert phi.dtype == torch.float32
assert alpha.dtype == torch.float32
assert bias.dtype == torch.float32
```

### 8.3 Contiguous 验证

```python
assert x.is_contiguous()
# phi 转置后需 contiguous（wrapper 中处理）
# bias 切片后需 contiguous（wrapper 中处理）
```

---

## 9. Kernel 签名设计

### 9.1 Tensor 标记

| 参数 | Shape | 标记 | 说明 |
|------|-------|------|------|
| x | `[B*S, N, D]` | `[DYNAMIC, STATIC, STATIC]` | BS 动态，N/D 静态 |
| phi_T | `[N*D, N²+2N]` | `[STATIC, STATIC]` | 固定权重 |
| bias_pre | `[N]` | `[STATIC]` | 固定偏置 |
| bias_post | `[N]` | `[STATIC]` | 固定偏置 |
| bias_comb | `[N*N]` | `[STATIC]` | 固定偏置 |
| h_in | `[B*S, D]` | `[DYNAMIC, STATIC]` | BS 动态，D 静态 |
| h_post | `[B*S, N]` | `[DYNAMIC, STATIC]` | BS 动态，N 静态 |
| h_res | `[B*S, N, N]` | `[DYNAMIC, STATIC, STATIC]` | BS 动态，N 静态 |

### 9.2 完整签名

```python
@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128},
    pass_options={"vec_nbuffer_setting": {-2: 1, -1: 4}},
    debug_options={"runtime_debug_mode": 1}
)
def mhc_pre_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    phi_T: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    bias_pre: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    bias_post: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    bias_comb: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    h_in: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    h_post: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_FP32),
    h_res: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    alpha_0: float = 1.0,
    alpha_1: float = 1.0,
    alpha_2: float = 1.0,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
):
    """mhc_pre kernel - Multi-Head Context 前处理算子."""
    # 实现
```

---

## 10. 测试设计

### 10.1 测试矩阵

| 测试名称 | B*S | N | D | 验证点 |
|---------|-----|---|----|----|
| 极小规模 | 8 | 4 | 128 | 基本功能 |
| 基础验证 | 128 | 4 | 5120 | 完整流程 |
| 小规模 | 256 | 4 | 128 | 功能验证 |
| 中等规模 | 1024 | 4 | 5120 | 性能验证 |
| 大规模 | 4096 | 4 | 2560 | 大规模验证 |

### 10.2 精度验证方法

```python
# 对比方法
h_in_impl_np = h_in_impl.cpu().float().numpy()
h_in_golden_np = h_in_golden.float().numpy()
assert_allclose(h_in_impl_np, h_in_golden_np, rtol=0.0078125, atol=0.0001)

# 三个输出全部验证
assert_allclose(h_post_impl_np, h_post_golden_np, rtol=RTOL, atol=ATOL)
assert_allclose(h_res_impl_np, h_res_golden_np, rtol=RTOL, atol=ATOL)
```

### 10.3 三态标记

- `[PRECISION_PASS]`：精度验证通过
- `[PRECISION_FAIL]`：精度验证失败

---

## 11. 实现约束

### 11.1 硬编码参数

| 参数 | 值 | 原因 |
|------|---|------|
| N | 8 | 注意力头数固定 |
| unroll_list | [128] | 循环展开长度 |

### 11.2 编译时确定

| 参数 | 标记 | 说明 |
|------|------|------|
| N | `pypto.STATIC` | 固定值 |
| D | `pypto.STATIC` | D 变化触发重编译 |

### 11.3 运行时确定

| 参数 | 来源 |
|------|------|
| BS | `x.shape[0]` |
| N | `x.shape[1]` 或 Python 常量 |
| D | `x.shape[2]` 或 Python 常量 |
| alpha | Python float（wrapper 中提取） |
| bias | wrapper 中预切片 |

---

## 12. 性能预期

### 12.1 计算复杂度

| 操作 | FLOPS | 说明 |
|------|-------|------|
| RMSNorm | 2 × B*S × N*D | square + sum |
| MatMul | 2 × B*S × N*D × (N²+2N) | 矩阵乘法（主导） |
| Branch Pre sigmoid | N × sigmoid FLOPs | sigmoid |
| Branch Pre 加权求和 | 2 × B*S × N × D | mul + sum |
| Branch Post sigmoid | N × sigmoid FLOPs | sigmoid |
| Branch Res linear | 2 × B*S × N² | mul + add |

**总 FLOPS**：
```
约 2 × B*S × N*D × (N²+2N) ≈ 2 × B*S × 40960 × 80 = 6.56M × B*S
```

### 12.2 内存访问量

| 操作 | 访问量 | 说明 |
|------|-------|------|
| 读 x | B*S × N × D × 2 bytes | BF16 |
| 读 phi_T | N*D × (N²+2N) × 4 bytes | FP32 固定权重 |
| 读 bias | (N²+2N) × 4 bytes | FP32 固定偏置 |
| 写 h_in | B*S × D × 2 bytes | BF16 |
| 写 h_post | B*S × N × 4 bytes | FP32 |
| 写 h_res | B*S × N × N × 4 bytes | FP32 |

**总访问量**：
```
≈ B*S × D × (2N + 2 + 4N + 4N) + 固定权重 ≈ B*S × 5120 × 66 bytes
```

### 12.3 计算强度

```
计算强度 = FLOPS / 访存量 ≈ (6.56M × B*S) / (B*S × 5120 × 66) ≈ 19.5 FLOP/byte
```

**性能特征**：
- **计算密集**：MatMul 占主导，计算强度较高
- **MatMul 优化关键**：Cube tile shapes 和 Split K 是核心
- **并行度高**：BS 轴完全并行

---

## 13. 文件清单

| 文件 | 职责 |
|------|------|
| `SPEC.md` | 需求规格 |
| `API_REPORT.md` | API 映射报告 |
| `DESIGN.md` | 本设计文档 |
| `mhc_pre_golden.py` | Golden 参考实现 |
| `mhc_pre_impl.py` | PyPTO kernel 实现 |
| `test_mhc_pre.py` | 精度验证测试 |
| `README.md` | 使用说明 |

---

## 14. 参考文档

- [PyPTO 编程指南](../../docs/pypto_programming_guide.md)
- [PyPTO API 参考](../../docs/api_reference.md)
- [MHC 论文](https://arxiv.org/abs/2406.07828)