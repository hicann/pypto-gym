# mhc_pre 算子需求规格

## 1. 算子概述

### 1.1 算子名称
`mhc_pre` - MHC (Multi-Head Context) 前处理算子

### 1.2 功能描述
`mhc_pre` 实现多头注意力机制的前处理阶段，包括特征归一化、矩阵变换和三分支分流处理。该算子将输入特征通过 RMSNorm、MatMul 和三分支处理，生成三个输出张量。

### 1.3 应用场景
- Transformer 架构中的多头注意力前处理
- Multi-Head Context (MHC) 系统的前处理阶段
- 大语言模型中的注意力计算预处理

---

## 2. 数学公式

### 2.1 计算流程（7 步）

```
输入: x [B*S, N, D] (BF16) + phi [N²+2N, N*D] (FP32) + alpha [3] (FP32) + bias [N²+2N] (FP32)

Step 1: Reshape & Float
  x_flat = x.reshape([B*S, N*D]).float()  # BF16 → FP32

Step 2: RMSNorm
  inv_rms = rsqrt(mean(x_flat²) + norm_eps)

Step 3: MatMul with Normalization
  h_mix = F.linear(x_flat, phi)
  weight = h_mix * inv_rms

Step 4: Split
  X_pre, X_post, X_comb = weight.split([N, N, N²], dim=-1)

Step 5: Branch Pre (加权求和生成 h_in)
  h_pre = sigmoid(X_pre * alpha[0] + bias[:N]) + hc_eps
  weighted_X = h_pre.unsqueeze(-1) * x.unflatten(-1, (N, D))
  h_in = sum(weighted_X, dim=1).to(bfloat16)

Step 6: Branch Post
  h_post = 2 * sigmoid(X_post * alpha[1] + bias[N:2*N])

Step 7: Branch Res
  h_res = (X_comb * alpha[2] + bias[2*N:]).view([B*S, N, N])

输出: h_in [B*S, D] (BF16), h_post [B*S, N] (FP32), h_res [B*S, N, N] (FP32)
```

### 2.2 关键公式详解

**RMSNorm (Root Mean Square Layer Normalization)**:
```
inv_rms = rsqrt(mean(x_flat²) + norm_eps)
```
- 目的：归一化输入特征，避免方差为零时除零
- `norm_eps`: 默认 1e-6

**Branch Pre (加权求和)**:
```
h_pre = sigmoid(X_pre * alpha[0] + bias[:N]) + hc_eps
h_in = sum(h_pre.unsqueeze(-1) * x.unflatten(-1, (N, D)), dim=1)
```
- sigmoid 输出范围：(0, 1)
- `h_pre` 范围：`(hc_eps, 1 + hc_eps)`，避免零值
- 加权求和：沿 N 维度聚合

**Branch Post (门控信号)**:
```
h_post = 2 * sigmoid(X_post * alpha[1] + bias[N:2*N])
```
- 输出范围：`(0, 2)`
- 用于后处理门控

**Branch Res (组合门控)**:
```
h_res = X_comb * alpha[2] + bias[2*N:].view(N, N)
```
- 纯线性变换，无 sigmoid
- reshape 为 3D: `[B*S, N, N]`

---

## 3. 输入输出规格

### 3.1 输入张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `x` | `[B*S, N, D]` | bfloat16 | 输入特征 tensor |
| `phi` | `[N²+2N, N*D]` | float32 | 权重矩阵（MatMul 用） |
| `alpha` | `[3]` | float32 | 缩放系数 [α₀, α₁, α₂] |
| `bias` | `[N²+2N]` | float32 | 偏置向量（三分支共用） |

### 3.2 输出张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `h_in` | `[B*S, D]` | bfloat16 | 加权输入（Branch Pre 输出） |
| `h_post` | `[B*S, N]` | float32 | 后处理门控信号（Branch Post 输出） |
| `h_res` | `[B*S, N, N]` | float32 | 组合门控信号（Branch Res 输出） |

### 3.3 参数

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `norm_eps` | float32 | 1e-6 | RMSNorm 的 epsilon，防止除零 |
| `hc_eps` | float32 | 1e-6 | sigmoid 输出的精度保护，避免饱和区 |

---

## 4. Shape 范围与约束

### 4.1 动态轴与静态轴

| 轴 | 范围 | 标记 | 说明 |
|----|------|------|------|
| B*S | {1024, 2048, 4096} | `DYNAMIC` | 动态轴，批大小 × 序列长度 |
| N | 8 | `STATIC` | 注意力头数，固定值 |
| D | 5120 | `STATIC` | 隐藏层维度，变化触发重编译 |

### 4.2 派生常量

| 常量 | 公式 | 典型值 | 说明 |
|------|------|--------|------|
| N*D | N × D | 40960 | 特征维度（x flatten 后） |
| N²+2N | N² + 2N | 80 | 权重矩阵 phi 的第一维 |
| N² | N × N | 64 | Branch Res 输出维度 |

### 4.3 约束条件

1. **N 固定为 8**：注意力头数不可变（当前实现硬编码）
2. **D 为 STATIC**：D 维度变化会触发 kernel 重编译
3. **B*S 为 DYNAMIC**：支持动态 shape，无需重编译
4. **内存连续性**：
   - `x` 必须是 contiguous 的
   - `phi` 转置后需 `.contiguous()`（wrapper 中处理）
   - `bias` 需提前切片并 contiguous（wrapper 中处理）
5. **精度约束**：
   - 输入 `x` 为 BF16，计算前转为 FP32
   - `h_in` 输出为 BF16（FP32 → BF16）
   - `h_post` 和 `h_res` 输出为 FP32
   - sigmoid 和 MatMul 仅支持 FP32
6. **MatMul 约束**：
   - `X_flat`: `[B*S, N*D]` FP32
   - `phi_T`: `[N*D, N²+2N]` FP32
   - 输出: `[B*S, N²+2N]` FP32

---

## 5. 精度要求

### 5.1 数据类型转换

| 转换点 | 输入类型 | 输出类型 | 说明 |
|--------|---------|---------|------|
| Step 1 | BF16 | FP32 | 输入特征转换 |
| Step 5 输出 | FP32 | BF16 | h_in 输出转换 |
| Step 6 输出 | FP32 | FP32 | h_post 保持 FP32 |
| Step 7 输出 | FP32 | FP32 | h_res 保持 FP32 |

### 5.2 精度容差

- **相对容差 (RTOL)**：0.0078125 (1/128)
- **绝对容差 (ATOL)**：0.0001

### 5.3 精度验证标准

输出结果与 golden 实现对比需满足：
```python
numpy.testing.assert_allclose(result, golden, rtol=0.0078125, atol=0.0001)
```

### 5.4 值域特性

| 输出 | 值域范围 | 说明 |
|------|---------|------|
| `h_pre` | `(hc_eps, 1 + hc_eps)` | sigmoid + hc_eps |
| `h_post` | `(0, 2)` | 2 × sigmoid |
| `h_res` | 无限制 | 纯线性变换 |

---

## 6. 性能要求

### 6.1 计算特点

| 特点 | 说明 |
|------|------|
| 混合算子 | MatMul (Cube) + Vector 操作 |
| 计算密集 | MatMul 占主要计算量 |
| 内存密集 | RMSNorm、sigmoid、sum 为内存密集 |

### 6.2 优化方向

1. MatMul 优化（Cube tile shape、split_k）
2. Loop Unroll 优化（BS 轴展开）
3. Tile Shape 优化（Vector 算子分块）
4. Bias 预切片（减少 kernel 内操作）

---

## 7. 测试验证要求

### 7.1 功能测试

| 测试名称 | B*S | N | D | 验证点 |
|---------|-----|---|----|----|
| 极小规模 | 8 | 4 | 128 | 基本功能验证 |
| 基础验证 | 128 | 4 | 5120 | 完整流程验证 |
| 小规模 | 256 | 4 | 128 | 功能验证 |
| 中等规模 | 1024 | 4 | 5120 | 性能验证 |
| 大规模 | 4096 | 4 | 2560 | 大规模验证 |

### 7.2 精度测试

- Golden 实现：纯 PyTorch 实现，作为精度基准
- 三态标记：`[PRECISION_PASS]` 或 `[PRECISION_FAIL]`
- 对比方法：与 golden 实现逐元素对比
- 验证项：
  - 三个输出的 shape 和 dtype 验证
  - 数值对比（最大差异、均值差异）
  - 数值稳定性（无 NaN/Inf）
  - sigmoid 值域检查（h_post ∈ [0, 2]）

### 7.3 边界测试

- 最小 B*S 值：8
- 最大 B*S 值：4096
- D 维度切换：5120 ↔ 2560（验证重编译）
- N 维度变化：测试中部分用例使用 N=4

---

## 8. 实现约束

### 8.1 API 映射要求

- 必须使用 PyPTO 框架实现
- 支持 PyPTO JIT 编译
- 支持动态 shape（B*S）
- 支持静态轴重编译（D）

### 8.2 内存管理

- 输入输出张量必须 contiguous
- Bias 需在 wrapper 中预切片
- Phi 需在 wrapper 中转置并 contiguous
- 中间计算使用 FP32，需考虑内存占用

### 8.3 兼容性

- PyPTO 版本要求：与 CANN 版本匹配
- 硬件要求：华为昇腾 AI 处理器
- 软件栈：CANN 8.5.0+

---

## 9. 参考实现

### 9.1 Golden 实现

位于 `mhc_pre_golden.py`，提供纯 PyTorch 参考实现：
```python
def mhc_pre_golden(x, phi, alpha, bias, norm_eps=1e-6, hc_eps=1e-6):
    T, N, D = x.shape
    ND = N * D
    
    # Step 1: Reshape & Float
    x_flat = x.reshape(T, ND).float()
    
    # Step 2: RMSNorm
    inv_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
    
    # Step 3: MatMul with Normalization
    h_mix = F.linear(x_flat, phi.float())
    weight = h_mix * inv_rms
    
    # Step 4: Split
    h_pre, h_post, h_res = weight.split([N, N, N*N], dim=-1)
    
    # Step 5-7: 三分支处理
    # ...（见完整实现）
    
    return h_in, h_post, h_res
```

### 9.2 参考文献

- [MHC 论文](https://arxiv.org/abs/2406.07828) - Multi-Head Context Attention