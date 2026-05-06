# mhc_post 算子需求规格

## 1. 算子概述

### 1.1 算子名称
`mhc_post` - MHC (Manifold-Constrained Hyper-Connections) 后处理融合算子

### 1.2 功能描述
`mhc_post` 实现注意力机制中的流间混合计算，属于 MHC 系统的后处理阶段。该算子将多个输入张量进行融合计算，完成逐样本动态权重广播乘法和流间混合加权求和。

### 1.3 应用场景
- Transformer 架构中的注意力机制增强
- 多流（multi-stream）注意力计算的后处理阶段
- 大语言模型中的 Manifold-Constrained Hyper-Connections 实现

---

## 2. 数学公式

### 2.1 计算公式

```python
h_post_term = h_post.unsqueeze(-1) * h_out.unsqueeze(-2)
h_comb_term = torch.sum(h_res.unsqueeze(-1) * x.unsqueeze(-2), dim=-3)
output = (h_post_term + h_comb_term).to(bfloat16)
```

### 2.2 展开形式

```
output[b*s, n, d] = h_post[b*s, n] * h_out[b*s, d] + Σ(k=0..N-1) h_res[b*s, k, n] * x[b*s, k, d]
```

### 2.3 计算步骤

1. **h_post_term 计算**：逐样本动态权重广播乘法
   - `h_post: [B*S, N]` → unsqueeze → `[B*S, N, 1]`
   - `h_out: [B*S, D]` → unsqueeze → `[B*S, 1, D]`
   - 广播乘法：`[B*S, N, 1] × [B*S, 1, D] → [B*S, N, D]`

2. **h_comb_term 计算**：逐样本流间混合加权求和
   - `h_res: [B*S, N, N]` → unsqueeze → `[B*S, N, N, 1]`
   - `x: [B*S, N, D]` → unsqueeze → `[B*S, N, 1, D]`
   - 广播乘法：`[B*S, N, N, 1] × [B*S, N, 1, D] → [B*S, N, N, D]`
   - 沿 dim=-3 求和：`[B*S, N, N, D] → [B*S, N, D]`

3. **融合输出**：逐元素相加并转换回 BF16
   - `result = h_post_term + h_comb_term`
   - 转换回 `bfloat16` 精度

---

## 3. 输入输出规格

### 3.1 输入张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `x` | `[B*S, N, D]` | bfloat16 | 输入 tensor，来自前序计算 |
| `h_res` | `[B*S, N, N]` | float32 | 流间混合权重矩阵，逐样本动态权重 |
| `h_out` | `[B*S, D]` | bfloat16 | 输出项数据 |
| `h_post` | `[B*S, N]` | float32 | 后处理权重，逐样本动态权重 |

### 3.2 输出张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `output` | `[B*S, N, D]` | bfloat16 | 融合计算结果 |

### 3.3 参数说明

- **B*S**：批大小 × 序列长度，动态维度
- **N**：注意力流数量，固定为 4
- **D**：隐藏层维度，静态轴

---

## 4. Shape 范围与约束

### 4.1 动态轴取值范围

| 轴 | 取值范围 | 说明 |
|----|---------|------|
| B*S | {1024, 2048, 4096} | 动态维度，批大小 × 序列长度 |
| N | 4 | **固定值**，注意力流数量 |
| D | {2560, 5120} | 静态轴，隐藏层维度（变化时触发重编译） |

### 4.2 约束条件

1. **N 固定为 4**：注意力流数量不可变，硬编码在 kernel 中
2. **D 为 STATIC 标记**：D 维度变化会触发 kernel 重编译
3. **B*S 为 DYNAMIC**：支持动态 shape，无需重编译
4. **内存连续性**：所有输入 tensor 必须是 contiguous 的
5. **精度约束**：
   - BF16 输入在计算前转为 FP32
   - FP32 中间结果最后转回 BF16 输出
   - sigmoid 和 sum 操作仅支持 FP32

### 4.3 典型配置

| 配置名称 | B*S | N | D | 用途 |
|---------|-----|---|----|----|
| P0_标准 | 1024 | 4 | 2560 | 标准推理场景 |
| P0_大D | 1024 | 4 | 5120 | 大模型配置 |
| P0_大BS | 4096 | 4 | 2560 | 长序列场景 |
| P0_最大 | 4096 | 4 | 5120 | 最大配置 |

---

## 5. 精度要求

### 5.1 数据类型转换

- 输入类型：`x` 和 `h_out` 为 bfloat16，`h_res` 和 `h_post` 为 float32
- 计算类型：所有计算在 float32 下进行
- 输出类型：转换为 bfloat16

### 5.2 精度容差

- **相对容差 (RTOL)**：0.0078125 (1/128)
- **绝对容差 (ATOL)**：0.0001

### 5.3 精度验证标准

输出结果与 golden 实现对比需满足：
```python
numpy.testing.assert_allclose(result, golden, rtol=0.0078125, atol=0.0001)
```

---

## 6. 性能要求

### 6.1 计算特点

- **纯 Vector 算子**：无矩阵乘法操作，纯向量运算
- **内存访问模式**：顺序访问为主，利于向量化
- **并行度**：B*S 维度完全并行

### 6.2 优化方向

1. 循环展开优化（BS 轴）
2. 向量化分块优化
3. 双缓冲策略
4. 内存访问模式优化

---

## 7. 测试验证要求

### 7.1 功能测试

| 测试名称 | B*S | N | D | 验证点 |
|---------|-----|---|----|----|
| 极小规模 | 8 | 4 | 128 | 基本功能验证 |
| 小规模 | 256 | 4 | 128 | 功能验证 |
| 中等规模 | 1024 | 4 | 5120 | 精度验证（大 D） |
| 大规模 | 4096 | 4 | 2560 | 性能验证 |

### 7.2 精度测试

- Golden 实现：纯 PyTorch 实现，作为精度基准
- 三态标记：`[PRECISION_PASS]` 或 `[PRECISION_FAIL]`
- 对比方法：与 golden 实现逐元素对比

### 7.3 边界测试

- 最小 B*S 值：1024
- 最大 B*S 值：4096
- D 维度切换：2560 ↔ 5120（验证重编译）

---

## 8. 实现约束

### 8.1 API 映射要求

- 必须使用 PyPTO 框架实现
- 支持 PyPTO JIT 编译
- 支持动态 shape（B*S）
- 支持静态轴重编译（D）

### 8.2 内存管理

- 输入输出张量必须 contiguous
- 中间计算使用 FP32，需考虑内存占用
- 支持 inplace 操作减少内存拷贝

### 8.3 兼容性

- PyPTO 版本要求：与 CANN 版本匹配
- 硬件要求：华为昇腾 AI 处理器
- 软件栈：CANN 8.5.0+

---

## 9. 参考实现

### 9.1 Golden 实现

位于 `mhc_post_golden.py`，提供纯 PyTorch 参考实现：
```python
def mhc_post_golden(x, h_res, h_out, h_post):
    h_out_fp32 = h_out.float()
    x_fp32 = x.float()
    h_post_term = h_post.unsqueeze(-1) * h_out_fp32.unsqueeze(-2)
    h_comb_term = torch.sum(h_res.unsqueeze(-1) * x_fp32.unsqueeze(-2), dim=-3)
    output = (h_post_term + h_comb_term).to(torch.bfloat16)
    return output
```

### 9.2 参考文献

- [MHC 论文](https://arxiv.org/abs/2406.07828) - Manifold-Constrained Hyper-Connections