# grouped_matmul_swiglu_quant 算子需求规格

## 1. 算子概述

### 1.1 算子名称

`grouped_matmul_swiglu_quant` - MXFP8 grouped matmul + SwiGLU + per-token INT8 量化融合算子。

### 1.2 功能描述

`grouped_matmul_swiglu_quant` 对路由后的 token 按 expert 执行 MXFP8 scaled grouped matmul，将 matmul 输出经过 SwiGLU 激活后进行 per-token INT8 动态量化，并输出量化结果和对应 scale。

### 1.3 应用场景

- MoE 模型中 expert 上投影后的 SwiGLU 激活。
- FP8/MXFP8 推理链路中的 grouped matmul 后处理。
- 下游 INT8 matmul 所需的 per-token 量化输入生成。

---

## 2. 数学公式

### 2.1 计算公式

```python
gmm_out_i = scaled_mm(a_i, b_i, scaled_a_i, scaled_b_i)
value_i, gate_i = gmm_out_i.chunk(2, dim=-1)
swiglu_i = value_i * sigmoid(value_i) * gate_i
scale_i = torch.amax(torch.abs(swiglu_i), dim=-1, keepdim=True) / 127
output_i = torch.clamp(torch.round(swiglu_i / scale_i), -127, 127).to(torch.int8)
output_scale_i = scale_i.squeeze(-1).to(torch.float32)
```

### 2.2 展开形式

```
gmm_out[m, n] = Σ(k=0..K-1) dequant(a[m, k]) * dequant(b[expert(m), k, n])
output[m, d] = Quant(SiLU(gmm_out[m, d]) * gmm_out[m, d + N/2])
```

其中 `d ∈ [0, N/2)`。

### 2.3 计算步骤

1. **expert 分组解析**：根据 `group_list` 顺序计算每个 expert 的 token 范围。
2. **MXFP8 scaled matmul**：调用 `pypto.scaled_mm` 计算 `[M_i, K] × [K, N] -> [M_i, N]`。
3. **SwiGLU 激活**：将输出切分为 value/gate，计算 `value * sigmoid(value) * gate`。
4. **BF16 往返对齐**：SwiGLU 输出先 cast 到 BF16，再 cast 回 FP32，对齐 golden。
5. **Per-token 量化**：按最后一维求最大绝对值，生成 INT8 输出和 FP32 scale。
6. **输出组装**：通过 `pypto.assemble` 写回全局输出。

---

## 3. 输入输出规格

### 3.1 输入张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `a` | `[M, K]` | FP8 E4M3 | token 输入 |
| `b` | `[E, K, N]` 或 `[E, N, K]` | FP8 E4M3 | expert 权重 |
| `scaled_a` | `[M, K/64, 2]` | E8M0FNU | token scale |
| `scaled_b` | `[E, K/64, N, 2]` 或 `[E, N, K/64, 2]` | E8M0FNU | expert 权重 scale |
| `group_list` | `[E]` | list[int] | 每个 expert 的 token 数 |

### 3.2 输出张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `out` | `[M, N/2]` | int8 | SwiGLU 后量化输出 |
| `out_quant` | `[M]` | float32 | per-token 量化 scale |

### 3.3 参数说明

- **M**：路由后 token 总数。
- **K**：matmul K 维。
- **N**：matmul 输出维，SwiGLU 后变为 `N/2`。
- **E**：expert 数量。

---

## 4. Shape 范围与约束

### 4.1 动态轴取值范围

| 轴 | 取值范围 | 说明 |
|----|---------|------|
| M | 16 | 当前内置 testcase6 覆盖 |
| K | 512 | 当前内置 testcase6 覆盖 |
| N | 7168 | 当前内置 testcase6 覆盖 |
| E | 2 | 当前内置 testcase6 覆盖 |

### 4.2 约束条件

1. **N 必须为偶数**：SwiGLU 需要二等分最后一维。
2. **group_list 总和必须等于 M**：保证输出组装覆盖全部 token。
3. **K 需要和 scale 对齐**：当前 scale 使用 `K // 64`，要求 K 能被 64 整除。
4. **输入 tensor 需连续**：建议进入 kernel 前保证 contiguous。
5. **量化 scale 不应为 0**：若某 token 全零，需额外处理除零风险；当前随机测试数据不会触发。

### 4.3 典型配置

| 配置名称 | M | K | N | E | group_list | 用途 |
|---------|---|---|---|---|------------|------|
| testcase6 | 16 | 512 | 7168 | 2 | `[7, 9]` | 非均匀 expert token 分布验证 |

---

## 5. 精度要求

### 5.1 数据类型转换

- 输入类型：`a`、`b` 为 FP8 E4M3。
- scale 类型：`scaled_a`、`scaled_b` 为 E8M0FNU。
- matmul 输出：FP32。
- SwiGLU 后对齐：BF16 → FP32。
- 量化输出：INT8。
- scale 输出：FP32。

### 5.2 精度容差

- **INT8 输出 RTOL**：0.001
- **INT8 输出 ATOL**：1
- **Scale 输出 RTOL**：0.0001
- **Scale 输出 ATOL**：0.0001

### 5.3 精度验证标准

```python
assert_allclose(golden, result, rtol=1e-3, atol=1)
assert_allclose(golden_quant, result_quant, rtol=1e-4, atol=1e-4)
```

---

## 6. 性能要求

### 6.1 计算特点

- **Cube 计算为主**：主要计算量来自 MXFP8 grouped matmul。
- **Vector 后处理**：SwiGLU、abs、amax、div、round、cast 属于 vector 计算。
- **按 expert 顺序处理**：当前 kernel 使用 `group_list` 顺序分段。

### 6.2 优化方向

1. cube tile shape 优化。
2. vector tile shape 优化。
3. expert 维并行化。
4. per-token 量化除零保护。
5. b_trans=True 路径补充测试。

---

## 7. 测试验证要求

### 7.1 功能测试

| 测试名称 | M | K | N | E | 验证点 |
|---------|---|---|---|---|--------|
| testcase6 | 16 | 512 | 7168 | 2 | MXFP8 GMM + SwiGLU + INT8 quant |

### 7.2 精度测试

- Golden 实现：PyTorch 参考实现。
- 对比方法：量化输出和 scale 分别逐元素对比。
- 通过标准：满足 `assert_allclose` 容差。

### 7.3 边界测试

- `group_list` 中存在 0 token expert。
- `b_trans=True`。
- `N` 较小但仍为偶数。
- SwiGLU 输出全零或接近全零。

---

## 8. 实现约束

### 8.1 API 映射要求

- 必须使用 PyPTO 新前端 `@pypto.frontend.jit`。
- 必须使用 `pypto.scaled_mm` 表达 MXFP8 scaled matmul。
- 必须使用 PyPTO vector API 表达 SwiGLU 和量化。
- 必须使用 `pypto.assemble` 组装分 expert 输出。

### 8.2 内存管理

- 输出 `out` 预分配为 `[M, N/2]` int8。
- 输出 `out_quant` 预分配为 `[M, 1]` FP32，返回前 squeeze。
- expert 分段输出按 `begin` 偏移写回。

### 8.3 兼容性

- PyPTO 版本要求：支持 `pypto.frontend.jit`。
- 硬件要求：华为昇腾 AI 处理器。
- 软件栈：支持 FP8/MXFP8 和 `torch_npu` 的 CANN 环境。

---

## 9. 参考实现

### 9.1 Golden 实现

位于 `tests/ops/experimental/matmul/grouped_matmul_swiglu_quant/gmm_swiglu_quant_golden.py` 中的 `compute_golden_result` 和 `gen_golden`：

```python
gmm_out = torch.matmul(x1_golden, weight_golden)
swiglu_out = swiglu(gmm_out)
swiglu_out = swiglu_out.to(torch.bfloat16).to(torch.float32)
quant_output, quant_scale_output = quant_pertoken(swiglu_out)
```
