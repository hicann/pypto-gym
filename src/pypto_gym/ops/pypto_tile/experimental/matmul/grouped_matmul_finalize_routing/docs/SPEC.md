# grouped_matmul_finalize_routing 算子需求规格

## 1. 算子概述

### 1.1 算子名称

`grouped_matmul_finalize_routing` - MoE grouped matmul finalize routing 融合算子。

### 1.2 功能描述

`grouped_matmul_finalize_routing` 实现 MoE 路由后 grouped matmul 的 finalize 阶段。该算子按 expert 执行 MXFP8 scaled matmul，并将每个 token 的 expert 输出按照 `logit` 和 `row_index` 累加到最终输出，同时支持 shared expert 输出叠加。

### 1.3 应用场景

- MoE 模型中的 expert 路由后处理。
- FP8/MXFP8 权重量化推理路径。
- grouped matmul 输出到 token 维度输出的 scatter-add finalize。

---

## 2. 数学公式

### 2.1 计算公式

```python
mm_i = scaled_mm(x1_i, x2_i, pertoken_scale_i, scale)
if has_logit:
    mm_i = mm_i * logit_i.unsqueeze(-1)
out.index_add_(0, row_index_i, mm_i)
if has_shared_input:
    out[offset:offset+batch] += shared_input * shared_input_weight
```

### 2.2 展开形式

```
out[row_index[t], n] += logit[t] * Σ(k=0..K-1) dequant(x1[t, k]) * dequant(x2[e, k, n])
```

其中：

- `t` 为路由后的 token 索引；
- `e` 为 token 所属 expert；
- `dequant` 表示通过 MXFP8 数据和 E8M0FNU scale 还原到 FP32 参与计算。

### 2.3 计算步骤

1. **expert 分组解析**：根据 `group_list` 或均匀分组方式确定每个 expert 的 token 范围。
2. **MXFP8 scaled matmul**：调用 `pypto.scaled_mm` 计算 `[M_i, K] × [K, N] -> [M_i, N]`。
3. **logit 加权**：将 `[M_i]` 形状的 logit 扩展为 `[M_i, 1]` 后与 matmul 输出相乘。
4. **row_index 累加**：按 `row_index` 将结果 scatter-add 到 `out`。
5. **shared input 叠加**：将 shared expert 输出按 `shared_input_weight` 加到指定输出区间。

---

## 3. 输入输出规格

### 3.1 输入张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `x1` | `[M, K]` | FP8 E4M3/E5M2 | 路由后 token 输入 |
| `x2` | `[E, K, N]` 或 `[E, N, K]` | FP8 E4M3/E5M2 | expert 权重 |
| `scale` | `[ceil(K/64), N, 2]` 或 `[N, ceil(K/64), 2]` | E8M0FNU | 权重 scale |
| `pertoken_scale` | `[M, ceil(K/64), 2]` | E8M0FNU | token scale |
| `group_list` | `[E]` | int64 | expert 分组信息 |
| `shared_input` | `[batch, N]` | bfloat16 | shared expert 输出 |
| `logit` | `[M]` | float32 | token 权重 |
| `row_index` | `[M]` | int64 | 输出行索引 |
| `out` | `[batch, N]` | float32 | 输出初始值 |

### 3.2 输出张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `output` | `[batch, N]` | float32 | finalize routing 输出 |

### 3.3 参数说明

- **batch**：最终输出行数。
- **M**：路由后 token 总数。
- **K**：matmul K 维。
- **N**：输出特征维。
- **E**：expert 数量。

---

## 4. Shape 范围与约束

### 4.1 动态轴取值范围

| 轴 | 取值范围 | 说明 |
|----|---------|------|
| batch | {64, 128, 256} | 输出 batch 维 |
| M | {128, 256, 768} | token 总数 |
| K | {5120, 6144, 7168, 8192} | matmul K 维 |
| N | 4096 | 输出列数 |
| E | {8, 16, 32} | expert 数量 |

### 4.2 约束条件

1. **transpose_x1=False**：当前实现不支持转置 `x1`。
2. **transpose_x2 支持 True/False**：当前内置测试覆盖 `True`。
3. **M 可被 E 均匀切分**：当前 kernel 使用 `M // E` 作为每个 expert 的 token 数。
4. **group_list 支持两种格式**：golden 支持前缀和和计数格式，kernel 当前按均匀切分执行。
5. **row_index 合法**：所有索引必须落在 `[0, batch)`。
6. **输入 tensor 需连续**：建议调用 kernel 前保证输入为 contiguous。

### 4.3 典型配置

| 配置名称 | batch | M | K | N | E | 用途 |
|---------|-------|---|---|---|---|------|
| P0_32E_K6144 | 128 | 768 | 6144 | 4096 | 32 | 标准 32 expert 场景 |
| P0_32E_K8192 | 256 | 768 | 8192 | 4096 | 32 | 大 K 场景 |
| P0_8E | 64 | 128 | 5120 | 4096 | 8 | 小 expert 数场景 |
| P0_E5M2 | 64 | 256 | 7168 | 4096 | 16 | FP8 E5M2 场景 |

---

## 5. 精度要求

### 5.1 数据类型转换

- 输入类型：`x1`、`x2` 为 FP8 E4M3/E5M2。
- scale 类型：`scale`、`pertoken_scale` 为 E8M0FNU。
- 中间计算：`pypto.scaled_mm` 输出 FP32。
- 输出类型：FP32。

### 5.2 精度容差

- **相对容差 (RTOL)**：0.001
- **绝对容差 (ATOL)**：0.001

### 5.3 精度验证标准

输出结果与 golden 实现对比需满足：

```python
numpy.testing.assert_allclose(result, golden, rtol=1e-3, atol=1e-3)
```

---

## 6. 性能要求

### 6.1 计算特点

- **Cube 计算为主**：主要计算量来自 grouped scaled matmul。
- **Vector 后处理**：logit 加权和 row_index accumulate 属于 vector/scatter 路径。
- **expert 并行**：expert 维度可并行执行。

### 6.2 优化方向

1. cube tile shape 优化。
2. vector tile shape 优化。
3. expert 维并行调度。
4. row_index scatter-add 写回冲突优化。
5. shared input 叠加融合进 kernel。

---

## 7. 测试验证要求

### 7.1 功能测试

| 测试名称 | batch | M | K | N | E | 验证点 |
|---------|-------|---|---|---|---|--------|
| case1 | 128 | 768 | 6144 | 4096 | 32 | FP8 E4M3 + 32 experts |
| case2 | 256 | 768 | 8192 | 4096 | 32 | 大 K + 大 batch |
| case3 | 64 | 128 | 5120 | 4096 | 8 | 小 M + 8 experts |
| case4 | 64 | 256 | 7168 | 4096 | 16 | FP8 E5M2 |

### 7.2 精度测试

- Golden 实现：PyTorch 参考实现。
- 对比方法：与 PyPTO 输出逐元素对比。
- 通过标准：`rtol=1e-3, atol=1e-3`。

### 7.3 边界测试

- `has_logit=False`。
- `has_shared_input=False`。
- `group_list_type=0`。
- 非均匀 expert token 分布。
- `shared_input_offset != 0`。

---

## 8. 实现约束

### 8.1 API 映射要求

- 必须使用 PyPTO JIT kernel 实现。
- 必须使用 `pypto.scaled_mm` 表达 MXFP8 scaled matmul。
- 必须使用 `pypto.index_put_(..., accumulate=True)` 表达 row_index scatter-add。

### 8.2 内存管理

- 输入 tensor 建议 contiguous。
- `out` 为 FP32 输出初值，支持原地累加。
- shared input 当前在 host 侧预处理到 `out` 初值。

### 8.3 兼容性

- PyPTO 版本要求：与 CANN 版本匹配。
- 硬件要求：华为昇腾 AI 处理器。
- 软件栈：支持 FP8/MXFP8 和 `torch_npu` 的 CANN 环境。

---

## 9. 参考实现

### 9.1 Golden 实现

位于 `tests/ops/experimental/matmul/grouped_matmul_finalize_routing/gmm_finalize_routing_golden.py` 中的 `gen_golden`：

```python
def gen_golden(inputs: FinalizeRoutingGoldenInputs) -> torch.Tensor:
    golden = inputs.out.clone()
    for expert_idx in range(cfg.num_experts):
        mm_result = _compute_mxfp8_matmul_golden(...)
        if cfg.has_logit:
            mm_result = mm_result * inputs.logit[start:end].to(torch.float32).unsqueeze(-1)
        golden.index_add_(0, inputs.row_index[start:end].to(torch.int64), mm_result)
    return golden
```
