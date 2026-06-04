# quant_batch_matmul 算子需求规格

## 1. 算子概述

### 1.1 算子名称

`quant_batch_matmul` — 量化 Batch MatMul 算子族（当前实现 **F-T pertensor** 变体）。

### 1.2 功能描述

在 batch 维度上对多组矩阵乘进行量化计算，将 FP8 输入与全局（pertensor）scale 相乘后量化为 INT8 输出。语义对齐 `aclnnQuantMatmulV5` 的 **F-T** 路径：

- **F**：`x1` 为 `[batch, m, k]`（左矩阵不转置）。
- **T**：`x2` 为 `[batch, n, k]`，计算时等价于 `x2^T` 参与 matmul（`b_trans=True`）。

### 1.3 应用场景

- 大模型推理中的 FP8 量化 batch matmul。
- 需要 INT8 输出、scale 为标量的 pertensor 量化路径。
- PyPTO tile 算子库中与 CANN QuantMatmul 对齐的实验性实现。

---

## 2. 数学公式

### 2.1 计算公式

```python
acc = torch.matmul(x1.float(), x2.float().transpose(-1, -2))   # [batch, m, n]
combined_scale = mask_fixpipe(x1_scale * x2_scale)             # 标量
out = round(clamp(acc * combined_scale, -128, 127)).to(int8)
```

当 `x1_scale is None` 时：`combined_scale = mask_fixpipe(x2_scale)`。

### 2.2 展开形式

```
out[b, m, n] = round( Σ_k x1[b,m,k] * x2[b,n,k] * s ),  s = mask_fixpipe(s1 * s2)
```

其中 `s1` 来自 `x1_scale`（可选），`s2` 来自 `x2_scale`。

### 2.3 计算步骤

1. **读取 scale**：从 `(1,)` 形状的张量取标量，相乘后做 fixpipe 掩码。
2. **Batch matmul**：对每个 `b`，计算 `[m,k] @ [k,n]`。
3. **Fixpipe 量化**：在 matmul 路径通过 `extend_params["scale"]` 完成缩放并输出 INT8。
4. **写回输出**：按 N 分块 assemble 到 `[batch, m, n]`。

---

## 3. 输入输出规格

### 3.1 输入张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `x1` | `[batch, m, k]` | FP8 E4M3 | 左矩阵 |
| `x2` | `[batch, n, k]` | FP8 E4M3 | 右矩阵（物理存 K 维） |
| `x1_scale` | `(1,)` 或 `None` | FP32 | 左 scale，可选 |
| `x2_scale` | `(1,)` | FP32 | 右 scale |

### 3.2 输出张量

| 名称 | Shape | DType | 说明 |
|------|-------|-------|------|
| `out` | `[batch, m, n]` | INT8 | 量化结果 |

### 3.3 参数说明

- **batch**：batch 维大小，≥ 1。
- **m / k / n**：矩阵乘逻辑维度。
- **n_block_size**：N 轴 kernel 内循环块大小。
- **m_tile_shape / k_tile_shape / n_tile_shape**：Cube tiling 配置。

---

## 4. Shape 范围与约束

### 4.1 动态轴取值范围

| 轴 | 取值范围（测试覆盖） | 说明 |
|----|---------------------|------|
| batch | 1 ~ 8 | 多 batch |
| m | 1 ~ 32 | 含 m=1 向量乘场景 |
| k | 128 ~ 7168 | 含 5120、6144 等 |
| n | 128 ~ 4096 | 需能被 n_block 友好切分 |

### 4.2 约束条件

1. **仅 F-T 布局**：不支持 `transpose_x1`/`transpose_x2` 配置切换。
2. **pertensor scale**：`x1_scale`、`x2_scale` 均为标量张量 `(1,)`。
3. **dtype 锁定**：输入 FP8E4M3，输出 INT8。
4. **fixpipe scale**：kernel 与 golden 使用相同位掩码 `0xFFFFE000`。
5. **尾块 N**：`valid_n = min(n - n_off, n_block)`，assemble 带 `valid_shape`。

### 4.3 典型配置

| 配置名称 | batch | m | k | n | x1Scale | 用途 |
|---------|-------|---|---|---|---------|------|
| default | 4 | 4 | 7168 | 2048 | yes | 默认回归用例 |
| small | 1 | 1 | 6144 | 768 | yes | 小 batch、大 K |
| wide_n | 8 | 4 | 5120 | 3072 | yes | 大 N、多 batch |

---

## 5. 精度要求

### 5.1 数据类型转换

- 输入：`x1`、`x2` 为 FP8 E4M3（PyPTO `DT_FP8E4M3`）。
- 中间：matmul + fixpipe scale（硬件/fixpipe 语义）。
- 输出：INT8（`DT_INT8`）。
- Golden：CPU FP32 matmul → scale → `round` → `clamp` → INT8。

### 5.2 精度容差

- **相对容差 (RTOL)**：0.001
- **绝对容差 (ATOL)**：0

### 5.3 精度验证标准

```python
numpy.testing.assert_allclose(result, golden, rtol=1e-3, atol=0)
```

逐点相对误差：`rel_err = |result - golden| / (|golden| + 1e-6)`，要求 `max(rel_err) <= 1e-3`。

---

## 6. 性能要求

### 6.1 计算特点

- **Cube 计算为主**：`pypto.matmul` 到 INT8。
- **Vector 辅助**：reshape、view、assemble 在 AIV 路径。
- **双层循环**：batch × N-block，任务数随 `batch * ceil(n / n_block)` 增长。

### 6.2 优化方向

1. 增大 `n_block_size`（在 L1 允许范围内）减少 N 循环次数。
2. 按 m/k/n 自适应 `m_tile_shape`、`k_tile_shape`。
3. 保持 F-4 展平，避免 kernel 内多余 3D reshape。
4. scale 通过 `extend_params` 融合，避免后处理 vector mul。

---

## 7. 测试验证要求

### 7.1 功能测试

| 验证点 | 说明 |
|--------|------|
| F-T layout | `x1`/`x2` 物理 shape 与 `get_x1_shape`/`get_x2_shape` 一致 |
| x1Scale=null | 仅 `x2_scale` 路径 |
| x1Scale 非空 | `x1_scale * x2_scale` 路径 |
| 输出 shape/dtype | `[batch, m, n]` INT8 |

### 7.2 精度测试

- Golden：`gen_golden`（PyTorch CPU）。
- 对比：NPU kernel 输出 vs golden。
- 通过标准：`rtol=1e-3, atol=0`。

### 7.3 边界测试

- `m=1` 极窄 M。
- `n` 非 `n_block` 整数倍（尾块 `valid_n`）。
- 大 `k`（5120、6144、7168）。
- 多 batch（8）。

---

## 8. 实现约束

### 8.1 API 映射要求

- 必须使用 PyPTO JIT kernel（`@pypto.frontend.jit`）。
- 必须使用 `pypto.matmul(..., pypto.DT_INT8, b_trans=True, extend_params={"scale": ...})`。
- 必须使用 `pypto.assemble` 写回分块结果。

### 8.2 内存管理

- Host 侧输入 `.npu().contiguous()`。
- 输出 `torch.zeros(..., int8).npu()` 预分配。
- `reshape(..., inplace=True)` 展平 x1/x2。

### 8.3 兼容性

- PyPTO 版本与 CANN 匹配。
- 硬件：华为昇腾 AI 处理器（测试标记 `@pytest.mark.soc("950", "910")`）。
- 环境变量：`TILE_FWK_DEVICE_ID`、`PTO_TILE_LIB_CODE_PATH`。

---

## 9. 参考实现

### 9.1 Golden 实现

位于 `tests/ops/experimental/matmul/quant_batch_matmul/quant_batch_matmul_golden.py`：

```python
def gen_golden(inputs, config):
    acc = torch.matmul(x1.float(), x2.float().transpose(-1, -2))
    _, golden_scale = compute_combined_scale(inputs.x1_scale, inputs.x2_scale)
    out = torch.round((acc * golden_scale).clamp(-128, 127))
    return out.to(torch.int8)
```

### 9.2 Host 入口

```python
def quant_batch_matmul(inputs, config) -> torch.Tensor:
    kernel_scale, _ = compute_combined_scale(inputs.x1_scale, inputs.x2_scale)
    launch_config = replace(config, combined_scale=kernel_scale)
    quant_batch_matmul_kernel(x1, x2, out, launch_config)
    return out
```
