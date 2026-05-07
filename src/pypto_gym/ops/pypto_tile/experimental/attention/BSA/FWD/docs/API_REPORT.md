# BSA Forward API 映射报告

## 概述

BSA Forward（`aclnnBlockSparseAttention`）前向算子实现块稀疏注意力计算，采用 Online Softmax 分块迭代策略。PyPTO 实现提供两条路径：

- **Sparse 路径**：通过 Mask2Idx 紧凑张量策略，内层循环仅遍历 `maxSel`（最大有效 KV 块数）而非 `numKB`（总 KV 块数）
- **Dense 路径**：当掩码全为 1 时，省去 mask 加载和 apply 开销，内层遍历 `numKB`

两条路径共享相同的 Online Softmax 迭代逻辑，输出 `attentionOut` 和 `softmaxLse`。

---

## PyTorch 操作分解

基于 `bsa_fwd_golden.py` 的 golden 实现，前向计算涉及以下核心操作：

| 阶段 | 操作 | 描述 |
|------|------|------|
| 数据转换 | `query.to(ftype)` | FP16 → FP32 精度提升 |
| 矩阵乘 | `torch.matmul(q_block, k_block.t())` | Q × K^T 计算注意力分数 |
| 标量乘 | `S * scale` | Softmax 缩放 (1/√d) |
| 行最大值 | `S.max(dim=-1).values` | 计算 amax（数值稳定） |
| 标量减 | `S - max.unsqueeze(-1)` | 减去行最大值 |
| 指数 | `torch.exp(...)` | 计算指数概率 |
| 行求和 | `P.sum(dim=-1)` | 计算 exp 归一化因子 |
| 矩阵乘 | `torch.matmul(P, v_block)` | P × V 加权求和 |
| 标量减 | `block_max - new_max` | Online softmax 校正因子 |
| 标量最大 | `torch.maximum(block_max, cur_max)` | 更新全局最大值 |
| 标量乘 | `block_sum * correction` | 校正旧累积值 |
| 标量加 | `sum_a + sum_b` | 合并 exp 求和 |
| 标量除 | `block_out / block_sum.unsqueeze(-1)` | 最终归一化输出 |
| 对数 | `torch.log(block_sum)` | 计算 LSE |
| 标量加 | `block_max + torch.log(block_sum)` | LSE = m + log(l) |

---

## PyPTO API 映射表

### 类型转换 API

| PyTorch 操作 | PyPTO API | 说明 |
|-------------|-----------|------|
| `tensor.to(torch.float32)` | `pypto.cast(tensor, pypto.DT_FP32)` | FP16 → FP32 精度提升 |
| `tensor.to(torch.float16)` | `pypto.cast(tensor, pypto.DT_FP16)` / `pypto.cast(tensor, dtype)` | FP32 → FP16 输出转换 |

### 维度操作 API

| PyTorch 操作 | PyPTO API | 说明 |
|-------------|-----------|------|
| `tensor.reshape(shape)` | `pypto.reshape(tensor, shape)` | 改变张量形状（如 `[BLOCK, D]` → `[1, BLOCK, D]`） |
| `tensor[offset:offset+size]` (2D切片) | `pypto.view(src, [rows, cols], [row_off, col_off])` | 按偏移量切片加载子块 |
| `tensor[...] = value` (写入子区域) | `pypto.assemble(src, [dim0_off, dim1_off, ...], dst)` | 将计算结果写入输出张量指定位置 |

### 运算 API

| PyTorch 操作 | PyPTO API | 说明 |
|-------------|-----------|------|
| `torch.matmul(A, B)` | `pypto.matmul(A, B, pypto.DT_FP32, a_trans=False, b_trans=True)` | 矩阵乘法，支持转置标志 |
| `A * scalar` | `pypto.mul(A, scalar)` | 标量乘法（softmax scale、校正因子） |
| `A + B` | `pypto.add(A, B)` | 标量/张量加法（累积器更新） |
| `A - B` | `pypto.sub(A, B)` | 标量/张量减法（exp 前减最大值） |
| `A / B` | `pypto.div(A, B)` | 标量除法（最终输出归一化 O/l） |
| `torch.exp(A)` | `pypto.exp(A)` | 指数运算（概率计算、校正因子） |
| `torch.log(A)` | `pypto.log(A)` | 对数运算（LSE = m + log(l)） |
| `torch.maximum(A, B)` | `pypto.maximum(A, B)` | 逐元素最大值（Online softmax max 更新） |

### 归约 API

| PyTorch 操作 | PyPTO API | 说明 |
|-------------|-----------|------|
| `tensor.max(dim=-1).values` | `pypto.amax(tensor, dim=-1, keepdim=True)` | 沿最后一维求最大值，保持维度 `[BLOCK, 1]` |
| `tensor.sum(dim=-1)` | `pypto.sum(tensor, dim=-1, keepdim=True)` | 沿最后一维求和，用于 exp 归一化因子 |

### 循环控制 API

| PyTorch 操作 | PyPTO API | 说明 |
|-------------|-----------|------|
| `for i in range(N)` | `for i in pypto.loop(N, name="...", idx_name="...")` | 编译期可识别的循环结构 |
| `i == 0` (首次迭代) | `pypto.is_loop_begin(i)` | 判断是否为循环首次迭代 |
| `i == N-1` (末次迭代) | `pypto.is_loop_end(i)` | 判断是否为循环末次迭代 |

### 中间张量创建 API

| PyTorch 操作 | PyPTO API | 说明 |
|-------------|-----------|------|
| `torch.zeros([shape])` | `pypto.tensor([shape], dtype, "name")` | 创建循环内中间累积器 |

### TileShape 控制 API

| API | 用途 | 配置 |
|-----|------|------|
| `pypto.set_vec_tile_shapes(*vtl)` | 设置向量操作 tile 形状 | 加载: `(128, 128)`，输出: `(16, 128, 128)` |
| `pypto.set_cube_tile_shapes(ct, ct, ct)` | 设置矩阵乘 tile 形状 | `([128,128], [128,128], [128,128])` |

### JIT 编译 API

| API | 用途 |
|-----|------|
| `@pypto.frontend.jit(**opts)` | PyPTO JIT 编译装饰器 |
| `pypto.Tensor([shape], dtype)` | 编译期张量类型注解 |

---

## 约束条件清单

### 数据类型约束

| 约束项 | 值 |
|--------|-----|
| Q / K / V / O 数据类型 | `DT_FP16` (FLOAT16) |
| softmaxLse 数据类型 | `DT_FP32` (FLOAT32) |
| 累积器数据类型 | `DT_FP32` (FP32 累积保证精度) |
| valid_mask 数据类型 | `DT_FP32` (wrapper 预转换，避免 BOOL→FP32 转换问题) |
| head_dim | 128 (固定) |

### Shape 约束

| 约束项 | 值 |
|--------|-----|
| Q 布局 | `[B, Hq, Sq, 128]` BNSD |
| K/V 布局 | `[B, Hkv, Skv, 128]` BNSD |
| GQA 头数约束 | `Hq >= Hkv` 且 `Hq % Hkv == 0` |
| block_shape_x | 64 的整数倍 |
| block_shape_y | ≥128 且为 64 的整数倍 |
| mask 形状 | `[B, Hq, ceil(Sq/X), ceil(Skv/Y)]` |

### 内存约束

| 约束项 | 说明 |
|--------|------|
| Cube TileShape 上限 | `[128, 128]`（CANN 9.0.0 安全配置） |
| Vector 加载 TileShape | `(128, 128)` |
| Vector 输出 TileShape | `(16, 128, 128)` 3-arg |
| 紧凑 KV 张量 | `[total_qblocks * maxSel * by, D]` |
| valid_mask 张量 | `[total_qblocks * maxSel * bx, by]` |

---

## 性能优化建议

1. **Dense 路径分离**：当 `block_sparse_mask.all() == True` 时自动切换到 dense kernel，省去 mask 加载、inv_mask 计算和 S_masked 应用开销

2. **紧凑 KV 策略**：sparse 路径通过 `_build_sparse_kv()` 预收集有效 KV 块，内层循环次数从 `numKB` 降低到 `maxSel`，稀疏率越低收益越大

3. **L1 复用**：`cube_l1_reuse_setting: {-1: 64}` 启用 Cube L1 缓存复用，Q 块在内层 KV 循环中仅加载一次

4. **调度模式**：`device_sched_mode: 3` 启用自动调度策略，优化多核任务分配

5. **Stitch 参数调优**：`stitch_function_num_initial=128`、`stitch_function_num_step=64` 控制子图切分粒度

6. **Kernel 缓存**：工厂函数 + 字典缓存模式，首次调用触发编译，后续同 shape 调用零编译开销

---

## Kernel 签名

### Sparse Forward Kernel

```python
@pypto.frontend.jit(**_make_jit_opts(cfg, total_outer=TOTAL_OUTER))
def kernel(
    q_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
    k_compact: pypto.Tensor([B * Hq * numQB * maxSel * by, D], pypto.DT_FP16),
    v_compact: pypto.Tensor([B * Hq * numQB * maxSel * by, D], pypto.DT_FP16),
    valid_mask: pypto.Tensor([B * Hq * numQB * maxSel * bx, by], pypto.DT_FP32),
    output_3d: pypto.Tensor([B * Hq, Sq, D], pypto.DT_FP16),
    lse_2d: pypto.Tensor([B * Hq, Sq], pypto.DT_FP32),
):
```

### Dense Forward Kernel

```python
@pypto.frontend.jit(**_make_jit_opts(cfg, total_outer=TOTAL_OUTER))
def kernel(
    q_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
    k_2d: pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
    v_2d: pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
    output_3d: pypto.Tensor([B * Hq, Sq, D], pypto.DT_FP16),
    lse_2d: pypto.Tensor([B * Hq, Sq], pypto.DT_FP32),
):
```

### 公共 Wrapper 签名

```python
def block_sparse_attention_forward(
    query, key, value, block_sparse_mask,
    actual_seq_lengths=None, actual_seq_lengths_kv=None,
    block_shape=None, cfg=DEFAULT_CONFIG,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns: (attention_out [B, Hq, Sq, D], softmax_lse [B, Hq, Sq])
```
