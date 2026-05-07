# BSA Backward API 映射报告

## 1. 概述

BSA Backward（`aclnnBlockSparseAttentionGrad`）实现块稀疏注意力反向传播，采用 **重计算（Recompute）策略**：不保存前向中间矩阵 S/P，仅利用前向输出的 O 和 LSE 在反向时重新计算 S 和 P。

为实现高效的梯度累积，反向传播拆分为 **两个独立 kernel**：
- **dQ kernel**：外层遍历 Q 块，内层遍历紧凑 KV 块，局部累积 dQ 后一次性写出
- **dK/dV kernel**：外层遍历 KV 块，内层遍历紧凑 Q 块，局部累积 dK/dV 后一次性写出

每个 kernel 均有 **sparse 和 dense 两种变体**，共 4 个 kernel 实现。

---

## 2. PyTorch 操作分解

基于 golden 实现（`bsa_bwd_golden.py`），反向传播的核心操作如下：

### 2.1 softmaxGrad（D_row）计算

```python
sg_block = (do_block * O_f[b, h_q, q_start:q_end, :]).sum(dim=-1)
```

- 逐元素乘法 `dO * O`
- 沿 D 维度求和，得到每行的 softmaxGrad 标量 `[BLOCK]`

### 2.2 S 矩阵重计算

```python
S = torch.matmul(q_block, k_block.t()) * scale
```

- 矩阵乘法 `Q @ K^T`，含 softmax scale

### 2.3 P 矩阵重计算

```python
P = torch.exp(S - lse_block.unsqueeze(-1))
```

- 指数运算 `exp(S * scale - LSE)`

### 2.4 dS 计算

```python
dS = P * (torch.matmul(do_block, v_block.t()) - sg_block.unsqueeze(-1))
```

- 矩阵乘法 `dO @ V^T` 得到 dP
- 逐元素乘法 `P * (dP - softmaxGrad)` 得到 dS

### 2.5 梯度累积

```python
dQ[b, h_q, q_start:q_end, :] += torch.matmul(dS, k_block) * scale
dK[b, h_kv, k_start:k_end, :] += torch.matmul(dS.t(), q_block) * scale
dV[b, h_kv, k_start:k_end, :] += torch.matmul(P.t(), do_block)
```

- dQ: `dS @ K * scale`
- dK: `dS^T @ Q * scale`
- dV: `P^T @ dO`（**无 scale**）

---

## 3. PyPTO API 映射表

### 3.1 类型转换 API

| PyTorch 操作 | PyPTO API | 说明 |
|-------------|-----------|------|
| `.to(torch.float32)` | `pypto.cast(tensor, pypto.DT_FP32)` | FP16 → FP32 精度提升 |
| `.to(torch.float16)` | `pypto.cast(tensor, dtype)` | FP32 → FP16 精度回退（dtype=DT_FP16） |

**关键场景**：
- `do_o_fp32 = pypto.cast(do_o, pypto.DT_FP32)`：dO*O 乘积转 FP32 后再 sum，确保 D_row 精度
- `dP_fp16 = pypto.cast(dP, dtype)`：dP 转 FP16 用于后续 matmul
- `dq_fp16 = pypto.cast(dq_new, dtype)`：累积结果 FP32→FP16 输出

### 3.2 维度操作 API

| PyTorch 操作 | PyPTO API | 说明 |
|-------------|-----------|------|
| `tensor[ofs:ofs+sz, :]` | `pypto.view(tensor, [BLOCK, D], [row_ofs, 0])` | 2D 切片访问（行偏移） |
| `output[ofs:ofs+sz, :] = value` | `pypto.assemble(value, [row_ofs, 0], output)` | 2D 写入（覆写语义） |
| `torch.zeros(...)` | `pypto.tensor([BLOCK, D], pypto.DT_FP32, "name")` | 局部累积器分配 |
| `acc[:] = new_val` | `acc[:] = new_val` | 累积器更新 |

### 3.3 运算 API

| PyTorch 操作 | PyPTO API | 说明 |
|-------------|-----------|------|
| `A @ B^T` | `pypto.matmul(A, B, DT_FP32, a_trans=False, b_trans=True)` | QK^T / dO@V^T |
| `A^T @ B` | `pypto.matmul(A, B, DT_FP32, a_trans=True, b_trans=False)` | dS^T@Q / P^T@dO |
| `A @ B` | `pypto.matmul(A, B, DT_FP32, a_trans=False, b_trans=False)` | dP@K（dQ）/ dP@Q（dK） |
| `A * B` | `pypto.mul(A, B)` | 逐元素乘法 |
| `A + B` | `pypto.add(A, B)` | 逐元素加法 |
| `A - B` | `pypto.sub(A, B)` | 逐元素减法 |
| `torch.exp(A)` | `pypto.exp(A)` | 指数运算（P 重计算） |
| `A * scalar` | `pypto.mul(A, softmax_scale)` | 标量乘法（scale 应用） |

### 3.4 归约 API

| PyTorch 操作 | PyPTO API | 说明 |
|-------------|-----------|------|
| `tensor.sum(dim=-1, keepdim=True)` | `pypto.sum(tensor, dim=-1, keepdim=True)` | 行级求和（softmaxGrad） |

**注意**：`pypto.sum` 的输出 dtype 与输入一致，因此必须在 cast 到 FP32 后再 sum：
```python
do_o_fp32 = pypto.cast(do_o, pypto.DT_FP32)  # FP16 → FP32
D_row = pypto.sum(do_o_fp32, dim=-1, keepdim=True)  # FP32 求和
```

### 3.5 累积器 API

| 操作 | PyPTO API | 说明 |
|------|-----------|------|
| 分配累积器 | `pypto.tensor([BLOCK, D], pypto.DT_FP32, "dq_acc")` | FP32 局部累积器 |
| 循环首次 | `pypto.is_loop_begin(idx)` | 首次迭代：初始化累积器 |
| 循环末次 | `pypto.is_loop_end(idx)` | 末次迭代：写出结果 |
| 累积器赋值 | `acc[:] = value` | 更新累积器 |

**累积模式**（避免 `pypto.assemble` 覆写语义冲突）：
```python
if pypto.is_loop_begin(v_idx):
    if pypto.is_loop_end(v_idx):       # 仅一次迭代
        assemble(cast(contrib, dtype), ...)
    else:
        dq_acc[:] = dq_contrib          # 初始化
else:
    dq_new = pypto.add(dq_acc, dq_contrib)
    if pypto.is_loop_end(v_idx):       # 最终迭代
        assemble(cast(dq_new, dtype), ...)
    else:
        dq_acc[:] = dq_new              # 继续累积
```

---

## 4. 约束条件清单

### 4.1 数据类型约束

| 参数 | 类型 | 说明 |
|------|------|------|
| Q, K, V, dO, O | FP16 (DT_FP16) | 输入/输出数据 |
| softmaxLse | FP32 (DT_FP32) | 前向 LSE |
| dQ, dK, dV | FP16 (DT_FP16) | 梯度输出 |
| blockSparseMask | BOOL/UINT8 | 块稀疏掩码 |
| D_row (softmaxGrad) | FP32 | **必须在 FP32 下计算** |
| dS, dP 内部累积 | FP32 | 梯度累积精度 |

### 4.2 Shape 约束

| 参数 | 约束 | 说明 |
|------|------|------|
| head_dim (D) | = 128 | 固定值 |
| block_shape_x (bx) | 64 的倍数 | Q 块大小，默认 256 |
| block_shape_y (by) | ≥128, 64 的倍数 | KV 块大小，默认 512 |
| Hq | ≥ Hkv, Hq % Hkv == 0 | GQA 约束 |
| Sq, Skv | 任意正整数 | 非对齐时自动 pad |

### 4.3 内存约束

| 紧凑张量 | Shape | 说明 |
|----------|-------|------|
| k_compact (dQ sparse) | `[B*Hq*numQB*maxSel*by, D]` | 紧凑 KV 数据 |
| v_compact (dQ sparse) | 同上 | 同上 |
| valid_mask (dQ sparse) | `[B*Hq*numQB*maxSel*bx, by]` | 稀疏掩码 |
| q_compact (dK/dV sparse) | `[B*Hkv*numKB*maxInner*bx, D]` | 紧凑 Q 数据 |
| do_compact (dK/dV sparse) | 同上 | 紧凑 dO 数据 |
| o_compact (dK/dV sparse) | 同上 | 紧凑 O 数据 |
| lse_compact (dK/dV sparse) | `[B*Hkv*numKB*maxInner*bx, 1]` | 紧凑 LSE（填充位=1e30） |
| inner_mask (dK/dV sparse) | `[B*Hkv*numKB*maxInner*bx, by]` | 内层掩码 |

### 4.4 Kernel 拆分约束

- dQ kernel 和 dK/dV kernel **必须拆分为独立 kernel**，因为 `pypto.assemble` 是覆写语义（非累加），多个 Q 块写同一 KV 位置会互相覆盖
- Dense 变体使用直接 2D 访问（无紧凑张量），减少 wrapper 开销
- Dense dQ 采用 sub-block 分割（256→2×128），增加外层并行度

---

## 5. 性能优化建议

### 5.1 Per-kernel L1 Reuse 调优

| Kernel | cube_l1_reuse_setting | 原因 |
|--------|----------------------|------|
| Dense dQ | {-1: 64} | 大 shape 内层循环长（numKB），高 L1 复用收益显著 |
| Sparse dQ | {-1: 16} | 内层循环短（maxSel 通常 1~3），过高 L1 浪费 |
| Sparse dK/dV | {-1: 16} | 中等内层循环，适度 L1 |
| Dense dK/dV | {-1: 16} | 小 shape 场景为主 |

### 5.2 Sub-block 分割（Dense dQ）

将每个 Q-block（bx=256）拆为 SUB_SPLIT=2 个 128 行子块：
- TOTAL_OUTER: `B*Hq*numQB*2`，外层任务数翻倍
- 提升多核并行度，尤其对小 shape（如 256×256：8→16 任务）
- matmul 的 M 维从 256 降为 128，单 task 计算量减少但整体吞吐提升

### 5.3 其他建议

- **Kernel 工厂缓存**：首次调用某 shape 触发 JIT 编译，后续调用命中缓存零编译开销
- **torch.npu.synchronize()**：紧凑张量构建后必须同步，确保 NPU 数据就绪
- **避免动态维度**：CANN 9.0.0 不支持 `pypto.DYNAMIC`，使用工厂函数 + 字典缓存模式

---

## 6. Kernel 签名

### 6.1 Sparse dQ Kernel

```python
def _get_dq_kernel(B, Hq, Hkv, Sq, D, numQB, maxSel, cfg):
    @pypto.frontend.jit(...)
    def kernel(
        q_2d:      pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        k_compact: pypto.Tensor([B * Hq * numQB * maxSel * by, D], pypto.DT_FP16),
        v_compact: pypto.Tensor([B * Hq * numQB * maxSel * by, D], pypto.DT_FP16),
        do_2d:     pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        o_2d:      pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        lse_2d:    pypto.Tensor([B * Hq * Sq, 1], pypto.DT_FP32),
        valid_mask:pypto.Tensor([B * Hq * numQB * maxSel * bx, by], pypto.DT_FP32),
        dq_2d:     pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
    ):
```

### 6.2 Dense dQ Kernel

```python
def _get_dense_dq_kernel(B, Hq, Hkv, Sq, Skv, D, numQB, numKB, cfg):
    @pypto.frontend.jit(...)
    def kernel(
        q_2d:  pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        k_2d:  pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        v_2d:  pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        do_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        o_2d:  pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        lse_2d:pypto.Tensor([B * Hq * Sq, 1], pypto.DT_FP32),
        dq_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
    ):
```

### 6.3 Sparse dK/dV Kernel

```python
def _get_dk_dv_kernel(B, Hq, Hkv, Sq, Skv, D, numQB, numKB, maxInner, cfg):
    @pypto.frontend.jit(...)
    def kernel(
        q_compact:  pypto.Tensor([B * Hkv * numKB * maxInner * bx, D], pypto.DT_FP16),
        do_compact: pypto.Tensor([B * Hkv * numKB * maxInner * bx, D], pypto.DT_FP16),
        o_compact:  pypto.Tensor([B * Hkv * numKB * maxInner * bx, D], pypto.DT_FP16),
        lse_compact:pypto.Tensor([B * Hkv * numKB * maxInner * bx, 1], pypto.DT_FP32),
        inner_mask: pypto.Tensor([B * Hkv * numKB * maxInner * bx, by], pypto.DT_FP32),
        k_2d:       pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        v_2d:       pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        dk_2d:      pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        dv_2d:      pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
    ):
```

### 6.4 Dense dK/dV Kernel

```python
def _get_dense_dk_dv_kernel(B, Hq, Hkv, Sq, Skv, D, numQB, numKB, cfg):
    @pypto.frontend.jit(...)
    def kernel(
        q_2d:  pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        do_2d: pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        o_2d:  pypto.Tensor([B * Hq * Sq, D], pypto.DT_FP16),
        lse_2d:pypto.Tensor([B * Hq * Sq, 1], pypto.DT_FP32),
        k_2d:  pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        v_2d:  pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        dk_2d: pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
        dv_2d: pypto.Tensor([B * Hkv * Skv, D], pypto.DT_FP16),
    ):
```
