# BSA Backward 设计文档

## 1. 设计概述

BSA Backward（`aclnnBlockSparseAttentionGrad`）是块稀疏注意力反向传播算子的 PyPTO 实现，面向华为昇腾 910B NPU。

### 1.1 核心设计决策

- **重计算策略**：不保存前向中间矩阵 S/P，仅利用前向输出的 O 和 softmaxLse 在反向时重新计算 S 和 P，节省显存
- **双 kernel 拆分**：将反向传播拆分为 dQ kernel 和 dK/dV kernel 两个独立 kernel，解决 `pypto.assemble` 覆写语义（非累加）导致的梯度覆盖问题
- **Dense/Sparse 双路径**：检测 mask 是否为全 1，自动选择 dense（无紧凑张量开销）或 sparse（紧凑张量跳过无效块）路径
- **FP32 累积**：所有梯度在 FP32 精度下累积，最终 cast 到 FP16 输出

### 1.2 Kernel 拆分必要性

| Kernel | 外层循环 | 内层循环 | 累积目标 | 拆分原因 |
|--------|---------|---------|---------|---------|
| dQ | Q 块 (`B*Hq*numQB`) | 紧凑 KV 块 (`maxSel`) | dQ 按 Q 块局部累积 | 每个 Q 块写入唯一位置，无冲突 |
| dK/dV | KV 块 (`B*Hkv*numKB`) | 紧凑 Q 块 (`maxInner`) | dK/dV 按 KV 块局部累积 | 多个 Q 块贡献同一 KV 块，必须在同一 kernel 内累积后一次性写出 |

若在单个 kernel 中以 Q 块→KV 块顺序计算，多个 Q 块写同一 KV 位置时 `pypto.assemble` 会互相覆盖。

---

## 2. 计算图设计

### 2.1 整体反向计算流程

```
输入: dO, Q, K, V, O, softmaxLse, blockSparseMask
  │
  ├─ Step 1: 预处理
  │    ├─ pad to block-aligned (Q/K/V/dO/O)
  │    ├─ reshape 4D → 2D
  │    ├─ LSE padding (填充位 = 1e30 → exp(S - 1e30) ≈ 0)
  │    └─ 分配 dQ/dK/dV 输出缓冲 (FP16, 零初始化)
  │
  ├─ Step 2: 路径选择
  │    └─ is_dense = _is_dense_mask(mask) AND is_aligned
  │
  ├─ Step 3: dQ 计算
  │    ├─ [Sparse] _build_sparse_kv() → 紧凑 K/V + valid_mask
  │    │              torch.npu.synchronize()
  │    │              _get_dq_kernel() → dQ kernel
  │    ├─ [Dense]  _get_dense_dq_kernel() → dense dQ kernel (sub-block split)
  │    └─ dQ kernel 内部:
  │         ├─ D_row = sum(cast(dO * O, FP32), dim=-1)    # softmaxGrad
  │         ├─ for each KV block:
  │         │    ├─ S = Q @ K^T * scale                   # 重计算
  │         │    ├─ P = exp(S - LSE)                       # 重计算
  │         │    ├─ P_masked = P * valid_mask               # 应用掩码
  │         │    ├─ dP = P_masked * (dO @ V^T - D_row)    # 注意力概率梯度
  │         │    └─ dq_acc += dP @ K * scale               # 梯度累积
  │         └─ assemble(cast(dq_acc, FP16)) → dq_2d
  │
  ├─ Step 4: dK/dV 计算
  │    ├─ [Sparse] _build_sparse_q_dkdv() → 紧凑 Q/dO/O/LSE + inner_mask
  │    │              torch.npu.synchronize()
  │    │              _get_dk_dv_kernel() → dK/dV kernel
  │    ├─ [Dense]  _get_dense_dk_dv_kernel() → dense dK/dV kernel
  │    └─ dK/dV kernel 内部:
  │         ├─ for each KV block (outer):
  │         │    ├─ for each Q block (inner):
  │         │    │    ├─ D_row = sum(cast(dO * O, FP32), dim=-1)
  │         │    │    ├─ S = Q @ K^T * scale
  │         │    │    ├─ P = exp(S - LSE)
  │         │    │    ├─ P_masked = P * inner_mask
  │         │    │    ├─ dP = P_masked * (dO @ V^T - D_row)
  │         │    │    ├─ dk_acc += dP^T @ Q * scale        # dK 梯度累积
  │         │    │    └─ dv_acc += P_masked^T @ dO          # dV 梯度累积（无 scale）
  │         │    └─ assemble(cast(dk_acc, FP16)) → dk_2d
  │         │       assemble(cast(dv_acc, FP16)) → dv_2d
  │
  └─ Step 5: 输出裁剪
       ├─ dq = dq_2d.reshape(B, Hq, Sq_pad, D)[:, :, :Sq, :]
       ├─ dk = dk_2d.reshape(B, Hkv, Skv_pad, D)[:, :, :Skv, :]
       └─ dv = dv_2d.reshape(B, Hkv, Skv_pad, D)[:, :, :Skv, :]
```

### 2.2 S 和 P 的重计算

在反向传播中，前向的 S 和 P 矩阵不保存，通过以下方式重计算：

```
S = Q_u @ K_v^T                                          # [bx, by]
S_scaled = S * softmax_scale                              # [bx, by]
P = exp(S_scaled - lse_u)                                 # [bx, by]  (lse_u: [bx, 1])
```

对于 sparse 路径，额外应用掩码：
```
P_masked = P * valid_mask + (1 - valid_mask) * 0.0        # 无效块置零
```

### 2.3 梯度计算图

```
dO_u ───┬── matmul(V_v^T) ──┐
         │                   │
O_u  ───┼── mul ── cast(FP32) ── sum(D) ── D_row
         │                                       │
         │                   ┌───────────────────┘
         │                   │
Q_u  ── matmul(K_v^T) ─┬─ mul(scale) ── S_scaled ── sub(lse) ── exp ── P
         │              │
K_v  ───┘              │
         │              │
V_v  ──────────────────┘
         │
         ├── P_masked = P * mask
         │
         ├── dP = P_masked * (dO@V^T - D_row)
         │
         ├── dQ_contrib = dP @ K * scale
         ├── dk_contrib = dP^T @ Q * scale
         └── dv_contrib = P_masked^T @ dO  (no scale)
```

---

## 3. Tiling 策略

### 3.1 全局 Tiling 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `_VEC_TILE_LOAD` | (128, 128) | 向量加载 tile shape |
| `_CUBE_TILE` | (128, 128) | Cube matmul tile shape |
| `_CUBE_TILE_LIST` | [128, 128] | Cube tile 列表 |
| `block_shape_x` | 256 | Q 块行数 |
| `block_shape_y` | 512 | KV 块行数 |

### 3.2 Per-kernel Pass Options

| Kernel | cube_l1_reuse_setting | 原因 |
|--------|----------------------|------|
| Dense dQ | {-1: 64} | 大 shape 内层循环长（numKB 大），高 L1 复用收益显著 |
| Sparse dQ | {-1: 16} | 内层循环短（maxSel 通常 1~3），L1=64 过大浪费编译时间 |
| Sparse dK/dV | {-1: 16} | 中等内层循环 |
| Dense dK/dV | {-1: 16} | 与 sparse dK/dV 相同 |

### 3.3 紧凑张量策略

**dQ kernel 用 `_build_sparse_kv()`**：
- 输入：block_sparse_mask, K_2d, V_2d
- 输出：k_compact, v_compact, valid_mask, maxSel
- 紧凑维度：按每个 Q 块的有效 KV 块集合紧凑排列
- Shape：`[B*Hq*numQB*maxSel*by, D]`（K/V），`[B*Hq*numQB*maxSel*bx, by]`（mask）

**dK/dV kernel 用 `_build_sparse_q_dkdv()`**：
- 输入：block_sparse_mask, Q_2d, dO_2d, O_2d, lse_2d
- 输出：q_compact, do_compact, o_compact, lse_compact, inner_mask, maxInner
- 紧凑维度：按每个 KV 块的有效 Q 块集合紧凑排列（含 GQA 分组）
- Shape：`[B*Hkv*numKB*maxInner*bx, D]`（Q/dO/O），`[B*Hkv*numKB*maxInner*bx, 1]`（LSE），`[B*Hkv*numKB*maxInner*bx, by]`（mask）

---

## 4. Loop 结构设计

### 4.1 Sparse dQ Kernel

```
TOTAL_OUTER = B * Hq * numQB

for outer in loop(TOTAL_OUTER):                          # 外层: 每个 Q 块
    u = outer % numQB
    h_q_idx = (outer // numQB) % Hq
    b_idx = outer // (numQB * Hq)
    bh_ofs = b_idx * Hq + h_q_idx
    q_row_ofs = bh_ofs * Sq + u * BLOCK

    # 加载 Q, dO, O, LSE 块
    q_block  = view(q_2d,  [BLOCK, D], [q_row_ofs, 0])
    do_block = view(do_2d, [BLOCK, D], [q_row_ofs, 0])
    o_block  = view(o_2d,  [BLOCK, D], [q_row_ofs, 0])
    lse_block= view(lse_2d,[BLOCK, 1], [q_row_ofs, 0])

    # 计算 softmaxGrad (D_row) — FP32 精度!
    do_o = mul(do_block, o_block)
    do_o_fp32 = cast(do_o, DT_FP32)
    D_row = sum(do_o_fp32, dim=-1, keepdim=True)         # [BLOCK, 1]

    # 分配 dQ 累积器
    dq_acc = tensor([BLOCK, D], DT_FP32, "dq_acc")

    for v_idx in loop(maxSel):                            # 内层: 紧凑 KV 块
        kv_row_ofs = outer * maxSel * KV_BLOCK + v_idx * KV_BLOCK
        k_block = view(k_compact, [KV_BLOCK, D], [kv_row_ofs, 0])
        v_block = view(v_compact, [KV_BLOCK, D], [kv_row_ofs, 0])

        # 重计算 S, P
        S = matmul(q_block, k_block, DT_FP32, b_trans=True)
        S_scaled = mul(S, softmax_scale)
        P = exp(sub(S_scaled, lse_block))

        # 应用掩码
        mask_float = view(valid_mask, [BLOCK, KV_BLOCK], [...])
        inv_mask = add(mul(mask_float, -1.0), 1.0)
        P_masked = add(mul(P, mask_float), mul(inv_mask, 0.0))

        # 计算 dP
        do_v = matmul(do_block, v_block, DT_FP32, b_trans=True)
        dP = mul(P_masked, sub(do_v, D_row))

        # dQ 梯度贡献
        dP_fp16 = cast(dP, dtype)
        dq_contrib = matmul(dP_fp16, k_block, DT_FP32)
        dq_contrib = mul(dq_contrib, softmax_scale)

        # 累积
        if is_loop_begin(v_idx):
            if is_loop_end(v_idx):
                assemble(cast(dq_contrib, dtype), [q_row_ofs, 0], dq_2d)
            else:
                dq_acc[:] = dq_contrib
        else:
            dq_new = add(dq_acc, dq_contrib)
            if is_loop_end(v_idx):
                assemble(cast(dq_new, dtype), [q_row_ofs, 0], dq_2d)
            else:
                dq_acc[:] = dq_new
```

### 4.2 Dense dQ Kernel（Sub-block 分割）

```
SUB_SPLIT = bx // 128   # = 2
SUB_BLOCK = bx // SUB_SPLIT  # = 128
TOTAL_OUTER = B * Hq * numQB * SUB_SPLIT

for outer in loop(TOTAL_OUTER):
    sub = outer % SUB_SPLIT
    u = (outer // SUB_SPLIT) % numQB
    h_q_idx = (outer // (SUB_SPLIT * numQB)) % Hq
    b_idx = outer // (SUB_SPLIT * numQB * Hq)
    q_row_ofs = (b_idx * Hq + h_q_idx) * Sq + u * bx + sub * SUB_BLOCK

    # 加载 SUB_BLOCK 行的 Q/dO/O/LSE
    q_sub = view(q_2d, [SUB_BLOCK, D], [q_row_ofs, 0])
    ...

    for v_blk in loop(numKB):
        kv_row_ofs = (b_idx * Hkv + h_kv_idx) * Skv + v_blk * KV_BLOCK
        # 直接 2D 访问 K/V，无掩码
        # 重计算 S, P（无 mask）
        # 累积 dq_acc
        ...
```

### 4.3 Sparse dK/dV Kernel

```
TOTAL_OUTER = B * Hkv * numKB

for outer in loop(TOTAL_OUTER):                          # 外层: 每个 KV 块
    v_blk = outer % numKB
    h_kv_idx = (outer // numKB) % Hkv
    b_idx = outer // (numKB * Hkv)
    kv_row_ofs = (b_idx * Hkv + h_kv_idx) * Skv + v_blk * KV_BLOCK

    # 加载 K, V 块
    k_block = view(k_2d, [KV_BLOCK, D], [kv_row_ofs, 0])
    v_block = view(v_2d, [KV_BLOCK, D], [kv_row_ofs, 0])

    # 分配 dK/dV 累积器
    dk_acc = tensor([KV_BLOCK, D], DT_FP32, "dk_acc")
    dv_acc = tensor([KV_BLOCK, D], DT_FP32, "dv_acc")

    for inner in loop(maxInner):                          # 内层: 紧凑 Q 块
        q_row_ofs = outer * maxInner * BLOCK + inner * BLOCK

        # 加载紧凑 Q/dO/O/LSE/inner_mask
        ...

        # D_row 计算 (同 dQ kernel)
        do_o = mul(do_block, o_block)
        do_o_fp32 = cast(do_o, DT_FP32)
        D_row = sum(do_o_fp32, dim=-1, keepdim=True)

        # 重计算 S, P（同 dQ kernel）
        S = matmul(q_block, k_block, DT_FP32, b_trans=True)
        S_scaled = mul(S, softmax_scale)
        P = exp(sub(S_scaled, lse_block))

        # 应用 inner_mask
        ...

        # dP 计算
        do_v = matmul(do_block, v_block, DT_FP32, b_trans=True)
        dP = mul(P_masked, sub(do_v, D_row))

        # dK 梯度贡献（含 scale）
        dP_fp16 = cast(dP, dtype)
        dk_contrib = matmul(dP_fp16, q_block, DT_FP32, a_trans=True)
        dk_contrib = mul(dk_contrib, softmax_scale)

        # dV 梯度贡献（无 scale！）
        P_fp16 = cast(P_masked, dtype)
        dv_contrib = matmul(P_fp16, do_block, DT_FP32, a_trans=True)

        # 双累积
        if is_loop_begin(inner):
            if is_loop_end(inner):
                assemble(cast(dk_contrib, dtype), [kv_row_ofs, 0], dk_2d)
                assemble(cast(dv_contrib, dtype), [kv_row_ofs, 0], dv_2d)
            else:
                dk_acc[:] = dk_contrib
                dv_acc[:] = dv_contrib
        else:
            dk_new = add(dk_acc, dk_contrib)
            dv_new = add(dv_acc, dv_contrib)
            if is_loop_end(inner):
                assemble(cast(dk_new, dtype), [kv_row_ofs, 0], dk_2d)
                assemble(cast(dv_new, dtype), [kv_row_ofs, 0], dv_2d)
            else:
                dk_acc[:] = dk_new
                dv_acc[:] = dv_new
```

### 4.4 Dense dK/dV Kernel

与 sparse dK/dV 结构相同，区别：
- 无紧凑张量，直接 2D 访问 Q/dO/O/LSE
- 内层循环边界为 `TOTAL_INNER = group * numQB`（GQA 分组内的所有 Q 块）
- 无 inner_mask 应用

---

## 5. 数据流设计

### 5.1 输入预处理

```python
# 1. Pad to block-aligned
Q_pad, Sq_pad = _pad_to_block_aligned(query, bx)
K_pad, Skv_pad = _pad_to_block_aligned(key, by)
V_pad, _       = _pad_to_block_aligned(value, by)
dO_pad, _      = _pad_to_block_aligned(dout, bx)
O_pad, _       = _pad_to_block_aligned(attention_out, bx)

# 2. Reshape 4D → 2D
q_2d  = Q_pad.reshape(B * Hq * Sq_pad, D)
k_2d  = K_pad.reshape(B * Hkv * Skv_pad, D)
v_2d  = V_pad.reshape(B * Hkv * Skv_pad, D)
do_2d = dO_pad.reshape(B * Hq * Sq_pad, D)
o_2d  = O_pad.reshape(B * Hq * Sq_pad, D)

# 3. LSE padding
lse_pad = torch.full([B, Hq, Sq_pad], 1e30, dtype=torch.float32, device=device)
lse_pad[:, :, :Sq] = softmax_lse
lse_2d = lse_pad.reshape(B * Hq * Sq_pad, 1)

# 4. Allocate outputs
dq_2d = torch.zeros(B * Hq * Sq_pad, D, dtype=torch.float16, device=device)
dk_2d = torch.zeros(B * Hkv * Skv_pad, D, dtype=torch.float16, device=device)
dv_2d = torch.zeros(B * Hkv * Skv_pad, D, dtype=torch.float16, device=device)
```

### 5.2 紧凑张量构建 — dQ 用 `_build_sparse_kv()`

三阶段构建：

**Phase 1: 收集有效 KV 索引**
```
for (b, h_q, u):
    valid_v = [v for v in range(numKB) if mask[b, h_q, u, v] == 1]
    if not valid_v: valid_v = [0]   # 保证至少一个，避免空循环
    maxSel = max(maxSel, len(valid_v))
```

**Phase 2: 构建紧凑张量**
```
k_compact [total_qblocks * maxSel * by, D]  ← 仅复制有效 KV 块
v_compact: 同上
valid_mask [total_qblocks * maxSel * bx, by] ← 有效位=1.0, 填充位=0.0
```

**Phase 3: 边界处理**
- 若 Skv < Skv_pad：最后 KV 块的填充行 → valid_mask 对应列置零
- 若 Sq < Sq_pad：最后 Q 块的填充行 → valid_mask 对应行置零

### 5.3 紧凑张量构建 — dK/dV 用 `_build_sparse_q_dkdv()`

三阶段构建：

**Phase 1: 收集有效 Q 索引（含 GQA 分组）**
```
for (b, h_kv, v_blk):
    valid_q = [(h_kv * group + g_idx, u)
               for g_idx in range(group)
               for u in range(numQB)
               if mask[b, h_kv * group + g_idx, u, v_blk] == 1]
    maxInner = max(maxInner, len(valid_q))
```

**Phase 2: 构建紧凑张量**
```
q_compact  [total_kv * maxInner * bx, D]    ← 紧凑 Q
do_compact [total_kv * maxInner * bx, D]    ← 紧凑 dO
o_compact  [total_kv * maxInner * bx, D]    ← 紧凑 O
lse_compact [total_kv * maxInner * bx, 1]   ← 紧凑 LSE（填充位=1e30）
inner_mask [total_kv * maxInner * bx, by]   ← 有效位=1.0, 填充位=0.0
```

**Phase 3: 边界处理**
- 若 Sq < Sq_pad：最后 Q 块的填充行 → inner_mask 对应行置零

### 5.4 输出组装

```python
dq = dq_2d.reshape(B, Hq, Sq_pad, D)[:, :, :Sq, :].contiguous()
dk = dk_2d.reshape(B, Hkv, Skv_pad, D)[:, :, :Skv, :].contiguous()
dv = dv_2d.reshape(B, Hkv, Skv_pad, D)[:, :, :Skv, :].contiguous()
```

---

## 6. 精度设计

### 6.1 FP32 累积策略

| 计算步骤 | 输入精度 | 累积精度 | 输出精度 | 说明 |
|---------|---------|---------|---------|------|
| dO * O | FP16 | FP16 | FP16 | 逐元素乘 |
| D_row = sum(dO*O) | **FP32** | FP32 | FP32 | **必须先 cast FP32 再 sum** |
| S = Q @ K^T | FP16 | FP32 | FP32 | matmul 输出 DT_FP32 |
| P = exp(S - LSE) | FP32 | - | FP32 | 指数运算 |
| dP = P * (dO@V^T - D_row) | FP32 | - | FP32 | 注意力概率梯度 |
| dq_acc 累积 | FP32 | FP32 | FP32 | 梯度累积 |
| dk_acc / dv_acc 累积 | FP32 | FP32 | FP32 | 梯度累积 |
| 输出 (assemble) | FP16 | - | FP16 | cast FP32→FP16 后写出 |

### 6.2 D_row 精度（关键）

D_row（softmaxGrad）是反向传播中最关键的精度控制点：

```python
do_o = pypto.mul(do_block, o_block)           # FP16 × FP16 → FP16
do_o_fp32 = pypto.cast(do_o, pypto.DT_FP32)   # 必须！否则 sum 返回 FP16
D_row = pypto.sum(do_o_fp32, dim=-1, keepdim=True)  # FP32 精度求和
```

**原因**：`pypto.sum(FP16)` 返回 FP16，而 FP16 的尾数仅 10 位，对 128 维度求和会丢失大量精度。

### 6.3 Scale 应用规则

| 梯度 | 是否乘 scale | 原因 |
|------|------------|------|
| dQ | 是 | `dQ += dS @ K * scale` |
| dK | 是 | `dK += dS^T @ Q * scale` |
| dV | **否** | `dV += P^T @ dO`（P 已包含 scale 效果） |

此规则严格对齐 BSA.md 规范。

### 6.4 LSE Padding 精度

填充位的 LSE 设为 `1e30`（而非 `float('inf')`），确保：
- `exp(S * scale - 1e30) ≈ 0`：填充行对 P 产生零贡献
- 避免 FP16 的 inf 值在 kernel 内引发 NaN

### 6.5 精度验证标准

| 输出 | atol | rtol | 说明 |
|------|------|------|------|
| dQ | 0.0001 | 0.0078125 | Query 梯度 |
| dK | 0.0001 | 0.0078125 | Key 梯度 |
| dV | 0.0001 | 0.0078125 | Value 梯度 |

---

## 7. 性能优化设计

### 7.1 Per-kernel L1 Reuse

```python
_BWD_DENSE_DQ_PASS_OPTS = {"cube_l1_reuse_setting": {-1: 64}}
_BWD_PASS_OPTS = {"cube_l1_reuse_setting": {-1: 16}}
```

- Dense dQ 保持 L1=64：大 shape 的内层循环（numKB）长，高 L1 复用收益显著
- 其他 kernel 降至 L1=16：小 shape 场景下过高 L1 浪费编译时间和内存

### 7.2 Sub-block 分割（Dense dQ）

```python
SUB_SPLIT = bx // 128   # = 2
SUB_BLOCK = bx // SUB_SPLIT  # = 128
TOTAL_OUTER = B * Hq * numQB * SUB_SPLIT
```

- 将 256 行 Q-block 分为 2 个 128 行子块
- 外层任务数翻倍，提升多核并行度
- 代价：matmul M 维从 256 降为 128，单 task 计算量略减

### 7.3 Kernel 工厂缓存

```python
_dq_cache = {}
_dkdv_cache = {}
_dq_dense_cache = {}
_dkdv_dense_cache = {}
```

- 首次调用某 shape 触发 JIT 编译
- 后续调用命中缓存直接执行，零编译开销
- 缓存 key 包含所有影响 kernel 签名的参数

### 7.4 Synchronize 策略

紧凑张量构建后必须调用 `torch.npu.synchronize()`：
- 确保 wrapper 层在 NPU 上的数据写入完成
- 避免 kernel 读到未就绪的紧凑张量数据

---

## 8. 测试设计

### 8.1 测试用例配置

| 用例 | B | Hq | Hkv | Sq | Skv | Sparsity | 路径 | 说明 |
|------|---|-----|-----|-----|------|----------|------|------|
| BWD Basic | 1 | 4 | 2 | 256 | 256 | 50% | Sparse | 基本场景 |
| BWD Dense | 1 | 4 | 4 | 256 | 256 | 100% | Dense | 全稠密掩码 |
| BWD GQA | 1 | 8 | 2 | 256 | 512 | 50% | Sparse | GQA 分组 |
| BWD Long Seq | 1 | 8 | 1 | 1024 | 1024 | 30% | Sparse | 长序列 |
| BWD NonAligned | 1 | 4 | 2 | 300 | 300 | 50% | Sparse | 非对齐（已知限制） |

### 8.2 验证方法

- 使用 `bsa_backward_golden()` 作为参考基准
- `torch.allclose(dq_impl, dq_golden, atol=0.0001, rtol=0.0078125)`
- 分别验证 dQ、dK、dV 三个输出

---

## 9. 测试性能预期

### 9.1 MHA Dense (B=1, Hq=Hkv=4)

| Shape | dQ Task Time | dK/dV Task Time | dQ AICore Util | dK/dV AICore Util |
|-------|-------------|-----------------|----------------|-------------------|
| 256×256 | ~78 us | ~87 us | ~42% | ~34% |
| 512×512 | ~109 us | ~112 us | ~40% | ~43% |
| 1024×1024 | ~174 us | ~216 us | ~63% | ~57% |

### 9.2 性能瓶颈与未来优化方向

1. **小 shape 并行度不足**：256×256 dQ 仅有 8 外层任务（sub-block 后），22 核利用率约 42%
2. **累积模式限制**：dK/dV 需跨 Q 块累积，无法简单增加外层并行
3. **Cube tile 限制**：`[128,128]` 是 CANN 9.0.0 支持的最大安全配置
4. **Dynamic Shape 受限**：CANN 9.0.0 不支持 PyPTO 动态维度，每种 shape 需独立编译
