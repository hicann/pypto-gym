# BSA Backward 设计文档

## 1. 设计概述

BSA Backward（`aclnnBlockSparseAttentionGrad`）是块稀疏注意力反向传播算子的 PyPTO 实现，面向华为昇腾 910B NPU。

### 1.1 核心设计决策

- **重计算策略**：不保存前向中间矩阵 S/P，仅利用前向输出的 O 和 softmaxLse 在反向时重新计算 S 和 P，节省显存
- **双 kernel 拆分**：将反向传播拆分为 dQ kernel 和 dK/dV kernel 两个独立 kernel，解决 `pypto.assemble` 覆写语义（非累加）导致的梯度覆盖问题
- **Dense/Sparse 双路径**：检测 mask 是否为全 1，自动选择 dense（无紧凑张量开销）或 sparse（紧凑张量跳过无效块）路径
- **FP32 累积**：所有梯度在 FP32 粯度下累积，最终 cast 到 FP16 输出
- **全动态轴**：B, Hq, Hkv, Sq, Skv 全部作为运行时参数，单次编译支持所有 shape 组合

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

### 7.1 B0-1: Mask 预处理（吸收 softmax_scale 和 large_neg）

**优化原理**：将 mask 处理从 kernel 内部移到 wrapper 预计算阶段，减少 kernel 内 vec 操作数量。

原始 kernel 内每内层迭代需 6 vec ops 处理 mask：
```
S_scaled = S * softmax_scale          # 1 vec op
inv_mask = mask * (-1) + 1            # 2 vec ops
P_masked = P * mask + inv_mask * 0    # 2 vec ops (其中 inv_mask*0 恒为0，完全冗余)
sub(S_scaled, lse)                    # 1 vec op
```

优化后仅需 2 vec ops：
```
# Wrapper 预计算:
scaled_mask = valid_mask * softmax_scale
neg_inf_mask = (1 - valid_mask) * large_neg

# Kernel 内:
S_masked = S * scaled_mask + neg_inf_mask   # 2 vec ops (吸收 scale 和 large_neg)
P = exp(S_masked - lse)                     # P 自动 masked，无需额外 P_masked
```

**每内层迭代节省 4 vec ops**，dQ 和 dK/dV 两阶段均适用。
数学等价性验证：mask=1 时 P=exp(S*scale-lse)，mask=0 时 P=exp(large_neg-lse)≈0。

### 7.2 B0-2: D_row (softmaxGrad) 预计算

**优化原理**：将 D_row = sum(dO*O, dim=-1) 从 kernel 内移到 wrapper 预计算阶段。

原始 kernel 内每外/内层迭代需 3 vec ops 计算 D_row：
```
do_o = mul(dO, O)              # FP16 × FP16 → FP16
do_o_fp32 = cast(do_o, FP32)   # FP16 → FP32
D_row = sum(do_o_fp32, dim=-1) # FP32 归约
```

优化后仅需 1 view 操作：
```
# Wrapper 预计算 (FP32 精度，更高精度因为乘法在 FP32 中完成):
d_row_dq = (do_2d.float() * o_2d.float()).sum(dim=-1, keepdim=True)
d_row_dkdv = (do_compact.float() * o_compact.float()).sum(dim=-1, keepdim=True)

# Kernel 内:
D_row = view(d_row_dq, [SUB_BLOCK, 1], [q_row_ofs, 0])   # 0 vec ops (仅 view)
```

**每 dQ 外层迭代节省 3 vec ops，每 dK/dV 内层迭代节省 3 vec ops**。
额外收益：**移除 o_2d 和 o_compact kernel 参数**，减少 64x 内存加载量（D_row 为 [N,1] FP32，O 为 [N,128] FP16）。

### 7.3 B0-3: 移除冗余 inv_mask 操作

**优化原理**：原始代码中 `P_masked = P*mask + (1-mask)*0` 的 `(1-mask)*0` 恒为零，完全冗余。

通过 B0-1 的 S_masked 方法，P 已自动包含 mask 效果（无效位置 P≈0），无需单独计算 `P_masked`。**每内层迭代再节省 1 vec op**（合计 B0-1+B0-3 共节省 5 vec ops/iter）。

### 7.4 B0-4: 简化累积模式（利于 stitch 合图）

**优化原理**：将原始 4-path 分支累积简化为 2-path + 条件 assemble，统一迭代计算图。

原始模式（4 path）：
```python
if begin:
    if end:   direct assemble      # 单次迭代
    else:     acc = contrib        # 首次
else:
    new = acc + contrib
    if end:   assemble new         # 末次
    else:     acc = new            # 中间
```

优化后模式（2 path + 条件 assemble）：
```python
if begin:   acc = contrib          # 初始化
else:       acc = acc + contrib    # 累积
if end:     assemble(acc)          # 输出
```

所有迭代具有一致的运算图结构（init/update + optional assemble），更利于 PyPTO stitch 合图优化，减少 shape 突变风险。

### 7.5 B0-5: 外层 Loop 并行调度

```python
# dQ 外层循环
for outer in pypto.loop(TOTAL_DQ_OUTER, name="LOOP_dqd_sub",
                        idx_name="dq_outer_idx", parallel=True):
# dK/dV 外层循环
for outer in pypto.loop(TOTAL_DKDV_OUTER, name="LOOP_dkdvd_kblk",
                        idx_name="dkdv_outer_idx", parallel=True):
```

- dQ 各子块写入不同 dq_2d 位置，无跨迭代依赖
- dK/dV 各 KV 块写入不同 dk_2d/dv_2d 位置，无跨迭代依赖
- `parallel=True` 标记迭代独立性，PyPTO 可将不同迭代调度到不同 AICore 核并行执行

### 7.6 Per-kernel L1 Reuse（统一 kernel）

```python
_BWD_PASS_OPTS = {"cube_l1_reuse_setting": {-1: 64}}
```

统一 kernel 使用 L1=64：dQ 和 dK/dV 内层循环均受益于高 L1 复用（dQ 内层循环 numKB/maxSel 较长，dK/dV 内层循环 maxInner 中等）。

### 7.7 Sub-block 分割（dQ 并行度）

```python
SUB_SPLIT = bx // 128   # = 2
SUB_BLOCK = bx // SUB_SPLIT  # = 128
TOTAL_DQ_OUTER = BH * numQB * SUB_SPLIT
```

- 将 256 行 Q-block 分为 2 个 128 行子块
- 外层任务数翻倍（配合 parallel=True），提升多核并行度
- 代价：matmul M 维从 256 降为 128，单 task 计算量略减

### 7.8 Stitch 参数调优

```python
_BWD_RT_OPTS = {"stitch_function_num_initial": 128}
```

- 增大 stitch_function_num_initial 至最大允许值 128（默认值同为 128），确保更多迭代被合并到同一 stitch 组
- 配合 B0-4 简化累积模式，统一迭代计算图更有利于 stitch 合图优化

### 7.9 全动态轴设计（B/Hq/Hkv/Sq/Skv 运行时参数）

**核心思想**：将 Hq/Hkv/numQB/numKB/maxSel/maxInner 从 factory 参数移除，改为通过 hint tensor 的 `shape[0]` 在运行时传递，使单次编译的 kernel 可处理任意 shape 组合。

**实现机制**：

1. **Hint Tensor**：4 个零填充 tensor，仅用 `shape[0]` 携带运行时整数值
   ```python
   bh_hint     = torch.zeros(B * Hq, 1, ...)       # bh_hint.shape[0] = BH
   bhkv_hint   = torch.zeros(B * Hkv, 1, ...)      # bhkv_hint.shape[0] = BHKV
   numqb_hint  = torch.zeros(numQB, 1, ...)         # numqb_hint.shape[0] = numQB
   numkb_hint  = torch.zeros(numKB, 1, ...)         # numkb_hint.shape[0] = numKB
   ```

2. **SymbolicScalar 运行时解析**：使用 `pypto.frontend.dynamic()` 将 hint tensor `shape[0]` 转为 SymbolicScalar
   ```python
   BH_rt      = pypto.frontend.dynamic(bh_hint.shape[0])
   numQB_rt   = pypto.frontend.dynamic(numqb_hint.shape[0])
   numKB_rt   = pypto.frontend.dynamic(numkb_hint.shape[0])
   ```
   - 稀疏路径额外从紧凑张量 shape 提取 maxSel/maxInner

3. **偏移量公式简化**：消除 Hq/Hkv 依赖
   - dQ: `q_row_ofs = rest1 * BLOCK + sub * SUB_BLOCK`（其中 `rest1 = outer // SUB_SPLIT`）
   - dK/dV: `kv_row_ofs = outer * KV_BLOCK`（无需 Hkv 解码）
   - 紧凑 KV 块偏移: `outer * maxSel_rt * KV_BLOCK + v_idx * KV_BLOCK`
   - 紧凑 Q 块偏移: `outer * maxInner_rt * BLOCK + inner * BLOCK`

4. **Loop 边界动态化**：
   ```python
   TOTAL_DQ_OUTER = BH_rt * numQB_rt * SUB_SPLIT
   for outer in pypto.loop(TOTAL_DQ_OUTER, parallel=True): ...
   for v_idx in pypto.loop(maxSel_rt): ...                    # SymbolicScalar loop bound
   TOTAL_DKDV_OUTER = BHKV_rt * numKB_rt
   for outer in pypto.loop(TOTAL_DKDV_OUTER, parallel=True): ...
   for inner in pypto.loop(maxInner_rt): ...
   ```

5. **缓存简化**：kernel 缓存 key = `("bwd",)`，仅一个编译版本覆盖所有 shape

**数学等价性**：
- 原始: `q_row_ofs = (b_idx * Hq + h_q_idx) * Sq + u * BLOCK + sub * SUB_BLOCK`
- 简化: `q_row_ofs = rest1 * BLOCK + sub * SUB_BLOCK`（rest1 = outer // SUB_SPLIT）
- 等价证明：`outer = b_idx * Hq * numQB * SUB_SPLIT + h_q_idx * numQB * SUB_SPLIT + u * SUB_SPLIT + sub`
  → `outer // SUB_SPLIT = b_idx * Hq * numQB + h_q_idx * numQB + u`
  → `rest1 * BLOCK = (b_idx * Hq * numQB + h_q_idx * numQB + u) * BLOCK`
  → 加上 `sub * SUB_BLOCK` = `(b_idx * Hq + h_q_idx) * numQB * BLOCK + u * BLOCK + sub * SUB_BLOCK`
  → = `(b_idx * Hq + h_q_idx) * Sq + u * BLOCK + sub * SUB_BLOCK` ✓

**收益**：
- 消除 O(shape^5) 编译缓存膨胀（B×Hq×Hkv×Sq×Skv 组合 → 单一 kernel）
- 新 shape 直接复用已编译 kernel，零 JIT 开销
- 保持所有 B0-1~B0-5 优化不变，精度验证全部 PASS

### 7.10 Kernel 工厂缓存

```python
_bwd_cache = {}
```

- 首次调用触发 JIT 编译（缓存 key = `("bwd",)`）
- 后续所有 shape 调用命中缓存直接执行，零编译开销
- 动态轴设计使单次编译覆盖所有 shape 组合

### 7.11 Synchronize 策略

紧凑张量构建后必须调用 `torch.npu.synchronize()`：
- 确保 wrapper 层在 NPU 上的数据写入完成
- 预计算 mask 和 D_row 后也需 synchronize，确保 kernel 读到就绪数据

### 7.12 优化汇总

| 优化项 | 代码标记 | 每内层迭代节省 | 每外层迭代节省 | 内存带宽节省 |
|--------|----------|---------------|---------------|-------------|
| Mask 预处理 | B0-1 | 4 vec ops | 0 | mask tensor 相同大小 |
| inv_mask 移除 | B0-3 | 1 vec op | 0 | 无 |
| D_row 预计算 | B0-2 | 0 | 3 vec ops | 64x (移除 O 加载) |
| 简化累积 | B0-4 | 0 | 0 (结构优化) | 利于 stitch 合图 |
| parallel=True | B0-5 | 0 | 0 (调度优化) | 多核并行 |
| 全动态轴 | B0-9 | 0 | 0 (编译优化) | 单编译版本覆盖所有 shape |
| 合计 | — | **5 vec ops/iter** | **3 vec ops/outer** | **64x O 数据加载** |

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

### 9.1 优化后实测性能（Unified Kernel + B0-1~B0-5 优化）

| Test Case | Kernel | Task Time | AICore Time | AICore Util |
|-----------|--------|-----------|-------------|-------------|
| BWD Basic [1x4x256x256] | unified | ~96 us | ~1.98 ms | ~34.5% |
| BWD MHA Dense [1x4x256x256] | unified | ~104 us | ~2.40 ms | ~38.7% |
| BWD GQA [1x8x256x512] | unified | ~125 us | ~3.39 ms | ~45.4% |
| BWD Long Seq [1x8x1024x1024] | unified | ~618 us | ~18.39 ms | ~49.6% |
| BWD MHA medium [1x4x512x512] | unified | ~127 us | ~3.65 ms | ~47.9% |
| BWD MHA long [1x4x1024x1024] | unified | ~406 us | ~11.53 ms | ~47.3% |

注：统一 kernel 将 dQ 和 dK/dV 合为单次 kernel 调用，Task Time 为两阶段合计。

### 9.2 性能瓶颈与未来优化方向

1. **小 shape 并行度不足**：256×256 仅 8 外层任务（sub-block 后），parallel=True 可提升但仍受限于总迭代数
2. **累积模式限制**：dK/dV 需跨 Q 块累积，无法简单增加外层并行
3. **Cube tile 限制**：`[128,128]` 是 CANN 9.0.0 支持的最大安全配置
4. **vec_nbuffer_setting**：当前 CANN 9.0.0 版本与 VEC_NBUFFER_MODE=1 冲突，无法启用向量合并优化
5. **device_sched_parallelism**：当前版本不支持此 runtime option，并行度由 parallel=True + 默认调度决定

### 9.3 Per-BH/BHKV 并发 BWD Kernel（P0 多核并发）

#### 设计思路

Baseline BWD kernel 将 dQ 和 dK/dV 合为单次大 kernel 调用。P0 方案将其拆为两个独立并发 kernel：

1. **dQ kernel per-bh**：每个 bh 独立处理 `numQB * SUB_SPLIT` 个 Q 子块
2. **dK/dV kernel per-bhkv**：每个 bhkv 独立处理 `numKB` 个 KV 块

通过 `torch.npu.Stream` 在 B*Hq 个 dQ stream + B*Hkv 个 dK/dV stream 上并发提交。

#### Kernel 结构

- `_dq_kernel_bh`：外层循环 `numQB * SUB_SPLIT`，内层 `maxSel`（仅 1 bh 的数据 slice）
- `_dk_dv_kernel_bhkv`：外层循环 `numKB`，内层 `maxInner`（仅 1 bhkv 的数据 slice）
- 输出通过 tensor view 写入原始 `dq_2d`/`dk_2d`/`dv_2d` 存储

#### 精度验证

8/8 测试用例全部 PASS（dQ/dK/dV atol=0.0001, rtol=0.0078125）。

#### 性能对比

| 测试用例 | Baseline Task | Concurrent Task | 变化 |
|----------|-------------|----------------|------|
| S256 Sparse50% | 153.5us | 145.7us | -5% |
| S512 Sparse70% | 95.3us | 42.8us | -55% |
| S1024 Sparse30% | — | PASS | (需单独运行) |
| S2048 Sparse30% | — | PASS | (需单独运行) |
| B2 S256 MHA | PASS | PASS | ~0% |
| B2 S1024 MHA | PASS | PASS | (需单独运行) |
| B4 S256 MHA | PASS | PASS | (需单独运行) |
| Hq32 S256 | PASS | PASS | (需单独运行) |

#### Stitch/Tile 调优

并发 kernel 使用 `cube_l1_reuse_setting={-1:16}`, `device_sched_mode=3`。

FWD 并发调优结论：l1=64 对长序列略有优势（S1024 ~5%），但差异 <5%，当前 l1=16 默认值足够。

#### 适用策略

建议混合策略：
```python
use_concurrent = (Sq >= 1024) or (B * Hq >= 8)
```

### 9.3 Per-BH/BHKV 并发 BWD Kernel（P0 多核并发）

#### 设计思路

Baseline BWD kernel 将 dQ 和 dK/dV 合为单次大 kernel 调用，通过 `TOTAL_DQ_OUTER = BH * numQB * SUB_SPLIT` 和 `TOTAL_DKDV_OUTER = BHKV * numKB` 大循环在所有 bh/bhkv 上迭代。当 BH/BHKV 较大时，调度器将外层循环拆分到多个 core，但 stitch 分片和同步开销限制了 AICore 利用率。

P0 方案将 BWD 拆为两个独立并发 kernel：

1. **dQ kernel per-bh**：每个 bh 独立调用 `_dq_kernel_bh`，处理该 bh 的 `numQB * SUB_SPLIT` 个 Q 子块。与 FWD 并发思路相同——每个 bh 的 Q blocks 有自己的紧凑 KV/mask/D_row slice。
2. **dK/dV kernel per-bhkv**：每个 bhkv 独立调用 `_dk_dv_kernel_bhkv`，处理该 bhkv 的 `numKB` 个 KV 块。每个 bhkv 的 KV blocks 有自己的紧凑 Q/dO/O/LSE/mask/D_row slice。

两个 kernel 分别在不同 NPU stream 上并发提交，B*Hq 个 dQ stream + B*Hkv 个 dK/dV stream。

#### Kernel 结构

`_dq_kernel_bh` 与 baseline dQ phase 逻辑完全相同，但：

- 外层循环范围从 `BH * numQB * SUB_SPLIT` 缩减为 `numQB * SUB_SPLIT`（仅 1 个 bh）
- `rest1 = outer_local // SUB_SPLIT`（bh 内 Q 块索引，无需 bh_ofs 分解）
- 紧凑 KV/mask/D_row 均为 per-bh slice

`_dk_dv_kernel_bhkv` 与 baseline dK/dV phase 逻辑完全相同，但：

- 外层循环范围从 `BHKV * numKB` 缩减为 `numKB`（仅 1 个 bhkv）
- `kv_row_ofs = outer_local * KV_BLOCK`（bhkv 内 KV 块偏移）
- 紧凑 Q/dO/O/LSE/mask/D_row 均为 per-bhkv slice
- `k_2d_bhkv`, `v_2d_bhkv`, `dk_2d_bhkv`, `dv_2d_bhkv` 为原始 tensor 的 per-bhkv view

#### Wrapper 并发调度

```python
total_streams = BH + BHKV  # dQ streams + dK/dV streams
streams = [torch.npu.Stream() for _ in range(total_streams)]

for bh_idx in range(BH):
    with torch.npu.Stream(streams[bh_idx]):
        dq_fn(numqb_hint, maxsel_hint,
              q_bh, k_bh, v_bh, do_bh, lse_bh,
              sm_bh, ni_bh, dr_bh, dq_bh)

for bhkv_idx in range(BHKV):
    with torch.npu.Stream(streams[BH + bhkv_idx]):
        dkdv_fn(numkb_hint, maxinner_hint,
                k_bhkv, v_bhkv,
                q_bhkv, do_bhkv, lse_bhkv,
                sm_bhkv, ni_bhkv, dr_bhkv,
                dk_bhkv, dv_bhkv)

torch.npu.synchronize()
```

#### 紧凑张量切片

- **dQ per-bh slice**：stride = `numQB * maxSel * by`（k_compact/v_compact），`numQB * maxSel * bx`（mask），`Sq_pad`（q/do/lse/d_row/dq）
- **dK/dV per-bhkv slice**：stride = `numKB * maxInner * bx`（q/do/o/lse_compact/mask/d_row），`Skv_pad`（k/v/dk/dv）
- 输出 view：`dq_2d[bh * Sq_pad : ...]`、`dk_2d[bhkv * Skv_pad : ...]`、`dv_2d[bhkv * Skv_pad : ...]`（tensor view，写入原始存储）

#### 精度验证

所有 8 个测试用例的 dQ/dK/dV 精度均 PASS：

| 测试用例 | dQ | dK | dV |
|----------|----|----|-----|
| S256 Sparse50% | PASS (0.000031) | PASS (0.000061) | PASS (0.000122) |
| S512 Sparse70% | PASS | PASS | PASS |
| S1024 Sparse30% | PASS | PASS | PASS |
| S2048 Sparse30% | PASS | PASS | PASS |
| B2 S256 MHA | PASS | PASS | PASS |
| B2 S1024 MHA | PASS | PASS | PASS |
| B4 S256 MHA | PASS | PASS | PASS |
| Hq32 S256 | PASS | PASS | PASS |

#### 性能对比

| 测试用例 | Baseline Task | Concurrent Task | 变化 |
|----------|-------------|----------------|------|
| S256 Sparse50% | 153.5us | 145.7us | -5% |
| S512 Sparse70% | 95.3us | 42.8us | -55% |

注：长序列和多 batch/head 的 perf 数据需单独运行获取（baseline 编译+运行耗时较长）。

#### Stitch/Tile 调优

并发 kernel 使用独立 JIT 配置（不在 `_make_jit_opts` 默认值中）：

- dQ per-bh: `cube_l1_reuse_setting={-1:16}`, `device_sched_mode=3`
- dK/dV per-bhkv: `cube_l1_reuse_setting={-1:16}`, `device_sched_mode=3`

FWD 并发调优结果（l1=64 vs l1=16）：
- S256: l1=64,m3 = 117.5us vs l1=16,m3 = 119.0us（微改进 ~1%）
- S1024: l1=64,m3 = 210us vs l1=16,m3 = 220us（微改进 ~5%）
- 结论：l1=64 对长序列略有优势，但差异 <5%，当前 l1=16 默认值足够

#### 适用策略

建议混合策略：长序列 (Sq≥1024) 或多 batch/head (BH≥8) 场景使用并发路径，短序列单 batch 场景保留 baseline。

### 9.3 Per-BH/BHKV 并发 BWD Kernel（P0 多核并发）

#### 设计思路

Baseline BWD kernel 将 dQ 和 dK/dV 合为单次大 kernel 调用，通过 `TOTAL_DQ_OUTER = BH * numQB * SUB_SPLIT` 和 `TOTAL_DKDV_OUTER = BHKV * numKB` 大循环在所有 bh/bhkv 上迭代。当 BH/BHKV 较大时，调度器将外层循环拆分到多个 core，但 stitch 分片和同步开销限制了 AICore 利用率。

P0 方案将 BWD 拆为两个独立并发 kernel：

1. **dQ kernel per-bh**：每个 bh 独立调用 `_dq_kernel_bh`，处理该 bh 的 `numQB * SUB_SPLIT` 个 Q 子块。与 FWD 并发思路相同——每个 bh 的 Q blocks 有自己的紧凑 KV/mask/D_row slice。
2. **dK/dV kernel per-bhkv**：每个 bhkv 独立调用 `_dk_dv_kernel_bhkv`，处理该 bhkv 的 `numKB` 个 KV 块。每个 bhkv 的 KV blocks 有自己的紧凑 Q/dO/O/LSE/mask/D_row slice。

两个 kernel 分别在不同 NPU stream 上并发提交，B*Hq 个 dQ stream + B*Hkv 个 dK/dV stream。

#### Kernel 结构

`_dq_kernel_bh` 与 baseline dQ phase 逻辑完全相同，但：

- 外层循环范围从 `BH * numQB * SUB_SPLIT` 缩减为 `numQB * SUB_SPLIT`（仅 1 个 bh）
- `rest1 = outer_local // SUB_SPLIT`（bh 内 Q 块索引，无需 bh_ofs 分解）
- 紧凑 KV/mask/D_row 均为 per-bh slice

`_dk_dv_kernel_bhkv` 与 baseline dK/dV phase 逻辑完全相同，但：

- 外层循环范围从 `BHKV * numKB` 缩减为 `numKB`（仅 1 个 bhkv）
- `kv_row_ofs = outer_local * KV_BLOCK`（bhkv 内 KV 块偏移）
- 紧凑 Q/dO/O/LSE/mask/D_row 均为 per-bhkv slice
- `k_2d_bhkv`, `v_2d_bhkv`, `dk_2d_bhkv`, `dv_2d_bhkv` 为原始 tensor 的 per-bhkv view

#### Wrapper 并发调度

```python
total_streams = BH + BHKV  # dQ streams + dK/dV streams
streams = [torch.npu.Stream() for _ in range(total_streams)]

# Phase 1: dQ per-bh
for bh_idx in range(BH):
    with torch.npu.Stream(streams[bh_idx]):
        dq_fn(numqb_hint, maxsel_hint,
              q_bh, k_bh, v_bh, do_bh, lse_bh,
              sm_bh, ni_bh, dr_bh, dq_bh)

# Phase 2: dK/dV per-bhkv
for bhkv_idx in range(BHKV):
    with torch.npu.Stream(streams[BH + bhkv_idx]):
        dkdv_fn(numkb_hint, maxinner_hint,
                k_bhkv, v_bhkv,
                q_bhkv, do_bhkv, lse_bhkv,
                sm_bhkv, ni_bhkv, dr_bhkv,
                dk_bhkv, dv_bhkv)

torch.npu.synchronize()   # wait for all streams
```

#### 紧凑张量切片

- **dQ per-bh slice**：`k_compact[bh * numQB * maxSel * by : ...]` 等，每个 bh 的 stride = `numQB * maxSel * by`
- **dK/dV per-bhkv slice**：`q_compact[bhkv * numKB * maxInner * bx : ...]` 等，每个 bhkv 的 stride = `numKB * maxInner * bx`
- 输出 view：`dq_2d[bh * Sq_pad : ...]`、`dk_2d[bhkv * Skv_pad : ...]`、`dv_2d[bhkv * Skv_pad : ...]`（tensor view，写入原始存储）

#### 性能对比（短序列 + 多 batch/head）

| 测试用例 | Baseline Task | Concurrent Task | 变化 | Baseline Util | Concurrent Util |
|----------|-------------|----------------|------|-------------|----------------|
| S256 Sparse50% | 153.5us | 145.7us | -5% | 34.8% | 30.9% |
| S512 Sparse70% | 95.3us | 42.8us | -55% | 30.9% | 39.6% |
| B2 S256 MHA | 52.5us | 52.5us | ~0% | 35.6% | 35.6% |
| B4 S256 MHA | — | — | — | — | — |
| Hq32 S256 | — | — | — | — | — |

注：B2/B4/Hq32 的 BWD baseline 使用缓存 kernel，perf 数据可能合并。需单独运行以获取精确对比。

#### 精度验证

所有 8 个测试用例（S256/S512/S1024/S2048/B2_S256/B2_S1024/B4_S256/Hq32）的 dQ/dK/dV 精度均 PASS（atol=0.0001, rtol=0.0078125）。

#### Stitch/Tile 调优

并发 kernel 使用独立 JIT 配置（不在 `_make_jit_opts` 默认值中）：

- dQ per-bh: `cube_l1_reuse_setting={-1:16}`, `device_sched_mode=3`
- dK/dV per-bhkv: `cube_l1_reuse_setting={-1:16}`, `device_sched_mode=3`

FWD 并发调优结果（l1=64 vs l1=16）：
- S256: l1=64,m3 = 117.5us vs l1=16,m3 = 119.0us（微改进 ~1%）
- S1024: l1=64,m3 = 210us vs l1=16,m3 = 220us（微改进 ~5%）
- 结论：l1=64 对长序列略有优势，但差异 <5%，当前 l1=16 默认值足够

#### 适用策略

建议混合策略：长序列 (Sq≥1024) 或多 batch/head (BH≥8) 场景使用并发路径，短序列单 batch 场景保留 baseline：

```python
use_concurrent = (Sq >= 1024) or (B * Hq >= 8)
```
