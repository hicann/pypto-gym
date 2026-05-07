# BSA Forward 设计文档

## 设计概述

### 算子定位

BSA Forward（`aclnnBlockSparseAttention`）是面向华为昇腾 NPU 的块稀疏注意力前向算子，专为长序列场景优化。通过块级稀疏掩码仅计算有效块对的注意力，将计算复杂度从 O(N²d) 降低到 O(N·S·d)（S 为有效块数）。

### 核心特性

1. **Mask2Idx 紧凑张量策略**：在 Python wrapper 层预计算每个 Q 块对应的有效 KV 块集合，构建紧凑 K/V 张量，使 kernel 内层循环仅迭代 `maxSel`（最大有效 KV 块数）而非 `numKB`（总 KV 块数）
2. **Online Softmax**：逐 KV 块迭代维护运行最大值 `m_j` 和运行指数和 `l_j`，跨块执行全局数值校正，保证分块计算与全量计算数学完全等价
3. **Dense/Sparse 双路径**：当掩码全为 1 时自动切换到 dense kernel，省去 mask 相关开销
4. **GQA 兼容**：支持 Hq ≥ Hkv 的分组查询注意力，通过 `h_kv = h_q // group` 映射
5. **FP32 累积**：所有中间计算（max、sum、O 累积）在 FP32 精度下完成，最终 cast 回 FP16 输出

---

## 计算图设计

### 总体计算流

```
输入: Q[B,Hq,Sq,D], K[B,Hkv,Skv,D], V[B,Hkv,Skv,D], mask[B,Hq,numQB,numKB]
                                    │
                              ┌─────┴──────┐
                              │ Pad + Reshape│  4D → 2D view, 块对齐填充
                              └─────┬──────┘
                                    │
                            ┌───────┴────────┐
                            │ is_dense_mask? │
                            └───┬────────┬───┘
                           Yes  │        │  No
                    ┌──────────┘        └──────────┐
                    ▼                              ▼
          ┌─────────────────┐          ┌──────────────────────┐
          │  Dense Kernel    │          │ _build_sparse_kv()    │
          │  Loop: numKB     │          │ → k/v_compact,        │
          │  No mask apply   │          │   valid_mask, maxSel  │
          └────────┬────────┘          └──────────┬───────────┘
                   │                              │
                   │                    ┌─────────┴──────────┐
                   │                    │ npu.synchronize()   │
                   │                    └─────────┬──────────┘
                   │                              │
                   │                    ┌─────────┴──────────┐
                   │                    │  Sparse Kernel      │
                   │                    │  Loop: maxSel only  │
                   │                    │  Mask apply per blk │
                   │                    └─────────┬──────────┘
                   │                              │
                   └──────────┬───────────────────┘
                              ▼
                    ┌─────────────────┐
                    │ 裁剪填充 + Reshape│  [B*Hq,Sq,D]→[B,Hq,Sq,D]
                    └─────────────────┘
                              │
                              ▼
                  Output: O[B,Hq,Sq,D], LSE[B,Hq,Sq]
```

### Sparse 路径计算图（单 outer 迭代）

```
                          ┌───────────┐
                          │ Q_u [BX,D]│
                          └─────┬─────┘
                                │
              ┌─────────────────┼─────────────────┐
              │           Inner Loop (maxSel)      │
              │                 │                  │
              │  ┌──────────────┼───────────────┐  │
              │  │         K_v, V_v             │  │
              │  │         valid_mask           │  │
              │  │              │               │  │
              │  │    ┌─────────┴─────────┐     │  │
              │  │    │ QK^T (Cube, FP32)  │     │  │
              │  │    └─────────┬─────────┘     │  │
              │  │              │               │  │
              │  │    ┌─────────┴─────────┐     │  │
              │  │    │ S * scale (Vec)    │     │  │
              │  │    └─────────┬─────────┘     │  │
              │  │              │               │  │
              │  │    ┌─────────┴─────────┐     │  │
              │  │    │ S_masked =         │     │  │
              │  │    │  S*mask+(1-mask)*Neg│    │  │
              │  │    └─────────┬─────────┘     │  │
              │  │              │               │  │
              │  │    ┌─────────┴─────────┐     │  │
              │  │    │ Online Softmax     │     │  │
              │  │    │ m_new=max(m,m_ij)  │     │  │
              │  │    │ α=exp(m-m_new)     │     │  │
              │  │    │ β=exp(m_ij-m_new)  │     │  │
              │  │    │ l_new=αl+βl_ij     │     │  │
              │  │    └─────────┬─────────┘     │  │
              │  │              │               │  │
              │  │    ┌─────────┴─────────┐     │  │
              │  │    │ PV (Cube, FP32)    │     │  │
              │  │    │ o_new = α*o+β*o_ij │     │  │
              │  │    └─────────┬─────────┘     │  │
              │  │              │               │  │
              │  │    [is_loop_end?] ── Yes ──→  │  │
              │  │       O = o_new / l_new       │  │
              │  │       LSE = m_new + log(l_new)│  │
              │  │              │               │  │
              │  └──────────────┼───────────────┘  │
              └─────────────────┼─────────────────┘
                                │
                                ▼
                     assemble → output_3d, lse_2d
```

---

## Tiling 策略

### 向量化 Tiling (TileShape)

| 操作阶段 | API | TileShape | 说明 |
|----------|-----|-----------|------|
| 加载 Q/K/V 块 | `pypto.set_vec_tile_shapes(128, 128)` | `(128, 128)` | 2D 切片加载 |
| QK^T 矩阵乘 | `pypto.set_cube_tile_shapes([128,128], [128,128], [128,128])` | `(128, 128)` × `(128, 128)` | Cube matmul |
| PV 矩阵乘 | 同上 | `(128, 128)` × `(128, 128)` | Cube matmul |
| 输出 cast + assemble | `pypto.set_vec_tile_shapes(16, 128, 128)` | `(16, 128, 128)` | 3-arg tile shape |

### JIT 参数配置

```python
# Pass Options
pass_options = {
    "cube_l1_reuse_setting": {-1: 64}   # Cube L1 缓存复用 64KB
}

# Runtime Options
runtime_options = {
    "device_sched_mode": 3,                # 自动调度模式
    "stitch_function_num_initial": 128,    # 子图切分初始数量
    "stitch_function_num_step": 64,        # 子图切分步进
    "stitch_function_inner_memory": 100,   # 子图内部内存限制
    "stitch_function_outcast_memory": 100, # 子图外部内存限制
}

# Debug Options
debug_options = {
    "runtime_debug_mode": 1               # 运行时调试模式
}
```

### 内存访问模式

1. **Q 块**：每个外层迭代加载一次 `[BLOCK, D]` 的 Q 子块，在内层 KV 循环中复用（通过 L1 reuse）
2. **K/V 块**：每次内层迭代加载 `[KV_BLOCK, D]` 的紧凑 K/V 子块
3. **valid_mask**：每次内层迭代加载 `[BLOCK, KV_BLOCK]` 的 mask 子块（仅 sparse 路径）
4. **输出 O**：每个外层迭代末尾 assemble `[1, BLOCK, D]` 到 output_3d
5. **输出 LSE**：每个外层迭代末尾 assemble `[1, BLOCK]` 到 lse_2d

---

## Loop 结构设计

### 外层循环（Q 块遍历）

```
for outer in pypto.loop(TOTAL_OUTER, name="LOOP_fwd_s_qblk", idx_name="outer_idx"):
    # TOTAL_OUTER = B * Hq * numQB

    # 索引分解（从扁平化 outer index 恢复多维索引）
    u = outer % numQB              # Q 块索引
    rest = outer // numQB
    h_q_idx = rest % Hq            # Q 头索引
    b_idx = rest // Hq             # batch 索引
    bh_ofs = b_idx * Hq + h_q_idx  # batch-head 偏移

    # Q 块行偏移
    q_row_ofs = bh_ofs * Sq + u * BLOCK

    # 初始化 FP32 累积器
    mi_update = [BLOCK, 1]  # 运行最大值
    li_update = [BLOCK, 1]  # 运行 exp 和
    oi_update = [BLOCK, D]  # 运行加权 V 和
```

### 内层循环（KV 块遍历）

**Sparse 路径**：
```
for v_idx in pypto.loop(maxSel, name="LOOP_fwd_s_kblk", idx_name="v_idx"):
    # maxSel = 每个 Q 块的最大有效 KV 块数
    kv_row_ofs = outer * maxSel * KV_BLOCK + v_idx * KV_BLOCK
```

**Dense 路径**：
```
for v_blk in pypto.loop(numKB, name="LOOP_fwd_d_kblk", idx_name="v_blk"):
    # numKB = ceil(Skv / block_shape_y)
    h_kv_idx = h_q_idx // group   # GQA 映射
    kv_row_ofs = (b_idx * Hkv + h_kv_idx) * Skv + v_blk * KV_BLOCK
```

### 索引分解示意

```
outer = 0, 1, 2, ..., B*Hq*numQB - 1

outer = b_idx * (Hq * numQB) + h_q_idx * numQB + u

分解:
  u       = outer % numQB
  rest    = outer // numQB
  h_q_idx = rest % Hq
  b_idx   = rest // Hq
```

---

## 数据流设计

### 输入预处理

1. **Padding**：将 Q/K/V 的序列长度填充到块对齐大小
   ```python
   numQB = ceil(Sq / bx);  Sq_pad = numQB * bx
   numKB = ceil(Skv / by); Skv_pad = numKB * by
   Q_pad, _ = _pad_to_block_aligned(query, bx)
   K_pad, _ = _pad_to_block_aligned(key, by)
   V_pad, _ = _pad_to_block_aligned(value, by)
   ```

2. **Reshape 4D → 2D**：展平 BNSD 的 B/H/S 维度
   ```python
   q_2d = Q_pad.reshape(B * Hq * Sq_pad, D)   # [B*Hq*Sq_pad, 128]
   k_2d = K_pad.reshape(B * Hkv * Skv_pad, D)  # [B*Hkv*Skv_pad, 128]
   v_2d = V_pad.reshape(B * Hkv * Skv_pad, D)
   ```

3. **Sparse 路径额外预处理**：
   ```python
   k_compact, v_compact, valid_mask, maxSel = _build_sparse_kv(
       block_sparse_mask, k_2d, v_2d,
       B, Hq, Hkv, Sq, Skv, Sq_pad, Skv_pad, numQB, numKB, bx, by, D, device)
   torch.npu.synchronize()  # 确保 wrapper 数据就绪
   ```

### 中间数据流

```
[加载阶段]                          [计算阶段]                        [输出阶段]
──────────                          ──────────                        ──────────

q_2d ─view──→ q_block [BX, D]      q_block ─matmul──→ S [BX, BY]    oi_new ─div──→ O_final [BX, D]
k_compact ─view──→ k_block [BY, D]         │                        O_final ─cast──→ O_fp16
v_compact ─view──→ v_block [BY, D]    S ─mul(scale)──→ S_scaled     O_fp16 ─reshape──→ [1, BX, D]
valid_mask ─view──→ mask [BX, BY]    S_scaled ─mask──→ S_masked      [1,BX,D] ─assemble──→ output_3d
                                      S_masked ─amax──→ m_ij         mi_new + log(li_new) → LSE
                                      S_masked - m_ij ─exp──→ P_ij   LSE ─reshape──→ [1, BX]
                                      P_ij ─sum──→ l_ij              [1, BX] ─assemble──→ lse_2d
                                      P_ij ─cast──→ P_fp16
                                      P_fp16 @ v_block ─matmul──→ o_ij
```

### 输出装配

```python
# 最终输出（is_loop_end 时执行）
O_final = pypto.div(oi_new, li_new)                    # 归一化 [BLOCK, D]
O_cast = pypto.cast(pypto.reshape(O_final, [1, BLOCK, D]), dtype)
pypto.assemble(O_cast, [bh_ofs, u * BLOCK, 0], output_3d)

lse_val = pypto.add(mi_new, pypto.log(li_new))         # LSE [BLOCK, 1]
lse_cast = pypto.reshape(lse_val, [1, BLOCK])
pypto.assemble(lse_cast, [bh_ofs, u * BLOCK], lse_2d)

# Wrapper 后处理
attention_out = output_3d[:, :Sq, :].reshape(B, Hq, Sq, D)
softmax_lse = lse_2d[:, :Sq].reshape(B, Hq, Sq)
```

---

## 精度设计

### FP32 累积路径

所有中间计算均在 FP32 精度下执行：

| 计算步骤 | 输入精度 | 输出精度 | 说明 |
|----------|---------|---------|------|
| QK^T (matmul) | FP16 × FP16 | **FP32** | `pypto.matmul(..., pypto.DT_FP32)` |
| Scale (mul) | FP32 × float | **FP32** | 标量乘法 |
| Mask apply | FP32 × FP32 | **FP32** | 算术掩码 |
| amax | FP32 | **FP32** | 行最大值 |
| exp | FP32 | **FP32** | 指数概率 |
| sum | FP32 | **FP32** | 行求和 |
| PV (matmul) | FP16(P) × FP16(V) | **FP32** | P 先 cast FP16，结果 FP32 |
| O 累积 | FP32 × FP32 | **FP32** | 校正因子乘法 + 加法 |
| 最终输出 | FP32 / FP32 | **FP32 → FP16** | 归一化后 cast 回 FP16 |
| LSE | FP32 + log(FP32) | **FP32** | 保持 FP32 输出 |

### 精度保障措施

1. **matmul FP32 累积**：`pypto.matmul(q, k, pypto.DT_FP32)` 确保矩阵乘在 FP32 精度下累积
2. **P 矩阵 FP16 下溢保护**：P 先 cast 到 FP16 再做 PV matmul，减少 FP32 matmul 的计算量
3. **Large negative masking**：无效位置使用 `-65504.0`（FP16 最小值）使 exp 下溢为零
4. **Online softmax 数值稳定**：减去行最大值后再 exp，避免数值溢出/下溢
5. **累积器初始化**：`mi_update`/`li_update`/`oi_update` 均为 FP32 中间张量

### 精度验证标准

| 输出 | atol | rtol |
|------|------|------|
| attentionOut (O) | 0.0001 | 0.0078125 |
| softmaxLse (LSE) | 0.0001 | 0.0078125 |

---

## 性能优化设计

### 参数设置

| 参数 | 值 | 作用 |
|------|-----|------|
| `cube_l1_reuse_setting` | `{-1: 64}` | Cube L1 缓存复用 64KB，Q 块在内层循环复用 |
| `device_sched_mode` | `3` | 自动调度模式，优化多核任务分配 |
| `stitch_function_num_initial` | `128` | 子图切分初始数量 |
| `stitch_function_num_step` | `64` | 子图切分步进粒度 |
| `stitch_function_inner_memory` | `100` | 子图内部内存预算 |
| `stitch_function_outcast_memory` | `100` | 子图外部内存预算 |
| `runtime_debug_mode` | `1` | 运行时调试模式 |

### 内存设计

1. **输出张量预分配**：`output_3d` 和 `lse_2d` 在 wrapper 层预分配并传入 kernel，避免 kernel 内动态分配
2. **紧凑 KV 一次构建**：`_build_sparse_kv()` 一次性构建紧凑 K/V 和 valid_mask，kernel 内直接按偏移访问
3. **累积器核内分配**：`mi_update`、`li_update`、`oi_update` 通过 `pypto.tensor()` 在 kernel 内创建

### 计算设计

1. **Dense/Sparse 路径分离**：根据 `is_dense_mask()` 结果选择不同 kernel，避免 sparse 路径的 mask 开销
2. **算术掩码替代 where**：使用 `S * mask + (1 - mask) * large_neg` 替代 `pypto.where(mask, S, neg)`，规避 CCE 编译限制
3. **Kernel 缓存复用**：工厂函数 + 字典缓存，避免同 shape 重复编译

---

## 测试设计

### 测试用例配置

| 测试名 | B | Hq | Hkv | Sq | Skv | 稀疏率 | 路径 |
|--------|---|-----|-----|-----|-----|--------|------|
| Basic Sparse50 | 1 | 4 | 2 | 256 | 256 | 50% | Sparse |
| GQA group4 | 1 | 8 | 2 | 256 | 512 | 40% | Sparse |
| GQA Hq32 Hkv8 | 1 | 32 | 8 | 256 | 256 | 50% | Sparse |
| Long Seq S1024 | 1 | 8 | 1 | 1024 | 1024 | 30% | Sparse |
| Sparse30% | 1 | 4 | 4 | 512 | 512 | 30% | Sparse |
| Sparse70% | 1 | 4 | 4 | 512 | 512 | 70% | Sparse |
| Dense 100% | 1 | 4 | 4 | 256 | 256 | 100% | Dense |
| Batch2 | 2 | 4 | 2 | 256 | 512 | 50% | Sparse |
| NonAligned | 1 | 4 | 2 | 300 | 300 | 50% | Sparse (SKIP) |
| Long Seq S2048 | 1 | 4 | 2 | 2048 | 2048 | 30% | Sparse |

### 测试覆盖维度

- **序列长度**：256/512/1024/2048/300（非对齐）
- **GQA 分组**：groupSize=1 (MHA), 2, 4
- **稀疏率**：30%/40%/50%/70%/100%
- **Batch**：1, 2
- **路径**：Sparse + Dense

---

## 测试性能预期

基于 Ascend 910B3 平台的基准性能数据（MHA Dense, B=1, Hq=Hkv=4）：

| Shape | Task Time | AICore Time | AICore Util |
|-------|-----------|-------------|-------------|
| 256×256 | ~75 us | ~1.3 ms | ~37% |
| 512×512 | ~92 us | ~2.6 ms | ~43% |
| 1024×1024 | ~180 us | ~6.5 ms | ~60% |

### 性能趋势

- AICore 利用率随序列长度增加而提升（256→1024: 37%→60%）
- 长序列（≥1024）场景下 AICore 利用率 >50%，计算密度较高
- 短序列（256）场景受限于并行度，22 核利用率约 37%

### 精度预期

| 指标 | 预期最大误差 | 容差阈值 |
|------|------------|---------|
| O max_diff | ≤ 0.000122 | atol=0.0001, rtol=0.0078125 |
| LSE max_diff | ≤ 0.000001 | atol=0.0001, rtol=0.0078125 |
