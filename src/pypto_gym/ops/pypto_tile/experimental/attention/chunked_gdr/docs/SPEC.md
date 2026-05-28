---
schema_version: 1
op_name: chunked_gated_delta_rule
supported_dtypes: [float32]
p0_shapes:
  - query: [128, 2, 128]
    key: [128, 2, 128]
    value: [128, 4, 128]
    beta: [128, 4]
    gate: [128, 4]
    states: [1, 4, 128, 128]
    mask: [128, 128]
    tril_mask: [128, 128]
    eye: [16, 128]
    act_seq_len: [2]
    core_attn_out: [128, 4, 128]
    last_state_data: [1, 4, 128, 128]
tolerance:
  type: standard
  atol: 0.001
  rtol: 0.001
dynamic_axes: ['T', 'B']
dynamic_axes_ranges: {'T': [1, 65536], 'B': [1, 64]}
shape_constraints: 'Nv % Nqk == 0; Nv >= 4; D == 128; L in {32, 64, 128}'
default_params: {'chunk_size': 'auto', 'head_dim': 128, 'eps': 1e-6, 'scale': '1/sqrt(128)', 'rtol': 1e-3, 'atol_abs': 0, 'atol_rel': 1e-3}
activation_stats:  # Observed value ranges from Qwen3-next real model inference
  query:  {max: 1.3655, min: -0.2785}   # formula: rand*(max+|min|)-(max+|min|) → all-negative
  key:    {max: 1.4664, min: -0.2785}   # same formula pattern; negative values valid in delta rule
  value:  {max: 1.6488, min: -0.2785}
  beta:   {max: 0.8927, min: -0.0889}   # formula: rand*(max-min)-(max-min); neg beta → |beta| as decay
  gate:   {max: 37.5452, min: -0.1343}  # formula: rand*(low+high)-(low+high); neg gate → exp(gate)∈(0,1)
perf_target: '首跑精度成功性能的 2 倍'
---

## 算子需求规范

### 1. 基础信息
- **算子名称**: chunked_gated_delta_rule
- **算子分类**: attention  <!-- 分块门控 Delta Rule 线性注意力 -->

### 1.1 功能描述

实现分块门控 Delta Rule 线性注意力机制（Chunked Gated Delta Rule Linear Attention），将传统 O(n²) 复杂度的 Attention 降低到 O(n)。

算子对输入序列按可配置 chunk_size (L=32/64/128) 进行分块，在每个 chunk 内计算：
1. L2 归一化后的 Q/K
2. 带门控衰减的预注意力矩阵 A = (kβ @ k^T) * D_decay * mask
3. 分块递推法求逆 (I - A)^{-1}
4. Value 和 Key 的累积衰减
5. 跨 chunk 的循环状态注意力更新

支持 GQA（Grouped Query Attention）模式，Nv 必须是 Nqk 的整数倍。

提供两个版本：
- **aligned 版本**: 序列长度整除 L 的情况
- **unaligned 版本**: 序列长度不整除 L，使用 fillpad + assemble 处理尾部不满 chunk

### 1.2 算法参数

| 参数 | 值 | 说明 |
|------|-----|------|
| chunk_size (L) | 32 / 64 / 128 | 分块长度，可配置（auto 自动选择最优 L） |
| head_dim (D) | 128 | 头维度，硬编码固定值 |
| scale | 1/sqrt(128) = 1/sqrt(D) | 缩放因子，kernel 内部计算 |
| eps | 1e-6 | L2 归一化的 epsilon |
| group | Nv // Nqk | GQA 分组比（运行时计算） |

### 1.3 数学公式

$$
\begin{aligned}
&\text{L2归一化: } \hat{q} = \frac{q}{\sqrt{\sum q_i^2 + \epsilon}}, \quad \hat{k} = \frac{k}{\sqrt{\sum k_i^2 + \epsilon}} \\
&\text{预注意力: } g_{cum} = \text{cumsum}(g) = \text{tril} \cdot g \\
&\quad D_{decay} = \exp((g_{cum} - g_{cum}^T) \cdot \text{tril}) \\
&\quad k_\beta = k \cdot \beta \\
&\quad A = (k_\beta \cdot k^T) \cdot D_{decay} \cdot \text{mask} \\
&\text{矩阵求逆: } A_{inv} = (I - A)^{-1} \quad (\text{分块递推法}) \\
&\text{累积衰减: } v_{out} = A_{inv} \cdot (v \cdot \beta) \\
&\quad k_{cumdecay} = A_{inv} \cdot (k_\beta \cdot \exp(g_{cum})) \\
&\text{循环状态: } v' = k_{cumdecay} \cdot S^T \\
&\quad o_{inter} = (q \cdot \exp(g_{cum})) \cdot S^T \\
&\quad o_{chunk} = o_{inter} + (q \cdot k^T \cdot D_{decay} \cdot \text{tril}) \cdot (v_{out} - v') \\
&\quad S_{new} = S \cdot \exp(g_{last}) + v^T \cdot k_{gexp} - v'^T \cdot k_{gexp} \\
&\quad \text{其中 } k_{gexp} = k \cdot \exp(g_{last} - g)
\end{aligned}
$$

### 2. 关键特性

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| 分块策略 (chunk_size=32/64/128) | ✓ 需要 | ✓ 高 | 序列按 L 分块，循环迭代处理每个 chunk，auto 自动选择最优 L | P0 |
| 门控衰减 (gated decay) | ✓ 需要 | ✓ 高 | gate 信号控制时序衰减，计算 decay_mask | P0 |
| 矩阵求逆 (分块递推法) | ✓ 需要 | ✓ 高 | 128×128 矩阵分 8×8 个 16×16 子块，递推求逆 (I-A)^{-1} | P0 |
| 循环状态 (recurrent state) | ✓ 需要 | ✓ 高 | 跨 chunk 的状态 S 递归更新，首尾衔接 | P0 |
| GQA 支持 | ✓ 需要 | ✓ 高 | Nv // Nqk 分组映射，nqk_idx = nv_idx // group | P0 |
| 动态 Shape (T, B) | ✓ 需要 | ✓ 高 | T 和 B 使用 pypto.DYNAMIC | P0 |
| 多输出 | ✓ 需要 | ✓ 高 | 同时输出 core_attn_out 和 last_state_data | P0 |
| valid_shape 处理 | ✓ 需要 | ✓ 高 | 处理不满 chunk 的尾部（actual_l < L） | P0 |
| unaligned 版本 | ✓ 需要 | ✓ 高 | fillpad + assemble 处理尾部不满 chunk | P0 |
| L2 归一化 | ✓ 需要 | ✓ 高 | query 和 key 的 L2 归一化 | P0 |
| causal mask | ✓ 需要 | ✓ 高 | 下三角掩码实现因果注意力 | P0 |
| 合图优化 (stitch) | ✓ 需要 | ✓ 高 | stitch_function_max_num=2 | P1 |
| combine_axis 优化 | ✓ 需要 | ✓ 高 | 循环维度合并优化 | P1 |
| UB 预取优化 (+0.0) | ✓ 需要 | ⚠ 中 | 编译器特性的零开销预取 | P2 |
| scope 内存管理 | ✓ 需要 | ⚠ 中 | sg_set_scope 管理 zeros_16/32/64 的生命周期 | P1 |
| BF16 输入支持 | ✗ 不需要 | ✓ 高 | 当前版本仅支持 FP32 输入输出 | P3 |

### 3. 算法描述

```
Algorithm: Chunked Gated Delta Rule Linear Attention (Forward)
────────────────────────────────────────────────────────────────
输入: query [T,Nqk,D], key [T,Nqk,D], value [T,Nv,D], beta [T,Nv], gate [T,Nv],
      states [B,Nv,D,D], mask [L,L], tril_mask [L,L], eye [16,L/16], act_seq_len [B+1]
输出: core_attn_out [T,Nv,D], last_state_data [B,Nv,D,D]

1. 计算 group = Nv // Nqk (GQA 分组比)
2. for b_idx = 0 to B-1:                          // batch 循环
     s = act_seq_len[b_idx+1] - act_seq_len[b_idx]  // 当前 batch 序列长度
     for nv_idx = 0 to Nv-1:                       // value head 循环
       nqk_idx = nv_idx // group                   // GQA 映射到 QK head
       bs_ofs = act_seq_len[b_idx]                 // batch 在扁平序列中的偏移
       初始化 last_state = states[b_idx, nv_idx]
       for s_idx = 0 to s, step L=128:             // chunk 循环
         actual_l = min(s - s_idx, L)              // 当前 chunk 实际长度
         使用 pypto.view 从全局 tensor 切片:
           query_view [actual_l, 1, D]
           key_view [actual_l, 1, D]
           value_view [actual_l, 1, D]
           gate_view [actual_l, 1]
           beta_view [actual_l, 1]
         reshape 为 2D: [actual_l, D] / [actual_l, 1]

         Step 1: L2 归一化
           query_norm = query_view / sqrt(sum(query², dim=-1) + eps)
           key_norm = key_view / sqrt(sum(key², dim=-1) + eps)

         Step 2: 预注意力计算
           gate_cum = tril_mask @ gate_view          // [L,1]
           decay_mask = exp((gate_cum - gate_cum^T) * tril_mask)  // [L,L]
           key_beta = key_norm * beta_view            // [L,D]
           A = matmul(key_beta, key_norm^T) * decay_mask * mask  // [L,L]

         Step 3: 矩阵求逆 (分块递推法)
           A_inv = inverse_pto(A, eye, zeros_16, zeros_32, zeros_64)
           // 将 128×128 分为 8×8 个 16×16 子块
           // 行递推求逆 → 逐步合并 16→32→64→128

         Step 4: 累积衰减计算
           v_out = A_inv @ (value_norm * beta_view)
           k_cumdecay = A_inv @ (key_beta * exp(gate_cum))

         Step 5: 循环状态注意力
           v_prime = k_cumdecay @ last_state^T
           o_inter = (query_norm * exp(gate_cum)) @ last_state^T
           attn = matmul(query_norm, key_norm^T)
           chunk_attn_value = (attn * decay_mask * tril_mask) @ v_out
           chunk_attn_vprime = (attn * decay_mask * tril_mask) @ v_prime
           chunk_attn_out = o_inter + chunk_attn_value - chunk_attn_vprime

           // 状态更新
           k_gexp = key_norm * exp(g_last - gate_view)
           state_new = last_state * exp(g_last) 
                      + matmul(v_out^T, k_gexp) 
                      - matmul(v_prime^T, k_gexp)

         Step 6: 写回
           last_state[:] = state_new
           core_attn_out[bs_ofs:bs_ofs+actual_l, nv_idx] = chunk_attn_out
           (unaligned 版本尾部 chunk 使用 fillpad + assemble)

3. last_state_data[b_idx, nv_idx] = last_state
4. return core_attn_out, last_state_data
```

### 4. 数据流图

```
     query [T,Nqk,D]    key [T,Nqk,D]    value [T,Nv,D]    beta [T,Nv]    gate [T,Nv]
          │                   │                 │                │              │
          │                   │                 │                │              │
     ┌────▼────┐         ┌────▼────┐           │                │              │
     │ L2Norm  │         │ L2Norm  │           │                │              │
     │ q_norm  │         │ k_norm  │           │                │              │
     └────┬────┘         └────┬────┘           │                │              │
          │                   │                 │                │              │
          │                   ├─────────────────┘                │              │
          │                   │ kβ = k*β                         │              │
          │                   │                                  │              │
          │              ┌────▼────┐                             │              │
          │              │Pre-Attn │◄─── tril_mask, mask ────────┤──────────────┤
          │              │ A=L×L   │                             │              │
          │              └────┬────┘                             │              │
          │                   │                                  │              │
          │              ┌────▼────┐                             │              │
          │              │Inverse  │◄─── eye                     │              │
          │              │A_inv    │                             │              │
          │              └────┬────┘                             │              │
          │                   │                                  │              │
          │              ┌────▼────────────────────┐             │              │
          │              │Cum-Decay                │◄────────────┘──────────────┤
          │              │ v_out, k_cumdecay       │  v*β, kβ*exp(g_cum)      │
          │              │         [L,D]           │             gate_cum      │
          │              └────┬──────┬─────────────┘             │              │
          │                   │      │                           │              │
          │                   │      │                           │              │
          │                   │      └──► v_prime = k_cum @ S^T │              │
          │                   │          o_inter = q*g_exp @ S^T │              │
          │                   │                           ┌─────▼─────┐        │
          │                   │                           │ Recurrent │◄── S₀  │
          │                   │                           │ State     │        │
          │                   │                           │ Attn      │        │
          │                   ├──────────────────────────►│           │        │
          │                   │                           │ chunk_out │        │
          │                   │                           │ S_new     │──► S_t+1│
          │                   │                           └─────┬─────┘        │
          │                   │                                 │              │
          │                   │                                 │              │
          ▼                   ▼                                 ▼              │
    ┌────────────────────────────────────────────┐                            │
    │        core_attn_out [T,Nv,D]              │                            │
    │        last_state_data [B,Nv,D,D]          │◄───────────────────────────┘
    └────────────────────────────────────────────┘
```

### 5. 输入输出规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------|-------|--------|--------|------|
| query | [T, Nqk, D] | DT_FP32 | T | ✓ 高 | 查询向量，D=128 固定 |
| key | [T, Nqk, D] | DT_FP32 | T | ✓ 高 | 键向量，D=128 固定 |
| value | [T, Nv, D] | DT_FP32 | T | ✓ 高 | 值向量，Nv % Nqk == 0 |
| beta | [T, Nv] | DT_FP32 | T | ✓ 高 | Beta 缩放因子 |
| gate | [T, Nv] | DT_FP32 | T | ✓ 高 | 门控衰减信号 |
| states | [B, Nv, D, D] | DT_FP32 | B | ✓ 高 | 初始循环状态矩阵 |
| mask | [L, L] (L=128) | DT_FP32 | — | ✓ 高 | 注意力掩码矩阵（下三角负值掩码） |
| tril_mask | [L, L] (L=128) | DT_FP32 | — | ✓ 高 | 下三角掩码矩阵 |
| eye | [16, 128] (aligned) / [16, 16] (unaligned) | DT_FP32 | — | ✓ 高 | 求逆用的特殊单位矩阵 |
| act_seq_len | [B+1] | DT_INT32 | B | ✓ 高 | 各 batch 累积序列长度索引 |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------|-------|--------|--------|------|
| core_attn_out | [T, Nv, D] | DT_FP32 | T | ✓ 高 | 注意力计算输出 |
| last_state_data | [B, Nv, D, D] | DT_FP32 | B | ✓ 高 | 更新后的循环状态矩阵 |

### 6. 数据类型支持

| Dtype | 支持 | atol_abs | atol_rel | rtol | 备注 |
|-------|------|----------|----------|------|------|
| float32 | ✓ P0 | 0 | 1e-3 | 1e-3 | 唯一支持的 dtype |

### 7. 精度要求
- **atol_abs**: 0
- **atol_rel**: 1e-3
- **rtol**: 1e-3
- 精度比较公式: `tolerance = atol_abs + atol_rel * |expected|`
- 所有 matmul 指定 `pypto.DT_FP32` 输出 dtype，确保 FP32 精度计算

### 8. 动态轴说明
- **动态轴**: ['T', 'B']
- **轴含义**: T = 总序列长度（所有 batch 的序列之和），B = batch 数
- **取值范围**: T ∈ [1, 65536], B ∈ [1, 64]
- **固定轴**: Nqk ∈ [1, 64], Nv ∈ [1, 64], D = 128, L = 128

### 9. 边界条件处理
- **零值**: 正常计算（L2Norm 中 eps=1e-6 防止除零）
- **极值**: 正常计算
- **NaN/Inf**: 正常计算（exp 运算可能在极端 gate 值下溢出/溢出）

### 10. 性能要求
- **性能目标**: 首跑精度成功性能的 2 倍
- **关键优化**: 
  - stitch_function_max_num=2（合图优化）
  - combine_axis=True（循环维度合并）
  - Cube TileShape 精细调整（小M小N、小M大K等场景）
  - unroll_list=[16,1] 循环展开
  - +0.0 UB 预取优化

### 11. 参考信息
- **参考实现**: 
  - PyPTO: `/mnt/workspace/gitCode/cann/mce/pypto_fork/pypto_6304/models/qwen3_next/gated_delta_rule_impl.py`
  - AscendC: `/mnt/workspace/gitCode/cann/mce/ops-transformer/attention/chunk_gated_delta_rule/`
  - original.md: `/mnt/workspace/gitCode/cann/mce/pypto_fork/pypto_6304/models/experimental/attention/chunked_gated_delta_rule/docs/original.md`
- **论文**: Gated Delta Rule (线性注意力变体)
- **类似算子**: Flash Attention, Linear Attention, Delta Rule

### 12. 应用场景
- **目标模型**: Qwen3-Next (GLA-based attention)
- **使用位置**: 注意力层，替代传统 softmax attention

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0 | 性能 | P0 | B=2, Nqk=2, Nv=4, D=128, T=4096 | q/k:[4096,2,128], v/beta/gate:[4096,4,128], states:[2,4,128,128] | attn_out:[4096,4,128], state:[2,4,128,128] | 核心性能场景，aligned (4096%128==0) |
| 功能_P0 | 功能 | P0 | B=2, Nqk=2, Nv=4, D=128, T=4097 | q/k:[4097,2,128], v/beta/gate:[4097,4,128], states:[2,4,128,128] | attn_out:[4097,4,128], state:[2,4,128,128] | unaligned 场景，尾部不满 chunk |
| 功能_P0_GQA | 功能 | P0 | B=1, Nqk=2, Nv=8, D=128, T=128 | q/k:[128,2,128], v/beta/gate:[128,8,128], states:[1,8,128,128] | attn_out:[128,8,128], state:[1,8,128,128] | GQA 模式 (group=4) |
| 功能_P0_single | 功能 | P0 | B=1, Nqk=2, Nv=2, D=128, T=128 | q/k:[128,2,128], v/beta/gate:[128,2,128], states:[1,2,128,128] | attn_out:[128,2,128], state:[1,2,128,128] | 最小配置 (group=1) |

---
*生成时间: 2026-05-27*
*确认状态: 已确认（基于 original.md 完整规格直接生成，已更新 L=32/64/128 和 Nv≥4 约束）*
*置信度说明: ✓ 高（基于 qwen3_next PyPTO 参考实现提取，原始代码验证）*