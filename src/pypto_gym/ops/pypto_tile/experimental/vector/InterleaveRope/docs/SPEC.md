---
schema_version: 1
op_name: interleave_rope
supported_dtypes: [float16, bfloat16]
p0_shapes: [[1, 1, 1024, 64], [1, 128, 2048, 64], [4, 128, 8192, 64]]
tolerance: {atol: 0.0001, rtol: 0.0078125}
dynamic_axes: ['B', 'S', 'N']
dynamic_axes_ranges: {B: [1, 4], S: [1, 8192], N: [1, 128]}
shape_constraints:
  - "x.shape == y.shape == [B, N, S, D]"
  - "cos.shape == sin.shape == [B, 1, S_cs, D]"
  - "S_cs == 1 or S_cs == S"
  - "D == 64 (constant)"
  - "cos.N == sin.N == 1"
  - "dtype(cos) == dtype(sin) == dtype(x) == dtype(y)"
  - "supported dtypes: float16, bfloat16"
  - "all tensors are contiguous (ND format)"
default_params: {}
perf_target: "首跑成功性能的 2 倍"
---

## 算子需求规范

### 1. 基础信息
- **算子名称**: interleave_rope
- **算子分类**: custom (RoPE / position embedding)

### 1.1 功能描述

对输入特征张量 x 在最后一维（特征维 D）上的相邻元素对 (x[2i], x[2i+1]) 应用旋转位置编码 (Rotary Position Embedding, RoPE)。
采用 **interleave** 模式：对相邻元素两两配对旋转（区别于 rotate_half 把 D 平分成前后两半的模式）。
sin/cos 矩阵使用 `pypto.gathermask` 在 D 维上以 PatternMode=1/2 间隔取数（取偶/奇位置），实现奇偶位置的有效系数提取。

**输出 layout**：本算子约定输出采用 **split-half layout**（不是 interleave 排布）——
计算的是 interleave 模式 RoPE，但旋转后的奇偶分量在输出张量中分开放置：

```
out[..., 0:D/2 ] = y_even = [y_origin[0], y_origin[2], ..., y_origin[D-2]]
out[..., D/2:D] = y_odd  = [y_origin[1], y_origin[3], ..., y_origin[D-1]]
```

下游 attention 在 Q/K 都按 split-half 排布时，`QK^T = Σ_d Q[d]·K[d]` 与原 interleave 输出数值等价
（点积对 D 维顺序不敏感）。该约定免除了 kernel 内 5D `reshape+concat` 重组的开销。
golden 同样输出 split-half 排布以保持与 impl 可对比。

### 1.2 算法参数

无（θ_i 通过 cos/sin 张量直接传入）。

### 1.3 数学公式

中间结果 `y_origin` 按 interleave 模式定义（$i \in [0, D/2)$，$D = 64$）：

$$
\begin{aligned}
y_{\mathrm{origin}}[\dots, 2i]   &= x[\dots, 2i] \cdot \cos\theta_i - x[\dots, 2i+1] \cdot \sin\theta_i \\
y_{\mathrm{origin}}[\dots, 2i+1] &= x[\dots, 2i] \cdot \sin\theta_i + x[\dots, 2i+1] \cdot \cos\theta_i
\end{aligned}
$$

最终输出（split-half layout）：

$$
\begin{aligned}
y[\dots, i]            &= y_{\mathrm{origin}}[\dots, 2i]   \quad (i \in [0, D/2)) \\
y[\dots, i + D/2]      &= y_{\mathrm{origin}}[\dots, 2i+1] \quad (i \in [0, D/2))
\end{aligned}
$$

### 2. 关键特性

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| interleave 配对 (相邻 2i / 2i+1) | ✓ 需要 | ✓ 高 | 用 gathermask PM=1 取偶位 (2i)，PM=2 取奇位 (2i+1)；区别于 rotate_half | P0 |
| gathermask 间隔取数 | ✓ 需要 | ✓ 高 | 应用于 cos、sin、x 的 D 维，将 D=64 拆为 D/2=32 | P0 |
| cos/sin 同 dtype 约束 | ✓ 需要 | ✓ 高 | dtype(cos)==dtype(sin)==dtype(x)，支持 fp16 / bf16 | P0 |
| cos/sin 双形态 S 维 (1 或 S) | ✓ 需要 | ✓ 高 | 当 S_cs==1 时沿 S broadcast；否则一一对应 | P0 |
| cos/sin N 维必须为 1 | ✓ 需要 | ✓ 高 | 强约束；沿 N broadcast | P0 |
| D 固定为 64 | ✓ 需要 | ✓ 高 | 常量；可在 kernel 中以编译期常量处理 | P0 |
| 动态轴 B (1~4) | ✓ 需要 | ✓ 高 | 入参动态轴 | P0 |
| 动态轴 S (1~8192) | ✓ 需要 | ✓ 高 | 主切分轴；tiling 沿 S 切分 | P0 |
| 多态 N (1 或 128) | ✓ 需要 | ✓ 高 | tiling 需考虑两种取值 | P0 |
| 连续张量约束 | ✓ 需要 | ✓ 高 | 不支持非连续 Tensor，可直接按 ND 内存布局处理 | P0 |
| 内部 fp32 累加 | ⚠ 推荐 | ⚠ 中 | bf16/fp16 cast 到 fp32 计算后再 cast 回原 dtype，满足 atol=1e-4 | P1 |

### 3. 算法描述

```
Algorithm: interleave_rope
────────────────────────────────────
输入: x   ∈ R^{B×N×S×D},        dtype ∈ {fp16, bf16}
      cos ∈ R^{B×1×S_cs×D},     dtype 同 x   (S_cs ∈ {1, S})
      sin ∈ R^{B×1×S_cs×D},     dtype 同 x   (S_cs ∈ {1, S})
输出: y   ∈ R^{B×N×S×D},        dtype 同 x

约束: D = 64; cos.N = sin.N = 1; 全部 contiguous

1. 用 gathermask 在 D 维上拆奇偶：
   x_even = gathermask(x,   PatternMode=1)   # [B, N, S,    D/2]   取 2i
   x_odd  = gathermask(x,   PatternMode=2)   # [B, N, S,    D/2]   取 2i+1
   c_half = gathermask(cos, PatternMode=1)   # [B, 1, S_cs, D/2]   (cos[2i] = cos[2i+1])
   s_half = gathermask(sin, PatternMode=1)   # [B, 1, S_cs, D/2]
2. 沿 N (必要时沿 S) broadcast c_half / s_half 到 [B, N, S, D/2]
3. （可选）cast 到 fp32:  xe, xo, ch, sh = cast_fp32(...)
4. 计算:
   ye = xe * ch - xo * sh                    # [B, N, S, D/2]
   yo = xe * sh + xo * ch                    # [B, N, S, D/2]
5. cast 回原 dtype 后，按 D 维 interleave 拼接:
   y[..., 2i]   = ye[..., i]
   y[..., 2i+1] = yo[..., i]
6. 返回 y ∈ R^{B×N×S×D}
```

### 4. 数据流图

```
   x   [B, N, S,    D=64]  bf16/fp16  (ND, contiguous)
   cos [B, 1, S|1, D=64]  same dtype  (N=1)
   sin [B, 1, S|1, D=64]  same dtype  (N=1)

   ┌── gathermask(x,   PM=1) ──▶ x_even [B, N, S,    32]
   x ─┤
   │  └── gathermask(x,   PM=2) ──▶ x_odd  [B, N, S,    32]
   │
   ├── gathermask(cos, PM=1) ──▶ c_half [B, 1, S|1, 32]
   └── gathermask(sin, PM=1) ──▶ s_half [B, 1, S|1, 32]

   broadcast c_half, s_half → [B, N, S, 32]
   (optional cast to fp32)

         ye = x_even * c_half - x_odd  * s_half     [B, N, S, 32]
         yo = x_even * s_half + x_odd  * c_half     [B, N, S, 32]

   interleave 重组 (y[2i]=ye[i], y[2i+1]=yo[i])
                    ▼
              y [B, N, S, D=64] bf16/fp16
```

### 5. 输入输出规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------|-------|--------|--------|------|
| x   | [B, N, S, D]    | float16 / bfloat16 | B, N, S | ✓ 高 | 待旋转特征张量；D=64；连续 ND |
| cos | [B, 1, S\|1, D] | 与 x 同 dtype       | B, S    | ✓ 高 | 旋转角余弦；N 必须=1；S 可为 1 或与 x 同；连续 ND |
| sin | [B, 1, S\|1, D] | 与 x 同 dtype       | B, S    | ✓ 高 | 旋转角正弦；N 必须=1；S 可为 1 或与 x 同；连续 ND |

**输出规格**:

| 变量 | Shape | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------|-------|--------|--------|------|
| y | [B, N, S, D] | 与 x 同 dtype | B, N, S | ✓ 高 | RoPE 旋转后的特征张量；D=64；连续 ND |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float16  | ✓ | 0.0001 | 0.0078125 | x/cos/sin/y 同 dtype |
| bfloat16 | ✓ | 0.0001 | 0.0078125 | x/cos/sin/y 同 dtype |
| float32  | ✗ | -      | -         | 不在支持列表 |

### 7. 精度要求
- **atol**: 0.0001
- **rtol**: 0.0078125

### 8. 动态轴说明
- **动态轴**: B, S, N
- **轴含义**:
  - B: batch
  - N: head/group 数
  - S: sequence length
  - D: 特征维 (常量 = 64)
- **取值范围**:
  - B ∈ [1, 4]
  - S ∈ [1, 8192]
  - N ∈ {1, 128}
  - D = 64

### 9. 边界条件处理
- **零值**: 正常计算
- **极值**: 正常计算（依赖 cos/sin 输入数值范围 [-1, 1]）
- **NaN/Inf**: 不做特殊处理，按 IEEE 浮点语义传递

### 10. 性能要求
- **性能目标**: 首跑成功性能的 2 倍

### 11. 参考信息
- **参考实现**: PyTorch RoPE (interleave 风格); LLaMA / GPT-NeoX RoPE 实现
- **论文**: RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
- **类似算子**: rotate_half_rope (rotate-half 模式)、apply_rotary_pos_emb

### 12. 应用场景
- **目标模型**: LLM 系列 (LLaMA / Qwen / GLM / GPT-NeoX 等)
- **使用位置**: Attention 模块 Q/K projection 之后，QK^T 之前

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 功能_P0_min      | 功能 | P0 | B=1, N=1,   S=1024, D=64 | x [1,1,1024,64], cos/sin [1,1,1024,64] | y [1,1,1024,64] | 单 head 短序列功能验证 |
| 功能_P0_typ      | 功能 | P0 | B=1, N=128, S=2048, D=64 | x [1,128,2048,64], cos/sin [1,1,2048,64] | y [1,128,2048,64] | 多头典型配置 |
| 功能_P0_Scs1     | 功能 | P0 | B=2, N=128, S=4096, D=64, S_cs=1 | x [2,128,4096,64], cos/sin [2,1,1,64] | y [2,128,4096,64] | cos/sin S=1 broadcast |
| 性能_P0_max      | 性能 | P0 | B=4, N=128, S=8192, D=64 | x [4,128,8192,64], cos/sin [4,1,8192,64] | y [4,128,8192,64] | 动态轴上限性能场景 |

---
*生成时间: 2026-04-30*
*确认状态: 已确认*
*置信度说明: ✓ 高（用户明确给出） / ⚠ 中（推断，需后续验证）*
