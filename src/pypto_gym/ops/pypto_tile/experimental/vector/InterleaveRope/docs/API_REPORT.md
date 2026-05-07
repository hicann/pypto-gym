---
schema_version: 1
op_name: interleave_rope
supported_dtypes: [float16, bfloat16]
dynamic_axes: ['B', 'S', 'N']
shape_constraints:
  - "x.shape == y.shape == [B, N, S, D=64]"
  - "cos.shape == sin.shape == [B, 1, S_cs, D=64], S_cs ∈ {1, S}"
  - "dtype(cos) == dtype(sin) == dtype(x)"
  - "all tensors contiguous (ND)"
tiling_required: true
feasibility: 可行
---

# API 探索报告 — interleave_rope

> **生成时间**: 2026-04-30

---

## 1. 概述

### 1.1 输入摘要

实现 interleave 风格 RoPE：对最后维 D=64 上的相邻元素对 (x[2i], x[2i+1]) 应用旋转：
- y[2i]   = x[2i]·cos_i − x[2i+1]·sin_i
- y[2i+1] = x[2i]·sin_i + x[2i+1]·cos_i

输入 x[B,N,S,64]、cos/sin[B,1,S|1,64]，dtype ∈ {fp16, bf16} 一致；输出 y[B,N,S,64]。
按用户要求采用 `pypto.gathermask` (PM=1/2) 拆奇偶。

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: 仅含 elementwise (mul/sub/add/cast) + gathermask + concat/reshape，无 matmul / reduction。

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | gather  | x_even = x[..., 0::2], x_odd = x[..., 1::2] | 沿 D 拆奇偶 |
| 2 | gather  | c_half = cos[..., 0::2], s_half = sin[..., 0::2] | 由于 cos/sin 在 interleave 模式下成对相同，仅需取偶位（或全用 PM=1）|
| 3 | cast    | xe, xo, ch, sh → fp32 | 累积精度 |
| 4 | mul/sub | ye = xe·ch − xo·sh | broadcast 沿 N、可能 S |
| 5 | mul/add | yo = xe·sh + xo·ch | broadcast 沿 N、可能 S |
| 6 | cast    | ye, yo → 原 dtype | |
| 7 | reshape+concat | y[..., 2i]=ye[..., i], y[..., 2i+1]=yo[..., i] | interleave 重组 |

> **说明**：interleave 模式下，按惯例 cos/sin 张量在 D 维上 cos[2i]==cos[2i+1]==cos(θ_i)（即对每对相邻元素共享同一 θ_i），因此 c_half/s_half 直接以 PM=1 取偶位即可。该假设需在 SPEC/golden 中明确定义。

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1a | x[..., 0::2] | `pypto.gathermask(x, pattern_mode=1)` | direct | ✓ (D=64, 64%2=0) |
| 1b | x[..., 1::2] | `pypto.gathermask(x, pattern_mode=2)` | direct | ✓ |
| 2a | cos[..., 0::2] | `pypto.gathermask(cos, pattern_mode=1)` | direct | ✓ |
| 2b | sin[..., 0::2] | `pypto.gathermask(sin, pattern_mode=1)` | direct | ✓ |
| 3 | cast→fp32 / fp32→bf16/fp16 | `pypto.cast(t, pypto.DT_FP32)` / `pypto.cast(t, dtype)` | direct | ✓ |
| 4 | xe·ch − xo·sh | `pypto.sub(pypto.mul(xe, ch), pypto.mul(xo, sh))` | direct | ✓ broadcast |
| 5 | xe·sh + xo·ch | `pypto.add(pypto.mul(xe, sh), pypto.mul(xo, ch))` | direct | ✓ broadcast |
| 6 | reshape | `pypto.reshape(ye, [B, N, S, D//2, 1])`, `pypto.reshape(yo, [B, N, S, D//2, 1])` | direct | ✓ |
| 7 | concat 末轴 | `pypto.concat([ye5d, yo5d], dim=-1)` → `pypto.reshape(_, [B, N, S, D])` | direct | ✓ |

可选 broadcast 加速（视实现需要）：

| 步骤 | API | 说明 |
|------|-----|------|
| 显式 broadcast | `pypto.expand_clone(c_half, [B, N, S, D//2])` | 当 elementwise 隐式 broadcast 不满足，或 S_cs=1 需要先扩到 S |

### 3.2 Substitute 配方

Interleave 重组（无专用 interleave API）：

```
方案 A（推荐，concat + reshape）:
    ye5d = pypto.reshape(ye, [B, N, S, D//2, 1])
    yo5d = pypto.reshape(yo, [B, N, S, D//2, 1])
    y5d  = pypto.concat([ye5d, yo5d], dim=-1)        # [B, N, S, D//2, 2]
    y    = pypto.reshape(y5d, [B, N, S, D])          # [B, N, S, D=64]

方案 B（备选，scatter_）:
    需构造 even_index = [0,2,...,62], odd_index = [1,3,...,63]
    pypto.scatter_(y, dim=-1, index=even_idx_tensor, src=ye)
    pypto.scatter_(y, dim=-1, index=odd_idx_tensor,  src=yo)
    （需额外 host 端 index 张量；不优于 A）
```

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | float16 / bfloat16 / float32 等 | fp16 或 bf16 | ✓ |
| contiguous | 必须连续 | SPEC 已约束连续 | ✓ |
| 非空 | 非空 Tensor | B≥1, N≥1, S≥1, D=64 | ✓ |
| format | ND | ND | ✓ |

来源：`docs/api/others/pypto-from_torch.md`

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| `gathermask` | dtype | DT_FP16 / DT_BF16 / DT_FP32 / 整型 | ✓ fp16/bf16 |
| `gathermask` | self.shape 尾轴 % 2 == 0 (PM=1/2) | D=64 | ✓ |
| `gathermask` | tile_shape 尾轴 % 2 == 0 (PM=1/2) | tile 尾轴定为偶数 | ✓ 设计阶段保证 |
| `gathermask` | view_shape 尾轴 % 2 == 0 | tile 尾轴定为偶数 | ✓ |
| `gathermask` | self 尾轴不做 view 切分 | D=64 不切 | ✓ |
| `mul/add/sub` | dtype | DT_FP16/BF16/FP32/INT16/INT32 | ✓ |
| `mul/add/sub` | broadcast | 多轴 broadcast 支持 | ✓ |
| `cast` | bf16 ↔ fp32 / fp16 ↔ fp32 | A2/A3/A5 均支持 | ✓ |
| `concat` | dtype | DT_FP16/BF16/FP32 | ✓ |
| `concat` | 除拼接维外 shape 一致 | 5D 末轴拼接，前 4 维一致 | ✓ |
| `reshape` | 元素总数一致 | B·N·S·(D/2)·2 = B·N·S·D | ✓ |
| `expand_clone` | 待扩 dim 必须为 1 | cos/sin N=1, S_cs∈{1,S} | ✓ |
| `set_vec_tile_shapes` | 维度数 ≤ 4，每维 > 0 | 4D 输出 | ⚠ 注意：interleave 重组 5D（B,N,S,D/2,2）阶段需另一组 tile，最多 4 维约束需在 design 阶段处理 |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes()` |

**关键考量**：
- 输出 4D `[B, N, S, D=64]`：tile 候选 `(1, n_tile, s_tile, 64)`，n_tile ∈ {1, 全 N}，s_tile 切 S。
- gathermask 阶段输入 4D `[B, N, S, 64]` → 输出 4D `[B, N, S, 32]`；按文档示例，TileShape 尾轴写 64（=2·32）。
- concat/reshape 重组阶段处于 5D 中间形态 `[B, N, S, 32, 2]`。tile 维度上限 4 维，需在 design 阶段评估是否走 4D 写回（直接以 `view + scatter`）或合并 (S,D/2) 为单维。
- 动态轴 B/S/N：tile 大小需在编译期为常量，因此采用"动态轴 + view 切分"策略。
- N 多态 {1,128}：可能需要按 N 走两条 tile 路径（design 阶段决策）。

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `models/deepseek_v4/common.py`（`rotate_half`, `inverse_rope_3d`） | models | 中 | 高 | RoPE 总体框架、fp32 内部累积、cos/sin reshape broadcast、tile_shape 多级设置 |
| `models/deepseek_v4/mla_prolog_v4_impl.py`（`rope_2d`, `rope_3d`） | models | 中 | 高 | reshape→transpose→reshape 的 RoPE 风格（**rotate_half 风格**，与本算子 interleave 形态相似但拆分方式不同），可参考 tile 配置 `(1,64,64)`/`(1,64,64,64)` |
| `models/deepseek_v32_exp/mla_prolog_quant_impl.py`（`rope_v2`, `rope_3d_v2`） | models（exp 同级） | 中 | 高 | `RopeTileShapeConfig` 抽象、平台感知 tile（DAV_3510 → `(1,128,64)`） |
| `models/glm_v4_5/glm_attention_pre_quant.py`（`rope_data`） | models | 低 | 高 | 明确 `o1=x1·cos − x2·sin, o2=x2·cos + x1·sin` 后 `pypto.concat([o1,o2], dim=2)`，是 chunk/rotate_half 风格而非 interleave |
| `examples/01_beginner/transform/add_scalar_loop_view_assemble.py` | examples | 低 | 高 | 4D `[B,N,S,D]` + `set_vec_tile_shapes(1,4,1,64)` + loop view/assemble 模板 |
| `examples/02_intermediate/basic_nn/ffn/ffn_module.py`（`gelu_activation_core`, `swiglu_activation_core`） | examples | 低 | 高 | bf16↔fp32 cast 模式：`x_fp32 = pypto.cast(x, DT_FP32)` → 计算 → `pypto.cast(_, DT_BF16)` |
| `examples/03_advanced/advanced_nn/attention/attention.py` | examples | 低 | 高 | 4D `[B,H,S,D]` 上 `set_vec_tile_shapes(1, 8, 16, HEAD_DIM)` |

> **重要差异**：仓内现有所有 RoPE 参考均为 **rotate_half / chunk** 风格（把 D 切成前后两半），**没有任何示例使用 gathermask + interleave 模式**。本算子是首例，需要从零设计 interleave 重组路径。

### 6.2 可复用模式

- **API 调用模式**：`from_torch` → `set_vec_tile_shapes(...)` → 计算流（gathermask + cast + mul/add/sub + concat/reshape）→ 写回。
- **Tiling 策略**：4D 输出 `[B,N,S,D]`，按 `(1, n_tile, s_tile, D)` 切；S 主切轴，N=128 时按子块切，N=1 时整 N。
- **Loop 结构**：典型 RoPE 不显式循环（编译期 tile 自动展开）；如出现 5D 中间形态，需在 design 阶段合并轴。
- **边界处理**：dtype mixed-precision（fp16/bf16 → fp32 → fp16/bf16）；S_cs=1 时显式 `expand_clone` 或依赖隐式 broadcast。

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| RoPE 风格 | rotate_half / chunk | interleave (相邻配对) | 用 gathermask PM=1/2 替代 view+chunk |
| 拆分 API | `pypto.view` + `chunk` | 用户指定 `pypto.gathermask` | 直接采纳 gathermask；记录与 view 方案的性能对比作为 perf 阶段备选 |
| 重组 API | `concat` 在 dim=-1（chunk 直接拼接） | 需 interleave 拼接 | 5D `concat` + `reshape` 方案 |
| cos/sin shape | `[seq, rope_dim]` 一致 | `[B,1,S\|1,D]` 4D | 通过 `expand_clone` / 隐式 broadcast 适配 |
| dtype | bf16 主流 | bf16 + fp16 双支持 | 内部统一 cast 到 fp32 |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 仓内无 interleave-RoPE 参考 | 现有实现均为 rotate_half | 从零设计；在 design 阶段画出 5D 中间形态及 tile 推导 |
| `set_vec_tile_shapes` 上限 4 维 vs 5D 中间形态 | concat 需要 `[B,N,S,D/2,2]` | 在 design 阶段确认：是否将 5D 视作仅 `reshape`（不参与 vector 计算），由 vector tile 仅作用于 4D 输入/输出阶段 |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| cos/sin 语义假设 | interleave 模式下默认 cos[2i]==cos[2i+1]；若用户 cos/sin 已展开成 D 长度且含偶奇不同值，需在 SPEC/golden 显式约定 |
| S_cs=1 vs S_cs=S | 两条数据路径；至少各保留一条 P0 用例 |
| N=1 vs N=128 | tile 配置可能不同；初版可统一 N 切块策略，perf 阶段再分化 |
| 精度 atol=1e-4 | 严格于通用 1e-3；必须 fp32 内部累积；避免直接 bf16/fp16 乘加 |
| gathermask 尾轴不做 view 切分 | 设计 tile 时 D 必须整段进 ub |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 列表 | `docs/api/operation/index.md` |
| gathermask 文档 | `docs/api/operation/pypto-gathermask.md` |
| concat 文档 | `docs/api/operation/pypto-concat.md` |
| reshape 文档 | `docs/api/operation/pypto-reshape.md` |
| cast 文档 | `docs/api/operation/pypto-cast.md` |
| mul / add / sub 文档 | `docs/api/operation/pypto-{mul,add,sub}.md` |
| expand_clone 文档 | `docs/api/operation/pypto-expand_clone.md` |
| scatter_ 文档（备选） | `docs/api/operation/pypto-scatter_.md` |
| set_vec_tile_shapes 文档 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| 参考实现 (rotate_half) | `models/deepseek_v4/common.py`, `models/deepseek_v4/mla_prolog_v4_impl.py`, `models/deepseek_v32_exp/mla_prolog_quant_impl.py`, `models/glm_v4_5/glm_attention_pre_quant.py` |
| 4D tile 模板 | `examples/01_beginner/transform/add_scalar_loop_view_assemble.py`, `examples/03_advanced/advanced_nn/attention/attention.py` |
| dtype cast 模板 | `examples/02_intermediate/basic_nn/ffn/ffn_module.py` |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**: 仓内无 interleave 风格 RoPE 参考实现，需自行设计 5D concat+reshape 的 interleave 重组路径；并在 design 阶段确认 tile 维度上限（≤4）下的拓扑（4D 计算 + 5D 仅 reshape）。
- **首选实现路径**:
  1. `gathermask(x, PM=1/2)` 拆奇偶
  2. `gathermask(cos/sin, PM=1)` 取偶位（基于 interleave 模式 cos/sin 假设）
  3. cast → fp32，计算 ye, yo，cast 回原 dtype
  4. `reshape(_, [B,N,S,D/2,1])` × 2 → `concat(_, dim=-1)` → `reshape(_, [B,N,S,D])`
- **次选**：用 `pypto.view` 直接做奇偶 stride 切分（已在仓内使用），作为 perf 阶段对比基线。
- **dtype**：仅支持 fp16 / bf16，内部 fp32 累积。
- **Tiling**：`set_vec_tile_shapes` 4D 模式作用于 [B,N,S,D]；S 为主切轴，N 按取值 (1/128) 决策。
