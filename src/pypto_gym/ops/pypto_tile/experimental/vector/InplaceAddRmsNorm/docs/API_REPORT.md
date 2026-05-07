---
schema_version: 1
op_name: inplace_add_rms_norm
supported_dtypes: [bfloat16]
dynamic_axes: ['B', 'S']
shape_constraints: 'B*S in [1024, 8192] or (B in [16, 144] and S == 1)'
tiling_required: vec
feasibility: feasible_with_caveats
---

# API 探索报告

> **生成时间**: 2026-05-06

---

## 1. 概述

### 1.1 输入摘要

`inplace_add_rms_norm` 融合算子，bfloat16 输入：
- 输入: `x1, x2 ∈ [B,S,H]`, `gamma ∈ [H]`, `eps`（标量）
- Inplace 输出（复用输入 buffer）：
  - `x1 ← y = RmsNorm(x1+x2) * gamma`
  - `x2 ← x_add = x1+x2`
  - `rstd ← 1/sqrt(mean((x1+x2)^2)+eps)`（独立 buffer）
- H = 7168 固定；B/S 动态 (B*S ∈ [1024,8192] 或 (B∈[16,144] 且 S==1))。
- 强制约束：Python wrapper 仅做参数透传，所有 reshape/broadcast/cast/add/reduce/rsqrt/mul 必须发生在 PyPTO kernel 内；inplace 写回必须 kernel 内显式实现。

### 1.2 算子分类

- **类型**: Vector
- **判断依据**: 全部为逐元素与 reduce 类操作（add、mul、sum、rsqrt、cast），无 matmul/conv 调用，仅需 `pypto.set_vec_tile_shapes`。

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | elementwise | `x_add = x1 + x2` | bf16 input；建议在 fp32 域累加用于后续平方+reduce |
| 2 | dtype cast | `x_fp32 = cast(x_add, fp32)` | 提升精度，避免 bf16 平方+sum 的累积误差 |
| 3 | elementwise | `sq = x_fp32 * x_fp32` | square |
| 4 | reduction  | `s  = sum(sq, dim=-1, keepdim=True)` | reduce H 维 |
| 5 | elementwise scalar | `ms = s / H` | mean (用 sum/count 替代 mean) |
| 6 | elementwise scalar | `t  = ms + eps` | 数值稳定 |
| 7 | elementwise | `rstd_fp32 = rsqrt(t)` | 1/sqrt(t) |
| 8 | elementwise (broadcast) | `y_fp32 = x_fp32 * rstd_fp32` | broadcast `[B,S,1]` 到 `[B,S,H]` |
| 9 | dtype cast | `gamma_fp32 = cast(gamma_bf16, fp32)` | 内部全 fp32 计算 |
| 10 | elementwise (broadcast) | `y_fp32 = y_fp32 * gamma_fp32` | broadcast `[H]` 到 `[B,S,H]` |
| 11 | dtype cast | `y_bf16 = cast(y_fp32, bf16)` | 写回输出类型 |
| 12 | dtype cast | `x_add_bf16 = cast(x_fp32, bf16)` | 写回 x2 buffer 用的 bf16 |
| 13 | dtype cast | `rstd_bf16 = cast(rstd_fp32, bf16)` | 写回 rstd buffer |
| 14 | assemble (write-back) | `assemble(y_bf16, [0,0,0], x1_out)` | x1 buffer 接收 y |
| 15 | assemble (write-back) | `assemble(x_add_bf16, [0,0,0], x2_out)` | x2 buffer 接收 x_add |
| 16 | assemble (write-back) | `assemble(rstd_bf16, [0,0,0], rstd_out)` | rstd 独立输出 buffer |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | x1 + x2 | `pypto.add(x1, x2)` 或 `x1 + x2` | direct | ✓ (BF16/FP32 支持，Shape 1-4 维) |
| 2 | cast bf16→fp32 | `pypto.cast(t, pypto.DT_FP32)` | direct | ✓ (BF16↔FP32 支持) |
| 3 | x*x | `pypto.mul(x, x)` 或 `x * x` | direct | ✓ |
| 4 | sum(_, dim=-1, keepdim=True) | `pypto.sum(t, dim=-1, keepdim=True)` | direct | ✓ (FP32/BF16 支持，1-4 维) |
| 5 | / H | `t / H` | direct | ✓ scalar broadcast |
| 6 | + eps | `t + eps` | direct | ✓ scalar broadcast |
| 7 | rsqrt | `pypto.rsqrt(t)` | direct | ✓ FP32/FP16/BF16 |
| 8/10 | broadcast mul | `pypto.mul(a, b)` | direct（broadcast 由广播规则自动处理） | ✓ "支持多维度广播到相同形状" |
| 9 | cast bf16→fp32 | `pypto.cast(gamma, pypto.DT_FP32)` | direct | ✓ |
| 11/12/13 | cast fp32→bf16 | `pypto.cast(t, pypto.DT_BF16)` | direct | ✓ FP32→BF16 默认 CAST_RINT |
| 14/15/16 | inplace 写回 | `pypto.assemble(src, offsets, out)` | direct | ✓ 把内核计算写入声明为输出参数的同一 GM buffer |
| - | 备选融合 | `pypto.rms_norm(input, gamma, eps)` | direct（仅 y） | ⚠ 不返回 rstd，且不支持复用 x_add；不能直接满足 spec |

注：`pypto.rms_norm` 单算子可用于 y 的计算路径，但**无法**返回 rstd 中间结果，也**无法**返回 add 结果，因此不能直接用，需要走分解路径。可在性能调优阶段评估"y 路径用 pypto.rms_norm + 旁路再计算 rstd"是否更快。

### 3.2 Substitute 配方

```
mean → sum / hidden_size       (PyPTO 没有独立 mean，与 layer_norm 示例一致)
broadcast gamma [H] → [B,S,H]  (mul 的广播规则自动处理；TileShape 与输出对齐)
inplace 写回                  (使用 assemble 把计算结果写入声明为输出参数的输入 buffer；
                                 inplace 共享 buffer 由 wrapper 在调 kernel 时把同一
                                 torch.Tensor 同时作为输入和输出参数传入)
```

---

## 4. 约束检查

### 4.1 入口约束 (`pypto.from_torch`)

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/FP64/INT*/UINT*/BOOL | bfloat16 | ✓ |
| shape | 非空 Tensor | [B,S,H], [H] 均非空 | ✓ |
| contiguous | 必须连续 | wrapper 不做 contiguous，需调用方保证或 spec 明示要求；测试输入 `torch.randn` 默认连续 | ✓（前置条件） |
| 维度数 | 1-4 维 | x1/x2 3 维, gamma 1 维 | ✓ |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| `pypto.add` | dtype | FP16/BF16/INT16/INT32/FP32 | ✓ BF16 入；FP32 中间 |
| `pypto.add` | shape 维度 | 1-4 维 | ✓ 3 维 |
| `pypto.mul` | dtype | FP16/BF16/INT16/INT32/FP32 | ✓ |
| `pypto.mul` | broadcast | 支持多维度广播 | ✓ ([B,S,1] * [B,S,H]，[H] * [B,S,H]) |
| `pypto.sum` | dtype | FP32/BF16/INT32/INT16 | ✓ FP32 路径 |
| `pypto.sum` | TileShape ≤ 64KB | tile 内 elements*sizeof(dtype) ≤ 65536 | ⚠ H=7168 单元素 fp32=4B 即 28672B/行；选 (1, 7168) 单行 ≈28KB ✓；多行如 (4, 7168)=112KB ✗，所以 reduce 阶段每个 vec tile 行数受限 |
| `pypto.sum` | 尾轴 32B 对齐 | 7168*4B=28672B ✓ FP32；7168*2B=14336B ✓ BF16 | ✓ 7168 是 8/16 倍数 |
| `pypto.rsqrt` | dtype | FP32/FP16/BF16 | ✓ |
| `pypto.cast` | BF16↔FP32 | 双向支持 | ✓ |
| `pypto.assemble` | 任意 dtype | PyPTO 全部 dtype | ✓ |

### 4.3 Tiling 约束

- `set_vec_tile_shapes`: 每维 > 0，最多 4 维。
- `pypto.sum` 还要求 **TileShape ≤ 64KB** 且**尾轴 32B 对齐**。
- 单行 fp32 `[1, H=7168]` = 28672B（≤64KB ✓，32B 对齐 ✓）。
- 一个常见可行 vec tile：`(1, 1, 7168)` for [B,S,H]，或 reshape 到 `[B*S, H]` 后用 `(1, 7168)` 或 `(4, 7168)`（4*7168*4B=112KB ✗，超 64KB）。
- 因此 reduce 阶段每个 tile 的行数应使 `行数 * H * 4B ≤ 64KB`，即 行数 ≤ 2（fp32）；bf16 路径行数 ≤ 4。
- 对于纯逐元素阶段（无 reduce），TileShape 约束放宽，可以更大行数提高吞吐，但需在调用 reduce API 之前重新 set。

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes(...)` |

**推荐 tile 策略（H=7168 不切，沿 B、S 切）**：

1. 把 `[B,S,H]` 在 kernel 内 `reshape` 成 `[B*S, H]`，或保持 3 维都可。沿 H 不切（H 必须整片以保证沿 -1 reduce 正确）。
2. 沿 B*S 维切 tile，遍历 tile 完成 add → square → reduce → rsqrt → mul gamma → cast → assemble。
3. 推荐 tile 大小：reduce 阶段 fp32 路径 `(1, 7168)` 或 `(2, 7168)`；逐元素阶段可放大到 `(4, 7168)`（bf16）。
4. 由于动态 shape：B*S 总长度未知，需用 `pypto.loop` 或 `pypto.loop_unroll` 沿 B*S 维循环切片，每片用固定 tile shape 编译。

**reduce + 动态 shape 兼容性**：`pypto.sum` 接受 1-4 维 tensor，dim 是常量（-1），即使 B、S 是 DYNAMIC，本算子的 reduce 是沿最后一维（H 是 STATIC=7168），编译期 concrete，因此 **不会触发 "invalid shape value: -1" 类型的报错**。但是建议参考 `models/deepseek_v4/hc_pre_impl.py` 的写法：先用 `pypto.view` 切出固定大小 tile（含 STATIC 维度的 view），再对 view 调 reduce，避免任何 DYNAMIC 维度直接进 reduce。

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py`（`rmsnorm_golden`/`rms_norm_core`/`rms_norm_kernel`） | examples | 高 | 高 | RMSNorm 计算骨架（square→sum→sqrt→div→mul gamma）；通过 `pypto.assemble(out, [0,0], output)` 写回；wrapper 仅传 torch tensor + jit kernel |
| `docs/tutorials/distributed/matmul_allreduce_rmsnorm.md`（`matmul_allreduce_add_rmsnorm_kernel`） | docs/tutorial | 极高 | 高 | **逐步骤完全对应本算子的 add → rmsnorm 段**：fp32 提升 / 残差 add / `mul→sum→add eps→sqrt→div`；同时返回 `out_tensor` 与 `residual_out` 的双输出模式；显式展示 inplace 写回（`out_tensor[bs:] = ...`、`residual_out[bs:] = ...`） |
| `models/deepseek_v4/hc_pre_impl.py`（`rms_norm_denom` + 主 kernel） | models/production | 中 | 高 | rms_denom helper；动态 shape 下 `pypto.loop_unroll` + `pypto.view` + `pypto.assemble` 范式；多输出 buffer (`y, post, comb`) 通过单 kernel 写回；torch.library 注册 wrapper（透传，不做预处理） |

**推荐首选**：`docs/tutorials/distributed/matmul_allreduce_rmsnorm.md`（业务相同：add+rmsnorm）+ `models/deepseek_v4/hc_pre_impl.py`（生产级动态 shape 模式与多输出写回）。

### 6.2 可复用模式

- **API 调用模式**：
  - `x_fp32 = pypto.cast(x_bf16, pypto.DT_FP32)` → 在 fp32 域做 add/square/sum/rsqrt → `cast(_, DT_BF16)` 回写。
  - 用 `pypto.sum(sq, -1, True)` + `mean = sum / H` 替代 mean。
  - `rstd = pypto.rsqrt(mean + eps)`。
  - `y = x_fp32 * rstd * gamma_fp32`（broadcast 自动）。
  - 写回：`pypto.assemble(y_bf16, [b_idx, 0, 0], x1_out)` —— 把声明为输出参数的 x1 buffer 直接当作目标。
- **Tiling 策略**：H 不切，沿 B*S 切；reduce 阶段 fp32 行数 ≤ 2；逐元素阶段可放大。
- **Loop 结构**：`pypto.loop` / `pypto.loop_unroll` 沿 B 或 (B*S) 维切；每个 tile 内顺序执行 add→square→reduce→rsqrt→mul→cast→assemble。
- **边界处理**：B*S 较小（如 1024）时减小 unroll；S==1 大 B 形态可视作 [B, H]，等价处理。

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| 输出数量 | layer_norm 示例输出 1 个；matmul_allreduce_rmsnorm 输出 2 个；hc_pre 输出 3 个 | 输出 3 个（且其中两个 inplace 写回输入） | 仿 hc_pre：把 x1_out, x2_out, rstd_out 都作为 kernel 形参，wrapper 中将 x1/x2 同时作为输入与输出 |
| Inplace buffer aliasing | layer_norm 示例使用 `torch.empty(shape)` 创建独立输出 | 用户硬约束：x1, x2 buffer 必须复用 | wrapper 中调用 kernel 时传入相同的 torch.Tensor 同时作为输入与输出形参；rstd 用 `torch.empty(...)` 新建 |
| Wrapper 预处理 | layer_norm 示例 wrapper 只构造 out 张量；hc_pre wrapper 也仅构造输出 | 用户硬约束：wrapper 仅做参数透传，不得 reshape/broadcast/cast | 严格遵守；reshape/broadcast 全部在 kernel 内 |
| Gamma broadcast | matmul_allreduce_rmsnorm 用 `pypto.reshape(gamma, [1, H], inplace=True)` 然后 mul 自动广播 | 必须 kernel 内做 | 在 kernel 入口对 gamma 做 `pypto.reshape(gamma, [1,1,H], inplace=True)` 或依靠 mul 的广播 |
| 算子命名"inplace" | 示例没有真正的 inplace 语义算子，是新建输出 | 必须 inplace（PyPTO kernel 仅向声明为输出的形参写回，由 wrapper 用同一 buffer 复用实现 inplace 语义） | 见下文风险 |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无 | 所有所需 API 均存在并满足约束 | 走分解路径实现 |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| Inplace 语义实现机制 | PyPTO kernel 没有"标记输入为输出"的专用 API。实现方式是：在 `@pypto.frontend.jit` kernel 签名中显式声明 `x1_out: pypto.Tensor(...)` 和 `x2_out: pypto.Tensor(...)`；在 wrapper 中（torch 侧）把同一个 torch.Tensor 同时作为 `x1`（输入）和 `x1_out`（输出）传入 kernel。kernel 内通过 `pypto.assemble(...) → x1_out` 写入。是否真正"零拷贝复用同一段 GM"取决于 PyPTO 编译器的别名分析；可能会读出旧值再写新值（语义 OK）但占用一份额外中间 UB。需在 stage 5/6 验证内存占用与精度。 |
| TileShape ≤ 64KB（reduce） | H=7168 + fp32=4B 单行 28KB；reduce tile 行数最多 2。逐元素阶段无此限制可以放大 tile 提性能 |
| 尾轴 32B 对齐 | H=7168，bf16=14336B，fp32=28672B，均 32B 对齐 ✓ |
| 动态 B、S 与编译期形状 | 通过 `pypto.loop`/`loop_unroll` 切片到 STATIC tile 后再调 reduce，规避 DYNAMIC 维进 reduce |
| `pypto.rms_norm` 不可直用 | 仅返回 y，不返回 rstd；且不接受预先 add 的输入也不输出 add 中间结果。必须走分解路径 |
| 数值精度 | bf16 直接 square+sum 易累积误差；建议在 fp32 域累加。matmul_allreduce_rmsnorm 教程也是先 cast fp32 |
| Wrapper 透传约束 | 用户禁止 wrapper 做任何 reshape/broadcast/cast/contiguous。所有 reshape gamma 与 cast 操作必须放在 kernel 内 |
| 读后写顺序（inplace） | 由于 x_add 既要写回 x2 也要参与后续 y 计算，需保证 x_add 的所有消费者（square、与 rstd*gamma 相乘）在 assemble→x2 前已"读完"，PyPTO 的图调度通过显式 SSA + assemble 端点天然满足；但开发时需注意不要在 kernel 内对 x1（输入）做读取后再用 assemble→x1，应通过中间 SSA tensor 传递 |
| 形态 [144, 1, 7168] | S=1，仍是合法 3 维 ([144,1,7168])；reduce dim=-1 不变；切 tile 沿 B 即可 |
| 性能（达到 2x） | 首跑后可考虑：合并子图（`set_pass_options(sg_set_scope=...)`）、调大 vec tile 行数（仅逐元素阶段）、`pypto.rms_norm` 替换 y 路径（仍需自算 rstd）、降低 fp32 中间 buffer 大小 |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` |
| `pypto.add` | `docs/api/operation/pypto-add.md` |
| `pypto.mul` | `docs/api/operation/pypto-mul.md` |
| `pypto.sum` | `docs/api/operation/pypto-sum.md` |
| `pypto.rsqrt` | `docs/api/operation/pypto-rsqrt.md` |
| `pypto.sqrt` | `docs/api/operation/pypto-sqrt.md` |
| `pypto.cast` | `docs/api/operation/pypto-cast.md` |
| `pypto.rms_norm`（备选） | `docs/api/operation/pypto-rms_norm.md` |
| `pypto.assemble` | `docs/api/operation/pypto-assemble.md` |
| `pypto.reshape` | `docs/api/operation/pypto-reshape.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Vec tile | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| Pass 选项 | `docs/api/config/pypto-set_pass_options.md` |
| 参考实现 1（推荐） | `docs/tutorials/distributed/matmul_allreduce_rmsnorm.md` |
| 参考实现 2 | `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` |
| 参考实现 3 | `models/deepseek_v4/hc_pre_impl.py`（`rms_norm_denom`、`hc_pre_kernel`、`npu_hc_pre`） |

---

## 9. 结论

- **可行性**: 可行（feasible）
- **主要问题**: 无阻断问题。
- **关键决策建议**：
  1. 走分解路径（add → cast fp32 → square → sum → /H → +eps → rsqrt → mul x_fp32 → mul gamma_fp32 → cast bf16），不使用 `pypto.rms_norm` 单算子（无法返回 rstd 与 add 中间结果）。
  2. Inplace 写回通过：kernel 形参显式声明 x1_out / x2_out / rstd_out，wrapper 中把同一 `torch.Tensor` 既作 x1 输入又作 x1_out 输出（x2 同理），rstd 用 `torch.empty([B,S,1], bfloat16)` 新建。
  3. 沿 B（或 B*S）维切 tile；H 不切；reduce 阶段 fp32 tile 行数 ≤ 2；逐元素阶段可放大；动态形状用 `pypto.loop_unroll` 处理。
  4. 全部 reshape / broadcast / cast / 计算都在 kernel 内完成，wrapper 仅做 `torch.library.impl` 注册和参数透传 + 新建 rstd buffer。
