---
schema_version: 1
op_name: apply_adam_w_v2
supported_dtypes: [bfloat16, float32]
dynamic_axes: ['K']
shape_constraints: {axis0: 7168, axis1: K (dynamic, 2048-24576)}
tiling_required: vec
feasibility: 可行
---

# API 探索报告

> **生成时间**: 2026-04-29

---

## 1. 概述

### 1.1 输入摘要

实现 AdamW 单步优化器更新算子 `apply_adam_w_v2`，对 [7168, K] 张量按 AdamW 公式（含 bias correction 和解耦 weight decay）原地更新 weight、m、v。weight/grad 支持 bf16/fp32，m/v 固定 fp32，中间累加器 fp32。1 轴 K 为动态轴（2048-24576），需沿 K 切分 loop。

### 1.2 算子分类

- **类型**: Vector（纯 element-wise + 标量广播 + 标量 cast，无 matmul）
- **判断依据**: 计算仅由逐元素 add/sub/mul/div/sqrt/cast 组成；无任何 cube/reduction 运算

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 0 | host scalar | bc1 = 1 - β1^t；bc2 = 1 - β2^t | host 侧用 Python 计算后作为标量传入 |
| 1 | cast (条件) | grad_f32 = cast(grad, fp32) if grad.dtype==bf16 | bf16 路径需要 |
| 2 | mul + scalar | β1 * m | 标量乘 tensor |
| 3 | mul + scalar | (1 - β1) * grad_f32 | 标量乘 tensor |
| 4 | add | m_t = ② + ③ | tensor 加 tensor |
| 5 | mul + self | grad_sq = grad_f32 * grad_f32 | 平方 |
| 6 | mul + scalar | β2 * v | 标量乘 |
| 7 | mul + scalar | (1 - β2) * grad_sq | 标量乘 |
| 8 | add | v_t = ⑥ + ⑦ | tensor 加 |
| 9 | div + scalar | m_hat = m_t / bc1 | 标量除 |
| 10 | div + scalar | v_hat = v_t / bc2 | 标量除 |
| 11 | sqrt | sqrt_v = sqrt(v_hat) | 逐元素 sqrt |
| 12 | add + scalar | denom = sqrt_v + ε | 标量加 |
| 13 | div | term1 = m_hat / denom | tensor 除 tensor |
| 14 | mul + scalar | term2 = λ * weight_f32 | 标量乘 |
| 15 | add | update = term1 + term2 | tensor 加 |
| 16 | mul + scalar / sub | w_new = weight_f32 - η * update | 标量乘 + 减 |
| 17 | cast (条件) | w_new_bf16 = cast(w_new, bf16) if weight.dtype==bf16 | 写回前 cast |
| 18 | assemble | 将 w_new / m_t / v_t 写回 weight/m/v | 三处 in-place 写回 |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| 1 | cast bf16↔fp32 | `pypto.cast(t, pypto.DT_FP32 / DT_BF16)` | direct | ✓ |
| 2-3,6-7,14,16 | scalar * tensor | `pypto.mul(tensor, scalar)` | direct | ✓ |
| 4,8,12,15 | tensor + tensor / scalar | `pypto.add(a, b)` | direct | ✓ |
| 5 | grad² | `pypto.mul(grad, grad)`（无 square 时退化） | direct | ✓ |
| 9-10 | tensor / scalar | `pypto.div(tensor, scalar)` | direct | ✓ |
| 11 | sqrt | `pypto.sqrt(tensor)` | direct | ✓ |
| 13 | tensor / tensor | `pypto.div(a, b)` | direct | ✓ |
| 16 | tensor - tensor | `pypto.sub(a, b)` | direct | ✓ |
| 18 | 写回 | `pypto.assemble(tile, [offset...], output)` 或 `output[:] = ...` | direct | ✓ |
| host | 1-β^t | Python 内置 `**`（host 侧计算 bc1/bc2 后作为 float 标量传入 kernel） | direct | ✓ |

### 3.2 Substitute 配方

无需 substitute；所有原子操作均有直接 API 支持。

> 备注：若 `pypto.div` 对 fp32 精度不足，可改用 `1.0 / bc1` 在 host 计算后用 `mul` 替代（rsqrt 类似适用于 v_hat^(-1/2)）。

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP16/BF16/FP32/INT*/BOOL | bf16 / fp32 (weight/grad), fp32 (m/v) | ✓ |
| 非空 tensor | 必须 | [7168, K], K≥2048 | ✓ |
| contiguous | 必须 | 默认 PyTorch 连续 | ✓ |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| `pypto.add/sub/mul/div` | dtype | FP16/BF16/FP32 | ✓ (中间路径全 fp32) |
| `pypto.add/sub/mul/div` | shape | 1-4 维，支持广播 | ✓ (2 维) |
| `pypto.sqrt` | dtype | FP16/BF16/FP32 | ✓ |
| `pypto.cast` | 支持矩阵 | FP32↔BF16 / FP32↔FP16 | ✓ |
| `set_vec_tile_shapes` | 每维 > 0，最多 4 维 | 2 维 (m1, n1) | ✓ |
| 动态 shape 兼容性 | 计算 API 接受含 DYNAMIC 的 tensor（loop tile 后变 concrete） | 经 view 切片后为 concrete tile | ✓ |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| Vector | `pypto.set_vec_tile_shapes(m_tile, n_tile)` |

**Tile 选择建议**:
- 0 轴固定 7168，可整切（7168 = 2^10 × 7 = 1024 × 7 = 256 × 28 = 128 × 56）
- 1 轴 K 动态，需在 loop 内通过 view + valid_shape 处理尾块
- 初步建议: `set_vec_tile_shapes(1, 1024)` 或 `set_vec_tile_shapes(1, 2048)`，按 K 沿轴切分 loop；具体由 Stage 4 design 推导

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `examples/01_beginner/transform/add_scalar_loop_view_assemble.py` | examples | 高 | 高 | loop + view + assemble 模板，标量+tensor 计算 |
| `examples/02_intermediate/controlflow/loop/loop.py` | examples | 高 | 高 | 动态轴 loop tiling + valid_shape 边界处理 |
| `examples/02_intermediate/controlflow/others/dynamic.py` | examples | 高 | 高 | DYNAMIC 维度声明 + tile size 推导 |
| `examples/01_beginner/compute/elementwise_ops.py` | examples | 高 | 高 | scalar 标量参数广播；`add(t, s)` / `mul(a,b,alpha=)` |
| `examples/01_beginner/transform/transform_ops.py` | examples | 高 | 高 | bf16↔fp32 cast 模板（含 dtype 映射字典） |
| `models/qat/qat_impl.py` | models | 中 | 高 | 多输入多输出 + 混合精度 + 标量广播 + multi-assemble |
| `models/deepseek_v4/compressor_impl.py` | models | 中 | 高 | 嵌套 loop + 动态 tile + state in-place 累加 |
| `examples/02_intermediate/basic_nn/ffn/ffn_module.py` | examples | 中 | 高 | mixed-precision 计算管线（cast→fp32→compute→cast back） |

**首选**: `add_scalar_loop_view_assemble.py` + `dynamic.py` + `transform_ops.py` 组合，分别覆盖 loop 模板、动态轴语义、混合精度 cast。

### 6.2 可复用模式

- **API 调用模式**: `pypto.cast` 入/出 + `pypto.mul/add/sub/div/sqrt` 中间 fp32 计算；标量参数直接以 Python `float` 传入 mul/add（无需 `pypto.full`）。
- **Tiling 策略**: 沿 K（1 轴）切 loop，0 轴 7168 不切；`set_vec_tile_shapes(1, n_tile)` 在 loop 体或 kernel 头部一次性配置。
- **Loop 结构**: 单层 loop。`k_loop = ceil(K / tile_k)`；用 `pypto.view(t, [7168, tile_k], [0, k_offset], valid_shape=[7168, valid_k])` 取 tile，计算后 `pypto.assemble(out_tile, [0, k_offset], output)` 写回。
- **边界处理**: 尾块 `valid_k = K - k_offset`，通过 `valid_shape` 让 PyPTO 自动处理；不需要显式 padding。
- **In-place 写回**: weight/m/v 三个输出独立 assemble；PyPTO kernel 签名将三者声明为输出 tensor，golden 在 host 端做 in-place 等价（torch 上原地更新或返回新值再 copy_）。

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| 输出数 | 单输出 (assemble 1 次) | 三输出 (weight/m/v) | 三次 `pypto.assemble`，对应 3 个输出 tensor |
| dtype 路径 | 单一 dtype | weight/grad 双路径 (bf16/fp32)；m/v 固定 fp32 | 在 kernel 内用 `if grad.dtype == BF16` 分支 cast；或在 host 提前 cast |
| 标量参数数量 | 1-2 个 | 6+ (β1, 1-β1, β2, 1-β2, η, λ, ε, bc1, bc2) | host 侧预计算 `1-β1`、`1-β2`、`bc1`、`bc2`，作为 float 标量参数传入 |
| Bias correction 标量来源 | 无 | 需 host 计算 β^t | host 侧 Python `**` 运算后传入；避免 device 端 pow + pow_i32 | 

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无 | — | — |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| bf16 精度退化 | weight/grad 在 bf16 路径下，cast→fp32 计算→cast→bf16 必须严格执行；`set_satmode` 关注溢出 |
| eps 与 bc1/bc2 量级 | t 较大时 bc1/bc2 趋近 1；t=1 时 (1-β2^1)=1-0.999=0.001 较小，注意 fp32 精度 |
| 三输出 in-place | weight/m/v 必须在同一 kernel 中完成，避免读写顺序导致依赖错误（先用 m 再覆盖 m） |
| 动态 K 兼容 | view + valid_shape 必须使用，避免直接对 DYNAMIC 维度做 concrete 计算 |
| set_vec_tile_shapes 时机 | 通常在 kernel 入口设定一次；若需 per-loop 调整需谨慎 |
| host 标量 dtype | Python float 传入时 PyPTO 视为 fp32；若 weight 为 fp32 路径时无碍；bf16 路径下 cast 自动处理 |
| 7168 是否需切分 | 由 Stage 4 决定；初步认为 0 轴可不切（tile m=1 需评估 vec UB 占用） |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API 存在性 | `docs/api/operation/index.md` |
| add | `docs/api/operation/pypto-add.md` |
| sub | `docs/api/operation/pypto-sub.md` |
| mul | `docs/api/operation/pypto-mul.md` |
| div | `docs/api/operation/pypto-div.md` |
| sqrt | `docs/api/operation/pypto-sqrt.md` |
| cast | `docs/api/operation/pypto-cast.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| Vector Tiling | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| DataType | `docs/api/datatype/DataType.md` |
| TileOpFormat | `docs/api/datatype/TileOpFormat.md` |
| 动态 shape | `docs/api/pypto-DYNAMIC.md` |
| Loop | `docs/api/controlflow/pypto-loop.md` |
| 参考: scalar+loop+assemble | `examples/01_beginner/transform/add_scalar_loop_view_assemble.py` |
| 参考: 动态 loop tile | `examples/02_intermediate/controlflow/loop/loop.py` |
| 参考: DYNAMIC 维度 | `examples/02_intermediate/controlflow/others/dynamic.py` |
| 参考: cast 模板 | `examples/01_beginner/transform/transform_ops.py` |
| 参考: 多输出 + 混合精度 | `models/qat/qat_impl.py` |
| 参考: 嵌套 loop + 动态 tile | `models/deepseek_v4/compressor_impl.py` |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**: 无阻断；实现需重点关注：(1) bf16/fp32 双路径的 cast 边界；(2) host 侧 bias correction 标量预计算；(3) 三输出 in-place 写回顺序；(4) 动态 K 轴的 loop + view + valid_shape 模板。
- **环境要求**（用户提供）: 运行算子前必须 `cd /mnt/workspace/gitCode/cann/pypto && source env_setup.sh`，将在 Stage 5 首次执行测试时落实。
