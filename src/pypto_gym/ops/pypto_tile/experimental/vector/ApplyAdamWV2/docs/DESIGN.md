---
schema_version: "2.1"
op_name: "apply_adam_w_v2"
status: draft
last_updated: "2026-04-29"

# 关键接口契约（详细形参在 §2 写明）
compute_kind: "vector"
dtypes: ["bf16", "fp32"]
dynamic_axes: ["K"]
precision: { rtol: 7.8125e-3, atol: 1e-4 }
---

# apply_adam_w_v2 设计方案

## 0. 背景与目标

实现 AdamW 单步优化器更新算子。给定 `weight`、`grad`（bf16 或 fp32）、`m`、`v`（fp32），按 AdamW 公式带 bias correction 更新三者。Shape 固定为 `[7168, K]`，K 为动态轴 ∈ [2048, 24576]。

输入参考：
- `SPEC.md` — 算法、shape、精度要求
- `API_REPORT.md` — 公式分解（§2）、API 映射（§3.1）、参考示例（§6）
- `apply_adam_w_v2_golden.py` — torch 参考实现

设计目标：
1. 在 PyPTO 上完整复现 AdamW 单步公式，精度满足 atol=1e-4、rtol=7.8125e-3。
2. 三个输出（weight/m/v）通过多次 assemble 完成 in-place 写回。
3. 动态轴 K 通过 `pypto.loop` + view/assemble + valid_shape 处理（含尾块）。
4. 支持 bf16/fp32 双精度路径，bf16 路径在 kernel 入口 cast→fp32，写回 weight 前 cast 回 bf16。

---

## 1. 计算图与精度路由

### 1.1 API 调用序列

> 序列基于 API_REPORT §2 的公式分解；所有中间 tensor 在 fp32 上计算。各步骤 shape 均为 tile 视角下的 `[m_tile, n_tile]`（默认 `[1, 1024]`）。

| 步骤 | 操作 | PyPTO API | 输入 dtype | 输出 dtype | 输出 shape | 备注 |
|------|------|-----------|------------|------------|-----------|------|
| 0 (host) | bc1=1-β1^t；bc2=1-β2^t；one_m_b1=1-β1；one_m_b2=1-β2 | Python `**` / 算术 | python float | python float | scalar | 在 wrapper 中预计算，传入 kernel |
| 1a | view 取 weight tile | `pypto.view(weight, [m,n], [0,k_off], valid_shape=[7168,vk])` | bf16/fp32 | bf16/fp32 | [m,n] | loop 内每次切 |
| 1b | view 取 grad tile | `pypto.view(grad, ...)` | bf16/fp32 | bf16/fp32 | [m,n] | 同上 |
| 1c | view 取 m tile | `pypto.view(m, ...)` | fp32 | fp32 | [m,n] | 同上 |
| 1d | view 取 v tile | `pypto.view(v, ...)` | fp32 | fp32 | [m,n] | 同上 |
| 2a | cast weight 至 fp32（bf16 路径） | `pypto.cast(w_tile, DT_FP32)` | bf16 | fp32 | [m,n] | fp32 路径下跳过 |
| 2b | cast grad 至 fp32（bf16 路径） | `pypto.cast(g_tile, DT_FP32)` | bf16 | fp32 | [m,n] | fp32 路径下跳过 |
| 3 | β1 * m_tile | `pypto.mul(m_tile, beta1)` | fp32 | fp32 | [m,n] | 标量乘 |
| 4 | (1-β1) * grad_f32 | `pypto.mul(grad_f32, one_m_b1)` | fp32 | fp32 | [m,n] | 标量乘 |
| 5 | m_new = ③ + ④ | `pypto.add(a, b)` | fp32 | fp32 | [m,n] | tensor + tensor |
| 6 | grad_sq = grad_f32 * grad_f32 | `pypto.mul(grad_f32, grad_f32)` | fp32 | fp32 | [m,n] | 平方 |
| 7 | β2 * v_tile | `pypto.mul(v_tile, beta2)` | fp32 | fp32 | [m,n] |  |
| 8 | (1-β2) * grad_sq | `pypto.mul(grad_sq, one_m_b2)` | fp32 | fp32 | [m,n] |  |
| 9 | v_new = ⑦ + ⑧ | `pypto.add(a, b)` | fp32 | fp32 | [m,n] |  |
| 10 | m_hat = m_new / bc1 | `pypto.div(m_new, bc1)` | fp32 | fp32 | [m,n] | 标量除；可改 mul(m_new, 1/bc1) |
| 11 | v_hat = v_new / bc2 | `pypto.div(v_new, bc2)` | fp32 | fp32 | [m,n] |  |
| 12 | sqrt_v = sqrt(v_hat) | `pypto.sqrt(v_hat)` | fp32 | fp32 | [m,n] |  |
| 13 | denom = sqrt_v + eps | `pypto.add(sqrt_v, eps)` | fp32 | fp32 | [m,n] | 标量加 |
| 14 | term1 = m_hat / denom | `pypto.div(m_hat, denom)` | fp32 | fp32 | [m,n] | tensor / tensor |
| 15 | term2 = λ * weight_f32 | `pypto.mul(w_f32, weight_decay)` | fp32 | fp32 | [m,n] | 标量乘 |
| 16 | update = term1 + term2 | `pypto.add(term1, term2)` | fp32 | fp32 | [m,n] |  |
| 17 | scaled = lr * update | `pypto.mul(update, lr)` | fp32 | fp32 | [m,n] | 标量乘 |
| 18 | w_new_f32 = w_f32 - scaled | `pypto.sub(w_f32, scaled)` | fp32 | fp32 | [m,n] | tensor - tensor |
| 19 | cast 回 bf16（bf16 路径） | `pypto.cast(w_new_f32, DT_BF16)` | fp32 | bf16 | [m,n] | fp32 路径跳过 |
| 20a | 写回 weight | `pypto.assemble(w_out_tile, [0,k_off], weight_out)` | bf16/fp32 | — | — | 三次写回之一 |
| 20b | 写回 m | `pypto.assemble(m_new, [0,k_off], m_out)` | fp32 | — | — |  |
| 20c | 写回 v | `pypto.assemble(v_new, [0,k_off], v_out)` | fp32 | — | — |  |

### 1.2 精度路由

```text
weight (bf16/fp32) ─► [view] ─► [cast→fp32 (仅 bf16 路径)] ─► fp32 计算管线 ─► [cast→bf16 (仅 bf16 路径)] ─► [assemble] ─► weight_out

grad   (bf16/fp32) ─► [view] ─► [cast→fp32 (仅 bf16 路径)] ─► fp32 计算管线（仅消费，不写回）

m (fp32) ─► [view] ─► fp32 计算管线 ─► m_new (fp32) ─► [assemble] ─► m_out
v (fp32) ─► [view] ─► fp32 计算管线 ─► v_new (fp32) ─► [assemble] ─► v_out

host: β1, β2, lr, weight_decay, eps, step ─► host 预计算：bc1, bc2, one_m_b1, one_m_b2, one_minus_beta1=…
```

| 转换位置 | 转换方向 | 原因 |
|---------|---------|------|
| 步骤 2a/2b（kernel 入口，仅 bf16 路径） | BF16 → FP32 | 中间 add/mul/div/sqrt 全部需在 fp32 累加，避免 bf16 精度损失（满足 atol=1e-4） |
| 步骤 19（写回 weight 前，仅 bf16 路径） | FP32 → BF16 | 与输入 weight dtype 保持一致 |
| host 端 bias correction | python float | β^t 在 device 端无直接 API（需 pow_i32 + pow），由 host 一次性算好作为 float 传入更稳更省 |

m_new、v_new 始终为 fp32，无需 cast。

### 1.3 替代方案（已排除）

| 替代方案 | 排除原因 |
|---------|---------|
| 在 device 端用 `pypto.pow` 计算 β^t | 多此一举：t 是 host 标量；host `**` 一次即可，避免 device 端 pow 链路与潜在精度问题 |
| 用 `rsqrt(v_hat)` 替代 `sqrt+div` | 需要 (sqrt_v + eps) 形式，rsqrt 不带加 eps，组合后并不更简单；保留 sqrt+add+div 与 golden 一致 |
| 把 weight 强制 cast 为 fp32 全程保留 | 浪费 UB；bf16 入参的 cast→fp32 必须发生，但只在 tile 范围内进行，不提升整体 dtype |
| 用 `pypto.full` 构造广播标量再 mul | 直接传 python float 给 `pypto.mul/add/div` 即可，PyPTO 自动处理；构造常量更耗 UB |
| 用 `output[:] = result` 写回 | 在三输出 + 多 tile + 动态 K 场景下 `assemble + offset` 语义最清晰、与官方示例 `add_scalar_loop_view_assemble.py` 模板一致 |

---

## 2. 数据规格

### 2.1 Kernel 函数签名

bf16 与 fp32 路径采用不同的 jit kernel（避免在 jit 体内做 dtype 分支）；wrapper 在 host 端做 dispatch。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def apply_adam_w_v2_kernel_fp32(
    weight: pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_FP32),    # in/out
    grad:   pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_FP32),    # in
    m:      pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_FP32),    # in/out
    v:      pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_FP32),    # in/out
    weight_out: pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_FP32),
    m_out:      pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_FP32),
    v_out:      pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_FP32),
    beta1: float, one_m_b1: float,
    beta2: float, one_m_b2: float,
    bc1: float, bc2: float,
    lr: float, weight_decay: float, eps: float,
):
    ...

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def apply_adam_w_v2_kernel_bf16(
    weight: pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_BF16),
    grad:   pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_BF16),
    m:      pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_FP32),
    v:      pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_FP32),
    weight_out: pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_BF16),
    m_out:      pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_FP32),
    v_out:      pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_FP32),
    beta1: float, one_m_b1: float,
    beta2: float, one_m_b2: float,
    bc1: float, bc2: float,
    lr: float, weight_decay: float, eps: float,
):
    ...
```

Host wrapper：

```python
def apply_adam_w_v2_wrapper(
    weight: torch.Tensor, grad: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
    beta1: float, beta2: float, lr: float, weight_decay: float, eps: float, step: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 1. host 标量预计算
    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    one_m_b1 = 1.0 - beta1
    one_m_b2 = 1.0 - beta2

    # 2. 上设备
    w_npu = pypto.from_torch(weight)
    g_npu = pypto.from_torch(grad)
    m_npu = pypto.from_torch(m)
    v_npu = pypto.from_torch(v)
    w_out = pypto.empty_like(w_npu)
    m_out = pypto.empty_like(m_npu)
    v_out = pypto.empty_like(v_npu)

    # 3. dtype dispatch
    if weight.dtype == torch.bfloat16:
        kernel = apply_adam_w_v2_kernel_bf16
    elif weight.dtype == torch.float32:
        kernel = apply_adam_w_v2_kernel_fp32
    else:
        raise TypeError(f"unsupported weight dtype: {weight.dtype}")

    kernel(w_npu, g_npu, m_npu, v_npu, w_out, m_out, v_out,
           beta1, one_m_b1, beta2, one_m_b2, bc1, bc2, lr, weight_decay, eps)

    return pypto.to_torch(w_out), pypto.to_torch(m_out), pypto.to_torch(v_out)
```

### 2.2 动态轴分析

| 维度名 | 是否动态 | 取值范围 / 常量 | 标注方式 |
|--------|---------|-----------------|---------|
| 0 轴 (M=7168) | 否 | 编译期常量 7168 | 字面量 `7168` |
| 1 轴 (K) | 是 | 运行期 [2048, 24576] | `pypto.DYNAMIC` |

### 2.3 值类型分析（避免 SymbolicScalar 误用）

| 变量 | 来源 | 类型 | 注意事项 |
|------|------|------|---------|
| `K = weight.shape[1]` | 动态轴 | SymbolicScalar | 仅可参与 `pypto.loop`、`min/max`、view/valid_shape 计算；不可做 `**`、Python `if`、list 下标、`range` |
| `7168` | 静态轴 | Python int | 常规使用 |
| `n_tile = 1024` | 设计常量 | Python int | 可做 list 下标、Python `range`、与 SymbolicScalar 做 `//`、`-`、`min` |
| `k_idx` | `pypto.loop` 索引 | SymbolicScalar | 用作 view offset；`k_off = k_idx * n_tile`，`valid_k = (K - k_off).min(n_tile)` |
| `beta1`, `bc1` 等 | wrapper 预计算 | Python float | 直接作为 jit 标量参数；进入 `pypto.mul/div/add` 时被识别为标量 |

不存在需要 `pypto.Element` 包装的标量（所有标量在 host 已预计算且直接接受 python float）。

---

## 3. Tiling 策略

### 3.1 算子类型

Vector（无 matmul、无 reduce）。仅需 `pypto.set_vec_tile_shapes`。

### 3.2 Tiling 推导

**同时驻留 UB 的 Tensor（默认 `tile = [1, 1024]`，bf16 路径最坏情况）**：

| Tensor | 用途 | shape | dtype | 字节数 (1 tile) |
|--------|------|-------|-------|-----------------|
| w_tile | 输入 weight | [1, 1024] | bf16 | 2 048 |
| g_tile | 输入 grad | [1, 1024] | bf16 | 2 048 |
| w_f32  | cast 后 weight | [1, 1024] | fp32 | 4 096 |
| g_f32  | cast 后 grad | [1, 1024] | fp32 | 4 096 |
| m_tile | 输入 m | [1, 1024] | fp32 | 4 096 |
| v_tile | 输入 v | [1, 1024] | fp32 | 4 096 |
| beta1_m | β1*m | [1, 1024] | fp32 | 4 096 |
| one_b1_g | (1-β1)*g | [1, 1024] | fp32 | 4 096 |
| m_new | m 更新 | [1, 1024] | fp32 | 4 096 |
| grad_sq | g*g | [1, 1024] | fp32 | 4 096 |
| beta2_v | β2*v | [1, 1024] | fp32 | 4 096 |
| one_b2_gs | (1-β2)*g² | [1, 1024] | fp32 | 4 096 |
| v_new | v 更新 | [1, 1024] | fp32 | 4 096 |
| m_hat | bias-corrected m | [1, 1024] | fp32 | 4 096 |
| v_hat | bias-corrected v | [1, 1024] | fp32 | 4 096 |
| sqrt_v | √v_hat | [1, 1024] | fp32 | 4 096 |
| denom | sqrt_v+eps | [1, 1024] | fp32 | 4 096 |
| term1 | m_hat/denom | [1, 1024] | fp32 | 4 096 |
| term2 | λw | [1, 1024] | fp32 | 4 096 |
| update | term1+term2 | [1, 1024] | fp32 | 4 096 |
| scaled | lr*update | [1, 1024] | fp32 | 4 096 |
| w_new_f32 | w-η*upd | [1, 1024] | fp32 | 4 096 |
| w_out_bf16 | cast 回 | [1, 1024] | bf16 | 2 048 |

合计估算：≈ 22 × 4 KB + 3 × 2 KB ≈ 94 KB（保守上界，编译器可复用 buffer 实际更低）。Atlas 800I A2 单核 UB ≈ 192 KB，留有充足余量。

**推导步骤**：
1. **尾轴对齐**：bf16 → 16 元素对齐；fp32 → 8 元素对齐。`n_tile = 1024` 同时满足两个对齐要求。
2. **UB 预算**：上面合计 < UB 容量。
3. **展开约束**：`(7168/1) × (K/1024) × tensor_count`。最坏 K=24576 → `7168 × 24 × ~25 ≈ 4.3M`，但 PyPTO 会在 loop 维度上自动 lower（loop 单步只展开 tile 内表达式），实际展开仅与 tile 内 op 数相关：≈ 25 ops × 1 = 25，远 < 18000。
4. **0 轴 m_tile = 1**：保留 m 轴一次完整覆盖（7168 全部并行处理），避免引入第二层 loop；7168 % 1 == 0 自然满足。

**最终 tile**：

```python
pypto.set_vec_tile_shapes(1, 1024)
```

### 3.3 替代方案

| 备选 tile | 否决理由 |
|-----------|---------|
| `(1, 2048)` | UB 占用接近翻倍（≈ 188 KB），单核接近容量上限，对 K=24576 切分次数减半但稳健性下降；保留为优化阶段候选 |
| `(1, 512)` | tile 太小，loop 次数翻倍（K/512），调度开销上升；初版不推荐 |
| `(7, 1024)` | 0 轴切到 7（7168/7=1024），UB 占用 ×7 必爆，否决 |
| `(8, 1024)` | 7168 % 8 = 0（=896），需要二层 loop 切 0 轴，复杂度上升而无明显收益；首版仅切 K |

> 备注：若性能阶段（pypto-op-perf-tune）需要进一步调优，可尝试 `(1, 2048)` 或在 0 轴引入 `pypto.loop(7168/n_m, ...)`；当前阶段以正确性优先。

---

## 4. Loop 与数据流

### 4.1 维度判定

| 轴 | 维度大小 | 编译期 / 运行期 | Loop 处理 |
|----|---------|----------------|----------|
| 0 (M=7168) | 编译期 7168 | 编译期已知 | 不需要 loop（tile 已覆盖）|
| 1 (K) | 运行期 [2048, 24576] | 运行期 | `pypto.loop((K + n_tile - 1) // n_tile, name="k_loop")` |

### 4.2 完整伪代码

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def apply_adam_w_v2_kernel_bf16(
    weight: pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_BF16),     # [7168, K]
    grad:   pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_BF16),     # [7168, K]
    m:      pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_FP32),     # [7168, K]
    v:      pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_FP32),     # [7168, K]
    weight_out: pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_BF16), # out
    m_out:      pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_FP32), # out
    v_out:      pypto.Tensor([7168, pypto.DYNAMIC], pypto.DT_FP32), # out
    beta1: float, one_m_b1: float,
    beta2: float, one_m_b2: float,
    bc1: float, bc2: float,
    lr: float, weight_decay: float, eps: float,
):
    M = 7168                    # Python int (静态)
    N_TILE = 1024               # Python int (设计常量)
    K = weight.shape[1]         # SymbolicScalar (动态)

    pypto.set_vec_tile_shapes(1, N_TILE)         # vec tile 配置（一次）

    # K-axis loop count (ceil division on SymbolicScalar)
    k_loops = (K + N_TILE - 1) // N_TILE         # SymbolicScalar
    for k_idx in pypto.loop(k_loops, name="k_loop"):
        k_off = k_idx * N_TILE                    # SymbolicScalar
        valid_k = (K - k_off).min(N_TILE)         # SymbolicScalar (尾块)

        # ---- view ----
        w_tile = pypto.view(weight, [M, N_TILE], [0, k_off],
                            valid_shape=[M, valid_k])    # [7168,1024] bf16
        g_tile = pypto.view(grad,   [M, N_TILE], [0, k_off],
                            valid_shape=[M, valid_k])    # bf16
        m_tile = pypto.view(m,      [M, N_TILE], [0, k_off],
                            valid_shape=[M, valid_k])    # fp32
        v_tile = pypto.view(v,      [M, N_TILE], [0, k_off],
                            valid_shape=[M, valid_k])    # fp32

        # ---- cast bf16 -> fp32 (only bf16 kernel) ----
        w_f32 = pypto.cast(w_tile, pypto.DT_FP32)        # [M,N_TILE] fp32
        g_f32 = pypto.cast(g_tile, pypto.DT_FP32)        # fp32

        # ---- moment updates (fp32) ----
        beta1_m   = pypto.mul(m_tile, beta1)              # fp32
        one_b1_g  = pypto.mul(g_f32,  one_m_b1)           # fp32
        m_new     = pypto.add(beta1_m, one_b1_g)          # fp32

        grad_sq   = pypto.mul(g_f32, g_f32)               # fp32
        beta2_v   = pypto.mul(v_tile, beta2)              # fp32
        one_b2_gs = pypto.mul(grad_sq, one_m_b2)          # fp32
        v_new     = pypto.add(beta2_v, one_b2_gs)         # fp32

        # ---- bias correction ----
        m_hat = pypto.div(m_new, bc1)                     # fp32
        v_hat = pypto.div(v_new, bc2)                     # fp32

        # ---- AdamW update ----
        sqrt_v = pypto.sqrt(v_hat)                        # fp32
        denom  = pypto.add(sqrt_v, eps)                   # fp32 (eps as scalar)
        term1  = pypto.div(m_hat, denom)                  # fp32 (tensor/tensor)
        term2  = pypto.mul(w_f32, weight_decay)           # fp32
        update = pypto.add(term1, term2)                  # fp32
        scaled = pypto.mul(update, lr)                    # fp32
        w_new_f32 = pypto.sub(w_f32, scaled)              # fp32

        # ---- cast back ----
        w_out_tile = pypto.cast(w_new_f32, pypto.DT_BF16) # bf16

        # ---- assemble (3 outputs) ----
        pypto.assemble(w_out_tile, [0, k_off], weight_out)   # bf16 写回
        pypto.assemble(m_new,      [0, k_off], m_out)        # fp32 写回
        pypto.assemble(v_new,      [0, k_off], v_out)        # fp32 写回
```

> fp32 路径与上面相同，去掉 `cast` 即可（步骤 2a/2b/19）。两个 kernel 共用大部分主体，可在实现阶段抽取公共函数或使用配置参数。

### 4.3 跨迭代状态（如有）

| 状态名 | 初始化 | 更新方式 | submit_before_loop |
|--------|--------|---------|--------------------|
| 无 | — | — | 否 |

每次 K tile 内的计算完全独立（m_new、v_new 都基于 m_tile、v_tile、g_tile 局部计算后直接写回），不存在跨迭代累加。因此**不需要** `submit_before_loop=True`，也不需要 `unroll_list`（或仅在性能调优阶段引入）。

### 4.4 尾块处理

- **方案**：`valid_shape=[7168, valid_k]`，其中 `valid_k = (K - k_off).min(N_TILE)`。
- K = 2048/8192/16384/24576 都是 1024 的整数倍 → 实际不会触发尾块；但 `valid_shape` 必须保留以兼容 K=2049 等非整除场景，并满足 `pypto.DYNAMIC` 的语义要求。
- 不需要 padding：PyPTO 通过 valid_shape 自动屏蔽尾部无效元素的写回。

---

## 5. 约束自检清单

| # | 约束 | 是否满足 | 备注 |
|---|------|---------|------|
| 1 | 所有 sum 输入已转 FP32 | N/A | 算子无 reduce/sum |
| 2 | matmul 两侧 dtype 一致 | N/A | 算子无 matmul |
| 3 | TileShape 维度数 = 操作数维度数 | ✓ | tile=`[1,1024]` 与 2D tensor 一致 |
| 4 | 尾轴满足对齐 | ✓ | 1024 同时满足 fp32(8) / bf16(16) 对齐 |
| 5 | 同阶段 UB 占用 ≤ 容量 | ✓ | 估算 ≈ 94 KB < 192 KB |
| 6 | 表达式展开 < 18000 | ✓ | tile 内 ~25 ops |
| 7 | 输出经 `[:]` / `assemble` 显式写回 | ✓ | 三处 `pypto.assemble` |
| 8 | 无 view/assemble 同张量回环 | ✓ | weight/m/v 输入与 weight_out/m_out/v_out 是独立 tensor |
| 9 | 动态轴标 `pypto.DYNAMIC` | ✓ | K 轴标注；M=7168 不标 |
| 10 | 动态 loop 提供 `unroll_list` | △ | 首版不提供（无跨迭代依赖）；perf 阶段可评估 |
| 11 | 跨迭代状态用 `submit_before_loop=True` | N/A | 无跨迭代状态 |
| 12 | 尾块用 `valid_shape` 处理 | ✓ | view 显式传入 valid_shape |
| 13 | 无 SymbolicScalar 用作 `**` / list index / Python `if` | ✓ | β^t 在 host 算；ceil-div 用整数算术；`(K-k_off).min(N_TILE)` 用 SymbolicScalar 方法 |

### 开放问题（延期至 Stage 5/调优阶段）

| # | 问题 | 影响范围 | 待解决方式 |
|---|------|---------|-----------|
| O1 | tile=(1,1024) 是否最优 | 性能 | Stage 5 通过 perf-tune 对比 (1,1024) vs (1,2048) |
| O2 | bf16 与 fp32 是否合并为单 kernel + dtype 分支 | 实现复杂度 | Stage 5 评估 PyPTO 对 dtype 分支支持情况，决定是否合并 |
| O3 | weight_out/m_out/v_out 与输入是否可直接复用同一 buffer（true in-place） | 内存 | Stage 5 验证 PyPTO 是否允许同一 tensor 同时入参/出参；若不允许，wrapper 在外部 `copy_` 回原 tensor |
| O4 | unroll_list 配置 | 性能 | perf 阶段评估，与 K 取值范围匹配 |
| O5 | 是否需要切 0 轴以提升并行 | 性能 | perf 阶段评估，若 7168 单核无法饱和则引入 m 轴 loop |

---

## 6. 验证方案

### 6.1 测试配置

| 用例 | 输入 shape | dtype | 重点验证 |
|------|----------|-------|---------|
| fp32_min | weight/grad/m/v: [7168, 2048] | fp32 | P0 功能基线（fp32 路径） |
| bf16_min | weight/grad: [7168, 2048] bf16；m/v: [7168, 2048] fp32 | mixed | bf16 cast 边界、最小 K |
| bf16_mid | weight/grad: [7168, 8192] bf16；m/v: fp32 | mixed | 典型 LLM 配置（K=8192） |
| bf16_large | weight/grad: [7168, 16384] bf16；m/v: fp32 | mixed | 中等大 K，loop 切 16 轮 |
| bf16_max | weight/grad: [7168, 24576] bf16；m/v: fp32 | mixed | 最大 K，loop 切分 24 轮压力测试 |
| fp32_max | weight/grad/m/v: [7168, 24576] | fp32 | fp32 全路径最大 case |
| step_t1 | [7168, 2048], step=1 | bf16/fp32 | 1-β^1 = 1-β（最大 bias correction 影响） |
| step_t100 | [7168, 8192], step=100 | bf16 | 中等 step |
| step_t1000 | [7168, 24576], step=1000 | bf16 | 大 step（bc1/bc2≈1） |

### 6.2 精度容忍度

| dtype | rtol | atol |
|-------|------|------|
| FP32  | 7.8125e-3 | 1e-4 |
| BF16  | 7.8125e-3 | 1e-4 |

> 容差来自 SPEC.md。在 `test_apply_adam_w_v2.py` 中以 `torch.allclose(out, golden, atol=1e-4, rtol=7.8125e-3)` 比较 weight、m、v 三路输出。

### 6.3 验证流程

1. 准备：`source env_setup.sh`（位于仓库根目录），确保 NPU 环境就绪。
2. 调用 `apply_adam_w_v2_wrapper(...)` 得到 `(w_out, m_out, v_out)`。
3. 调用 `apply_adam_w_v2_golden(...)` 得到对照参考。
4. 三输出分别用 `torch.allclose` 校验，失败时打印 max abs / rel error。
5. 覆盖 SPEC §12 中的 P0 配置（fp32_min, bf16_min, bf16_mid, bf16_max）以及上面 §6.1 的边界 step 用例。

### 6.4 性能验证（perf 阶段，非 Stage 5 必须）

- 单 case 跑 100 次取平均，目标：相对首跑性能 2× 提升（SPEC §10）。
- 重点观察 K=24576 case 的耗时与 UB 利用率。

---

## 完成报告

设计状态：已收敛（含 5 项延期至 Stage 5/perf 阶段的开放问题）

迭代过程：
- 第 1 轮：API 调用链 ~21 步（含 host 预计算与 cast），cast 2 处（仅 bf16 路径）
- 第 2 轮：Tiling = vec，tile = `(1, 1024)`，UB 占用 ≈ 94 KB
- 第 3 轮：单层 K-loop（动态轴），无跨迭代依赖，三 assemble 写回
- 第 4 轮：13 项约束自检全部通过或 N/A

回退记录：无（一次收敛）

开放问题：见 §5 的 O1–O5（均不阻塞 Stage 5 实现，属于优化与边界确认事项）。
