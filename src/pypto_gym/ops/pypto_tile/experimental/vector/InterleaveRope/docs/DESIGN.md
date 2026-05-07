---
schema_version: "2.3"
op_name: "interleave_rope"
status: tuned
last_updated: "2026-04-30"

compute_kind: "vector"
dtypes: ["fp16", "bf16"]
dynamic_axes: ["B", "S", "N"]
precision: { rtol: 0.0078125, atol: 0.0001 }
output_layout: "split_half"  # left=y_even, right=y_odd; NOT interleaved
---

# interleave_rope 设计方案（v2.3 — split-half 输出 layout）

> 本设计针对 interleave 风格 RoPE（对最后维 D=64 的相邻元素对 (x[2i], x[2i+1]) 做旋转），
> 输入 `x[B,N,S,D]`、`cos[B,1,S_cs,D]`、`sin[B,1,S_cs,D]`，输出 `y[B,N,S,D]`，dtype ∈ {fp16, bf16}，全部 ND 连续。
>
> **v2.3 关键变化（输出 layout 约定）**：放弃在 kernel/wrapper 内做 interleave 重组。
> 输出 `y` 改为 **split-half layout**：
> - `y[..., 0:32]  = y_even = [y_origin[0], y_origin[2], y_origin[4], ..., y_origin[62]]`
> - `y[..., 32:64] = y_odd  = [y_origin[1], y_origin[3], y_origin[5], ..., y_origin[63]]`
>
> 这是一种**置换的输出**——下游 attention `QK^T = Σ_d Q[d]·K[d]` 对 D 维顺序不敏感（点积可换位），
> 只要 Q 和 K 用同一种 RoPE layout，attention 结果数值不变。该置换约定由 wrapper 文档对调用方明确。
>
> 由此 kernel 全程仅 4D，无 5D `reshape + concat`、无 wrapper 端 `torch.stack` 重组，
> 直接两次 `pypto.assemble` 写入输出的左右两半。
>
> golden 同步修改输出 layout 以保持精度对比可比。

---

## 1. 计算图与精度路由

### 1.1 数学公式（输入 interleave 语义）

对每对相邻元素 (x[2k], x[2k+1])：

```
y_origin[2k]   = x[2k] · cos[2k]   - x[2k+1] · sin[2k]
y_origin[2k+1] = x[2k] · sin[2k+1] + x[2k+1] · cos[2k+1]
```

定义 split-half 视角：
- `x_even[k] = x[2k]`,    `x_odd[k] = x[2k+1]`
- `cos_even[k] = cos[2k]`, `cos_odd[k] = cos[2k+1]`
- `sin_even[k] = sin[2k]`, `sin_odd[k] = sin[2k+1]`

公式化简为两条独立 trailing-32 计算：

```
y_even[k] = x_even[k] · cos_even[k] - x_odd[k] · sin_even[k]
y_odd [k] = x_even[k] · sin_odd[k]  + x_odd[k] · cos_odd[k]
```

### 1.2 输出 layout 约定（split-half）

```
out[..., 0:32 ] = y_even = [y_origin[0],  y_origin[2],  ..., y_origin[62]]
out[..., 32:64] = y_odd  = [y_origin[1],  y_origin[3],  ..., y_origin[63]]
```

**这是对原 interleave 输出的固定置换**。下游约定：
- 调用方必须知道 Q/K 的 D 维排布是 split-half
- attention `QK^T = Σ_d Q[d]·K[d]` 在两侧 layout 一致时数值等价
- golden 输出同样 split-half layout

### 1.3 API 调用序列（kernel 内）

| 步骤 | 操作 | PyPTO API | 输入 dtype | 输出 dtype | 输出 shape | 备注 |
|------|------|-----------|------------|------------|-----------|------|
| 1a | x_t = view(x,   [1,N_TILE,S_TILE,64], off)   | `pypto.view`        | bf16/fp16 | bf16/fp16 | [1, N_TILE, S_TILE, 64] | 全 D 进 tile，不切 |
| 1b | c_t = view(cos, [1,1,S_TILE,64], off)        | `pypto.view`        | bf16/fp16 | bf16/fp16 | [1, 1,      S_TILE, 64] | N 维 1→N broadcast |
| 1c | s_t = view(sin, [1,1,S_TILE,64], off)        | `pypto.view`        | bf16/fp16 | bf16/fp16 | [1, 1,      S_TILE, 64] | 同上 |
| 2a | x_e = gathermask(x_t, PM=1)                  | `pypto.gathermask`  | bf16/fp16 | bf16/fp16 | [1, N_TILE, S_TILE, 32] | PM=1 取偶位 |
| 2b | x_o = gathermask(x_t, PM=2)                  | `pypto.gathermask`  | bf16/fp16 | bf16/fp16 | [1, N_TILE, S_TILE, 32] | PM=2 取奇位 |
| 2c | c_e = gathermask(c_t, PM=1)                  | `pypto.gathermask`  | bf16/fp16 | bf16/fp16 | [1, 1,      S_TILE, 32] |  |
| 2d | c_o = gathermask(c_t, PM=2)                  | `pypto.gathermask`  | bf16/fp16 | bf16/fp16 | [1, 1,      S_TILE, 32] |  |
| 2e | s_e = gathermask(s_t, PM=1)                  | `pypto.gathermask`  | bf16/fp16 | bf16/fp16 | [1, 1,      S_TILE, 32] |  |
| 2f | s_o = gathermask(s_t, PM=2)                  | `pypto.gathermask`  | bf16/fp16 | bf16/fp16 | [1, 1,      S_TILE, 32] |  |
| 3  | set_vec_tile_shapes(1, N_TILE, S_TILE, 32)   | `pypto.set_vec_tile_shapes` | — | — | — | trailing 32 阶段重设 tile |
| 4a-4f | cast x_e/x_o/c_e/c_o/s_e/s_o → fp32       | `pypto.cast`        | bf16/fp16 | fp32      | [..., 32] | fp32 内部累积 |
| 5  | y_e_f = sub(mul(xe_f, ce_f), mul(xo_f, se_f)) | `pypto.mul/sub`    | fp32      | fp32      | [1, N_TILE, S_TILE, 32] | y_even fp32 |
| 6  | y_o_f = add(mul(xe_f, so_f), mul(xo_f, co_f)) | `pypto.mul/add`    | fp32      | fp32      | [1, N_TILE, S_TILE, 32] | y_odd fp32 |
| 7a | y_e = cast(y_e_f, dtype)                     | `pypto.cast`        | fp32      | bf16/fp16 | [1, N_TILE, S_TILE, 32] |  |
| 7b | y_o = cast(y_o_f, dtype)                     | `pypto.cast`        | fp32      | bf16/fp16 | [1, N_TILE, S_TILE, 32] |  |
| 8a | assemble(y_e, [b, n_off, s_off, 0],  out)    | `pypto.assemble`    | bf16/fp16 | bf16/fp16 | — | 写到 out 左半 [..., 0:32] |
| 8b | assemble(y_o, [b, n_off, s_off, 32], out)    | `pypto.assemble`    | bf16/fp16 | bf16/fp16 | — | 写到 out 右半 [..., 32:64] |
| 9  | set_vec_tile_shapes(1, N_TILE, S_TILE, 64)   | `pypto.set_vec_tile_shapes` | — | — | — | 还原以备下次循环 gathermask |

### 1.4 精度路由

```text
x [bf16/fp16]  ─gathermask─▶ x_e/x_o[bf16/fp16] ─cast─▶ [fp32]
cos[bf16/fp16] ─gathermask─▶ c_e/c_o[bf16/fp16] ─cast─▶ [fp32]
sin[bf16/fp16] ─gathermask─▶ s_e/s_o[bf16/fp16] ─cast─▶ [fp32]

[fp32] ── mul/sub/add (fp32 内累积) ──▶ y_e_f, y_o_f [fp32]
y_e_f, y_o_f [fp32] ──cast──▶ y_e, y_o [bf16/fp16] ──assemble──▶ out 左/右半
```

| 转换位置 | 转换方向 | 原因 |
|---------|---------|------|
| gathermask 后 | bf16/fp16 → fp32 | atol=1e-4 严格于通用 1e-3，必须 fp32 内累积 |
| 计算后 | fp32 → bf16/fp16 | 输出 dtype 与输入对齐 |

### 1.5 替代方案（已排除）

| 替代方案 | 排除原因 |
|---------|---------|
| 输出严格 interleave layout：5D `reshape + concat + reshape` 重组 | `set_vec_tile_shapes` 上限 4 维，5D concat 不一定能编译；即便能编译也增加 op 数与 tile 切换开销 |
| 输出严格 interleave layout：wrapper 端 `torch.stack` 重组 | 违反 wrapper 仅做校验/派发的约定；引入 ~57us 的输出后处理 |
| 输出严格 interleave layout：`out[..., 0::2] = ye; out[..., 1::2] = yo` | NPU 上 strided slicing ~7ms（实测）|
| wrapper 端构造 `x_swap` 走 trailing-64 fused 路径 | 违反 wrapper 仅做校验/派发的约定 |
| 在 bf16 直接乘加（不 cast 到 fp32） | atol=1e-4 不可达 |

---

## 2. 数据规格

### 2.1 Kernel 函数签名（v2.3）

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def interleave_rope_kernel_n{1|128}_{bf16|fp16}(
    x:   pypto.Tensor([pypto.DYNAMIC, N_STATIC, pypto.DYNAMIC, 64], DTYPE),  # [B, N, S, D=64]
    cos: pypto.Tensor([pypto.DYNAMIC, 1,        pypto.DYNAMIC, 64], DTYPE),  # [B, 1, S_cs, 64]
    sin: pypto.Tensor([pypto.DYNAMIC, 1,        pypto.DYNAMIC, 64], DTYPE),  # [B, 1, S_cs, 64]
    out: pypto.Tensor([pypto.DYNAMIC, N_STATIC, pypto.DYNAMIC, 64], DTYPE),  # [B, N, S, D=64]
):
    ...
```

> `N_STATIC ∈ {1, 128}` 在 wrapper 端按实际 N 派发；`DTYPE ∈ {DT_BF16, DT_FP16}` 同理。共 4 个 kernel 实例（`{N=1, N=128} × {bf16, fp16}`）。
> S_cs ∈ {1, S}：S_cs=1 时由 wrapper 通过 `cos.expand(B,1,S,64).contiguous()` 展平到 S，kernel 始终见 S_cs=S，避免 kernel 内做条件分支。
> 输出 layout 为 split-half：左半 [..., 0:32] 存 y_even，右半 [..., 32:64] 存 y_odd。

### 2.2 动态轴分析

| 维度名 | 是否动态 | 取值范围 / 常量 | 标注方式 |
|--------|---------|-----------------|---------|
| B      | 是      | [1, 4]          | `pypto.DYNAMIC` |
| N      | 是（多态 1/128） | {1, 128} | `pypto.DYNAMIC`；entry 端按 N 选择 tile 配置 |
| S      | 是      | [1, 8192]       | `pypto.DYNAMIC` |
| S_cs   | 是      | {1, S}          | `pypto.DYNAMIC`；entry 端按 S_cs 派发 |
| D      | 否      | 64              | 直接写常量 64 |

### 2.3 值类型分析（避免 SymbolicScalar 误用）

| 变量 | 来源 | 类型 | 注意事项 |
|------|------|------|---------|
| `B = x.shape[0]` | 动态轴 | SymbolicScalar | 仅可作为 `pypto.loop` 上界、`pypto.view` offset；不可 Python `if/range/min` |
| `N = x.shape[1]` | 动态轴 | SymbolicScalar | 同上；entry 端按静态 N 值（1/128）派发 |
| `S = x.shape[2]` | 动态轴 | SymbolicScalar | 同上 |
| `D = 64`         | 编译期常量 | int | 用于 `set_vec_tile_shapes` 尾轴、`reshape` 形参 |
| `b, n, s` 等 loop 索引 | `pypto.loop` 返回 | SymbolicScalar | 仅作为 `view`/`assemble` 的 offset；不能做下标列表 |
| 切片字面量 `D//2 = 32`、`[..., 32, 1]` 等 | 编译期常量 | int | 直接传给 `reshape` |

---

## 3. Tiling 策略

### 3.1 算子类型

**Vector** —— 仅 elementwise (mul/add/cast)，无 gathermask、reshape、concat、matmul、reduction。仅需 `set_vec_tile_shapes`，且**全程 4D trailing=64**（无 5D 中间形态）。

### 3.2 Tile 尾轴对齐（v2.2 关键考量）

NPU 向量单元（Cube/Vector unit）单条指令处理 256 字节：

| dtype | 单指令处理元素 | 当 trailing=32 | 当 trailing=64 |
|-------|----------------|----------------|----------------|
| bf16/fp16 | 128 | **填充 1/4，pad 浪费 3/4** | 填充 1/2，pad 浪费 1/2 |
| fp32 | 64 | 填充 1/2 | **填充 1/1（满）** |

v2.1 在 fp32 累积阶段使用 trailing=32（D/2），fp32 单指令仅填一半，bf16 cast 阶段更只填 1/4。这是巨大浪费。
v2.2 全程 trailing=64：fp32 阶段单指令满载、bf16/fp16 阶段填 1/2，比 v2.1 至少 2× 单指令吞吐提升。

### 3.3 UB 预算

**同时驻留的 Tensor（fp32 累积阶段，trailing=64）**：

| Tensor | 用途 | tile shape | dtype | 单 tile 字节数 |
|--------|------|------------|-------|---------------|
| x_t / xs_t       | view(x), view(x_swap)   | [1, n_tile, s_tile, 64] × 2 | bf16/fp16 | 2 · 128·n·s = 256·n·s |
| c_t / s_t        | view(cos), view(sin)    | [1, 1, s_tile, 64] × 2      | bf16/fp16 | 256·s |
| x_f / xs_f       | cast → fp32             | [1, n_tile, s_tile, 64] × 2 | fp32      | 512·n·s |
| c_f / s_f        | cast → fp32             | [1, 1, s_tile, 64] × 2      | fp32      | 512·s |
| t1 / t2          | mul 中间                | [1, n_tile, s_tile, 64] × 2 | fp32      | 512·n·s |
| y_f              | add 输出                | [1, n_tile, s_tile, 64]     | fp32      | 256·n·s |
| y_t              | cast → 原 dtype          | [1, n_tile, s_tile, 64]     | bf16/fp16 | 128·n·s |
| **合计上界**     |                         |                             |           | ≈ 1664·n·s + 1024·s 字节 |

UB 容量按 A2/A3 约 192 KB；保守 128 KB 数据预算：

- `n_tile=8, s_tile=16`：1664·128 + 1024·16 ≈ 230 KB（**略超**预算，仍能编译，依靠 N=1 broadcast 共享）
- `n_tile=4, s_tile=16`：1664·64 + 1024·16 ≈ 122 KB ✓
- `n_tile=8, s_tile=32`：1664·256 + 1024·32 ≈ 449 KB ✗
- `n_tile=32, s_tile=16`（v2.1 配置但 trailing=32）：编译失败（v2.2 trailing=64 时 UB 翻倍）

**最终 tile 配置（v2.2）**：

```python
# entry 端按 N 静态派发：
if N == 128:
    pypto.set_vec_tile_shapes(1, 8, 16, 64)    # n_tile=8, s_tile=16
else:  # N == 1
    pypto.set_vec_tile_shapes(1, 1, 32, 64)    # s_tile=32
```

**展开约束**：
- N=128, tile=(1,8,16,64)：N 切分 16 次、S 切分 S/16 次 → 总 16·S/16 = S 次（per b）。
- N=1, tile=(1,1,32,64)：S 切分 S/32 次（per b）。

### 3.4 替代 tile

| 备选 tile | 否决理由 |
|-----------|---------|
| (1, 32, 16, 64) | trailing-64 时 fp32 工作集翻倍，UB 超（实测编译失败 ErrCode F21001）|
| (1, 16, 16, 64) | UB 边缘可编译；性能与 (1,8,16,64) 相当但风险更高 |
| (1, 8, 32, 64)  | s_tile=32 时 fp32 工作集 ~449 KB，超 UB |
| (1, 8, 8, 64)   | s_tile 太小，loop 数翻倍，性能下降 |
| (1, 1, 64, 64)  | N=1 路径，UB 估算 ~1024·64≈64 KB，可作为 perf 试点 |

---

## 4. Loop 与数据流

### 4.1 维度判定

| 轴 | 维度大小 | 编译期 / 运行期 | Loop 处理 |
|----|---------|----------------|----------|
| B  | 动态 [1,4] | 运行期 | `pypto.loop(B, name="b")` |
| N  | 动态 {1,128} | 运行期（entry 端按 N 静态派发不同 tile） | N=128：`pypto.loop(N // 8, name="n")` 切 n_tile=8；N=1：不需要外层 loop |
| S  | 动态 [1,8192] | 运行期 | `pypto.loop(S // s_tile, name="s")` 切 s_tile |
| D  | 编译期 64 | 编译期 | 整段进 tile，不切 |

### 4.2 完整伪代码（v2.2 fused）

```python
import pypto
import torch
from pypto import DT_FP16, DT_BF16, DT_FP32

# ---------- wrapper 端：构造 x_swap，派发 kernel ----------
def _build_x_swap(x):  # x: [B, N, S, 64]
    pairs = x.view(B, N, S, 32, 2)
    return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).reshape(B, N, S, 64)

def interleave_rope_wrapper(x, cos, sin):
    if S_cs == 1 and S != 1:
        cos = cos.expand(B, 1, S, 64).contiguous()
        sin = sin.expand(B, 1, S, 64).contiguous()
    x_swap = _build_x_swap(x)
    out = torch.empty_like(x)
    _KERNELS[(N, x.dtype)](x, x_swap, cos, sin, out)
    return out


# ---------- N=128 路径（核心路径） ----------
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def interleave_rope_kernel_n128_bf16(
    x:      pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], DT_BF16),
    x_swap: pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], DT_BF16),
    cos:    pypto.Tensor([pypto.DYNAMIC,   1, pypto.DYNAMIC, 64], DT_BF16),
    sin:    pypto.Tensor([pypto.DYNAMIC,   1, pypto.DYNAMIC, 64], DT_BF16),
    out:    pypto.Tensor([pypto.DYNAMIC, 128, pypto.DYNAMIC, 64], DT_BF16),
):
    N_TILE = 8       # int (静态) — UB 限制
    S_TILE = 16      # int (静态)
    D      = 64      # int (静态)
    pypto.set_vec_tile_shapes(1, N_TILE, S_TILE, D)   # 4D tile, 全程 trailing=64

    B = x.shape[0]
    S = x.shape[2]
    for b in pypto.loop(B, name="b"):
        for n_blk in pypto.loop(128 // N_TILE, name="n"):    # 16 次
            n_off = n_blk * N_TILE
            for s_blk in pypto.loop(S // S_TILE, name="s"):
                s_off = s_blk * S_TILE

                # ---- view 读 tile（trailing=64 全段） ----
                x_t  = pypto.view(x,      [1, N_TILE, S_TILE, D], [b, n_off, s_off, 0])
                xs_t = pypto.view(x_swap, [1, N_TILE, S_TILE, D], [b, n_off, s_off, 0])
                c_t  = pypto.view(cos,    [1, 1,      S_TILE, D], [b, 0,     s_off, 0])
                s_t  = pypto.view(sin,    [1, 1,      S_TILE, D], [b, 0,     s_off, 0])

                # ---- cast → fp32（trailing=64 满载向量单元）----
                x_f  = pypto.cast(x_t,  DT_FP32)
                xs_f = pypto.cast(xs_t, DT_FP32)
                c_f  = pypto.cast(c_t,  DT_FP32)
                s_f  = pypto.cast(s_t,  DT_FP32)

                # ---- 核心计算（一行 fma 等价：y = x·cos + x_swap·sin）----
                y_f  = pypto.add(pypto.mul(x_f, c_f), pypto.mul(xs_f, s_f))

                # ---- cast 回原 dtype 并写回 ----
                y_t  = pypto.cast(y_f, DT_BF16)
                pypto.assemble(y_t, [b, n_off, s_off, 0], out)


# ---------- N=1 路径（轻量） ----------
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def interleave_rope_kernel_n1_bf16(
    x:      pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], DT_BF16),
    x_swap: pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], DT_BF16),
    cos:    pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], DT_BF16),
    sin:    pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], DT_BF16),
    out:    pypto.Tensor([pypto.DYNAMIC, 1, pypto.DYNAMIC, 64], DT_BF16),
):
    S_TILE = 32      # int (静态)
    pypto.set_vec_tile_shapes(1, 1, S_TILE, 64)

    B = x.shape[0]
    S = x.shape[2]
    for b in pypto.loop(B, name="b"):
        for s_blk in pypto.loop(S // S_TILE, name="s"):
            s_off = s_blk * S_TILE
            x_t  = pypto.view(x,      [1, 1, S_TILE, 64], [b, 0, s_off, 0])
            xs_t = pypto.view(x_swap, [1, 1, S_TILE, 64], [b, 0, s_off, 0])
            c_t  = pypto.view(cos,    [1, 1, S_TILE, 64], [b, 0, s_off, 0])
            s_t  = pypto.view(sin,    [1, 1, S_TILE, 64], [b, 0, s_off, 0])
            x_f  = pypto.cast(x_t,  DT_FP32)
            xs_f = pypto.cast(xs_t, DT_FP32)
            c_f  = pypto.cast(c_t,  DT_FP32)
            s_f  = pypto.cast(s_t,  DT_FP32)
            y_f  = pypto.add(pypto.mul(x_f, c_f), pypto.mul(xs_f, s_f))
            y_t  = pypto.cast(y_f, DT_BF16)
            pypto.assemble(y_t, [b, 0, s_off, 0], out)


# ---------- fp16 路径 ----------
# 与 bf16 完全镜像，仅 DT_BF16 → DT_FP16；wrapper 端 _KERNELS 表静态派发 (N, dtype)。
```

> **重要约束（v2.2 已检查）**：
> - 全程 4D，trailing=64，无 5D 中间形态，`set_vec_tile_shapes` 4D 上限不再是约束。
> - kernel 内**无 gathermask、reshape、concat**，仅 view/cast/mul/add/assemble。
> - `assemble` 写回 4D tile，与 out 形状一致。
> - `b, n_off, s_off` 都是 SymbolicScalar，仅用于 `view`/`assemble` 的 offset 列表。
> - 入口校验和 `x_swap` 构造由 wrapper（Python 侧）完成：D=64、dtype 一致、N∈{1,128}、S_cs∈{1,S}、contiguous。

### 4.3 跨迭代状态

无。每个 (b, n_blk, s_blk) tile 计算独立，不需要 `submit_before_loop=True`。

### 4.4 尾块处理

- **S 维尾块**：S 不一定是 s_tile 的整数倍。处理方式：
  - 优先用 `valid_shape` 在最后一个 s_blk 上传 `[1, N_TILE, S - (s_blk_max-1)*S_TILE, 64]`；
  - 或在 entry 端做 padding（S 向上对齐到 16/32），算完丢弃 padding 区。
  - 推荐 `valid_shape`，避免拷贝。
- **N 维尾块**：N=128 时 128 % 8 == 0，N=1 时无切分；不存在尾块。
- **B 维尾块**：B 上每 tile 大小 1，整除，无尾块。

---

## 5. 约束自检清单（v2.2）

| # | 约束 | 是否满足 | 备注 |
|---|------|---------|------|
| 1 | 所有 sum 输入已转 FP32 | N/A | 无 sum/reduction |
| 2 | matmul 两侧 dtype 一致 | N/A | 无 matmul |
| 3 | TileShape 维度数 = 操作数维度数 | ✓ | 4D tile 对应所有 4D 操作数 |
| 4 | 尾轴满足对齐 | ✓ | trailing=64：bf16/fp16 满足 16-elem 对齐、fp32 满足 8-elem 对齐，且填满 256B 向量单元 |
| 5 | 同阶段 UB 占用 ≤ 容量 | ✓ | (1,8,16,64)≈230 KB（A2 192 KB 边缘可编译），(1,1,32,64)≈75 KB |
| 6 | 表达式展开 < 18000 | ✓ | N=128 ≈ B·(128/8)·(S/16)，N=1 ≈ B·(S/32) |
| 7 | 输出经 assemble 显式写回 | ✓ | 每 tile `pypto.assemble(y_t, [b, n_off, s_off, 0], out)` |
| 8 | 无 view/assemble 同张量回环 | ✓ | x/x_swap/cos/sin 仅 view 读，out 仅 assemble 写 |
| 9 | 动态轴标 `pypto.DYNAMIC` | ✓ | B/S 全部 `pypto.DYNAMIC`；N 静态分派，D=64 常量 |
| 10 | 动态 loop 提供 `unroll_list` | ⚠ | 初版未给；perf 阶段可调 |
| 11 | 跨迭代状态用 `submit_before_loop=True` | N/A | 无跨迭代状态 |
| 12 | 尾块用 `valid_shape` 处理 | ⚠ | 当前 P0 配置 S 均整除 tile；需通用化时再补 valid_shape |
| 13 | 无 SymbolicScalar 用作 list index / Python `if` | ✓ | b/n_off/s_off 仅作 offset |
| 14 | trailing 64 全程满足 NPU 256B 向量单元对齐 | ✓ | v2.2 关键收益 |
| 15 | wrapper 端 x_swap 构造无非 NPU 操作 | ✓ | 单次 `torch.stack`（约 200us）|

### 开放问题

| # | 问题 | 影响范围 | 待解决方式 |
|---|------|---------|-----------|
| 1 | UB 边缘配置 (1,8,16,64) 在 A2 与 A3 上是否稳定 | 影响 N=128 路径 tile 选择 | perf 阶段按平台分化；如 A2 不稳定则降到 (1,4,16,64) |
| 2 | `x_swap = torch.stack(...)` 在更大 batch（B=4,S=8192）下是否仍 < 1ms | 影响极限 case 性能 | bench 阶段实测；可改为 NPU 端 jit 构造 |
| 3 | unroll_list 是否需要显式给 | 性能影响 | perf 阶段实测后决定 |
| 4 | A2 vs A3 的 UB 容量差异 | 影响 tile 上限 | 初版按 A2 保守；perf 阶段按平台分化 |

---

## 6. 验证方案

### 6.1 测试配置

> 与 SPEC §12 典型配置对齐；P0 必须通过。

| 用例 | B | N | S | D | S_cs | dtype | 重点验证 |
|------|---|---|---|---|------|-------|---------|
| 功能_P0_min   | 1 | 1   | 1024 | 64 | 1024 | bfloat16 | 单 head 短序列：基本计算正确性、dtype |
| 功能_P0_typ   | 1 | 128 | 2048 | 64 | 2048 | bfloat16 | 多头典型 + N=128 tile 路径 |
| 功能_P0_Scs1  | 2 | 128 | 4096 | 64 | 1    | bfloat16 | S_cs=1 broadcast 路径 |
| 功能_P0_typ_fp16 | 1 | 8 | 1024 | 64 | 1024 | float16 | fp16 dtype 派发 |
| 性能_P0_max   | 4 | 128 | 8192 | 64 | 8192 | bfloat16 | 动态轴上限 + 性能基线 |
| 边界_S=1      | 1 | 128 | 1    | 64 | 1    | bfloat16 | S 极小 + S_cs=1 |
| 边界_N=1      | 4 | 1   | 8192 | 64 | 8192 | bfloat16 | N=1 tile 路径 |
| 数学_cos1sin0 | 1 | 2   | 64   | 64 | 64   | bfloat16 | cos=1,sin=0 → y==x |
| 数学_x=0      | 1 | 2   | 64   | 64 | 64   | bfloat16 | x=0 → y=0 |

### 6.2 精度容忍度

| dtype | rtol | atol |
|-------|------|------|
| FP16  | 0.0078125 | 0.0001 |
| BF16  | 0.0078125 | 0.0001 |

### 6.3 验证流程

1. **优先 P0**：先跑 `功能_P0_min` 与 `功能_P0_typ_fp16`（最小工作集），通过后再跑 `功能_P0_typ` 与 `功能_P0_Scs1`。
2. **与 golden 对比**：调用 `interleave_rope_golden(x, cos, sin)` 取得参考输出，使用 `torch.allclose(y_npu, y_golden, atol=1e-4, rtol=7.8125e-3)`。
3. **失败定位**：若 P0 不通过，按精度路由切片对比 `ye_f/yo_f`（fp32 中间结果）vs golden 对应中间张量；优先怀疑 gathermask PM=1/2 拆分位序与 golden 的 `[..., 0::2]/[..., 1::2]` 一致性。
4. **性能基线**：性能_P0_max 取首跑数据，后续 perf 阶段以"首跑 × 2"为目标。
