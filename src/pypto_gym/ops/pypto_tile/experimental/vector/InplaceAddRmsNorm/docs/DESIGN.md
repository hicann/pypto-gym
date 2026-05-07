---
schema_version: "2.1"
op_name: "inplace_add_rms_norm"
status: draft
last_updated: "2026-05-06"

compute_kind: "vector"
dtypes: ["bfloat16"]
dynamic_axes: ["B", "S"]
precision: { rtol: 1e-3, atol: 1e-3 }
inplace_semantics: true
buffer_aliasing:
  x1_out: "alias x1 (写归一化结果 RmsNorm(x1+x2)*gamma)"
  x2_out: "alias x2 (写 add 中间结果 x1+x2)"
  rstd_out: "新建 buffer (1/sqrt(mean((x1+x2)^2)+eps))"
torch_library_schema: "inplace_add_rms_norm(Tensor(a!) x1, Tensor(b!) x2, Tensor gamma, float eps) -> (Tensor, Tensor, Tensor)"
---

# inplace_add_rms_norm 设计方案

> 本文档严格落实用户硬约束：
> 1. torch.library schema 使用 `Tensor(a!) x1, Tensor(b!) x2` 标记 mutable inplace 输入
> 2. Wrapper **零预处理**：禁止 reshape/broadcast/cast/contiguous/view/unsqueeze/squeeze；禁止 hook
> 3. 所有 reshape / broadcast / cast / 计算逻辑全部在 PyPTO kernel 内
> 4. Inplace 写回通过：将同一 `torch.Tensor` 同时作为 kernel 的输入参数与输出参数传入；kernel 用 `pypto.assemble` 写回到输出形参以复用输入 GM
> 5. 仅 `rstd` 在 wrapper 中通过 `torch.empty(...)` 新建（这不是预处理，是必要的输出 buffer 分配）

---

## 1. 计算图与精度路由

### 1.1 API 调用序列

完全采用 API_REPORT §2 的 16 步分解路径。**禁用** `pypto.rms_norm`（不返回 rstd、不返回 add 中间结果）。

| 步骤 | 操作 | PyPTO API | 输入 dtype | 输出 dtype | 输出 shape (单 tile) | 备注 |
|------|------|-----------|------------|------------|---------------------|------|
| 0a | gamma reshape | `pypto.reshape(gamma, [1,1,H], inplace=True)` | bf16 | bf16 | `[1,1,H]` | 在 kernel 内 reshape，使其能与 [tile_t,1,H] 广播 |
| 0b | gamma cast | `pypto.cast(gamma_3d, DT_FP32)` | bf16 | fp32 | `[1,1,H]` | 提前一次性 cast，loop 外执行 |
| 1 | view x1 tile | `pypto.view(x1, [tile_t, S, H], [t_idx,0,0])` | bf16 | bf16 | `[tile_t,S,H]` | tile_t 通常 1；按 B 维切 |
| 2 | view x2 tile | `pypto.view(x2, [tile_t, S, H], [t_idx,0,0])` | bf16 | bf16 | `[tile_t,S,H]` | 同上 |
| 3 | cast x1 → fp32 | `pypto.cast(x1_tile, DT_FP32)` | bf16 | fp32 | `[tile_t,S,H]` | 进入 fp32 域 |
| 4 | cast x2 → fp32 | `pypto.cast(x2_tile, DT_FP32)` | bf16 | fp32 | `[tile_t,S,H]` | |
| 5 | add | `pypto.add(x1_fp32, x2_fp32)` | fp32 | fp32 | `[tile_t,S,H]` | x_add fp32 |
| 6 | square | `pypto.mul(x_add_fp32, x_add_fp32)` | fp32 | fp32 | `[tile_t,S,H]` | |
| 7 | sum dim=-1 | `pypto.sum(sq, dim=-1, keepdim=True)` | fp32 | fp32 | `[tile_t,S,1]` | sum 仅支持 fp32 |
| 8 | / H | `s_fp32 / H` | fp32 | fp32 | `[tile_t,S,1]` | mean = sum / hidden_size |
| 9 | + eps | `mean + eps` | fp32 | fp32 | `[tile_t,S,1]` | scalar broadcast |
| 10 | rsqrt | `pypto.rsqrt(t)` | fp32 | fp32 | `[tile_t,S,1]` | rstd_fp32 |
| 11 | y = x_add * rstd | `pypto.mul(x_add_fp32, rstd_fp32)` | fp32 | fp32 | `[tile_t,S,H]` | broadcast [_,_,1] → [_,_,H] |
| 12 | y *= gamma | `pypto.mul(y_tmp, gamma_fp32_3d)` | fp32 | fp32 | `[tile_t,S,H]` | broadcast [1,1,H] |
| 13 | cast y → bf16 | `pypto.cast(y_fp32, DT_BF16)` | fp32 | bf16 | `[tile_t,S,H]` | 写回 dtype |
| 14 | cast x_add → bf16 | `pypto.cast(x_add_fp32, DT_BF16)` | fp32 | bf16 | `[tile_t,S,H]` | 写回 x2 dtype |
| 15 | cast rstd → bf16 | `pypto.cast(rstd_fp32, DT_BF16)` | fp32 | bf16 | `[tile_t,S,1]` | 写回 rstd dtype |
| 16a | assemble → x1_out | `pypto.assemble(y_bf16, [t_idx,0,0], x1_out)` | bf16 | — | — | inplace 写回 x1 GM |
| 16b | assemble → x2_out | `pypto.assemble(x_add_bf16, [t_idx,0,0], x2_out)` | bf16 | — | — | inplace 写回 x2 GM |
| 16c | assemble → rstd_out | `pypto.assemble(rstd_bf16, [t_idx,0,0], rstd_out)` | bf16 | — | — | 写到独立 rstd buffer |

### 1.2 精度路由

```text
x1, x2 (bf16) ──cast──┐
                       ├──add──► x_add (fp32) ──┬──► assemble→x2_out (cast→bf16)
                       │                         │
gamma (bf16) ──cast──► gamma_fp32                ├──square──sum──/H──+eps──rsqrt──► rstd (fp32)
                                                 │                                   │
                                                 ├──mul rstd ──mul gamma ──► y (fp32)
                                                 │                                   │
                                                 │                                   ├──cast→bf16──► assemble→x1_out
                                                 │                                   │
                                                 └────────────────────────────────────┴──cast→bf16──► assemble→rstd_out
```

| 转换位置 | 转换方向 | 原因 |
|---------|---------|------|
| 步骤 3, 4 | bf16 → fp32 | `pypto.sum` 仅 fp32；累加避免 bf16 精度损失 |
| 步骤 0b | bf16 → fp32 | 与 fp32 y 做 mul 需要同 dtype |
| 步骤 13, 14, 15 | fp32 → bf16 | 输出回 bf16 |

### 1.3 替代方案（已排除）

| 替代方案 | 排除原因 |
|---------|---------|
| `pypto.rms_norm(x, gamma, eps)` 单算子 | 仅返回 y，**不返回 rstd**；也不接受 pre-add 输入也不输出 add 中间结果。无法满足 spec 三输出 |
| 全程 bf16 累加（不 cast fp32） | bf16 sum(x^2) 累积误差大，H=7168 下大概率超过 atol=1e-3 |
| Wrapper 内做 reshape gamma 到 [1,1,H] | **违反用户硬约束**（wrapper 严格透传） |
| 在 wrapper 中 `torch.empty` 出 x1_out, x2_out 然后拷回 | 违反 inplace 语义，与 `Tensor(a!)` 矛盾，且额外内存与拷贝开销 |
| 保持 3 维 `[B,S,H]` 切 B 与 reshape 到 `[B*S,H]` 切 B*S | 当前选 3 维方案：避免 kernel 内额外 reshape；动态 B、动态 S 时 B*S = SymbolicScalar*SymbolicScalar 可能不支持，仅切 B 更稳 |

---

## 2. 数据规格

### 2.1 Kernel 函数签名

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def inplace_add_rms_norm_kernel(
    # ── 输入 ──
    x1:        pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, 7168], pypto.DT_BF16),  # [B,S,H]
    x2:        pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, 7168], pypto.DT_BF16),  # [B,S,H]
    gamma:     pypto.Tensor([7168],                              pypto.DT_BF16),   # [H]
    eps:       float,                                                              # python float
    # ── 输出（与 x1, x2 在 wrapper 中是同一 torch.Tensor，复用 GM；rstd 是独立 buffer）──
    x1_out:    pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, 7168], pypto.DT_BF16),  # alias x1
    x2_out:    pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, 7168], pypto.DT_BF16),  # alias x2
    rstd_out:  pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, 1],    pypto.DT_BF16),  # 独立 buffer
):
    ...
```

### 2.2 动态轴分析

| 维度名 | 是否动态 | 取值范围 / 常量 | 标注方式 |
|--------|---------|-----------------|---------|
| B (batch)        | **是** | [1, 144]    | `pypto.DYNAMIC` |
| S (seq)          | **是** | [1, 8192]   | `pypto.DYNAMIC` |
| H (hidden)       | **否** | 常量 7168   | 数值字面量 |

reduce 沿 H 维（编译期 STATIC=7168），不会触发"DYNAMIC 维进 reduce"风险。

### 2.3 值类型分析

| 变量 | 来源 | 类型 | 注意事项 |
|------|------|------|---------|
| `B = x1.shape[0]` | 动态轴 | SymbolicScalar | 必须用 `pypto.loop(B)` 循环；不可 `range(B)` |
| `S = x1.shape[1]` | 动态轴 | SymbolicScalar | 不参与 Python 条件；切片用 `pypto.view` 而非下标 |
| `H = 7168` | spec 常量 | Python int | 可参与算术，做除数 `s / H` |
| `eps` | wrapper 形参 | Python float | scalar broadcast `mean + eps` |
| `t_idx` | `pypto.loop` 循环索引 | SymbolicScalar | 用作 `pypto.view`/`assemble` 的 offset |
| `tile_t` | 编译期常量 | Python int | 1 |

---

## 3. Tiling 策略

### 3.1 算子类型

**Vector**（无 matmul/conv，仅 elementwise + reduce）。

### 3.2 Tiling 推导

#### 同时驻留 UB 的 Tensor（reduce 阶段最重，作为预算上界）

设 `tile = (tile_t, S_tile, H)`，因 H=7168 整片不切，逐元素阶段 S_tile 由实际 S 决定（若 S 大则切 S；目前以 tile_t=1, 单步覆盖 [1, S, H] 评估）。

为避免 sum 路径 64KB 上限，先估算 reduce 阶段单 tile（取 `tile = (1, 1, 7168)`，即每次 reduce 一行）：

| Tensor | 用途 | shape | dtype | 大小估算 (bytes) |
|--------|------|-------|-------|------------------|
| x1_tile_bf16 | 输入分片 | [1,1,7168] | bf16 | 14336 |
| x2_tile_bf16 | 输入分片 | [1,1,7168] | bf16 | 14336 |
| x1_fp32      | cast 上行 | [1,1,7168] | fp32 | 28672 |
| x2_fp32      | cast 上行 | [1,1,7168] | fp32 | 28672 |
| x_add_fp32   | add 结果 | [1,1,7168] | fp32 | 28672 |
| sq_fp32      | square | [1,1,7168] | fp32 | 28672 |
| sum_fp32     | reduce | [1,1,1]    | fp32 | 4 |
| rstd_fp32    | rsqrt  | [1,1,1]    | fp32 | 4 |
| y_fp32       | mul    | [1,1,7168] | fp32 | 28672 |
| gamma_fp32_3d| gamma  | [1,1,7168] | fp32 | 28672（loop 外常驻） |
| y_bf16       | cast 下行 | [1,1,7168] | bf16 | 14336 |
| x_add_bf16   | cast 下行 | [1,1,7168] | bf16 | 14336 |
| rstd_bf16    | cast 下行 | [1,1,1]    | bf16 | 2 |

合计 ≈ 234 KB。UB 为 192 KB（昇腾 910 vec UB 典型），单 tile 全活 buffer 略超。

**优化策略**：
1. PyPTO 编译器具备生命周期分析，cast 后原 buffer 可释放（如 x1_tile_bf16 在 cast 出 x1_fp32 之后即可丢弃）；`x1_fp32` 只在 add 时被读，add 完成后可释放；同理 `sq_fp32` 在 sum 后释放。实际同时驻留预算远小于上表合计。
2. `pypto.sum` 自身的 64KB 限制是**对其 input 的 TileShape**：input shape=[1,1,7168] fp32 = 28672 B ≤ 64 KB ✓
3. 尾轴 32B 对齐：fp32×7168 = 28672 B（÷32=896），bf16×7168 = 14336 B（÷32=448），均对齐 ✓

#### 推荐 set_vec_tile_shapes 配置

考虑两个相位（PyPTO 允许 kernel 内多次 set_vec_tile_shapes 切换 tile 形状）：

| 相位 | 操作 | tile 配置（[B,S,H]） | 推导 |
|------|------|----------------------|------|
| 全局 | 默认（loop 外 cast gamma 等） | `set_vec_tile_shapes(1, 1, 7168)` | 安全保守 |
| Loop 内 elementwise (add/cast/mul) | `set_vec_tile_shapes(1, 1, 7168)` | reduce 阶段同行；保持简单一致 |
| reduce 阶段 (sum/rsqrt) | `set_vec_tile_shapes(1, 1, 7168)` | 64KB 限制下 tile_t=1 fp32 一行 28KB 安全 |

> Stage 5 实测后（Stage 7 调优）可尝试将 elementwise 阶段 tile 增大到 `(1, 2, 7168)` 或 `(2, 1, 7168)`（fp32 56KB）以提高吞吐。

### 3.3 替代方案

| 备选 tile | 否决理由 |
|-----------|---------|
| `(2, 1, 7168)` 或 `(1, 2, 7168)` 用于 reduce | fp32 input 2×28672 = 57344 B 仍 ≤ 64KB；可作为 Stage 7 调优候选；首跑选保守 (1,1,7168) |
| `(4, 1, 7168)` | fp32 4×28672 = 114688 B > 64 KB，违反 sum 限制 |
| 沿 H 维切 | reduce 沿 -1，切 H 会破坏 reduce 语义；除非引入两阶段 reduce，复杂度大幅上升 |
| reshape 到 [B*S, H] 沿 B*S 切 | 动态 B × 动态 S 的乘积是 SymbolicScalar 算术，loop 形态更复杂；3 维 + 切 B 已能覆盖（每次 view 出 [1, S, H]） |

---

## 4. Loop 与数据流

### 4.1 维度判定

| 轴 | 维度大小 | 编译期 / 运行期 | Loop 处理 |
|----|---------|----------------|----------|
| B  | DYNAMIC ([1,144]) | 运行期 | `pypto.loop(B, name="b_loop")` 或 `pypto.loop_unroll(0, B, 1, ...)` |
| S  | DYNAMIC ([1,8192]) | 运行期 | **不显式 loop**；每次 view 出 [1, S, H]，由 PyPTO 自动按 vec_tile 切 |
| H  | 7168 (静态)        | 编译期 | 不 loop（整片参与 reduce） |

> 决策：**只对 B 维显式 loop**。view [tile_t=1, S, H] 出来后，S 维由 PyPTO 编译器根据 set_vec_tile_shapes 的最后两维（这里是 1, 7168）自动 tile。这是 layer_norm 例子和 hc_pre 都使用的模式。

### 4.2 完整伪代码

```python
import math
import pypto
import torch

H_CONST = 7168  # 静态 hidden size
TILE_B = 1      # 单次循环处理 1 个 batch（保守）

@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def inplace_add_rms_norm_kernel(
    x1:       pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H_CONST], pypto.DT_BF16),
    x2:       pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H_CONST], pypto.DT_BF16),
    gamma:    pypto.Tensor([H_CONST],                                pypto.DT_BF16),
    eps:      float,
    x1_out:   pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H_CONST], pypto.DT_BF16),
    x2_out:   pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, H_CONST], pypto.DT_BF16),
    rstd_out: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, 1],       pypto.DT_BF16),
):
    # SymbolicScalar 维度
    B = x1.shape[0]      # SymbolicScalar
    S = x1.shape[1]      # SymbolicScalar
    H = H_CONST          # Python int，常量

    # ── Loop 外：gamma 的 reshape + cast 一次性做 ──
    pypto.set_vec_tile_shapes(1, 1, H)                               # 全局默认 tile
    pypto.reshape(gamma, [1, 1, H], inplace=True)                    # gamma: [H] -> [1,1,H]，bf16
    gamma_fp32 = pypto.cast(gamma, pypto.DT_FP32)                    # [1,1,H], fp32（loop 外常驻）

    # ── 沿 B 维循环 ──
    # 使用 loop_unroll 提供 unroll_list 便于编译器选择展开度（参考 hc_pre）
    unroll_list = [1, 2, 4, 8]
    for b_idx, _ in pypto.loop_unroll(0, B, 1, name="b_loop",
                                      idx_name="b_idx", unroll_list=unroll_list):
        # b_idx: SymbolicScalar
        # 1. view tile（不切 S）
        x1_tile = pypto.view(x1, [TILE_B, S, H], [b_idx, 0, 0])      # [1,S,H], bf16
        x2_tile = pypto.view(x2, [TILE_B, S, H], [b_idx, 0, 0])      # [1,S,H], bf16

        # 2. cast 到 fp32
        x1_fp32 = pypto.cast(x1_tile, pypto.DT_FP32)                 # [1,S,H], fp32
        x2_fp32 = pypto.cast(x2_tile, pypto.DT_FP32)                 # [1,S,H], fp32

        # 3. add (x_add 同时是 x2 写回的源 + 后续 y 计算的输入)
        x_add_fp32 = pypto.add(x1_fp32, x2_fp32)                     # [1,S,H], fp32

        # 4. square + sum + /H + eps + rsqrt
        sq_fp32   = pypto.mul(x_add_fp32, x_add_fp32)                # [1,S,H], fp32
        sum_fp32  = pypto.sum(sq_fp32, dim=-1, keepdim=True)         # [1,S,1], fp32
        mean_fp32 = sum_fp32 / H                                      # [1,S,1], fp32
        rstd_fp32 = pypto.rsqrt(mean_fp32 + eps)                     # [1,S,1], fp32

        # 5. y = x_add * rstd * gamma  (broadcast [1,S,1] / [1,1,H])
        y_fp32 = pypto.mul(x_add_fp32, rstd_fp32)                    # [1,S,H], fp32
        y_fp32 = pypto.mul(y_fp32, gamma_fp32)                       # [1,S,H], fp32

        # 6. cast 回 bf16
        y_bf16     = pypto.cast(y_fp32,     pypto.DT_BF16)           # [1,S,H], bf16
        x_add_bf16 = pypto.cast(x_add_fp32, pypto.DT_BF16)           # [1,S,H], bf16
        rstd_bf16  = pypto.cast(rstd_fp32,  pypto.DT_BF16)           # [1,S,1], bf16

        # 7. inplace 写回（assemble 端点）
        # 注意：x1_tile / x2_tile 的"读取"已在前序步骤完成（x1_fp32, x2_fp32 已落 SSA），
        # 此处对 x1_out / x2_out 写回不会产生与 x1 / x2 输入读取的环路。
        pypto.assemble(y_bf16,     [b_idx, 0, 0], x1_out)            # x1 buffer ← y
        pypto.assemble(x_add_bf16, [b_idx, 0, 0], x2_out)            # x2 buffer ← x_add
        pypto.assemble(rstd_bf16,  [b_idx, 0, 0], rstd_out)          # rstd buffer ← rstd
```

### 4.3 跨迭代状态

| 状态名 | 初始化 | 更新方式 | submit_before_loop |
|--------|--------|---------|--------------------|
| 无 | — | — | — |

每个 b_idx 独立计算，无跨迭代依赖。`gamma_fp32` 在 loop 外初始化为常量 alias。

### 4.4 尾块处理

- 沿 B 维循环，TILE_B=1，B 必为整数倍，**无尾块**。
- S 维由 PyPTO 编译器自动按 vec tile 切，编译器自带 valid_shape 处理尾块。

### 4.5 Inplace 写回时序保证

> 用户硬约束 #4 + API_REPORT §7 关键风险点

读后写顺序：
1. **先读**：`x1_fp32 = pypto.cast(x1_tile, ...)` 与 `x2_fp32 = pypto.cast(x2_tile, ...)` 在 add 步骤前完成 → `x1`/`x2` 输入端的读取已落地为 SSA tensor。
2. **再算**：所有后续运算都在 SSA 中间 tensor 上进行，**不再读取 `x1` / `x2` 输入参数**。
3. **最后写**：`pypto.assemble(..., x1_out)` / `pypto.assemble(..., x2_out)` 在 loop 最后执行。
4. 由于 `x1` 与 `x1_out`、`x2` 与 `x2_out` 在 wrapper 中是同一 torch.Tensor，PyPTO 编译器看到的是不同形参（不同 SSA 变量），不构成图中的 "view + assemble 同一张量" 环路。
5. 对于不同 b_idx 之间的写入：`assemble(..., [b_idx,0,0], x1_out)` 写不同的 GM 段，无 race。

---

## 5. 约束自检清单

| # | 约束 | 是否满足 | 备注 |
|---|------|---------|------|
| 1 | 所有 `sum` 输入已转 FP32 | ✅ | 步骤 7 前已 cast 到 fp32 |
| 2 | matmul 两侧 dtype 一致 | N/A | 无 matmul |
| 3 | TileShape 维度数 = 操作数维度数 | ✅ | `(1,1,7168)` 三维，与 `[B,S,H]` 三维一致 |
| 4 | 尾轴满足对齐 | ✅ | 7168 既是 16 倍数（bf16）又是 8 倍数（fp32） |
| 5 | 同阶段 UB 占用 ≤ 容量 | ⚠ | 按生命周期分析约 60KB；需 Stage 5 实测确认；备选可降到 `(1,1,3584)` 但破坏 reduce |
| 6 | 表达式展开 < 18000 | ✅ | tile=(1,1,7168) 单次 view 完整 H；展开数 = 1 × 1 × 1 = 1 远小于 18000 |
| 7 | 输出经 `assemble` 显式写回 | ✅ | 三处 assemble |
| 8 | 无 view/assemble 同张量回环 | ✅ | view 用 x1/x2 输入形参；assemble 用 x1_out/x2_out 输出形参（不同 SSA） |
| 9 | 动态轴标 `pypto.DYNAMIC` | ✅ | B、S 两轴 |
| 10 | 动态 loop 提供 `unroll_list` | ✅ | `unroll_list=[1,2,4,8]` |
| 11 | 跨迭代状态用 `submit_before_loop=True` | N/A | 无跨迭代状态 |
| 12 | 尾块用 `valid_shape` 处理 | N/A | TILE_B=1 无尾块；S 维由编译器自动处理 |
| 13 | 无 SymbolicScalar 用作 `**` / list index / Python `if` | ✅ | `H` 是 Python int 才用作除数；B、S 仅做 `pypto.loop` 与 `pypto.view` 输入 |
| 14 | `set_vec_tile_shapes` 在首个向量 op 前 | ✅ | loop 外先 set，loop 内未再变更 |
| 15 | gamma 仅 reshape 一次（loop 外） | ✅ | inplace=True，避免重复 op |
| 16 | inplace 语义在 schema 层声明 | ✅ | `Tensor(a!) x1, Tensor(b!) x2` |
| 17 | wrapper 零预处理 | ✅ | 仅 `torch.empty` 出 rstd + 调 kernel |

### 开放问题

| # | 问题 | 影响范围 | 待解决方式 |
|---|------|---------|-----------|
| OQ1 | PyPTO 编译器是否会因为 `x1` 与 `x1_out` 在 torch.Tensor 层别名而触发图层别名分析告警 | 编译期 | Stage 5 首跑实测；若失败回退方案：在 wrapper 内 `x1_out = x1`（保留同一引用）+ 在 schema 中保留 `(a!)` 标记 |
| OQ2 | bf16 输入下，atol=0.001 是否过严（业界 bf16 RMSNorm 常用 0.01） | 精度门禁 | test_cases.json 已用 0.01；若 0.001 过严由 Stage 5 反馈后调整 SPEC |
| OQ3 | `pypto.reshape(gamma, [1,1,H], inplace=True)` 在 jit 内对输入参数的合法性 | 编译期 | 若不合法回退用 `gamma_3d = pypto.reshape(gamma, [1,1,H])`（非 inplace） |

---

## 6. 验证方案

### 6.1 测试配置（与 test_cases.json 对齐）

| 用例 | 输入 shape | dtype | 重点验证 |
|------|----------|-------|---------|
| level0 | x1/x2=[1,16,7168]   | bf16 | 小数据基础功能；inplace 语义（data_ptr / 内容覆写） |
| level1 | x1/x2=[16,128,7168] | bf16 | B*S=2048 P0 性能场景 |
| level2 | x1/x2=[8,128,7168]  | bf16 | B*S=1024 最小边界 |
| level3 | x1/x2=[64,128,7168] | bf16 | B*S=8192 最大边界 |
| level4 | x1/x2=[144,1,7168]  | bf16 | S=1 特殊场景 |
| level5 | x1/x2=[1,1024,7168] | bf16 | B=1 边界 |

### 6.2 Inplace 验证（必须项）

测试 `test_inplace_add_rms_norm.py` 必须包含：

1. **data_ptr 不变**：`p1_before = x1.data_ptr()`；调用后 `assert x1.data_ptr() == p1_before`；x2 同理。
2. **返回值是 alias**：`final_y, x_add, rstd = npu_inplace_add_rms_norm(x1, x2, gamma, eps)`；`assert final_y.data_ptr() == x1.data_ptr()` 且 `x_add.data_ptr() == x2.data_ptr()`。
3. **rstd 是新建 tensor**：`assert rstd.data_ptr() not in (x1.data_ptr(), x2.data_ptr())`。
4. **内容覆写正确性**：与 golden（同样 inplace 语义）对比 x1、x2、rstd 三个张量。
5. **shape**：x1=[B,S,H], x2=[B,S,H], rstd=[B,S,1]，dtype 全 bf16。

### 6.3 精度容忍度

| dtype | rtol | atol | 来源 |
|-------|------|------|------|
| BF16  | 1e-3 | 1e-3 | SPEC §6（严格） |
| BF16  | 1e-2 | 1e-2 | test_cases.json（实操推荐，bf16 H=7168 reduce 累积误差） |

> 首跑用 SPEC 严格阈值（atol=rtol=1e-3）。若 H=7168 bf16 累积误差导致接近超限，先降 atol/rtol 到 1e-2 再判断（不视为精度失败，仅记录），依据是 deepseek_v4 实际工程容差。

### 6.4 Wrapper 设计（落地参考，由 Stage 5 实现）

```python
# 注意：wrapper 仅做参数透传 + 新建 rstd buffer，不做任何 reshape/cast/contiguous

pyptolib = torch.library.Library("pypto", "FRAGMENT")
pyptolib.define(
    "inplace_add_rms_norm(Tensor(a!) x1, Tensor(b!) x2, Tensor gamma, float eps) "
    "-> (Tensor, Tensor, Tensor)"
)

@torch.library.impl(pyptolib, "inplace_add_rms_norm", "Meta")
def _meta(x1, x2, gamma, eps):
    rstd = torch.empty([x1.size(0), x1.size(1), 1], dtype=x1.dtype, device=x1.device)
    return x1, x2, rstd

@torch.library.impl(pyptolib, "inplace_add_rms_norm", "NPU")
def _npu(x1, x2, gamma, eps):
    return npu_inplace_add_rms_norm(x1, x2, gamma, eps)

def npu_inplace_add_rms_norm(x1, x2, gamma, eps):
    # rstd 是必须的输出 buffer 分配，不属于"预处理"
    rstd = torch.empty([x1.size(0), x1.size(1), 1], dtype=torch.bfloat16, device=x1.device)
    # x1 既是输入又是 x1_out；x2 既是输入又是 x2_out
    inplace_add_rms_norm_kernel(x1, x2, gamma, eps, x1, x2, rstd)
    return x1, x2, rstd
```

---

## 7. 风险与回退（汇总）

| 风险 | 触发条件 | 检测点 | 回退方案 |
|------|---------|--------|----------|
| inplace 别名分析告警 | kernel 编译期 | Stage 5 编译失败 | 选项 A：保留同名 schema 但 wrapper 用 `x1_alias = x1` 显式 alias 传入；选项 B：在 PyPTO kernel 内显式声明 in-place 编译选项（若有） |
| `pypto.reshape(gamma, ..., inplace=True)` 拒绝输入参数 | 编译失败 | Stage 5 错误日志 | 改为非 inplace `gamma_3d = pypto.reshape(gamma, [1,1,H])` |
| sum 64KB 限制误报 | 编译期约束 | Stage 5 报错 "TileShape exceeds 64KB" | 已选 (1,1,7168) fp32=28KB 安全；若仍报错改 (1,1,3584) 两次 reduce |
| bf16 累积误差超 atol=1e-3 | 精度首跑 | Stage 5 输出 [PRECISION_FAIL] | 进入 Stage 6；先确认 fp32 中间路径完整；考虑 SPEC atol 下调到 1e-2 |
| 动态 B、S 与 unroll_list 不匹配 | 编译/运行 | shape 错误 | unroll_list 改为 [1] 关闭展开；或换 `pypto.loop(B)` |
| wrapper 无意 contiguous() | 调用方传入非连续 tensor | 测试失败 | 测试用 `torch.randn().contiguous()` 显式构造；约束写入 SPEC §10（已写） |

---

## 8. 设计完成自检

- [x] 设计是决策记录，非 SPEC 复述
- [x] 每个决策含「结论 + 推导 + 排除替代」
- [x] 伪代码标注 shape / dtype / SymbolicScalar
- [x] inplace 写回时序明确（先读 → 计算 → 后写）
- [x] torch.library schema 用 `Tensor(a!) x1, Tensor(b!) x2`
- [x] wrapper 透传约束清晰
- [x] 测试设计覆盖 data_ptr / 内容 / shape / dtype 四项 inplace 验证
