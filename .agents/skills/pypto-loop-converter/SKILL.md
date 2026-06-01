---
name: pypto-loop-converter
description: 识别 PyPTO kernel 代码或待迁移代码中的 Python for 循环，将其转换为 pypto.loop / pypto.loop_unroll 形式，消除 Python 层逐次调用开销。触发词：循环转换、loop 转换、Python loop、for 循环优化、loop_unroll、消除 Python 循环、批量处理循环。
---

# PyPTO Loop 转换

识别 Python for 循环并转换为 `pypto.loop` / `pypto.loop_unroll`，将循环逻辑下沉到 NPU kernel 内部执行，消除 Python 层 per-iteration 调度开销。

---

## 核心概念

### 为什么需要 Loop 转换？

PyPTO JIT kernel 在 NPU 上整体编译执行。如果循环留在 Python 层，每次迭代都会触发一次完整的 kernel launch（含 host-device 同步）。将循环纳入 kernel 内部后，所有迭代在一次 kernel launch 中完成。

**性能影响示例**（LLaDA2 MoE，E=64 experts）：

| 方式 | 延迟 | 说明 |
|------|------|------|
| Python `for e in range(E)` + 逐 expert 调用 | 5,972 us | 64 次 kernel launch |
| `pypto.loop` over experts in single kernel | 584 us | 1 次 kernel launch，**10.2x 加速** |

### 两种 Loop API

| API | 用途 | trip count |
|-----|------|-----------|
| `pypto.loop(start, end, step, ...)` | 固定 trip count 或编译时已知范围的循环 | 静态或 symbolic |
| `pypto.loop_unroll(start, end, step, ..., unroll_list=[...])` | 动态 trip count 且需自动处理尾块的循环 | 动态 (DYNAMIC) |

**选择规则**：
- 循环次数**编译时确定**（如 `num_experts`，由非 tensor 参数传入） → `pypto.loop`
- 循环次数**运行时才知道**（如 `batch_size`，来自 `tensor.shape[0]`） → `pypto.loop_unroll`

---

## 所需输入

| 信息 | 用途 |
|------|------|
| 待转换的 Python 代码（含 for 循环） | 识别循环模式 |
| 循环变量的语义（batch 维度、expert 维度等） | 确定 loop/loop_unroll 选择 |
| tensor shape 信息 | tile shape 配置 |
| 是否已有 PyPTO kernel 框架 | 确定是新建还是改造 |

---

## 工作流程

### 阶段一：检测 Python 循环

在目标代码中扫描以下模式：

#### 检测模式 1：Host 侧逐次 kernel 调用

```python
# BEFORE: Python 层循环调用 kernel — 每次迭代触发一次 kernel launch
for e in range(num_experts):
    expert_ffn(sorted_tokens[start[e]:start[e+1]], w13[e], w2[e], out[e])
```

**信号**：`for ... in range(...)` 包含 PyPTO kernel 函数调用或 `torch` 操作。

#### 检测模式 2：Kernel 内部 Python range 循环

```python
@pypto.frontend.jit(...)
def my_kernel(x, w, result, num_items: int):
    for i in range(num_items):          # <-- Python range，编译时展开
        tile = x[i:i+1, :]
        out = pypto.matmul(tile, w)
        result[i:i+1, :] = out
```

**信号**：`@pypto.frontend.jit` 装饰函数内部存在 `for ... in range(...)` 且 range 参数是非 tensor 的 int 参数。

#### 检测模式 3：动态 batch 维度遍历

```python
@pypto.frontend.jit(...)
def my_kernel(x, result):
    bs = x.shape[0]                     # 动态
    for i in range(bs):                 # <-- 不可行：bs 是 SymbolicScalar
        ...
```

**信号**：range 参数来自 `tensor.shape[dim]`，属于 `SymbolicScalar`，Python `range()` 会报 TypeError。

---

### 阶段二：确定转换策略

根据循环变量来源选择 API：

| 循环变量来源 | API 选择 | unroll_list |
|-------------|---------|-------------|
| 非 tensor int 参数（`num_experts: int`） | `pypto.loop` | 不需要 |
| `tensor.shape[dim]`（batch size 等动态维度） | `pypto.loop_unroll` | 根据可能的 batch 范围设定，如 `[1, 2, 4, 8, 16, 32, 64]` |
| 两个 tensor 元素之差（`cumsum[e+1] - cumsum[e]`） | 外层 `pypto.loop` + 内层 `pypto.loop_unroll` | 内层需要 unroll_list |

---

### 阶段三：执行转换

#### 转换规则

**规则 1**：`range(N)` → `pypto.loop(0, N, 1, name=..., idx_name=...)`

```python
# BEFORE
for e in range(num_experts):
    # process expert e

# AFTER
for e_idx in pypto.loop(0, num_experts, 1,
                        name="EXPERT_LOOP", idx_name="e_idx"):
    # process expert e_idx
```

**规则 2**：动态 batch → `pypto.loop_unroll` + `tile_batch`

```python
# BEFORE
bs = x.shape[0]
for i in range(bs):
    tile = x[i:i+1, :]
    ...

# AFTER
bs = x.shape[0]
for bs_idx, tile_batch in pypto.loop_unroll(
    0, bs, 1,
    name="LOOP_BATCH_L0",
    idx_name="bs_idx",
    unroll_list=[1, 2, 4, 8, 16, 32, 64],
):
    tile = pypto.view(x, [tile_batch, hidden_size],
                      [bs_idx, 0],
                      valid_shape=[tile_batch, hidden_size])
    ...
```

**规则 3**：索引操作 → `pypto.view` / `pypto.assemble`

```python
# BEFORE (Python indexing)
tile = x[i:i+batch, :]
result[i:i+batch, :] = out

# AFTER (PyPTO view + assemble)
tile = pypto.view(x, [batch, W], [i, 0], valid_shape=[batch, W])
pypto.assemble(out, [i, 0], result)
```

**规则 4**：嵌套循环 — 外层 `pypto.loop` + 内层 `pypto.loop_unroll`

```python
# 典型 MoE 场景：固定 expert 数 + 动态 per-expert token 数
for e_idx in pypto.loop(0, num_experts, 1,
                        name="EXPERT_LOOP", idx_name="e_idx"):
    e_start = expert_cumsum[e_idx]
    e_end = expert_cumsum[e_idx + 1]
    n_e = e_end - e_start

    for tok_idx, tile_batch in pypto.loop_unroll(
        0, n_e, 1,
        name="LOOP_TOKEN", idx_name="tok_idx",
        unroll_list=[1, 2, 4, 8, 16, 32, 64],
    ):
        tile_x = pypto.view(sorted_tokens, [tile_batch, H],
                            [e_start + tok_idx, 0])
        # ... compute ...
        pypto.assemble(out, [e_start + tok_idx, 0], result)
```

**规则 5**：loop 内部必须重新设置 tile shapes

```python
for bs_idx, tile_batch in pypto.loop_unroll(...):
    # 每次迭代 tile_batch 可能不同，必须重设 tile shapes
    pypto.set_cube_tile_shapes(
        [tile_batch, tile_batch],
        [K_tile, K_tile * 2],
        [N_tile, N_tile],
        True,
    )
    pypto.set_vec_tile_shapes(min(tile_batch, vec_first), width)
    # ... compute ...
```

---

### 阶段四：验证

1. **语法检查**：确认转换后代码无 Python 语法错误
2. **精度验证**：运行测试用例，对比 golden 输出
   ```bash
   export TILE_FWK_DEVICE_ID=14
   python3 test_{op}.py
   ```
3. **性能验证**：对比转换前后延迟
   ```bash
   python3 benchmark/{op}/bench_{op}.py
   ```

---

## 完整转换案例

### 案例：MoE per-expert dispatch → Grouped GEMM

**转换前**：Python 循环逐 expert 调用（`pypto_patch.py` host 层）

```python
# Host-side Python dispatch: E kernel launches
for expert_idx in range(self.num_experts):
    mask = (topk_ids == expert_idx)
    expert_tokens = hidden_states[mask]
    if expert_tokens.shape[0] == 0:
        continue
    w13 = self._w13_stacked[expert_idx]
    w2  = self._w2_stacked[expert_idx]
    out = torch.empty_like(expert_tokens)
    llada2_expert_ffn(expert_tokens, w13, w2, out)
    result[mask] = out
```

**转换后**：单 kernel 内 `pypto.loop` over experts（`llada2_moe_grouped_gemm_impl.py`）

```python
@pypto.frontend.jit(
    runtime_options={"device_sched_mode": 1, "stitch_function_max_num": 64},
    pass_options={
        "cube_l1_reuse_setting": {-1: 2},
        "cube_nbuffer_setting": {-1: 2},
        "vec_nbuffer_setting": {-1: 2},
    },
)
def grouped_gemm_kernel(
    sorted_tokens: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16, format=ND),
    w13_flat:      pypto.Tensor([], pypto.DT_BF16, format=ND),
    w2_flat:       pypto.Tensor([], pypto.DT_BF16, format=ND),
    expert_cumsum: pypto.Tensor([], pypto.DT_INT32, format=ND),
    result:        pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16, format=ND),
    num_experts:   int,
    hidden_size:   int,
    intermediate_size: int,
):
    two_i = intermediate_size * 2
    pypto.experimental.set_operation_options(combine_axis=True)

    for e_idx in pypto.loop(0, num_experts, 1,
                            name="EXPERT_LOOP", idx_name="e_idx"):
        e_start = expert_cumsum[e_idx]
        e_end = expert_cumsum[e_idx + 1]
        n_e = e_end - e_start

        for tok_idx, tile_batch in pypto.loop_unroll(
            0, n_e, 1,
            name="LOOP_TOKEN", idx_name="tok_idx",
            unroll_list=[1, 2, 4, 8, 16, 32, 64],
        ):
            pypto.set_cube_tile_shapes(...)
            w13_e = pypto.view(w13_flat, [hidden_size, two_i],
                               [e_idx * hidden_size, 0])
            w2_e = pypto.view(w2_flat, [intermediate_size, hidden_size],
                              [e_idx * intermediate_size, 0])
            tile_x = pypto.view(sorted_tokens, [tile_batch, hidden_size],
                                [e_start + tok_idx, 0])

            gate_up = pypto.matmul(tile_x, w13_e, pypto.DT_FP32)

            pypto.set_vec_tile_shapes(min(tile_batch, 13), intermediate_size)
            sw = _swiglu_silu(gate_up)
            sw = pypto.cast(sw, pypto.DT_BF16)

            pypto.set_cube_tile_shapes(...)
            down = pypto.matmul(sw, w2_e, pypto.DT_FP32)
            pypto.set_vec_tile_shapes(min(tile_batch, 13), hidden_size)
            out = pypto.cast(down, pypto.DT_BF16)
            pypto.assemble(out, [e_start + tok_idx, 0], result)
```

**Host 侧准备**（weight 重排 + token 排序）：

```python
# Weight flattening (done once at init)
w13_flat = torch.cat([w13[e] for e in range(E)], dim=0)  # [E*H, 2I]
w2_flat  = torch.cat([w2[e] for e in range(E)], dim=0)   # [E*I, H]

# Token sorting (done per forward)
sort_indices = topk_ids.argsort(stable=True)
sorted_tokens = hidden_states[sort_indices]
expert_cumsum = torch.zeros(E+1, dtype=torch.int32, device=dev)
for e in range(E):
    expert_cumsum[e+1] = expert_cumsum[e] + (topk_ids == e).sum()
```

**结果**：E=64 时，5,972 us → 584 us（**10.2x 加速**）。

---

### 案例：Gemma-4 GeGLU FFN — 静态 range() → pypto.loop

**模型**：Gemma-4-31B-it fused GeGLU activation kernel

**场景**：intermediate_size = 21504 = 3072 × 7，内层 tile 循环 trip count 编译时已知（N_TILES = 7）。

**转换前**：Python `range()` 内层 tile 循环（`geglu_ffn_original.py`）

```python
I_DIM   = 21504
I_TILE  = 3072
N_TILES = I_DIM // I_TILE   # 7

@pypto.frontend.jit
def geglu_ffn_kernel(
    gate_proj: pypto.Tensor([pypto.DYNAMIC, I_DIM], pypto.DT_BF16),
    up_proj:   pypto.Tensor([pypto.DYNAMIC, I_DIM], pypto.DT_BF16),
    out:       pypto.Tensor([pypto.DYNAMIC, I_DIM], pypto.DT_BF16),
):
    M = gate_proj.shape[0]
    for m_idx in pypto.loop(M, name="M_LOOP", idx_name="m_idx"):
        for t in range(N_TILES):                          # <-- Python range
            i_off = t * I_TILE
            g_tile = pypto.view(gate_proj, [ROW_TILE, I_TILE],
                                [m_idx, i_off], valid_shape=[ROW_TILE, I_TILE])
            u_tile = pypto.view(up_proj,   [ROW_TILE, I_TILE],
                                [m_idx, i_off], valid_shape=[ROW_TILE, I_TILE])
            res = _geglu_tile(g_tile, u_tile)
            pypto.assemble(res, [m_idx, i_off], out)
```

**转换后**：`pypto.loop` 静态循环（`geglu_ffn_converted.py`）

```python
@pypto.frontend.jit
def geglu_ffn_kernel(
    gate_proj: pypto.Tensor([pypto.DYNAMIC, I_DIM], pypto.DT_BF16),
    up_proj:   pypto.Tensor([pypto.DYNAMIC, I_DIM], pypto.DT_BF16),
    out:       pypto.Tensor([pypto.DYNAMIC, I_DIM], pypto.DT_BF16),
):
    M = gate_proj.shape[0]
    for m_idx in pypto.loop(M, name="M_LOOP", idx_name="m_idx"):
        for t_idx in pypto.loop(0, N_TILES, 1,            # <-- pypto.loop
                                name="TILE_LOOP", idx_name="t_idx"):
            i_off = t_idx * I_TILE
            g_tile = pypto.view(gate_proj, [ROW_TILE, I_TILE],
                                [m_idx, i_off], valid_shape=[ROW_TILE, I_TILE])
            u_tile = pypto.view(up_proj,   [ROW_TILE, I_TILE],
                                [m_idx, i_off], valid_shape=[ROW_TILE, I_TILE])
            res = _geglu_tile(g_tile, u_tile)
            pypto.assemble(res, [m_idx, i_off], out)
```

**转换要点**：

1. N_TILES = 7 是编译时常量（`int`，非 tensor） → 使用 `pypto.loop`（非 `loop_unroll`）
2. `t` (Python int) → `t_idx` (SymbolicScalar)：索引偏移 `t * I_TILE` → `t_idx * I_TILE` 自动兼容
3. 外层 `pypto.loop(M)` 已有，仅需转换内层

**结果**：

| Shape | Original (range) | Converted (pypto.loop) | Speedup | Accuracy |
|-------|-----------------|----------------------|---------|----------|
| [4, 21504] | 300 us (p10=289, p90=335) | 299 us (p10=294, p90=309) | **1.00x** | PASS (diff=0.0) |

- **吞吐**：1.00x — 静态 trip count 下无调度开销差异，符合预期
- **方差**：p90-p10 从 46 us → 15 us（**3x 改善**）— 循环完全在 NPU 侧执行，消除 host-device 同步抖动
- **精度**：bit-exact（max diff = 0.0）

---

## 约束与注意事项

### 必须遵守

1. **`pypto.loop` 返回 SymbolicScalar**：不能用作 Python list 索引。用 `pypto.view` 代替 `tensor[sym]`。
2. **每次迭代重设 tile shapes**：`pypto.loop_unroll` 每次迭代 `tile_batch` 可能不同，必须在 loop body 开头调用 `set_cube_tile_shapes` / `set_vec_tile_shapes`。
3. **输出必须用 `pypto.assemble`**：不能 `result[i] = out`，必须 `pypto.assemble(out, offsets, result)`。
4. **`unroll_list` 必须覆盖所有可能的 tile 大小**：从 1 到预期最大 batch，按 2 的幂递增。通常 `[1, 2, 4, 8, 16, 32, 64]`。
5. **动态轴必须标注 `pypto.DYNAMIC`**：loop 涉及的 tensor 动态维度在注解中标为 `pypto.DYNAMIC`。
6. **`name` 和 `idx_name` 必须全局唯一**：嵌套 loop 的 name/idx_name 不能重复。

### 禁止

1. **禁止在 kernel 内使用 Python `range()`**：`range(symbolic_scalar)` 会抛 TypeError。
2. **禁止 `pypto.loop(1)` 或 `pypto.loop(常量)` 作为空循环**：门禁 OL43 检查。
3. **禁止 loop 内 `result = out`（rebind）**：必须用 `result[:] = out` 或 `pypto.assemble`。
4. **禁止将 weight reshape 放在 loop 内部**：静态 weight 的 view 可以在 loop 外完成（如果偏移不依赖 loop 变量）。

---

## Checklist

1. [ ] 所有 Python `for ... in range(...)` 循环已替换为 `pypto.loop` 或 `pypto.loop_unroll`
2. [ ] `loop`/`loop_unroll` 的 `name` 和 `idx_name` 参数已填写且唯一
3. [ ] 动态维度（`tensor.shape[dim]`）使用 `pypto.loop_unroll` + `unroll_list`
4. [ ] 静态维度（int 参数）使用 `pypto.loop`
5. [ ] 循环内索引操作已替换为 `pypto.view` + `pypto.assemble`
6. [ ] 每次 loop 迭代开头已调用 `set_cube_tile_shapes` / `set_vec_tile_shapes`
7. [ ] kernel 函数的 tensor 注解中动态轴标注为 `pypto.DYNAMIC`
8. [ ] 精度测试通过（`[PRECISION_PASS]`）
9. [ ] 性能对比完成（转换前 vs 转换后延迟）
