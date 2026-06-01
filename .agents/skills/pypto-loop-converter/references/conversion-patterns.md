# Loop 转换模式速查表

## 1. pypto.loop — 静态/编译时已知 trip count

### API 签名

```python
for idx in pypto.loop(start, end, step, name="LOOP_NAME", idx_name="idx"):
    # idx 是 SymbolicScalar，不是 Python int
    ...
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `start` | int / SymbolicScalar | 起始值（含） |
| `end` | int / SymbolicScalar | 结束值（不含） |
| `step` | int | 步长 |
| `name` | str | loop 节点名称，全局唯一 |
| `idx_name` | str | 迭代变量名称 |

### 适用场景

- `num_experts: int` 参数传入的循环
- 固定层数循环
- 编译时已知的维度遍历

### 示例

```python
# Expert 维度循环（num_experts 是 int 参数）
for e_idx in pypto.loop(0, num_experts, 1,
                        name="EXPERT_LOOP", idx_name="e_idx"):
    w_e = pypto.view(w_flat, [H, I], [e_idx * H, 0])
    # ...
```

---

## 2. pypto.loop_unroll — 动态 trip count + 自动尾块处理

### API 签名

```python
for idx, tile_size in pypto.loop_unroll(
    start, end, step,
    name="LOOP_NAME",
    idx_name="idx",
    unroll_list=[1, 2, 4, 8, 16, 32, 64],
):
    # idx: 当前偏移（SymbolicScalar）
    # tile_size: 当前 tile 大小（来自 unroll_list）
    ...
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `start` | int / SymbolicScalar | 起始值（含） |
| `end` | SymbolicScalar | 结束值（不含），通常来自 `tensor.shape[dim]` |
| `step` | int | 步长（通常为 1） |
| `name` | str | loop 节点名称 |
| `idx_name` | str | 迭代变量名称 |
| `unroll_list` | list[int] | tile 大小列表，从小到大排列 |

### unroll_list 工作原理

编译器为 `unroll_list` 中每个值生成一个特化版本的 loop body。运行时根据剩余元素数选择最大可用 tile：

```
总共 37 个元素，unroll_list=[1, 2, 4, 8, 16, 32, 64]

第 1 次迭代: tile=32, 处理 [0:32)
第 2 次迭代: tile=4,  处理 [32:36)
第 3 次迭代: tile=1,  处理 [36:37)
```

### 适用场景

- batch 维度遍历（`tensor.shape[0]` 是动态的）
- per-expert token 数（来自 cumsum 差值）
- 任何运行时才确定的循环次数

### 推荐 unroll_list

| 场景 | 推荐 unroll_list |
|------|-----------------|
| batch 维度 (≤64) | `[1, 2, 4, 8, 16, 32, 64]` |
| batch 维度 (≤128) | `[1, 2, 4, 8, 16, 32, 64, 128]` |
| sequence 维度 (≤512) | `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]` |
| 小范围 (≤8) | `[1, 2, 4, 8]` |

---

## 3. 索引操作转换

### Python indexing → pypto.view

```python
# BEFORE: Python slice
tile = x[i:i+batch, :]

# AFTER: pypto.view
tile = pypto.view(x, [batch, W], [i, 0])
# 如果 batch 是动态的，需要 valid_shape:
tile = pypto.view(x, [batch, W], [i, 0], valid_shape=[batch, W])
```

### pypto.view 参数

```python
pypto.view(
    tensor,           # 源 tensor
    shape,            # [行数, 列数] — tile 的形状
    offsets,          # [行偏移, 列偏移] — 起始位置
    valid_shape=None, # 动态场景下的有效形状
)
```

### 输出写回 → pypto.assemble

```python
# BEFORE
result[i:i+batch, :] = out

# AFTER
pypto.assemble(out, [i, 0], result)
```

---

## 4. tile shapes 设置

### 规则：每次 loop 迭代开头必须设置

```python
for bs_idx, tile_batch in pypto.loop_unroll(...):
    # 1) Cube tile shapes（matmul 前）
    pypto.set_cube_tile_shapes(
        [tile_batch, tile_batch],       # M 轴（batch）
        [K_tile, K_tile * 2],           # K 轴
        [N_tile, N_tile],               # N 轴
        True,                           # l1_reuse
    )

    gate_up = pypto.matmul(tile_x, w13, pypto.DT_FP32)

    # 2) Vec tile shapes（activation 前）
    pypto.set_vec_tile_shapes(
        min(tile_batch, VEC_FIRST),     # 行方向 tile
        intermediate_size,              # 列方向 width
    )

    activated = activation_fn(gate_up)

    # 3) 切换 cube tile shapes（第二个 matmul）
    pypto.set_cube_tile_shapes(
        [tile_batch, tile_batch],
        [K2_tile, K2_tile * 2],
        [N2_tile, N2_tile],
        False,
    )
    down = pypto.matmul(activated, w2, pypto.DT_FP32)
```

### UB 限制

vec tile shapes 受 192 KB UB 限制：

```
VEC_FIRST * width * num_intermediates * sizeof(FP32) < 192 KB

SiLU 需要 ~7 个 FP32 中间变量:
  VEC_FIRST < 192000 / (28 * width)

示例:
  width=512  → VEC_FIRST ≤ 13
  width=4096 → VEC_FIRST ≤ 1
```

---

## 5. SymbolicScalar 陷阱

### 不可用作 Python list 索引

```python
# WRONG: TypeError: list indices must be integers
weights = [w1, w2, w3, w4]
for e_idx in pypto.loop(0, 4, 1, ...):
    w = weights[e_idx]      # SymbolicScalar 不是 int

# RIGHT: flatten 所有 weights 到单个 tensor，用 pypto.view 切片
w_flat = torch.cat(weights, dim=0)
for e_idx in pypto.loop(0, 4, 1, ...):
    w = pypto.view(w_flat, [H, I], [e_idx * H, 0])
```

### 不可用于 Python 条件判断

```python
# WRONG
for e_idx in pypto.loop(0, E, 1, ...):
    if n_tokens > 0:        # SymbolicScalar 不支持 Python bool 判断
        ...

# RIGHT: 使用 pypto.cond 或确保 loop body 对空 slice 是安全的
```

---

## 6. 嵌套 Loop 模式

### 外 pypto.loop + 内 pypto.loop_unroll

```python
# 外层：expert 维度（固定）
for e_idx in pypto.loop(0, num_experts, 1,
                        name="EXPERT_LOOP", idx_name="e_idx"):
    e_start = expert_cumsum[e_idx]
    e_end = expert_cumsum[e_idx + 1]
    n_e = e_end - e_start

    # 内层：token 维度（动态）
    for tok_idx, tile_batch in pypto.loop_unroll(
        0, n_e, 1,
        name="LOOP_TOKEN", idx_name="tok_idx",
        unroll_list=[1, 2, 4, 8, 16, 32, 64],
    ):
        # 绝对偏移 = e_start + tok_idx
        tile = pypto.view(x, [tile_batch, H], [e_start + tok_idx, 0])
        # ...
        pypto.assemble(out, [e_start + tok_idx, 0], result)
```

### name 唯一性

嵌套 loop 的 `name` 和 `idx_name` 必须互不相同：

```python
# WRONG
for i in pypto.loop(0, N, 1, name="LOOP", idx_name="idx"):
    for j in pypto.loop_unroll(0, M, 1, name="LOOP", idx_name="idx", ...):  # 重复！

# RIGHT
for i in pypto.loop(0, N, 1, name="LOOP_OUTER", idx_name="outer_idx"):
    for j in pypto.loop_unroll(0, M, 1, name="LOOP_INNER", idx_name="inner_idx", ...):
```

---

## 7. Host 侧数据准备模式

当将 Python 循环下沉到 kernel 时，通常需要在 host 侧预处理数据：

### Weight Flattening

```python
# 从 per-expert weight list 转为 flattened tensor
w13_list = [expert.gate_up_weight for expert in experts]  # E 个 [H, 2I]
w13_flat = torch.cat(w13_list, dim=0)                      # [E*H, 2I]

# kernel 内用 pypto.view 按 expert index 切片
w13_e = pypto.view(w13_flat, [H, two_i], [e_idx * H, 0])
```

### Token Sorting

```python
# 按 expert assignment 排序 tokens
sort_indices = topk_ids.flatten().argsort(stable=True)
sorted_tokens = hidden_states[sort_indices]

# 计算 cumulative sum 作为 expert 边界
expert_cumsum = torch.zeros(E + 1, dtype=torch.int32, device=dev)
for e in range(E):
    expert_cumsum[e + 1] = expert_cumsum[e] + (topk_ids == e).sum()
```
