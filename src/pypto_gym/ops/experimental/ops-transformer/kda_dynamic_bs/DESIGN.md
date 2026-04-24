---
schema_version: "2.1"
op_name: "kda_dynamic_bs"
status: reviewed
last_updated: "2026-04-17"
compute_kind: "vector"
dtypes: ["fp32"]
dynamic_axes: ["B", "S"]
precision: { rtol: 1e-3, atol: 1e-3 }
---

# kda_dynamic_bs 设计方案

## 1. 计算图与精度路由

### 1.1 API 调用序列

| 步骤 | 操作 | PyPTO API | 输入 dtype | 输出 dtype | 输出 shape | 备注 |
|---|---|---|---|---|---|---|
| 1 | 切片取 token | `pypto.view` | FP32 | FP32 | `[1,1,D]` | `B/S` 为 SymbolicScalar offset |
| 2 | 形状变换 | `pypto.reshape` | FP32 | FP32 | `[D,1]` / `[1,D]` | 用于广播计算 |
| 3 | 单轴扩展 | `pypto.expand_clone` | FP32 | FP32 | `[D,D]` | 仅一维从 1 扩展 |
| 4 | 外积与状态更新 | `pypto.mul` + `pypto.add` | FP32 | FP32 | `[D,D]` | `state = state*alpha + outer*beta` |
| 5 | 输出归约 | `pypto.sum(dim=0, keepdim=True)` | FP32 | FP32 | `[1,D]` | `sum` 限制为 FP32 |
| 6 | 写回 | `pypto.assemble` | FP32 | FP32 | `output`/`last_state` | 显式写回 |

### 1.2 精度路由

```text
输入(FP32) → 递推状态更新(FP32) → 归约(FP32) → 输出(FP32)
```

| 转换位置 | 转换方向 | 原因 |
|---|---|---|
| 无 | 无 | 本实现全链路 FP32，避免额外 cast |

### 1.3 替代方案（已排除）

| 替代方案 | 排除原因 |
|---|---|
| 用 `matmul(k_col, v_row)` 计算外积 | `K=1` 场景下 cube 对齐约束复杂，v1 先走 vector 路径 |
| 4D/3D 直接整块计算 | 双动态轴下更容易触发 shape/tiling 复杂约束 |

---

## 2. 数据规格

### 2.1 Kernel 函数签名

```python
@pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
def kda_kernel(
    query: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D], pypto.DT_FP32),
    key: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D], pypto.DT_FP32),
    value: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D], pypto.DT_FP32),
    alpha: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D], pypto.DT_FP32),
    beta: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC, D], pypto.DT_FP32),
    last_state: pypto.Tensor([pypto.DYNAMIC, D, D], pypto.DT_FP32),
):
    ...
```

### 2.2 动态轴分析

| 维度名 | 是否动态 | 取值范围 / 常量 | 标注方式 |
|---|---|---|---|
| `B` | 是 | 运行时决定 | `pypto.DYNAMIC` |
| `S` | 是 | 运行时决定 | `pypto.DYNAMIC` |
| `D` | 否 | 编译期常量 | Python int |

### 2.3 值类型分析（避免 SymbolicScalar 误用）

| 变量 | 来源 | 类型 | 注意事项 |
|---|---|---|---|
| `b_dyn` | `query.shape[0]` | SymbolicScalar | 只能用于 `pypto.loop`/offset |
| `s_dyn` | `query.shape[1]` | SymbolicScalar | 不能用于 Python `range` |
| `d` | wrapper 入参 | int | 用于 `shape` 与 tile 配置 |

---

## 3. Tiling 策略

### 3.1 算子类型

Vector

### 3.2 Tiling 推导

- **同时驻留 UB 的 Tensor**：

| Tensor | 用途 | shape | dtype | 大小估算 |
|---|---|---|---|---|
| `state` | 递推状态 | `[D,D]` | FP32 | `D*D*4` bytes |
| `outer` | 当前外积 | `[D,D]` | FP32 | `D*D*4` bytes |
| `alpha_expand` | 衰减扩展 | `[D,D]` | FP32 | `D*D*4` bytes |
| `beta_expand` | 更新扩展 | `[D,D]` | FP32 | `D*D*4` bytes |
| `weighted_state` | 输出前中间量 | `[D,D]` | FP32 | `D*D*4` bytes |

- **推导步骤**：
  1. dtype 为 FP32，按 vector 处理
  2. v1 先用单 tile `[D, D]`，便于验证正确性
  3. 如后续出现 UB 压力，再拆分 `D` 轴分块

- **最终 tile**：

```python
pypto.set_vec_tile_shapes(D, D)
```

### 3.3 替代方案

| 备选 tile | 否决理由 |
|---|---|
| `[64, 64]` 固定 tile + 额外内层分块 | 复杂度更高，v1 先保证可读性与正确性 |

---

## 4. Loop 与数据流

### 4.1 维度判定

| 轴 | 维度大小 | 编译期 / 运行期 | Loop 处理 |
|---|---|---|---|
| `B` | DYNAMIC | 运行期 | `pypto.loop(b_dyn)` |
| `S` | DYNAMIC | 运行期 | `pypto.loop(s_dyn, unroll_list=[16,1])` |
| `D` | 常量 | 编译期 | 不建 loop |

### 4.2 完整伪代码

```python
@pypto.frontend.jit
def kda_kernel(query, key, value, alpha, beta, output, last_state):
    b_dyn = query.shape[0]  # SymbolicScalar
    s_dyn = query.shape[1]  # SymbolicScalar

    for b_idx in pypto.loop(b_dyn, name="LOOP_B_KDA", idx_name="b_idx"):
        pypto.set_vec_tile_shapes(D, D)
        state = pypto.full(size=[D, D], fill_value=0.0, dtype=pypto.DT_FP32)

        for s_idx in pypto.loop(s_dyn, name="LOOP_S_KDA", idx_name="s_idx", unroll_list=[16, 1]):
            q_3d = pypto.view(query, [1, 1, D], [b_idx, s_idx, 0], valid_shape=[1, 1, D])
            k_3d = pypto.view(key, [1, 1, D], [b_idx, s_idx, 0], valid_shape=[1, 1, D])
            v_3d = pypto.view(value, [1, 1, D], [b_idx, s_idx, 0], valid_shape=[1, 1, D])
            a_3d = pypto.view(alpha, [1, 1, D], [b_idx, s_idx, 0], valid_shape=[1, 1, D])
            b_3d = pypto.view(beta, [1, 1, D], [b_idx, s_idx, 0], valid_shape=[1, 1, D])

            q_col = pypto.reshape(q_3d, [D, 1], valid_shape=[D, 1])
            k_col = pypto.reshape(k_3d, [D, 1], valid_shape=[D, 1])
            v_row = pypto.reshape(v_3d, [1, D], valid_shape=[1, D])
            a_col = pypto.reshape(a_3d, [D, 1], valid_shape=[D, 1])
            b_col = pypto.reshape(b_3d, [D, 1], valid_shape=[D, 1])

            k_expand = pypto.expand_clone(k_col, [D, D])
            v_expand = pypto.expand_clone(v_row, [D, D])
            outer = k_expand * v_expand

            a_expand = pypto.expand_clone(a_col, [D, D])
            b_expand = pypto.expand_clone(b_col, [D, D])
            state = state * a_expand + outer * b_expand

            q_expand = pypto.expand_clone(q_col, [D, D])
            weighted_state = state * q_expand
            out_row = pypto.sum(weighted_state, dim=0, keepdim=True)  # [1, D]
            out_3d = pypto.reshape(out_row, [1, 1, D], valid_shape=[1, 1, D])
            pypto.assemble(out_3d, [b_idx, s_idx, 0], output)

        state_3d = pypto.reshape(state, [1, D, D], valid_shape=[1, D, D])
        pypto.assemble(state_3d, [b_idx, 0, 0], last_state)
```

### 4.3 跨迭代状态

| 状态名 | 初始化 | 更新方式 | submit_before_loop |
|---|---|---|---|
| `state` | `full([D,D], 0.0)` | `state = state * a + outer * b` | 否 |

### 4.4 尾块处理

- `B/S` 以单 token 视图处理（`[1,1,D]`），不需要额外尾块 padding

---

## 5. 约束自检清单

| # | 约束 | 是否满足 | 备注 |
|---|---|---|---|
| 1 | 所有 sum 输入已转 FP32 | ✓ | 全链路 FP32 |
| 2 | matmul 两侧 dtype 一致 | N/A | v1 无 matmul 主路径 |
| 3 | TileShape 维度数 = 操作数维度数 | ✓ | `[D,D]` |
| 4 | 尾轴满足对齐 | ✓ | Vector 路径 |
| 5 | 同阶段 UB 占用 ≤ 容量 | ⚠ | `D` 过大需二次调优 |
| 6 | 表达式展开 < 18000 | ✓ | 结构简单 |
| 7 | 输出经 `[:]` / `assemble` 显式写回 | ✓ | 使用 `assemble` |
| 8 | 无 view/assemble 同张量回环 | ✓ | 输入只读，输出单向写 |
| 9 | 动态轴标 `pypto.DYNAMIC` | ✓ | `B/S` 已标注 |
| 10 | 动态 loop 提供 `unroll_list` | ✓ | `S` 轴设置 |
| 11 | 跨迭代状态用 `submit_before_loop=True` | N/A | 无显式跨图依赖要求 |
| 12 | 尾块用 `valid_shape` 处理 | ✓ | `view/reshape` 显式写入 |
| 13 | 无 SymbolicScalar 用作 `**` / list index / Python `if` | ✓ | 遵守 |

### 开放问题

| # | 问题 | 影响范围 | 待解决方式 |
|---|---|---|---|
| 1 | `D` 很大时 vector 实现性能可能不足 | 大模型长维度场景 | Stage 7 进入 cube/混合优化 |

---

## 6. 验证方案

### 6.1 测试配置

| 用例 | 输入 shape | dtype | 重点验证 |
|---|---|---|---|
| Case 1 | `[1, 16, 64]` | FP32 | 基础正确性 |
| Case 2 | `[2, 31, 64]` | FP32 | 非整齐 S 的动态行为 |
| Case 3 | `[4, 127, 64]` | FP32 | 更大动态范围 |

### 6.2 精度容忍度

| dtype | rtol | atol |
|---|---|---|
| FP32 | 1e-3 | 1e-3 |
