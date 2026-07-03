# PyPTO-Pro Kernel 编码常见问题与避坑指南

> **目标读者**：pypto-pro-op-develop 子代理。

---

## 1. 编译期问题

### 1.1 `pl.Scalar` 中间变量赋值可能触发 `'Var' object is not iterable`

**症状**：
```
TypeError: 'pypto.pypto_impl.ir.Var' object is not iterable
```
发生在 kernel 编译阶段（JIT 解析时），非运行时错误。

**根因**：`pl.Scalar` 表达式（包括 `pl.min()`、`pl.max()`、`a // 2` 等）赋值给 Python 局部变量时，某些场景下 JIT AST 解析器会错误地尝试将其展开为可迭代对象，触发上述错误。

**重要**：赋值给中间变量**通常可以编译通过**——pro_ops FA perf 样例中存在大量此类写法且均编译通过（如 `actual_sq = pl.min(...)`、`actual_sq_half = actual_sq // 2`、`actual_sq_pad = ((actual_sq + 31) // 32) * 32` 等，见 EXPLORE_REPORT §4 定位的尾块处理样例）。

**修复方法**：若编译报 `'Var' object is not iterable`，将触发错误的赋值表达式改为直接内联到 `pl.*` API 调用参数中：

```python
# 若此行触发 "not iterable"：
actual_sq_half = actual_sq // 2
pl.set_validshape(qk_vec, actual_skv, actual_sq_half)

# 改为直接内联：
pl.set_validshape(qk_vec, actual_skv, actual_sq // 2)
```

**适用范围**：此问题可能影响所有返回 `pl.Scalar` 或 `pl.Scalar` 表达式的 API，包括 `pl.min()`、`pl.max()`、`pl.add()`、`pl.sub()`、`pl.mul()`、`pl.div()`、`pl.exp()`、`pl.sel()`、`pl.cmp()`、`a // 2` 等。**`pl.range()` 的参数中使用中间变量是安全的**（如 `for ki in pl.range(0, skv_tiles)` 中的 `skv_tiles`），因为 `pl.range()` 的参数是编译期求值的。

**参考**：
- EXPLORE_REPORT §4 定位的尾块处理样例 — 中间变量赋值合法编译
- EXPLORE_REPORT §4 定位的 unalign 样例 — `is_tail` 模式

---

### 1.2 tile 常量写进 kernel 函数内 → `Unsupported kwarg type for key: memref_size`

**症状**：
```
Unsupported kwarg type for key: memref_size
```
JIT 解析器拒绝 kernel 函数，编译失败（非运行时错误）。

**根因**：PyPTO-Pro JIT 把 kernel 函数体内所有 Python 赋值语句当作 PyPTO IR 语句处理。`TILE_SIZE = 32768` 写在函数内部后，`pl.make_tile(tt, size=TILE_SIZE)` 中对该变量的引用不被支持（IR 层拿到的是一个非法 kwarg 类型）。

**修复**：所有 tile 尺寸 / 公式编译期常量移到模块级（`@pl.jit` 装饰器之前）。参考 pro_ops `test_add.py`。

**设计阶段预防**：design SKILL R4 步骤 6 + design-template §4 骨架已标注"常量声明必须在 kernel 函数外"。

---

## 2. 运行时问题

### 2.1 `valid_shape` 可能跨迭代残留——每轮外层迭代前重置持久化 tile 的有效尺寸

**症状**：kernel 不挂死，但输出出现大面积零值或错误值。常见特征：
- 某些 tile 维度整除时通过，存在尾块时失败
- 第一个循环迭代正确，后续迭代输出错误
- 多轮外层迭代场景下，某些迭代明显继承了上一轮尾块的有效区域

**根因**：`valid_shape` 可能在迭代间持久化。当某次迭代设置 `set_validshape(tile, tail_c, tail_w)` 处理尾块后，若下一次迭代未重新设置，该 tile 可能仍使用上次的尾块尺寸，导致**本该加载完整 tile 却只读了尾块大小的数据**，或本应作用于全尺寸 tile 的计算只覆盖局部区域。

**主规则**：凡是跨外层迭代持久化在 UB 中的 tile，无论是否重新 `load_tile`，每轮外层迭代开始时都应显式 `set_validshape` 重置为全尺寸；若本轮存在尾块，再在具体读写/计算前设置本次实际有效尺寸。

**正确写法 1**（每次 `load_tile` 前显式设置本次实际有效尺寸）：

```python
# ✅ 正确：load_tile 前显式设置本次实际有效尺寸
pl.set_validshape(tile_x, valid_c, valid_w)  # 本次实际有效区域
pl.load_tile(tile_x, x, [b, c_tile, h, w_tile], tile_dims=[1, 3])
```

**扩展场景**：持久化系数/scratch tile 的 `valid_shape` 跨外层迭代残留

同样原理适用于**不经过 `load_tile` 但跨外层迭代持久化在 UB 中的 tile**。`valid_shape` 可能作用于对该 tile 的**所有操作**（含 `expands`、`row_expand_mul`、`mul`、`add` 等），不仅是 `load_tile`。

一个典型现象是：某轮尾块把系数 tile 设成窄尺寸，下一轮未重置即继续参与计算，导致操作仍只作用于局部区域，其余位置读到垃圾数据。

**正确写法 2**（外层迭代开始时先重置持久化 tile，再按需设置尾块）：

```python
# ❌ 错误：上一轮尾块设窄了 coeff 的 valid_shape，下一轮未重置
for c in pl.range(core_id, C_val, num_cores):
    # 加载系数到 coeff_a/coeff_b ...
    for n_tile in pl.range(...):
        pl.set_validshape(coeff_a, valid_m, 1)     # tail 轮设为 (2, 1)
        pl.row_expand_mul(out, x, coeff_a)          # 下一 channel：coeff_a 仍是 (2, 1)！
    pl.system.bar_all()

# ✅ 正确：每个 channel 迭代开始时重置系数 tile
for c in pl.range(core_id, C_val, num_cores):
    # 加载系数到 coeff_a/coeff_b ...
    pl.set_validshape(coeff_a, TILE_M, 1)           # ⚡ 重置到全尺寸
    for n_tile in pl.range(...):
        pl.set_validshape(coeff_a, valid_m, 1)     # 再设本次实际尾块尺寸（如需）
        pl.row_expand_mul(out, x, coeff_a)
    pl.system.bar_all()
```

---

### 2.2 精度错误但 kernel 不挂死

**症状**：kernel 跑完不报错，但 `max|diff|` 远大于预期精度阈值（如 atol=5e-3 时差值达数十），不匹配率很高。

**检查清单**：

1. **padding 维度与实际计算维度是否一致？**
   - Cube tile 的方形对齐约束（如 D 需要 pad 到 128）会导致物理维度 ≠ 逻辑维度
   - 所有依赖维度值的常量（如 `scale = 1/sqrt(d)`）必须用**逻辑维度**（pad 前的真实值），不能用 tile 维度
     - DESIGN.md §2.1 已定义逻辑维度与 pad 维度的常量表，kernel 中的 scale/norm 类常量全部引用逻辑维度
2. **是否含 exp/log/sqrt 等超越函数导致溢出→NaN？**
   - 症状特征：`Greatest absolute difference: nan`，少量元素触发（溢出元素占比小但污染对应位置）
   - 思路：低精度 dtype（如 fp16）的表示上限远低于 fp32，超越函数（exp 尤甚）在较小的输入下即溢出为 +inf，后续 `inf/inf` → NaN 传播
   - 修复思路：在超越函数前用 `pl.mins`/`pl.maximum` 把输入截断到"该 dtype 下不溢出的安全区间"。安全阈值须按**目标 dtype 上限 + 具体公式**反推（如 exp 的阈值 ≈ ln(dtype_max)），不套用固定数值。设计阶段预防见 design §1 数值安全边界

---

### 2.3 `tile_dims` 最外层维度 stride 过大 → 507035 / 507015 🚨

> 本节为**诊断清单**。tile_dims stride 的设计阶段预防详见 design skill R2 步骤 4（检查表 + 修复方法）和 design-template.md §2.1。

**症状**：整除 shape 通过，尾块 shape **运行时直接挂死**，报 aicore / vector core 异常：
- 错误码 507015（`AICORE_EXCEPTION`，头文件 `device_error_code.h`）
- 错误码 507035（`VECTOR_CORE_EXCEPTION`，头文件 `device_error_code.h`）

> 注：507015/507035 是通用的 aicore/vector-core 异常码，并非本坑的专属错误码；本坑只是其常见触发原因之一。专属的读越界码是 507042/507044（`*_TRAP_READ_OVERFLOW`），但实际报错以框架抛出的 507015/507035 更常见。

**根因**：`load_tile(tile, tensor, [...], tile_dims=[d0, d1])` 中，DMA 以 dim d0 作为"行"维度遍历。若 d0 在张量布局中的 **stride**（行间跳跃字节数）过大，尾块 DMA 读取地址远超物理分配范围，触发硬件异常。

**诊断步骤**（与设计 R2 步骤 4 对称）：
1. 确认整除 shape PASS、尾块 FAIL，且错误码为 507035 或 507015
2. 计算 `tile_dims` 第一个维度在张量布局中的 stride = 该维度之后的所有维度乘积 × dtype 字节数
3. 若 stride 超过 EXPLORE_REPORT §7 探测的 stride 经验阈值，且该维度存在尾块 → 根因确认
4. 修复：permute 张量使 tile_dims 最外层维度的 stride 最小化（详见 design skill R2 步骤 4）

---

### 2.4 Acc 物理 tile <fractal 时的 pad 需求（注意 fractal 自动设置，问题在物理尺寸不足）

> 本条为**动态轴含极小维度时的泛化风险提示**，非固定 shape 算子的常见故障。设计阶段预防详见 design skill R2 步骤 2 的 Acc fractal 约束项与 R8 检查表。

**背景**：Acc(L0C) 的 FP32/INT32 tile 的 `fractal=1024` 由框架**自动设置**，用户**无需手写** `fractal=` 参数（证据：`$PYPTO_DEVKIT_DIR/docs/api/SIMD-API/基础数据结构/TileType.md:38` "Acc 的 FP32/INT32 自动设为 1024"；`:58` "Acc 的 FP32/INT32 自动设置 `fractal=1024`"）。因此本坑的关键**不在**"是否写了 fractal"，而在 **Acc tile 的物理 M×N 尺寸是否够一个 fractal**。

**症状**：kernel 不报参数错误，但当 matmul 的某个维度逻辑值极小（如动态轴允许 `S_q=1`、`M` 退化为 1）导致 Acc 物理 tile 的 `M×N×dtype_bytes < 1024` 时，Acc→Vec 的 `move` 可能读到不足一个 fractal 的数据，出现数据错误或搬运异常。

**根因**：L0C 以 fractal（FP32 = 1024 bytes = 256 元素）为最小搬运/累加粒度。若物理 tile 面积不足一个 fractal，硬件按 fractal 粒度读写时会越出实际有效区域。注意：满 tile 化方案（如 M pad 到 128）下 Acc 物理尺寸恒 ≥ fractal，通常不触发；仅当 tile 维度本身随极小逻辑维度退化时才出现。

**修复方法**：
1. 保证 Acc tile 的**物理** shape（而非逻辑 shape）满足一个 fractal 的最小值。例如 Acc FP32 需 `M×N ≥ 256` 元素，若 `N=128` 则 `M` 至少 pad 到 `2`。
2. pad tile shape 后，运行时用 `set_validshape` 将有效区域限制回真实逻辑尺寸，保证计算结果正确。
3. 区分**逻辑维度**（可能极小）与**物理 tile 维度**（须满足 fractal）——公式常量（如 scale）用逻辑维度，tile 声明用 pad 后的物理维度。

---

### 2.5 归约输出 `[M,1]` 未设 `layout=pl.DN`

**症状**：编译期报 layout 非法组合 `ValueError`，或运行/精度异常。

**根因**：行向归约类 API（`row_max`/`row_sum`/`row_reduce`/`row_expand_*` 等）的 `[行数,1]` 输出（及 `row_expand_*` 的 `[行数,1]` 行向量输入）**硬性要求 `layout=pl.DN`**（证据：`$PYPTO_DEVKIT_DIR/docs/api/SIMD-API/计算API/数学函数/row_max.md:25`、`row_sum.md:25`、`row_expand_sub.md:27`、`row_expand_div.md:27` 明文"须设 `layout=pl.DN`"）。声明 tile 时漏写 `layout=pl.DN` 即触发。注意 Pro 用 `layout=pl.DN`，不是老 pypto 的 `blayout=2`/`ColMajor`。（对称的列向 `col_max`/`col_sum` 输出为 `[1,列数]`，其 layout 要求以对应 API 文档为准，不套用本条。）

**修复**：归约输出 tile 声明加 `layout=pl.DN`：
```python
tile_out = pl.make_tile(pl.TileType(shape=[64, 1], dtype=pl.DT_FP32,
                                     target_memory=pl.MemorySpace.Vec, layout=pl.DN),
                        addr=..., size=256)
```
参考 `row_max.md:50-51`。设计阶段预防见 design SKILL R2 归约输出 layout 强制项。

---

### 2.6 `[M,1]` DN 归约输出直接参与逐元素运算 → 需双视图

**症状**：把 `layout=pl.DN` 的 `[M,1]` 归约输出直接传给 `sub`/`div`/`maximum` 等 tile×tile 逐元素 API，出现 layout 不匹配或结果错误。

**根因**：归约输出是 `[M,1]` DN 布局，而逐元素 tile×tile 运算的操作数需默认 `pl.ND` 布局。二者布局不同，不能用同一个 tile 变量直接跨用。

**修复**：在**同一 UB 地址**声明一对双视图——一个 `[M,1] layout=pl.DN`（供归约写），一个 `[1,M]` 默认 ND（供逐元素运算读）。归约用 DN 视图，逐元素用 ND 视图：
```python
reduce_dst    = pl.make_tile(pl.TileType(shape=[M, 1], dtype=..., layout=pl.DN), addr=VA, size=...)
reduce_dst_rm = pl.make_tile(pl.TileType(shape=[1, M], dtype=...),               addr=VA, size=...)  # 同址 ND
pl.row_max(reduce_dst, qk_vec, tmp_vec)          # DN 视图写
pl.maximum(reduce_dst_rm, reduce_dst_rm, gmax)   # ND 视图参与逐元素
```
证据：`pro_ops/fa/test_fa_performance.py:478-483`（reduce_dst/reduce_dst_rm 共用 addr）、`491-496`（gsum/gsum_rm）。设计阶段预防见 design SKILL R2 双视图触发器 + design-template §3 双视图对表。
