---
name: pypto-pro-op-design
description: Stage 3 架构设计。通过 9 轮迭代式约束收敛，基于 Stage 1 产物（SPEC.md、EXPLORE_REPORT.md），产出 DESIGN.md。核心输出为 tile 级别数据流图——决定 Phase 划分、API 映射、Tile 规划、UB 空间布局、循环与 Section 结构、分核策略、核间流水、尾块处理。每一步决策必须有 API 文档或已有样例做证据，严禁猜测。触发词：生成设计方案、tile 数据流、DESIGN.md、tile 级别设计、UB 空间规划。
---

# PyPTO-Pro Stage 3 — 迭代式方案设计

通过 9 轮迭代式约束收敛（R0-R8），目标是生成可直接翻译为 kernel 代码的 DESIGN.md。

**核心原则**：
- 每个决策必须包含**结论 + 推导过程 + 证据来源**
- 力求后续 Agent 拿到 DESIGN.md 即可确定 kernel 的完整结构与关键决策；API 签名等细节仍须由 coder 以 API 文档原文为准确认（EXPLORE_REPORT 仅为派生的先行速查，不作签名权威），运行验证暴露设计失误时可据实修正
- 每轮发现的矛盾必须回溯修正前序决策，不允许累积到 R8 再处理

## 输入

| 来源 | 路径 | 用途 |
|------|------|------|
| 算子规格 | `custom/<op>/SPEC.md` | 公式、shape、dtype、动态轴 |
| 资料探索报告 | `custom/<op>/EXPLORE_REPORT.md` | API 映射与约束（§3）、相似样例与可复用模式（§4）、教程设计指导（§5）、Tile/同步策略建议（§6）、环境常量快照（§7：UB 容量/event_id 上限/stride 阈值等） |
| 全量资料索引 | `custom/<op>/PRO_MATERIAL_INDEX.md` | API 文档（§A）、pro_ops 样例（§B）、教程（§C）的精确路径定位 |

## 输出

- **`custom/<op>/DESIGN.md`**，基于 [templates/design-template.md](templates/design-template.md)，核心交付物为 §9"Tile 数据流全景图"

---

## 前置：维度契约

在切 Phase 之前先确认 kernel 的输入/输出维度契约——这是后续所有轮（Tile 规划、循环结构）的前提，不属于 Phase 划分本身。

明确 kernel 固定处理的维度（如"kernel 只处理 2D `[M,N]`"）。若 SPEC.md 要求支持 1D 或任意多维，必须在此定义 host 端适配方案（如 1D `[L]` → host reshape `[L,1]` → kernel `[M,N]` → 输出 reshape 回 `[L]`），不能默默只实现 2D 而遗漏 SPEC 要求的其他维度。产出填入 DESIGN.md §0 的维度契约栏（kernel 固定维度 + host 适配规则表）。

---

## 迭代设计流程

设计是**问题驱动**的迭代，不是线性填表。每一轮聚焦一个核心问题，发现矛盾时回溯修正前序决策。

### R0：Phase 划分

**核心问题**：整体计算流可以分解为几个 Phase？每个 Phase 做什么？

**流程**（按优先级）：

1. **数据依赖**：需要前面 Phase 的完整结果才能开始计算的，必须划分为不同 Phase。例如跨 N-tile 的 softmax：必须先遍历所有 tile 算全局 max（Pass1），才能算 exp(x-max) 并累加全局 sum（Pass2），最后做除法（Pass3）。⚠️ **归约轴可能超单 tile 时必须走这种多 Phase/多 tile 归约（online/两遍法），禁止用超大 TILE 假设单 tile 装得下来规避——那会丧失泛化性（归约轴超出即不可用）**。多 tile 归约的具体状态更新方式随算子而异（softmax 的 running max/sum、layernorm 的 running mean/var 等），以对应 golden 与 pro_ops 样例为准
2. **Section 分隔**：Cube 和 Vector 使用不同引擎，必须划分到不同 Phase。Section 间需明确数据传递方式（via GM workspace）

**输出**：

- Phase 列表（Phase1/2/3...），每个标注：目的、输入依赖、输出、涉及的 Section
- 如果只有一个 Phase（如 N ≤ TILE_N 的 softmax），标注"单 Phase"并说明为什么不需要拆分
- 归约轴容量结论：归约轴是否可能超单 tile + 是否采用多 tile 归约（含归约算子时必填）

---

### R1：API 映射

**核心问题**：每个数学步骤具体用哪些 API？需要在 R0 的 Phase 划分基础上，将 API 调用链进一步细化到 Phase 内部的每一步操作。

**流程**：

1. 从 EXPLORE_REPORT.md §3 提取已确认的 API 映射
2. **逐一查阅每个 API 的文档原文**（通过 `PRO_MATERIAL_INDEX.md` §A 定位路径），确认并记录：
   - **功能**：API 做什么、语义是否与数学步骤完全匹配
   - **签名**：参数顺序、位置参数 vs 关键字参数（以 API 文档原文为准，EXPLORE_REPORT §3.3 已汇总部分）
   - **使用约束**：dtype 限制、shape 要求、layout 要求、MemorySpace 要求、tmp tile 是否可与输入重叠等（以 API 文档原文为准，EXPLORE_REPORT §3.3 已汇总部分）
   - **特性特点**：in-place 支持、尾块行为、与 `set_validshape` 的交互等
   - **数值安全边界**（API 链含 exp/log/sqrt/reciprocal/tanh 等超越/非线性函数时强制）：分析该 API 在目标 dtype 下的溢出边界。例如 fp16 max≈65504，exp(11.09)≈65504，输入 >11 即溢出为 +inf，后续 inf/inf → NaN 传播。须在 §1 数值安全边界栏记录"输入范围→是否溢出→防护措施"（如 exp 前插 `pl.mins(_, 11.0)` 截断）
3. 在每个 Phase 内部，将计算步骤展开为具体的 API 序列
4. 标注每个 API 的输入 tile、输出 tile
5. 若 EXPLORE_REPORT.md 中有 unsupported 步骤，在此确认替代路径或阻断

**注意**：本轮只确定计算 API（`load_tile`/`matmul`/`row_max`/`exp` 等），不涉及同步 API。同步点位置在 R4 标注，具体同步 API 由 R5/R6 填入。伪代码中同步点用占位注释 `// sync: {目的}` 标注。

**输出**：Phase 级 API 调用序列（伪代码形式），例如：

```
Phase 1 — 全局 max 归约:
  for j in range(n_tile_num):
    load_tile(tile_a, x, [i, j])          // sync: 等MTE2完成
    row_max(redu_col, tile_a, tile_tmp)   // sync: V pipe内依赖
    maximum(redu_rm, redu_rm, gmax_rm)
    muls(gmax_rm, redu_rm, 1.0)
```

---

### R2：Tile 规划

**核心问题**：需要哪些 tile？每个 tile 的 shape、dtype、layout 是什么？

**流程**：

0. 按模板 §2.1 的格式填写关键常量。列出所有 tile 尺寸（TS、TD 等）和派生常量（SCALE 等）。tile 尺寸是编译期对齐值，运行时通过 R7 的 `pl.min(D - offset, TILE)` + `set_validshape` 自动处理所有尾块场景（维度 < tile、整除、N+1 块且尾块不满）。公式常量若依赖实际维度值而非 tile 尺寸，应使用实际值
1. 根据 R1 的 API 序列，列出所有涉及的 tile
2. 对每个 tile，确定：
   - **shape**：由 API 操作数要求决定（如 `row_max(dst[M,1], src[M,N], tmp[M,N])` → 需 `[M,N]` 输入 + `[M,N]` scratch + `[M,1]` 输出）
   - **dtype**：由精度需求决定（如 matmul 累加用 FP32，存储/传输用 FP16）
   - **layout**：默认 `pl.ND`（行优先），或 `pl.DN`（维度转置）/`pl.NZ`/`pl.ZN` 等枚举，具体约束以 EXPLORE_REPORT §3.3 / API 文档为准。**术语**：Pro 用 `layout=pl.DN` 而非老 pypto 的 `blayout=2`/`ColMajor`（证据：`TileType.md:24`）
   - **归约输出 layout 强制**（🚨 常见遗漏）：行向归约类 API（row_max/row_sum/row_reduce/row_expand_* 等）的 `[行数,1]` 输出**须设 `layout=pl.DN`**（证据：`row_max.md:25`、`row_sum.md:25`、`row_expand_sub.md:27` 明文"须设 `layout=pl.DN`"；对称的列向 col_* 输出为 `[1,列数]`，layout 要求以其 API 文档为准，不套用此规则）。**若该归约输出后续需参与 tile×tile 逐元素运算**（需默认 ND 布局），**必须在同地址声明 DN + ND 双视图对**——在 §3 双视图表中登记（证据：`pro_ops/fa/test_fa_performance.py:478-483` 的 `reduce_dst`(DN) / `reduce_dst_rm`(ND) 共用 addr）。具体 layout 以对应 API 文档为准，不假设 DN 归约输出可直接参与逐元素运算
   - **大小**：`prod(shape) × dtype_bytes`
   - **Acc fractal 约束**（🚨 阻断级）：Acc(L0C) FP32/INT32 tile 的 `fractal=1024` 由框架自动设置，**无需手写**（见 EXPLORE_REPORT §3.4 / §5.2 及 `TileType.md`）；但必须保证 Acc tile 的**物理 M×N** 满足一个 fractal（FP32 ≥ 256 元素 = 1024 bytes）。当某维度逻辑值极小、可能使物理 tile 退化到 <fractal 时（如动态轴允许 `S_q=1`），tile shape 须 pad 到最小合规尺寸，运行时用 `set_validshape` 限制有效区。注意区分**逻辑维度**（可能极小）与**物理 tile 维度**（须满足 fractal）。
3. 标注所有数据 tile（涉及 `load_tile`/`store_tile` 的 tile）的 `valid_shape=[-1,-1]`
4. **`tile_dims` stride 安全性检查** 🚨（经验性检查项——在当前 pro_ops 样例与已知故障现象中，是 aicore/vector core 异常的常见根因之一）：
   - 如果设计中使用了 `load_tile` + `tile_dims=[d0, d1]`（覆盖多维度），检查被覆盖的最外层维度 d0 在 3D 张量布局中的 **stride**（行间跳跃字节数）
   - **低风险信号**：被覆盖的最外层维度 stride ≤ tile 面积（如 3D 布局 `[C,B,N]` + `tile_dims=[1,2]`，dim 1 stride = N ≈ 10KB 级别）。在当前样例与已知故障现象中，这类布局更容易让 DMA 尾块读取地址保持在物理分配范围内，并由 `set_validshape` 裁剪
    - **高风险信号**：被覆盖的最外层维度 stride 巨大（如 3D 布局 `[B,C,N]` + `tile_dims=[0,2]`，dim 0 stride = C×N ≈ MB 级别）。在当前样例与已知故障现象中，这类布局更容易让 DMA 读取该维度尾块时每行跳跃过大，地址超出预期分配范围，运行时可能表现为 aicore exception（如 507015）或 vector core exception（如 507035）
   - **修复方法**：对张量做 permute，将 stride 小的维度移到 `tile_dims` 的最外层。上例中 `[B,C,N]` → permute 为 `[C,B,N]`，`tile_dims` 从 `[0,2]` 改为 `[1,2]`，最外层 stride 从 C×N 降到 N
   - **检查清单**：对每个 `load_tile` / `store_tile` 的 `tile_dims`：
     1. 计算 dim d0 在张量布局中的 stride
     2. 若 stride 超过 EXPLORE_REPORT §7 探测的 stride 经验阈值，标记为 **❌ 高风险**，优先考虑 permute 布局或调整 tile_dims，并结合本次 API 文档 / 样例证据确认修复路径

**输出**：Tile 属性表：

| 用途 | 变量名 | shape | dtype | layout | 大小 | 备注 |
|------|--------|-------|-------|--------|------|------|
| 输入暂存 | `tile_a` | `[64,128]` | FP32 | `—` | 32768 | valid_shape=[-1,-1] |
| ... | ... | ... | ... | ... | ... | ... |

> `—` 表示 layout 列留空即用默认 `pl.ND`（行优先）；非默认布局（如 `pl.DN`）显式写出。

---

### R3：UB 空间布局

**核心问题**：tile 放在 UB 的哪个地址？如何分配管理？

**流程**：

1. 基于 R2 的 tile 大小，从 `0x00000` 开始连续排列地址，不重叠，对齐到 32 字节
2. 标注双视图对（同地址、不同 layout 视角）
3. 分配方式首选 `make_tile_group` + `auto_mutex`，由框架自动管理 buffer 切换与同步；`make_tile` 手动分配为次选
4. 验证所有 tile 总大小不超过 UB 上限（上限值以 EXPLORE_REPORT §7 探测的 UB 容量为准）
5. **R3↔R6 联合决策**：涉及核间流水时，double buffer 会让地址占用翻倍，因此地址表按 buffer 数参数化；是否需要 PONG 地址、buffer 数取值、以及哪些 tile 受其影响，均在 R3 先显式标出，待 R6 确定 pipeline 深度后回填确认

**输出**：UB 地址映射表：

| 用途 | 变量名 | shape | dtype | layout | 地址 | 大小 | 备注 |
|------|--------|-------|-------|--------|------|------|------|
| 输入暂存 | `tile_a` | `[64,128]` | FP32 | `—` | `0x00000` | 32768 | valid_shape=[-1,-1] |
| ... | ... | ... | ... | ... | ... | ... | ... |

> `—` 表示 layout 列留空即用默认 `pl.ND`（行优先）；非默认布局（如 `pl.DN`）显式写出。

**UB 总用量**: {∑ 大小} bytes / {UB 上限} bytes = {百分比}

---

### R4：循环与 Section 结构

**核心问题**：怎么把 Phase、tile 申请、计算操作组织成完整的循环结构？

**流程**：

1. 设计 M-tile 循环（外层 SPMD 跨核）和 N-tile 循环（内层）的嵌套关系
2. 根据 R0 的 Phase 划分，决定循环嵌套层级与 section 位置：
   - 如果所有 N-tile 的某个 Phase 可以独立完成（不需要等全部 N-tile 的上一 Phase 结果），则 Phase 内部只需 N-tile 循环
   - 如果需要跨 N-tile 归约，则每个 Phase 是一个完整的 N-tile 遍历
3. 将 R3 的空间规划（tile 声明）和 R1 的 API 序列嵌入到循环结构中
4. **SPMD 原语获取位置由 Phase 数量决定**：`pl.get_block_idx()` 和 `pl.get_block_num()` 是全局 SPMD 原语。
   - **单 section 算子**（Phase 数 = 1）：可在该 section 内部获取（EXPLORE_REPORT §4/§5 定位的单 section 样例与教程均在 section 内调用）
   - **多 section 算子**（Phase 数 > 1）：需在 jit 函数体内、所有 section 声明之前获取一次，通过闭包变量共享给各 section（EXPLORE_REPORT §4 定位的多 section 样例）
   两种写法均合法。`num_cores = pl.get_block_num()`（动态获取），因为 core 数量因芯片而异。DESIGN.md §3 的伪代码须按实际 Phase 结构体现。
5. **标注同步点位置**（只标注位置和目的，具体同步 API 由 R5/R6 填入）：
   - 标注每个需要同步的位置（如"load_tile 后需等 MTE2 完成"）
   - 标注同步目的（pipe 间依赖、核间依赖、sub-block 间依赖）
6. **常量声明位置**：所有 tile 尺寸/公式编译期常量（TS、TD、SCALE 等）必须声明在 kernel 函数**外部**（模块级）。写进 kernel 函数体内会被 JIT 当作 IR 语句处理，触发编译错误 `Unsupported kwarg type for key: memref_size`（见 develop pitfalls §1.2）。伪代码骨架须把常量块画在 `@pl.jit` 装饰器之前

**输出**：完整的 kernel 伪代码骨架：
- 动态维度声明（DynVar）
- tile 声明（make_tile_group）
- section 声明
- SPMD 参数获取（含位置说明）
- 完整的循环嵌套结构（含同步点占位标注）

---

### R5：分核策略

**核心问题**：work item 如何分配到各物理核？

**流程**：

1. **分核方式**：当前默认使用 strided loop 分配 work item 到各物理核：
   ```python
   for i in pl.range(core_id, m_tile_num, num_cores):
   ```
   在当前样例中，各物理核独立处理不同 tile、通常无核间数据依赖（参考 EXPLORE_REPORT §4 定位的 strided loop 样例）

2. **bar_all 屏障**（条件性）：仅当核间有数据依赖时使用。按当前样例与默认分核方式，若各核通过 strided loop 独立处理 work item，通常不需要 `bar_all`

**输出**：
- strided loop 分核方案
- 是否需要 `bar_all` 的判断与理由

---

### R6：同步与核间流水

**条件性**：同步规划始终需要；其中 cross_core 流水仅当算子涉及跨 section/sub-block 数据传递时需要（R0 Phase 划分含多 section 且 section 间有数据流）。

**核心问题**：Phase 内/Phase 间需要哪些同步？若存在 cross_core 流水，Cube 和 Vector 之间如何传递数据、event_id 如何隔离？

**注意**：在当前收集到的 pro_ops 样例中，cross_core 通信体现为同物理核的 Cube↔Vector sub-block 间流水（通过 GM FIFO），而非不同物理核间通信。若后续文档或样例出现不同语义，以最新证据为准。

**流程**：

1. **汇总同步点**：承接 R4 的同步占位，统一决定每个位置使用哪类同步：
   - `bar_v`：V pipe 内前后依赖的计算之间
   - `sync_src/sync_dst`：`load_tile`/`move`/`matmul` 等 pipe 级依赖
   - `set_cross_core`/`wait_cross_core`：仅用于 cross_core 流水
   - `auto_mutex` 的同步语义也在本轮说明：若 R3 选择 `make_tile_group` + `auto_mutex`，则说明其覆盖的 buffer 切换/互斥边界，以及哪些依赖仍需额外同步

2. **确认流水方向**：若存在 cross_core 流水，由 R0 Phase 划分确定方向（如 Cube→Vector 通过 GM FIFO 传递 qk/p/pv）

3. **决定 pipeline 深度**：QK_PRELOAD 深度和 FIFO_SIZE（= pipeline 预取深度 + 1）。此决策影响 R3 地址规划（需预留 double buffer / PONG 地址），据此回填 R3 的 buffer 数与地址占用

4. **event_id FIFO 隔离**：从当前样例用法看，`set_cross_core` 的 event_id 可按 flag 语义建模——同一 work item 生命周期内同一 event_id 在样例中通常只见一次消费，多组流水靠轮转隔离而非计数。当前可按 `_READY_IDS[task_id % FIFO_SIZE]` 槽位轮转隔离来设计——多组流水（如 QK/P/PV）各占一段不重叠的 event_id 范围。event_id 上限统一以 `custom/<op>/EXPLORE_REPORT.md` §7 的本次探测结果为准；Stage 3 只消费该结论，不单独设定固定数值（参考样例中的 `assert 3*FIFO_SIZE <= {event_id上限}`）。官方 cross_core 文档（`SIMD-API/计算API/同步控制/set_cross_core_wait_cross_core.md`）未明文描述此 flag 语义，上述为基于当前样例的经验性推断，如遇冲突以文档或更新样例为准。

5. **多 work-item-per-core 场景**：算子 shape 放大后通常会出现多 work-item-per-core。若采用当前 cross_core 流水模型，应在设计阶段给出 event_id 隔离方案，并在 DESIGN.md 中记录 FIFO 深度和 event_id 分配表

6. **标注 cross_core 同步点**：
    - `pl.system.set_cross_core`（produce 端）：在数据产生之后
    - `pl.system.wait_cross_core`（consume 端）：在数据使用之前
    （set/wait 位置参考 EXPLORE_REPORT §4 定位的流水样例）

**输出**：
- 在 R4 的伪代码骨架中填入 `bar_v` / `sync_src` / `sync_dst` / `set_cross_core` / `wait_cross_core`
- `auto_mutex` 覆盖范围与剩余同步责任说明
- pipeline 深度和 FIFO_SIZE
- event_id 分配表（各组流水占用的 event_id 范围 + max_event_id）

**特殊证据要求**：event_id 分配方案需有样例引用。回填后的 R3 地址规划需能覆盖 pipeline 深度对应的 buffer 数。

---

### R7：尾块处理

**核心问题**：如何处理维度不整除 tile 尺寸的尾块？

**流程**：

1. **ceiling division 计算 tile 数**：`m_tile_num = (M + TILE_M - 1) // TILE_M`（覆盖维度 < tile、整除、N+1 块尾块不满所有场景，参考 EXPLORE_REPORT §4 定位含尾块处理的样例）
2. **pl.min 计算尾块有效尺寸**：`valid_m = pl.min(M - m_off, TILE_M)`（满 tile = TILE，尾块 = 余数）
3. **set_validshape 运行时告知硬件**：`pl.set_validshape(tile, valid_m, valid_n)`
4. **跨迭代恒等值初始化**：如 `expands(gmax, -1e9)` 消除首迭代分支
5. **条件分支选择 tile 形状**（如需）：可用 `is_tail = pl.min(1, TS - actual)` 标志位 + 模块级常量（参考 EXPLORE_REPORT §4 定位的 unalign 样例）

**注意**：如果算子涉及核间流水（R6），尾块的有效 shape（如 `actual_sq`/`actual_skv`）是流水 ctx 的一部分，必须随 ctx 在 event_id FIFO 中流转。consume 端通过错位取 ctx 获取对应 task 的有效 shape（参考 EXPLORE_REPORT §4 定位的深预计算样例中 ctx_arr → 错位取尾块 shape 的模式）。尾块处理须与 R6 联合设计，不能独立后置。

编码期 `pl.Scalar` 内联相关注意事项见 develop pitfalls §1.1。

**输出**：在 R4 的伪代码骨架中填入尾块处理代码。

---

### R8：综合评估

**核心问题**：整体设计是否准确、可靠、泛化？

**流程**：

> 评估时每项检查须标注 ✅（通过）或 ❌（不通过）；orchestrator 以评估表中无 ❌ 作为设计通过判据。

| 维度 | 检查项 | 结果 | 不通过时的处理 |
|------|--------|------|----------------|
| **准确性** | API 调用链是否完整实现了数学公式的每一步 | 回到 R1 补充 |
| | 数据依赖是否正确（Phase 顺序、sync 位置） | 回到 R0 或 R4 调整 |
| | dtype 选择是否能保证精度（如累加用 FP32） | 回到 R2 调整 |
| | 归约类 API 的 `[M,1]`/`[1,N]` 输出已设 `layout=pl.DN`；若参与逐元素运算已建 DN+ND 双视图 | 回到 R2 补 layout / 双视图 |
| | Acc tile 物理 M×N×dtype_bytes ≥ fractal（FP32 ≥ 1024 bytes；动态轴含极小维度时尤需检查） | 回到 R2 pad tile shape |
| **泛化性** | 是否正确处理了尾块（ceiling division + set_validshape） | 回到 R7 补充 |
| | 归约轴可能超单 tile 时已采用 online/两遍法（非 TILE_N 大值兜底，R0 已判断） | 回到 R0 重设归约方案 |
| | 循环边界是否正确（ceiling division、valid_m/valid_n 计算） | 回到 R4 修正 |
| | 超越函数（exp/log/sqrt 等）在目标 dtype 范围内无溢出（§1 数值安全边界已分析） | 回到 R1 补溢出防护 |
| | 跨 tile 状态是否正确初始化和持久化（如 expands 恒等值、muls 拷贝） | 回到 R1 或 R7 修正 |
| | 同步策略在 SPEC §8 动态轴全范围下是否正确（num_cores、pipeline defer、event 分配是否随 total_work 变化而保持隔离） | 回到 R5/R6 修正 |
| **一致性** | R0-R7 各轮输出是否存在矛盾（如 API 需要的 tile 在 R2 中缺失） | 回溯到矛盾产生的轮次修正 |
| | 证据链是否完整（每个决策都有来源） | 补充缺失的文档引用或样例路径 |
| | `tile_dims` 最外层维度 stride 不超过 EXPLORE_REPORT §7 探测阈值（R2 步骤 4 已检查） | 回到 R2 调整布局 |
| **条件性检查** | 如跳过 R6，确认该算子确实无跨核数据传递 | 回到 R0 重新评估 |

**迭代规则**：
- 发现问题数 ≤ 3，修复后重新走 R8
- 发现问题数 > 3，回到问题最早出现的轮次，重新过后续轮
- 最多 5 次完整 R0-R8 迭代

**输出**：评估结论 + 修改记录（如有）。

---

### Tile 数据流全景图

R8 评估通过后，将 R0-R7 各轮的分散产出串成一张全景图（填入模板 §9）。这张图应能单页展示算子中所有数据流的全貌。

**绘制方法**：

1. **起点**：GM 中的输入张量
2. **数据搬运**：标注每个 `load_tile` 和 `store_tile`，并标注流水线（MTE2 / MTE3）
3. **数据流向**：用箭头 `→` 连接每个操作，箭头标注 API 名（如 `row_max`、`exp`）。同一块 tile 被多个操作串联使用时用 `├─` 表示分支
4. **跨 Phase 持久化**：tile 在某个 Phase 中写入、在后续 Phase 中读取的，用 `---` 虚线表示数据跨越 Phase 边界。标注 tile 变量名和地址
5. **双视图转换**：如有，标注 `[双视图]` 转换点（`pl.DN` ↔ 默认 `pl.ND`）
6. **输出**：最终 `store_tile` 写入 GM 的输出张量

**验证**：全景图中每一块 tile 和每一个操作都必须能在 R3 的地址映射表、R1 的 API 序列中找到对应条目。缺失或矛盾则回溯修正。

---

## 设计原则

1. **每个决策必须有证据**：API 文档引用、样例代码路径、数学推导三者至少占其一
2. **Tile 数据流图是给 coder 的初步设计**：coder 拿到 DESIGN.md 应能确定 kernel 的完整结构与关键决策；API 签名等细节须以 API 文档原文为准确认（EXPLORE_REPORT 仅为派生的先行速查，不作签名权威）；运行验证暴露设计失误时 coder 可据实修正
3. **地址分配精确到字节**：不写"大约"、"若干"
4. **参考样例优先于自行设计**：遇到同步策略、tile 尺寸等决策时，优先查阅 pro_ops 下相似算子，复用成熟模式
5. **每轮发现问题立即回溯**：不允许累积到 R8 再一起修