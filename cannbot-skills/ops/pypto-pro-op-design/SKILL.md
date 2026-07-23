---
name: pypto-pro-op-design
description: Stage 3 架构设计。通过 9 轮迭代式约束收敛，基于 Stage 1 产物（SPEC.md、EXPLORE_REPORT.md），产出 DESIGN.md。核心输出为 tile 级别数据流图——决定 Phase 划分、API 映射、Tile 规划、片上空间布局（UB/L1/L0）、循环与 Section 结构、分核策略、核间流水、尾块处理。每一步决策必须有 API 文档、教学文档或官方指定算子做证据，严禁猜测。触发词：生成设计方案、tile 数据流、DESIGN.md、tile 级别设计、片上空间规划、UB 空间规划。
---

# PyPTO-Pro Stage 3 — 迭代式方案设计

通过 9 轮迭代式约束收敛（R0-R8），**目标** 是生成可直接翻译为 kernel 代码的 DESIGN.md。

**核心原则**：
- 每个决策必须包含**结论 + 推导过程 + 证据来源**
- 力求后续 Agent 拿到 DESIGN.md 即可确定 kernel 的完整结构与关键决策；API 签名等细节仍须由 coder 以 API 文档原文为准确认（EXPLORE_REPORT 仅为派生的先行速查，不作签名权威），运行验证暴露设计失误时可据实修正
- 每轮发现的矛盾必须回溯修正前序决策，不允许累积到 R8 再处理
- 本 skill 以**思维方法指导**为主，不教具体写法——具体 API 用法、tile 配置、同步写法等请查阅 API 文档（`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/`）、教学文档（`$PYPTO_DEVKIT_DIR/docs/pypto_pro/guide`）、官方指定算子（见 `PRO_MATERIAL_INDEX.md` §B），理解后据实设计

## 两条性能强制（设计阶段须落实）

> 完整定义见 `pypto-pro-material-explore` SKILL「两条性能强制」节。设计阶段须在 R3（地址分配）和 R1（API 映射）中落实：
> 1. 所有需要 buffer 切换/轮转的 tile 一律用 `make_tile_group` + `auto_mutex`，`make_tile` 仅限单次使用 scratch tile。手动 sync 的严格界限（auto_mutex 管辖范围、跨核同步用 `set_cross_core`/`wait_cross_core`、mutex_id 与 event_id 独立命名空间）见 `pypto-pro-material-explore` SKILL「两条性能强制」节。R3 落实 buffer 管理方式，R6 落实 cross_core 同步方案。
> 2. Vector 数值计算用 `vf.*` 手写（完整理由见 `pypto-pro-material-explore` SKILL「两条性能强制」节）。

## 输入

| 来源 | 路径 | 用途 |
|------|------|------|
| 算子规格 | `custom/<op>/SPEC.md` | 公式、shape、dtype、动态轴；末尾「kernel 契约补充」节的辅助张量暂存 / cast 边界链 / 累加语义（分别喂给 R2 tile 规划 / §1 数值边界 / R6 同步），及标注「留待 design」的 MAY-DESIGN 字段 |
| Golden 参考 | `custom/<op>/{op}_golden.py` | 函数签名参考（影响 R7.5 测试 case 规划）；增量验证模式下需暴露中间量辅助函数（如 `{op}_golden_stage1`），设计阶段应知晓此依赖 |
| 资料探索报告 | `custom/<op>/EXPLORE_REPORT.md` | API 映射与约束（§3）、相似样例与可复用模式（§4）、教程设计指导（§5）、Tile/同步策略建议（§6）、环境常量快照（§7：UB 容量/event_id 上限/对齐要求等） |
| 全量资料索引 | `custom/<op>/PRO_MATERIAL_INDEX.md` | API 文档（§A）、官方指定算子（§B）、教程（§C）的精确路径定位 |

## 输出

- **`custom/<op>/DESIGN.md`**，基于 [templates/design-template.md](templates/design-template.md)，核心交付物为 §10"Tile 数据流全景图"

---

## 前置：维度契约

在切 Phase 之前先确认 kernel 的输入/输出维度契约——这是后续所有轮（Tile 规划、循环结构）的前提，不属于 Phase 划分本身。

明确 kernel 固定处理的维度（如"kernel 只处理 2D `[M,N]`"）。若 SPEC.md 要求支持 1D 或任意多维，必须在此定义 host 端适配方案（如 1D `[L]` → host reshape `[L,1]` → kernel `[M,N]` → 输出 reshape 回 `[L]`），不能默默只实现 2D 而遗漏 SPEC 要求的其他维度。产出填入 DESIGN.md §0 的维度契约栏（kernel 固定维度 + host 适配规则表）。

---

## 迭代设计流程

设计是**问题驱动**的迭代，不是线性填表。每一轮聚焦一个核心问题（回溯原则见核心原则第 3 条）。

### R0：Phase 划分

**核心问题**：整体计算流可以分解为几个 Phase？每个 Phase 做什么？

**流程**（按优先级）：

1. **数据依赖**：需要前面 Phase 的完整结果才能开始计算的，必须划分为不同 Phase。例如跨 N-tile 的 softmax：必须先遍历所有 tile 算全局 max（Pass1），才能算 exp(x-max) 并累加全局 sum（Pass2），最后做除法（Pass3）。⚠️ **归约轴可能超单 tile 时，须采用多 Phase/多 tile 归约（online/两遍法）以保证泛化性——归约轴超出单 tile 时仍须正确工作**。多 tile 归约的具体状态更新方式随算子而异（softmax 的 running max/sum、layernorm 的 running mean/var 等），以对应 golden 与官方指定算子为准；归约实现须用 vf 手写（见上方「两条性能强制」）
2. **Section 分隔**：Cube 和 Vector 使用不同引擎，必须划分到不同 Phase。Section 间需明确数据传递方式（via GM workspace）

**输出**：

- Phase 列表（Phase1/2/3...），每个标注：目的、输入依赖、输出、涉及的 Section
- 如果只有一个 Phase（如 N ≤ TILE_N 的 softmax），标注"单 Phase"并说明为什么不需要拆分
- 归约轴容量结论：归约轴是否可能超单 tile + 是否采用多 tile 归约（含归约算子时必填）

---

### R1：API 映射

**核心问题**：每个数学步骤具体用哪些 API？需要在 R0 的 Phase 划分基础上，将 API 调用链进一步细化到 Phase 内部的每一步操作。

**⚠️ 性能强制（影响本轮映射）**：Vector 数值计算步骤须映射到 `vf.*` 指令序列。Cube 步骤照常映射 `pl.*` Cube API（`pl.matmul`/`pl.matmul_acc` 等）。具体 vf 指令的选用以 vf API 文档与官方指定算子中的 vf 写法为准。

**流程**：

1. 从 EXPLORE_REPORT.md §3 提取已确认的 API 映射（vec 步骤应为 `vf.*` 序列）
2. **逐一查阅每个 API 的文档原文**（通过 `PRO_MATERIAL_INDEX.md` §A 定位路径），确认并记录：
   - **功能**：API 做什么、语义是否与数学步骤完全匹配
   - **签名**：参数顺序、位置参数 vs 关键字参数（以 API 文档原文为准，EXPLORE_REPORT §3.3 已汇总部分）
   - **使用约束**：dtype 限制、shape 要求、layout 要求、MemorySpace 要求、tmp tile 是否可与输入重叠等（以 API 文档原文为准，EXPLORE_REPORT §3.3 已汇总部分）
   - **特性特点**：in-place 支持、尾块行为、与 `set_validshape` 的交互等
   - **数值安全边界**（API 链含 exp/log/sqrt/reciprocal/tanh 等超越/非线性函数时强制）：分析该 API 在目标 dtype 下的溢出边界。例如 fp16 max≈65504，exp(11.09)≈65504，输入 >11 即溢出为 +inf，后续 inf/inf → NaN 传播。须在 §1 数值安全边界栏记录"输入范围→是否溢出→防护措施"（如 exp(x) 可以写作 (exp(x - max))/exp(max)）。**注**：Vector 数值计算中的非线性须用 vf 指令实现（如 `vf.exp_sub`），其溢出行为以 vf API 文档为准
3. 在每个 Phase 内部，将计算步骤展开为具体的 API 序列（vec 步骤为 `vf.*` 序列，cube 步骤为 `pl.*` 序列）
4. 标注每个 API 的输入 tile、输出 tile
  5. **Vector 数值计算必须映射到 `vf.*` 指令序列，无例外。** 若某步骤找不到直接对应的 vf API，优先尝试用其他 vf API 组合 + 循环结构手动实现（参考 EXPLORE_REPORT §3 中的组合方案）。"复杂"不是放弃 VF 的理由。仅当穷尽 vf 组合方案仍不可行时，标注 unsupported——此时算子开发失败，**不得以 `pl.*` 计算替代 `vf.*` 完成 Vector 数值计算**

**注意**：本轮只确定计算 API ，不涉及同步 API。

**输出**：Phase 级 API 调用序列（伪代码形式）

---

### R2：Tile 规划

**核心问题**：需要哪些 tile？每个 tile 的 shape、dtype、layout 是什么？

**流程**：

0. 按模板 §2.1 的格式填写关键常量。列出所有 tile 尺寸（TS、TD 等）和派生常量（SCALE 等）。tile 尺寸是编译期对齐值，运行时通过 R7 设计的策略处理尾块场景（维度 < tile、整除、N+1 块且尾块不满）。公式常量若依赖实际维度值而非 tile 尺寸，应使用实际值
1. 根据 R1 的 API 序列，列出所有涉及的 tile
2. 对每个 tile，确定：
   - **shape**：由 API 操作数要求决定
   - **dtype**：由用户需求决定
   - **内存空间（`target_memory`）**：由该 tile 所参与的 API 决定——vector 计算 tile 用 `Vec`(UB)；含 cube/matmul 时，`load_tile` 的 GM 落点用 `Mat`(L1)，matmul 左/右操作数用 `Left`(L0A)/`Right`(L0B)，累加输出用 `Acc`(L0C)。此列直接决定 R3 的分节归属
   - **layout**：默认按内存空间取（`Vec` 无约束，`Mat`/`Left`/`Right`/`Acc` 见 `TileType.md` 默认布局表），具体约束以 EXPLORE_REPORT §3.3 / API 文档为准。
   - **大小**：`prod(shape) × dtype_bytes`
   - **Acc fractal 约束**：必须保证 Acc tile 的**物理 M×N** 满足一个 fractal（FP32 ≥ 256 元素 = 1024 bytes）。当某维度逻辑值极小、可能使物理 tile 退化到 <fractal 时，tile shape 须 pad 到最小合规尺寸，运行时用 `set_validshape` 限制有效区。

**输出**：Tile 属性表：

| 用途 | 变量名 | shape | dtype | 内存空间 | layout | 大小 | 备注 |
|------|--------|-------|-------|---------|--------|------|------|
| 输入暂存 | `tile_a` | `[64,128]` | FP32 | UB(Vec) | `—` | 32768 | ... |
| ... | ... | ... | ... | ... | ... | ... | ... |

> 内存空间取 `Vec`(UB)/`Mat`(L1)/`Left`(L0A)/`Right`(L0B)/`Acc`(L0C)；`—` 表示 layout 列留空即用该内存空间的默认布局。

---

### R3：片上空间布局

**核心问题**：每块 tile 放在哪个内存空间的哪个地址？如何分配管理？

**内存空间**：tile 按 R2 确定的 `target_memory` 落到不同片上空间，**每个空间独立寻址、独立限容**——地址各自从 `0x00000` 起算，同一 addr 值在不同空间是不同物理位置。

- `Vec`(UB)：vector 计算用；纯 vector 算子只涉及此空间（`TileType.md` 参数范围表：`Vec` 对应 UB）
- `Mat`(L1)、`Left`(L0A)、`Right`(L0B)、`Acc`(L0C)：含 cube/matmul 的算子涉及。`matmul` 的操作数内存空间是**硬性约束**——`lhs` 只能 `Left`(L0A)、`rhs` 只能 `Right`(L0B)、`dst` 只能 `Acc`(L0C)，放错空间即报错
- 典型数据流：`GM --load--> L1(Mat) --move--> L0A/L0B --matmul--> L0C(Acc)`，结果既可从 Acc 直接 `store` 回 GM，也可先 `move` 到 UB 再后处理/store

**流程**：

1. 按 `target_memory` 把 R2 的 tile 分组，**每个内存空间各自从 `0x00000` 开始**连续排列地址，不重叠。UB/L1 首地址须 32 字节对齐（证据：`load_tile.md` 参数范围表 "L1、UB buffer 首地址必须 32 字节对齐"）；L0A/L0B/L0C 的对齐以对应 API 文档 / 官方指定算子为准，不套用 32 字节
2. 标注同地址不同 layout 的 tile 对（如有）
3. 分配方式应使用 `make_tile_group` + `auto_mutex`，由框架自动管理 buffer 切换与同步（见上方「两条性能强制」）。
4. **逐空间**验证该空间上的 tile 总大小不超过其容量上限。容量值以 EXPLORE_REPORT §7 探测记录为准——§7 必含 UB 容量；含 cube 时须补探 L1/L0 各空间容量（§7 未记录则回退 material-explore 补测，不得在此臆测数值）
5. **R3↔R6 联合决策**：涉及核间流水时，double buffer 会让地址占用翻倍，因此地址表按 buffer 数参数化；是否需要 PONG 地址、buffer 数取值、以及哪些 tile 受其影响，均在 R3 先显式标出，待 R6 确定 pipeline 深度后回填确认

**输出**：片上地址映射表（**按内存空间分节**，纯 vector 算子只有 UB 一节）：

| 内存空间 | 用途 | 变量名 | shape | dtype | layout | 地址 | 大小 | 备注 |
|---------|------|--------|-------|-------|--------|------|------|------|
| UB(Vec) | 输入暂存 | `tile_a` | `[64,128]` | FP32 | `—` | `0x00000` | 32768 | ... |
| L1(Mat) | A 矩阵暂存 | `a_l1` | `[128,128]` | FP16 | `—` | `0x00000` | 32768 | ... |
| L0C(Acc) | 累加结果 | `acc` | `[128,128]` | FP32 | `—` | `0x00000` | 65536 | Acc FP32 自动 fractal |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

> `—` 表示 layout 列留空即用该内存空间的默认布局（Vec 无约束；其余空间见 `TileType.md` 默认布局表）；非默认布局显式写出，具体值以 API 文档为准。

**各空间总用量**（逐空间列出，无对应 tile 的空间可省略）:
- UB(Vec): {∑ 大小} / {§7 UB 容量} = {百分比}
- L1(Mat) / L0A / L0B / L0C（如有 cube）: {∑ 大小} / {§7 对应容量} = {百分比}——各空间容量来自 §7 探测记录

---

### R4：循环与 Section 结构

**核心问题**：怎么把 Phase、tile 申请、计算操作组织成完整的循环结构？

**流程**：

1. 设计 M-tile 循环（外层 SPMD 跨核）和 N-tile 循环（内层）的嵌套关系
2. 根据 R0 的 Phase 划分，决定循环嵌套层级与 section 位置，具体写法参考 EXPLORE_REPORT.md 中指定的官方算子样例
3. 将 R3 的空间规划（tile 声明）和 R1 的 API 序列嵌入到循环结构中
4. `pl.get_block_idx()` 和 `pl.get_block_num()` 是全局 SPMD 原语，具体获取位置请见官方指定算子（PRO_MATERIAL_INDEX §B）。
5. 伪代码骨架须把编译期常量块（TS、TD、SCALE 等）画在 `@pl.jit` 装饰器之前（模块级）；具体声明位置的编码规范由 develop skill 负责

**输出**：完整的 kernel 伪代码骨架：
- 动态维度声明（具体声明 API 以 `docs/` API 文档和官方指定算子样例为准，不臆测）
- tile 声明（make_tile_group）
- section 声明
- SPMD 参数获取（含位置说明）
- 完整的循环嵌套结构

> **完整性要求**：伪代码中核心计算步骤必须写完整（具体 API 调用或明确的算法步骤），不得用 `...` 留空或标注"需 coder 实现"。`...` 仅允许用于 sync 占位（R6 填入）和 R7 尾块占位（已标注"R7 填入"）。若某步骤的 API 或实现方式未知，说明 R1 API 映射不完整，应回到 R1 补充或标注为 unsupported 触发回退。

---

### R5：分核策略

**核心问题**：work item 如何分配到各物理核？

> 📌 **权威依据（必读，一切以此为准）**：`$PYPTO_DEVKIT_DIR/docs/pypto_pro/guide/编程指南/编程模型/AI-Core-SIMD编程/基于Tile的Python编程/多核切分与Tiling.md`。分核策略全部照该文档执行，与经验推断冲突时以该文档为准。

**本轮须在 DESIGN.md §5 落实的产出**：
- 分核方案
- host 侧 `num_cores` 计算式

**输出**：DESIGN.md §5（引用 多核切分与Tiling.md，填入上述两项产出）

---

### R6：同步与核间流水

**条件性**：同步规划始终需要；其中 cross_core 流水仅当算子涉及跨 section/sub-block 数据传递时需要（R0 Phase 划分含多 section 且 section 间有数据流）。

**核心问题**：Phase 内/Phase 间需要哪些同步？若存在 cross_core 流水，Cube 和 Vector 之间如何传递数据、event_id 如何隔离？

**注意**：在当前官方指定算子中，cross_core 通信体现为同物理核的 Cube↔Vector sub-block 间通过片上共享 buffer（L1/Mat 或 UB/Vec）的 N-buffer 轮转传递数据，配合 `set_cross_core`/`wait_cross_core` 做同步，而非不同物理核间通信。若后续文档或样例出现不同语义，以最新证据为准。

**流程**：

1. **标注并汇总同步点**：基于 R4 的伪代码骨架，先标注每个需要同步的位置（如"load_tile 后需等 MTE2 完成"）及其目的（pipe 间依赖、核间依赖、sub-block 间依赖），再统一决定每个位置使用哪类同步：
   - `bar_v`：V pipe 内前后依赖的计算之间
   - `sync_src/sync_dst`：`load_tile`/`move`/`matmul` 等 pipe 级依赖
   - `set_cross_core`/`wait_cross_core`：仅用于 cross_core 流水
   - `auto_mutex` 的同步语义也在本轮说明：若 R3 选择 `make_tile_group` + `auto_mutex`，则说明其覆盖的 buffer 切换/互斥边界，以及哪些依赖仍需额外同步
2. **确认流水方向**：若存在 cross_core 流水，由 R0 Phase 划分确定方向（如 FA 中 Cube→Vector 通过片上 buffer 传递 qk/pv，Vector→Cube 传递 p）
3. **决定 pipeline 深度**：pipeline 预取深度（以实际算子为准，如 FA 的 QK_PRELOAD）和 FIFO_SIZE（= pipeline 预取深度 + 1）。此决策影响 R3 地址规划（需预留 double buffer / PONG 地址），据此回填 R3 的 buffer 数与地址占用
4. **event_id 隔离**：多组流水各占一段不重叠的 event_id 范围（流水组以实际算子为准，如 FA 的 QK/P/PV），具体隔离方案以官方指定算例的实际用法为准（参考 EXPLORE_REPORT §4 定位的样例）。event_id 取值范围为 `[0, 16)`（API 文档 `set_cross_core_wait_cross_core.md` 参数范围表）。
5. **多 work-item-per-core 场景**：算子 shape 放大后通常会出现多 work-item-per-core。若涉及 cross_core 流水，应在设计阶段给出 event_id 隔离方案，并在 DESIGN.md 中记录 FIFO 深度和 event_id 分配表
6. **标注 cross_core 同步点**：
    - `pl.system.set_cross_core`（produce 端）：在数据产生之后
    - `pl.system.wait_cross_core`（consume 端）：在数据使用之前
    （set/wait 位置参考 EXPLORE_REPORT §4 定位的官方指定算子）

**输出**：
- 在 R4 的伪代码骨架中填入 `bar_v` / `sync_src` / `sync_dst` / `set_cross_core` / `wait_cross_core`
- `auto_mutex` 覆盖范围与剩余同步责任说明
- pipeline 深度和 FIFO_SIZE
- event_id 分配表（各组流水占用的 event_id 范围）

**特殊证据要求**：event_id 分配方案需有官方指定算子引用。回填后的 R3 地址规划需能覆盖 pipeline 深度对应的 buffer 数。

---

### R7：尾块处理

**核心问题**：如何处理维度不整除 tile 尺寸的尾块？

> 📌 **权威依据（必读，一切以此为准）**：`$PYPTO_DEVKIT_DIR/docs/pypto_pro/guide/编程指南/编程模型/AI-Core-SIMD编程/基于Tile的Python编程/尾块处理.md`。尾块的完整机制全部照该文档执行，与经验推断冲突时以该文档为准。核心模型：**物理形状固定（永远满块可复用），有效形状随位置变化**。

**本轮须在 DESIGN.md §7 落实的产出**：将该文档的尾块机制落到本算子的伪代码骨架。

**输出**：DESIGN.md §7（引用 尾块处理.md 落地尾块代码）

---

### R7.5：目标测试 case 规划

**核心问题**：本设计要交给 develop 验证的具体测试 shape 是哪几个？

设计阶段必须**基于已确定的 tile 切分（R2 tile 尺寸）与尾块方案（R7）**，把测试 case 连同具体 shape 一次性确定下来，写入 DESIGN.md §8，供 develop 直接实现——develop 开工前即已知这些 case，不再自行重算。
**流程**：

1. **确定 case 集合**：
   - 若 SPEC.md / 用户已给出目标 case，则**在其基础上追加**下述 4 个基础 case。
   - 若无用户指定 case，则**直接确定为下述 4 个基础 case，不询问用户**。
2. **4 个基础 case（按 tile 切分推导具体 shape，取值以 R2 tile 尺寸为基准）**：

   | case | 覆盖场景 | shape 取法（示例） |
   |------|---------|-------------------|
   | 全整除 | 所有 tile 维度均整除 | `[TILE_A, TILE_B]` |
   | 单轴尾块 | 一个 tile 维度存在尾块 | `[TILE_A + 尾块余数, TILE_B]` |
   | 双轴尾块 | 两个 tile 维度均存在尾块 | `[TILE_A + 尾块余数, TILE_B - 尾块余数]` |
   | 跨多 tile + 尾块 | 动态轴跨越 2–3 个 tile 并带尾块，验证跨 tile 循环与状态持久化 | `[2~3 × TILE_A + 尾块余数, TILE_B - 尾块余数]`（取触发多 tile 迭代的最小规模即可，不必放大，避免拖慢编译/执行或触发 OOM） |

3. **逐个验证 design 可适配**：对每个 case，核对当前 design（R2 tile 规划、R4 循环边界、R7 尾块处理、归约方案等）能否正确处理该 shape。**若发现某 case 适配不了，必须回溯对应轮次具体解决**（如尾块处理缺陷回 R7、循环边界错误回 R4、归约轴超单 tile 回 R0），**不得反过来删改 case 迁就 design**。
4. **例外**：若算子凑不出 4 个有区分度的 case，则以覆盖到的边界类别（整除 / 尾块 / 跨多 tile）为准，不为凑数量制造冗余用例；此时在 DESIGN.md §8 与 MEMORY.md 说明实际可覆盖的 case 数与原因。

**输出**：DESIGN.md §8 的"目标测试 case"表（case 名 / 具体 shape / 覆盖场景 / design 适配确认），至少 4 个（单轴算子按例外处理）。
---

### R8：综合评估

**核心问题**：整体设计是否准确、可靠、泛化？

**流程**：

> 评估时每项检查须标注 ✅（通过）或 ❌（不通过）；orchestrator 以评估表中无 ❌ 作为设计通过判据。

| 维度 | 检查项 | 结果 | 不通过时的处理 |
|------|--------|------|----------------|
| **准确性** | API 调用链是否完整实现了数学公式的每一步 | 回到 R1 补充 |
| | §4 伪代码核心计算步骤无 `...` 留空（`=` 赋值处的 `...`、标注"需 coder 实现"等均为不通过；sync 占位 `// sync: {目的}` 和 R7 尾块占位除外） | 回到 R1 补充 API 或标注 unsupported 触发回退 |
| | 数据依赖是否正确（Phase 顺序、sync 位置） | 回到 R0 或 R4 调整 |
| | dtype 选择是否能保证精度（如 matmul 累加用 FP32） | 回到 R2 调整 |
| | 归约类 API 的 `[M,1]`/`[1,N]` 输出已设 `layout` | 回到 R2 补 layout |
| | Acc tile 物理 M×N×dtype_bytes ≥ fractal（FP32 ≥ 1024 bytes；动态轴含极小维度时尤需检查） | 回到 R2 pad tile shape |
| **泛化性** | 目标测试 case（≥4，单轴算子按例外）已按 tile 切分确定具体 shape，且逐个验证 design 可适配（R7.5 已完成） | 回到 R7.5 补充 / 回溯适配不了的轮次 |
| | 是否正确处理了尾块 | 回到 R7 补充 |
| | 循环边界是否正确（ceiling division、valid_m/valid_n 计算） | 回到 R4 修正 |
| | 超越函数（exp/log/sqrt 等）在目标 dtype 范围内无溢出（§1 数值安全边界已分析） | 回到 R1 补溢出防护 |
| | 跨 tile 状态是否正确初始化和持久化 | 回到 R0 或 R1 修正 |
| | 同步策略在 SPEC §8 动态轴全范围下是否正确（num_cores、pipeline defer、event 分配是否随 total_work 变化而保持隔离） | 回到 R5/R6 修正 |
| **一致性** | R0-R7 各轮输出是否存在矛盾（如 API 需要的 tile 在 R2 中缺失） | 回溯到矛盾产生的轮次修正 |
| | 证据链是否完整（每个决策都有来源） | 补充缺失的文档引用或官方指定算子路径 |
| | 每个内存空间（UB/L1/L0A/L0B/L0C）的 tile 总用量分别不超过各自容量上限（R3 逐空间验证，含 cube 时须查 L1/L0） | 回到 R3 重排地址 / R2 缩 tile |
| **条件性检查** | 如跳过 R6，确认该算子确实无跨核数据传递 | 回到 R0 重新评估 |

**迭代规则**：
- 发现问题数 ≤ 3，修复后重新走 R8
- 发现问题数 > 3，回到问题最早出现的轮次，重新过后续轮
- 最多 5 次完整 R0-R8 迭代

**输出**：评估结论 + 修改记录（如有）。

---

### Tile 数据流全景图

R8 评估通过后，将 R0-R7 各轮的分散产出串成一张全景图（填入模板 §10）。这张图应能单页展示算子中所有数据流的全貌。

**绘制方法**：

1. **起点**：输入张量
2. **数据搬运**：标注 `load_tile` 、 `store_tile` 与流水线（MTE2 / MTE3）
3. **数据流向**：用箭头 `→` 连接每个操作，箭头标注 API 名。同一块 tile 被多个操作串联使用时用 `├─` 表示分支
4. **跨 Phase 持久化**：tile 在某个 Phase 中写入、在后续 Phase 中读取的，用 `---` 虚线表示数据跨越 Phase 边界。标注 tile 变量名和地址
5. **输出**：输出张量

**验证**：全景图中每一块 tile 和每一个操作都必须能在 R3 的地址映射表、R1 的 API 序列中找到对应条目。缺失或矛盾则回溯修正。

---

## 设计原则

1. **每个决策必须有证据**：API 文档引用、官方指定算子路径、教学文档、数学推导至少占其一
2. **Tile 数据流图是给 coder 的初步设计**：coder 拿到 DESIGN.md 应能确定 kernel 的完整结构与关键决策；API 签名等细节须以 API 文档原文为准确认（EXPLORE_REPORT 仅为派生的先行速查，不作签名权威）；运行验证暴露设计失误时 coder 可据实修正
3. **地址分配精确到字节**：不写"大约"、"若干"
4. **两条性能强制不可违背**（见上方「两条性能强制」节）
5. **模板中标注「性能强制」的措辞不可改写**：若发现 vf.* 路径不可行，须按 R1 流程第 5 步标注 unsupported 并触发回退，**不得在 DESIGN.md 中将「vf.* 指令序列」改写为「pl.* Vector API」或其他非 vf 措辞**。篡改模板措辞以匹配非 vf 方案视为设计失误
6. **参考官方指定算子优先于自行设计，拥有最高优先级**：遇到同步策略、tile 尺寸、vf 指令组合等决策时，优先查阅官方指定算子（PRO_MATERIAL_INDEX §B）中相似者，复用成熟模式；无相似时以 API 文档 / 教学文档为准