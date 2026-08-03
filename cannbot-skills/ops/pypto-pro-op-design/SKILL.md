---
name: pypto-pro-op-design
description: Stage 3 架构设计。通过 9 轮迭代式约束收敛，基于 Stage 1 产物（SPEC.md、EXPLORE_REPORT.md），产出 DESIGN.md。核心输出为 tile 级别数据流图——决定 Module 划分、API 映射、Tile 规划、片上空间布局（UB/L1/L0）、循环与 Section 结构、分核策略、核间同步、尾块处理。每一步决策必须有 API 文档、教学文档或官方指定算子做证据，严禁猜测。触发词：生成设计方案、tile 数据流、DESIGN.md、tile 级别设计、片上空间规划、UB 空间规划。
---

# PyPTO-Pro Stage 3 — 迭代式方案设计

通过 9 轮迭代式约束收敛（R0-R8），**目标** 是生成可直接翻译为 kernel 代码的 DESIGN.md。

**核心原则**：
- 每个决策必须包含**结论 + 推导过程 + 证据来源**
- 力求后续 Agent 拿到 DESIGN.md 即可确定 kernel 的完整结构与关键决策；API 签名等细节仍须由 coder 以 API 文档原文为准确认（EXPLORE_REPORT 仅为派生的先行速查，不作签名权威），运行验证暴露设计失误时可据实修正
- 每轮发现的矛盾必须回溯修正前序决策，不允许累积到 R8 再处理
- 本 skill 以**思维方法指导**为主，不教具体写法——具体 API 用法、tile 配置、同步写法等请查阅 API 文档（`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/`）、教学文档（`$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials`）、官方指定算子（见 `PRO_MATERIAL_INDEX.md` §B），理解后据实设计

## 两条性能强制（设计阶段须落实）

> 完整定义见 `../../references/performance-constraints.md`。设计阶段须在 R3（地址分配）和 R1（API 映射）中落实：
> 1. 所有需要 buffer 切换/轮转的 tile 一律用 `make_tile_group` + `auto_mutex`，`make_tile` 仅限单次使用 scratch tile。手动 sync 的严格界限（auto_mutex 管辖范围、跨核同步用 `set_cross_core`/`wait_cross_core`、mutex_id 与 event_id 独立命名空间）见该文件。R3 落实 buffer 管理方式，R6 落实 cross_core 同步方案。
> 2. Vector 数值计算用 `vf.*` 手写（完整理由见该文件）。

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

在切 Module 之前先确认 kernel 的输入/输出维度契约——这是后续所有轮（Tile 规划、循环结构）的前提，不属于 Module 划分本身。

明确 kernel 固定处理的维度（如"kernel 只处理 2D `[M,N]`"）。若 SPEC.md 要求支持 1D 或任意多维，必须在此定义 host 端适配方案（如 1D `[L]` → host reshape `[L,1]` → kernel `[M,N]` → 输出 reshape 回 `[L]`），不能默默只实现 2D 而遗漏 SPEC 要求的其他维度。产出填入 DESIGN.md §0 的维度契约栏（kernel 固定维度 + host 适配规则表）。

---

## 迭代设计流程

设计是**问题驱动**的迭代，不是线性填表。每一轮聚焦一个核心问题（回溯原则见核心原则第 3 条）。

### R0：Module 划分

**核心问题**：整体计算流可以分解为几个 Module？每个 Module 做什么？

**流程**（按优先级）：

1. **数据依赖**：需要前面 Module 的完整结果才能开始计算的，必须划分为不同 Module。例如跨 N-tile 的 softmax：必须先遍历所有 tile 算全局 max（Pass1），才能算 exp(x-max) 并累加全局 sum（Pass2），最后做除法（Pass3）。
2. **Section 分隔**：Cube 和 Vector 使用不同引擎，必须划分到不同 Module。Section 间需明确数据传递方式（via GM workspace）

**输出**：

- Module 列表（Phase1/2/3...），每个标注：目的、输入依赖、输出、涉及的 Section
- 如果只有一个 Module，标注"单 Module"并说明为什么不需要拆分
- **`module_interfaces.yaml` 契约**（机器可读，single source of truth）——产出到 `custom/<op>/module_interfaces.yaml`，包含以下字段：
  - `module_count`：Module 总数
  - `is_fusion`：是否为融合算子（同时含 cube 和 vec section → `true`）。按 R0 定义，`is_fusion=true` 隐含 `module_count >= 2`（Cube 和 Vector 必须划分到不同 Module）。此字段决定 Stage 4 走 L0（`false`，一口气开发）还是 L1（`true`，逐 Module 循环）
  - `has_cross_core`：是否涉及 cross_core 跨核流水（来自 §6，信息记录用，不影响分流判据）
  - `modules[]`：每个 Module 的 `id` / `name` / `description` / `section`（`cube` 或 `vector`）/ `golden_steps`（该 Module 对应的数学步骤列表，供 mathematician 切分 golden 用）/ `inputs`（source 为 `primary` 或 `module_<j>`，`j < 当前 id`）/ `outputs` / `golden_stage_fn`
  - `final_outputs`：每个 golden 返回值对应到产出 Module
  - `composition_verification`：atol / rtol / seeds / shapes
  - 骨架由脚本生成：`python ./scripts/gen_module_interfaces.py custom/<op>/<op>_golden_cpu.py --spec custom/<op>/SPEC.md --op <op> --design custom/<op>/DESIGN.md > custom/<op>/module_interfaces.yaml`，自动填 `schema_version` / `op` / `primary_inputs` / `composition_verification`，architect 填标 `TODO` 的判断部分
  - 产出后须自验：`python ./scripts/validate_module_yaml.py custom/<op>/module_interfaces.yaml --json`，返回 `"status": "PASS"` 才算完成

---

### R1：API 映射

**核心问题**：每个数学步骤具体用哪些 API？在 R0 的 Module 划分基础上，将 API 调用链细化到 Module 内部每一步操作。

**流程**：

1. **提取初步映射**：从 EXPLORE_REPORT.md §3 获取已确认的 API 映射（vec 步骤为 `vf.*` 序列，cube 步骤为 `pl.*` Cube API）。
2. **逐个核实 API 文档**（通过 `PRO_MATERIAL_INDEX.md` §A 定位路径），确认并记录：
   - **功能**：API 做什么、语义是否与数学步骤完全匹配
   - **签名**：参数顺序、位置参数 vs 关键字参数（以 API 文档原文为准，EXPLORE_REPORT §3.3 已汇总部分）
   - **使用约束**：dtype 限制、shape 要求、layout 要求、MemorySpace 要求、tmp tile 是否可与输入重叠等（以 API 文档原文为准，EXPLORE_REPORT §3.3 已汇总部分）
   - **特性特点**：in-place 支持、尾块行为、与 `set_validshape` 的交互等
3. **标注输入/输出 tile**：为每个 API 标注输入 tile 与输出 tile（供 R2 tile 规划消费）
4. **数值安全边界**（条件性，API 链含 exp/log/sqrt/reciprocal/tanh 等超越/非线性函数时强制）：分析该 API 在目标 dtype 下的溢出边界，在 §1 数值安全边界栏记录"输入范围→是否溢出→防护措施"。例如 fp16 max≈65504，exp(11.09)≈65504，输入 >11 即溢出为 +inf，后续 inf/inf → NaN 传播；防护如 exp(x) 写作 exp(x-max)。注：Vector 数值计算中的非线性须用 vf 指令实现（如 `vf.exp_sub`），其溢出行为以 vf API 文档为准

**注意**：本轮只确定计算 API，不涉及同步 API。

**输出**：Module 级 API 调用序列（伪代码形式）

---

### R2：Tile 规划

**核心问题**：需要哪些 tile？每个 tile 的 shape、dtype、layout 是什么？

**流程**：

0. 按模板 §2.1 的格式填写关键常量。列出所有 tile 尺寸（TS、TD 等）和派生常量（SCALE 等）。tile 尺寸是编译期对齐值，运行时通过 R7 设计的策略处理尾块场景（维度 < tile、整除、N+1 块且尾块不满）。公式常量若依赖实际维度值而非 tile 尺寸，应使用实际值
1. 根据 R1 的 API 序列，列出所有涉及的 tile
2. 对每个 tile，确定：
   - **shape**：由 API 操作数要求决定
   - **dtype**：由用户需求决定
   - **内存空间（`target_memory`）**：由该 tile 所参与的 API 决定，此列直接决定 R3 的分节归属
   - **layout**：默认按内存空间取，具体约束以 EXPLORE_REPORT §3.3 / API 文档为准。
   - **大小**：`prod(shape) × dtype_bytes`

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

1. 按 `target_memory` 把 R2 的 tile 分组，**每个内存空间各自从 `0x00000` 开始**连续排列地址，不重叠。UB/L1 首地址须 32 字节对齐；L0A/L0B/L0C 的对齐以对应 API 文档 / 官方指定算子为准。
2. 标注同地址不同 layout 的 tile 对（如有）
3. 分配方式应使用 `make_tile_group` + `auto_mutex`，由框架自动管理 buffer 切换与同步（见上方「两条性能强制」）。
4. **逐空间**验证该空间上的 tile 总大小不超过其容量上限。容量值以 EXPLORE_REPORT §7 探测记录为准——§7 必含 UB 容量；含 cube 时须补探 L1/L0 各空间容量（§7 未记录则回退 material-explore 补测，不得在此臆测数值）
5. **double buffer 地址规划**：`make_tile_group` 的 buffer 数 > 1 时地址占用按倍数放大，须在地址表中显式反映（buffer 数、受影响 tile、是否需 PONG 地址）。buffer 数取值参照官方指定算子中相似算子的实际配置

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

**核心问题**：本算子的循环嵌套、section 划分、SPMD 原语获取应如何组织？

> 各算子在循环与 Section 结构上差异极大（单/多 section、单/多 Module、是否含 cross_core 流水、SPMD 原语位置等），不存在通用设计方法。**必须**以官方指定算子的实际写法为主要参考，不自行臆造。

**流程**：

1. **首选参考 EXPLORE_REPORT.md §4 定位的有参考价值的样例推荐**：研读该样例的循环嵌套、section 声明、SPMD 原语获取位置等写法，作为本算子结构的主要参考
2. **参考范围扩展**：若 §4 定位的样例与本算子结构差异较大或细节不足，可在 `../pypto-pro-material-explore/references/official_samples.md` 清单中的其他官方指定算子中寻找结构更相似的样例参考
3. **基于样例确定本算子结构**：综合 R0 Module 划分、R1 API 序列、R3 空间规划，参照样例写法确定本算子的 section 划分、循环嵌套、SPMD 原语位置
4. 编译期常量块（TS、TD、SCALE 等）须置于 `@pl.jit` 装饰器之前（模块级）；具体声明位置的编码规范由 develop skill 负责

**输出**：填入模板 §4：
- 参考样例路径与可复用结构点
- 本算子的 section 划分、循环嵌套、SPMD 原语位置

---

### R5：分核策略

**核心问题**：work item 如何分配到各物理核？

> 📌 **权威依据（必读，一切以此为准）**：`$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials/programming_guide/programming_model/AI_Core_SIMD_programming/tile_based_python_programming/multi_core_partitioning_and_Tiling.md`。分核策略全部照该文档执行，与经验推断冲突时以该文档为准。

**本轮须在 DESIGN.md §5 落实的产出**：
- 分核方案
- host 侧 `num_cores` 计算式

**输出**：DESIGN.md §5（引用 multi_core_partitioning_and_Tiling.md，填入上述两项产出）

---

### R6：核间同步（cross_core）

**核心问题**：若算子含 Cube↔Vector 跨核数据传递，需手动插入哪些 `set_cross_core`/`wait_cross_core` 同步点？

> 当前 PyPTO-Pro 框架下，核内同步（pipe 间依赖、buffer 互斥等）由 `auto_mutex` 自动管理，**无需设计阶段关心**。**唯一需要手动插入同步的是 cross_core**——即同物理核的 Cube↔Vector sub-block 间通过片上共享 buffer（L1/Mat 或 UB/Vec）传递数据时的 `set_cross_core`/`wait_cross_core`。同步分工与命名空间（`mutex_id` vs `event_id` 独立）见 `../../references/performance-constraints.md`「强制 1」的同步分工表。
>
> **条件性**：仅当 R0 Module 划分含多 section 且 section 间有数据流时才需要 cross_core 同步。单 section 算子（纯 vec / 纯 cube）无跨核数据传递，本节填"不涉及 cross_core"即可。

**⚠️ cross_core 同步关键规则（必读，违反将导致编译失败或运行错误）**：

> **必读参考样例**：`$PYPTO_DEVKIT_DIR/pro_ops/` 中的 `lightning_indexer` 系列（如 `test_quant_lightning_indexer_vf.py`）是 cube↔vec 双向跨核同步的官方验证样例，包含完整的 `set_cross_core`/`wait_cross_core` 用法。设计 cross_core 同步时**必须**参照该样例。

1. **pipe 类型规则**：cube section 使用 `pipe=pl.PipeType.FIX`，vector section 使用 `pipe=pl.PipeType.V`。
2. **`set_cross_core`/`wait_cross_core` 与 `set_intra_block`/`wait_intra_block` 的关系**：`set_cross_core`/`wait_cross_core` 是用户编写的 API；在 a5 平台上，编译器会自动将 cube↔vector 的 `set_cross_core`/`wait_cross_core` 编译为底层的 `set_intra_block`/`wait_intra_block`（因为 cube↔vector 在同一 AI core block 内）。**用户不需要也不应该直接调用 `set_intra_block`/`wait_intra_block`**——始终使用 `set_cross_core`/`wait_cross_core`
3. **event_id 管理**：`auto_mutex` 的 `mutex_id` 与 `cross_core` 的 `event_id` 是**独立的命名空间**，可以共存（部分数值重叠不影响正确性——框架会正确区分）。`event_id` 取值范围 `[0, 16)`，多组流水各占不重叠区段

**流程**：

1. **确认是否涉及 cross_core**：由 R0 Module 划分判断。无跨 section 数据流 → 填"不涉及"，结束本轮
2. **参照官方指定算子确定同步方案**：涉及 cross_core 时，研读 EXPLORE_REPORT §4 定位的官方指定算子（如 FA、lightning_indexer 类）中 `set_cross_core`/`wait_cross_core` 的实际插入位置、pipe 类型、event_id 分配，参照其写法确定本算子的同步方案。
3. **event_id 隔离**：多组流水各占一段不重叠的 event_id 范围（`[0, 16)`），具体隔离方案以官方指定算子的实际用法为准

**输出**：填入模板 §6：
- 是否涉及 cross_core（不涉及则填"不涉及"并结束）
- 涉及时：cross_core 同步点表（位置 / set 或 wait / event_id）+ event_id 分配表

> 同步点的具体插入位置与参数以官方指定算子原文为准，不臆造。

---

### R7：尾块处理

**核心问题**：如何处理维度不整除 tile 尺寸的尾块？

> 📌 **权威依据（必读，一切以此为准）**：`$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials/programming_guide/programming_model/AI_Core_SIMD_programming/tile_based_python_programming/tail_block_handling.md`。尾块的完整机制全部照该文档执行，与经验推断冲突时以该文档为准。

**本轮须在 DESIGN.md §7 落实的产出**：将该文档的尾块机制落到本算子的循环与 Section 结构（填入 §7 尾块处理方案）。

**输出**：DESIGN.md §7（引用 tail_block_handling.md 落地尾块代码）

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
| | §4 已参照官方样例确定循环与 Section 结构（含参考样例路径与结构说明） | 回到 R4 补充 |
| | 数据依赖是否正确（Module 顺序、sync 位置） | 回到 R0 或 R4 调整 |
| | dtype 选择是否能保证精度（如 matmul 累加用 FP32） | 回到 R2 调整 |
| | 归约类 API 的 `[M,1]`/`[1,N]` 输出已设合适的 `layout` | 回到 R2 补 layout |
| **泛化性** | 目标测试 case（≥4，单轴算子按例外）已按 tile 切分确定具体 shape，且逐个验证 design 可适配（R7.5 已完成） | 回到 R7.5 补充 / 回溯适配不了的轮次 |
| | 是否正确处理了尾块 | 回到 R7 补充 |
| | 循环边界是否正确（ceiling division、valid_m/valid_n 计算） | 回到 R4 修正 |
| | 超越函数（exp/log/sqrt 等）在目标 dtype 范围内无溢出（§1 数值安全边界已分析） | 回到 R1 补溢出防护 |
| | 跨 tile 状态是否正确初始化和持久化 | 回到 R0 或 R1 修正 |
| | cross_core 同步方案是否正确（涉及跨 section 时）：同步点位置与 event_id 隔离是否参照官方指定算子 | 回到 R6 修正 |
| **一致性** | R0-R7 各轮输出是否存在矛盾（如 API 需要的 tile 在 R2 中缺失） | 回溯到矛盾产生的轮次修正 |
| | 证据链是否完整（每个决策都有来源） | 补充缺失的文档引用或官方指定算子路径 |
| | 每个内存空间（UB/L1/L0A/L0B/L0C）的 tile 总用量分别不超过各自容量上限（R3 逐空间验证，含 cube 时须查 L1/L0） | 回到 R3 重排地址 / R2 缩 tile |
| **条件性检查** | 若 §6 填"不涉及 cross_core"，确认该算子确实无跨 section 数据传递 | 回到 R0 重新评估 |

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
4. **跨 Module 持久化**：tile 在某个 Module 中写入、在后续 Module 中读取的，用 `---` 虚线表示数据跨越 Module 边界。标注 tile 变量名和地址
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