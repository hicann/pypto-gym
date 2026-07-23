---
name: pypto-pro-material-explore
description: PyPTO-Pro 资料探索。构建 PRO_MATERIAL_INDEX.md 资料索引（API 文档 + 官方指定算子样例 + 教程），基于索引从三个方向依次探索（API 映射/约束检查、指定样例参考、教程设计模式），产出 EXPLORE_REPORT.md。触发词：资料探索、API 探索、查找 API、PyPTO-Pro 有没有 xxx、支持什么 dtype、约束是什么、API 映射、可行性分析、这个算子能做吗、pl.api。
---

# pypto-pro-material-explore

构建 PyPTO-Pro 资料索引，基于索引从三个方向探索，为算子开发提供 API 映射、约束检查、样例参考和可行性分析。

## 输入

`custom/<op>/SPEC.md`（由 Stage 1 Step 1 的 `pypto-pro-intent-understand` 产出）。从 SPEC.md 中提取算子计算逻辑、shape、dtype 等需求信息。

## 输出

- **`custom/<op>/PRO_MATERIAL_INDEX.md`**：资料索引（§A/§C 动态扫描 + §B 官方指定样例固定清单）
- **`custom/<op>/EXPLORE_REPORT.md`**：三方向探索报告，使用 [templates/explore_report.md](templates/explore_report.md) 模板

---

## 两条性能强制（贯穿全流程，务必先读）

以下两条是性能约定，探索阶段的 API 映射须遵守，否则下游 design/develop 产出的算子性能可能不可接受：

1. **buffer 管理使用 `make_tile_group` + `auto_mutex`**：所有需要 buffer 切换/轮转的 tile（含 double buffer）一律用 `make_tile_group` 创建，由框架自动管理 buffer 轮转与互斥同步。`make_tile` 仅限**单次使用 scratch tile**——写入一次、读取一次、不参与任何 buffer 切换/轮转循环（如一次性中间结果暂存、不迭代的归约标量结果）。**禁止用 `make_tile` + 手动 `sync_src`/`sync_dst` 管理 buffer 轮转**。

   **auto_mutex 与跨核同步的分工**：
   - **auto_mutex 管核内**：auto_mutex=True 时，框架对 `make_tile_group` 的 tile 自动插入 `mutex_lock`/`mutex_unlock`（基于 mutex_id，取值 [0,31]），覆盖同一核内 pipe 间的 buffer 互斥。此时禁止对已由 auto_mutex 管理的 tile 再手动加 `sync_src`/`sync_dst`——手动叠加会产生依赖环导致 AICore timeout 死锁。
   - **跨核同步用 `set_cross_core`/`wait_cross_core`（手动）**：AIC↔AIV 跨核数据依赖由 `set_cross_core`/`wait_cross_core`（基于 event_id，取值 [0,16)）处理，auto_mutex 不覆盖。mutex_id 与 event_id 是两个独立命名空间（FA 样例中两者数值重叠共存无报错），不构成冲突。

2. **Vector 数值计算用 `vf.*` 手写**：所有 Vector 数值计算（含逐元素、归约、广播、非线性等）一律用 `vf.*` 指令手写，在 `section_vector()` 内通过 `@pl.vector_function` 执行。`pl.*` 计算 SAPI 不得用于 Vector 数值计算——无论在何处编写。`pl.load`/`pl.store`/`pl.load_tile`/`pl.store_tile`/`pl.range`/`pl.get_block_idx` 等数据搬运与控制流 API 不受此限。两种等价写法（以 vf API 文档与官方指定算子为准）：`@pl.vector_function` 装饰器（通常定义在模块级，从 `section_vector()` 内调用，见 FA；也可在工厂函数内定义，见 lightning），或 `@pl.inline` + `with pl.section_vf():` 块（见 vf API 文档示例）。vec 探索主路径是 **vf API 文档**。

   > **vf.\* 指令在 VF 寄存器中完成计算，寄存器不占 UB 空间**——因此 VF 路径的 UB 占用不会高于 Memory 级路径。UB 超限不是放弃 VF 的理由（UB 超限处理见 design SKILL R3 重排地址 / R2 缩 tile）。

---

## Step 1：构建资料索引

PyPTO-Pro 资料处于持续更新中，**每次执行必须重新扫描 §A/§C**；§B 为官方指定样例清单（从 [references/official_samples.md](references/official_samples.md) 读取，该清单是整个工作流的统一官方样例索引来源）。

> **核心理念**：先建图，后按图索骥。索引一次构建，全流程复用。

### 索引覆盖范围

| 资料类别 | 目录 | 扫描方式 |
|----------|------|----------|
| API 文档 | `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/`（递归） | `find` 获取所有 `.md` 文件 |
| 官方指定算子样例 | 见 [references/official_samples.md](references/official_samples.md) | 读取清单文件，不扫描 a5 全目录 |
| 教程文档 | `$PYPTO_DEVKIT_DIR/docs/pypto_pro/guide`（递归） | `find` 获取所有 `.md` 文件 |

> **§B 说明**：官方指定算子样例是**唯一的算子写法参考来源**，`$PYPTO_DEVKIT_DIR/pro_ops/` 下其余文件不得作为样例参考或索引对象（orchestrator 资源缓存准备时已按清单清理，仅保留清单内文件）。清单后续可能增减，增减时**只改 [references/official_samples.md](references/official_samples.md)**，无需改动其他文件。

### 生成方式

以 [templates/pro_material_index.md](templates/pro_material_index.md) 为骨架，§A/§C 通过 bash 命令扫描缓存动态填充数据行，§B 直接复制 [references/official_samples.md](references/official_samples.md) 的清单内容（核对是否与官方最新指定一致）。

### 输出要求

- 路径使用相对路径（从仓库根目录算起）
- 每个子类别标题后标注 `（N 文档）` / `（N 文件）` 的计数
- §A 按目录路径自动分组展示（按 `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/` 下实际子目录分节）
- §B 复制 `references/official_samples.md` 清单，不动态扫描
- §C 列出 `$PYPTO_DEVKIT_DIR/docs/pypto_pro/guide` 下全部 `.md`
- 若某个扫描目录不存在或为空，保留空表（标注 `<!-- 空 -->`）

---

## Step 2：三方向依次探索

> **探索目标**：为 `custom/<op>/SPEC.md` 中的算子需求服务——三个方向的探索都围绕 SPEC.md 中的算子公式、shape、dtype、计算逻辑展开，目的是验证可行性、确定 API 映射、提取约束、找到可复用样例。脱离 SPEC.md 的泛化探索无意义。

**以下所有探索的搜索以 `custom/<op>/PRO_MATERIAL_INDEX.md` 为权威目录**——从中查找目标路径，而非在文件系统中盲目 grep。若索引中未找到所需资料，再回退到文件系统补充搜索。

直接使用 `read`/`glob`/`grep` 依次串行完成以下三个方向的探索，不要拆分或 dispatch 子代理。逐方向完成后进入下一方向，最后统一汇总。

### 方向 1：API 文档与约束

**搜索范围**：`$PYPTO_DEVKIT_DIR/docs/`（基于索引 §A）

**任务**：

1. 从索引 §A.1 查 `$PYPTO_DEVKIT_DIR/docs/pypto_api_list.md` 获取 API 总索引
2. 将算子计算逻辑分解为原子操作序列，对每个操作从索引 §A 中查找对应 API 调用链
   - **公式分解以 SPEC 为准**：以 SPEC.md 中用户给出的数学公式为基准进行分解，不自行推导替代公式。仅当某步骤需要数值近似实现（如 erf/sigmoid 无直接 API，需多项式近似）时才进行近似推导，并代入 2-3 个已知正确值验证（如 erf(0)=0、erf(1)≈0.843），在 EXPLORE_REPORT §2 记录验证结果
3. **vec 步骤的 API 映射主路径是 vf API 文档**：算子中所有 Vector 数值计算（elementwise、归约、非线性、排序等），须从 §A 中查找对应的 `vf.*` 指令文档，映射到 vf 指令序列。Cube 步骤（matmul 等）照常映射 `pl.*` Cube API。
4. 逐文档读取相关 API 文档，提取约束：dtype 支持、shape 范围、layout 要求、MemorySpace 约束
5. **vf 指令参数语义强制记录**：对映射到的每个 `vf.*` 指令，读其文档全文，记录：每个参数的语义（如 offset 单位是字节还是元素、layout 参数的可选取值及行为）、寄存器级行为（如 `vf.astype` 如何在 BF16/FP32 寄存器间映射、输入输出寄存器数量关系）、与 `pl.*` 同名 API 的差异（如 `vf.gather` vs `pl.gather` 的参数签名差异）。若文档含 layout/参数范围等约束表则逐一记录。vf 文档结构以实际为准。
6. 提取 Tile 规格约束（TileType 文档）、MemorySpace 约束、DataType 枚举值
7. **探测关键常量**：从 API 文档与教学文档中提取硬件/版本相关常量——UB 容量上限（直接查 `多核切分与Tiling.md` §5.2，A5/DAV_3510 为 248KB）、cross_core event_id 上限（`max_event_id` 默认值）、地址对齐要求、Cube tile 对齐要求等，记录值 + 文档路径
  8. 未找到直接对应的 vf API → **优先尝试用其他 vf API 组合 + 循环结构手动实现**。在 EXPLORE_REPORT §3 中记录组合方案及可行性分析依据。仅当穷尽 vf 组合方案仍不可行时，才标记 unsupported 并说明已尝试的组合路径
9. **动态维度声明方式**：从 API 文档（如 Tensor 数据结构文档）和官方指定算子样例中确认动态维度的正确声明方式，在 EXPLORE_REPORT §3 中记录

**返回**：API 映射表（数学步骤 → API 调用链，vec 步骤为 `vf.*` 序列、Cube 步骤为 `pl.*` 序列）、约束检查结果（含 vf 指令参数语义/寄存器级行为）、动态维度声明方式、环境常量（UB 容量/event_id 上限/对齐要求/Cube tile 约束 + 来源路径）、证据路径列表

### 方向 2：官方指定算子样例

**搜索范围**：PRO_MATERIAL_INDEX.md §B 中的官方指定算子（固定清单）

> **注意**：指定算子是官方精选的实现参考，是最主要的写法参考。a5 目录下其余文件不得参考，质量无保障。

**任务**：

1. **全量阅读**索引 §B 中的所有官方指定算子样例（不遗漏任何一个）。每个样例按其 cube/vec 组成分三类记录参考价值：
   - **纯 Cube 样例**（matmul 类）：cube tile 管理、K 累加链、matmul/matmul_acc phase 用法、L1/L0 tile 布局、set_mm_layout_transform
   - **纯 Vec 样例**（elementwise 类、vf_api 类）：vf 指令组合、make_tile_group 双缓冲、auto_mutex 同步、load_align/store_align 模式
   - **VC 融合样例**（FA 类、lightning_indexer 类）：section_cube→acc_to_vec→section_vector 衔接写法、cross_core 流水、多 Phase 分块、bufid 管理
2. 根据当前算子的 cube/vec 组成，从对应分类中提取**直接可复用模式**（tile_group 用法、vf 指令组合、循环结构、同步策略、cross_core 流水等）
3. 从所有样例（含非对应分类）中提取**通用写法参考**——API 用法（如 vf.gather/vf.astype/vf.reduce_* 的参数语义与调用方式）、分核策略、同步事件管理、尾块处理等，这些写法与算子类型无关，任何样例都可能示范
4. **可复用模式列举**：按 cube/vec 组成分类列出可复用模式，标注来源样例。具体写法以样例源码与 API 文档为准
5. **探测关键常量**：从样例中提取 UB 容量上限（如 `assert {addr} <= {N}*1024`）；记录值 + 样例路径

**返回**：按 cube/vec 组成分类的样例参考表（路径 + 可复用点）、可复用模式（直接可复用 + 通用写法参考）、关键常量（UB 容量/stride 经验阈值 + 来源路径）

> **vf 写法学习路径**：官方指定算子未示范某 vf 指令时，直接查 vf API 文档（`PRO_MATERIAL_INDEX.md` §A 中 vf 相关文档）学习其用法——API 文档是 vf 指令写法的完整权威来源，不依赖样例逐一示范。注意：样例中的调用名与 API 文档名可能不完全一致（如样例用 `vf.exp_sub`，API 文档名为 `vf.fused_exp_sub`），查找时以实际 API 文档为准。

### 方向 3：教程与设计指南

**搜索范围**：`$PYPTO_DEVKIT_DIR/docs/pypto_pro/guide`（基于索引 §C）

**任务**：

1. 从索引 §C 中获取全部教程文档路径清单（以实际扫描结果为准，不预设固定文件列表）
2. 逐文档读取，提取与当前算子相关的设计模式：tile 尺寸建议、尾块处理、多核分摊等

**返回**：适用的设计模式、关键约束、参考的教程章节

### 探索结果汇总

三个方向依次探索完成后：

1. 合并 API 映射与约束检查结果（方向 1）——注意 vec 步骤须为 `vf.*` 序列
2. 合并样例搜索结果（方向 2），按 cube/vec 组成分类整理样例参考表与可复用模式
3. 合并教程建议（方向 3），补充设计策略
4. **合并关键常量**：汇总方向 1（event_id 上限/对齐要求/Cube tile 约束）和方向 2（UB 容量/stride 阈值）探测到的常量，填入 EXPLORE_REPORT §7 环境常量快照表，标注来源路径
5. 若存在多个高质量参考，列出 Top 3 并说明推荐首选及理由

### 生成报告

基于 [templates/explore_report.md](templates/explore_report.md) 模板生成 `EXPLORE_REPORT.md`。

---

## Checklist

### PRO_MATERIAL_INDEX.md

1. 文件存在且 §A/§C 为本次重新扫描生成、§B 为从 `references/official_samples.md` 复制的清单（非拷贝模板的 find 结果）
2. 三个一级章节（`§A` / `§B` / `§C`）存在且内容不为空
3. `§A` API 文档数量与 `find $PYPTO_DEVKIT_DIR/docs/pypto_pro/api/ -name "*.md" | wc -l` 结果一致（不遗漏任何文档）
4. `§A` 按 `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/` 下实际子目录分组，无遗漏
5. `§B` 与 `references/official_samples.md` 清单一致，不含 pro_ops 下其余文件
6. `§C` 下列出 `$PYPTO_DEVKIT_DIR/docs/pypto_pro/guide` 下全部 `.md` 文件
7. 所有路径为相对路径

### EXPLORE_REPORT.md

1. 文件存在
2. 以下章节存在且内容不为空：
   - `## 1. 概述`
   - `## 3. API 文档探索`（须包含 §3.1 API 映射结果 + §3.3 API 约束；vec 步骤映射到 `vf.*`，非高层 `pl.*`）
   - `## 4. 算子样例探索`（可标注「无匹配」但不可缺失 §4.1–§4.3 三个子章节）
   - `## 5. 教程与设计指南探索`（须遍历 §C 中索引的全部教程文档并给出适用性评估）
   - `## 6. Tile / 同步策略建议`（综合 §3+§4+§5 三个来源）
   - `## 7. 环境常量快照`（须含 UB 容量、event_id 上限、对齐要求等，标注来源路径）
   - `## 8. 风险评估`
   - `## 9. 证据索引`（须包含 §9.1 API 文档 / §9.2 算子样例 / §9.3 教程文档）
   - `## 10. 结论`
3. 无 "unsupported" 阻断项（或虽有但已给出替代方案）

---

## 错误处理

| 场景 | 处理 |
|------|------|
| 输入无法解析 | 引导用户提供公式或代码 |
| API 不存在 | **优先尝试其他 vf API 组合 + 循环结构手动实现**（记录组合方案与可行性依据）；仅当穷尽 vf 组合仍不可行时标记 unsupported，在风险中说明已尝试路径 |
| 约束不满足 | 标记 ✗，在风险中给出替代方案 |
| 无匹配样例 | 在「参考实现」章节标注「无匹配」，不阻断流程 |
| vec 步骤映射到非 vf 指令 | 纠正为 `vf.*` 指令序列，以 vf API 文档为准 |
