---
name: pypto-pro-op-plan
description: Stage 1 算子规划。首先加载 pypto-pro-intent-understand 产出 SPEC.md，之后加载 pypto-pro-material-explore 构建 PRO_MATERIAL_INDEX.md 全量资料索引并产出 EXPLORE_REPORT.md，初始化 MEMORY.md。
---

# PyPTO-Pro 复杂 Kernel — Stage 1 规划

## 规划流程

> 先需求、后探索：资料探索须基于 SPEC.md 中的算子公式、shape、dtype 等明确需求进行，确保探索具有目的性，避免盲目搜索。
>
> **执行方式**：本 skill 由单个子代理在同一 session 内**串行加载** `pypto-pro-intent-understand` 和 `pypto-pro-material-explore`，非嵌套 dispatch 子子代理。先完成 intent-understand 产出 SPEC.md，再加载 material-explore 基于 SPEC.md 进行探索。

### Step 1：需求总结

**必须**加载 skill `pypto-pro-intent-understand`，按其流程产出 `SPEC.md`。

> `pypto-pro-intent-understand` 是 PyPTO-Pro 专属的需求理解组件，产出的是**通用 SPEC**（公式、dtype、shape、关键特性、优先级等）。它不覆盖 kernel 层的若干契约字段——这些由下面的 Step 1.5 在其产物基础上补齐。

### Step 1.5：Pro kernel 契约补充

在 Step 1 产出通用 SPEC.md **之后**，基于它补齐 kernel 层特有的契约信息。产物**追加到 SPEC.md 末尾的「## kernel 契约补充」节**（同时在 MEMORY.md 记一份裁定摘要），供 Step 2 探索与后续 design / develop 消费。

#### 1. 字段归属裁定（三档）

对每个契约字段判定「由谁定」，避免子代理在 Stage 1 凭空猜测本该由用户或 design 决定的事：

| 档位 | 含义 | 处理 |
|------|------|------|
| **ASK** | 公式无法消解、须用户拍板 | 已由 Step 1 的 intent-understand 采集；此处只核对是否齐全 |
| **MAY-DESIGN** | 既不猜也不问用户，交由 design 阶段设计 | 在补充节标注「留待 design」，**不在 Stage 1 定值** |
| **MAY-ASSUME** | 可取仓库 / 对齐默认 | 取默认并注明 |

逐字段归属：

- **ASK**（Step 1 应已确认）：目标公式、输入/输出 dtype。
- **MAY-DESIGN**（标注留待 design，不问用户）：输入张量 **shape**、**拓扑**（纯 vector / cube→vec / …，由 design R0 Module/Section 决定）、**尾块行为**（design R7/R7.5）、**tile 族 / 切分 / 片上地址**（design R2/R3）。
- **MAY-ASSUME**（取默认并注明）：**设备**默认 a5；**matmul 累加 dtype** 默认 float（与 golden 的 `.float()` 累加对齐）。

#### 2. 三个 kernel 契约字段（SPEC 通用模板未覆盖，此处必补）

这三项直接影响 tile 规划、精度与多核写回，通用 SPEC 模板无对应字段，须在补充节显式写出（无则注明「不涉及」）：

| 字段 | 说明 | 缺省判断依据 |
|------|------|-------------|
| **辅助张量暂存策略** | bias / mask / scale 向量等：调用侧预展开成满 tensor，还是 kernel 内用 vf 广播指令实现 | 影响 tile 规划；歧义时归 ASK 向用户确认。|
| **cast 边界链** | input → matmul 累加 → 后处理 → 输出 各段 dtype 及降精度时点 | 精度正确性关键；累加段默认 float |
| **累加语义** | 跨多次启动 / 多核：覆盖写 vs 原子累加（后者需输出预清零） | 多核归约场景必判，默认覆盖写 |

#### 3. 无用户应答时（自动化运行）

用户无法实时回复时，ASK 字段无法当场确认：仍按 intent-understand 写出该问的问题，再为每个未决字段取最佳推测默认值，并在 SPEC 补充节与 MEMORY.md **显式记录每个假设**，供后续追溯与用户事后复核。MAY-DESIGN 字段无需在此处理，正常流转到 design。

### Step 2：资料探索

**必须**加载 skill `pypto-pro-material-explore`，先构建覆盖全流程的资料索引 `PRO_MATERIAL_INDEX.md`（后续所有 Stage 均以该索引为权威目录，不再依赖盲目 grep），再基于索引从三个方向依次探索：API 文档（公式分解、`pl.*`/`vf.*` API 映射、约束验证 dtype / layout / MemorySpace / tile shape）、官方指定算子（典型算子实现参考）、教学文档（设计模式与关键约束），产出 `EXPLORE_REPORT.md`。

### 必要规划文件

`custom/<算子名称>/MEMORY.md` 最迟在 Step 1.5 创建（用于承接契约裁定摘要），全流程持续追加。Stage 1 结束时必须包含：

- 任务摘要
- **契约裁定摘要**（来自 Step 1.5：三档归属结论 + 三个 kernel 契约字段的取值 / 假设）
- 参考位置（包括 `PRO_MATERIAL_INDEX.md` 的路径及索引中的官方指定算子路径）
- **PyPTO-Pro API 映射**（来自 Step 2）：每个数学步骤 → `pl.*` API 调用链
- 规范化 golden 状态
- 已冻结条目
- 尝试历史
- 阻塞列表
