---
name: pypto-pro-op-plan
description: Stage 1 算子规划。首先加载 pypto-intent-understand 产出 SPEC.md，之后加载 pypto-pro-material-explore 构建 PRO_MATERIAL_INDEX.md 全量资料索引并产出 EXPLORE_REPORT.md，初始化 MEMORY.md。
---

# PyPTO-Pro 复杂 Kernel — Stage 1 规划

## 规划流程

> 先需求、后探索：资料探索须基于 SPEC.md 中的算子公式、shape、dtype 等明确需求进行，确保探索具有目的性，避免盲目搜索。

### Step 1：需求总结

**必须**加载 skill `pypto-intent-understand`，按其流程产出 `SPEC.md`。

### Step 2：资料探索

**必须**加载 skill `pypto-pro-material-explore`，先构建覆盖全流程的资料索引 `PRO_MATERIAL_INDEX.md`（后续所有 Stage 均以该索引为权威目录，不再依赖盲目 grep），再基于索引从三个方向并行探索：API 文档（公式分解、`pl.*` API 映射、约束验证 dtype / layout / MemorySpace / tile shape）、pro_ops 算子样例（结构相似实现参考）、教程文档（设计模式与关键约束），产出 `EXPLORE_REPORT.md`。

### 必要规划文件

先创建 `custom/<算子名称>/MEMORY.md`，随后立刻继续实现。

记忆文件必须包含：

- 任务摘要
- 参考位置（包括 `PRO_MATERIAL_INDEX.md` 的路径及索引中的 pro_ops 相似样例路径）
- **PyPTO-Pro API 映射**（来自 Step 2）：每个数学步骤 → `pl.*` API 调用链
- 规范化 golden 状态
- 已冻结条目
- 尝试历史
- 阻塞列表
