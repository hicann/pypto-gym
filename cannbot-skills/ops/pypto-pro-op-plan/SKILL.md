---
name: pypto-pro-op-plan
description: 将算子需求整理为 PyPTO-Pro 开发所需的规格、资料证据、kernel 契约和知识选择。用于建立可验证、可复用的完整算子规划产物。
---

# PyPTO-Pro 算子规划

整理需求、实现约束和资料证据，生成彼此一致的规划产物。需求规格化与资料探索分别复用
`pypto-pro-intent-understand` 和 `pypto-pro-material-explore`。

## 输入与产物

输入：用户已授权的算子需求、目标版本 devkit 缓存、KB 根目录和共享 `performance-constraints.md` 的路径。
已有安装可分别从 `$PYPTO_DEVKIT_DIR`、`$CANNBOT_CONFIG_ROOT/pypto-pro-op-kb/` 和
`$CANNBOT_CONFIG_ROOT/references/performance-constraints.md` 定位。所复用的两个 Skill 及其脚本须可加载。

使用任务指定的输出目录 `<op-dir>`，默认 `custom/<op>/`；目录名 `<op>` 与 SPEC 算子名一致。在其中生成：

| 产物 | 内容 |
|------|------|
| `SPEC.md` | 数学语义、公开接口、P0 cases |
| `PRO_MATERIAL_INDEX.md` | 本次目标版本资料目录 |
| `EXPLORE_REPORT.md` | API 可行性、约束和证据 |
| `KB_SELECTION.json` | 当前 class 的知识路由结果 |

调用上述 Skill 时传入本次 SPEC、输出目录、缓存和共享规则的实际路径，替换命令中的默认路径；
需求确认与资料阅读遵循各自 Skill 的规则。

## 生成与校验

复用已提供且校验有效的产物，按以下依赖补齐：先明确规格，再核实 API 可行性，最后依据完整事实生成知识选择。

### 1. 冻结需求语义

加载 `pypto-pro-intent-understand`，按已有授权生成或更新 `SPEC.md` 并验证。

### 2. 补充 kernel 契约

在 `SPEC.md` 末尾增加 `## kernel 契约补充`，记录通用需求模板尚未表达的实现语义：

| 字段 | 记录要求 |
|------|--------------|
| 辅助张量语义 | 确认是公开输入、模型参数还是内部临时量 |
| cast 边界链 | 记录输入、累加、后处理、输出各段的语义 dtype |
| 累加/写回语义 | 确认覆盖写、跨块累加或原子累加的数学要求 |
| 目标设备 | 运行时或 build 配置已指定 target 时使用指定值；未指定时默认 A5，并在 SPEC 标注这是默认假设 |

涉及数学语义或公开接口的修订按 intent-understand 校验；硬件实现的可选方案及待核实条件记入资料报告。

### 3. 探索目标版本资料

加载 `pypto-pro-material-explore`，以包含 kernel 契约的 `SPEC.md` 为输入，重新扫描资料
索引并生成 `PRO_MATERIAL_INDEX.md` 与 `EXPLORE_REPORT.md`。结论须保持 SPEC 的数学语义；
若发现公式、接口或 P0 case 有问题，返回 intent-understand，按其规则修订和校验。

### 4. 冻结知识选择

阅读输入 KB 根下的 `CONTRACT.md`、`ROUTER.md` 与 `topology-map.json`，为每个 class
逐 class 生成 `KB_SELECTION.json`：flat 布局落在 `<op-dir>/KB_SELECTION.json`，此时
`class_id` 必须为字面量 `"."`；split 布局逐一落在
`<op-dir>/<class>/KB_SELECTION.json`，`class_id` 必须等于该 class 目录名。

执行规则：

1. 从公式的计算拓扑和已确认 properties 路由，禁止按算子名称猜选。
2. **允许命中零个或多个拓扑，不做唯一选择、不做覆盖**：融合算子同时符合
   `multi-phase-fusion` 与其组成部分（如 `cube-matmul`、`row-reduction`）时全部选入。
   若公式不符合任何已声明拓扑，必须如实记录 `topologies: []`，不得强行选择最接近的类别。
   `[]` 只表示已完成公式路由且没有当前键命中，不能表示未知、未分析或跳过；信息不足时必须
   继续补充分析依据，任何实际命中都不得遗漏。
   数组中的每个元素都必须是 `topology-map.json.topologies` 的当前键；不维护本地枚举副本。
3. 收集全部命中 `topologies` 的并集（空并集合法）、properties、已确认 target 和 mandatory
   触发的全部 constraints（去重合并，不得截断、不得只收其一）。拓扑数组为空时，后三类
   路由仍须正常执行。
4. optional pattern 候选来自全部命中拓扑与适用 property modifier 路由结果的并集；只保留
   适用前提成立且会产生独立、具体设计作用的条目，不限数量。
5. 无适用 pattern 时设置 `no_matching_pattern: true`，但不得删除 required constraints。
6. 每条引用使用 KB 根相对路径，记录 class-specific reason 与当前文件 SHA-256；不得记录
   安装前缀、绝对路径或占位哈希。
7. `properties` 只来自 SPEC、cases 或已确认环境事实。target 未指定时按默认 A5
   触发 `constraints/arch-a5.md`；已明确为非 A5 时不触发。

布局、`class_id` 与完整字段合同以所提供 KB 的 `CONTRACT.md` 为准；路由算法和数据分别以同一 KB 的
`ROUTER.md` 与 `topology-map.json` 为准。

### 5. 收尾自检

将 `<skill-dir>`、`<kb-root>`、`<缓存绝对路径>` 和 `<op-dir>` 替换为实际路径，保留引号，运行只读检查器：
它复用 SPEC/索引校验器，检查报告结构、引用及样例/指南路径覆盖，以及 KB_SELECTION 的字段、路径和哈希。
另按公式步骤、接口约束和当前风险核对报告证据是否充分。

```bash
python "<skill-dir>/scripts/validate_plan.py" \
  --op-dir "<op-dir>" \
  --devkit "<缓存绝对路径>" \
  --kb-root "<kb-root>"
```

输出非 0 时一次性修正全部 `[FAIL]` 后重跑。

## 完成条件

- 四类规划产物均存在；
- SPEC 校验通过，探索结论与其数学语义一致；
- INDEX 的 §A/§B/§C 与本次缓存一致；
- EXPLORE_REPORT 没有未解决的 `unsupported` 阻断；
- KB_SELECTION 的 `schema_version` 等于 `topology-map.json` 中当前的
  `contract.contract_version`，所有路径和哈希真实可复核；
- 规格语义、资料结论与知识选择相互一致。

返回产物路径、已完成的校验及其结果、剩余风险和需要补齐的证据。
