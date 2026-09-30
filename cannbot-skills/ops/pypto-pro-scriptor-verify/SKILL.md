---
name: pypto-pro-scriptor-verify
description: 对给定阶段的 PyPTO-Pro 产物做独立语义与执行验收，并封存与当前候选及原始证据绑定的报告。
---

# 独立验收

**目标**：独立核对候选是否满足冻结合同和给定阶段的要求，生成可追溯、可复算且不改变工作流状态的封存报告。

**输入与路径**：共同输入为算子目录、验收阶段及冻结的 SPEC/Golden；candidate/accept 另接收预期同步模式、合同列出的全部 `KB_SELECTION.json`、已解析 baseline/`exit_criteria`、DSL/导出候选和本次原始检查与性能证据。optimize 还必须接收完整的知识卡片覆盖账本及本轮关联的实验 `experiment_id`。阶段检查项、报告格式和封存命令以 handoff 为准。

**方法**：独立核对公式、reference、输入域和 KB 必需约束；candidate/accept 再核对单 runtime kernel、host 边界、DSL/导出对应关系和同步模式。用 snapshot 绑定产物，记录有位置与依据的语义审阅，实际执行相应 check；candidate/accept 对全部 P0 完整 shape 调用公开 wrapper 上板并按 SPEC 精度逐 case 比较。核对逐案 `input_special_values` 是否实际出现在本次 wrapper 输入，不能以有限替代样本或另一个同步模式的诊断代替原 case；worker 的直接 `OpExec` 证据若与冻结模式不符须写入 findings 并拒绝 PASS。独立复算性能证据后按 handoff 执行 observe/seal；缩小 probe 和模拟只证明诊断范围。

优化轮另核对本轮假设、实际改动及与已验证最佳候选的可比证据，在现有 review 中说明 keep/reject 理由；检查实际接口、shape/dtype 与来源，不能用旧原型分数或 case 名称代替本轮测量，也不以 worker 的“已充分优化/不支持”断言代替证据。收尾前按模式入口独立核对有效轮数、基础项覆盖、选定候选的目标结果及方案穷尽依据，在本次 review 的 `findings` 保留结论与证据；工具总轮数或达标本身不证明可结束。

逐项核对知识卡片：正常完成 optimize 时，Active 表中的每个 `item_id` 都必须出现在账本并到达终态；用户停止、预算耗尽或阻断时，保留 pending/unknown/blocked 并在 findings 中说明。卡片正文、源码/生成物、必要 profiler 证据和适用性判断必须对应。适用且能力可用的 item 缺少改码、正确性、可比性能或机制证据时返回 FAIL；标记“不适用/不支持/已覆盖”时必须有依据和 `relations`，能力未知不能关闭。共享实验可以复用测量，但每个实验 `experiment_id` 都要有独立结论。verifier 复核冻结合同下的轮内性能证据；accept 只验收最终候选，不追加独立 perf-tune。

**异常处理**：设备或环境异常保留命令、日志和影响范围并交给 `pypto-pro-environment-check`，条件与证据未变化时不重复执行。`stage=prepare` 仅按兼容合同检查数学输入与 Golden。

**产物与验收**：返回语义审阅、封存报告路径、逐 case 必需检查、用户目标准出结果和实际 verdict。optimize review 还要引用知识卡片账本路径及 SHA-256。证据缺失为 UNKNOWN，失败或阻断保持 FAIL/BLOCKED；candidate/accept 仅在完整输入域、全部 P0 真机精度及其他必需项通过时为 PASS，性能只按冻结 `exit_criteria` 判定。

## 官方 AscendC 源用例验收补充路径

本节只在 handoff 明确给出 `workflow_mode=source_ascendc_case` 时启用；否则继续执行上文 formal/scriptor 验收，不改变其输入、阶段或 verdict 规则。source-led 验收不要求伪造 formal 合同，但必须接收 `SOURCE_CASE_CONTRACT.json`、`SOURCE_MAPPING.json`、独立 CPU reference、源用例输入和 A5 运行 manifest。

启用后先读[源用例功能与精度合同](../pypto-pro-scriptor-develop/references/ascendc-source-functional-precision.md)，独立核对其中的入口边界、布局探针和证据要求。

独立复核源实现、host/tiling/infer-shape（若有）与映射文件，确认 shape、dtype、layout、轴/边界、累加与 cast 顺序以及声明支持范围一致。实际在 A5 NPU 上完成编译、加载、launch、同步和 D2H；以非零 sentinel 初始化输出，检查完整输出已由 kernel 覆盖，再用不导入 PyPTO 的 CPU reference 比较全部元素并记录容差及最大误差。存在 host 侧计算或输出修补、未完成完整输出比较、launch 次数超出合同、或关键源信息为 unknown 时，不得给出 PASS。

source-led 只有上述证据全部通过时才返回 PASS，并在报告中标记 `single_case_only`；证据缺失为 UNKNOWN，执行或精度失败为 FAIL，环境阻断为 BLOCKED。该补充路径只验收功能和精度，性能按独立 handoff 处理。
