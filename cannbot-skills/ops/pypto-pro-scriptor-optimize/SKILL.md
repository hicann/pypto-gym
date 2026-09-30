---
name: pypto-pro-scriptor-optimize
description: 基于冻结合同和已验证候选，验证一个有证据的优化方向，更新 ascriptor DSL，并生成可独立核验的精度与性能证据。
---

# 优化候选

**目标**：依据当前候选的真实证据验证一个优化假设，在数学、接口和交付边界不变的前提下改善已解析的性能目标。

**输入与路径**：接收冻结的 SPEC、Golden、合同列出的全部 `KB_SELECTION.json`、当前与最佳 DSL/导出候选、全部 P0 检查与性能记录、已解析 baseline/`exit_criteria`、知识卡片覆盖账本、历史证据、本轮假设和用户限制；历史合同没有 KB 选择时沿用其既有输入。读取 develop 约定和 handoff；新上下文按 `doctor.source_root` 从固定源码的 `agent/` 执行 `tools/build_kernel_context.py --print`（直接读取，不做来源哈希校验；已读相同内容则不重复），随后读取源码快照 ROUTER 指向的 optimize playbook。专项 owner 资料只在瓶颈或症状匹配时展开。只执行分配的本轮实验；整个 optimize 的最低轮数和停止判定由主 agent 按模式入口负责，worker 不因本轮达标或无收益自行结束流程。

**知识卡片来源（必须覆盖）**：主 agent 在第一轮前读取 `$CANNBOT_CONFIG_ROOT/skills/pypto-pro-op-perf-tune/SKILL.md` 的 `knowledge_card` 规则、该 Skill 下 `references/knowledge-cards/index.md` 的 Active 表和全部卡片正文，并建立完整账本，保存为 `custom/<op>/reports/scriptor-optimization-ledger.json`，并在顶层记录 `active_index_sha256`。worker 不启动完整 `pypto-pro-op-perf-tune`，只执行 dispatch 指定的一张或一组卡片实验。账本至少保留 `item_id`、`source_kind`、`source_ref`、`case_scope`、`applicability`、`expected_metric`、`status`、`evidence`、`relations`；`source_ref` 含权威文件、原子锚点和内容 SHA-256；卡片项使用 `item_id`，实验关联另用 `experiment_id`；正常收尾前每个 item 都必须到达 optimization-playbook 定义的终态，用户停止、预算耗尽或阻断时保留实际的 pending/unknown/blocked。

适用且能力已确认的卡片，必须在 Scriptor 普通 optimize 轮中完成“改 DSL → 完整正确性 → 可比性能 → 机制核对 → keep/reject”。预期收益小或 profiler 未直接显示该现象都不能跳过；能力未知先补证。只有机制不成立、能力已证实不支持或同一改动已由其它实验 `experiment_id` 覆盖时，才可不再单独改码，并为每个实验 `experiment_id` 写结论和 `relations`。

**方法**：先核对候选、case、设备、统计口径和原始证据的可比性，再按瓶颈选择工作分配、复用、容量/布局或依赖方向；修改 DSL 后按冻结的 `delivery_sync_mode` 重新 export，完成 wrapper smoke、全部 P0 candidate check 和同口径测量。自行写的 PyPTO-Pro `OpExec` 真机诊断也须显式传 `sync_mode=delivery_sync_mode` 并核对源码与 manifest，不能沿用底层 manual 默认或生成另一模式的对照。保持每个 P0 的特殊输入分布。小 shape、少核 sim/pipesim 或模型只用于定位，修复后回到完整 P0 真机；环境问题携证据交给 `pypto-pro-environment-check`，仅在条件、实现或证据变化且预算允许时重试。

用源码快照 API 和样例确认优化机制的合法表达，不照搬 PyPTO 写法；报错先区分用法错误、环境故障和能力缺口。声称“不适用/不支持”须给出对应代码、条件或最小复现，不能凭一次失败关闭方向；失败候选先留给 verifier 封存，恢复顺序按模式入口执行。

**产物与验收**：返回本轮卡片 `item_id`/`experiment_id`、假设与实际改动、命令/日志、全部 P0 逐 case 真机精度、可复算性能比较和下一步；给出候选、账本更新及原始报告的明确路径，指标绑定当前候选、case、单位和来源。优化轮复用 Scriptor 已冻结的 case、设备和计时合同；同一候选的有效证据可按哈希复用，不能另起一套末尾 perf-tune 采集。verifier review 必须引用账本路径及 SHA-256；账本变化后重新审阅封存。SPEC/Golden/KB、输入域、容差、单 runtime kernel 与 host 边界保持不变；缺失项保留 UNKNOWN，由独立 verifier 封存实际结论，主 agent 汇总轮次证据。结论只覆盖实际验证的方向，未试方向如实保留，不凭几次参数扫描或工具停止码宣称基础优化和卡片覆盖已完成。
