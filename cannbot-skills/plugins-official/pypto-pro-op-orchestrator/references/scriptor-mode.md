---
name: pypto-pro-scriptor-mode
description: 用户选择 scriptor 模式时，先完成可上板原型，再完成 ascriptor DSL 实现、优化与验收；优化阶段吸收 pypto-pro-op-perf-tune 的知识卡片规则，不在末尾另起一套性能调优流程。
---

# Scriptor 模式

你仍是 `pypto-pro-op-orchestrator`。先完成需求、Golden、设计，再快速写一个可上板原型，然后用 ascriptor 实现并验收。原型满足 SPEC、编译通过、上板正常运行，精度尽力修复（全算子最多 3 轮）；后段恢复全部精度要求。原型只供实现参考，不能作为 Golden 或最终交付。**状态推进默认必须严格按 `scriptor-handoff-contract.md` 的「新任务动作序列」执行，只允许用其中的动作流转。** 步骤确实无法完成时，不要求强行走完序列：不得静默跳过或改用序列外方式代替，应记录具体阻断原因，并在任务收尾时一并汇报。

## 先看：主流程总览

1. **识别/恢复**：确定 `custom/<op>` 并先建立 Todo；再读取本文件、handoff、`scriptor-install.json` 并运行 `doctor`。已有 `.scriptor/state.json` 先 `status`；已有深度编排状态先停止自动切换并明确处理。
2. **准备**：完成 Plan；在 Golden 自验证前写入并冻结 `SPEC.exit_criteria`（用户未指定性能目标时默认所有 P0 的 `golden_reference_ratio ≥ 1.0×`；用户明确不要性能目标则为 `null`，用户已指定的目标原样保留）；完成 Golden 自验证和 Design，首次写原型前运行 `bootstrap-check`。
3. **Pro 原型**：主 agent 覆盖全部 P0 真机运行。仅数值精度最多修复 3 轮；编译、设备、同步或输入合同失败记为阻断，不计精度轮。保存 prototype、日志和 `pro-bootstrap.md`；数值精度 3 轮仍未通过时写 `UNRESOLVED` 后交 implement。
4. **DSL implement**：按合同执行 `init(from_pro)`（复核 bootstrap 回执，冻结 `delivery_sync_mode` 和优化策略）→ `worker(develop)` → export/check(candidate)。独立 verifier 全量检查并 PASS 后才 `complete_implement`；从此恢复完整 SPEC 精度要求。
5. **DSL optimize**：用户明确不优化时按合同跳过并记录；启用时，首轮前读取 perf-tune 的 `knowledge_card` 规则、Active 表和正文，建立 `item_id` 账本（详见下文）。基础项沿用原 Scriptor optimize playbook，只新增 knowledge-card 来源；默认至少 10 个有效轮且无上限，均走 `begin_round → worker → verifier → finish_round`，复用冻结证据，同一候选按哈希复用，新候选或条件变化重测。正常收尾须由主 agent/verifier 核对基础项、账本和停止依据；用户停止或预算按合同结束，阻断用 `blocked` 交接。
6. **accept**：从选定 DSL 按冻结的 `delivery_sync_mode`（默认 `auto_mutex`）重新导出；accept verifier 完成最终全量 P0 验收（精度、输入域、单 runtime kernel、host 边界、独立运行、基础时延和准出条件）后执行 `complete_accept(report)`。accept 仍须完成这些最终检查；同一冻结合同下最终候选的有效证据可复用，但不重复采集 baseline、Golden 或 final formal compare。新任务不走 sync-trial，完成后不另起 perf-tune。
7. **交付**：`complete_accept` 后，若启用优化则核对优化轮数、基础项和卡片账本；始终核对最终报告和 `outcome`。`done` 仅表示归档，FAIL/UNKNOWN 必须如实保留。再从选定 DSL 导出并运行 `delivery-check`，只有 `status=PASS` 才算交付通过，`STRUCTURE_PASS` 不足以通过。

硬约束：

- 只有主 agent 推进状态；阶段交接、工具停止码或“看起来已达标”都不能代替证据。
- 用户停止、预算限制或真实阻断须保留未完成项，并如实交接。
- accept 后不强制调用完整 `pypto-pro-op-perf-tune`；只有用户明确要求独立性能调优时才调用。

## 启动时建立并维护 Todo list（强制）

确定 `custom/<op>/` 后、第一次写原型或调用 CLI 前，主 agent 必须建立可见 Todo list（优先使用宿主 Todo 工具；没有时写入 `custom/<op>/reports/scriptor-todo.md`）。Todo 至少列出上面 7 个阶段、知识卡片逐卡覆盖、最终 `accept` 和 `delivery-check`，并为每项记录状态、证据路径和下一步。进入下一状态前更新；恢复会话时从 `status`、账本和报告重建，不清空历史。没有证据不能勾选完成；`pending/unknown/blocked` 必须保留到有结论或如实交付。

## 入口与恢复

- 按主入口完成公共资料准备，复用 `pypto-pro-docs-search` 与目标版本 KB，并按自主开发规则关闭 OpenCode 自动 lint。此模式不读 `orchestration.md`、不创建 `.orchestrator_state.json`，不调用 Stage/Module 状态工具。API、同步、容量、host 边界和数学合同约束仍生效。
- 完整模式当前使用 OpenCode。算子目录为 `custom/<op>/`；读取安装记录 `scriptor-install.json`，使用其中的 Python 运行 `$CANNBOT_CONFIG_ROOT/scriptor/scripts/scriptor.py --config-root $CANNBOT_CONFIG_ROOT`（下称 CLI）。缺少或更新安装资源时按 OpenCode 安装入口同步，重启会话并核对实际读取的入口与 `doctor` 返回的源码身份；源码更新后须刷新快照索引和 OpenCode 工作流资源，再核对 `doctor` 的源码路径与 ID。
- 已有 `.scriptor/state.json` 时先调用 `scriptor_transition` 的 `status`，沿 `next_request` 继续；不重复开发原型或重置轮数。旧 `fresh/optimize_existing` 任务处于 prepare 时，按交接合同的旧任务说明恢复。已有深度编排状态时停止自动切换，先明确处理旧任务。
- 仅主 agent 推进状态。前段的规划、Golden 与设计可自主执行或调度对应 Pro 子 agent，dispatch 仍带 `workflow_mode=scriptor-bootstrap`；原型由主 agent 按下文流程直接编写。后段由 worker 和独立 verifier 分工。
- 可并行分配评测脚手架和安装/流程梳理；在 `custom/<op>/reports/pro-bootstrap.md` 交接环境与安装路径、全部 case 映射、精度判据、baseline 来源/计时口径及可执行命令。主 agent 核对后继续负责实现、全量评测与预算推进，子 agent 完成前置工作不代表任务完成。

## Pro 原型：复用技能，不做前置验收

按依赖顺序完成；有效的已有产物先复核复用，不机械重做。

| 工作 | Skill / 可用角色 | 交付 |
|---|---|---|
| 需求、资料与 KB 选择 | `pypto-pro-op-plan` / planner | SPEC、资料索引、探索报告、KB_SELECTION |
| 独立数学参考 | `pypto-pro-golden-generate` / mathematician | CPU/NPU Golden、有效的 `GOLDEN_VALIDATION.json`；`collect_golden_perf=false` |
| 设计 | `pypto-pro-op-design` / architect | DESIGN、DESIGN_BINDINGS、Module 接口；以可实现为目标 |
| 上板原型 | 主 agent 直接编写（流程见下） | `test_<op>.py`、依赖和上板日志；融合算子也直接交完整 kernel |

交给 Golden skill 的请求须带 `workflow_mode=scriptor-bootstrap`：CPU Golden
在 FP32 下计算后，按每个 P0 的 SPEC `output_dtypes` 舍入输出，并提供纯 CPU 的
`_make_inputs(device)` 覆盖全部 P0。普通 Pro 的“CPU Golden 总返回 FP32”模板
不适用；用 Golden skill 的 Scriptor 专用模板。`bootstrap-check` 会独立调用
CPU Golden 逐案核对输入和输出 shape/dtype 及特殊值，失败须在原型前修复并重新
自验证 Golden。

设计照常调用 `pypto-pro-op-design`；它的 Module 划分与 Module 接口在 scriptor 模式不生效，可以完全忽略：后续原型按单个 kernel 一次写完，不要求按 Module 分段开发。

**原型阶段门禁**：Plan、Golden 自验证和 Design 均完成后，在首次编写
`test_<op>.py` 或 `prototype/` 前，执行 CLI
`bootstrap-check --op-dir custom/<op>`，保存 `.scriptor/receipts/pro-bootstrap.json`。
该命令核对 SPEC、Plan 索引与 KB 选择、Golden 回执、Design 三件套并绑定其哈希；
封存前逐案核对 SPEC 声明的 P0 输入、输出和 CPU Golden；回执只绑定这些
算子开发输入，不读取仓外任务文件。
`init(entry_mode=from_pro)` 必须复核该回执。检查失败先修复缺项，禁止用先写原型、
删除原型后补回执的方式绕过顺序。

封存后若确需修改 SPEC/Golden/Design，主 agent 在修改前调用
`bootstrap-restart --op-dir custom/<op> --reason '<具体原因>'`。CLI 会归档旧回执、
原型、编译目录与运行日志并移走活动原型；随后修复并自验证 Golden，再运行
`bootstrap-check` 生成带旧回执出处的新封存。若原型曾存在，必须用新合同重新运行
全部 P0 Pro 原型，重新写 `reports/pro-bootstrap.md`，其中写一行
`Bootstrap receipt SHA-256: <新 bootstrap-check 返回的 receipt_sha256>`，再交接
`init(from_pro)`。不得手工删除、移动 `.scriptor` 或旧回执；旧证据留在
`reports/bootstrap-restarts/`。若已误改封存输入，仍用该命令恢复并在报告中保留
原始违约与修复经过，不把恢复后的回执写成首次无误通过。已有 Scriptor state 的
合同变更仍用 `rollback_prepare`。

规划时逐项采用用户给出的 baseline、性能目标和优化要求。没有可比用户 baseline 时，以本任务 NPU Golden 为性能参考，并冻结首个正确 PyPTO 候选作为优化 baseline；未指定目标时默认每个 P0 case 的 `golden_reference_ratio ≥1.0×`。Scriptor optimize 默认至少完成 10 轮有效优化，无轮数上限：达标也须满足最低轮数和下文基础优化要求；未达标则继续，直到有充分证据确认暂无可行方案。用户明确的停止、不优化或预算限制优先，未满足默认要求时如实说明；显式项目要求次于 prompt，工具或旧安装的默认上限不视为用户限制。此处只确定规则，性能测量留到后段。

在 Golden 自验证前把性能要求写入 SPEC 的 `exit_criteria`，`perf_target` 保持 null，字段合同见 handoff。默认按 handoff 的冻结 compare 口径，对全部 P0 逐 case 验收硬件 `golden_reference_ratio >= 1.0`；用户指定的 baseline、目标、范围和组合逻辑原样保留。用户明确不要性能目标时保持 `exit_criteria=null`；明确不优化时关闭后段优化。恢复旧任务沿用冻结合同和状态。

这些技能负责自己的产物校验，不另外调度 Stage verifier 或 Scriptor preparer/prepare verifier。Golden 必须按公式独立实现并完成原有自验证；放宽的是原型 kernel 的精度，不能修改 SPEC、Golden、输入域或容差来迁就错误。

新任务默认在 SPEC 写 `"tolerance":{"policy":"pro_scheme_a"}`，前后段统一按已安装 Pro `precision_compare.py` 的方案 A 对独立 CPU Golden 比较。源任务附带的 MARE/MERE 等诊断不自动成为准出门槛；用户明确指定其他精度要求时先落实到合同与评测。旧合同保留原判据，变更须走下文的回退与重新自验证，不能改写旧报告。

**同步交付政策**：新 Scriptor 任务默认只生成 `auto_mutex=True` 的 PyPTO-Pro
候选与最终包；`export` 默认 `--sync-mode auto_mutex`。仅用户明确要求 manual 时，
在 `init` 写 `delivery_sync_mode=manual` 且 `manual_requested_by_user=true`，
后续显式 `export --sync-mode manual`。不得自行生成 manual 对照或在 auto_mutex
发射、精度、真机、性能准出失败后回退 manual；保存失败证据并报告阻断。
底层 Ascriptor `OpExec`/`compile_kernel` 的历史默认仍是 manual，因此此工作流
必须显式使用 auto_mutex 导出入口；不能把底层默认误认为交付政策。
DSL worker 自写的 smoke、完整 P0 或特殊值诊断若直接使用 `OpExec(..., launcher="pypto")`，
也必须显式传 `sync_mode=init.delivery_sync_mode` 并核对实际 manifest 和装饰器；
只对正式 export 指定模式不足以约束这些独立运行。错误模式的证据保留并报告阻断。
原任务的 Inf/NaN case 用 SPEC `p0_cases[].input_special_values` 逐输入声明，
有限 `value_range` 仅约束同一输入的有限元素，不能替换原始特殊输入。

原型由主 agent 直接编写：以 SPEC.md 为需求基准，运行 CLI `doctor`，用返回的 `agent_router` 路径读已安装源码的 `agent/zh-CN/ROUTER.md`（需要英文时读 `agent/en/ROUTER.md`），按 ROUTER 进 author playbook，按需加载写 kernel 的参考文档和可运行样例；再结合 DESIGN.md 的设计细节及其引用的 KB 模板、约束、样例写 `test_<op>.py`（`pypto_pro.language` kernel，DSL 重写是后段 implement 的事）。规范尽量守：单 `@pl.jit` kernel、公开 wrapper、一次启动、host 边界不变。
新任务的 Pro 原型也显式使用 `@pl.jit(auto_mutex=True)`；用户明确要求 manual 时才用
`auto_mutex=False`。原型出现原生同步能力问题时留下定位证据，不改用 manual 绕过。

原型测试覆盖全部 SPEC P0，按当前环境选择本地或远端真机。每个 case 先调用公开 wrapper 并完成 NPU 同步，再独立比较 CPU Golden；只捕获比较步骤的数值断言以继续其余 case，不让首个精度失败中止全量运行。编译、设备调用、同步、输入合同或参考实现异常均是阻断，不能吞掉或回退 CPU 冒充上板。性能测量留到后段。

精度不通过时，使用 `pypto-pro-precision-debug`，全算子最多 3 轮“诊断 → 修复 → 重编译上板复测”。首次对比不计修复轮；在 `custom/<op>/reports/pro-bootstrap.md` 简短记录当前原型路径、逐 P0 运行/精度结果、实际命令与日志、已用轮数和遗留问题。每轮开始前记账，结束后补结果；换 case、子 agent 或恢复会话不重置预算，子 agent 只执行分配的一轮。精度通过立即交接；3 轮后仍有数值误差，写 `UNRESOLVED` 和具体问题，直接交接 implement。编译或设备运行仍失败时修复运行问题或如实报告阻断，不把它计作成功交接。

## 交接到 implement

主 agent 根据当前源码对应的日志确认全部 P0 已完成设备运行；修改原型后要重新上板，旧日志不替代本次结果。将原型及本地依赖按原相对布局保存到 `prototype/`，确认副本后移走根 `test_<op>.py`，让 DSL exporter 接管该路径。保留已有备份，重交接时使用新的子目录并更新记录。SPEC、Golden 和自验证回执继续留在原位。

调用 `scriptor_transition`：

```json
{"action":"init","entry_mode":"from_pro","opDir":"custom/<op>","delivery_sync_mode":"auto_mutex","optimization":{"enabled":true,"max_iterations":null,"stop_on_criteria":false,"time_budget_s":null,"max_no_improvement_rounds":null}}
```

工具复用已有 SPEC 校验与 Golden 回执，冻结数学输入并直接进入 `implement`，不另加 prepare 验收。上例显式落实本模式默认策略，再按规划阶段已确认的用户/项目要求逐项覆盖；不能省略参数而继承工具的 10 轮上限与达标早停。`max_iterations=null` 表示无限，最低有效轮数由主 agent 按下文证据核对，工具没有 `min_iterations` 字段。多个 dtype/接口 class 共用全算子轮数，但基础优化须覆盖全部声明范围。恢复已有任务保留策略和历史；用户要求改用本规则时，结束优化前通过 `update_optimization` 更新并记录真实原因、不清零轮数，已进入 accept/done 则按 handoff 的新请求入口处理。

`init` 之前不得运行 `export`/`check` 并据其结果推进实现；`export` 拒绝缺少状态的
工作目录并核对冻结的交付同步模式，单独跑通 `check` 不代表状态流程已推进。

## DSL 实现、优化与验收

1. **implement**：调度 `pypto-pro-scriptor-worker(mode=develop)`，加载 `pypto-pro-scriptor-develop`。传入四类 Plan 产物、Golden 与回执、Design 产物、原型备份和 `custom/<op>/reports/pro-bootstrap.md`；原型提供 API/布局参考，数学以 SPEC/Golden 为准。DSL worker 从 `doctor.source_root` 直接读取源码快照 `agent/context/kernel-authoring.zh-CN.md` 作为起点，再按任务边界查 owner；Pro 原型阶段仍按上文资料流程，不把 DSL 包当作 Pro API 契约。写 DSL 与 `scriptor/task.py`，按 `init.delivery_sync_mode` 导出独立 PyPTO-Pro wrapper（默认 auto_mutex），完整 shape 上板。再把同步模式和 worker 原始证据传给 `pypto-pro-scriptor-verifier(stage=candidate)`；只有真实 PASS 才 `complete_implement(report)`。前段 3 轮精度豁免在这里结束，DSL 候选起按 SPEC 容差逐 case 真机比对。
2. **optimize**：读取 `status`，按有效轮数、基础优化、知识卡片覆盖和停止条件决定继续或收尾，不能仅凭达标或工具已计 10 轮结束。
   - **建立账本**：第一次 `begin_round` 前读取 `$CANNBOT_CONFIG_ROOT/skills/pypto-pro-op-perf-tune/SKILL.md` 的 `knowledge_card` 规则、该 Skill 下 `references/knowledge-cards/index.md` 的 Active 表及全部卡片正文；不启动独立 perf-tune。将账本保存为 `custom/<op>/reports/scriptor-optimization-ledger.json`，在顶层记录 `active_index_sha256`，沿用 optimization-playbook 的 `source_kind` 与终态；逐卡记录 `item_id/source_kind/source_ref/case_scope/applicability/expected_metric/status/evidence/relations`，其中 `source_ref` 含权威文件、原子锚点和内容 SHA-256，实验关联另记 `experiment_id`。
   - **执行卡片**：适用且能力已确认的卡片，必须在普通优化轮中完成改码、正确性、可比性能和机制核对；能力未知先补证。只有机制不成立、能力已证实不支持或同一改动已由其它实验 `experiment_id` 覆盖时，才可关闭，并为每个实验 `experiment_id` 写结论和 `relations`。
   - **执行轮次**：每轮把卡片/假设、当前及最佳候选、baseline/性能参考、`exit_criteria`、历史证据和用户限制传给 worker；worker 使用 `pypto-pro-scriptor-optimize`，verifier 独立复算并封存 candidate，最后 `finish_round`。没有用户 baseline 时按冻结合同建立一次参考；采集失败要修复或报告阻断，不能用缺指标跳过优化。
   - **复用与停止**：优化轮与 accept 共享冻结的 case、设备和计时合同，同一候选的有效证据按哈希复用，不重复采集 baseline/Golden/final。保持 `stop_on_criteria=false`，由 verifier 按可比证据 keep/reject；只有有效轮数、基础项和账本均闭合后，才调用 `finish_optimization`，未达标如实保留。
3. **accept**：从选定 DSL 按冻结的 `delivery_sync_mode` 重新导出，把同一模式、版本和最终原始证据传给 verifier `stage=accept`，验证精度、完整输入域、单 runtime kernel、host 边界、独立运行、基础时延和用户准出条件；`complete_accept(report)` 核对报告及导出模式并生成 `reports/final/final.md`（另有 JSON/CSV）。新任务不执行旧的 `begin_sync_trial`/`complete_sync_trial`；旧状态仍按其原合同恢复。`done` 只表示归档，`target_unmet/target_unknown` 必须保留 FAIL/BLOCKED。

后段命令与报告格式按 `$CANNBOT_CONFIG_ROOT/references/scriptor-handoff-contract.md`。源码/证据改变后重新验证；模型结果不替代真机验收。修改数学合同时先 `rollback_prepare(reason)` 交回 Pro，完成更新及自验证后再次 `init(entry_mode=from_pro)`，保留优化历史；worker 不静默改合同。除已留证的环境/实现能力阻断或用户停止外，主 agent 按 `next_request` 持续推进到 Scriptor 归档，不把原型上板、首版 candidate 或阶段交接当作最终交付。

## Scriptor 轮数与调优要求

最低 10 轮只计已完成、具有新假设或有依据的参数对照、实际 DSL 改动及独立封存结果的实验；有证据的失败或无收益也计入。工具在 `begin_round` 即记账，不能直接用其总数代替有效轮数；空跑、纯重测、无变化重试和环境排障不凑数，也不删除其历史记录。

按源码快照 optimize playbook 核查工作分配、数据复用、tile/容量/布局、流水/依赖及阶段内部开销，结合选中 KB 和全部 P0 瓶颈安排实验。适用的基础优化须实际验证，有收益的改动落实到选定候选；不适用或无收益须有依据，不能用十次单一参数微调代替基础项核查。

完成最低轮数、基础项已处理且 Active 知识卡片逐卡闭合后，选定候选达标才可正常收尾；未达标或明确无性能目标时继续探索，直至可行方向逐项已有实验结论或不适用证据，并经独立 verifier 复核确认暂无可行方案。连续无收益、差距大、暂时想不到方案或缺指标都不能单独证明方案已尽；仍有待验证方向就继续。verifier 在新的 `reports/reviews/` 审阅记录的 `findings` 中汇总有效轮次、基础项结论、知识卡片覆盖、选定候选和实际停止依据，保留原始证据引用，不回写已封存报告。不足最低轮数或卡片闭合要求却确实无法继续时如实交接未完成/阻断，不能空跑凑数或宣称已满足要求。

每轮留下“瓶颈与假设 → 实际 DSL 改动 → 全 P0 精度与可比性能证据 → keep/reject 理由 → 下一步”。只写计划、重跑同一候选不算验证了新优化方向；轮外扫描或修复记到 `reports/diagnostics/`，采用时须在正式轮内复验。设备、case/dtype、计时口径或竞争负载不可比时先修复测量条件，不据污染数据判定算法收益或极限。

失败候选先保留代码与原始日志，由 verifier 按真实 checks 封存并 `finish_round`，再恢复最佳 DSL、重新导出后尝试下一方向；`finish_round` 不自动恢复代码。若导出失败、尚无 checks，留在当前轮先核对源码快照 API/样例并修复用法；一次编译失败不足以判定能力不支持。确有源码能力阻断时保存最小复现和错误位置，如实交接未完成状态，不编造报告或修改 state。

不得因会话/上下文将结束、目标差距大或一次失败擅自关闭优化、缩减轮数/输入域或放宽目标。需要换上下文时交接当前状态、实际产物路径、已用预算和下一步，从 `status` 恢复；仅按用户真实变更调整策略，不能把自身限时写成用户停止。

## 收尾判定

主 agent 读取子任务交回的实际产物与报告，核对 `.scriptor/state.json` 的 `current_stage=done`、`reports/final/final.md`、有效轮数、基础项审阅和知识卡片账本。未到 done 是 Scriptor 流程未完成；已归档但目标 FAIL/UNKNOWN 则如实报告未达标/未知，并列出未完成的优化项。`done` 只表示本次流程归档，不自动触发另一套性能 Skill。

以上一阶段选定的候选、冻结合同和已有证据交接；代码仍只改 `scriptor/` 中的 DSL，再 export，不能手改生成文件或缩减接口、输入域、精度要求。每版重新验证精度与可比性能，在 Scriptor 的优化账本、轮次报告和 verifier review 中记录最终候选及新证据；`reports/final/` 保留 Scriptor 阶段的历史结论，不回写旧报告或轮次。

最终交付和用户要求的评分须对应最终选定 DSL 导出的同一版本，核对公开接口、全部 case 的实际 shape/dtype、候选身份及报告路径；旧原型、局部 cell、case 名称或不同版本的最佳指标不能替代完整候选。已有同一候选的有效证据先核对复用，最终返回 DSL/导出代码、对应报告、实际验证范围和未达目标。

## 最终交付区

`custom/<op>/` 是开发、状态和原始证据区；最终可运行包单独放在 `delivery/<op>/`。读取源码快照
`doctor.source_root/agent/zh-CN/references/pypto-pro.md` 的「交付区」，以其
`kernels/`、`golden_cpu.py`、`wrapper.py`、`test.py`、`DESIGN.md`、`REPORT.md`
作为面向用户的最终结构。冻结 SPEC/Golden、`scriptor/`、`generated/`、
`.scriptor/`、`reports/` 留在 `custom/<op>/`，不复制进交付包；
`reports/final/` 保留 Scriptor 阶段的历史验收。Pro 交接报告固定放在
`custom/<op>/reports/pro-bootstrap.md`，不散落到项目级 `reports/`。

完成 Scriptor 验收并闭合优化来源账本后，才从最终选定 DSL 重新 export 并形成
精简交付包：每个 case 映射到包内字节与 `generated/<case>/<manifest.entry>` 一致的
内核源码，优先平铺为 `kernels/<variant>.py`；重复源码可共享，所有包内 kernel 源码
至少被一个 case 命中。`wrapper.py` 真正路由包内 `kernels/`，`golden_cpu.py` 独立，
`test.py` 是唯一用户入口，必须通过公开 wrapper 逐 case 上板并与 CPU reference 比较，
用 `--output` 写 `test_results.json`：`status=PASS`、`case_count`、逐 case
`name/status/kernel`。测试辅助模块与精度比较器若被 import，必须随包提供；不能
引用工作目录、`.opencode` 或另一算子目录。`CASES` 的 shape/dtype/标量参数
必须与原任务及最终导出一一对应；多 dtype 共用这一交付
入口和验收范围。逐 case 的 `input_dtypes`/`output_dtypes` 与 `input_special_values`
要和原任务一一对应；交付检查会观察实际传入包内 wrapper 的特殊值，单列 case 名称
或另跑 manual 诊断不能替代 auto_mutex 交付测试。
跨 rank 的公开 tensor 使用顶层 `shape:["..."]` 并在每个 P0 case 写明具体 shape；
不能靠无状态的旁路算子目录冒充一个已验收产物。

Scriptor 独立 verifier 在工作目录验证 DSL `OpExec` 与内部导出 wrapper；最终
`delivery/<op>/test.py` 测试用户实际收到的公开 wrapper，不依赖开发工具。
在同一台目标机器上验证最终包并保留每 case × launcher、实际选定同步模式、
manifest 与真实评分证据；不支持的阶段写清原因和替代验证。`DESIGN.md` 同步最终
生效结构；`REPORT.md` 只把实际交付版本写为最终，列出完整源码/导出 SHA-256、
失败或回退的候选、剩余问题与证据路径。先定稿报告，再清理交付包内的 `.tmp/`、
`.DS_Store` 和运行输出；工作目录的 `.scriptor/`、`reports/` 与原始证据保留。
最后在目标机执行
`scriptor.py --config-root <实际配置根> delivery-check --op-dir custom/<op> --delivery-dir delivery/<op>`，
默认门禁先核对包结构、导出身份与 SPEC P0 case，再把**仅交付包**复制到隔离目录运行
`test.py --output test_results.json`，核对退出码、20 例等逐案结果、公开 wrapper 的
实际调用记录、命中内核与运行前后文件哈希；
CLI 会把 `ASCEND_WORK_PATH` 与临时目录设在隔离包之外，并在检查后清理 JIT 产物；
交付 `test.py` 不必自行创建持久的编译缓存目录。
新任务还要求 `.scriptor/state.json` 已 `done`，最终导出模式与初始化冻结的交付模式一致；
原始 stdout/stderr/JSON 存在 `custom/<op>/reports/delivery-check/`。
只有返回 `status=PASS` 才是本轮交付通过；`--structure-only` 仅供诊断，返回
`STRUCTURE_PASS`，不代表真机验收。检查失败或内容与最终候选不一致时，
不宣称完成交付，即使 Scriptor 状态已 `done`。
