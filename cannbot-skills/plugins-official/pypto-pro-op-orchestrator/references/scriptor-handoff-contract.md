# Scriptor 交接与可执行入口

新任务统一从 `pypto-pro-op-orchestrator` 的 `references/scriptor-mode.md` 进入。Pro 完成需求、Golden 与可上板原型后使用 `init(entry_mode=from_pro)` 直接交接 implement，不再调度独立 prepare/preparer/verifier。

仅当旧 `fresh/optimize_existing` 任务的 status 返回 prepare 时，复用 `pypto-pro-golden-generate`（缺 SPEC 时调用 intent）复核补齐数学输入，再调度现有 verifier `stage=prepare`，用实际报告 `complete_prepare(report)`。保留旧状态与预算，不重开发 Pro 原型或恢复 preparer。

OpenCode 的 `$CANNBOT_CONFIG_ROOT/scriptor-install.json` 仅记录所选 Python、
工作流资源和本仓源码快照的绝对路径；它不是 Ascriptor 包的安装记录。
本文所有命令在目标项目目录执行：

```bash
<python> $CANNBOT_CONFIG_ROOT/scriptor/scripts/scriptor.py --config-root $CANNBOT_CONFIG_ROOT doctor
```

`doctor.source_root` 指向插件内 `resources/ascriptor/`。该目录直接包含
`library/agent/kernels`，不复制到项目中，也不依赖私有 Git 仓库或 wheel。
`sources.json` 和 `sources-index.json` 对整棵交付源码逐文件记录 SHA-256；
source ID 由这些文件和 `product-view.json` 确定。CLI 激活时校验完整快照，
并从快照的 `library/` 导入 Ascriptor。修改源码或更新插件后，先刷新快照索引，
再刷新 OpenCode 工作流注册资源并核对 `doctor` 的源码路径与 ID。

DSL 新上下文从 `doctor.source_root/agent` 运行
`python tools/build_kernel_context.py --print`，直接读取浓缩起点，无需来源哈希校验；
已有相同内容不重读。随后依据操作、精度、布局和生命周期选择 owner 资料。
历史验收不选择当前运行版本，也不能替代新设备验收。

## 任务产物

`SPEC.md`、`<op>_golden.py`、`<op>_golden_cpu.py` 是冻结输入；`scriptor/task.py` 和其本地 DSL 模块是开发源；`generated/` 与 `test_<op>.py` 是导出交付；`reports/runs/` 保存原始执行，`reports/reviews/` 和 `reports/sealed/` 属于 verifier。`.scriptor/state.json` 与 checkpoints 只由状态工具维护，`reports/final/` 由工具根据最终验收生成。
这些是 Scriptor 工作视图，留在 `custom/<op>/`。最终面向用户的独立 `delivery/<op>/` 包须有源码快照中的 Ascriptor
`agent/zh-CN/references/pypto-pro.md#delivery-area` 规定的 `kernels/`、
`golden_cpu.py`、`wrapper.py`、`test.py`、最终 `DESIGN.md` 和 `REPORT.md`。
`test_<op>.py` 是导出器的内部公开 wrapper，`delivery/<op>/test.py` 是完整交付包的唯一执行入口；
两者职责不同。`reports/pro-bootstrap.md`、原始运行和最终评分留在工作区，并由
`delivery/<op>/REPORT.md` 引用。交付包内的 `.tmp/` 仅在报告定稿后清理。

### Pro 交接

前段按模式文件开发和上板，在算子目录的 `reports/pro-bootstrap.md` 记录已用精度轮数、当前源码的逐 P0 运行/精度结果、命令、日志与遗留问题。主 agent 保留 `prototype/` 下的原型及依赖副本，确认后释放根 `test_<op>.py`；这些是实现参考，不能作为数学 oracle。

该记录同时交接源码/环境路径、完整 case 映射（含重复规格的独立 case ID）、精度策略、baseline 来源与计时口径、实际评测命令。并行准备者完成后由主 agent 整合并继续实现和评测，不另设前置 verifier。

初始化请求及默认优化参数统一见 `scriptor-mode.md` 的「交接到 implement」。

首次原型前执行 `bootstrap-check --op-dir custom/<op>`，由 CLI 核对 Plan、Golden 和 Design 产物并封存前段哈希；原型或评分先行属于流程违约。CLI 交接入口是 `state --request-json '<实际请求>'`。`init(entry_mode=from_pro)` 复核 bootstrap 回执和 Golden skill 的 `check_receipt`，冻结数学输入并直接进入 implement；前段运行、轮数和备份由主 agent 按 skill 指导负责，不需要额外 prepare verifier。
新回执还须记录 CPU Golden 的逐 P0 ABI 探针 PASS：使用它自己的纯 CPU `_make_inputs(device)`，
独立核对全部输入、特殊值及输出 shape/dtype 与 SPEC 一致。Golden skill 的普通
FP32 输出合同不适用于 Scriptor；CPU 计算可提升到 FP32，但返回值必须按该 case 的
`output_dtypes` 舍入。探针失败先修正并重新自验证 Golden，不能先写 Pro 原型。

`bootstrap-check` 逐案核对 SPEC P0 与 CPU Golden 的输入、输出、dtype、shape
和特殊值，并将算子开发输入的哈希写入回执。核对失败先修正 SPEC 或 Golden，
再开始原型。

封存后、`init` 前的 Plan/Golden/Design 修订须由主 agent 调用 `bootstrap-restart --op-dir custom/<op> --reason '<原因>'`。CLI 归档旧回执和已有原型/日志，移走活动原型并留下受保护的恢复记录；修订、自验证后重新执行 `bootstrap-check`，再按新回执重跑完整 Pro 原型。新 `reports/pro-bootstrap.md` 须写 `Bootstrap receipt SHA-256: <新回执哈希>`，否则 `init` 不接受。不得通过文件命令删除或移动 `.scriptor` 来重置回执。
若输入已经误改，恢复记录与最终报告要如实说明原来的顺序违约；新回执证明恢复后的顺序，不抹去旧失败。

已有状态先用 `status` 恢复。数学合同变更时先 `rollback_prepare(reason)` 交回 Pro upstream，更新并完成自验证后再次 `init(entry_mode=from_pro)`；只允许这种重入，优化预算和已用轮数保留。后段 candidate/accept 继续使用下文严格封存报告。

### task.py

定义 `make_case(case)`。输入 `case` 是 SPEC machine-contract 中的一项 `p0_cases`，函数每次生成独立输入、输出缓冲及执行配置；使用确定 seed，满足 SPEC 的 dtype、shape、value_range 和语义限制。

```python
def make_case(case):
    # 根据 SPEC 和已安装源码的 typed DSL 构造当前 case。
    return {
        "kernel": typed_kernel,
        "args": [input_tensor, output_tensor],
        "input_indices": {"x": 0},
        "output_indices": {"y": 1},
        "block_dim": 1,
        "bindings": {},
        "output_initialization": {"y": "empty"},
        "workspace_initialization": "empty",
    }
```

`workspace_initialization` 与 `output_initialization` 只能为 `"empty"`；省略时也按
`"empty"`。生成的公开 wrapper 只用 `torch.empty` 分配，不能调用 `torch.zeros`。
kernel 必须在每次调用中先写后读所有使用的输出和 workspace 区域，包括 padding
与空分区。需要初始零值的算法须在 kernel 中实现并验收；仅把清零搬到全局缓存
不能证明首次调用和并发调用符合合同。

该片段描述返回结构，不是可直接运行的算子。`args` 按 typed kernel 的原始签名顺序，返回的 kernel 输出顺序与 SPEC 公开输出顺序一致。`bindings` 是需要发射时特化的显式 scalar 绑定；静态形状无此需求时为空。需要其他初始化 ABI 的任务应提供明确扩展并验收。输入/输出映射名称必须与 SPEC 一致。

reference 只来自 `<op>_golden_cpu`；task.py 不生成 expected。副作用/alias 需在语义审阅中核对，完整输出覆盖应经 poison/初始化检查。

### TensorList public IO

Declare `is_list: true` on the SPEC input. Each P0 `input_shapes[name]` is an
ordered array of concrete member shapes, for example `[[8, 64], [16, 64]]`.
`task.make_case` supplies one non-empty list/tuple at the original `GMList`
parameter position. Members have the declared dtype, one common positive rank,
contiguous storage and one device. Their shapes must also satisfy the typed DSL.
The ordinary finite `value_range` applies to each member's finite elements.
`input_special_values[name]` describes the union of specials observed across all
members. TensorList outputs use `is_list: true` and nested `output_shapes[name]`;
one list is one public output, including an arity-one list. Empty lists remain
unsupported. Output allocation, serialization, Golden normalization and precision
comparison retain each member's identity and order.

Non-finite scalar attrs use JSON strings `"inf"` / `"+inf"` / `"-inf"` / `"nan"`.
The adapter decodes them for Golden calls and matches the corresponding runtime
float values (including NaN). Tasks pass these values through an explicit floating
kernel parameter to VF calls; the standalone emitter materializes their specialized
kernel expressions as typed runtime scalars rather than undefined bare names.
JSON evidence stays strict and uses string encodings for these constants.

Export specializes each declared case to its exact member shapes and arity.
Every member becomes an independent tensor argument; the wrapper reads metadata,
allocates outputs and launches once. It does not loop over kernel launches or
repack tensor data. Public names may differ from DSL names. `input_members` in
`export.json` is keyed by public names; the emitter's list map uses IR names.
The generated launcher embeds the shared metadata validator and imports neither
Scriptor nor Ascriptor. Test-data value checks stay outside the public wrapper.

Delivery `test.py` CASES must copy the export's `input_is_list` / `output_is_list`
maps and per-case `input_members` / `output_members` alongside shapes/dtypes/params.
The delivery checker compares these fields against the selected export. A finite
case export does not claim arbitrary runtime list-length coverage.

### 开发命令

```bash
<python> $CANNBOT_CONFIG_ROOT/scriptor/scripts/scriptor.py --config-root $CANNBOT_CONFIG_ROOT export --op-dir custom/<op>
<python> $CANNBOT_CONFIG_ROOT/scriptor/scripts/scriptor.py --config-root $CANNBOT_CONFIG_ROOT check --op-dir custom/<op> --stage candidate
```

自动导出具体 P0 的 PyPTO-Pro 源码、manifest、独立 wrapper。它只声明这些 case 的支持；广泛动态域不能靠有限 case 分派冒充通过。现有非工具生成的 wrapper 不被静默覆盖。

新 `from_pro` 任务的 `export` 默认 `--sync-mode auto_mutex`，生成源码必须是
`@pl.jit(auto_mutex=True)`；模式写入导出索引并与 backend manifest 核对。
仅当用户明确要求 manual，`init` 才设置
`"delivery_sync_mode":"manual","manual_requested_by_user":true`，后续导出显式加
`--sync-mode manual`。无此状态时手动指定 manual 会被拒绝。旧任务保留原有状态和证据。
`check` 输出原始检查 JSON 的相对路径，`execution_policy="hardware_first"`。默认对完整声明 case
直接执行交付 wrapper 的真实 NPU 路径，再与独立 reference 比较、读取 profile；不先执行 DSL
functional 模拟或 pipesim。两项模型检查保留 NOT_RUN 与原因，不阻塞已经通过的硬件验收。
新策略仍要求每个 case 的 hardware、standalone 和 latency 全部通过；缺卡/编译失败不触发大 shape 模拟回退。
历史无 execution_policy 的报告保留原有必需检查，封存不能改写策略来绕过失败。

新任务用 SPEC 的 `tolerance={"policy":"pro_scheme_a"}` 选择已安装 Pro `precision_compare.py` 的方案 A。`check` 记录比较器路径/哈希和逐输出结果，并核对 SPEC 输出集合、shape/dtype；数值判定采用该引擎。历史 `{atol,rtol}` 合同仍按逐元素容差检查；切换须先 `rollback_prepare` 更新合同、完成 Golden 自验证再交接，历史报告不重判。

精度/性能问题需要模型诊断时，在 `reports/diagnostics/` 保存独立小 probe、命令及日志，保留
原始 case/shape/核数和缩小依据。使用最少合法核数及足够的 slot 周转，保留相关 CV 参与者与
核间关系；重新构造 kernel/launch，不盲目修改启动核数。修复后完整 shape 上板复验，诊断
结果不替代完整域精度或性能结论。默认不自动导入小模型指标作为原 case 的准出值。

## 验证报告

先用 `snapshot --op-dir custom/<op>` 取得 `artifact_hash`，写 `reports/reviews/<stage>.json`：

```json
{
  "artifact_hash": "<snapshot 返回的实际哈希>",
  "checks": {
    "formula": "PASS",
    "reference_independent": "PASS",
    "domain_coverage": "PASS",
    "single_runtime_kernel": "PASS",
    "host_boundary": "PASS",
    "dsl_export_correspondence": "PASS"
  },
  "findings": ["具体文件/符号、实际检查与结论"],
  "external_metrics_sha256": null
}
```

新模式的 candidate/accept 要求全部项；失败项写 FAIL，不能照抄示例。源文件变化后重做审阅。历史 prepare 报告仅要求前三项。

优化轮使用 `stage=candidate` 的 review，且必须在 `findings` 中引用 `custom/<op>/reports/scriptor-optimization-ledger.json` 及其 SHA-256；账本变化后重新审阅并封存。

```bash
<python> $CANNBOT_CONFIG_ROOT/scriptor/scripts/scriptor.py --config-root $CANNBOT_CONFIG_ROOT check --op-dir custom/<op> --stage candidate
<python> $CANNBOT_CONFIG_ROOT/scriptor/scripts/scriptor.py --config-root $CANNBOT_CONFIG_ROOT seal --op-dir custom/<op> --checks reports/runs/<本次运行>/checks.json --review reports/reviews/candidate.json --output reports/sealed/candidate.json
```

`candidate/accept` 用相应 stage；优化候选可传 `--recommendation keep`。封存会检查源码身份和证据哈希；最终 stage 会评估准出条件。状态推进只接收封存报告。

## 状态与预算

OpenCode 主 agent 使用 `scriptor_transition(request=<JSON字符串>)`。相同实现的 CLI 可用于其他宿主的复现或验证：

```bash
<python> $CANNBOT_CONFIG_ROOT/scriptor/scripts/scriptor.py --config-root $CANNBOT_CONFIG_ROOT state --request-json '{"action":"status","opDir":"custom/example"}'
```

请求是 JSON 对象，也可通过 `--request <文件路径>` 读取已有 JSON 文件，无需专门创建请求文件。`init` / `restart_optimization` 显式传入 `scriptor-mode.md` 已解析的策略，避免继承工具默认的 10 轮上限与达标早停。工具只识别 `enabled/max_iterations/stop_on_criteria/time_budget_s/max_no_improvement_rounds`；最低有效轮数、基础优化覆盖和方案穷尽由主 agent 与独立 verifier 按入口证据核查，不伪造工具字段。恢复已有任务不重置策略或轮数。

新任务动作序列：Pro 原型 → init(entry_mode=from_pro，默认 delivery_sync_mode=auto_mutex) → complete_implement(report) → finish_optimization → complete_accept(report)。启用优化时，在 finish_optimization 前按 begin_round → worker/verifier → finish_round(report) 循环，并在循环中闭合知识卡片覆盖账本。state/status 可恢复当前阶段。已有 DSL 先保留并核对当前合同，不重新生成数学参考；已完成任务收到新的 Scriptor 优化请求时，通过 restart_optimization 重新验证候选后优化。旧状态中的同步试验动作仍按旧合同恢复。

全局排名的并列窗口随最高实测分数变化时，较早的已测候选可能重新成为首选。
在两轮之间，恢复该候选的完整产物及其绑定的 observations 后，可调用
`select_best(report, reason)` 选择它。该动作只接受历史已结算且未修改的 PASS 报告，
重新验证源码、原始证据和同步模式；它不重写历史 keep/reject、不重复计入轮数，也不
重新采集同一执行证据。`reason` 应记录冻结排名规则、最高分和实际选择依据。

默认状态响应包含当前阶段、轮数、候选/最终目标准出来源和 `next_request` / `next_command`。工具按已保存策略决定 finish 或 begin，不能自行识别最低有效轮数与方案穷尽；活动轮次需先返回 candidate 验证。带 report 的请求给出常规路径示例，主编排用 verifier 实际返回的路径替换，不能把提示当成验证已经运行。

本模式约定的收尾条件经独立 verifier 复核后，主 agent 用 `finish_optimization(reason=user_stop)` 表达按预定条件停止；该兼容值不表示用户临时叫停。真实触发条件、有效轮次与证据必须见本次 review 的 `findings`，不能仅凭 `user_stop` 判定流程充分完成。不为收尾缩小 `max_iterations`、打开达标早停或捏造用户变更；工具本身不强制这些入口要求。用户实际叫停也用此动作，但如实记录未完成项。核对选定 `best_report` 的目标结果，不用可能来自被拒候选的 `last_criteria` 代替；最终结果仍由 accept 验证实际恢复的版本。

CLI 仅在摘要缺少所需字段时加 `--full`（兼容 `--full-state`）；OpenCode 请求可加 `"detail":"full"`。status 只读，不生成报告或改写历史。无需手填轮次 ID、报告摘要或哈希；Scriptor 要求的 Todo list 只跟踪阶段、证据和下一步，不代替 state、verifier report 或本合同。

`begin_round` 仍立即计入一轮，失败/无收益不退款。工具在 `finish_round` 内记录报告摘要和原始检查身份：重复提交已结算报告不再结算下一轮；对同一次检查改名/重新封存也不能冒充新执行。verifier 可以复用输出文件名，新执行由原始检查身份区分；工具在 `.scriptor/receipts/` 保存只读快照，`best_report` 指向快照，不要求 agent 手动改名。v1 历史缺少完成时摘要时，不补造历史完整性证明；无法区分同一候选的旧执行时明确报错。

每个请求都必须包含 `opDir`，后续阶段不会从 cwd 或上一次调用猜测。例如：

```json
{"action":"complete_implement","opDir":"custom/example","report":"reports/sealed/candidate.json"}
```

进行中的 prompt 预算调整用 `update_optimization`，传 optimization 覆盖值和用户变更 reason，已消耗轮数不清零。已结束任务的新 Scriptor 优化请求用 `restart_optimization`，显式传入本次已解析策略，保存前次历史，并重新核验已有 SPEC/Golden/候选。普通断点恢复用 status，不初始化或重置计数。

用户明确的预算耗尽或允许的早停可 finish，但不据此宣称满足默认最低轮数、基础优化和知识卡片覆盖要求。关闭达标早停后，verifier 的 keep/reject 决定候选取舍。新模式的 `rollback_prepare(reason)` 交回 Pro upstream，保留历史轮次，不调用已删除的 preparer。环境阻断用 `blocked(reason)` 记录真实原因，环境恢复后继续当前阶段。

项目升级源码后，用 `migrate_sources` 显式迁移已有状态：请求必须给出
`from_source_id`、`to_source_id` 和用户升级原因 `reason`，且不能有进行中的优化轮次。
工具核对来源状态和当前安装的目标身份，保存迁移前完整状态；新模式已在 upstream 时保持该阶段，完成 Pro 更新及自验证后再 init(from_pro)；其余新模式任务重验冻结输入和 Golden 回执后回 implement，历史四阶段仍退回 prepare。
优化预算和已消耗轮次保留。SPEC、Golden、DSL、生成文件和旧报告不改写，旧报告仍绑定旧源码身份，
不能作为新版本的通过证据。迁移不会启动新的优化或硬件任务。

### 固定交付同步模式

新任务在 `init(entry_mode=from_pro)` 冻结 `delivery_sync_mode`，默认 auto_mutex。
从首个 DSL 候选、每轮优化到最终 accept，导出和独立 verifier 报告都必须采用同一模式；
worker 直接 `OpExec(..., launcher="pypto")` 的 smoke、完整 P0 与诊断也须显式使用
冻结模式，工具会检查工作区中保留的 PyPTO-Pro manifest 和生成源码，拒绝模式混用；
`complete_implement`、`finish_round`、`complete_accept` 均核对模式，accept 还核对实际
`generated/export.json` 和 backend manifest。新任务不执行 `begin_sync_trial`、
`complete_sync_trial` 或 `abort_sync_trial`，不生成 manual 对照；旧状态仍按旧流程恢复。
auto_mutex 发射失败、真机精度失败或目标阻断时保存原始证据并如实报告，不能回退 manual
冒充完成。用户明确要求 manual 时才按初始化记录的授权仅生成 manual；变更既有任务同步
模式属于新的用户请求，不能静默覆写当前已冻结状态。

### 最终收尾与结果

`complete_accept` 核对当前最佳候选、冻结输入、全部必需检查、独立语义审阅、原始运行与指标证据。用户目标和实现验证分开记录：

| 必需验证 | 用户准出 | outcome | overall verdict |
|---|---|---|---|
| PASS | PASS / NOT_REQUESTED | accepted | PASS |
| PASS | FAIL | target_unmet | FAIL |
| PASS | UNKNOWN | target_unknown | BLOCKED |
| 未通过 | 任意 | 拒绝完成验收 | 保留实际验证结果 |

FAIL/UNKNOWN 的收尾还要求存在与已保存策略一致的优化结束记录；`user_stop` 须按上文区分约定条件触发与用户实际叫停，并保留真实依据。`current_stage=done` 表示本次流程归档；只有 outcome/criteria 能说明目标结果，不能仅用 done 判定达标。封存的 overall verdict 不改写。

收尾自动生成 `reports/final/final.md`、`final.json`、`final.csv`。默认只读 Markdown，它包括最终 case 时延、用户目标值/实测值、原有 all/any/聚合范围、轮数与停止原因、代码/证据入口；JSON 保留完整字段，CSV 供用户导出。数据仅来自同一个最终验收，不混入历史候选的逐 case 最佳指标，也不额外计算 Benchmark 分数。

报告记录状态历史可证明的本次 workflow wall time；无法从宿主获得的 author/阶段时间保留未知，不拿总时间代替。派生视图生成失败会明确返回 `results.status=unavailable`，不改变已写入的验收事实；修复原因后可运行：

```bash
<python> $CANNBOT_CONFIG_ROOT/scriptor/scripts/scriptor.py --config-root $CANNBOT_CONFIG_ROOT report --op-dir custom/<op>
```

该命令重新核对当前完成状态及证据，只重建派生文件。新完成状态记录验收报告摘要；旧 v1 完成状态保持原字节，重新验证可得证据后生成视图，并标注历史摘要/逐 case 详情的实际可用范围。重新准备或开启新优化时清除当前收尾结论，原有历史和 checkpoints 保留。

## Scriptor 优化阶段的知识卡片与性能证据

Scriptor 的 `optimize` 在第一轮前读取 `$CANNBOT_CONFIG_ROOT/skills/pypto-pro-op-perf-tune/SKILL.md` 的
`knowledge_card` 规则、该 Skill 下 `references/knowledge-cards/index.md` 的 Active 表及全部卡片正文，建立逐 `item_id` 的覆盖账本，固定保存为 `custom/<op>/reports/scriptor-optimization-ledger.json` 并在顶层记录 `active_index_sha256`，其 `source_ref` 含权威文件、原子锚点和内容 SHA-256，并把每个适用且能力可用的
卡片纳入普通 `begin_round → worker → verifier → finish_round` 闭环。能力未知先补证；只有机制不成立、能力已证实不支持
或同一改动已经由其它 `experiment_id` 完成实验时才可以关闭该 item，而且每个 `experiment_id` 都要有自己的结论与 `relations`。

这一步复用 Scriptor 已冻结的 case、设备、计时和候选证据。accept 仍对最终选定候选执行完整验收；已有同候选有效证据可复用，但不得为同一候选重复采集 baseline、Golden 或 final formal compare，也不得在 accept 之后
强制再启动完整 `pypto-pro-op-perf-tune`。用户另行明确要求独立
性能调优时，才把它作为新的、独立的请求处理；`reports/final/` 仍只表示 Scriptor 验收事实。

末尾候选取舍与用户评分完成后，在项目根执行：

```bash
<python> <实际安装的 .opencode/scriptor/scripts/scriptor.py> --config-root <实际 .opencode> delivery-check --op-dir custom/<op> --delivery-dir delivery/<op>
```

该命令核对精简包必需文件、最终导出的 source/hash、`test.py` 的字面量
`CASES` 与 `kernels/` 的覆盖映射、`REPORT.md` 的完整 SHA-256，拒绝工作区状态、
`.tmp/`、`.DS_Store`、符号链接和未命中的内核。随后在隔离副本运行
`test.py --output test_results.json`，核对逐案 PASS、公开 wrapper 实际调用、
选中 kernel 与文件字节未变；
新任务先核对 Scriptor 已完成且导出模式等于初始化记录的默认 auto_mutex 或用户明确要求的 manual；
原始日志与机器结果留在 `custom/<op>/reports/delivery-check/`。
只做静态诊断可加 `--structure-only`，其结果为 `STRUCTURE_PASS`，不能作为最终 PASS。
`REPORT.md` 的 `Final export SHA-256:` 和逐源码 `Final source SHA-256 (scriptor/文件.py):`
字段必须唯一且与当前导出相等；在历史实验段顺带提到当前哈希不算最终身份声明。
隔离测试由 CLI 将 PyPTO JIT 工作目录放在交付包外并在结束后清理，不要求
`delivery/<op>/test.py` 自行创建遗留的编译缓存。
多个 dtype 须逐 case 写入 SPEC 并核对导出；不能用另一个未验收算子目录
补成表面通过。

## 准出条件与指标

SPEC 可选字段 `exit_criteria` 为 null、单条件或 `{"all":[...]}` / `{"any":[...]}`。非空树不能包含空组；没有条件不产生达标结论。

```json
{
  "id": "vector_busy",
  "metric": "vector_pipe_utilization_pct",
  "operator": ">=",
  "threshold": 70,
  "unit": "%",
  "basis": "model",
  "cases": ["p0"],
  "aggregation": "each",
  "source": "pipesim.mean_active_vector_utilisation"
}
```

70 仅用于展示格式，不是默认目标。operator 支持 `< <= > >= == between`，between 的 threshold 为 `[lower,upper]`。aggregation 为 each/min/max/mean，basis 为 hardware/model；cases 必须属于 SPEC。多要求默认 all，按用户明确的逻辑保留 any/嵌套组合。

指标口径（基础时延/利用率由 `check` 采集；性能 Skill 的正式结果由 `observe` 导入）：

| metric / basis | source | 口径 |
|---|---|---|
| latency_us / hardware | msprof.Task Duration(us).min_after_warmup | 同一任务 32 次执行丢弃前 5 次，保留最小时延和中位数原始记录 |
| case_speedup / hardware | 冻结 formal baseline/final compare | 同协议 baseline duration / candidate duration，单位 x |
| golden_reference_ratio / hardware | 冻结 formal Golden compare | NPU Golden 单次 E2E / PyPTO target-kernel，单位 x；仅 `valid_for_target_met=true` 时用于准出 |
| vector_pipe_utilization_pct / hardware | msprof.aiv_vec_ratio.mean_after_warmup | 列存在时记录对应比率均值乘 100 |
| vector_pipe_utilization_pct / model（显式 observe） | 由模型报告声明 | 默认不采集；必须绑定所声明 case 的真实模型工作量，缩小诊断不能冒充完整 case |

每个指标绑定当前候选、case、单位、依据与源报告哈希。不混用模型 cycles、host wall time 和真机时延；缺字段为 UNKNOWN，不换一个近似指标放行。

启用优化或有性能目标时，按以下任务合同复用现有评测脚手架：

- **目标**：对当前候选的全部 P0 生成可复算性能证据。可比的用户 baseline 用于 `case_speedup`；否则以 NPU Golden 为性能参考，并用首个正确 PyPTO 候选作为优化 baseline。无用户数值目标时默认要求 `golden_reference_ratio ≥1.0`，采集失败则保留 UNKNOWN。
- **输入与路径**：输入为冻结 SPEC/Golden、已通过 `check` 的公开 wrapper、checks.json 及已有 baseline/候选证据。优化轮与 accept 共用冻结 manifest、精确 Op Name、设备、warm-up、repeats 和单次 target launch；全部 P0 的 baseline/final formal compare 和 NPU Golden 参考均复用 `pypto-pro-op-perf-tune` 的 evidence protocol/逐 case 采集器，Golden 只建立一次，同一候选按 artifact hash 复用，不另建计时协议。
- **产物**：复用 Scriptor 优化轮和 accept 已产生的性能 evidence、日志、报告和原始归档；指标源绑定当前 `artifact_hash`、本次 `check_report` 及原始 evidence，路径统一相对算子目录，不要求另生成独立 perf-tune 报告。
- **验收**：verifier 按冻结口径独立复算并核对 case、候选、baseline 和 Golden 参考身份，用 `observe` 导入有效指标，将 observations 的 `sha256` 写入 review 后 `seal`。候选、证据或条件变化后重新测量；最终 accept 不复用其它候选的数据，也不把多个候选的指标拼成一个结果。

性能及 Roofline 等外部模型统一使用 `observe`：每项描述含 metric、case、unit、basis、source、evidence_file 和 pointer，源报告绑定当前 `artifact_hash` 并保留模型定义、参数来源及性能 Skill 原始产物。

```bash
<python> $CANNBOT_CONFIG_ROOT/scriptor/scripts/scriptor.py --config-root $CANNBOT_CONFIG_ROOT observe --op-dir custom/<op> --request metric-sources.json
```

`observe` 只导入现有数据，不生成或猜测模型参数。

## 运行环境

使用现有 Python；安装不执行 pip/conda、不升级 Torch/CANN。CPU 模拟需要已安装源码声明的 Torch/NumPy；实际能力以 check 为准。

Scriptor 检查必须在具备 PyPTO-Pro、torch_npu、NPU 和 msprof 的设备机器本机执行。未提交的 `$CANNBOT_CONFIG_ROOT/scriptor.local.json` 可设置 `timeout`，也可用 `board`、`boards_file` 指定**本机**设备条目；若该条目没有 `local: true`，检查立即拒绝。`ASCRIPTOR_BOARD`、`ASCRIPTOR_BOARDS` 与 check 参数同样只用于选择本机条目，不触发 SSH、传输或远端解包。主编排者负责选择并检查健康设备；连接信息不能进入归档和公共报告。

最终 driver 只复制交付源码与输入到隔离目录，阻止 ascriptor 导入，实际调用公开 wrapper；独立 CPU Golden 的比较在控制端完成。

### Explicit quantized integer tolerance

A source task that permits quantized integer differences can declare
`{"atol": 0.001, "rtol": 0.001, "integer_atol": 1}`. The optional non-negative
integer budget applies only to non-Boolean integer outputs up to 32 bits; subtraction
is widened before comparison. Omission preserves exact equality. Boolean and 64-bit
integer outputs remain exact. Floating outputs use only atol/rtol. A task with
additional per-dtype MERE/MARE rules must retain and enforce those original rules
as mandatory final accuracy metrics; this field does not replace them.
