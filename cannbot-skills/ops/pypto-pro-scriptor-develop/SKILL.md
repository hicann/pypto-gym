---
name: pypto-pro-scriptor-develop
description: 基于冻结的 SPEC、Golden 与 KB 合同，用安装态 ascriptor DSL 实现或修复单 runtime kernel，导出独立 PyPTO-Pro wrapper 并生成逐 case 证据。
---

# 编写与修复

**目标**：把已确认的设计实现为 ascriptor DSL。数学以 SPEC 和独立 Golden 为准，原型只提供 API、布局和已知问题参考；核心计算在一个 runtime kernel 内完成，host 只承担合同允许的分配、参数和调用。

**输入与路径**：完整 Pro 交接包含 `custom/<op>/` 下四类 Plan 产物（`SPEC.md`、`PRO_MATERIAL_INDEX.md`、`EXPLORE_REPORT.md`、全部 `KB_SELECTION.json`）、CPU/NPU Golden 与 `GOLDEN_VALIDATION.json`、`DESIGN.md`、`DESIGN_BINDINGS.json`、`module_interfaces.yaml`、`prototype/` 和 `reports/pro-bootstrap.md`；兼容任务使用已有冻结合同、候选和可用资料。先审查公式、接口、P0 cases、资料充分性及 KB/Design 绑定。按 handoff 的 task ABI 运行 CLI `doctor`，从返回的 `source_root` 读取 `agent/AGENTS.md`，再在 `source_root/agent` 用已安装 Python 执行 `tools/build_kernel_context.py --print`。该命令直接输出浓缩起点，不依赖来源哈希校验，也可直接读取该 Markdown 文件；中文单 kernel author 的通用路线已经包含其中，只按当前任务继续读 PyPTO-Pro 专题、触发的 owner API 和一个匹配样例。指南缺失时沿同一源码目录的 ROUTER 和 owner 原文继续。

**方法**：在 `scriptor/` 编写 DSL 和 `make_case(case)`，按调用方给定的同步模式 export；先完成公开 wrapper 的 import、compile、launch、sync smoke，再对全部 P0 完整 shape 上板并逐 case 按 SPEC 精度比较。每个直接发射或上板的 PyPTO-Pro 诊断也必须显式传入冻结模式，例如 `OpExec(kernel, launcher="pypto", sync_mode=delivery_sync_mode)`；底层省略参数会生成 manual，不能把这种运行当作 auto_mutex 候选证据。使用固定模式的正式 export/check 作为完整 P0 证据，并核对所有诊断 manifest 与 `@pl.jit(auto_mutex=...)`；错误模式的产物保留并报告，不能删证据后宣称通过。`make_case(case)` 必须生成 SPEC 声明的 `input_special_values`，不得因有限 `value_range` 把原任务 Inf/NaN 换为有限样本。需要性能证据时只返回当前候选可复算的原始记录。

**按症状诊断**：API、导出或编译错误查源码位置、日志和固定源码；数值或性能问题用保留症状的最小 probe 定位，随后回到全部 P0；设备异常携证据交给 `pypto-pro-environment-check`。仅在条件、实现或证据变化且预算允许时重试；模型诊断不替代最终真机验收。

**产物与验收**：返回可复现 DSL、task、导出源码/wrapper、命令、日志及逐 case 证据，供主 agent 最终整理 `custom/<op>/` 的 Ascriptor 交付视图；`generated/` 和 `test_<op>.py` 不代替最终 `kernels/`、`wrapper.py` 和 `test.py`。smoke、全部 P0 真机精度和声明输入域覆盖均通过才完成；SPEC/Golden/KB、单 runtime kernel 与 host 边界保持不变，阻断和 library/backend 缺口按实际范围报告。

## 官方 AscendC 源用例转化补充路径

本节只在 handoff 明确给出 `workflow_mode=source_ascendc_case` 时启用；未给出该标记时，以上 formal/scriptor 工作流及其输入要求保持不变。source-led 转化不要求调用方补造 `SPEC`、`DESIGN`、`DESIGN_BINDINGS`、Module 或 KB 合同，也不改变正式 scriptor 模式的职责。

启用后先读[源用例功能与精度合同](references/ascendc-source-functional-precision.md)，按其中的入口边界、布局探针和证据要求执行。

source-led 输入必须包含官方 AscendC 实现、可用的 host/tiling/infer-shape 信息（若仓库提供）、一个可运行的源用例和 A5 PyPTO-Pro 环境。先把源实现和用例冻结成 `SOURCE_CASE_CONTRACT.json`，记录输入/输出 shape、dtype、layout、轴与边界、累加和 cast 顺序、别名/偏移规则及源用例的数值域；同时生成 `SOURCE_MAPPING.json`，逐项说明源语义到 PyPTO-Pro API 的对应关系。缺失信息要标为 `unknown` 并阻断相应结论，不能凭猜测补齐。

实现时保持一个 `@pl.jit` runtime kernel 和一个公开 wrapper；host 只做合同允许的设备分配、参数校验、launch 和同步，不在 host 侧计算或修补输出。必须先完成导入、编译、加载和 launch smoke，再在真实 A5 NPU 上运行源用例；输出先用非零 sentinel 初始化，D2H 后检查完整输出已被 kernel 覆盖。参考实现独立于 PyPTO（建议 CPU FP32），逐元素比较全部输出并记录容差、最大绝对/相对误差、NaN/Inf 处理和失败位置。

交付至少包括可复现源码与命令、`RUN_MANIFEST.json`、`FUNCTIONAL_RESULT.json`、`PRECISION_RESULT.json` 和问题清单。只有真实编译/加载/同步/D2H、sentinel 检查及完整独立精度比较全部通过，才能将该用例记为 `functional_pass` 和 `precision_pass`；结论必须标记为 `single_case_only`，不得外推为全算子覆盖。性能属于独立流程，本路径只记录功能和精度证据。
