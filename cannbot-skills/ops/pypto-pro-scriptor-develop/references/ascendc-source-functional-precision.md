# AscendC 源码用例到 A5 PyPTO-Pro：功能与精度闭环

本参考用于一种源代码驱动的迁移：上游只有官方 AscendC（通常来自 A2/A3）、op host/tiling、infershape 和至少一个可运行用例，尚未形成完整的 PyPTO-Pro `SPEC`/`DESIGN` 合同。目标是把**这个明确用例**无 host 中途接管地打通到真实 A5 PyPTO-Pro kernel，并给出可复核的功能和精度结论。

这条路径由 Port playbook 选择，随后调用 `pypto-pro-scriptor-develop` 实现和 `pypto-pro-scriptor-verify` 验收。它不是 `pypto-pro-op-develop` 的入口；后者只适用于已经冻结 `DESIGN`、`DESIGN_BINDINGS` 和 Module 合同的 formal 实现任务。

## 1. 入口判定和输入

只有同时具备以下材料才进入 source-led：

- 官方 AscendC kernel、op host/tiling、infershape 或已编译源实现，带 revision 或文件 SHA-256；
- 一个能实际运行的源实现用例，以及确定性的输入生成或输入文件；
- 输出 shape、dtype、layout、无效区域和源实现的验收规则；
- 可用的 A5 CANN/PyPTO-Pro 环境和 NPU。

调用方显式标记 `workflow_mode=source_ascendc_case`。缺少可执行用例、输入语义或输出语义时返回 `input_contract_gap`，不能按算子名猜公式，也不能伪造 `SPEC`、`DESIGN`、`DESIGN_BINDINGS`、Module 或 `KB_USAGE`。

本路径只用于新建的源用例功能/精度任务，不创建或推进 formal Scriptor 状态。已有 `.scriptor/state.json`、冻结 formal 合同，或用户要求完整 Scriptor/优化流程时，仍从原入口恢复或执行；不能以本路径的 PASS 代替 `complete_accept`、`delivery-check` 或性能准出。用户另有性能目标时，保留为独立 handoff 和未完成项，不能因本路径不测性能而宣称整个任务完成。

## 2. 先冻结源用例合同

写入 `SOURCE_CASE_CONTRACT.json`，至少包含：

- case ID、源文件、revision/hash、kernel/host/tiling/infershape 入口；
- 输入和输出完整 shape、dtype、layout、stride、元素数和 byte footprint；
- flatten/逆映射、batch/row/column 到 work-item 的分配和 GM offset；
- 每个布局变换、转置/反转、矩阵化 scan 或归约方向的显式映射；对上三角/下三角、行/列主序和尾块不能凭名称推断；
- 公式、累加 dtype、cast/round 顺序、归约顺序、padding、mask、无效区域和 store 规则；
- 输入 hash、源输出或 golden hash、独立 CPU reference 命令；
- A5 SOC、CANN/PyPTO-Pro 版本、逻辑/物理 device 映射。

源码、host/tiling、infershape 和测试表有冲突时返回 `source_contract_mismatch`，记录冲突位置，不能静默选择“看起来合理”的一方。

## 3. A5 实现边界

从源实现恢复语义后重新设计 A5 数据流，不逐行翻译 A2/A3 的 UB、block stride、CCE 指令或硬件流水。实现必须满足：

1. 交付文件恰好一个 `@pl.jit` runtime kernel，wrapper 恰好一次 launch，launch 不在 host 循环中。
2. 公式、索引、dtype 转换、padding、mask、无效区填充和 store 全部在 kernel 内完成；host 只做合同检查、合法 shape/stride 推导、分配和一次启动。
3. 不调用 AscendC、CPU/Numpy/Torch 结果，不在 kernel 后用 host 计算修补输出；CPU 只生成独立 golden。
4. 功能测试至少一次用非零或 NaN sentinel 初始化输出，检查全部元素。不能用 `torch.zeros` 的预填值掩盖 kernel 未写出的无效区域。
5. 目标后端必须是 A5 PyPTO-Pro（`compile_kernel(..., backend="pypto_pro")`、`OpExec(..., launcher="pypto")`）。CCE 只能用于诊断，不能代替交付后端。

同步模式沿用 handoff 并写入源用例合同；未指定时使用 `auto_mutex`。`OpExec` 显式传入该 `sync_mode`，导出用支持此参数的 `emit_module(..., sync_mode=...)`，核对生成的 `@pl.jit(auto_mutex=...)` 与运行 manifest。`compile_kernel` 不接收 `sync_mode`，其默认 manual 产物只证明该模式，不能代替 auto_mutex 验收。

## 4. 验证顺序

按顺序执行并保留每一步的命令、退出码、日志和产物 hash：

1. 静态检查 kernel 数量、wrapper launch 次数、host-loop launch、输入输出地址范围和 work-item 覆盖。
2. 编译和最小手算用例，先隔离索引、对角线、边界和无效区；对于任何布局/转置/scan 映射，使用单位脉冲、单调序列或上/下三角基准分别验证方向，并把通过的映射写入 `SOURCE_MAPPING.json`。
3. 目标用例在 A5 NPU 上真实编译、加载、launch、同步和 D2H。
4. 用非零 sentinel 重跑同一用例，检查所有无效区域确由 kernel 写出。
5. 用不导入 ascriptor/PyPTO 的独立 CPU FP32 reference 比较完整输出，记录 max/mean error、fail count、finite、NaN/Inf 和 invalid-region 统计。
6. 若源实现允许，补一个不同边界属性的诊断用例；只有一个源用例时标记 `single_case_only`，不能外推全算子覆盖。

编译失败、设备失败、同步失败和输入合同失败分别归类；环境问题交给环境检查，不改写成算法精度问题。只把稳定复现且修复后消失的现象写入 `confirmed issue`。

## 5. 固定产物和状态

每个 source-led case 至少留下：

```text
SOURCE_CASE_CONTRACT.json
SOURCE_MAPPING.json
FUNCTIONAL_RESULT.json
PRECISION_RESULT.json
RUN_MANIFEST.json
ISSUES.md
```

功能和精度状态分开记录。只有真实 A5 NPU 运行、完整输出比较、sentinel 检查均通过且没有未解释告警，才能写 `functional_pass` 和 `precision_pass`。性能不在本路径内；未测量时写 `performance_not_in_scope`，不能用性能结果替代功能或精度证据。

## 6. 证据边界

具体 case 的输入、输出 hash、误差、日志和问题清单属于该 case 的 `RUN_MANIFEST.json`、`FUNCTIONAL_RESULT.json`、`PRECISION_RESULT.json` 与 `ISSUES.md`，不应固化到本 skill。可以用已验证 case 检查本流程是否可执行，但不得把某个 case 的 shape、阈值、问题或结果外推为通用规则。
