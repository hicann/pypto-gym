---
name: pypto-pro-op-perf-tune
description: 优化并验收已通过 Stage 4 的 PyPTO-Pro 单 kernel 实现；冻结规格与测量协议，先闭合已选 KB 缺口、通用优化手段、符合条件的性能知识卡片和模板优化项，再根据当前瓶颈自主提出并验证新的优化项，最终闭合全部实验与证据。也可按既有 msprof 文档独立执行性能采集、比较和瓶颈分析。
---

# PyPTO-Pro Stage 5 性能优化

本 Skill 只定义 Stage 5 的优化来源、执行顺序和交付边界。具体账本、实验、重开、最佳版本选择
与停止规则统一见[优化项实验闭环](references/optimization-playbook.md)。性能采集和分析按需读取：

- 标准、compare、quick、batch 采集与主 bound 判定：[msprof 指南](references/msprof-guide.md)；
- 环境支持时的 `msprof op` 补充诊断：[msprof op 指南](references/msprof-op-guide.md)；
- CSV 字段与阈值边界：[CSV 字段参考](references/csv_fields_reference.md)；
- Stage 5 manifest、Golden、归档和时间线合同：[证据协议](references/evidence-protocol.md)。

用户只要求采集、比较或诊断时，进入对应参考页即可，不启动 Stage 5 状态流程。Stage 5 由
`pypto-pro-op-optimizer` 执行、`stage5-check` 独立验收；本 Skill 不读写编排状态。

## 1. 目标与边界

保持外部功能合同和 Stage 4 铁律不变，在 Stage 5 内找全、落实并关闭全部优化项；冻结 baseline
与本轮正式评价的候选只在正确、合规时纳入排名，留下冻结聚合指标最优的版本与可复算性能证据。用户数值目标
或默认 Golden 参考是理想目标，用于指引优化和披露差距，不是 Stage 5 能否交付的硬门禁。

- 进入 Stage 5 后不返回 Stage 1、2、3 或 4。代码、知识落实、设计一致性和证据问题都在本阶段
  修复并重新验收。
- 环境阻断、用户中止或冻结输入之间存在无法合法消解的矛盾时，报告阻断；机器 Stage 等待时保持
  `in_progress`，确定终止时由编排器调用 `fail_stage(5)`。
- 只修改可维护的 Python DSL 和必要的最终事实记录。`build/**`、`kernel.cpp`、trace 与指令生成物
  只读，只能作为证据。

## 2. 开始前读取与冻结

必须读取并交叉核对：

- `SPEC.md`、Golden、`EXPLORE_REPORT.md`、`PRO_MATERIAL_INDEX.md`、`MEMORY.md`；
- `DESIGN.md`、`DESIGN_BINDINGS.json`、`module_interfaces.yaml`；
- 最终 `test_{op}.py` 与全部 Stage 4 P0 测试；
- `$CANNBOT_CONFIG_ROOT/references/performance-constraints.md` 中 Stage 5 仍须遵守的性能强制；
- 每个 class 的 `KB_SELECTION.json`、其中选中的每个 pattern/constraint 原文及对应
  `KB_USAGE.json`；
- [通用优化手段](references/general-optimization-methods.md)；
- [知识卡片索引](references/knowledge-cards/INDEX.md)及其中全部 active/eligible 卡片正文；
- [模板优化项索引](templates/INDEX.md)及其中全部 active/eligible 模板。

只有本节明确允许更新的最终源码、性能交付物和 owner-schema 事实记录可在 Stage 5 补齐或纠正；
其它必需输入只读。只读或冻结输入缺失、损坏、无法解析且无可恢复原始字节时，不猜测、不重建，
报告 `stage5_contract_blocked`。

始终冻结：

- SPEC 数学语义、I/O、支持范围、P0、输入分布、精度门与用户目标；
- Golden 数学实现和 Stage 4 测试语义；
- `KB_SELECTION.json` 的选择、引用、理由和内容哈希；
- `module_interfaces.yaml`、staged Module、公开 wrapper 的 callable/签名/I/O/optional 语义与 host
  边界合同，以及单 kernel/单 launch 等 Stage 4 铁律；wrapper body 只能在这些合同内调整；
- `PERFORMANCE_CASES.json` 及其中的目标 case，在任何 Stage 5 采集或改码前冻结；
- 设备、seed、warm-up、repeats、精确 `Op Name` 和计时范围在首次正式采集前冻结；baseline 与
  Golden 合同生成后不再修改。

Stage 5 可以更新最终 `test_{op}.py`、性能交付物，以及为匹配最终接受实现所必需的
`DESIGN.md`、`DESIGN_BINDINGS.json` 和 `KB_USAGE.json`。事实记录不得删除、降级或
改写已选 KB 义务，也不得用文档修改掩盖代码未落实；未接受的实验不能写成最终事实。更新前按需
读取 [Design Skill](../pypto-pro-op-design/SKILL.md)、[Develop Skill](../pypto-pro-op-develop/SKILL.md)
及 [KB 合同](../pypto-pro-op-kb/CONTRACT.md)，只按各自产物 owner 定义的 schema 同步最终事实。

## 3. 优化项来源与顺序

优化项只有以下五类，按顺序处理。第一次修改代码前，按[优化项实验闭环](references/optimization-playbook.md)
建立来源覆盖账本。

### 3.1 `kb_selected`：已选 KB 的未落实要点

逐 class、逐行读取 `KB_SELECTION.json.optional_patterns` 与 `required_constraints` 指向的原文，
先按 Design Skill 的 source-first 合同核对并修正 `DESIGN_BINDINGS.json`。通过 owner schema 自检后
形成本轮原子 requirement 清单；`kb_selected` 分母只包含其中 `kind=obligation` 的原子项，precondition 与
validation scope 作为适用性和验证证据保留。再把每条义务与最终代码、DESIGN/BINDINGS 和 KB_USAGE
核对：已落实、未触发和真实缺口都进入账本；前提触发但代码未落实的缺口必须最终真实落实，不能
因无性能收益而拒绝。

### 3.2 `general_method`：通用优化手段

枚举[通用优化手段](references/general-optimization-methods.md)索引中的全部 active/eligible 原子项。
明确不适用或目标/API 不支持时用证据关闭；其它可能适用项逐个进入受控实验。通用项仍要与本次
selected KB 去重，同一改动可以共享实验，但每个 source id 单独关闭。

### 3.3 `knowledge_card`：知识卡片

只按[知识卡片索引](references/knowledge-cards/INDEX.md)枚举 active/eligible 原子项，不扫描目录猜测。
INDEX 为空时如实记录数量为 0；未来晋升为 active/eligible 的卡片自动进入相同闭合分母。可能
适用项逐个实验，其它项用明确的适用性或能力证据关闭。

### 3.4 `template`：模板优化项

只按[模板优化项索引](templates/INDEX.md)枚举 active/eligible 原子项，不扫描目录或把未登记的
`.tmpl` 文件加入清单。INDEX 定义优化意图、适用门槛和稳定 ID，模板文件只提供实现骨架；退役、
负向或不 eligible 的模板不进入闭合分母。可能适用项逐个实验，其它项用明确的适用性或能力证据
关闭。采用模板前必须按当前代码适配，并重新证明正确性、机制和性能。

### 3.5 `bottleneck_derived`：当前瓶颈自主项

前四类预置来源全部闭合后，才可根据当前接受实现的 profiler 与源码、生成物或 Tile DAG 证据
提出自主项。每项必须指向尚未尝试的具体瓶颈、使用与当前代码哈希匹配的新鲜分析、与前四类和
既有自主项去重，并按同一受控闭环验证。接受改动后必须在新代码上刷新瓶颈证据，再建立下一项。

profiler、代码分析、平台资料和 general knowledge 可以共同解释新鲜瓶颈并形成假设；平台资料和
general knowledge 本身不是优化项来源，假设必须先按前述证据去重并登记为 `bottleneck_derived`，
再修改代码。自主改动使预置项结论失效时，先重开并闭合预置项。

## 4. Stage 5 主流程

1. **确认输入与正确性**：完成上述读取，对未修改的 Stage 4 最终代码重跑完整正确性并预热
   JIT/编译。
2. **冻结执行合同**：开始任何 Stage 5 采集或改码前，从 SPEC P0 与既有测试生成
   `PERFORMANCE_CASES.json`：`cases` 覆盖全部 P0，`optimization_target` 记录本轮排名使用的非空
   目标 case 子集及选择方式。该选择必须由调用方在本 Skill 启动前确定；dispatch 缺失或不合法时，
   optimizer 立即停止并报告调用方，待补齐后再启动，不自行猜测或二次提问。无用户数值目标时，
   按证据协议采集并冻结一次 Golden 理想参考。未生成性能
   Golden 合同时记录 `unavailable`，不能据此否定有效的 PyPTO baseline→candidate 选择证据；已有
   合同矛盾或不可复算时必须修复证据。
3. **建立来源覆盖账本**：先按 §3.1 核对并形成 selected-KB 原子 requirement 清单，再枚举其中全部
   obligation、全部通用方法、active/eligible 知识卡片和模板优化项；账本字段和关闭状态按实验
   闭环填写。
4. **建立正式 baseline**：discovery 确认唯一、完整 lowering `Op Name`，再按 manifest 逐 case、
   逐 repeat 采集七指标 baseline 并归档原始证据。
5. **分析瓶颈并排序**：使用既有 Roofline、Vector/Cube/Scalar/MTE、分层带宽、冲突、逐核及
   Tile DAG/时间线方法判断适用性、优先级和验证指标；ratio 只负责指路。
6. **闭合预置来源**：依次处理 `kb_selected`、`general_method`、`knowledge_card`、`template`。每个
   待实验项执行“改源码 → 完整正确性 → quick → 必要时 formal compare → 机制核对 → 接受或恢复
   → 写回账本”。
7. **进入自主优化**：确认四类预置来源全部闭合，并刷新当前最佳实现的瓶颈证据；证据不足先补证，
   再根据当前瓶颈逐个建立 `bottleneck_derived`，接受改动后重新分析。
8. **闭合与停止**：关闭全部来源和已创建项，并在最终代码上执行 fresh candidate sweep。性能目标
   差距和残留瓶颈用于提出、排序候选；仍有合法未尝试项就继续，确认没有新候选时记录候选闭合与
   最佳版本选择证据。达到或未达到理想目标本身都不能提前结束或阻止交付。
9. **同步最终事实**：按上述 owner 合同，只依据最终接受代码更新 DESIGN/BINDINGS/KB_USAGE，
   逐项核对引用、file/symbol、验证方法与代码一致。
10. **最终验收**：对同一最终实现连续执行完整正确性、独立 final formal compare 和每个 P0 的
    补充 timeline；若终验使预置项结论失效，返回第 6 步重开，若发现新的自主项则返回第 7 步。
    确认无新项后再交 `stage5-check`；中途改代码就重做最终验收。

## 5. 交付物

| 交付 | 要求 |
|---|---|
| `test_{op}.py` | 正确、合规的冻结 baseline 与本轮正式评价候选中，按冻结聚合指标选出的最佳实现 |
| `PERFORMANCE_CASES.json` | SPEC P0 与既有测试的一一映射，以及冻结的目标 case；baseline/final 共用 |
| `PERFORMANCE_REPORT.md` | 冻结合同、目标 case、来源账本、逐 case baseline/final/目标、实验、候选比较、终态和证据路径 |
| `performance.json` / `performance.log` / `perf_report.md` | final formal compare 产物；quick 不能替代 |
| `docs/perf/round_NNN/` | baseline/final 原始 CSV、measurement/collection 与逐 case timeline |
| 最终事实记录 | 与接受代码一致的 DESIGN、DESIGN_BINDINGS 和 KB_USAGE |

verifier FAIL 后保持 Stage 5 `in_progress`，把原始失败证据交回 optimizer 定点修复并重新执行受影响
验证；不得建议或执行任何 `rollback_to_stage`。
