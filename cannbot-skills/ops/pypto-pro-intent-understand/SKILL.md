---
name: pypto-pro-intent-understand
description: 将自然语言、官方 API、论文或用户代码中的 PyPTO-Pro 算子需求整理为可执行的 SPEC.md。用于 Stage 1 的需求澄清、公式与接口冻结、特性优先级和验收规格定义；不要用于 API 可行性探索、tile/Module 设计、golden 或 kernel 实现。
---

# PyPTO-Pro 需求规格化

只负责回答“算什么、接口是什么、怎样算正确”。不要选择 PyPTO-Pro API、计算拓扑、
Module、tile、同步或 kernel 写法；这些分别属于 material-explore 和 Stage 3/4。

## 输入与输出

- 输入：用户陈述，以及用户明确给出的公式、代码、测试、论文或官方 API 引用。
- 输出：`custom/<op>/SPEC.md`，使用 [SPEC 模板](templates/spec-template.md)。
- 若 `SPEC.md` 已存在，不得静默覆盖；先让用户确认覆盖，或在无人值守任务中保留原文件并返回 `blocked`。

按输入形态恢复事实，不改变同一证据标准：标准 API 以目标版本官方定义为准；外部 URL
必须先读取原文，读取失败时只形成低置信度待确认草稿；代码/测试按实际签名和控制流提取；
自定义描述缺公式或最小输入输出合同时直接列出缺失项。不要把模型记忆升级成外部材料证据。

## 事实与决策规则

按以下优先级采用事实：

1. 用户明确陈述及用户提供的测试；
2. 项目已经批准的接口合同；
3. 目标版本官方文档或标准 API；
4. 用户提供的论文、源码或参考实现；
5. 模型知识只可形成待确认草稿，不可冒充已确认事实。

为每项非显然结论记录来源和置信度：

- `✓ 高`：用户事实、已批准合同或目标版本官方资料；
- `⚠ 中`：从论文、源码或用户代码分析得到；
- `❓ 低`：推断，只能等待确认或按无人值守规则披露。

不得静默为公式、输入输出、shape、dtype、optional 参数、动态轴或边界语义填值。

## 工作流

### 1. 提取最小语义合同

识别并整理公开接口的 rank、shape、dtype、动态轴和语义：

- 合法算子名：小写字母开头，只含小写字母、数字和下划线；
- 数学公式或等价的逐步算法；
- 每个公开输入、输出及可选参数的名称、rank、shape、dtype、动态轴和语义；
- 每个输出 shape/dtype 相对输入与参数的推导关系；rank-0 tensor 的 shape 明确写 `[]`；
- machine-contract 的每个输入/输出都写可解析的闭区间 `value_range: [min, max]`；界限必须有语义
  依据且为有限数，供 Stage 3 做强制数值安全分析，未知时不得猜测或用模板示例替代；
- 零值、极值、NaN/Inf、空维度及尾部元素的数学行为；
- 精度标准、功能优先级和至少一组可执行 P0 典型配置。

公开接口是否要求辅助 tensor 由调用方预展开/预广播，属于输入 shape 关系，必须在这里保留；
kernel 内如何据此 broadcast、切 tile 或索引属于 Stage 3，不能提前设计。

公式不能表达分块、循环、递推、在线更新或状态依赖时，补充带编号的算法步骤。
算法描述定义数学过程，不提前规定 tile 或硬件实现。

### 2. 识别会改变公开语义的特性

只列当前算子实际涉及的特性，例如 mask、量化、融合、动态 shape、混合精度、随机性、
稀疏性或数值稳定策略。每项记录：

- 是否需要；
- 来源与置信度；
- P0/P1/P2/P3 优先级；
- 对公式、接口或验收的影响。

P0/P1 必须进入首个版本；P2 可选；P3 明确暂缓。不要在本 skill 中判断 PyPTO-Pro
实现复杂度或 API 支持度，交给 material-explore。

对 optional 参数执行四项语义分析：在公式中的位置、计算作用、与其他参数的依赖/互斥、
省略时的默认语义。不要把 kernel 实现方式混入这项分析。

### 3. 集中处理歧义

- 交互模式：一次展示数据流摘要、规格清单、关键歧义及拟采用默认值，整体确认；最多两轮。
- 用户输入已经完整时，直接展示确认稿，不机械追问已给出的事实。
- 用户明确要求不提问或持续执行时，进入无人值守模式；仅对非阻塞字段使用可追溯默认，
  并在 SPEC“自动决策”中记录采用值、来源、理由和影响。
- 若数学语义或最小输入输出合同仍无法确定，不猜测，返回 `blocked` 和缺失项。

允许使用的默认值必须先披露再持久化。shape、dtype 和动态轴只有在目标标准接口确有默认
或用户确认后才可采用；没有依据时保持未决并阻断，不能用示例模板值替代。

### 4. 生成并自检 SPEC

复制 [SPEC 模板](templates/spec-template.md)，逐项替换占位符；删除不适用的可选行，禁止把
模板占位符或示例值留在成品中。`json machine-contract` fenced block 是唯一机器事实源；
正文只解释公式、接口语义、边界和证据，不重复 shape、dtype、容差或 P0 字段值：

- `op_name` 对应算子名；
- `formula` 用等式或编号伪代码步骤定义每个公开输出；复杂算法仍须让每个输出成为赋值或箭头目标，正文只解释符号与依据；
- `supported_dtypes` 按首次出现顺序列出本 SPEC 公开输入输出实际使用的全部 canonical dtype；同一接口的另一组 dtype 组合拆为独立 class/SPEC，避免下游静默丢失 index、mask 或量化辅助 tensor 的 dtype；
- `inputs` / `outputs` 按公开签名顺序完整记录 name、shape、dtype、有限 value_range；
- `p0_cases` 逐案记录 name、params、input_shapes、output_shapes；校验器会从第一项导出
  只供旧调用方使用的内存字段 `p0_shapes`，SPEC 不再双写该字段；
- `default_params` 只包含公开签名中已确认的标量默认参数；没有则写 `{}`；
- `tolerance`、动态轴范围、shape 约束和性能目标直接记录已确认裁定；

所有 P0 的映射 key 与顺序分别等于 `default_params`、`inputs`、`outputs`；具体 shape 必须
等于把本 case 输入/参数代入合同 shape 表达式后的结果。rank-0 tensor 写 `[]`。tile/kernel
shape 仍由 Stage 3 决定。每个动态 shape 符号须在至少一个输入维中单独出现，保证 P0
具体 shape 能确定地绑定该符号；其他输入/输出可使用由它组成的表达式。

运行模板旁的校验器；非零退出时修复后重跑：

```bash
python "$CANNBOT_CONFIG_ROOT/skills/pypto-pro-intent-understand/scripts/validate_spec.py" \
  custom/<op>/SPEC.md
```

## 完成条件

- 算子名、数学语义和最小输入输出合同均已确定；
- shape、dtype、动态轴、optional 参数和边界行为已确认或按无人值守规则留有证据；
- 复杂算子含可恢复的算法步骤；
- 所有功能均标优先级，P0/P1 没有遗漏；
- 所有 P0 配置均可供 golden 构造输入，且输出 shape 与机器合同公式一致；
- 所有机器合同输入/输出均有有依据、可解析的有限 `value_range`；
- SPEC 恰有一个严格 JSON machine-contract、正文没有重复机器字段、无占位符或未披露默认值；
- `validate_spec.py` 通过。
