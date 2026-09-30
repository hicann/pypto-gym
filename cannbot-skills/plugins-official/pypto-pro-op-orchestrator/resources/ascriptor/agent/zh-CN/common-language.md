# Kernel 工作的共同语言

每个 kernel 上下文在 router 后读一次，用这些词填写 [preflight](references/authoring-preflight.md)。
首次编写再读[执行模型](concepts.md)。精确操作属于 [library API](../../library/docs/api/README.md)。

| 术语 | 工作含义 |
|---|---|
| Facade / device profile | 编写词汇 / 编译器与模型目标；`ascriptor.a5` 的 canonical ID 为 `950` |
| Backend / launcher | 源码生成目标 / 执行机制 |
| Stage / launch | 内部数据流阶段 / 一次 runtime 调用 |
| Core owner / work item | 对逻辑输出区域负责的参与者 / 分配给它的工作项 |
| Logical extent / footprint / allocation | 有效元素 / 指令实际地址 / 拥有的 backing 字节 |
| GM / UB / L1 / L0A,B,C | 公共全局存储 / 向量本地存储 / cube 暂存、操作数与累加器 |
| View 或 reinterpret / cast | 地址或位的解释 / 带 rounding 的数值转换 |
| Var / Reg / MaskReg | 可变标量单元 / VF 寄存器 / lane predicate |
| Slot / event credit | 物理存储版本 / 发布或复用权限 |
| Producer / last reader | 写者 / 复用前必须退役的最后物理读取 |
| Pipeline warmup / steady state / drain | 填充调度 / 重复常规工作项调度 / 消费剩余有效工作并发布最后输出 |
| Precision boundary | 有序的累加、materialize、cast、rounding 与 saturation 决定 |
| Criteria met / headroom | 约定准出满足 / 进一步可改进空间 |

混合流水按依赖与可用 slot 推导各 stage 的工作索引、lookahead 和 drain。
CVC/VCV 描述共享物理资源的图；独立引擎数与缓冲深度须另行推导，见[流水方法](references/pipeline-model.md)。
物理访问查 [storage](../../library/docs/api/storage.md) 和[内存边界](references/memory-and-tails.md#vector-tail)。
窄 view 提供地址范围；指令 predicate 与 footprint 仍须显式选择。

<a id="evidence"></a>
## 证据与结论

| 结论 | 应保留的证据 |
|---|---|
| 声明操作或支持组合 | 当前 owner 声明及精确 device/dtype/layout/case/backend 范围 |
| 生成源码 / 厂商编译 / 设备执行 | 分别记录 emit、compile、run 及实际源码与依赖身份 |
| 算术正确 | 运行时生成的独立 reference、完整输出与被拒绝的错误输出对照 |
| 同步正确 | Balance 以及 lowered 物理访问、hazard、deadlock、归属与最后 reader 生命周期 |
| 计算重叠 | 同核实际 stage/item 区间，排除 DMA 与同步 |
| 硬件性能 | 同设备测量；模型 cycles、CPU 时间与诊断采集各自记录原单位 |
| 指南效果 | 限定任务的新上下文试验；改进结论需要可比较且重复的基线/候选试验 |

报告结果时使用本表。缺失证据保持未知；候选失败或目录缺项后，进入聚焦源码与 probe 调查。
性能任务区分 unique、requested 与实测 HBM bytes，按 [Roofline](references/roofline.md)解释。
专题保留自身约束与反例。每个 checker 或试验说明实际验证了什么，
以及哪些阶段或输入范围仍未测试。
