# 实现前的契约

从用户公式/reference、当前 API 和单元源码推导字段，记录一次，避免重复问卷。只有证据
无法决定可观察语义才需要提问；普通 tile、标识符和文件安排由实现者决定。

| 字段 | 必须表达 |
|---|---|
| 数学 | 运算、括号、常数、reduction 轴、边界行为 |
| Public ABI | 命名 IO、顺序、shape/symbol、dtype、stride/layout、物理存储 |
| Mutation | 输出初始化、in-place alias、禁止重叠、返回值 |
| Domain | 最小/最大 shape、空输入、对齐、tail、有效 core 数 |
| Precision | 累加 dtype、每处 cast/rounding/saturation、非有限值行为 |
| Comparison | 各输出 exact/bit/numeric 规则、容差及理由 |
| Delivery | device/backend、launch 数、允许的 host 工作 |
| Ownership | workspace producer/consumer、生命周期、已初始化/已定义范围 |
| Verification | seed/case、独立 reference、精确命令 |

多 stage 再记录 DAG、stage signature、边界 layout、producer 完成/consumer 就绪、saved
state schema/version，以及逐 stage/端到端预算。保存状态即使在 forward/backward 内部也
是输出契约；说明 materialize/recompute、分配和释放归属。必需 preparation 在每个发布单元内。

Host preparation 必须在任务约定内：不得把要求在 kernel 内完成的计算移到 host，或为让候选
通过而静默 cast、重排、变换输入。允许的 preparation 要记录对数值、dtype、layout 的影响、
执行位置，以及是否计入比较和计时范围；必需步骤随单元交付，并对原始任务输入用独立 reference
验证。即使只改 shape，也必须保持约定 ABI 与元素解释；不存在无条件安全的预处理白名单。

使用[编写模板](../../templates/authoring-contract.md)和[分解模板](../../templates/decomposition-plan.md)。
两个 owner 都没有机器 contract。library API 样例与 kernel demo 使用同样四个文件，
由 [RFC 0012 第 3 节](../../../library/docs/rfc/0012-product-contracts.md) 规定。
目录算什么写在 `metadata.json` 的 `formula` 或 `surface` 里，拿什么核对写在 `reference.py` 的
代码里。叙述模板对两者都不另造协议。

备选的几份源都在同一个 `kernel.py` 里，`main.py` 用 `--variant`（或按该目录的轴命名的
`--pattern`、`--mode`）选其一，每个 case 自带由它自己的算术证成的容差，且写在做比较的地方。
两份不许漂移开的源要分别写出来而不是共用，这样改动其中一份不会悄悄改到另一份。某个 variant
的输出集合或精度确实不同时，就写在 `main.py` 里、紧挨着读它的那段比较——`register_groups`
与 `cast_formats` 是两个这样做的例子。

`reference.py` 导出 `make_inputs(case) -> dict` 与 `reference(inputs) -> dict`，流水型还导出
`reference_stages(inputs) -> dict`；`main.py` 放 `execute`、比较，以及比较这些中间结果的
`--stages`。reference 不修改共享输入、不调用 simulator，输入每次由 case 的 seed 重新生成，
不支持的执行组合明确报错，不回退 reference。

公开 ABI 使用 typed `GM[dtype, dims]`，必要时按声明使用 `GMList[dtype, dims, count]`。
未类型化 placeholder 不是新公开契约。不编造 `shape_bindings`；symbol 由 typed 维度与签名
顺序的显式 scalar 绑定，以相应版本声明为准。

数学相等不等于浮点执行相同。非 reduction 轴划分可保持元素归属，在 materialized 边界拆分
可保持该边界；fusion、reduction split、recompute、atomic merge、cast 移动与分配律仍需
符合精度契约。不存在跨架构/格式通用的“atomic 只支持 float”或固定最小 matmul tile。
错误契约可凭源码证据、明确版本改动和回归修正，不能仅为迎合失败输出调整 expected。
