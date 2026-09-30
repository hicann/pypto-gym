# Kernel 编写浓缩起点

本页是跨任务的**起步索引与检查表**，覆盖 Ascriptor DSL 的共同语言、单 kernel 编写路径、常见物理边界和验证层级。它不列举所有 opcode、overload、设备形态或后端缺口；需要某个操作的精确签名与支持域时，打开该操作的 owner API、一个匹配的可运行样例及其 `study_for` / `do_not_copy_when`。源码与现行规范始终优先。本页不构成硬件验收或性能证据。

适用：编写、实质修改、分解、调试和优化 NPU kernel 的开场。本页的路线图、共同语言、单 kernel author 步骤和通用 preflight 表满足**中文单 kernel 编写**的初始阅读；完成当前任务命中的 owner 查询即可，不为重复而重读 router/author。分解、调试、优化等其他路线仍读所选 playbook；英文路线读英文 router/playbook。纯文档维护、迁移、Pro 反向导入和厂商算子移植仍按各自 playbook。已有相同的启动注入内容时，不再通过工具重复读取。若本页与现行规范不一致，以[中文](../zh-CN/ROUTER.md)或[英文](../en/ROUTER.md) router 和 owner 原文为准。

在本快照的 `agent/` 目录运行 `python tools/build_kernel_context.py --print`
直接读取全文，也可直接打开本文件。浓缩指南不依赖来源哈希或刷新步骤；修改 owner
文档时按内容需要同步本页。编译器源码快照与验收产物继续使用各自的完整性校验。

## 1. 先定路线与来源

- 本仓自包含快照的 `sources.json` 和 `sources-index.json` 决定源码身份；
  `library/`、`kernels/`、`agent/` 必须一起使用。
  [pyproject.toml](../../library/pyproject.toml)声明版本，
  [状态](../../library/docs/status.md)说明验证范围。
- 用户给公式、reference 或模型 → [author](../zh-CN/playbooks/author.md)；多个 launch 的分解 → [decompose](../zh-CN/playbooks/decompose.md)；已有分解实现 → [implement-decomposition](../zh-CN/playbooks/implement-decomposition.md)；错误/hazard/交接审查 → [debug](../zh-CN/playbooks/debug.md)；正确候选的性能改进 → [optimize](../zh-CN/playbooks/optimize.md)。只选与任务一致的 playbook，不把整个画廊预读进上下文。
- `library` 拥有公开 API、规格、编译器/runtime 与 defect；`kernels` 拥有算法 demo、公式和比较契约；`agent` 拥有路线与工作流。冲突先找 owner；目录/索引是导航，不是支持证明。[API 总入口](../../library/docs/api/README.md)、[API manifest](../../library/docs/api/manifest.json)、[kernel 画廊索引](../../kernels/index.json)各司其职。
- 每个 kernel 新上下文的共同词义：facade 是编写词汇，device profile 是模型/编译目标，backend 生成源码，launcher 运行某阶段；stage 是核内数据流段，launch 是一次 runtime 调用；core owner 是对某一逻辑输出区域负责的参与者；logical extent 是有效元素，footprint 是**指令实际访问**，allocation 是拥有的物理 backing。View/reinterpret 改地址或位解释，cast 改数值。Slot 是物理缓冲版本，event credit 是发布/复用许可，不能相互代替。详见[共同语言](../zh-CN/common-language.md)。

## 2. 写代码前填一份任务契约

把决定与依据写在忽略的 `tmp/<task>/`，可用[契约模板](../templates/authoring-contract.md)。从用户的输入和独立数学 reference 推导，未决项只阻塞依赖它的实现。

| 决定 | 至少写清 |
| --- | --- |
| 语义 | 公式与括号顺序；shape/domain、非有限值；cast/rounding/saturation；规约顺序；输入输出 alias；误差预算。 |
| 交付 | 一个 kernel launch 还是约定的多 launch composition；允许的 host 工作；完整 workload 与停止条件。不要把用户要求的算术移到 host。 |
| ABI | `GM[dtype, shape]` 参数次序、显式标量、输出由调用者分配并传入、`return` 命名可观察结果；动态 shape 的绑定。 |
| 归属 | device/core 数、每核工作项与输出区域、各输出唯一或合法共享的写者；最后一块、空路径和跨核会合。 |
| 存储 | 每级容量、UB/L1/L0 与 GM 的逻辑范围、实际 footprint、物理 pitch/alignment、slot 数、初值与最后 reader。 |
| 验证 | 运行时输入、独立 reference、完整输出/guard、负对照、sim/pipesim/emit/硬件的目标阶段和执行环境。 |

单 launch 的公式留在 kernel 中；host 可以造输入、分配输出、dispatch、比较，以及交付契约允许的准备。`@kernel` 后的 Python 函数会被 AST 编为 Surface IR，不是普通可直接调用的 host 函数；`kernel.ir()` 查看解析结果。`range` 是 device 循环，即使边界为字面量；`unroll` 是编写期展开。`Var(dtype=i32)` 是可变标量单元，普通 `=` 只是重新绑定 Python 名字。精确静态子集、helper、分支与动态索引读[Authoring API](../../library/docs/api/authoring.md)和[Reference](../../library/docs/api/reference.md)。

## 3. 最小数据路径与设备差异

一个 A5 固定形状起步锚点来自 [AXPB owner 样例](../../library/examples/api/axpb/kernel.py)：typed GM 输入/输出 → UB staging → `@vf` 寄存器计算 → GM 写回。下方只展示形态；原样例的完整 imports、独立 reference、case 和限制在[目录](../../library/examples/api/axpb)及[metadata](../../library/examples/api/axpb/metadata.json)。它只覆盖一核、`[1,64]` FP32，不能直接推断新 shape 的 tail、核数或性能。

```python
from ascriptor.a5 import DT, GM, Position, Reg, Tensor, add, auto_sync, f32, kernel, muls, vf

@vf
def axpb_vf(x_ub: Tensor, y_ub: Tensor, o_ub: Tensor):
    x = Reg(DT.float)
    y = Reg(DT.float)
    x <<= x_ub[0]
    y <<= y_ub[0]
    scaled = Reg(DT.float)
    muls(scaled, x, 2.0)
    total = Reg(DT.float)
    add(total, scaled, y)
    o_ub[0] <<= total

@kernel(mode="vec", block_dim=1)
def axpb(x: GM[f32, (1, 64)], y: GM[f32, (1, 64)], o: GM[f32, (1, 64)]):
    ux = Tensor(DT.float, [1, 64], Position.UB)
    uy = Tensor(DT.float, [1, 64], Position.UB)
    uo = Tensor(DT.float, [1, 64], Position.UB)
    with auto_sync():
        ux <<= x
        uy <<= y
        axpb_vf(ux, uy, uo)
        o <<= uo
    return o
```

`<<=` 按源/目标空间选择搬运或赋值；`auto_sync` 处理受支持的同侧依赖，不是跨 cube/vector 或多参与者的万能屏障。A5 `Reg`/`MaskReg`/`@vf` 的 load、store 分布及 mask 见[Registers API](../../library/docs/api/registers.md)。`@simt` 的线程与 atomic 归属另查[SIMT API](../../library/docs/api/simt.md)。Cube 的输入先入 L1，操作数入 L0A/B，累加器在 L0C；`matmul` 的逻辑式与物理限制看[Cube API](../../library/docs/api/cube.md)，从 L0C 排到向量侧要另算 sub-block 与转换限制。

| 目标 | 正确起点 |
| --- | --- |
| A5 | `ascriptor.a5`，register/VF 或 SIMT；`mode="vec"`、`"cube"`、`"mix"` 按所需数据流选择。A5PR 的 `ascriptor.a5pr` 仍是单独实验 facade。 |
| A2/A3 | `ascriptor.a2` / `ascriptor.a3`，CCE 与 tensor-vector UB 操作；没有 A5 的 `@vf`/`@simt` 形式。先读[A2/A3 入口](../zh-CN/references/a2-a3.md)与[A2 Vector API](../../library/docs/api/a2-vectors.md)。 |
| PyPTO-Pro 目标 | 先读[专题](../zh-CN/references/pypto-pro.md)：每个 shape 先 emit，按标量绑定；交付默认显式 `auto_mutex=True`，manual 仅在用户要求时生成；开发与原始证据在 `custom/<op>/`，精简可运行包在 `delivery/<op>/`。 |

另一个跨 facade 的小锚点是 [cube_matmul](../../library/examples/api/cube_matmul/kernel.py)：固定 16×16 FP16 输入，L1 操作数、L0C FP32 累加、ND FP32 输出。它的 `make_kernel(device)` 按设备取 `api.kernel`；底下这段只表示一次 tile 的完整流向，**不**表示任意 K、bias、split-K 或多核已经处理：

```python
@api.kernel(mode="cube", block_dim=1)
def cube_matmul(x: GM[f16, (16, 16)], y: GM[f16, (16, 16)], o: GM[f32, (16, 16)]):
    lhs = Tensor(DT.half, [16, 16], Position.L1)
    rhs = Tensor(DT.half, [16, 16], Position.L1)
    accum = Tensor(DT.float, [16, 16], Position.L0C)
    with auto_sync():
        lhs <<= x
        rhs <<= y
        matmul(accum, lhs, rhs)
        o <<= accum
    return o
```

若目标是 A2/A3 vector，先看 [a2_fma](../../library/examples/api/a2_fma/kernel.py) 的 UB tensor 操作：`adds(acc, c, 0.0)` 在 `muladddst(acc, a, b)` 前初始化 accumulator；后者读取旧 `acc`，不是纯 `a*b`。样例用双 slot 覆盖反复复用和 tail，混合精度路径的 repeat/stride 依 dtype 推导；换 dtype 或 tile 时重新推导，不照抄常数。[metadata](../../library/examples/api/a2_fma/metadata.json)列出它能教的边界与不能复制的情况。

Facade 中有名字不意味着所有 dtype、layout、stride、backend 形式都被接受。既有 kernel 是某个 domain 的证据，不是新形式的授权。选 primitive 时走[library API 样例](../../library/examples/api/README.md)，选完整算法走[kernels 画廊](../../kernels/README.md)及其 `index.json`；先看候选 `metadata.json` 的 `study_for` / `do_not_copy_when`，再读**一条**匹配样例的源码、reference、main。仅有词组时在 agent checkout 用 `python tools/select_example.py --query '<词组>' --device a5 --language zh-CN`，把 device 改为实际系列。

### 特征出现时直接展开

先用下面的入口找**当前操作**的精确契约；不要沿链接递归预读。若任务边界变化，再回来选下一行。[preflight 原表](../zh-CN/references/authoring-preflight.md)给出更完整的触发条件。

| 任务特征 | 下一处 owner 证据 | 要解决的问题 |
| --- | --- | --- |
| A5 VF、register、mask、distribution | [Registers API](../../library/docs/api/registers.md)及同 dtype 样例 | load/store 物理 lane、谓词范围、初始化与可用形式。 |
| A2/A3 vector 与 repeat | [A2/A3 入口](../zh-CN/references/a2-a3.md)、[A2 Vectors](../../library/docs/api/a2-vectors.md) | UB 指令的 count/repeat/stride 与持续 mask 状态；不能套用 A5 寄存器。 |
| 普通 tail、view、DMA、padding | [内存边界](../zh-CN/references/memory-and-tails.md#vector-tail)、[Storage API](../../library/docs/api/storage.md) | valid、指令 footprint、allocated backing 与物理行距分别推导。 |
| Packed、cast、量化 | [Packed 写回](../zh-CN/references/memory-and-tails.md#packed-writeback)、[Formats API](../../library/docs/api/formats.md)、[精度](../zh-CN/references/precision.md) | 位序、舍入、carrier、store distribution 与实际拥有的字节。 |
| 规约、广播、在线状态 | [数值模式](../zh-CN/references/numerical-patterns.md#reason-reduction)、[Registers API](../../library/docs/api/registers.md) | 结果保留在哪些 lane、何时 broadcast、identity 与 cast 次序。 |
| SIMT、atomic、参与者会合 | [SIMT 起步](../zh-CN/references/patterns.md#simt-start)、[SIMT API](../../library/docs/api/simt.md) | 线程/核输出归属、原子贡献者与 barrier 范围。 |
| Sort/TopK | [记录检查](../zh-CN/references/patterns.md#sort-start)、[Sorting API](../../library/docs/api/sorting.md) | score–ID 配对、tie 顺序及 record footprint。 |
| Cube、bias、MX、L0C 排空 | [Cube API](../../library/docs/api/cube.md)、[排空设备事实](../zh-CN/references/facts-device.md#排到向量侧这一步没有安全的默认值) | L1/L0 几何、累加/输出 dtype、sub-block 分配与转换。 |
| 跨 cube/vector、slot 多次复用 | [交接调用序列](../zh-CN/references/cross-side-handoff.md)、[同步专题](../zh-CN/references/synchronization.md) | producer/last reader、发布/归还、credit 与真实 slot 数。 |
| 多 stage/多 launch、性能 | [分解](../zh-CN/playbooks/decompose.md)、[流水方法](../zh-CN/references/pipeline-model.md)、[Roofline](../zh-CN/references/roofline.md) | stage ABI、warmup/drain、HBM bytes 与测量目标。 |
| Attention 或 softmax 状态 | [Attention 专题](../zh-CN/references/attention-authoring.md)、一个精确 canonical case | 在线递推、mask 次序、布局、输出与数值预算。 |

## 4. 物理范围、尾块与精度

先定 tiling、各 core 的输出归属及数据流，再写最小完整路径。每项 DMA、cast、buffer、同步边都要对应语义、footprint、容量、精度或依赖；不能以“样例也这么写”为理由。每个访问分别列出 `valid` 元素、指令真实 footprint、owned allocation、source/destination stride、是否执行、最后一行/slot；view 只改变逻辑窗口，不自动缩小 DMA/register 指令或提供 mask。[Storage API](../../library/docs/api/storage.md)与[普通尾块检查](../zh-CN/references/memory-and-tails.md#vector-tail)是首读。

- 普通 A5 register↔UB、GM↔UB 指令的 **UB 侧起址 32-byte 对齐**；GM offset 遵循自己的规则。多行搬运逐行检查 UB 物理 pitch，完整 padding footprint 应在自身 backing 内。合法逻辑切片不保证邻接分配不被访问。全关 mask 也不豁免一条已执行指令的 base alignment。
- `Reg` 的 load/store footprint 随 dtype 与 distribution 变化；连续 load 可能读满寄存器，store predicate 也可能按块取整。按具体 API 选择 mask，不从窄 view 猜。无效 lane 在首次消费前初始化、屏蔽或由明确 identity 填充。
- Packed 格式区分逻辑 lane、载体字节、位序、cast/pack、UB pitch 和 GM 写入范围。`reinterpret` 不做数值转换；受支持的窄格式与 bit codec 看[Formats API](../../library/docs/api/formats.md)。比较原始 carrier bytes，并检查行尾、下一行、guard。`ub_to_gm_pad` 等操作的 payload/gap 单位须按[Storage API](../../library/docs/api/storage.md#transfer-units-and-initialized-regions)核对，不可凭名称猜。
- Reduction 先在首次受影响操作前应用 mask/identity；在线状态、softmax 的 max/sum 次序、workspace downcast、saved state 与输出 cast 都是语义边界。保持原公式括号顺序和声明精度；`allclose` 阈值需有推导，exact/bitwise 也需有理由。[精度专题](../zh-CN/references/precision.md)列出相应边界，不能把旧任务的数值预算搬给新任务。

## 5. 同步、复用和流水

每个 slot 写明 producer、全部 consumer、**最后物理 reader**、何时可覆盖，以及使它退役的 pipe。正向 publish 只保证本轮生产先于消费，不能自动保证下一轮覆盖前旧 reader 已结束。`auto_sync` 排序可覆盖的同侧依赖；跨侧 handoff 先按[调用序列](../zh-CN/references/cross-side-handoff.md)和[同步 owner API](../../library/docs/api/synchronization.md)选设备支持的协议。A2/A3 的 workspace bridge 与 A5 片上 handoff 不可互换。[同步专题](../zh-CN/references/synchronization.md)解释 slot、event credit 和最后 reader。

`Tensor/DBuff/TBuff/QBuff` 分别有 1/2/3/4 个物理 slot；event 数不是存储容量。重复 CVC/VCV 或其他混合流水先画 stage × work-item 依赖，推导 warmup、steady、drain，按最大同时存活角色算容量，再验证 lookahead 和尾项。[通用流水方法](../zh-CN/references/pipeline-model.md)提供推导。至少让活跃 core 真正复用 slot 多次；单次 multicore smoke 不能证明 reuse。对带条件的 producer，分别检查零轮、一轮、多轮、跳过生产者和最后 drain。`pipesim` 查 lowered event balance/hazard/deadlock，但 VF 内部的 register 次序和硅片行为另需证据；警告不得压掉后声称同步通过。

## 6. 按阶段运行，按证据交付

用**实际执行的解释器**确认 `ascriptor` 版本/导入路径和所选源码，先看 `ascriptor doctor`。`OpExec` 的调用按签名传入所有 tensor（**含调用者分配的输出**），显式标量随后传；比较返回值而不是只看传入的缓冲。对需要保留初始化的输出用 `seed_outputs=True`；完整覆盖型输出先 poison，以便发现未写 lane。[执行页](../zh-CN/references/development-execution.md)给出调用、设备与 launcher 边界；`sim`/`pipesim`/`board` 等都在调用它的**本机**运行，设备任务要把完整脚本放到分配的机器上运行。

开发脚本的核心调用只需如下形态；`x`、`y` 和已分配且 poison 的 `o` 是同一解释器创建的真实输入。对每个 launcher 用相同独立 reference 比较实际返回值，并保留不同阶段的结果：

```python
from ascriptor.runtime import OpExec, compile_kernel

artifact = compile_kernel(axpb, backend="cce")  # 只发射源码
got = OpExec(axpb, launcher="sim", seed_outputs=True)(x, y, o)
assert_same_contract(got, independent_reference(x, y))
```

这里的 `assert_same_contract` 与 `independent_reference` 由任务编写，不能调用被测 kernel 或其 codec。若任务目标是 PyPTO-Pro，先将 `backend` 改为 `"pypto_pro"` 并提供需要的 `bindings`，再按其专题验证对应 case；`compile_kernel` 返回的 artifact 仍未经过厂商编译。

建议验证顺序：检查 `kernel.ir()` 的签名与数据路径；用 `compile_kernel(entry, backend=...)` 尽早生成目标源码（emit 不是厂商编译）；根据任务的硬件优先规则跑完整 workload，或者在明确的成本例外下先做本地诊断；用真实文件和运行时生成的输入做 `OpExec(..., launcher="sim")` 的独立 reference 比较，再用 `launcher="pipesim"` 查物理同步。模型失败进入[debug](../zh-CN/playbooks/debug.md)定位最小 owner 层；硬件不可用则明确记为未验证。PyPTO-Pro 先 emit 尤其重要，因为操作数的具体形式可能仅在该阶段被拒绝。

独立 reference 不 import kernel、编译器或被测 codec，按用户契约生成 shape/domain 与边界输入。逐个核对输出个数/名称、shape、dtype、全部有效 lane、未写区、alias/输入不变；必要时比较字节、NaN/Inf、signed zero。用故意错误的输出（如全零、错误位序、单元素损坏或遗漏 stage）确认比较器会拒绝。保留第一份失败、最小复现和修复前后证据；不要删 case、换 backend 或放宽阈值换取通过。

| 能报告的结论 | 必需证据 |
| --- | --- |
| 语义/算术 | 独立 reference 与完整输出比较；范围明确，负对照会失败。 |
| 同步 | `pipesim` 的 lowered balance/hazard/deadlock、归属和最后 reader；超出模型范围另列。 |
| Backend | 源码 emit、厂商编译、设备执行**分别**报告；emit 通过不等于上板通过。 |
| 性能 | 选定 artifact、真实设备与完整 workload 的测量；模型 cycles、CPU 时间和诊断输出不能冒充硬件时延。 |

最终报告交付路径、source/依赖身份、实际环境、case × stage、通过/失败、阻塞项与下一边界；在 `tmp/<task>/` 留[工作流检查点](../templates/workflow-checkpoint.md)。设备任务按[硬件优先流程](../zh-CN/runtime-and-maintenance.md#hardware-first)检查设备、锁和隔离；机器配置在外部忽略文件中，报告前清理板上输出。
