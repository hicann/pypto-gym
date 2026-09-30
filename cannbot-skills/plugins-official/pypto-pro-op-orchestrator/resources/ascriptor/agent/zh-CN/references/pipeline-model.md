# 推导多阶段缓冲流水

用于重复的混合计算：CVC、VCV、CVCV、VCVC 及更长依赖图。阶段名字帮助选择样例，不
决定固定缓冲配方。从[preflight](authoring-preflight.md)开始，填写
[流水计划](../../templates/pipeline-plan.md)；涉及性能选择时同时读[Roofline](roofline.md)。

## 定义依赖图与工作

为每个 stage 记录资源与参与者、输入输出、数学精度和工作映射。边可能连接同一工作项、
上一项、多项 producer 或不变共享输入。记录所有 reader，包含更晚阶段与延迟标量状态。

区分全局工作项、核间分配、本核流水轮次、各阶段自己的工作索引。推导每个活跃核的工作
项数，包含空闲核和不均分配。增加核数可能使每核只有一项，从而消除跨项 lookahead 空间。

两次 cube 阶段共享 M 及关联搬运资源；多个 VF 阶段也共享对应 vector 资源。跨项重叠时
仍须保持项内依赖。递推可以限制某阶段提前，同时允许独立输入预取。

## 选择合法调度

一一对应的阶段可用 `k_s = t - d_s` 描述本核第 t 轮 stage s 的工作项，并使用
`0 <= k_s < N_s` 的有效条件。`d_s` 来自依赖、资源和容量，不必等于 stage 编号。分块
粒度不同时显式定义 `k_s = f_s(t)`、速率、join 和有效输入，通用延迟表本身不能证明可执行。

### 相邻两阶段为一组

一一对应的交替流水按相邻两阶段分组：第 g 组处理 `t-g` 项，最后一组可以只有一个阶段。
阶段从 0 编号时，`d_s = s // 2`。CVCV 的一步 lookahead 写成：

```text
for i in range(N + 1):
    if i < N:
        C1(i); V1(i)
    if i > 0:
        C2(i - 1); V2(i - 1)
```

这里的调用代表所属执行侧的工作与必要同步。Mix kernel 会
[拆成 Cube/Vector 两侧](../../../library/ascriptor/passes/split_sides.py)，阶段的实际开始时间由
依赖、资源和事件决定。同一个 `if` 不构成两侧共同完成的屏障，buffer 仍按工作项身份选择，
容量与复用仍按最后 reader 的实际完成事件推导。

| 数据流 | 分组映射 | 必须验证 |
|---|---|---|
| C1 → V → C2 | C1(t)、V(t)、C2(t-1) | V 的结果保留至下一组 C2；C1/C2 共享 cube |
| V1 → C → V2 | V1(t)、C(t)、V2(t-1) | C 的结果保留至下一组 V2；两次 VF 共享 vector |
| C1 → V1 → C2 → V2 | C1(t)、V1(t)、C2(t-1)、V2(t-1) | 两组交错，保留跨项版本与最后 reader 同步；一步 drain |
| V1 → C1 → V2 → C2 | V1(t)、C1(t)、V2(t-1)、C2(t-1) | 双向 handoff、最终 cube 操作数寿命与资源共享 |
| C1 → V1 → C2 → V2 → C3 | C1(t)、V1(t)、C2(t-1)、V2(t-1)、C3(t-2) | 三组独立索引与完整排空 |

若共 G 组，每组用 `0 <= t-g < N` 判断有效工作，遍历 `range(N + G - 1)` 完成启动与排空。
这些是逻辑工作分组；实际重叠须通过 trace 验证。跨项状态、额外 reader、不同处理速率
或逐轮屏障需要重新核对依赖及容量。

明确同轮 producer/consumer 和资源使用者的发射顺序。Producer 不能阻塞在只有后续、
因而无法到达的 consumer 才能归还的容量上。只增加 credit、不改变存储与调度不能解决
这种环；必要时改为 consumer 先发射、缩短 lookahead 或分配独立存储。

概念上的计划展开如下：

```text
遍历包含启动与排空的每个流水轮次：
    按证明过的顺序访问阶段
    计算此阶段自己的工作映射
    若工作有效且所有依赖满足：
        获取容量，执行并发布，完成必需读取后的回收
```

这是推理方法，不是 DSL 支持的动态 stage dispatcher 语法。实现须使用当前支持的静态/
原生控制流并验证生成 IR。[CVC 教程](cube-vector-cube.md)展示如何映射到具体源码。

## 推导物理存储和 credit

对每条边和角色标出受保护的写入、全部 reader、最后物理 reader 所在 pipe，以及允许
覆盖的 release。所需槽位等于该调度下同时存活物理版本数的最大值。总容量计入 alias、
padding footprint、subblock pitch 和长期常驻输入。

计数依据是已证明的发射/完成偏序及复用约束，不能只看逻辑轮次标签的差。若调度允许在
前项最后 reader 退役前发布新项，就必须容纳两个版本；若已有两个旧版本仍然存活，要
无停顿地发布新项就需三个槽。提前退役或容量等待可以改变存活数与实际排程，stage 个数
本身不能选择 DBuff/TBuff/QBuff。推导给出的是充分深度，不是最小深度：接受不同停顿的
更小深度也可能合法，上面的推导并不排除它。

写入使用 producer 的工作项身份，读取使用 consumer 对应的同一版本，不能用当前生产
索引读取延迟数据。不同 lifetime family 分开计数。后续 cube 再次读取的 K/V 必须保留到
最后一次读取退役；rescale 等递推元数据也必须跟随其消费项。

Event/mutex credit 描述权限与最大 outstanding 工作，不分配存储。初始 credit、
lock/ready/wait/free、物理轮转和最终 drain 必须匹配。正向发布保证 producer 完成，容量
在最后 reader 之后归还，可能涉及不同 pipe。保护依据是实际的存储角色而不是 stage 边界：
在不需要该容量的工作之前就去获取 handoff 容量是合法的，代价是挡住了被它阻塞的独立计算。当前接口与反向复用边见[同步](synchronization.md)。

## 延迟数据与标量元数据一起保留版本

延迟 consumer 可能同时需要 tensor 及其对应的 scale、mask、length 或 normalizer。
即使它们采用不同的物理 slot family，也要按同一工作项身份保留。Tensor 有双槽但 scale
仍写同一个共享位置，可能通过常量输入，却破坏后续非均匀工作项。

分别处理不同递推：下一 producer 的规约状态可以提前，前一 consumer 的输出状态仍可
等待，前提是对应元数据未被覆盖。除了 slot 回绕，还检查工作项切换。
[Attention 专题](attention-authoring.md)将该原则用于 online max/sum、延迟 rescale 和
PV 累加；写出 lookahead 循环本身不会消除输出更新阶段。

## 完成首尾处理

逐阶段推导有效轮次，最后一个必需消费者完成后才能结束。`N+1` 只是一步 lookahead 的
一种写法；非均匀延迟或速率必须分别计算最终有效工作项。检查每个必需 stage 对每项
恰好执行一次，包含状态更新与最终 store，首尾无效阶段正确跳过。

带 guard 的 producer 跳过不代表此前 pending token 消失，应单独处理旧工作。Join 等待
全部必需 producer，共享输入在所有 consumer 退役后释放。递推保持原算术和 rounding
顺序。超容量、缺少受支持 handoff 或违反递推时，要修改调度或给出有证据的具体限制。

## 验证实际排程

1. 对独立 reference 验证数值、存储与精度 ABI。覆盖单项、深度边界、同活跃核多次周转、
   不均分配和支持的 tail；零工作仅在 contract 内测试。
2. 检查 lowered hazard、deadlock 和 event balance，保留提前覆盖、漏 drain、延迟状态
   索引错误的负例。漏最终 drain 正是数值对照不能省的理由：它能通过 hazard 与 deadlock
   检查，只是少输出一个工作项。
3. 研究排程时使用匹配串行对照；可能存在模块缓存串用时在新进程分别运行候选和对照。
4. 将实际 compute task 区间映射到 stage/item，并保留 source/IR 身份。核内重叠使用同一
   mixed core 对应的 cube/vector 参与者，排除 DMA 和同步。
5. 先求 vector lane 区间并集，与 cube 求交，再对交集求并后计时。直接累加同时执行的
   lane 会重复计数。源码顺序和 DMA 正交集不证明跨项计算重叠。
6. 部署性能另做匹配硬件测量。合法多缓冲仍可能被 wait 串行化；更多重叠也可能伴随
   可删除流量或较差端到端性能。

一个 CVC 排程在源码顺序上看起来是提前生产 C1，trace 里真正重叠的却可能是另一对阶段。
所以阶段标签要用 trace 核对，不能从源码读出来。trace 自己取：在
[混合流水 demo](../../../kernels/ascriptor_kernels/tutorials/mixed_pipeline) 里对某个
`cvc_pipeline_*` case 跑 `--launcher pipesim`，再把区间映射回 stage 和 item。单项 case
可以合法地没有交集，并付出额外启动控制成本。

## 迁移到新依赖图

增加或调整 stage、改变 tile 速率、增加 reader 后，重新推导映射、资源顺序、最后读者、
存储与 drain，并保留原公式及精度边界。从最近的[模式](patterns.md)起步，再测试至少一组
未照搬该实例的配置。泛化意味着能推导并验证变化后的图，不能以各图都重复同一 delay/depth
作为泛化证据。

增加晚消费者时，参考[残差教学 demo](../../../kernels/ascriptor_kernels/tutorials/late_reader)：
P 在早、晚两个 reader 之间持续保留——激活立刻读 P，残差加法在第二次 matmul 之后再读一次，
槽位生命周期由最后一个 reader 决定。P 的三槽环与其他独立双槽说明了按图推导的生命周期和
不同 drain。要用 `--launcher pipesim` 跑它，不能只跑 `sim`：功能模拟器给每次 launch 独立
存储，会接受真实流水不允许的生命周期。
