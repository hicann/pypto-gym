# Kernel 候选的实现前检查

在[共同语言](../common-language.md)和任务 playbook 后、实现或实质改变架构前读取。
复用[编写契约](../../templates/authoring-contract.md)记录一次；普通选择从公式、当前声明
与样例推导，不重复向用户发问卷。

## 修改前必须确定

| 决定 | 具体结果 |
|---|---|
| 可观察语义 | 公式、cast/rounding、规约顺序、alias、非有限值与 shape domain |
| 交付 | 一个 runtime launch 或约定 composition；允许的 host 工作 |
| ABI | Typed GM 维度与 dtype、标量绑定、显式输出分配与返回值 |
| 工作归属 | 可用且允许的核数、分配公式、vector 参与者、每活跃核工作项数 |
| 存储 | 逻辑范围、指令 footprint、物理 pitch/alignment、slot 及逐层容量 |
| 初始化 | 首次累加与后续更新、padding、输出 seeding、原地规则 |
| 精度 | 每个 materialize/有损边界、独立 reference 与比较预算 |
| 同步 | Producer、全部 reader、最后 reader 所在 pipe、发布与复用边 |
| 目标 | 正确性、受限排程研究或开放性能优化 |
| 验证 | 完整工作负载先上板；问题驱动的小 shape/少核诊断、匹配对照与停止条件 |

Runtime ABI 与静态子集由当前 library 所有，不能照搬旧 `shape_bindings`、生成命名或
常量分支限制。使用 specialization、helper、控制流前读
[authoring](../../../library/docs/api/authoring.md)。Facade 中有名字不代表所有形式都支持。

## 由任务触发的读取

先读“首读”，只有右栏特征出现时才展开。无需沿所有链接递归预读；实现中遇到新边界再返回本表。
一般签名查 [reference](../../../library/docs/api/reference.md) 或按名字查 [manifest](../../../library/docs/api/manifest.json)。

| 当前边界 | 首读 | 何时展开 |
|---|---|---|
| 普通 vector 行/tail | [内存短检查](memory-and-tails.md#vector-tail)和一个同 dtype 样例 | Packed、NZ、subview 或 slot 再读具体边界 |
| Register/VF、mask | [Registers API](../../../library/docs/api/registers.md)中所用操作 | 新 distribution 或 mask 状态查对应样例 |
| Reduction/broadcast | [规约段](numerical-patterns.md#reason-reduction)；`cadd` 把结果留在 lane 0，广播回全部 lane 是 `ub_to_reg_single`，见[分布表](../../../library/docs/api/registers.md) | 改累加顺序、敏感 cast 或比较预算时读[精度](precision.md) |
| Packed/cast/rounding | Packed 写回先[推导访问范围](memory-and-tails.md#packed-writeback)；格式查 [Formats API](../../../library/docs/api/formats.md) | 分开推导 carrier 放置、store distribution 和拥有字节；rounding/saturation 查[精度](precision.md) |
| SIMT | [归属检查](patterns.md#simt-start)和 [SIMT API](../../../library/docs/api/simt.md)中所用操作 | 推导参与者/线程归属、atomic 贡献者与会合范围 |
| Sort/topk | [记录检查](patterns.md#sort-start)和 [Sorting API](../../../library/docs/api/sorting.md)中所用操作 | 固定输出顺序/tie、record footprint 与 score–ID 配对 |
| DMA/view/padding | [Storage API](../../../library/docs/api/storage.md)中所用传输 | 非连续/变 pitch 时读物理 view 小节 |
| 按索引/按行访问（动态下标、`Var.GetValueFrom`） | [索引与切片](../../../library/docs/api/storage.md#indexing-and-slicing)与[标量段](../../../library/docs/api/authoring.md#scalar-values-and-memory) | 需要 padding 或钳位的索引要自己定合法但被屏蔽的取值 |
| Cube、bias、MX | [Cube API](../../../library/docs/api/cube.md)中所用族 | 跨侧发布后进入同步路线 |
| Cube 向向量侧排空（`l0c_to_ub`、`ub <<= l0c`） | [设备事实](facts-device.md#排到向量侧这一步没有安全的默认值)与 [Cube API](../../../library/docs/api/cube.md#draining-l0c-into-the-vector-side) 里的该算子 | 顺路转换或 requant 的排空**被迫**用 `SINGLE`；平搬里只有跨 M 的归约才需要它 |
| Slot 复用、跨侧或多 reader | 调用序列读[跨侧交接](cross-side-handoff.md)，生命周期读[同步](synchronization.md)和 [Synchronization API](../../../library/docs/api/synchronization.md) | 重复混合图再读[流水方法](pipeline-model.md) |
| 性能目标 / 多 launch | [Roofline](roofline.md) / [分解](../playbooks/decompose.md) | 按任务目标进入对应 playbook |
| Launch/IR/返回值 | [Execution API](../../../library/docs/api/execution.md) | 身份检查区分 canonical device ID 与 facade 别名 |

## 实现与交接

编码前确定 tiling 与数据流，先构建最小完整路径，再按任务的执行流程验证逐步加入的 stage。
每项操作、cast、buffer、同步边与数据搬运都必须能对应到所需语义、精度边界、物理存储需求
或 producer/consumer 依赖。非显然的选择记入任务契约，照抄已有代码不能作为依据。
修复或优化已有 canonical 单元时，仍以该单元为修改对象。

重复混合流水需推导合法的 multi-buffer/lookahead、各 stage 索引及 drain。用单项检查启动收尾，
让活跃 core 多次周转 slot，并检查有效 tail 和完整输出。算术/布局/搬运不变的排程对照与
改变复用、网格或 tiling 的实验分别记录。按[证据表](../common-language.md#evidence)报告结论。
每个优化结果同时说明约定准出与剩余空间，不追加新的成功门槛。
交接前把版本、已读路径、不变量、失败和下一边界写入 scratch 的[检查点](../../templates/workflow-checkpoint.md)。
