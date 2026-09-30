# 优化已经正确的 kernel

新上下文先读一次[共同语言](../common-language.md)，实质修改前完成
[preflight](../references/authoring-preflight.md)。搜索前冻结目标、允许改变的维度、比较范围
和停止条件。

先按[硬件优先](../runtime-and-maintenance.md#hardware-first)在目标设备与独立 reference 比较，
有精度/性能问题后才用小 shape、少核模型诊断。PyPTO-Pro 优化保持交付选定的
auto_mutex 模式，用户明确要求 manual 时才改用 manual；不要为收尾自行生成第二模式。
有错误先[调试](debug.md)。声明指标：部署性能使用 board
latency；研究调度可明确使用模型 cycles。功能 interpreter 的 wall time 不是这两者。

## 修改前先分析成本

按[Roofline](../references/roofline.md)填写[性能记录](../../templates/performance-analysis.md)。
记录 cube MAC/FLOPs、独立 vector/转换工作、唯一与请求字节、复用次数、tile/容量、允许
和活跃核数、每核工作、模型资源成本与关键等待。注明频率/带宽来源、缓存范围和缺项。
每次调参前填写模板中前后两列的分派、分配与重复工作表；发射前合计所有 slot，并同时
记录调用次数和处理行数/字节。更小 tile 可能增加调用而不减少算术。
历史结果、开发采集与最终测量分别保留实际身份和证据范围。
仅 attention 任务按[专项专题](../references/attention-authoring.md)选择相关状态和布局实例；
其他任务沿用下述通用分析。

| 调查维度 | 问题 |
|---|---|
| 工作分配 | 独立工作有多少，活跃核数受什么限制，每核剩几项？ |
| 复用 | 哪些输入不随 tile 改变，重读几次，能否驻留？ |
| Tile、容量与 layout | Primitive 粒度、物理 pitch、缓冲版本与常驻数据是否兼容？ |
| 流水与依赖 | 谁在等待，哪些独立工作能提前，哪个 reader 让 slot 退役？ |
| 阶段内部 | 是否在关键路径上，哪些指令、搬运、标量工作能影响总时长？ |

这是分析顺序，不要求每个维度都修改。保留任务限制，单核排程与全设备优化回答不同
问题。混合阶段按[通用流水](../references/pipeline-model.md)和
[CVC 教程](../references/cube-vector-cube.md)推导工作索引、buffer lifetime 和 drain。
遇到 slot/window 拒绝时，先用[成对 API 检查](../../../library/examples/api/cube_vector_roundtrip#slot-and-window-boundary-checks)，
保留精确参数、带源位置的错误及 backend/stage 结果。

1. 记录 source/contract/library revision、设备/backend/toolchain、shape、core 数、warmup、
   repeat、命令与指标。确认 import/产物确实对应待测源码，使用独立 scratch。
2. 统一测量 composition 与相关 stage。模型关键路径/pipe occupancy 只来自同一版本模型，
   cycles 不与 board 微秒或另一版本混合。
3. 每次选一个有根据的变动：删除重复流量、保持片上中间值、选择合法 tile、拆开重叠 buffer
   角色、将独立 load 与明确归属边重叠。layout 标记不会自行完成数据打包。
4. reduction split 在实现前规定 partial layout、merge owner/mechanism、launch、可见性、
   初始化和运算顺序。功能 atomic 结果不证明 board merge 合法。
5. 保持 precision/alias/shape contract；先检查正确性、tail、同 core slot 复用，再测相同
   case。违约或不可比较的“加速”回退。
6. 报告前后分布与实际限制。occupancy/带宽饱和提示瓶颈，不能证明每份流量均必要。

使用单元的 `profile` 入口：先验证正确性，再按声明 warmup/repeat 测同一实现。缺少 board
测量就写未测量，旧“最快”不是当前结果。新优化任务遵循自身明确约定的输入域和比较范围。
稳定发现记录在 canonical kernel 的 support/comparison 元数据；agent 链接该所有者。先修
canonical 单元；存在 library 教学副本时，再按 revision/content digest 导出、同步并验证。

## 解释结果与停止

每轮保留五行：瓶颈证据、假设、预期空间、精度/layout/归属约束、结果及剩余限制。
纯排程对照匹配算术、VF 本体、layout、搬运、tile、核数；驻留、网格和 tiling 实验记录
改变的工作与整体收益。每次实质改动后重算成本图景。

要求计算重叠时使用同核实际 stage/item 区间，排除 DMA/同步，对同时 vector lane 求并。
预取加速不自动证明跨项计算重叠。每核一项展示分配，每核多项才展示反复复用。

保留样本数、warmup、分布、原始记录、缓存条件和参与者分母。预期存在计算却 total cycles
全为零的无效采样不能证明利用率，单个未使用 pipe 为零可以合法。有效时延与不可用计数
分开。模型比率不是 HBM 带宽达成率，未标定多核共享竞争的模型不能预测 board 扩展收益，
驻留与多核收益也不能相乘。

按已约定准出和预算停止，剩余空间另报，不在成功后新增重叠/利用率门槛。跨上下文保留
[检查点](../../templates/workflow-checkpoint.md)。
