# 归属、事件与复用

`auto_sync()` 插入同侧依赖；跨 cube/vector 要使用该设备接受的显式协议。A5 片上 handoff
和 A2-family workspace bridge 能力不同，精确签名从公开声明查，不跨 facade 照搬。
该协议的四个调用、以及消费者在其中的位置，见[跨侧交接](cross-side-handoff.md)；
本页讲的是同一个 mutex 的生命周期与声明规则。
各复用 slot 写明 producer、首/最后 consumer、复用点和真正让资源退役的 pipe。正向 publish
只保证 producer→consumer；覆盖仍被 reader 使用的 slot 前可能另需 loop-carried 依赖。

**一次 `set()` 只覆盖它之前发出的生产者工作。** 守着两块缓冲的事件必须在两块都写完之后 set，
不能夹在中间：夹在中间时握手看上去仍然完整——一个 set、一个 wait、balance 正确、pipe 也对
——但消费者只被排在第一次搬运之后。功能模拟看不见它，数值甚至可以逐位精确；只有 hazard
列表会报。

`Tensor/DBuff/TBuff/QBuff` 分别提供 1/2/3/4 个 local slot。从调度求最大同时存活角色数。
重叠 beat 数乘同时 live role 数在全部重叠时是保守界，不是万能下界；证明生命周期不相交后
才能共用 slot。event/mutex credit 不会分配存储；rotation 与 credit 相符，不同 lifetime
family 使用不同 counter。

现在 mutex **必须自己说清楚**：`depth=` 与 `guards=` 二者必居其一，且都是 keyword-only
——因为信用数属于**被交接的那块缓冲**，而声明处此前根本没提到过任何缓冲，对一块未知缓冲
不存在正确的默认值。优先写 `guards=<那块缓冲>`，信用数直接从它的槽位数读出，不会与它脱节；
只有当该 mutex 每轮转要跑多次时才另外写上 `depth=`，而信用数超过槽位数时会有 lint 点名。
信用多于槽位意味着 producer 可以抢回 consumer 仍在读的槽——**结果是错的数，不是 hang**。

这件事现在**对每个 kernel 都查**，不再只查含有 `auto_sync` 区域的那些。
这个检查此前住在"插核内 event"的那个 pass 里，而那个 pass 在函数里一个区域都没有时会直接返回
——于是**全程手写同步的 kernel，也就是作者亲自在跑 mutex 协议的那一类，反而一点都没被查**。
现在它属于 `crosssync` pass：信用数超过一次 hand-back 相隔的周期数是 **error**，
未被覆盖的跨侧边是 **warning**，而 `autosync_cross_side=off` 两者都关
——想让 pipe 模型把那个 race 演给你看时用它。

A2 族上，`auto_sync` 把同侧交接当作**一套协议**规划，而不是每条依赖一套：成对的
`ready` / `valid` **slot session**——窗口在生产 pipe 开始写处打开，在消费 pipe 最后一次读之后收口，
信用数就是该窗口轮转的物理槽数（受 `sync_depth` 约束），见
[RFC-0005 §5](../../../library/docs/rfc/0005-autosync-on-ir.md#5-as-implemented-a2a3-slot-sessions-2026-09-16)。
同一 pipe 的连续操作合并进已打开的窗口，因此内核花掉的 flag 跟随角色切换，而不是依赖边条数。
这里不再需要 lease、fragment 枚举或手工抬高深度：旧 planner 必须为共享 L0A/L0B scratch
单独证明的那套协议，现在是每次交接的默认形态，先前的 lease 方案随之删除。作者仍然自己负责的部分不变
——跨侧归属、GM/workspace 协议、同一 pipe 上重叠的写，以及作为生产者的标量 pipe（它无法 set flag）。

两个习惯依然有回报：按调度**实际轮转的槽数**声明缓冲，因为这个数就是信用数；每个生命周期 family
用各自的 counter，因为 session 从该 counter 读轮转。遇到无法规划的形态——消费者跑在同一窗口的
发布之前、窗口打开期间的 `break`、某成员在两个 pipe 之间双向流动——pass 会带上两个操作的位置报错，
修法是显式事件对，或让生产者与消费者落在同一个窗口里。

还要核对 planner 实际读出的轮转：在 A2 家族上，信用数就是从它数出来的，读不出来的 session 退到
1 信用。`cf.for` 归纳变量不会被当作槽位索引读取（`765e4ec` 上它也保守回退为一轮距离）；
套一层只读 `Var` 会被折叠掉。已有计数器对照 (`docs/migration/fragments/for-iv-slot-distance-20260909/README.md`)
在循环外初始化显式 Cell、每轮末无条件递增一次，使现有分析识别两槽轮转，数学与存储保持
不变。这是有范围的源码变通；不能据此手工提高 event 深度，也不能仅凭多缓冲声明假定已形成重叠。

当 wait 跨过无关工作时，检查 IR 中保留下来的 predicate、Cell 写入和 canonical allocation
root。M10-068说明
独立分配可能被跨分支保守合并；拆分仍要通过 alias 与事件预算证明。Descriptor 复制还须
在 native lowering 中保留赋值时的旧值。[M10-070](../../../library/docs/upstream.md#a5-up-036)
记录了 Surface/Lowered 正确、native loop backedge 却丢失 snapshot 的问题。已记录的
kernel 整数复制适配有明确范围，不能据此使用 float identity 算术，或认定所有上游 scalar
形式已经正确。

为延迟 stage `d` 明写消费 work item 与 warmup/drain guard。提前一轮 producer 通常需要
最终 drain。下游最后 reader 可能延长原 source lifetime，不能在中间 stage 完成时提前 free。
VF local load/store 也可能需要 `vf_barrier`，kernel 事件未必覆盖 VF 全部内存次序。
带 guard 的 carried event 要区分上一轮仍 pending 的 token 与当前 producer 条件。当前
producer 跳过时仍可能需要 wait；首轮可能无需 wait，drain 应恰好消费剩余状态。检查零轮、
一轮、多轮和 producer 交替跳过的路径。声明 event depth 本身不能证明最大 outstanding 数。
按[证据表](../common-language.md#evidence)核对同步结论，检查 op provenance 和物理访问。明确最后 reader 后才缩短临界区。未解释 warning 保持 open；不以固定重试数
阻止已授权修复。

按实际 set 完成和配对 wait 开始的时间检查 token 生命周期，而非模拟器构建队列的顺序。
位于未来的 wait 尚未释放物理 flag。M10-069
按参与者、有向 pipe channel 和分配的 ID 检查重叠生命周期，包含不同 event 名复用同一 ID。
计数平衡和没有 memory hazard 不能替代该检查；缺少物理绑定仍属于证据不完整。另一个历史
[共享 publication 分析](../../../library/docs/rfc/0005-autosync-on-ir.md#55-why-no-run-ahead-analysis-acknowledgement-cell-or-mirror-is-needed)已随 edge planner 退役；timeline checker 仅验收自身模型范围。

编辑 Lowered IR 或 emitted artifact 得到的是诊断对照。正常源码编译器需独立验证，保留
其 runtime/artifact 身份，并检查实际生成的 native 代码。发射字节相同可以证明观察与源码
的对应关系，不能把旧硬件采集改标为新运行。归因于删除依赖前，比较实际 stage 区间的
union 和完整输出。

至少让同一 active core 复用 slot 多于一次。M-tiled attention 可用
`ceil(BH * ceil(S1 / TILE_M) / active_core_count) > 1`，递归项目按实际 chunk 分配公式或支持的
单 core 运行。每 core 仅一项的 multicore smoke 不能证明复用。
**确认缺陷属于 library 之后**，所有者是 `ascriptor/passes/`、
`ascriptor/frontend/rules_sync.py`、`ascriptor/backends/sim/pipesim.py`；详细诊断与模型契约在
library `docs/diagnosing-sync.md` 及 `docs/rfc/0006-lowering-pipeline.md`。手写事件不在这份
名单上：自己声明 `SEvent` 的 kernel 自己负责它们的位置，报出来的 hazard 在 lowered IR 证明
不是之前，都算 kernel 的。缺边或模型错误进入[调试](../playbooks/debug.md)。

完整推导例子见 [CVC 生命周期表](cube-vector-cube.md#具体生命周期表)。其他阶段序列按
[通用流水](pipeline-model.md)，变更图后重算全部工作映射、最后 reader、容量和 drain。
