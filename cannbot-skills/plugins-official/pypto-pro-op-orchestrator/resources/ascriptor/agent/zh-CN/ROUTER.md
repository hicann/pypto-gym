# 选择一项任务

使用本快照的 [sources.json](../../sources.json) 和
[sources-index.json](../../sources-index.json) 确定完整源码身份。Kernel 编写、实质修改、
分解、调试或优化先按 [AGENTS.md](../AGENTS.md) 读一次[浓缩起点](../context/kernel-authoring.zh-CN.md)，
其中的共同语言摘要完成本轮起步；若缺失或过期，改读[共同语言](common-language.md)。
然后选一条 playbook。实现前完成[preflight](references/authoring-preflight.md)，按触发条件读专题和一个可运行样例。
首次编写还读[执行模型](concepts.md)。选择一种语言即可，不预读 RFC 全集或整个画廊。
纯文档维护可直接进入维护路线。

查询操作与签名从 [API 总入口](../../library/docs/api/README.md)开始。

**找样例只有一条路，起点是一个判断。** primitive 去 [API 样例目录](../../library/examples/api/README.md)；完整算法去 [kernel 画廊](../../kernels/README.md)，按 `topology`、`device`，或 `formula` 与 `tags` 里的词过滤它的 `index.json`。两边都一样：候选自己的 `study_for` 和 `do_not_copy_when` 写明它值得学什么、什么时候不要照抄——先读这两项再打开源码，并且只读一条相关条目，不要预读全集。手里只有一个词组或刚写下的符号时，在 agent checkout 里运行 `python tools/select_example.py --query '<词组>' --device a5 --language zh-CN` 会替你走同一条路，经由[主题索引](../index/README.md)。`--device` 写你要写的那个系列：默认是 `a5`，声明了别的系列的 demo 只会列在 `rejected` 里。下面那些会给你样例的页面都是指回这里，不是另开一扇门。

| 任务 | 路线 | 交付 |
|---|---|---|
| 从公式、reference 或模型编写一个 kernel | [编写](playbooks/author.md) | 约定 launch 与独立 reference；目标是 PyPTO-Pro 时另读下表的专题，交付形态是[交付区](references/pypto-pro.md#delivery-area) |
| 规划多个 runtime kernel | [分解](playbooks/decompose.md) | 可执行 reference DAG 与 stage ABI |
| 实现已有分解 | [按分解实现](playbooks/implement-decomposition.md) | 分别验证 leaf 与 composition |
| 排查错误或 hazard——或判断别人交过来的 kernel 是不是做完了 | [调试](playbooks/debug.md) | 最小复现、原因与回归；交接件则是每个阶段分别给出的结论，以及还没有被证明的部分 |
| 改进已经正确的 kernel | [优化](playbooks/optimize.md) | 相同契约下可比较的测量 |
| 修复 library 或指南 | 维护 | 所属层修复与 defect 更新 |
| 迁移或重构项目 | [迁移](playbooks/migrate.md) | 逐文件去留与隔离单元 |
| 把已有 PyPTO Pro kernel 转成 IR | [导入](playbooks/import-pypto-pro.md) | 已验证的 Lowered IR、可复核的 export 与定位到源码的拒绝 |
| 把厂商 AscendC 算子重新实现到另一个设备族 | [移植](playbooks/port-vendor-operator.md) | 从三个上游来源还原的语义，以及一个自带证据的重新推导的 kernel |

新建任务若只要求将官方 AscendC 源用例迁移到 A5 PyPTO-Pro 并验收功能/精度，使用
[移植的源用例交接](playbooks/port-vendor-operator.md#source-only-handoff)。已有 formal Scriptor
任务和完整流程请求仍走原入口；源用例交接不调用要求冻结设计的 `pypto-pro-op-develop`。

A2/A3 CCE 使用 `ascriptor.a2` 和 `ascriptor.a3`，先读
[A2/A3 编写入口](references/a2-a3.md)，使用其 tensor-vector 词汇。A5 提供
register/VF 与 SIMT 接口；具体后端和设备支持需按当前任务验证。

按需读取：[设备事实与由它推出的规则](references/facts-device.md)、
[契约](references/authoring-contract.md)、[内存与尾块](references/memory-and-tails.md#vector-tail)、
[同步](references/synchronization.md)、[精度](references/precision.md)、
[simulator 白盒](references/simulator-white-box.md)、[归属与源码](references/ownership.md)、
[模式](references/patterns.md)、[术语](references/terms.md)。
| 当前问题 | 直接证据路径 |
|---|---|
| **怎么把这个 kernel 跑起来？上板怎么跑？** | [开发期怎么跑](references/development-execution.md) —— 先读这一条，它只有一页 |
| **数值是精确的，那是不是就做完了？** | 不是：[证据表](common-language.md#evidence) 写明每个阶段能证明什么，算术正确只是其中一行 |
| **目标是 PyPTO-Pro，有什么不同？** | [目标 PyPTO-Pro](references/pypto-pro.md) |
| Attention 数学、布局和在线状态如何组合？ | [Attention 编写专题](references/attention-authoring.md)，再读一个精确 canonical case。这是一篇绑定 MLA demo 的长文；如果你要的只是 online softmax 的递推和它的次序约束，直接去 [固定数值顺序](references/attention-authoring.md#固定数值顺序)，例子看 [数值模式](references/numerical-patterns.md) 的 softmax 一行 |
| 时间花在哪里？ | [Roofline 与成本模板](references/roofline.md) |
| Cube 算完的结果怎么交到向量侧？ | [设备事实](references/facts-device.md#排到向量侧这一步没有安全的默认值)，再读 [Cube API](../../library/docs/api/cube.md#draining-l0c-into-the-vector-side) |
| 哪几个调用把一块缓冲交到另一侧？ | [跨侧交接](references/cross-side-handoff.md)，再读[跨侧归属](../../library/docs/api/synchronization.md#cross-side-ownership) |
| 搬入很忙，是否有重复工作？ | [流量与复用](references/roofline.md#cvc-演算删除工作与改善排程) |
| 重复 CVC/VCV/CVCV/VCVC 如何重叠？ | [通用流水方法](references/pipeline-model.md)，再读 [CVC 推导](references/cube-vector-cube.md) |
| Slot 何时可以覆盖？ | [同步](references/synchronization.md)与[具体生命周期](references/cube-vector-cube.md#具体生命周期表) |
| 哪个 primitive/样例适合此边界？ | [数值模式](references/numerical-patterns.md)给出模式名；[样例选择器](references/patterns.md#选择一个样例)就是上面那条找样例的路，不是第二条 |

[English entry](../en/ROUTER.md)
