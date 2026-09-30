# 迁移或重构项目

保留算法、精度、ABI 与 launch 拓扑，让隔离出来的 demo 目录自带可运行入口和自己的独立
reference。旧快照只读作为证据，不能继续成为 runtime import、reference provider 或维护入口。
按[源码归属](../references/ownership.md)定位各层事实所有者。

1. 清点各源文件及链接依赖，包括文档、生成器、helper、旧 driver。逐份完整阅读，记录路径、
   完整 SHA-256 与实际 UTC；hashing 不等于 validity。
2. 对照当前源码、规范和测量范围评估每份 claim，分别 retain/adapt/rewrite/retire/regenerate，
   说明理由和目标。退役路线有替代；历史测量保留日期。
3. 恢复独立 full/stage ref 和确定输入，保留修正后的 kernel body、[cast/rounding](../references/precision.md)、in-place
   初始化与 [saved-state](decompose.md)。不能用 interpreter 输出代替缺失 ref。
4. 整理成恰好四个文件的 demo 目录，放在 kernels checkout 的 `ascriptor_kernels/<area>/<name>`
   下：`kernel.py`
   （DSL 及它需要的 host 侧表，不含 torch、不做 launch）、`reference.py`（只用 torch，永不
   import ascriptor，导出 `make_inputs(case)` 与 `reference(inputs)`）、`main.py`（`OpExec`
   入口、精度比较，case 以字面量 `CASES` 列表形式存在）、`metadata.json`（只做导航）。
   把重复 driver 合进这一个 `main.py`，有意义的算法变体与 stage 检查用 `--variant`、
   `--stages` 显式表达，而不是再开一个入口。最小 API 教学样例是另一个去处，也是同样四个文件，
   按 [RFC 0012 第 3 节](../../../library/docs/rfc/0012-product-contracts.md) 规定的样例合同。
5. [验证各 leaf 和 composition](implement-decomposition.md)，然后把该目录单独复制到 scratch，
   在那里用声明依赖运行 `python main.py --list` 与 `python main.py`。library API 样例照同样方式
   拷贝，没有导出步骤。离开原来那个仓就跑不起来的目录不算做完。缺少 case/ref 必须失败；
   预期 tensor 由 case 的 seed 在运行时生成，不作为记录数据随仓发布。
6. 验证目标代码、精确命令、链接和每种语言；完整读源不能关闭目标迁移。新发现资源进入
   pending 队列，报告实际检查和未完项。

Backward demo 的必要 forward preparation 在本目录内：它在自己的 `reference.py` 里准备
saved state，不 import 相邻的 forward 项目。A3 使用共用 A2 家族，并自己取得新的 A3 board
证据。Attention 全变体放在画廊里；按 `metadata.json` 写的用途选 demo，不按"记录范围"——
没有这种记录；任务内的独立副本仍要写明它来自哪个目录。
使用[迁移记录模板](../../templates/migration-record.md)。协作任务的主 agent 统一负责来源快照、
目录切换、公共契约与索引、环境绑定、硬件、release pin 与私有发布。
