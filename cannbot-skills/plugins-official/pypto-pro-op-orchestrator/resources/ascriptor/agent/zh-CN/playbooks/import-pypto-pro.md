# 导入一个 PyPTO Pro kernel

把一个已特化的 Pro 设备 kernel 转成已验证的 `lowered/1`，而不是重新规划它。导入保留类型化操作数、
各参与方内部的顺序、控制流、物理存储、数值模式与同步。它不是普通 lowering：**不得**重新分配内存、
重排同步、重新编号事件或合并指令。契约是
[RFC-0015](../../../library/docs/rfc/0015-pypto-pro-import.md)；已经接纳的范围看生成的
[导入证据索引](../../../library/docs/pypto-pro-import-support.md)，它的方向是 Pro → IR，
基线是反方向的[后端支持](../../../library/docs/pypto-pro-support.md)。没有命令行入口：
`ascriptor dump-ir` 读的是 IR，不是 Pro bundle。

1. 导出一次，到处导入。导出需要装好 Pro 与固定 profile；导入既不需要 Pro 也不需要 torch。
   跨越这条边界的产物就是那份 bundle JSON。

   ```python
   from ascriptor.importers.pypto_pro import dumps, export_kernel
   bundle = export_kernel(pro_jit_kernel, directions={"x": "input", "y": "output"}, block_dim=1)
   ```

   `directions` 是调用方显式标注的，绝不从参数名去猜；只要有一个 `inout` 参数，后面每个 executor
   都被强制 `seed_outputs=True`。bundle 记录了 Pro 源文件名与它的 SHA-256，因此存下来的 export
   始终可以对着它的来源复核。

   `directions` 和 `block_dim` 都是原样写进 bundle 的 `abi`。export 只校验参数名存在，不从
   kernel 体里推导任何东西，也不拿它去核对；没写到的参数记为 `unknown`。所以这两项来自调用方
   怎么启动、怎么使用这个 kernel，而不是来自读它的源码——而且写错了不会被拒绝，只会得到一个
   能干净导入、但含义已经不同的 bundle。

2. 导入之后读账，而不是读打印出来的文本。

   ```python
   from ascriptor.importers.pypto_pro import import_kernel, loads, prepare_import
   entry = import_kernel(loads(text))
   ```

   `entry.ir().attrs["import_ledger"]` 每个源节点一行，带 `target_ids` 与 `translated` 或
   `declaration` 的处置。RFC-0015 要求每一个被接纳的操作都在那里有交代，所以"某一行源码去了哪里"
   要靠这份台账回答，而不是靠 diff 生成的代码。`prepare_import(bundle).operations` 在任何目标被
   构造之前就列出源操作，`ImportPlan.require_converters` 会在源码位置上拒掉没有转换器的那一个。

3. 拒绝是一种结果，不是障碍。`ProImportError` 会点名 Pro 的源码位置，例如
   `pro_p6_vf_memory.py:362:9: VF memory access inside VF control flow needs its own cursor and
   footprint rule`。**不得**为了绕过它去改导出的 bundle、放松 verifier 或近似这个形式。
   应当把它归到它的所属方：

   | 拒绝指向什么 | 归属何处 |
   |---|---|
   | Pro 有、IR 没有对应 opcode 的形式 | 经维护把 owner RFC、注册表、verifier、模型与后端一并扩展 |
   | IR 有 opcode、但没有转换器 | 导入器本身与 RFC-0015 的接纳小节，并补一个 export fixture |
   | 形式真实存在、但硅上语义没实测过 | 保留拒绝；先测量，后接纳 |
   | Pro 自身的缺陷或未文档化行为 | `library/docs/upstream.md` 里的一条 `A5-UP-*` |
   | 我们自己转换错了 | 走[调试](debug.md)路线，记入 `library/docs/defects/` |

4. 按固定顺序验证，并且让各阶段互不替代：功能模型、带 hazard 与死锁报告的流水模型、
   打印出的后端产物，最后才是设备。任何一个阶段都不替另一个阶段作数；涉及正确性、同步或支持范围的
   告警，在[调试](debug.md)路线解决之前，阻塞它所涉及的那条结论。

   ```python
   from ascriptor.backends.sim.pipesim import simulate
   from ascriptor.runtime import compile_kernel

   entry.executor(launcher="sim")(*tensors)
   simulate(entry.ir(), tensors, processes=False, check_gm=True)
   compile_kernel(entry, backend="cce")
   ```

5. 证明存下来的 bundle 仍然就是那份 export。重新导出同一个 Pro kernel，用
   `importers.pypto_pro.session.same_export` 比较。逐字节相等是错的检查，profile 一旦改版就会
   全例失败：记录里的 `producer` 可以是固定 profile 的某个已验证前驱，它在另一个身份下承载同样的内容。

6. 设备验收是原生 Pro kernel 与导入得到的 CCE 产物在同一批生成输入上的对比，在分配到的卡与锁下进行，
   遵循[运行时](../runtime-and-maintenance.md#hardware-first)的硬件优先流程。模型一致不构成设备验收。

接纳新形式需要真实的 Pro export fixture、接纳与拒绝检查，以及重新生成的
[支持索引](../../../library/docs/pypto-pro-import-support.md)。完成这些检查后才能声明支持。
手写的 bundle 不构成任何证据。仍然被拒绝的要连同源码位置与原因一起记录：索引会统计拒绝数，
没有解释的那一条是未完成项，不是已经关闭的边界。
