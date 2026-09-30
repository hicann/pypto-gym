# 运行、检查与维护 kernel

写的时候调用什么、脚本怎么上卡，见[开发期怎么跑](references/development-execution.md)；这一页是围绕它们的流程。
用本快照的 [sources.json](../../sources.json) 选择源码。四个 facade 维护编写声明，
`ascriptor.runtime` 负责执行和源码编译。每次 compile/run 显式选择 backend，不修改进程级
全局 target。

| 操作 | 当前入口 | 成功能证明什么 |
|---|---|---|
| 独立期望值 | 目录的 `reference.py`，它不 import facade、编译器、后端或模拟器 | 生成输入与 reference 不变量，不导入模拟器 |
| 功能执行 | `OpExec(entry, launcher="sim")` | 受支持的算术和 Surface 语义 |
| Lowered pipe 执行 | `OpExec(entry, launcher="pipesim")` | 模型下的动态 event/hazard/deadlock 检查；三项任一不过即抛错 |
| 源码生成 | `compile_kernel(entry, backend=..., block_dim=..., bindings=...)` 或 `ascriptor compile` | backend 能表达这份 Lowered IR |
| 本地 CANN 执行 | `OpExec(..., launcher="aclnn")` | 厂商编译与声明的设备执行 |
| CANN 模拟器 | `OpExec(..., launcher="cannsim")` | 厂商模拟执行，单独于宿主 pipesim 记录 |
| 本机的卡 | `OpExec(..., launcher="board")`，在板上运行 | 在这台机器上厂商编译并实际执行 |
| 本机的 PyPTO-Pro | `OpExec(..., launcher="pypto")`，在板上运行 | 生成的 PyPTO-Pro 源码在这台机器的卡上执行 |

`compile_kernel` 没有 `output=`：它返回一个 `Artifacts`，生成的文本在它的 `files` 映射里，
落到哪由调用方决定。下文命令里的 `--output` 是 unit runner 的 CLI 旗标，不是这个 API 的参数。

<a id="hardware-first"></a>
## 完整工作负载先上板，模拟用于诊断

硬件 kernel 任务先在分配的空闲卡上持锁编译、运行全部声明 shape，与独立 reference
比较并测量目标工作负载。静态 IR/访问范围检查可以先做；完整 shape 的 functional
模拟和 pipesim 不作为上板前置步骤。

如果完整候选的一次设备迭代需要数小时，可先用完整 shape 的模型检查。
记录实测单 case 设备耗时和调整顺序的依据。这只调整顺序：最终仍需在分配的
设备上对完整 shape 和独立参考完成验收。

出现精度或性能问题后，再构造能保留可疑原因的小 shape：dtype/cast、layout/tail、
初始化、slot 复用或相关核间交接。用真实脚本、有界 timeout 和最少合法核数运行
sim/pipesim。按这个核数重新构造 kernel/launch，保留必要的 Cube/Vector 参与者、
核间关系及足够的 slot 周转次数，不能只改启动核数而破坏工作分配。
记录原始与诊断的 shape/核数，以及缩小后保留的条件和无法说明的范围。

修复后回到原始完整 shape 和部署核数上板复验。小模型不能证明完整工作负载通过，
也不能提供它的时延或利用率。无可用硬件时明确记录阻断。这里禁止的只有一件事：把完整 shape 的模拟结果当成对方要的硬件
验收，或者用它代替"设备不可用"的如实报告。它不禁止在 `sim` 和 `pipesim` 上跑完整 shape——
那就是[开发期怎么跑](references/development-execution.md)的日常循环，只要几秒，它能证明什么
写在[证据表](common-language.md#evidence)里。有明确范围的 smoke/probe 留给诊断用。用户明确指定的模型研究或 simulator 维护按其独立约定执行。

<a id="where-each-step-runs"></a>
## 每一步在哪台机器上跑

从本仓快照直接完成源码检查、lower 和模型验证。真机检查在持有分配板卡的
机器本地运行，把相同快照的 `library/` 加入 `PYTHONPATH`。本机
`ASCRIPTOR_BOARDS` 条目必须设置 `"local": true`；连接字段会被拒绝。
在同一进程中生成输入和独立参考、执行算子并比较输出。工作目录、锁和产物
都留在本机的隔离任务目录；记录解释器、源码路径、source ID、设备和完整负载。

[执行 API](../../library/docs/api/execution.md#running-the-whole-unit-on-the-device-machine)
给出了本机调用方式。模型结果不能代替设备验收。

<a id="sync-closeout"></a>
## PyPTO-Pro 交付同步政策

新交付默认使用 `sync_mode="auto_mutex"`，从首个候选到最终包都要核对实际导出源码
的 `@pl.jit(auto_mutex=True)`、manifest 与真机结果。底层 `OpExec` 的历史默认仍是
manual，调用者须显式传入交付模式：

```python
delivery = OpExec(kernel, launcher="pypto", sync_mode="auto_mutex")
manual = OpExec(kernel, launcher="pypto", sync_mode="manual")  # 仅用户明确要求时
```

manual 指后端保留显式同步映射，DSL 仍可使用 `auto_sync`；只有用户明确要求时才
生成它。auto_mutex 由 PyPTO 插入本地锁，编译器保留的跨核协议与 barrier 仍有效；
[native 策略](../../library/docs/rfc/0013-pypto-native-synchronization.md)说明它不是每种
别名访问形式都支持。按同一设备、完整 shape 和用户精度/性能准出验证实际 auto_mutex
产物。发射、编译、精度或真机检查失败时保存原因、源码位置与日志，报告交付阻断，
不得自动改用 manual 冒充完成。仅编译成功不足以验收。

基础源码不依赖 tensor 包即可生成支持的源码；数值入口使用声明的 host extras。CANN、
驱动、厂商 PTO/PyPTO 工具另外配置。backend/launcher 组合：两个 owner 的目录都不声明，跑一遍看它怎么说。两种情况下都必须执行所请求的路径或报告缺口。emit 成功不证明厂商编译，PyPTO 的支持范围也不能由历史上的
两项不支持列表推定。

基础源码使用标准库编码 DMA padding 字面量。精确舍入和 NaN 位模式查
[所属约束与回归](../../library/docs/api/storage.md#base-install-padding-literals)。
主机字面量编码、设备运算与 VF 转换分别保持各自的验证范围。

在 `examples/api/axpb` 的一份拷贝里——四个文件，别无其他——以下命令无需 CANN，即可检查同一个独立验证的样例：

```bash
python main.py
python main.py --launcher pipesim
ascriptor compile kernel.py::axpb --backend cce -o tmp/runtime/emit
ascriptor dump-ir kernel.py::axpb --after all --explain
```

`main.py --launcher pipesim` 生成输入、运行规范 lowering pipeline、检查 event balance、启用 GM hazard，
并比较实际返回输出。调查 trace/状态时，把当前 library 的 `examples/api/axpb` 复制到一个原本为空的
scratch 目录，再把下面的[配方](../templates/pipe_axpb.py)放进该目录、保存为 `pipe_axpb.py`：它
import 该目录自己的 `kernel` 模块，所以是放进去而不是放旁边。清除 `PYTHONPATH`，用已验收的
源码环境执行实际文件，不读取 archive 或记录的预期张量。

```bash
python pipe_axpb.py --output tmp/runtime/trace
```

下面的 `PassManager/check_balance/simulate` 是可供调查的源码实现入口，不是 facade export
或 `OpExec` launcher。固定 domain 为一个 A5 core、连续 FP32 `(1,64)` 输入和独立输出。
有界整数值使 `2*x+y` 在 FP32 中精确，NaN output seeding 检测未写 lane；reference 表达式
独立于 kernel 和 simulator。

每个 case 要求空 balance/hazard、无 deadlock、返回值 exact、输入不变，并拒绝两个错误输出。
trace/schedule 在结果断言前保存；若 interpreter 先抛异常，则可能还没有 result，应先调查
首个异常。trace event 的 `time_domain` 是 `cycle`，查看器 display-unit 元数据不表示硅片时间。

`check_balance` 对已解析的 literal loop 使用实际次数；无法解析的 loop 使用有界 `rounds`
（默认四次）。有界探索通过不证明任意 symbolic 迭代次数。pipe scheduler 将同一 lane 的同 pipe
FIFO 视为有序，但这不证明每种设备指令的 writeback/landing 要求都已满足。保留设备需要的
barrier，对争议硬件行为另做验证。后续见 library [同步诊断](../../library/docs/diagnosing-sync.md)。

CLI 显示真实 op ID 和源码来源。先从 dump 选择 ID，再运行
`ascriptor explain kernel.py::axpb --op ID`；其中 `ID` 要替换，不能当成固定示例编号。
生成语句的 `// #N` 将编译器诊断与 Lowered IR 联系起来。需要编译期特化的 backend 显式
传入 bindings；typed GM 的 shape symbol 确定运行时维度，不靠标量值猜测。

运行时在 `out_dir` 下组织生成工程、harness 与数据，并用源码 hash 决定是否重编译。
实际输出取 launcher 返回值，带初值输出必须声明 seeding。本地运行目录使用独立 output identity。
scratch 中的输入输出文件是本地临时产物，不是发布所依赖的记录数据集。

硬件访问来自私有配置，不能放入公开文字或源码。操作员在启动前用 `npu-smi` 检查选定
设备。运行时 advisory `flock` 协调参与此协议的任务，不执行旧的重复空闲采样协议。
使用指定环境、单个分配的可见设备和自己的 workspace。超时需要诊断，重试前先看首次
build/run 失败；旧脚本的保护 override flags 不是当前产品支持的配置方式。

修复编译器或模拟器前，先读所属 library RFC；契约确实有误时，在同一变更里先修订 RFC。
按争议行为定位所有者：

| 行为 | 当前源码所有者 |
|---|---|
| facade 名称、签名、DSL lowering | `ascriptor/a*.py`、`frontend/dsl.py`、`frontend/rules_*.py` |
| 类型、操作数、访问与验证 | `ascriptor/ir/` |
| 依赖、event、layout、资源分配 | `ascriptor/passes/` |
| CCE 打印与硬件 wrapper | `ascriptor/backends/cce/` |
| PTO ISA / PyPTO 打印 | `ascriptor/backends/pto_isa/`、`ascriptor/backends/pypto_pro/` |
| 运行时绑定、工程、harness、launcher | `ascriptor/runtime/` |
| 数值、物理访问和调度模型 | [模拟器白盒所有者](references/simulator-white-box.md) |

新操作应在 registry、前端规则、适用 lowering、模型语义和 backend 中如实描述类型、
读写效果、scope 与设备能力，不能增加手写旧 autosync session 或 inline 后门。pass 使用
`Rewriter` 保留来源并解释决策；无法打印时报告带源码位置的操作。先用生成的正例和负例
验证修改的语义，再运行受影响的整合 gate。

library 维护小型编译器/运行时回归与 defect，kernels 维护完整算法及案例，agent 维护流程
与双语路线。历史测试数不能证明新仓 gate。精确契约可以采用 exact/bitwise 比较；非零
预算需要精度依据和错误输出对照。详细协议见[精度](references/precision.md)与
维护。
