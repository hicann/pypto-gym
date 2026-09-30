# 精确探索 simulator

允许在新 library checkout 阅读 handler、跟踪状态、临时 logging/assertion，并修复确认的
defect。内部 API 可读可改，只是不作为普通 facade 导出。记录 checkout revision 与实际 import
path，明确实验所用模型。
Functional interpreter 从 Surface IR 求值；pipe simulation 执行 Lowered IR，记录功能访问，
调度 pipe/event 依赖并检测 hazard/deadlock。cycles 使用带版本成本模型。任何一种单独结果
均不能证明 CANN 编译或硅片实测。

先运行生成输入的 [pipe 配方](../runtime-and-maintenance.md)：完整实际文件包含 lowering、
balance/GM 检查、真实输出比较与 trace 保存。随后查 handler 时保留独立 reference 和 reduced case。

1. 保留确定输入、输出初始化和独立 ref，缩到首个错误 op/stage，同时保留触发问题的 tail、
   alias、非有限值和复用条件。
2. 降低问题用 `--after all --explain` dump IR，记录 ID、opcode、location、origin、dtype、
   operand、attribute、scope。对实际 ID 用 `explain --op`；查 emitter/编译器时对应 `// #N`。
3. 按问题定位源码，只跟踪相关 handler/helper：

| 问题 | library 源码 |
|---|---|
| 参数、shape symbol、输出 seeding | `ascriptor/backends/sim/launch.py` |
| Dispatch、memory view、register、mask、actor | `ascriptor/backends/sim/interp.py` |
| DMA bytes、padding、layout | `ascriptor/backends/sim/dma_ops.py` |
| A2 repeat/stride | `ascriptor/backends/sim/vec_ops.py` |
| A5 VF | `ascriptor/backends/sim/interp.py`, `ascriptor/backends/sim/vf_ops.py` |
| Rounding/saturation | `ascriptor/backends/sim/cast_rounding.py`, `cast_saturation.py` |
| Access overlap、token、vector-clock hazard | `ascriptor/backends/sim/pipesim.py` |
| Analytical cycle 参数 | `ascriptor/backends/sim/timing/` |

在 library checkout 可运行的只读定位命令：

```bash
rg -n 'class MemRef|class RegRef|class MaskRef|class Machine|def run_block' ascriptor/backends/sim/interp.py
rg -n 'class Access|class Scheduler|def _check_hazards|def simulate' ascriptor/backends/sim/pipesim.py
```

4. 跟踪一个 op 的前后状态：core/side/sub-block，`MemRef.base/slot/offsets/extents/gm_strides/view_offset`、
   backing allocation、resolved byte index、`RegRef.bytes/dtype/valid`、`MaskRef.bits` 和完整的
   `MaskRef.physical()` 快照，以及 round mode。只看逻辑 mask view 会漏掉 pack/unpack 保留的
   细粒度 bit。
   调度问题记录 `Task.op.id/pipe/accesses/clock`、消费 token 和最后 reader。不要 dump 无关
   完整 tensor 或机器配置。
5. 在首个争议转换加 assertion，runner 放 ignored `tmp/<task>/` 的实际文件，不从 stdin 启动。
   每个 case、stage 和输入调用使用独立 output 目录，防止后一次覆盖前一次证据，并遵守
   进程预算。先捕获 actor 原始异常，再分析 wait。只有剩余假设是执行成本
   而非缺依赖时，timeout retry 才有诊断价值。
6. 记录推导规则、revision、源码位置、精确复现/结果，标为 **model-derived**。硬件重要时由
   主 agent 调度同输入/ref 的最小实验，另记 **silicon-measured**、toolchain 与 domain。
7. 硬件与模型不一致时保留两份证据并缩小 case。比较声明契约、独立 reference、模型和生成
   指令，再修复违背预期契约的所属层。不能靠裁剪访问、
   修改 expected 或 timing cost 让错误 kernel 通过。临时插桩清理或转成有依据的维护诊断，
   最终交付最小修复、回归、defect status 和 fixed version。

后续走维护。探索与正式修复分开；疑似 defect 就可以先记录，不必
等完全确定后才保留证据。

可用以下维护中的案例练习完整流程。它们来自 M10 项目的独立生成输入，不依赖记录数据：

| 案例与首个争议边界 | 最小修复及回归 |
|---|---|
| M10-006：末轴 FP32 slice 应填满一个 `[1,32]` UB row，lowered DMA 却发出 32 次带 padding 的 scalar burst | 先检查 descriptor，再执行。load/store 都应保留单次连续 128-byte burst；以生成的 `arange` 数据比较 Surface 与 Lowered 输出。 |
| M10-006：reshape 后有行距的 GM rectangle 被误报重叠，也可能漏掉后续行 | 检查 resolved offset、shape、stride 和覆盖区间。保留不重叠、真实重叠、stride 空洞、后续行及不同 pitch 的对照。32-byte 冲突粒度不变。 |
| M10-007：双槽 fill/drain 的数值与 pipe 执行通过，静态 checker 却报告 overflow | literal `range(2)` 必须按两次执行，不能套 symbolic 代表次数。保留 zero/reverse range、真实第三 token overflow、empty wait、未完全 drain 以及长循环终止对照。 |
| [M10-008](../../../library/docs/rfc/0005-autosync-on-ir.md#55-why-no-run-ahead-analysis-acknowledgement-cell-or-mirror-is-needed)：key 与延迟 score publication 共用一个 ready event | 该 edge-planner 缺陷属于历史问题；A2/A3 slot session 与 A5 local mutex 已替换旧 planner。score 专属 `SEvent(Pipe.V, Pipe.MTE3)` 与 reuse guard 仍是有效显式协议，项目 pipe suite 仅验收各自范围。 |
| [M10-051](../../../library/docs/api/mask-write-semantics.md)：非活动 predicate bit 和 pack/unpack 与 native 输出不一致 | 同时检查逻辑 lane 和完整 256 个物理 bit。确诊后的 native 契约修正旧模型/reference 公式；保留非零的非活动 destination、两个 pack 半区和 alias 对照。未观察的 producer 细粒度 bit 和 b64 原始字段不在测量范围内。 |
| M10-053：Surface 通过 cell 舍入平方根，Lowered 却暴露未舍入值 | `scalar.sqrt` 应在操作处舍入到声明结果类型。历史 `sqrtf` 链接失败属于 native 指令选择问题。保留动态输入独立 reference；普通 CCE/PTO BF16 转换与 sqrt 已由 M10-055 后续修复，PyPTO 限制仍属上游 gap。 |
| M10-054：native 舍入改变负零 bit，而共享 IR/模型/reference 已一致 | 修复目标 wrapper 对零结果符号的处理。保留非零/NaN bit 和原有 `fmod` 中间步骤；模型通过与修复后的真实 board 验证分别记录。 |
| M10-057：通过最后一个 64-half 行 view 发出的 128-half store 被两个模型静默放过 | 在 `Interp.op_vf_store_cont` 中跟踪 backing allocation、组合后的 origin/offset、distribution 和选中的活动 destination。写入前核验所有绝对地址，实际写入使用同一组地址。保留恰好到边界、LOWHALF、empty mask、稀疏 lane、negative base 和报错前不部分写入的对照；短 view 不提供隐式 mask。 |
| M10-060：新的 bounds guard 发现 V2/V4 packed tail store 从 byte 8,128 写到 8,256，超过 8,192-byte tile | 完整 HiFloat8 ZERO carrier 会 pack 成 128 byte，而行只拥有 64 byte。在两个受影响的 tail helper 中显式使用 HiFloat8 LOWHALF mask，并保留完整 64-byte guard。合法的 128-half state store 与 FP32 pack4 路径不变；原矩阵失败保留。这属于 kernel footprint 修复，与 M040、M059 不同。 |

在包含相应修复的 checkout 和已接受环境中运行：

```bash
python -c 'import ascriptor; print(ascriptor.__version__, ascriptor.__file__)'
python -m pytest -q -n 0 tests/runtime/test_dma_footprints.py tests/passes/test_balance_loops.py
```

第一个文件的回归区分 descriptor、数值和物理
重叠；第二个文件的回归区分 literal token 守恒、
未定界循环与重放终止。验收 installed wheel 时，将选中的测试复制到源码包外的 ignored 目录，
清除 `PYTHONPATH`，用分配的不可变环境执行并记录实际 import origin。源码执行与安装包执行
分别记录，不能互相替代。

诊断时可在 descriptor 或 access interval 被消费前临时加 assertion，记录一个 op 的已解析
状态。先在受影响版本展示反例，再修改所属规则，重跑原始 case 与 negative control。
不能缩小真实物理冲突：两个 core 各写不同的四字节值，仍可能落在同一个 32-byte tracked block。
若根因是 ownership 或 ordering，应修复对应协议。
