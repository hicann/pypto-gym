# 调试 kernel

新上下文先读[共同语言](../common-language.md)，实现或实质修改 kernel 前完成
[preflight](../references/authoring-preflight.md)。

从实际观察到的结果出发。结果来自板子时，按[硬件优先诊断](../runtime-and-maintenance.md#hardware-first)
排查；**没有板子时**——要验收的交接件、工作机、根本没分配卡——从能看见该症状的那个阶段开始，
hazard 就是 `pipesim`，并在报告里写明设备执行未被证明，而不是把"没跑"当成通过。
运行 sim/pipesim 前，
先缩小 shape 和活跃核数，保留可疑 tail、复用或核间依赖，并说明小 probe 为何仍能触发问题。
保留生成的失败 case 和独立 reference，定位第一个错误边界。simulator 出错也可能是库 defect。
已授权任务允许阅读、修复新 library，包括 simulator 内部实现。

影响正确性、同步或支持范围的 warning 即使在执行成功时也是诊断证据。保留原始消息、源码或
op 位置、版本与触发 case，确定它来自 kernel 错误、不支持的形式还是诊断本身的局限。
修复所属层，或用契约、源码依据和针对性检查说明该警告为何不适用于当前 case。
未解决前，将受影响的验证标为未证实；屏蔽消息或仅获得数值一致的输出不能消除同步或支持警告。

| 症状 | 下一份证据 | 所属层 |
|---|---|---|
| 所有 case 都错 | 公式、typed ABI、transpose、cast 顺序 | 单元 / frontend |
| 只有 tail 错 | 物理访问及首个受 mask 影响的 reduction | 单元 / DMA / vector model |
| 单 tile 对、复用错 | [同步检查](../references/synchronization.md)：最后 reader、slot rotation、event provenance | 单元 / passes |
| 功能对、hazard/deadlock | [同步检查](../references/synchronization.md)：Lowered op ID、event balance、重叠访问 | 先看单元，再看 passes / pipe model |
| 编译 gap | 带位置的 op 和生成产物 | backend / lowering |
| board 不同 | 首个不同 stage 和最小 board 实验 | kernel / backend / model |

在 library checkout 检查 smoke，或替换成实际失败源文件与符号：

```bash
ascriptor dump-ir examples/api/axpb::axpb --after all --explain
```

读出 op ID 后再使用 `ascriptor explain PATH::NAME --op N`；路径和 N 是从 dump
取得的替换值，不是可照抄的参数。生成 CCE 语句中的 `// #N` 对应 IR 来源。
按 [simulator 白盒指南](../references/simulator-white-box.md) 查 handler、跟踪 tensor/register
状态、加断言和临时插桩。模型推断和硅片实测分开记录；数值正确不证明时序，event 平衡也
可能仍有内存 hazard。
遇到 predicate 字段、online MX subnormal、scalar 平方根或 SIMT 零符号问题时，按当前
[精度边界](../references/precision.md)及所属契约、回归和验收记录确定所属层，并保留已支持的 domain。

短行 VF store 先查生成指令的 distribution，以及 `PAT_ALL` 或显式 half predicate，不能仅凭
view shape 判断。M057
中的最后 slot 越界曾被两个模型静默裁剪；修复后模型拒绝活动地址越过 allocation，kernel
仍须选择符合自身写回契约的 mask。低精度数值不同时，分别比较 cast 前数值和声明的数学
目标：M058恢复源码定义的
FP32 目标；另一半看 [V8 P-stage demo](../../../kernels/ascriptor_kernels/attention/a5_v8_p_stage)——
它的 `main.py` 写明该 stage 实测的 native `exp` 一个 ULP 预算，`reference.py` 的
`reference_candidates` 把这个区间变成一组可接受的 HiFloat8 编码，而不是放宽容差。
这些观察都不授权任意修改容差；该预算属于 A5 上的这个 stage，不是通用 native-exp 精度结论。

缩小到一个操作/view、输出或 stage，但保留失败初始化和复用条件。首个边界明确后再加回
下游输出。不要靠重复增大 timeout 或猜 cast 掩盖 race；先看 actor 首个异常，再解释后续 wait。
及时创建 [defect](../../templates/defect.md)，修复最小所属层，规范错误先修规范，添加生成
输入回归并重跑原失败及相邻边界。不能削弱 ref、静默丢弃越界活动访问或改变 timing 常数让错误 kernel
通过。经维护完成 fixed version 和过时 workaround 清理。
