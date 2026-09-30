# 访问范围与尾块

<a id="vector-tail"></a>
## 普通 vector 尾块的起步检查

对连续普通 dtype 的逐元素行，先完成下面五步，再打开匹配的 owner 样例：

1. 写出每行 valid 元素、GM stride、UB 物理 pitch 与 allocation/slot 大小。
2. 合成 UB allocation、slot 和 offset 的实际起址，检查指令要求的 32-byte 对齐。
3. 分别推导 load、计算和 store 的活动 lane/字节。按 distribution 选显式 predicate，
   将实际访问范围限制在自身 backing 内；GM 有效范围与 UB 物理范围分别计算。
4. 在首个会消费 padding 的操作之前初始化或屏蔽无效 lane；规约使用对应 identity。
5. 用不同的行值、输出 poison、相邻 guard 和最后一行验证完整写回。随后执行功能与 pipesim 检查。

从[非对齐行](../../../library/examples/api/unaligned_rows)或[mask 样例](../../../library/examples/api/mask_semantics)
选择相同 dtype/distribution 的 case。规约再读[数值模式的规约段](numerical-patterns.md#reason-reduction)。
Packed/HiFloat8、NZ、subview、split、slot 或跨侧协议出现时，继续读下列对应边界；普通尾块不要求通读全部专题。

<a id="specific-boundaries"></a>
## 指令与布局的具体边界

每个操作区分逻辑 live 元素、实际访问字节和 backing allocation。mask/窄 view 不自动缩小
指令 footprint；按 dtype/layout 和指令推导，不明确时查源码模型及生成指令。

A5 FP16 寄存器的无 mask `NORM_B16` store 即使使用 64 元素行 view，也会写 128 个 lane。
只写 64 个 lane 时，应创建 `low = MaskReg(f16, init_mode=MaskType.LOWHALF)`，再显式传给
`reg_to_ub_normal(dst_row, values, low)`。分配范围内的完整 store 仍可能覆盖下一逻辑行；
越过分配末端时则必须在写任何字节前报错。
M057 修复按 distribution
和活动 destination 地址检查实际 allocation，不把 view extent 变成 mask。回归保留相邻
未写字节和最后一个 slot；未测量 domain 的 packed-mask activation 仍按模型约定记录。
[窄 tile roundtrip](../../../library/examples/api/cube_vector_roundtrip)提供 FP16/BF16
64 元素 case、物理地址表和 slot 回绕对照。load 一侧并不对称：`vf.load_cont` 根本没有
predicate，所以要读[按 distribution 的 footprint 表](../../../library/docs/api/registers.md#load-compute-and-store)，
而不是逐个 case 自己的数字。

<a id="packed-writeback"></a>
## 推导 packed 写回

1. 从逻辑值追踪 cast/rounding、carrier lane 放置和 pack/unpack bit order，
   查所用[格式](../../../library/docs/api/formats.md)及其 owner 样例。
2. 先把显式 mask 应用于 carrier lane，再推导 store distribution 的目标字节区间。
   检查每行/slot 的拥有范围和 allocation，包含尾块。
3. 比较原始 packed byte、相邻 guard 和最后一行；相邻 state store 单独检查，
   每个 mask 均按自身 dtype 与 distribution 选择。

M060也说明必须考虑
distribution：完整的 256-byte HiFloat8 ZERO-layout carrier 经 `PACK_B16`（CCE 为
`PK_B16`）写出 128 个 packed byte，而 V2/V4 tail 的单 key probability 行只拥有 64 byte。
显式 HiFloat8 LOWHALF mask 选择前 128 个 carrier byte，得到所需的 64-byte 行。相邻的
128-half state buffer 仍需要合法的完整 256-byte `NORM_B16` store。逐个推导 store 范围，
不能因 tail 修复就把每个 store 都减半。

## 搬运与物理布局边界

Cache coherence 操作有各自的 cache-line 范围，不能从 DMA burst 或 simulator 的访问冲突
block 推断。Scalar GM publication 需要对应版本的指令与独立设备证据，见已审阅的
缓存范围说明（历史记录） (`docs/migration/fragments/docs-closure.vendor-review.md`)。
正数 burst 的访问终点为 `(n_burst - 1) * step + burst_len`；空搬运 footprint 为零，不能
盲套公式。burst 间隙不读取，带 padding 的 destination 可能写入比 source payload 更大的
对齐范围。非空非负 stride view 的元素覆盖为 `1 + sum((span - 1) * stride)`，各维和 parent
storage 分别检查。实际 consumer 决定 stride 是否支持，不能把任意 view 当成 DMA。

NZ/ZZ 要满足物理 panel pitch/alignment，逻辑 `M*N` 不够。view/reinterpret/layout 标记
描述存储，不自行完成打包或数值转换。packed/exotic carrier 写清 bit order 和逻辑/载体范围。
指令要求完整 local tile 时，tail 保持物理形状，valid extent 单独携带。限制 GM 读写，初始化
或 mask 所有后续计算可见 padded lane。Bridge 的 split 属于 ABI：A2-family workspace 固定
物理半块不能单方面改成 valid row 的紧凑分割。

Unaligned store chain 可能把不足整块的内容留在独立 state，直到必需的 final flush。跟踪
cursor mutation、prime/flush 配对及最后 partial block；cursor 前进量不能证明指令访问范围。
按所选 overload 的规则处理，并比较相邻未写字节。

在首个受影响 reduction 前应用 identity。Online softmax 在 max 前 mask key/causal 列，
无效 query 行在 exp 前 mask 或不消费；按契约精度更新 sum 后才 cast probability。全 masked
行行为明确。有限 sentinel 仅在满足 score domain 与语义时合法，不能宣称所有 inf 都不可用，
或默认允许 public output 的 NaN。

A2 `count` 是 counter mode 的总元素数，`count_per_rep` 是每次 repeat 内的 live lane 数，
两者互斥。多 chunk 用 repeat 与 repeat stride。当前 IR 在每个 op 上携带 mode；不能按源码
顺序推测跨分支、零轮循环的运行状态。fp32 每 32-byte block
为 8 lane；group32/stride4 起点是 0、32、64、96，不是 full-row reduction。检查整条指令链的
scratch footprint：scalar 结果的下游仍可能访问完整对齐 vector。保持[精度](precision.md)和
[同步](synchronization.md)。
当前源码入口：`ascriptor/frontend/rules_mem.py`、`ascriptor/frontend/rules_vec.py`、
`ascriptor/backends/sim/dma_ops.py`、
`vec_ops.py`、`interp.py` 的 `MemRef/Machine` 与 `pipesim.py` 的 `Access`。
这些是模型源码入口，不是新硬件测量；有冲突进入[白盒调试](simulator-white-box.md)。
检查最小有效/首个无效范围、单/多 burst、对齐/tail、未写区域 canary 和同 core 重复使用。
不通过裁剪 view 或削弱 guard 隐藏越界。

混合流水应在同一存储层合计常驻权重与分别推导的 slot family，见
[CVC 容量计算](cube-vector-cube.md#具体生命周期表)。权重驻留和加深流水可能竞争同一容量。

## 对齐 UB 指令起址

普通 register↔UB、GM↔UB 指令要求 UB 侧起址为 32-byte 对齐；地址须合成 allocation、
buffer slot、view/reinterpret 与显式 offset 后检查。多行 DMA 的每个物理 UB 行起址也须
对齐，逻辑 valid 列数不能决定行 pitch。GM 侧仍按元素寻址：合法的 GM 16-byte offset
不违反 UB 的对齐规则。

逻辑 payload 与物理块分别记录。每个 UB allocation/slot 拥有 allocator 向上对齐后的
backing。四个 FP32 的 DMA payload 可以用一个完整 32-byte carrier、四个 valid 元素，
但完整 padding footprint 必须位于自身 backing 内，不能借用相邻 allocation 的 padding。
Register distribution 与 predicate 仍决定活动 lane 的数据效果，窄 view 不提供隐式 mask。

已经执行的普通 register/UB 指令即使 mask 全关，也必须满足 base alignment；mask 抑制
对应数据效果，不能豁免起址前提。Scalar `.single()`/`.single_value()` 按元素对齐，显式
unaligned 指令族保留自身的 state、范围和 flush 规则。零 burst DMA，以及零轮 VF 循环等
从未执行的指令不产生访问。这些区别由 [IR 内存合同](../../../library/docs/rfc/0001-ir.md)维护。

历史 M065 adapter 从合法 GM 16-byte offset 向对齐 UB 接收四个 FP32，却生成了
非法的物理类型 `Vec<float, 1, 4>`；carrier 回归保留这一边界。修复在 adapter 的[完整物理块 carrier 与 valid shape](../../../library/docs/rfc/0011-pto-isa-backend.md)，
保留 kernel 的合法地址。更改 kernel 布局前先诊断生成的 carrier；模型通过与 native 编译
通过分别记录。

## 从 view 追到物理 tile

Base alignment 与物理行距是两项独立约束。FP32 UB parent `[16,256]` 的左右列窗口
`[16,128]` 应有以下地址：

| 窗口 | 正确行 0 / 行 1 起址，byte | 若错误物化为紧密 `[16,128]` tile |
|---|---|---|
| 左侧，列 0–127 | 0 / 1,024 | 0 / 512 |
| 右侧，列 128–255 | 512 / 1,536 | 512 / 1,024 |

这些起址都满足 32-byte 对齐，却不能证明紧密 carrier 保留了 parent pitch。按
allocation/slot → 逻辑 view 起点和 stride → backend 物理 tile、valid shape 与 consumer
offset 逐层追踪。Valid shape 描述 live 元素，本身不保留更宽的行距；最终物理 footprint
也必须位于真实 parent allocation 内。

功能/pipe 模型可以保留 lowered view，而 emitter 仍可能丢失行距。FIX 还要追踪显式
`N_dst`：模型写入和 trace 字节区间必须按同一 descriptor，不能代换为 parent 的逻辑 stride。
用行列标号、至少两行、
左右两个 offset、parent padding poison、多次 slot 复用和完整读回暴露映射差异。
[FIX 目标对照](../../../library/examples/api/cube_vector_roundtrip#fix-destinations-retain-the-parent-ub-pitch)
并列保留连续 UB、带行距窗口与独立 UB 三种 case。
[M066](../../../library/examples/api/cube_vector_roundtrip#fix-destinations-retain-the-parent-ub-pitch)拥有 backend 修复及精确
证据。保留已支持的 subview；backend 无法表达时，保留带源位置的明确拒绝及参数，不能
静默改行距，也不能笼统拒绝全部 subview。一个 backend 的拒绝不证明另一个也有同类失败。

L0B 有独立的坐标映射：IR 使用 `[N,K]`，native PyPTO `Right` 使用 `[K,N]`。
Allocation、reinterpret、slot 选择与 slice 必须始终携带这层映射，并且只交换一次；
选择静态 slot 时不能再次交换已经转换过的 shape。Shape 正确也不能证明物理行距正确：
partial-N 窗口跨越多个 K fractal 时，仍保留 parent N pitch 带来的间隙。物化时必须保留
这些地址，否则保留带源位置的 backend 拒绝。UB 的行块规则不能直接套用到 Right 的
fractal 布局。具体可支持的窗口与各 stage 证据见
[owner 映射规则](../../../library/docs/rfc/0011-pto-isa-backend.md)和
M067 对照。

把 slot/window 失败转成 API 规则前，先用现有成对边界检查：它的
源码与运行它的测试，
以及它出自的[roundtrip 样例](../../../library/examples/api/cube_vector_roundtrip)。K256 的 b16 PV
在 `splitn=64/128` 时，每个隐式 L0B slot 分别需要 32/64 KiB，而该 slot 容量为 32 KiB。
前者通过两个模型和三个 emitter，后者被 pipesim、PyPTO 拒绝，CCE/PTO-ISA 仅发射源码；
发射不是合法 runtime 结果。同组 probe 还给出了合法的同字节 UINT8 `[32,64]` → FP16
`[32,32]` L0B reinterpret；分别记录同时 splitn/splitk 的 lowering 拒绝，以及 PyPTO 接受
`[16,64][:,16:32]` L0C strip、对 `[32,64][:16,16:32]` 给出 located gap。逐项读取精确 case
和 stage；这些模型/发射对照不建立 native 资格，也不支持全面禁用 reinterpret/subview。

## 按寄存器分布选择 mask

Predicate 在 store distribution 打包前选择物理 register lane。
[Attention 样例](attention-authoring.md)使用的 FP32→b16 ZERO-layout 转换中，64 个转换值
占用 128 个 b16 register 位置。`reg_to_ub_downsample` 会打包这 64 个值；此时再加 b16
LOWHALF mask 只会留下 32 个。相反，dense 的 64 值 b16 load 接 normal store 时，需要
显式的 64-lane mask。相同逻辑宽度不代表相同的 mask/store。

把转换后的位置、deinterleave/packing、store distribution 和目标 pitch 连成一条地址链。
比较原始 staging 字节与邻接 guard，不能只看可能隐藏丢失或重复 lane 的最终规约结果。
