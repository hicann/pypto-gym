# 编写 attention kernel

**本页很长，且大部分在讲 MLA。直接取你要的那一节：**

| 你要的 | 章节 |
|---|---|
| online softmax 的递推，以及它各步必须保持的次序 | [固定数值顺序](#固定数值顺序) |
| 从 MLA 合同迁到 MHA 或 GQA 的 head 映射 | [将合同迁移到 MHA 或 GQA](#将合同迁移到-mha-或-gqa) |
| 取不连续的 KV 行 | [按索引选取 KV 行](#按索引选取-kv-行) |
| 把 causal mask 写成 predicate | [推导 causal predicate](#推导-causal-predicate) |
| 哪块 buffer 归谁、用什么布局 | [写清布局和归属](#写清布局和归属) |
| 你的场景属于哪一种已命名调度 | [按场景选择状态和调度](#按场景选择状态和调度) |

递推的实例看[数值模式](numerical-patterns.md)的 softmax 一行：整体的 prefill demo，
以及只有向量侧的那个。

在 [preflight](authoring-preflight.md) 后使用本专题。这里把 attention 的数学与状态连接到
已有[内存](memory-and-tails.md)、[流水](pipeline-model.md)和[成本](roofline.md)规则，
通用方法仍由各专题维护。先读任务自己的精确数学与布局合同。下文实例来自
[标准 MLA demo](../../../kernels/ascriptor_kernels/attention/a5_mla_fp16_bf16/metadata.json)，
每个 case 的精确参数就写在它的
[`main.py`](../../../kernels/ascriptor_kernels/attention/a5_mla_fp16_bf16/main.py)里，
`python main.py --list` 打印 case id 及其选择的排程；迁移到 MHA/GQA 前，
先核对家族迁移表，再决定是否沿用行分组、K 复用或 slot 数量。

| 既有 MLA 教学 case / 来源 | 路线 | 核对重点 |
|---|---|---|
| `decode_single_query` / 7 | `decode_splitkv` | 支付 partial state 和 KV-split merge 成本前，先数独立行工作。 |
| `prefill_causal` / 1 | `prefill_resident` | 跨 query item 复用驻留 K，完整单 KV tile 删除递推。 |
| `prefill_bf16_dn448` / 3 | `prefill_resident` | 保留 BF16 舍入边界，在所选 backend 验证可整除的 feature 分片。 |
| `decode_bnsd_two_queries` / 19 | `decode_preload` | 正确分组物理行，保留延迟 PV 需要的旧 KV/P/rescale 版本。 |
| `decode_large_batch` / 10 | `online_paired` | 行工作已填满 launch 时统计 KV 重读，context 配对不跨 batch/core 边界。 |
| `prefill_bf16_bnsd_long` / 15 | `online_prefetch` | 恢复 BNSD query 位置，用 M128 和分片 PV 处理多个 KV tile。 |
| `decode_bf16_causal_large_batch` / 18 | `online_paired` | 独立保留每个 BF16 context 的 causal 状态，成对分区后重新计算工作量。 |

该 demo 包含最初的 7/1/3/19 和后续的 10/15/18，共七个精确完整场景；其余 29 个
case 是四个串行对照和 25 个缩小的 model case，不增加 benchmark 来源场景。逐项读
`main.py` 里每个 case 的参数：BF16+BNSD 只有一个明确声明的长 prefill case，不代表
任意 shape 支持，demo 没有列出的官方场景也仍在它之外。
[a5_mla](../../../kernels/ascriptor_kernels/attention/a5_mla/metadata.json) 的 E4M3 MLA
decode 保留不同的 16× 输出约定，使用
[语义选择器](patterns.md#选择一个样例)区分。

需要具体成本对照时，[MLA 记录](mla-cost-case.md)保留
[case 10 的 split/复用实验](mla-cost-case.md#case-10填写调参记录)与
[七场景资格边界](mla-cost-case.md#从开发观测到最终资格)。仅当前问题需要对应比较时读取；
case 编号用于定位历史证据，不用于替新合同选择实现。

## 将合同迁移到 MHA 或 GQA

选择行组前先记录 KV-head 映射 `g(hq)`。MLA demo 的 Nkv1 和 `V=K_nope` 是它自身的条件；
标准 MHA 按 head 独立读取传入的 K 和 V，不继承这些条件。

| 决策 | 本 MLA 教学合同 | MHA / GQA 任务需要确定的条件 |
|---|---|---|
| Q-head → KV-head | `g(hq)=0` | MHA 为 `g(hq)=hq`；GQA 按声明的映射。仅声明连续等长分组、且 `G=Hq/Hkv` 为整数时才使用 `floor(hq/G)`。 |
| K/V 数值关系 | 所有合法输入的 V 均等于 K_nope | 新合同没有规定相等时保留两个操作数；某个测试里数值相同不够。 |
| 真实 storage alias | 数值相等不要求共享分配 | 按存储层核对 backing range/index map；GM alias 不代表 L1 也共享，实际 local alias 共享分配及全部读取生命周期。 |
| 可共享 K/V 的 query 行 | 同 batch 共享唯一 KV head，causal 位置仍可不同 | 复用同一个逻辑 KV-head payload 要求相同 `(batch,g(hq))`；显式多 head 打包必须保留各 head 的独立 K/V 段并排除跨 head 贡献。 |
| L1 K 退役 | 包含 QK 和延迟 PV 操作数搬运 | K/V 独立时，K 的最后读取者可以是 QK 的最后一次 operand copy；驻留复用还须包含所有相关 query item。 |
| L1 V 退役 | 使用保留的 K 分配 | 独立 V 保活至最后一次 PV operand copy，包含延迟 PV；slot 数量可以与 K 不同。 |
| 物理寻址 | 使用该 demo 的 Nkv1 offset | 按实际 Q/K/V head 数、序列长度和 stride 计算；地址公式不等于 backend tile 支持证明。 |

对维度为 B、S、H、D 的 tensor，元素 offset 与 stride 为：

| Layout | 元素 offset | Sequence stride / head stride |
|---|---|---|
| BSND | `((b*S+s)*H+h)*D+d` | `H*D` / `D` |
| BNSD | `((b*H+h)*S+s)*D+d` | `D` / `S*D` |

Q 使用 `S=SQ,H=Hq,D=Dqk`；K 使用 `S=SKV,H=Hkv,D=Dqk`；V 使用
`S=SKV,H=Hkv,D=Dv`。Output 保留 Q 的 head/sequence 位置，feature 维为 Dv；乘元素字节
数后才是 byte offset。BSND 中固定 head 的 sequence tile 跨度为 `H*D`；把相邻 head
当作同一个逻辑 KV head 会改变 MHA 数学。显式 head 打包需要逐行证明 QK mask 和 PV
value 归属。GQA 即使 head 分组可共享，也可能需要
strided load。继续按已有的[view→物理 tile 规则](memory-and-tails.md#从-view-追到物理-tile)核验。

按新的 head/item 映射填写下文释放表，再从[存活版本与 credit](pipeline-model.md#推导物理存储和-credit)
推导 slot。独立 K1/V2/P2 是需要证明的候选排程，不能由 MHA 名称直接推出。
[独立 K/V primitive](../../../library/examples/api/independent_kv_slots#independent-last-readers)
提供有界生命周期对照，并分别记录执行阶段。
[MHA 成本记录](mha-cost-case.md#已完成的-case-6-排程对照)包含已完成的源码匹配 phase-order
对照，并分开记录模型区间、native 计时与可复算的历史几何。其中 MHA case 编号和上文
MLA 编号属于不同 benchmark。

<a id="indexed-kv-rows"></a>
## 按索引选取 KV 行

稀疏或 top-k 合同给出的 KV 行是索引，不是窗口。这只改变 tile 的拼装方式，
不改变上面的数学，所以按三个判断读完这条路线就回到这里。它的归属是
[按索引 gather 行的样例单元](../../../library/examples/api/indexed_row_gather)：
一个槽的索引用 `Var.GetValueFrom` 取进 cell，cell 成为 GM 视图的行下标，
一次传输搬一行。

| 已记录的事实 | 它决定什么 |
|---|---|
| 每张表按自己的行宽 gather | 合同把两者分开时总有 `Dv < Dk`，没有一个宽度同时对得上两张表，而用错宽度有两种截然不同的结果。K 与 V 各自声明时会被拒绝（`slice [0:64) outside extent 48`）；value 行存成 `Dk` 宽行的 `Dv` 前缀时，同样的超宽读合法且静默，返回的是下一个字段的列。样例的 `tests/check_row_widths.py` 把两者都记了下来。 |
| gather 的 copy 是无条件的 | 补到 tile 高度的槽同样会被解引用，所以它的索引必须是合法行。一个槽之所以是 padding，是因为 mask 丢掉它的分数，不是因为跳过了 copy。 |
| clamp 过的目标行合法但是错的 | 循环跑的是补齐后的槽数，而目标只有 `live` 行，于是 `min(row, live - 1)` 让地址留在 tensor 内部并覆盖掉最后一个真实行。写出前的守卫要从行数推，绝不从 tensor 声明的高度推。 |

然后是代价：gather 出来的一行就是一个传输描述符，所以 1024 行的 top-k 是 1024 个，
而连续窗口只要一个。在选 tile 高度之前把它算进[传输预算](roofline.md)，
动态下标接受什么形式见[索引与切片规则](../../../library/docs/api/storage.md#indexing-and-slicing)。

**没有单元覆盖的部分。** 上面这段传输是按索引选行的 attention kernel 里今天唯一有归属的部分，
本页不暗示其余部分已被覆盖：右下角对齐的 causal 几何下全被屏蔽的行要输出 0 且不产生 NaN、
GQA 让同一 `(b, n2)` 的多个 query 共用一份 gather 结果、`Dv < Dk` 时 `PV` 乘积的列分块，
以及它们与下面在线状态的组合。每一项都从它自己的合同推导，并记录你确立了什么。
最接近的可运行材料是
[presence-mask demo](../../../kernels/ascriptor_kernels/attention/a5_presence_mask/metadata.json)，
它重建的是稠密谓词，不是搬行。

<a id="causal-mask"></a>
## 推导 causal predicate

先固定合同中的 query/key 位置与对齐方式。逐元素 causal 为
`k_position <= q_position`；右下对齐序列的局部下标满足 `j <= i + SKV - SQ`。
[Block32 demo](../../../kernels/ascriptor_kernels/attention/a2_block32_causal/metadata.json)使用左上对齐坐标，
条件为 `floor(k_position / 32) <= floor(q_position / 32)`，因此 query 0 可见同块内的 key 31。
分块后保留所选 predicate，检查块边界与不等长序列。该 demo 是 A2/A3 demo，自身不给出任何
backend 结论；当前任务按自己的 mask 合同选择。

## 固定数值顺序

普通 MLA 先合并 nope 与 rope 两个乘积，再应用合同中的 scale。
先 mask 再求最大值，无效 P 位置贡献零。一个 Vector register
chunk 不等于一行 softmax，行最大值和行和必须覆盖该行所有 chunk。

分块 KV 的递推顺序应明确写出：

```text
m_new = max(m_old, rowmax(score))
alpha = exp(m_old - m_new)
P = exp(score - m_new)，无效位置为零
l_new = alpha * l_old + sum(P)                 # FP32，在 P cast 前
O_new = alpha * O_old + cast_input_dtype(P) @ V # FP32 累加与状态
out = cast_output_dtype(O_final / l_final)
```

MLA demo 的 [`kernel.py`](../../../kernels/ascriptor_kernels/attention/a5_mla_fp16_bf16/kernel.py)
中的 `serial_nd` 排程（`make_serial_nd_kernel`）是正确性起点。
单个 KV tile 可以删除旧状态重缩放与重复累加，见同一文件里的
`prefill_resident` 排程（`make_prefill_resident_kernel`），但仍保留未归一化 P
的 cast 和最终除法。把归一化移过低精度 cast 会改变舍入，不能仅凭代数等价就认定通过。
独立 reference、原 benchmark comparator 与匹配候选分块顺序的数值归因模型分别维护。

首 tile 的专用分支可以在读取旧状态前，直接建立 maximum、sum 和 rescale。先证明实际
首写覆盖所有后续读者使用的 lane，再删除冗余初始化调用及闲置 VF。对 64-lane FP32
state，应检查完整 store 的分布与 mask，包含 padding 和仍执行该 VF 的 inactive 参与者。
首个 product 路径需另外检查：要么不读取旧状态，要么已为这些读者合法初始化。把未初始化
值乘零不等于没有读取。物理 Cube 操作数仍需要的 Q/P padding 初始化应保留。

使用非零输入、同核连续单 tile query、state-slot 回绕以及 tail/idle 对照，记录首写与
随后的实际 state load，并检查独立 reference。已有初始化删除对照减少了 VF 工作，
却没有缩短模型总周期；一次 native 开发采集也不能证明可重复加速。分别保留工作量、
模型周期与实测性能。

用整数改 FP32 指数位不能通用地替换 rescale：零或 normal→subnormal 转换不满足该位
等价关系。例如 V=K_nope=0、rope score 变化时，输出仍须为零。当前保持 FP32 状态；
采用其他表示需要独立数值合同及边界对照。

## 写清布局和归属

Query 行起点在 BSND 为 `((b*SQ+i)*Nq+h)*D`，在 BNSD 为
`((b*Nq+h)*SQ+i)*D`。Singleton 轴可能使 view 等价，SQ>1 会让差异可观察；两轴都使用
非均匀值测试。在已声明的 noncausal、Nkv=1 路径中，同 batch 的物理 query 行可连续分组
并共享 K。扩展 causal 时必须恢复每行自己的 query 位置，不能对不同位置套同一个有效
key 数。

本 MLA demo 中 V 与 K_nope 数值相等，不要求两者共享分配，因此按合同复用已经搬入的 K 做 PV 是合法的。
保持输入不可修改、输出 contiguous；除非任务明确允许并计时，否则单 kernel 设计不在
host 侧物化布局转换。

逻辑行数、Cube 物理 M、每个 Vector 拥有的行数分别记录。初始化物理 matmul 会读取的
padding，每个 Vector 仅发布其拥有的行。用
[窄 tile roundtrip](../../../library/examples/api/cube_vector_roundtrip)的地址表和
guard 定位 packing 问题。根据实际[寄存器分布](memory-and-tails.md#按寄存器分布选择-mask)
推导 store；LOWHALF 不是所有 64 值转换的统一修复。
独立于逻辑 payload 和 mask，保持 [UB 指令起址对齐](memory-and-tails.md#对齐-ub-指令起址)。
从 GM 16-byte offset 向对齐 UB 读取少量 state 是合法的，adapter 必须正确表示物理 carrier。
同时[从 view 追到物理 tile](memory-and-tails.md#从-view-追到物理-tile)：两个 score 窗口的
基址可以都对齐，却仍需要保留较宽 parent 的行距。诊断 FIX→UB 时，分别核对逻辑 view、
生成 carrier 与所选 backend 的结果。
显式 QK/PV 操作数的 L0B 窗口也按该通用章节核对 IR→Right 映射与 fractal 行距，
不能直接沿用 UB 行的物理坐标。

Causal 裁剪应使用最大的真实 query 位置，证明省略的 key 对全部参与行都不可见；QK、
softmax、P 发布与 PV 使用同一有效范围，并检查实际 P/V 操作数读取。逻辑 score 变小
不自动意味着物理 carrier 也变小。
[静态容量契约](../../../library/docs/rfc/0013-pypto-native-synchronization.md#static-local-capacity)禁止 adapter
从 raw-byte 存储轴猜测 typed L0 形状；历史 M10-072 曾因此静默截断合法请求。FIX 还必须携带
物理源 pitch：动态 valid M 与动态 `M_src` 是不同要求。受保护的 backend 会拒绝未证明
的 typed 上界或无法表示的动态 FIX pitch。需要时使用受支持的静态分支与正确尺寸的
carrier，再核对合计容量、事件预算和 native 精度。保留 located refusal，不凭模型
通过宣称动态形式已经受支持。

## 按场景选择状态和调度

下文具体实现排程属于该 MLA demo；[MHA 分析](mha-cost-case.md)按自己的 head 映射
和证据范围进行。

按目标 shape 和实际 factory 参数填写[前后调参表](../../templates/performance-analysis.md#fill-before-each-tuning-change)。
添加 KV split 前先数完整行工作，再合计私有 partial 和 merge owner。
[历史 case 10 记录](mla-cost-case.md#case-10填写调参记录)说明小 decode 的 launch/split 数
不能直接迁移到大 batch，也把较小 tile 的容量降低与全部核已有工作时增加的调用/KV 请求
分别列出。

[`kernel.py`](../../../kernels/ascriptor_kernels/attention/a5_mla_fp16_bf16/kernel.py)
中的 `decode_preload` 流水（`_build_decode`）使用 M64/N128，
把 FP32 输出状态保留在 UB，将当前 KV tile 的 QK/softmax 与上一 tile 的 PV/累加分组。
K、P、score 和延迟 rescale 使用双 slot，product 使用单 slot。Running max/sum 跟随 softmax
顺序，输出 accumulator 消费对应旧版本 rescale。按
[通用流水推导](pipeline-model.md#相邻两阶段为一组)处理启动、排空和 credit。
下表里的每条边界，就是消费侧 `free` 必须落在其之后的那个点；落实它的调用见
[跨侧交接](cross-side-handoff.md)。「UB 中的 P」正是本次会话写错的那一条——
槽在 PV matmul 读到之前就被归还了。

| 角色 | 必须保留的释放边界 |
|---|---|
| L1 中的 Q | 复用该 query tile 的所有 KV tile 已完成最后一次 QK 操作数搬运 |
| QK/PV 共用的 K | 延迟 PV 的最后 MTE1 读取，而非 QK 完成 |
| L1 中独立的 K | 最后一次 QK 操作数搬运，包含声明的全部驻留 query 复用 |
| L1 中独立的 V | 最后一次 PV 操作数搬运，包含延迟 PV |
| UB 中的 P | 向 L1 发布该版本的搬运已完成源读取 |
| L1 中的 P | 对应版本已完成 PV 操作数搬运 |
| Rescale factor | 与其匹配的延迟 accumulator 更新已读完 |
| L0C 中的 score/product | FIX 发布完成后才能覆盖 |
| 输出 accumulator | 所属 query item 内按递推顺序更新，最后归一化 |
| Context 共用的显式 L0B | 所有参与 context 对该 fragment 的 MMAD 读取均已完成 |
| 与 output 共用的 UB product | 最终转换与输出 MTE3 读取完成，随后经过 Vector barrier |

[延迟数据与标量元数据一起保留版本](pipeline-model.md#延迟数据与标量元数据一起保留版本)。
Prefill 的 score/PV 生命周期串行时，可以共享一个 L0C 分配；两代 score/PV 同时存活时
需要独立存储。删除不再需要的状态可让更大的 row block 放入片上，但增加 M 也会减少
独立 block 数，必须逐场景重算每个活跃核的工作量。

在断定"这个 row block 放不下"之前，先看 score 的排空用的是哪个 `dual_mode`。
用 `SPLITM` 时，一个 128 行的 L0C tile 落成两个 64 行的 tile、每个 vector sub-block 一个，
所以它占的 UB 是行数暗示的**一半**。用 `SINGLE` 时整块 128 行落进单个 sub-block 的 UB，
按这个口径算出来的预算会把 M128 判成放不下——而它其实放得下；与此同时 vector 吞吐也减半，
所以这两笔损失是**叠加**而不是互换。下文那些 M128 场景的前提就是 split 排空；
什么时候用不了它，见[设备事实](facts-device.md#排到向量侧这一步没有安全的默认值)。
最初两个短 prefill 场景使用完整 N128 KV tile，FP16 为 M128、BF16 为 M64；同 batch 内，该核
多个 query item 共享驻留 KV。Dn448 采用可整除的 feature 分片；如果所选 backend 无法
物化窗口，仅有数学上的 tail 支持仍不够。
按[成对 slot/window 对照](../../../library/examples/api/cube_vector_roundtrip#slot-and-window-boundary-checks)
核对实际边界：K256 的 b16 PV 隐式 L0B 操作数需要 `256 * splitn * 2` 字节。缩小其他
buffer 不能修复超出 L0B slot 的操作数，一个窗口被拒绝也不代表所有 subview 都无效。

同一
[`kernel.py`](../../../kernels/ascriptor_kernels/attention/a5_mla_fp16_bf16/kernel.py)
中的 `online_prefetch` 路线（`_build_online_prefetch`）以 M128/N128 服务来源 15。它先加载 K(t)，再消费 PV(t-1)，随后产生 QK(t)，保留启动与
最终排空。两个 K slot 服务一组 score/P/state context，PV 的两个 256 列分片限制 product
存储。多个 KV tile 仍保留 FP32 递推；只有每个真实行都能看到整个 tile 时，才选择不带
mask 的 softmax。首 tile 可以直接建立分子。

`online_paired` 路线（`_build_online_paired`）以每个 pack
一个或两个 M64/N128 context 服务来源 10 和 18。参与 context 共享 K/RopeK 和显式 L0B
搬运，各自保留 P、score、FP32 state 和 product credit。配对不跨 batch 或 core 边界。
一个 context 先结束 causal prefix 时，在另一方继续执行期间仍持有最终 product credit。
两条新路线都让最终 FP32 UB product 与 b16 output 共用存储；该 alias 将最后读者的生命
周期延长至输出 DMA。

对 `I` 个原始 M64 item、`C` 个 core，paired 路线使用 `T=ceil(I/C)`、`q=2-(T%2)`；
`mla_online_paired` 里的 `ceil_items`、`quantum`、`group_count` 就是这个表达式，
各核区间由它推出。该分区保持每核 item 上界 T，因而保持固定 M64 物理行槽位的上界；
它不保证一般形状的有效行数、causal FLOPs 或时延不变。q=1 时原区间完全不变。
声称负载更均衡或工作更少前，应枚举各核实际 KV prefix 与请求次数。

`decode_splitkv` 排程（`_build_split_decode`）必须声明
partial output/max/sum 的布局、初始化、merge owner 和完成协议。其小 decode 场景采用
M64/N256、八个 KV 分区，16 个 producer item 对应 32 个独立的四行 merge item。
16-Cube/32-Vector launch 与这些工作项数匹配；launch 与设备总核数分别记录。每个分区
只有一个 tile，可直接发布 PV 分子，删除旧状态递推；发布操作让交接的最后读者变为
MTE3，而非 Vector 更新使用的 V endpoint。

所有 Vector（包括 idle owner）都参与 all-Vector 完成屏障，再由 merge 读取私有 workspace；
原子更新或 fence 本身不能建立该 rendezvous。对 partial `(m_j, l_j, O_j)`，按
`m = max_j(m_j)`、`w_j = exp(m_j-m)`、
`out = cast(sum_j(w_j*O_j) / sum_j(w_j*l_j))` 合并。四行 merge 把跨分区的 max 和 sum 读取
批量合并为两条 strided DMA；一个行组不能跨 producer 的 Vector-state 或 M-tile 边界。
重新调参时保持这些归属、最终数值顺序和单 runtime kernel 约定。

## 把观察整理成可复用结果

归因于 overlap 前，先[比较重新分块后的工作量](roofline.md#重新分块后先比较工作量)。
增加 KV tile 会改变递推和 FIX 流量次数；N128 流水与 N256 串行不是纯排程对照。
用实际同核 stage/item 区间验证指定计算阶段的重叠，按[证据表](../common-language.md#evidence)解释结论。

保留错误 layout、遗漏 drain、错误延迟 slot、遗漏输出和输入写入负控。模型小 case 保持
对应实现分支，并覆盖多次回绕、每核多 item、padding 和 idle core。按证据表记录各阶段，保留原 native comparator。Frontend 接受的动态尾窗
仍可能在 backend tile materialization 失败。

MHA 读取
[demo 目录](../../../kernels/ascriptor_kernels/attention/a5_mha_fp16_bf16/metadata.json)
和各 case 在
[`main.py`](../../../kernels/ascriptor_kernels/attention/a5_mha_fp16_bf16/main.py)
中声明的精确参数。公开 FIA
源码的固定版本可提供分块或阶段间距线索，但不能证明已安装 `torch_npu` binary 实际选择
了哪个 kernel；软件包、硬件和实际 dispatch 需另外绑定。demo 不声明任何逐场景的数值、
launch 或性能状态，因此每一项都要自己取得：`python main.py --case <id>` 得到精度检查，
涉及硬件的结论只能在装有该卡的机器上用 `--launcher board` 取得。
[MHA 成本记录](mha-cost-case.md)把实测 FIA 目标与 case 6 排程对照作为历史记录保留，
并给出恢复它们的路径。

MLA demo 同样不记录源码身份、资格和计时。`python main.py` 在功能模拟器下跑每个 case
的精度检查，`--launcher pipesim` 在事件/冒险模型下跑下降后的流水，硬件结果只能用
`--launcher board` 取得。既有单元的 validation、performance 与优化记录属于历史，
可从 library 的
retired attention exports
按 `examples/attention/a5_mla_fp16_bf16/` 恢复；引用前逐项读取其 backend、硬件、variant、
精确 case 和测量协议，不把时延或资格迁移到新 shape 或新设备。
[证据边界](mla-cost-case.md#从开发观测到最终资格)保留 MLA 计时里程碑及其来源，
[历史布局实验](mla-cost-case.md#一份完成的性能记录)说明如何保留一个正确性修复，同时如实记录
没有确立加速或主瓶颈的结果。
