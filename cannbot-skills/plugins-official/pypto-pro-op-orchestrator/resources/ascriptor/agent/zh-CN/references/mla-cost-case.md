# MLA：记录调参改变的成本

普通 FP16/BF16 MLA 从
[标准 MLA demo](../../../kernels/ascriptor_kernels/attention/a5_mla_fp16_bf16/metadata.json)
开始。最初四个基础场景分别教授单 query decode、causal prefill、BF16/Dn448，以及双 query BNSD
直接寻址。当前七场景 demo 新增大 batch FP16 decode、长 BF16/BNSD prefill 和大 batch BF16
causal decode。各 case 的精确参数就声明在
[`main.py`](../../../kernels/ascriptor_kernels/attention/a5_mla_fp16_bf16/main.py)
的 `CASES` 列表里，逐项读取；
这些 dtype/layout/shape 组合不代表任意笛卡尔积支持。诊断 P 发布、LOWHALF store 和物理 NZ
pitch 时，使用[窄 tile roundtrip](../../../library/examples/api/cube_vector_roundtrip)。

**本页数字的来源。** demo 不记录任何测量。下文引用的每一项观测都属于被它取代的旧单元，
并且都能从 library 的
retired attention exports
历史记录不属于本源码快照。下文括号中的路径相对于历史
`examples/attention/`。恢复一份记录得到的是原始字节与身份，永远不是当前执行结论。

## 一份完成的性能记录

历史 case 7 记录（`a5_mla_fp16_bf16/study/historical-case7.json`）
拥有下表观测的身份和来源摘要。这些是较早候选的 PyPTO-Pro 硬件观测，与当前教学源码的
验证分别保留。每行均通过了当时记录的精度检查。

| 候选 / 采集范围 | 硬件时延，μs | 结果范围 |
|---|---:|---|
| R8 ND，局部对照 | 63.00 | 局部布局实验的参照 |
| R8 compact NZ，局部对照 | 63.51 | 未观察到端到端收益 |
| R16 compact NZ，局部对照 | 64.03 | 行分组和工作分配也同时改变 |
| R8 ND，全套采集 | 63.38 | 另一次采集，不能替换局部的 63.00 |

发布的 case 7 baseline 为 15.7 μs，其字段与来源保留在历史记录中；发布表允许用 proxy
补齐缺项，未逐项标明是否为同卡 native 实测。这些候选结果均是单次采集，不是同一个
候选的三次独立重复。几项小差异不足以证明统计显著的变慢，也不能定位主要耗时阶段。

- **瓶颈证据：** 有完整 kernel 时延；阶段实测、有效 counters 和主瓶颈仍为 UNKNOWN。
- **假设：** 将 ND 发布改为 compact NZ，可能减少 P 搬运或 packing 成本。正确 LOWHALF
  store 只证明访问范围正确。
- **预期空间：** 在测到改变的工作量及其关键路径占比前保持 UNKNOWN；packing 正确不能
  预测加速。
- **约束：** 保持 score scale、online state 和 P 转换边界、causal/layout 语义、物理
  footprint，以及发布、消费与复用的同步边。
- **结果：** 局部 R8 NZ 没有可观察的端到端收益。撤回“P 发布已经确认为主瓶颈”的说法，
  其成本仍是需要测量的假设。

## 重算工作量

该单 query case 为 `B=1`、`Nq=128`、`SKV=2048`、`Dn=512`、`Dr=64`。一个 MAC 计两个
FLOPs，因此有效 QK 与 PV 合计为 `2 * 128 * 2048 * (512 + 64 + 512) = 570425344`
cube FLOPs。softmax、state 更新、规约与 cast 单独计数。

| 数量 | 本次可确定的事实 / 应继续记录的内容 |
|---|---|
| 独立 row block | R8 为 16，R16 为 8 |
| 活跃 Cube core | 在当时的 28-Cube 目标上分别最多 16 或 8；实际活跃仍需证据 |
| 每核工作 | 从实际 block 分配计算；更少 block 改变可用独立工作量 |
| 补齐后的计算 | 源码三次 matmul 均用 M16；R8 只有 8 行有效，等效 cube FLOPs 为 1140850688，R16 为 570425344。属于源码工作量推导，不是硬件 counters |
| KV 迭代与 state | KV64 使每 row block 有 32 轮：R8 共 512 次 block/tile 迭代，R16 为 256 次，每次处理的行数不同。Vector 工作仍需单独统计 |
| 流量 | 分清唯一输入值、与 K 数值相等但独立分配的 V、实际请求搬运和每核重读 |
| 驻留 | 按物理 pitch 统计同时存活的 L1/UB 角色，包括尚未完成最后读取的 buffer |
| 实测 HBM/L2 字节、阶段时间 | 本次观测中为 UNKNOWN |

R16 使用双方 Vector，同时将独立 block 数减半，不能作为只改变参与者数量的纯对照。
prefill 每核有许多 item 时，复用与排程机会不同；这个 decode 结果不能证明 prefill
布局或流水的收益。

先固定行分组比较 ND/NZ。改变分组时，将 padding、双方有效行、block 数和流量变化一起
报告。按 [Roofline](roofline.md) 和[性能模板](../../templates/performance-analysis.md)
保留源码身份及未决项。model cycles、硬件 μs、requested bytes、实测 HBM/L2 流量分别
记录，独立阶段时间不能直接相加当作融合 kernel 的时延。

当前正确性就是 `python main.py --case <id>` 对所跑 case 打印的结果；demo 本身不持有任何
时延和资格。旧单元的 `validation.json` 与 `performance.json` 作为历史记录从同一份归档恢复，
引用前逐项读取其 backend、硬件、variant、精确 case 和测量协议；历史观察不扩大 demo 实际检查的范围。

## Case 10：填写调参记录

历史 fresh-evaluation 记录（`a5_mla_fp16_bf16/study/historical-fresh-case10.json`）
绑定了以下三份候选的 DSL、生成 tile 和原始报告。这是 2026-09-08 对旧 library/guide
快照的评测，它计时的不是 gallery 今天持有的任何东西。三份候选在相同开发 seed 下均通过当时的
首次/最终精度及单 runtime 检查。每个时延来自一次原 profiler/parser 采集，不是三次独立
采集的中位数。发布的 `baseline_perf_us` 为 223.995 μs，元数据仍保留未逐项标记 proxy
补齐值的可能性。

Case 10 为 FP16/BSND，`B=60, SQ=1, Nq=128, SKV=2048, Dn=512, Dr=64`，noncausal、
Nkv1，共 7,680 个独立输出行。下表 S 为 M64/N256、八个 KV 分区和四行 merge；P64/P32
分别为 M64/N128、M32/N128 preload。三者都没有 query 行 padding，KiB/MiB 按 1024 换算。

| 分派与观测 | S | P64 | P32 |
|---|---:|---:|---:|
| 历史时延，μs | 825.20 | 291.82 | 435.18 |
| 精度 / 单 runtime kernel | PASS / PASS | PASS / PASS | PASS / PASS |
| 发布的 1.0× 目标 | 未达标 | 未达标 | 未达标 |
| Row group / producer item | 120 / 960 | 120 / 120 | 240 / 240 |
| Launch Cube / Vector | 16 / 32 | 28 / 56 | 28 / 56 |
| 分配到 producer 工作的 Cube 核 | 16 | 28 | 28 |
| 最忙 Cube：item / producer 行访问次数 | 60 / 3,840 | 5 / 320 | 9 / 288 |
| 该 Cube 触及的不同逻辑行 | 512 行的部分 KV | 320 个完整行 | 288 个完整行 |
| Merge item / 最忙 Vector item / 输出行 | 1,920 / 60 / 240 | N/A | N/A |

Split 的行访问次数包含同一行的多个 KV 分区，不是新增独立输出行。Producer 按 floor
区间均衡分配，merge 按 Vector index 分配四行组。表中的活跃核由源码分派和 launch 推导，
不代表 occupancy counter。

归因前先记录 buffer。单元格为物理 `shape×slot`；Q/K/P 与最终 output staging 为 FP16。
Score、product、online/merge state 和所有私有 GM partial 均为 FP32。
UB 数量按**每个 Vector**计，L1/L0C 按每个 Cube 计。这些保存
源码的容量合计与生成代码物理地址区间的并集一致，typed alias 不重复算作新分配。

| 分配 | S | P64 | P32 |
|---|---|---|---|
| Q / L1 | 64×512×1 | 64×512×1 | 32×512×1 |
| RopeQ / L1 | 64×64×1 | 64×64×1 | 32×64×1 |
| 复用为 V 的 K / L1 | 256×512×1 | 128×512×2 | 128×512×2 |
| RopeK / L1 | 256×64×1 | 128×64×2 | 128×64×2 |
| P / L1 | 64×256×1 | 64×128×2 | 32×128×2 |
| Score / L0C | 64×256×1 | 64×128×2 | 32×128×2 |
| Product / L0C | 64×512×1 | 64×512×1 | 32×512×1 |
| Score / UB | 32×256×1 | 32×128×2 | 16×128×2 |
| Product / UB | 32×512×1 | 32×512×1 | 16×512×1 |
| Compact-NZ P / UB | 33×256×1 | 33×128×2 | 17×128×2 |
| Max、sum / UB，各自 | 1×64×1 | 1×64×1 | 1×64×1 |
| Rescale / UB | 1×64×1，未使用递推 | 1×64×2 | 1×64×2 |
| Accumulator / UB | 32×512×1，用于 merge | 32×512×1 | 16×512×1 |
| Output staging / UB | 32×512×1 | 32×512×1 | 16×512×1 |
| Merge max/sum/weight / UB，各自 | 8×64×1 | 无 | 无 |
| Partial output / GM | 960×64×512×1 | 无 | 无 |
| Partial max/sum / GM，各自 | 1,920×1×64×1 | 无 | 无 |
| **L1 总量，KiB** | **392** | **392** | **340** |
| **L0C 总量，KiB** | **192** | **192** | **96** |
| **每个 Vector 的 UB 总量，KiB** | **215.25** | **209.5** | **105.5** |
| **私有 GM 总量，MiB** | **120.9375** | **0** | **0** |

Compact P 的每个 NZ 列多一个物理 padding 行。Max/sum/rescale 的 64 元素行中分别有
32/32/16 个 live entry；merge state 每行有四个 live entry。每个候选的 L0A、L0B **各自**
还保留两个 32-KiB shortcut slot，即每个 operand memory 为 64 KiB，typed view 与之
共用地址。保持 K 双 slot、只将 N128 改为 N256，K 本身就占 512 KiB L1；加上
Q/RopeQ/RopeK/P，M64/M32/M16 分别需要 712/644/610 KiB，已经超过 512-KiB L1 容量。
这是按源码推导的容量约束，不是新增 backend bug 结论。

| 请求的重复工作 | S | P64 | P32 |
|---|---:|---:|---:|
| KV 迭代：每 producer / 全局 / 最忙 Cube | 1 / 960 / 60 | 16 / 1,920 / 80 | 16 / 3,840 / 144 |
| 每 batch 跨 row group 的完整 KV 重读 | 2 | 2 | 4 |
| 最忙 Cube 的完整 KV 等效读取 | 7.5 | 5 | 9 |
| Q + RopeQ 请求，MiB | 67.5 | 8.4375 | 8.4375 |
| K + RopeK 请求，MiB | 270 | 270 | 540 |
| 最忙 Cube 的 K + RopeK 请求，MiB | 16.875 | 11.25 | 20.25 |
| Score / product FIX group 发布，各自 | 960 | 1,920 | 3,840 |
| Score / product FIX 字节，MiB | 60 / 120 | 60 / 240 | 60 / 240 |
| P 的 Vector 发布 / MiB | 1,920 / 30 | 3,840 / 30 | 7,680 / 30 |
| Softmax VF 调用 / 行 max-and-sum 更新 | 1,920 / 61,440 | 3,840 / 122,880 | 7,680 / 122,880 |
| 输出递推 VF 调用 / 行更新 | 0 / 0 | 3,840 / 122,880 | 7,680 / 122,880 |
| Partial GM 发布 | 5,760 | 0 | 0 |
| Merge state / product 的 GM load 调用 | 3,840 / 15,360 | 0 / 0 | 0 / 0 |
| Merge partial-row 累加次数 | 61,440 | 0 | 0 |
| Partial output GM 写 / 读，MiB | 120 / 120 | 0 / 0 | 0 / 0 |
| Partial max+sum GM 写 / 读，MiB | 0.9375 / 0.46875 | 0 / 0 | 0 / 0 |
| All-Vector 集体屏障 | 1 | 0 | 0 |
| 最终输出 GM store / MiB | 1,920 / 7.5 | 240 / 7.5 | 480 / 7.5 |

这里的 FIX group 发布是一次源码 Cube 搬运向两个 Vector 分发，不是 vendor 指令计数。
GM 字节表示逻辑请求 payload；较短 UB 目标的 padding footprint 可能更大。对照没有测量
HBM/L2 流量，也没有确认主要硬件瓶颈。

**Split → P64：** 120 个完整行 item 已能让全部 28 个 Cube 核有工作。Preload 删除了
partial GM 存储/merge 和八倍 Q 重读，同时也改变 N、递推、FIX 流量、缓冲与 launch。
这证明的是整体实现收益，不能把全部提升归因于 merge 或 launch 单项。

**P64 → P32：** 原假设为增加并行度或减少行浪费；实际没有 query 行 padding 可删，分配
工作的 Cube 核仍为 28。最忙核的输出行数减少，但 row group 翻倍让全局 KV 请求翻倍。
P/FIX **字节**、总的逐行递推和有效 Cube 工作不变，发布/VF **调用次数**增加。实测退化
否定了这个历史 case 的调参选择，不代表更小 M 普遍无益。已确认瓶颈仍为 UNKNOWN，
下一候选继续填写[同一张前后对照模板](../../templates/performance-analysis.md#fill-before-each-tuning-change)。

## 从开发观测到最终资格

历史优化 study（`a5_mla_fp16_bf16/study/optimization.md`）把所选路线与工作量、
生命周期变化相连。七个完整来源为 7/1/3/19/10/15/18；demo 的四个串行对照和 25 个缩小
model case 不增加 benchmark 来源场景。以下记录分别使用：

| 记录 | 可以确定的范围 |
|---|---|
| 历史四场景性能（`a5_mla_fp16_bf16/study/stage1-performance.json`） | 保留早期源码/runtime 与四个精确场景，不给三个新增场景提供资格。 |
| Fresh 开发观测（`a5_mla_fp16_bf16/study/fresh-observations.json`） | 每次分别绑定源码、编译 runtime、产物和原 parser 采集，PMU 尝试有自己的范围。 |
| 历史性能记录（`a5_mla_fp16_bf16/performance.json`） | 保留当时声明的范围和状态；每个最终时延取相应源码/runtime 下三次独立原 parser 采集的中位数。它声明的是 gallery 已不再持有的那份源码，demo 自己不声明任何范围。 |

2026-09-08 的计时里程碑，记录针对的是当时那份源码：21 次正式 profile 全部通过，七个场景
各自三次采集的 median 均达到固定发布 baseline，七个使用 73510929 的独立 fresh-seed 检查
也全部通过。这七个精确组合共有 28 次已完成的 native 采集——`a5_mla_fp16_bf16/evidence/`
下的 21 份 `fresh-round-<case>-profile-{1,2,3}.json` 和 7 份
`fresh-round-<case>-fresh-seed.json`。完整
导出/隔离检查
（`docs/migration/fragments/mla-fresh-export-20260908.json`，在 library 的
`completed-migration` 归档里）也已通过，针对的是当时已提交的 owner。以上全部绑定该记录
`source_identity` 中的文件哈希，而 demo 不持有其中任何一个；它不为今天能跑的任何东西提供资格。

一次采集中的五个 active task 不等于三次独立采集。逐 case 读取准确性、单 runtime 和
recommendation 状态；最终计时资格按[证据表](../common-language.md#evidence)关联对应采集。未启动的尝试不提供时延，也不能确定所请求的 metric 是否受支持。

M128→paired 对照同时改变编译 runtime、行分块、存储和排程，不能用其时延差单独归因于
L0B 共享。后续 paired→quantum 对照保持编译 runtime 不变，只改变可执行源码中的分区
表达式；其中的单次采集仍不能确立统计意义上的效果。

Pair quantum 论证（`a5_mla_fp16_bf16/study/optimization.md`，"Pair quantum and its exact
bound"）保持原每核 item 上界，因而保持固定 M 的物理行槽位上界；它所论证的分区表达式在
demo 里仍然存在，即 `mla_online_paired` 中的 `ceil_items` 与 `quantum`。有效行、causal
工作和 KV 请求仍须分别枚举。来源 10 的 q=1 保留原区间与计数工作，观测时延不同不能归因于分区工作
减少。没有对应实测场景的 paired 静态成本仍只属于静态证据。

按观测记录读取 counter 定义和 parser 来源。`cube_utilization(%)` 表示任务区间与硬件
容量下的总 Cube 执行周期覆盖，不是 useful-FLOP 或 MAC Roofline 效率。Pipe ratio 可以
重叠且分母不同；本次未独立核对所安装 CANN parser 的字节与公开 parser 源码完全一致。
PMU 采集不进入最终时延 median。
