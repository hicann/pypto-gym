# MHA：区分重建成本与实测性能

先完成 [MLA → MHA/GQA 合同迁移](attention-authoring.md#将合同迁移到-mha-或-gqa)，继续使用
现有[性能记录](../../templates/performance-analysis.md)。下文 MHA case 编号属于另一个
benchmark，不等于同编号的 MLA case。曾经实测过的完整 kernel 性能范围恰好为：

| MHA 来源 / 声明 case | Dtype / layout | B / SQ / SKV / H / D | Causal | 官方 FIA 实测 median，µs |
|---|---|---|---|---|
| 6 / `cross_bf16_short_kv` | BF16 / BSND | 2 / 512 / 128 / 16 / 128 | 否 | `18.070` |
| 7 / `prefill_fp16_causal_long` | FP16 / BSND | 4 / 1024 / 1024 / 32 / 128 | 是 | `159.360` |
| 14 / `decode_fp16_large_batch` | FP16 / BSND | 128 / 1 / 128 / 32 / 128 | 否 | `180.910` |

上表三行 shape 是现存的：它们就是
[MHA demo 的 `main.py`](../../../kernels/ascriptor_kernels/attention/a5_mha_fp16_bf16/main.py)
中 `CASES` 的 `full_shape` 条目，`python main.py --list` 会列出它们。µs 那一列不是：
demo 不记录任何测量。

**本页 µs 的来源。** 下文每个实测数字都属于被该 demo 取代的旧单元，都能从 library 的
retired attention exports
完整恢复——历史记录不在本源码快照内。括号中的路径相对于历史
`examples/attention/`。恢复一份记录得到的是原始字节与身份，永远不是当前执行结论。

维护者在
[D-258](../../../library/docs/decisions.md) 中选择了官方 FIA 实测目标（`a5_mha_fp16_bf16/fia-targets.json`），
各目标来自同卡、同测量协议的三次独立 FIA 采集中位数。用户随后接受当时冻结的实现收尾，
并结束继续优化。那份历史性能记录（`a5_mha_fp16_bf16/performance.json`）
分别保留每例实测 median、严格 FIA 比较和用户接受状态；
`user_accepted_current_version` 不表示胜过 FIA。只有全部所选场景严格更快时才作性能推荐；
否则发布的 selection 仅覆盖已验证正确性。三例均独立传入 Q/K/V，head 映射为恒等映射。
保留三个原 parser 观测、median 和 min/max，不通过舍入把持平或较慢结果变为通过。
下文历史 case6 排程对照保留原 source/runtime，与最终共同单元资格分开记录。
下文历史 case 15 只是算术反例，不属于完整 kernel 性能范围。
已发布的 baseline 元数据及早期比较保留各自原始身份；Pipe 插桩耗时不计入时延目标。

## 历史证据允许说明什么

维护者提供的 2026-09-08 MHA xhigh 改进建议记录了两组几何与成本数字。历史 R7 源码、
原始 `final-source-costs.json` 以及 validation/profile 报告已被外部清理，目前不可用。
本页没有检查或重跑这些文件，只按建议中的 shape、tiling 和 causal 规则复算。数字与
建议表一致，不代表缺失实现已经验证。

假设普通 MHA 的每个 Q head 对应一个独立 K/V head，K/V 使用不同存储，输入输出元素为
两字节，QK/value 的维度都为 D。每个 head/query tile 对每个访问的完整 KV tile 各加载
一次 K 和 V，只省略完全不可见的 causal 尾部 tile。这些条件重建的是请求 payload 模型，
不是实际 HBM/L2 transaction。

| 复算数量 | 历史 MHA 7 | 历史 MHA 15 |
|---|---:|---:|
| B / SQ / SKV / H / D | 4 / 1024 / 1024 / 32 / 128 | 60 / 1 / 512 / 16 / 256 |
| 逻辑 M / 物理 M / KV N | 128 / 128 / 128 | 1 / 16 / 256 |
| 独立 head sequence，`B*H` | 128 | 960 |
| 逻辑输出行，`B*H*SQ` | 131,072 | 960 |
| 每 head 的 query tile | 8 | 1 |
| 全局 KV tile 访问次数 | 4,608 | 1,920 |
| 唯一 K+V 字节 | 67,108,864 | 503,316,480 |
| 按所述加载规则推导的 K+V payload 字节 | 301,989,888 | 503,316,480 |
| 唯一 Q+output 字节 | 67,108,864 | 983,040 |
| 未被 mask 的 pair 上的 QK+PV FLOPs | 34,393,292,800 | 503,316,480 |
| 整个物理 tile 隐含的 QK+PV FLOPs | 38,654,705,664 | 8,053,063,680 |
| 实测 HBM/L2 字节 / Vector pipe utilization | UNKNOWN | UNKNOWN |

一个 MAC 计两个 FLOPs，因此这里每个 query/key pair 的 QK+PV 工作为 `4*D`。
右下对齐 causal 下，query i 有 `v_i=min(SKV,max(0,SKV-SQ+i+1))` 个可见 key；
noncausal 行的全部 key 均可见。有效 FLOPs 为 `4*D*B*H*sum_i(v_i)`。

Case 7 每个 head 有八个 M128 query tile，对应 KV-prefix 长度 1 到 8，故访问次数为
`4*32*(1+2+...+8)=4608`。每 head 的精确 causal pair 数为 `1024*1025/2`，
有效 FLOPs 为 `4*128*(4*32)*(1024*1025/2)`；整 tile FLOPs 为
`4*128*128*128*4608`。唯一 KV 字节为 `2*4*32*1024*128*2`，推导的 KV 请求为
`4608*2*128*128*2` 字节。

Case 15 的 `60*16` 个 head 各有一个 query、两个 N256 KV tile，因此共
`60*16*2=1920` 次访问。SQ1 下，右下对齐 causal 和 noncausal 都允许全部 512 个 key，
建议未明确的历史 causal 标志不影响这些计数。有效 FLOPs 为 `4*256*60*16*1*512`，
物理 tile FLOPs 为 `4*256*16*256*1920`；唯一 KV 字节为 `2*60*16*512*256*2`，
恰好等于重建请求量。Q 加 output 为 `2*60*16*1*256*2` 字节。

Case 7 的 4.5× KV 请求比例不证明 HBM 流量也放大 4.5 倍；case 15 的物理工作放大
16 倍也不能单独证明 Cube 是瓶颈。缺失源码的实际 launch/core 分配、buffer allocation
和存活 slot 数量均为 UNKNOWN，不能仅凭几何填写。Vector/state 工作单独计算，继续
按已有 [Roofline 规则](roofline.md)分析；请求字节不能替代实测流量，名义容量也不能
替代可持续吞吐。

## 容量算式仍需要生命周期证明

按所述物理 M16/N256/D256 和两字节元素，一个 L1 slot 分别为 Q=8,192 B、K=131,072 B、
V=131,072 B、P=8,192 B：

| 说明性 L1 安排 | 总字节 | 算术可以确定什么 |
|---|---:|---|
| Q1 / K2 / V2 / P2 | 548,864 | 超过所述 524,288-B L1 容量 |
| Q1 / K1 / V2 / P2 | 417,792 | 此 L1 合计能容纳；仅凭算式不能证明排程或其他存储层 |
| Q1 / K1 / V1 / P1 | 278,528 | L1 合计更小，可能的串行化不是实测结果 |

建议中报告“分配 V 时 532,480 B”恰好等于尚未加 P 的 Q1+K2+V2。复算该数字不等于
重跑已经不可用的历史 allocator 过程。K1 的合法性要求覆盖前完成最后一次 QK 操作数读取；
V 和 P 仍可能服务延迟 PV。K 同时承担 V 角色时，更晚的 reader 会改变这个结论。
沿用现有[最后读取者与 slot 推导](pipeline-model.md#推导物理存储和-credit)，并合计所有
存储层，再按[证据表](../common-language.md#evidence)分别检查同步、backend 与计时。
[独立 K/V slot 样例](../../../library/examples/api/independent_kv_slots#independent-last-readers)
为这些策略维护自己的有界源码/模型/emission/native 记录，不是完整 attention 或第四个性能场景。
其当前证据确认这些分配、过大策略的定位拒绝，以及六个 PyPTO native 的逐位通过结果。
[五策略计时示例](../../../library/examples/api/independent_kv_slots#preliminary-device-timing)
各保留一次独立采集，其中较少 slot 或 lookahead 也有更慢的结果。这些证据属于新的
primitive，历史 R7 程序仍不可用。

## 已完成的 case 6 排程对照

已保存的 native 对照（`a5_mha_fp16_bf16/evidence/schedule-study/native-summary.json`）
只交换**当前 item 的 QK/softmax**与**前一 item 的 PV/finish**两段顺序。恢复这一处 phase
交换后，两份源 AST 相等；其他执行文件、head 映射、输入、M128/N128、grid、slot/credit
数量、cache tag 与数值顺序一致。这是两种 P2 排程的对照。更早的单槽 resident 实现具有
不同的 V 流量，不能当作本次归因的 control。
逐核源码成本记录（`a5_mha_fp16_bf16/evidence/schedule-study/requested-cost.json`）
绑定了以下完整 case 的数量。

| 完整 case 的共同数量 | 两种排程相同 |
|---|---|
| 工作 / launch | 16,384 个逻辑行；128 items；28 个活跃 Cube / 56 Vector；每 Cube 4–5 items，最多640个输出行 |
| L1 | Q、K 各128×128 BF16 单槽；V、P 同 shape 各双槽；总计196,608 B |
| L0C | Score 128×128 FP32 双槽；product 同 shape 单槽；总计196,608 B |
| L0A / L0B | 每个存储层各拥有两个32-KiB操作数slot，即各65,536 B；raw/typed alias不重复占用存储 |
| 每 Vector UB | Score 64×128 FP32 ×2；product 64×128 FP32 ×1；P 65×128 BF16 ×2；denominator 1×64 FP32 ×2；output 64×128 BF16 ×1；总计148,480 B |
| Credit | QK 2；P 发布2；product 1 |
| GM 请求 | Q加载128次，K加载52次，V加载92次；K+V payload为4,718,592 B，Q+output为8,388,608 B |
| Cube 工作 | 128次QK和128次PV tile product；1,073,741,824 FLOPs，也是此noncausal完整case的有效dense数量 |
| FIX / 发布 payload | Score FIX 8,388,608 B；product FIX 8,388,608 B；P发布4,194,304 B |
| 实测硬件stage时间 / HBM/L2字节 / pipe utilization | 本case6对照中仍为UNKNOWN |

P的额外UB行提供NZ pitch。FP32 denominator保留自己的item slot直到finish完成；P的L1
reader退役，并不代表denominator或output UB也已退役。K与每个V slot有独立的head tag，
因此shape相同仍可能有不同加载次数。这些是源码请求，不是实测HBM流量。迁移数量前先读
两份源码与对照证明（`a5_mha_fp16_bf16/evidence/schedule-study/source-proof.json`）。

| Native phase顺序 | 三次独立采集的median，µs | Median / range，µs |
|---|---|---|
| 当前QK/softmax，然后前一项PV/finish | 12.25、12.38、12.08 | 12.25 / 12.08–12.38 |
| 前一项PV/finish，然后当前QK/softmax | 15.14、15.32、15.27 | 15.27 / 15.14–15.32 |

每次采集保留原profiler/parser、三次warmup与五次active步骤，并绑定精确的
source/artifact/runtime/input；原精度和每个active步骤恰好一个目标kernel均通过。
QK-first相对PV-first control为**1.2465×加速**，本对照时延下降**19.78%**。加速比为
`15.27/12.25`，下降幅度为 `(15.27-12.25)/15.27`，即都以15.27 µs为比较对象。
13.55 µs是case6已发布的历史baseline元数据，
另行实测的官方FIA median为18.07 µs。两者均不改写这两组开发采集及其执行身份；
该单元的最终结果仍由它自己的历史性能记录单独绑定，demo 不绑定其中任何一项。

模型区间记录（`a5_mha_fp16_bf16/evidence/schedule-study/model-intervals.json`）
使用更小的BF16 case：B1/SQ257/SKV128/H2/D128，单Cube。六个items分别覆盖每个head的
query行0–127、128–255及只有一行的尾块256；head/query/slot图例与两份原始64阶段区间均
保留。对每个Cube stage/item，先与对应Vector区间求交，再对**两个Vector参与者取并集**。
下表汇总记录中各item pair的并集长度。

| 模型phase顺序 | Softmax(当前) ∩ PV(前一项)，并集cycles | Finish(前一项) ∩ QK(当前)，并集cycles | 完整模型cycles |
|---|---:|---:|---:|
| 当前QK优先 | 1,797 | 0 | 37,922 |
| 前一项PV优先 | 0 | 1,261 | 40,774 |

这些区间证明模型排程，不是硬件stage时长；其并集或模型总cycles之差都不能换算成节省的
native微秒。Case7另行采集的vendor PMU列只描述它自己的源码与诊断协议，不能填写上表
尚缺的case6硬件单元格。

## 单独解释 case 7 的 PMU 诊断

三份开发诊断（`a5_mha_fp16_bf16/evidence/diagnostics/index.json`）
分别保留 factory、原始 CSV 和源码身份。以下是五个插桩目标任务的中位数，列名沿用
vendor 的原始输出：

| 已采集的 case7 源码 | `aic_mac_ratio` | `aic_mte2_time(us)` | `aiv_vec_time(us)` | `cube_utilization(%)` |
|---|---:|---:|---:|---:|
| `preload-v1` | 0.270 | 114.215 | 207.300 | 94.696 |
| `preload-cache256-v1` | 0.300 | 56.521 | 194.690 | 95.158 |
| `preload-transposed-v1` | 0.358 | 56.613 | 128.111 | 95.252 |

保留这些 vendor 定义与分母。它们不能确定关键路径上的等待，也不等于 peak-FLOPs
效率；不能把 `cube_utilization(%)` 改称 MAC 效率。实测 HBM/L2 字节与可持续带宽仍为
UNKNOWN。插桩时延不参与性能资格；这些特定源码的诊断不能填入 case6 的 stage 时间，
也不能把后续硬件绑定变化前后的差异归因于代码改动。

历史改进建议还报告 N256 VF 展开没有稳定收益；对应源码对照和原始时延样本在这里不可用。
保留其标签为 **reported / raw evidence unavailable**，不称为已复现退化，也不能当成
循环优化收益上限的实测证明。当前 kernel 的瓶颈需要新的 source-matched 对照。
