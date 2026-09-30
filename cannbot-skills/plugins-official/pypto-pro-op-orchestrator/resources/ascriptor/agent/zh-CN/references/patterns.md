# 选择数据流模式

先推导数学与契约。模式解释组合，精确 API 从可运行 demo 和当前声明取得。kernels owner 只
提供可运行 demo，不提供实测 support，也没有后端支持矩阵——某个 demo 对得上只是起点，不是
支持声明。生成的索引按 owner 重建，不复制成另一套事实。

| 数据流 | 保留的不变量 |
|---|---|
| Cube-only matmul | 输出 tile 唯一 owner；K 初始化/累加明确 |
| A2 row reduce/broadcast | group 汇总为 row scalar；scratch 满足下游 footprint |
| A2 独立 group reduce | count 与 repeat stride 分开，不合并独立 group |
| A5 packed cast | 稀疏 register 位置、live reinterpret alias、pack/unpack 顺序 |
| Online softmax | score mask 在 max 前；sum precision 与 delayed value cast |
| Cube/vector bridge | 设备物理 layout 和 publish/consume/reuse |
| Lookahead/drain | 各 stage 自己的 work index，drain 产生最后输出 |
| Slot 多角色 | 同时 live 存储足够，credit 与 rotation 匹配 |

Exact uint2 检查所有 256 种四值组合与 256 carrier byte、bit order 和输入 domain。
FP4 覆盖 signed zero、tie mode、defined lane 与 packing 边界；decode scratch 可能需要完整
register store footprint，即使逻辑有效值更少。
Attention 变体可能同时改变 precision 和 schedule；先选择
[逐元素、右下对齐或 Block32 predicate](attention-authoring.md#causal-mask)。half/hif8/FP8
probability 改变 delayed value contract，float row sum 可保持；public rowmax/rowsum 与 saved
state 要比较。score/PV lifetime 重叠则保持不同 scratch。旧最快/硬件结论不转成新结果。

模式不覆盖所需能力时查 facade、frontend/lowering、simulator 和邻近可组合 primitive；生成
最小 probe，必要时补 emission/board 证据。候选失败或画廊里没有条目不能证明没有合法组合。
按需读[内存](memory-and-tails.md)、[精度](precision.md)、[同步](synchronization.md)和
[调试](../playbooks/debug.md)。
遇到对应旧失败症状时查历史经验，其原验证范围不变，不代表新支持声明。

<a id="simt-start"></a>
## SIMT 组合的起步检查

1. 按 [SIMT 身份规则](../../../library/docs/api/simt.md)将
   `(core, vector participant, thread, iteration)` 映射到逻辑输入/输出元素。
   普通输出证明唯一写者；atomic 输出先数贡献者并初始化目标。
2. 对每个共享值写出 producer、consumer 和必需的会合范围，分别选择访问排序与会合机制；
   跨侧发布遵循[同步](synchronization.md)规则。
3. 覆盖工作量小于/大于线程数、tail 和多个参与者。除数值外，检查 output poison 与实际
   writer trace；重复写相同值也可能隐藏归属错误。从
   [SIMT atomic 样例](../../../library/examples/api/simt_atomics)或匹配的 owner 样例起步。

<a id="sort-start"></a>
## Sort 或 topk 组合的起步检查

有限 FP32、最大值、无序 TopK 可先看公共
[`radix_topk` 复合接口](../../../library/docs/api/sorting.md#register-radix-selection)
和[完整算法 demo](../../../kernels/ascriptor_kernels/algorithms/a5_radix_topk)。阈值搜索在
寄存器内完成；一次调用由前端编译出可检查的 VF 基础 IR。先核对固定容量、count/k、padding
与 ties，毋须自行复制拼装整段算法；后端能不能跑，用该目录的 `main.py` 带上你要的 backend
跑一遍，而不是去查表。

1. 在任务合同中固定选出数量、输出有序/无序、tie 策略与允许的 score domain，
   再选择 [record 操作](../../../library/docs/api/sorting.md)。
2. 同时追踪 score 与 ID 位，推导 record/scratch footprint，证明每路 merge 输入已满足
   所需顺序，并保留 padding 与 tail 的初始化。
3. 在合同允许的 tie 规范化之前，检查入选集合、所需顺序、ID 重数与 score–ID 配对。
   覆盖重复分数和边界尺寸，拒绝缺失/重复 ID 及故意错配的记录；
   [record 样例](../../../library/examples/api/sort_records)提供 primitive 对照。

## 选择一个样例

重复 CVC/VCV/CVCV/VCVC 或更长图先读[通用方法](pipeline-model.md)，再读
[CVC 推导](cube-vector-cube.md)和[混合流水 demo](../../../kernels/ascriptor_kernels/tutorials/mixed_pipeline)
中的匹配结构。[数值方法](numerical-patterns.md)覆盖 tail、规约与 cast；
[Roofline](roofline.md)把资源忙碌与可删除工作、排程空间联系起来。

在 agent checkout 用小索引选择，避免预读整个画廊：

```bash
python tools/select_example.py --pattern VCVC --language zh-CN --limit 1
python tools/select_example.py --query '尾块 访问范围' --device a5 --limit 2
python tools/select_example.py --pattern mla --language zh-CN
python tools/select_example.py --pattern p-publish --language zh-CN
```

`--query`、`--pattern`、`--language`、`--device`（`a5|a2|a3`）和 `--limit` 就是全部接口。
没有 `--dtype`、`--layout`、`--backend` 筛选了：目录不声明 dtype、layout 和 backend，
没有东西可筛。

两个 owner 的答案是同一个，因为两边都是四文件目录：指南、**目录**、它的 `formula` 或
`surface`、`topology`、`tags`、case **数量**、一行 `run`，以及一句 `support_scope`：它不记录
任何后端结果——它在哪里跑过，取决于你在哪台机器上跑它。未匹配时先使用返回的 `fallback`
查 API/样例总目录，再进入聚焦源码与 probe 调查；未匹配不证明缺少能力。

`--pattern mla` 选中 [MLA demo](../../../kernels/ascriptor_kernels/attention/a5_mla_fp16_bf16)。
它的精确 shape、数值约定和排程都在那个目录里：`python main.py --list` 逐条打印 case、
它用的排程，以及 model shape 的 case 代表哪个 full shape；`--variant` 只保留使用某一种排程
的 case。[Attention 专题](attention-authoring.md)把该入口接入完整编写方法，成本实例聚焦具体
实验。指定 pattern 也不能绕过语义冲突；列出的 case 是一个精确 shape，不声明任意笛卡尔积
支持。参见[完成的成本实例](mla-cost-case.md)。

某个后端能不能跑某个目录，索引里没有，选择器里也没有。在有卡的机器上用你需要的
`--launcher` 和 `--backend` 跑那个目录的 `main.py`。确实跑不了时，理由是该目录自己
`main.py` 里的注释，写明拒绝的是什么、在哪，并且那个 case 会带着这条理由被跳过，而不是
被放宽或删掉；索引只记录"存在这样一条注释"——demo 用 `pypto_pro_note`，API 样例用
`refusals`，后者还写明是哪个后端或 launcher。

需要历史 [E4M3 MLA](../../../kernels/ascriptor_kernels/attention/a5_mla) 时使用
`--pattern mla-e4m3-16x`。结果保留原 Q/K/P 格式和 16× 输出约定；旧名 `mla_hif8` 只是精确
导航别名，不是 HiFloat8 dtype 声明。P 发布诊断使用 `p-publish`，它选中 library 的
cube/vector roundtrip 及该目录自身能证明的东西，不继承完整 MLA 的结论。

## 检索覆盖与范围

[导航索引说明](../../index/README.md)区分自动生成的[完整目录](../../index/kernels.json)与
[双语主题词表](../../index/patterns.json)。可用精确 unit ID、owner 路径或中英主题词查询，
例如 `--query SIMT`、`--query 量化`、`--pattern api.event_depths`。
API/签名查[总页](../../../library/docs/api/README.md)，primitive 查[样例目录](../../../library/examples/api/README.md)，
完整算法查 [kernel 画廊](../../../kernels/README.md)；两个 owner 的目录都不记录验收阶段，实测
只在 library receipt 里，由 release 记录划范围。

结果的 `matched` 解释命中来源。`related` 只提供相关读物；矩阵乘积归一化不是 LayerNorm，
SIMT 转置样例融合了 matmul，无序 topk 不保证排序输出。`deferred_material` 保留延期范围，
现有 GELU/SwiGLU 不能转成 A5 支持。`reason` 区分没有候选、相关参考、延期、筛选冲突和元数据不足。
`fallback` 保留查询条件并给出总入口；没有 dtype/layout 的结构化事实时保持未知，不从文件名猜测。
