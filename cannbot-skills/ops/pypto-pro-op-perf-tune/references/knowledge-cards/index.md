---
okf_version: "0.2"
---

# 性能优化知识卡片库

本目录是一个采用 [Open Knowledge Format（OKF）v0.2](https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md)
组织的 PyPTO-Pro 性能优化方法库，供性能调优时查阅和实验。卡片说明适用条件、改法、技术限制与验证方法，
内容质量在合入前由技术人员把关。

本文件是卡片清单的**唯一入口**，调优时只从 Active 表获取候选，不通过扫描目录自动采用卡片。
实际卡片按稳定类别放入子目录。初始内置 14 张 VF/VEC 卡片，位于 `vec/`，编号为 `vec-01` 至
`vec-14`，均以 `status=stable` 登记到 Active 表。

## Active items

新卡默认登记到本表，合入前由技术人员 review。调优时按本表列出候选，
由使用者结合卡片描述、当前算子和目标能力判断适用性。

| item_id | 卡片与锚点 | status | bound_hint | applicability | target/api_gate |
|---|---|---|---|---|---|
| `vec-01` | [Vec Tile / VF 内融合，消除 GM 往返](vec/vec-01-ub-fusion.md) | `stable` | `mixed` | 相邻 Vector 链的中间结果无外部消费者，生成物仍有中间 GM 回写再读入，且 Vec 容量可容纳融合 live set | 仅限已核验 VF、TileGroup 与 auto_mutex 的 Ascend 950PR 或 950DT 工具链；其它 SoC 重新查表并验证 |
| `vec-02` | [减少、批量或就近执行 Cast](vec/vec-02-reduce-cast.md) | `stable` | `compute` | 存在冗余 Cast、可批量或就近转换的机会，或不必要的转换中转 Tile，且 dtype、rounding、布局与值域允许相应改写 | 仅限 Ascend 950PR 或 950DT；Tile 级 pl.cast 需当前 ST 支持，UNPK/PACK 须有明确分布模式与端到端 value ST |
| `vec-03` | [`vf.mul_dst_add` 融合乘加](vec/vec-03-fused-instr.md) | `stable` | `compute` | 存在 x 乘 weight 加 bias，编译结果未自动融合，中间乘积无其他消费者且 dtype 与 mask 兼容 | 仅在当前 Ascend 950PR 或 950DT PyPTO-Pro 公开 Vf.mul_dst_add 语义和生成 vmadd 均已复核时采用 |
| `vec-04` | [标量 round-trip 改为寄存器广播与批量计算](vec/vec-04-scalar-roundtrip-vectorize.md) | `stable` | `mixed` | 每行归约后存在标量往返或不必要的 Tile store→load；消费者可用同 VF full、跨阶段单 B32 广播，或已通过八哨兵 value ST 的 BLK 路径 | 仅限 Ascend 950PR 或 950DT；Vf.full 与 BRC_B32 可作保守路径，BLK 和 stride-zero 多行布局须另证 |
| `vec-05` | [消除中间 Vec Tile 暂存（寄存器内直算）](vec/vec-05-eliminate-ub-staging.md) | `stable` | `mixed` | 中间 Vec Tile 仅供紧邻 VF 步骤使用且无跨循环、输出或 bank-conflict 角色，生成物存在 store-load | 仅限 Ascend 950PR 或 950DT；使用已核验的 Vf.full、reduce、shuffle、mem_bar 与 Cast，布局能力须补 value ST |
| `vec-06` | [不变量广播一次 + 多行 VL merge](vec/vec-06-broadcast-once-vl-merge.md) | `stable` | `mixed` | 外提时数据须在复用循环内不变；VL merge 时须证明 R、packed 布局及 Tile、tiling、mask、dispatch 一致 | 仅限 Ascend 950PR 或 950DT；BRC_B32 是保守路径；BLK 须八哨兵 value ST；VL merge 须完整 packed/蝶式布局与尾 mask value ST |
| `vec-07` | [`vf.reduce_*` 硬件树形归约](vec/vec-07-low-latency-reduce.md) | `stable` | `compute` | 当前归约由标量循环或线性依赖链实现，结果只需单值或可广播，且 mask 与 dtype 满足 reduce 接口 | 仅限 Ascend 950PR 或 950DT；reduce、full 与 FIRST_ELEMENT 的语义、对齐和 dtype 支持均须核验 |
| `vec-08` | [归约累加链多路展开](vec/vec-08-multi-accumulator-unroll.md) | `stable` | `compute` | 结合性能指标及源码、生成物或 trace 判断单累加器 RAW 链是瓶颈且 spill 不是主因，归约长度足以摊销多累加器与最终合并 | 仅限 Ascend 950PR 或 950DT；示例所需 mul_add_dst MERGING 当前文档不支持，采用该示例前须另证；另核验 reduce、动态 pl.range 与展开度 |
| `vec-09` | [外层循环下沉进 VF](vec/vec-09-loop-sink-into-vf.md) | `stable` | `mixed` | kernel 按行重复调用同一 VF，行间无依赖，setup 跨行不变且合并后的寄存器与 Tile 生命周期可容纳 | 仅限 Ascend 950PR 或 950DT；确认 vector_function 内动态 pl.range、offset 与逐行 mask 语义后采用 |
| `vec-10` | [小定长循环显式展开](vec/vec-10-small-loop-unroll.md) | `stable` | `mixed` | trip count 编译期已知且很小，展开后的数据、offset 与 mask 等价 | 仅限 Ascend 950PR 或 950DT；只使用已核验 VF 基础算术，是否源码展开由当前 parser、生成物和 TilingKey 决定 |
| `vec-11` | [减少超越函数计算量（PyPTO-Pro 能力门控）](vec/vec-11-reduce-compute-lut.md) | `stable` | `compute` | 超越函数主导热点，且存在以下机会之一：重复调用复用、公开融合接口或有当前 API、编译产物和精度证据的 LUT/近似方案 | 仅限 Ascend 950PR 或 950DT；采用目标版本支持的公开超越函数或融合接口，所需 precision/LUT 能力未明确支持时记录 capability gap |
| `vec-12` | [寄存器溢出时按条件拆分 VF](vec/vec-12-preg-split.md) | `stable` | `scheduling` | 目标路径生成物或 trace 出现 spill/reload，分支条件可证明，且专用 VF 能真正删除整段逻辑和活跃值 | 仅限 Ascend 950PR 或 950DT；通过 TilingKey 或合法控制流分流，RegTraitNumTwo 仅作后端风险模型 |
| `vec-13` | [VF 尾块统一 mask](vec/vec-13-vf-tail-unified-mask.md) | `stable` | `mixed` | full 与 tail 代码体重复，逐拍 active 可由 total 减 offset 精确重算，offset 与 mask 使用同一元素粒度，且不违反已选 KB 中前提成立的 full-mask 外提义务 | 仅限 Ascend 950PR 或 950DT；须核验 vf.update_mask 不回写 Python 标量，lane 常量按 dtype 与生成物确定 |
| `vec-14` | [对齐分段 Tile 布局](vec/vec-14-aligned-split-copy.md) | `stable` | `memory` | 非 32B 段起点在生成物或 trace 中造成对齐退化，且独立 Tile 及 padding 可被 Vec 容量容纳 | 仅限 Ascend 950PR 或 950DT；须核验 pl.load/store 元素 offset、Tile physical/valid shape、TileGroup 地址与对齐合同 |

## ID 与状态规则

- 按方法主题选择类别，优先沿用已有分类；新增类别使用简短英文名。
- `item_id` 使用 `<类别>-<两位递增序号>`，文件名使用 `<item_id>-<短名>.md`，放入类别目录。
  编号从已登记条目中该类别的最大编号递增，新类别从 `01` 开始；已登记 ID 保持不变且不复用。
- `general-*` 由[通用优化手段](../general-optimization-methods.md)独占。
- 修改卡片时同步索引中的 ID、状态、适用条件和 API 限制，方便查阅和 review。
- 索引按 `status` 分组：`stable` 为 Active，`draft` 为 Draft，`deprecated` 为 Retired；有条目时列出对应分组。
- 退役卡保留稳定 ID、历史链接和退役原因，编号不复用。
- 调优时按现有账本记录卡片 ID、文件位置和本轮结论；Draft、Retired 不作为候选。

字段、来源和状态说明见 [OKF 格式约定](PROFILE.md)。

## 贡献入口

第一次贡献先按[贡献指南](CONTRIBUTING.md)准备事实包，并使用其中可复制的提示词让 Agent 生成
卡片，默认以 Active 提交；[带注释示例](examples/annotated-card.md)展示了证据不足时如何保留语义而不补猜。
内容按[卡片模板](CARD_TEMPLATE.md)组织，合入前交技术人员 review。
采用卡片调优的执行流程见[性能调优 Skill](../../SKILL.md)。

`templates/` 是独立的模板优化项来源，其生效项由
[模板优化项索引](../../templates/INDEX.md)单独枚举，与本索引的知识卡片分别计数。

## Bundle resources

* [OKF 格式约定](PROFILE.md) - 卡片字段、可选参考资料和状态的简明说明。
* [贡献指南](CONTRIBUTING.md) - 准备事实包、驱动 Agent 生成卡片和提交人工评审的入口。
* [卡片模板](CARD_TEMPLATE.md) - 新建知识卡时使用的结构与默认字段。
* [带注释示例](examples/annotated-card.md) - 展示事实不足时如何保留语义和待确认项。
