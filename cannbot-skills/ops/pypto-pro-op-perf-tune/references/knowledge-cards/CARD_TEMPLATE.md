## 使用说明（贡献卡片时删除本节）

将本文件复制到对应类别子目录；先分配 `item_id=<类别>-<两位递增序号>`，再命名为
`<item_id>-<短名>.md`，并按目录深度修正相对链接。新卡先以 draft 状态加入
[INDEX.md](INDEX.md)，完成文末检查后再晋升为 active。
`general-*` 前缀由[通用优化手段](../general-optimization-methods.md)独占。

# 技术卡片 <id>：<一个原子优化动作>

- **item_id**：<类别>-<两位递增序号；与文件名前缀一致>
- **lifecycle**：draft
- **stage5_eligible**：no
- **适用范围**：<VEC / Cube / Scalar / 访存 / 调度；可多选>
- **一句话**：<在什么条件下，把什么结构改成什么结构>
- **目标/API 门**：<支持的 SoC、版本或必须现场核验的能力>

## 何时用（诊断特征）

列出可从当前源码、生成物或 profiler 直接核对的特征；ratio 和历史经验只能作为线索。

## 何时不适用

列出可以关闭为 `not_applicable` 的明确条件和容易误判的相邻情形。

## 原理

解释改动减少或重排的工作、搬运、依赖、冲突或固定开销。一个卡片只承载一个主要机制；独立
动作拆成新卡，并用稳定 ID 声明关系。

## 怎么改（before / after）

当前公开 PyPTO-Pro API 能表达核心动作时，active 卡片必须至少给出一个 PyPTO-Pro `after`
片段；能用最小 before/after 清楚表达时同时给出两者。代码 fence 标为以下一种：

- `结构示意`：只展示数据流，明确缺少哪些上下文且不可直接编译或交付；
- `嵌入片段`：使用已核验的公开 `pl.*` / `vf.*` API，并说明嵌入的 kernel/VF 上下文、
  目标版本以及 dtype、shape、tail 等必要前提；
- `能力门控伪码`：API 尚未确认，禁止直接复制实现。核心动作只能这样表达时，本卡保持
  `lifecycle=draft`、`stage5_eligible=no`，直到能力与数值验证补齐；可选增强必须与已验证的
  核心动作明确分开。

不得臆造 PyPTO-Pro API，也不得把生成 C++ 或 AscendC 写法伪装成 Python API。
示例不能替代当前算子的完整正确性、quick 筛选和 formal compare。

## 性能与验证指标

写明预期指标、正确性覆盖、quick 筛选和 formal compare。外部或历史数字必须标注来源及
`unverified_external_historical`，当前算子未复现前不能当收益承诺。

## 红线（Stop rules）

列出精度、语义、dtype/layout、容量、tail、同步、并发、wrapper、目标版本和恢复条件。卡片不能
放宽 SPEC、Stage 4 铁律或统一测量协议。

## 关系与事实锚点

- **事实锚点**：<官方文档、源码、ST 或可复核实验；写清版本和适用边界>
- **与 KB 的关系**：<无重合；或列出 KB 路径及本卡独特增量>
- **与通用优化手段的关系**：<无重合；或列出 general item id 及本卡独特增量>
- **与模板优化项的关系**：<无重合；或列出 template item id 及本卡独特增量>
- **卡片关系**：<requires / enables / conflicts / overlaps + card id>

## Active 晋级检查

- [ ] ID、文件名和标题一致，编号未复用，且未使用 `general-*`。
- [ ] 是可执行、可证伪的单一优化动作，适用与不适用条件可由当前证据判断。
- [ ] 目标/API 能力有事实锚点；核心动作不是仅由能力门控伪码表达，没有伪 API。
- [ ] 公开 API 能表达核心动作时，已有符合上述边界的 PyPTO-Pro `after`；before/after、正确性
      和性能验证方法完整，示例等级明确。
- [ ] 历史数字未冒充当前设备结论。
- [ ] 已与 active 卡片、active 模板、KB 和通用优化手段去重，或写清独特增量。
- [ ] 所有相对链接有效，关系使用稳定 ID。
