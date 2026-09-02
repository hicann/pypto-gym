# 性能优化知识卡片库

本目录是 Stage 5 的可扩展优化项来源，不属于 `pypto-pro-op-kb`。KB 承载已选择的
pattern/constraint 合同；知识卡承载贡献者提供、需要在当前算子上重新判断和实测的原子优化方法。

`INDEX.md` 是卡片生命周期和 Stage 5 枚举的唯一入口。实际卡片按类别放入子目录，例如
`single-core-pipeline/`；目录和文件的存在本身不使卡片生效。当前库为空，因此不创建空类别目录。

**当前知识卡片计数：active item 为 0，draft item 为 0。**

## Active atomic items

Stage 5 只枚举本表中 `lifecycle=active && stage5_eligible=yes` 的原子项。`active` 表示卡片完整到
可以验证，不表示一定适用、一定有收益或示例可直接运行。

| item_id | 卡片与锚点 | lifecycle | stage5_eligible | bound_hint | hard_applicability_gates | target/api_gate |
|---|---|---|---|---|---|---|

## Draft items（不进入 Stage 5）

新卡先登记到本表并保持 `lifecycle=draft`、`stage5_eligible=no`。完成事实锚点、边界及
[CARD_TEMPLATE.md](CARD_TEMPLATE.md)规定的跨来源去重审查后，才可晋升到 active 表。
核心动作只能用尚未验证的能力门控伪码表达时必须继续留在 Draft；可选增强存在能力门控时，须
与已验证、可执行的核心动作明确分开。

| item_id | 卡片与锚点 | lifecycle | stage5_eligible | bound_hint | hard_applicability_gates | target/api_gate |
|---|---|---|---|---|---|---|

## ID 与晋级规则

- `item_id` 使用 `<类别>-<两位递增序号>`，文件名使用 `<item_id>-<短名>.md`，索引路径必须包含
  类别目录；已进入索引的 ID 只增不改、不得复用。当前库为空，各类别从 `01` 开始。
- `general-*` 由[通用优化手段](../general-optimization-methods.md)独占。
- active 行必须完整填写七列，并与卡片中的 `item_id`、`lifecycle` 和 `stage5_eligible` 一致。
- Stage 5 记录 INDEX 行身份、卡片文件/原子锚点和内容哈希；贡献者不手工固定运行时哈希。
- `lifecycle!=active` 或 `stage5_eligible!=yes` 的项不进入 Stage 5 闭合分母。

## 贡献入口

复制并填写 [CARD_TEMPLATE.md](CARD_TEMPLATE.md)，先加入 Draft 表；完成模板末尾的晋级检查后，
再由维护者更新卡片状态并移动到 Active 表。

`templates/` 不属于知识卡片库，而是独立的 Stage 5 模板优化项来源；其生效项只由
[模板优化项索引](../../templates/INDEX.md)枚举，不计入本 INDEX 的 card 分母。
