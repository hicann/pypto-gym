# Issue 类型映射规则

定义断裂点类型到 GitCode Issue 的映射关系和标题模板。

## 映射规则

| 断裂点类型 | Issue 类型 | 标签 |
|------------|------------|------|
| D1, D2, D5 | Documentation | `documentation` |
| D3, D6 | Bug Report | `bug`, `documentation` |
| A1, A2, A4, A5 | Bug Report | `bug` |
| A3 | Feature Request | `enhancement` |
| E1, E2, E3, E4 | Bug Report | `bug` |
| C1, C2, C4 | Bug Report 或 Feature Request | 视具体情况判断 |
| C5, C6 | Bug Report 或 Feature Request | 视具体情况判断 |

### C 类断裂点的 Issue 类型判定

C 类断裂点（行为模式类）的 Issue 类型需要根据具体情况判断：
- 如果根因明确指向框架 bug（如 API 返回错误导致重试）→ Bug Report
- 如果根因指向缺少功能或文档（如没有示例导致过度探索）→ Feature Request 或 Documentation
- 如果根因不明确 → Bug Report（默认）

## Issue 标题模板

| 断裂点类型 | 标题模板 |
|------------|----------|
| D1 | `[Doc] {实体名} 文档缺失` |
| D2 | `[Doc] {实体名} 文档不完整：{缺失内容}` |
| D3 | `[Doc] {实体名} 文档与实际行为不一致` |
| D5 | `[Doc] {实体名} 缺少代码示例` |
| D6 | `[Doc] {实体名} 示例代码无法运行` |
| A1 | `[Bug] {实体名} API 不存在` |
| A2 | `[Bug] {实体名} API 行为异常` |
| A3 | `[Feature] {实体名} 参数限制：{具体限制}` |
| A4 | `[Bug] {实体名} 类型不匹配` |
| A5 | `[Bug] {实体名} 边界行为未定义` |
| E1 | `[Bug] {实体名} 操作失败无错误提示` |
| E2 | `[Bug] {实体名} 错误信息模糊` |
| E3 | `[Bug] {实体名} 错误信息误导` |
| E4 | `[Bug] {实体名} 错误信息无修复建议` |
| C1 | `[Bug] {实体名} 导致反复重试` |
| C2 | `[Bug] {实体名} 信息不足导致决策困难` |
| C4 | `[Bug] {实体名} 信息分散导致过度探索` |
| C5 | `[Bug] Session 意外中断` |
| C6 | `[Bug] Session 需要用户介入` |

## 模板使用说明

- `{实体名}`：替换为具体的实体名称，如 `pypto.reshape`、`pypto.Tile`
- `{缺失内容}`：替换为具体缺失的信息类型，如"参数说明"、"返回值类型"
- `{具体限制}`：替换为具体的参数限制描述，如"不支持 float16 输入"
- Session 级断裂点（C5、C6）的标题不包含实体名
