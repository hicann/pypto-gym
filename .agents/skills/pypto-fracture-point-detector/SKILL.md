---
name: pypto-fracture-point-detector
description: 分析当前 session 上下文，识别 PyPTO 框架或文档不完善导致的断裂点，产出可转化为 Issue 的结构化报告。当用户在 pypto 相关 skill 运行结束后提到"断裂点"、"识别断裂点"、"检测断裂点"、"fracture point"时触发此 skill。也适用于用户对 session 中遇到的问题进行复盘、想要生成问题报告、或希望改进 pypto 框架/文档质量的场景。
---

# PyPTO 断裂点识别

分析当前 session 的对话历史，识别由 PyPTO 框架或文档不完善导致的"断裂点"——即 skill 运行过程中的失败、重试、效率低下等问题——并产出结构化的 Markdown 报告。

## 核心概念

- **断裂点**：session 中因 pypto 框架或文档不完善导致的问题点
- **实体**：session 中涉及的 API（如 `pypto.add`）、文件、概念等对象
- **信号**：断裂点的可观察表现（如搜索失败、反复重试、操作报错）
- **置信度**：断裂点判定的可靠程度，低置信度的断裂点会被过滤掉

共 19 种断裂点类型，分为文档类（D1-D3, D5-D6）、API/框架类（A1-A5）、错误信息类（E1-E4）、行为模式类（C1-C2, C4-C6）。完整定义见 `references/fracture-points.md`。

## 使用场景

**典型场景**：用户在主 agent 会话中调用此 skill，分析当前会话。

```
用户工作流程：
1. 用户在 session 中开发 PyPTO 算子（可能调用多个 subagent）
2. 开发完成后，用户说："识别断裂点"或"生成断裂点报告"
3. skill 分析当前 session（默认）或当前 session 及其所有子会话（scope=full）
4. skill 输出断裂点报告
```

**两种分析模式**：

| 模式 | 分析范围 | 数据来源 |
|------|---------|---------|
| `scope=current`（默认） | 仅当前 session | LLM 直接回顾对话历史 |
| `scope=full` | 当前 session + 所有子会话 | SQLite 数据库查询 |

## 工作流程

按以下步骤顺序执行。默认使用 `scope=current` 模式。

### 步骤 0：确定分析模式

根据参数确定分析范围：

- **无参数或 `scope=current`（默认）**：只分析当前 session，跳过步骤 0.1-0.3，直接进入核心分析步骤
- **`scope=full`**：获取当前 session 及其所有子会话的原始数据，执行步骤 0.1-0.3

#### 子会话数据获取（scope=full）

> **重要说明**：子会话数据获取功能依赖于 opencode 框架的特定实现（SQLite 数据库）。在其他 agent 框架下，此功能可能不可用或需要重新实现数据访问层。若当前环境不支持数据库访问，应使用 `scope=current` 模式。

当使用 `scope=full` 时，通过 opencode 的 SQLite 数据库读取会话数据。默认路径为 `~/.local/share/opencode/opencode.db`，可通过环境变量 `OPENCODE_DB_PATH` 覆盖。

#### 步骤 0.1：获取当前 Session ID

```python
from scripts.session_db import get_current_session_id, get_session_title

current_session_id = get_current_session_id()
current_session_title = get_session_title(current_session_id)
```

#### 步骤 0.2：确认 Session 正确性

**情况 A：推断的 session 标题与当前对话内容匹配**

继续执行。

**情况 B：推断的 session 标题与当前对话内容不匹配**

```python
from scripts.session_db import list_recent_root_sessions

recent_sessions = list_recent_root_sessions(limit=10)
# 根据当前对话内容判断正确的 session
```

仅在无法确定时才需要用户确认。

#### 步骤 0.3：获取子会话数据

```python
from scripts.session_db import get_child_sessions, get_session_parts

# 获取子会话列表
children = get_child_sessions(current_session_id)

# 获取每个子会话数据
child_parts = [get_session_parts(c.id) for c in children]
```

主 session 数据直接回顾当前对话历史即可。对每个会话执行核心分析步骤 1-8，然后汇总到步骤 5。

---

### 核心分析步骤（步骤 1-10）

以下步骤为两种模式共用，用于分析单个 session：

#### 步骤 1：读取各个 Session（主session + 子Session） 上下文

读取 session 的完整对话历史，重点关注：
- 工具调用及其返回结果（尤其是错误和空结果）
- 用户消息中的修正、澄清、不满表达
- 搜索/读取操作的频次和模式
- 任务是否最终完成

**成功标准**：完成回顾后，应能列出 session 中涉及的所有关键实体和操作。

#### 步骤 2：实体识别

从对话历史中识别所有涉及的 pypto 实体。识别模式见 `references/entity-patterns.md`，主要包括：
- **API**：匹配 `pypto.xxx` 模式的 API 调用
- **文件**：涉及的 `.py`、`.md` 等文件路径
- **概念**：tile、tensor、pass、codegen、ub、gm 等框架概念

为每个实体确定复杂度等级（简单/中等/复杂），规则见 `references/entity-complexity.md`。

**成功标准**：完成识别后，应产出实体列表，每个实体标注复杂度等级。

#### 步骤 3：信号检测

按实体聚合，逐一检测 10 类信号。每个信号对应一个或多个断裂点类型。检测规则和阈值见 `references/detection-rules.md`。

关键信号包括：
- **搜索失败**：搜索返回空或"未找到" → D1, D5
- **操作失败**：工具调用返回 Error/Exception/Failed → A1, A2, E1, E2
- **重复操作**：同一实体上相同操作出现 ≥3 次（简单实体） → C1
- **过度探索**：搜索/读取次数超过阈值 → C4
- **用户介入**：用户消息中包含修正或手动指导 → C6
- **任务未完成**：session 最终未达到预期目标 → C5

阈值会根据实体复杂度加权调整，复杂实体的阈值更宽松。

**成功标准**：完成检测后，应产出信号列表，每个信号标注对应断裂点类型。

#### 步骤 4：断裂点判定

将检测到的信号匹配到 19 种断裂点类型，并评估置信度。

置信度评估基于 4 个条件：
1. 有明确错误信息（Error/Exception/Failed）
2. 多个信号佐证（≥2 个信号触发同一断裂点）
3. 证据完整（能清晰复现问题）
4. 用户明确反馈（表达困惑或不满）

判定规则：
- 满足 ≥2 个条件 → 高置信度 → 输出
- 满足 1 个条件 → 中置信度 → 输出
- 不满足任何条件 → 低置信度 → **剔除**

#### 步骤 5：去重与合并

同一实体 + 同一断裂点类型 = 同一个断裂点。合并所有触发的信号和证据片段。

**scope=full 模式**：合并所有 session（主 + 子）的断裂点，为每个断裂点标记来源 session（"主 Session" 或 "子 Session-{short_id}"）。

#### 步骤 6：二次校验（根因归属判定）

> **目的**：排除因模型自身原因导致的误报，确保断裂点根因确实归属于框架或文档。

对去重后的候选断裂点，**必须实际读取相关文档**后执行检查。具体检查项和判定规则见 `references/detection-rules.md` 中的"二次校验规则"章节。

检查结果处理：
- 根因归属 PyPTO框架/PyPTO文档/PyPTO样例/Agent框架 → 计入断裂点总数，输出到报告正文
- 根因归属模型自身 → 标记为"模型能力"，在报告末尾单独列出，不计入断裂点总数

#### 步骤 7：关联标记

相同实体的不同断裂点自动标记为关联。例如 `pypto.reshape` 同时有 D1（文档缺失）和 C1（反复重试），它们互相关联。

#### 步骤 8：置信度过滤

剔除所有低置信度的断裂点。只有中/高置信度的断裂点进入最终报告。

#### 步骤 9：环境信息获取

执行以下命令获取环境信息，**任何命令失败则对应字段标记为"未知"**，不影响报告生成：

| 信息 | 命令 |
|------|------|
| CANN 版本 | `echo $ASCEND_HOME_PATH \| grep -oP 'cann-\K[\d.]+'` |
| PyPTO Commit | `git merge-base HEAD origin/master 2>/dev/null && git log -1 --format='%H %ci' $(git merge-base HEAD origin/master) \|\| echo "Unknown"` |
| 服务器类型 | `lspci -n -D \| grep '19e5:d80[23]' \| sed 's/.*d80\([23]\).*/A\1/' 2>/dev/null \|\| echo "Unknown"` |
| Python 版本 | `python --version 2>/dev/null \|\| echo "Unknown"` |
| 操作系统 | `cat /etc/os-release 2>/dev/null \| grep PRETTY_NAME \|\| echo "Unknown"` |

#### 步骤 10：报告生成

根据检测结果生成报告：

**有断裂点的情况**：
- 使用 `templates/report-template.md` 中的模板生成报告
- 文件名：`fracture-point-YYYY-MM-DD-HHMMSS.md`
- 保存到当前工作目录
- 断裂点按优先级排序：致命 > 高 > 中
- 每个断裂点包含：类型、优先级、根因归属、Issue 建议、证据片段、优化建议
- Issue 类型映射和标题模板见 `references/issue-mapping.md`
- `scope=full` 模式下，报告包含"子会话分析概览"章节，每个断裂点标记来源 session

**无断裂点的情况**：
- 默认不生成报告文件，在屏幕直接输出摘要：session 基本信息、操作统计、涉及实体列表、"未检测到断裂点"提示
- 当上游流程要求产出文件（例如 benchmark 批量分析）时，仍需生成报告文件，断裂点总数为 0，并注明"未检测到断裂点"

## 重要提醒

- 证据片段必须是 session 中的原文引用，不要编造或概括
- 每个断裂点的"问题描述"应清晰到能让不了解 session 上下文的人理解发生了什么
- 优化建议要具体、可操作，指向具体的文件或 API
- Session 级断裂点（C5、C6）不针对特定实体，单独列为一个章节
- `scope=full` 模式使用"最近活跃推断策略"获取当前 session ID，在多 session 同时活跃时可能不准确，报告中需说明此限制
- 分析子会话时，完整读取用户消息、助手回复、推理过程和工具调用链
- **二次校验**：必须执行步骤 6 的两项检查，验证根因是否确实归属于框架或文档。断裂点分类包含：PyPTO框架、PyPTO文档、PyPTO样例、Agent框架、模型能力。前四个是问题，模型能力是非问题（不应计入断裂点总数）。
- **benchmark 日志分析**：当本 skill 被 benchmark 汇总流程调用时，每个断裂点必须额外标注"聚合分类"（Type-1/Type-2/Type-3）和"分类依据"，供综合报告过滤统计使用。
