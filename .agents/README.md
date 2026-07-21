# PyPTO Agent Team

预配置的 9 智能体团队，覆盖昇腾 NPU 算子从零到一的端到端开发，由显式的 Stage 1–7 状态机驱动，并由自动化 lint 门禁守护。

```
Stage 1 规划 → 2 算法 → 3 架构 → 4 设计
        → 5 构造 → 6 验证 → 7 调优
```

本仓库提供：

- **`AGENTS.md`** — 项目入口，项目概述、通用原则、入口路径速查
- **`.opencode/agents/`** — agent 定义（算子开发团队为 1 个 primary 编排者 + 8 个专职 sub-agent）
- **`.opencode/plugins/`** — 状态机插件与 OL lint 守卫（TypeScript）
- **`.agents/skills/`** — 专家技能集，索引见 [`skills/README.md`](skills/README.md)
- **`.agents/hooks/pypto-op-lint/`** — Python lint 引擎：58 条规则、5 个维度，在每次文件写入与每次测试执行后自动运行

同时支持 [opencode](https://opencode.ai) 与 [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview)。

> **使用 Claude Code 的用户请先参考 [Claude Code 配置](#claude-code-配置)** 完成一次性迁移。opencode 用户无需任何额外配置。

---

## 快速上手 — 实际怎么跑起来

> **最关键的一步**：会话开始时，切换到 `pypto-op-orchestrator` agent。如果跳过这一步，默认 agent 不会调度 9 智能体团队，会自己包揽所有工作。

| 工具 | 切换到编排者的方法 |
|:---|:---|
| **opencode** | 仓库根目录启动 `opencode`，按 **`Tab`**，选 `pypto-op-orchestrator`。无需任何安装 — agent / skill / plugin / lint 全部自动发现。 |
| **Claude Code** | 需要一次性配置（见下方 [Claude Code 配置](#claude-code-配置)）。然后在仓库根目录运行 `claude --agent pypto-op-orchestrator`。 |

接着用自然语言描述算子（数学公式、规格文档或论文链接）。编排者会读取 `AGENTS.md`，加载操作手册（[`pypto-orchestration-manual`](skills/pypto-orchestration-manual/SKILL.md)），按 Stage 1–7 推进 — 在每个 stage 调度对应专职 agent，并在每次 Stage Stop 时跑 OL lint 引擎。

如果你跳过这一步，对默认 agent 直接说「帮我做一个 PyPTO 算子」，得到的是单 agent 答复，会绕过门禁、lint 引擎和状态机。**永远先切换到编排者。**

> 本仓库另有一个 primary 编排者 [`pypto-pro-op-orchestrator`](../.opencode/agents/pypto-pro-op-orchestrator.md)，走 PyPTO-Pro Stage 1–4 路线，配套 `pypto-pro-*` 系列 skill。本文描述的是 `pypto-op-orchestrator` 的 Stage 1–7 全流程，两者不要混用。

---

## 快速开始

切换到编排者之后，用以下任一方式描述目标。

**方式 1 — 数学公式**

```
我要开发一个名为 sinh 的算子。公式：(e^x - e^(-x)) / 2。
输入：shape 为 [b, s, n, d] 的 float32 tensor。输出 shape 相同。
精度：atol=2.5e-5, rtol=5e-3。
```

**方式 2 — 算子规格文档**

```
请根据 ./docs/my_operator_spec.md 中的方案文档开发对应的 PyPTO 算子。
```

**方式 3 — 算子论文**

```
请将 https://arxiv.org/abs/2205.14135 (Flash Attention) 中描述的算子实现为 PyPTO kernel。
```

编排者将自动按 Stage 1–7 推进。

更多需求示例见 [`user_in.md`](user_in.md)。

---

## 9 智能体团队

| Agent | Mode | Stage | 职责 |
|:---|:---:|:---:|:---|
| [`pypto-op-orchestrator`](../.opencode/agents/pypto-op-orchestrator.md) | primary | 1–7 | 入口。推进 stage、强制门禁、调度 sub-agent。本身不直接做领域工作。 |
| [`pypto-op-planner`](../.opencode/agents/pypto-op-planner.md) | subagent | 1 | 将用户需求翻译为 `SPEC.md` + `API_REPORT.md`；初始化 `MEMORY.md`。 |
| [`pypto-op-mathematician`](../.opencode/agents/pypto-op-mathematician.md) | subagent | 2 | 产出 PyPTO 友好的 `<op>_golden.py` 参考实现与 Golden 函数清单。 |
| [`pypto-op-architect`](../.opencode/agents/pypto-op-architect.md) | subagent | 3 | 产出 `DESIGN.md`：tiling 策略、loop 结构、性能目标表。 |
| [`pypto-op-designer`](../.opencode/agents/pypto-op-designer.md) | subagent | 4 | 将 kernel 拆分为语义模块，定义 `module_interfaces.yaml`。 |
| [`pypto-op-coder`](../.opencode/agents/pypto-op-coder.md) | subagent | 5 | 每次调度只写一个 impl 文件。先 per-module 累计构建 (`modules/<op>_module<k>_impl.py`)，最后一个模块通过 verify 后做 cleanup 把累计 impl 整理成 `<op>_impl.py` 并写 `README.md`。从不写测试，从不调试。 |
| [`pypto-op-verifier`](../.opencode/agents/pypto-op-verifier.md) | subagent | 4–7 | 仅评判。运行 `detailed_tensor_compare`、布局检查、prefix-eval、回归检查。分类失败原因。从不调查、从不修复。 |
| [`pypto-op-debugger`](../.opencode/agents/pypto-op-debugger.md) | subagent | 5（按需） | 一次加载一个调试子技能，定位根因，给出补丁建议。补丁由 coder 应用。 |
| [`pypto-op-optimizer`](../.opencode/agents/pypto-op-optimizer.md) | subagent | 7 | Stage 7 性能调优执行者。按编排器的 stage 参数加载 skill `pypto-op-perf-tune` 分步执行（S1 环境检查 / S2 数据采集 / S3 性能分析 / S4_FRONTEND 开箱调优 / S4_SWIMLANE 深度调优 / S4_INCORE 核内调优），返回结构化结果供编排者验证 |

Agent 之间通过两个产物交换信息：
- `custom/<op>/MEMORY.md` — 共享叙事（所有 agent 读写）
- `custom/<op>/.orchestrator_state.json` — 机器可读状态（仅编排者写入；由 lint 插件强制保护）

更多细节（职责边界、完成判据、信息隔离）见 [`AGENTS.md`](../AGENTS.md) 与操作手册 [`pypto-orchestration-manual`](skills/pypto-orchestration-manual/SKILL.md)。

> 除上述 9 个 agent 外，本仓库还有 [`pypto-kernel-validator`](../.opencode/agents/pypto-kernel-validator.md)（算子产物校验：反作弊 + 精度 + 性能，输出统一 JSON 报告）与 PyPTO-Pro 路线的 [`pypto-pro-op-orchestrator`](../.opencode/agents/pypto-pro-op-orchestrator.md)，二者不属于 Stage 1–7 团队编制。

---

## Stage 1–7 工作流

| Stage | 名称 | Agent | 输入 | 产出 |
|:---:|:---|:---|:---|:---|
| 1 | Planning | planner | 用户需求 | `SPEC.md`, `API_REPORT.md` |
| 2 | Algorithm | mathematician | `SPEC.md` | `<op>_golden.py` |
| 3 | Architecture | architect | `<op>_golden.py` | `DESIGN.md` |
| 4 | Design | designer（+ verifier 搭脚手架） | `DESIGN.md` | `module_interfaces.yaml`、scaffolding |
| 5 | Construction | coder ↔ verifier ↔ debugger | `module_interfaces.yaml` | 每次 Phase M_k 产出一个 `_module<k>_impl.py`；最后 cleanup 阶段产出 `<op>_impl.py`、`test_<op>.py`、`README.md` |
| 6 | Verification | verifier | `<op>_impl.py` | 布局 / 结构 / 端到端 PASS/FAIL 裁决 |
| 7 | Optimization | optimizer + verifier | Stage 6 已通过 | 优化后的 impl |

### Stage 5 内部循环（per-module M_k）

```
coder 写模块 M_k
        │
        ▼
verifier 评判（detailed_tensor_compare + prefix-eval --up-to-module k + 布局检查）
        │
        ├── PASS ──► 下一个模块 M_{k+1}
        │
        └── FAIL ──► debugger 调查
                          │
                          ▼
                     在 MEMORY.md 中给出补丁建议
                          │
                          ▼
                     coder 应用补丁 ──► 回到 verifier
```

M_{k+1} 不能在 M_k 通过之前启动。

---

## Agent 跑出来的产物

每个算子在 `custom/<op>/` 目录下生成。下方是 Stage 1–7 完整跑通后的文件结构，已标注每个文件由哪个 stage 产出。

```
custom/<op>/
├─ MEMORY.md                              ← 共享叙事；所有 agent 读写（编排者在 S1 初始化）
├─ .orchestrator_state.json               ← 机器可读状态机（仅编排者写入）
│
├─ SPEC.md                                ← S1：结构化算子规格（公式、shape、dtype、容差）
├─ API_REPORT.md                          ← S1：PyPTO API 映射、不支持算子、规避方案
│
├─ <op>_golden.py                         ← S2：纯 PyTorch fp32 参考实现 + Golden 函数清单
│
├─ DESIGN.md                              ← S3：tiling 策略、loop 结构、数值稳定性档案、
│                                              Layers A–L、性能目标表
│
├─ module_interfaces.yaml                 ← S4：语义模块分解 + 每模块契约
│
├─ modules/
│  ├─ <op>_module<k>_golden.py            ← S4：每模块 torch golden（verifier 搭脚手架时建立）
│  ├─ <op>_module<k>_impl.py              ← S5：每模块 PyPTO impl（每次 Phase M_k 调度产一个）
│  └─ test_<op>_module<k>.py              ← S4：每模块测试（adversarial harness，verifier 编写）
│
├─ <op>_impl.py                           ← S5（cleanup）：集成 PyPTO impl（最终 kernel）
├─ test_<op>.py                           ← S5（cleanup）：端到端测试
├─ README.md                              ← S5（cleanup）：算子级 README（用法、配置、已知约束）
│
└─ eval/
   ├─ module_interfaces.yaml              ← S4：机器可读契约（lint OL50 / OL51 交叉验证）
   ├─ adversarial_cases.json              ← S4：verifier 生成的边界用例
   ├─ evaluation_report.json              ← S5/S6：verifier 裁决（PASS/FAIL + failure_category）
   └─ prefix_eval_results.json            ← S5：prefix 评测（<op>_impl.py 至 module k）
```

**Stage 7 不会新增文件** — 它原地修改 `<op>_impl.py`。优化报告追加到 `MEMORY.md`；中间产物（swimlane / leafhash dump）按需生成在同级的 `<op>_perf/` 目录下。

| Stage | 新增文件 | 一句话说明 |
|:---:|:---|:---|
| 1 | `MEMORY.md`, `SPEC.md`, `API_REPORT.md` | 框定问题与 API 表面 |
| 2 | `<op>_golden.py` | kernel 必须匹配的位级精度参考 |
| 3 | `DESIGN.md` | 实现方案（tiling、loop、稳定性） |
| 4 | `module_interfaces.yaml`、`modules/<op>_module<k>_golden.py`、`modules/test_<op>_module<k>.py`、`eval/adversarial_cases.json` | 契约 + 脚手架，使每个模块能独立构造与评判 |
| 5 | `modules/<op>_module<k>_impl.py`、`eval/prefix_eval_results.json`；最后一个 M_k 通过后 cleanup：`<op>_impl.py`, `test_<op>.py`, `README.md` | 每模块 impl 累计构建；最后整理出最终 kernel + 端到端测试 |
| 6 | （无 — 仅做门禁） | 布局、结构性、端到端精度门禁 |
| 7 | （修改 `<op>_impl.py`） | 调优后的 kernel；性能报告写入 `MEMORY.md` |

Lint 引擎会把 `eval/module_interfaces.yaml` 中的契约与 impl 做交叉验证（OL50 参数顺序、OL51 输出数量）、强制 `_golden.py` / `_impl.py` / `test_*.py` 三文件分离（D3 规则）、并约束 impl 的 Layer 结构（D1 规则，如 OL45 / OL57 / OL58）。任一不通过，对应的 Write/Edit 在工具边界即被拦截 — 详见下方 [Lint 与状态机](#lint-与状态机--自动门禁)。

---

## Skills

每个 skill 是 `.agents/skills/<name>/` 下的一个目录，包含入口 `SKILL.md`，可选子目录 `references/`、`scripts/`、`templates/`。

**完整的 skill 分类索引见 [`skills/README.md`](skills/README.md)。** 与 Stage 1–7 团队直接相关的入口：

| Stage | Skill |
|:---:|:---|
| 编排 | [`pypto-orchestration-manual`](skills/pypto-orchestration-manual/SKILL.md) — 编排者启动手册；[`pypto-memory-template`](skills/pypto-memory-template/SKILL.md) — `MEMORY.md` 必填结构 |
| 1 | [`pypto-intent-understand`](skills/pypto-intent-understand/SKILL.md)、[`pypto-api-explore`](skills/pypto-api-explore/SKILL.md)、[`pypto-op-plan`](skills/pypto-op-plan/SKILL.md) |
| 2 | [`pypto-golden-generate`](skills/pypto-golden-generate/SKILL.md) |
| 3 | [`pypto-op-design`](skills/pypto-op-design/SKILL.md) |
| 4–5 | [`pypto-op-construct`](skills/pypto-op-construct/SKILL.md)、[`pypto-op-develop`](skills/pypto-op-develop/SKILL.md) |
| 5–6 | [`pypto-op-verify`](skills/pypto-op-verify/SKILL.md)、[`pypto-op-review`](skills/pypto-op-review/SKILL.md) |
| 7 | [`pypto-op-perf-tune`](skills/pypto-op-perf-tune/SKILL.md)（子技能：[`tune-frontend`](skills/pypto-op-perf-tune/tune-frontend/SKILL.md)、[`tune-swimlane`](skills/pypto-op-perf-tune/tune-swimlane/SKILL.md)、[`tune-incore`](skills/pypto-op-perf-tune/tune-incore/SKILL.md)、[`perf-analyzer`](skills/pypto-op-perf-tune/perf-analyzer/SKILL.md)） |
| 调试与精度 | [`pypto-general-debug`](skills/pypto-general-debug/SKILL.md)、[`pypto-precision-compare`](skills/pypto-precision-compare/SKILL.md)、[`pypto-precision-debug`](skills/pypto-precision-debug/SKILL.md) |
| 知识与资料 | [`pypto-op-knowledge`](skills/pypto-op-knowledge/SKILL.md)、[`pypto-docs-search`](skills/pypto-docs-search/SKILL.md) |

---

## Lint 与状态机 — 自动门禁

两个 opencode 插件在每次 tool 调用时自动运行。**任何 agent 都不需要手动调用它们。**

### `pypto-op-lint.ts`

在每次相关的 tool 事件后运行 OL lint 引擎（`.agents/hooks/pypto-op-lint/`）：

| 事件 | 触发条件 | 执行内容 |
|:---|:---|:---|
| `tool.execute.after`（Write/Edit） | 文件名匹配 `*_impl.py`、`*_golden.py` 或 `test_*.py` | `post-edit` hook — 用 58 条 OL 规则校验；命中 S0/S1 规则时**拦截**本次 tool 调用 |
| `tool.execute.after`（Bash） | 命令匹配 `python test_*.py` | `post-bash` hook — 解析 stdout/stderr/exit code 并产出裁决 |
| `tool.execute.before`（Bash） | 命令尝试写入 `.orchestrator_state.json` | **拦截** — 该文件仅允许编排者通过状态机插件修改 |

规则定义在 [`hooks/pypto-op-lint/rules.json`](hooks/pypto-op-lint/rules.json)（v2.0.0，58 条）。五个维度：

| 维度 | 条数 | 覆盖范围 |
|:---|:---:|:---|
| **D1** | 24 | 框架约束合规 — 装饰器、签名 shape、JIT 要求、Layer 结构 |
| **D2** | 14 | 工件完整性与流程合规 — 每个 stage 的必备文件 |
| **D3** | 4 | 三文件分离 — golden / impl / test 的边界（golden 中禁止 import pypto 与 `.T`、test 中不含 kernel 实现） |
| **D4** | 6 | 测试规范 — adversarial 覆盖、tolerance 模式（`atol/rtol` 或 `mare/mere/rmse` 矩阵） |
| **D5** | 10 | 跨文件一致性 — spec / design / 模块契约与 impl、test 一致 |

严重级别：S0（致命，10 条）→ S1（必修，35 条）→ S2（警告，11 条）→ S3（信息，2 条）。

### `pypto-state-transition.ts`

守护 stage 间转移，发出 Phase M_k 循环事件。支持 `rollback_to_stage`，让单个失控模块不会污染其余流水线。核心逻辑放在 `lib/state-transition-core.ts`，与 opencode 解耦，可独立单测。

两个插件都在 `.opencode/plugins/__tests__/` 下提供单元测试。

---

## Claude Code 配置

opencode 自动发现 `.opencode/agents/`、`.agents/skills/`、`.opencode/plugins/`。Claude Code 使用不同的目录结构（`.claude/agents/`、`.claude/skills/`、`CLAUDE.md`、`.claude/settings.json`），需要一次性迁移配置。

> **一次性配置**，在仓库根目录运行。仅在新增 agent/skill 或刷新 hook 配置时需要重跑。

### 1. 创建 `.claude/` 目录结构

```bash
# 1. 创建 Claude Code 目录结构
mkdir -p .claude/skills .claude/agents

# 2. 复制项目指令文件
cp AGENTS.md CLAUDE.md

# 3. 复制 Skills 到 Claude Code 目录
cp -r .agents/skills/* .claude/skills/

# 4. 复制 Agents 到 Claude Code 目录
cp -r .opencode/agents/* .claude/agents/

# 5. 复制 lint hook 配置
cp .agents/settings.json .claude/settings.json
```

[`settings.json`](settings.json) 把同一份 Python lint 引擎接到 Claude Code 的 hooks 上（`PostToolUse` → post-edit / post-bash，`PreToolUse` → pre-edit-backup，`Stop` → 交付门禁），与 opencode 下 `pypto-op-lint.ts` 触发的是同一个引擎、同一套 58 条规则。

它**只接线 hook，不预批任何工具权限** —— Claude Code 会按其默认权限模型，在 Write / Edit / Bash 前逐次征求确认（只读操作如文件读取、Grep 默认放行）。如需减少确认次数，自行在 `.claude/settings.json` 中追加 `permissions.allow` 规则，语法见 [Claude Code 权限文档](https://code.claude.com/docs/en/permissions)。

### 2. 指定 Agent 启动 Claude Code

使用 `--agent` 参数让 Claude Code 启动时直接进入编排者：

```bash
claude --agent pypto-op-orchestrator
```

启动后，在对话中描述算子开发任务即可。

### 3. 验证配置

启动后，确认编排者已激活：
- 它的开场白会提到 Stage 1–7 与 9 智能体团队
- 输入 `/agents`，确认列表里能看到全部 9 个 `pypto-op-*` agent
- 编辑任意 `*_impl.py` 写一些违规内容；OL lint hook 应触发并给出告警

如果上述任一项失败，最常见原因：
- 工作目录不在仓库根（Claude Code 从 `cwd` 向上检索 `.claude/`）
- `.claude/settings.json` JSON 格式错误 — 用 `jq . .claude/settings.json` 校验
- agent 的 `mode:` frontmatter 行让 Claude Code 报警告（这是 opencode 专用字段）。如果遇到，按下行去掉 `mode:`：`for f in .opencode/agents/*.md; do sed '/^mode:/d' "$f" > ".claude/agents/$(basename "$f")"; done`

> **当前限制**：**状态机不随上述步骤迁移**。`.opencode/plugins/pypto-state-transition.ts` 向 opencode 注册的是一个 `state_transition` **工具**，而 Claude Code 不加载 `.opencode/plugins/`；hooks 也替代不了工具 —— hook 由生命周期事件自动触发，模型无法主动调用它。因此当前在 Claude Code 下，lint 门禁可用（步骤 5），但 stage 迁移与 `rollback_to_stage` 需要人工把关。
>
> 这是**尚未实现**，而非做不到。Claude Code 支持自定义工具，正规路径是 [MCP server](https://code.claude.com/docs/en/mcp)：把 `state_transition` 以 MCP 协议（stdio）暴露，可单独注册，也可通过 [plugin](https://code.claude.com/docs/en/plugins-reference) 的 `mcpServers` 字段随插件分发（工具名形如 `mcp__plugin_<plugin>_<server>__state_transition`）。改造成本不高：核心逻辑已解耦在 [`lib/state-transition-core.ts`](../.opencode/plugins/lib/state-transition-core.ts)（483 行，零外部 import），opencode 耦合只存在于薄适配层，换一个 MCP 适配层即可复用。欢迎贡献。

---

## 单技能模式（不走编排者）

如果只想用某一项能力 — 比如对已有 kernel 跑一次 `pypto-precision-compare` — 可以不通过编排者，直接调用单个 skill。它读取的是同一份 `SKILL.md`，但不会推进 Stage 1–7，也不会触发 lint 门禁。

这种模式适合诊断、orchestrated run 之后的补丁修复，以及一次性的分析工作。

---

## 仓库布局

```
pypto-gym/
├─ AGENTS.md                          ← 项目入口（项目概述、通用原则、入口速查）
├─ .opencode/
│  ├─ agents/                         ← agent 定义（markdown frontmatter + body）
│  └─ plugins/                        ← 状态机 + OL lint 插件（TypeScript + 单测）
└─ .agents/
   ├─ README.md                       ← 当前文件
   ├─ skills/                         ← 专家技能，每个含 SKILL.md（索引见 skills/README.md）
   ├─ hooks/pypto-op-lint/            ← Python lint 引擎（58 规则、12 单测、JSON 事件日志）
   ├─ settings.json                   ← Claude Code lint hook 配置（拷到 .claude/settings.json）
   └─ user_in.md                      ← 用户需求提示词模板
```

---

## 常见问题

<details>
<summary><b>AGENTS.md、Skills 和 Agents 有什么区别？</b></summary>

| 维度 | AGENTS.md | Skills | Agents |
|:---|:---|:---|:---|
| 作用 | 项目级自定义规范 | 特定任务的执行流程 | 编排和隔离执行复杂任务 |
| 加载方式 | 自动加载，对所有对话生效 | 按需加载，调用时才生效 | Orchestrator 主导，Subagent 被调度 |
| 内容 | 通用开发规范和原则 | 具体任务的步骤、工具、验证标准 | 状态机、工件契约、重试策略 |
| 执行模式 | 规则约束 | 直接执行 | Primary 编排 + Subagent 隔离执行 |

三者配合使用：AGENTS.md 定义"怎么做才对"，Skills 定义"怎么一步步做完"，Agents 定义"怎么编排和隔离执行"。

</details>

<details>
<summary><b>什么时候用 Orchestrator，什么时候直接用 Skill？</b></summary>

- **完整算子开发**：切换到 `pypto-op-orchestrator` agent，走 Stage 1–7
- **单步任务**：直接调用对应 Skill，如只需生成 Golden 就调用 `pypto-golden-generate`
- **调试修复**：直接调用调试类 Skill，如 `pypto-precision-debug`、`pypto-general-debug`

</details>

<details>
<summary><b>其他 AI 工具兼容性</b></summary>

本项目支持多种 AI 编程工具，包括 [OpenCode](https://opencode.ai)、[Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview)、Cursor、Codex 等。

**Claude Code 目录结构映射**：

| 组件 | OpenCode | Claude Code |
|:---|:---|:---|
| 项目指令 | `AGENTS.md` | `CLAUDE.md` |
| Skills | `.agents/skills/` | `.claude/skills/` |
| Agents | `.opencode/agents/` | `.claude/agents/` |
| Hook | `.opencode/plugins/pypto-op-lint.ts` | `.claude/settings.json`（由 `.agents/settings.json` 拷贝，仅含 lint hook） |
| 自定义工具 | `.opencode/plugins/` 直接注册（如 `state_transition`） | 须经 [MCP server](https://code.claude.com/docs/en/mcp) 暴露，可打包进 plugin 的 `mcpServers`；本仓尚未提供 |

**格式兼容性**：
- **SKILL.md**：YAML frontmatter + Markdown，两种工具完全兼容
- **Agents**：YAML frontmatter + Markdown，`mode: primary` 为 OpenCode 特有字段，Claude Code 会忽略

</details>

<details>
<summary><b>Lint / 门禁没有触发怎么办？</b></summary>

两种工具共享同一份 Python lint 引擎（`.agents/hooks/pypto-op-lint/pypto_op_lint.py`），但触发方式不同：

- **opencode**：由 `.opencode/plugins/pypto-op-lint.ts` 自动加载，仓库根目录启动时自动发现，无需配置
- **Claude Code**：由 `.claude/settings.json` 中的 hooks 配置触发（从 `.agents/settings.json` 拷贝，见 [Claude Code 配置](#claude-code-配置) 步骤 5）

如果仍未触发：
- 确认工作目录在仓库根（插件与 hook 均按仓库根解析）
- 确认被编辑的文件名匹配 `*_impl.py`、`*_golden.py` 或 `test_*.py` — 只有这三类文件会触发 post-edit lint
- 手动跑一次 lint 引擎复现：`python3 .agents/hooks/pypto-op-lint/pypto_op_lint.py --help`
- Claude Code 下另需确认 `.claude/settings.json` 已拷贝且 JSON 合法（`jq . .claude/settings.json`）

**状态机**（stage 迁移、`rollback_to_stage`）目前仅在 opencode 下生效 —— 它是一个工具而非 hook，Claude Code 下需经 MCP server 暴露，本仓尚未提供。详见 [Claude Code 配置](#claude-code-配置) 的当前限制。

</details>

---

## 进一步阅读

- [`AGENTS.md`](../AGENTS.md) — 项目概述、通用原则、入口路径速查
- [`skills/README.md`](skills/README.md) — 全部 skill 的分类索引与使用指南
- [`pypto-orchestration-manual`](skills/pypto-orchestration-manual/SKILL.md) — 编排者启动手册（调试编排逻辑时先读它）
- [`hooks/pypto-op-lint/rules.json`](hooks/pypto-op-lint/rules.json) — 权威 OL 规则集（v2.0.0，58 条）
