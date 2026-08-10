# CANNBot PyPTO-Pro 算子开发快速入门指南

## 概述

CANNBot PyPTO-Pro 算子开发模式适用于通过 PyPTO-Pro 开发 Ascend NPU 算子。采用 4 阶段工作流驱动，覆盖从需求理解到代码实现的完整流程，通过独立 verifier 在每个 Stage 结束时执行检查清单，确保各阶段产出符合质量要求。

### 与 PyPTO 开发的区别

| 对比维度 | PyPTO-Pro 算子开发（本模式） | PyPTO 算子开发 |
|---------|---------------------------|--------------|
| 适用场景 | PyPTO-Pro 框架算子开发 | PyPTO 框架算子开发 |
| 编程语言 | Python（PyPTO-Pro API） | Python（PyPTO API） |
| 开发内容 | PyPTO-Pro kernel + golden + test | PyPTO kernel + golden + test |
| 阶段数 | 4 阶段工作流 | 7 阶段状态机驱动 |
| 状态管理 | `.orchestrator_state.json`（`state_transition` 工具 + lint 机械门禁 + verifier 语义门禁） | `.orchestrator_state.json`（`state_transition` 工具 + lint 门禁） |
| 性能调优 | 按需参考 | Stage 7 独立调优阶段 |

## 一、环境搭建

### 前置条件

- 已安装 CANN Toolkit（建议 ≥ 9.0.0），具体版本配套关系请查阅 [CANN Release Notes](https://www.hiascend.com/cann/document)
- 已安装匹配版本的 CANN、torch/torch_npu 和 PyPTO-Pro
- NPU 可见，且 PyPTO-Pro devkit 资料能够按 skill 指引装配
- 已安装 OpenCode、Claude Code、TRAE、Cursor、Copilot、CodeArts 等受支持的 AI 编程工具

### OpenCode（推荐）

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator
bash init.sh project opencode   # 项目级（默认）
bash init.sh global opencode    # 全局级
```

### 其他工具（资源安装，不含自动状态机/硬门禁）

> **自动门禁支持边界**：`state_transition` 工具、写入后 lint 和 Stage/Module 自动硬门禁目前通过 OpenCode 插件提供。下列其他工具的 `init.sh` 适配仅安装 skills、agents 与提示词资源，不会获得同等的 OpenCode 自动门禁能力。需要完整 4 阶段状态机和 fail-closed lint 流程时，请使用 OpenCode。

<details>
<summary>Claude Code</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator
bash init.sh project claude     # 项目级
bash init.sh global claude      # 全局级
```

</details>

<details>
<summary>TRAE</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator
bash init.sh project trae       # 项目级
bash init.sh global trae        # 全局级
```

安装后自动检测 TRAE 环境，生成 `.trae/`（TRAE IDE）、`.marscode/`（TRAE Plugin）或 `.traecli/`（TRAE CLI）目录，结构与 Claude/OpenCode 基本一致。

</details>

<details>
<summary>Cursor</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator
bash init.sh project cursor     # 项目级
bash init.sh global cursor      # 全局级
```

安装后在项目根目录生成 `.cursor/` 目录，结构与 Claude/OpenCode 基本一致。

</details>

<details>
<summary>Copilot</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator
bash init.sh project copilot    # 项目级
bash init.sh global copilot     # 全局级
```

安装后在项目根目录生成 `.github/` 目录（项目级）或 `~/.copilot/` 目录（全局级），AGENTS.md 自动注入 VS Code Copilot 上下文。

</details>

<details>
<summary>CodeArts</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator
bash init.sh project codearts     # 项目级
bash init.sh global codearts      # 全局级
```

安装后在项目根目录生成 `.codeartsdoer/` 目录（项目级）或 `~/.codeartsdoer/` 目录（全局级），包含 skills/、agents/ 和 AGENTS.md。

</details>

### 在其他目录执行

`init.sh` 支持通过完整路径调用，无需先 `cd` 到插件目录。第三个参数指定目标项目路径，省略则安装到当前目录：

```bash
# 安装到当前目录
bash /path/to/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/init.sh project opencode

# 安装到指定项目
bash /path/to/pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator/init.sh project opencode /path/to/your_project_path
```

### 验证安装

```bash
# OpenCode
opencode agent list
# 应看到 pypto-pro-op-planner / pypto-pro-op-mathematician / pypto-pro-op-architect / pypto-pro-op-coder / pypto-pro-op-verifier

# Claude Code
ls .claude/
# 应看到 skills/ agents/ CLAUDE.md cannbot-manifest.json

# TRAE
ls .trae/      # TRAE IDE
ls .marscode/  # TRAE Plugin（init.sh 自动检测）
ls .traecli/   # TRAE CLI（init.sh 自动检测）
# 应看到 skills/ agents/ cannbot-manifest.json
# AGENTS.md 位于项目根目录

# Cursor
ls .cursor/
# 应看到 skills/ agents/ cannbot-manifest.json
# AGENTS.md 位于项目根目录
```

## 二、快速上手

### 启动

```bash
# OpenCode
opencode

# Claude Code
claude
```

> **TRAE 用户**：TRAE 通过 IDE、VS Code 插件或 CLI 启动。init.sh 会自动检测 TRAE IDE（`~/.trae-cn`）、Plugin（`~/.marscode`）或 CLI（`~/.traecli`）并安装到对应目录。安装完成后在 IDE 中直接打开项目即可。
>
> **Cursor 用户**：Cursor 通过 IDE 启动，`.cursor/` 目录中的配置会自动加载。安装完成后在 IDE 中直接打开项目即可。

### 开发算子示例

在交互界面中输入算子开发需求，CANNBot 会自动启动 4 阶段流程：

```
使用 PyPTO-Pro 开发 softmax 算子，支持 [1, 128]、[4, 2048] 和 [32, 4096] 的 float16 输入。
```

### 核心工作流

采用 4 阶段流程，每个 Stage 由独立 verifier 执行检查清单，确保各阶段产出符合质量要求：

```
Stage 1: 需求规划与资料索引 → Stage 2: NPU/CPU Golden（性能采集可选）
    → Stage 3: Tile 数据流设计 → Stage 4: Kernel 实现与精度验证
```

每一阶段通过 verifier 检查后才可进入下一阶段。verifier 失败时，orchestrator 将失败项反馈给对应 Stage 的子代理修正。Pro 流程**使用** `custom/<op>/.orchestrator_state.json` 状态机推进 Stage（`state_transition` 工具，随 OpenCode 插件安装；其他工具下按 AGENTS.md 降级协议手工维护同一账本）。详见 AGENTS.md「共享状态与 state_transition 工具」。

Stage 2 默认只生成并验证 `{op}_golden.py`（NPU）与 `{op}_golden_cpu.py`（CPU FP32），不采集 NPU golden 性能。若需要性能报告，请在需求中明确说明“采集 NPU golden 性能”或“生成 GOLDEN_PERF_REPORT.md”；编排器才会启用 profiling。

### 产出物示例

PyPTO-Pro 算子开发模式下，CANNBot 会在 `custom/<op>/` 目录下生成以下文件：

```
custom/<op>/
├── SPEC.md                    # 需求规格
├── EXPLORE_REPORT.md          # 资料探索与可行性报告
├── PRO_MATERIAL_INDEX.md      # PyPTO-Pro 资料索引
├── MEMORY.md                  # 任务摘要与协作记录
├── {op}_golden.py             # NPU Golden 参考实现
├── {op}_golden_cpu.py         # CPU 更高精度 Golden（供精度校验）
├── GOLDEN_PERF_REPORT.md      # 可选：用户明确要求时生成的 Golden 基准性能报告
├── DESIGN.md                  # Tile 数据流设计文档
└── test_{op}.py               # kernel 与测试单文件
```

## 三、可用技能

| Skill | 用途 | 触发阶段 |
|-------|------|---------|
| `pypto-pro-op-plan` | 串行组织需求理解与资料探索 | Stage 1 |
| `pypto-pro-intent-understand` | 需求意图理解与规格生成 | Stage 1 |
| `pypto-pro-material-explore` | 资料索引与可行性探索 | Stage 1 |
| `pypto-pro-environment-check` | 环境检查与 smoke 测试 | 按需 |
| `pypto-pro-golden-generate` | Golden 参考实现生成 | Stage 2 |
| `pypto-pro-op-design` | Tile 数据流设计 | Stage 3 |
| `pypto-pro-op-develop` | Kernel 实现与精度验证 | Stage 4 |
| `pypto-pro-op-perf-tune` | 性能约束与调优参考 | 按需 |
| `pypto-docs-search` | 算子 API 文档与参考实现检索 | 按需 |

| Agent | 用途 | 负责阶段 |
|-------|------|---------|
| `pypto-pro-op-planner` | 需求理解与资料索引 | Stage 1 |
| `pypto-pro-op-mathematician` | NPU/CPU Golden；按需采集基准性能 | Stage 2 |
| `pypto-pro-op-architect` | Tile 数据流设计 | Stage 3 |
| `pypto-pro-op-coder` | Kernel 实现与精度验证 | Stage 4 |
| `pypto-pro-op-verifier` | 每个 Stage 的独立检查 | Stage 1–4 |

## 四、常见问题

### Q: 如何查看帮助信息？

```bash
bash init.sh --help
```

### Q: 项目级和全局安装如何选择？

- **项目级**：适合多项目开发，每个项目可以有不同配置
- **全局**：适合单一项目，全局生效

### Q: 如何更新？

```bash
cd pypto-gym/cannbot-skills/plugins-official/pypto-pro-op-orchestrator && bash init.sh project opencode
```

### Q: PyPTO 和 PyPTO-Pro 如何选择？

| 场景 | 推荐模式 |
|------|---------|
| 使用 PyPTO 框架开发算子 | PyPTO |
| 使用 PyPTO-Pro API，并按 4 阶段逐项检查 | PyPTO-Pro |

---

## 总结

1. PyPTO-Pro 通过 4 阶段工作流覆盖从需求规划到 Kernel 实现的完整流程：需求规划与资料索引→NPU/CPU Golden（性能采集可选）→Tile 数据流设计→Kernel 实现与精度验证
2. 使用 `init.sh` 一键安装（OpenCode 推荐），支持项目级和全局级
3. `opencode` / `claude` 是核心交互指令
4. 每个 Stage 由独立 verifier 执行检查清单，确保质量门禁
5. 产物全部写入 `custom/<op>/`，含 SPEC、资料报告、Golden、设计文档与 Kernel 实现
