# CANNBot PyPTO-Pro 算子开发快速入门指南

## 概述

CANNBot PyPTO-Pro 算子开发模式适用于通过 PyPTO-Pro 开发 Ascend NPU 算子。默认由 agent 自主安排开发与验证；用户明确要求深度编排时，采用 5 阶段流程。

### 与 PyPTO 开发的区别

| 对比维度 | PyPTO-Pro 算子开发（本模式） | PyPTO 算子开发 |
|---------|---------------------------|--------------|
| 适用场景 | PyPTO-Pro 框架算子开发 | PyPTO 框架算子开发 |
| 编程语言 | Python（PyPTO-Pro API） | Python（PyPTO API） |
| 开发内容 | PyPTO-Pro kernel + golden + test | PyPTO kernel + golden + test |
| 开发流程 | 默认自主开发；深度编排为 5 阶段流程 | 7 阶段状态机驱动 |
| 状态管理 | 深度编排使用 `.orchestrator_state.json`（`state_transition` 工具 + 阶段门禁） | `.orchestrator_state.json`（`state_transition` 工具 + lint 门禁） |
| 性能调优 | 按需优化；深度编排在 Stage 5 执行 | Stage 7 独立调优阶段 |

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

> **自动门禁支持边界**：`state_transition` 工具、写入前 lint 和 Stage/Module 自动硬门禁目前通过 OpenCode 插件提供。下列其他工具的 `init.sh` 适配仅安装普通 Pro 的 skills、agents 与提示词资源，不提供 Scriptor CLI、下游 worker/verifier 或 Scriptor 验收流程。需要完整 5 阶段状态机、fail-closed lint 或 Scriptor 模式时，请使用 OpenCode。

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
# 应看到 pypto-pro-op-planner / pypto-pro-op-mathematician / pypto-pro-op-architect / pypto-pro-op-coder / pypto-pro-op-optimizer / pypto-pro-op-verifier

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

在交互界面中输入算子开发需求，agent 默认自主安排开发与验证：

```
使用 PyPTO-Pro 开发 softmax 算子，支持 [1, 128]、[4, 2048] 和 [32, 4096] 的 float16 输入。
```

### Scriptor 模式（OpenCode）

拉取更新后，使用已有 Python >=3.10，通过本插件的 `init.sh project opencode <项目路径>` 安装或刷新一次资源；工作流资源是项目副本，刷新时需重跑入口；Ascriptor 编译器始终从本插件源码快照直接加载。此入口注册 CLI、精度调试技能及下游 worker/verifier。使用以下提示词即可选路：

```text
使用 scriptor 模式开发这个 PyPTO-Pro 算子：<公式、接口、shape/dtype、输入约束>。
前段只要求可编译上板，精度最多修复 3 轮；后段启用优化，遵循入口默认轮数。
```

主入口读取 [scriptor-mode.md](references/scriptor-mode.md)：需求/Golden/设计/可上板原型 → DSL implement → optimize → accept → delivery-check。前段无性能开发、无独立 prepare 或 verifier；原型由主 agent 参考源码快照的 ROUTER/playbook 文档与 DESIGN.md 直接编写，满足 SPEC 且编译上板跑通；精度不通过时全算子最多修复 3 轮，3 轮后未解决的精度问题随证据交接。DSL 候选及最终验收仍须通过精度。后段按用户要求优化；无可比用户 baseline 时以 NPU Golden 为参考、首个正确 PyPTO 候选用于排名，未指定目标时默认每个 P0 ≥1.0×。Scriptor optimize 默认至少 10 轮有效优化且无上限，必须完成基础优化和 Active 卡片逐卡覆盖；优化阶段复用冻结合同和已有证据，accept 后不强制再调用完整 `pypto-pro-op-perf-tune`，用户明确要求独立调优时才另行发起。用户明确的停止、预算或不优化要求优先，具体规则见入口。
新 Scriptor 交付默认从首个 DSL 候选到最终包只生成 `auto_mutex=True`；仅用户明确要求 manual 时在初始化状态记录授权并显式生成 manual。auto_mutex 失败时保留证据并报告阻断，不自动回退 manual；底层 Ascriptor API 的历史默认不决定本工作流的交付模式。
`custom/<op>/` 保留 Scriptor 状态与原始证据；最终精简可运行包放在独立的
`delivery/<op>/`，含 `kernels/`、`golden_cpu.py`、`wrapper.py`、`test.py`、
最终 `DESIGN.md`、`REPORT.md` 与测试必需的本地依赖。用
`delivery-check --op-dir custom/<op> --delivery-dir delivery/<op>` 核对并在隔离副本
真机执行交付入口；只有返回 `status=PASS` 才算本轮通过。仅有 `generated/`、
`test_<op>.py`、历史 `reports/final/` 或静态 `STRUCTURE_PASS` 不代表交付完成。

agent 保留原型副本，公开 wrapper 由 DSL 导出器接管；Scriptor 阶段报告在
`custom/<op>/reports/final/final.md`。优化账本和轮次证据随算子保留。
OpenCode 初始化只注册工作流、Skills、Hooks 和 CLI，不安装或复制 Ascriptor 编译器。
源码直接取自本插件的 `resources/ascriptor/`；数值运行需要 Torch/NumPy，真机还需要
CANN、PyPTO-Pro、torch_npu 与 msprof。

维护者修改快照后运行 `scripts/sync_ascriptor_sources.py` 刷新完整文件索引，
并用 `--check` 验证 SHA-256、Markdown 链接、inline 脚本命令和关键 JSON 路径。
`product-view.json` 声明此自包含快照的交付范围；`sources.json` 与
`sources-index.json` 是可复核的源码身份。无需获取私有仓库或 wheel。
DSL worker 从 `doctor.source_root/agent` 执行
`python tools/build_kernel_context.py --print`，直接读取浓缩起点，无需来源哈希校验。
首发 0.1.0 是源码交付；设备和性能结果需按当前任务独立验证。

### 深度编排工作流（按需）

需要原 5 阶段流程时，在上述提示词前加上“启用深度编排”。agent 按 [入口说明](AGENTS.md) 自动加载 [编排协议](references/orchestration.md)；在 OpenCode 中开启自动 lint 并初始化或恢复状态机。

```
Stage 1: 需求规划与资料索引 → Stage 2: NPU/CPU Golden（性能采集可选）
    → Stage 3: Tile 数据流设计 → Stage 4: Kernel 实现与精度验证
    → Stage 5: 可比基线驱动的性能优化与证据验收
```

Stage 1、3、4、5 通过 verifier 后才可推进；Stage 2 在 mathematician 满足 golden skill 完成条件后，经 `complete_stage(2)` 的原有 lint 门禁推进，用户明确要求 Stage 2 verifier 时才增加独立复核。失败项交回对应子代理修正。深度编排使用 `custom/<op>/.orchestrator_state.json` 状态机推进 Stage（`state_transition` 工具由 OpenCode 插件提供）。详见 [编排协议的“共享状态与 state_transition 工具”](references/orchestration.md)。

Stage 2 默认生成并验证 `{op}_golden.py`（NPU）与 `{op}_golden_cpu.py`（CPU FP32），并生成 `GOLDEN_VALIDATION.json` 回执，不采集 NPU golden 性能。用户明确要求“采集 NPU golden 性能”时，必须额外生成 `GOLDEN_PERF_REPORT.md`；性能采集不自动启用 Stage 2 verifier。Stage 2 不负责 Stage 5 机器合同。若 SPEC 未记录用户数值目标，Stage 5 optimizer 会使用 `pypto-pro-op-perf-tune` 自带的 `collect_golden_reference.py`，基于现有 Golden 和冻结 case 清单生成一次 Stage 5 专用报告，不回滚或重跑 Stage 2。

Stage 5 优先采用用户在 SPEC 中明确给出的可复算性能目标。用户未给数值目标时，把 `PERFORMANCE_CASES.json` 中每个 P0 case 的 `Golden 每迭代 NPU E2E / PyPTO 最终 target-kernel >= 1.0` 作为默认理想参考。该比值不是 PyPTO baseline→final 同口径加速比；用户目标和默认理想参考都不是能否交付的硬门禁，达到、未达到或不可用都必须如实报告。

Stage 5 由 optimizer 完整加载 `pypto-pro-op-perf-tune`：审计 selected KB 全部原子点并补齐
未落实缺口，闭合全部 Active 知识卡及 active/eligible 通用方法与模板项；四类预置来源全部闭合后，按当前
瓶颈建立并闭合 `bottleneck_derived`；冻结 baseline 与正式评价候选只在正确、合规时纳入排名，
最终保留冻结聚合指标最优版本，再独立重采并验收。
采集协议、来源账本、逐项实验、最佳版本选择、停止条件和交付物均以
[`pypto-pro-op-perf-tune`](../../ops/pypto-pro-op-perf-tune/SKILL.md) 为准；Stage 5 修复只在本 Stage 内完成，不回退。

### 深度编排产出物示例

自主开发的产物和目录按用户交付要求确定。以下为深度编排的 `custom/<op>/` 产物示例，按实际执行的阶段和选项生成：

```
custom/<op>/
├── SPEC.md                    # 需求规格
├── EXPLORE_REPORT.md          # 资料探索与可行性报告
├── PRO_MATERIAL_INDEX.md      # PyPTO-Pro 资料索引
├── {op}_golden.py             # NPU Golden 参考实现
├── {op}_golden_cpu.py         # CPU 更高精度 Golden（供精度校验）
├── GOLDEN_VALIDATION.json     # 与当前 SPEC/Golden 哈希绑定的成功回执
├── GOLDEN_PERF_REPORT.md      # Stage 2 用户按需报告，或 Stage 5 默认目标人读报告
├── GOLDEN_PERF_REPORT.json    # 仅 Stage 5 默认目标分支生成的机器合同
├── DESIGN.md                  # Tile 数据流设计文档
├── test_{op}.py               # Stage 4 kernel；Stage 5 保留冻结聚合指标最优的正确、合规版本
├── PERFORMANCE_CASES.json     # Stage 5 执行 case 清单（SPEC P0 与既有测试一一对应）
├── PERFORMANCE_REPORT.md      # Stage 5 baseline/final、来源账本、候选比较与关闭记录
├── performance.json           # Stage 5 结构化性能结果
├── performance.log            # Stage 5 采集日志
├── perf_report.md             # Stage 5 脚本分析报告
└── docs/perf/round_NNN/       # Stage 5 七指标与逐 case instruction timeline 归档
```

## 三、可用技能

自主开发按需使用专业资源；完整 skills 列表见 [入口的可用资料](AGENTS.md)。下表列出深度编排的阶段分工。

| Skill | 用途 | 触发阶段 |
|-------|------|---------|
| `pypto-pro-op-plan` | 串行组织需求理解与资料探索 | Stage 1 |
| `pypto-pro-intent-understand` | 需求意图理解与规格生成 | Stage 1 |
| `pypto-pro-material-explore` | 资料索引与可行性探索 | Stage 1 |
| `pypto-pro-environment-check` | 环境检查与 smoke 测试 | 按需 |
| `pypto-pro-golden-generate` | Golden 参考实现生成与用户按需 NPU 性能采集 | Stage 2 |
| `pypto-pro-op-design` | Tile 数据流设计 | Stage 3 |
| `pypto-pro-op-develop` | Kernel 实现与精度验证 | Stage 4 |
| `pypto-pro-op-perf-tune` | 默认 Golden 目标采集、性能分析与优化闭环 | Stage 5 |
| `pypto-pro-docs-search` | Pro API、指南与指定样例的独立缓存装配及检索 | 会话开始装配，之后按需检索 |

| Agent | 用途 | 负责阶段 |
|-------|------|---------|
| `pypto-pro-op-planner` | 需求理解与资料索引 | Stage 1 |
| `pypto-pro-op-mathematician` | NPU/CPU Golden；按需采集基准性能 | Stage 2 |
| `pypto-pro-op-architect` | Tile 数据流设计 | Stage 3 |
| `pypto-pro-op-coder` | Kernel 实现与精度验证 | Stage 4 |
| `pypto-pro-op-optimizer` | 可比证据驱动的性能优化 | Stage 5 |
| `pypto-pro-op-verifier` | 独立阶段检查，调度规则见 [编排协议](references/orchestration.md) | Stage 1–5 |

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
| 使用 PyPTO-Pro API 开发算子 | PyPTO-Pro |

---

## 总结

1. 默认自主开发与验证；用户明确要求时启用 5 阶段深度编排
2. 使用 `init.sh` 一键安装（OpenCode 推荐），支持项目级和全局级
3. `opencode` / `claude` 是核心交互指令
4. 按用户需求交付实现与验证结果；深度编排产物写入 `custom/<op>/`
