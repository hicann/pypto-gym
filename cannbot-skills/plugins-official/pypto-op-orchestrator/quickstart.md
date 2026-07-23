# CANNBot PyPTO 算子开发快速入门指南

## 概述

CANNBot PyPTO 算子开发模式适用于通过 PyPTO 开发自定义算子。采用 7 阶段状态机驱动，9 智能体团队协作，覆盖从需求理解到性能调优的完整开发流程，支持断点续跑与失败恢复。每个阶段完成门禁校验后才能进入下一阶段，Stage 5+ 通过 `MEMORY.md` 协作账本记录 pass/fail 与推理过程。

### 与 PyPTO-Pro 开发的区别

| 对比维度 | PyPTO 算子开发（本模式） | PyPTO-Pro 算子开发 |
|---------|------------------------|-------------------|
| 适用场景 | PyPTO 框架算子开发 | PyPTO-Pro 框架算子开发 |
| 编程语言 | Python（PyPTO API） | Python（PyPTO-Pro API） |
| 开发内容 | PyPTO kernel + golden + test | PyPTO-Pro kernel + golden + test |
| 阶段数 | 7 阶段状态机驱动 | 4 阶段工作流 |
| 状态管理 | `.orchestrator_state.json` 状态文件 | 调度顺序隐式管理 |
| 性能调优 | Stage 7 独立调优阶段 | 按需参考 |

## 一、环境搭建

### 前置条件

- 已安装 CANN Toolkit（建议 ≥ 9.0.0），具体版本配套关系请查阅 [CANN Release Notes](https://www.hiascend.com/cann/document)
- 已安装 PyPTO，版本需与 CANN 配套。通过 PyPI 安装时，CANN 与 PyPTO 版本对应关系查阅 [PyPI 安装](https://pypto.gitcode.com/install/build_and_install.html#pypi)；CANN 9.1.0 版本推荐使用源码编译安装，参阅 [源码编译安装](https://pypto.gitcode.com/install/build_and_install.html)
- 已配置 NPU 设备（支持 Ascend 910/950 PR 等芯片）
- 已安装 OpenCode、Claude Code、TRAE、Cursor、Copilot、CodeArts 等受支持的 AI 编程工具

### OpenCode（推荐）

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project opencode   # 项目级（默认）
bash init.sh global opencode    # 全局级
```

### 其他工具

<details>
<summary>Claude Code</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project claude     # 项目级
bash init.sh global claude      # 全局级
```

</details>

<details>
<summary>TRAE</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project trae       # 项目级
bash init.sh global trae        # 全局级
```

安装后自动检测 TRAE 环境，生成 `.trae/`（TRAE IDE）、`.marscode/`（TRAE Plugin）或 `.traecli/`（TRAE CLI）目录，结构与 Claude/OpenCode 基本一致。

</details>

<details>
<summary>Cursor</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project cursor     # 项目级
bash init.sh global cursor      # 全局级
```

安装后在项目根目录生成 `.cursor/` 目录，结构与 Claude/OpenCode 基本一致。

</details>

<details>
<summary>Copilot</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project copilot    # 项目级
bash init.sh global copilot     # 全局级
```

安装后在项目根目录生成 `.github/` 目录（项目级）或 `~/.copilot/` 目录（全局级），AGENTS.md 自动注入 VS Code Copilot 上下文。

</details>

<details>
<summary>CodeArts</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator
bash init.sh project codearts     # 项目级
bash init.sh global codearts      # 全局级
```

安装后在项目根目录生成 `.codeartsdoer/` 目录（项目级）或 `~/.codeartsdoer/` 目录（全局级），包含 skills/、agents/ 和 AGENTS.md。

</details>

### 在其他目录执行

`init.sh` 支持通过完整路径调用，无需先 `cd` 到插件目录。第三个参数指定目标项目路径，省略则安装到当前目录：

```bash
# 安装到当前目录
bash /path/to/pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator/init.sh project opencode

# 安装到指定项目
bash /path/to/pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator/init.sh project opencode /path/to/your_project_path
```

### 验证安装

```bash
# OpenCode
opencode agent list
# 应看到 pypto-op-planner / pypto-op-mathematician / pypto-op-architect / pypto-op-designer / pypto-op-coder / pypto-op-verifier / pypto-op-debugger / pypto-op-optimizer

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

在交互界面中输入算子开发需求，CANNBot 会自动启动 7 阶段流程：

```
帮我开发一个 softmax 算子，支持 float16 数据类型，shape 主要是 [1,128]、[4,2048]、[32,4096]
```

### 核心工作流

采用 7 阶段状态机驱动，9 智能体团队协作，确保算子开发质量：

```
Stage 1: 需求规划与 API 可行性 → Stage 2: Golden 生成
    → Stage 3: 架构设计 → Stage 4: 模块分解与契约
    → Stage 5: 按模块编码闭环 → Stage 6: 最终 E2E 验证
    → Stage 7: 性能调优
```

- **Stage 1**（planner）：完成需求规划和 API 可行性，产出 SPEC.md、API_REPORT.md
- **Stage 2**（mathematician）：生成 `<op>_golden.py` 参考实现与 GOLDEN_PERF_REPORT.md
- **Stage 3**（architect）：生成 DESIGN.md 架构设计文档
- **Stage 4**（designer）：产出 `eval/module_interfaces.yaml` 模块接口契约；L1 时 verifier 再生成模块验证脚手架
- **Stage 5**（coder、verifier、debugger）：按模块闭环，逐模块完成编码→验证→修复，产出 `modules/`、集成 `<op>_impl.py`、`test_<op>.py` 与 README.md；MEMORY.md 从 Stage 5 开始写入
- **Stage 6**（verifier）：最终 E2E 精度验证与 layout 校验
- **Stage 7**（optimizer）：性能采集、分析与迭代调优，verifier 回归确认精度无损，生成 `<op>_tuning_report.md`

每个阶段完成门禁校验后才能进入下一阶段。支持断点续跑和失败恢复，详见 AGENTS.md。

### 产出物示例

PyPTO 算子开发模式下，CANNBot 会在 `custom/<op>/` 目录下生成以下文件：

```
custom/<op>/
├── SPEC.md                    # 需求规格
├── API_REPORT.md              # API 可行性报告
├── <op>_golden.py             # Golden 参考实现
├── GOLDEN_PERF_REPORT.md      # Golden 基准性能报告
├── DESIGN.md                  # 架构设计文档
├── <op>_impl.py               # 集成 PyPTO kernel 实现
├── test_<op>.py               # 端到端测试入口
├── README.md                  # 实现说明
├── MEMORY.md                  # 阶段协作记录（Stage 5+）
├── <op>_tuning_report.md      # 性能调优报告（Stage 7）
├── .orchestrator_state.json   # 流程状态（自动维护）
├── eval/
│   ├── module_interfaces.yaml # 模块接口契约
│   ├── test_inputs.py         # 对抗测试输入
│   ├── adversarial_suite.json # 对抗测试套件
│   └── adversarial_runner.py  # 对抗测试执行器
├── modules/                   # 模块文件（L1 路径）
└── history_version/           # 版本备份
```

## 三、可用技能

| Skill | 用途 | 触发阶段 |
|-------|------|---------|
| `pypto-op-plan` | 串行组织需求理解与 API 可行性探索 | Stage 1 |
| `pypto-intent-understand` | 需求意图理解与规格生成 | Stage 1 |
| `pypto-api-explore` | API 可行性探索与分析 | Stage 1 |
| `pypto-golden-generate` | Golden 参考实现生成 | Stage 2 |
| `pypto-op-design` | 算子架构设计与模块分解 | Stage 3–4 |
| `pypto-op-construct` | 模块构建脚手架 | Stage 4–5 |
| `pypto-op-develop` | 算子代码实现 | Stage 5 |
| `pypto-op-verify` | 模块、E2E 与回归验证 | Stage 4–7 |
| `pypto-general-debug` | 通用错误定位与修复 | Stage 5 |
| `pypto-precision-debug` | 精度问题代码层排查 | Stage 5 |
| `pypto-precision-compare` | 精度中间结果对比分析 | Stage 5（辅助） |
| `pypto-op-perf-tune` | 算子性能分析与自动调优 | Stage 7 |
| `pypto-op-review` | 设计与实现评审 | Stage 3–7 |
| `pypto-docs-search` | 算子 API 文档、参考实现与 golden 检索 | 按需 |
| `pypto-memory-template` | MEMORY.md 协作账本模板 | Stage 5+ |
| `pypto-op-knowledge` | 算子开发知识库 | 按需 |
| `pypto-op-monitor` | 过程监控与状态追踪 | 全阶段 |
| `pypto-orchestration-manual` | 编排策略与门禁定义 | 全阶段 |

| Agent | 用途 | 负责阶段 |
|-------|------|---------|
| `pypto-op-planner` | 需求规划与 API 可行性 | Stage 1 |
| `pypto-op-mathematician` | Golden 参考实现 | Stage 2 |
| `pypto-op-architect` | 架构、tiling 与 loop 设计 | Stage 3 |
| `pypto-op-designer` | 模块分解与接口契约 | Stage 4 |
| `pypto-op-coder` | Kernel 实现 | Stage 5 |
| `pypto-op-verifier` | 独立裁决与检查 | Stage 4–7 |
| `pypto-op-debugger` | 失败定位与补丁建议 | Stage 5 |
| `pypto-op-optimizer` | 性能采集与调优 | Stage 7 |

## 四、断点续跑与恢复

CANNBot 通过 `.orchestrator_state.json` 维护全局状态，支持：

| 场景 | 使用方式 |
|------|---------|
| 中断后继续 | 再次输入算子名，自动从上次中断处续跑 |
| 失败后重试 | 输入"继续开发 {算子名}"，从失败阶段恢复 |
| 查看状态 | 查看 `custom/<op>/.orchestrator_state.json` |

## 五、常见问题

### Q: 如何查看帮助信息？

```bash
bash init.sh --help
```

### Q: 项目级和全局安装如何选择？

- **项目级**：适合多项目开发，每个项目可以有不同配置
- **全局**：适合单一项目，全局生效

### Q: 如何更新？

```bash
cd pypto-gym/cannbot-skills/plugins-official/pypto-op-orchestrator && bash init.sh project opencode
```

### Q: PyPTO 和 PyPTO-Pro 如何选择？

| 场景 | 推荐模式 |
|------|---------|
| 使用 PyPTO 框架开发算子 | PyPTO 算子开发 |
| 使用 PyPTO-Pro API 开发算子 | PyPTO-Pro 算子开发 |
| 快速验证算子可行性 | PyPTO 算子开发 |
| 原型开发和概念验证 | PyPTO 算子开发 |

---

## 总结

1. PyPTO 算子开发模式通过 7 阶段状态机与 9 智能体团队实现端到端自动化：需求规划→Golden 生成→架构设计→模块分解→编码闭环→E2E 验证→性能调优
2. 使用 `init.sh` 脚本一键安装（OpenCode 推荐），支持项目级和全局级
3. `opencode` / `claude` 是核心交互指令
4. 所有阶段通过门禁驱动，支持断点续跑与失败恢复
5. 产出物包含完整的 SPEC、API 报告、Golden 参考、架构设计、模块契约、模块实现、E2E 测试、调优报告与流程状态文件
