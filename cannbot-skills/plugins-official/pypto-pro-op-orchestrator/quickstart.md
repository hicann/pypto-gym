# CANNBot PyPTO-Pro 算子开发快速入门指南

## 概述

CANNBot PyPTO-Pro 算子开发模式适用于通过 PyPTO-Pro 开发 Ascend NPU 算子。采用 5 阶段工作流驱动，覆盖从需求理解、代码实现到证据化性能优化的完整流程，通过独立 verifier 在每个 Stage 结束时执行检查清单，确保各阶段产出符合质量要求。

### 与 PyPTO 开发的区别

| 对比维度 | PyPTO-Pro 算子开发（本模式） | PyPTO 算子开发 |
|---------|---------------------------|--------------|
| 适用场景 | PyPTO-Pro 框架算子开发 | PyPTO 框架算子开发 |
| 编程语言 | Python（PyPTO-Pro API） | Python（PyPTO API） |
| 开发内容 | PyPTO-Pro kernel + golden + test | PyPTO kernel + golden + test |
| 阶段数 | 5 阶段工作流 | 7 阶段状态机驱动 |
| 状态管理 | `.orchestrator_state.json`（`state_transition` 工具 + lint 机械门禁 + verifier 语义门禁） | `.orchestrator_state.json`（`state_transition` 工具 + lint 门禁） |
| 性能调优 | Stage 5：可比基线、优化循环与证据验收 | Stage 7 独立调优阶段 |

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

> **自动门禁支持边界**：`state_transition` 工具、写入前 lint 和 Stage/Module 自动硬门禁目前通过 OpenCode 插件提供。下列其他工具的 `init.sh` 适配仅安装 skills、agents 与提示词资源，不会获得同等的 OpenCode 自动门禁能力。需要完整 5 阶段状态机和 fail-closed lint 流程时，请使用 OpenCode。

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

在交互界面中输入算子开发需求，CANNBot 会自动启动 5 阶段流程：

```
使用 PyPTO-Pro 开发 softmax 算子，支持 [1, 128]、[4, 2048] 和 [32, 4096] 的 float16 输入。
```

### 核心工作流

采用 5 阶段流程，每个 Stage 由独立 verifier 执行检查清单，确保各阶段产出符合质量要求：

```
Stage 1: 需求规划与资料索引 → Stage 2: NPU/CPU Golden（性能采集可选）
    → Stage 3: Tile 数据流设计 → Stage 4: Kernel 实现与精度验证
    → Stage 5: 可比基线驱动的性能优化与证据验收
```

每一阶段通过 verifier 检查后才可进入下一阶段。verifier 失败时，orchestrator 将失败项反馈给对应 Stage 的子代理修正。Pro 流程**使用** `custom/<op>/.orchestrator_state.json` 状态机推进 Stage（`state_transition` 工具，随 OpenCode 插件安装；其他工具下按 AGENTS.md 降级协议手工维护同一账本）。详见 AGENTS.md「共享状态与 state_transition 工具」。

Stage 2 默认只生成并验证 `{op}_golden.py`（NPU）与 `{op}_golden_cpu.py`（CPU FP32），不采集 NPU golden 性能。用户在需求中明确说明“采集 NPU golden 性能”时，可额外生成一份可选的 `GOLDEN_PERF_REPORT.md`；Stage 2 不负责 Stage 5 机器合同。若 SPEC 未记录用户数值目标，Stage 5 optimizer 会使用 `pypto-pro-op-perf-tune` 自带的 `collect_golden_reference.py`，基于现有 Golden 和冻结 case 清单生成一次 Stage 5 专用报告，不回滚或重跑 Stage 2。

Stage 5 优先采用用户在 SPEC 中明确给出的可复算性能目标。用户未给数值目标时，默认要求 `PERFORMANCE_CASES.json` 中每个 P0 case 的 `Golden 每迭代 NPU E2E / PyPTO 最终 target-kernel >= 1.0`。该比值是明确定义的 Golden 跨实现参考目标，不是 PyPTO baseline→final 同口径加速比。

性能提取顺序固定为：先重跑 Stage 4 正确性并完成 JIT/编译预热；用一个受控 case 做 discovery profile，读取真实完整 lowering `Op Name`；再按 manifest 逐 case formal compare，逐 repeat 采集七组 msprof 指标并归档；据此做 Roofline、PyPTO-Pro Tile 依赖和 Scalar/流水分析；经过单变量 A/B 与兼容组合后，对最终版本独立重采，并为 final 每个 P0 case 补采 instruction timeline。补充时间线只计算唯一 target task 窗口内同一监控 lane 的 pipe 区间重叠，还需与当前 Tile DAG/slot/stage 映射联合验证，不能用其受扰动时延替代 formal compare。Stage 4 runner 没有逐 case selector时，多 case manifest 必须逐 case 用 `test_function` 指向既有无参测试函数，由 Stage 5 适配器选择；单 case 可直接运行整个 runner，但必须只有一次可唯一归属的 target launch。均不修改 Stage 1–4 模板。

数值目标达到后还要过健康终态门禁：每个 P0 case 的关键路径必须是有效计算、必要数据搬运或二者平衡，Scalar/等待不能主导；合法可重叠的搬算必须由可归属 target kernel 的 timeline 与当前 Tile DAG 共同证明。`op_summary` 中多个 pipe ratio 同时高、不同核/lane 并行或没有 Tile 映射的区间相交都只用于诊断，不能证明 overlap 或 Roofline。optimizer 会建立候选覆盖账本：数值目标或健康终态未通过时，须继续尝试全部适用原子思路、有意义的兼容/协同组合，并在瓶颈变化后重开受影响的旧候选；不能因少数实验无收益提前结束。

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
├── GOLDEN_PERF_REPORT.md      # Stage 2 用户按需报告，或 Stage 5 默认目标人读报告
├── GOLDEN_PERF_REPORT.json    # 仅 Stage 5 默认目标分支生成的机器合同
├── DESIGN.md                  # Tile 数据流设计文档
├── test_{op}.py               # Stage 4 kernel；Stage 5 原地保留最快正确版本
├── PERFORMANCE_CASES.json     # Stage 5 执行 case 清单（SPEC P0 与既有测试一一对应）
├── PERFORMANCE_REPORT.md      # Stage 5 逐 case baseline/final、Golden 参考比值、候选覆盖与优化记录
├── performance.json           # Stage 5 结构化性能结果
├── performance.log            # Stage 5 采集日志
├── perf_report.md             # Stage 5 脚本分析报告
└── docs/perf/round_NNN/       # Stage 5 七指标与逐 case instruction timeline 归档
```

## 三、可用技能

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
| `pypto-docs-search` | 算子 API 文档与参考实现检索 | 按需 |

| Agent | 用途 | 负责阶段 |
|-------|------|---------|
| `pypto-pro-op-planner` | 需求理解与资料索引 | Stage 1 |
| `pypto-pro-op-mathematician` | NPU/CPU Golden；按需采集基准性能 | Stage 2 |
| `pypto-pro-op-architect` | Tile 数据流设计 | Stage 3 |
| `pypto-pro-op-coder` | Kernel 实现与精度验证 | Stage 4 |
| `pypto-pro-op-optimizer` | 可比证据驱动的性能优化 | Stage 5 |
| `pypto-pro-op-verifier` | 每个 Stage 的独立检查 | Stage 1–5 |

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
| 使用 PyPTO-Pro API，并按 5 阶段逐项检查 | PyPTO-Pro |

---

## 总结

1. PyPTO-Pro 通过 5 阶段工作流覆盖从需求规划到性能优化的完整流程：需求规划与资料索引→NPU/CPU Golden（性能采集可选）→Tile 数据流设计→Kernel 实现与精度验证→证据化性能优化
2. 使用 `init.sh` 一键安装（OpenCode 推荐），支持项目级和全局级
3. `opencode` / `claude` 是核心交互指令
4. 每个 Stage 由独立 verifier 执行检查清单，确保质量门禁
5. 产物全部写入 `custom/<op>/`，含 SPEC、资料报告、Golden、设计文档、Kernel 实现与可复算性能证据
