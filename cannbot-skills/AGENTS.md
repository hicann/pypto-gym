# CANNBot Agent 开发平台 - 工程师指南

## 核心职责

作为 **Skills 开发者**，从 **Agent 架构** 视角出发，构建专业、高效的 PyPTO 算子开发智能体能力模块：

1. **Skills 开发与优化** — 创建可复用的技能模块，为 PyPTO 算子开发 Agent 提供专业能力支持
2. **Agents 创建** — 设计专业化子代理，实现职责分工和模块化开发
3. **业务工作流** — 构建多 Agent 协作编排，覆盖 PyPTO/PyPTO-Pro 算子开发、模型适配、仓库治理等完整场景
4. **效果评测** — 持续验证 Skills 和 Agents 的实际效果，优化交付质量

## 核心原则

开发 Skills、Agents 和 Plugins 时遵循以下原则：

### 1. 信息来源可信

技术信息必须来自可信源，禁止编造：
- **可信来源**：PyPTO 官方文档与代码仓、CANN 安装路径下的文件、用户明确提供的信息
- **禁止行为**：编造 API/参数、推测未验证的行为
- **不确定时**：明确标注，引导用户验证

### 2. 渐进式披露

模块设计分层，按需展开：
- SKILL.md 只写核心流程与决策规则
- 详细参考放 `references/` 目录
- 用户需要时再深入引导

### 3. 简洁精炼

输出直击要点：
- 先给结论，再说原因
- 使用列表、表格等结构化表达
- 一个段落只讲一个要点

## 架构设计

```
CANNBot Skills — PyPTO-Gym 架构（自底向上）
┌─────────────────────────────────────────┐
│        Plugins（应用编排层）             │  编排多个 Agent 协同工作
│    plugins-official/*/ 目录              │
├─────────────────────────────────────────┤
│        Agents（角色执行层）              │  定义做什么（职责范围）
│    plugins-official/*/agents/ 目录       │
├─────────────────────────────────────────┤
│        Skills（知识能力层）              │  定义怎么做（具体实现）
│    ops/、model/、infra/ 下的各 Skill 子目录 │
├─────────────────────────────────────────┤
│        References（知识层）              │  定义如何做得更好
│    内嵌在 Skills 中的最佳实践            │
├─────────────────────────────────────────┤
│    Infrastructure（基础设施层）          │  提供底层工具支持
│    infra/ 目录下的工具类 Skill           │
└─────────────────────────────────────────┘
     ↑ 效果评测横向覆盖所有层级 ↑
```

> 注：`ops/`、`model/`、`infra/` 分别对应算子开发、模型适配、仓库治理三大业务领域。

## 项目结构

```
cannbot-skills/
├── ops/                      # 算子 Skills（PyPTO classic + PyPTO-Pro）
│   ├── pypto-api-explore/
│   ├── pypto-op-design/
│   ├── pypto-pro-op-develop/
│   └── ...                   # 共 26 个 Skill 子目录
├── model/                    # 模型适配与推理优化 Skills
│   ├── hf-npu-e2e-workflow/
│   ├── pypto-convert-model/
│   └── pypto-fused-op-integration/
├── infra/                    # 仓库治理 Skills
│   └── pypto-static-check-repire/
├── plugins-official/         # 官方 Plugin（Plugin 配置 + Agents + 安装入口）
│   ├── pypto-op-orchestrator/       # PyPTO classic 算子开发
│   │   ├── agents/                  # Agent 定义（.md）
│   │   ├── hooks/                   # 运行时 Hook 与状态机
│   │   ├── AGENTS.md                # Plugin 配置
│   │   ├── init.sh                  # 安装入口
│   │   └── quickstart.md            # 快速入门
│   ├── pypto-pro-op-orchestrator/   # PyPTO-Pro 算子开发
│   ├── pypto-model-tools/           # 模型适配工具集
│   └── pypto-infra-tools/           # 基础设施治理工具
├── AGENTS.md                 # 本文件（开发者指南）
└── README.md                 # 项目说明与技能索引
```

## Skills 分类

本仓库 Skills 按业务领域与功能性质分类：

### 算子开发（ops/）

| 类别 | 技能（示例） | 说明 |
|------|------------|------|
| 编排与知识类 | `pypto-orchestration-manual`、`pypto-intent-understand`、`pypto-op-knowledge`、`pypto-api-explore`、`pypto-docs-search`、`pypto-memory-template` | 编排入口、需求理解、领域知识、API 速查、文档检索、经验复用 |
| 方案与开发类 | `pypto-op-plan`、`pypto-op-design`、`pypto-op-develop`、`pypto-op-construct` | 实施计划、方案设计、代码开发、工程脚手架 |
| 验证与调优类 | `pypto-golden-generate`、`pypto-precision-compare`、`pypto-precision-debug`、`pypto-op-verify`、`pypto-op-review`、`pypto-op-perf-tune` | Golden 生成、精度对比与排查、功能验证、代码检视、性能调优 |
| 监控与调试类 | `pypto-op-monitor`、`pypto-general-debug` | 任务监控、通用诊断 |
| Pro 专属类 | `pypto-pro-intent-understand`、`pypto-pro-material-explore`、`pypto-pro-op-plan`、`pypto-pro-op-design`、`pypto-pro-op-develop`、`pypto-pro-op-perf-tune`、`pypto-pro-golden-generate`、`pypto-pro-environment-check` | PyPTO-Pro 精简流程的全部八项技能 |

### 模型适配（model/）

| 技能 | 说明 |
|------|------|
| `hf-npu-e2e-workflow` | HF 模型到昇腾 NPU 端到端迁移 |
| `pypto-fused-op-integration` | 融合算子入网集成与整网验证 |
| `pypto-convert-model` | PyTorch/ONNX/safetensors 模型格式互转 |

### 仓库治理（infra/）

| 技能 | 说明 |
|------|------|
| `pypto-static-check-repire` | Python 代码静态规范检查与自动修复 |

## 详细规范

### Skill 目录结构

每个 Skill 子目录按需包含以下内容：

- `SKILL.md` — 必需，核心技能说明，包含触发条件、执行流程与决策规则
- `references/` — 可选，详细参考文档，按需引用的深度资料
- `scripts/` — 可选，技能使用的自动化脚本
- `templates/` — 可选，工程模板与配置样例

跨 Skill 引用必须指向本仓实际存在的目录，并使用目标工具可解析的相对路径；Plugin 只声明和安装自身工作流需要的 Skills 与 Agents。
