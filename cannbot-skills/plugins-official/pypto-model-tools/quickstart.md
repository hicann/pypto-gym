# PyPTO 模型工具快速入门

## 概述

`pypto-model-tools` 提供三个主模型处理 skill，并随 `pypto-fused-op-integration` 附带安装其依赖的 8 个算子支撑 skill（共 11 个），分别覆盖：

- 从 HuggingFace 模型卡到 NPU 端到端迁移与验证
- PyPTO 融合算子替换与整网集成（含 8 个算子支撑 skill，仅集成流程按需调用）
- 模型格式互转（PyTorch ↔ ONNX ↔ safetensors）与数值校验

这些 skill 是独立的，不依赖也不触发 PyPTO 或 PyPTO-Pro 的算子开发状态机。

## 一、环境搭建

### 前置条件

- 已安装 CANN Toolkit（建议 ≥ 9.0.0），具体版本配套关系请查阅 [CANN Release Notes](https://www.hiascend.com/cann/document)
- 已安装 PyPTO（需要融合算子集成时）和 torch/torch_npu
- 已配置 NPU 设备（支持 Ascend 910/950 PR 等芯片）
- 已安装 OpenCode、Claude Code、TRAE、Cursor、Copilot、CodeArts 等受支持的 AI 编程工具
- 需要 HuggingFace 下载时，确保网络可达或已配置 HF 镜像

### OpenCode（推荐）

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-model-tools
bash init.sh project opencode   # 项目级（默认）
bash init.sh global opencode    # 全局级
```

验证：

```bash
# 主模型 skill（3 个）
ls .opencode/skills/hf-npu-e2e-workflow/SKILL.md
ls .opencode/skills/pypto-convert-model/SKILL.md
ls .opencode/skills/pypto-fused-op-integration/SKILL.md
# 算子支撑 skill（8 个，融合算子集成按需调用）
ls .opencode/skills/pypto-intent-understand/SKILL.md
ls .opencode/skills/pypto-api-explore/SKILL.md
ls .opencode/skills/pypto-golden-generate/SKILL.md
ls .opencode/skills/pypto-op-design/SKILL.md
ls .opencode/skills/pypto-op-develop/SKILL.md
ls .opencode/skills/pypto-precision-compare/SKILL.md
ls .opencode/skills/pypto-precision-debug/SKILL.md
ls .opencode/skills/pypto-op-perf-tune/SKILL.md
```

### 其他工具

<details>
<summary>Claude Code</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-model-tools
bash init.sh project claude     # 项目级
bash init.sh global claude      # 全局级
```

</details>

<details>
<summary>TRAE</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-model-tools
bash init.sh project trae       # 项目级
bash init.sh global trae        # 全局级
```

安装后自动检测 TRAE 环境，生成 `.trae/`（TRAE IDE）、`.marscode/`（TRAE Plugin）或 `.traecli/`（TRAE CLI）目录，结构与 Claude/OpenCode 基本一致。

</details>

<details>
<summary>Cursor</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-model-tools
bash init.sh project cursor     # 项目级
bash init.sh global cursor      # 全局级
```

安装后在项目根目录生成 `.cursor/` 目录，结构与 Claude/OpenCode 基本一致。

</details>

<details>
<summary>Copilot</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-model-tools
bash init.sh project copilot    # 项目级
bash init.sh global copilot     # 全局级
```

安装后在项目根目录生成 `.github/` 目录（项目级）或 `~/.copilot/` 目录（全局级），AGENTS.md 自动注入 VS Code Copilot 上下文。

</details>

<details>
<summary>CodeArts</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-model-tools
bash init.sh project codearts     # 项目级
bash init.sh global codearts      # 全局级
```

安装后在项目根目录生成 `.codeartsdoer/` 目录（项目级）或 `~/.codeartsdoer/` 目录（全局级），包含 skills/、agents/ 和 AGENTS.md。

</details>

### 在其他目录执行

`init.sh` 支持通过完整路径调用，无需先 `cd` 到插件目录。第三个参数指定目标项目路径，省略则安装到当前目录：

```bash
# 安装到当前目录
bash /path/to/pypto-gym/cannbot-skills/plugins-official/pypto-model-tools/init.sh project opencode

# 安装到指定项目
bash /path/to/pypto-gym/cannbot-skills/plugins-official/pypto-model-tools/init.sh project opencode /path/to/your_project_path
```

### 验证安装

```bash
# OpenCode
ls .opencode/skills/hf-npu-e2e-workflow/SKILL.md
ls .opencode/skills/pypto-convert-model/SKILL.md
ls .opencode/skills/pypto-fused-op-integration/SKILL.md
ls .opencode/skills/pypto-intent-understand/SKILL.md
ls .opencode/skills/pypto-api-explore/SKILL.md
ls .opencode/skills/pypto-golden-generate/SKILL.md
ls .opencode/skills/pypto-op-design/SKILL.md
ls .opencode/skills/pypto-op-develop/SKILL.md
ls .opencode/skills/pypto-precision-compare/SKILL.md
ls .opencode/skills/pypto-precision-debug/SKILL.md
ls .opencode/skills/pypto-op-perf-tune/SKILL.md
# 每个 skill 目录应包含 SKILL.md

# Claude Code
ls .claude/
# 应看到 skills/（含 11 个子目录） CLAUDE.md cannbot-manifest.json

# TRAE
ls .trae/      # TRAE IDE
ls .marscode/  # TRAE Plugin（init.sh 自动检测）
ls .traecli/   # TRAE CLI（init.sh 自动检测）
# 应看到 skills/（含 11 个子目录） cannbot-manifest.json
# AGENTS.md 位于项目根目录

# Cursor
ls .cursor/
# 应看到 skills/（含 11 个子目录） cannbot-manifest.json
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

### 使用示例

**端到端模型迁移**

```
使用 hf-npu-e2e-workflow，把模型 Qwen/Qwen3-8B 下载到本地，在 Ascend NPU 上跑通并采集端到端吞吐。
```

**融合算子集成**

```
使用 pypto-fused-op-integration，把已有 PyPTO 融合算子接入当前 HuggingFace 模型，完成基线、融合后精度和性能对比。
```

**模型格式转换**

```
使用 pypto-convert-model，将 ./model.pt 转成 ONNX，再做一次 round-trip 数值验证。
```

## 三、安装内容

| 内容 | 说明 |
|------|------|
| 主技能（3 个） | 来自 `model/`，覆盖模型迁移、融合算子集成和格式转换 |
| 算子支撑技能（8 个） | 来自 `ops/`，仅由 `pypto-fused-op-integration` 按需调用 |
| 配置入口 | `AGENTS.md` / `CLAUDE.md` |

## 四、可用技能

| Skill | 用途 | 典型输入 |
|-------|------|---------|
| `hf-npu-e2e-workflow` | 下载 HF 模型 → 适配 NPU → 跑通推理 → 采集吞吐 | 模型 ID（如 `Qwen/Qwen3-8B`） |
| `pypto-fused-op-integration` | 模型 NPU 迁移 + 融合算子替换 + 整网验证 | 模型路径、融合算子目录 |
| `pypto-convert-model` | PyTorch → ONNX → safetensors 互转，含 round-trip 数值校验 | 输入文件路径 |
| `pypto-intent-understand` | 需求分析与规范化 | 融合目标描述 |
| `pypto-api-explore` | PyPTO API 速查与用法示例 | API 名称 |
| `pypto-golden-generate` | Golden 参考实现生成 | 被替换算子逻辑 |
| `pypto-op-design` | 算子方案设计 | 输入/输出 tensor 规格 |
| `pypto-op-develop` | PyPTO kernel 实现 | 算子设计方案 |
| `pypto-precision-compare` | 精度对比 | impl 输出 vs golden |
| `pypto-precision-debug` | 精度问题排查 | 精度不通过的算子 |
| `pypto-op-perf-tune` | 性能分析与自动调优 | 已通过精度的 kernel |

后 8 个为算子支撑 skill，仅由 `pypto-fused-op-integration` 按需调用，不独立激活。本插件不包含 Subagent。

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
cd pypto-gym/cannbot-skills/plugins-official/pypto-model-tools && bash init.sh project opencode
```

### Q: 模型下载慢怎么办？

配置 HuggingFace 镜像环境变量：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 端到端迁移和单点 skill 如何选择？

| 场景 | 推荐方式 |
|------|---------|
| 从 HuggingFace 模型到 NPU 推理的完整链路 | `hf-npu-e2e-workflow` skill |
| 已有 NPU 模型，仅需替换融合算子 | `pypto-fused-op-integration` skill |
| 仅需模型文件格式转换 | `pypto-convert-model` skill |

---

## 总结

1. model 插件提供 3 个主 skill，并安装 8 个算子支撑 skill（共 11 个），覆盖 NPU 模型全流程
2. 算子支撑 skill 仅由 `pypto-fused-op-integration` 按需调用，不独立激活
3. 使用 `init.sh` 一键安装（OpenCode 推荐），支持项目级和全局级
4. `opencode` / `claude` 是核心交互指令；IDE 类工具（TRAE / Cursor / Copilot / CodeArts）打开项目即自动加载
5. 不依赖 PyPTO / PyPTO-Pro 状态机
