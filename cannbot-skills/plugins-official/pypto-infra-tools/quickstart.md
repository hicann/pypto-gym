# PyPTO 静态规范修复快速入门

## 概述

`pypto-static-check-repire` 用于读取静态检查 Excel 报表（如 CodeCheck 导出的问题清单），定位 Python 源码中的规范违规项，执行最小化修复并生成变更记录。该 skill 面向 PyPTO-Gym 门禁导出的 Excel 问题清单，不按通用 linter 名称推测修复规则。

## 一、环境搭建

### 前置条件

- 已安装 OpenCode、Claude Code、TRAE、Cursor、Copilot、CodeArts 等受支持的 AI 编程工具
- 已有静态检查报表文件（`.xlsx` 格式）
- 报表中明确列出文件路径、行号、规则编号和违规描述

### OpenCode（推荐）

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-infra-tools
bash init.sh project opencode   # 项目级（默认）
bash init.sh global opencode    # 全局级
```

验证：

```bash
ls .opencode/skills/pypto-static-check-repire/SKILL.md
```

### 其他工具

<details>
<summary>Claude Code</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-infra-tools
bash init.sh project claude     # 项目级
bash init.sh global claude      # 全局级
```

</details>

<details>
<summary>TRAE</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-infra-tools
bash init.sh project trae       # 项目级
bash init.sh global trae        # 全局级
```

安装后自动检测 TRAE 环境，生成 `.trae/`（TRAE IDE）、`.marscode/`（TRAE Plugin）或 `.traecli/`（TRAE CLI）目录，结构与 Claude/OpenCode 基本一致。

</details>

<details>
<summary>Cursor</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-infra-tools
bash init.sh project cursor     # 项目级
bash init.sh global cursor      # 全局级
```

安装后在项目根目录生成 `.cursor/` 目录，结构与 Claude/OpenCode 基本一致。

</details>

<details>
<summary>Copilot</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-infra-tools
bash init.sh project copilot    # 项目级
bash init.sh global copilot     # 全局级
```

安装后在项目根目录生成 `.github/` 目录（项目级）或 `~/.copilot/` 目录（全局级），AGENTS.md 自动注入 VS Code Copilot 上下文。

</details>

<details>
<summary>CodeArts</summary>

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym/cannbot-skills/plugins-official/pypto-infra-tools
bash init.sh project codearts     # 项目级
bash init.sh global codearts      # 全局级
```

安装后在项目根目录生成 `.codeartsdoer/` 目录（项目级）或 `~/.codeartsdoer/` 目录（全局级），包含 skills/、agents/ 和 AGENTS.md。

</details>

### 在其他目录执行

`init.sh` 支持通过完整路径调用，无需先 `cd` 到插件目录。第三个参数指定目标项目路径，省略则安装到当前目录：

```bash
# 安装到当前目录
bash /path/to/pypto-gym/cannbot-skills/plugins-official/pypto-infra-tools/init.sh project opencode

# 安装到指定项目
bash /path/to/pypto-gym/cannbot-skills/plugins-official/pypto-infra-tools/init.sh project opencode /path/to/your_project_path
```

### 验证安装

```bash
# OpenCode
ls .opencode/skills/pypto-static-check-repire/SKILL.md
# 应看到 SKILL.md

# Claude Code
ls .claude/
# 应看到 skills/ CLAUDE.md cannbot-manifest.json

# TRAE
ls .trae/      # TRAE IDE
ls .marscode/  # TRAE Plugin（init.sh 自动检测）
ls .traecli/   # TRAE CLI（init.sh 自动检测）
# 应看到 skills/ cannbot-manifest.json
# AGENTS.md 位于项目根目录

# Cursor
ls .cursor/
# 应看到 skills/ cannbot-manifest.json
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

在交互界面中输入修复需求：

```
使用 pypto-static-check-repire，读取仓库根目录 ./codecheck.xlsx，按报表逐项处理 Python 静态规范问题，并把已修复项和跳过原因追加记录到仓库根目录 ./static.md。
```

执行目录必须是 `pypto-gym` Git 仓库根目录。skill 只处理报表列出的范围，并按自身规则跳过 `src/pypto_gym/transformers/`，以及 kernel 文件中的超大函数、超大深度函数和超大圈复杂度问题。

### 核心工作流

```
校验 pypto-gym 与 Excel 报表 → 解析问题清单 → 应用豁免规则
    → 逐项最小修复或跳过 → 将结果追加到 static.md
```

- 每一项都记录文件、行号、原始问题、处理结果和原因
- `static.md` 采用追加方式，不覆盖已有记录
- Git 仅用于环境校验；skill 不负责提交修改

### 产出物

- 修复：直接修改对应的 Python 源文件
- 记录：仓库根目录 `static.md`（追加式，记录每项处理结果和原因）

## 三、可用技能

| Skill | 用途 | 触发时机 |
|-------|------|---------|
| `pypto-static-check-repire` | 按静态检查报表逐项修复 Python 规范违规 | 用户指定报表路径和修复范围 |

本插件不包含 Subagent，skill 通过描述匹配自动激活。

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
cd pypto-gym/cannbot-skills/plugins-official/pypto-infra-tools && bash init.sh project opencode
```

### Q: 哪些问题会跳过？

确认无需修改的问题、`src/pypto_gym/transformers/` 下的问题，以及 kernel 文件中的超大函数、超大深度函数和超大圈复杂度问题会跳过，并在 `static.md` 中记录原因。

### Q: 修复记录保存在哪里？

全部记录追加到仓库根目录 `static.md`。

---

## 总结

1. `pypto-static-check-repire` 按报表逐项修复静态规范问题
2. 使用 `init.sh` 一键安装（OpenCode 推荐），支持项目级和全局级
3. `opencode` / `claude` 是核心交互指令；IDE 类工具（TRAE / Cursor / Copilot / CodeArts）打开项目即自动加载
4. 每个修复或跳过项都追加到仓库根目录 `static.md`，便于追溯
