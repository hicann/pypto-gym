---
name: pypto-pro-docs-search
description: 装配和检索 PyPTO-Pro 的本地 API、编程指南、快速入门与官方指定算子样例。用于 Pro 工作流查 API 用法、指南依据或参考样例；不用于普通 PyPTO 文档与算子仓检索。
---

# pypto-pro-docs-search

按关键词在当前 Pro 缓存中发现资料位置，再读取全文。已有 `PRO_MATERIAL_INDEX.md` 时先按其路径取证；需要补充发现时使用下方限定范围。

## 资源缓存

`$PYPTO_DEVKIT_DIR` 默认是当前项目的 `.devkit/`。Pro 工作流由 orchestrator 在会话开始时装配一次，并向所有子代理传递同一缓存的绝对路径。

| 子目录 | 内容 |
|---|---|
| `docs/` | Pro API、编程指南、快速入门、简介及配套图片 |
| `pro_ops/` | 官方指定算子样例，仅装配统一样例清单内文件，入口见 [`references/sample-index.md`](references/sample-index.md) |

指南保留上游 `guide` 目录结构，简介使用共享的 `introduction.md`。缓存不包含 Tensor API/专属指南目录、`docs/install/` 或普通 PyPTO 算子与测试。

## 准备缓存

首次或需要更新时，在项目根目录运行本技能的独立脚本。已安装资源根由 `init.sh` 渲染的 `$CANNBOT_CONFIG_ROOT` 提供。

```bash
export PYPTO_DEVKIT_DIR="$(pwd)/.devkit"
CANNBOT_ROOT="$CANNBOT_CONFIG_ROOT"
python "$CANNBOT_ROOT/skills/pypto-pro-docs-search/scripts/sync_devkit.py" \
    --samples "$CANNBOT_ROOT/skills/pypto-pro-material-explore/references/official_samples.md"
```

在同一调用末尾增加 `--check` 只校验已有缓存：`READY` 且退出码为 0 表示就绪；`NEED_PROVISION` 且退出码为 4 表示需要装配。正常装配按清单复制样例并校验结果，无需再清理 `pro_ops/`。

脚本默认从 `https://gitcode.com/cann/pypto.git` 获取 Pro 资料。`PYPTO_SRC` 可指定已有 PyPTO 工作树，`PYPTO_SRC_URL` 可覆盖远端源，`--pin <git-ref>` 从远端获取指定版本。成功后写入 `MANIFEST.json` 和就绪标记，记录缓存来源与版本，以目标运行环境对应的资料为准。

在 Pro 工作流中，装配由 orchestrator 负责，子代理只检索同一缓存。

**子代理路径传递**：命令的路径参数直接代入编排器给定的绝对路径并加引号；Python 从环境读取路径时，仅将给定的 `PYPTO_DEVKIT_DIR`、`CANNBOT_CONFIG_ROOT` 原样传入单条命令，如 `PYPTO_DEVKIT_DIR="<缓存绝对路径>" CANNBOT_CONFIG_ROOT="<资源根绝对路径>" python "<脚本绝对路径>"`。不使用 `export`、猜测或更换路径，也不设置其他环境变量；未收到路径时返回 `env_error` 交回编排器。

## 检索

使用原生 `Glob` / `Grep`，按资料类型限定路径：

- **API 名称与签名**：在 `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/` 递归查找 `*<API名>*.md`，再读取参数、类型与约束全文。
- **编程与用法**：分别检索 `$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/`、`$PYPTO_DEVKIT_DIR/docs/guide/quick_start/pro/` 与 `$PYPTO_DEVKIT_DIR/docs/guide/introduction.md`，不扩大到整个 `docs/guide/`。
- **参考样例**：依据 [`references/sample-index.md`](references/sample-index.md) 中的统一清单，在 `$PYPTO_DEVKIT_DIR/pro_ops/` 查找关键词或 API 调用，只读取清单内样例。

命中后直接 `Read` 全文，返回实际命中的缓存相对路径和相关事实。

## 边界

已知确切路径时直接读取；本技能用于按关键词发现、跨文件定位。资料未说明的接口能力不能凭名称或普通 PyPTO 的同名 API 推断。

## 缓存未就绪时

`--check` 返回 `NEED_PROVISION`，或下游发现文档、指定样例缺失时，报告具体缺项，由 orchestrator 按“准备缓存”处理。子代理不自行同步、切换缓存或改用普通 PyPTO 同步器及在线索引。离线无法装配时报告缺项，不编造路径或索引。

## 详细索引（按入口键）

- 知道 **API 名 / 功能类别** → [`references/api-index.md`](references/api-index.md)
- 查 **编程指南 / 快速入门 / 调试调优** → [`references/doc-index.md`](references/doc-index.md)
- 找 **官方指定样例** → [`references/sample-index.md`](references/sample-index.md)
