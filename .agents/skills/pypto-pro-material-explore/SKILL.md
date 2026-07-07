---
name: pypto-pro-material-explore
description: PyPTO-Pro 资料探索。构建 PRO_MATERIAL_INDEX.md 全量资料索引（API 文档 + pro_ops 算子样例 + 教程），基于索引从三个方向并行探索（API 映射/约束检查、pro_ops 相似样例、教程设计模式），产出 EXPLORE_REPORT.md。触发词：资料探索、API 探索、查找 API、PyPTO-Pro 有没有 xxx、支持什么 dtype、约束是什么、API 映射、可行性分析、这个算子能做吗、pl.api。
---

# pypto-pro-material-explore

构建 PyPTO-Pro 全量资料索引，基于索引从三个方向探索，为算子开发提供 API 映射、约束检查、样例参考和可行性分析。

> **资料来源：devkit 缓存**。所有 API 文档 / pro_ops 样例 / 教程均在本地缓存 `$PYPTO_DEVKIT_DIR`（默认 `${XDG_CACHE_HOME:-$HOME/.cache}/pypto-devkit`）下，不在当前工作仓库内。首次或需要更新时先运行 skill `pypto-docs-search` 的 `scripts/sync_devkit.py` 装配缓存——三个源 URL（`PYPTO_SRC_URL` / `PYPTO_GYM_URL` / `PYPTO_PRO_OPS_URL`）**全部指向 `https://gitcode.com/gaoxiang618/pypto.git`**，确保 docs（含 `pypto_pro/` 教程）与 pro_ops（a5 样例）从含 PyPTO-Pro 资料的源仓拉取。缓存目录结构：
>
> | 缓存子目录 | 内容 | 对应源仓路径 |
> |---|---|---|
> | `$PYPTO_DEVKIT_DIR/docs/api/` | API 文档 | 源仓 `docs/zh/api/` |
> | `$PYPTO_DEVKIT_DIR/docs/pypto_pro/` | 教程文档 | 源仓 `docs/zh/pypto_pro/` |
> | `$PYPTO_DEVKIT_DIR/docs/pypto_api_list.md` | API 总索引 | 源仓 `docs/zh/pypto_api_list.md` |
> | `$PYPTO_DEVKIT_DIR/pro_ops/` | PyPTO-Pro 算子样例 | 源仓 `python/tests/ut/block/frontend/a5/` |

## 输入

`custom/<op>/SPEC.md`（由 Stage 1 Step 1 的 `pypto-intent-understand` 产出）。从 SPEC.md 中提取算子计算逻辑、shape、dtype 等需求信息。

## 输出

- **`custom/<op>/PRO_MATERIAL_INDEX.md`**：全量资料索引（每次执行重新扫描生成，不直接拷贝模板）
- **`custom/<op>/EXPLORE_REPORT.md`**：三方向探索报告，使用 [templates/explore_report.md](templates/explore_report.md) 模板

---

## Step 0：确保 devkit 缓存就绪

扫描前先确认缓存已装配（`$PYPTO_DEVKIT_DIR` 下存在 `docs/ pro_ops/`）。缺失时运行 `pypto-docs-search` 的装配脚本——三个源 URL 全部指向 `https://gitcode.com/gaoxiang618/pypto.git`：

```bash
CACHE="${PYPTO_DEVKIT_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/pypto-devkit}"
if [ ! -d "$CACHE/docs/api" ] || [ ! -d "$CACHE/pro_ops" ]; then
  PYPTO_SRC_URL=https://gitcode.com/gaoxiang618/pypto.git \
  PYPTO_GYM_URL=https://gitcode.com/gaoxiang618/pypto.git \
  PYPTO_PRO_OPS_URL=https://gitcode.com/gaoxiang618/pypto.git \
    python3 <pypto-docs-search skill 目录>/scripts/sync_devkit.py
fi
```

装配失败（离线受限等）时向用户反馈，不得凭空编造索引。

---

## Step 1：构建全量资料索引

PyPTO-Pro 资料处于持续更新中，**每次执行必须重新扫描仓库**生成 `PRO_MATERIAL_INDEX.md`，不得直接拷贝或复用已有索引。

> **核心理念**：先建图，后按图索骥。索引一次构建，全流程复用。

### 索引覆盖范围

| 资料类别 | 目录 | 扫描方式 |
|----------|------|----------|
| API 文档 | `$PYPTO_DEVKIT_DIR/docs/api/`（递归） | `grep -rl "pypto_pro\|PyPTO-Pro"` 过滤所有含标记的 `.md` |
| 算子样例 | `$PYPTO_DEVKIT_DIR/pro_ops/` | `find` 获取所有 `.py` 文件 |
| 教程文档 | `$PYPTO_DEVKIT_DIR/docs/pypto_pro/`（递归） | `find` 获取所有 `.md` 文件 |

> **不遗漏保证**：API 文档以 `grep -rl "pypto_pro\|PyPTO-Pro"` 关键字过滤为唯一扫描方式——任何含 pypto_pro 标记的文档都会被纳入，不依赖预设目录列表。新增子目录或文档时无需修改扫描命令。

### 生成方式

以 [templates/pro_material_index.md](templates/pro_material_index.md) 为骨架（章节结构 + 扫描命令 + 表头），**通过 bash 命令扫描当前仓库动态填充数据行**：

- **§A API 文档**：先 `grep -rl` 获取全部含 pypto_pro 标记的文档清单，再按目录路径自动分组展示（SIMD-API / Utils-API / SIMT-API / 其余）
- **§B 算子样例**：`find` 获取全部 `.py`，按 pro_ops 子目录拆分（以实际扫描结果为准）
- **§C 教程文档**：`find` 获取 `$PYPTO_DEVKIT_DIR/docs/pypto_pro/` 下全部 `.md`（覆盖整个 pypto_pro 目录，以实际扫描结果为准）

### 输出要求

- 路径使用缓存内绝对路径（`$PYPTO_DEVKIT_DIR/...`，保证下游 Stage 可直接 `Read`）
- 每个子类别标题后标注 `（N 文档）` / `（N 文件）` 的计数
- §B 按 pro_ops 子目录拆分为独立小节，子目录名按字典序排列
- 若某个扫描目录不存在或为空，保留空表（标注 `<!-- 空 -->`）

---

## Step 2：三方向并行探索

> **探索目标**：为 `custom/<op>/SPEC.md` 中的算子需求服务——三个方向的探索都围绕 SPEC.md 中的算子公式、shape、dtype、计算逻辑展开，目的是验证可行性、确定 API 映射、提取约束、找到可复用样例。脱离 SPEC.md 的泛化探索无意义。

**以下所有子代理的搜索以 `custom/<op>/PRO_MATERIAL_INDEX.md` 为权威目录**——从中查找目标路径，而非在文件系统中盲目 grep。若索引中未找到所需资料，再回退到文件系统补充搜索。

将探索任务拆分为**三个并行的 Explore subagent**，在**同一条消息中同时发起**。

### Explore subagent 1：API 文档与约束

**搜索范围**：`$PYPTO_DEVKIT_DIR/docs/`（基于索引 §A）

**任务**：

1. 从索引 §A.1 查 `$PYPTO_DEVKIT_DIR/docs/pypto_api_list.md` 获取 API 总索引
2. 将算子计算逻辑分解为原子操作序列，对每个操作从索引 §A 中查找对应 `pl.*` API 调用链（单个 API 或多个 API 组合）
3. 逐文档读取相关 API 文档，提取约束：dtype 支持、shape 范围、layout 要求（`layout=pl.ND/DN/NZ/ZN` 等枚举）、MemorySpace 约束
4. **归约类 API 的 layout 要求强制记录**：对 row_max / row_sum / row_reduce / col_max / col_sum / col_reduce 以及 row_expand_* / col_expand_* 系列，必须读其文档"参数范围"表并逐一记录输出/行向量 tile 的 layout 要求。已知行向归约的 `[行数,1]` 输出明文要求 `layout=pl.DN`（证据：`$PYPTO_DEVKIT_DIR/docs/api/SIMD-API/计算API/数学函数/row_max.md:25`、`row_sum.md:25`、`row_expand_sub.md:27`）；列向 col_* 输出为 `[1,列数]`，layout 要求以其自身文档为准，不套用行向规则。漏读会导致 Stage 3 遗漏双视图设计
5. 提取 Tile 规格约束（TileType 文档）、MemorySpace 约束、DataType 枚举值
6. **探测关键常量**：从 API 文档中提取硬件/版本相关常量——cross_core event_id 上限（`max_event_id` 默认值）、地址对齐要求、Cube tile 对齐要求等，记录值 + 文档路径
7. 未找到对应 API → 标记 unsupported，尝试 pl.* 组合或 vf 替代方案

**返回**：API 映射表（数学步骤 → pl.* 调用链）、约束检查结果（含归约类 API 的 layout 要求）、环境常量（event_id 上限/对齐要求/Cube tile 约束 + 来源路径）、证据路径列表

### Explore subagent 2：pro_ops 算子样例

**搜索范围**：`$PYPTO_DEVKIT_DIR/pro_ops/`（基于索引 §B）

**任务**：

1. 从索引 §B 中获取全部 pro_ops 子目录及文件清单（以实际扫描结果为准），按算子类型定位候选子目录
2. 按 API 使用进一步筛选：搜索 `pl.row_max\|pl.matmul\|pl.exp\|pl.load_tile` 等关键 API
3. 提取可复用模式：tile_group 用法、双视图技巧、循环结构等
4. 遍历所有候选，收集**所有匹配的参考实现**，不要找到一个就停止
5. **参考完整度甄别**（选参考时最关键）：对每个候选样例标注是否覆盖以下生产级模式——① 多 tile 归约（归约轴 > 单 tile）② 双视图（同地址 DN + ND tile 对）③ online 状态更新（running max/sum）。**优先选完整生产级实现，而非简化或走不同引擎路径的版本**。
6. **探测关键常量**：从样例中提取 UB 容量上限（如 `assert {addr} <= {N}*1024`），从 `tile_dims` 用法中推断 stride 经验阈值；记录值 + 样例路径

**返回**：每个匹配实现的路径、相似度、**完整度标注（是否覆盖多tile归约/双视图/online）**、可复用点；关键常量（UB 容量/stride 经验阈值 + 来源路径）；若无匹配标注「无匹配」

> **注意**：pro_ops/ 下文件是 API 用法参考 + 功能测试，非 production 标准。

### Explore subagent 3：教程与设计指南

**搜索范围**：`$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials/`（基于索引 §C）

**任务**：

1. 从索引 §C 中获取全部教程文档路径清单（以实际扫描结果为准，不预设固定文件列表）
2. 逐文档读取，提取与当前算子相关的设计模式：sync 策略选择、tile 尺寸建议、双视图适用场景、尾块处理

**返回**：适用的设计模式、关键约束、参考的教程章节

### 并行探索结果汇总

等待三个 subagent 全部返回后：

1. 合并 API 映射与约束检查结果（subagent 1）
2. 合并样例搜索结果（subagent 2），选择**最佳匹配**
3. 合并教程建议（subagent 3），补充设计策略
4. **合并关键常量**：汇总 subagent 1（event_id 上限/对齐要求/Cube tile 约束）和 subagent 2（UB 容量/stride 阈值）探测到的常量，填入 EXPLORE_REPORT §7 环境常量快照表，标注来源路径
5. 若存在多个高质量参考，列出 Top 3 并说明推荐首选及理由

### 生成报告

基于 [templates/explore_report.md](templates/explore_report.md) 模板生成 `EXPLORE_REPORT.md`。

---

## Checklist

### PRO_MATERIAL_INDEX.md

1. 文件存在且为本次重新扫描生成（非直接拷贝模板）
2. 三个一级章节（`§A` / `§B` / `§C`）存在且内容不为空
3. `§A` API 文档数量与 `grep -rl "pypto_pro\|PyPTO-Pro" "$PYPTO_DEVKIT_DIR/docs/api/" --include="*.md" | wc -l` 结果一致（不遗漏任何含标记的文档）
4. `§A` 下至少包含 SIMD-API 各子类别、Utils-API、SIMT-API
5. `§B` 下按 pro_ops 子目录拆分表格
6. `§C` 下列出 `$PYPTO_DEVKIT_DIR/docs/pypto_pro/` 下全部 `.md` 文件
7. 所有路径为缓存内路径（`$PYPTO_DEVKIT_DIR/...`，下游 Stage 可直接 `Read`）

### EXPLORE_REPORT.md

1. 文件存在
2. 以下章节存在且内容不为空：
   - `## 1. 概述`
   - `## 3. API 文档探索`（须包含 §3.1 API 映射结果 + §3.3 API 约束）
   - `## 4. 算子样例探索`（可标注「无匹配」但不可缺失 §4.1–§4.3 三个子章节）
   - `## 5. 教程与设计指南探索`（须遍历 §C 中索引的全部教程文档并给出适用性评估）
   - `## 6. Tile / 同步策略建议`（综合 §3+§4+§5 三个来源）
   - `## 7. 环境常量快照`（须含 UB 容量、event_id 上限、对齐要求等，标注来源路径）
   - `## 8. 风险评估`
   - `## 9. 证据索引`（须包含 §9.1 API 文档 / §9.2 算子样例 / §9.3 教程文档）
   - `## 10. 结论`
3. 无 "unsupported" 阻断项（或虽有但已给出替代方案）

---

## 错误处理

| 场景 | 处理 |
|------|------|
| 输入无法解析 | 引导用户提供公式或代码 |
| API 不存在 | 标记 unsupported，在风险中说明，尝试 pl.* 组合替代 |
| 约束不满足 | 标记 ✗，在风险中给出替代方案 |
| 无匹配样例 | 在「参考实现」章节标注「无匹配」，不阻断流程 |
