---
schema_version: 2
op_name: {operator_name}
feasibility: {feasibility}
---

# PyPTO-Pro 资料探索报告

> **生成时间**: {timestamp}
> **全量资料索引**: `custom/{operator_name}/PRO_MATERIAL_INDEX.md`

---

<!-- REQUIRED -->
## 1. 概述

### 1.1 输入摘要

{输入内容摘要}

### 1.2 算子分类

- **计算引擎类型**: {Vector / Cube / VF / 混合}
- **判断依据**: {公式中是否含 matmul（Cube）、仅逐元素/归约（Vector）、或需 VF 指令}

### 1.3 可能会涉及的 API 类别

> 以下为常见类别提示，实际 API 以本次 PRO_MATERIAL_INDEX §A 扫描结果为准。

| 类别 | 是否涉及 | 关键 API |
|------|---------|----------|
| 数据搬运 | {是/否} | `load_tile` / `store_tile` / `load` / `store` / `move` |
| 矢量计算 — 逐元素 | {是/否} | `add` / `sub` / `mul` / `div` / `maximum` / `relu` / `neg` / `cast` |
| 矢量计算 — 数学 | {是/否} | `exp` / `muls` / `expands` |
| 矢量计算 — 归约 | {是/否} | `row_max` / `row_sum` / `col_max` / `col_sum` / `row_reduce` / `col_reduce` |
| 矢量计算 — 广播 | {是/否} | `row_expand_sub` / `row_expand_div` / `col_expand_sub` |
| 矩阵计算（Cube） | {是/否} | `matmul` / `matmul_acc` |
| VF 计算 | {是/否} | `vf.add` / `vf.mul` / `vf.reduce` / `vf.cast` |
| 同步控制 | {auto/手动} | `pl.system.sync_src/sync_dst` / `pl.system.bar_v` / `pl.system.bar_all` / `mutex_lock/unlock` |
| 系统访问 | 是 | `get_block_idx` / `get_block_num` |
| 控制流 | 是 | `section_vector` / `section_cube` / `pl.range` |
| 工具 | {是/否} | `DynVar` / `set_validshape` |

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | pl.* 调用链 | 说明 |
|------|----------|----------|------------|------|
| {n} | {op_type} | {math_expr} | `pl.{api1}` → `pl.{api2}` | {desc} |

---

<!-- REQUIRED -->
## 3. API 文档探索

> **来源**: Explore subagent 1 — 基于 `PRO_MATERIAL_INDEX.md` §A

### 3.1 API 映射结果

| 步骤 | 数学表达 | PyPTO-Pro pl.* 调用链 | 状态 | 约束满足 |
|------|----------|-----------------------|------|----------|
| {n} | {expr} | `pl.{api1}` → `pl.{api2}` | {直接可用/需组合/不支持} | {✓/⚠/✗} |

### 3.2 替代方案

<!-- 仅标注"不支持"或"需组合"时填写 -->

### 3.3 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| {api} | {constraint} | {requirement} | {✓/⚠/✗} |

### 3.4 MemorySpace 约束

| 操作 | MemorySpace | 约束 |
|------|-------------|------|
| {op} | {space} | {constraint} |

---

<!-- REQUIRED -->
## 4. 算子样例探索

> **来源**: Explore subagent 2 — 基于 `PRO_MATERIAL_INDEX.md` §B
> **注意**: pro_ops/ 下文件是 API 用法参考 + 功能测试，非 production 标准

### 4.1 匹配样例

<!-- 无匹配时填写：无匹配参考实现 -->

| # | 示例路径 | 索引 §B.x | 相似度 | 完整度（多tile归约/双视图/online） | 可复用点 |
|---|----------|-----------|--------|-----------------------------------|----------|
| {n} | `pro_ops/{category}/{file}` | {B.x} | {高/中/低} | {✅多tile归约 ✅双视图 ✅online / 部分 / 简化路径} | {可复用点} |

> **选参考原则**：优先选完整度高的生产级实现，而非仅相似度高的简化/异引擎版本。若最相似样例是简化路径，须在此标注并另找完整参考。

### 4.2 可复用模式

- **pl.* 调用链**：{api_usage}
- **Tile 配置**：{tile_config}
- **同步策略**：{sync_pattern}
- **循环结构**：{loop_pattern}

### 4.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| {diff} | {example} | {need} | {suggestion} |

---

<!-- REQUIRED -->
## 5. 教程与设计指南探索

> **来源**: Explore subagent 3 — 基于 `PRO_MATERIAL_INDEX.md` §C

### 5.1 适用的设计模式

<!-- 遍历 PRO_MATERIAL_INDEX.md §C 中索引的全部教程文档（以索引实际扫描结果为准），逐行填写 -->

| 教程文档（索引 §C） | 设计模式 | 适用性 |
|---------------------|----------|--------|
| `{tutorial_path}` | {模式} | {对本算子的指导意义} |

### 5.2 来自教程的关键约束与建议

| 来源 | 约束/建议 | 影响 |
|------|----------|------|
| {tutorial} | {constraint} | {impact} |

---

<!-- REQUIRED -->
## 6. Tile / 同步策略建议

> **综合来源**: §3 API 约束 + §4 样例参考 + §5 教程指导

### 6.1 Tile 规格建议

| 维度 | 建议值 | 依据 |
|------|--------|------|
| {dim} | {value} | {来源章节} |

### 6.2 同步策略建议

- **推荐方案**: {auto_mutex / 手动 sync + bar_all}
- **理由**: {来源}

### 6.3 双视图需求

- **是否需要**: {是 / 否}
- **判定规则**: 归约类 API（row_max/row_sum 等）输出 `[行数,1]` 须设 `layout=pl.DN`（证据 `row_max.md:25`）；若该输出后续要参与 tile×tile 逐元素运算（需默认 ND 布局），则**必须**在同地址声明 DN + ND 双视图对
- **原因**: {来源}

---

<!-- REQUIRED -->
## 7. 环境常量快照

> **来源**: Explore subagent 1（API 文档约束）+ Explore subagent 2（pro_ops 样例 assert）
> 从当前仓库的 API 文档和 pro_ops 样例中提取硬件/版本相关常量。下游 Stage 3/4 全部引用本节，不再各自写死。

| 常量 | 探测值 | 来源路径 | 备注 |
|------|--------|----------|------|
| UB 容量 | {N} KB | {pro_ops 样例 assert 路径} | {如 `assert {addr} <= {N}*1024`} |
| cross_core event_id 上限 | {N} | {API 文档路径} | 本次资料探索确认的上限值；Stage 3/4 涉及 event_id 分配时统一引用本项，不在下游重复定义固定数值 |
| 地址对齐要求 | {N} 字节 | {API 文档路径} | 本次资料探索确认的对齐要求；Stage 3/4 的地址规划与代码实现统一引用本项，不在下游重复固化 |
| Cube tile 对齐 | {N} | {教程/文档路径} | {tile 尺寸对齐要求} |
| tile_dims stride 经验阈值 | {N} KB | {pro_ops 样例推断} | 非文档明文，为基于样例的经验推断值 |

---

<!-- REQUIRED -->
## 8. 风险评估

### 8.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| {issue} | {reason} | {suggestion} |

### 8.2 注意事项

| 注意点 | 说明 |
|--------|------|
| {warning} | {desc} |

---

<!-- REQUIRED -->
## 9. 证据索引

> **全量资料索引**: `custom/{operator_name}/PRO_MATERIAL_INDEX.md`

### 9.1 API 文档证据

| 信息 | 路径 |
|------|------|
| {api} 文档 | `{path}` |

### 9.2 算子样例证据

| 信息 | 路径 |
|------|------|
| {样例} | `{path}` |

### 9.3 教程文档证据

| 信息 | 路径 |
|------|------|
| {教程} | `{path}` |

---

<!-- REQUIRED -->
## 10. 结论

- **可行性**: {可行 / 需调整 / 不可行}
- **主要问题**: {main_issue}
