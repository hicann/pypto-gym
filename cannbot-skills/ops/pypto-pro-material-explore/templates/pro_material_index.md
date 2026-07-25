# PyPTO-Pro 全部资料索引

> **自动生成时间**: {timestamp}
> **说明**: 本索引 §A/§C 由扫描命令动态生成，§B 为官方指定算子固定清单。每次执行须重新扫描 §A/§C，§B 以官方最新指定清单为准。

---

## §A API 文档（`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/`）

> 搜索范围：`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/` 递归获取所有 `.md` 文档

### 扫描命令

```bash
# 主扫描：获取所有 PyPTO-Pro API 文档（现已统一归档至 $PYPTO_DEVKIT_DIR/docs/pypto_pro/api/）
find $PYPTO_DEVKIT_DIR/docs/pypto_pro/api/ -name "*.md" | sort

# API 总索引（固定）
# $PYPTO_DEVKIT_DIR/docs/pypto_api_list.md

# 按目录路径自动分组（以 $PYPTO_DEVKIT_DIR/docs/pypto_pro/api/ 下实际子目录为准）
```

### A.1 API 总索引

| 文档 | 路径 |
|------|------|
| PyPTO-Pro API 总索引 | `$PYPTO_DEVKIT_DIR/docs/pypto_api_list.md` |

### A.2+ 按扫描结果填充

<!-- 将 find 扫描结果按目录路径分组为多个三级标题 `### A.x {分类名}（N 文档）`。按 `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/` 下实际子目录拆分。若无结果保留空表（标注 `<!-- 空 -->`）。 -->

| # | 文档 | 路径 | 子类别 |
|---|------|------|--------|
| {n} | {name} | `{relative_path}` | {category} |

---

## §B 官方指定算子样例

> **重要**：以下为官方明确允许 agent 开发算子时参考的算子代码，是**唯一的算子写法参考来源**。`$PYPTO_DEVKIT_DIR/pro_ops/` 下其余文件**不得**作为样例参考或索引对象（orchestrator 资源缓存准备时已按清单清理，仅保留清单内文件）。
>
> **生成方式**：直接复制 `.opencode/skills/pypto-pro-material-explore/references/official_samples.md` 的清单内容。清单是该统一索引来源的唯一维护点——增删样例时只改该文件，无需改动其他文件。

### 扫描命令

```bash
# 直接读取统一清单文件（orchestrator 资源缓存准备时已按此清单清理 pro_ops/）
cat .opencode/skills/pypto-pro-material-explore/references/official_samples.md
```

### 按清单填充

<!-- 将 references/official_samples.md 的表格内容复制到此处。路径保持 $PYPTO_DEVKIT_DIR/pro_ops/... 形式。 -->

| # | 算子名称 | 缓存相对路径 | 类型 | 描述 |
|---|---------|-------------|------|------|
| {从 official_samples.md 复制} | | | | |

---

## §C 教程文档（`$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials`）

### 扫描命令

```bash
# 覆盖 $PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials 目录
find $PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials -name "*.md" | sort
```

### 按扫描结果填充

<!-- 将扫描结果填入下表。以实际扫描结果为准，不预设固定文件列表。 -->

| # | 文档 | 路径 |
|---|------|------|
| {n} | {name} | `{relative_path}` |
