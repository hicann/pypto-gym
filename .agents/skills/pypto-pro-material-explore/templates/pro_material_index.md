# PyPTO-Pro 全部资料索引

> **自动生成时间**: {timestamp}
> **说明**: 本索引由 Step 1 的 bash 扫描命令动态生成，覆盖 devkit 缓存 `$PYPTO_DEVKIT_DIR` 下全部 PyPTO-Pro 相关资料。每次执行须重新扫描，不得直接拷贝本模板。路径统一为缓存内绝对路径（`$PYPTO_DEVKIT_DIR/...`），下游 Stage 可直接 `Read`。

---

## §A API 文档（`$PYPTO_DEVKIT_DIR/docs/api/`）

> 搜索范围：`$PYPTO_DEVKIT_DIR/docs/api/` 递归搜索所有含 "pypto_pro" / "PyPTO-Pro" 关键字的 `.md` 文档

### 扫描命令

```bash
# 主扫描：获取所有含 pypto_pro 标记的 API 文档（唯一权威来源，不预设目录）
grep -rl "pypto_pro\|PyPTO-Pro" "$PYPTO_DEVKIT_DIR/docs/api/" --include="*.md" | sort

# API 总索引（固定）
# $PYPTO_DEVKIT_DIR/docs/pypto_api_list.md

# 按目录路径自动分组：
#   SIMD-API/  → 按子目录（基础数据结构/计算API/...）分节
#   Utils-API/ → 独立分节
#   SIMT-API/  → 独立分节
#   其余       → 归入"其他"分节
```

### A.1 API 总索引

| 文档 | 路径 |
|------|------|
| PyPTO-Pro API 总索引 | `$PYPTO_DEVKIT_DIR/docs/pypto_api_list.md` |

### A.2+ 按扫描结果填充

<!-- 将 grep -rl 扫描结果按目录路径分组为多个三级标题 `### A.x {分类名}（N 文档）`。分类按 SIMD-API 子目录 / Utils-API / SIMT-API / 其他 拆分。若无结果保留空表（标注 `<!-- 空 -->`）。 -->

| # | 文档 | 路径 | 子类别 |
|---|------|------|--------|
| {n} | {name} | `{cache_path}` | {category} |

---

## §B 算子样例（`$PYPTO_DEVKIT_DIR/pro_ops/`）

> 按 pro_ops 子目录拆分（以实际扫描结果为准，不预设固定子目录列表）

### 扫描命令

```bash
find "$PYPTO_DEVKIT_DIR/pro_ops" -name "*.py" | sort
```

### 按子目录填充

<!-- 将扫描结果按子目录分组。每个子目录一个三级标题 `### B.x {子目录名}（N 文件）`，子目录名按字典序排列。 -->

| # | 文件 | 路径 |
|---|------|------|
| {n} | {name} | `{cache_path}` |

---

## §C 教程文档（`$PYPTO_DEVKIT_DIR/docs/pypto_pro/`）

### 扫描命令

```bash
# 覆盖整个 pypto_pro 目录（含 tutorials/ 及未来可能新增的子目录）
find "$PYPTO_DEVKIT_DIR/docs/pypto_pro" -name "*.md" | sort
```

### 按扫描结果填充

<!-- 将扫描结果填入下表。以实际扫描结果为准，不预设固定文件列表。 -->

| # | 文档 | 路径 |
|---|------|------|
| {n} | {name} | `{cache_path}` |
