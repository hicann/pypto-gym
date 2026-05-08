# 断裂点识别报告

## 1. 摘要

- **会话时间**: {session_time}
- **检测时间**: {detection_time}
- **断裂点总数**: {total_count} 个
- **按优先级分布**:
  - 致命: {critical_count} 个
  - 高: {high_count} 个
  - 中: {medium_count} 个
- **按根因归属分布**:
  - 文档: {doc_count} 个
  - 框架: {framework_count} 个
  - 两者: {both_count} 个
- **按 Issue 类型分布**:
  - Bug Report: {bug_count} 个
  - Documentation: {doc_issue_count} 个
  - Feature Request: {feature_count} 个

## 2. 环境信息

- **CANN 版本**: {cann_version}
- **PyPTO Commit**: {pypto_commit}
- **服务器类型**: {server_type}
- **Python 版本**: {python_version}
- **操作系统**: {os_info}

## 3. 子会话分析概览

<!-- 如无子会话或 scope=current，省略此章节 -->

- **分析模式**: 完整分析（scope=full）
- **子会话总数**: {child_session_count} 个
- **子会话断裂点总数**: {child_fp_count} 个

| 子会话短 ID | 标题 | 断裂点数 | 致命 | 高 | 中 |
|-------------|------|----------|------|----|----|
| {child_short_id} | {child_title} | {child_fp_total} | {child_critical} | {child_high} | {child_medium} |

<!-- 每个子会话一行，按创建时间排序 -->

## 4. 优先修复列表

| 优先级 | 编号 | 类型 | 实体 | 来源 Session | Issue 类型 | 置信度 |
|--------|------|------|------|--------------|------------|--------|
| {priority} | {fp_id} | {fp_type} | {entity} | {source_session} | {issue_type} | {confidence} |

<!-- source_session: "主 Session" 或 "子 Session-{short_id}" -->
<!-- 按优先级排序：致命 > 高 > 中，同优先级内按置信度排序：高 > 中 -->

## 5. Session 级断裂点

<!-- 如无 Session 级断裂点（C5、C6），省略此章节 -->

### 5.1 [{fp_id}] {type_code}-{type_name}

- **类型**: {type_code}-{type_name}
- **优先级**: {priority}
- **根因归属**: 两者
- **Issue 类型**: {issue_type}
- **建议 Issue 标题**: `{issue_title}`
- **置信度**: {confidence}
- **聚合分类**: {aggregate_type}
- **分类依据**: {aggregate_reason}

#### 问题描述

{description}

#### 证据片段

```
{evidence}
```

#### 优化建议

{suggestion}

## 6. 实体级断裂点详情

### 6.1 [{fp_id}] {type_code}-{type_name}: {entity}

- **类型**: {type_code}-{type_name}
- **优先级**: {priority}
- **根因归属**: {root_cause}
- **Issue 类型**: {issue_type}
- **建议 Issue 标题**: `{issue_title}`
- **置信度**: {confidence}
- **来源 Session**: {source_session}
- **可能关联**: {related_fps}
- **聚合分类**: {aggregate_type}
- **分类依据**: {aggregate_reason}

#### 问题描述

{description}

#### 证据片段

```
{evidence}
```

#### 复现步骤

1. {step1}
2. {step2}
3. {step3}

#### 期望行为

{expected}

#### 实际行为

{actual}

#### 相关链接

- {links}

#### 优化建议

1. {suggestion1}
2. {suggestion2}

---

<!-- 每个断裂点使用 --- 分隔，重复 6.x 的结构 -->
