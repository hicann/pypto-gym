---
schema_version: 1
op_name: kda_dynamic_bs
supported_dtypes: ["DT_FP32"]
dynamic_axes: ["B", "S"]
shape_constraints: "query/key/value/alpha/beta 均为 [B,S,D]，其中 B/S 为动态轴，D 为静态正整数"
tiling_required: true
feasibility: "可行"
---

# API 探索报告

> **生成时间**: 2026-04-17

---

## 1. 概述

### 1.1 输入摘要

- 目标算子：KDA 风格递推注意力前向核心
- 关键需求：`B/S` 双动态轴
- 计算特征：以递推状态 `state[D,D]` 为核心，逐 token 更新并输出
- 数据类型：FP32

### 1.2 算子分类

- **类型**: `Vector`
- **判断依据**:
  - 本实现采用 `expand_clone + mul + sum` 构造外积与加权归约
  - 不依赖 cube matmul 主路径

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|---|---|---|---|
| 1 | shape/index | `x[b,t,:]` | 用 `pypto.view` 取当前 token |
| 2 | shape | `[1,1,D] -> [D,1]/[1,D]` | 用 `pypto.reshape` 准备广播 |
| 3 | elementwise | `Outer = k_col * v_row` | 通过 `expand_clone` 生成 `[D,D]` 后逐元素乘 |
| 4 | elementwise | `state = state*alpha + Outer*beta` | 递推状态更新 |
| 5 | reduction | `out = sum(state * q_col, dim=0)` | 生成 `[1,D]` 输出 |
| 6 | writeback | `assemble` | 写回 `output[b,t,:]` 与 `last_state[b,:,:]` |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|---|---|---|---|---|
| 1 | token 切片 | `pypto.view` | direct | ✓ |
| 2 | 形状变换 | `pypto.reshape` | direct | ✓ |
| 3 | 一维扩展 | `pypto.expand_clone` | direct | ✓ |
| 4 | 逐元素更新 | `pypto.mul`, `pypto.add` | direct | ✓ |
| 5 | 归约求和 | `pypto.sum` | direct | ✓（FP32） |
| 6 | 循环控制 | `pypto.loop` | direct | ✓ |
| 7 | 写回输出 | `pypto.assemble` | direct | ✓ |
| 8 | tile 设置 | `pypto.set_vec_tile_shapes` | direct | ✓ |

### 3.2 Substitute 配方

本算子无 substitute 需求。

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|---|---|---|---|
| dtype | FP32 | FP32 | ✓ |
| contiguous | 必须 | wrapper 侧检查 | ✓（需确保） |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|---|---|---|---|
| `pypto.view` | shape 类型 | `List[int]`，不可含 SymbolicScalar | ✓ |
| `pypto.expand_clone` | 广播维限制 | 仅允许一维从 1 扩展 | ✓ |
| `pypto.sum` | dtype | 仅 FP32 | ✓ |
| `pypto.loop` | 索引类型 | 返回 SymInt，不可做 Python list 索引 | ✓ |
| `pypto.assemble` | 偏移合法性 | offsets < out.shape | ✓ |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|---|---|
| Vector | `pypto.set_vec_tile_shapes(...)` |

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|---|---|---|---|---|
| `examples/02_intermediate/controlflow/others/dynamic.py` | examples | 高 | 高 | 双动态轴 loop + view/assemble + valid_shape 模式 |
| `models/qwen3_next/gated_delta_rule_impl.py` | models | 中 | 高 | 递推状态更新思路、按 batch/seq 逐步推进 |
| `models/experimental/ops-transformer/flash_attention_score/flash_attention_score_impl.py` | models/experimental | 中 | 中 | 动态 attention 组织方式（参考） |

### 6.2 可复用模式

- **API 调用模式**：`loop -> view -> reshape -> compute -> assemble`
- **Tiling 策略**：每个 token tile 采用固定 `[D,D]` vec tile
- **Loop 结构**：外层 `B`，内层 `S`
- **边界处理**：`B/S` 全动态，单 token 视图无需尾块补齐

### 6.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|---|---|---|---|
| 外积计算 | 多用 `matmul` | `k[:,None] * v[None,:]` | 用 `expand_clone + mul` 实现，规避 K=1 matmul 约束 |
| 输出生成 | `matmul` 或直接写回 | `sum(q*state, dim=0)` | 用 FP32 `sum` 路径 |
| 动态轴数 | 单动态或多动态混合 | 明确双动态 `B/S` | kernel 注解显式标 `pypto.DYNAMIC` |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|---|---|---|
| `D` 过大导致 UB 压力 | `state` 与多个 `[D,D]` 中间量并存 | v1 聚焦 `D=64/128`，后续做性能分块优化 |
| 输入不连续引发错误 | `from_torch` 连续性要求 | wrapper 强制 `contiguous` 检查 |

### 7.2 注意事项

| 注意点 | 说明 |
|---|---|
| `pypto.view` 的 `shape` 不可含符号值 | 动态只放在 `offsets` |
| `sum` 只支持 FP32 | 全链路固定 FP32 |
| 同图避免读写回环 | 输出只写 `output/last_state`，不回写输入 |

---

## 8. 证据索引

| 信息 | 文档路径 |
|---|---|
| `view` 约束 | `docs/api/operation/pypto-view.md` |
| `assemble` 约束 | `docs/api/operation/pypto-assemble.md` |
| `expand_clone` 约束 | `docs/api/operation/pypto-expand_clone.md` |
| `loop` 约束 | `docs/api/controlflow/pypto-loop.md` |
| `set_vec_tile_shapes` | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| 动态轴示例 | `examples/02_intermediate/controlflow/others/dynamic.py` |
| 递推参考实现 | `models/qwen3_next/gated_delta_rule_impl.py` |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**: v1 为正确性优先实现，性能未优化；`D` 建议先使用典型值（64/128）
