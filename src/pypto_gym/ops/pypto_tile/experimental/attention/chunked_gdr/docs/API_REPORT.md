---
schema_version: 1
op_name: chunked_gated_delta_rule
supported_dtypes: [float32]
dynamic_axes: ['T', 'B']
shape_constraints: 'Nv % Nqk == 0; D == 128; L == 128'
tiling_required: true
feasibility: feasible
---

# API 探索报告

> **生成时间**: 2026-05-22

---

## 1. 概述

### 1.1 输入摘要

chunked_gated_delta_rule 算子实现分块门控 Delta Rule 线性注意力机制，核心计算包含：L2 归一化、预注意力矩阵计算、分块递推矩阵求逆、累积衰减计算、循环状态注意力更新。输入为 10 个 tensor（query, key, value, beta, gate, states, mask, tril_mask, eye, act_seq_len），输出为 2 个 tensor（core_attn_out, last_state_data）。全部使用 FP32 dtype。支持动态 Shape (T, B) 和 GQA。

### 1.2 算子分类

- **类型**: 混合 (Cube + Vector)
- **判断依据**: 算子包含大量 `pypto.matmul`（Cube 操作）和向量运算（sqrt, exp, sum, mul, div 等），属于 Cube+Vector 混合算子。需要同时配置 `set_cube_tile_shapes` 和 `set_vec_tile_shapes`。

---

## 2. 公式分解

| 步骤 | 操作类型 | 数学表达 | 说明 |
|------|----------|----------|------|
| 1 | elementwise+reduction | `q_norm = q / sqrt(sum(q²) + eps)` | L2 归一化，向量操作 + 归约 |
| 2 | matmul | `g_cum = tril @ g` | 门控累积，矩阵乘法 |
| 3 | elementwise | `decay_mask = exp((g_cum - g_cum^T) * tril)` | 衰减掩码，向量操作 |
| 4 | elementwise | `k_beta = k * beta` | 加权键，逐元素乘 |
| 5 | matmul | `A = k_beta @ k^T * decay_mask * mask` | 预注意力矩阵，矩阵乘法 + 逐元素乘 |
| 6 | matmul+elementwise | `A_inv = inverse_pto(A)` | 分块递推矩阵求逆，多次 matmul + 逐元素操作 |
| 7 | elementwise+matmul | `v_out = A_inv @ (v * beta)` | 累积衰减，逐元素乘 + matmul |
| 8 | elementwise+matmul | `k_cumdecay = A_inv @ (k_beta * exp(g_cum))` | 累积衰减，逐元素乘 + matmul |
| 9 | matmul | `v_prime = k_cumdecay @ S^T` | 循环状态 matmul |
| 10 | matmul | `o_inter = q*g_exp @ S^T` | 循环状态 matmul |
| 11 | matmul | `attn = q @ k^T` | 注意力矩阵 matmul |
| 12 | matmul | `chunk_attn_value/vprime = attn * decay * tril @ v/v_prime` | 分块注意力 matmul |
| 13 | elementwise+matmul | `S_new = S*exp(g_last) + v^T@kg - v'^T@kg` | 状态更新，逐元素 + matmul |

---

## 3. API 映射

### 3.1 映射结果

| 步骤 | 数学表达 | PyPTO API | 映射级别 | 约束满足 |
|------|----------|-----------|----------|----------|
| L2归一化 | `q / sqrt(sum(q²)+eps)` | `pypto.sqrt`, `pypto.sum`, 逐元素`*`/`/` | direct | ✓ |
| 门控累积 | `tril @ g` | `pypto.matmul(tril, g, DT_FP32)` | direct | ✓ |
| 衰减掩码 | `exp((g_cum - g_cum^T) * tril)` | `pypto.exp`, `.transpose()`, 逐元素`*` | direct | ✓ |
| 加权键 | `k * beta` | 逐元素 `*` | direct | ✓ |
| 预注意力 | `k_beta @ k^T` | `pypto.matmul(k_beta, k, DT_FP32, b_trans=True)` | direct | ✓ |
| 矩阵求逆 | `inverse_pto(A)` | `pypto.matmul` + `pypto.concat` + `pypto.view` + 逐元素运算（分块递推法） | substitute（组合实现） | ✓ |
| 累积衰减v | `A_inv @ v_beta` | `pypto.matmul(A_inv, v_beta, DT_FP32)` | direct | ✓ |
| 累积衰减k | `A_inv @ k_beta*g_exp` | `pypto.matmul(A_inv, weighted_k, DT_FP32)` | direct | ✓ |
| 循环状态 | `k_cum @ S^T` | `pypto.matmul(k_cum, S, DT_FP32, b_trans=True)` | direct | ✓ |
| 注意力矩阵 | `q @ k^T` | `pypto.matmul(q, k, DT_FP32, b_trans=True)` | direct | ✓ |
| 分块注意力 | `attn_masked @ v` | `pypto.matmul(attn_masked, v, DT_FP32)` | direct | ✓ |
| 状态更新 | `v^T @ kg` | `pypto.matmul(v, kg, DT_FP32, a_trans=True)` | direct | ✓ |
| 子视图切片 | 3D→2D切片 | `pypto.view(query, [l,1,d], [bs_ofs,nqk_idx,0], valid_shape=[actual_l,1,d])` | direct | ✓ |
| reshape动态 | 3D→2D | `pypto.reshape(view, [l,d], valid_shape=[actual_l,d])` | direct | ✓ |
| 循环控制 | 3层嵌套循环 | `pypto.loop(0, s, l, name, idx_name, unroll_list)` | 需注意：unroll_list需用`pypto.loop_unroll` | ⚠ |
| 不满chunk填充 | 尾块padding | `pypto.fillpad(tensor, "constant", 0.0)` | direct | ⚠（仅1-2维） |
| 输出拼接 | 尾块回写 | `pypto.assemble(chunk_out, [bs_ofs, nv_idx, 0], core_attn_out)` | direct | ✓ |
| 广播克隆 | gate扩展 | `pypto.expand_clone(gate[l-1:l,:], (dv, 1))` | direct | ✓ |
| 零矩阵创建 | zeros_16/32/64 | `pypto.full(size=[16,16], fill_value=0.0, dtype=DT_FP32)` | direct | ✓ |
| 拼接 | 求逆中concat | `pypto.concat(list, dim=1)` | direct | ⚠（validShape不自动推导） |
| 循环终止 | 最后chunk判断 | `pypto.is_loop_end(s_idx)` | direct | ✓ |
| JIT入口 | kernel装饰器 | `pypto.frontend.jit(runtime_options={...})` | direct | ✓ |
| TileShape配置 | cube+vec | `pypto.set_cube_tile_shapes`, `pypto.set_vec_tile_shapes` | direct | ✓ |
| 内存管理 | scope控制 | `pypto.set_pass_options(sg_set_scope=1/-1)` | direct | ✓ |
| 合轴优化 | combine_axis | `pypto.experimental.set_operation_options(combine_axis=True)` | direct | ✓ |

### 3.2 Substitute 配方

```
矩阵求逆 (inverse_pto): 分块递推法实现
  - 将 128×128 矩阵分为 8×8 个 16×16 子块
  - 使用 inverse_pto_min_length 逐行递推（concat + view + reshape + sum + 逐元素运算）
  - 使用 inverse_matmul 逐步合并（pypto.matmul 应用 Schur 补公式）
  - 需要 zeros_16, zeros_32, zeros_64 零矩阵用于填充右上角
```

```
loop(unroll_list): 需使用 pypto.loop_unroll 替代 pypto.loop
  - pypto.loop 不支持 unroll_list 参数
  - pypto.loop_unroll 支持 unroll_list 配置，返回 (idx, unroll_factor) 元组
  - 注意：unroll_list 会增加编译出的图数量
```

---

## 4. 约束检查

### 4.1 入口约束

| 约束项 | 要求 | 输入值 | 结果 |
|--------|------|--------|------|
| dtype | FP32/BF16/FP16/INT32等 | 全部 DT_FP32（act_seq_len为DT_INT32） | ✓ |
| contiguous | 必须连续 | 所有输入 tensor contiguous | ✓（需确保 from_torch 传入连续 tensor） |

### 4.2 API 约束

| API | 约束项 | 要求 | 结果 |
|-----|--------|------|------|
| matmul | FP32不支持NZ格式 | 使用ND格式 | ✓ |
| matmul | 需先设置cube_tile_shapes | 每次matmul前调用 | ✓（需在设计阶段规划） |
| matmul | 左右矩阵dtype一致 | FP32+FP32 | ✓ |
| matmul | FP32场景16元素对齐 | TileShape每维16元素对齐 | ✓（128/64等均满足） |
| view | shape不支持SymbolicScalar | shape必须为List[int] | ✓（子切片shape为固定值l=128等） |
| view | offsets和valid_shape支持SymbolicScalar | 动态offset使用循环索引 | ✓ |
| reshape | 动态轴不支持-1推导 | 显式指定所有维度 | ✓（使用tensor.shape获取维度） |
| fillpad | 仅支持1-2维 | 不满chunk数据需先reshape到2D | ⚠（需在实现中处理） |
| concat | validShape不自动推导 | 需手动计算确保正确 | ⚠（需在实现中处理） |
| sum | 尾轴32bytes对齐 | FP32场景8元素对齐 | ✓（D=128满足） |
| sum | TileShape不超过64KB | D=128×128×4=64KB刚好满足 | ⚠（临界值，需验证） |
| full | fill_value和dtype类型匹配 | 0.0(float)+DT_FP32 | ✓ |
| loop | 不支持unroll_list | 需用loop_unroll | ⚠（需调整实现） |

---

## 5. Tiling 需求

| 算子类型 | 需调用 API |
|----------|-----------|
| 混合(Cube+Vector) | `pypto.set_cube_tile_shapes()` + `pypto.set_vec_tile_shapes()` |

**Cube TileShape 配置要点**:
- FP32 场景所有维度需16元素对齐（mL0/mL1/kL0/kL1/nL0/nL1）
- 不同matmul场景需动态切换配置：
  - 大M大N: `[128,128],[128,128],[128,128]`
  - 小M小N: `[128,128],[128,128],[64,64]`
  - 小M大K: `[64,64],[128,128],[128,128]`

**Vector TileShape 配置要点**:
- 维度数与当前operation输出维度一致
- 每维 > 0，最多4维
- sum归约需保证尾轴32bytes对齐

---

## 6. 参考实现

### 6.1 匹配示例

| 示例路径 | 来源 | 相似度 | 置信度 | 可复用点 |
|----------|------|--------|--------|----------|
| `models/qwen3_next/gated_delta_rule_impl.py` | models | **高** | 极高 | 算子直接实现：分块矩阵求逆、aligned/unaligned双路径、状态递推、全部API使用模式 |
| `models/qwen3_next/qwen3_next_gated_delta_rule.py` | models | **高** | 极高 | PyTorch golden参考、测试入口、数据生成、动态分发逻辑 |
| `models/glm_v4_5/glm_attention.py` | models | **中** | 高 | GQA head映射模式、sg_set_scope分阶段内存管理、is_loop_end分支、assemble分块拼接 |
| `models/kimi/block_attn_res_impl.py` | models | **中** | 高 | sg_set_scope=1/-1规约前后内存管理、reshape(valid_shape)动态形状 |
| `models/arctic/sum_lstm.py` | models | **中** | 高 | 循环状态递推模式、双输出tensor、多函数拆分 |
| `examples/03_advanced/advanced_nn/attention/attention.py` | examples | **高** | 高 | matmul+loop+view(valid_shape)+Cube/Vector混合组合模式 |
| `examples/02_intermediate/controlflow/others/dynamic.py` | examples | **高** | 高 | DYNAMIC+ceil_div+min边界+view/assemble完整组合 |
| `examples/02_intermediate/controlflow/condition/condition.py` | examples | **高** | 高 | is_loop_begin/is_loop_end边界条件用法 |
| `examples/01_beginner/transform/transform_ops.py` | examples | **中** | 高 | view(valid_shape)、concat、assemble基础用法 |
| `examples/01_beginner/compute/matmul_ops.py` | examples | **中** | 高 | matmul+a_trans/b_trans+cube_tile_shapes用法 |

### 6.2 可复用模式

- **API 调用模式**: `pypto.view + valid_shape + pypto.reshape + pypto.matmul` 的标准动态切片→计算→回写流程
- **Tiling 策略**: cube_tile_shapes 根据不同matmul的M/N/K动态切换，vec_tile_shapes 统一设置(128,128)
- **Loop 结构**: 三重嵌套循环（B→Nv→S），S循环使用loop_unroll配置unroll_list=[16,1]
- **边界处理**: `actual_l = (s - s_idx).min(l)` 计算实际chunk长度，is_loop_end判断是否最后一个chunk，fillpad填充不满chunk
- **内存管理**: `sg_set_scope=1` 在inverse_pto阶段包裹zeros_16/32/64分配，scope结束后释放
- **状态递推**: `last_state[:] = cur_state` 在chunk间传递状态

### 6.3 差异分析

| 差异点 | 参考做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| loop参数 | qwen3_next使用`pypto.loop(unroll_list=[16,1])` | `pypto.loop`不支持unroll_list | 使用`pypto.loop_unroll`替代 |
| stitch配置 | qwen3_next使用`stitch_function_max_num: 2` | original.md建议128 | 保持2（参考生产级实现），后续性能调优时调整 |
| fillpad维度 | 直接对2D tensor fillpad | 如果输入3D需先reshape | 参考unaligned版本：先reshape到2D再fillpad |
| concat validShape | qwen3_next未特别处理 | 需手动计算确保正确 | 在inverse_pto_min_length中手动计算 |

---

## 7. 风险评估

### 7.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无阻断问题 | 所有核心API存在且FP32约束满足 | — |

### 7.2 注意事项

| 注意点 | 说明 |
|--------|------|
| fillpad仅支持1-2维 | unaligned版本中对3D tensor做fillpad时，需先reshape到2D再fillpad再reshape回3D |
| loop不支持unroll_list | 必须使用`pypto.loop_unroll`替代，注意unroll_list会增加编译图数量 |
| concat不自动推导validShape | 在inverse_pto的concat调用中需手动计算validShape |
| sum尾轴32bytes对齐 | L2Norm中sum(-1)归约的尾轴需8元素对齐（D=128满足） |
| matmul FP32不支持NZ格式 | 确保所有输入tensor使用ND格式 |
| matmul需先设置cube_tile_shapes | 每次matmul调用前必须先调用set_cube_tile_shapes |
| reshape动态轴不支持-1 | 需从tensor.shape获取SymbolicScalar显式指定维度 |
| matmul不直接接收DYNAMIC tensor | 需通过loop+view切片传入concrete shape子视图 |

---

## 8. 证据索引

| 信息 | 文档路径 |
|------|----------|
| API存在性 | `docs/api/operation/index.md` |
| pypto.matmul文档 | `docs/api/operation/pypto-matmul.md` |
| pypto.view文档 | `docs/api/operation/pypto-view.md` |
| pypto.reshape文档 | `docs/api/operation/pypto-reshape.md` |
| pypto.loop文档 | `docs/api/controlflow/pypto-loop.md` |
| pypto.loop_unroll文档 | `docs/api/controlflow/pypto-loop_unroll.md` |
| pypto.fillpad文档 | `docs/api/operation/pypto-fillpad.md` |
| pypto.assemble文档 | `docs/api/operation/pypto-assemble.md` |
| pypto.concat文档 | `docs/api/operation/pypto-concat.md` |
| pypto.sum文档 | `docs/api/operation/pypto-sum.md` |
| pypto.exp文档 | `docs/api/operation/pypto-exp.md` |
| pypto.sqrt文档 | `docs/api/operation/pypto-sqrt.md` |
| pypto.expand_clone文档 | `docs/api/operation/pypto-expand_clone.md` |
| pypto.is_loop_end文档 | `docs/api/controlflow/pypto-is_loop_end.md` |
| pypto.frontend.jit文档 | `docs/api/config/pypto-frontend-jit.md` |
| pypto.set_vec_tile_shapes文档 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| pypto.set_cube_tile_shapes文档 | `docs/api/config/pypto-set_cube_tile_shapes.md` |
| pypto.set_pass_options文档 | `docs/api/config/pypto-set_pass_options.md` |
| pypto.full文档 | `docs/api/operation/pypto-full.md` |
| DataType枚举 | `docs/api/datatype/DataType.md` |
| 入口约束 | `docs/api/others/pypto-from_torch.md` |
| 主要参考实现 | `models/qwen3_next/gated_delta_rule_impl.py` |
| Golden参考 | `models/qwen3_next/qwen3_next_gated_delta_rule.py` |
| 动态shape示例 | `examples/02_intermediate/controlflow/others/dynamic.py` |
| Attention示例 | `examples/03_advanced/advanced_nn/attention/attention.py` |

---

## 9. 结论

- **可行性**: 可行
- **主要问题**: 需注意 `pypto.loop` 不支持 unroll_list（需用 `loop_unroll` 替代），`fillpad` 仅支持 1-2 维（需 reshape 处理），`concat` 不自动推导 validShape（需手动计算）。其余所有核心 API 均存在且 FP32 dtype 支持满足要求。参考实现 `models/qwen3_next/gated_delta_rule_impl.py` 覆盖全部所需 API 模式，可直接作为实现参考。