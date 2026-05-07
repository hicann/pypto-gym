---
schema_version: 1
op_name: inplace_add_rms_norm
category: fused-norm
supported_dtypes: [bfloat16]
axes_list: ['B', 'S']
axes_ranges: {B: [1, 144], S: [1, 8192]}
shape_constraints: 'B*S in [1024, 8192] or (B in [16, 144] and S == 1)'
p0_shapes: [[16, 128, 7168], [8, 128, 7168], [64, 128, 7168], [144, 1, 7168]]
performance_target: '首跑精度成功性能的 2 倍'
atol: 0.001
rtol: 0.001
tolerance: {atol: 0.001, rtol: 0.001}
default_params: {eps: 1.0e-6}
inplace_semantics: true
buffer_aliasing:
  x1_out: 'normalized output (RmsNorm(x1+x2) * gamma)'
  x2_out: 'add result (x1+x2)'
  rstd_out: '1/sqrt(mean((x1+x2)^2) + eps)'
user_constraints:
  no_hooks: true
  wrapper_passthrough_only: true
  all_compute_in_pypto_kernel: true
---

# inplace_add_rms_norm 算子规格

## 1. 概述

`inplace_add_rms_norm` 是一个**带 inplace 语义**的融合算子，先对两个输入做 elementwise add，再对相加结果沿最后一维做 RMSNorm。所有输出**原地写回输入 buffer**，不分配新输出张量。常用于 LLM 残差加 RMSNorm 的融合场景，避免重复内存分配。

## 2. 数学公式

输入：
- `x1`, `x2` ∈ R^{B×S×H}
- `gamma` ∈ R^{H}
- `eps` ∈ R (标量)

计算：
```
x_add = x1 + x2                                # shape [B, S, H]
ms    = mean(x_add^2, dim=-1, keepdim=True)    # shape [B, S, 1]
rstd  = 1 / sqrt(ms + eps)                     # shape [B, S, 1]
y     = x_add * rstd * gamma                   # shape [B, S, H]
```

**Inplace 写回（核心语义）**：
- `x1` ← `y`        （归一化结果写回 x1 buffer）
- `x2` ← `x_add`    （add 结果写回 x2 buffer）
- `rstd` ← `rstd`   （独立输出 buffer）

## 3. 关键特性

| 特性 | 状态 | 实现说明 | 优先级 |
|------|------|----------|--------|
| add+rmsnorm 融合 | ✓ 需要 | 单 kernel 内完成 add → square → reduce → rsqrt → mul gamma | P0 |
| **Inplace 写回 x1（归一化结果）** | ✓ 需要 | 输出张量与 x1 共享 buffer | P0 |
| **Inplace 写回 x2（add 结果）** | ✓ 需要 | 中间相加结果与 x2 共享 buffer | P0 |
| 输出 rstd | ✓ 需要 | 独立 buffer，反向需要 | P0 |
| bfloat16 输入 | ✓ 需要 | 输入/输出全部 bf16；内部累加可用 fp32 | P0 |
| 动态 shape (B, S) | ✓ 需要 | 沿 B、S 切 tile；H=7168 固定 | P0 |
| dropout / mask | ✗ 不需要 | - | - |

## 4. 数据流图

```
   x1 [B,S,H] bf16        x2 [B,S,H] bf16        gamma [H] bf16        eps (float)
       │                       │                       │                  │
       └────────── + ──────────┘                       │                  │
                   │                                   │                  │
                   ▼                                   │                  │
           x_add [B,S,H] ──────► (写回 x2 buffer, inplace)                │
                   │                                   │                  │
                   ▼                                   │                  │
            x_add^2 (per element)                      │                  │
                   │                                   │                  │
                   ▼                                   │                  │
            mean over H                                │                  │
                   │                                   │                  │
                   ▼                                   │                  │
              + eps; rsqrt                             │                  │
                   │                                   │                  │
                   ▼                                   │                  │
           rstd [B,S,1] ───► (独立输出 buffer)         │                  │
                   │                                   │                  │
                   ▼                                   ▼                  │
                       y = x_add * rstd * gamma  [B,S,H] bf16
                                  │
                                  ▼
                           写回 x1 buffer (inplace)
```

## 5. 输入输出规格

### 输入（同时也是 inplace 输出 buffer）

| 名称 | shape | dtype | 动态轴 | 角色 | 说明 |
|------|-------|-------|--------|------|------|
| x1 | [B, S, H] | bfloat16 | B, S | 输入 + inplace 输出 | 加法左输入；调用后内容被覆盖为归一化结果 |
| x2 | [B, S, H] | bfloat16 | B, S | 输入 + inplace 输出 | 加法右输入；调用后内容被覆盖为 x1+x2 |
| gamma | [H] | bfloat16 | - | 只读输入 | RMSNorm 缩放权重 |
| eps | scalar | float | - | 标量 | 数值稳定项，默认 1e-6 |

### 输出（实际语义）

| 名称 | shape | dtype | 来源 buffer | 说明 |
|------|-------|-------|-------------|------|
| x1 (覆盖) | [B, S, H] | bfloat16 | 复用 x1 输入 buffer | 归一化结果 = RmsNorm(x1+x2) * gamma |
| x2 (覆盖) | [B, S, H] | bfloat16 | 复用 x2 输入 buffer | x1+x2 加法结果 |
| rstd | [B, S, 1] | bfloat16 | 独立 buffer | 1/sqrt(mean((x1+x2)^2)+eps) |

H 固定为 7168。

## 6. 精度要求

- atol = 0.001
- rtol = 0.001
- 由于 bf16 表达范围有限，square+reduce 阶段建议在内部用 fp32 累加，最终 cast 回 bf16。

## 7. 动态轴

| 轴 | 含义 | 范围 |
|----|------|------|
| B | batch | [1, 144] |
| S | sequence length | [1, 8192] |

形状约束：`B*S ∈ [1024, 8192]` 或 `(B ∈ [16, 144] 且 S == 1)`。

P0 形状：
- [16, 128, 7168]
- [8, 128, 7168]
- [64, 128, 7168]
- [144, 1, 7168]

## 8. 边界条件

| 条件 | 处理 |
|------|------|
| zero | 正常计算（x=0 时 rstd = 1/sqrt(eps)） |
| inf | 正常传播 |
| nan | 正常传播 |

## 9. 性能目标

首跑精度通过的性能为基线，目标 2x。

## 10. 用户约束（强制）

- **禁止使用/编写 PyTorch hook 函数**
- **Python wrapper 仅做参数透传**：不得在 wrapper 内进行任何 reshape、broadcast、cast、view、contiguous、unsqueeze、squeeze 等预处理
- **所有计算逻辑必须在 PyPTO kernel 中完成**：包括 add、square、reduce、rsqrt、乘 gamma、必要的 dtype 提升与回落、broadcast gamma [H] 到 [B,S,H]
- **Inplace 写回必须在 kernel 内显式实现**：x1 buffer 写归一化结果、x2 buffer 写 add 结果、rstd 写独立 buffer

## 11. 参考与场景

- 应用：LLM 残差加 RMSNorm 融合（DeepSeek、LLaMA 类残差路径）
- 参考：标准 RMSNorm 公式 + elementwise add 融合 + inplace buffer 复用

## 12. 典型配置

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 性能_P0_a | 性能 | P0 | eps=1e-6 | x1/x2=[16,128,7168], gamma=[7168] | x1=[16,128,7168], x2=[16,128,7168], rstd=[16,128,1] | B*S=2048 |
| 性能_P0_b | 性能 | P0 | eps=1e-6 | x1/x2=[8,128,7168], gamma=[7168]  | 同上结构 | B*S=1024 最小边界 |
| 性能_P0_c | 性能 | P0 | eps=1e-6 | x1/x2=[64,128,7168], gamma=[7168] | 同上结构 | B*S=8192 最大边界 |
| 性能_P0_d | 性能 | P0 | eps=1e-6 | x1/x2=[144,1,7168], gamma=[7168]  | 同上结构 | S=1 特殊场景 |
| 功能_P0   | 功能 | P0 | eps=1e-6 | x1/x2=[1,16,7168], gamma=[7168]   | 同上结构 | 小数据量基础功能 |

## 13. 生成时间

2026-05-06
