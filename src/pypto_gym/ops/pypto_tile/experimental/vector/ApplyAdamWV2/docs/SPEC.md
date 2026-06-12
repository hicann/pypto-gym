---
schema_version: 1
op_name: apply_adam_w_v2
supported_dtypes: [bfloat16, float32]
p0_shapes: [[7168, 2048], [7168, 8192], [7168, 16384], [7168, 24576]]
tolerance: {atol: 0.0001, rtol: 0.0078125}
dynamic_axes: ['K']
dynamic_axes_ranges: {K: [2048, 24576]}
shape_constraints: {axis0: 7168, axis1: K (dynamic)}
default_params: {beta1: 0.9, beta2: 0.999, lr: 0.001, weight_decay: 0.01, eps: 1.0e-8, step: 1}
perf_target: 首跑精度成功后性能的 2 倍
---

## 算子需求规范

### 1. 基础信息
- **算子名称**: apply_adam_w_v2
- **算子分类**: custom (optimizer update, element-wise + scalar broadcast)

### 1.1 功能描述

实现 AdamW 优化器的单步更新。给定权重 `weight`、梯度 `grad`、一阶矩 `m`、二阶矩 `v`，按 AdamW 公式带 bias correction 原地更新 `weight`、`m`、`v`。weight 与 grad 支持 bf16/fp32，m 与 v 固定为 fp32；中间计算使用 fp32 累加器。0 轴固定 7168，1 轴 K 为动态轴（2048-24576），需通过 loop 切分。

### 1.2 算法参数

| 参数 | 类型 | 含义 | 典型值 |
|------|------|------|--------|
| beta1 (β1) | float | 一阶矩指数衰减率 | 0.9 |
| beta2 (β2) | float | 二阶矩指数衰减率 | 0.999 |
| lr (η) | float | 学习率 | 1e-3 |
| weight_decay (λ) | float | 权重衰减系数 | 0.01 |
| eps (ε) | float | 防除零平滑常数 | 1e-8 |
| step (t) | int | 当前步数（用于 bias correction） | ≥ 1 |

### 1.3 数学公式

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t = m_t / (1 - \beta_1^t) \\
\hat{v}_t = v_t / (1 - \beta_2^t) \\
w_t = w_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda w_{t-1} \right)
$$

### 2. 关键特性

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| in_place 更新 (w/m/v) | ✓ 需要 | ✓ 高 | 原地写回三个 tensor | P0 |
| bias_correction | ✓ 需要 | ✓ 高 | 使用 1 - β^t 修正一阶/二阶矩 | P0 |
| mixed_precision | ✓ 需要 | ✓ 高 | weight/grad 支持 bf16 与 fp32 | P0 |
| accumulator_dtype | ✓ 需要 | ✓ 高 | 中间矩估计与更新使用 fp32 累加 | P0 |
| loop tiling on K | ✓ 需要 | ✓ 高 | K 为动态轴，需要沿 K 切分 loop | P0 |
| weight_decay (λ) | ✓ 需要 | ✓ 高 | AdamW 解耦权重衰减 | P0 |
| dropout / 随机性 | ✗ 不需 | ✓ 高 | - | - |

### 3. 算法描述

```
Algorithm: apply_adam_w_v2 (single step AdamW update)
─────────────────────────────────────────────────────
输入:
  weight ∈ R^{7168×K} (bf16/fp32, in-place)
  grad   ∈ R^{7168×K} (bf16/fp32, in)
  m      ∈ R^{7168×K} (fp32, in-place)
  v      ∈ R^{7168×K} (fp32, in-place)
标量: β1, β2, η, λ, ε, t

输出: 更新后的 weight, m, v (原地)

1. 计算 bias-correction 标量（host 侧 / scalar）:
     bc1 = 1 - β1^t
     bc2 = 1 - β2^t
2. 沿 K 轴切分为 N_tiles 个 tile（tile 大小由 tiling 推导决定）
3. for tile_idx = 1 to N_tiles:
     3.1 加载 weight_tile, grad_tile, m_tile, v_tile（cast 到 fp32 累加器）
     3.2 m_new = β1 * m_tile + (1 - β1) * grad_tile
     3.3 v_new = β2 * v_tile + (1 - β2) * grad_tile^2
     3.4 m_hat = m_new / bc1
     3.5 v_hat = v_new / bc2
     3.6 update = m_hat / (sqrt(v_hat) + ε) + λ * weight_tile
     3.7 w_new = weight_tile - η * update
     3.8 写回 weight_tile (cast 回原 dtype), m_tile = m_new, v_tile = v_new
4. return
```

### 4. 数据流图

```
┌─────────────────────────┐    ┌─────────────────────────┐
│   weight[7168, K]       │    │    grad[7168, K]        │
│   bf16 / fp32 (in/out)  │    │    bf16 / fp32 (in)     │
└──────────┬──────────────┘    └────────────┬────────────┘
           │                                │
           │       ┌────────────────────────┘
           │       │
           │       ▼                        ┌──────────────────┐
           │  ┌──────────┐                  │   m[7168, K]     │
           │  │ cast→fp32│                  │   fp32 (in/out)  │
           │  └────┬─────┘                  └────────┬─────────┘
           │       │                                 │
           │       ├──────────────────┐              │
           │       ▼                  ▼              ▼
           │  ┌─────────────────────────────────────────┐
           │  │ m_t = β1*m + (1-β1)*g                   │
           │  │ v_t = β2*v + (1-β2)*g²                  │
           │  └────────────┬────────────────────────────┘
           │               │       ┌──────────────────┐
           │               │       │   v[7168, K]     │
           │               │       │   fp32 (in/out)  │
           │               │       └──────────────────┘
           │               ▼
           │  ┌─────────────────────────────────────────┐
           │  │ m_hat = m_t / (1 - β1^t)                │
           │  │ v_hat = v_t / (1 - β2^t)                │
           │  │ update = m_hat/(√v_hat+ε) + λ*w         │
           │  │ w_t = w - η * update                    │
           │  └────────────┬────────────────────────────┘
           ▼               ▼
    ┌──────────────┐  ┌──────────────┐
    │ weight (out) │  │ m (out),     │
    │              │  │ v (out)      │
    └──────────────┘  └──────────────┘

切分策略: loop 沿 K 轴（动态轴, 2048-24576）切分
```

### 5. 输入输出规格

**输入规格**:

| 变量 | Shape | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------|-------|--------|--------|------|
| weight | [7168, K] | bfloat16 / float32 | K | ✓ | 权重，原地更新 |
| grad   | [7168, K] | bfloat16 / float32 | K | ✓ | 当前步梯度 |
| m      | [7168, K] | float32            | K | ✓ | 一阶矩，原地更新 |
| v      | [7168, K] | float32            | K | ✓ | 二阶矩，原地更新 |
| beta1  | scalar    | float              | -  | ✓ | 一阶矩衰减率 |
| beta2  | scalar    | float              | -  | ✓ | 二阶矩衰减率 |
| lr     | scalar    | float              | -  | ✓ | 学习率 η |
| weight_decay | scalar | float          | -  | ✓ | 权重衰减 λ |
| eps    | scalar    | float              | -  | ✓ | 平滑常数 ε |
| step   | scalar    | int                | -  | ✓ | 当前步 t (≥1) |

**输出规格**（均为原地更新）:

| 变量 | Shape | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------|-------|--------|--------|------|
| weight | [7168, K] | bfloat16 / float32 | K | ✓ | 与输入 weight 同 dtype，更新后写回 |
| m      | [7168, K] | float32            | K | ✓ | 更新后的一阶矩 |
| v      | [7168, K] | float32            | K | ✓ | 更新后的二阶矩 |

### 6. 数据类型支持

| Dtype 组合 (weight=grad) | 支持 | atol | rtol | 备注 |
|--------------------------|------|------|------|------|
| float32  | ✓ | 0.0001 | 0.0078125 | 全 fp32 |
| bfloat16 | ✓ | 0.0001 | 0.0078125 | w/g bf16；m/v 仍 fp32；中间 fp32 |

### 7. 精度要求
- **atol**: 0.0001
- **rtol**: 0.0078125

### 8. 动态轴说明
- **动态轴**: ['K']
- **轴含义**: 0 轴固定为 7168；1 轴 K 表示参数列方向，需 loop 切分
- **取值范围**: K ∈ [2048, 24576]

### 9. 边界条件处理
- **零值**: 正常计算（v_hat 为 0 时由 ε 兜底）
- **极值**: 正常计算
- **NaN/Inf**: 正常计算（不做特殊处理）

### 10. 性能要求
- **性能目标**: 首跑精度成功后性能的 2 倍

### 11. 参考信息
- **参考实现**: PyTorch `torch.optim.AdamW`（带 bias correction）
- **论文**: Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019
- **类似算子**: apply_adam, apply_adam_w

### 12. 应用场景
- **目标模型**: 大语言模型训练（LLM optimizer step）
- **使用位置**: 优化器 step，每次反向传播后调用一次

**典型配置**:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| 功能_P0_fp32_min | 功能 | P0 | β1=0.9,β2=0.999,η=1e-3,λ=0.01,ε=1e-8,t=1 | weight/grad/m/v: [7168,2048] fp32/fp32/fp32/fp32 | 同输入 | 最小 K，全 fp32 |
| 功能_P0_bf16_min | 功能 | P0 | 同上 | weight/grad: [7168,2048] bf16；m/v: [7168,2048] fp32 | 同输入 | 最小 K，bf16 路径 |
| 性能_P0_bf16_mid | 性能 | P0 | 同上 t=100 | weight/grad: [7168,8192] bf16；m/v: [7168,8192] fp32 | 同输入 | 中等 K，典型 LLM 配置 |
| 性能_P0_bf16_max | 性能 | P0 | 同上 t=1000 | weight/grad: [7168,24576] bf16；m/v: [7168,24576] fp32 | 同输入 | 最大 K，loop 切分压力测试 |

---
*生成时间: 2026-04-29*
*确认状态: 已确认*
*置信度说明: ✓ 高（自身知识库/框架知识） / ⚠ 中（外部材料提取，需确认）*

## 2026-06-11 整改同步

- Shape 范围扩展为动态 `M/K` 的 2D tensor。
- 新增 `level5=[37,1500]`、`level6=[5,17]`、`level7=[33,2053]` 泛化验证。
- `[7168,K]` 既有大 M 网络规格继续走 large-M 策略，避免影响已有性能规格。
