# KDA 算子需求规格（`kda_dynamic_bs`）

## 1. 基础信息

- 算子名称：`kda_dynamic_bs`
- 目标：实现 KDA（Kimi Delta Attention）风格的递推注意力核心前向算子
- 版本：`v1`
- 优先级：
  - `P0`：支持 `B/S` 双动态轴
  - `P0`：输出逐 token 的注意力结果 `output`
  - `P0`：输出每个 batch 的最终递推状态 `last_state`
  - `P1`：支持自定义 `initial_state`（本版本暂不支持，固定为零状态）

## 2. 数学定义

对每个 batch `b`、时间步 `t`，定义：

- `q_{b,t}, k_{b,t}, v_{b,t}, alpha_{b,t}, beta_{b,t} ∈ R^D`
- `S_{b,t} ∈ R^{D×D}`（递推状态矩阵）

递推与输出公式：

1. `Outer_{b,t} = k_{b,t}[:, None] * v_{b,t}[None, :]`
2. `S_{b,t} = S_{b,t-1} * alpha_{b,t}[:, None] + Outer_{b,t} * beta_{b,t}[:, None]`
3. `y_{b,t} = Σ_i ( q_{b,t}[i] * S_{b,t}[i, :] )`

初始状态：

- `S_{b,-1} = 0`

## 3. 输入输出规格

### 3.1 输入

| 名称 | Shape | dtype | 说明 |
|---|---|---|---|
| `query` | `[B, S, D]` | FP32 | 查询向量 |
| `key` | `[B, S, D]` | FP32 | 键向量 |
| `value` | `[B, S, D]` | FP32 | 值向量 |
| `alpha` | `[B, S, D]` | FP32 | 状态衰减门控 |
| `beta` | `[B, S, D]` | FP32 | 更新门控 |

### 3.2 输出

| 名称 | Shape | dtype | 说明 |
|---|---|---|---|
| `output` | `[B, S, D]` | FP32 | 每个时间步的输出 |
| `last_state` | `[B, D, D]` | FP32 | 每个 batch 最终状态 |

## 4. 动态轴与约束

- 动态轴：
  - `B`：batch 轴，`pypto.DYNAMIC`
  - `S`：sequence 轴，`pypto.DYNAMIC`
- 静态轴：
  - `D`：编译期常量（由 wrapper 构建 kernel 时确定）
- 约束：
  - 所有输入 shape 必须一致（除输出维度变化外）
  - dtype 仅支持 `torch.float32`
  - `D > 0`

## 5. 典型配置

| 配置名 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|---|---|---|---|---|---|---|
| `case_small` | 功能 | P0 | `B=1,S=16,D=64` | `[1,16,64]` | `[1,16,64]` + `[1,64,64]` | 基础正确性 |
| `case_mid` | 功能 | P0 | `B=2,S=31,D=64` | `[2,31,64]` | `[2,31,64]` + `[2,64,64]` | 非整齐序列长度 |
| `case_large` | 功能 | P1 | `B=4,S=127,D=64` | `[4,127,64]` | `[4,127,64]` + `[4,64,64]` | 大一些的动态规模 |

## 6. 精度要求

- 对比对象：`kda_golden.py` 的纯 PyTorch 实现
- 容忍度：`rtol=1e-3, atol=1e-3`

## 7. 非目标（v1 不覆盖）

- 不支持 BF16/FP16
- 不支持外部传入初始状态
- 不包含 backward 与性能调优目标
