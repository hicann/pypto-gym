# engram

## 概述

`engram` 是一个面向 transformer 模型的高性能 **Key-Value 记忆门控算子**，基于 `pypto_pro` DSL 在 Ascend NPU 上实现。该算子从 token embedding 出发计算 key/value 投影，通过 RMS 归一化的点积打分导出内容感知的门控信号，并将门控作用于 value 投影，为下游层产出门控输出。

算子采用 **Cube-Vector (CV) 并行执行**架构 —— Cube 子核执行矩阵乘法（key/value 投影），Vector 子核并发执行 RMS 归一化、打分、门控计算和广播乘法，两个子核间通过乒乓事件同步。

## 接口签名

```python
value_out, score_back, key_back, value_back, gate_back = engram_wrapper(
    hidden_states,       # [B, S, M, H]  bf16
    embeddings,          # [B, S, De]    bf16
    key_proj_weights,    # [M, De, H]    bf16
    value_proj_weights,  # [De, H]       bf16
    key_gamma,           # [M, H]        bf16
    query_gamma,         # [M, H]        bf16
)
```

### 输入

| 名称                  | 形状            | 类型   | 说明                                   |
|-----------------------|-----------------|--------|----------------------------------------|
| `hidden_states`       | `[B, S, M, H]`  | BF16   | Query 激活（每个头一个）               |
| `embeddings`          | `[B, S, De]`    | BF16   | Token 嵌入，作为 key/value 投影的输入  |
| `key_proj_weights`    | `[M, De, H]`    | BF16   | 按头的 key 投影权重矩阵                |
| `value_proj_weights`  | `[De, H]`       | BF16   | 共享的 value 投影权重矩阵              |
| `key_gamma`           | `[M, H]`        | BF16   | 按头的可学习 key 缩放系数 (γ_k)        |
| `query_gamma`         | `[M, H]`        | BF16   | 按头的可学习 query 缩放系数 (γ_q)      |

**维度说明**：`B` = 批次大小，`S` = 序列长度，`M` = 头数，`De` = 嵌入维度，`H` = 隐藏维度（1280 或 2560）。

### 输出

| 名称          | 形状           | 类型   | 说明                                  |
|---------------|----------------|--------|---------------------------------------|
| `value_out`   | `[B, S, M, H]` | BF16   | 门控 value 输出 (gate ⊙ value_proj)   |
| `score_back`  | `[B, S, M, 1]` | FP32   | 按头 score，用于反向传播              |
| `key_back`    | `[B, S, M, H]` | FP32   | Key 投影，用于反向传播                |
| `value_back`  | `[B, S, H]`    | FP32   | Value 投影，用于反向传播              |
| `gate_back`   | `[B, S, M, 1]` | FP32   | 门控值，用于反向传播                  |

## 算法

对每个头 `h` 和每个 token（第 `m` 行），依次执行：

1. **Key/Value 投影**（Cube 子核）：
   - `key_proj = emb · key_proj_weights[h]` — 按头 key 投影
   - `value_proj = emb · value_proj_weights` — 共享 value 投影

2. **RMS 归一化**（Vector 子核）：
   - `rms_k = 1 / sqrt(mean(key_proj²) + ε)`
   - `rms_q = 1 / sqrt(mean(query²) + ε)`

3. **Score 计算与门控**（Vector 子核）：
   - `score = sum(γ_k ⊙ rms_k ⊙ key_proj ⊙ γ_q ⊙ rms_q ⊙ query) / sqrt(H)`
   - `gate = sigmoid(sign(score) · sqrt(max(|score|, 1e-6)))`

   门控函数采用"有符号平方根 + sigmoid"的非线性形式，输出值域为 (0, 1)，可根据内容自适应地放行或抑制 value 信息。

4. **门控输出**（Vector 子核）：
   - `value_out = gate · value_proj`

## 架构设计

### CV 并行执行

每个 block 内的 2 个子核分工如下：

| 子核            | Section | 职责                                              |
|-----------------|---------|---------------------------------------------------|
| 子核 0 (AIC)    | Cube    | MatMul：emb × weight → key_back、value_back，写入 GM |
| 子核 1 (AIV)    | Vector  | 加载 key_back/value_back/hidden，四阶段 VF 流水线 → value_out |

### 跨核同步（乒乓协议）

Cube 和 Vector 之间按头同步，采用深度为 2 的乒乓机制防止死锁：

```
Cube（每个 head h）:                        Vector（每个 head h）:
  计算 key_back[h]                            wait K_READY[h%2]
  set K_READY[h%2]  ───────────────────→      加载并处理 head h
  wait ACK[h%2]     ←───────────────────      set ACK[h%2]
```

- `K_READY{0,1}`（cube→vec）：通知 key_back 已写入 GM
- `ACK{0,1}`（vec→cube）：背压信号，确保 Vector 消费完毕前 Cube 不会覆写同一槽位的事件 ID

### AIV Split

Vector section 内部将奇偶行 tile 分配到 2 个子核进一步并行处理。

### H 分片

当 `H = 2560` 时，分为 2 个 1280 的 chunk 处理。RMS 累加和 score 点积跨 chunk 累加，中间结果暂存 UB。UB 总占用约 206 KB，在 248 KB 上限以内。

## Tiling 常量

| 符号          | 值    | 说明                             |
|---------------|-------|----------------------------------|
| `TILE_M`      | 64    | Cube M 方向 tile（每次 MatMul 的行数） |
| `TILE_K`      | 128   | Cube K 方向 tile（内积维度）          |
| `TILE_N`      | 128   | Cube N 方向 tile（输出列数）          |
| `TILE_M_VEC`  | 8     | Vector M 方向 tile（每次 VF 处理行数） |
| `H_CHUNK`     | 1280  | H 维度分片大小                        |
| `CLAMP_VALUE` | 1e-6  | 开方前 score 绝对值下限               |
| `RMS_EPS`     | 1e-6  | RMS 归一化的 epsilon                  |

UB 总占用：约 206 KB（< 248 KB 上限）。

## 支持的配置

- 隐藏维度 `H`：1280 或 2560
- 头数 `M`：任意（受显存限制）
- 嵌入维度 `De`：任意
- 多核：最多 32 核，自适应 `num_cores = min(32, ceil(M / TILE_M))`

## 文件结构

```
engram/
├── engram.py    # 算子 kernel（pypto_pro DSL）+ host 封装
└── README.md    # 本文件
```

## 使用方法

```python
from pypto_gym.ops.pypto_pro.experimental.ops_transformer.engram import engram_wrapper

# 所有输入需在 NPU 设备上，除标注外均为 BF16
value_out, score_back, key_back, value_back, gate_back = engram_wrapper(
    hidden_states, embeddings,
    key_proj_weights, value_proj_weights,
    key_gamma, query_gamma,
)
```

首次调用时通过 `@pl.jit` 触发 JIT 编译。

## 精度说明

- **内部计算**：全程 FP32（Vector 运算、累加器、MatMul 累加器）
- **I/O**：输入和 `value_out` 使用 BF16；所有 `*_back` 输出使用 FP32（保留梯度精度用于反向传播）
- 门控计算中的 sigmoid 和 sqrt 使用 FP32 以保证数值稳定性
