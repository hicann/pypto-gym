# engram

## 概述

`engram` 是一个面向 transformer 模型的高性能 **Key-Value 记忆门控算子**，基于 `pypto_pro` DSL 在 Ascend NPU 上实现。该算子从 token embedding 出发计算 key/value 投影，通过 RMS 归一化的点积打分导出内容感知的门控信号，并将门控作用于 value 投影，为下游层产出门控输出。

算子采用 **Cube-Vector (CV) 并行执行**架构 —— Cube 子核执行矩阵乘法（key/value 投影），Vector 子核并发执行 RMS 归一化、打分、门控计算和广播乘法，两个子核间通过乒乓事件同步。

## 接口签名

```python
value_out, score_back, key_back, value_back, gate_back = engram_forward_wrapper(
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
├── engram_forward_impl.py    # 算子 kernel（pypto_pro DSL）+ host 封装
└── README.md    # 本文件
```

## 使用方法

```python
from pypto_gym.ops.pypto_pro.experimental.ops_transformer.engram import engram_forward_wrapper

# 所有输入需在 NPU 设备上，除标注外均为 BF16
value_out, score_back, key_back, value_back, gate_back = engram_forward_wrapper(
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

---

# engram_backward

## 概述

`engram_backward` 是 `engram`（正向 Key-Value 记忆门控算子）的**反向传播算子**，基于 `pypto_pro` DSL 在 Ascend NPU 上实现。它接收上游对门控输出 `value_out` 的梯度 `grad_output`，以及正向缓存的中间量（`scores` / `gates` / `keys` / `value`），沿正向 7 步计算链**逆向求导**，输出 6 个梯度：对 `hidden_states`、`embeddings`、`key_proj_weights`、`value_proj_weights`、`key_gamma`、`query_gamma` 的梯度。

算子延续正向的 **Cube-Vector (CV) 并行执行**架构，但数据流方向相反：**Vector 段先跑**（产出 `grad_value_ws` / `grad_key_ws` 等立方操作数工作区与 `grad_hidden` / `grad_γ`），通过两个分相事件（`GV_DONE` / `GK_DONE`）将工作区交接给 **Cube 段**（消耗工作区，产出 `grad_emb` / `grad_W_v` / `grad_W_k`）。Vector 段利用 32 核 × 2 AIV = 64 个 worker，Cube 段利用 32 个 AIC。

## 接口签名

```python
(grad_hidden_states, grad_embeddings, grad_key_proj_weights,
 grad_value_proj_weights, grad_key_gamma, grad_query_gamma) = engram_backward_wrapper(
    grad_output,         # [B, S, M, H]  bf16   上游梯度 ∂L/∂value_out
    hidden_states,       # [B, S, M, H]  bf16   正向输入（query 激活）
    embeddings,          # [B, S, De]    bf16   正向输入
    key_proj_weights,    # [M, De, H]    bf16   正向输入（host 内部预转置）
    value_proj_weights,  # [De, H]       bf16   正向输入（host 内部预转置）
    key_gamma,           # [M, H]        bf16   正向输入
    query_gamma,         # [M, H]        bf16   正向输入
    scores,              # [B, S, M, 1]  fp32   正向缓存 (Step4 输出)
    gates,               # [B, S, M, 1]  fp32   正向缓存 (Step5 输出)
    keys,                # [B, S, M, H]  bf16   正向缓存 (Step1 输出)
    value,               # [B, S, H]     bf16   正向缓存 (Step6 输出)
    clamp_value=1e-6,
    eps=1e-6,
)
```

### 输入

| 名称                  | 形状            | 类型   | 说明                                       |
|-----------------------|-----------------|--------|--------------------------------------------|
| `grad_output`         | `[B, S, M, H]`  | BF16   | 上游梯度 ∂L/∂value_out                     |
| `hidden_states`       | `[B, S, M, H]`  | BF16   | 正向 Query 激活                            |
| `embeddings`          | `[B, S, De]`    | BF16   | 正向 Token 嵌入                            |
| `key_proj_weights`    | `[M, De, H]`    | BF16   | 正向按头 key 投影权重（host 预转置为 wk_t）|
| `value_proj_weights`  | `[De, H]`       | BF16   | 正向共享 value 投影权重（host 预转置为 wv_t）|
| `key_gamma`           | `[M, H]`        | BF16   | 正向 key 缩放系数 γ_k                      |
| `query_gamma`         | `[M, H]`        | BF16   | 正向 query 缩放系数 γ_q                    |
| `scores`              | `[B, S, M, 1]`  | FP32   | 正向缓存：Step4 score（**必须 FP32**）     |
| `gates`               | `[B, S, M, 1]`  | FP32   | 正向缓存：Step5 gate（**必须 FP32**）      |
| `keys`                | `[B, S, M, H]`  | BF16   | 正向缓存：Step1 key 投影                   |
| `value`               | `[B, S, H]`     | BF16   | 正向缓存：Step6 value 投影                 |

**约束**：`scores` 和 `gates` 必须为 `torch.float32`（wrapper 会校验并报错）；其余激活/权重输入为 BF16。`H` 仅支持 1280 或 2560（TilingKey 特化）。

### 输出

| 名称                     | 形状            | 类型   | 说明                                       |
|--------------------------|-----------------|--------|--------------------------------------------|
| `grad_hidden_states`     | `[B, S, M, H]`  | BF16   | ∂L/∂hidden_states（Step3 逆向，对 query） |
| `grad_embeddings`        | `[B, S, De]`    | BF16   | ∂L/∂embeddings（Step6+Step1 逆向累加）    |
| `grad_key_proj_weights`  | `[M, De, H]`    | BF16   | ∂L/∂key_proj_weights（Step1 逆向）        |
| `grad_value_proj_weights`| `[De, H]`       | BF16   | ∂L/∂value_proj_weights（Step6 逆向）      |
| `grad_key_gamma`         | `[M, H]`        | BF16   | ∂L/∂key_gamma（Step2 逆向）               |
| `grad_query_gamma`       | `[M, H]`        | BF16   | ∂L/∂query_gamma（Step3 逆向）             |

**维度说明**：与正向一致 —— `B` 批次、`S` 序列、`M` 头数（动态 1–16）、`De` 嵌入维度、`H` 隐藏维度（1280 或 2560）。内部 `B·S` 合并为 `BS` 轴。

## 算法

沿正向 7 步链**逆向**求导（Step7 → Step1），逐头 `m ∈ [0, M)`：

1. **Step7 逆向**（Vector，门控广播乘法 `O = g · V`）：
   - `grad_gates[m] = Σ_h(grad_output · value)` → `[B,S,M,1]`
   - `grad_value    = Σ_m(grad_output · gates)` → `[B,S,H]`

2. **Step6 逆向**（Cube，`V = E @ W_v`）：
   - `grad_emb_from_v, grad_value_proj_weights = linear_bw(grad_value, E, W_v)`

3. **Step5 逆向**（Vector，signed_sqrt_gate）：
   - `grad_score = grad_gate · g(1−g) · mask / (2·√max(|s|,c) + 1e-12)`，`mask = (|s|>c)`

4. **Step4 逆向**（Vector，缩放点积）：
   - `grad_normed_key   = grad_score · (1/√H) · normed_query`
   - `grad_normed_query = grad_score · (1/√H) · normed_key`（对称）

5. **Step3 / Step2 逆向**（Vector，RMSNorm，3-pass）：
   - `(grad_hidden_m, grad_γ_q) = rms_norm_bw(grad_normed_query, hidden, γ_q)`
   - `(grad_key_m,    grad_γ_k) = rms_norm_bw(grad_normed_key,   key,    γ_k)`

6. **Step1 逆向**（Cube，`K = E @ W_k[m]`）：
   - `grad_emb_from_k, grad_W_k[m] = linear_bw(grad_key_m, E, W_k[m])`

7. **跨头累加**：`grad_embeddings = grad_emb_from_v + Σ_m grad_emb_from_k`

其中 `linear_bw` 对应 `grad_x = grad_output @ W^T`、`grad_W = x^T @ grad_output`；`rms_norm_bw` 按 `rms = √(mean(x²)+ε)`、`n = x/rms` 重算后求 `grad_x = (grad_n − n·mean(grad_n·n))/rms`、`grad_γ = Σ grad_x̂·n`。

## 架构设计

### 单 kernel 双 section（Vector 先 → Cube 后）

与正向「Cube 产 key/value、Vector 消费」相反，反向是 **Vector 先产工作区、Cube 后消费**：

| Section | 子核        | 职责                                                                 |
|---------|-------------|----------------------------------------------------------------------|
| Vector  | AIV (×64 worker) | 产 `grad_value_ws` / `grad_key_ws` / `grad_hidden` / `grad_γ`        |
| Cube    | AIC (×32)   | 消耗工作区，产 `grad_emb` / `grad_W_v` / `grad_W_k`                  |

### Vector 段：Pass A + Phase B（B-key / B-query）

Vector 段切分为两相，各自产出 Cube 所需的工作区，并各自发一个跨核事件，让 Cube 的对应消费者能尽早启动、与 Vector 的重相**重叠**：

| 相            | 产出                                   | 事件         | Cube 消费者（可提前启动）|
|---------------|----------------------------------------|--------------|--------------------------|
| Pass A        | `grad_value_ws`                        | `GV_DONE`    | Nest1-val + Nest2        |
| Phase B-key   | `grad_key_ws` + `grad_γ_k`             | `GK_DONE`    | Nest1-key + Nest3        |
| Phase B-query | `grad_hidden` + `grad_γ_q`             | （无，覆盖写）| —                        |

- **Pass A** 用专用大 tile（`[TILE_BS_VEC_A, H_CHUNK]`，行数为 B 相的 2 倍），`bs_start` 迭代减半。
- **Phase B-key 先于 B-query**：让 `GK_DONE` 尽早 fire，Nest1-key/Nest3 能与 B-query 的重度 per-head RMS 反向重叠。
- **grad_score 缓存**：B-key 算完 `gs` 后存入 `gscore_ws`，B-query 直接 load，省掉 M_H 次 Step7b+Step5 重算。

### 跨核同步（V → Cube 双事件）

```
Vector (Phase A 完成):                     Cube:
  sync_all(MIX)                              sync_all(MIX)
  set_cross_core(MTE3, GV_DONE) ──────→     wait_cross_core(MTE2, GV_DONE)   # Nest1-val + Nest2 可启动
  ... Phase B-key ...                        ... Nest2 与 Phase B 重叠 ...
  sync_all(MIX)                              sync_all(MIX)
  set_cross_core(MTE3, GK_DONE) ──────→     wait_cross_core(MTE2, GK_DONE)   # Nest1-key + Nest3 可启动
```

> **注意**：两个 `sync_all(MIX)` 是必需的（经板上验证，移除会死锁——`wait_cross_core(INTRA_BLOCK)` 依赖所有 AIV+AIC 到达对称汇合点）。`GK_DONE` 必须用全局 MIX 汇合，因为两个 AIV subblock 分担 `grad_key_ws` 生产，cube 必须等**全部** subblock 完成。

### Cube 段：3 个 Nest（K-reduction cover-write / atomic-add）

Cube 段用 BF16 `[128,128,128]` tile，L0C FP32 累加器。L1/L0 双缓冲让 K 链的 GM→L1 (MTE2) + L1→L0 (MTE1) 与上一片 matmul (M) 三流水重叠。三个 Nest 的数学对应：

| Nest      | 数学操作                                          | 输出                    | 写法                |
|-----------|---------------------------------------------------|-------------------------|---------------------|
| Nest1-val | `grad_value_ws @ wv_t`                            | `grad_emb_acc` (FP32)   | atomic-add（同槽）  |
| Nest2     | `emb_t @ grad_value_ws`                           | `grad_value_proj_weights`| cover-write (K=BS) |
| Nest1-key | `Σ_m grad_key_ws[m] @ wk_t[m]`                    | `grad_emb_acc` (FP32)   | atomic-add（同槽）  |
| Nest3     | `emb_t @ grad_key_ws[m]`                          | `grad_key_proj_weights` | cover-write (K=BS) |

- **Scheme A（grad_emb）**：value 贡献（Nest1-val）+ 每头 key 贡献（Nest1-key）都 atomic-add 进**同一个** `grad_emb_acc[BS, De]` FP32 槽（M-strided 划分保证每地址只被一个核写，atomic-add 退化为本地 FP32 累加，无跨核竞争）。host 端一次性 `grad_emb_acc → BF16` 得到 `grad_embeddings`。替代了旧的 `[M_H+1, BS, De]`（~71MB）cover-write workspace + Phase3 vector 求和。
- **grad_γ RMW**：`grad_γ_q` / `grad_γ_k` 按 per-subblock FP32 GM workspace（`[num_cores·2, M, H]`）做 tile RMW，host 端 `sum(dim=0)` 归约后 cast。

### TilingKey 双字段自适应（8 个编译变体）

- **HMode**（1 bit）：`0 → H=1280`、`1 → H=2560`，绑定 Vector sub-tile 行数（`[8,1280]` 与 `[4,2560]` 字节同构，UB 布局不变但 `bs_start` 迭代减半，有效带宽 ~2×）。
- **BSMode**（2 bit）：按 `BS` 阈值选 `V_TILE`（`0/1/2/3 → 128/64/32/16`，阈值 `BS ≥ 8192/4096/2048/<2048`）。大 BS 档位恰好 `ceil(BS/V_TILE) = 64` 填满全部 worker；`BS < 2048` 一律用最小 `V_TILE=16` 最大化占用（worker 数不足 64 由 `bsi < n_bs_v` 守卫跳过空槽，结果正确但 vector 利用率下降）。

`HMode`/`BSMode` 经 parser ConstInt fast path 只编译 taken 分支，使 tile 形状成为编译期常量。`[rows,64]` / `[1,H_CHUNK]` tile 的 UB 地址按 HMode 用 parser-foldable 算术选择特化（H=1280 行数翻倍，需 0x800 stride 避免 overlap）。

## Tiling 常量

| 符号              | 值         | 说明                                          |
|-------------------|------------|-----------------------------------------------|
| `CUBE_MN`         | 128        | Cube M/N 输出 tile（BF16 `[128,128]`=32KB）   |
| `CUBE_K`          | 128        | Cube K tile（BF16，L0A/L0B 双缓冲）           |
| `V_TILE_BS0..3`   | 128/64/32/16| Vector worker BS-row tile（BSMode 选择）     |
| `TILE_BS_VEC`     | 4 (H=2560) / 8 (H=1280) | Vector sub-row tile（HMode 选择）|
| `TILE_BS_VEC_A`   | 8 (H=2560) / 16 (H=1280)| Pass A 专用大 tile（2× B 相行数）|
| `H_CHUNK`         | 2560 / 1280| Vector 全 H 单 tile                           |
| `CLAMP_VALUE`     | 1e-6       | signed_sqrt_gate `|s|` 下限（对齐正向）        |
| `RMS_EPS`         | 1e-6       | RMSNorm epsilon（对齐正向）                    |
| `GATE_EPS`        | 1e-12      | signed_sqrt_gate 分母防零（对齐 golden）       |

UB 总占用：约 236.5 KB（H=1280 high-water，< 248 KB 上限）。Cube L1 占用 128 KB（< 512 KB）。

## 支持的配置

- 隐藏维度 `H`：**仅** 1280 或 2560（TilingKey 特化，wrapper 校验）
- 头数 `M`：动态 1–16
- 嵌入维度 `De`：任意
- `BS = B·S`：任意；小 BS（< 2048）时 Vector 利用率下降（worker 难以填满）
- 多核：固定 32 核（32 AIC × 2 AIV = 64 vector worker），靠 `bsi < n_bs` / `flat < total_*` 边界守卫跳过空槽

## 文件结构

```
engram/
├── engram_forward_impl.py     # 正向算子 kernel + host 封装
├── engram_backward_impl.py    # 反向算子 kernel（pypto_pro DSL）+ host 封装
├── engram_golden.py           # 前向/反向 golden 参考实现（torch，CPU FP64 真值）
└── README.md                  # 本文件
```

## 使用方法

```python
from pypto_gym.ops.pypto_pro.experimental.ops_transformer.engram import (
    engram_forward_wrapper,
    engram_backward_wrapper,
)

# 1) 正向：拿到 value_out 和反向所需的 cache（scores/gates/keys/value）
value_out, score_back, key_back, value_back, gate_back = engram_forward_wrapper(
    hidden_states, embeddings,
    key_proj_weights, value_proj_weights,
    key_gamma, query_gamma,
)
# score_back / gate_back / key_back / value_back 形状见正向输出表（需对齐到本算子期望的轴）

# 2) 反向：传入 grad_output 与正向缓存（scores/gates 必须 FP32）
grad_hs, grad_emb, grad_kpw, grad_vpw, grad_kg, grad_qg = engram_backward_wrapper(
    grad_output,
    hidden_states, embeddings,
    key_proj_weights, value_proj_weights,
    key_gamma, query_gamma,
    score_back.float(), gate_back.float(),   # scores / gates 必须 FP32
    key_back, value_back,                     # keys / value 保持 BF16
)
```

首次调用时通过 `@pl.jit` 触发 JIT 编译（按 `HMode × BSMode` 共 8 个特化变体）。

## 精度说明

- **内部计算精度**：Vector 段运算（RMS 反向 3-pass、grad_γ RMW、grad_score/grad_gate）全程 FP32；Cube 段为 BF16 操作数 × FP32 L0C 累加器（对齐 A5 BF16 matmul 4× 算力）。
- **关键中间量存储**：`grad_value_ws` / `grad_key_ws` 由 Vector 段以 FP32 算出，**存入 GM 时降为 BF16**（作为 Cube matmul 操作数）。这是性能取舍（避免 FP32 cube 的 4× 算力损失），但在极端抵消点会带来精度边界（见下）。
- **I/O**：输入激活/权重/缓存为 BF16（`scores`/`gates` 强制 FP32）；6 个梯度输出为 BF16（由内部 FP32 结果在写出/host 端一次性 cast 得到）。
- **精度参考**：`engram_golden.py` 为前向/反向 CPU FP64 数学真值；板上跑 `python engram_backward_impl.py`（或对应 test）做 max_diff/rel 自检，容差 `atol=1e-3, rtol=2e-2`。
- `1/√H` 在 FP32 vector 寄存器内构造（避免 BF16 标量路径丢精度）；`H` 为精确整数，`mean = sq / H` 用整数除法避免倒数误差。
