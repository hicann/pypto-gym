# GDR Backward (PyPTO Kernel)

基于 PyPTO 框架实现的 Gated Delta Rule 反向传播算子，运行于 Ascend NPU。

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 文件说明

| 文件 | 位置 | 说明 |
|------|------|------|
| `gdr_bwd_impl.py` | `src/.../qwen3_5/gdr_bwd/` | Kernel 实现 |
| `gdr_bwd_golden.py` | `tests/ops/qwen3_5/gdr_bwd/` | Golden 参考实现 |
| `test_gdr_bwd.py` | `tests/ops/qwen3_5/gdr_bwd/` | 测试用例 |

配套前向算子详见 [`gdr_fwd/`](../gdr_fwd/) 目录。

## 算法概述

Gated Delta Rule 是一种带门控的线性注意力变体。反向采用两遍扫描 (two-pass) 策略：PASS-1 正向串行扫描快照每个 chunk 的进入状态 `S_in`，PASS-2 反向串行扫描从 `A_inv` + `S_i` 重算 `w/v_new` 并写出全部梯度，避免缓存中间量。使用 **reverse-cumsum** (`tril^T @ d_gc`) 计算 gate 梯度，避免 `total - cumsum` 的相消误差。

### 数学公式

前向定义（本反向所对应的前向），单序列、单 value 头 `hv`，chunk `i`：

```
γ[t]   = Σ_{s<=t} g[i*BT+s]                       # chunk-local inclusive 前缀和
A[t,s] = beta[t] · exp(γ[t]-γ[s]) · (k[t]·k[s])    # 严格下三角 (t > s)
Tinv   = (I + A)^{-1}                              # 单位下三角，精确逆
u      = Tinv @ (beta ⊙ v)
w      = Tinv @ (beta ⊙ exp(γ) ⊙ k)
v_new  = u - w @ S_i                                # S_i = 进入 chunk i 时的状态 [K,V]
P[t,s] = (q[t]·k[s]) · exp(γ[t]-γ[s])               # 含对角 (t >= s)
o      = scale · ( exp(γ) ⊙ (q @ S_i) + P @ v_new )
S_i+1  = exp(γ[L-1]) · S_i + kᵀ @ ( v_new ⊙ exp(γ[L-1]-γ) )
```

### 语义约定

- **Q/K 侧 (H 头)**: q/k — 归一化后的（l2norm 在 kernel 内完成）
- **V 侧 (HV 头)**: v/g/beta — `HV == H`（不支持 GVA）
- **chunk_size (BT)**: 分块大小，必须为 2 的幂
- **A**: 前向保存的 `(I+L)^{-1}`，直接消费不重算（D1/D7）

### 循环结构

```
PASS 1: forward serial scan      — 快照 entering state S_in[i] → sin_ws
  seq_head_loop                  — 遍历 (序列 n, head h)
    chunk_loop                   — 序列按 BT 正向分块
      fwd_chunk_body             — 计算状态递推，快照 S_in

PASS 2: backward reverse scan    — carry dS，写出 5 个梯度 + dh0
  seq_head_loop                  — 遍历 (序列 n, head h)
    chunk_loop (reverse)         — 序列按 BT 反向分块
      bwd_chunk_body             — 重算 w/v_new，链式传播梯度
```

### 计算流程 (per chunk, reverse scan)

```
1. S_i     = sin_ws[chunk]                        # 从 PASS-1 缓存读取
2. v_new   = u - w @ S_i                           # 从 A_inv + S_i 重算
3. d_attn  = do @ v_new^T                          # o = qg@S_i + attn@v_new 的偏导
4. dS_out  = dS·e^{g_last} + qg^T@do - w^T@dv_new  # carry 递推
5. d_A_inv = du@v_beta^T + dw@k_beta_g^T            # 矩阵求逆规则
6. dA0     = -A_inv^T @ d_A_inv @ A_inv^T           # 严格下三角
7. dg      = tril^T @ d_gcum                        # reverse-cumsum (NS#5)
8. dq/dk   = 各路梯度汇总 + (可选) L2norm VJP
```

## Kernel 签名

```python
chunk_gated_delta_rule_backward_wrapper(
    q,                          # [B, T, H, K]      BF16  — 归一化后的 query
    k,                          # [B, T, H, K]      BF16  — 归一化后的 key
    v,                          # [B, T, HV, V]     BF16  — value
    g,                          # [B, T, HV]        FP32  — chunk-local cumsum gate
    beta,                       # [B, T, HV]        BF16  — 写入强度
    A,                          # [B, T, HV, BT]    FP32  — 前向保存的 (I+L)^{-1}
    scale,                      # float             — q 缩放系数
    initial_state,              # [N, HV, K, V]     FP32  — 初始状态
    do,                         # [B, T, HV, V]     BF16  — 上游 dL/do
    dht,                        # [N, HV, K, V]     FP32  — final_state 的上游梯度
    cu_seqlens=None,            # [N+1]             INT32 — varlen 累积长度
    chunk_size=128,             # int               — 分块大小
    g_is_natural_cumsum=False,  # bool              — g 是否为自然对数 cumsum
    q_rstd=None,                # [B, T, H]         FP32  — L2norm rstd(q)
    k_rstd=None,                # [B, T, H]         FP32  — L2norm rstd(k)
) -> (dq, dk, dv, db, dg, dh0, dA_log, ddt_bias)
```

**Shape 约束**:

| Symbol | Meaning | Supported |
|--------|---------|-----------|
| `D` | Per-head dimension (K=V) | 128 |
| `BT` | Chunk size | power of 2 (推荐 128) |
| `HV` | Value head count | == H (不支持 GVA) |
| `A` | 前向保存的 `(I+L)^{-1}` | 不能为 None |

## Dtype 转换流程

Kernel 内部严格控制 BF16/FP32 转换以平衡精度和性能：

| 阶段 | 操作 | Dtype |
|------|------|-------|
| 输入 | q/k/v/beta/do | BF16 |
| 输入 | g/A/initial_state/dht/q_rstd/k_rstd | FP32 |
| forward recompute | w/v_new 重算 | BF16 操作数 → FP32 累加 |
| 梯度 matmul | dq/dk/dv/db 各路 | BF16 操作数 → FP32 累加 |
| reverse-cumsum | tril^T @ d_gcum | FP32 全程（最 fragile 梯度） |
| 输出 | dq/dk/dv/db | BF16 |
| 输出 | dg/dh0 | FP32 |

## 测试用例

通过 `CASES` 字典集中定义，支持 `--case` 按名选择：

| 用例 | B | T | H | D | BT | varlen 段数 | 说明 |
|------|---|---|---|----|-----|------------|------|
| `t256` | 1 | 256 | 16 | 128 | 128 | 2 | 小规模冒烟 |
| `t1024` | 1 | 1024 | 16 | 128 | 128 | 7 | P0 规模 |
| `t4096` | 1 | 4096 | 16 | 128 | 128 | 15 | 大序列 |
| `varlen64_t32k_h8_bt64` | 1 | 32768 | 8 | 128 | 128 | 64 | 性能优化目标 |

### 精度校验

使用 `detailed_tensor_compare` 进行精度验证，按梯度分量分别设置门限：

| 梯度 | rtol | atol | 说明 |
|------|------|------|------|
| `dq` | 1e-1 | 1e-1 | dS carry 累积 + bf16 量化 |
| `dk` | 3e-2 | 3e-2 | dS carry 累积较轻 |
| `dv` | 3e-3 | 3e-3 | 无 dS carry 依赖 |
| `db` | 1e-1 | 1e-1 | beta 梯度链较长 |
| `dg` | 2.0 | 2.0 | gate 梯度跨 chunk 累积最严重 |

校验输出包括 5 个梯度：`dq`, `dk`, `dv`, `db`, `dg`（`dh0` 可选）。

Golden reference 严格模拟 kernel 内部的 bf16 量化路径（q/k/v/beta/do/A 均 bf16 量化后计算），确保对比基准与硬件行为一致。

## 运行方式

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=0

# 精度校验（默认 t1024）
python tests/ops/qwen3_5/gdr_bwd/test_gdr_bwd.py --case t1024 --mode precision

# 性能测量
python tests/ops/qwen3_5/gdr_bwd/test_gdr_bwd.py --case t4096 --mode perf --iters 10

# 精度 + 性能
python tests/ops/qwen3_5/gdr_bwd/test_gdr_bwd.py --case t1024 --mode both
```

**运行参数**:

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--device` | NPU 设备 ID | `TILE_FWK_DEVICE_ID` 环境变量 |
| `--case` | 用例名 | `t1024` |
| `--mode` | `precision` / `perf` / `both` | `precision` |
| `--iters` | 性能测量迭代次数 | 1 |
| `--warmup` | 性能测量 warmup 次数 | 2 |
| `--seed` | 随机种子 | 0 |

## 添加新用例

在 `test_gdr_bwd.py` 的 `CASES` 字典中追加一条：

```python
"my_case": dict(
    B=1, T=2048, H=16, D=128, CHUNK=128,
    VARLEN=[0, 512, 1024, 2048],
),
```

## 分块配置

**实现文件默认配置** (`gdr_bwd_impl.py`):
```python
_VT0, _VT1 = 128, 128    # Vector tile 形状
_CT = 128                 # Cube tile 边长
_SCHED = 0                # 调度模式
```

**说明**:
- `(128, 128)` 覆盖一个完整 chunk 行块，实测 -38% wall
- 行轴超过 64 会退化（`[BT,*]` tile 不能超过 BT=64 行）

## 与前向传播配合

反向 kernel 消费前向保存的残差：

```python
# 前向传播（请求反向残差）
o, final_state, residuals = chunk_gated_delta_rule_wrapper(
    q, k, v, g, beta, return_bwd_residuals=True, use_qk_l2norm_in_kernel=True)

# 反向传播（消费残差）
dq, dk, dv, db, dg, dh0, _, _ = chunk_gated_delta_rule_backward_wrapper(
    residuals["q_hat"], residuals["k_hat"], v, residuals["g_cum"],
    beta, residuals["A_inv"], scale, initial_state,
    do, dht, g_is_natural_cumsum=True,
    q_rstd=residuals["q_rstd"], k_rstd=residuals["k_rstd"])
```

**注意**: 前向的 `g` 是 log 域 per-token gate，反向的 `g` 是 chunk-local cumsum 后的 gate。`g_is_natural_cumsum=True` 时跳过 `* ln2` 换算。

## 依赖

- Python 3.x
- PyTorch + torch_npu
- PyPTO (`pypto` 包)
