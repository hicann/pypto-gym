# GDR Forward (PyPTO Kernel)

基于 PyPTO 框架实现的 Gated Delta Rule 前向传播算子，运行于 Ascend NPU。

## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 文件说明

| 文件 | 位置 | 说明 |
|------|------|------|
| `gdr_fwd_impl.py` | `src/.../qwen3_5/gdr_fwd/` | Kernel 实现 |
| `gdr_fwd_golden.py` | `tests/ops/qwen3_5/gdr_fwd/` | Golden 参考实现 |
| `test_gdr_fwd.py` | `tests/ops/qwen3_5/gdr_fwd/` | 测试用例 |

配套反向算子详见 [`gdr_bwd/`](../gdr_bwd/) 目录。

## 算法概述

Gated Delta Rule 是一种带门控的线性注意力变体，通过 chunk-parallel 策略在保持线性复杂度的同时实现高效计算。前向将序列按 `chunk_size` 分块，每块内并行计算 gate/decay 矩阵、层次化求逆 `(I+A)^{-1}`、WY 表示和输出，块间串行递推状态 `S`。使用 **层次化批量求逆**（8×16 叶子 + 3 级块合并）计算 `(I+A)^{-1}`。

### 数学公式

单序列、单 value 头 `hv`，chunk `i`（局部下标 t/s，`G = HV // H`，`h = hv // G`）：

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

- **Q/K 侧 (H 头)**: q/k — 头数 `H`，head dim `K`
- **V 侧 (HV 头)**: v/g/beta — 头数 `HV`（`HV % H == 0` 时触发 GVA）
- **chunk_size (BT)**: 分块大小，固定 128

### 循环结构

```
seq_head_loop            — 遍历 (序列 n, value 头 hv)，合并为一层
  chunk_loop             — 序列按 BT 分块
    l2norm (可选)        — q/k 的 L2 归一化
    gate_and_a           — γ 前缀和 + decay 矩阵 + A 矩阵
    inverse              — 8×16 层次化求逆 (I+A)^{-1}
    wy                   — WY 表示 u/w
    out_and_state        — 输出 o + 状态递推 S_{i+1}
```

### 计算流程 (per chunk)

```
1. γ[t]   = Σ_{s<=t} g[s]                              # chunk-local inclusive 前缀和
2. A[t,s] = beta[t] · exp(γ[t]-γ[s]) · (k[t]·k[s])     # 严格下三角 (t > s)
3. Tinv   = (I + A)^{-1}                               # 层次化求逆
4. u      = Tinv @ (beta ⊙ v)                          # WY 表示
5. w      = Tinv @ (beta ⊙ exp(γ) ⊙ k)
6. v_new  = u - w @ S_i                                # S_i = 进入 chunk i 时的状态
7. P[t,s] = (q[t]·k[s]) · exp(γ[t]-γ[s])               # 含对角 (t >= s)
8. o      = scale · ( exp(γ) ⊙ (q @ S_i) + P @ v_new )
9. S_i+1  = exp(γ[L-1]) · S_i + kᵀ @ ( v_new ⊙ exp(γ[L-1]-γ) )
```

## Kernel 签名

```python
chunk_gated_delta_rule_wrapper(
    q,                          # [B, T, H, K]     BF16  — query
    k,                          # [B, T, H, K]     BF16  — key
    v,                          # [B, T, HV, V]    BF16  — value
    g,                          # [B, T, HV]       FP32  — log 域遗忘门
    beta,                       # [B, T, HV]       BF16  — 写入强度
    scale=None,                 # float            — q 缩放，None → K**-0.5
    initial_state=None,         # [N, HV, K, V]    FP32  — 初始状态
    output_final_state=False,   # bool             — 是否返回 final_state
    use_qk_l2norm_in_kernel=False,  # bool         — kernel 内 L2 归一化
    cu_seqlens=None,            # [N+1]            INT64 — varlen 累积长度
    chunk_size=128,             # int              — 分块大小
) -> (o, final_state)           # o: [B,T,HV,V] BF16, final_state: [N,HV,K,V] FP32
```

**Shape 约束**:

| Symbol | Meaning | Supported |
|--------|---------|-----------|
| `D` | Per-head dimension (K=V) | 128 |
| `BT` | Chunk size | 128 |
| `HV` | Value head count | divisible by H (GVA) |

## Dtype 转换流程

Kernel 内部严格控制 BF16/FP32 转换以平衡精度和性能：

| 阶段 | 操作 | Dtype |
|------|------|-------|
| 输入 | q/k/v/beta | BF16 |
| 输入 | g/initial_state | FP32 |
| gate/decay | γ cumsum + exp decay | FP32 全程 |
| 求逆 | (I+A)^{-1} | FP32 → BF16 (舍入点 2) |
| WY matmul | Tinv @ v_beta / Tinv @ k_beta_g | BF16 操作数 → FP32 累加 |
| WY 落盘 | u / w | BF16 (舍入点 3/4) |
| 输出 matmul | q@S_i, P@v_new | BF16 操作数 → FP32 累加 |
| 输出 | o | BF16 |
| 状态 | final_state | FP32 |

## 测试用例

通过 `_CASES` 列表集中定义，支持 `--case` 按名选择：

| 用例 | group | B | T | H | HV | D | BT | 说明 |
|------|-------|---|---|---|----|----|-----|------|
| `bt128` | precision | 2 | 512 | 4 | 4 | 128 | 128 | chunk_size=128 |
| `depth256_T32K_H4` | long | 1 | 32768 | 4 | 4 | 128 | 128 | 链深 256（T=32K） |
| `varlen_T32K_H8` | precision | 1 | 32768 | 8 | 8 | 128 | 128 | 64 段 varlen + l2norm |

### 精度校验

使用 `detailed_tensor_compare` 进行双档精度验证：

```python
# 主判据：kernel(bf16) vs golden(emulate_bf16=True)
MAIN_ATOL_O, MAIN_RTOL_O = 2e-3, 2e-3    # o
MAIN_ATOL_S, MAIN_RTOL_S = 5e-3, 5e-3    # final_state
MAIN_L2_GATE = 3e-3

# 参考判据：kernel(bf16) vs golden(emulate_bf16=False, fp32 真值)
REF_ATOL, REF_RTOL = 3e-2, 3e-2
REF_L2_GATE = 1e-2
```

校验输出包括：
- **o (output)**: 前向输出
- **final_state**: 最终状态（`output_final_state=True` 时）

Golden reference 严格模拟 kernel 内部的 bf16 舍入点（`emulate_bf16=True`），确保对比基准与硬件行为一致。

## 运行方式

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=0

# 运行全部用例（P 精度 + L 长序列）
python tests/ops/qwen3_5/gdr_fwd/test_gdr_fwd.py

# 按名选择用例
python tests/ops/qwen3_5/gdr_fwd/test_gdr_fwd.py --case bt128

# 列出所有可用用例
python tests/ops/qwen3_5/gdr_fwd/test_gdr_fwd.py --list-cases

# 仅执行 kernel（不跑 golden，供 msprof profiling）
python tests/ops/qwen3_5/gdr_fwd/test_gdr_fwd.py --no-verify
```

## 添加新用例

在 `test_gdr_fwd.py` 的 `_CASES` 列表中追加一条 dict：

```python
dict(name="my_case", group="precision", kind="pair", desc="自定义用例",
     build=dict(b=2, t=1024, h=4, hv=4, d=128, seed=42),
     call=dict(chunk_size=128)),
```

`build` 支持的可选参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `b` | 批次数量 | — |
| `t` | 序列长度 | — |
| `h` | q/k 头数 | — |
| `hv` | v 头数 | = `h` |
| `d` | 每个头维度 | 128 |
| `seed` | 随机种子 | 0 |
| `g_range` | gate 采样范围 | (-0.10, -0.001) |
| `with_state` | 是否构造 initial_state | False |

## 分块配置

**实现文件默认配置** (`gdr_fwd_impl.py`):
```python
_VEC_TR = 64    # Vector tile 行数
_VEC_TC = 128   # Vector tile 列数
_KDA_MIN = 16   # 求逆叶子块边长
```

**说明**:
- Kernel 的 Cube tile 形状为 `[128, 128]`
- 求逆走 8×16 叶子 + 3 级块合并（BT=128 专用路径）

## 与反向传播配合

前向输出 `g_cum` / `A_inv` / `q_hat` / `k_hat` / `q_rstd` / `k_rstd` 可作为反向残差返回：

```python
# 前向传播（请求反向残差）
o, final_state, residuals = chunk_gated_delta_rule_wrapper(
    q, k, v, g, beta, return_bwd_residuals=True, use_qk_l2norm_in_kernel=True)

# 反向传播（消费残差）
# 详见 gdr_bwd/ 目录
```

## 依赖

- Python 3.x
- PyTorch + torch_npu
- PyPTO (`pypto` 包)
