# Engram 算子（PyPTO Kernel）

基于 PyPTO 框架实现的 Engram 前向与反向算子，运行于 Ascend NPU。
Engram 是一种单头记忆检索注意力机制，支持 Key/Value 投影 + RMSNorm + sign-sqrt gate + 门控融合输出。

## 产品支持情况

- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

> **约束**：当前实现仅支持 `num_heads=1`（单头），通过 `combine_axis=True` 和 `batch_size*seq_len` 轴合并优化访存。

## 文件说明

| 文件 | 说明 |
|------|------|
| `engram_forward_impl.py` | 前向 Kernel 实现（线性投影、RMSNorm、sign-sqrt gate、Host 封装） |
| `engram_backward_impl.py` | 反向 Kernel 实现（RMSNorm/Linear 子模块、主 kernel、Host 封装） |

测试文件位于 `tests/ops/experimental/ops_transformer/engram/`：

| 文件 | 说明 |
|------|------|
| `engram_forward_golden.py` | 前向 PyTorch Golden 参考实现 |
| `test_engram_forward.py` | 前向精度测试 |
| `test_engram_backward.py` | 反向精度测试 + Golden 参考实现 |

---

## 前向算子

### 算法概述

对每个 token 计算 Key/Value 投影，对 Key 做 RMSNorm 后与 query（即 `hidden_states`）
点积得到 score，再用 `sigmoid(sign_sqrt(score))` 生成 gate，最终输出门控后的 value
以及若干中间结果（供反向使用）。

### 计算公式

```
key       = embeddings @ Wk.T + bk                          # [total_seq, hidden_dim]
value     = embeddings @ Wv.T + bv                          # [total_seq, hidden_dim]
normed_key = rms_norm(key, key_gamma, eps=1e-6)             # [total_seq, hidden_dim]
score     = sum(normed_key * hidden_states, dim=-1) / sqrt(hidden_dim)  # [total_seq]
gate      = sigmoid(sign(score) * sqrt(|score| + 1e-4))     # [total_seq]
value_out = gate * value                                     # [total_seq, hidden_dim]
```

其中 `sign_sqrt(x) = sign(x) * sqrt(|x| + eps)`，`rms_norm` 与 `pypto.rms_norm` 对齐。

### 前向输入参数

| 参数名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| hidden_states | (batch_size, seq_len, num_heads=1, hidden_dim) | FP32 | Query 输入 |
| embeddings | (batch_size, seq_len, hidden_dim) | FP32 | Key/Value 线性层共同输入 |
| key_proj_weights | (num_heads=1, hidden_dim, hidden_dim) | FP32 | Key 投影权重 |
| key_proj_bias | (num_heads=1, hidden_dim) | FP32 | Key 投影偏置 |
| value_proj_weights | (hidden_dim, hidden_dim) | FP32 | Value 投影权重 |
| value_proj_bias | (hidden_dim,) | FP32 | Value 投影偏置 |
| key_gamma | (num_heads=1, hidden_dim) | FP32 | Key RMSNorm gamma |

### 前向输出参数

| 参数名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| value_out | (batch_size, seq_len, num_heads, hidden_dim) | FP32 | 主输出：门控后的 value |
| score_back | (batch_size, seq_len, num_heads, 1) | FP32 | 中间结果 score（供反向使用） |
| key_back | (batch_size, seq_len, num_heads, hidden_dim) | FP32 | 中间结果 key（投影后、归一化前） |
| value_back | (batch_size, seq_len, hidden_dim) | FP32 | 中间结果 value（投影后、门控前） |
| gate_back | (batch_size, seq_len, num_heads, 1) | FP32 | 中间结果 gate |

### 前向 Kernel 概览

| 函数 | 功能 |
|------|------|
| `linear(tensor, weight, bias)` | `tensor @ weight.T + bias`，cube tile [128,128],[64,256],[128,128] |
| `sign_sqrt(tensor, eps)` | `sign(x) * sqrt(\|x\| + eps)`，用 ge/where/neg/sqrt 组合实现 |
| `engram_forward_kernel(...)` | 主 NPU kernel，batch/seq 双层循环，tile=256 |
| `pypto_engram_forward(...)` | Host 封装：reshape → kernel → reshape 回原始 shape |

### 前向 Pass 配置

```python
_PASS_OPTIONS = {
    "cube_l1_reuse_setting": {"DEFAULT": 1},
    "vec_nbuffer_setting": {"DEFAULT": 1},
    "auto_mix_partition": 1,
}
_RUNTIME_OPTIONS = {
    "stitch_function_max_num": 128,
    "max_workspace_kb": 1048907,
}
```

---

## 反向算子

### 算法概述

反向传播分为两条主路径：

1. **Value 路径**：`d_value → value_linear_backward → d_embeddings`
2. **Key/Query 路径**：
   - `d_gate → dz → d_score`（sigmoid + sign_sqrt 激活的链式法则）
   - `d_score → d_query_normed, d_key_normed`（点积反传）
   - `d_key_normed → rms_norm_backward → d_key_lineared`
   - `d_key_lineared → key_linear_backward → d_embeddings`（累加）

### 反向计算公式

$$d\_value = \text{grad\_out} \cdot gate$$

$$d\_gate = \text{grad\_out} \cdot \text{value\_lineared}$$

$$d\_score = \left(\sum_d d\_gate \cdot gate(1-gate)\right) \cdot \frac{0.5}{\sqrt{|score|+\varepsilon}}$$

$$d\_query\_normed = \frac{d\_score \cdot \text{key\_normed}}{\sqrt{\text{feat\_dim}}}, \quad
d\_key\_normed = \frac{d\_score \cdot \text{query\_normed}}{\sqrt{\text{feat\_dim}}}$$

### 反向输入参数

| 参数 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| grad_out | (batch_size, seq_len, num_heads, hidden_dim) | FP32 | 上游梯度 |
| hidden_states | (batch_size, seq_len, num_heads, hidden_dim) | FP32 | Query（前向输入） |
| embeddings | (batch_size, seq_len, hidden_dim) | FP32 | Key/Value 线性层输入 |
| key_w | (num_heads, hidden_dim, hidden_dim) | FP32 | Key 投影权重 |
| value_w | (hidden_dim, hidden_dim) | FP32 | Value 投影权重 |
| key_gamma | (num_heads, hidden_dim) | FP32 | Key RMSNorm gamma |
| query_gamma | (num_heads, hidden_dim) | FP32 | Query RMSNorm gamma |
| key_lineared | (batch_size, seq_len, num_heads, hidden_dim) | FP32 | Key 线性层输出（前向保存） |
| value_lineared | (batch_size, seq_len, hidden_dim) | FP32 | Value 线性层输出（前向保存） |
| gate | (batch_size, seq_len, num_heads, 1) | FP32 | Gate 激活值（前向保存） |
| score | (batch_size, seq_len, num_heads, 1) | FP32 | 点积 score（前向保存） |

### 反向输出参数

| 参数 | 形状 | 数据类型 | 初始化要求 | 说明 |
|------|------|----------|-----------|------|
| d_hidden | (batch_size, seq_len, num_heads, hidden_dim) | FP32 | zeros | Query 梯度 |
| d_embeddings | (batch_size, seq_len, hidden_dim) | FP32 | zeros | Embeddings 梯度（key + value 路径之和） |
| d_key_w | (num_heads, hidden_dim, hidden_dim) | FP32 | zeros | Key 权重梯度 |
| d_key_b | (num_heads, hidden_dim) | FP32 | zeros | Key bias 梯度 |
| d_value_w | (hidden_dim, hidden_dim) | FP32 | zeros | Value 权重梯度 |
| d_value_b | (hidden_dim,) | FP32 | zeros | Value bias 梯度 |
| d_key_gamma | (num_heads, hidden_dim) | FP32 | zeros | Key RMSNorm gamma 梯度 |

### 反向 Kernel 概览

| 函数 | 功能 |
|------|------|
| `rms_norm_backward_module(dy, x, gamma, eps)` | RMSNorm 反向，输出 dx [tile, hidden_dim] 和 d_gamma [hidden_dim] |
| `linear_backward_module(dy, x, weight)` | 线性层反向，输出 dx、d_weight、db |
| `engram_backward_kernel(...)` | 主 NPU kernel，flatten 为 [total_seq, hidden_dim] 后处理 |
| `engram_backward_pto(...)` | Host 封装：reshape → kernel → reshape 回原始 shape |

### 反向 Pass 配置

```python
pass_options = {
    "cube_l1_reuse_setting": {-1: 16},
    "vec_nbuffer_setting": {-2: 1, -1: 4},
}
runtime_options = {
    "stitch_function_max_num": 12,
    "max_workspace_kb": 5469824,
}
```

---

## 维度说明

| 符号 | 含义 |
|------|------|
| batch_size | Batch 数量（动态） |
| seq_len | 序列长度（动态） |
| num_heads | 注意力头数，当前固定为 1 |
| hidden_dim | 隐藏维度（静态） |
| total_seq | `batch_size * seq_len`，kernel 内 flatten 后的序列总长度 |

## 分块配置

| 操作 | tile shape |
|------|-----------|
| 主循环 tile（seq_len 轴） | 256（`unroll_list=[256]`） |
| Vec tile（2D） | [64, 256] 或 [64, 512] |
| Vec tile（reduce） | [16, 1024] 或 [1024] |
| Cube tile | [128, 128], [64, 256], [128, 128] |

---

## 运行测试

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=0

# 前向测试
python tests/ops/experimental/ops_transformer/engram/test_engram_forward.py

# 反向测试
python tests/ops/experimental/ops_transformer/engram/test_engram_backward.py

# 使用 pytest
pytest tests/ops/experimental/ops_transformer/engram/ -v
```

### 测试用例

**前向：**

| 用例 | batch_size | seq_len | hidden_dim | num_heads | 说明 |
|------|-----------|---------|----------|-----------|------|
| 默认 | 1 | 1024 | 1024 | 1 | 256 对齐，单头 |

**反向：**

| 用例 | batch_size | seq_len | hidden_dim | num_heads | 说明 |
|------|-----------|---------|----------|-----------|------|
| 默认 | 1 | 8192 | 1024 | 1 | 标准序列长度 |
| pytest param 1 | 1 | 8192 | 1024 | 1 | 标准 |
| pytest param 2 | 1 | 1024 | 512 | 1 | 小规模 |
| pytest param 3 | 1 | 2048 | 768 | 1 | 中规模 |

### 精度校验

**前向**（与 `engram_forward_golden.py` 对比）：

```python
# value_out / score_back / key_back / value_back
atol = 5e-2, rtol = 1e-2
# gate_back
atol = 1e-3, rtol = 1e-3
```

**反向**（与 PyTorch golden 对比，校验全部 7 个输出）：

```python
atol = 1e-3, rtol = 1e-3
```

---

## 依赖

- Python 3.x
- PyTorch + torch_npu
- PyPTO (`pypto` 包)
- NumPy
