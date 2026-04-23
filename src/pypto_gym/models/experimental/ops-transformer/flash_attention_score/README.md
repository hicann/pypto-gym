# Flash Attention Score

## 概述

Flash Attention Score 是一个高效的注意力机制实现，采用 **Online Softmax** 算法实现分块计算，支持完整的训练场景。

### 功能特性

-  **Online Softmax 分块计算** - 避免存储完整 attention matrix
-  **支持动态轴** - Batch size、Query seq len、KV seq len 均为动态维度
-  **反向传播支持** - 输出 softmax_max 和 softmax_sum 中间结果
-  **Dropout 支持** - 训练场景正则化
-  **位置编码支持** - 支持 4 种 pse_type 模式
-  **注意力掩码** - 支持因果掩码和自定义掩码
-  **Scale 可配置** - 支持自定义 scale_value 参数（Stage 3）
-  **多数据类型** - 支持 BF16、FP32 数据类型（Stage 3）
-  **高精度** - 使用 FP32 进行中间计算，满足精度标准

---

## Kernel 概览

提供两组 API：

### 1. 统一接口（推荐）

通过 `dtype` 参数自动选择对应的底层 kernel：

```python
from flash_attention_score_impl import (
    flash_attention_score_with_mask,
    flash_attention_score_with_pse_and_dropout
)

# BF16 (默认)
flash_attention_score_with_mask(..., dtype="bf16")

# FP32
flash_attention_score_with_mask(..., dtype="fp32")
```

### 2. 底层 Kernel（高级用户）

直接调用特定数据类型的 kernel：

| Kernel | 功能 | 数据类型 | 适用场景 |
|--------|------|---------|---------|
| `flash_attention_score_kernel_with_mask_origin` | 基础 mask (无中间输出) | BF16 | 简化推理场景 |
| `flash_attention_score_kernel_with_mask` | 基础 mask 支持 | BF16 | 推理场景 |
| `flash_attention_score_kernel_with_mask_fp32` | 基础 mask 支持 | FP32 | 推理场景 |
| `flash_attention_score_kernel_with_pse_and_dropout` | Mask + PSE + Dropout | BF16 | 训练场景 |
| `flash_attention_score_kernel_with_pse_and_dropout_fp32` | Mask + PSE + Dropout | FP32 | 训练场景 |

**数据类型策略：**
- BF16: 输入 -> FP32 中间计算 -> 输出（带 cast）
- FP32: 输入 -> FP32 中间计算 -> 输出（无 cast）

---

## 数学公式

### Kernel 1: with_mask

$$
\text{attention\_out} = \text{Softmax}\left(\frac{Q @ K^T}{\sqrt{d}} \cdot \text{mask}\right) @ V
$$

### Kernel 2: with_pse_and_dropout

**pseType = 1:**
$$
\text{attention\_out} = \text{Dropout}\left(\text{Softmax}\left(\text{Mask}\left(\text{scale} \cdot (\text{pse} + Q @ K^T), \text{atten\_mask}\right)\right), \text{keep\_prob}\right) @ V
$$

**pseType ≠ 1 (0, 2, 3):**
$$
\text{attention\_out} = \text{Dropout}\left(\text{Softmax}\left(\text{Mask}\left(\text{scale} \cdot Q @ K^T + \text{pse}, \text{atten\_mask}\right)\right), \text{keep\_prob}\right) @ V
$$

---

## 参数规格

### Kernel 0: flash_attention_score_kernel_with_mask_origin

基础版本，只输出 attention_out，不输出中间结果（softmax_max/softmax_sum），不支持反向传播。

| 参数 | 类型 | Shape | 数据类型 | 说明 |
|------|------|-------|----------|------|
| query | 输入 | [B, N, Sq, D] | BF16 | Query 张量 |
| key | 输入 | [B, N, Skv, D] | BF16 | Key 张量 |
| value | 输入 | [B, N, Skv, D] | BF16 | Value 张量 |
| atten_mask | 输入 | [Sq, Skv] | FP32 | 注意力掩码，1=不参与，0=参与 |
| output | 输出 | [B, N, Sq, D] | BF16 | Attention 输出 |

**特点：**
- 固定 scale = 1/√D，不可配置
- 仅支持 BF16 数据类型
- 无中间输出，不适用于反向传播

### Kernel 1: flash_attention_score_kernel_with_mask

| 参数 | 类型 | Shape | 数据类型 | 说明 |
|------|------|-------|----------|------|
| query | 输入 | [B, N, Sq, D] | BF16 | Query 张量 |
| key | 输入 | [B, N, Skv, D] | BF16 | Key 张量 |
| value | 输入 | [B, N, Skv, D] | BF16 | Value 张量 |
| atten_mask | 输入 | [Sq, Skv] | FP32 | 注意力掩码，1=不参与，0=参与 |
| output | 输出 | [B, N, Sq, D] | BF16 | Attention 输出 |
| softmax_max | 输出 | [B, N, Sq, 1] | FP32 | Softmax 最大值，用于反向 |
| softmax_sum | 输出 | [B, N, Sq, 1] | FP32 | Softmax 指数和，用于反向 |
| scale_value | 属性 | - | float | 缩放系数，默认 1/√D (Stage 3) |

### Kernel 2: flash_attention_score_kernel_with_pse_and_dropout

| 参数 | 类型 | Shape | 数据类型 | 说明 |
|------|------|-------|----------|------|
| query | 输入 | [B, N, Sq, D] | BF16 | Query 张量 |
| key | 输入 | [B, N, Skv, D] | BF16 | Key 张量 |
| value | 输入 | [B, N, Skv, D] | BF16 | Value 张量 |
| atten_mask | 输入 | [Sq, Skv] | FP32 | 注意力掩码，1=不参与，0=参与 |
| pse | 输入 | [B, N, Sq, Skv] | BF16 | 位置编码 |
| drop_mask | 输入 | [Sq, Skv] | FP32 | Dropout 掩码，1=保留，0=丢弃 |
| output | 输出 | [B, N, Sq, D] | BF16 | Attention 输出 |
| softmax_max | 输出 | [B, N, Sq, 1] | FP32 | Softmax 最大值，用于反向 |
| softmax_sum | 输出 | [B, N, Sq, 1] | FP32 | Softmax 指数和，用于反向 |
| pse_type | 属性 | - | int | PSE 模式 (0, 1, 2, 3) |
| keep_prob | 属性 | - | float | Dropout 保留概率，默认 1.0 |
| scale_value | 属性 | - | float | 缩放系数，默认 1/√D (Stage 3) |

### 固定参数

| 参数 | 值 | 说明 |
|------|-----|------|
| N (Num heads) | 8 | 注意力头数 |
| D (Head dim) | 64 | 每个头的维度 |
| Block size | 64 | 分块大小 |

### 动态轴

- **B (Batch size)**: 动态维度，运行时可变
- **Sq (Query sequence length)**: 动态维度，运行时可变
- **Skv (KV sequence length)**: 动态维度，运行时可变

---

## pseType 模式说明

| pseType | 计算公式 | 约束 |
|---------|---------|------|
| 1 | `scale * (pse + Q @ K^T)` | 无 |
| 0, 2, 3 | `scale * Q @ K^T + pse` | pseType=2 或 3 时：Sq == Skv |

**注意**: pseType 0, 2, 3 计算逻辑相同，区别仅在约束条件。

---

## 反向传播支持

两个 kernel 均输出中间结果用于反向传播：

```python
# 前向传播
flash_attention_score_kernel_with_mask(
    query, key, value, atten_mask,
    output, softmax_max, softmax_sum
)

# 反向传播使用中间结果
# grad_q, grad_k, grad_v = flash_attention_backward(
#     grad_output, query, key, value, output, softmax_max, softmax_sum
# )
```

---

## 使用示例

### 统一接口（推荐）

```python
import torch
import math
from flash_attention_score_impl import flash_attention_score_with_mask

# 准备输入（BF16）
query = torch.randn(2, 8, 64, 64, dtype=torch.bfloat16, device='npu:0')
key = torch.randn(2, 8, 128, 64, dtype=torch.bfloat16, device='npu:0')
value = torch.randn(2, 8, 128, 64, dtype=torch.bfloat16, device='npu:0')
atten_mask = torch.zeros(64, 128, dtype=torch.float32, device='npu:0')

# 准备输出
output = torch.empty(2, 8, 64, 64, dtype=torch.bfloat16, device='npu:0')
softmax_max = torch.empty(2, 8, 64, 1, dtype=torch.float32, device='npu:0')
softmax_sum = torch.empty(2, 8, 64, 1, dtype=torch.float32, device='npu:0')

# 调用统一接口（BF16）
flash_attention_score_with_mask(
    query, key, value, atten_mask,
    output, softmax_max, softmax_sum,
    scale_value=1.0 / math.sqrt(64),
    dtype="bf16"  # 可选，默认 bf16
)

# 或者使用 FP32
query_fp32 = torch.randn(2, 8, 64, 64, dtype=torch.float32, device='npu:0')
# ... 其他 FP32 张量 ...

flash_attention_score_with_mask(
    query_fp32, key_fp32, value_fp32, atten_mask,
    output_fp32, softmax_max, softmax_sum,
    scale_value=1.0 / math.sqrt(64),
    dtype="fp32"  # 显式指定 FP32
)
```

### 底层 Kernel（高级用户）

#### BF16 示例

```python
import torch
import math
from flash_attention_score_impl import flash_attention_score_kernel_with_mask

# 输入
query = torch.randn(2, 8, 64, 64, dtype=torch.bfloat16, device='npu:0')
key = torch.randn(2, 8, 128, 64, dtype=torch.bfloat16, device='npu:0')
value = torch.randn(2, 8, 128, 64, dtype=torch.bfloat16, device='npu:0')
atten_mask = torch.zeros(64, 128, dtype=torch.float32, device='npu:0')

# 输出
output = torch.empty(2, 8, 64, 64, dtype=torch.bfloat16, device='npu:0')
softmax_max = torch.empty(2, 8, 64, 1, dtype=torch.float32, device='npu:0')
softmax_sum = torch.empty(2, 8, 64, 1, dtype=torch.float32, device='npu:0')

# 调用 kernel
flash_attention_score_kernel_with_mask(
    query, key, value, atten_mask,
    output, softmax_max, softmax_sum,
    scale_value=1.0 / math.sqrt(64)
)
```

#### 训练场景 (with_pse_and_dropout) - 统一接口

```python
import torch
import math
from flash_attention_score_impl import flash_attention_score_with_pse_and_dropout

# 输入
query = torch.randn(2, 8, 64, 64, dtype=torch.bfloat16, device='npu:0')
key = torch.randn(2, 8, 128, 64, dtype=torch.bfloat16, device='npu:0')
value = torch.randn(2, 8, 128, 64, dtype=torch.bfloat16, device='npu:0')
atten_mask = torch.zeros(64, 128, dtype=torch.float32, device='npu:0')

# PSE 和 Dropout
pse = torch.randn(2, 8, 64, 128, dtype=torch.bfloat16, device='npu:0')
drop_mask = torch.ones(64, 128, dtype=torch.float32, device='npu:0')

# 输出
output = torch.empty(2, 8, 64, 64, dtype=torch.bfloat16, device='npu:0')
softmax_max = torch.empty(2, 8, 64, 1, dtype=torch.float32, device='npu:0')
softmax_sum = torch.empty(2, 8, 64, 1, dtype=torch.float32, device='npu:0')

# 调用统一接口
flash_attention_score_with_pse_and_dropout(
    query, key, value, atten_mask, pse, drop_mask,
    output, softmax_max, softmax_sum,
    pse_type=0, keep_prob=1.0,
    scale_value=1.0 / math.sqrt(64),
    dtype="bf16"  # 或 "fp32"
)
```

---

## 运行测试

```bash
# 设置环境变量
export TILE_FWK_DEVICE_ID=0

# 运行全部测试（所有数据类型）
python flash_attention_score.py

# 测试指定数据类型
python flash_attention_score.py --dtype bf16
python flash_attention_score.py --dtype fp32

# 仅测试 with_mask_origin kernel
python flash_attention_score.py --kernel mask_origin

# 仅测试 with_mask kernel
python flash_attention_score.py --kernel mask

# 仅测试 with_pse_and_dropout kernel
python flash_attention_score.py --kernel pse_dropout

# 测试自定义 scale_value
python flash_attention_score.py --scale_value 0.2

# 使用 sim 模式
python flash_attention_score.py --run_mode sim
```

**测试参数说明：**
- `--dtype`: 数据类型（all/bf16/fp32），默认 all
- `--kernel`: Kernel 类型（all/mask_origin/mask/pse_dropout），默认 all
- `--scale_value`: 自定义 scale 值，默认 1/√D
- `--run_mode`: 运行模式（npu/sim），默认 npu

---

## 测试结果

### BF16 精度验证

| Kernel | pseType | 最大差异 | 平均差异 | 状态 |
|--------|---------|---------|---------|------|
| with_mask_origin | - | 0.000977 | 0.000000 | 通过 |
| with_mask | - | 0.001953 | 0.000000 | 通过 |
| with_pse_and_dropout | 0 | 0.001953 | 0.000000 | 通过 |
| with_pse_and_dropout | 1 | 0.000488 | 0.000000 | 通过 |

**精度标准**: `rtol=0.0078125, atol=0.0001` (BF16)

### FP32 精度验证

| Kernel | pseType | 最大差异 | 平均差异 | 状态 |
|--------|---------|---------|---------|------|
| with_mask | - | 0.003866 | 0.000219 | 通过 |
| with_pse_and_dropout | 0 | 0.007114 | 0.000390 | 通过 |
| with_pse_and_dropout | 1 | 0.003890 | 0.000301 | 通过 |

**精度标准**: `rtol=0.01, atol=0.003` (FP32)

### Scale 可配置验证

| Scale 值 | Kernel | 状态 |
|---------|--------|------|
| 默认 (1/√D) | with_mask | 通过 |
| 0.2 (自定义) | with_mask | 通过 |
| 0.2 (自定义) | with_pse_and_dropout | 通过 |

---

## Dropout 实现说明

采用 **Inverted Dropout** 策略，在训练时对保留的激活值进行缩放：

$$
p_{\text{scaled}} = \frac{p \cdot \text{drop\_mask}}{\text{keep\_prob}}
$$

其中：
- `drop_mask`: 掩码张量，1=保留，0=丢弃
- `keep_prob`: 保留概率（例如 0.8 表示保留 80%）
- 缩放因子 `1/keep_prob` 保持期望值不变

**注意**：Golden Reference 实现必须与 Kernel 保持一致的 dropout 应用时机（在 softmax 计算过程中应用，而非事后跳过）。

---

## 文件结构

```
flash_attention_score/
├── flash_attention_score_impl.py     # Kernel 实现
│   ├── flash_attention_score_with_mask (统一接口) 
│   ├── flash_attention_score_with_pse_and_dropout (统一接口) 
│   ├── flash_attention_score_kernel_with_mask_origin (BF16, 无中间输出)
│   ├── flash_attention_score_kernel_with_mask (BF16)
│   ├── flash_attention_score_kernel_with_mask_fp32 (FP32)
│   ├── flash_attention_score_kernel_with_pse_and_dropout (BF16)
│   └── flash_attention_score_kernel_with_pse_and_dropout_fp32 (FP32)
├── flash_attention_score.py          # 测试套件 + Golden Reference
├── README.md                         # 本文档
└── UPGRADE_PLAN.md                   # 功能规划
```

**推荐使用统一接口，通过 dtype 参数选择数据类型。**

---

## 已知限制

| 限制 | 说明 | 规划 |
|------|------|------|
| 固定维度 | Num heads=8, Head dim=64 | 待优化 |
| drop_mask dtype | 使用 FP32 而非 UINT8 | 已知问题 |
| FP8 支持 | 不支持 | Stage 4 |

### drop_mask 使用 FP32 说明

AscendC 文档要求 `drop_mask` 使用 UINT8 类型，但 PyPTO 实现中使用 FP32。原因是 UINT8 类型在当前 PyPTO 版本中存在兼容性问题。FP32 实现在功能上完全等价，仅内存占用略高。

---

## 开发进度

| Stage | 功能 | 状态 |
|-------|------|------|
| Stage 1 | 反向传播中间结果 (softmax_max, softmax_sum) | 已完成 |
| Stage 2 | Dropout + PSE 支持 | 已完成 |
| Stage 3 | Scale 可配置 + BF16/FP32 数据类型 | 已完成 |
| Stage 4 | FP8 量化支持 | 计划中 |

**Stage 3 详细说明：**
- Scale 可配置：支持自定义 scale_value 参数
- BF16 数据类型：完全支持，精度验证通过
- FP32 数据类型：完全支持，精度验证通过

---

## 参考资料
- [Online Softmax 论文](https://arxiv.org/abs/2006.04768)
- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)