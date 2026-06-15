# RMSNorm

RMS Normalization (Root Mean Square Normalization) 算子，对输入张量沿特征维度 (dim=1) 计算 RMS 并归一化。常用于 Transformer / LLM 架构中的归一化层。

---

## 数学公式

$$
\text{out}[b, c, h, w] = \frac{x[b, c, h, w]}{\sqrt{\frac{1}{C}\sum_{j=0}^{C-1} x[b, j, h, w]^2 + \varepsilon}}
$$

其中 C=64 (num_features)，ε=eps。

## 接口

```python
def RMSNorm_wrapper(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    ...
```

## 参数说明

| 参数 | dtype | shape | 说明 |
|------|-------|-------|------|
| `x` | float32 | [B, 64, 256, 256] | 输入张量, B 为动态 batch 维度 |
| `eps` | float | 标量 | 防止除零的小常数, 默认 1e-5 |
| 返回值 | float32 | [B, 64, 256, 256] | RMS 归一化后的输出, shape 与输入一致 |

## 约束条件

- 输入维度: 4D [B, 64, 256, 256]，其中 B 为动态轴 (范围 [1, 1024])
- 输入 dtype: float32
- 特征维度固定为 dim=1, size=64
- 空间维度固定为 H=W=256
- 硬件平台: 华为昇腾 AI 处理器 (Ascend 910 系列)

## 支持规格

| 项目 | 支持范围 |
|------|---------|
| dtype | float32 |
| 输入维度 | 4D [B, 64, 256, 256] |
| 动态轴 | B (dim=0) |
| 硬件平台 | Ascend 910 系列 |

## 使用示例

```python
import torch
from RMSNorm_impl import RMSNorm_wrapper

# 构造输入数据
x = torch.randn(16, 64, 256, 256, dtype=torch.float32)

# 调用算子
y = RMSNorm_wrapper(x, eps=1e-5)

print(y.shape)  # torch.Size([16, 64, 256, 256])
```

## 目录结构

```
.
├── RMSNorm_impl.py      # 算子实现（含 RMSNorm_wrapper 接口）
├── RMSNorm_golden.py    # Golden 参考实现（精度基准）
├── test_RMSNorm.py      # 精度测试入口
├── test_cases.json      # 测试用例配置
└── README.md            # 本文件
```

## 运行方式

```bash
# 设置环境
export TILE_FWK_DEVICE_ID=0

# 运行精度测试（默认遍历所有用例）
python3 test_RMSNorm.py

# 运行单个用例
python3 test_RMSNorm.py case_001

# 列出所有用例
python3 test_RMSNorm.py --list
```
