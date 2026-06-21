# Sigmoid

对输入张量逐元素应用 Sigmoid 激活函数，将任意实数映射到 (0, 1) 区间。常用于神经网络的输出层（二分类概率）或门控机制。

---


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 数学公式

$$
\text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + e^{-x}}
$$

逐元素公式: `out[i, j] = 1 / (1 + exp(-x[i, j]))`

## 接口

```python
def Sigmoid_wrapper(x: torch.Tensor) -> torch.Tensor:
    ...
```

## 参数说明

| 参数 | dtype | shape | 说明 |
|------|-------|-------|------|
| `x` | float32 | [B, 16384] | 输入张量，任意实数。B 为动态轴，取值范围 [1, 65536] |
| 返回值 | float32 | [B, 16384] | 输出张量，值域 (0, 1) |

## 约束条件

- 输入必须是 2D float32 张量，shape 为 [B, 16384]
- B（batch 维度）为动态轴，取值范围 [1, 65536]
- 需要昇腾 NPU 环境和 CANN 工具链

## 支持规格

| 项目 | 支持范围 |
|------|---------|
| dtype | float32 |
| 输入维度 | 2D [B, 16384] |
| 硬件平台 | 昇腾 910 系列 |

## 使用示例

```python
import torch
from Sigmoid_impl import Sigmoid_wrapper

# 构造输入数据
x = torch.randn([16, 16384], dtype=torch.float32)

# 调用算子
y = Sigmoid_wrapper(x)

print(y.shape)  # torch.Size([16, 16384])
print(y.min(), y.max())  # 值域在 (0, 1) 范围内
```

## 目录结构

```
.
├── Sigmoid_impl.py      # 算子实现（含 Sigmoid_wrapper 接口）
├── Sigmoid_golden.py    # Golden 参考实现（精度基准）
├── test_Sigmoid.py      # 精度测试入口
├── test_cases.json      # 测试用例配置
└── README.md            # 本文件
```

## 运行方式

```bash
# 设置环境
export TILE_FWK_DEVICE_ID=0

# 运行精度测试（默认遍历所有用例）
python3 test_Sigmoid.py

# 运行单个用例
python3 test_Sigmoid.py p0_core

# 列出所有用例
python3 test_Sigmoid.py --list
```
