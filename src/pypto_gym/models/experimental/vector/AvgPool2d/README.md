# avg_pool2d

## 功能说明

`avg_pool2d` 算子实现了 2D 平均池化操作。该算子对输入张量应用滑动窗口进行平均池化，支持 SAME 和 VALID 两种填充模式。

该算子对应 TensorFlow 的 `tf.nn.avg_pool2d` 操作，常用于卷积神经网络中的下采样操作，减少特征图的空间维度。

## 数学公式

$$
\text{output}[n, c, oh, ow] = \frac{1}{k_h \times k_w} \sum_{i=0}^{k_h-1} \sum_{j=0}^{k_w-1} \text{input}[n, c, oh \times s_h + i, ow \times s_w + j]
$$

其中：
- $\text{input}$：输入张量 (batch_size, channels, in_h, in_w)
- $\text{output}$：输出张量 (batch_size, channels, out_h, out_w)
- $k_h, k_w$：池化窗口的高度和宽度
- $s_h. s_w$：步长的高度和宽度
- $oh, ow$：输出特征图的位置索引

对于 SAME padding：
$$
out_h = \lceil \frac{in_h}{s_h} \rceil, \quad out_w = \lceil \frac{in_w}{s_w} \rceil
$$

对于 VALID padding:
$$
out_h = \lceil \frac{in_h - k_h + 1}{s_h} \rceil. \quad out_w = \lceil \frac{in_w - k_w + 1}{s_w} \rceil
$$

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
|--------|-----------|------|----------|----------|-------------|
| input_tensor | 输入 | 输入特征图 | float32 | ND | [batch_size, channels, in_h, in_w] |
| output_result | 输出 | 输出特征图 | float32 | ND | [batch_size, channels, out_h, out_w] |
| kernel_size | 属性 | 池化窗口大小 | tuple | - | (k_h, k_w) |
| stride | 属性 | 步长大小 | tuple | - | (s_h, s_w)，默认为 kernel_size |
| padding_mode | 属性 | 填充模式 | string | - | 'SAME' 或 'VALID' |

## 实现原理

该算子通过以下步骤实现：

1. 使用 `PoolParams` dataclass 封装所有池化参数（batch_size, channels, kernel_size, stride, padding 等）
2. 使用 `pypto.loop_unroll` 对 batch*channel 维度进行分块处理，支持 [8, 4, 2, 1] 等多种分块大小
3. 对每个输出位置 (oh, ow)：
   - 计算输入特征的滑动窗口范围，考虑 padding 偏移
   - 对窗口内的数据进行高度方向的求和（reduce）
   - 对宽度方向进行求和并计算平均值
   - 使用 `pypto.assemble` 将结果写入输出位置

## 核心优化

1. **参数封装优化**：使用 `@dataclass(frozen=True)` 封装池化参数，提升代码可读性和类型安全性
2. **向量化计算**：Golden 实现使用 `np.mean(window, axis=(2,3))` 进行批量计算，减少循环层数
3. **动态 shape 支持**：支持 batch_size 和 channels 的动态 shape
4. **循环展开**：支持多种 unroll 大小以适应不同输入规模

## 调用示例

```Python
import torch
import pypto

# 准备数据
batch_size, channels, in_h, in_w = 2, 3, 6, 6
x = torch.rand([batch_size, channels, in_h, in_w], dtype=torch.float32)
y = torch.empty([batch_size, channels, 3, 3], dtype=torch.float32)

# 创建并调用算子
kernel = avg_pool_2d(
    shape=x.shape,
    kernel_size=(2, 2),
    stride=(2, 2),
    padding_mode='SAME',
    run_mode='npu',
    dynamic=True
)
kernel(x, y)

print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")
```

## 支持的测试用例

| batch_size | channels | in_h | in_w | kernel_size | stride | padding_mode | out_h | out_w |
|-----------|----------|------|------|-------------|--------|--------------|-------|-------|
| 2 | 3 | 6 | 6 | (2, 2) | (2, 2) | SAME | 3 | 3 |
| 4 | 8 | 12 | 12 | (3, 3) | (2, 2) | VALID | 5 | 5 |

## 详细实现

- 详见 [avg_pool2d.py](./avg_pool2d.py)
