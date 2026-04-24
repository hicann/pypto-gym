# mHC 算子实现

本目录包含 mHC (Multi-Head Computation) 算子集的 PyPTO 实现，用于在华为昇腾 AI 处理器上执行多路流计算。

## 算子概述

mHC 算子集包含三个核心算子，实现多路流之间的转换和混合操作：

| 算子 | 数学公式 | 功能描述 | 输入 | 输出 |
|------|----------|----------|------|------|
| **mhc_pre** | `out[b,s,d] = Σ_n h[n] × x[b*N+n,s,d]` | N路流 → 1路（加权求和） | x: [batch*N, seq, dim]<br>h: [num_streams] | out: [batch, seq, dim] |
| **mhc_post** | `out[b*N+n,s,d] = x[b,s,d] × h[n]` | 1路 → N路（广播缩放） | x: [batch, seq, dim]<br>h: [num_streams] | out: [batch*N, seq, dim] |
| **mhc_res** | `out[b*N+t,s,d] = Σ_r h[r,t] × x[b*N+r,s,d]` | N路 → N路（流间混合） | x: [batch*N, seq, dim]<br>h: [N, N] | out: [batch*N, seq, dim] |

### 符号说明

- **batch**: 批次大小
- **num_streams (N)**: 流的数量
- **seq**: 序列长度
- **dim**: 隐藏层维度
- **b, n, r, t**: 索引变量

## 代码结构

```
mhc/
├── mhc_post.py    # mhc_post 算子实现 (1路 → N路)
├── mhc_pre.py    # mhc_pre 算子实现 (N路 → 1路)
├── mhc_res.py    # mhc_res 算子实现 (N路 → N路)
└── README.md     # 本文档
```

## 运行方法

### 环境准备

```bash
# 配置 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置设备 ID
export TILE_FWK_DEVICE_ID=0
```

### 执行测试

```bash
# 测试 mhc_post 算子
python3 mhc_post.py

# 测试 mhc_pre 算子
python3 mhc_pre.py

# 测试 mhc_res 算子
python3 mhc_res.py

```

## 核心实现

### mhc_post 算子 (1路 → N路)

```python
@pypto.jit(...)
def mhc_post_kernel(x: pypto.Tensor, h: pypto.Tensor, out: pypto.Tensor) -> None:
    # 1. 扩展 x 到 [batch, num_streams, seq, dim]
    x_expanded = pypto.unsqueeze(x, dim=1)
    x1 = pypto.expand_clone(x_expanded, [batch, num_streams, seq, dim])

    # 2. 扩展 h 到 [batch, num_streams, seq, dim]
    h_expanded = pypto.reshape(h, [1, num_streams, 1, 1])
    h1 = pypto.concat([h_expanded] * batch, dim=0)
    h2 = pypto.concat([h1] * seq, dim=-2)
    h3 = pypto.concat([h2] * dim, dim=-1)

    # 3. 逐元素乘法
    out_expanded = pypto.mul(x1, h3)

    # 4. 变形输出
    result = pypto.reshape(out_expanded, [batch * num_streams, seq, dim])
    pypto.assemble(result, [0, 0, 0], out)
```

### mhc_pre 算子 (N路 → 1路)

```python
@pypto.jit(...)
def mhc_pre_kernel(x: pypto.Tensor, h: pypto.Tensor, out: pypto.Tensor) -> None:
    # 1. 变形 x 到 [batch, num_streams, seq, dim]
    x_reshaped = pypto.reshape(x, [batch, num_streams, seq, dim])

    # 2. 扩展 h 到 [batch, num_streams, seq, dim]
    h_expanded = pypto.reshape(h, [1, num_streams, 1, 1])
    h1 = pypto.concat([h_expanded] * batch, dim=0)
    h2 = pypto.concat([h1] * seq, dim=-2)
    h3 = pypto.concat([h2] * dim, dim=-1)

    # 3. 逐元素乘法
    weighted = pypto.mul(x_reshaped, h3)

    # 4. 在 num_streams 维度求和
    result = pypto.sum(weighted, dim=1)
    pypto.assemble(result, [0, 0, 0], out)
```

### mhc_res 算子 (N路 → N路)

```python
@pypto.jit(...)
def mhc_res_kernel(x: pypto.Tensor, h: pypto.Tensor, out: pypto.Tensor) -> None:
    # 1. 变形 x 和 h
    x_reshaped = pypto.reshape(x, [batch, num_streams, 1, seq * dim])
    h_reshaped = pypto.reshape(h, [1, num_streams, num_streams, 1])

    # 2. 广播到相同形状
    x_expanded = pypto.expand_clone(x_reshaped, [batch, num_streams, num_streams, seq * dim])
    h_expanded1 = pypto.expand_clone(h_reshaped, [batch, num_streams, num_streams, seq * dim])

    # 3. 逐元素乘法
    weighted = pypto.mul(x_expanded, h_expanded1)

    # 4. 在 num_streams 维度求和
    out_expanded = pypto.sum(weighted, 1, True)
    out1 = pypto.reshape(out_expanded, [batch * num_streams, seq, dim])
    pypto.assemble(out1, [0, 0, 0], out)
```

### 精度容差配置

| 数据类型 | 相对误差 (rtol) | 绝对误差 (atol) |
|----------|-----------------|-----------------|
| float32  | 1e-4            | 1e-4            |

## 注意事项

1. **数据类型**: 目前支持 float32
2. **精度验证**: 使用 numpy.testing.assert_allclose 进行精度验证
3. **动态轴**: 输入 tensor 使用空列表表示无动态轴
4. **设备**: 测试数据生成在 NPU 上，结果与 CPU 上的 Golden 结果对比

## 已知限制

1. **sum 操作**: pypto.sum 只支持 DT_FP32 类型
2. **expand_clone**: 只支持单维广播操作
3. **assemble**: 循环内不能对同一输出同时使用 assemble 和普通操作
