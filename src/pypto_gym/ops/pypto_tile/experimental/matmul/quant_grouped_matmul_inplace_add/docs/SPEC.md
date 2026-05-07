# quant_grouped_matmul_inplace_add 算子需求规格

## 1. 算子概述

### 1.1 算子名称
`quant_grouped_matmul_inplace_add` - 基于MXFP8量化的分组矩阵乘法就地加法算子

### 1.2 功能描述
本算子实现了基于MXFP8量化的分组矩阵乘法就地加法操作，支持多分组场景下的高效梯度累积计算。将GroupedMatMul和InplaceAdd融合，使用MXFP8量化方式，支持按K轴分组进行矩阵乘法计算并就地累加到输出张量。

### 1.3 应用场景
- Transformer模型中的梯度累积计算
- 专家混合模型(MoE)的专家并行计算
- 多分组线性层的权重更新计算
- 大语言模型训练中的量化矩阵乘法

---

## 2. 数学公式

### 2.1 计算公式

对于每个分组 i（i ∈ [0, num_groups-1]），计算公式为：

$$
y_i = y_i + \left(\left(a_{k_i} \times scale_{a_i}\right) @ \left(b_{k_i} \times scale_{b_i}\right)\right)
$$

其中：
- `k_i` 为第 i 个分组的 K 轴块，范围 `[i×k_block, (i+1)×k_block]`，`k_block = K / num_groups`
- `a_{k_i}` 为左矩阵的第 i 个 K 轴切片：`a[begin:end, :]`，形状 `[k_block, M]`
- `b_{k_i}` 为右矩阵的第 i 个 K 轴切片：`b[begin:end, :]`，形状 `[k_block, N]`
- `scale_{a_i}` 和 `scale_{b_i}` 为对应的 MXFP8 量化缩放因子
- `@` 表示矩阵乘法
- 输出 `y` 形状为 `[num_groups, M, N]`，每个分组对应一个输出矩阵

### 2.2 参数关系

**参数关系**：
- 输入矩阵 a：形状 `[K, M]`（左矩阵转置）
- 输入矩阵 b：形状 `[K, N]`（右矩阵不转置）
- 输出矩阵 y：形状 `[num_groups, M, N]`（每个分组一个输出）

---

## 3. 输入输出规格

### 3.1 输入参数

| 参数名 | 类型 | 形状 | 描述 |
|--------|------|------|------|
| a | Tensor | `[K, M]` | 左矩阵（FP8E4M3 或 FP8E5M2） |
| b | Tensor | `[K, N]` | 右矩阵（FP8E4M3 或 FP8E5M2） |
| scaled_a | Tensor | `[(K//64)+num_groups, M, 2]` | 左矩阵缩放因子（FP8E8M0） |
| scaled_b | Tensor | `[(K//64)+num_groups, N, 2]` | 右矩阵缩放因子（FP8E8M0） |
| y | Tensor | `[num_groups, M, N]` | 输出矩阵（FP32，同时作为输入初始值） |
| tile_config | ShapeConfig | 配置对象 | 包含分组数、形状和分块参数 |

### 3.2 输出参数

| 参数名 | 类型 | 形状 | 描述 |
|--------|------|------|------|
| y | Tensor | `[num_groups, M, N]` | 累加后的输出矩阵（FP32） |

### 3.3 ShapeConfig 配置类

```python
@dataclass
class ShapeConfig:
    ori_shape: list          # [M, K, N]
    num_groups: int          # 分组数量
    m_tile_shape: list       # Cube M 轴切分 [mL1, mL0]
    k_tile_shape: list       # Cube K 轴切分 [kL1, kL0]
    n_tile_shape: list       # Cube N 轴切分 [nL1, nL0]
    vector_tile_shape: list  # Vector 切分配置
    in_dtype: pypto.DataType # 输入数据类型
    a_trans: bool = True     # 左矩阵转置（固定为 True）
    b_trans: bool = False    # 右矩阵不转置（固定为 False）
    a_format_nz: bool = False
    b_format_nz: bool = False
    c_format_nz: bool = False
    description: str = ""
```

---

## 4. Shape 范围与约束

### 4.1 动态轴取值范围

| 轴 | 取值范围 | 说明 |
|----|---------|------|
| M | {768, 1024, 2048} | 左矩阵列维度（内轴） |
| K | {6144, 8192, 16384} | 矩阵乘公共维度（切分轴） |
| N | {4096, 5120, 8192} | 右矩阵列维度（内轴） |
| num_groups | {32, 64} | K轴分组数量 |

### 4.2 关键约束条件

**⚠️ 重要：K 轴必须满足双重对齐条件**

```
K 必须是 64 × num_groups 的倍数
```

**约束说明**：
1. **MX 量化约束**：K 轴必须 64 对齐（每 64 个元素共享一个缩放因子）
2. **均匀切分约束**：K 轴必须能被 `num_groups` 整除（均匀分组）
3. **综合约束**：`K = 64 × num_groups × k_block_count`（例如：6144 = 64 × 32 × 3）

### 4.3 内轴对齐约束

**内轴对齐说明**：
- FP8 数据的内轴必须满足 32 字节对齐（即 32 个元素）
- M 和 N 维度必须 ≥ 32 且为 32 的倍数

### 4.4 典型配置

| 配置名称 | M | K | N | num_groups | 用途 |
|---------|---|---|----|-----------|------|
| P0_标准 | 768 | 6144 | 4096 | 32 | 标准训练场景 |
| P0_大K | 1024 | 16384 | 4096 | 64 | 大模型配置 |
| P0_大M | 2048 | 6144 | 5120 | 32 | 大隐藏层场景 |
| P0_最大 | 2048 | 16384 | 8192 | 64 | 最大配置 |

---

## 5. 支持的数据类型

### 5.1 数据类型支持表

| 数据类型 | PyPTO DataType | Torch dtype | 格式说明 | 适用场景 |
|----------|----------------|-------------|----------|----------|
| 输入数据 | DT_FP8E4M3 | torch.float8_e4m3fn | 4位指数+3位尾数，精度更高 | 训练场景，精度优先 |
| 输入数据 | DT_FP8E5M2 | torch.float8_e5m2 | 5位指数+2位尾数，动态范围更大 | 推理场景，动态范围优先 |
| 缩放因子 | DT_FP8E8M0 | torch.float8_e8m0fnu | 8位纯指数格式，仅包含指数部分 | MX 量化专用 |
| 输出数据 | DT_FP32 | torch.float32 | 标准 FP32 格式 | 累加输出 |

---

## 6. 矩阵格式约束

### 6.1 当前实现配置

**当前实现配置**：
- **左矩阵 a**：转置格式（`a_trans=True`）
  - 形状：`[K, M]`
  - Scale 形状：`scaled_a = [(K//64)+num_groups, M, 2]`
  - 内轴：M 维度（需 32 字节对齐）

- **右矩阵 b**：非转置格式（`b_trans=False`）
  - 形状：`[K, N]`
  - Scale 形状：`scaled_b = [(K//64)+num_groups, N, 2]`
  - 内轴：N 维度（需 32 字节对齐）

---

## 7. MXFP8 量化说明

### 7.1 MX 量化格式

MX 量化（Microscaling Quantization）是一种基于块缩放的量化格式：

- **量化块大小**：每 64 个元素（K 轴）共享一个缩放因子
- **缩放因子格式**：FP8E8M0（8位纯指数），仅包含指数部分，隐含 mantissa=1.0
- **数据格式**：FP8E4M3FN（4位指数+3位尾数）或 FP8E5M2（5位指数+2位尾数）
- **Scale 存储格式**：连续存储，分组间有 1 元素间隔，总长度 `(K//64) + num_groups`

---

## 8. 精度要求

### 8.1 数据类型转换路径

- 输入类型：`a` 和 `b` 为 FP8E4M3 或 FP8E5M2
- Scale类型：`scaled_a` 和 `scaled_b` 为 FP8E8M0
- 计算类型：缩放后的矩阵乘法在 FP32 下进行
- 输出类型：累加结果为 FP32

### 8.2 精度容差

- **相对容差 (RTOL)**：0.01（MXFP8量化精度）
- **绝对容差 (ATOL)**：0.001

### 8.3 精度验证标准

输出结果与 golden 实现对比需满足：
```python
numpy.testing.assert_allclose(result, golden, rtol=0.01, atol=0.001)
```

---

## 9. 性能要求

### 9.1 计算特点

- **量化矩阵乘法**：使用MXFP8量化格式，减少内存占用和计算开销
- **K轴分组**：支持多分组并行计算，提升并行度
- **就地加法**：输出张量同时作为输入，减少内存拷贝
- **融合计算**：通过 `scaled_mm` 算子实现缩放和矩阵乘法的融合

### 9.2 优化方向

1. 循环并行化优化（parallel loop）
2. NBuffer配置优化（cube_nbuffer_setting, vec_nbuffer_setting）
3. TileShape优化（M/K/N轴分块）
4. 批量累加优化（batch add）

---

## 10. 测试验证要求

### 10.1 功能测试矩阵

| 测试名称 | M | K | N | num_groups | 验证点 |
|---------|---|---|----|-----------|-------|
| 极小规模 | 32 | 64 | 32 | 1 | 基本功能验证 |
| 小规模 | 128 | 128 | 128 | 2 | 分组功能验证 |
| 中等规模 | 768 | 6144 | 4096 | 32 | 标准场景验证 |
| 大规模 | 2048 | 16384 | 8192 | 64 | 性能验证 |

### 10.2 精度测试

- Golden 实现：纯 PyTorch 实现，作为精度基准
- 三态标记：`[PRECISION_PASS]` 或 `[PRECISION_FAIL]`
- 对比方法：与 golden 实现逐元素对比

### 10.3 边界测试

- 最小 M/N 值：32（内轴对齐）
- 最大 K 值：16384
- num_groups 切换：32 ↔ 64（验证重编译）
- K轴对齐验证：验证 K % (64 × num_groups) == 0

---

## 11. 实现约束

### 11.1 API 映射要求

- 必须使用 PyPTO 框架实现
- 支持 PyPTO JIT 编译
- 支持 parallel loop 并行计算
- 支持 MXFP8 量化格式

### 11.2 内存管理

- 输入输出张量必须 contiguous
- 中间计算使用 FP32，需考虑内存占用
- 支持 inplace 操作减少内存拷贝
- Scale tensor 按MXFP8格式存储

### 11.3 兼容性

- PyPTO 版本要求：与 CANN 版本匹配
- 硬件要求：华为昇腾 AI 处理器
- 产品支持：Ascend 950PR/Ascend 950DT
- 软件栈：CANN 8.5.0+

---

## 12. 参考实现

### 12.1 参考文献

- MXFP8 量化规范文档
- PyPTO API 文档
- 性能优化指南

### 12.2 相关算子

- `scaled_mm`：MXFP8量化矩阵乘法基础算子
- `pypto.loop`：并行循环算子
- `pypto.add`：逐元素加法算子