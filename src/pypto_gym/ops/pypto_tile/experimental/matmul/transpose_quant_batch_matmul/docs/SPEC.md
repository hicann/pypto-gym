# transpose_quant_batch_matmul 算子需求规格

## 1. 算子概述

### 1.1 算子名称
`transpose_quant_batch_matmul` - 基于MXFP8量化的批量矩阵乘法（带转置）算子

### 1.2 功能描述
本算子实现了基于MXFP8量化的批量矩阵乘法操作，支持多种perm组合（permX1, permX2, permY）和输出dtype（FP16/BF16）。M轴动态化——同一编译kernel可支持不同M大小的输入。外层LOOP_B使用parallel=True并行切分，内层使用scaled_mm API实现MX量化硬件加速。

### 1.3 应用场景
- Transformer模型中的批量矩阵乘法计算
- 大语言模型推理中的量化矩阵乘法
- 专家混合模型(MoE)的批量专家计算
- 动态batch场景下的量化矩阵乘法

---

## 2. 数学公式

### 2.1 计算公式

对于每个batch索引 b（b ∈ [0, B-1]），计算公式为：

$$
out[:, b, :] = permY\left(permX1(x1)[:, b, :] \times permX2(x2)[b, :, :]\right)
$$

其中：
- `x1` 形状 `[M, B, K]`，FP8格式，M轴动态
- `x2` 形状 `[B, K, N]`（permX2=[0,1,2]）或 `[B, N, K]`（permX2=[0,2,1]）
- `x1Scale` 形状 `[M, B, K//64, 2]`，E8M0格式
- `x2Scale` 形状取决于permX2：`[B, K//64, N, 2]` 或 `[B, N, K//64, 2]`
- `permX1=[1,0,2]`：x1从 `[M,B,K]` 变为 `[B,M,K]`
- `permX2=[0,1,2]`：x2保持 `[B,K,N]`；`permX2=[0,2,1]`：x2从 `[B,N,K]` 转为 `[B,K,N]`
- `permY=[1,0,2]`：输出从 `[B,M,N]` 变为 `[M,B,N]`
- `×` 表示MXFP8量化矩阵乘法（反量化 → FP32 matmul → 输出dtype）

### 2.2 参数关系

**参数关系**：
- 输入 x1：形状 `[M, B, K]`（FP8E4M3 或 FP8E5M2）
- 输入 x2：形状 `[B, K, N]` 或 `[B, N, K]`（取决于permX2）
- 输出 out：形状 `[M, B, N]`（FP16 或 BF16，取决于dtype参数）

---

## 3. 输入输出规格

### 3.1 输入参数

| 参数名 | 类型 | 形状 | 描述 |
|--------|------|------|------|
| x1 | Tensor | `[M, B, K]` | 左矩阵（FP8E4M3 或 FP8E5M2）— M轴动态 |
| x2 | Tensor | `[B, K, N]` 或 `[B, N, K]` | 右矩阵（FP8E4M3 或 FP8E5M2） |
| x1Scale | Tensor | `[M, B, K//64, 2]` | 左矩阵缩放因子（FP8E8M0） |
| x2Scale | Tensor | `[B, K//64, N, 2]` 或 `[B, N, K//64, 2]` | 右矩阵缩放因子（FP8E8M0） |
| permX1 | List[int] | 配置参数 | x1的perm组合，默认 `[1, 0, 2]` |
| permX2 | List[int] | 配置参数 | x2的perm组合，默认 `[0, 1, 2]` |
| permY | List[int] | 配置参数 | 输出的perm组合，默认 `[1, 0, 2]` |
| dtype | int | 配置参数 | 输出dtype：1=FP16, 27=BF16 |
| tile_config | ShapeConfig | 配置对象 | 包含batch_size、形状和分块参数 |

### 3.2 输出参数

| 参数名 | 类型 | 形状 | 描述 |
|--------|------|------|------|
| out | Tensor | `[M, B, N]` | 输出矩阵（FP16 或 BF16） |

### 3.3 ShapeConfig 配置类

```python
@dataclass
class ShapeConfig:
    ori_shape: list          # [M, K, N] — M仅用于golden/wrapper
    batch_size: int          # B轴大小（编译期固定）
    m_tile_shape: list       # Cube M 轴切分
    k_tile_shape: list       # Cube K 轴切分
    n_tile_shape: list       # Cube N 轴切分
    vector_tile_shape: list  # Vector 切分配置
    num_k_groups: int = 1    # K维度split组数
    num_n_groups: int = 1    # N维度split组数
    in_dtype: pypto.DataType = pypto.DT_FP8E4M3
    out_dtype: pypto.DataType = pypto.DT_BF16
    permX1: List[int] = None  # 默认 [1, 0, 2]
    permX2: List[int] = None  # 默认 [0, 1, 2]
    permY: List[int] = None   # 默认 [1, 0, 2]
    description: str = ""
```

---

## 4. Shape 范围与约束

### 4.1 动态轴取值范围

| 轴 | 取值范围 | 说明 |
|----|---------|------|
| M | 动态（8 ~ 32768） | 同一编译kernel支持不同M |
| K | {128, 512} | 矩阵乘公共维度 |
| N | {128, 512} | 右矩阵列维度 |
| B | {128} | batch维度（编译期固定） |

### 4.2 关键约束条件

**⚠️ MX量化约束**

```
K 必须是 64 的倍数
```

**约束说明**：
1. **MX 量化约束**：K 轴必须 64 对齐（每 64 个元素共享一个缩放因子）
2. **Scale 形状约束**：x1Scale 的 K 维度 = K//64，x2Scale 形状取决于 permX2

### 4.3 内轴对齐约束

- FP8 数据的内轴必须满足 32 字节对齐
- K 维度必须 ≥ 64 且为 64 的倍数

### 4.4 典型配置

| 配置名称 | M | K | N | B | permX2 | out_dtype | 用途 |
|---------|---|---|---|----|--------|-----------|------|
| P0_小M | 8 | 128 | 512 | 128 | [0,2,1] | BF16 | 小M动态验证 |
| P0_中M | 128 | 128 | 512 | 128 | [0,1,2] | FP16 | 中等规模验证 |
| P0_大M | 8192 | 128 | 512 | 128 | [0,1,2] | FP16 | 大M性能验证 |
| P0_超大M | 32768 | 128 | 512 | 128 | [0,1,2] | FP16 | 超大规模验证 |

---

## 5. 支持的数据类型

### 5.1 数据类型支持表

| 数据类型 | PyPTO DataType | Torch dtype | 格式说明 | 适用场景 |
|----------|----------------|-------------|----------|----------|
| 输入数据 | DT_FP8E4M3 | torch.float8_e4m3fn | 4位指数+3位尾数 | 训练/推理 |
| 输入数据 | DT_FP8E5M2 | torch.float8_e5m2 | 5位指数+2位尾数 | 动态范围优先 |
| 缩放因子 | DT_FP8E8M0 | torch.float8_e8m0fnu | 8位纯指数格式 | MX 量化专用 |
| 输出数据 | DT_FP16 | torch.float16 | 半精度浮点 | FP16输出 |
| 输出数据 | DT_BF16 | torch.bfloat16 | BF16格式 | BF16输出 |

---

## 6. 矩阵格式约束

### 6.1 当前实现支持的perm组合

**perm组合说明**：

- **permX1=[1, 0, 2]**：x1从 `[M,B,K]` 变为 `[B,M,K]`（batch first）
- **permX2=[0, 1, 2]**：x2保持 `[B,K,N]` 布局（K,N顺序，不需要b_trans）
- **permX2=[0, 2, 1]**：x2从 `[B,N,K]` 转为 `[B,K,N]`（需要b_trans=True, scale_b_trans=True）
- **permY=[1, 0, 2]**：输出从 `[B,M,N]` 变为 `[M,B,N]`

---

## 7. MXFP8 量化说明

### 7.1 MX 量化格式

MX 量化（Microscaling Quantization）是一种基于块缩放的量化格式：

- **量化块大小**：每 64 个元素共享一个缩放因子
- **缩放因子格式**：FP8E8M0（8位纯指数），隐含 mantissa=1.0
- **数据格式**：FP8E4M3FN 或 FP8E5M2
- **Scale 存储格式**：x1Scale `[M, B, K//64, 2]`，x2Scale 形状取决于permX2

---

## 8. 精度要求

### 8.1 数据类型转换路径

- 输入类型：`x1` 和 `x2` 为 FP8E4M3 或 FP8E5M2
- Scale类型：`x1Scale` 和 `x2Scale` 为 FP8E8M0
- 计算类型：缩放后的矩阵乘法在 FP32 下进行
- 输出类型：FP16 或 BF16（取决于dtype参数）

### 8.2 精度容差

- **相对容差 (RTOL)**：1e-3
- **绝对容差 (ATOL)**：1e-3

### 8.3 精度验证标准

```python
numpy.testing.assert_allclose(result, golden, rtol=1e-3, atol=1e-3)
```

---

## 9. 性能要求

### 9.1 计算特点

- **MXFP8 量化**：使用scaled_mm API实现MX量化硬件加速
- **B轴并行**：parallel loop实现batch并行计算
- **M轴动态**：同一编译kernel支持不同M大小，减少重编译开销
- **perm组合支持**：通过permX2决定是否使用b_trans，避免数据搬运

### 9.2 优化方向

1. B轴并行循环优化（parallel=True）
2. NBuffer配置优化
3. TileShape优化
4. Cache策略优化（NONE_CACHEABLE）

---

## 10. 测试验证要求

### 10.1 功能测试矩阵

| 测试名称 | M | K | N | B | permX2 | in_dtype | out_dtype | 验证点 |
|---------|---|---|---|----|--------|----------|-----------|-------|
| test1 | 8 | 128 | 512 | 128 | [0,2,1] | FP8E5M2 | BF16 | 小M+b_trans验证 |
| test2 | 128 | 128 | 512 | 128 | [0,1,2] | FP8E4M3 | FP16 | 中M+标准验证 |
| test3 | 8192 | 128 | 512 | 128 | [0,1,2] | FP8E4M3 | FP16 | 大M性能验证 |
| test4 | 32768 | 128 | 512 | 128 | [0,1,2] | FP8E4M3 | FP16 | 超大规模验证 |

### 10.2 精度测试

- Golden 实现：纯 PyTorch 实现，作为精度基准
- 三态标记：`[PRECISION_PASS]` 或 `[PRECISION_FAIL]`
- 对比方法：与 golden 实现逐元素对比

### 10.3 边界测试

- M轴动态验证：8 → 128 → 8192 → 32768（同一kernel）
- permX2切换：[0,1,2] ↔ [0,2,1]（验证b_trans路径）
- dtype切换：FP16 ↔ BF16
- in_dtype切换：FP8E4M3 ↔ FP8E5M2

---

## 11. 实现约束

### 11.1 API 映射要求

- 必须使用 PyPTO 框架实现
- 支持 PyPTO JIT 编译
- 支持 parallel loop 并行计算
- 支持 MXFP8 量化格式
- 支持 M轴动态化

### 11.2 内存管理

- 输入输出张量必须 contiguous
- NONE_CACHEABLE策略用于输入张量
- 输出dtype由tile_config.out_dtype决定
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
- `pypto.set_cube_tile_shapes`：Cube分块配置
- `pypto.set_vec_tile_shapes`：Vector分块配置