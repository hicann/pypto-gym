# quant_grouped_matmul_inplace_add 算子设计文档

## 1. 设计概述

### 1.1 算子定位
`quant_grouped_matmul_inplace_add` 是基于MXFP8量化的分组矩阵乘法就地加法算子，用于多分组场景下的高效梯度累积计算。

### 1.2 核心特性
- **MXFP8 量化**：数据支持 FP8E4M3FN 和 FP8E5M2 格式，缩放因子使用 FP8E8M0 格式
- **K 轴分组**：K 轴按 `num_groups` 均匀切分，每个分组对应不同的权重矩阵块
- **就地加法**：输出张量同时作为输入，执行就地加法操作 `y = y + result`，减少内存拷贝
- **融合计算**：通过 `scaled_mm` 算子实现缩放和矩阵乘法的融合计算
- **新前端写法**：使用 PyPTO 新前端 API，支持类型注解和自动类型推导

---

## 2. 计算图设计

### 2.1 分组策略可视化

```
输入矩阵 a [K, M]              输入矩阵 b [K, N]
     |                              |
     ├─ Group 0 [0:k_block, :]      ├─ Group 0 [0:k_block, :]
     ├─ Group 1 [k_block:2k, :]     ├─ Group 1 [k_block:2k, :]
     ├─ Group 2 [2k:3k, :]          ├─ Group 2 [2k:3k, :]
     ...                            ...
     └─ Group i [i*k:(i+1)*k, :]    └─ Group i [i*k:(i+1)*k, :]
     
Scale a [(K//64)+num_groups, M, 2]  Scale b [(K//64)+num_groups, N, 2]
     |                              |
     ├─ Group 0 scale [offset:offset+length, :, :]
     ├─ Group 1 scale [offset+length+1:..., :, :]
     ...
     
     ↓ scaled_mm(a_i, b_i, scale_i) ↓
     
     mm_result_tensor[i] [M, N]  ← 第i个分组的矩阵乘结果
     
     ↓ parallel loop (num_groups) ↓
     
     收集所有分组结果 → mm_result_tensor [num_groups, M, N]
     
     ↓ batch add (循环结束后) ↓
     
     y = y + mm_result_tensor  ← 就地加法（融合）
     
输出: y [num_groups, M, N] (FP32)
```

**数据流说明**：

| 分支 | 输入 | 计算 | 输出 | Shape变化 |
|------|------|------|------|-----------|
| **Group i** | a[i×k:(i+1)×k, :] [k_block, M] | × b[i×k:(i+1)×k, :] [k_block, N] | mm_result[i] [M, N] | [k_block, M] × [k_block, N] → [M, N] |
| **Scale提取** | scaled_a[offset:offset+length, :, :] | 对应Group i | scale_a_i [scale_length, M, 2] | MXFP8量化scale |
| **累加融合** | y [num_groups, M, N] | + mm_result_tensor | y (updated) | 就地加法 |

### 2.2 Scale Offset计算公式

对于第 i 个分组（i ∈ [0, num_groups-1]）：

```python
# K轴切片范围
begin = i * k_block
end = (i + 1) * k_block

# Scale offset计算（关键：分组间有1元素间隔）
scale_offset = begin // 64 + i  # 加上分组索引 i
scale_length = k_block // 64

# Scale切片
scaled_a_i = scaled_a[scale_offset : scale_offset + scale_length, :, :]
scaled_b_i = scaled_b[scale_offset : scale_offset + scale_length, :, :]
```

**Scale offset计算原理**：
- MXFP8格式要求每64个元素共享一个scale，因此基础offset = begin // 64
- 分组间有1元素间隔，因此额外偏移 = i（分组索引）
- 总offset = begin // 64 + i

---

## 3. Tiling策略

### 3.1 Cube Tiling配置

**分块配置**：

| 轴 | Tile Shape | 说明 |
|----|-----------|------|
| M轴 | `[mL1, mL0]` 例如 `[128, 128]` | 左矩阵列维度分块 |
| K轴 | `[kL1, kL0]` 例如 `[64, 192]` | 公共维度分块（MXFP8量化要求kL0=64） |
| N轴 | `[nL1, nL0]` 例如 `[256, 1024]` | 右矩阵列维度分块 |

**分块原理**：
- **K轴分块约束**：kL0必须为64，满足MXFP8量化块大小要求
- **M/N轴分块**：考虑L1/L2 cache容量，平衡计算强度和内存访问
- **固定分组**：k_block = K / num_groups，每个分组的K轴大小固定

### 3.2 Vector Tiling配置

**分块配置**：

| 配置 | Tile Shape | 说明 |
|------|-----------|------|
| vector_tile_shape | `[1, 32, 512, 2]` | Vector核4维分块配置 |

**分块原理**：
- 第一维：1（分组维度，每组独立）
- 第二维：32（M轴分块，内轴对齐）
- 第三维：512（N轴分块）
- 第四维：2（辅助维度）

### 3.3 Tile Shape优化分析

**性能提升案例**：

| 配置 | 执行时间 | 核心利用率 | 提升 |
|------|---------|-----------|------|
| 基线配置 | 1709 us | 32.58% | - |
| 优化配置（k_tile=[64,192], n_tile=[256,1024]） | 1342 us | 52.22% | +21.5% |

**优化原理**：
- **k_tile_shape=[64,192]**：kL0=64满足MXFP8量化块要求，kL1=192覆盖完整分组
- **n_tile_shape=[256,1024]**：优化N轴分块，提升内存访问效率
- **cube_nbuffer_setting={-1:4}**：最后一维4缓冲，提升数据复用

---

## 4. Loop结构设计

### 4.1 Parallel Loop实现

```python
# 并行循环：每个分组独立计算
for i in pypto.loop(num_groups, parallel=True):
    begin = i * k_block
    end = (i + 1) * k_block
    scale_offset = begin // 64 + i
    scale_length = k_block // 64
    
    # 提取当前分组的输入和权重
    x = a[begin:end, :]
    weight = b[begin:end, :]
    scaled_x = scaled_a[scale_offset : scale_offset + scale_length, :, :]
    scaled_weight = scaled_b[scale_offset : scale_offset + scale_length, :, :]
    
    # 计算量化矩阵乘法
    mm_result_tensor[i] = pypto.scaled_mm(
        x, weight, pypto.DT_FP32, 
        scaled_x, scaled_weight,
        a_trans=True, b_trans=False
    )
```

**并行策略**：
- `parallel=True`：启用并行执行，所有分组同时计算
- 每个分组独立计算，无数据依赖
- 结果存储在中间tensor `mm_result_tensor`

### 4.2 Batch Add优化策略

**优化对比**：

| 策略 | 实现方式 | 性能 |
|------|---------|------|
| **逐组累加**（低效） | 在循环内执行 `y[i] = y[i] + mm_result[i]` | 循环内多次累加，性能损失 |
| **批量累加**（高效） | 循环结束后执行 `y = y + mm_result_tensor` | 一次批量累加，性能最优 |

**实现代码**：
```python
# 批量累加（循环结束后）
y[:,:,:] = pypto.add(y, mm_result_tensor)
```

**性能提升原因**：
- 避免循环内的多次累加开销
- PyPTO优化器可识别批量累加模式，生成更优指令
- 内存访问模式更连续

---

## 5. 数据流设计

### 5.1 分组切片逻辑

```python
# 输入验证
assert K % (64 * num_groups) == 0  # K轴双重对齐
assert M >= 32 and M % 32 == 0    # M轴对齐
assert N >= 32 and N % 32 == 0    # N轴对齐

# 计算分组参数
k_block = K // num_groups  # 每个分组的K轴大小

# 分组循环
for i in range(num_groups):
    # K轴切片
    begin = i * k_block
    end = (i + 1) * k_block
    
    # 矩阵切片（a_trans=True, b_trans=False）
    a_slice = a[begin:end, :]  # [k_block, M]
    b_slice = b[begin:end, :]  # [k_block, N]
    
    # Scale切片（关键计算）
    scale_offset = begin // 64 + i  # 分组间1元素间隔
    scale_length = k_block // 64
    
    scaled_a_slice = scaled_a[scale_offset : scale_offset + scale_length, :, :]
    scaled_b_slice = scaled_b[scale_offset : scale_offset + scale_length, :, :]
```

### 5.2 就地加法实现

```python
# 创建中间tensor（FP32）
mm_result_tensor = pypto.tensor([num_groups, M, N], pypto.DT_FP32)

# 循环计算（结果存储到中间tensor）
for i in pypto.loop(num_groups, parallel=True):
    mm_result_tensor[i] = pypto.scaled_mm(...)

# 就地加法（y作为输入和输出）
y[:,:,:] = pypto.add(y, mm_result_tensor)
```

**就地加法优势**：
- 减少内存拷贝（y既是输入又是输出）
- 降低内存峰值（无需额外的累加结果tensor）
- 提升性能（PyPTO优化器可识别inplace模式）

---

## 6. 精度设计

### 6.1 MXFP8量化路径

```
输入精度: FP8E4M3/E5M2 (a, b) + FP8E8M0 (scaled_a, scaled_b)
     ↓
缩放精度: a × scaled_a → FP32 (隐式)
         b × scaled_b → FP32 (隐式)
     ↓
计算精度: FP32 (矩阵乘法)
     ↓
累加精度: FP32 (y = y + mm_result_tensor)
     ↓
输出精度: FP32 (y)
```

### 6.2 精度保证措施

| 措施 | 说明 |
|------|------|
| FP32中间计算 | 避免FP8累加误差放大 |
| MXFP8量化格式 | FP8E4M3/E5M2保证输入精度，FP8E8M0保证scale精度 |
| 就地加法 | FP32累加，保证精度 |

### 6.3 精度验证标准

```python
RTOL = 0.01  # MXFP8量化精度容差
ATOL = 0.001
numpy.testing.assert_allclose(result, golden, rtol=RTOL, atol=ATOL)
```

---

## 7. 性能优化设计

### 7.1 NBuffer配置

```python
@pypto.frontend.jit(
    pass_options={
        "cube_nbuffer_setting": {-1: 4},
        "vec_nbuffer_setting": {-2: 1, -1: 4}
    },
    runtime_options={
        "stitch_function_max_num": 8
    }
)
```

**参数说明**：
- `cube_nbuffer_setting={-1:4}`：Cube核最后一维使用4缓冲，提升数据复用
- `vec_nbuffer_setting={-2:1,-1:4}`：Vector核倒数第二维1缓冲、最后一维4缓冲
- `stitch_function_max_num=8`：函数拼接上限，控制编译复杂度

### 7.2 性能数据

**Case1性能对比**：

| 指标 | 基准性能 | 优化后性能 | 提升 |
|------|---------|-----------|------|
| 执行时间 | 1709 us | 1342 us | 21.5% |
| 核心利用率 | 32.58% | 52.22% | +19.64% |
| 任务数量 | 7746 | 2370 | -69.4% |

**优化配置**：
- NBuffer 配置：`cube_nbuffer_setting={-1:4}`, `vec_nbuffer_setting={-2:1,-1:4}`
- TileShape 优化：`k_tile_shape=[64,192]`, `n_tile_shape=[256,1024]`

### 7.3 TileShape优化原理

**关键优化点**：

1. **k_tile_shape=[64,192]**：
   - kL0=64：满足MXFP8量化块大小要求（每64元素共享scale）
   - kL1=192：覆盖完整分组（k_block=192），减少边界处理

2. **n_tile_shape=[256,1024]**：
   - nL0=1024：优化N轴内存访问，提升cache命中率
   - nL1=256：控制L1 cache占用

3. **cube_nbuffer_setting={-1:4}**：
   - 最后一维4缓冲：N轴数据复用，减少重复加载

---

## 8. Kernel签名设计

### 8.1 ShapeConfig配置类

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

### 8.2 Kernel函数签名

```python
@pypto.frontend.jit(
    pass_options={
        "cube_nbuffer_setting": {-1: 4},
        "vec_nbuffer_setting": {-2: 1, -1: 4}
    },
    runtime_options={
        "stitch_function_max_num": 8
    }
)
def scaled_matmul_kernel(
    a: pypto.Tensor(),            # 左矩阵 [K, M]
    b: pypto.Tensor(),            # 右矩阵 [K, N]
    scaled_a: pypto.Tensor(),     # 左矩阵缩放因子 [(K//64)+num_groups, M, 2]
    scaled_b: pypto.Tensor(),     # 右矩阵缩放因子 [(K//64)+num_groups, N, 2]
    y: pypto.Tensor(),            # 输出矩阵 [num_groups, M, N]
    tile_config: ShapeConfig      # 配置参数
) -> None
```

---

## 9. 测试设计

### 9.1 测试矩阵

| 测试名称 | M | K | N | num_groups | k_block | 验证点 |
|---------|---|---|----|-----------|---------|-------|
| test_极小规模 | 32 | 64 | 32 | 1 | 64 | 基本功能 |
| test_小规模 | 128 | 128 | 128 | 2 | 64 | 分组功能 |
| test_中等规模 | 768 | 6144 | 4096 | 32 | 192 | 标准场景 |
| test_大规模 | 2048 | 16384 | 8192 | 64 | 256 | 性能验证 |

### 9.2 精度验证方法

```python
# 对比方法
result_np = result.cpu().float().numpy()
golden_np = golden_output.float().numpy()
assert_allclose(result_np, golden_np, rtol=0.01, atol=0.001)
```

### 9.3 三态标记

- `[PRECISION_PASS]`：精度验证通过
- `[PRECISION_FAIL]`：精度验证失败

---

## 10. 计算复杂度分析

### 10.1 计算FLOPS

| 操作 | FLOPS | 说明 |
|------|-------|------|
| scaled_mm (每个分组) | 2 × k_block × M × N | 矩阵乘法 |
| 总scaled_mm (num_groups分组) | num_groups × 2 × k_block × M × N = 2 × K × M × N | 所有分组 |
| inplace add | num_groups × M × N | 逐元素加法 |

**总 FLOPS**：
```
约 2 × K × M × N + num_groups × M × N
≈ 2 × 6144 × 768 × 4096 + 32 × 768 × 4096
≈ 38.6 GFLOPS（Case1）
```

### 10.2 内存访问量

| 操作 | 访问量 | 说明 |
|------|-------|------|
| 读 a | K × M × 1 byte | FP8数据 |
| 读 b | K × N × 1 byte | FP8数据 |
| 读 scaled_a | (K//64+num_groups) × M × 2 × 1 byte | FP8E8M0 |
| 读 scaled_b | (K//64+num_groups) × N × 2 × 1 byte | FP8E8M0 |
| 读 y (输入) | num_groups × M × N × 4 bytes | FP32 |
| 写 y (输出) | num_groups × M × N × 4 bytes | FP32 |

**总访问量**：
```
≈ K × (M + N) + 2 × (K//64+num_groups) × (M + N) + 8 × num_groups × M × N bytes
```

### 10.3 计算强度

```
计算强度 = FLOPS / 访存量
≈ (2 × K × M × N) / (K × (M+N) + 8 × num_groups × M × N)
≈ 高计算强度（矩阵乘法为主）
```

---

## 11. 文件清单

| 文件 | 职责 |
|------|------|
| `SPEC.md` | 需求规格 |
| `API_REPORT.md` | API映射报告 |
| `DESIGN.md` | 本设计文档 |
| `README.md` | 使用说明 |
| `quant_grouped_matmul_inplace_add.py` | Kernel实现 |

---

## 12. 参考文档

- [PyPTO编程指南](../../../docs/pypto_programming_guide.md)
- [MXFP8量化规范](../../../docs/tutorials/quantization/)
- [性能优化指南](../../../docs/tutorials/debug/performance.md)