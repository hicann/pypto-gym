# transpose_quant_batch_matmul 算子设计文档

## 1. 设计概述

### 1.1 算子定位
`transpose_quant_batch_matmul` 是基于MXFP8量化的批量矩阵乘法（带转置）算子，用于动态M轴场景下的高效量化矩阵乘法计算，支持多种perm组合和输出dtype。

### 1.2 核心特性
- **MXFP8 量化**：数据支持 FP8E4M3FN 和 FP8E5M2 格式，缩放因子使用 FP8E8M0 格式
- **M轴动态化**：同一编译kernel支持不同M大小，运行时通过shape[0]获取实际M值
- **B轴并行**：LOOP_B使用parallel=True，每个batch独立计算
- **perm组合支持**：通过permX2决定scaled_mm的b_trans参数，硬件内完成转置
- **输出dtype灵活**：支持FP16和BF16输出

---

## 2. 计算图设计

### 2.1 Batch并行计算可视化

```
输入 x1 [M, B, K] (FP8)               输入 x2 [B, K, N] 或 [B, N, K] (FP8)
     |                                       |
     ├─ b_idx=0 → x1[:,0,:] [M,K]           ├─ b_idx=0 → x2[0,:,:] [K,N] 或 [N,K]
     ├─ b_idx=1 → x1[:,1,:] [M,K]           ├─ b_idx=1 → x2[1,:,:]
     ├─ b_idx=2 → x1[:,2,:] [M,K]           ├─ b_idx=2 → x2[2,:,:]
     ...                                    ...
     
     x1Scale [M, B, K//64, 2] (E8M0)       x2Scale (shape varies by permX2)
     |                                       |
     ├─ b_idx=0 → x1Scale[:,0,:,:]          ├─ b_idx=0 → x2Scale[0,...]
     ...                                    ...
     
     ↓ LOOP_B (parallel=True) ↓
     
     For each b_idx:
       x1_slice = x1[:, b_idx, :]           # [M, K]
       x1_scale_slice = x1Scale[:, b_idx, :, :]  # [M, K//64, 2]
       
       if permX2 == [0, 1, 2]:
         mm_result = scaled_mm(x1_slice, x2[b_idx,:,:], out_dtype,
                               x1_scale_slice, x2Scale[b_idx,:,:,:])
       else:
         mm_result = scaled_mm(x1_slice, x2[b_idx,:,:], out_dtype,
                               x1_scale_slice, x2Scale[b_idx,:,:,:],
                               b_trans=True, scale_b_trans=True)
       
       out[:, b_idx, :] = mm_result          # 写回3D输出
     
输出: out [M, B, N] (FP16 或 BF16)
```

**数据流说明**：

| 分支 | 输入 | 计算 | 输出 | Shape变化 |
|------|------|------|------|-----------|
| **batch b** | x1[:,b,:] `[M,K]` | scaled_mm × x2[b,:,:] | mm_result `[M,N]` | `[M,K]×[K,N]→[M,N]` 或 `[M,K]×[N,K]^T→[M,N]` |
| **Scale提取** | x1Scale[:,b,:,:] `[M,K//64,2]` | 对应batch b | scale slice | MXFP8量化scale |
| **写入输出** | out[:,b,:] `[M,N]` | mm_result赋值 | out (updated) | 2D→3D切片写入 |

### 2.2 perm组合对计算路径的影响

```
permX2 = [0, 1, 2] (K,N顺序):
  x2 shape: [B, K, N]
  x2Scale shape: [B, K//64, N, 2]
  scaled_mm参数: 无b_trans
  计算路径: x1_slice [M,K] × x2[b,:,:] [K,N] → [M,N]

permX2 = [0, 2, 1] (N,K反序):
  x2 shape: [B, N, K]
  x2Scale shape: [B, N, K//64, 2]
  scaled_mm参数: b_trans=True, scale_b_trans=True
  计算路径: x1_slice [M,K] × x2[b,:,:] [N,K]^T → [M,N]
```

---

## 3. Tiling策略

### 3.1 Cube Tiling配置

**分块配置**：

| 轴 | Tile Shape | 说明 |
|----|-----------|------|
| M轴 | `[mL1, mL0]` 例如 `[256, 256]` | 左矩阵行维度分块 |
| K轴 | `[kL1, kL0]` 例如 `[128, 128]` | 公共维度分块 |
| N轴 | `[nL1, nL0]` 例如 `[256, 256]` | 右矩阵列维度分块 |

**分块原理**：
- **M轴分块**：M动态，tile shape覆盖典型M值范围
- **K轴分块**：K较小（128），tile shape覆盖完整K维度
- **N轴分块**：考虑L1/L2 cache容量和计算强度

### 3.2 Vector Tiling配置

**分块配置**：

| 配置 | Tile Shape | 说明 |
|------|-----------|------|
| vector_tile_shape | `[1, 128, 256, 32]` | Vector核4维分块配置 |

**分块原理**：
- 第一维：1（batch维度由loop处理）
- 第二维：128（M轴分块）
- 第三维：256（N轴分块）
- 第四维：32（辅助维度）

---

## 4. Loop结构设计

### 4.1 Parallel Loop实现

```python
# B轴并行循环
for b_idx in pypto.loop(B, name="LOOP_B", idx_name="b_idx", parallel=True):
    x1_slice = x1[:, b_idx, :]              # [M, K] — 从3D切出2D
    x1_scale_slice = x1Scale[:, b_idx, :, :] # [M, K//64, 2]

    if permX2 == [0, 1, 2]:
        mm_result = pypto.scaled_mm(
            x1_slice, x2[b_idx, :, :], out_dtype,
            x1_scale_slice, x2Scale[b_idx, :, :, :]
        )
    else:
        mm_result = pypto.scaled_mm(
            x1_slice, x2[b_idx, :, :], out_dtype,
            x1_scale_slice, x2Scale[b_idx, :, :, :],
            b_trans=True, scale_b_trans=True
        )

    out[:, b_idx, :] = mm_result             # 2D写回3D切片
```

**并行策略**：
- `parallel=True`：所有batch同时计算
- 每个batch独立计算，无数据依赖
- 3D→2D切片取出数据，2D→3D切片写回结果

### 4.2 3D→2D切片策略

**切片设计**：

| 操作 | 维度变化 | 说明 |
|------|---------|------|
| x1切片 | `[M,B,K] → [:,b_idx,:]` = `[M,K]` | 从3D取2D分片 |
| x1Scale切片 | `[M,B,K//64,2] → [:,b_idx,:,:]` = `[M,K//64,2]` | Scale同步切片 |
| x2切片 | `[B,K,N] → [b_idx,:,:]` = `[K,N]` | 从3D取2D分片 |
| x2Scale切片 | `[B,K//64,N,2] → [b_idx,:,:,:]` = `[K//64,N,2]` | Scale同步切片 |
| 结果写入 | `[M,N] → [:,b_idx,:]` | 2D写回3D位置 |

**设计优势**：
- 3D→2D切片让scaled_mm直接处理2D输入，符合API设计
- 2D→3D写回避免了维度不匹配报错
- 切片操作与parallel loop天然配合

---

## 5. 数据流设计

### 5.1 Wrapper预处理流程

```python
def transpose_quant_batch_matmul(inputs):
    x1 = inputs.x1
    x2 = inputs.x2
    x1Scale = inputs.x1Scale
    x2Scale = inputs.x2Scale
    tile_config = inputs.tile_config
    
    M, B, K = x1.shape
    N = x2.shape[-1] if permX2 == [0,1,2] else x2.shape[1]
    
    # 不permute，直接移动到NPU（kernel内部处理perm）
    x1 = x1.npu()
    x1Scale = x1Scale.npu()
    x2 = x2.npu()
    x2Scale = x2Scale.npu()
    
    # 构造输出tensor
    torch_out_dtype = torch.float16 if out_dtype == DT_FP16 else torch.bfloat16
    out_batch = torch.zeros(M, B, N, dtype=torch_out_dtype).npu()
    
    # 调用kernel
    transpose_quant_batch_mat_mul_kernel(x1, x2, x1Scale, x2Scale, out_batch, tile_config)
    
    return out_batch
```

**设计要点**：
- 输入不permute，直接.npu()移动——perm由scaled_mm的b_trans参数在硬件内处理
- 输出tensor预分配 `[M,B,N]`，kernel直接写入切片位置
- M值从shape[0]动态获取，不依赖tile_config.ori_shape

### 5.2 Golden计算流程

```python
# Step 1: FP8 → FP32
x1_fp32 = x1.float()
x2_fp32 = x2.float()

# Step 2: E8M0 Scale → FP32
x1_scale_fp32 = x1Scale.float()
x2_scale_fp32 = x2Scale.float()

# Step 3-4: permute
x1_perm = x1_fp32.permute(permX1)     # [M,B,K] → [B,M,K]
x2_perm = x2_fp32.permute(permX2)     # 按permX2处理

# Step 5: Scale reshape + broadcast (repeat_interleave 32x)
x1_scale_broadcast = ...  # [B, M, K] (每32元素共享1个scale)
x2_scale_broadcast = ...  # [B, K, N] 或 [B, N, K] (取决于permX2)

# Step 6-7: 反量化 + matmul
x1_dequant = x1_perm * x1_scale_broadcast
x2_dequant = x2_perm * x2_scale_broadcast
result = torch.matmul(x1_dequant, x2_dequant)  # [B, M, N]

# Step 8-9: 输出permute + dtype cast
output_perm = result.permute(permY)    # [B,M,N] → [M,B,N]
output = output_perm.half() 或 .bfloat16()  # dtype cast
```

---

## 6. 精度设计

### 6.1 MXFP8量化路径

```
输入精度: FP8E4M3/E5M2 (x1, x2) + FP8E8M0 (x1Scale, x2Scale)
     ↓
缩放精度: x1 × x1Scale → FP32 (隐式, scaled_mm内部)
          x2 × x2Scale → FP32 (隐式, scaled_mm内部)
     ↓
计算精度: FP32 (矩阵乘法, scaled_mm内部)
     ↓
输出精度: FP16 或 BF16 (由out_dtype参数决定)
```

### 6.2 精度保证措施

| 措施 | 说明 |
|------|------|
| scaled_mm内部FP32计算 | 矩阵乘法在FP32精度下进行 |
| MXFP8量化格式 | FP8E4M3/E5M2保证输入精度，FP8E8M0保证scale精度 |
| BF16输出低数据范围 | data_range=0.05（减少量化误差） |

### 6.3 精度验证标准

```python
RTOL = 1e-3
ATOL = 1e-3
numpy.testing.assert_allclose(result, golden, rtol=RTOL, atol=ATOL)
```

---

## 7. 性能优化设计

### 7.1 JIT配置

```python
@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1},
    pass_options={
        "auto_mix_partition": 1,
        "cube_l1_reuse_setting": {-1: 2},
        "cube_nbuffer_setting": {-1: 2},
        "vec_nbuffer_setting": {-2: 1, -1: 16},
    },
    runtime_options={"stitch_function_max_num": 512, "device_sched_mode": 0}
)
```

**参数说明**：
- `auto_mix_partition=1`：自动混合分区
- `cube_l1_reuse_setting={-1:2}`：Cube核最后一维L1复用2
- `cube_nbuffer_setting={-1:2}`：Cube核最后一维2缓冲
- `vec_nbuffer_setting={-2:1,-1:16}`：Vector核倒数第二维1缓冲、最后一维16缓冲
- `stitch_function_max_num=512`：函数拼接上限
- `device_sched_mode=0`：设备调度模式

### 7.2 Cache策略

```python
x1.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
x2.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
x1Scale.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
x2Scale.set_cache_policy(pypto.CachePolicy.NONE_CACHEABLE, True)
```

**NONE_CACHEABLE原理**：
- 输入数据一次性读取，无需缓存复用
- 减少L1缓存占用，为输出tensor腾出空间
- 对单次读取场景无性能损失

### 7.3 性能参考数据

| 配置 | M | K | N | B | 预估性能 |
|------|---|---|---|----|---------|
| test1 | 8 | 128 | 512 | 128 | ~108 us |
| test2 | 128 | 128 | 512 | 128 | ~122 us |
| test3 | 8192 | 128 | 512 | 128 | ~9583 us |
| test4 | 32768 | 128 | 512 | 128 | ~40635 us |

---

## 8. Kernel签名设计

### 8.1 ShapeConfig配置类

```python
@dataclass
class ShapeConfig:
    ori_shape: list          # [M, K, N]
    batch_size: int          # B轴大小
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

### 8.2 Kernel函数签名

```python
@pypto.frontend.jit(
    debug_options={"runtime_debug_mode": 1},
    pass_options={
        "auto_mix_partition": 1,
        "cube_l1_reuse_setting": {-1: 2},
        "cube_nbuffer_setting": {-1: 2},
        "vec_nbuffer_setting": {-2: 1, -1: 16},
    },
    runtime_options={"stitch_function_max_num": 512, "device_sched_mode": 0}
)
def transpose_quant_batch_mat_mul_kernel(
    x1: pypto.Tensor(),        # 左矩阵 [M, B, K] (FP8)
    x2: pypto.Tensor(),        # 右矩阵 [B, K, N] 或 [B, N, K] (FP8)
    x1Scale: pypto.Tensor(),   # 左矩阵缩放因子 [M, B, K//64, 2] (E8M0)
    x2Scale: pypto.Tensor(),   # 右矩阵缩放因子 (E8M0)
    out: pypto.Tensor(),       # 输出矩阵 [M, B, N]
    tile_config: ShapeConfig   # 配置参数
) -> None
```

---

## 9. 测试设计

### 9.1 测试矩阵

| 测试名称 | M | K | N | B | permX2 | in_dtype | out_dtype | 验证点 |
|---------|---|---|---|----|--------|----------|-----------|-------|
| test1 | 8 | 128 | 512 | 128 | [0,2,1] | FP8E5M2 | BF16 | 小M+b_trans |
| test2 | 128 | 128 | 512 | 128 | [0,1,2] | FP8E4M3 | FP16 | 中M+标准 |
| test3 | 8192 | 128 | 512 | 128 | [0,1,2] | FP8E4M3 | FP16 | 大M性能 |
| test4 | 32768 | 128 | 512 | 128 | [0,1,2] | FP8E4M3 | FP16 | 超大规模 |

### 9.2 精度验证方法

```python
# 对比方法
result_np = result.float().cpu().numpy()
golden_np = golden.float().cpu().numpy()
assert_allclose(result_np, golden_np, rtol=1e-3, atol=1e-3)
```

### 9.3 三态标记

- `[PRECISION_PASS]`：精度验证通过
- `[PRECISION_FAIL]`：精度验证失败

---

## 10. 计算复杂度分析

### 10.1 计算FLOPS

| 操作 | FLOPS | 说明 |
|------|-------|------|
| scaled_mm (每个batch) | 2 × M × K × N | 矩阵乘法 |
| 总scaled_mm (B个batch) | B × 2 × M × K × N | 所有batch |

**总 FLOPS**：
```
B × 2 × M × K × N
= 128 × 2 × M × 128 × 512
= 128 × M × 131072
```

### 10.2 内存访问量

| 操作 | 访问量 | 说明 |
|------|-------|------|
| 读 x1 | M × B × K × 1 byte | FP8数据 |
| 读 x2 | B × K × N × 1 byte | FP8数据 |
| 读 x1Scale | M × B × (K//64) × 2 × 1 byte | FP8E8M0 |
| 读 x2Scale | B × (K//64) × N × 2 × 1 byte | FP8E8M0 |
| 写 out | M × B × N × 2 bytes | FP16/BF16 |

---

## 11. 文件清单

| 文件 | 职责 |
|------|------|
| `SPEC.md` | 需求规格 |
| `API_REPORT.md` | API映射报告 |
| `DESIGN.md` | 本设计文档 |
| `README.md` | 使用说明 |
| `transpose_quant_batch_matmul_impl.py` | Kernel实现 |
| `transpose_quant_batch_matmul_golden.py` | Golden参考实现 |
| `test_transpose_quant_batch_matmul.py` | 测试文件 |

---

## 12. 参考文档

- [PyPTO编程指南](../../../docs/pypto_programming_guide.md)
- [MXFP8量化规范](../../../docs/tutorials/quantization/)
- [性能优化指南](../../../docs/tutorials/debug/performance.md)