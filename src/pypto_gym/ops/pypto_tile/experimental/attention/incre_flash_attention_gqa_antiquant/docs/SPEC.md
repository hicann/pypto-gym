# Incre Flash Attention GQA Anti-Quant 算子规格文档

## 1. 算子概述

**算子名称**：`incre_flash_attention_gqa_antiquant`  
**功能描述**：增量式 Incre Flash Attention with Grouped Query Attention (GQA) 和反量化 (Anti-Quantization)  
**应用场景**：大语言模型自回归生成场景，支持 Paged KV Cache、分组查询注意力及 KV Cache 反量化

### 1.1 核心特性

- **增量式计算**：适用于自回归生成，每次只计算新增 token 的注意力
- **Paged KV Cache**：支持非连续内存访问，提高内存利用率
- **分组查询注意力 (GQA)**：支持 query heads 和 kv heads 数量不同的场景
- **反量化 (Anti-Quantization)**：支持 FP8 格式的 KV Cache，通过反量化缩放因子还原为 BF16 进行计算
- **Flash Attention**：采用在线 Softmax 算法，降低内存开销

## 2. 数学原理

### 2.1 标准注意力计算

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

### 2.2 GQA 注意力计算

对于 GQA，query heads 分组共享 KV heads：

```
group = n1 // n2
Q_i ∈ [b, n1, s1, d]  →  按 group 分组
K_j ∈ [b, n2, s2, d]  →  每个 KV head 被多个 query head 组共享
V_j ∈ [b, n2, s2, d]

对于第 i 个 query head，对应的 KV head 为 j = i // group
Attention_i = softmax(Q_i × K_j^T / √d) × V_j
```

### 2.3 反量化计算

KV Cache 以 FP8 格式存储，计算前需要通过反量化还原为 BF16。为保证精度，反量化在 FP32 下执行乘法：

```
步骤:
  1) 将 FP8 KV 数据 cast 为 FP32
  2) 按 n2_idx 取出当前 KV head 的 BF16 缩放因子，并 cast 为 FP32
  3) FP32 × FP32 逐元素相乘（缩放因子沿 sequence 维度广播）
  4) 将 FP32 结果 cast 为 BF16，作为 matmul 输入

数学表达式:
  K_dequant[s, d] = cast_to_fp32(K_fp8[s, d]) × cast_to_fp32(key_antiquant_scale[n2_idx, d])
  V_dequant[s, d] = cast_to_fp32(V_fp8[s, d]) × cast_to_fp32(value_antiquant_scale[n2_idx, d])

其中:
  key_antiquant_scale:   [n2, d]  (BF16)，按 n2_idx 取 [d] 向量后广播至 [s2_tile, d]
  value_antiquant_scale: [n2, d]  (BF16)，按 n2_idx 取 [d] 向量后广播至 [s2_tile, d]
```

### 2.4 在线 Softmax 算法

采用 Flash Attention 的在线 Softmax 算法，分块计算注意力：

对于每个块 i：
```
m_i = max(score_i)
l_i = sum(exp(score_i - m_i))
m_new = max(m_old, m_i)
l_new = exp(m_old - m_new) * l_old + exp(m_i - m_new) * l_i
output_new = (exp(m_old - m_new) * output_old + exp(m_i - m_new) * output_i) / l_new
```

## 3. 输入输出规格

### 3.1 输入参数

| 参数名 | 数据类型 | 形状 | 说明 |
|--------|----------|------|------|
| query | DT_BF16 | [b, n1, s1, d] | 查询张量 (BNSD 布局) |
| key | DT_FP8E4M3 | [block_num, n2, block_size, d] | Key Cache (Paged 格式，FP8 存储) |
| value | DT_FP8E4M3 | [block_num, n2, block_size, d] | Value Cache (Paged 格式，FP8 存储) |
| key_antiquant_scale | DT_BF16 | [n2, d] | Key 反量化缩放因子 |
| value_antiquant_scale | DT_BF16 | [n2, d] | Value 反量化缩放因子 |
| kv_actual_seqs | DT_INT32 | [b] | 每个批次的实际 KV 序列长度 |
| block_table | DT_INT32 | [b, max_blocks_per_query] | 块表，映射逻辑块到物理块索引 |

### 3.2 配置参数

#### 3.2.1 IFAKernelCfg

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| n1 | int | 查询头数量 |
| d | int | 头维度 |
| block_num | int | 总 KV 块数 |
| n2 | int | Key/Value 头数量 |
| block_size | int | 每个块的大小 |
| b | int | 批次大小 |
| s1 | int | 查询序列长度 |
| group | int | 分组数 (n1 // n2) |
| softmax_scale | float | Softmax 缩放因子 (d^(-0.5)) |

#### 3.2.2 AttentionTileConfig

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| g_tile | int | 组维度切分大小 (group) |
| s2_tile | int | KV 序列维度切分大小 (2048) |
| c1_tile | list | C1 阶段（Q×K^T）切分配置 |
| v1_tile | list | V1 阶段（反量化 + Softmax）切分配置 |
| c2_tile | list | C2 阶段（Softmax×V）切分配置 |
| v2_tile | list | V2 阶段（输出更新）切分配置 |

### 3.3 输出参数

| 参数名 | 数据类型 | 形状 | 说明 |
|--------|----------|------|------|
| atten_out | DT_BF16 | [b, n1, s1, d] | 注意力输出 (BNSD 布局) |

## 4. 约束条件

### 4.1 数据类型约束

- `query`: 必须为 DT_BF16
- `key`, `value`: 必须为 DT_FP8E4M3（量化存储的 KV Cache）
- `key_antiquant_scale`, `value_antiquant_scale`: 必须为 DT_BF16
- `kv_actual_seqs`, `block_table`: 必须为 DT_INT32
- 中间计算使用 DT_FP32 保证精度
- 反量化计算流程：FP8 → cast(FP32)，BF16_scale[n2_idx] → cast(FP32)，FP32 × FP32 → cast(BF16)，scale 按 n2_idx 取值后沿 sequence 维度广播

### 4.2 形状约束

- `n1` 必须能被 `n2` 整除（分组查询注意力）
- `s1` 通常为 1~3（增量生成场景）
- `block_size` 为 128
- `s2_tile` (2048) 应能被 `block_size` (128) 整除
- `d` 支持 128

### 4.3 内存约束

- Block Table 中的无效索引用 -1 表示，计算时使用 `.max(0)` 处理
- 需要足够的 workspace 内存存储中间结果（out_update, sum_update, max_update）
- KV Cache 以 FP8 存储，反量化后转为 BF16 进行计算

### 4.4 性能约束

- 建议使用 Tiling 策略优化内存访问
- 支持多核并行计算 (`device_sched_mode=1`)
- 支持流水线优化 (`vec_nbuffer_setting`, `cube_l1_reuse_setting`)
- 内层 S2 循环支持展开 (`unroll_list=[8, 1]`)

## 5. 实现细节

### 5.1 计算流程

1. **初始化阶段**：从输入张量提取计算参数 (`IFAKernelCfg`)，获取 Tiling 配置
2. **数据预处理**：将 Q, K, V reshape 为 2D 格式
3. **反量化阶段**：
   - 按 n2_idx 从 `[n2, d]` 的 scale 张量取出当前 KV head 的 `[d]` 向量
   - 将 FP8 K/V 组装缓冲区 cast 为 FP32
   - 将 BF16 scale cast 为 FP32
   - FP32 × FP32 乘法（scale 沿 sequence 广播），结果 cast 为 BF16
4. **C1 阶段**：计算 Q × K_dequant^T，得到注意力分数
5. **V1 阶段**：在线 Softmax 计算（缩放、最大值、指数和）
6. **C2 阶段**：计算 Softmax × V_dequant
7. **V2 阶段**：在线 Softmax 输出更新和最终归一化

### 5.2 循环结构

```
for b_idx in batch_size:
    for s1_idx in s1:
        cur_seq_len = kv_actual_seqs[b_idx] - (s1 - 1 - s1_idx)
        for n2_idx in n2:
            for group_idx in (n1 // n2) // g_tile:
                初始化 online softmax 状态
                组装 qi (query 组)
                for s2_idx in s2_loop:
                    组装 K/V block (反量化)
                    [C1 -> V1 -> C2 -> V2]
                    最终归一化并输出
```

### 5.3 Tiling 策略

- **Group Tiling**：将 query heads 按 group 分组处理，g_tile = group
- **Sequence Tiling**：将 KV 序列分块处理，s2_tile = 2048
- **Vector/Cube Tiling**：
  - c1_tile = [[128, 128], [128, 128], [128, 128]]  (Q×K^T)
  - v1_tile = [128, 2048]  (反量化 + Softmax)
  - c2_tile = [[128, 128], [128, 128], [128, 128]]  (Softmax×V)
  - v2_tile = [128, 128]  (输出更新)

## 6. 测试用例

### 6.1 测试场景

| 测试用例名称 | 批次大小 | 查询头数 | KV头数 | KV序列长度 | 头维度 | 说明 |
|-------------|----------|----------|--------|-----------|--------|------|
| test_1b2k | 1 | 8 | 1 | 2048 | 128 | 小批次短序列 |
| test_8b2kqs2 | 8 | 8 | 1 | 2048 | 128 | 中等批次，s1=2 |
| test_16b4kqs3 | 16 | 8 | 1 | 4096 | 128 | 大批次，s1=3 |
| test_32b8k_d256 | 32 | 8 | 1 | 8192 | 256 | 大批次长序列，d=256 |
| test_64b2k_kvn2 | 64 | 8 | 2 | 2048 | 128 | 大批次，n2=2 |
| test_4b16k | 4 | 64 | 8 | 16384 | 128 | 小批次长序列，GQA |
| test_64b16k | 64 | 64 | 8 | 16384 | 128 | 大批次长序列，GQA |

### 6.2 精度要求

- 绝对容差 (atol): 0.0001
- 相对容差 (rtol): 0.0078125 (1/128)
- 最大误差比例: 0.5%

## 7. 性能优化建议

### 7.1 内存优化

- 使用 Paged KV Cache 减少内存碎片
- KV Cache 以 FP8 存储，减少内存占用
- 采用 Flash Attention 算法降低内存占用
- 优化 Tiling 策略平衡计算和内存访问

### 7.2 计算优化

- 利用分组查询注意力减少计算量
- 使用在线 Softmax 避免存储完整注意力矩阵
- 反量化在计算前即时执行，避免额外存储开销
- 优化块大小和切分策略

### 7.3 并行优化

- 利用多核并行计算 (`device_sched_mode=1`)
- 优化流水线减少等待时间 (`vec_nbuffer_setting`, `cube_l1_reuse_setting`)
- 内层循环展开优化 (`unroll_list=[8, 1]`)

## 8. 使用示例

```python
import torch
from incre_flash_attention_gqa_antiquant_impl import incre_flash_attention_gqa_antiquant

query = torch.randn(b, n1, s1, d, dtype=torch.bfloat16)
key_cache = torch.randn(block_num, n2, block_size, d, dtype=torch.float8_e4m3fn)
value_cache = torch.randn(block_num, n2, block_size, d, dtype=torch.float8_e4m3fn)
key_antiquant_scale = torch.randn(n2, d, dtype=torch.bfloat16)
value_antiquant_scale = torch.randn(n2, d, dtype=torch.bfloat16)
kv_actual_seqs = torch.tensor([s2] * b, dtype=torch.int32)
block_table = gen_block_table(...)

output = incre_flash_attention_gqa_antiquant(
    query=query,
    key=key_cache,
    value=value_cache,
    key_antiquant_scale=key_antiquant_scale,
    value_antiquant_scale=value_antiquant_scale,
    kv_actual_seqs=kv_actual_seqs,
    block_table=block_table,
)
```

## 9. 参考资料

- Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- Multi-Query Attention / Grouped Query Attention
- PagedAttention: Efficient Memory Management for LLM Inference
- FP8 Quantization for KV Cache Optimization