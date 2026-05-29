# Incre Flash Attention MLA 算子规格文档

## 1. 算子概述

**算子名称**：`incre_flash_attention_mla`  
**功能描述**：增量式 Incre Flash Attention for Multi-Head Latent Attention (MLA)  
**应用场景**：大语言模型自回归生成场景，支持 Paged KV Cache 和分组查询注意力 (GQA)

### 1.1 核心特性

- **增量式计算**：适用于自回归生成，每次只计算新增 token 的注意力
- **Paged KV Cache**：支持非连续内存访问，提高内存利用率
- **分组查询注意力 (GQA)**：支持 query heads 和 kv heads 数量不同的场景
- **RoPE 支持**：支持旋转位置编码 (Rotary Position Embedding)
- **Flash Attention**：采用在线 Softmax 算法，降低内存开销

## 2. 数学原理

### 2.1 标准注意力计算

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

### 2.2 MLA 注意力计算

对于 MLA，需要将 query 和 key 分为非旋转部分和旋转部分：

```
Q_full = [Q_nope, Q_rope]
K_full = [K_nope, K_rope]

Attention = softmax(Q_full × K_full^T / √d) × V
```

### 2.3 在线 Softmax 算法

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
| query | DT_BF16 | [batch_size, s1, n1, q_d] | 查询张量（非旋转部分） |
| key | DT_BF16 | [block_num, n2, block_size, kv_d] | Key Cache (Paged 格式) |
| value | DT_BF16 | [block_num, n2, block_size, kv_d] | Value Cache (Paged 格式) |
| query_rope | DT_BF16 | [batch_size, s1, n1, q_rope_d] | 查询旋转位置编码部分 |
| key_rope | DT_BF16 | [block_num, n2, block_size, k_rope_d] | Key 旋转位置编码 Cache |
| kv_actual_seqs | DT_INT32 | [batch_size] | 每个批次的实际 KV 序列长度 |
| block_table | DT_INT32 | [batch_size, max_blocks_per_query] | 块表，映射逻辑块到物理块索引 |

### 3.2 配置参数

#### 3.2.1 MlaConfig

| 参数名 | 数据类型 | 默认值 | 说明 |
|--------|----------|--------|------|
| layout | str | "BSND" | 输入张量布局 |
| b | int | 32 | 批次大小 |
| n1 | int | 128 | 查询头数量 |
| s1 | int | 1 | 查询序列长度（增量生成通常为 1） |
| q_d | int | 512 | 查询头维度（非旋转部分） |
| q_rope_d | int | 64 | 查询旋转位置编码维度 |
| n2 | int | 1 | Key/Value 头数量 |
| s2 | int | 4096 | Key/Value 序列长度（最大值） |
| kv_d | int | 512 | Key/Value 头维度 |
| k_rope_d | int | 64 | Key 旋转位置编码维度 |
| block_size | int | 128 | Paged Cache 块大小 |
| softmax_scale | float | 576^(-0.5) | Softmax 缩放因子 |

#### 3.2.2 AttentionTileConfig

| 参数名 | 数据类型 | 说明 |
|--------|----------|------|
| g_tile | int | 组维度切分大小 |
| s2_tile | int | KV 序列维度切分大小 |
| v0_tile | list | V0 阶段（组装计算）切分配置 |
| c1_tile | list | C1 阶段（Q×K^T）切分配置 |
| v1_tile | list | V1 阶段（Softmax）切分配置 |
| c2_tile | list | C2 阶段（Softmax×V）切分配置 |
| v2_tile | list | V2 阶段（输出更新）切分配置 |
| v2_update_tile | list | V2 更新阶段切分配置 |

### 3.3 输出参数

| 参数名 | 数据类型 | 形状 | 说明 |
|--------|----------|------|------|
| attention_output | DT_BF16 | [batch_size, s1, n1, q_d] | 注意力输出 |

## 4. 约束条件

### 4.1 数据类型约束

- `query`, `key`, `value`, `query_rope`, `key_rope`: 必须为 DT_BF16
- `kv_actual_seqs`, `block_table`: 必须为 DT_INT32
- 中间计算使用 DT_FP32 保证精度

### 4.2 形状约束

- `n1` 必须能被 `n2` 整除（分组查询注意力）
- `s1` 通常为 1（增量生成场景）
- `q_d + q_rope_d` = `kv_d + k_rope_d`（完整查询/键维度）
- `block_size` 128
- `s2_tile` 应能被 `block_size` 整除

### 4.3 内存约束

- Key Cache 和 Value Cache 共享同一内存（value = key）
- Block Table 中的无效索引用 -1 表示
- 需要足够的 workspace 内存存储中间结果

### 4.4 性能约束

- 建议使用 Tiling 策略优化内存访问
- 支持多核并行计算
- 支持流水线优化

## 5. 实现细节

### 5.1 计算流程

1. **V0 阶段 (MLA_V0)**：组装 Key 和 Key_Rope
   - 根据 block_table 从 Paged Cache 中提取 Key 和 Key_Rope
   - 将两部分拼接成完整的 Key 向量

2. **C1 阶段 (MLA_C1)**：计算注意力分数
   - 组装 Query 和 Query_Rope
   - 计算 Q × K^T

3. **V1 阶段 (MLA_V1)**：在线 Softmax
   - 缩放注意力分数
   - 计算局部最大值和指数和
   - 存储中间状态用于后续更新

4. **C2 阶段 (MLA_C2)**：加权值聚合
   - 计算 Softmax × V

5. **V2 阶段 (MLA_V2)**：输出更新
   - 使用在线 Softmax 算法更新输出
   - 处理最后一个块或中间块的不同逻辑

### 5.2 循环结构

```
for batch_idx in batch_size:
    for s1_idx in s1:
        for n2_idx in kv_heads:
            for group_idx in (query_heads // kv_heads):
                for s2_idx in s2_tiles:
                    [V0 -> C1 -> V1 -> C2 -> V2]
```

### 5.3 Tiling 策略

- **Group Tiling**：将 query heads 分组处理，降低内存压力
- **Sequence Tiling**：将 KV 序列分块处理，支持长序列
- **Vector Tiling**：优化向量操作的切分策略
- **Cube Tiling**：优化矩阵乘法的切分策略

## 6. 测试用例

### 6.1 测试场景

| 测试用例名称 | 批次大小 | 查询头数 | KV序列长度 | 说明 |
|-------------|----------|----------|-----------|------|
| test_incre_flash_attention_mla_1b4k | 1 | 128 | 4096 | 小批次长序列 |
| test_incre_flash_attention_mla_8b4k | 8 | 128 | 4096 | 中等批次 |
| test_incre_flash_attention_mla_16b8k | 16 | 128 | 8192 | 大批次长序列 |
| test_incre_flash_attention_mla_32b2k | 32 | 128 | 2048 | 大批次短序列 |
| test_incre_flash_attention_mla_32b4k | 32 | 128 | 4096 | 大批次中等序列 |

### 6.2 精度要求

- 绝对容差 (atol): 0.0001
- 相对容差 (rtol): 0.0078125 (1/128)
- 最大误差比例: 0.5%

## 7. 性能优化建议

### 7.1 内存优化

- 使用 Paged KV Cache 减少内存碎片
- 采用 Flash Attention 算法降低内存占用
- 优化 Tiling 策略平衡计算和内存访问

### 7.2 计算优化

- 利用分组查询注意力减少计算量
- 使用在线 Softmax 避免存储完整注意力矩阵
- 优化块大小和切分策略

### 7.3 并行优化

- 利用多核并行计算
- 优化流水线减少等待时间
- 使用异步数据传输

## 8. 使用示例

```python
import torch
from incre_flash_attention_mla_impl import incre_flash_attention_mla

# 配置参数
config = MlaConfig(
    b=32, n1=128, s1=1, q_d=512, q_rope_d=64,
    n2=1, s2=4096, kv_d=512, k_rope_d=64,
    block_size=128, softmax_scale=576**-0.5
)

tile_config = AttentionTileConfig(
    g_tile=128, s2_tile=2048,
    v0_tile=[128, 576],
    c1_tile=[[128, 128], [128, 128], [128, 128]],
    v1_tile=[8, 2048],
    c2_tile=[[128, 128], [128, 128], [128, 128]],
    v2_tile=[64, 512],
    v2_update_tile=[32, 512]
)

# 执行计算
output = incre_flash_attention_mla(
    query=query,
    key=key_cache,
    value=value_cache,
    query_rope=query_rope,
    key_rope=key_rope_cache,
    kv_actual_seqs=kv_actual_seqs,
    block_table=block_table,
    kernel_config=config,
    tile_config=tile_config
)
```

## 9. 参考资料

- Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- Multi-Query Attention / Grouped Query Attention
- PagedAttention: Efficient Memory Management for LLM Inference