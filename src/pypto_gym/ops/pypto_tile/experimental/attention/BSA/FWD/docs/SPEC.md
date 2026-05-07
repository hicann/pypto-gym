# BSA Forward 需求规格

## 算子概述

### 名称

`aclnnBlockSparseAttention`

### 功能描述

BSA Forward 算子实现块稀疏注意力的前向计算。通过块级稀疏掩码仅计算有效块对的注意力，同时兼容 Flash Attention 的 IO 感知优化与数值稳定机制。采用 Online Softmax 分块迭代策略，逐 KV 块维护运行最大值和运行指数和，跨块执行全局数值校正，保证分块计算与全量计算数学完全等价。

### 应用场景

- 视频生成（如字节 Seedance2.0）
- 长上下文 LLM 推理
- 高分辨率 ViT
- 任何需要长序列注意力且可利用稀疏性的场景

---

## 数学公式

### 前向计算公式

$$S_{uv} = \frac{Q_u K_v^\top}{\sqrt{d}}, \quad M[u][v] = 1$$

$$l_{u,i} = \text{LSE over all valid } v \text{ of } S_{uv}[i,:]$$

$$P_{uv} = \exp(S_{uv} - l_u \cdot \mathbf{1}^\top), \quad M[u][v] = 1$$

$$O_u = \sum_{v:M[u][v]=1} P_{uv} V_v$$

### Online Softmax 迭代过程

对每个 Q 块 $u$，沿 KV 方向逐块迭代（仅遍历掩码为 1 的 KV 块），设当前为第 $j$ 个有效 KV 块：

$$m_j = \max(m_{j-1},\ \text{rowmax}(S_j / \sqrt{d}))$$

$$l_j = l_{j-1} \cdot \exp(m_{j-1} - m_j) + \text{rowsum}(\exp(S_j / \sqrt{d} - m_j))$$

$$\tilde{O}_j = \tilde{O}_{j-1} \cdot \exp(m_{j-1} - m_j) + \exp(S_j / \sqrt{d} - m_j) \cdot V_j$$

**最终输出**：

$$O_u = \tilde{O}_J / l_J$$

$$\text{LSE}_u = m_J + \log(l_J)$$

其中：
- $Q_u$ 为按 `blockShapeX` 划分的 Q 块，$K_v$ / $V_v$ 为按 `blockShapeY` 划分的 K/V 块
- $M \in \{0,1\}^{B \times H_q \times B_q \times B_k}$ 为块稀疏掩码
- $d = 128$（固定 head_dim）
- $m_j$、$l_j$、$\tilde{O}_j$ 分别为第 $j$ 步的运行最大值、运行指数和、运行加权 V 和
- $J$ 为最后一个有效 KV 块的索引

---

## 输入输出规格

### 输入张量

| 序号 | 名称 | 数据类型 | 形状 | 必选 | 说明 |
|------|------|----------|------|------|------|
| 0 | query | FP16 | `[B, H_q, Sq, 128]` | 是 | Query 张量，BNSD 布局 |
| 1 | key | FP16 | `[B, H_kv, Skv, 128]` | 是 | Key 张量，BNSD 布局 |
| 2 | value | FP16 | `[B, H_kv, Skv, 128]` | 是 | Value 张量，BNSD 布局 |
| 3 | blockSparseMask | BOOL/UINT8 | `[B, H_q, ceil(Sq/X), ceil(Skv/Y)]` | 是 | 块稀疏掩码，1=有效块对 |
| 4 | actualSeqLengths | INT64 | `[B]` | 否 | 每 batch 的 Q 实际序列长度，默认为 Sq |
| 5 | actualSeqLengthsKv | INT64 | `[B]` | 否 | 每 batch 的 KV 实际序列长度，默认为 Skv |
| 6 | blockShape | INT64 | `[2]` | 否 | `[blockShapeX, blockShapeY]`，默认 `[256, 512]` |

### 输出张量

| 序号 | 名称 | 数据类型 | 形状 | 说明 |
|------|------|----------|------|------|
| 0 | attentionOut | FP16 | `[B, H_q, Sq, 128]` | 注意力输出，与 Q 同 Shape |
| 1 | softmaxLse | FLOAT32 | `[B, H_q, Sq]` | 行级 Log-Sum-Exp 值 |

### 算子属性

| 名称 | 类型 | 约束 | 说明 |
|------|------|------|------|
| numKeyValueHeads | INT | = H_kv | KV 头数 |
| scaleValue | FLOAT | = 1/√128 ≈ 0.0884 | 缩放因子 |
| maskType | INT | = 0 | 掩码类型（无额外掩码） |

---

## Shape 范围与约束

### 参数范围

| 参数 | 范围/约束 | 典型值 |
|------|-----------|--------|
| head_dim (D) | **128**（固定） | 128 |
| blockShapeX | **64 的整数倍**（最小 64） | 256 |
| blockShapeY | **≥128 且为 64 的整数倍** | 512 |
| H_q | ≥ H_kv，H_q % H_kv == 0 | 4, 8, 32 |
| H_kv | ≥ 1 | 1, 2, 4, 8 |
| Sq | ≥ 1 | 256, 512, 1024, 2048 |
| Skv | ≥ 1 | 256, 512, 1024, 2048 |
| B | ≥ 1 | 1, 2 |

### 严格约束

1. **数据类型**：Q/K/V/O 必须为 **FP16**；softmaxLse 必须为 **FP32**；blockSparseMask 为 BOOL/UINT8
2. **数据布局**：统一使用 **BNSD** 布局，张量内存布局必须为紧凑布局，禁止非连续步长张量
3. **GQA 头数**：H_q ≥ H_kv，且 H_q % H_kv == 0
4. **序列长度**：支持 Q/KV 序列长度非 blockShape 整数倍，分块数自动向上取整
5. **维度对齐硬约束**：head_dim = 128（固定），blockShapeX 为 64 的整数倍，blockShapeY 最小为 128 且为 64 的整数倍
6. **稀疏掩码 Shape**：`[B, H_q, ceil(Sq / blockShapeX), ceil(Skv / blockShapeY)]`
7. **预留参数**：`attenMaskOptional` 传入 `nullptr`，`maskType` = 0

### GQA 兼容说明

设 Q 头数为 H_q，KV 头数为 H_kv，分组数 G = H_q / H_kv。每 G 个 Q 头共享 1 个 KV头：
- `kvHeadIdx = qHeadIdx / groupSize`
- 同一组内的 Q 头共享同一份 KV 数据

### 典型配置

| 场景 | B | Hq | Hkv | Sq | Skv | 稀疏率 |
|------|---|-----|-----|-----|-----|--------|
| 基础 MHA | 1 | 4 | 4 | 256 | 256 | 50% |
| GQA group4 | 1 | 8 | 2 | 256 | 512 | 40% |
| GQA 大头 | 1 | 32 | 8 | 256 | 256 | 50% |
| 长序列 | 1 | 8 | 1 | 1024 | 1024 | 30% |
| Dense | 1 | 4 | 4 | 256 | 256 | 100% |
| 多 Batch | 2 | 4 | 2 | 256 | 512 | 50% |
| 超长序列 | 1 | 4 | 2 | 2048 | 2048 | 30% |

---

## 精度要求

### 容差标准

| 输出 | atol | rtol | 说明 |
|------|------|------|------|
| attentionOut (O) | 0.0001 | 0.0078125 | 前向注意力输出 |
| softmaxLse | 0.0001 | 0.0078125 | 前向 LSE 值 |

### 精度保障机制

1. **FP32 累积**：所有中间计算（QK^T、softmax max/sum/exp、PV、O 累积）在 FP32 精度下完成
2. **Online Softmax 数值稳定**：减去行最大值后再 exp，避免数值溢出
3. **Large negative masking**：无效 KV 块位置使用 `-65504.0`（FP16 最小值）使 exp 下溢为零
4. **累积器精度**：`mi_update`、`li_update`、`oi_update` 均为 FP32

### 典型实测精度

| 输出 | 最大误差 | 来源 |
|------|---------|------|
| O max_diff | ≤ 0.000122 | FP16 矩阵乘累积误差 |
| LSE max_diff | ≤ 0.000001 | FP32 在线 softmax 数值稳定 |

---

## 性能要求

### 目标平台

- **硬件**：昇腾 910B (Atlas A2)
- **CANN**：9.0.0+
- **数据类型**：FP16 (Q/K/V/O), FP32 (softmaxLse)

### 性能基准（MHA Dense, B=1, Hq=Hkv=4）

| Shape | Task Time | AICore Time | AICore Util |
|-------|-----------|-------------|-------------|
| 256×256 | ~75 us | ~1.3 ms | ~37% |
| 512×512 | ~92 us | ~2.6 ms | ~43% |
| 1024×1024 | ~180 us | ~6.5 ms | ~60% |

### 性能优化措施

| 优化项 | 说明 |
|--------|------|
| Dense/Sparse 双路径 | 掩码全 1 时无 mask 开销 |
| 紧凑 KV 策略 | 内层循环仅遍历有效 KV 块 |
| L1 缓存复用 | Q 块在内层循环复用 |
| Kernel 缓存 | 首次编译后缓存复用 |

---

## 测试验证要求

### 测试覆盖

| 维度 | 覆盖范围 |
|------|---------|
| 序列长度 | 256, 512, 1024, 2048 |
| GQA 分组 | Hq:Hkv = 4:4(MHA), 8:2, 32:8, 8:1, 4:2 |
| 稀疏率 | 30%, 40%, 50%, 70%, 100%(dense) |
| Batch | 1, 2 |
| 路径 | Sparse, Dense |
| head_dim | 128（固定） |

### 测试用例清单

| # | 测试名 | B | Hq | Hkv | Sq | Skv | 稀疏率 | 路径 |
|---|--------|---|-----|-----|-----|-----|--------|------|
| 1 | Basic Sparse50 | 1 | 4 | 2 | 256 | 256 | 50% | Sparse |
| 2 | GQA group4 | 1 | 8 | 2 | 256 | 512 | 40% | Sparse |
| 3 | GQA Hq32 Hkv8 | 1 | 32 | 8 | 256 | 256 | 50% | Sparse |
| 4 | Long Seq S1024 | 1 | 8 | 1 | 1024 | 1024 | 30% | Sparse |
| 5 | Sparse30% | 1 | 4 | 4 | 512 | 512 | 30% | Sparse |
| 6 | Sparse70% | 1 | 4 | 4 | 512 | 512 | 70% | Sparse |
| 7 | Dense 100% | 1 | 4 | 4 | 256 | 256 | 100% | Dense |
| 8 | Batch2 | 2 | 4 | 2 | 256 | 512 | 50% | Sparse |
| 9 | NonAligned | 1 | 4 | 2 | 300 | 300 | 50% | Sparse |
| 10 | Long Seq S2048 | 1 | 4 | 2 | 2048 | 2048 | 30% | Sparse |

### 精度验证方法

对每个测试用例，比较 PyPTO 实现输出与 Golden 参考实现输出：

```python
torch.testing.assert_close(impl_out, golden_out, atol=0.0001, rtol=0.0078125)
torch.testing.assert_close(impl_lse, golden_lse, atol=0.0001, rtol=0.0078125)
```

### 已知限制

- **非对齐序列**：当 Sq 或 Skv 不是 blockShape 整数倍时，零填充行在 softmax 中可能获得非零权重，导致精度偏差。根治方案需在 kernel 中实现边界块的 valid_shape 裁剪
