# BSA Backward 需求规格

## 1. 算子概述

- **算子名称**：`aclnnBlockSparseAttentionGrad`
- **功能描述**：计算块稀疏注意力（Block Sparse Attention）的反向传播，输出 dQ/dK/dV 梯度
- **应用场景**：视频生成（如 Seedance2.0）、长上下文 LLM、高分辨率 ViT 等需要长序列稀疏注意力的场景
- **实现策略**：基于前向保存的 O 和 softmaxLse 重计算 S 和 P，拆分为 dQ kernel + dK/dV kernel 两个独立 kernel

---

## 2. 数学公式

### 2.1 softmaxGrad

$$D_i = \sum_{t=1}^{d} dO_{i,t} \cdot O_{i,t}$$

对每行 i，计算 dO 和 O 的逐元素乘积沿 head_dim 维度的求和。

### 2.2 dP 计算

$$dP_{uv} = dO_u \cdot V_v^\top$$

### 2.3 P 重计算

$$P_{uv} = \exp\left(\frac{S_{uv}}{\sqrt{d}} - \text{LSE}_u\right)$$

其中 $S_{uv} = Q_u K_v^\top$，$\text{LSE}_u$ 为前向传播保存的 Log-Sum-Exp 值。

### 2.4 dS 计算

$$dS_{uv} = P_{uv} \odot \left(dP_{uv} - D_u \cdot \mathbf{1}^\top\right)$$

### 2.5 梯度累积

$$dQ_u \mathrel{+}= \frac{1}{\sqrt{d}} \cdot dS_{uv} \cdot K_v, \quad \forall v: M[u][v] = 1$$

$$dK_v \mathrel{+}= \frac{1}{\sqrt{d}} \cdot dS_{uv}^\top \cdot Q_u, \quad \forall u: M[u][v] = 1$$

$$dV_v \mathrel{+}= P_{uv}^\top \cdot dO_u, \quad \forall u: M[u][v] = 1$$

**关键细节**：
- dQ 和 dK 需乘 scaleValue ($1/\sqrt{d}$)，dV 不需要（P 已包含 scale 效果）
- 梯度使用 FP32 精度累加，最终 cast 到 FP16 输出
- D_row（softmaxGrad）必须在 FP32 精度下计算

---

## 3. 输入输出规格

### 3.1 输入张量

| 序号 | 名称 | 数据类型 | 形状 | 必选 | 说明 |
|------|------|----------|------|------|------|
| 0 | dout | FP16 | `[B, H_q, Sq, 128]` | 是 | 来自上层的梯度 |
| 1 | query | FP16 | `[B, H_q, Sq, 128]` | 是 | 前向 Query 张量，BNSD 布局 |
| 2 | key | FP16 | `[B, H_kv, Skv, 128]` | 是 | 前向 Key 张量，BNSD 布局 |
| 3 | value | FP16 | `[B, H_kv, Skv, 128]` | 是 | 前向 Value 张量，BNSD 布局 |
| 4 | attentionOut | FP16 | `[B, H_q, Sq, 128]` | 是 | 前向注意力输出 O |
| 5 | softmaxLse | FP32 | `[B, H_q, Sq]` | 是 | 前向 softmax LSE 值 |
| 6 | blockSparseMask | BOOL/UINT8 | `[B, H_q, ceil(Sq/bx), ceil(Skv/by)]` | 是 | 块稀疏掩码，1=有效块对 |
| 7 | blockShape | INT64 | `[2]` | 否 | `[blockShapeX, blockShapeY]`，默认 [256, 512] |
| 8 | actualSeqLengths | INT64 | `[B]` | 否 | 每batch的Q实际序列长度 |
| 9 | actualSeqLengthsKv | INT64 | `[B]` | 否 | 每batch的KV实际序列长度 |

### 3.2 输出张量

| 序号 | 名称 | 数据类型 | 形状 | 说明 |
|------|------|----------|------|------|
| 0 | dq | FP16 | `[B, H_q, Sq, 128]` | Query 梯度，与 Q 同 shape |
| 1 | dk | FP16 | `[B, H_kv, Skv, 128]` | Key 梯度，与 K 同 shape |
| 2 | dv | FP16 | `[B, H_kv, Skv, 128]` | Value 梯度，与 V 同 shape |

### 3.3 算子属性

| 名称 | 类型 | 约束 | 说明 |
|------|------|------|------|
| qInputLayout | INT | = 1 (BNSD) | Q 输入布局 |
| kvInputLayout | INT | = 1 (BNSD) | KV 输入布局 |
| numKeyValueHeads | INT | = H_kv | KV 头数 |
| maskType | INT | = 0 | 掩码类型 |
| scaleValue | FLOAT | = 1/sqrt(128) ≈ 0.0884 | 缩放因子 |
| preTokens | INT | = 2147483647 | 预留参数 |
| nextTokens | INT | = 2147483647 | 预留参数 |

---

## 4. Shape 范围与约束

### 4.1 维度约束

| 约束维度 | 约束值 | 说明 |
|----------|--------|------|
| head_dim (D) | = 128 | 固定值，不可修改 |
| block_shape_x | 64 的倍数 | Q 块行数，默认 256 |
| block_shape_y | ≥128, 64 的倍数 | KV 块行数，默认 512 |
| Hq | ≥ Hkv, Hq % Hkv == 0 | GQA 分组查询约束 |
| 数据类型 | Q/K/V/O/dO/dQ/dK/dV = FP16, softmaxLse = FP32 | 统一数据类型 |
| 数据布局 | BNSD | 统一布局 |
| 目标芯片 | Ascend 910B (Atlas A2) | 硬件平台 |

### 4.2 序列长度

- Sq 和 Skv 支持任意正整数
- 非对齐时自动向上取整到 block_shape 的整数倍（zero-padding）
- 分块数量：`numQB = ceil(Sq / block_shape_x)`，`numKB = ceil(Skv / block_shape_y)`

### 4.3 稀疏掩码约束

- Shape: `[B, H_q, ceil(Sq/block_shape_x), ceil(Skv/block_shape_y)]`
- 类型: BOOL 或 UINT8
- 每个 Q 块至少有一个有效 KV 块（避免空行导致 softmax 异常）
- 稀疏率 30%~70% 为推荐区间

### 4.4 典型配置

| 场景 | B | Hq | Hkv | Sq | Skv | sparsity | block_shape |
|------|---|-----|-----|-----|------|----------|-------------|
| 基本 | 1 | 4 | 2 | 256 | 256 | 50% | [256, 512] |
| Dense | 1 | 4 | 4 | 256 | 256 | 100% | [256, 512] |
| GQA group4 | 1 | 8 | 2 | 256 | 512 | 40% | [256, 512] |
| GQA Hq32 Hkv8 | 1 | 32 | 8 | 256 | 256 | 50% | [256, 512] |
| 长序列 1024 | 1 | 8 | 1 | 1024 | 1024 | 30% | [256, 512] |
| 长序列 2048 | 1 | 4 | 2 | 2048 | 2048 | 30% | [256, 512] |
| 批量 | 2 | 4 | 2 | 256 | 512 | 50% | [256, 512] |

---

## 5. 精度要求

### 5.1 精度标准

| 输出 | atol | rtol | 说明 |
|------|------|------|------|
| dQ | 0.0001 | 0.0078125 | Query 梯度 |
| dK | 0.0001 | 0.0078125 | Key 梯度 |
| dV | 0.0001 | 0.0078125 | Value 梯度 |

### 5.2 精度实现要求

1. **D_row（softmaxGrad）** 必须在 FP32 精度下计算：`cast(dO*O, FP32)` 后再 `sum(dim=-1)`
2. **梯度累积** 使用 FP32 精度，最终 cast 到 FP16 输出
3. **dV 不乘 scale**（P 已包含 scale 效果），dQ 和 dK 需乘 scaleValue
4. **LSE padding** 使用 `1e30`（非 inf），确保 `exp(S*scale - 1e30) ≈ 0`

### 5.3 实测精度

| 用例 | dQ max_diff | dK max_diff | dV max_diff | 结果 |
|------|------------|------------|------------|------|
| BWD Basic (50%) | 0.000031 | 0.000061 | 0.000122 | PASSED |
| BWD Dense (100%) | 0.000031 | 0.000031 | 0.000122 | PASSED |
| BWD GQA (50%) | 0.000031 | 0.000031 | 0.000122 | PASSED |
| BWD Long Seq (30%) | 0.000031 | 0.000061 | 0.000244 | PASSED |

所有用例 max_diff 远小于容差阈值。

---

## 6. 性能要求

### 6.1 性能目标

| Shape | dQ Task Time | dK/dV Task Time | 目标 AICore Util |
|-------|-------------|-----------------|------------------|
| 256×256 | <100 us | <100 us | >35% |
| 512×512 | <150 us | <150 us | >40% |
| 1024×1024 | <250 us | <300 us | >55% |

### 6.2 优化措施

| 优化项 | 说明 |
|--------|------|
| Dense/Sparse 双路径 | 全稠密掩码时省去紧凑张量构建和掩码加载开销 |
| Sub-block 分割 | Dense dQ 将 256 行分为 2×128 子块，提升并行度 |
| Per-kernel L1 reuse | Dense dQ 使用 L1=64，其余使用 L1=16 |
| Kernel 工厂缓存 | 首次编译后缓存，后续零编译开销 |

---

## 7. 测试验证要求

### 7.1 功能测试

| 序号 | 测试场景 | 覆盖要点 | 预期结果 |
|------|---------|---------|---------|
| 1 | 基本稀疏 (50%) | Sparse dQ + dK/dV 路径 | 精度通过 |
| 2 | 全稠密 (100%) | Dense dQ + dK/dV 路径 | 精度通过 |
| 3 | GQA (Hq=8, Hkv=2) | GQA 分组映射正确性 | 精度通过 |
| 4 | 长序列 (1024) | 长序列梯度正确性 | 精度通过 |
| 5 | 批量 (B=2) | 多 batch 独立性 | 精度通过 |

### 7.2 精度测试

- 所有测试使用 `torch.allclose(actual, expected, atol=0.0001, rtol=0.0078125)` 验证
- 分别检查 dQ、dK、dV 三个输出的精度
- 记录 max_diff 作为精度指标

### 7.3 边界测试

| 场景 | 状态 | 说明 |
|------|------|------|
| 非对齐序列 | 已知限制 | Sq/Skv 非 block_size 整数倍时，padding 导致 softmax 偏差 |
| 空稀疏行 | 已规避 | 每个 Q 块保证至少一个有效 KV 块 |
| 极端稀疏率 (30%/70%) | 已覆盖 | 测试用例已包含 30% 和 70% 稀疏率 |

### 7.4 性能测试

- 采集 Task Time、AICore Time、AICore Util 指标
- 覆盖 256×256、512×512、1024×1024 三种典型 shape
- 验证 sub-block 分割和 L1 reuse 优化的有效性
