# BSA Backward 算子说明

## 1. 算子语义

BSA Backward（`aclnnBlockSparseAttentionGrad`）是块稀疏注意力（Block Sparse Attention）的反向传播算子，用于计算注意力机制对 Q/K/V 的梯度。基于前向传播保存的 O（注意力输出）和 softmaxLse（Log-Sum-Exp），通过重计算策略恢复中间矩阵 S 和 P，计算 dQ/dK/dV 梯度。

---

## 2. 数学公式

### 2.1 softmaxGrad

$$D_i = \sum_{t=1}^{d} dO_{i,t} \cdot O_{i,t}$$

其中 $D_i$ 为行级 softmaxGrad 标量，沿 head_dim 维度求和。

### 2.2 P 重计算

$$P_{uv} = \exp\left(\frac{Q_u K_v^\top}{\sqrt{d}} - \text{LSE}_u\right)$$

从前向保存的 S 和 LSE 重建注意力概率矩阵。

### 2.3 dS 计算

$$dS_{uv} = P_{uv} \odot \left(dO_u V_v^\top - D_u\right)$$

### 2.4 梯度累积

$$dQ_u \mathrel{+}= \frac{1}{\sqrt{d}} \cdot dS_{uv} \cdot K_v$$

$$dK_v \mathrel{+}= \frac{1}{\sqrt{d}} \cdot dS_{uv}^\top \cdot Q_u$$

$$dV_v \mathrel{+}= P_{uv}^\top \cdot dO_u$$

**关键**：dQ 和 dK 需要乘以 scale（$1/\sqrt{d}$），dV 不需要。

---

## 3. 计算流程

### 3.1 整体流程

```
Step 1: 输入预处理
  ├─ Pad Q/K/V/dO/O 到 block-aligned 大小
  ├─ Reshape 4D [B,H,S,D] → 2D [B*H*S,D]
  ├─ LSE padding (填充位 = 1e30)
  └─ 分配 dQ/dK/dV 输出缓冲

Step 2: 路径判断
  └─ dense_mask AND aligned → Dense / Sparse

Step 3: dQ 计算
  ├─ [Sparse] 构建紧凑 KV + valid_mask → dQ kernel
  └─ [Dense]  直接 2D 访问 + sub-block split → dense dQ kernel

Step 4: dK/dV 计算
  ├─ [Sparse] 构建紧凑 Q/dO/O/LSE + inner_mask → dK/dV kernel
  └─ [Dense]  直接 2D 访问 → dense dK/dV kernel

Step 5: 输出裁剪
  └─ 裁剪 padding，reshape 回 [B,H,S,D]
```

### 3.2 Kernel 内计算（以 dQ sparse 为例）

对每个 Q 块 `u`：
1. 加载 Q_u, dO_u, O_u, LSE_u
2. 计算 D_row = sum(dO_u * O_u)（FP32 精度）
3. 遍历有效 KV 块：
   - 重计算 S = Q_u @ K_v^T * scale
   - 重计算 P = exp(S - LSE_u)
   - 应用掩码 P_masked = P * valid_mask
   - 计算 dP = P_masked * (dO_u @ V_v^T - D_row)
   - 累积 dq_acc += dP @ K * scale
4. 写出 dQ_u = cast(dq_acc, FP16)

---

## 4. 输入输出规格

### 4.1 输入张量

| 名称 | 数据类型 | 形状 | 说明 |
|------|----------|------|------|
| dout | FP16 | `[B, H_q, Sq, 128]` | 来自上层的梯度 |
| query | FP16 | `[B, H_q, Sq, 128]` | 前向 Query |
| key | FP16 | `[B, H_kv, Skv, 128]` | 前向 Key |
| value | FP16 | `[B, H_kv, Skv, 128]` | 前向 Value |
| attentionOut | FP16 | `[B, H_q, Sq, 128]` | 前向注意力输出 O |
| softmaxLse | FP32 | `[B, H_q, Sq]` | 前向 softmax LSE |
| blockSparseMask | BOOL/UINT8 | `[B, H_q, ceil(Sq/bx), ceil(Skv/by)]` | 块稀疏掩码 |

### 4.2 可选参数

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| actualSeqLengths | INT64 `[B]` | None (全部 Sq) | 每batch的Q实际序列长度 |
| actualSeqLengthsKv | INT64 `[B]` | None (全部 Skv) | 每batch的KV实际序列长度 |
| blockShape | INT64 `[2]` | [256, 512] | `[blockShapeX, blockShapeY]` |

### 4.3 输出张量

| 名称 | 数据类型 | 形状 | 说明 |
|------|----------|------|------|
| dq | FP16 | `[B, H_q, Sq, 128]` | Query 梯度 |
| dk | FP16 | `[B, H_kv, Skv, 128]` | Key 梯度 |
| dv | FP16 | `[B, H_kv, Skv, 128]` | Value 梯度 |

---

## 5. Shape 范围与约束

### 5.1 维度约束

| 参数 | 约束 | 说明 |
|------|------|------|
| head_dim (D) | = 128 | 固定值 |
| Hq | ≥ Hkv, Hq % Hkv == 0 | GQA 约束 |
| block_shape_x | 64 的倍数 | Q 块大小，默认 256 |
| block_shape_y | ≥128, 64 的倍数 | KV 块大小，默认 512 |

### 5.2 数据类型约束

- Q/K/V/O/dO/dQ/dK/dV: **FP16**
- softmaxLse: **FP32**
- blockSparseMask: **BOOL** 或 **UINT8**

### 5.3 布局约束

- 统一 **BNSD** 布局：`[Batch, HeadNum, SeqLen, HeadDim]`
- 张量内存布局必须为紧凑布局，禁止非连续步长张量

### 5.4 典型配置

| 场景 | B | Hq | Hkv | Sq | Skv | sparsity |
|------|---|-----|-----|-----|------|----------|
| 基本 | 1 | 4 | 2 | 256 | 256 | 50% |
| Dense | 1 | 4 | 4 | 256 | 256 | 100% |
| GQA | 1 | 8 | 2 | 256 | 512 | 50% |
| 长序列 | 1 | 8 | 1 | 1024 | 1024 | 30% |
| 批量 | 2 | 4 | 2 | 256 | 512 | 50% |

---

## 6. 实现特点

### 6.1 Kernel 拆分

反向传播拆分为 **dQ kernel** 和 **dK/dV kernel** 两个独立 kernel：
- `pypto.assemble` 是覆写语义（非累加），多个 Q 块写同一 KV 位置会互相覆盖
- dQ kernel：外层 Q 块 × 内层紧凑 KV 块，每个 Q 块写入唯一位置
- dK/dV kernel：外层 KV 块 × 内层紧凑 Q 块，多个 Q 块的贡献在同一 kernel 内累积

### 6.2 Dense/Sparse 双路径

- **Dense 路径**：当 mask 全 1 且序列对齐时，直接 2D 访问，省去紧凑张量构建和掩码加载开销
- **Sparse 路径**：使用紧凑张量跳过无效块，减少内层循环迭代次数

### 6.3 FP32 累积

所有梯度在 FP32 精度下累积，最终 cast 到 FP16 输出。特别是 D_row（softmaxGrad）**必须**先 cast 到 FP32 再 sum。

### 6.4 Sub-block 分割

Dense dQ kernel 将 256 行 Q-block 分为 2 个 128 行子块，外层任务数翻倍，提升多核并行度。

### 6.5 测试覆盖

- 基本稀疏场景、Dense 场景、GQA 分组、长序列（1024）
- 精度容差：atol=0.0001, rtol=0.0078125
- 已知限制：非对齐序列（Sq/Skv 非 block_size 整数倍）精度不满足要求
