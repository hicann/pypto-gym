# LightningIndexer

## 产品支持情况
| 产品 | 是否支持 |
|:----------------------------|:-----------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品| √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品| √ |

## 功能说明

LightningIndexer 算子基于一系列操作得到每一个 token 对应的 Top-k 个位置。

### 计算公式

$$
Indices = \text{Top-}k(W \odot \text{ReLU}(Q_{index} @ K_{index}^T))
$$

对于某个 token 对应的 Index Query $Q_{index} \in \mathbb{R}^{B \times S_q \times N \times D}$，给定上下文 Index Key $K_{index} \in \mathbb{R}^{B \times S_{kv} \times N \times D}$，以及权重 $W \in \mathbb{R}^{B \times S_q \times N}$，其中 $B$ 为 batch size，$N$ 为头数，$D$ 为每个头的维度，$S_q$ 是 query 的序列长度，$S_{kv}$ 是 key 的序列长度。

### 计算步骤

1. **矩阵乘法**: 计算 $Q_{index} @ K_{index}^T$ 得到注意力分数矩阵
2. **ReLU 激活**: 对分数矩阵应用 ReLU 激活函数
3. **加权**: 将权重 $W$ 与激活后的分数矩阵相乘
4. **Top-k**: 取每个位置的 top-k 个索引

## 函数原型

```python
def lightning_indexer_kernel(
    query: pypto.Tensor,      # [B, Sq, N, D]
    key: pypto.Tensor,        # [B, Skv, N, D]
    weights: pypto.Tensor,    # [B, N, Sq, 1]
    indices: pypto.Tensor,    # [B, Sq, N, topk] 输出
):
    ...
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
|--------|-----------|------|
| query | 输入 | Query tensor，shape 为 [B, Sq, N, D]，数据类型为 BF16 |
| key | 输入 | Key tensor，shape 为 [B, Skv, N, D]，数据类型为 BF16 |
| weights | 输入 | 权重 tensor，shape 为 [B, N, Sq, 1]，数据类型为 BF16 |
| indices | 输出 | Top-k 索引输出，shape 为 [B, Sq, N, topk]，数据类型为 INT32 |

## 约束说明

- 仅支持 BF16 数据类型
- topk 值需要满足 TileShape 尾轴对齐要求（32 bytes 对齐）
- 输入 tensors 需要是连续的（contiguous）
- 支持动态 batch size（B 维度）

## 编译运行指南

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0

# 设置 PTO 库路径
export PTO_TILE_LIB_CODE_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/aarch64-linux
```

### 编译

```bash
cd /mnt/workspace/gitCode/cann/pypto
python3 build_ci.py -f python3 --disable_auto_execute
pip install --force-reinstall build_out/pypto-0.1.1-cp310-cp310-linux_aarch64.whl
```

### 运行测试

```bash
cd /mnt/workspace/gitCode/cann/pypto/custom/lightning_indexer

# 默认 batch size = 1
python3 lightning_indexer.py

# 指定 batch size（支持动态 batch）
python3 lightning_indexer.py --batch_size 4
```

## 测试结果

### 测试用例配置

- Batch Size (B): 1
- Number of Heads (N): 8
- Query Sequence Length (Sq): 64
- Key Sequence Length (Skv): 64
- Head Dimension (D): 128
- Top-k: 8

### 测试输出

```
Input query shape: torch.Size([1, 64, 8, 128])
Input key shape: torch.Size([1, 64, 8, 128])
Input weights shape: torch.Size([1, 8, 64, 1])
Output indices shape: torch.Size([1, 64, 8, 8])
Exact match: True
Match ratio: 100.00% (4096/4096)
✓ LightningIndexer test completed
```

### 精度验证

- 与 PyTorch 参考实现对比
- 所有 4096 个索引值完全匹配
- Match ratio: 100%

## 已知限制

1. weights tensor 需要在传入前预先 transpose 并 expand 维度
2. topk 操作需要 FP32 数据类型，内部会进行类型转换
3. 其他维度（Sq, Skv, N, D）目前仍为静态维度

## API 映射关系

| 数学操作 | PyPTO API | 说明 |
|---------|-----------|------|
| $Q @ K^T$ | `pypto.matmul(q, k, b_trans=True)` | 矩阵乘法，K 需要转置 |
| ReLU | `pypto.relu(scores)` | 激活函数 |
| $\odot$ | `pypto.mul(a, b)` | 元素乘法（支持广播） |
| 类型转换 | `pypto.cast(x, pypto.DT_FP32)` | BF16 转 FP32 |
| Top-k | `pypto.topk(x, k, dim=-1)` | 获取前 k 个最大值的索引 |
| 转置 | `pypto.transpose(x, dim0, dim1)` | 维度交换 |

## 文件结构

```
custom/lightning_indexer/
├── lightning_indexer.py    # 算子实现和测试
└── README.md               # 本文档
```
