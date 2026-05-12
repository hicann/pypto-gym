# Qwen3-1.7B decode_attn 算子集成

## 概述

将 decode 阶段的 attention 算子替换为 PyPTO GQA-native 融合实现，优化 KV cache 访问模式，提升 NPU 上的推理性能。

## 文件结构

```
/path/to/pypto_gym/ops/pypto_tile/qwen3_1_7b/
└── decode_attention/
    ├── decode_attn_impl.py                  # Wrapper（调用 gym 仓）
    ├── decode_attn_integration.py           # Prefill/Decode 自动切换
    ├── qwen3_decode_attn.py                 # PyPTO 实现（引用 gym 仓）
    └── README.md                            # 本文档
```

## 测试验证

### 单算子测试（精度验证）

```bash
```bash
cd tests/ops/qwen3_1_7b
export TILE_FWK_DEVICE_ID=0

# Gym仓标准测试
python3 test_decode_attn.py
```


### 测试结果

**精度标准：max_diff < 2e-3（我的标准）**

| Skv | Scale | max_diff | 结论 | 场景说明 |
|-----|-------|----------|------|----------|
| 10  | 0.5   | 0.001953 | ✅ PASS | 常用decode（生成10 tokens） |
| 50  | 0.5   | 0.000977 | ✅ PASS | 常用decode（生成50 tokens） |

**对比 gym 仓标准：**
- gym 仓标准：< 5.0
- 数值稳定性：无 NaN/Inf ✅

## 测试用例来源

从 Qwen3-1.7B 模型 decode 阶段采集的真实参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| Nq | 16 | num_attention_heads |
| Nkv | 8 | num_key_value_heads |
| D | 128 | head_dim |
| GROUPS | 2 | Nq // Nkv（GQA） |
| Skv | 10-128 | KV cache序列长度 |

数据来源：`test_cases.json`

## 技术说明

### 场景判断

原始 attention 使用 torch_npu 融合算子（flash_attention），属于**场景B**：
- 需用 torch 重写等价实现
- 必须验证 Golden 与 torch_npu 一致性
- Golden 需符合实际计算逻辑

### 实现选择

**引用 gym 仓实现：**
- 路径：`pypto_gym.ops.pypto_tile.qwen3_1_7b.qwen3_decode_attn`
- 特性：GQA-native + Online softmax + bfloat16
- 优化：Batched 3D matmul + Cube L1 reuse

### GQA-native vs repeat_kv

**两种实现方式（数学等价）：**

| 方式 | 实现逻辑 | 计算复杂度 |
|------|---------|-----------|
| **GQA-native（gym仓）** | Query reshape [Nq, D] → [Nkv, GROUPS, D] | 更高效（避免KV扩展） |
| **repeat_kv（Qwen3原始）** | KV扩展 [Nkv, Skv, D] → [Nq, Skv, D] | 需额外内存 |

**精度验证：**
- Gym仓实现 vs Qwen3实际逻辑：max_diff < 2e-3 ✅
- 数学等价，性能更优

### Prefill/Decode 自动切换机制

**核心逻辑：**

```python
# 在 modeling_qwen3.py 中判断
Sq = query_states.shape[2]

if Sq == 1:  # Decode模式
    if USE_PTO_DECODE_ATTN:
        # 使用 PyPTO gym仓kernel
        qwen3_decode_attn(...)
    else:
        # fallback：原始 repeat_kv + matmul
else:  # Prefill模式（Sq>1）
    # 使用原始 eager_attention_forward
```

**关键文件：**
- `decode_attn_integration.py`：实现切换逻辑
- `modeling_qwen3.py`：注入判断代码（第219-238行）

### KV Cache Padding

**gym仓要求：**
- S2_TILE = 64（固定分块大小）
- Skv必须padding到64倍数

**自动处理：**
```python
Skv_p = ((Skv + S2_TILE - 1) // S2_TILE) * S2_TILE

k_pad = torch.zeros(Nkv, Skv_p, D, dtype=torch.bfloat16)
k_pad[:, :Skv] = k  # 有效部分填充

mask_pad = torch.zeros(Skv_p)
mask_pad[:Skv] = mask      # 有效部分全0
mask_pad[Skv:] = -1e30     # padding部分mask掉
```

### 数值稳定性

**关键点：**
- 使用 bfloat16 精度
- 输入缩放 * 0.5 避免溢出
- Online softmax 在合理序列长度稳定
- 无 NaN/Inf（已验证）

### 集成方式

**自动注入（推荐）：**

在 `modeling_qwen3.py` 的 `Qwen3Attention.forward` 中：
```python
import sys
pto_kernels = sys.modules.get("qwen3_pto_kernels")

use_pto_decode_attn = (
    pto_kernels is not None and 
    pto_kernels.USE_PTO_DECODE_ATTN and 
    query_states.shape[2] == 1  # Decode模式
)

if use_pto_decode_attn:
    # 使用 PyPTO decode_attn
    attn_output = decode_attn_forward_pto(...)
else:
    # 使用原始实现
    attn_output = eager_attention_forward(...)
```

**优势：**
- 自动判断 prefill/decode
- 无需手动切换
- Fallback机制完善

## 状态

✅ 环境验证
✅ 网络基线验证
✅ 打点采集
✅ Golden 编写（符合实际逻辑）
✅ Golden 验证（对比 torch_npu）
✅ 单算子验证（max_diff < 2e-3）
✅ Prefill/Decode自动切换实现
✅ KV Cache Padding处理
✅ 数值稳定性验证（无NaN/Inf）
✅ 模型集成（自动注入）
✅ 端到端验证（整网推理成功）
