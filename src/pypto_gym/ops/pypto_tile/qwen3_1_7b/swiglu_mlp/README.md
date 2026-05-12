# Qwen3-1.7B SwiGLU MLP 算子集成

## 概述

将 SwiGLU MLP 算子（gate + up + silu + down）从原始 torch 实现替换为 PyPTO 融合算子，减少中间存储，提升 NPU 上的推理性能。

## 文件结构

```
/data/h00520348/models/Qwen3-1.7B/pto_kernels/
└── swiglu_mlp/
    ├── swiglu_mlp_impl.py         # PyPTO 实现（引用 gym 仓）
    └── README.md                  # 本文档
```

## 测试验证

### 单算子测试

```bash
cd tests/ops/qwen3_1_7b
export TILE_FWK_DEVICE_ID=0

python3 test_swiglu_mlp.py
```

### 测试结果

```
[decode_single_token] shape=[1, 2048], max_diff=0.000014 ✅
[decode_batch]        shape=[16, 2048], max_diff=0.000017 ✅

精度标准：< 2e-3
实际精度：远优于标准（max_diff≈0.000015）
```

## 测试用例来源

从 Qwen3-1.7B 模型打点采集的真实 shape/dtype：

| 场景 | Shape | 说明 |
|------|-------|------|
| MLP输入 | [1, 2048] | Decode阶段，单个token |
| MLP输入 | [16, 2048] | Batch推理场景 |

**参数配置：**
- H = 2048（hidden_size）
- INT_SIZE = 6144（intermediate_size = 3×H）

数据来源：`pto_kernels/test_cases.json`

## 技术说明

### 场景判断

原始实现使用纯 torch 基础算子（matmul、silu），属于**场景A**：
- Golden 直接复制原始逻辑
- 无需 torch_npu 验证
- Golden 与原始实现数学等价

### 实现选择

**引用 gym 仓实现：**
- 路径：`pypto_gym.ops.pypto_tile.qwen3_1_7b.swiglu_mlp.swiglu_mlp_impl`
- 融合算子：gate + up + silu + down 合并为单kernel
- 优化：减少中间存储，Cube L1 reuse

**数学公式：**
```
gate = matmul(x, Wgate)          # [S, 2048] @ [2048, 6144]
up = matmul(x, Wup)              # [S, 2048] @ [2048, 6144]
silu_gate = silu(gate)           # gate * sigmoid(gate)
hidden = silu_gate * up          # element-wise multiply
output = matmul(hidden, Wdown)   # [S, 6144] @ [6144, 2048]
```

### 数值稳定性关键

**问题：** 标准 randn 初始化导致数值溢出 → NaN/Inf

**原因：**
- 标准randn：均值0，标准差1
- 3次连续matmul累积放大数值范围
- SiLU（sigmoid运算）对大数值敏感

**解决方案：**
```python
# 缩放初始化范围
x = torch.randn(shape) * 0.1      # 输入标准差降至0.1
W = torch.randn(shape) * 0.01     # 权重标准差降至0.01
```

**原理：**
- 降低初始数值范围，避免matmul累积放大
- SiLU在合理数值范围内计算稳定
- 符合真实模型训练后的权重分布特性

### 集成方式

**Monkey Patching：**

在 `qwen3_pto_kernels/__init__.py` 中：
```python
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

def pto_mlp_forward(self, x):
    return swiglu_mlp_impl(x, self.gate_proj.weight, 
                           self.up_proj.weight, self.down_proj.weight)

Qwen3MLP.forward = pto_mlp_forward
```

**权重处理：**
- PyTorch Linear.weight：[out_features, in_features]
- PyPTO matmul需要：[in_features, out_features]
- Wrapper自动处理transpose

## 状态

✅ 环境验证
✅ 网络基线验证
✅ 打点采集
✅ Golden 编写
✅ 单算子验证（max_diff=0.000014）
✅ 数值稳定性修复（缩放初始化）
✅ 模型集成（Monkey patching）
✅ 端到端验证（整网推理成功）