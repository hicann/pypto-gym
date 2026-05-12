# Qwen3-1.7B RMSNorm 算子集成

## 概述

将 RMSNorm 算子从原始 torch 实现替换为 PyPTO 融合算子，提升 NPU 上的推理性能。

## 文件结构

```
pypto_gym/ops/pypto_tile/qwen3_1_7b/
└── rms_norm/
    ├── rms_norm_impl.py           # PyPTO 实现（引用 gym 仓）
    └── README.md                  # 本文档
```

## 测试验证

### 单算子测试

```bash
cd tests/ops/qwen3_1_7b
export TILE_FWK_DEVICE_ID=0
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa

python3 test_rms_norm.py
```

### 测试结果

```
[input_layernorm] shape=[1, 1, 2048], max_diff=0.000000 ✅
[q_norm]          shape=[1, 1, 16, 128], max_diff=0.000000 ✅
[k_norm]          shape=[1, 1, 8, 128], max_diff=0.000000 ✅

精度标准：< 2e-3
实际精度：完美对齐（max_diff=0）
```

## 测试用例来源

从 Qwen3-1.7B 模型打点采集的真实 shape/dtype：

| 场景 | Shape | 说明 |
|------|-------|------|
| input_layernorm | [1, 1, 2048] | Decode阶段，hidden_size=2048 |
| q_norm | [1, 1, 16, 128] | Qwen3特有，Q头归一化 |
| k_norm | [1, 1, 8, 128] | Qwen3特有，KV头归一化 |

## 技术说明

### 场景判断

原始实现使用纯 torch 基础算子（pow、mean、rsqrt），属于**场景A**：
- Golden 直接复制原始代码
- 无需 torch_npu 验证
- Golden 与原始实现数学等价

### 实现选择

**引用 gym 仓实现：**
- 路径：`pypto_gym.ops.pypto_tile.qwen3_1_7b.rms_norm.rms_norm_impl`
- API：`pypto.rms_norm` 融合算子
- 优化：自动 TileShape 设置 + L1 reuse

**关键修改：**
- 原gym仓签名：`(hidden_states, weight, output, epsilon)`
- 修正签名：`(hidden_states, weight, output, eps)` （参数名修正）
- TileShape：根据rank动态设置 `[128 for _ in range(rank)]`

### 集成方式

**sys.modules 注入（推荐）：**

在 `modeling_qwen3.py` 中注入：
```python
import sys
sys.modules["transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm"] = pto_kernels.rms_norm.rms_norm_impl
```

**优势：**
- 无需修改模型代码
- 全局生效（所有RMSNorm实例）
- 自动fallback机制

## 状态

✅ 环境验证
✅ 网络基线验证
✅ 打点采集
✅ Golden 编写
✅ 单算子验证（max_diff=0）
✅ 模型集成（sys.modules注入）
✅ 端到端验证（整网推理成功）