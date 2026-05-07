# Qwen3-VL-8B RMSNorm 算子集成

## 概述

将 RMSNorm 算子从原始 torch 实现替换为 PyPTO 融合算子，提升 NPU 上的推理性能。

## 文件结构

```
pypto_gym/ops/pypto_tile/qwen3_vl_8b_instruct_unredacted_max/
└── rms_norm/
    ├── rms_norm_impl.py           # PyPTO 实现
    └── README.md                  # 本文档
```

## 测试验证

```bash
cd tests/ops/qwen3_vl_8b_instruct_unredacted_max
export TILE_FWK_DEVICE_ID=0
python3 test_rms_norm.py
```

### 测试用例来源

从 Qwen3-VL-8B 模型打点采集的真实 shape/dtype：
- prefill 阶段：[1, 11, 2048]
- decode 阶段：[1, 1, 2048]

## 技术说明

### 场景判断

原始实现使用纯 torch 基础算子（pow、mean、rsqrt），属于**场景A**：
- Golden 直接复制原始代码
- 无需 torch_npu 验证

### 实现选择

使用 PyPTO 内置 `pypto.rms_norm` 融合算子实现。

## 状态

✅ 环境验证
✅ 网络基线验证
✅ 打点采集
✅ Golden 编写
⏳ 单算子验证
⏳ 模型集成
⏳ 端到端验证
