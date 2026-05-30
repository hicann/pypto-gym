# Phi-3-mini-4k-instruct RMSNorm 算子集成

## 概述

将 RMSNorm 算子从原始 torch 实现替换为 PyPTO 融合算子，提升 NPU 上的推理性能。

## 文件结构

```
pypto_gym/ops/pypto_tile/phi_3_mini_4k_instruct/
├── __init__.py
└── rms_norm/
    ├── __init__.py
    ├── rms_norm_impl.py     # PyPTO kernel 实现 (D=3072)
    └── README.md            # 本文档
```

## 测试验证

```bash
cd tests/ops/phi_3_mini_4k_instruct

python3 test_rms_norm.py
```

## 测试用例来源

从 Phi-3-mini-4k-instruct 模型打点采集的真实 shape/dtype：
- decode 阶段：[1, 1, 3072]
- prefill 短：[1, 5, 3072]
- prefill 中：[1, 128, 3072]
- prefill 最大：[1, 4096, 3072]

## 技术说明

### 场景判断

原始实现使用纯 torch 基础算子（pow、mean、rsqrt），属于**场景A**：
- Golden 直接复制原始代码
- 无需 torch_npu 验证

### 实现选择

使用 PyPTO 手动 tiling 实现，TILE_M=4，支持 DYNAMIC 序列长度。

### ACLGraph 支持

已注册 `torch.library` 自定义算子：
- **op name**: `pypto::rms_norm_phi3`
- **Meta**: 返回 `empty_like`，适配动态 shape
- **NPU backend**: 调用 `rms_norm_wrapper`（PyPTO JIT kernel）
- **调用入口**: `rms_norm_pypto(hidden_states, weight)` → `torch.ops.pypto.rms_norm_phi3`

模型级 torch.compile 封装通过 `--use-acl-graph` 启用。

## 性能数据

> 测试环境: Ascend 910B2, CANN 25.5.0

| 算子 | 输入 shape | 输入 dtype | 状态 |
|------|-----------|-----------|------|
| rms_norm (decode) | [1, 1, 3072] | float16 | ✅ 精度通过 |
| rms_norm (prefill 5) | [1, 5, 3072] | float16 | ✅ 精度通过 |
| rms_norm (prefill 128) | [1, 128, 3072] | float16 | ✅ 精度通过 |
| rms_norm (prefill 4096) | [1, 4096, 3072] | float16 | ✅ 精度通过 |

## 状态

✅ 环境验证
✅ 网络基线验证
✅ 打点采集
✅ Golden 编写
✅ 单算子精度验证
✅ 模型集成 (sys.modules 注入)
✅ aclgraph 支持 (torch.library + torch.compile)
✅ 端到端验证 (Eager + ACLGraph 模式均通过)
⏳ 单算子性能调优
