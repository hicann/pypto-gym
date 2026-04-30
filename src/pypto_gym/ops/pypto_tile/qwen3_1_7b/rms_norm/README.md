# Qwen3-1.7B RMSNorm 算子集成

## 概述

将 RMSNorm 算子从原始 torch 实现替换为 PyPTO 融合算子，提升 NPU 上的推理性能。

## 文件结构

```
pto_kernels/
├── __init__.py                    # USE_PTO 开关
├── rms_norm/
│   ├── __init__.py                # wrapper 导出
│   ├── rms_norm_impl.py           # PyPTO 实现
│   ├── rms_norm_golden.py         # PyTorch 参考实现
│   ├── README.md                  # 本文档
│   └── test/
│       ├── test_rms_norm.py       # 测试脚本
│       └── test_cases.json        # 真实测试用例
```

## 使用方法

### 启用 PTO 算子

```bash
python3 scripts/ask_Qwen3-1.7B.py --prompt "你好" --use-pto
```

### 禁用 PTO（使用原始 torch 实现）

```bash
python3 scripts/ask_Qwen3-1.7B.py --prompt "你好"
```

## 测试验证

```bash
cd /mnt/workspace/gitCode/cann/models/pure/Qwen3-1.7B
export TILE_FWK_DEVICE_ID=0
python3 pto_kernels/rms_norm/test/test_rms_norm.py
```

### 测试用例来源

从 Qwen3-1.7B 模型打点采集的真实 shape/dtype：
- prefill 阶段：[1, 11, 2048], [1, 11, 16, 128], [1, 11, 8, 128]
- decode 阶段：[1, 1, 2048], [1, 1, 16, 128], [1, 1, 8, 128]

## 集成修改

### modeling_qwen3.py 修改

1. 导入 sys.modules 获取 pto_kernels
2. Qwen3RMSNorm.forward 使用条件分支调用 wrapper

### 推理脚本修改

添加 `--use-pto` 参数和 sys.modules 注入逻辑。

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
