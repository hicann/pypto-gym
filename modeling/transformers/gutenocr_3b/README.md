# GutenOCR-3B PyPTO算子集成说明

## 基本信息

| 项目 | 值 |
|------|------|
| HuggingFace原版 | [rootsautomation/GutenOCR-3B](https://huggingface.co/rootsautomation/GutenOCR-3B) |
| 权重目录 | /path/to/gutenocr_3b |
| 代码来源 | transformers包（内置，model_type=gutenocr_3b_vl） + PyPTO算子集成 |
| transformers版本 | 4.40.0+ |
| PyPTO版本 | CANN 9.0.0 |
| 运行命令 | `python3 scripts/ask_gutenocr_3b_compile.py --device 1 --batch 16 --use_dynamic_config` |

## PyPTO算子集成状态

| 算子 | 集成状态 | 性能提升 | 推荐启用 | 说明 |
|------|---------|---------|---------|------|
| SwiGLU MLP | ✅ 已集成 | +14% (Batch≥16) | ✅ 推荐 | 融合算子消除3个kernel launch |
| MRoPE | ✅ 已集成 | +5% (小batch) | ⚠️ Batch≤4 | 小batch有效，大batch固化开销抵消 |
| RMSNorm | ✅ 已集成 | +60% (单算子) | ❌ 不推荐 | sys.modules固化开销抵消优化 |
| Softmax | ⚠️ 已集成 | - | ❌ 禁用 | overhead抵消优化 |

## 性能数据（标准prompt真实场景）

| Batch | Baseline吞吐 | PyPTO吞吐 | 加速 | 最优模式 |
|-------|-------------|----------|------|---------|
| 1 | 25.14 | 27.08 | +7.8% | PyPTO Eager |
| 4 | 22.55 | 25.66 | +13.7% | PyPTO Eager |
| 8 | 20.91 | 24.27 | +15.7% | PyPTO Eager |
| 16 | 16.60 | 17.12 | +3.1% | PyPTO Eager |

**平均加速**: +10.1%

## 目录结构

```
gutenocr_3b/
├── config.json
├── model*.safetensors
├── model.safetensors.index.json
├── tokenizer.json / tokenizer_config.json
├── vocab.json / merges.txt
├── generation_config.json
├── modeling_gutenocr_3b_vl.py              # 模型文件（已集成PyPTO算子）
├── pto_kernels/                        # PyPTO算子库
│   ├── __init__.py                     # 算子开关配置
│   ├── rms_norm/                       # RMSNorm算子
│   ├── mrope/                          # MRoPE算子
│   ├── swiglu_mlp/                     # SwiGLU融合算子
│   └── softmax/                        # Softmax算子（禁用）
└── scripts/
    ├── ask_gutenocr_3b_compile.py      # 主推理脚本
    ├── dynamic_pto_config.py           # 动态配置模块
    ├── setup_cann9_env.sh              # CANN环境配置
    ├── test_batch16_standard_prompt.py # 性能测试脚本
    ├── 当前脚本使用指南.md              # 脚本详细说明
    ├── 最终总结报告.md                  # 集成总结报告
    └── README.md                        # 本文件
```

## 使用方法

### 环境设置

```bash
# 设置CANN 9.0.0环境
source scripts/setup_cann9_env.sh

# 验证环境
echo $ASCEND_TOOLKIT_HOME
echo $PTO_TILE_LIB_CODE_PATH
```

### 推理命令

```bash
# Baseline Eager（基准）
python3 scripts/ask_gutenocr_3b_compile.py --batch 16 --device 1 --warmup 20

# PyPTO Eager（推荐）
python3 scripts/ask_gutenocr_3b_compile.py --batch 16 --use_dynamic_config --device 1 --warmup 20

# PyPTO + aclgraph（仅Batch=4推荐）
python3 scripts/ask_gutenocr_3b_compile.py --batch 4 --use_dynamic_config --use_acl_graph --device 1

# 简化prompt（仅用于算子调试）
python3 scripts/ask_gutenocr_3b_compile.py --batch 16 --use_dynamic_config --simple_prompt --device 1
```

### 性能测试

```bash
# 标准prompt真实性能测试
python3 scripts/test_batch16_standard_prompt.py

# 多次稳定性测试
python3 scripts/test_batch16_multiple.py

# aclgraph对比测试
python3 scripts/test_aclgraph_comparison.py
```

### 算子手动配置

在 `pto_kernels/__init__.py` 中设置：

```python
# 推荐配置（所有batch）
USE_PTO_RMS_NORM = False   # 禁用
USE_PTO_MROPE = False      # Batch≥8禁用
USE_PTO_SWIGLU_MLP = True  # 启用
USE_PTO_SOFTMAX = False    # 禁用

# 小batch配置（Batch≤4）
USE_PTO_MROPE = True       # 启用MRoPE
```

## 推荐配置

### 生产环境推荐

所有batch统一使用PyPTO Eager模式：

| Batch | 推荐命令 | 预期吞吐 | 加速 |
|-------|---------|---------|------|
| 1 | `--batch 1 --use_dynamic_config` | 27.08 | +7.8% |
| 4 | `--batch 4 --use_dynamic_config` | 25.66 | +13.7% |
| 8 | `--batch 8 --use_dynamic_config` | 24.27 | +15.7% |
| 16 | `--batch 16 --use_dynamic_config` | 17.12 | +3.1% |

### 算子推荐

**启用**:

- SwiGLU MLP（所有batch）
- MRoPE（仅Batch≤4）

## 核心特性

### 动态配置

根据batch自动选择最优算子组合：

```python
from dynamic_pto_config import DynamicPTOConfig

config = DynamicPTOConfig.get_optimal_config(batch_size=16)
# 自动设置: use_rms_norm=False, use_mrope=False, use_swiglu=True
```

### SwiGLU融合算子

融合gate_proj、up_proj、down_proj为1个kernel，消除3个独立kernel launch开销。
