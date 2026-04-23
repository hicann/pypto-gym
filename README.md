# PyPTO-Gym

[简体中文](README.md)

PyPTO-Gym 是基于 [PyPTO](https://gitcode.com/cann/pypto) 编程框架构建的算子/模型样例仓库。它收录了一批用 PyPTO 写成的高性能融合算子与典型大模型结构实现，作为 PyPTO 的"算子体操场"，方便开发者学习、复用、压测与对比。

> 本仓原为 `pypto/models/` 目录，现已拆分为独立仓，与 PyPTO 主仓解耦演进。

## 🚀 概述

PyPTO-Gym 的定位类似 NVIDIA 的 [TileGym](https://github.com/NVIDIA/TileGym) 之于 cuTile —— 一个围绕编程框架的算子示例和基准库。区别在于：

- **硬件目标**：华为昇腾（Ascend）AI 处理器
- **编程框架**：PyPTO，基于 Tile 的编程模型
- **内容**：端到端可运行的融合算子样例 + 大模型关键结构（Attention、MoE、LSTM、Delta-Rule 等）

## ✨ 特性

- 覆盖 DeepSeek V3.2、GLM V4.5、Qwen3-Next、Arctic、QAT 等模型的关键算子实现
- 提供实验性目录 `experimental/` 收录 Attention、Matmul、Vector、Distributed 等基础算子的开发态样例
- 每个模型目录自包含：impl 文件 + pytest 测试 + README，可独立运行
- 复用 PyPTO 自带的多卡/多 SoC 测试调度 `conftest.py`（`@pytest.mark.soc`、`@pytest.mark.world_size`）

## 📦 安装

### 前置依赖

- 昇腾 CANN 环境
- [PyPTO](https://gitcode.com/cann/pypto) ≥ 0.2.0（必选）
- PyTorch + torch_npu（根据昇腾版本选择）

### 从源码安装

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym
pip install -e .
```

开发模式附加依赖：

```bash
pip install -e ".[dev]"
```

## ⚡️ 快速上手

### 1. 运行单个模型的测试

```bash
# Arctic LSTM
pytest src/pypto_gym/models/arctic -v

# GLM V4.5 Attention
pytest src/pypto_gym/models/glm_v4_5/glm_attention.py -v

# DeepSeek V3.2 Sparse Flash Attention
pytest src/pypto_gym/models/deepseek_v32_exp/deepseekv32_sparse_flash_attention_quant.py -v

# Qwen3-Next Gated Delta Rule
pytest src/pypto_gym/models/qwen3_next -v
```

### 2. 运行全部（非 experimental）测试

```bash
pytest
```

`pytest.ini` 默认排除 `experimental/` 目录；如需运行实验性算子：

```bash
pytest src/pypto_gym/models/experimental/<op_name> -v
```

### 3. 多卡 / 指定 SoC

```bash
# 指定 NPU device id
pytest src/pypto_gym/models/<model> --device 0

# 多卡（2 卡）分布式样例
pytest src/pypto_gym/models/experimental/distributed --device 0 1 --cards-per-case 2
```

## 🔍 目录结构

```
pypto-gym/
├── docs/                                    # 文档资源（规划中）
├── src/
│   └── pypto_gym/
│       ├── __init__.py
│       └── models/                          # 模型 / 算子样例根目录
│           ├── arctic/                      # Arctic LSTM
│           │   ├── sum_lstm.py
│           │   ├── test_sum_lstm.py
│           │   └── README.md
│           ├── deepseek_v32_exp/            # DeepSeek V3.2 实验算子
│           │   ├── deepseekv32_sparse_flash_attention_quant.py
│           │   ├── deepseekv32_mla_prolog_quant.py
│           │   ├── deepseekv32_lightning_indexer_quant.py
│           │   └── ...
│           ├── glm_v4_5/                    # GLM V4.5
│           │   ├── glm_attention.py
│           │   ├── glm_moe_fusion.py
│           │   ├── glm_select_experts.py
│           │   └── ...
│           ├── qat/                         # 量化感知训练
│           ├── qwen3_next/                  # Qwen3-Next Gated Delta Rule
│           └── experimental/                # 实验性算子（默认不跑）
│               ├── attention/
│               ├── distributed/
│               ├── flash_attention_score_grad/
│               ├── matmul/
│               ├── ops-transformer/
│               └── vector/
├── tests/                                   # 共享测试工具（conftest 聚合入口）
├── conftest.py                              # pytest 调度（多卡 / 多 SoC 筛选）
├── pytest.ini
├── pyproject.toml
├── setup.py
├── requirements.txt
├── LICENSE
├── SECURITY.md
└── README.md
```

## 🧩 添加新算子

1. 在 `src/pypto_gym/models/` 下新建子目录（若是通用算子，放入 `experimental/` 对应子类）。
2. 编写 impl 文件，按需拆分 `*_impl.py` 和业务入口 `*.py`。
3. 同目录增加 `test_*.py` 或在业务入口内直接写 `def test_xxx()`（参考 `qwen3_next/`）。
4. 用 `@pytest.mark.soc("950", "910")` 标注适用 SoC，用 `@pytest.mark.world_size(N)` 标注多卡需求。
5. 补一份 `README.md` 说明算子语义、shape 范围与预期性能。

## 🔗 关联资源

- [PyPTO 主仓](https://gitcode.com/cann/pypto)
- [PyPTO 文档中心](https://pypto.gitcode.com)
- [PyPTO 贡献指南](https://gitcode.com/cann/pypto/blob/master/CONTRIBUTION.md)

## 📝 相关信息

- [许可证](LICENSE)：CANN Open Software License Agreement Version 2.0
- [安全声明](SECURITY.md)

## 联系我们

- **问题反馈**：通过 GitCode Issues 提交
- **功能建议**：通过 GitCode 讨论区交流
