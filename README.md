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

## 📦 环境准备

### 系统要求

| 组件 | 版本要求 |
|------|---------|
| 华为昇腾 CANN | ≥ 8.5.0 |
| Python | 3.10+ |
| PyTorch | 2.7.x |
| torch_npu | 与 PyTorch 版本配套 |
| [PyPTO](https://gitcode.com/cann/pypto) | ≥ 0.2.1（需从源码编译安装） |
| [pto-isa](https://gitcode.com/cann/pto-isa) | 与 PyPTO 主仓同步的最新版本 |

### 第一步：安装 CANN 环境

按照昇腾官方文档安装 CANN toolkit，安装完成后加载环境变量：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 第二步：安装 torch_npu

torch_npu 需要与 PyTorch 版本严格对应，按照昇腾官方发布的配套表安装：

```bash
pip install torch torch_npu -i https://mirrors.aliyun.com/pypi/simple/
```

torch_npu 依赖 `scipy` 和 `decorator`，需一并安装：

```bash
pip install scipy decorator -i https://mirrors.aliyun.com/pypi/simple/
```

### 第三步：克隆并编译安装 pto-isa

pto-isa 提供底层 ISA 接口头文件，PyPTO 编译时依赖。

```bash
git clone https://gitcode.com/cann/pto-isa.git
```

> pto-isa 无需单独编译，仅需将仓库路径通过环境变量 `PTO_TILE_LIB_CODE_PATH` 指定给 PyPTO 编译系统即可。

### 第四步：克隆并编译安装 PyPTO

```bash
git clone https://gitcode.com/cann/pypto.git
cd pypto

# 设置 pto-isa 路径（必须在编译前设置）
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa

# 编译 C++ 扩展（约 1~2 分钟）
python setup.py build_ext --inplace

# 可编辑模式安装
pip install -e . -i https://mirrors.aliyun.com/pypi/simple/
```

### 第五步：设置运行时环境变量

每次运行前需设置以下环境变量：

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 指定运行的 NPU 设备 ID（根据实际可用 chip 设置）
export TILE_FWK_DEVICE_ID=0

# 指定 pto-isa 代码路径（用于 JIT 编译）
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
```

推荐将上述内容保存为 `env_setup.sh`，每次执行 `source env_setup.sh` 即可。

### 第六步：安装 pypto-gym

```bash
git clone https://gitcode.com/cann/pypto-gym.git
cd pypto-gym
pip install -e . -i https://mirrors.aliyun.com/pypi/simple/
```

开发模式附加依赖：

```bash
pip install -e ".[dev]" -i https://mirrors.aliyun.com/pypi/simple/
```

## ⚡️ 快速上手

### 1. 验证环境

```bash
source env_setup.sh
python -c "import pypto; import torch_npu; print('pypto:', pypto.__version__); print('npu available:', torch_npu.npu.is_available())"
```

### 2. 运行单个模型的测试

```bash
source env_setup.sh

# GLM V4.5 Attention
pytest src/pypto_gym/models/glm_v4_5/glm_attention.py -v

# DeepSeek V3.2 MLA Prolog
pytest src/pypto_gym/models/deepseek_v32_exp/deepseekv32_mla_prolog_quant.py -v

# Qwen3-Next Gated Delta Rule
pytest src/pypto_gym/models/qwen3_next -v

# QAT 量化感知训练
pytest src/pypto_gym/models/qat -v
```

### 3. 运行全部（非 experimental）测试

```bash
source env_setup.sh
pytest -v
```

`pytest.ini` 已配置：
- `testpaths`：`src/pypto_gym/models` 和 `tests`
- `norecursedirs`：自动排除 `experimental/` 目录
- `python_files`：匹配 `test_*.py glm_*.py deepseekv32_*.py qwen3_next_*.py`

如需运行实验性算子：

```bash
pytest src/pypto_gym/models/experimental/<op_name> -v
```

### 4. 多卡 / 指定 SoC

```bash
# 指定 NPU device id（覆盖 TILE_FWK_DEVICE_ID 环境变量）
pytest src/pypto_gym/models/glm_v4_5 -v --device 1

# 多卡（2 卡）分布式样例
pytest src/pypto_gym/models/experimental/distributed --device 0 1 --cards-per-case 2
```

### 5. 用例筛选说明

测试用例通过 `@pytest.mark.soc` 标注适用芯片（`"950"` 对应 910B/910C，`"910"` 对应 910A），conftest.py 会根据当前设备的 soc_version 自动过滤不适配的用例（显示为 `SKIPPED`）。部分规模较大的用例通过 `@pytest.mark.skip(reason="large test case")` 标注，需手动移除 skip 标注后运行。

## 🔧 常见问题排查

**Q: 报错 `key: runtime.stitch_cfgcache_size does not exist`**

该 key 为新版 PyPTO 引入，需确保 PyPTO 编译版本与 impl 文件版本一致。建议始终保持 pypto、pypto-gym、pto-isa 三仓同步到最新版本后重新编译安装 PyPTO。

**Q: 报错 `NPU out of memory`**

部分算子（如 sparse_flash_attention、gated_delta_rule）的 `stitch_function_max_num` 参数会影响 workspace 大小，公式为 `workspace = totalSlot × (stitch_function_max_num + 1) × parallelism`。可在 impl 文件对应 `@pypto.frontend.jit` 的 `runtime_options` 中降低该值（如从 128 降至 1）以减少内存占用，代价是降低并行度。

**Q: 报错 `npu_format_cast ACL error 500001`**

TBE（Tensor Boost Engine）初始化失败，通常由缺少 Python 依赖导致。执行以下命令修复：

```bash
pip install scipy decorator -i https://mirrors.aliyun.com/pypi/simple/
```

**Q: 部分 GLM 用例报 `TypeError: set_pass_options() got an unexpected keyword argument 'pg_upper_bound'`**

`pg_upper_bound` 在新版 PyPTO 中已改为自动推导，从 `set_pass_options()` 调用中移除该参数即可。建议同步 pypto 主仓最新的 impl 文件。

**Q: 编译 PyPTO 时找不到 pto-isa 头文件**

确认 `PTO_TILE_LIB_CODE_PATH` 指向 pto-isa 仓库根目录（含 `include/` 子目录），且 pto-isa 版本与 PyPTO 兼容（建议两仓同步到最新）。

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
