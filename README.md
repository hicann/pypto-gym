简体中文 | [English](README.en.md)

# PyPTO-Gym

PyPTO-Gym 是基于 [PyPTO](https://gitcode.com/cann/pypto) 编程框架构建的算子/模型样例仓库。它收录了一批用 PyPTO 写成的高性能融合算子与典型大模型结构实现，作为 PyPTO 的"算子体操场"，方便开发者学习、复用、压测与对比。

> 本仓原为 `pypto/models/` 目录，现已拆分为独立仓，与 PyPTO 主仓解耦演进。

## 概述

PyPTO-Gym 的定位类似 NVIDIA 的 [TileGym](https://github.com/NVIDIA/TileGym) 之于 cuTile —— 一个围绕编程框架的算子示例和基准库。区别在于：

- **硬件目标**：华为昇腾（Ascend）AI 处理器
- **编程框架**：PyPTO，基于 Tile 的编程模型
- **内容**：端到端可运行的融合算子样例 + 大模型关键结构（Attention、MoE、LSTM、Delta-Rule 等）

## 特性

- 覆盖 DeepSeek V3.2、GLM V4.5、Qwen3-Next、Qwen3-1.7B、Arctic、QAT 等模型的关键算子实现
- 提供实验性目录 `experimental/` 收录 Attention、Matmul、Vector、Distributed 等基础算子的开发态样例
- 算子实现与测试分离：kernel 实现位于 `src/pypto_gym/ops/pypto_tile/<model>/`，对应测试位于 `tests/ops/<model>/`
- 复用 PyPTO 自带的多卡/多 SoC 测试调度 `conftest.py`（`@pytest.mark.soc`、`@pytest.mark.world_size`）
- 内置 Benchmark 子系统：基于 KernelBench 数据集的端到端自动化评测，含反作弊验证、精度校验与性能测试，支持 LLM 驱动的批量算子生成与回归

## 环境准备

pypto-gym 无需单独安装。请先参照 PyPTO 文档完成环境部署：

- [环境部署](https://gitcode.com/cann/pypto/blob/master/docs/install/prepare_environment.md)：介绍项目基础环境的搭建，包括软件包和第三方依赖的获取和安装。
- [编译安装](https://gitcode.com/cann/pypto/blob/master/docs/install/build_and_install.md)：环境部署后，介绍如何快速获取或编译 PyPTO 软件包并安装。

PyPTO 环境就绪后，克隆本仓并设置运行时环境变量：

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 指定运行的 NPU 设备 ID（根据实际可用 chip 设置）
export TILE_FWK_DEVICE_ID=0

# 指定 pto-isa 代码路径（用于 JIT 编译）
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
```

推荐将上述内容保存为 `env_setup.sh`，每次执行 `source env_setup.sh` 即可。


## 快速上手

### 1. 验证环境

```bash
source env_setup.sh
python -c "import pypto; import torch_npu; print('pypto:', pypto.__version__); print('npu available:', torch_npu.npu.is_available())"
```

### 2. 运行单个模型的测试

```bash
source env_setup.sh

# GLM V4.5 Attention
pytest tests/ops/glm_v4_5 -v

# DeepSeek V3.2 MLA Prolog
pytest tests/ops/deepseek_v32_exp -v

# Qwen3-Next Gated Delta Rule
pytest tests/ops/qwen3_next -v

# QAT 量化感知训练
pytest tests/ops/qat -v

# Qwen3-1.7B 融合算子
pytest tests/ops/qwen3_1_7b -v
```

### 3. 运行全部（非 experimental）测试

```bash
source env_setup.sh
pytest -v
```

`pytest.ini` 已配置：
- `testpaths`：`tests/ops`
- `norecursedirs`：自动排除 `experimental/` 目录
- `python_files`：匹配 `test_*.py`

如需运行实验性算子：

```bash
pytest src/pypto_gym/ops/pypto_tile/experimental/<op_name> -v
```

### 4. 多卡 / 指定 SoC

```bash
# 指定 NPU device id（覆盖 TILE_FWK_DEVICE_ID 环境变量）
pytest tests/ops/glm_v4_5 -v --device 1

# 多卡（2 卡）分布式样例
pytest src/pypto_gym/ops/pypto_tile/experimental/distributed --device 0 1 --cards-per-case 2
```

### 5. 用例筛选说明

测试用例通过 `@pytest.mark.soc` 标注适用芯片（`"950"` 对应 910B/910C，`"910"` 对应 910A），conftest.py 会根据当前设备的 soc_version 自动过滤不适配的用例（显示为 `SKIPPED`）。部分规模较大的用例通过 `@pytest.mark.skip(reason="large test case")` 标注，需手动移除 skip 标注后运行。

## Benchmark

`benchmark/` 目录提供了基于 KernelBench 数据集的端到端自动化评测子系统，用于批量验证 PyPTO 算子生成流程的正确性与性能。KernelBench 完整测试用例集已内置于 `benchmark/KernelBench/`，无需额外下载。

核心能力：
- **批量算子生成**：对接 PyPTO 的 7 阶段 LLM agent 工作流（`pypto-op-orchestrator`），自动生成目标算子的 PyPTO kernel 实现
- **多层验证**：含反作弊检测（AST / 模式 / 运行时三层）、精度校验（与 PyTorch golden 对比）、性能测试（端到端加速比）
- **实时监控**：内置 TUI dashboard，运行中可查看各 case 状态与进度
- **报告沉淀**：自动产出 per-case `result.json` 与全局 `summary.md` / `summary.json`

### 前置准备

```bash
# 下载 PyPTO 源码（算子生成阶段需要）
bash benchmark/scripts/download_pypto.sh
```

### 快速运行

```bash
# 使用内置配置运行单个 case（默认后台运行 + 自动打开实时监控）
python -m benchmark run --config configs/relu.yaml

# 前台阻塞模式（适合 CI / 调试）
python -m benchmark run --config configs/relu.yaml --foreground

# 后台运行但不自动进入 monitor TUI
python -m benchmark run --config configs/relu.yaml --no-auto-monitor
```

如需只跑 PyPTO 生成流程、暂不进入 KernelVerifier，可在配置中设置：

```yaml
verifier:
  skip: true
```

仓库根目录也提供一键启动脚本：

```bash
# 运行单个 case（relu）
bash benchmark/scripts/single_quick_start.sh

# 运行 PyPTO 精选评测集
bash benchmark/scripts/pypto_quick_start.sh
```

### 查看结果

```bash
# 运行结束后，在输出目录下查看报告
cat <root_dir>/report/summary.md       # 全局 Markdown 报告
cat <root_dir>/report/summary.json     # 全局 JSON 结果

# 从既有 report 重新生成汇总报告
python -m benchmark summary <root_dir>/report

# 运行中或事后重连实时监控
python -m benchmark monitor <root_dir>/state
```

详细配置说明、架构设计、监控面板使用等参见 `benchmark/` 目录下的文档与 `benchmark/README.md`。

## 常见问题排查

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

## 目录结构

```
pypto-gym/
├── benchmark/                                # KernelBench 自动化评测子系统
│   ├── configs/                              # YAML 配置文件
│   ├── docs/                                 # 架构 / 配置 / 监控等文档
│   ├── scripts/                              # 下载 PyPTO 源码等辅助脚本
│   ├── KernelBench/                          # 内置完整 KernelBench 测试用例集
│   ├── verifier/                             # 反作弊 + 精度 + 性能验证模块
│   └── README.md
├── docs/                                    # 文档资源（规划中）
├── modeling/                                # 模型端到端执行脚本与样例输入
│   └── transformers/                        # Qwen3-1.7B 推理示例
│       ├── infer.py
│       ├── bench_qwen3_1_7b.sh
│       ├── README.md
│       └── sample_inputs/
├── src/
│   └── pypto_gym/
│       ├── __init__.py
│       ├── ops/                             # 算子样例根目录
│       │   ├── pypto_tile/                  # Tile 算子实现
│       │   │   ├── arctic/                  # Arctic LSTM
│       │   │   │   ├── sum_lstm.py
│       │   │   │   └── README.md
│       │   │   ├── deepseek_v32_exp/        # DeepSeek V3.2 实验算子
│       │   │   │   ├── lightning_indexer_prolog_quant_impl.py
│       │   │   │   ├── lightning_indexer_quant_impl.py
│       │   │   │   ├── mla_indexer_prolog_quant_impl.py
│       │   │   │   ├── mla_prolog_quant_impl.py
│       │   │   │   ├── sparse_attention_antiquant_impl.py
│       │   │   │   ├── sparse_flash_attention_quant_impl.py
│       │   │   │   ├── utils/
│       │   │   │   └── README.md
│       │   │   ├── glm_v4_5/                # GLM V4.5
│       │   │   │   ├── glm_attention_impl.py
│       │   │   │   ├── glm_attention_fusion_impl.py
│       │   │   │   ├── glm_attention_pre_quant_impl.py
│       │   │   │   ├── glm_ffn_common_interface.py
│       │   │   │   ├── glm_ffn_shared_expert_quant_impl.py
│       │   │   │   ├── glm_gate_impl.py
│       │   │   │   ├── glm_moe_fusion_impl.py
│       │   │   │   ├── glm_select_experts_impl.py
│       │   │   │   ├── utils/
│       │   │   │   ├── integrated_example.md
│       │   │   │   └── README.md
│       │   │   ├── qat/                     # 量化感知训练
│       │   │   │   ├── qat_impl.py
│       │   │   │   └── README.md
│       │   │   ├── qwen3_1_7b/              # Qwen3-1.7B 融合算子
│       │   │   │   ├── qwen3_pre_attn_fused.py
│       │   │   │   ├── qwen3_k3_post_attn.py
│       │   │   │   ├── qwen3_decode_attn.py
│       │   │   │   ├── qwen3_iter1a_kernel.py
│       │   │   │   ├── qwen3_iter1b_kernel.py
│       │   │   │   ├── qwen3_k2_qk_rope.py
│       │   │   │   ├── k3_post_attn.py
│       │   │   │   ├── __init__.py
│       │   │   │   └── README.md
│       │   │   └── qwen3_next/              # Qwen3-Next Gated Delta Rule
│       │   │       ├── gated_delta_rule_impl.py
│       │   │       └── README.md
│       │   └── experimental/                # 实验性算子（默认不跑）
│       │       ├── attention/
│       │       ├── distributed/
│       │       ├── matmul/
│       │       ├── ops_transformer/
│       │       └── vector/
│       └── transformers/                    # HuggingFace 模型结构定义
│           └── qwen3_1_7b/
├── tests/                                   # 测试用例
│   └── ops/                                 # 与 ops/ 一一对应
│       ├── arctic/test_sum_lstm.py
│       ├── deepseek_v32_exp/test_*.py
│       ├── glm_v4_5/test_*.py
│       ├── qat/test_qat.py
│       ├── qwen3_1_7b/test_*.py
│       │   └── conftest.py
│       ├── qwen3_next/test_gated_delta_rule.py
│       └── README.md
├── conftest.py                              # pytest 调度（多卡 / 多 SoC 筛选）
├── pytest.ini
├── pyproject.toml
├── setup.py
├── requirements.txt
├── LICENSE
├── SECURITY.md
└── README.md
```

## 添加新算子

1. 在 `src/pypto_gym/ops/pypto_tile/` 下新建子目录（若是通用算子，放入 `experimental/` 对应子类）。
2. 编写 kernel 实现文件，命名建议为 `*_impl.py`，对外暴露入口函数 / 配置类。
3. 在 `tests/ops/` 下新建同名子目录，添加 `test_*.py`，通过绝对路径引用 kernel。
4. 用 `@pytest.mark.soc("950", "910")` 标注适用 SoC，用 `@pytest.mark.world_size(N)` 标注多卡需求。
5. 补一份 `README.md` 说明算子语义、shape 范围与预期性能，以及对应测试文件路径。

## 关联资源

- [PyPTO 主仓](https://gitcode.com/cann/pypto)
- [PyPTO 文档中心](https://pypto.gitcode.com)
- [PyPTO 贡献指南](https://gitcode.com/cann/pypto/blob/master/CONTRIBUTION.md)

## 相关信息

- [许可证](LICENSE)：CANN Open Software License Agreement Version 2.0
- [安全声明](SECURITY.md)

## 联系我们

- **问题反馈**：通过 GitCode Issues 提交
- **功能建议**：通过 GitCode 讨论区交流
