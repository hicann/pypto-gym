# PyPTO-Gym

## 🔥最新动态
- 2026/06：PyPTO-GYM 项目首次上线，本仓原为 [PyPTO](https://gitcode.com/cann/pypto) 仓的 `models` 目录，现已拆分为独立仓，与主仓解耦演进。

## 🚀概述

PyPTO-Gym 是基于 PyPTO 编程框架构建的样例仓库，面向华为昇腾（Ascend）AI 处理器，使用 PyPTO 的 Tile 编程模型，提供融合算子开发样例和大模型适配样例。这些样例均基于CANNBot完成设计、开发与调优工作。本仓作为 PyPTO 的"算子训练场"，方便开发者学习、复用、压测与对比。

- **硬件目标**：华为昇腾（Ascend）AI 处理器
- **编程框架**：PyPTO，基于 Tile 的编程模型
- **内容**：融合算子样例 + 大模型适配样例

## 核心特性

- **大模型核心算子样例**：覆盖 DeepSeek V3.2 / V4、GLM V4.5、Gemma4-31B-it、Qwen3 系列（1.7B / 3.5-9B / 3.6-27B / Next / VL-8B）、LLaDA2-MoE 等模型的关键算子实现
- **实验性算子样例**：基于实验性目录 `experimental`，收录 Attention、Matmul、Vector、Distributed 等基础算子的开发态样例
- **大模型适配样例**：提供融合算子入网适配样例，以及端到端模型推理与性能基准脚本
- **Agent能力**：提供模型整网适配skills，提升大模型对接易用性

## 环境准备

当前仓库支持的 CANN 版本如下：

| CANN 版本 | 支持情况 |
|----------|----------|
| 9.1.0 | 支持 |

pypto-gym 无需单独安装。请先参照 PyPTO 文档完成环境部署：

- [环境部署](https://gitcode.com/cann/pypto/blob/master/docs/zh/install/prepare_environment.md)：介绍项目基础环境的搭建，包括软件包和第三方依赖的获取和安装。
- [编译安装](https://gitcode.com/cann/pypto/blob/master/docs/zh/install/build_and_install.md)：环境部署后，介绍如何快速获取或编译 PyPTO 软件包并安装。

PyPTO 环境就绪后，克隆本仓并设置运行时环境变量：

```bash
# 加载 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 指定运行的 NPU 设备 ID（根据实际可用 chip 设置）
export TILE_FWK_DEVICE_ID=0

# 指定 pto-isa 代码路径（当pypto使用源码编译安装时需要关注）
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
```

推荐将上述内容保存为 `env_setup.sh`，每次执行 `source env_setup.sh` 即可。

## ⚡️快速上手

### 1. 运行部分算子测试

```bash
# 直接执行 GLM V4.5 Attention
python tests/ops/glm_v4_5/test_glm_attention.py

# 使用 pytest 执行 GLM V4.5 Attention
pytest tests/ops/glm_v4_5/test_glm_attention.py

# 使用 pytest 执行 GLM V4.5 目录所有算子测试
pytest tests/ops/glm_v4_5 -v --forked
```

### 2. 运行全部算子测试

```bash
pytest -v --forked
```

具体测试范围参考 [pytest.ini](./pytest.ini) 配置

### 3. 多卡 / 指定 SoC

```bash
# 指定 NPU device id（覆盖 TILE_FWK_DEVICE_ID 环境变量）
pytest tests/ops/glm_v4_5 -v --forked --device 1

# 多卡（2 卡）分布式样例
pytest tests/ops/experimental/distributed --device 0 1 --cards-per-case 2
```

### 4. 用例筛选说明

测试用例通过 `@pytest.mark.soc` 标注适用芯片，conftest.py 会根据当前设备的 soc_version 自动过滤不适配的用例（显示为 `SKIPPED`）。部分规模较大的用例默认已使用 `@pytest.mark.skip(reason="large test case")` 标注，需手动移除 skip 标注后运行。

## 常见问题排查

**Q: 报错 `NPU out of memory`**

部分算子（如 sparse_flash_attention、gated_delta_rule）的 `stitch_function_max_num` 参数会影响 workspace 大小，公式为 `workspace = totalSlot × (stitch_function_max_num + 1) × parallelism`。可在 impl 文件对应 `@pypto.frontend.jit` 的 `runtime_options` 中降低该值（如从 128 降至 1）以减少内存占用，代价是降低并行度。

**Q: 报错 `npu_format_cast ACL error 500001`**

TBE（Tensor Boost Engine）初始化失败，通常由缺少 Python 依赖导致。执行以下命令修复：

```bash
pip install scipy decorator -i https://mirrors.aliyun.com/pypi/simple/
```

**Q: 编译 PyPTO 时找不到 pto-isa 头文件**

确认 `PTO_TILE_LIB_CODE_PATH` 指向 pto-isa 仓库根目录（含 `include/` 子目录），且 pto-isa 版本与 PyPTO 兼容（建议两仓同步到最新）。

## 目录结构

```
pypto-gym/
├── docs/                                    # 文档资源
├── modeling/                                # 模型端到端执行脚本
│   └── transformers/                        # 大模型推理示例
│       ├── infer.py                         # 通用推理入口
│       ├── deepseek-v2-lite-chat/           # DeepSeek V2 Lite Chat
│       ├── gemma4_31b_it/                   # Gemma4-31B-it
│       ├── llada2_moe/                      # LLaDA2-MoE
│       ├── qwen3_1_7b/                      # Qwen3-1.7B
│       ├── qwen3_5_9b/                      # Qwen3.5-9B
│       ├── ...
│       └── sample_inputs/                   # 公共样例输入
├── src/
│   └── pypto_gym/
│       ├── ops/                             # 算子样例根目录
│       │   ├── pypto_tile/                  # Tile 算子实现
│       │   │   ├── arctic/                  # Arctic LSTM Speculator
│       │   │   ├── deepseek_v2_lite_chat/   # DeepSeek V2 Lite Chat MLA Prolog
│       │   │   ├── deepseek_v32_exp/        # DeepSeek V3.2 MLA / Sparse Attention / Lightning Indexer
│       │   │   ├── deepseek_v4/             # DeepSeek V4 MLA / Compressor / Window Attention
│       │   │   ├── gemma4_31b_it/           # Gemma4-31B-it GQA Decode Attention / Softmax
│       │   │   ├── glm_v4_5/                # GLM V4.5 Attention / MoE / FFN / Gate
│       │   │   ├── llada2_moe/              # LLaDA2-MoE Gate / Expert FFN / Grouped GEMM
│       │   │   ├── qat/                     # 量化感知训练（对称/非对称，per-tensor/channel/group）
│       │   │   ├── qwen3_1_7b/              # Qwen3-1.7B RMSNorm + RoPE
│       │   │   ├── qwen3_5_9b/              # Qwen3.5-9B Gated Delta Rule
│       │   │   ├── ...
│       │   │   └── experimental/            # 实验性算子
│       │   │       ├── attention/           # BSA / Chunked GDR / Flash Attention 等
│       │   │       ├── distributed/         # 分布式配置与分析
│       │   │       ├── matmul/              # GMM / 量化矩阵乘 / MXFP8 等
│       │   │       ├── ops_transformer/     # Flash Attention / MLA Prolog / SwiGLU / Sparse Attention 等
│       │   │       └── vector/              # RMSNorm / RoPE / AdamW / Sigmoid / MoE 等
│       │   └── ...
│       └── transformers/                    # 各个模型结构定义
│           ├── gemma4_31b_it/
│           ├── llada2_moe/
│           ├── qwen3_1_7b/
│           ├── qwen3_5_9b/
│           ├── ...
│           └── spatial_ssrl_3b/
├── tests/                                   # 算子测试用例
│   └── ops/                                 # 与 src/pypto_gym/ops/ 对应
│       ├── arctic/                          # Arctic LSTM 算子测试
│       ├── deepseek_v32_exp/                # DeepSeek V3.2 算子测试
│       ├── deepseek_v4/                     # DeepSeek V4 算子测试
│       ├── gemma4_31b_it/                   # Gemma4-31B-it 算子测试
│       ├── glm_v4_5/                        # GLM V4.5 算子测试
│       ├── llada2_moe/                      # LLaDA2-MoE 算子测试
│       ├── qat/                             # QAT 算子测试
│       ├── utils/                           # 测试工具函数
│       └── experimental/                    # 实验性算子测试
│           ├── attention/
│           ├── distributed/
│           ├── matmul/
│           ├── ops_transformer/
│           └── vector/
├── conftest.py                              # pytest 调度（多卡 / 多 SoC 筛选）
├── CONTRIBUTING.md                          # 贡献指南
├── pytest.ini
├── pyproject.toml
├── setup.py
├── requirements.txt
├── LICENSE
├── SECURITY.md
└── README.md
```

## 算子总览

| 模型目录 | 算子 | 说明 |
|---------|------|------|
| `arctic/` | sum_lstm | LSTM 推测器（Arctic-Inference），融合输入、RMSNorm、GELU、门控与细胞状态更新 |
| `deepseek_v2_lite_chat/` | mla_prolog | MLA Prolog 预计算 |
| `deepseek_v32_exp/` | mla_prolog_quant, lightning_indexer_prolog_quant, sparse_flash_attention_quant, sparse_attention_antiquant, mla_indexer_prolog_quant, lightning_indexer | MLA Prolog、Lightning Indexer、稀疏注意力等 6 个量化/非量化算子 |
| `deepseek_v4/` | mla_prolog_v4, mla_prolog_quant_v4, lightning_indexer_prolog_quant_v4, compressor, compress_flash_attention, sparse_compress_flash_attention, win_attention, hc_pre | MLA Prolog、压缩器、压缩/稀疏 Flash Attention、窗口注意力等 8 个算子 |
| `gemma4_31b_it/` | gqa_decode_attn, attn_softmax | GQA 解码注意力（KV 头均值，带宽降低 75%）、3-pass Softmax |
| `glm_v4_5/` | attention_pre_quant, attention, attention_fusion, gate, select_experts, ffn_shared_expert_quant, moe_fusion | 注意力（含量化前处理与融合）、MoE 门控/专家选择/FFN/融合等 7 个算子 |
| `gutenocr_3b/` | swiglu_mlp, rms_norm, mrope | SwiGLU MLP、RMSNorm、多模态 RoPE |
| `llada2_moe/` | gate_select, expert_ffn, moe_grouped_gemm | MoE 门控选择、单专家 FFN、分组 GEMM（9.2x 端到端加速） |
| `phi_3_mini_4k_instruct/` | rms_norm | RMSNorm（D=3072），支持 ACLGraph |
| `qat/` | symmetric_per_tensor, symmetric_per_channel, asymmetric_per_group | 量化感知训练三模式（前向 + 反向） |
| `qwen3_1_7b/` | rms_norm_rope | RMSNorm + RoPE 融合 |
| `qwen3_5_9b/` | gated_delta_rule | Chunk Gated Delta Rule 注意力（D=128, Nv=32, Nqk=16） |
| `qwen3_6_27b/` | gated_delta_rule | Chunk Gated Delta Rule 注意力（D=128, Nv=48, Nqk=16） |
| `qwen3_next/` | gated_delta_rule | Chunk Gated Delta Rule，线性复杂度 O(n)，支持 1K-1M+ 序列长度 |
| `qwen3_vl_8b_instruct_.../` | rms_norm | RMSNorm（hidden_size=2048） |
| `spatial_ssrl_3b/` | rms_norm, rope | RMSNorm + RoPE（Vision 2D / 多模态 3D） |
| `experimental/vector/` | ApplyAdamWV2, GatherPaKvCache, QkvRmsNormRopeCache | AdamW 更新、Paged Attention KV cache gather、QKV RMSNorm RoPE cache 融合 |

## 添加新算子

1. 在 `src/pypto_gym/ops/pypto_tile/` 下新建子目录（若是通用算子，放入 `experimental/` 对应子类）。
2. 编写 kernel 实现文件，命名建议为 `*_impl.py`，对外暴露入口函数 / 配置类。
3. 在 `tests/ops/` 下新建同名子目录，添加 `test_*.py`，通过绝对路径引用 kernel。
4. 用 `@pytest.mark.soc("950", "910")` 标注适用 SoC，用 `@pytest.mark.world_size(N)` 标注多卡需求。
5. 编写 `README.md` 说明算子语义、shape 范围与预期性能，以及对应测试文件路径。

详细规范参见 [CONTRIBUTING.md](CONTRIBUTING.md)。

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
