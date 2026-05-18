[简体中文](README.md) | English

# PyPTO-Gym

PyPTO-Gym is an operator / model example library built on the [PyPTO](https://gitcode.com/cann/pypto) programming framework. It collects a set of high-performance fused operators and typical LLM structure implementations written in PyPTO, serving as a "kernel gym" for PyPTO — a place to learn, reuse, benchmark, and compare operators.

> This repository was originally the `pypto/models/` directory, now split out as an independent repository decoupled from the PyPTO core.

## Overview

PyPTO-Gym is positioned similarly to NVIDIA's [TileGym](https://github.com/NVIDIA/TileGym) for cuTile — an operator example and benchmark library built around a programming framework. The differences are:

- **Hardware target**: Huawei Ascend AI processors
- **Programming framework**: PyPTO, based on the Tile programming model
- **Content**: End-to-end runnable fused operator examples + key LLM structures (Attention, MoE, LSTM, Delta-Rule, etc.)

## Features

- Covers key operator implementations for DeepSeek V3.2, GLM V4.5, Qwen3-Next, Qwen3-1.7B, Arctic, QAT, and other models
- Provides an `experimental/` directory with development-stage examples of fundamental ops such as Attention, Matmul, Vector, and Distributed
- Operator implementations separated from tests: kernel implementations in `src/pypto_gym/ops/pypto_tile/<model>/`, corresponding tests in `tests/ops/<model>/`
- Reuses PyPTO's built-in multi-device / multi-SoC test scheduling `conftest.py` (`@pytest.mark.soc`, `@pytest.mark.world_size`)
- Built-in Benchmark subsystem: end-to-end automated evaluation based on the KernelBench dataset, with anti-cheat verification, accuracy validation, and performance testing, supporting LLM-driven batch operator generation and regression

## Environment Setup

PyPTO-Gym does not need to be installed separately. Please first complete environment deployment following the PyPTO documentation:

- [Environment Preparation](https://gitcode.com/cann/pypto/blob/master/docs/install/prepare_environment.md): Setting up the basic environment, including obtaining and installing software packages and third-party dependencies.
- [Build and Install](https://gitcode.com/cann/pypto/blob/master/docs/install/build_and_install.md): After environment preparation, how to quickly obtain or compile the PyPTO package and install it.

Once the PyPTO environment is ready, clone this repository and set the runtime environment variables:

```bash
# Load the CANN environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Specify the NPU device ID to use (set according to available chips)
export TILE_FWK_DEVICE_ID=0

# Specify the pto-isa code path (for JIT compilation)
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
```

It is recommended to save the above as `env_setup.sh` and run `source env_setup.sh` each time.

## Quick Start

### 1. Verify Environment

```bash
source env_setup.sh
python -c "import pypto; import torch_npu; print('pypto:', pypto.__version__); print('npu available:', torch_npu.npu.is_available())"
```

### 2. Run Tests for a Single Model

```bash
source env_setup.sh

# GLM V4.5 Attention
pytest tests/ops/glm_v4_5 -v

# DeepSeek V3.2 MLA Prolog
pytest tests/ops/deepseek_v32_exp -v

# Qwen3-Next Gated Delta Rule
pytest tests/ops/qwen3_next -v

# QAT (Quantization-Aware Training)
pytest tests/ops/qat -v

# Qwen3-1.7B fused operators
pytest tests/ops/qwen3_1_7b -v
```

### 3. Run All (non-experimental) Tests

```bash
source env_setup.sh
pytest -v
```

`pytest.ini` has been configured with:
- `testpaths`: `tests/ops`
- `norecursedirs`: automatically excludes the `experimental/` directory
- `python_files`: matches `test_*.py`

To run experimental operators:

```bash
pytest src/pypto_gym/ops/pypto_tile/experimental/<op_name> -v
```

### 4. Multi-device / Specific SoC

```bash
# Specify NPU device ID (overrides TILE_FWK_DEVICE_ID env variable)
pytest tests/ops/glm_v4_5 -v --device 1

# Multi-device (2 devices) distributed example
pytest src/pypto_gym/ops/pypto_tile/experimental/distributed --device 0 1 --cards-per-case 2
```

### 5. Test Case Filtering

Test cases use `@pytest.mark.soc` to annotate supported chips (`"950"` for 910B/910C, `"910"` for 910A). The conftest.py automatically filters out incompatible cases (shown as `SKIPPED`) based on the current device's soc_version. Some large-scale cases are annotated with `@pytest.mark.skip(reason="large test case")` and need the skip annotation manually removed before running.

## Benchmark

The `benchmark/` directory provides an end-to-end automated evaluation subsystem based on the KernelBench dataset, used for batch verification of the correctness and performance of PyPTO operator generation workflows. The complete KernelBench test case set is built into `benchmark/KernelBench/` — no additional download required.

Core capabilities:
- **Batch operator generation**: Integrates with PyPTO's 7-stage LLM agent workflow (`pypto-op-orchestrator`) to automatically generate PyPTO kernel implementations for target operators
- **Multi-layer verification**: Includes anti-cheat detection (AST / pattern / runtime three layers), accuracy validation (comparison against PyTorch golden), and performance testing (end-to-end speedup ratio)
- **Real-time monitoring**: Built-in TUI dashboard for viewing case status and progress during execution
- **Report archival**: Automatically generates per-case `result.json` and global `summary.md` / `summary.json`

### Prerequisites

```bash
# Download PyPTO source code (required for the operator generation phase)
bash benchmark/scripts/download_pypto.sh
```

### Quick Run

```bash
# Run a single case using a built-in config (default: background + auto-open monitor)
python -m benchmark run --config configs/relu.yaml

# Foreground blocking mode (suitable for CI / debugging)
python -m benchmark run --config configs/relu.yaml --foreground

# Background mode without auto-entering monitor TUI
python -m benchmark run --config configs/relu.yaml --no-auto-monitor
```

To run only the PyPTO generation flow and skip KernelVerifier, set:

```yaml
verifier:
  skip: true
```

One-command quick start scripts are also available:

```bash
# Run a single case (ReLU)
bash benchmark/scripts/single_quick_start.sh

# Run the curated PyPTO benchmark set
bash benchmark/scripts/pypto_quick_start.sh
```

### Viewing Results

```bash
# After execution completes, check the report in the output directory
cat <root_dir>/report/summary.md       # Global Markdown report
cat <root_dir>/report/summary.json     # Global JSON results

# Regenerate summary reports from existing results
python -m benchmark summary <root_dir>/report

# Reattach to the real-time monitor during or after execution
python -m benchmark monitor <root_dir>/state
```

For detailed configuration instructions, architecture design, and monitor panel usage, see the documentation and `benchmark/README.md` under the `benchmark/` directory.

## FAQ

**Q: Error `key: runtime.stitch_cfgcache_size does not exist`**

This key was introduced in a newer version of PyPTO. Ensure that the PyPTO build version matches the impl file version. It is recommended to sync pypto, pypto-gym, and pto-isa to the latest version and recompile/install PyPTO.

**Q: Error `NPU out of memory`**

For some operators (e.g. sparse_flash_attention, gated_delta_rule), the `stitch_function_max_num` parameter affects workspace size, with the formula `workspace = totalSlot × (stitch_function_max_num + 1) × parallelism`. You can reduce this value in the `runtime_options` of the corresponding `@pypto.frontend.jit` decorator in the impl file (e.g., from 128 to 1) to decrease memory usage, at the cost of reduced parallelism.

**Q: Error `npu_format_cast ACL error 500001`**

TBE (Tensor Boost Engine) initialization failed, usually caused by missing Python dependencies. Run the following command to fix:

```bash
pip install scipy decorator -i https://mirrors.aliyun.com/pypi/simple/
```

**Q: Some GLM cases report `TypeError: set_pass_options() got an unexpected keyword argument 'pg_upper_bound'`**

`pg_upper_bound` has been changed to automatic derivation in the new PyPTO version. Remove this parameter from the `set_pass_options()` call. It is recommended to sync with the latest impl files from the PyPTO main repository.

**Q: Cannot find pto-isa header files when compiling PyPTO**

Make sure `PTO_TILE_LIB_CODE_PATH` points to the root of the pto-isa repository (containing the `include/` subdirectory), and that the pto-isa version is compatible with PyPTO (it is recommended to keep both repositories synced to the latest).

## Directory Structure

```
pypto-gym/
├── benchmark/                                # KernelBench automated evaluation subsystem
│   ├── configs/                              # YAML configuration files
│   ├── docs/                                 # Architecture / config / monitoring documentation
│   ├── scripts/                              # Helper scripts (PyPTO source download, quick start, etc.)
│   ├── KernelBench/                          # Built-in complete KernelBench test case set
│   ├── verifier/                             # Anti-cheat + accuracy + performance verification module
│   └── README.md
├── docs/                                    # Documentation resources (planned)
├── modeling/                                # End-to-end model execution scripts and sample inputs
│   └── transformers/                        # Qwen3-1.7B inference example
│       ├── infer.py
│       ├── bench_qwen3_1_7b.sh
│       ├── README.md
│       └── sample_inputs/
├── src/
│   └── pypto_gym/
│       ├── __init__.py
│       ├── ops/                             # Operator examples root directory
│       │   ├── pypto_tile/                  # Tile operator implementations
│       │   │   ├── arctic/                  # Arctic LSTM
│       │   │   │   ├── sum_lstm.py
│       │   │   │   └── README.md
│       │   │   ├── deepseek_v32_exp/        # DeepSeek V3.2 experimental operators
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
│       │   │   ├── qat/                     # Quantization-Aware Training
│       │   │   │   ├── qat_impl.py
│       │   │   │   └── README.md
│       │   │   ├── qwen3_1_7b/              # Qwen3-1.7B fused operators
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
│       │   └── experimental/                # Experimental operators (not run by default)
│       │       ├── attention/
│       │       ├── distributed/
│       │       ├── matmul/
│       │       ├── ops_transformer/
│       │       └── vector/
│       └── transformers/                    # HuggingFace model structure definitions
│           └── qwen3_1_7b/
├── tests/                                   # Test cases
│   └── ops/                                 # One-to-one mapping with ops/
│       ├── arctic/test_sum_lstm.py
│       ├── deepseek_v32_exp/test_*.py
│       ├── glm_v4_5/test_*.py
│       ├── qat/test_qat.py
│       ├── qwen3_1_7b/test_*.py
│       │   └── conftest.py
│       ├── qwen3_next/test_gated_delta_rule.py
│       └── README.md
├── conftest.py                              # pytest scheduling (multi-device / multi-SoC filtering)
├── pytest.ini
├── pyproject.toml
├── setup.py
├── requirements.txt
├── LICENSE
├── SECURITY.md
└── README.md
```

## Adding a New Operator

1. Create a new subdirectory under `src/pypto_gym/ops/pypto_tile/` (for general-purpose operators, place in the corresponding subcategory under `experimental/`).
2. Write the kernel implementation file; the recommended naming convention is `*_impl.py`, exposing entry functions / configuration classes.
3. Create a subdirectory with the same name under `tests/ops/` and add `test_*.py`, referencing the kernel via absolute path.
4. Use `@pytest.mark.soc("950", "910")` to annotate supported SoCs, and `@pytest.mark.world_size(N)` for multi-device requirements.
5. Add a `README.md` describing the operator semantics, shape ranges, expected performance, and the corresponding test file path.

## Related Resources

- [PyPTO Main Repository](https://gitcode.com/cann/pypto)
- [PyPTO Documentation Center](https://pypto.gitcode.com)
- [PyPTO Contribution Guide](https://gitcode.com/cann/pypto/blob/master/CONTRIBUTION.md)

## Additional Information

- [License](LICENSE): CANN Open Software License Agreement Version 2.0
- [Security Statement](SECURITY.md)

## Contact Us

- **Issue Reporting**: Submit via GitCode Issues
- **Feature Suggestions**: Discuss via GitCode Discussions
