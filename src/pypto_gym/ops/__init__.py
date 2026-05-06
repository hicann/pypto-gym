# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text of the License.
"""pypto-gym operator library.

Tile operator implementations live under ``pypto_gym/ops/pypto_tile/``,
with corresponding tests under ``tests/ops/`` - one directory per model.

Stable subdirectories under ``pypto_tile/``:

* ``arctic``            — Arctic LSTM (sum-LSTM)
* ``deepseek_v32_exp``  — DeepSeek V3.2: Sparse Flash Attention / MLA / Lightning Indexer
* ``glm_v4_5``          — GLM V4.5 attention / MoE / FFN
* ``qat``               — Quantization Aware Training kernels
* ``qwen3_1_7b``        — Qwen3-1.7B fused operators
* ``qwen3_next``        — Qwen3-Next Gated Delta Rule
* ``experimental``      — Unstable / WIP operators (excluded from the default pytest path)

LLM model definitions built on top of these operators live under
``pypto_gym/transformers/``. The execution scripts (inference entrypoints,
benchmarks, Dockerfiles, sample inputs) sit at the repo root under ``modeling/``,
mirroring TileGym's top-level ``modeling/`` layout.

NOTE: The ``pypto_tile/`` directory is an implicit namespace package — it has
no ``__init__.py`` so that pytest ``norecursedirs`` can exclude ``experimental/``
without breaking imports when tests are run under ``pytest tests/ops/``.
"""

STABLE_OPS = (
    "arctic",
    "deepseek_v32_exp",
    "glm_v4_5",
    "qat",
    "qwen3_1_7b",
    "qwen3_next",
)
EXPERIMENTAL_DIR = "experimental"

__all__ = ["STABLE_OPS", "EXPERIMENTAL_DIR"]
