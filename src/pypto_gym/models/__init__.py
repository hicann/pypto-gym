# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text of the License.
"""pypto-gym model & operator library.

Each subdirectory under ``pypto_gym/models`` hosts a self-contained model /
fused-operator example built on PyPTO.

Stable subdirectories:

* ``arctic``            — Arctic LSTM (sum-LSTM)
* ``deepseek_v32_exp``  — DeepSeek V3.2: Sparse Flash Attention / MLA / Lightning Indexer
* ``glm_v4_5``          — GLM V4.5 attention / MoE / FFN
* ``qat``               — Quantization Aware Training kernels
* ``qwen3_next``        — Qwen3-Next Gated Delta Rule
* ``experimental``      — Unstable / WIP operators (excluded from the default pytest path)

NOTE: Model subdirectories are *not* Python sub-packages. They use sibling imports
(e.g. ``from sum_lstm import ...``) and are designed to be loaded by ``pytest <dir>``,
which adds the directory to ``sys.path``. Adding an ``__init__.py`` to a model
directory would break those sibling imports — keep them as plain directories.
"""

STABLE_MODELS = (
    "arctic",
    "deepseek_v32_exp",
    "glm_v4_5",
    "qat",
    "qwen3_next",
)
EXPERIMENTAL_DIR = "experimental"

__all__ = ["STABLE_MODELS", "EXPERIMENTAL_DIR"]
