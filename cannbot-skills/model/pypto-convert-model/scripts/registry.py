# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Registry of the 9 experiment models.

Each entry: id -> dict with category, source (hf/toy), repo, input spec.
input_spec drives the dummy-input generator for cross-format comparison.
"""

REGISTRY = {
    # ---- CNN ----
    "resnet18": {
        "category": "cnn",
        "source": "hf",
        "repo": "microsoft/resnet-18",
        "loader": "image-classification",
        "input": {"kind": "image", "shape": (1, 3, 224, 224)},
    },
    "mobilenet_v2": {
        "category": "cnn",
        "source": "hf",
        "repo": "google/mobilenet_v2_1.0_224",
        "loader": "image-classification",
        "input": {"kind": "image", "shape": (1, 3, 224, 224)},
    },
    "efficientnet_b0": {
        "category": "cnn",
        "source": "timm",
        "repo": "timm/efficientnet_b0.ra_in1k",
        "loader": "timm",
        "input": {"kind": "image", "shape": (1, 3, 224, 224)},
    },
    # ---- Transformer-MLP (MLP-only image classifiers) ----
    "mlp_mixer": {
        "category": "mlp",
        "source": "timm",
        "repo": "timm/mixer_b16_224.goog_in21k_ft_in1k",
        "loader": "timm",
        "input": {"kind": "image", "shape": (1, 3, 224, 224)},
    },
    "gmlp": {
        "category": "mlp",
        "source": "timm",
        "repo": "timm/gmlp_s16_224.ra3_in1k",
        "loader": "timm",
        "input": {"kind": "image", "shape": (1, 3, 224, 224)},
    },
    "resmlp": {
        "category": "mlp",
        "source": "timm",
        "repo": "timm/resmlp_12_224.fb_in1k",
        "loader": "timm",
        "input": {"kind": "image", "shape": (1, 3, 224, 224)},
    },
    # ---- MoE ----
    "switch_base_8": {
        "category": "moe",
        "source": "hf",
        "repo": "google/switch-base-8",
        "loader": "seq2seq-lm",
        "input": {"kind": "text", "prompt": "translate English to German: Hello world."},
    },
    "qwen_moe": {
        "category": "moe",
        "source": "hf",
        "repo": "Qwen/Qwen1.5-MoE-A2.7B",
        "loader": "causal-lm",
        "input": {"kind": "text", "prompt": "Hello, the quick brown"},
        "size_warning_gb": 14,
    },
    "toy_moe": {
        "category": "moe",
        "source": "toy",
        "repo": "toy_topk_moe",
        "loader": "toy",
        "input": {"kind": "tensor", "shape": (2, 64)},
    },
    "toy_soft_moe": {
        "category": "moe",
        "source": "toy",
        "repo": "toy_soft_moe",
        "loader": "toy",
        "input": {"kind": "tensor", "shape": (2, 64)},
    },
    "toy_switch_moe": {
        "category": "moe",
        "source": "toy",
        "repo": "toy_switch_moe",
        "loader": "toy",
        "input": {"kind": "tensor", "shape": (2, 64)},
    },
}

TARGET_FORMATS = ["onnx", "pt", "safetensors"]


def by_category(cat: str) -> list[str]:
    return [k for k, v in REGISTRY.items() if v["category"] == cat]
