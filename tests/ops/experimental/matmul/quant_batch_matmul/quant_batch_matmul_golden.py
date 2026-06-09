# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


#
# PyPTO quant_batch_matmul F-T golden reference implementation.
# Kernel and host wrapper live in quant_batch_matmul_impl.py.

import pypto
import torch

from experimental.matmul.quant_batch_matmul.quant_batch_matmul_impl import (
    QuantBatchMatmulConfig,
    QuantBatchMatmulInputs,
    compute_combined_scale,
)


def _torch_dtype_from_pypto(_dtype):
    return torch.int8


def gen_golden(
    inputs: QuantBatchMatmulInputs,
    config: QuantBatchMatmulConfig,
) -> torch.Tensor:
    """Reference F-T batch result computed with PyTorch on CPU."""
    x1 = inputs.x1.cpu()
    x2 = inputs.x2.cpu()

    acc = torch.matmul(x1.float(), x2.float().transpose(-1, -2))

    _, golden_scale = compute_combined_scale(inputs.x1_scale, inputs.x2_scale)
    out = torch.round((acc * golden_scale).clamp(-128, 127))
    return out.to(_torch_dtype_from_pypto(config.out_dtype))
