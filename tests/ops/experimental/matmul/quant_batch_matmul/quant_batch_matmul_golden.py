#!/usr/bin/env python3
# coding: utf-8
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
