#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: BNTrainingReduce.

Pure PyTorch golden for BatchNorm training reduce: sum(x) and sum(x*x)
over N/H/W dimensions, output per-channel [1, C, 1, 1].
"""

import torch
import torch.nn as nn

FORMULA = "sum_out[0, c, 0, 0] = sum(x[n, c, h, w]); sq_sum_out[0, c, 0, 0] = sum(x[n, c, h, w]^2)"
DYNAMIC_AXIS = ["N", "H", "W"]


class Model(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sum_out = torch.sum(x, dim=(0, 2, 3), keepdim=True)
        sq_sum_out = torch.sum(x * x, dim=(0, 2, 3), keepdim=True)
        return sum_out, sq_sum_out


def get_inputs():
    N, C, H, W = 400, 8, 1, 1
    x = torch.randn(N, C, H, W, dtype=torch.float32)
    return [x]


def get_init_inputs():
    return []
