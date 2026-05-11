#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: ApplyRMSProp.

Pure PyTorch golden for RMSProp optimizer step:
  ms_new = ms + (grad^2 - ms) * (1 - rho)
  mom_new = mom * momentum + (grad * lr) / sqrt(ms_new + epsilon)
  var_new = var - mom_new
"""

import math
import torch
import torch.nn as nn

FORMULA = "ms(t+1) = rho*ms(t) + (1-rho)*grad^2; mom(t+1) = momentum*mom(t) + lr*grad / sqrt(ms(t+1)+eps); var(t+1) = var(t) - mom(t+1)"
DYNAMIC_AXIS = ["ROWS", "COLS"]


class Model(nn.Module):
    def __init__(
        self,
        lr: float = 0.001,
        rho: float = 0.9,
        momentum: float = 0.9,
        epsilon: float = 1e-7,
    ):
        super().__init__()
        self.lr = lr
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon

    def forward(
        self,
        var: torch.Tensor,
        ms: torch.Tensor,
        mom: torch.Tensor,
        grad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_sq = grad * grad
        ms_new = ms + (grad_sq - ms) * (1.0 - self.rho)
        mom_new = mom * self.momentum + (grad * self.lr) / torch.sqrt(
            ms_new + self.epsilon
        )
        var_new = var - mom_new
        return var_new, ms_new, mom_new


def get_inputs():
    rows, cols = 16, 16
    var = torch.randn(rows, cols, dtype=torch.float32)
    ms = torch.rand(rows, cols, dtype=torch.float32)  # non-negative
    mom = torch.randn(rows, cols, dtype=torch.float32)
    grad = torch.randn(rows, cols, dtype=torch.float32)
    return [var, ms, mom, grad]


def get_init_inputs():
    return [0.001, 0.9, 0.9, 1e-7]
