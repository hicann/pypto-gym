#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn

FORMULA = "y[b,s,h] = (x1[b,s,h] + x2[b,s,h]) * rstd[b,s,1] * gamma[h], rstd = 1/sqrt(mean(x_add^2) + eps)"
DYNAMIC_AXIS = ["B", "S"]


class Model(nn.Module):
    def __init__(self, h: int = 7168, eps: float = 1e-6):
        super().__init__()
        self.h = h
        self.eps = eps
        self.gamma = nn.Parameter(torch.randn(h, dtype=torch.bfloat16))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1_fp32 = x1.to(torch.float32)
        x2_fp32 = x2.to(torch.float32)
        gamma_fp32 = self.gamma.to(torch.float32)

        x_add_fp32 = x1_fp32 + x2_fp32
        ms_fp32 = (x_add_fp32 * x_add_fp32).mean(dim=-1, keepdim=True)
        rstd_fp32 = torch.rsqrt(ms_fp32 + self.eps)
        y_fp32 = x_add_fp32 * rstd_fp32 * gamma_fp32

        y_bf16 = y_fp32.to(torch.bfloat16)
        x_add_bf16 = x_add_fp32.to(torch.bfloat16)
        rstd_bf16 = rstd_fp32.to(torch.bfloat16)

        return y_bf16, x_add_bf16, rstd_bf16


def get_inputs():
    B, S, H = 16, 128, 7168
    x1 = torch.randn(B, S, H, dtype=torch.bfloat16)
    x2 = torch.randn(B, S, H, dtype=torch.bfloat16)
    return [x1, x2]


def get_init_inputs():
    return [7168, 1e-6]
