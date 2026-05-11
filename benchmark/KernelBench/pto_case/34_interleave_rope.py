#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn

FORMULA = "y[b,n,s,2i] = x[b,n,s,2i]*cos[b,1,s_cs,2i] - x[b,n,s,2i+1]*sin[b,1,s_cs,2i]; y[b,n,s,2i+1] = x[b,n,s,2i]*sin[b,1,s_cs,2i+1] + x[b,n,s,2i+1]*cos[b,1,s_cs,2i+1]"
DYNAMIC_AXIS = ["B", "S"]


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.to(torch.float32)
        cos_f = cos.to(torch.float32)
        sin_f = sin.to(torch.float32)

        x_even = x_f[..., 0::2]
        x_odd = x_f[..., 1::2]
        cos_even = cos_f[..., 0::2]
        cos_odd = cos_f[..., 1::2]
        sin_even = sin_f[..., 0::2]
        sin_odd = sin_f[..., 1::2]

        y_even = x_even * cos_even - x_odd * sin_even
        y_odd = x_even * sin_odd + x_odd * cos_odd
        y = torch.cat((y_even, y_odd), dim=-1)

        return y.to(orig_dtype)


def get_inputs():
    B, N, S, D = 1, 128, 2048, 64
    x = torch.randn(B, N, S, D, dtype=torch.bfloat16)
    cos = torch.randn(B, 1, S, D, dtype=torch.bfloat16)
    sin = torch.randn(B, 1, S, D, dtype=torch.bfloat16)
    return [x, cos, sin]


def get_init_inputs():
    return []
