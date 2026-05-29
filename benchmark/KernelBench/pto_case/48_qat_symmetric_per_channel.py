#!/usr/bin/env python3
# coding: utf-8

import math
import torch
import torch.nn as nn


FORMULA = "out[n, m] = clamp(round(x[n, m] / s[n, 1]), min_v, max_v) * s[n, 1]"
DYNAMIC_AXIS = ["N"]


class Model(nn.Module):
    def __init__(self, bit: int = 4, eps: float = 1e-4):
        super().__init__()
        self.bit = bit
        self.eps = eps

    def forward(self, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        min_v = float(-2 ** (self.bit - 1))
        max_v = float(2 ** (self.bit - 1) - 1)
        eps_t = torch.tensor(self.eps, device=weight.device, dtype=torch.float32)

        w_f32 = weight.float()
        s_f32 = scale.float()
        protected_scale = torch.where(s_f32 > eps_t, s_f32, eps_t)

        weight_normalized = w_f32 / protected_scale
        weight_rounded = (weight_normalized.round() - weight_normalized).detach() + weight_normalized
        clamped = torch.clamp(weight_rounded, min_v, max_v)
        output = clamped * protected_scale
        return output.to(torch.bfloat16)


def get_inputs():
    n = 38344
    m = 2048
    torch.manual_seed(33)
    weight = torch.empty(n, m, dtype=torch.bfloat16).uniform_(-5.0, 5.0) / math.sqrt(m)
    scale = torch.empty(n, 1, dtype=torch.bfloat16).uniform_(-5.0, 5.0)
    return [weight, scale]


def get_init_inputs():
    return [4, 1e-4]
