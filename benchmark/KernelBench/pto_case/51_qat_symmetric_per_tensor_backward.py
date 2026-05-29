#!/usr/bin/env python3
# coding: utf-8

import math
import torch
import torch.nn as nn


FORMULA = "dW[n,m]=dY*mask; dS[1,1]=(sum(dY*clamp)+sum(dY*mask*(-W/s)))*sclip"
DYNAMIC_AXIS = ["N"]


class Model(nn.Module):
    def __init__(self, bit: int = 8, eps: float = 1e-4):
        super().__init__()
        self.bit = bit
        self.eps = eps

    def forward(
        self, grad_output: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        min_v = float(-2 ** (self.bit - 1))
        max_v = float(2 ** (self.bit - 1) - 1)
        eps_t = torch.tensor(self.eps, device=weight.device, dtype=torch.float32)

        go_f32 = grad_output.float()
        w_f32 = weight.float()
        s_f32 = scale.float()

        protected_scale = torch.where(s_f32 > eps_t, s_f32, eps_t)

        normalized = w_f32 / protected_scale
        rounded = torch.round(normalized)
        clamped_data = torch.clamp(rounded, min_v, max_v)

        mask = ((rounded >= min_v) & (rounded <= max_v)).float()
        scale_mask = (s_f32 >= eps_t).float()

        grad_weight = (go_f32 * mask).to(torch.bfloat16)

        grad_scale_mul = (go_f32 * clamped_data).sum(dim=(0, 1), keepdim=True)
        grad_scale_div = (go_f32 * mask * (-w_f32 / protected_scale)).sum(dim=(0, 1), keepdim=True)
        grad_scale = ((grad_scale_mul + grad_scale_div) * scale_mask).to(torch.bfloat16)

        return grad_weight, grad_scale


def get_inputs():
    n = 38344
    m = 2048
    torch.manual_seed(33)
    weight = torch.empty(n, m, dtype=torch.bfloat16).uniform_(-5.0, 5.0) / math.sqrt(m)
    scale = torch.empty(1, 1, dtype=torch.bfloat16).uniform_(-5.0, 5.0)
    torch.manual_seed(77)
    grad_output = torch.empty(n, m, dtype=torch.bfloat16).uniform_(-5.0, 5.0) / math.sqrt(m)
    return [grad_output, weight, scale]


def get_init_inputs():
    return [8, 1e-4]
