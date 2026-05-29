#!/usr/bin/env python3
# coding: utf-8

import math
import torch
import torch.nn as nn


FORMULA = "out[n, m] = QAT_asym_group(x[n, m]; scale[g, 1], offset[g, 1], bit, eps, clip_val)"
DYNAMIC_AXIS = ["N"]


class Model(nn.Module):
    def __init__(self, group_size: int = 128, bit: int = 3, eps: float = 1e-4, clip_val: float = 0.99):
        super().__init__()
        self.group_size = group_size
        self.bit = bit
        self.eps = eps
        self.clip_val = clip_val

    def forward(self, weight: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        n_levels = 2 ** (self.bit - 1)
        shift = 0.5
        clip_val = self.clip_val
        eps_t = torch.tensor(self.eps, device=weight.device, dtype=torch.float32)

        orig_shape = weight.shape
        weight_in = weight.float()
        scale_in = scale.float()
        offset_in = offset.float()

        protected_scale = torch.where(scale_in > eps_t, scale_in, eps_t)
        alpha = protected_scale * n_levels

        num_groups = weight.numel() // self.group_size
        weight_2d = weight_in.view(num_groups, self.group_size)

        weight_shifted = weight_2d - offset_in
        weight_clipped = torch.clamp(weight_shifted / alpha, -clip_val, clip_val) * n_levels - shift
        weight_rounded = (weight_clipped.round() - weight_clipped).detach() + weight_clipped
        weight_unshifted = weight_rounded + shift
        weight_denorm = weight_unshifted / n_levels
        output_2d = weight_denorm * alpha + offset_in

        return output_2d.view(orig_shape).to(torch.bfloat16)


def get_inputs():
    n = 768
    m = 2048
    group_size = 128
    groups_per_row = m // group_size
    num_groups = n * groups_per_row
    torch.manual_seed(33)
    weight = torch.empty(n, m, dtype=torch.bfloat16).uniform_(-5.0, 5.0) / math.sqrt(m)
    scale = torch.empty(num_groups, 1, dtype=torch.bfloat16).uniform_(-5.0, 5.0)
    offset = torch.empty(num_groups, 1, dtype=torch.bfloat16).uniform_(-5.0, 5.0)
    return [weight, scale, offset]


def get_init_inputs():
    return [128, 3, 1e-4, 0.99]
