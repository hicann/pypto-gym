#!/usr/bin/env python3
# coding: utf-8

import math
import torch
import torch.nn as nn


FORMULA = "dW[n,m]=dY*nclip*STE_in; dS[g,1]=sum(dY*(W_denorm-W_norm*nclip)*n_levels*sclip); dO[g,1]=sum(dY*(1-nclip))"
DYNAMIC_AXIS = ["N"]


class Model(nn.Module):
    def __init__(self, group_size: int = 128, bit: int = 3, eps: float = 1e-4, clip_val: float = 0.99):
        super().__init__()
        self.group_size = group_size
        self.bit = bit
        self.eps = eps
        self.clip_val = clip_val

    def forward(
        self, grad_output: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_levels = 2 ** (self.bit - 1)
        shift = 0.5
        clip_val = self.clip_val
        eps_t = torch.tensor(self.eps, device=weight.device, dtype=torch.float32)

        num_groups = scale.shape[0]
        group_size = self.group_size

        go_f32 = grad_output.view(num_groups, group_size).float()
        w_f32 = weight.view(num_groups, group_size).float()
        s_f32 = scale.float()
        o_f32 = offset.float()

        protected_scale = torch.where(s_f32 > eps_t, s_f32, eps_t)
        alpha = protected_scale * n_levels

        weight_shifted = w_f32 - o_f32
        weight_norm = weight_shifted / alpha
        weight_clipped = torch.clamp(weight_norm, -clip_val, clip_val)
        weight_scaled = weight_clipped * n_levels
        weight_shifted2 = weight_scaled - shift
        weight_rounded = torch.round(weight_shifted2)
        weight_unshifted = weight_rounded + shift
        weight_denorm = weight_unshifted / n_levels

        mask = ((weight_norm >= -clip_val) & (weight_norm <= clip_val)).float()
        inv_mask = 1.0 - mask

        scale_mask = (s_f32 > eps_t).float()

        grad_weight = (go_f32 * mask).to(torch.bfloat16).view(weight.shape)

        grad_offset = (go_f32 * inv_mask).sum(dim=1, keepdim=True)
        term_diff = weight_denorm - weight_norm * mask
        grad_alpha = (go_f32 * term_diff).sum(dim=1, keepdim=True)
        grad_scale = (grad_alpha * n_levels * scale_mask).to(torch.bfloat16)

        grad_offset = grad_offset.to(torch.bfloat16)

        return grad_weight, grad_scale, grad_offset


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
    torch.manual_seed(77)
    grad_output = torch.empty(n, m, dtype=torch.bfloat16).uniform_(-5.0, 5.0) / math.sqrt(m)
    return [grad_output, weight, scale, offset]


def get_init_inputs():
    return [128, 3, 1e-4, 0.99]
