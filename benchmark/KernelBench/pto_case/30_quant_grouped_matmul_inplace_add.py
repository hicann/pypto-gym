#!/usr/bin/env python3
# coding: utf-8

import math

import torch
import torch.nn as nn

FORMULA = "y[g, m, n] = y[g, m, n] + scaled_matmul(a[g*k_b:(g+1)*k_b, :], b[g*k_b:(g+1)*k_b, :], scale_a, scale_b)"
DYNAMIC_AXIS = ["M", "G"]


class Model(nn.Module):
    def __init__(self, m: int = 768, k: int = 6144, n: int = 4096, num_groups: int = 32):
        super().__init__()
        self.m = m
        self.k = k
        self.n = n
        self.num_groups = num_groups

        self.w = nn.Parameter(
            torch.randn((k, n), dtype=torch.float32).uniform_(0, 1)
        )
        scale_ks = k // 64 + num_groups
        self.scaled_w = nn.Parameter(
            torch.randn((scale_ks, n, 2), dtype=torch.float32).uniform_(0, 1)
        )

    def forward(self, a: torch.Tensor, scaled_a: torch.Tensor, y_init: torch.Tensor) -> torch.Tensor:
        k_block = self.k // self.num_groups
        result = y_init.clone()

        for i in range(self.num_groups):
            begin = i * k_block
            end = (i + 1) * k_block
            scale_offset = begin // 64 + i
            scale_length = k_block // 64

            x = a[begin:end, :]
            weight = self.w[begin:end, :]
            scaled_x = scaled_a[scale_offset : scale_offset + scale_length, :, :]
            scaled_w = self.scaled_w[scale_offset : scale_offset + scale_length, :, :]

            golden_temp = self._compute_single_matmul(x, weight, scaled_x, scaled_w)
            result[i] = result[i] + golden_temp

        return result

    def _compute_single_matmul(self, x, weight, scaled_x, scaled_w):
        x_golden = torch.swapaxes(x, -1, -2)
        sx = torch.swapaxes(scaled_x, -1, -2)
        if sx.ndim == 3:
            sx = sx.reshape(sx.shape[0] * sx.shape[1], sx.shape[2])
        sx = torch.swapaxes(sx, -1, -2)

        sw = torch.swapaxes(scaled_w, -1, -2)
        if sw.ndim == 3:
            sw = sw.reshape(sw.shape[0] * sw.shape[1], sw.shape[2])

        k_dim = x_golden.shape[-1]
        if math.ceil(k_dim / 32) % 2 != 0:
            sx = sx[:, :-1]
            sw = sw[:-1, :]

        sx_bc = torch.repeat_interleave(sx, repeats=32, dim=-1)
        sw_bc = torch.repeat_interleave(sw, repeats=32, dim=-2)

        x_pad_len = sx_bc.shape[-1] - x_golden.shape[-1]
        w_pad_len = sw_bc.shape[-2] - weight.shape[-2]

        x1_pad = [0, x_pad_len]
        for _ in range(x_golden.ndim - 1):
            x1_pad += [0, 0]
        x_padded = nn.functional.pad(x_golden, x1_pad, mode="constant", value=0)

        w_pad = [0, 0]
        w_pad += [0, w_pad_len]
        for _ in range(weight.ndim - 2):
            w_pad += [0, 0]
        w_padded = nn.functional.pad(weight, w_pad, mode="constant", value=0)

        x_scaled = x_padded.to(torch.float32) * sx_bc.to(torch.float32)
        w_scaled = w_padded.to(torch.float32) * sw_bc.to(torch.float32)

        return torch.matmul(x_scaled, w_scaled)


def get_inputs():
    m = 768
    k = 6144
    n = 4096
    num_groups = 32
    k_div = k // 64 + num_groups

    a = torch.randn((k, m), dtype=torch.float32).uniform_(0, 1)
    scaled_a = torch.randn((k_div, m, 2), dtype=torch.float32).uniform_(0, 1)
    y_init = torch.randn((num_groups, m, n), dtype=torch.float32)

    return [a, scaled_a, y_init]


def get_init_inputs():
    return [768, 6144, 4096, 32]
