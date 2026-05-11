#!/usr/bin/env python3
# coding: utf-8

import math

import torch
import torch.nn as nn

FORMULA = "out[m, n] = grouped_mxfp8_scaled_matmul(a[m, :], b[group(m), :, :], scaled_a, scaled_b)"
DYNAMIC_AXIS = ["M"]


class Model(nn.Module):
    def __init__(self, k: int = 512, n: int = 7168, group_list: list = None):
        super().__init__()
        if group_list is None:
            group_list = [7, 9]
        self.group_list = group_list
        self.num_groups = len(group_list)

        self.b = nn.Parameter(
            torch.randn((self.num_groups, k, n), dtype=torch.float32).uniform_(0, 1)
        )
        self.scaled_b = nn.Parameter(
            torch.randn((self.num_groups, k // 64, n, 2), dtype=torch.float32).uniform_(0, 1)
        )

    def forward(self, a: torch.Tensor, scaled_a: torch.Tensor) -> torch.Tensor:
        result = []
        begin = 0
        end = 0

        for i in range(self.num_groups):
            if self.group_list[i] <= 0:
                continue
            begin = end
            end = end + self.group_list[i]

            x = a[begin:end, :]
            weight = self.b[i]
            scaled_x = scaled_a[begin:end, :, :]
            scaled_w = self.scaled_b[i]

            golden_temp = self._compute_single_matmul(x, weight, scaled_x, scaled_w)
            result.append(golden_temp)

        return torch.cat(result, dim=0)

    def _compute_single_matmul(
        self, x, weight, scaled_x, scaled_w,
    ):
        if scaled_x.ndim == 3:
            scaled_x = scaled_x.reshape(scaled_x.shape[0], scaled_x.shape[1] * scaled_x.shape[2])

        scaled_w = torch.swapaxes(scaled_w, -1, -2)
        if scaled_w.ndim == 3:
            scaled_w = scaled_w.reshape(scaled_w.shape[0] * scaled_w.shape[1], scaled_w.shape[2])

        k_dim = x.shape[-1]
        if math.ceil(k_dim / 32) % 2 != 0:
            scaled_x = scaled_x[:, :-1]
            scaled_w = scaled_w[:-1, :]

        scaled_x_bc = torch.repeat_interleave(scaled_x, repeats=32, dim=-1)
        scaled_w_bc = torch.repeat_interleave(scaled_w, repeats=32, dim=-2)

        x_pad_len = scaled_x_bc.shape[-1] - x.shape[-1]
        w_pad_len = scaled_w_bc.shape[-2] - weight.shape[-2]

        x1_pad = [0, x_pad_len]
        for _ in range(x.ndim - 1):
            x1_pad += [0, 0]
        x_padded = nn.functional.pad(x, x1_pad, mode="constant", value=0)

        w_pad = [0, 0]
        w_pad += [0, w_pad_len]
        for _ in range(weight.ndim - 2):
            w_pad += [0, 0]
        w_padded = nn.functional.pad(weight, w_pad, mode="constant", value=0)

        x_scaled = x_padded.to(torch.float32) * scaled_x_bc.to(torch.float32)
        w_scaled = w_padded.to(torch.float32) * scaled_w_bc.to(torch.float32)

        return torch.matmul(x_scaled, w_scaled)


def get_inputs():
    m = 16
    k = 512
    a = torch.randn((m, k), dtype=torch.float32).uniform_(0, 1)
    scaled_a = torch.randn((m, k // 64, 2), dtype=torch.float32).uniform_(0, 1)
    return [a, scaled_a]


def get_init_inputs():
    return [512, 7168, [7, 9]]
