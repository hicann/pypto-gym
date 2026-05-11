#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn

FORMULA = "out[m, n] = sum_b( dequant(x1[b, m, k]) @ dequant(x2[b, k, n]) )"
DYNAMIC_AXIS = ["B", "M"]


class Model(nn.Module):
    def __init__(self, n: int = 128):
        super().__init__()
        self.n = n

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x1_scale: torch.Tensor,
        x2_scale: torch.Tensor,
    ) -> torch.Tensor:
        """INT8 quantized matmul with per-batch scale and batch reduce-sum.

        Args:
            x1: [batch, m, k] int8
            x2: [batch, k, n] int8
            x1_scale: [batch, m] float32
            x2_scale: [n] bfloat16

        Returns:
            [m, n] bfloat16
        """
        batch, m, k = x1.shape
        n = x2.shape[-1]

        result = torch.zeros((m, n), dtype=torch.float32, device=x1.device)

        for i in range(batch):
            matmul_result = torch.matmul(x1[i].float(), x2[i].float())
            scale_bc = x1_scale[i].unsqueeze(1).float() * x2_scale.unsqueeze(0).float()
            result += matmul_result * scale_bc

        return result.to(torch.bfloat16)


def get_inputs():
    b, m, k, n = 2, 128, 128, 128
    x1 = torch.randint(-10, 10, (b, m, k), dtype=torch.int8)
    x2 = torch.randint(-10, 10, (b, k, n), dtype=torch.int8)
    x1_scale = torch.randn((b, m), dtype=torch.float32).uniform_(0.5, 1.5)
    x2_scale = torch.randn((n,), dtype=torch.bfloat16).uniform_(0.5, 1.5)
    return [x1, x2, x1_scale, x2_scale]


def get_init_inputs():
    return [128]
