#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import math

FORMULA = "out[m, n] = x[m, k] @ W[n, k]^T"
DYNAMIC_AXIS = ["M"]


class Model(nn.Module):
    def __init__(self, num_router_experts: int = 160, hidden_size: int = 5120):
        super().__init__()
        self.gate_weight = nn.Parameter(
            torch.randn(num_router_experts, hidden_size, dtype=torch.float32) / math.sqrt(hidden_size)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.matmul(hidden_states, self.gate_weight.t())


def get_inputs():
    bs = 64
    hidden_size = 5120
    return [torch.randn(bs, hidden_size, dtype=torch.float32) / math.sqrt(hidden_size)]


def get_init_inputs():
    return [160, 5120]
