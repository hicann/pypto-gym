#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: ScatterNdSub.

Pure PyTorch golden for scatter_nd_sub: target[indices[i]] -= updates[i].
Supports repeated indices with accumulation.
"""

import torch
import torch.nn as nn

FORMULA = "target[indices[i], :] = target[indices[i], :] - updates[i, :]"
DYNAMIC_AXIS = ["K"]


class Model(nn.Module):
    def forward(
        self,
        target: torch.Tensor,
        indices: torch.Tensor,
        updates: torch.Tensor,
    ) -> torch.Tensor:
        result = target.clone()
        for i in range(indices.shape[0]):
            idx = indices[i].item()
            result[idx] = result[idx] - updates[i]
        return result


def get_inputs():
    M = 10000
    N = 16
    K = 256

    target = torch.rand(M, N, dtype=torch.float32) * 10.0
    indices = torch.randint(0, min(M, N), (K, 1), dtype=torch.int32)
    updates = torch.rand(K, N, dtype=torch.float32) * 2.0

    return [target, indices, updates]


def get_init_inputs():
    return []
