#!/usr/bin/env python3
# coding: utf-8

import math

import torch
import torch.nn as nn

FORMULA = "out[row_index[t], n] += logit[t] * scaled_matmul(x1[t,:], x2[expert(t),:,:], scale, pertoken_scale)"
DYNAMIC_AXIS = ["M", "B"]


class Model(nn.Module):
    def __init__(
        self,
        batch: int = 128,
        k: int = 6144,
        n: int = 4096,
        num_experts: int = 32,
        transpose_x2: bool = True,
        shared_input_weight: float = 1.0,
        shared_input_offset: int = 0,
        has_logit: bool = True,
        has_shared_input: bool = True,
    ):
        super().__init__()
        self.batch = batch
        self.k = k
        self.n = n
        self.num_experts = num_experts
        self.transpose_x2 = transpose_x2
        self.shared_input_weight = shared_input_weight
        self.shared_input_offset = shared_input_offset
        self.has_logit = has_logit
        self.has_shared_input = has_shared_input

        scale_k = (k + 63) // 64

        if transpose_x2:
            self.x2 = nn.Parameter(
                torch.randn((num_experts, n, k), dtype=torch.float32).uniform_(0, 1)
            )
            self.scale_w = nn.Parameter(
                torch.randn((n, scale_k, 2), dtype=torch.float32).uniform_(0, 1)
            )
        else:
            self.x2 = nn.Parameter(
                torch.randn((num_experts, k, n), dtype=torch.float32).uniform_(0, 1)
            )
            self.scale_w = nn.Parameter(
                torch.randn((scale_k, n, 2), dtype=torch.float32).uniform_(0, 1)
            )

    def forward(
        self,
        x1: torch.Tensor,
        pertoken_scale: torch.Tensor,
        group_list: torch.Tensor,
        shared_input: torch.Tensor,
        logit: torch.Tensor,
        row_index: torch.Tensor,
    ) -> torch.Tensor:
        m = x1.shape[0]
        out = torch.zeros((self.batch, self.n), dtype=torch.float32, device=x1.device)

        if self.has_shared_input:
            shared_start = self.shared_input_offset
            shared_end = shared_start + shared_input.shape[0]
            out[shared_start:shared_end, :] = (
                out[shared_start:shared_end, :]
                + shared_input.to(torch.float32) * self.shared_input_weight
            )

        for expert_idx in range(self.num_experts):
            start = int(group_list[:expert_idx].sum().item())
            end = start + int(group_list[expert_idx].item())
            if end <= start:
                continue

            x = x1[start:end, :]
            p_scale = pertoken_scale[start:end, :, :]
            weight = self.x2[expert_idx]
            ws = self.scale_w

            if self.transpose_x2:
                weight = weight.transpose(-1, -2).contiguous()
                ws = ws.transpose(0, 1).contiguous()

            mm_result = self._compute_mxfp8_matmul(x, weight, p_scale, ws)

            if self.has_logit:
                mm_result = mm_result * logit[start:end].to(torch.float32).unsqueeze(-1)

            out.index_add_(0, row_index[start:end].to(torch.int64), mm_result)

        return out

    def _compute_mxfp8_matmul(self, x, weight, xs, ws):
        if xs.ndim == 3:
            xs = xs.reshape(xs.shape[0], xs.shape[1] * xs.shape[2])

        ws = torch.swapaxes(ws, -1, -2)
        if ws.ndim == 3:
            ws = ws.reshape(ws.shape[0] * ws.shape[1], ws.shape[2])

        k_dim = x.shape[-1]
        if math.ceil(k_dim / 32) % 2 != 0:
            xs = xs[:, :-1]
            ws = ws[:-1, :]

        xs = torch.repeat_interleave(xs, repeats=32, dim=-1).to(torch.float32)
        ws = torch.repeat_interleave(ws, repeats=32, dim=-2).to(torch.float32)

        return torch.matmul(x.to(torch.float32) * xs, weight.to(torch.float32) * ws)


def get_inputs():
    batch = 128
    m = 768
    k = 6144
    n = 4096
    num_experts = 32
    scale_k = (k + 63) // 64

    x1 = torch.randn((m, k), dtype=torch.float32).uniform_(0, 1)
    pertoken_scale = torch.randn((m, scale_k, 2), dtype=torch.float32).uniform_(0, 1)

    base = m // num_experts
    counts = torch.full((num_experts,), base, dtype=torch.int64)
    counts[: m - base * num_experts] += 1
    group_list = counts

    shared_input = torch.randn((batch, n), dtype=torch.float32).uniform_(-0.5, 0.5).to(torch.bfloat16)
    logit = torch.randn((m,), dtype=torch.float32).uniform_(0, 1)
    row_index = torch.arange(m, dtype=torch.int64) % batch

    return [x1, pertoken_scale, group_list, shared_input, logit, row_index]


def get_init_inputs():
    return [128, 6144, 4096, 32, True, 1.0, 0, True, True]
