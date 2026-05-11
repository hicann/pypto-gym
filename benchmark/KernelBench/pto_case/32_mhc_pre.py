#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

FORMULA = "h_in[b,d], h_post[b,n], h_res[b,n,n] = mhc_pre(x[b,n,d], phi, alpha, bias)"
DYNAMIC_AXIS = ["B"]


class Model(nn.Module):
    def __init__(self, n: int = 8, d: int = 5120, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.n = n
        self.d = d
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps

        n2_plus_2n = n * n + 2 * n
        nd = n * d

        self.phi = nn.Parameter(torch.randn(n2_plus_2n, nd, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.randn(3, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(n2_plus_2n, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        T, N, D = x.shape
        x_flat = x.reshape(T, N * D).float()

        inv_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)

        h_mix = F.linear(x_flat, self.phi.float())
        weight = h_mix * inv_rms

        h_pre, h_post, h_res = weight.split([N, N, N * N], dim=-1)
        h_res = h_res.unflatten(-1, (N, N))

        h_pre_out = F.sigmoid(h_pre * self.alpha[0] + self.bias[:N].unsqueeze(0)) + self.hc_eps
        y = torch.sum(h_pre_out.unsqueeze(-1) * x_flat.unflatten(dim=-1, sizes=(N, -1)), dim=1)

        h_post_out = 2 * F.sigmoid(h_post * self.alpha[1] + self.bias[N:2*N].unsqueeze(0))
        h_res_out = h_res * self.alpha[2] + self.bias[2*N:].view(N, N).unsqueeze(0)

        return y.to(torch.bfloat16), h_post_out, h_res_out


def get_inputs():
    bs = 1024
    N = 8
    D = 5120
    x = torch.randn(bs, N, D, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [8, 5120, 1e-6, 1e-6]
