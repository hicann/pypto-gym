#!/usr/bin/env python3
# coding: utf-8

import math
import torch
import torch.nn as nn

FORMULA = "out[t,d], post[t,hc], comb[t,hc,hc] = hc_pre(x[t,hc*d], hc_fn, hc_scale, hc_base)"
DYNAMIC_AXIS = ["T"]


class Model(nn.Module):
    def __init__(self, hc: int = 4, d: int = 4096, sinkhorn_iters: int = 20):
        super().__init__()
        self.hc = hc
        self.d = d
        self.sinkhorn_iters = sinkhorn_iters
        self.norm_eps = 1e-6
        self.hc_eps = 1e-6
        mix_hc = (2 + hc) * hc
        hc_dim = hc * d

        self.hc_fn = nn.Parameter(torch.randn(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.randn(3, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.randn(1, mix_hc, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = x.shape[0]
        x_flat = x.reshape((t, self.hc * self.d)).to(torch.float32)

        rms_denom = torch.sqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)

        hc_fn = self.hc_fn.to(torch.float32)
        res = torch.matmul(x_flat, hc_fn.transpose(0, 1))
        res = res / rms_denom

        pre, post, comb = self._sinkhorn_split(res, self.hc_scale, self.hc_base.squeeze(0))

        mul_res = pre.reshape(t, self.hc, 1) * x_flat.reshape(t, self.hc, self.d)
        out = mul_res.sum(-2)

        return out.to(torch.bfloat16), post, comb

    def _sigmoid(self, x):
        return 1.0 / (1.0 + torch.exp(-x))

    def _sinkhorn_split(self, x, hc_scale, hc_base):
        t = x.shape[0]
        hc = self.hc

        pre = x[:, :hc] * hc_scale[0] + hc_base[:hc]
        pre = self._sigmoid(pre) + self.hc_eps

        post = x[:, hc: 2*hc] * hc_scale[1] + hc_base[hc: 2*hc]
        post = 2.0 * self._sigmoid(post)

        comb = (x[:, 2*hc:] * hc_scale[2] + hc_base[2*hc:]).reshape(t, hc, hc)
        row_max = comb.amax(-1, keepdim=True)
        comb = torch.exp(comb - row_max)

        row_sum = comb.sum(-1, keepdim=True)
        comb = comb / row_sum + self.hc_eps
        col_sum = comb.sum(-2, keepdim=True)
        comb = comb / (col_sum + self.hc_eps)

        for _ in range(self.sinkhorn_iters - 1):
            row_sum = comb.sum(-1, keepdim=True)
            comb = comb / (row_sum + self.hc_eps)
            col_sum = comb.sum(-2, keepdim=True)
            comb = comb / (col_sum + self.hc_eps)

        return pre, post, comb


def get_inputs():
    t = 1024
    hc = 4
    d = 4096
    x = torch.randn(t, hc * d, dtype=torch.bfloat16)
    return [x]


def get_init_inputs():
    return [4, 4096, 20]
