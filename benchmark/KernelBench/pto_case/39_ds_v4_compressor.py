#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn

FORMULA = "out[d] = compressor(x[t,h], wkv, wgate, sin, cos, weight) = sum(softmax(x@wgate) * rope(x@wkv)) * weight"
DYNAMIC_AXIS = ["T"]


class Model(nn.Module):
    def __init__(self, h=512, d=128, rope_head_dim=64):
        super().__init__()
        self.d = d
        self.rope_head_dim = rope_head_dim
        self.wkv = nn.Parameter(torch.randn(d, h, dtype=torch.float32))
        self.wgate = nn.Parameter(torch.randn(d, h, dtype=torch.float32))
        self.weight = nn.Parameter(torch.ones(d, dtype=torch.float32))

    def forward(self, x, sin, cos):
        rdim = self.rope_head_dim
        xf = x.to(torch.float32)
        kv = torch.matmul(xf, self.wkv.T)
        sc = torch.matmul(xf, self.wgate.T)
        w = nn.functional.softmax(sc, dim=-1)
        weighted = w * kv

        rope_part = weighted[:, :rdim]
        rest = weighted[:, rdim:]
        sf = sin.to(torch.float32)
        cf = cos.to(torch.float32)
        h2 = rdim // 2
        r1, r2 = rope_part[:, :h2], rope_part[:, h2:]
        roped = torch.zeros_like(rope_part)
        roped[:, :h2] = r1 * cf[:, :h2] - r2 * sf[:, :h2]
        roped[:, h2:] = r2 * cf[:, h2:] + r1 * sf[:, h2:]

        out = torch.zeros(x.shape[0], self.d, dtype=torch.float32, device=x.device)
        out[:, :rdim] = roped
        out[:, rdim:] = rest
        compressed = out.sum(dim=0) * self.weight
        return compressed.unsqueeze(0).to(torch.bfloat16)


def get_inputs():
    t, h, d, rdim = 16, 512, 128, 64
    x = torch.randn(t, h, dtype=torch.bfloat16)
    sin = torch.randn(t, rdim, dtype=torch.bfloat16)
    cos = torch.randn(t, rdim, dtype=torch.bfloat16)
    return [x, sin, cos]


def get_init_inputs():
    return [512, 128, 64]
