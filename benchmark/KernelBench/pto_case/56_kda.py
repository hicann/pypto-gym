#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn


FORMULA = "o[B,T,HV,V], S[B,HV,K,V] = KDA(Q[B,T,H,K], K[B,T,H,K], V[B,T,HV,V], g[B,T,HV,K], beta[B,T,HV], S0[B,HV,K,V])"
DYNAMIC_AXIS = ["T"]


class Model(nn.Module):
    def __init__(self, chunk_size: int = 64, output_final_state: bool = True):
        super().__init__()
        self.chunk_size = chunk_size
        self.output_final_state = output_final_state

    def forward(self, q, k, v, g, beta, initial_state):
        dtype = v.dtype
        b_sz, t, h, k_dim = q.shape
        hv = v.shape[2]
        v_dim = v.shape[-1]
        g_val = hv // h
        bt = self.chunk_size
        nt = t // bt
        scale = k_dim ** -0.5

        q, k = [x.to(torch.float32) for x in (q, k)]
        v, g, beta = [x.to(torch.float32) for x in (v, g, beta)]

        q = q.reshape(b_sz, nt, bt, h, k_dim).permute(0, 3, 1, 2, 4)
        k = k.reshape(b_sz, nt, bt, h, k_dim).permute(0, 3, 1, 2, 4)
        v = v.reshape(b_sz, nt, bt, hv, v_dim).permute(0, 3, 1, 2, 4)
        g = g.reshape(b_sz, nt, bt, hv, k_dim).permute(0, 3, 1, 2, 4)
        beta = beta.reshape(b_sz, nt, bt, hv).permute(0, 3, 1, 2)

        q = q.repeat_interleave(g_val, dim=1) * scale
        k = k.repeat_interleave(g_val, dim=1)

        g = g.cumsum(-2)

        mask = torch.triu(torch.ones(bt, bt, dtype=torch.bool, device=q.device), diagonal=0)

        a_mat = torch.zeros(*g.shape[:-1], bt, dtype=torch.float32, device=q.device)
        for i in range(bt):
            k_i = k[..., i, :]
            g_i = g[..., i:i + 1, :]
            a_mat[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g_i).exp(), k_i)
        a_mat = a_mat * beta[..., None]
        a_mat = -a_mat.masked_fill(mask, 0)

        for i in range(1, bt):
            a_mat[..., i, :i] = (
                a_mat[..., i, :i].clone()
                + (a_mat[..., i, :, None].clone() * a_mat[..., :, :i].clone()).sum(-2)
            )
        a_mat = (a_mat + torch.eye(bt, dtype=torch.float32, device=q.device)) * beta[..., None, :]

        w = a_mat @ (g.exp() * k)
        u = a_mat @ v

        s = k.new_zeros(b_sz, hv, k_dim, v_dim)
        if initial_state is not None:
            s = s + initial_state.to(torch.float32)

        o = torch.zeros_like(v)
        mask2 = torch.triu(torch.ones(bt, bt, dtype=torch.bool, device=q.device), diagonal=1)

        for i in range(0, nt):
            q_i = q[:, :, i]
            k_i = k[:, :, i]
            u_i = u[:, :, i]
            g_i = g[:, :, i]
            w_i = w[:, :, i]

            aqk = torch.zeros(b_sz, hv, bt, bt, dtype=torch.float32, device=q.device)
            for j in range(bt):
                k_j = k[:, :, i, j]
                g_j = g[:, :, i, j:j + 1, :]
                aqk[..., j] = torch.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).exp(), k_j)
            aqk = aqk.masked_fill(mask2, 0)

            v_i = u_i - w_i @ s
            o[:, :, i] = (q_i * g_i.exp()) @ s + aqk @ v_i

            s = s * g_i[:, :, -1].exp().reshape(b_sz, hv, k_dim, 1)
            s = s + (g_i[:, :, -1:] - g_i).exp().mul(k_i).permute(0, 1, 3, 2) @ v_i

        if not self.output_final_state:
            s = None

        o = o.permute(0, 2, 1, 3, 4).reshape(b_sz, t, hv, v_dim)
        return o.to(dtype), s


def get_inputs():
    b_sz = 1
    t = 128
    h = 2
    hv = 4
    k_dim = 64
    v_dim = 64

    q = torch.randn(b_sz, t, h, k_dim, dtype=torch.float32) / math.sqrt(k_dim)
    k = torch.randn(b_sz, t, h, k_dim, dtype=torch.float32) / math.sqrt(k_dim)
    v = torch.randn(b_sz, t, hv, v_dim, dtype=torch.float32) / math.sqrt(v_dim)
    g = torch.randn(b_sz, t, hv, k_dim, dtype=torch.float32) * 0.1
    beta = torch.randn(b_sz, t, hv, dtype=torch.float32).sigmoid()
    initial_state = torch.zeros(b_sz, hv, k_dim, v_dim, dtype=torch.float32)
    return [q, k, v, g, beta, initial_state]


def get_init_inputs():
    return [64, True]
