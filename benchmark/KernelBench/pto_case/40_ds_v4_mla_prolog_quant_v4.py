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

import torch
import torch.nn as nn

FORMULA = "q_out[t,n,d], kv_out[t,d], q_a_quant[t,qlr], q_a_scale[t,1] = mla_prolog_quant(x[t,h], wq_a, wq_b, wkv, cos, sin, gamma_cq, gamma_ckv)"
DYNAMIC_AXIS = ["T"]


class Model(nn.Module):
    def __init__(self, h=2048, num_heads=32, head_dim=192, q_lora_rank=256, qk_rope_head_dim=64):
        super().__init__()
        self.h = h
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.wq_a = nn.Parameter(torch.empty(h, q_lora_rank, dtype=torch.bfloat16).uniform_(-0.1, 0.1))
        self.wq_b_raw = nn.Parameter(torch.empty(q_lora_rank, num_heads * head_dim, dtype=torch.bfloat16).uniform_(-0.1, 0.1))
        self.w_kv = nn.Parameter(torch.empty(h, head_dim, dtype=torch.bfloat16).uniform_(-0.1, 0.1))
        self.gamma_cq = nn.Parameter(torch.empty(q_lora_rank, dtype=torch.bfloat16).uniform_(-1, 1))
        self.gamma_ckv = nn.Parameter(torch.empty(head_dim, dtype=torch.bfloat16).uniform_(-1, 1))

    def forward(self, x, cos, sin):
        t = x.shape[0]
        hd = self.head_dim
        nh = self.num_heads
        rdim = self.qk_rope_head_dim

        wq_b_int8, wq_b_scale = self._quant(self.wq_b_raw, is_pertoken=False)
        q_a = torch.matmul(x.to(torch.float32), self.wq_a.to(torch.float32))
        q_ln = self._rms_norm(q_a, self.gamma_cq)
        q_quant, q_scale = self._quant(q_ln, is_pertoken=True)
        q_b = torch.matmul(q_quant.to(torch.float32), wq_b_int8.to(torch.float32))
        q_deq = q_b.to(torch.float32) * q_scale
        q_deq = q_deq * wq_b_scale
        q_r = self._rms_norm_new(q_deq.reshape(t, nh, hd)).to(torch.bfloat16)

        kv_a = torch.matmul(x.to(torch.float32), self.w_kv.to(torch.float32))
        kv_norm = self._rms_norm(kv_a, self.gamma_ckv).reshape(t, hd).to(torch.bfloat16)

        q_pe = q_r[:, :, -rdim:]
        k_pe = kv_norm[:, -rdim:].reshape(t, 1, rdim)
        qe, ke = self._rope(q_pe, k_pe, cos, sin)

        qo = torch.cat([q_r[:, :, :-rdim], qe], -1)
        ko = torch.cat([kv_norm[:, :-rdim], ke.reshape(t, rdim)], -1)
        return qo, ko, q_quant.to(torch.float32), q_scale

    def _rms_norm(self, x, g, eps=1e-6):
        x_dtype = x.dtype
        mean_coff = 1.0 / x.shape[-1]
        gf = g.to(torch.float32)
        xf = x.to(torch.float32)
        square = xf * xf
        mean_res = square * mean_coff
        reduce_sum = torch.sum(mean_res, dim=-1, keepdims=True) + eps
        reduce_sqrt = torch.sqrt(reduce_sum)
        res_div = xf / reduce_sqrt
        res = res_div * gf
        if x_dtype != torch.float32:
            res = res.to(x_dtype)
        return res

    def _rms_norm_new(self, x, eps=1e-6):
        x_dtype = x.dtype
        mean_coff = 1.0 / x.shape[-1]
        xf = x.to(torch.float32)
        square = xf * xf
        mean_res = square * mean_coff
        reduce_sum = torch.sum(mean_res, dim=-1, keepdims=True) + eps
        reduce_sqrt = torch.sqrt(reduce_sum)
        res = xf / reduce_sqrt
        if x_dtype != torch.float32:
            res = res.to(x_dtype)
        return res

    def _quant(self, x, is_pertoken=True):
        xf = x.to(torch.float32)
        abs_res = torch.abs(xf)
        reduce_idx = -1 if is_pertoken else -2
        max_value = torch.max(abs_res, dim=reduce_idx, keepdims=True)[0]
        scale_quant = 127.0 / max_value
        out_fp32 = xf * scale_quant
        out_int32 = torch.round(out_fp32).to(torch.int32)
        out_fp16 = out_int32.to(torch.float16)
        out_int8 = torch.trunc(out_fp16).to(torch.int8)
        scale_dequant = 1.0 / scale_quant
        return out_int8, scale_dequant

    def _rotate_half(self, x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _rope(self, q, k, cos, sin, unsqueeze_dim=1):
        input_dtype = q.dtype
        q_clone = q.clone()
        k_clone = k.clone()
        t, nq, d = q.shape
        t_, nk, d_ = k.shape
        q = q.reshape(t, nq, d//2, 2).permute(0, 1, 3, 2).reshape(t, nq, d)
        k = k.reshape(t_, nk, d_//2, 2).permute(0, 1, 3, 2).reshape(t_, nk, d)
        qt = self._rotate_half(q)
        kt = self._rotate_half(k)
        qn = qt.reshape(t, nq, 2, d//2).permute(0, 1, 3, 2).reshape(t, nq, d)
        kn = kt.reshape(t_, nk, 2, d_//2).permute(0, 1, 3, 2).reshape(t_, nk, d)
        qn = qn.to(torch.float32)
        kn = kn.to(torch.float32)
        cos_f32 = torch.unsqueeze(cos, dim=unsqueeze_dim).to(torch.float32)
        sin_f32 = torch.unsqueeze(sin, dim=unsqueeze_dim).to(torch.float32)
        qe = q_clone * cos_f32 + qn * sin_f32
        ke = k_clone * cos_f32 + kn * sin_f32
        if input_dtype != torch.float32:
            qe, ke = qe.to(input_dtype), ke.to(input_dtype)
        return qe, ke


def get_inputs():
    t, h, rdim = 128, 2048, 64
    x = torch.empty(t, h, dtype=torch.bfloat16).uniform_(-1, 1)
    c = torch.empty(t, rdim, dtype=torch.bfloat16).uniform_(-1, 1)
    s = torch.empty(t, rdim, dtype=torch.bfloat16).uniform_(-1, 1)
    return [x, c, s]


def get_init_inputs():
    return [2048, 32, 192, 256, 64]
