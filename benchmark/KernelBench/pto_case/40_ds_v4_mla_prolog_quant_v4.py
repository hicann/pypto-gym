#!/usr/bin/env python3
# coding: utf-8

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
        self.wq_b = nn.Parameter(torch.empty(q_lora_rank, num_heads * head_dim, dtype=torch.bfloat16).uniform_(-0.1, 0.1))
        self.w_kv = nn.Parameter(torch.empty(h, head_dim, dtype=torch.bfloat16).uniform_(-0.1, 0.1))
        self.gamma_cq = nn.Parameter(torch.empty(q_lora_rank, dtype=torch.bfloat16).uniform_(-1, 1))
        self.gamma_ckv = nn.Parameter(torch.empty(head_dim, dtype=torch.bfloat16).uniform_(-1, 1))
        self.wq_b_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(self, x, cos, sin):
        t = x.shape[0]
        hd = self.head_dim
        nh = self.num_heads
        rdim = self.qk_rope_head_dim

        q_a = torch.matmul(x.to(torch.float32), self.wq_a.to(torch.float32))
        q_ln = self._rms_norm(q_a, self.gamma_cq).to(torch.bfloat16)
        q_quant, q_scale = self._quant(q_ln)
        q_b = torch.matmul(q_quant.to(torch.float32), self.wq_b.to(torch.float32))
        q_deq = q_b * q_scale * self.wq_b_scale
        q_r = self._rms_norm_new(q_deq.reshape(t, nh, hd)).to(torch.bfloat16)

        kv_a = torch.matmul(x.to(torch.float32), self.w_kv.to(torch.float32))
        kv_r = self._rms_norm(kv_a, self.gamma_ckv).reshape(t, hd).to(torch.bfloat16)

        q_pe = q_r[:, :, -rdim:]
        k_pe = kv_r[:, -rdim:].reshape(t, 1, rdim)
        qe, ke = self._rope(q_pe, k_pe, cos, sin)

        qo = torch.cat([q_r[:, :, :-rdim], qe], -1)
        ko = torch.cat([kv_r[:, :-rdim], ke.reshape(t, rdim)], -1)
        return qo, ko, q_quant.to(torch.float32), q_scale

    def _rms_norm(self, x, g, eps=1e-6):
        xf = x.to(torch.float32); gf = g.to(torch.float32)
        return (xf * torch.rsqrt((xf*xf).mean(-1, keepdim=True) + eps)) * gf

    def _rms_norm_new(self, x, eps=1e-6):
        xf = x.to(torch.float32)
        return xf * torch.rsqrt((xf*xf).mean(-1, keepdim=True) + eps)

    def _quant(self, x):
        xf = x.to(torch.float32)
        mv = torch.max(torch.abs(xf), dim=-1, keepdims=True)[0]
        sc = 127.0 / (mv + 1e-12)
        return torch.round(xf * sc).clamp(-127, 127).to(torch.int8), 1.0 / sc

    def _rope(self, q, k, cos, sin):
        qf = q.to(torch.float32); kf = k.to(torch.float32)
        cf = cos.to(torch.float32).unsqueeze(1); sf = sin.to(torch.float32).unsqueeze(1)
        d2 = qf.shape[-1] // 2
        q1, q2 = qf[..., :d2], qf[..., d2:]
        k1, k2 = kf[..., :d2], kf[..., d2:]
        c1, c2 = cf[..., :d2], cf[..., d2:]
        s1, s2 = sf[..., :d2], sf[..., d2:]
        qr = torch.cat([q1*c1 - q2*s1, q2*c2 + q1*s2], -1)
        kr = torch.cat([k1*c1 - k2*s1, k2*c2 + k1*s2], -1)
        return qr.to(torch.bfloat16), kr.to(torch.bfloat16)


def get_inputs():
    t,h,rdim=128,2048,64
    x=torch.empty(t,h,dtype=torch.bfloat16).uniform_(-1,1)
    c=torch.empty(t,rdim,dtype=torch.bfloat16).uniform_(-1,1)
    s=torch.empty(t,rdim,dtype=torch.bfloat16).uniform_(-1,1)
    return [x,c,s]


def get_init_inputs():
    return [2048,32,192,256,64]
