#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn

FORMULA = "q_out[t,n,d], kv_out[t,d] = mla_prolog(x[t,h], wq_a[h,qlr], wq_b[qlr,n*d], wkv[h,d], cos, sin, gamma_cq, gamma_ckv)"
DYNAMIC_AXIS = ["T"]


class Model(nn.Module):
    def __init__(
        self, h: int = 2048, num_heads: int = 32, head_dim: int = 192,
        q_lora_rank: int = 256, qk_rope_head_dim: int = 64
    ):
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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        t = x.shape[0]
        hd = self.head_dim
        nh = self.num_heads

        q_a = torch.matmul(x.to(torch.float32), self.wq_a.to(torch.float32))
        q_a_ln = self._rms_norm(q_a, self.gamma_cq).to(torch.bfloat16)
        q_b = torch.matmul(q_a_ln, self.wq_b)
        q_reshape = self._rms_norm_new(q_b.reshape(t, nh, hd)).to(torch.bfloat16)

        kv_a = torch.matmul(x.to(torch.float32), self.w_kv.to(torch.float32))
        kv_ln = self._rms_norm(kv_a, self.gamma_ckv).reshape(t, hd).to(torch.bfloat16)

        rdim = self.qk_rope_head_dim
        q_pe = q_reshape[:, :, -rdim:]
        k_pe = kv_ln[:, -rdim:].reshape(t, 1, rdim)
        qr, kr = self._rope(q_pe, k_pe, cos, sin)

        q_out = torch.cat([q_reshape[:, :, :-rdim], qr], -1)
        kv_out = torch.cat([kv_ln[:, :-rdim], kr.reshape(t, rdim)], -1)
        return q_out, kv_out

    def _rms_norm(self, x, gamma, eps=1e-6):
        x_f = x.to(torch.float32)
        g_f = gamma.to(torch.float32)
        ms = (x_f * x_f).mean(-1, keepdim=True)
        return (x_f * torch.rsqrt(ms + eps)) * g_f

    def _rms_norm_new(self, x, eps=1e-6):
        x_f = x.to(torch.float32)
        ms = (x_f * x_f).mean(-1, keepdim=True)
        return x_f * torch.rsqrt(ms + eps)

    def _rope(self, q, k, cos, sin):
        qf = q.to(torch.float32)
        kf = k.to(torch.float32)
        cf = cos.to(torch.float32).unsqueeze(1)
        sf = sin.to(torch.float32).unsqueeze(1)
        d = qf.shape[-1]
        qc = qf.reshape(-1, qf.shape[1], d//2, 2).permute(0, 1, 3, 2).reshape_as(qf)
        kc = kf.reshape(-1, kf.shape[1], d//2, 2).permute(0, 1, 3, 2).reshape_as(kf)
        q1, q2 = qc.chunk(2, dim=-1)
        k1, k2 = kc.chunk(2, dim=-1)
        qr = qf * cf + torch.cat((-q2, q1), -1) * sf
        kr = kf * cf + torch.cat((-k2, k1), -1) * sf
        return qr.to(torch.bfloat16), kr.to(torch.bfloat16)


def get_inputs():
    t, h, rope_dim = 128, 2048, 64
    x = torch.empty(t, h, dtype=torch.bfloat16).uniform_(-1, 1)
    cos = torch.empty(t, rope_dim, dtype=torch.bfloat16).uniform_(-1, 1)
    sin = torch.empty(t, rope_dim, dtype=torch.bfloat16).uniform_(-1, 1)
    return [x, cos, sin]


def get_init_inputs():
    return [2048, 32, 192, 256, 64]
