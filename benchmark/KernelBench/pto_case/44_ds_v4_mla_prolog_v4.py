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
        x_dtype = x.dtype
        mean_coff = 1.0 / x.shape[-1]
        gf = gamma.to(torch.float32)
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
    t, h, rope_dim = 128, 2048, 64
    x = torch.empty(t, h, dtype=torch.bfloat16).uniform_(-1, 1)
    cos = torch.empty(t, rope_dim, dtype=torch.bfloat16).uniform_(-1, 1)
    sin = torch.empty(t, rope_dim, dtype=torch.bfloat16).uniform_(-1, 1)
    return [x, cos, sin]


def get_init_inputs():
    return [2048, 32, 192, 256, 64]
