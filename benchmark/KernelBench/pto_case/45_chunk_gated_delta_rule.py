#!/usr/bin/env python3
# coding: utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

FORMULA = "core_attn_out[t,nv,d], final_state[b,nv,d,d] = chunk_gated_delta_rule(q[t,nqk,d], k[t,nqk,d], v[t,nv,d], beta[t,nv], gate[t,nv], state[b,nv,d,d])"
DYNAMIC_AXIS = ["T", "B"]


class Model(nn.Module):
    def __init__(self, t: int = 2048, b: int = 2, nqk: int = 2, nv: int = 4, d: int = 128, l: int = 128):
        super().__init__()
        self.t = t
        self.b = b
        self.nqk = nqk
        self.nv = nv
        self.d = d
        self.l = l

    @staticmethod
    def _sub_inverse(attn, chunk_size):
        for index in range(1, chunk_size):
            line = attn[..., index, :index].clone()
            sub = attn[..., :index, :index].clone()
            attn[..., index, :index] = line + (line.unsqueeze(-1) * sub).sum(-2)
        return attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    @staticmethod
    def _sub_cycle(query, key, value, decay_mask, k_cumdecay, g, last_recurrent_state, total_sequence_length, chunk_size):
        attn_out = torch.zeros_like(value).to(query.device)
        attn_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

        for index in range(0, total_sequence_length // chunk_size):
            q_index, k_index, v_index = query[:, :, index], key[:, :, index], value[:, :, index]
            attn = (q_index @ k_index.transpose(-1, -2) * decay_mask[:, :, index]).masked_fill_(attn_mask, 0)
            v_new = v_index - (k_cumdecay[:, :, index]) @ last_recurrent_state
            attn_out[:, :, index] = (q_index * g[:, :, index, :, None].exp()) @ last_recurrent_state + attn @ v_new
            last_recurrent_state = last_recurrent_state * g[:, :, index, -1, None, None].exp() + \
                (k_index * (g[:, :, index, -1, None] - g[:, :, index]).exp()[..., None]).transpose(-1, -2) @ v_new

        return attn_out, last_recurrent_state

    @staticmethod
    def _sub(query, key, value, g, beta, chunk_size, initial_state, output_final_state, use_qk_l2norm_in_kernel):
        b, n, s, d = value.shape

        initial_state = initial_state.transpose(3, 2)
        if use_qk_l2norm_in_kernel:
            query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
            key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        query, key, value = [F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value)]
        beta, g = [F.pad(x, (0, pad_size)) for x in (beta, g)]

        total_sequence_length = sequence_length + pad_size
        query = query * (1 / (query.shape[-1] ** 0.5))

        v_beta, k_beta = [x * beta.unsqueeze(-1) for x in (value, key)]
        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
            for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

        attn = Model._sub_inverse(attn, chunk_size)

        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

        if initial_state is None:
            last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=query.device).to(value)
        else:
            last_recurrent_state = initial_state.to(value)

        attn_out, last_recurrent_state = Model._sub_cycle(query=query, key=key, value=value,
            decay_mask=decay_mask, k_cumdecay=k_cumdecay, g=g, last_recurrent_state=last_recurrent_state,
            total_sequence_length=total_sequence_length, chunk_size=chunk_size)

        if not output_final_state:
            last_recurrent_state = None
        attn_out = attn_out.reshape(attn_out.shape[0], attn_out.shape[1], -1, attn_out.shape[-1])
        attn_out = attn_out[:, :, :sequence_length].transpose(1, 2).contiguous()

        last_recurrent_state = last_recurrent_state.transpose(3, 2)

        return attn_out, last_recurrent_state

    def forward(self, query, key, value, beta, gate, initial_state):
        chunk_size = self.l
        output_final_state = True
        use_qk_l2norm_in_kernel = True

        t, n1, d = query.shape
        t, n, d = value.shape
        batch = self.b

        query = query.repeat_interleave(n // n1, dim=1)
        key = key.repeat_interleave(n // n1, dim=1)

        final_state = torch.zeros([batch, n, d, d], dtype=torch.float32, device=query.device)

        query, key, value, beta, gate_t = \
            [x.transpose(0, 1).contiguous().to(torch.float32) for x in (query, key, value, beta, gate)]
        final_attn = torch.zeros([t, n, d], dtype=torch.float32, device=query.device)

        seq_len_per_batch = t // batch
        act_seq_len = [i * seq_len_per_batch for i in range(batch + 1)]

        for b_idx in range(batch):
            s = act_seq_len[b_idx + 1] - act_seq_len[b_idx]
            b_ofs = act_seq_len[b_idx]
            seg_s = 128
            pad_size = (chunk_size - s % chunk_size) % chunk_size
            pad_seq_length = s + pad_size
            batch_query, batch_key, batch_value = \
                [F.pad(x[:, b_ofs:b_ofs + s], (0, 0, 0, pad_size)) for x in (query, key, value)]
            batch_beta, batch_g = [F.pad(x[:, b_ofs:b_ofs + s], (0, pad_size)) for x in (beta, gate_t)]
            result_list = []
            recurrent_state = initial_state[b_idx:b_idx + 1, ...]
            for s_idx in range(0, pad_seq_length, seg_s):
                chunk_query, chunk_key, chunk_value = \
                    [x[:, s_idx:s_idx + seg_s, :].reshape(1, n, seg_s, d) for x in (batch_query, batch_key, batch_value)]
                chunk_gate, chunk_beta = [x[:, s_idx:s_idx + seg_s].reshape(1, n, seg_s) for x in (batch_g, batch_beta)]
                cur_attn, cur_state = Model._sub(query=chunk_query, key=chunk_key, value=chunk_value,
                    g=chunk_gate, beta=chunk_beta, chunk_size=chunk_size, initial_state=recurrent_state,
                    output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
                result_list.append(cur_attn.squeeze(0))
                recurrent_state = cur_state
            batch_attn = torch.cat(result_list, dim=0)[:s]
            final_attn[b_ofs:b_ofs + s] = batch_attn
            final_state[b_idx:b_idx + 1, ...] = recurrent_state

        return final_attn, final_state


def get_inputs():
    t = 2048
    b = 2
    nqk = 2
    nv = 4
    d = 128
    l = 128

    query = torch.rand([t, nqk, d], dtype=torch.float32) * (1.3655 + 0.2785) - (1.3655 + 0.2785)
    key = torch.rand([t, nqk, d], dtype=torch.float32) * (1.4664 + 0.2785) - (1.4664 + 0.2785)
    value = torch.rand([t, nv, d], dtype=torch.float32) * (1.6488 + 0.2785) - (1.6488 + 0.2785)
    beta = torch.rand([t, nv], dtype=torch.float32) * (0.8927 - 0.0889) - (0.8927 - 0.0889)
    gate = torch.rand([t, nv], dtype=torch.float32) * (-0.1343 + 37.5452) - (-0.1343 + 37.5452)
    states = torch.zeros([b, nv, d, d], dtype=torch.float32)

    return [query, key, value, beta, gate, states]


def get_init_inputs():
    return [2048, 2, 2, 4]
