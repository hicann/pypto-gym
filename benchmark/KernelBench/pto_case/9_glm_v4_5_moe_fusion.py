#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: GLM V4.5 MoE Fusion — gate + select_experts + ffn_shared_expert_quant combined."""

import math
import torch
import torch.nn as nn

FORMULA = (
    "router_logits[m, e] = hidden_states[m, h] @ W_gate[e, h]^T; "
    "topk_weights, topk_ids = GroupTopK( Sigmoid(router_logits) + bias, k ); "
    "x_quant, x_scale = PerTokenQuant(hidden_states); "
    "up = (x_quant @ w13) * x_scale * w13_scale; "
    "swiglu = SiLU(gate) * up; "
    "down_quant, down_scale = PerTokenQuant(swiglu); "
    "out[m, h] = (down_quant @ w2) * down_scale * w2_scale"
)
DYNAMIC_AXIS = ["M"]


class Model(nn.Module):
    def __init__(self, num_router_experts: int = 160, hidden_size: int = 5120,
                 intermediate_size: int = 192, top_k: int = 8,
                 renormalize: bool = True, topk_group: int = 1,
                 num_expert_group: int = 1):
        super().__init__()
        self.num_router_experts = num_router_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.renormalize = renormalize
        self.topk_group = topk_group
        self.num_expert_group = num_expert_group

        self.gate_weight = nn.Parameter(
            torch.randn(num_router_experts, hidden_size, dtype=torch.float32)
            / math.sqrt(hidden_size))
        self.e_score_bias = nn.Parameter(
            torch.randn(num_router_experts, dtype=torch.bfloat16)
            / math.sqrt(num_router_experts))

        w13_raw = torch.randn(hidden_size, intermediate_size * 2, dtype=torch.float32) * 0.01
        w13_sym, w13_scale_val = self._per_channel_quantize(w13_raw)
        self.register_buffer('w13', w13_sym)
        self.w13_scale = nn.Parameter(w13_scale_val.reshape(-1).to(torch.bfloat16))

        w2_raw = torch.randn(intermediate_size, hidden_size, dtype=torch.float32) * 0.01
        w2_sym, w2_scale_val = self._per_channel_quantize(w2_raw)
        self.register_buffer('w2', w2_sym)
        self.w2_scale = nn.Parameter(w2_scale_val.reshape(-1).to(torch.bfloat16))

    @staticmethod
    def _per_channel_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_val = x.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-8)
        scale = 127.0 / max_val
        q = (x * scale).round().clip(-128, 127).to(torch.int8)
        deq_scale = 1.0 / scale
        return q, deq_scale

    @staticmethod
    def _per_token_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_val = x.float().abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        scale = 127.0 / max_val
        q = (x.float() * scale).round().clamp(-128, 127).to(torch.int8)
        deq_scale = 1.0 / scale
        return q, deq_scale

    def _select_experts(self, router_logits: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        bs = router_logits.shape[0]
        ne = router_logits.shape[1]
        original_weights = router_logits.sigmoid()
        bias_2d = self.e_score_bias.unsqueeze(0).to(router_logits.dtype)
        topk_weights_add = original_weights + bias_2d
        group_unit = ne // self.num_expert_group
        tw_view = topk_weights_add.view(bs, self.num_expert_group, group_unit)
        grouped_weights = tw_view.max(dim=-1).values
        topk_group_indices = torch.topk(grouped_weights, k=self.topk_group, dim=-1, sorted=False)[1]
        topk_group_mask = torch.zeros_like(grouped_weights)
        topk_group_mask.scatter_(1, topk_group_indices, 1.0)
        tgm_expand = topk_group_mask.unsqueeze(-1).expand(bs, self.num_expert_group, group_unit)
        topk_weight_mask = tgm_expand.reshape(bs, -1)
        topk_weights_fill = topk_weights_add.masked_fill(~topk_weight_mask.bool(), 0.0)
        topk_ids = torch.topk(topk_weights_fill, k=self.top_k, dim=-1, sorted=False)[1].to(torch.int32)
        topk_weights_gather = original_weights.gather(1, topk_ids)
        if self.renormalize:
            topk_weights_out = topk_weights_gather / topk_weights_gather.sum(dim=-1, keepdim=True)
        else:
            topk_weights_out = topk_weights_gather
        return topk_weights_out, topk_ids

    def _ffn_shared_expert(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x_dtype = hidden_states.dtype
        x_quant, x_scale = self._per_token_quantize(hidden_states)
        up_proj_i32 = x_quant.float() @ self.w13.float()
        w13s = self.w13_scale.float().unsqueeze(0)
        up_proj = (up_proj_i32.float() * w13s * x_scale.float()).to(x_dtype)
        gate, up = up_proj.chunk(2, dim=-1)
        swiglu_out = gate.sigmoid() * gate * up
        down_quant, down_scale = self._per_token_quantize(swiglu_out)
        down_proj_i32 = down_quant.float() @ self.w2.float()
        w2s = self.w2_scale.float().unsqueeze(0)
        out = (down_proj_i32.float() * w2s * down_scale.float()).to(x_dtype)
        return out

    def forward(self, hidden_states: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = torch.matmul(hidden_states.float(), self.gate_weight.t())
        topk_weights, topk_ids = self._select_experts(router_logits.to(torch.float32))
        ffn_res = self._ffn_shared_expert(hidden_states)
        return topk_weights, topk_ids, ffn_res


def get_inputs():
    bs = 32
    hidden_size = 5120
    x = torch.randn(bs, hidden_size, dtype=torch.bfloat16) / math.sqrt(hidden_size)
    return [x]


def get_init_inputs():
    return [160, 5120, 192, 8, True, 1, 1]
