#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import math

FORMULA = (
    "topk_weights[m, k] = renormalize(gather(sigmoid(logits[m, n]), "
    "topk(masked_fill(group_top_k(sigmoid(logits) + bias, num_expert_group, topk_group), 0.0), k))), "
    "topk_ids[m, k] = topk_ids(...)"
)
DYNAMIC_AXIS = ["M"]


class Model(nn.Module):
    def __init__(
        self,
        num_router_experts: int = 160,
        top_k: int = 8,
        renormalize: bool = True,
        topk_group: int = 1,
        num_expert_group: int = 1,
    ):
        super().__init__()
        self.num_router_experts = num_router_experts
        self.top_k = top_k
        self.renormalize = renormalize
        self.topk_group = topk_group
        self.num_expert_group = num_expert_group

        self.e_score_correction_bias = nn.Parameter(
            torch.randn(num_router_experts, dtype=torch.bfloat16) / math.sqrt(num_router_experts)
        )

    def forward(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bs = router_logits.shape[0]
        ne = router_logits.shape[1]

        original_weights = router_logits.float().sigmoid()
        bias_2d = self.e_score_correction_bias.unsqueeze(0).float()
        topk_weights_g_add = original_weights + bias_2d

        # Group top-k selection
        group_unit = ne // self.num_expert_group
        tw_view = topk_weights_g_add.view(bs, self.num_expert_group, group_unit)
        grouped_weights = tw_view.max(dim=-1).values
        topk_group_indices = torch.topk(
            grouped_weights.to(torch.float32),
            k=self.topk_group,
            dim=-1,
            sorted=False,
        )[1]

        topk_group_mask = torch.zeros_like(grouped_weights)
        topk_group_mask.scatter_(1, topk_group_indices, 1)
        tgm_unsquee = topk_group_mask.unsqueeze(-1)
        tgm_expand = tgm_unsquee.expand(bs, self.num_expert_group, group_unit)
        topk_weight_mask = tgm_expand.reshape(bs, -1)
        logical_not_tmp = ~topk_weight_mask.bool()
        topk_weights_fill = topk_weights_g_add.masked_fill(logical_not_tmp, 0.0)

        # Top-k experts
        topk_ids_int64 = torch.topk(
            topk_weights_fill.to(torch.float32),
            k=self.top_k,
            dim=-1,
            sorted=False,
        )[1]
        topk_ids_int32 = topk_ids_int64.to(torch.int32)
        topk_weights_gather = original_weights.gather(1, topk_ids_int64)

        # Renormalize
        if self.renormalize:
            topk_weights_out = topk_weights_gather / topk_weights_gather.sum(dim=-1, keepdim=True)
        else:
            topk_weights_out = topk_weights_gather

        return topk_weights_out, topk_ids_int32


def get_inputs():
    bs = 32
    ne = 160
    return [torch.randn(bs, ne, dtype=torch.float32) / math.sqrt(ne)]


def get_init_inputs():
    return [160, 8, True, 1, 1]
