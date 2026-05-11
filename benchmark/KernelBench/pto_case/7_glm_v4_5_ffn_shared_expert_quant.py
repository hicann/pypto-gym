#!/usr/bin/env python3
# coding: utf-8
"""KernelBench case: GLM V4.5 FFN Shared Expert Quant — quantized SwiGLU FFN for MoE shared expert."""

import math
import torch
import torch.nn as nn

FORMULA = (
    "x_quant, x_scale = PerTokenQuant(x[m, h]); "
    "up = (x_quant @ w13[h, i*2]) * x_scale * w13_scale[i*2]; "
    "swiglu = SiLU(up[:, :i]) * up[:, i:]; "
    "down_quant, down_scale = PerTokenQuant(swiglu); "
    "out[m, h] = (down_quant @ w2[i, h]) * down_scale * w2_scale[h]"
)
DYNAMIC_AXIS = ["M"]


class Model(nn.Module):
    def __init__(self, hidden_size: int = 5120, intermediate_size: int = 192):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Quantize random weights to int8 per-channel
        w13_raw = torch.randn(hidden_size, intermediate_size * 2, dtype=torch.float32) * 0.01
        w13_symmetric, w13_scale_val = self._per_channel_quantize(w13_raw)
        self.register_buffer('w13', w13_symmetric)
        self.w13_scale = nn.Parameter(w13_scale_val.reshape(-1).to(torch.bfloat16))

        w2_raw = torch.randn(intermediate_size, hidden_size, dtype=torch.float32) * 0.01
        w2_symmetric, w2_scale_val = self._per_channel_quantize(w2_raw)
        self.register_buffer('w2', w2_symmetric)
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x_dtype = hidden_states.dtype

        # Per-token quantization of input
        x_quant, x_scale = self._per_token_quantize(hidden_states)

        # Up projection (quantized matmul) — use float for matmul to avoid int32 ACL failure
        up_proj_i32 = x_quant.float() @ self.w13.float()
        w13s = self.w13_scale.float().unsqueeze(0)
        up_proj = (up_proj_i32.float() * w13s * x_scale.float()).to(x_dtype)

        # SwiGLU
        gate, up = up_proj.chunk(2, dim=-1)
        swiglu_out = gate.sigmoid() * gate * up

        # Down projection
        down_quant, down_scale = self._per_token_quantize(swiglu_out)
        down_proj_i32 = down_quant.float() @ self.w2.float()
        w2s = self.w2_scale.float().unsqueeze(0)
        out = (down_proj_i32.float() * w2s * down_scale.float()).to(x_dtype)

        return out


def get_inputs():
    bs = 2
    hidden_size = 5120
    x = torch.randn(bs, hidden_size, dtype=torch.bfloat16) * 0.01 / math.sqrt(hidden_size)
    return [x]


def get_init_inputs():
    return [5120, 192]
