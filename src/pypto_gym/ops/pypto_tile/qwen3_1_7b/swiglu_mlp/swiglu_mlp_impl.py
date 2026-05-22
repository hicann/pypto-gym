#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Qwen3 SwiGLU MLP PyPTO Kernel

数学公式：
    gate = matmul(x, Wgate)         # [S, INT_SIZE]
    up = matmul(x, Wup)             # [S, INT_SIZE]
    silu_gate = silu(gate)          # g / (1 + e^-g)
    hidden = silu_gate * up         # [S, INT_SIZE]
    output = matmul(hidden, Wdown)  # [S, H]
"""

import pypto
import torch
from torch._dynamo import allow_in_graph

H = 2048          # hidden_size
INT_SIZE = 6144   # intermediate_size
BS_TILE = 8       # batch tile size


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128},
    pass_options={"cube_l1_reuse_setting": {-1: 4}},
)
def swiglu_mlp_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, H], pypto.DT_BF16),
    Wgate: pypto.Tensor([H, INT_SIZE], pypto.DT_BF16),
    Wup: pypto.Tensor([H, INT_SIZE], pypto.DT_BF16),
    Wdown: pypto.Tensor([INT_SIZE, H], pypto.DT_BF16),
    output: pypto.Tensor([pypto.DYNAMIC, H], pypto.DT_BF16),
):
    """SwiGLU MLP kernel"""
    S = x.shape[0]
    bs_loop = (S + BS_TILE - 1) // BS_TILE
    
    for bs_idx in pypto.loop(bs_loop, name="LOOP_BS_MLP", idx_name="bs_idx"):
        cur_bs = (S - bs_idx * BS_TILE).min(BS_TILE)
        
        # View input tile
        x_tile = pypto.view(x, [BS_TILE, H], [bs_idx * BS_TILE, 0],
                           valid_shape=[cur_bs, H])
        
        # Gate projection
        pypto.set_cube_tile_shapes([32, 32], [128, 512], [256, 256])
        gate_fp32 = pypto.matmul(x_tile, Wgate, pypto.DT_FP32)
        
        # Up projection
        up_fp32 = pypto.matmul(x_tile, Wup, pypto.DT_FP32)
        
        # SiLU(gate) = gate * sigmoid(gate) (使用sigmoid避免exp溢出)
        pypto.set_vec_tile_shapes(1, INT_SIZE)
        sigmoid_gate = pypto.sigmoid(gate_fp32)  # 数值稳定
        silu_gate = pypto.mul(gate_fp32, sigmoid_gate)
        
        # hidden = silu_gate * up
        hidden_fp32 = pypto.mul(silu_gate, up_fp32)
        hidden_buf = pypto.tensor([BS_TILE, INT_SIZE], pypto.DT_BF16, "hidden_buf")
        hidden_buf[:] = pypto.cast(hidden_fp32, pypto.DT_BF16)
        
        # Down projection
        pypto.set_cube_tile_shapes([32, 32], [128, 512], [128, 128])
        out_fp32 = pypto.matmul(hidden_buf, Wdown, pypto.DT_FP32)
        out_bf = pypto.cast(out_fp32, pypto.DT_BF16)
        
        # Assemble output（固定BS_TILE切片，依赖valid_shape机制）
        output[bs_idx * BS_TILE: bs_idx * BS_TILE + BS_TILE, :] = out_bf


@allow_in_graph
def swiglu_mlp_impl(x: torch.Tensor, Wgate: torch.Tensor, Wup: torch.Tensor, Wdown: torch.Tensor) -> torch.Tensor:
    """
    SwiGLU MLP wrapper
    
    Args:
        x: input tensor [batch, seq_len, hidden_size] or [S, H] (2D)
        Wgate: gate projection weight [hidden_size, intermediate_size] - 需要transpose给matmul
        Wup: up projection weight [hidden_size, intermediate_size] - 需要transpose给matmul
        Wdown: down projection weight [intermediate_size, hidden_size] - 直接使用
    
    Returns:
        output tensor with same shape as x
    
    Note:
        PyTorch Linear的weight shape是[out, in]
        gate_proj.weight: [6144, 2048] (out=intermediate_size, in=hidden_size)
        需要transpose为 [2048, 6144] 才能用于 matmul(x, W)
    """
    # Handle 3D input (transformers uses [batch, seq, hidden])
    input_shape = x.shape
    if x.dim() == 3:
        batch, seq, hidden = input_shape
        x_2d = x.view(batch * seq, hidden)
    else:
        x_2d = x
    
    # 权重处理：PyTorch Linear weight需要transpose
    # gate_proj.weight: [6144, 2048] → transpose to [2048, 6144]
    # up_proj.weight: [6144, 2048] → transpose to [2048, 6144]
    # down_proj.weight: [2048, 6144] → transpose to [6144, 2048]
    Wgate_t = Wgate.T.contiguous() if Wgate.shape[0] == INT_SIZE else Wgate  # 确保是[H, INT_SIZE]且contiguous
    Wup_t = Wup.T.contiguous() if Wup.shape[0] == INT_SIZE else Wup
    Wdown_t = Wdown.T.contiguous() if Wdown.shape[0] == H else Wdown  # 确保是[INT_SIZE, H]且contiguous
    
    # Convert to bfloat16
    x_bf = x_2d.to(torch.bfloat16).contiguous()
    Wgate_bf = Wgate_t.to(torch.bfloat16).contiguous()
    Wup_bf = Wup_t.to(torch.bfloat16).contiguous()
    Wdown_bf = Wdown_t.to(torch.bfloat16).contiguous()
    
    # Allocate output
    output_bf = torch.empty_like(x_bf)
    
    # Call kernel
    swiglu_mlp_kernel(x_bf, Wgate_bf, Wup_bf, Wdown_bf, output_bf)
    
    # Restore original shape
    if x.dim() == 3:
        output = output_bf.to(x.dtype).view(input_shape)
    else:
        output = output_bf.to(x.dtype)
    
    return output


if __name__ == "__main__":
    import os
    os.environ.setdefault('PTO_TILE_LIB_CODE_PATH', '/data/h00520348/pto-isa')
    
    torch.manual_seed(42)
    
    # Test with 2D input
    x = torch.randn(16, H, dtype=torch.bfloat16, device="cpu")
    Wgate = torch.randn(H, INT_SIZE, dtype=torch.bfloat16, device="cpu")
    Wup = torch.randn(H, INT_SIZE, dtype=torch.bfloat16, device="cpu")
    Wdown = torch.randn(INT_SIZE, H, dtype=torch.bfloat16, device="cpu")
    
    output = swiglu_mlp_impl(x, Wgate, Wup, Wdown)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")