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
Qwen3 Decode Attention Wrapper

使用 gym仓已有的 qwen3_decode_attn 实现（GQA-native, online softmax）
仅适用于 Decode 模式（Sq=1）
"""

import sys
import torch

from pypto_gym.ops.pypto_tile.qwen3_1_7b.qwen3_decode_attn import qwen3_decode_attn

# Qwen3-1.7B 参数
Nq = 16   # num_attention_heads
Nkv = 8   # num_key_value_heads  
D = 128   # head_dim
GROUPS = Nq // Nkv  # =2


def decode_attention_wrapper(
    query: torch.Tensor,    # [Nq=16, D=128]
    key_cache: torch.Tensor,  # [Nkv=8, Skv, D=128]
    value_cache: torch.Tensor,  # [Nkv=8, Skv, D=128]
    mask: torch.Tensor,     # [Skv] causal mask
) -> torch.Tensor:
    """
    Decode Attention wrapper
    
    Args:
        query: [Nq, D] (已经经过RoPE)
        key_cache: [Nkv, Skv, D] KV cache中的K
        value_cache: [Nkv, Skv, D] KV cache中的V
        mask: [Skv] causal mask
    
    Returns:
        output: [Nq, D]
    
    Note:
        仅适用于Decode模式（Sq=1）
        使用GQA-native实现，8个KV heads服务16个Q heads
    """
    # 转换为 bfloat16 并确保 contiguous
    query_bf = query.to(torch.bfloat16).contiguous()
    key_bf = key_cache.to(torch.bfloat16).contiguous()
    value_bf = value_cache.to(torch.bfloat16).contiguous()
    
    # 构造mask（gym仓实现期望 [Nkv, GROUPS, Skv]）
    Skv = key_cache.shape[1]
    mask_3d = mask.unsqueeze(0).unsqueeze(0).expand(Nkv, GROUPS, Skv).to(torch.float32).contiguous()
    
    # 调用gym仓的decode_attn
    output_bf = torch.empty_like(query_bf)
    
    qwen3_decode_attn(query_bf, key_bf, value_bf, mask_3d, output_bf)
    
    return output_bf.to(query.dtype)


if __name__ == "__main__":
    import os
    os.environ.setdefault('PTO_TILE_LIB_CODE_PATH', '/data/h00520348/pto-isa')
    os.environ.setdefault('TILE_FWK_DEVICE_ID', '5')
    
    import torch_npu
    torch.npu.set_device(5)
    
    # 测试decode_attn
    query = torch.randn(Nq, D, dtype=torch.bfloat16, device='npu:5')
    key_cache = torch.randn(Nkv, 128, D, dtype=torch.bfloat16, device='npu:5')
    value_cache = torch.randn(Nkv, 128, D, dtype=torch.bfloat16, device='npu:5')
    mask = torch.zeros(128, dtype=torch.float32, device='npu:5')
    
    # Warmup
    output = decode_attention_wrapper(query, key_cache, value_cache, mask)
    torch.npu.synchronize()
    
    print(f"Query shape: {query.shape}")
    print(f"Key cache shape: {key_cache.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print("✓ decode_attn wrapper测试通过")