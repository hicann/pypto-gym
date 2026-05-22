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
Qwen3 Attention - 兼容prefill/decode模式

关键：
- prefill模式（Sq>1）：使用原始eager_attention_forward
- decode模式（Sq=1）：使用gym仓qwen3_decode_attn
"""

import torch
import math
import sys
from typing import Optional

from pypto_gym.ops.pypto_tile.qwen3_1_7b.qwen3_decode_attn import qwen3_decode_attn

# Qwen3-1.7B参数
Nq = 16
Nkv = 8
D = 128
GROUPS = Nq // Nkv
S2_TILE = 64


def decode_attn_forward_pto(
    query_states: torch.Tensor,  # [batch, Nq, 1, D]
    key_states: torch.Tensor,    # [batch, Nkv, Skv, D]
    value_states: torch.Tensor,  # [batch, Nkv, Skv, D]
    attention_mask: torch.Tensor,  # [batch, 1, 1, Skv] or None
    scaling: float,
) -> torch.Tensor:
    """
    Decode Attention（PyPTO实现）
    
    Args:
        query_states: [batch, Nq, 1, D] - decode模式下Sq=1
        key_states: [batch, Nkv, Skv, D] - KV cache中的K
        value_states: [batch, Nkv, Skv, D] - KV cache中的V
        attention_mask: [batch, 1, 1, Skv] causal mask
        scaling: 1/sqrt(D)
    
    Returns:
        attn_output: [batch, 1, Nq, D] (注意：需要后续transpose)
    
    Note:
        仅适用于decode模式（Sq=1）
        使用gym仓GQA-native实现
    """
    batch = query_states.shape[0]
    Skv = key_states.shape[2]
    
    # 确保是decode模式
    assert query_states.shape[2] == 1, f"decode_attn only for Sq=1, got {query_states.shape}"
    
    # 提取数据（squeeze seq_len维度）
    # query: [batch, Nq, 1, D] → [batch, Nq, D] → [Nq, D]（假设batch=1）
    # 实际decode模式batch通常=1
    if batch == 1:
        query = query_states.squeeze(2).squeeze(0)  # [Nq, D]
        key = key_states.squeeze(0)  # [Nkv, Skv, D]
        value = value_states.squeeze(0)  # [Nkv, Skv, D]
    else:
        # 多batch情况：需要逐batch处理（gym仓kernel不支持batch>1）
        raise NotImplementedError("decode_attn不支持batch>1")
    
    # KV cache padding（gym仓要求S2_TILE=64倍数）
    Skv_p = ((Skv + S2_TILE - 1) // S2_TILE) * S2_TILE
    
    key_pad = torch.zeros(Nkv, Skv_p, D, dtype=torch.bfloat16, device=query.device)
    value_pad = torch.zeros(Nkv, Skv_p, D, dtype=torch.bfloat16, device=query.device)
    key_pad[:, :Skv] = key.to(torch.bfloat16)
    value_pad[:, :Skv] = value.to(torch.bfloat16)
    
    # Mask处理
    # attention_mask: [batch, 1, 1, Skv] → [Skv_p]
    # decode模式下，有效部分全0，padding部分-1e30
    if attention_mask is not None:
        mask_1d = attention_mask.squeeze().to(torch.float32)  # [Skv]
    else:
        mask_1d = torch.zeros(Skv, dtype=torch.float32, device=query.device)
    
    # 扩展到padding长度
    mask_pad = torch.zeros(Skv_p, dtype=torch.float32, device=query.device)
    mask_pad[:Skv] = mask_1d
    mask_pad[Skv:] = -1e30  # padding部分mask掉
    
    # Gym仓要求mask shape: [Nkv, GROUPS, Skv_p]
    mask_3d = mask_pad.view(1, 1, Skv_p).expand(Nkv, GROUPS, Skv_p).contiguous()
    
    # 调用gym仓kernel
    query_bf = query.to(torch.bfloat16)
    output_bf = torch.empty(Nq, D, dtype=torch.bfloat16, device=query.device)
    
    qwen3_decode_attn(query_bf, key_pad, value_pad, mask_3d, output_bf)
    
    # 恢复shape：[Nq, D] → [batch, 1, Nq, D]
    attn_output = output_bf.to(query_states.dtype).unsqueeze(0).unsqueeze(1)
    
    return attn_output


def qwen3_attention_forward_with_decode(
    module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    Qwen3 Attention Forward - 兼容prefill/decode
    
    自动判断模式：
    - decode（Sq=1）且USE_PTO_DECODE_ATTN=True → 使用PyPTO
    - 其他情况 → 使用原始eager_attention_forward
    """
    # 判断是否是decode模式
    Sq = query_states.shape[2]
    is_decode_mode = (Sq == 1)
    
    # 检查PyPTO开关
    pto_kernels = sys.modules.get("qwen3_pto_kernels")
    use_pto_decode = (
        pto_kernels is not None and 
        pto_kernels.USE_PTO_DECODE_ATTN and 
        is_decode_mode
    )
    
    if use_pto_decode:
        # 使用PyPTO decode_attn
        try:
            attn_output = decode_attn_forward_pto(
                query_states, key_states, value_states,
                attention_mask, scaling
            )
            # decode模式下不需要返回attn_weights
            return attn_output, None
        except Exception as e:
            # 失败时fallback到原始实现
            print(f"[Warning] decode_attn失败，fallback: {e}")
            use_pto_decode = False
    
    # 使用原始eager_attention_forward（直接复制逻辑）
    # 注意：这里不能用from .modeling_qwen3，因为相对导入会失败
    # 直接实现repeat_kv和attention计算
    
    # repeat_kv
    batch, num_key_value_heads, slen, head_dim = key_states.shape
    if module.num_key_value_groups > 1:
        key_states = key_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, module.num_key_value_groups, slen, head_dim
        ).reshape(batch, num_key_value_heads * module.num_key_value_groups, slen, head_dim)
        value_states = value_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, module.num_key_value_groups, slen, head_dim
        ).reshape(batch, num_key_value_heads * module.num_key_value_groups, slen, head_dim)
    
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    
    return attn_output, attn_weights


print(f"[decode_attn_integration] 已加载，支持prefill/decode自动切换")