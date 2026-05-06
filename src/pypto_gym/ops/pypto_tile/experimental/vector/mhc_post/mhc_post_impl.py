#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use the file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
mhc_post 算子实现

功能：MHC (Manifold-Constrained Hyper-Connections) 后处理融合算子
公式：
    h_post_term = h_post.unsqueeze(-1) * h_out.unsqueeze(-2)
    h_comb_term = torch.sum(h_res.unsqueeze(-1) * x.unsqueeze(-2), dim=-3)
    output = (h_post_term + h_comb_term).to(bfloat16)

展开形式：
    output[b*s, n, d] = h_post[b*s, n] * h_out[b*s, d] + sum_{k=0}^{N-1} h_res[b*s, k, n] * x[b*s, k, d]

输入：
    - x: [B*S, N, D] 输入 tensor (BF16), N=4, D∈{2560,5120}
    - h_res: [B*S, N, N] 流间混合权重矩阵 (FP32)
    - h_out: [B*S, D] 输出项数据 (BF16)
    - h_post: [B*S, N] 后处理权重 (FP32)

输出：
    - output: [B*S, N, D] 融合计算结果 (BF16)

注意事项：
    - sigmoid 和 sum 仅支持 FP32，所有计算在 FP32 下进行
    - 纯 Vector 算子，无 matmul 操作
"""

import pypto
import torch
import torch_npu

@pypto.frontend.jit(pass_options={"vec_nbuffer_setting": {-2: 1, -1: 8}})
def mhc_post_kernel_bf16(
    x: pypto.Tensor([pypto.DYNAMIC, 4, pypto.STATIC], pypto.DT_BF16),       # [B*S, N, D] BF16
    h_res: pypto.Tensor([pypto.DYNAMIC, 4, 4], pypto.DT_FP32),              # [B*S, N, N] FP32
    h_out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),      # [B*S, D] BF16
    h_post: pypto.Tensor([pypto.DYNAMIC, 4], pypto.DT_FP32),                # [B*S, N] FP32
    output: pypto.Tensor([pypto.DYNAMIC, 4, pypto.STATIC], pypto.DT_BF16),  # [B*S, N, D] BF16 (输出)
):
    
    """mhc_post kernel BF16 版本。
    
    实现融合计算：
    - h_post_term: h_post × h_out 广播乘法
    - h_comb_term: Σ(k=0..N-1) h_res[:,k,n] × x[:,k,d] 累加求和
    - output: h_post_term + h_comb_term

    BS=4096, N=4, D=2560

    Args:
        x: [B*S, N, D] 输入 tensor (BF16), N=4, D 由 STATIC 标记（变化时触发重编译）
        h_res: [B*S, N, N] 流间混合权重矩阵 (FP32)
        h_out: [B*S, D] 输出项数据 (BF16)
        h_post: [B*S, N] 后处理权重 (FP32)
        output: [B*S, N, D] 输出 tensor (BF16)
    """
    # === 获取动态形状值 ===
    pypto.experimental.set_operation_options(combine_axis=True)

    BS = x.shape[0]   # SymbolicScalar（动态轴）
    N = 4             # Python int（固定值）
    D = x.shape[2]    # SymbolicScalar（STATIC 标记的轴，返回符号值）
    

    h_post1 = pypto.reshape(h_post, [BS, N, 1], inplace=True)
    h_out1 = pypto.reshape(h_out, [BS, 1, D], inplace=True)
    h_res1 = pypto.reshape(h_res, [BS, N, N, 1], inplace=True)
    x1 = pypto.reshape(x, [BS, N, 1, D], inplace=True)

    for bs_idx, unroll_length in pypto.loop_unroll(0, BS, 1, name="LOOP_BS", idx_name="bs_idx", unroll_list=[128]):
        x_slice = x1[bs_idx: bs_idx + unroll_length, :, :, :]           
        h_res_slice = h_res1[bs_idx: bs_idx + unroll_length, :, :, :]   
        h_out_slice = h_out1[bs_idx: bs_idx + unroll_length, :, :]      
        h_post_slice = h_post1[bs_idx: bs_idx + unroll_length, :, :]   

        pypto.set_vec_tile_shapes(1, N, 1, 1280)         #尾轴不切最好，或者1280.unroll_length改成1    
        x_fp32 = pypto.cast(x_slice, pypto.DT_FP32)                

        pypto.set_vec_tile_shapes(1, N, 1280) 
        h_out_fp32 = pypto.cast(h_out_slice, pypto.DT_FP32)        
        
        h_post_term = pypto.mul(h_post_slice, h_out_fp32)           
        
        pypto.set_vec_tile_shapes(1, N, N, 1280)          # 1,4,2,2560 /1280      1,4,1,2560 /1280      4*2560*4=40K
        weighted = pypto.mul(h_res_slice, x_fp32)                    
        
        h_comb_term = pypto.sum(weighted, dim=1, keepdim=False)   
        
        pypto.set_vec_tile_shapes(1, N, 1280)
        result_fp32 = pypto.add(h_post_term, h_comb_term)          

        result_bf16 = pypto.cast(result_fp32, pypto.DT_BF16)      
        pypto.assemble(result_bf16, [bs_idx, 0, 0], output)        

# ─────────────────────────────────────────────
# Wrapper 函数（导出接口）
# ─────────────────────────────────────────────

def mhc_post_wrapper(
    x: torch.Tensor,
    h_res: torch.Tensor,
    h_out: torch.Tensor,
    h_post: torch.Tensor,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """算子 wrapper，供 test_mhc_post.py 调用。

    负责：
    1. 验证输入 shape/dtype
    2. 将输入 reshape 为 [B*S, ...] 格式
    3. 调用 JIT kernel（直接传递 torch tensor）
    4. 将输出 reshape 回原始格式

    Args:
        x: [B, S, N, D] 输入 tensor (BF16)
           N 固定为 4，D 为 2560 或 5120
        h_res: [B, S, N, N] 流间混合权重矩阵 (FP32)
        h_out: [B, S, D] 输出项数据 (BF16)
        h_post: [B, S, N] 后处理权重 (FP32)
        output: 可选输出 tensor，如未提供则自动构造

    Returns:
        output: [B, S, N, D] 融合计算结果 (BF16)
    """
    # 验证输入
    assert x.is_contiguous(), "x must be contiguous"
    assert h_res.is_contiguous(), "h_res must be contiguous"
    assert h_out.is_contiguous(), "h_out must be contiguous"
    assert h_post.is_contiguous(), "h_post must be contiguous"
    
    # 验证 shape
    B = x.shape[0]
    S = x.shape[1]
    N = x.shape[2]
    D = x.shape[3]
    
    assert N == 4, f"N must be 4, got {N}"
    
    assert h_res.shape == (B, S, N, N), \
        f"h_res shape mismatch: expected {(B, S, N, N)}, got {h_res.shape}"
    assert h_out.shape == (B, S, D), \
        f"h_out shape mismatch: expected {(B, S, D)}, got {h_out.shape}"
    assert h_post.shape == (B, S, N), \
        f"h_post shape mismatch: expected {(B, S, N)}, got {h_post.shape}"
    
    # 验证 dtype
    assert x.dtype == torch.bfloat16, \
        f"x dtype must be bfloat16, got {x.dtype}"
    assert h_res.dtype == torch.float32, \
        f"h_res dtype must be float32, got {h_res.dtype}"
    assert h_out.dtype == torch.bfloat16, \
        f"h_out dtype must be bfloat16, got {h_out.dtype}"
    assert h_post.dtype == torch.float32, \
        f"h_post dtype must be float32, got {h_post.dtype}"
    
    # 将输入 reshape 为 [B*S, ...]
    BS = B * S
    x_reshaped = x.view(BS, N, D).contiguous()
    h_res_reshaped = h_res.view(BS, N, N).contiguous()
    h_out_reshaped = h_out.view(BS, D).contiguous()
    h_post_reshaped = h_post.view(BS, N).contiguous()
    
    # 构造输出 tensor (reshaped)
    if output is None:
        output_reshaped = torch.empty(BS, N, D, dtype=torch.bfloat16, device=x.device)
    else:
        assert output.shape == (B, S, N, D), \
            f"output shape mismatch: expected {(B, S, N, D)}, got {output.shape}"
        assert output.dtype == torch.bfloat16, \
            f"output dtype must be bfloat16, got {output.dtype}"
        output_reshaped = output.view(BS, N, D).contiguous()

    # 直接传递 torch tensor 给 kernel
    mhc_post_kernel_bf16(
        x_reshaped, h_res_reshaped, h_out_reshaped, h_post_reshaped,
        output_reshaped
    )
    
    # 将输出 reshape 回原始格式
    return output_reshaped.view(B, S, N, D)