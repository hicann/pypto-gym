# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------



"""PyPTO moe_finalize_routing_v2 kernel implementation.

关键设计：
  - wrapper 预处理索引查找（创建 indexed_x 和 indexed_bias）- 数据准备
  - wrapper 为可选参数创建默认值 tensor（全0 或全1）
  - 单一 kernel 处理所有参数组合，通过 has_residual/has_bias/has_scales 标志判断
  - kernel 内部完成所有核心计算逻辑
"""

import pypto
import torch
from typing import Optional
import sys


# 从环境变量获取 run_mode
def _get_run_mode():
    """从命令行参数获取 run_mode。"""
    for arg in sys.argv:
        if arg == "--run_mode=sim" or arg == "--run-mode=sim":
            return pypto.RunMode.SIM
        if arg == "--run_mode=npu" or arg == "--run-mode=npu":
            return pypto.RunMode.NPU
    # 默认返回 NPU
    return pypto.RunMode.NPU


global_run_mode = _get_run_mode()

# 注意：PyPTO 在 sim 模式下仍然要求 tensor 在 NPU 设备上
# 所以即使是 sim 模式，kernel 也使用 NPU run_mode
kernel_run_mode = pypto.RunMode.NPU  # 始终使用 NPU 模式


# ─────────────────────────────────────────────
# 统一 JIT Kernel - 支持所有参数组合
# ─────────────────────────────────────────────

@pypto.frontend.jit(
    runtime_options={"run_mode": kernel_run_mode},
    pass_options={"vec_nbuffer_setting": {-2: 1, -1: 8}}
)
def moe_finalize_routing_v2_kernel(  # pylint: disable=huawei-too-many-arguments
    indexed_x: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    indexed_bias: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    scales: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    x1: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    x2: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),
    K: int,
    H: int,
    has_x1: bool,
    has_x2: bool,
    has_bias: bool,
    has_scales: bool,
):
    """MoE 路由聚合 kernel - 统一版本，支持所有参数组合。

    计算逻辑（通用公式）：
      out[i, :] = x1[i, :] + x2[i, :] + Σ_k(weighted_output[i*K+k, :])

      where weighted_output[i*K+k, :] = scales[i,k] * (indexed_x[i*K+k,:] + indexed_bias[i*K+k,:])

    Args:
        indexed_x: [NUM_ROWS*K, H] BF16 - 预处理好的 tensor
        indexed_bias: [NUM_ROWS*K, H] BF16 - 预处理好的 bias（或全0）
        scales: [NUM_ROWS, K] BF16 - 路由权重（或全1）
        x1: [NUM_ROWS, H] BF16 - 残差连接1（或全0）
        x2: [NUM_ROWS, H] BF16 - 残差连接2（或全0）
        out: [NUM_ROWS, H] BF16 - 输出（已初始化为 0）
        K: 编译期常量，专家数
        H: 编译期常量，hidden size
        has_x1: 是否有 x1
        has_x2: 是否有 x2
        has_bias: 是否有 bias
        has_scales: 是否有 scales
    """

    NUM_ROWS = out.shape[0]  # SymbolicScalar（动态轴）
    tile_h = H
    pypto.set_vec_tile_shapes(1, tile_h)

    # 主计算循环：遍历 NUM_ROWS
    for i in pypto.loop(NUM_ROWS, name="rows_loop", idx_name="i", unroll_list=[64, 16, 4, 1]):
        # 创建临时 FP32 累加器（遵循 execution-constraints.md 第5.4节）
        out_row_fp32 = pypto.tensor([1, tile_h], pypto.DT_FP32, "out_row_fp32")

        # 内循环：遍历 K 个专家（静态轴使用 Python for）
        for k in range(K):
            idx_pos = i * K + k

            # 获取专家输出行
            dst_row_bf16 = pypto.view(indexed_x, [1, tile_h], [idx_pos, 0], valid_shape=[1, H])
            dst_row_fp32 = pypto.cast(dst_row_bf16, pypto.DT_FP32)

            # 添加专家偏置（如果 has_bias=True）
            if has_bias:
                bias_row_bf16 = pypto.view(indexed_bias, [1, tile_h], [idx_pos, 0], valid_shape=[1, H])
                bias_row_fp32 = pypto.cast(bias_row_bf16, pypto.DT_FP32)
                dst_row_fp32 = pypto.add(dst_row_fp32, bias_row_fp32)

            # 应用路由权重（如果 has_scales=True）
            if has_scales:
                scale_bf16 = pypto.view(scales, [1, 1], [i, k], valid_shape=[1, 1])
                scale_fp32 = pypto.cast(scale_bf16, pypto.DT_FP32)
                dst_row_fp32 = pypto.mul(dst_row_fp32, scale_fp32)

            # 累加到当前行（判断是否为第一个专家）
            if k == 0:
                # 第一个专家：初始化累加器
                if has_x1:
                    # 添加 x1
                    x1_row_bf16 = pypto.view(x1, [1, tile_h], [i, 0], valid_shape=[1, H])
                    x1_row_fp32 = pypto.cast(x1_row_bf16, pypto.DT_FP32)
                    dst_row_fp32 = pypto.add(dst_row_fp32, x1_row_fp32)

                if has_x2:
                    # 添加 x2
                    x2_row_bf16 = pypto.view(x2, [1, tile_h], [i, 0], valid_shape=[1, H])
                    x2_row_fp32 = pypto.cast(x2_row_bf16, pypto.DT_FP32)
                    dst_row_fp32 = pypto.add(dst_row_fp32, x2_row_fp32)

                # 初始化为残差 + 专家输出（如果有残差）
                out_row_fp32[:] = dst_row_fp32
            else:
                # 后续专家：累加
                out_row_fp32[:] = out_row_fp32 + dst_row_fp32

        # 写回输出
        out_row_bf16 = pypto.cast(out_row_fp32, pypto.DT_BF16)
        pypto.assemble(out_row_bf16, [i, 0], out)


# ─────────────────────────────────────────────
# Wrapper 函数（导出接口）
# ─────────────────────────────────────────────

def moe_finalize_routing_v2_wrapper(
    expanded_x: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    x1: Optional[torch.Tensor] = None,
    x2: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expert_idx: Optional[torch.Tensor] = None,
    drop_pad_mode: int = 2,
) -> torch.Tensor:
    """算子 wrapper，供 test 调用。

    Args:
        expanded_x: [NUM_ROWS*K, H] BF16 - MoE FFN 输出
        expanded_row_idx: [NUM_ROWS*K] INT32 - 行索引
        x1: 残差连接1，可选，shape=(NUM_ROWS, H)，BF16
        x2: 残差连接2，可选，shape=(NUM_ROWS, H)，BF16
        bias: 专家偏置，可选，shape=(E, H)，BF16
        scales: 路由权重，可选，shape=(NUM_ROWS, K)，BF16
        expert_idx: 专家索引，可选，shape=(NUM_ROWS, K)，INT32
        drop_pad_mode: 索引排列方式（0-3）

    Returns:
        输出 [NUM_ROWS, H] BF16
    """
    # ========================================
    # 1. 参数校验
    # ========================================

    # 约束3：x1Optional 未提供时，x2Optional 不能提供
    if x1 is None and x2 is not None:
        raise ValueError("x2 cannot be provided when x1 is None (constraint #3)")

    # 约束5：biasOptional 存在时，expertIdxOptional 必须存在
    if bias is not None and expert_idx is None:
        raise ValueError("expert_idx must be provided when bias is provided (constraint #5)")

    # ========================================
    # 2. 计算 K 和 NUM_ROWS
    # ========================================

    bsk = expanded_row_idx.shape[0]
    H = expanded_x.shape[-1]

    # 确定 K 值和 NUM_ROWS（与 golden 保持一致的推导逻辑）
    if scales is not None:
        K = scales.shape[1]
        NUM_ROWS = bsk // K
    elif x1 is not None:
        NUM_ROWS = x1.shape[0]
        K = bsk // NUM_ROWS
    elif x2 is not None:
        NUM_ROWS = x2.shape[0]
        K = bsk // NUM_ROWS
    elif expert_idx is not None:
        NUM_ROWS = expert_idx.shape[0]
        K = expert_idx.shape[1]
    else:
        # 无任何辅助信息，默认 K=1
        K = 1
        NUM_ROWS = bsk

    # ========================================
    # 3. 预处理：索引查找（构造 indexed_x）
    # ========================================

    indexed_x = torch.zeros((NUM_ROWS * K, H), dtype=torch.bfloat16, device=expanded_x.device)

    # 构造索引位置数组
    if drop_pad_mode == 0 or drop_pad_mode == 1:
        # 按列排列
        idx_positions = torch.arange(NUM_ROWS * K, dtype=torch.int32, device=expanded_x.device)
        idx_positions = idx_positions.reshape(K, NUM_ROWS).T.flatten()
    else:
        # 按行排列（默认：drop_pad_mode = 2 或 3）
        idx_positions = torch.arange(NUM_ROWS * K, dtype=torch.int32, device=expanded_x.device)

    # 获取索引值
    expanded_idx_values = expanded_row_idx[idx_positions]

    # 处理特殊情况（drop_pad 和 drop_less）
    if drop_pad_mode == 1 or drop_pad_mode == 3:
        # drop_pad 场景：值为 -1 时跳过
        mask = expanded_idx_values != -1
    elif drop_pad_mode == 0 or drop_pad_mode == 2:
        # drop_less 场景：越界时跳过
        mask = (expanded_idx_values >= 0) & (expanded_idx_values < expanded_x.shape[0])
    else:
        mask = torch.ones(NUM_ROWS * K, dtype=torch.bool, device=expanded_x.device)

    # 批量索引查找
    valid_idx_values = expanded_idx_values[mask]
    indexed_x_flat = indexed_x.view(NUM_ROWS * K, H)

    if valid_idx_values.numel() > 0:
        indexed_x_flat[mask] = expanded_x[valid_idx_values]

    # ========================================
    # 4. 预处理：bias 查找（如果 bias 存在）
    # ========================================

    indexed_bias = None
    if bias is not None and expert_idx is not None:
        indexed_bias = torch.zeros((NUM_ROWS * K, H), dtype=torch.bfloat16, device=expanded_x.device)

        # 预处理 bias：根据 expert_idx 查找对应的 bias 行
        indexed_bias_by_row = torch.index_select(bias, 0, expert_idx.view(-1))

        # 根据 drop_pad_mode 重新排列 indexed_bias（与 indexed_x 的排列方式一致）
        if drop_pad_mode == 0 or drop_pad_mode == 1:
            # 按列排列：需要 transpose 再 flatten
            indexed_bias = indexed_bias_by_row.view(NUM_ROWS, K, H).transpose(0, 1).reshape(NUM_ROWS * K, H)
        else:
            # 按行排列：直接使用 indexed_bias_by_row
            indexed_bias = indexed_bias_by_row

    # ========================================
    # 5. 初始化输出 tensor（始终初始化为 0）
    # ========================================

    output = torch.zeros((NUM_ROWS, H), dtype=torch.bfloat16, device=expanded_x.device)

    # ========================================
    # 6. 为可选参数创建默认值 tensor
    # ========================================

    # indexed_bias：如果不存在，创建全0 tensor
    indexed_bias_param = indexed_bias
    if indexed_bias_param is None:
        indexed_bias_param = torch.zeros((NUM_ROWS * K, H), dtype=torch.bfloat16, device=expanded_x.device)

    # scales：如果不存在，创建全1 tensor
    scales_param = scales
    if scales_param is None:
        scales_param = torch.ones((NUM_ROWS, K), dtype=torch.bfloat16, device=expanded_x.device)

    # x1, x2：如果不存在，创建全0 tensor
    x1_param = x1
    if x1_param is None:
        x1_param = torch.zeros((NUM_ROWS, H), dtype=torch.bfloat16, device=expanded_x.device)

    x2_param = x2
    if x2_param is None:
        x2_param = torch.zeros((NUM_ROWS, H), dtype=torch.bfloat16, device=expanded_x.device)

    # 判断标志
    has_x1_param = (x1 is not None)
    has_x2_param = (x2 is not None)
    has_bias_param = (bias is not None)
    has_scales_param = (scales is not None)

    # ========================================
    # 7. 调用统一的 kernel（支持所有参数组合）
    # ========================================

    moe_finalize_routing_v2_kernel(
        indexed_x,
        indexed_bias_param,
        scales_param,
        x1_param,
        x2_param,
        output,
        K,
        H,
        has_x1_param,
        has_x2_param,
        has_bias_param,
        has_scales_param,
    )

    return output