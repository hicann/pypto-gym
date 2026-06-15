# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------



"""PyPTO moe_finalize_routing_v2 golden reference implementation.

算子名称: moe_finalize_routing_v2
数学公式: 
  expertid = expertIdx[i,k]
  out(i,j) = x1_{i,j} + x2_{i,j} + Σ_{k=0}^{K}
    (scales_{i,k} * (
        expandedX_{expandedRowIdx_{i+k*num_rows},j} + bias_{expertid,j}
    ))

功能说明:
  该算子用于 MoE（Mixture of Experts）模型的最终路由聚合阶段。
  将多个专家网络的输出根据路由权重进行加权求和，并可选地加上残差连接和专家偏置。

参数说明:
  - expanded_x: MoE FFN输出，shape为 (NUM_ROWS*K, H)，dtype=BFLOAT16
  - expanded_row_idx: 行索引，用于查找 expanded_x 中的行，shape为 (NUM_ROWS*K,)，dtype=INT32
  - x1: 残差连接1，可选，shape为 (NUM_ROWS, H)，dtype=BFLOAT16
  - x2: 残差连接2，可选，shape为 (NUM_ROWS, H)，dtype=BFLOAT16
  - bias: 专家偏置，可选，shape为 (E, H)，dtype=BFLOAT16
  - scales: 路由权重，可选，shape为 (NUM_ROWS, K)，dtype=BFLOAT16
  - expert_idx: 专家索引，可选，shape为 (NUM_ROWS, K)，dtype=INT32
  - drop_pad_mode: 控制扩展行索引的排列方式，取值范围 [0, 3]
    - 0: drop less 场景，expandedRowIdx 按列排列
    - 1: drop pad 场景，expandedRowIdx 按列排列
    - 2: drop less 场景，expandedRowIdx 按行排列
    - 3: drop pad 场景，expandedRowIdx 按行排列

返回值:
  - out: 聚合后的输出结果，shape为 (NUM_ROWS, H)，dtype=BFLOAT16

"""

from typing import Optional

import torch

# ─────────────────────────────────────────────
# Golden 参考实现（纯 torch）
# ─────────────────────────────────────────────


def moe_finalize_routing_v2_golden(
    expanded_x: torch.Tensor,
    expanded_row_idx: torch.Tensor,
    x1: Optional[torch.Tensor] = None,
    x2: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    expert_idx: Optional[torch.Tensor] = None,
    drop_pad_mode: int = 2,
) -> torch.Tensor:
    """PyTorch 参考实现。

    仅使用 torch 标准操作，不依赖 pypto。
    使用 float32 进行中间计算以保证精度，最后转换回 bfloat16。

    Args:
        expanded_x: MoE FFN输出，shape为 (NUM_ROWS*K, H)，dtype=BFLOAT16
        expanded_row_idx: 行索引，shape为 (NUM_ROWS*K,)，dtype=INT32
        x1: 残差连接1，可选，shape为 (NUM_ROWS, H)，dtype=BFLOAT16
        x2: 残差连接2，可选，shape为 (NUM_ROWS, H)，dtype=BFLOAT16
        bias: 专家偏置，可选，shape为 (E, H)，dtype=BFLOAT16
        scales: 路由权重，可选，shape为 (NUM_ROWS, K)，dtype=BFLOAT16
        expert_idx: 专家索引，可选，shape为 (NUM_ROWS, K)，dtype=INT32
        drop_pad_mode: 控制扩展行索引的排列方式，默认值为 2

    Returns:
        out: 聚合后的输出结果，shape为 (NUM_ROWS, H)，dtype=BFLOAT16
    """
    # 使用 float32 进行中间计算以保证精度
    out_dtype = torch.float32

    # 获取基本维度
    bsk = expanded_row_idx.shape[0]
    h = expanded_x.shape[-1]

    # 处理空 tensor 的情况
    if h == 0:
        return torch.tensor([], dtype=torch.bfloat16)

    # 将 expanded_x reshape 为 2D (NUM_ROWS*K, H)
    expanded_x = expanded_x.reshape(-1, h)

    # 确定 K 值和 num_rows
    # 优先从 scales 推导；若无 scales，则从 x1/x2 的行数推导 num_rows，再反推 K
    if scales is not None:
        K = scales.shape[1]
        num_rows = bsk // K
    elif x1 is not None:
        num_rows = x1.shape[0]
        K = bsk // num_rows
    elif x2 is not None:
        num_rows = x2.shape[0]
        K = bsk // num_rows
    elif expert_idx is not None:
        num_rows = expert_idx.shape[0]
        K = expert_idx.shape[1]
    else:
        # 无任何辅助信息，默认 K=1
        K = 1
        num_rows = bsk

    # 初始化输出 tensor（在与输入相同的设备上）
    out = torch.zeros((num_rows, h), dtype=out_dtype, device=expanded_x.device)

    # 添加残差连接 x1
    if x1 is not None:
        out = out + x1.to(out_dtype)

    # 添加残差连接 x2
    if x2 is not None:
        out = out + x2.to(out_dtype)

    # 主计算循环：遍历 num_rows 和 K
    for i in range(num_rows):
        for k in range(K):
            # 根据 drop_pad_mode 计算索引位置
            if drop_pad_mode == 0 or drop_pad_mode == 1:
                # 按列排列
                expanded_row_idx_idx = k * num_rows + i
            else:
                # 按行排列
                expanded_row_idx_idx = i * K + k

            # 获取 expanded_row_idx_value
            expanded_row_idx_value = expanded_row_idx[expanded_row_idx_idx].item()

            # drop_pad 场景：跳过 padding 位置（值为 -1）
            if expanded_row_idx_value == -1:
                continue

            # drop_less 场景：跳过越界索引
            if drop_pad_mode != 1 and drop_pad_mode != 3:
                if expanded_row_idx_value >= expanded_x.shape[0]:
                    continue

            # 从 expanded_x 中获取目标行
            dst_row = expanded_x[int(expanded_row_idx_value), :].to(out_dtype)

            # 添加专家偏置（如果 bias 和 expert_idx 存在）
            if bias is not None and expert_idx is not None:
                expert_id = int(expert_idx[i, k].item())
                dst_row = dst_row + bias[expert_id, :].to(out_dtype)

            # 应用路由权重（如果 scales 存在）
            if scales is not None:
                dst_row = dst_row * scales[i, k].to(out_dtype)

            # 累加到输出
            out[i, :] = out[i, :] + dst_row

    # 返回结果（转换回 bfloat16）
    return out.to(torch.bfloat16)