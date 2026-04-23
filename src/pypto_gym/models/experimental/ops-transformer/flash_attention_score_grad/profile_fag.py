#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""采集性能数据 (生成泳道图)"""
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_attention_score_grad_golden import (
    ForwardDataConfig, generate_forward_data,
)
from flash_attention_score_grad_impl import (
    NUM_HEADS, HEAD_DIM, S_TILE,
    flash_attention_score_grad_kernel_profile,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
))


if __name__ == "__main__":
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    import torch_npu
    torch.npu.set_device(device_id)

    batch_size, num_heads, seq_len, head_dim = 2, 8, 1024, 64
    device = f"npu:{device_id}"
    fwd_cfg = ForwardDataConfig(
        batch_size, num_heads, seq_len, head_dim, device=device)
    fwd = generate_forward_data(fwd_cfg)
    q, k, v, dy_t = fwd.q, fwd.k, fwd.v, fwd.dy
    sm, ss, ao, scale = fwd.softmax_max, fwd.softmax_sum, fwd.attention_out, fwd.scale

    q_flat = q.reshape(-1, head_dim).contiguous()
    k_flat = k.reshape(-1, head_dim).contiguous()
    v_flat = v.reshape(-1, head_dim).contiguous()
    dy_flat = dy_t.reshape(-1, head_dim).contiguous()
    ao_flat = ao.reshape(-1, head_dim).contiguous()
    sm_flat = sm.reshape(-1, 8).contiguous()
    ss_flat = ss.reshape(-1, 8).contiguous()
    dq_flat = torch.empty_like(q_flat)
    dk_flat = torch.empty_like(k_flat)
    dv_flat = torch.empty_like(v_flat)
    batch_tensor = torch.zeros(
        batch_size, dtype=torch.int32, device=device)

    logger.info("Running profile: B=%d, N=%d, S=%d, D=%d",
                batch_size, num_heads, seq_len, head_dim)
    flash_attention_score_grad_kernel_profile(
        q_flat, k_flat, v_flat, dy_flat,
        sm_flat, ss_flat, ao_flat,
        dq_flat, dk_flat, dv_flat,
        batch_tensor, scale,
    )
    logger.info("Done. Check output/ for swimlane data.")
