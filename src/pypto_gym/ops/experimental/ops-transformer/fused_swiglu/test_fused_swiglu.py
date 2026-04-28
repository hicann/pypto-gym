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
"""
Fused SwiGLU Operator Test
"""
import os
import math
import logging
import torch
import numpy as np
from numpy.testing import assert_allclose


logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        logging.info("Please set TILE_FWK_DEVICE_ID")
        return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])


def golden_fused_swiglu_fwd(x, w_g, w_fc, b_g, b_fc):
    gate = x.float() @ w_g.float() + b_g
    fc = x.float() @ w_fc.float() + b_fc
    gate_silu = gate * torch.sigmoid(gate)
    y = (gate_silu * fc).to(torch.bfloat16)
    return y


def test_fwd(m, k, n, device_id):
    logging.info(f"\n=== Forward Test [{m}, {k}] @ [{k}, {n}] ===")
    device = f'npu:{device_id}'
    np.random.seed(0)
    torch.manual_seed(0)

    x = torch.randn(m, k, dtype=torch.bfloat16, device=device) / math.sqrt(m)
    w_g = torch.randn(k, n, dtype=torch.bfloat16, device=device) / math.sqrt(k)
    w_fc = torch.randn(k, n, dtype=torch.bfloat16, device=device) / math.sqrt(k)
    b_g = torch.randn(1, n, dtype=torch.bfloat16, device=device) / math.sqrt(n)
    b_fc = torch.randn(1, n, dtype=torch.bfloat16, device=device) / math.sqrt(n)
    y_golden = golden_fused_swiglu_fwd(x, w_g, w_fc, b_g, b_fc)
    y_out = torch.empty(m, n, dtype=torch.bfloat16, device=device)

    from fused_swiglu_impl import fused_swiglu_fwd_kernel
    fused_swiglu_fwd_kernel(x, w_g, w_fc, b_g, b_fc, y_out)

    assert_allclose(y_out.cpu().float().numpy(), y_golden.cpu().float().numpy(), rtol=0.0078125, atol=0.0001)
    logging.info(f"\n=== Forward Pass ===")


def main():
    device_id = get_device_id()
    if device_id is None:
        return
    torch.npu.set_device(device_id)
    test_fwd(220000, 512, 1024, device_id)


if __name__ == "__main__":
    main()