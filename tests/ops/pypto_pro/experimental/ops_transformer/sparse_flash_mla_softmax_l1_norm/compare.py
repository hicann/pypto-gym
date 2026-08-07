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
"""Shared comparison helpers for sparse_flash_mla_softmax_l1_norm tests."""

import math

import torch


def _compare(npu_data, golden_data, rtol=1e-2, atol=1e-2):
    """比对 NPU 输出与 CPU FP32 golden，返回 (passed, max_abs, max_rel)。

    输出为 FP32，且部分 mask 场景下大量位置保持 0（未参与计算），因此直接用
    rtol/atol 逐元素判断，避免对 0 值做相对误差放大。
    """
    npu = npu_data.float().cpu()
    golden = golden_data.float().cpu()
    if npu.shape != golden.shape:
        return False, float("inf"), float("inf")

    diff = (npu - golden).abs()
    max_abs = diff.max().item()
    denom = golden.abs().clamp_min(1e-6)
    max_rel = (diff / denom).max().item()

    passed = torch.allclose(npu, golden, rtol=rtol, atol=atol)
    return passed, max_abs, max_rel


def _max_abs_diff(npu_data, golden_data):
    return (npu_data.float().cpu() - golden_data.float().cpu()).abs().max().item()