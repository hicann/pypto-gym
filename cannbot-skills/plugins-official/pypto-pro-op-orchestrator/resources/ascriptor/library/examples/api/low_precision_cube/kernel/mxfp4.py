# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Preserved mxfp4_carrier_matmul device body; numerical references are independent and local."""

from ascriptor.a5 import DT, FP4_E1M2_MAX_VALUE, FP4_E2M1_MAX_VALUE, GMTensor, Position, Tensor, Var, auto_sync, e8m0_to_fp32, fp32_to_fp4_e1m2, fp32_to_fp4_e2m1, fp4_e1m2_to_fp32, fp4_e2m1_to_fp32, gm_to_l1_mx_scale, kernel, matmul_mx, GM, f32, i32, u8
M_VALUE = 16
N_VALUE = 16
K_VALUE = 64
K_CARRIER_VALUE = (K_VALUE + 1) // 2
SCALE_COLS_VALUE = 2 * ((K_VALUE + 63) // 64)

@kernel(mode='cube', block_dim=1)
def mxfp4_carrier_matmul_kernel(a_carrier: GM[u8, (16, 32)], b_carrier: GM[u8, (16, 32)], scale_a_gm: GM[u8, (1, 32)], scale_b_gm: GM[u8, (1, 32)], out: GM[f32, (16, 16)], dummy: i32):
    l1a_u8 = Tensor(DT.uint8, [M_VALUE, K_CARRIER_VALUE], Position.L1)
    l1b_u8 = Tensor(DT.uint8, [N_VALUE, K_CARRIER_VALUE], Position.L1)
    l1a = l1a_u8.reinterpret(DT.fp4_e2m1, name='l1a_fp4_e2m1')
    l1b = l1b_u8.reinterpret(DT.fp4_e1m2, name='l1b_fp4_e1m2')
    scale_a = Tensor(DT.uint8, [M_VALUE, SCALE_COLS_VALUE], Position.L1)
    scale_b = Tensor(DT.uint8, [N_VALUE, SCALE_COLS_VALUE], Position.L1)
    l0c = Tensor(DT.float, [M_VALUE, N_VALUE], Position.L0C)
    with auto_sync():
        l1a_u8[:, :] <<= a_carrier[:, :]
        l1b_u8[:, :] <<= b_carrier[:, :]
        gm_to_l1_mx_scale(scale_a, scale_a_gm)
        gm_to_l1_mx_scale(scale_b, scale_b_gm)
        matmul_mx(l0c, l1a, l1b, scale_a, scale_b, m=M_VALUE, n=N_VALUE, k=K_VALUE, is_init=True)
        out[:, :] <<= l0c
    return out
