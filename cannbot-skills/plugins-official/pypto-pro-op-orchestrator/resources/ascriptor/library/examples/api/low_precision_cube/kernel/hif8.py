# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Preserved hif8_carrier_matmul device body; numerical references are independent and local."""

from ascriptor.a5 import *
M_VALUE = 32
N_VALUE = 64
K_VALUE = 64
SPLIT_N_VALUE = 32
SPLIT_K_VALUE = 32

@kernel(mode='cube', block_dim=1)
def hif8_carrier_matmul_kernel(x_carrier: GM[u8, ('M', 'K')], y_carrier: GM[u8, ('N', 'K')], z: GM[f32, ('M', 'N')], M: i32, N: i32, K: i32):
    l1x_carrier = Tensor(DT.uint8, [M, K], Position.L1)
    l1y_carrier = Tensor(DT.uint8, [N, K], Position.L1)
    l0a_carrier = Tensor(DT.uint8, [M, K], Position.L0A)
    l0b_carrier = Tensor(DT.uint8, [N, K], Position.L0B)
    l0c = Tensor(DT.float, [M, N], Position.L0C)
    l1x_hif8 = l1x_carrier.reinterpret(DT.hif8, name='l1x_hif8')
    l1y_hif8 = l1y_carrier.reinterpret(DT.hif8, name='l1y_hif8')
    l0a_hif8 = l0a_carrier.reinterpret(DT.hif8, name='l0a_hif8')
    l0b_hif8 = l0b_carrier.reinterpret(DT.hif8, name='l0b_hif8')
    with auto_sync():
        l1x_carrier <<= x_carrier[:, :]
        l1y_carrier <<= y_carrier[:, :]
        l0a_hif8 <<= l1x_hif8
        l0b_hif8 <<= l1y_hif8
        mmad(l0c, l0a_hif8, l0b_hif8, M=M, N=N, K=K, is_init=True)
        z[:, :] <<= l0c
    return z

@kernel(mode='cube', block_dim=1)
def hif8_carrier_matmul_matrix_kernel(a_nt_carrier: GM[u8, (32, 64)], a_t_carrier: GM[u8, (64, 32)], b_nt_carrier: GM[u8, (64, 64)], b_t_carrier: GM[u8, (64, 64)], z_nn_nosplit: GM[f32, (32, 64)], z_nn_splitn: GM[f32, (32, 64)], z_nn_splitk: GM[f32, (32, 64)], z_nt_nosplit: GM[f32, (32, 64)], z_nt_splitn: GM[f32, (32, 64)], z_nt_splitk: GM[f32, (32, 64)], z_tn_nosplit: GM[f32, (32, 64)], z_tn_splitn: GM[f32, (32, 64)], z_tn_splitk: GM[f32, (32, 64)], z_tt_nosplit: GM[f32, (32, 64)], z_tt_splitn: GM[f32, (32, 64)], z_tt_splitk: GM[f32, (32, 64)], dummy: i32):
    l1a_nt_carrier = Tensor(DT.uint8, [M_VALUE, K_VALUE], Position.L1)
    l1a_t_carrier = Tensor(DT.uint8, [K_VALUE, M_VALUE], Position.L1)
    l1b_nt_carrier = Tensor(DT.uint8, [N_VALUE, K_VALUE], Position.L1)
    l1b_t_carrier = Tensor(DT.uint8, [K_VALUE, N_VALUE], Position.L1)
    l0c = Tensor(DT.float, [M_VALUE, N_VALUE], Position.L0C)
    l1a_nt_hif8 = l1a_nt_carrier.reinterpret(DT.hif8, name='l1a_nt_hif8')
    l1a_t_hif8 = l1a_t_carrier.reinterpret(DT.hif8, name='l1a_t_hif8')
    l1b_nt_hif8 = l1b_nt_carrier.reinterpret(DT.hif8, name='l1b_nt_hif8')
    l1b_t_hif8 = l1b_t_carrier.reinterpret(DT.hif8, name='l1b_t_hif8')
    with auto_sync():
        l1a_nt_carrier <<= a_nt_carrier[:, :]
        l1a_t_carrier <<= a_t_carrier[:, :]
        l1b_nt_carrier <<= b_nt_carrier[:, :]
        l1b_t_carrier <<= b_t_carrier[:, :]
        matmul(l0c, l1a_nt_hif8, l1b_nt_hif8, m=M_VALUE, n=N_VALUE, k=K_VALUE, is_init=True)
        z_nn_nosplit[:, :] <<= l0c
        matmul(l0c, l1a_nt_hif8, l1b_nt_hif8, splitn=SPLIT_N_VALUE, m=M_VALUE, n=N_VALUE, k=K_VALUE, is_init=True)
        z_nn_splitn[:, :] <<= l0c
        matmul(l0c, l1a_nt_hif8, l1b_nt_hif8, splitk=SPLIT_K_VALUE, m=M_VALUE, n=N_VALUE, k=K_VALUE, is_init=True)
        z_nn_splitk[:, :] <<= l0c
        matmul(l0c, l1a_nt_hif8, l1b_t_hif8.T, m=M_VALUE, n=N_VALUE, k=K_VALUE, is_init=True)
        z_nt_nosplit[:, :] <<= l0c
        matmul(l0c, l1a_nt_hif8, l1b_t_hif8.T, splitn=SPLIT_N_VALUE, m=M_VALUE, n=N_VALUE, k=K_VALUE, is_init=True)
        z_nt_splitn[:, :] <<= l0c
        matmul(l0c, l1a_nt_hif8, l1b_t_hif8.T, splitk=SPLIT_K_VALUE, m=M_VALUE, n=N_VALUE, k=K_VALUE, is_init=True)
        z_nt_splitk[:, :] <<= l0c
        matmul(l0c, l1a_t_hif8.T, l1b_nt_hif8, m=M_VALUE, n=N_VALUE, k=K_VALUE, is_init=True)
        z_tn_nosplit[:, :] <<= l0c
        matmul(l0c, l1a_t_hif8.T, l1b_nt_hif8, splitn=SPLIT_N_VALUE, m=M_VALUE, n=N_VALUE, k=K_VALUE, is_init=True)
        z_tn_splitn[:, :] <<= l0c
        matmul(l0c, l1a_t_hif8.T, l1b_nt_hif8, splitk=SPLIT_K_VALUE, m=M_VALUE, n=N_VALUE, k=K_VALUE, is_init=True)
        z_tn_splitk[:, :] <<= l0c
        matmul(l0c, l1a_t_hif8.T, l1b_t_hif8.T, m=M_VALUE, n=N_VALUE, k=K_VALUE, is_init=True)
        z_tt_nosplit[:, :] <<= l0c
        matmul(l0c, l1a_t_hif8.T, l1b_t_hif8.T, splitn=SPLIT_N_VALUE, m=M_VALUE, n=N_VALUE, k=K_VALUE, is_init=True)
        z_tt_splitn[:, :] <<= l0c
        matmul(l0c, l1a_t_hif8.T, l1b_t_hif8.T, splitk=SPLIT_K_VALUE, m=M_VALUE, n=N_VALUE, k=K_VALUE, is_init=True)
        z_tt_splitk[:, :] <<= l0c
    return (z_nn_nosplit, z_nn_splitn, z_nn_splitk, z_nt_nosplit, z_nt_splitn, z_nt_splitk, z_tn_nosplit, z_tn_splitn, z_tn_splitk, z_tt_nosplit, z_tt_splitn, z_tt_splitk)
