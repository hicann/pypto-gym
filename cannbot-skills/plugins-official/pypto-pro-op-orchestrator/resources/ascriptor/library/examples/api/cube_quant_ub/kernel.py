# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Seven explicit SINGLE-subblock quantized L0C-to-UB bridge cases."""
from ascriptor.a5 import *

QF_SCALE,QF_OFFSET=0.5,8
RQ_SCALE,RQ_OFFSET=0.5,0
DQ_SCALE=0.25
SC_SCALE=0.5


def _cvmutex():
    """FIX(cube fixpipe) -> V(vector) bridge for the L0C->UB store + the UB->GM read-back."""
    return CvMutex(0, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)

@kernel(mode='mix', block_dim=1)
def qf_i8(x: GM[f16, ('M', 'K')], y: GM[f16, ('N', 'K')], z: GM[i8, ('M', 'N')], M: i32, N: i32, K: i32):
    cvmutex = _cvmutex()
    l1x = Tensor(DT.half, [M, K], Position.L1)
    l1y = Tensor(DT.half, [N, K], Position.L1)
    l0c = Tensor(DT.float, [M, N], Position.L0C)
    ub = Tensor(z.dtype, [M, N], Position.UB)
    with auto_sync():
        l1x <<= x[:, :]
        l1y <<= y[:, :]
        matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
        cvmutex.lock()
        ub <<= l0c.requant(scale=QF_SCALE, offset=QF_OFFSET).subblk(0)
        cvmutex.ready()
        cvmutex.wait()
        if GetSubBlockIdx() == 0:
            z[0:M, 0:N] <<= ub[0:M, 0:N]
        cvmutex.free()
    return z

@kernel(mode='mix', block_dim=1)
def qf_u8(x: GM[f16, ('M', 'K')], y: GM[f16, ('N', 'K')], z: GM[u8, ('M', 'N')], M: i32, N: i32, K: i32):
    cvmutex = _cvmutex()
    l1x = Tensor(DT.half, [M, K], Position.L1)
    l1y = Tensor(DT.half, [N, K], Position.L1)
    l0c = Tensor(DT.float, [M, N], Position.L0C)
    ub = Tensor(z.dtype, [M, N], Position.UB)
    with auto_sync():
        l1x <<= x[:, :]
        l1y <<= y[:, :]
        matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
        cvmutex.lock()
        ub <<= l0c.requant(scale=QF_SCALE, offset=QF_OFFSET).subblk(0)
        cvmutex.ready()
        cvmutex.wait()
        if GetSubBlockIdx() == 0:
            z[0:M, 0:N] <<= ub[0:M, 0:N]
        cvmutex.free()
    return z

@kernel(mode='mix', block_dim=1)
def rq_i8(x: GM[i8, ('M', 'K')], y: GM[i8, ('N', 'K')], z: GM[i8, ('M', 'N')], M: i32, N: i32, K: i32):
    cvmutex = _cvmutex()
    l1x = Tensor(DT.int8, [M, K], Position.L1)
    l1y = Tensor(DT.int8, [N, K], Position.L1)
    l0c = Tensor(DT.int, [M, N], Position.L0C)
    ub = Tensor(z.dtype, [M, N], Position.UB)
    with auto_sync():
        l1x <<= x[:, :]
        l1y <<= y[:, :]
        matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
        cvmutex.lock()
        ub <<= l0c.requant(scale=RQ_SCALE, offset=RQ_OFFSET).subblk(0)
        cvmutex.ready()
        cvmutex.wait()
        if GetSubBlockIdx() == 0:
            z[0:M, 0:N] <<= ub[0:M, 0:N]
        cvmutex.free()
    return z

@kernel(mode='mix', block_dim=1)
def deq_f16(x: GM[i8, ('M', 'K')], y: GM[i8, ('N', 'K')], z: GM[f16, ('M', 'N')], M: i32, N: i32, K: i32):
    cvmutex = _cvmutex()
    l1x = Tensor(DT.int8, [M, K], Position.L1)
    l1y = Tensor(DT.int8, [N, K], Position.L1)
    l0c = Tensor(DT.int, [M, N], Position.L0C)
    ub = Tensor(DT.half, [M, N], Position.UB)
    with auto_sync():
        l1x <<= x[:, :]
        l1y <<= y[:, :]
        matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
        cvmutex.lock()
        ub <<= l0c.requant(scale=DQ_SCALE).subblk(0)
        cvmutex.ready()
        cvmutex.wait()
        if GetSubBlockIdx() == 0:
            z[0:M, 0:N] <<= ub[0:M, 0:N]
        cvmutex.free()
    return z

@kernel(mode='mix', block_dim=1)
def scaled_f16(x: GM[f16, ('M', 'K')], y: GM[f16, ('N', 'K')], z: GM[f16, ('M', 'N')], M: i32, N: i32, K: i32):
    cvmutex = _cvmutex()
    l1x = Tensor(DT.half, [M, K], Position.L1)
    l1y = Tensor(DT.half, [N, K], Position.L1)
    l0c = Tensor(DT.float, [M, N], Position.L0C)
    ub = Tensor(DT.half, [M, N], Position.UB)
    with auto_sync():
        l1x <<= x[:, :]
        l1y <<= y[:, :]
        matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
        cvmutex.lock()
        ub <<= l0c.requant(scale=SC_SCALE).subblk(0)
        cvmutex.ready()
        cvmutex.wait()
        if GetSubBlockIdx() == 0:
            z[0:M, 0:N] <<= ub[0:M, 0:N]
        cvmutex.free()
    return z

@kernel(mode='mix', block_dim=1)
def relu_i8(x: GM[f16, ('M', 'K')], y: GM[f16, ('N', 'K')], z: GM[i8, ('M', 'N')], M: i32, N: i32, K: i32):
    cvmutex = _cvmutex()
    l1x = Tensor(DT.half, [M, K], Position.L1)
    l1y = Tensor(DT.half, [N, K], Position.L1)
    l0c = Tensor(DT.float, [M, N], Position.L0C)
    ub = Tensor(z.dtype, [M, N], Position.UB)
    with auto_sync():
        l1x <<= x[:, :]
        l1y <<= y[:, :]
        matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
        cvmutex.lock()
        ub <<= l0c.relu().requant(scale=QF_SCALE, offset=QF_OFFSET).subblk(0)
        cvmutex.ready()
        cvmutex.wait()
        if GetSubBlockIdx() == 0:
            z[0:M, 0:N] <<= ub[0:M, 0:N]
        cvmutex.free()
    return z

@kernel(mode='mix', block_dim=1)
def split_i8(x: GM[f16, ('M', 'K')], y: GM[f16, ('N', 'K')], z: GM[i8, ('M', 'N')], M: i32, N: i32, K: i32):
    cv0 = CvMutex(0, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    cv1 = CvMutex(1, depth=1, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    l1x = Tensor(DT.half, [M, K], Position.L1)
    l1y = Tensor(DT.half, [N, K], Position.L1)
    l0c = Tensor(DT.float, [M, N], Position.L0C)
    ub0 = Tensor(z.dtype, [16, N], Position.UB)
    ub1 = Tensor(z.dtype, [16, N], Position.UB)
    with auto_sync():
        l1x <<= x[:, :]
        l1y <<= y[:, :]
        matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
        cv0.lock()
        ub0 <<= l0c[0:16, :].requant(scale=QF_SCALE, offset=QF_OFFSET).subblk(0)
        cv0.ready()
        cv1.lock()
        ub1 <<= l0c[16:32, :].requant(scale=QF_SCALE, offset=QF_OFFSET).subblk(0)
        cv1.ready()
        cv0.wait()
        if GetSubBlockIdx() == 0:
            z[0:16, :] <<= ub0[0:16, :]
        cv0.free()
        cv1.wait()
        if GetSubBlockIdx() == 0:
            z[16:32, :] <<= ub1[0:16, :]
        cv1.free()
    return z

ENTRIES = {'qf_i8': qf_i8, 'qf_u8': qf_u8, 'rq_i8': rq_i8, 'deq_f16': deq_f16, 'scaled_f16': scaled_f16, 'relu_i8': relu_i8, 'split_i8': split_i8}
