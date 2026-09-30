# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F403, F405, F841
"""Reviewed production closures; host inputs/references are local and independent.

Pure-vector teaching units explicitly launch one vector participant. Original
mode=vec bodies, masks, byte footprints and overlapping-write barriers remain.
"""

# SHA256: 1f1a1a5e38639b470a4ce6c382bea95ffbad6dea138e044bc56123ee9785e26f

from ascriptor.a5 import *


@vf()
def gatherb64_vf(src_ub: Tensor, dst_ub: Tensor, idx_u: Tensor):
    ui = Reg(DT.uint32)
    ub_to_reg_normal(ui, idx_u)
    d = Reg(DT.int64)
    ub_to_reg_gatherb(d, src_ub, ui)
    reg_to_ub_normal(dst_ub, d)


@kernel(mode="vec", block_dim=1)
def gatherb64(src: GM[i64, (1, 64)], idx: GM[i32, (1, 64)], out: GM[i64, (1, 32)]):
    src_ub = Tensor(DT.int64, [1, 64], Position.UB)  # 16 blocks of 32 bytes (four int64 each)
    dst_ub = Tensor(DT.int64, [1, 32], Position.UB)  # 8 blocks
    idx_s = Tensor(DT.int, [1, 64], Position.UB)
    idx_u = reinterpret(idx_s, DT.uint32)
    with auto_sync():
        src_ub <<= src
        idx_s <<= idx
        gatherb64_vf(src_ub, dst_ub, idx_u)
        out <<= dst_ub
    return out
