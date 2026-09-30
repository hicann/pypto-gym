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

# SHA256: 3586df8f502cce3ba1fd69c5945f13319ab2410506a379d3d89f03ccec1e876d

from ascriptor.a5 import *


@kernel(mode="vec", block_dim=1)
def gm_view(x: GM[f32, (4, 48)], o: GM[f32, (4, 48)]):
    ub = Tensor(DT.float, [4, 48], Position.UB, name="gv_ub")
    ubw = Tensor(DT.float, [4, 32], Position.UB, name="gv_win")
    v = x.view([4, 32], strides=[48, 1], offset=8)
    vo = o.view([4, 32], strides=[48, 1], offset=16)
    with auto_sync():
        ub[:, :] <<= x
        o <<= ub  # baseline: o = x (every byte written)
        ubw[:, :] <<= v[:, :]  # the padded-row read window
        bar_mte3()  # the baseline's bursts must all land before this one starts
        vo[:, :] <<= ubw[:, :]  # the shifted write window
    return o


# SHA256: 45f163b639719f6811bd1a2623a9a3b30248b20107018a9ecb3da51b3f0a12da


@kernel(mode="vec", block_dim=1)
def gm_view_gather(x: GM[f32, (4, 48)], o: GM[f32, (4, 12)]):
    ub = Tensor(DT.float, [4, 16], Position.UB, name="gvg_ub")  # 64-byte rows: the UB port steps 32-byte blocks
    v = x.view([4, 12], strides=[48, 4])  # every 4th element of each padded row
    with auto_sync():
        ub[:, 0:12] <<= v[:, :]
        o <<= ub[:, 0:12]
    return o


@kernel(mode="vec", block_dim=1)
def gm_view_rank3(x: GM[f32, (4, 48)], o: GM[f32, (4, 8)]):
    ub = Tensor(DT.float, [4, 8], Position.UB, name="gvr_ub")
    v = x.view([2, 2, 8], strides=[96, 16, 1])  # a 3D window landing row-major in [4, 8]
    with auto_sync():
        ub[:, :] <<= v
        o <<= ub
    return o
