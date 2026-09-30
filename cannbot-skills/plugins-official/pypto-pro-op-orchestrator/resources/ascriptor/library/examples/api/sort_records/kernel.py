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

# SHA256: 68d78d1ce319f8ca860f171cbb810f09416296711e81b066871e93515a65dd3e

from ascriptor.a5 import *


@kernel(mode="vec", block_dim=1)
def sort_family(x: GM[f32, (4, 32)], idx: GM[u32, (4, 32)], y_sort: GM[f32, (4, 64)],
    y_merge4: GM[f32, (4, 64)], y_merge2: GM[f32, (2, 64)]):
    ub_x = Tensor(DT.float, [4, 32], Position.UB)
    ub_i = Tensor(DT.uint32, [4, 32], Position.UB)
    ub_s = Tensor(DT.float, [4, 64], Position.UB)
    ub_m4 = Tensor(DT.float, [4, 64], Position.UB)
    ub_m2 = Tensor(DT.float, [2, 64], Position.UB)
    with auto_sync():
        ub_x <<= x
        ub_i <<= idx
        sort32(ub_s, ub_x, ub_i, 4)  # four lists of 32 records, each sorted
        y_sort <<= ub_s
        mergesort4(ub_m4, ub_s, 32, 1)  # the four lists -> one list of 128 records
        y_merge4 <<= ub_m4
        mergesort_2seq(ub_m2, ub_s[0:1, :], ub_s[1:2, :], 32, 32)  # the first two lists -> 64 records
        y_merge2 <<= ub_m2
    return y_sort, y_merge4, y_merge2
