# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One vector launch of the public four-level MSD radix top-k composite."""

from ascriptor.a5 import DT, GM, Position, Tensor, auto_sync, f32, i32, kernel, radix_topk

MAXN = 4096
MAXK = 512


@kernel(mode="vec",block_dim=1)
def radix_kernel(keys: GM[f32,(1,4096)], out_val: GM[f32,(1,512)], out_idx: GM[i32,(1,512)],
                 count: i32, k: i32):
    keys_ub = Tensor(DT.float, [1, MAXN], Position.UB, name="rs_keys")
    cand = Tensor(DT.float, [1, MAXK], Position.UB, name="rs_cand")
    cidx = Tensor(DT.int, [1, MAXK], Position.UB, name="rs_cidx")
    with auto_sync():
        keys_ub[:, :] <<= keys[0:1, :]
        radix_topk(cand, cidx, keys_ub, count, k)
        out_val[0:1, :] <<= cand
        out_idx[0:1, :] <<= cidx
    return out_val, out_idx
