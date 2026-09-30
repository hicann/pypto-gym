# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One A2/A3 vector core gathers 32-byte source blocks, weights them and adds two taps."""

import importlib

from ascriptor.a2 import *
from functools import lru_cache

S0 = 512    # source halves, i.e. 1024 bytes = 32 addressable blocks
OFF = 32    # byte offsets, eight consumed per gather repeat
S1 = 16     # weights in the ABI; only the first four reach the output
DST = 256   # two 128-value output rows
EPB = 16    # halves per 32-byte block, which is also brcb's block width


def bilinear_interp(src0_gm: GM[f16, (1, 512)], offset_gm: GM[i32, (1, 32)], src1_gm: GM[f16, (1, 16)],
                    dst_gm: GM[f16, (1, 256)], inner_rep: i32):
    src0_ub = Tensor(DT.half, [1, S0], Position.UB)
    off_ub = Tensor(DT.int, [1, OFF], Position.UB)
    src1_ub = Tensor(DT.half, [1, S1], Position.UB)
    gathered = Tensor(DT.half, [1, S0], Position.UB)      # 4 reps x 128
    wbrcb = Tensor(DT.half, [1, S1 * EPB], Position.UB)   # 16 weights -> 16 blocks
    prod = Tensor(DT.half, [1, S0], Position.UB)
    dst_ub = Tensor(DT.half, [1, DST], Position.UB)

    with auto_sync():
        src0_ub <<= src0_gm[:, :]
        off_ub <<= offset_gm[:, :]
        src1_ub <<= src1_gm[:, :]
        off_u32 = reinterpret(off_ub, DT.uint32)

        # 1. gather the sampled blocks: rep r gathers src0 blocks offset[r*8 .. r*8+7].
        #    c220 vgatherb needs uint16 operands for 16-bit data, so reinterpret half <->
        #    uint16 around the block gather (same pattern as element `gather`, which callers
        #    reinterpret to an integer dtype). gathered_u16 aliases `gathered`, so the half
        #    view below sees the gathered bytes. Without this the generated `Gatherb` does not
        #    compile on real HW ("1st parameter maybe need a type '__ubuf__ unsigned short *'").
        gathered_u16 = reinterpret(gathered, DT.uint16)
        src0_u16 = reinterpret(src0_ub, DT.uint16)
        gather_block(gathered_u16, src0_u16, off_u32, repeat=inner_rep)

        # 2. broadcast each weight src1[k] into a full 16-half block
        brcb(wbrcb, src1_ub)

        # 3. prod[rep r, block b] = src1[r] * gathered[r*8+b]
        #    repeatMode=false: one weight per rep reused across its 8 blocks
        #    -> weight operand blk_stride=0 (reuse within rep), rep_stride=1 (advance 1 block/rep)
        mul(prod, wbrcb, gathered, repeat=inner_rep, src1_blk_stride=0, src1_rep_stride=1)

        # 4. accumulate the 2 horizontal taps for each vertical row
        add(dst_ub[:, 0:128], prod[:, 0:128], prod[:, 128:256])
        add(dst_ub[:, 128:256], prod[:, 256:384], prod[:, 384:512])

        # 5. store
        dst_gm[:, :] <<= dst_ub

    return dst_gm

@lru_cache(maxsize=2)
def kernel_for(device):
    """Bind the one body to A2 or A3. `mode="vec"` with `block_dim=1` is the correction this
    unit carries: the source ran the same body under implicit MIX, where both vector
    participants owned the whole output and wrote the same bytes twice."""
    if device not in ("a2", "a3"):
        raise ValueError("The weighted block-gather recipe supports A2/A3")
    api = importlib.import_module("ascriptor." + device)
    return api.kernel(mode="vec", block_dim=1)(bilinear_interp)
