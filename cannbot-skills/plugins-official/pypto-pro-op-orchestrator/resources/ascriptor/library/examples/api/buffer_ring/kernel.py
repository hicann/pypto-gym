# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two-slot UB and GM rings with an explicit GM producer/consumer hand-off.

Five rows travel through two UB slots and two GM slots, so both rings wrap twice and the fifth beat
takes slot 0 for the third time. The MTE3-to-MTE2 event orders the GM write before the read that
follows it; `auto_sync()` covers the on-chip buffers and their reuse. One vector core owns every row.
"""

from ascriptor.a5 import GM, DBuff, GMBuff, Pipe, Position, SEvent, auto_sync, f32, kernel


@kernel(mode="vec", block_dim=1)
def buffer_ring(x: GM[f32, (5, 64)], o: GM[f32, (5, 64)]):
    incoming = DBuff(f32, [1, 64], Position.UB)
    outgoing = DBuff(f32, [1, 64], Position.UB)
    workspace = GMBuff(f32, [1, 64], slots=2, per_core=False)
    gm_ready = SEvent(Pipe.MTE3, Pipe.MTE2)
    with auto_sync():
        for beat in range(5):
            source_slot = incoming[beat]
            target_slot = outgoing[beat]
            source_slot <<= x[beat : beat + 1, :]
            workspace[beat] <<= source_slot
            gm_ready.set()
            gm_ready.wait()
            target_slot <<= workspace[beat]
            o[beat : beat + 1, :] <<= target_slot
    return o
