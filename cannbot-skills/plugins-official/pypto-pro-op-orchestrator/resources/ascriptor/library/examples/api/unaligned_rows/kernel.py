# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-row Load/StoreUnAlign variants with explicit physical read slack."""

import ascriptor.a5 as api


def make_rows(rows, width, mode="stream"):
    if rows <= 0 or width <= 0 or mode not in ("stream", "once"):
        raise ValueError("positive row geometry and stream/once mode required")
    cells = rows * width
    capacity = ((cells + 64 + 7) // 8) * 8
    chunks = (width + 63) // 64

    @api.vf()
    def walk(src: api.Tensor, dst: api.Tensor):
        value = api.Reg(api.i32)
        load_state = api.unalign_reg_for_load()
        store_state = api.unalign_reg_for_store()
        for row in range(rows):
            if mode == "stream":
                source = api.ub_cursor(src[0:1, row * width :])
                target = api.ub_cursor(dst[0:1, row * width :])
                api.ub_to_reg_unalign_pre(load_state, source)
            for chunk in range(chunks):
                count = api.Min(64, width - chunk * 64)
                if mode == "stream":
                    api.ub_to_reg_unalign(value, load_state, source, count)
                else:
                    api.ub_to_reg_unalign_once(value, src[0:1, row * width + chunk * 64 :])
                api.adds(value, value, row)
                if mode == "stream":
                    api.reg_to_ub_unalign(target, value, store_state, count)
                else:
                    api.reg_to_ub_unalign_once(dst[0:1, row * width + chunk * 64 :], value, count)
            if mode == "stream":
                api.reg_to_ub_unalign_post(target, store_state)

    @api.kernel(mode="vec", block_dim=1)
    def unaligned_rows(x: api.GM[api.i32, (1, cells)], o: api.GM[api.i32, (1, cells)]):
        source = api.Tensor(api.i32, [1, capacity], api.Position.UB)
        target = api.Tensor(api.i32, [1, cells], api.Position.UB)
        with api.auto_sync():
            source[0:1, :cells] <<= x
            walk(source, target)
            o <<= target
        return o

    return unaligned_rows
