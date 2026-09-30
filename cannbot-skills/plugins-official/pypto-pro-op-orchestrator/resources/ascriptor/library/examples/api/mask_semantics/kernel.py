# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Execution masks preserve old predicate bits; selector masks choose both branches."""

import ascriptor.a5 as api


def make_masks(count):
    @api.vf()
    def masks(x: api.Tensor, y: api.Tensor, output: api.Tensor):
        a = api.Reg(api.i32)
        b = api.Reg(api.i32)
        value = api.Reg(api.i32)
        one = api.Reg(api.i32)
        zero = api.Reg(api.i32)
        one.fill(1)
        zero.fill(0)
        a <<= x[0]
        b <<= y[0]
        remaining = api.Var(count, dtype=api.u32)
        gate = api.MaskReg(api.i32)
        api.update_mask(gate, remaining)
        short = api.MaskReg(api.i32, init_mode=api.MaskType.LOWEST32)
        pred = api.MaskReg(api.i32, init_mode=api.MaskType.ALL)
        api.compare(pred, a, 0, api.CompareMode.GT, gate)
        value <<= pred.select(one, zero)
        output[0] <<= value
        api.select(value, a, b, gate)
        output[64] <<= value
        inverted = api.MaskReg(api.i32, init_mode=api.MaskType.ALL)
        api.mask_not(inverted, pred, gate)
        value <<= inverted.select(one, zero)
        output[128] <<= value
        both = api.MaskReg(api.i32, init_mode=api.MaskType.ALL)
        api.mask_and(both, pred, short, gate)
        value <<= both.select(one, zero)
        output[192] <<= value
        either = api.MaskReg(api.i32, init_mode=api.MaskType.NONE)
        api.mask_or(either, pred, short, gate)
        value <<= either.select(one, zero)
        output[256] <<= value
        different = api.MaskReg(api.i32, init_mode=api.MaskType.ALL)
        api.mask_xor(different, pred, short, gate)
        value <<= different.select(one, zero)
        output[320] <<= value
        copied = api.MaskReg(api.i32, init_mode=api.MaskType.NONE)
        api.mask_mov(copied, pred, gate)
        value <<= copied.select(one, zero)
        output[384] <<= value
        selected = api.MaskReg(api.i32, init_mode=api.MaskType.NONE)
        api.mask_sel(selected, pred, short, gate)
        value <<= selected.select(one, zero)
        output[448] <<= value
        packed = api.MaskReg(api.i32)
        unpacked = api.MaskReg(api.i32)
        api.mask_pack(packed, pred, low_part=True)
        value <<= packed.select(one, zero)
        output[512] <<= value
        api.mask_unpack(unpacked, packed, low_part=True)
        value <<= unpacked.select(one, zero)
        output[576] <<= value
        left = api.MaskReg(api.i32)
        right = api.MaskReg(api.i32)
        api.mask_interleave(left, right, pred, short)
        value <<= left.select(one, zero)
        output[640] <<= value
        value <<= right.select(one, zero)
        output[704] <<= value
        restored0 = api.MaskReg(api.i32)
        restored1 = api.MaskReg(api.i32)
        api.mask_deinterleave(restored0, restored1, left, right)
        value <<= restored0.select(one, zero)
        output[768] <<= value
        value <<= restored1.select(one, zero)
        output[832] <<= value
        value.fill(remaining)
        output[896] <<= value

    @api.kernel(mode="vec", block_dim=1)
    def mask_semantics(x: api.GM[api.i32, (1, 64)], y: api.GM[api.i32, (1, 64)],
        o: api.GM[api.i32, (15, 64)]):
        ux = api.Tensor(api.i32, [1, 64], api.Position.UB)
        uy = api.Tensor(api.i32, [1, 64], api.Position.UB)
        uo = api.Tensor(api.i32, [15, 64], api.Position.UB)
        with api.auto_sync():
            ux <<= x
            uy <<= y
            masks(ux, uy, uo)
            o <<= uo
        return o

    return mask_semantics
