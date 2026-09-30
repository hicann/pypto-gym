# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scalar mask state and register prefix counts have separate meanings."""

import ascriptor.a5 as api


def make_spr(count, pattern):
    @api.vf()
    def observe(output: api.Tensor):
        mask = api.MaskReg(api.i32)
        api.move_mask_spr(mask)
        seven = api.Reg(api.i32)
        zero = api.Reg(api.i32)
        result = api.Reg(api.i32)
        seven.fill(7)
        zero.fill(0)
        api.select(result, seven, zero, mask)
        output[0] <<= result

    @api.kernel(mode='vec', block_dim=1)
    def spr(dummy: api.GM[api.i32, (1, 8)], output: api.GM[api.i32, (3, 64)]):
        source = api.Tensor(api.i32, [1, 8], api.Position.UB)
        observed = api.Tensor(api.i32, [3, 64], api.Position.UB)
        with api.auto_sync():
            source <<= dummy
            api.set_mask_by_count(count)
            observe(observed[0:1, :])
            api.reset_mask()
            output[0:1, :] <<= observed[0:1, :]
            api.set_mask(0, pattern)
            observe(observed[1:2, :])
            api.reset_mask()
            output[1:2, :] <<= observed[1:2, :]
            observe(observed[2:3, :])
            output[2:3, :] <<= observed[2:3, :]
        return output

    return spr


@api.vf()
def prefix_vf(values: api.Tensor, selector: api.Tensor, output: api.Tensor):
    data = api.Reg(api.i32)
    select = api.Reg(api.i32)
    mask = api.MaskReg(api.i32)
    select <<= selector[0]
    api.compare(mask, select, 0, api.CompareMode.GT)
    data <<= values[0]
    api.unsqueeze(data, mask)
    output[0] <<= data


@api.kernel(mode='vec', block_dim=1)
def prefix(values: api.GM[api.i32, (2, 64)], selector: api.GM[api.i32, (1, 64)],
    output: api.GM[api.i32, (2, 64)]):
    src = api.Tensor(api.i32, [1, 64], api.Position.UB)
    sel = api.Tensor(api.i32, [1, 64], api.Position.UB)
    dst = api.Tensor(api.i32, [1, 64], api.Position.UB)
    with api.auto_sync():
        sel <<= selector
        for row in range(2):
            src <<= values[row:row + 1, :]
            prefix_vf(src, sel, dst)
            output[row:row + 1, :] <<= dst
    return output
