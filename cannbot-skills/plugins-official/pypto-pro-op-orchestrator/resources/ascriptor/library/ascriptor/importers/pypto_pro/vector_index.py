# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Increasing single-register FP32/INT32 index ramps."""
from ...ir import Ident
from .vector_predicates import reg, scalar

TARGETS = {'vf.arange': ('vf.arange',)}


def convert(ctx, node, args):
    o = ctx.o
    attrs = ctx.attrs(node, {'dtype', 'index_order'})
    o.need(len(args) == 2 and reg(args[0]), node, 'Arange requires an FP32/INT32 register and scalar start')
    dst, start = args
    if 'dtype' in attrs:
        o.need(o.dt(attrs['dtype'], node) == dst.type.dtype, node, 'Arange dtype contradicts destination')
    order = attrs.get('index_order', 0)
    o.need(type(order) is int and order == 0, node, 'Only increasing arange is admitted')
    o.need(scalar(start, dst.type.dtype), node, 'Arange start must match its destination scalar type')
    if dst.type.dtype.name == 'i32' and type(start) is int:
        o.need(-2**31 <= start < 2**31, node, 'INT32 arange start is outside its signed domain')
    ctx.emit(TARGETS['vf.arange'][0], node, (dst,), attrs={'v': start, 'mode': Ident('increase')})
    return None
