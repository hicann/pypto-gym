# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fused exp-difference and FP32-to-FP16 multiply-cast registers."""
from ...ir import Ident, Value
from ...ir.types import MaskType, RegType, ScalarType, dtype

TARGETS = {'vf.exp_sub': ('vf.expsub',), 'vf.muls_cast': ('vf.mulscast',)}
LAYOUTS = {0: 'zero', 1: 'one'}


def width(value):
    return value.type.dtype.name if isinstance(value, Value) and isinstance(value.type, RegType) and value.type.n == 1 else None


def convert(ctx, node, args):
    o, name = ctx.o, node['fields']['name']
    layout = ctx.attrs(node, {'layout'}).get('layout', 0)
    o.need(type(layout) is int and layout in LAYOUTS, node, 'Fused cast layout must be ZERO or ONE')
    o.need(len(args) == 4 and isinstance(args[3], Value) and isinstance(args[3].type, MaskType), node,
           f'{name} needs a destination, two sources and a predicate')
    dst, src, other, pred = args
    if name == 'vf.exp_sub':
        o.need(width(dst) == 'f32' and width(src) in {'f32', 'f16'} and isinstance(other, Value) and other.type == src.type,
               node, 'exp_sub needs matching FP32/FP16 sources and an FP32 destination')
        # FP32 layout ONE has no documented effect but prints PART_ODD; FP16 predicates are sampled per source lane.
        o.need(pred.type == MaskType(32) and layout == 0 if width(src) == 'f32' else pred.type == MaskType(16), node,
               'exp_sub admits FP32 with a b32 predicate and layout ZERO, or FP16 with a b16 predicate')
    else:
        o.need(width(src) == 'f32' and width(dst) == 'f16' and pred.type == MaskType(32), node,
               'muls_cast maps an FP32 register with a b32 predicate to FP16')
        o.need(type(other) is float or isinstance(other, Value) and other.type == ScalarType(dtype('f32')), node,
               'muls_cast requires an FP32 scalar')
    ctx.emit(TARGETS[name][0], node, (dst, src, other), attrs={'mask': pred, 'layout': Ident(LAYOUTS[layout])})
    return None
