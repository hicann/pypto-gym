# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""In-place FP32 fused accumulators; A5 rounds each exact expression once."""
from ...ir import Value
from ...ir.types import RegType, dtype
from .vector_predicates import mask, zeroing

TARGETS = {'vf.axpy': ('vf.axpy',), 'vf.mul_add_dst': ('vf.muladddst',), 'vf.mul_dst_add': ('vf.muldstadd',)}
F32 = RegType(dtype('f32'))


def convert(ctx, node, args):
    from .vector import fp32_scalar

    o, name = ctx.o, node['fields']['name']
    zeroing(o, node, ctx.attrs(node, {'mode'}))
    o.need(len(args) == 4 and all(isinstance(v, Value) and v.type == F32 for v in args[:2]) and mask(args[3]), node,
           'Fused accumulators require FP32 registers and a b32 predicate')
    dst, source, other = args[:3]
    if name == 'vf.axpy':
        o.need(fp32_scalar(other), node, 'Axpy requires an FP32 scalar or an exactly representable literal')
    else:
        o.need(isinstance(other, Value) and other.type == F32, node, 'Fused multiply-add requires FP32 registers')
    # Native registers start uninitialized; the model's zero-filled declaration must not hide that read.
    o.need(dst in ctx.written, node, 'In-place accumulator destination must be written earlier in this VF body')
    ctx.emit(TARGETS[name][0], node, (dst, source, other), attrs={'mask': args[3]})
    return None
