# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""FP32 SIMT elementary functions and integral rounding inside SIMT functions (RFC-0015).

Pro prints one A5 formula per call. exp, exp2, log1p, sin, cos, tanh and rsqrt keep their target operation,
whose CCE shim prints the same formula. Pro's log and log2 formulas differ from the forward shim, so they
expand into target operations in Pro's order: a positive subnormal operand is scaled by exp(23) before the
logarithm, and log2 divides by log(2). Integral rounding follows ROUNDING: 'positive_zero' makes a zero result
+0, as the A5 builtins return it for runtime and literal operands alike (measured), and 'refuse' rejects the
call; LITERAL_ROUNDING = False rejects literal operands.
"""
from ...ir import Ident, Value
from ...ir.types import ScalarType, dtype
from .scalar_ops import scalar

F32, B1 = ScalarType(dtype('f32')), ScalarType(dtype('b1'))
DIRECT = ('exp', 'exp2', 'log1p', 'sin', 'cos', 'tanh', 'rsqrt')
INTEGRAL = ('rint', 'round', 'floor', 'ceil', 'trunc')
SUBNORMAL_LIMIT, SCALE = 2.0 ** -126, 23.0  # Pro's 1.17549435e-38f and its exp(23.0f) scale
ROUNDING = 'positive_zero'
LITERAL_ROUNDING = True
LOG = ('simt.log', 'simt.exp', 'scalar.cmp', 'scalar.and', 'scalar.mul', 'scalar.sub', 'scalar.select', 'scalar.const')
TARGETS = {**{f'simt.{n}': (f'simt.{n}', 'scalar.const') for n in DIRECT}, 'simt.log': LOG, 'simt.log2': (*LOG, 'scalar.div'),
           **{f'simt.{n}': (f'simt.{n}', 'scalar.const', 'scalar.cmp', 'scalar.select') for n in INTEGRAL}}
OPERAND = 'SIMT math admits one FP32 operand and result; FP16/BF16 variants have no target'


def convert(ctx, node, args):
    o, name = ctx.o, node['fields']['name']
    short = name.removeprefix('simt.')
    ctx.attrs(node)
    typ = o.node(node['fields']['type'])
    o.need(len(args) == 1 and typ['kind'] == 'ScalarType' and o.dt(typ['fields']['dtype'], node).name == 'f32'
           and (type(args[0]) is float or scalar(args[0]) == F32), node, OPERAND)
    if short in INTEGRAL:
        o.need(ROUNDING == 'positive_zero', node,
               'A5 rounding builtins and the target model differ in zero signs; this import policy refuses SIMT rounding')
        o.need(LITERAL_ROUNDING or type(args[0]) is not float, node,
               'this import policy refuses a literal SIMT rounding operand; pass the operand at run time')
    x = ctx.snapshot(args[0], node, F32)
    o.need(isinstance(x, Value) and x.type == F32, node, OPERAND)
    if short in {'log', 'log2'}:
        return logarithm(ctx, node, x, short == 'log2')
    result = ctx.emit(name, node, (x,), typ=F32)
    if short in INTEGRAL:  # The target keeps the operand's zero sign (RFC-0001 §6.3); A5 returns +0.
        zero = ctx.emit('scalar.cmp', node, (result, 0.0), typ=B1, attrs={'pred': Ident('eq')})
        result = ctx.emit('scalar.select', node, (zero, 0.0, result), typ=F32)
    return result


def logarithm(ctx, node, x, base2):
    """((x > 0.0f && x < 1.17549435e-38f) ? (__logf(__expf(23.0f) * x) - 23.0f) : __logf(x)), then / __logf(2.0f)."""
    def emit(opcode, *operands, typ=F32, pred=None):
        return ctx.emit(opcode, node, operands, typ=typ, attrs={'pred': Ident(pred)} if pred else None)

    positive = emit('scalar.cmp', x, 0.0, typ=B1, pred='gt')
    subnormal = emit('scalar.and', positive, emit('scalar.cmp', x, SUBNORMAL_LIMIT, typ=B1, pred='lt'), typ=B1)
    scaled = emit('scalar.sub', emit('simt.log', emit('scalar.mul', emit('simt.exp', SCALE), x)), SCALE)
    result = emit('scalar.select', subnormal, scaled, emit('simt.log', x))
    return emit('scalar.div', result, emit('simt.log', 2.0)) if base2 else result
