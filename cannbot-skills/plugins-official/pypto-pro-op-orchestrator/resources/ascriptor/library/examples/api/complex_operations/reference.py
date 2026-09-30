# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent real-component arithmetic; no DSL or complex execution helper."""

import torch

OPS = ('add', 'sub', 'mul', 'div', 'adds', 'muls', 'abs', 'dup')


def make_inputs(case):
    generator = torch.Generator().manual_seed(case['seed'])
    rows, dataset = case['parameters']['rows'], case['parameters']['dataset']
    inputs = {}
    for kind, columns, dtype in [('64', 32, torch.complex64), ('32', 64, torch.complex32)]:
        shape = (rows, columns)
        if dataset == 'source':
            a = torch.complex(torch.randn(shape, generator=generator), torch.randn(shape, generator=generator))
            b = torch.complex(torch.randn(shape, generator=generator) + 2.5, torch.randn(shape, generator=generator) - 2)
        elif dataset == 'literal':
            av = [0j, 3+4j, -3+4j, 3-4j, -3-4j, 1+0j, 0+1j, -.5+.25j]
            bv = [1+0j, 2+1j, 2-1j, -2+1j, -2-1j, 0+1j, 1+0j, .5+.5j]
            a = torch.tensor(av).repeat(rows * columns // 8).reshape(shape)
            b = torch.tensor(bv).repeat(rows * columns // 8).reshape(shape)
        else:
            a = torch.complex(torch.randint(-16, 17, shape, generator=generator).float() / 8,
                torch.randint(-16, 17, shape, generator=generator).float() / 8)
            b = torch.complex(torch.randint(8, 25, shape, generator=generator).float() / 8,
                torch.randint(-16, 17, shape, generator=generator).float() / 8)
            if dataset == 'zero':
                a.zero_()
        inputs['a' + kind], inputs['b' + kind] = a.to(dtype), b.to(dtype)
    validate(inputs)
    return inputs


def validate(inputs):
    rows = None
    for kind, columns, dtype in [('64', 32, torch.complex64), ('32', 64, torch.complex32)]:
        for name in ['a', 'b']:
            x = inputs.get(name + kind)
            if not isinstance(x, torch.Tensor) or x.dtype != dtype or x.ndim != 2 or x.shape[1] != columns or not x.is_contiguous():
                raise ValueError('complex dtype and complete contiguous register row must agree')
            rows = x.shape[0] if rows is None else rows
            if x.shape[0] != rows or not 1 <= rows <= 64:
                raise ValueError('every input must share the declared positive row count')
            magnitude = x.to(torch.complex64).abs()
            if not bool(torch.isfinite(magnitude).all()) or bool((magnitude > (8 if name == 'a' else 12)).any()):
                raise ValueError('finite inputs must remain in the bounded reference domain')
            if name == 'b' and bool((magnitude < .125).any()):
                raise ValueError('complex division denominator is outside the conditioned domain')


def reference(inputs):
    validate(inputs)
    outputs = {}
    for kind in ['64', '32']:
        a, b = inputs['a' + kind].to(torch.complex128), inputs['b' + kind].to(torch.complex128)
        ar, ai, br, bi = a.real, a.imag, b.real, b.imag
        denominator = br * br + bi * bi
        components = {'add': (ar + br, ai + bi), 'sub': (ar - br, ai - bi),
            'mul': (ar * br - ai * bi, ar * bi + ai * br),
            'div': ((ar * br + ai * bi) / denominator, (ai * br - ar * bi) / denominator),
            'adds': (ar + 1.5, ai - .5), 'muls': (ar * 2 - ai, ar + ai * 2),
            'dup': (torch.full_like(ar, 3), torch.full_like(ai, -2))}
        for op, values in components.items():
            outputs[f'c{kind}_{op}'] = torch.stack(values, dim=-1).float()
        outputs[f'c{kind}_abs'] = torch.sqrt(ar * ar + ai * ai).float()
    return outputs
