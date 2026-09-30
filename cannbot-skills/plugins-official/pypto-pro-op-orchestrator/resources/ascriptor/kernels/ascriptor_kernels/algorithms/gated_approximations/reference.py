# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch reference: the per-mode source operation order in FP32, rounded
once to each variant's storage dtype, plus the mode table the host reads."""

import math
from collections import namedtuple

import torch

# ----------------------------------------------------------------------------------------------------
# settings.py
# One row per source mode: storage dtype, formula, UB tile capacity and the original elementwise budget.
# ----------------------------------------------------------------------------------------------------

Settings = namedtuple('Settings', 'dtype mode max_tile')
VARIANTS = {
    'gelu_tanh_f32': Settings('float32', 'tanh', 6656),
    'gelu_tanh_f16': Settings('float16', 'tanh', 6656),
    'gelu_tanh_bf16': Settings('bfloat16', 'tanh', 6656),
    'gelu_erf_f32': Settings('float32', 'erf', 5120),
    'gelu_erf_bf16': Settings('bfloat16', 'erf', 5120),
    'swiglu_f32': Settings('float32', 'swiglu', 8192),
    'swiglu_f16': Settings('float16', 'swiglu', 6144),
    'swiglu_bf16': Settings('bfloat16', 'swiglu', 6144),
}


def settings(variant):
    if variant not in VARIANTS:
        raise ValueError('No such complete activation mode')
    return VARIANTS[variant]

# ----------------------------------------------------------------------------------------------------
# reference.py
# Generated CPU operands and independent, operation-associated activation formulas.
# ----------------------------------------------------------------------------------------------------

CHUNK = 262144


def geometry(parameters, block_dim):
    s = settings(parameters['variant'])
    shape = parameters['shape']
    if not isinstance(shape, (tuple, list)) or not shape or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError('Shape must contain positive integer dimensions')
    if s.mode == 'swiglu' and shape[-1] < 2:
        raise ValueError('The last dimension must contain both halves')
    output_shape = tuple(shape[:-1]) + (shape[-1] // 2,) if s.mode == 'swiglu' else tuple(shape)
    n = math.prod(output_shape)
    if n > 16777216:
        raise ValueError('Output extent exceeds the retained source domain')
    tile = parameters.get('tile_len', min(s.max_tile, n))
    if type(tile) is not int or not 1 <= tile <= s.max_tile:
        raise ValueError('Tile length must fit this mode\'s declared UB allocations')
    if type(block_dim) is not int or not 1 <= block_dim <= 40:
        raise ValueError('Require1to40 runtime vector owners')
    tiles = (n + tile - 1) // tile
    per_core = (tiles + block_dim - 1) // block_dim
    itemsize = 4 if s.dtype == 'float32' else 2
    if tiles > per_core and per_core * tile * itemsize % 32:
        raise ValueError('Each active owner must start on a physical32-byte boundary')
    beta = parameters.get('beta', 1.0)
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or not 0 <= beta <= 2:
        raise ValueError('Require a finite beta in[0,2]')
    return tuple(shape), output_shape, n, tile, float(beta)


def make_inputs(case):
    p = case['parameters']
    s = settings(p['variant'])
    shape, output_shape, n, tile, beta = geometry(p, case.get('block_dim', 1))
    generator = torch.Generator().manual_seed(case['seed'])
    bound = 1 if s.mode == 'swiglu' else 10
    x = torch.empty(shape, dtype=getattr(torch, s.dtype)).uniform_(-bound, bound, generator=generator)
    pattern = p.get('mode', 'random')
    if pattern == 'positive':
        x.fill_(0.5 if s.mode == 'swiglu' else 1.0)
    elif pattern == 'mode_band' and s.mode != 'swiglu':
        x.fill_(-2.5)
        x.flatten()[0] = 1.0
    elif pattern == 'small_output' and s.mode == 'swiglu':
        x.fill_(0.0625)
    elif pattern != 'random':
        raise ValueError('Unknown generated activation pattern')
    return {'variant':p['variant'], 'x':x, 'shape':shape, 'output_shape':output_shape, 'n':n, 'tile_len':tile,
            'beta':beta, 'block_dim':case.get('block_dim', 1)}


def chunks(n):
    for offset in range(0, n, CHUNK):
        yield slice(offset, min(n, offset + CHUNK))


def split_inputs(inputs):
    x = inputs['x']
    half = x.shape[-1] // 2
    return x[..., :half].contiguous().flatten(), x[..., half:2 * half].contiguous().flatten()


def source_formula(x, other=None, beta=1.0, *, mode):
    """Each ordinary Torch FP32 operation rounds separately; no compiler or model imports."""
    x = x.float()
    if mode == 'swiglu':
        neg_beta = torch.tensor(-beta, dtype=torch.float32)
        den = torch.exp(x * neg_beta) + 1.0
        return (x * other.float()) / den
    if mode == 'tanh':
        squared = x * x
        cubic = squared * x
        corrected = x + cubic * 0.044715
        exponent = corrected * (-2 * math.sqrt(2 / math.pi))
        return x / (torch.exp(exponent) + 1.0)
    if mode != 'erf':
        raise ValueError('No such complete activation mode')
    z = x * (1 / math.sqrt(2))
    exponential = torch.exp(-(z * z))
    magnitude = z.abs()
    halfabs = magnitude * (1 / math.sqrt(2))
    t = 1.0 / (magnitude * 0.3275911 + 1.0)
    polynomial = t * 1.061405429 + (-1.453152027)
    polynomial = polynomial * t + 1.421413741
    polynomial = polynomial * t + (-0.284496736)
    polynomial = polynomial * t + 0.254829592
    polynomial = polynomial * t
    tail = (halfabs * polynomial) * exponential
    return (x * 0.5 + halfabs) - tail


def reference(inputs):
    s = settings(inputs['variant'])
    result = torch.empty(inputs['n'], dtype=getattr(torch, s.dtype))
    x0, x1 = split_inputs(inputs) if s.mode == 'swiglu' else (inputs['x'].flatten(), None)
    for sl in chunks(inputs['n']):
        result[sl] = source_formula(x0[sl], None if x1 is None else x1[sl], inputs['beta'], mode=s.mode).to(getattr(torch, s.dtype))
    return {'output':result.reshape(inputs['output_shape'])}
