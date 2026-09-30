# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent arbitrary-precision Python fmix32, and the deterministic bit patterns it is fed."""

import math
import torch

MASK = (1 << 32) - 1


def signed(bits):
    bits &= MASK
    return bits - (1 << 32) if bits >= 1 << 31 else bits


def fmix32(value):
    h = value & MASK
    h ^= h >> 16
    h = (h * 0x85ebca6b) & MASK
    h ^= h >> 13
    h = (h * 0xc2b2ae35) & MASK
    h ^= h >> 16
    return signed(h)


def geometry(shape, tile, cores):
    """Flatten the case's shape and refuse a geometry this kernel cannot own: more than
    1048576 values, a tile outside 1..8192, or a split whose per-core run of tiles does not
    start on a 32-byte GM boundary (per_core * tile * 4 bytes)."""
    if not isinstance(shape,(list,tuple)) or not shape or any(type(n) is not int or n<=0 for n in shape):
        raise ValueError('Require positive integer dimensions')
    n=math.prod(shape)
    if n>1048576 or type(tile) is not int or not 1<=tile<=8192 or type(cores) is not int or not 1<=cores<=40:
        raise ValueError('Invalid bounded tensor, tile or runtime vector count')
    tiles=(n+tile-1)//tile;per_core=(tiles+cores-1)//cores
    if tiles>per_core and per_core*tile*4%32:
        raise ValueError('Active owners must start on a physical32-byte boundary')
    return n


def make_inputs(case):
    p=case['parameters'];cores=case.get('block_dim',1);shape=tuple(p['shape']);n=geometry(shape,p['tile_len'],cores)
    generator=torch.Generator().manual_seed(case['seed'])
    mode=p['mode']
    if mode=='random':
        x=torch.randint(-(1<<31),(1<<31)-1,shape,dtype=torch.int64,generator=generator).int()
    elif mode=='counter':
        seed=p['counter_seed']
        if type(seed) is not int or not 0<=seed<=MASK:raise ValueError('Counter seed is a UINT32 payload')
        x=torch.tensor([signed(seed^i) for i in range(n)],dtype=torch.int32).reshape(shape)
    elif mode=='boundaries':
        values=[0,1,2,0xffff,0x10000,0x7fffffff,0x80000000,0x80000001,0xfffffffe,0xffffffff,0xaaaaaaaa,0x55555555]
        x=torch.tensor([signed(values[i%len(values)]) for i in range(n)],dtype=torch.int32).reshape(shape)
    else:raise ValueError('Unknown generated hash input')
    return {'x':x,'n':n,'tile_len':p['tile_len'],'block_dim':cores,'shape':shape}


def reference(inputs):
    """One Python integer at a time, in arbitrary precision. Nothing here is a tensor
    operation, so no 32-bit overflow, no float rounding and no dtype promotion can be shared
    with the kernel: every output bit is derived a second, independent way."""
    return {'output': torch.tensor([fmix32(v) for v in inputs['x'].flatten().tolist()],
                                   dtype=torch.int32).reshape(inputs['shape'])}
