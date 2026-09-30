# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch/IEEE reference: finite E4M3 nearest-even encoding, biased-exponent
E8M0 scale selection, and the FP64 dot product of the decoded operands."""

import bisect
import math
import struct
import torch

# ----------------------------------------------------------------------------------------------------
# formats.py
# Independent finite E4M3 RNE and IEEE FP32 exponent selection; no execution codec.
# ----------------------------------------------------------------------------------------------------

def fp32(value):
    return struct.unpack('<f',struct.pack('<f',value))[0]


def bits32(value):
    return struct.unpack('<I',struct.pack('<f',value))[0]


def positive_e4m3(code):
    exponent,fraction=divmod(code,8)
    if exponent==0:
        return fraction*2.0**-9
    if code==127:
        raise ValueError('E4M3 NaN is outside the finite online domain')
    return (8+fraction)*2.0**(exponent-10)


VALUES=tuple(positive_e4m3(code) for code in range(127))


def encode_e4m3(value):
    value=fp32(value)
    if not math.isfinite(value) or abs(value)>2:
        raise ValueError('The source normalization must produce finite magnitudes at most2')
    sign=(bits32(value)>>24)&128
    magnitude=abs(value)
    upper=bisect.bisect_left(VALUES,magnitude)
    if upper==0:
        code=0
    elif VALUES[upper]==magnitude:
        code=upper
    else:
        lower=upper-1
        left,right=magnitude-VALUES[lower],VALUES[upper]-magnitude
        code=lower if left<right else upper if right<left else lower if lower%2==0 else upper
    return code|sign


def decode_e4m3(code):
    value=positive_e4m3(code&127)
    return -value if code&128 else value


def quantize_row(values):
    if len(values)!=64:
        raise ValueError('The retained online recipe contains two32-value groups')
    payload=[];scales=[];dequantized=[]
    for group in (values[:32],values[32:]):
        exponent=max((bits32(value)>>23)&255 for value in group)
        if exponent==255:
            raise ValueError('NaN/Inf has no finite online scale contract')
        # Raw FP32 exponent-bits0 is zero; the VF explicitly clamps normalization
        # to2^-127, which is also the MX consumer meaning of E8M0 byte0.
        scale=2.0**(exponent-127)
        codes=[encode_e4m3(fp32(value/scale)) for value in group]
        payload.extend(codes);scales.append(exponent)
        dequantized.extend(fp32(decode_e4m3(code)*scale) for code in codes)
    return payload,scales,dequantized

# ----------------------------------------------------------------------------------------------------
# reference.py
# Exact source generators, independent payload/scales and quantized matrix products.
# ----------------------------------------------------------------------------------------------------

def geometry(parameters):
    variant=parameters['variant']
    rows=16 if variant=='nd' else 32 if variant=='transposed' else None
    if rows is None:
        raise ValueError('Select one of the two complete physical online layouts')
    shape=(rows,64) if variant=='nd' else (64,rows)
    required={'M':rows,'N':rows,'K':64,'input_rows':shape[0],'input_cols':shape[1]}
    if any(type(parameters.get(key)) is not int or parameters[key]!=value for key,value in required.items()):
        raise ValueError('Geometry must match this corrected fixed-capacity source signature')
    return rows,shape


def make_inputs(case):
    p=dict(case['parameters']);rows,shape=geometry(p);cores=case.get('block_dim',1)
    if type(cores) is not int or cores not in (1,2,3):
        raise ValueError('Only the declared active/idle group valuations are supported')
    generator=torch.Generator().manual_seed(case['seed'])
    # Source RNG uses physical input layout directly, x then y, both FP32 times16.
    x=torch.randn(shape,dtype=torch.float32,generator=generator)*16.0
    y=torch.randn(shape,dtype=torch.float32,generator=generator)*16.0
    pattern=p['pattern']
    if pattern=='small_output':x*=2**-16;y*=2**-16
    elif pattern!='random':
        a=torch.empty((rows,64));b=torch.empty_like(a)
        if pattern=='midpoints':
            values=[1+1/16,1+3/16,-1-1/16,-1-3/16,2**-10,-(2**-10),1.75,-0.0]*8
            a[:]=torch.tensor(values);b[:]=torch.tensor(values[::-1])
        elif pattern=='midpoint_sweep':
            samples=[]
            for left,right in zip(VALUES[:64],VALUES[1:65]):
                midpoint=(left+right)/2;word=bits32(midpoint)
                for offset in (-1,0,1):
                    value=struct.unpack('<f',struct.pack('<I',word+offset))[0]
                    samples.extend((value,-value))
            for row in range(rows):
                for group in range(2):
                    start=(row*2+group)*31
                    a[row,group*32:(group+1)*32]=torch.tensor([samples[(start+i)%len(samples)] for i in range(31)]+[1.75])
                    b[row,group*32:(group+1)*32]=torch.tensor([samples[(start+i+97)%len(samples)] for i in range(31)]+[1.75])
        elif pattern=='group_exponents':
            line=torch.linspace(-1.75,1.75,32)
            for row in range(rows):
                for group in range(2):
                    a[row,group*32:(group+1)*32]=line*2.0**((row%5)*6-12+group*3)
                    b[row,group*32:(group+1)*32]=line.flip(0)*2.0**((row%4)*5-10-group*2)
        elif pattern=='last_group_scale':
            a.fill_(1);b.fill_(1);a[-1,32:]=2**20;b[-1,32:]=2**-12
        elif pattern=='zero':
            a.zero_();a[:,1::2]=-0.0;b.fill_(1)
        elif pattern=='scale_byte_zero':
            a.fill_(2**-127);b.fill_(2**40)
        elif pattern=='subnormal_payload':
            a[:]=torch.tensor([2**-149,-(2**-149),2**-127,-(2**-127)]*16);b.fill_(2**40)
        else:
            raise ValueError('Unknown generated online-MX pattern')
        x=a if p['variant']=='nd' else a.T.contiguous()
        y=b if p['variant']=='nd' else b.T.contiguous()
    return {'x':x,'y':y,'parameters':p,'block_dim':cores}


def quantize(inputs,name):
    data=inputs[name] if inputs['parameters']['variant']=='nd' else inputs[name].T.contiguous()
    tuples=[quantize_row(row) for row in data.tolist()]
    codes=torch.tensor([x[0] for x in tuples],dtype=torch.uint8)
    scales=torch.tensor([x[1] for x in tuples],dtype=torch.uint8)
    reconstructed=torch.tensor([x[2] for x in tuples],dtype=torch.float32)
    if inputs['parameters']['variant']=='transposed':codes=codes.T.contiguous()
    return codes,scales,reconstructed


def reference_stages(inputs):
    x,sx,_=quantize(inputs,'x');y,sy,_=quantize(inputs,'y')
    return {'payload_x':x,'scales_x':sx,'payload_y':y,'scales_y':sy}


def reference(inputs):
    _,_,x=quantize(inputs,'x');_,_,y=quantize(inputs,'y')
    return {'output':(x.double()@y.double().T).float()}
