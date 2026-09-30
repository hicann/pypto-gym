# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch quaternion/rotation formula and the physical [4, N] -> [9, N] host ABI."""

import torch


# ----------------------------------------------------------------------------------------------------
# settings.py
# ----------------------------------------------------------------------------------------------------

KIND='quaternion'
LAYOUT='channels'
SOURCE_SIZES=[64, 17, 90]
DEVICES=('a2', 'a3')

# ----------------------------------------------------------------------------------------------------
# reference.py
# Independent quaternion/matrix references and complete physical host ABIs.
# ----------------------------------------------------------------------------------------------------

def geometry(n, cores, variant):
    if type(n) is not int or not 1<=n<=4096 or type(cores) is not int or not 1<=cores<=(20 if LAYOUT=='channels' else 32):
        raise ValueError('Require bounded positive N and a core count within this facade profile')
    if variant=='single_vector_unaligned':
        if LAYOUT!='channels' or cores!=1 or n%8==0:
            raise ValueError('Named unaligned variant requires one vector and an unaligned channel pitch')
    elif variant!='source_mix':
        raise ValueError('Unknown physical-owner variant')
    elif LAYOUT=='channels' and n>64 and n%8:
        raise ValueError('Original MIX cannot share an unaligned physical channel pitch across writers')


def random_values(n, generator):
    if KIND=='quaternion':return {'quaternion':torch.randn((n,4),generator=generator)}
    return {'cov00':1+torch.rand(n,generator=generator),'cov11':1+torch.rand(n,generator=generator),
            'cov01':.1*torch.randn(n,generator=generator)}


def make_inputs(case):
    p=case['parameters'];n=p['N'];cores=case.get('block_dim',1);variant=p['variant'];geometry(n,cores,variant)
    generator=torch.Generator().manual_seed(case['seed']);mode=p['mode']
    if mode=='source':
        position=p['source_position']
        if type(position) is not int or not 0<=position<len(SOURCE_SIZES) or SOURCE_SIZES[position]!=n:
            raise ValueError('Invalid source RNG position')
        for count in SOURCE_SIZES[:position+1]:values=random_values(count,generator)
    else:
        values=random_values(n,generator)
        if KIND=='quaternion':
            if mode=='axes':
                axes=torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,1],[-1,0,0,-1],[1,1,1,1],[2,-1,3,-4]],dtype=torch.float32)
                values['quaternion']=axes[torch.arange(n)%len(axes)].contiguous()
            elif mode=='scaled':values['quaternion']*=torch.linspace(.25,4,n)[:,None]
            elif mode!='random':raise ValueError('Unknown quaternion generator')
        elif mode=='diagonal':values['cov01'].zero_()
        elif mode=='indefinite':
            values={'cov00':torch.full((n,),1.),'cov01':torch.full((n,),2.),'cov11':torch.full((n,),1.)}
        elif mode=='small_offdiagonal':values['cov01'].fill_(2**-16)
        elif mode!='random':raise ValueError('Unknown inverse generator')
    if KIND=='quaternion' and LAYOUT=='channels':values['quaternion']=values['quaternion'].t().contiguous()
    return {**values,'N':n,'block_dim':cores,'variant':variant}


def quaternion_rows(inputs):
    q=inputs['quaternion']
    return q.t().contiguous() if LAYOUT=='channels' else q


def covariance_matrix(inputs):
    return torch.stack((inputs['cov00'],inputs['cov01'],inputs['cov01'],inputs['cov11']),-1).reshape(-1,2,2).double()


def validate_inputs(inputs,case=None):
    n=inputs['N'];geometry(n,inputs['block_dim'],inputs['variant'])
    specs={'quaternion':(4,n) if LAYOUT=='channels' else (n,4)} if KIND=='quaternion' else {k:(n,) for k in ('cov00','cov01','cov11')}
    for key,shape in specs.items():
        x=inputs[key]
        if x.shape!=shape or x.dtype!=torch.float32 or x.device.type!='cpu' or not x.is_contiguous() or not torch.isfinite(x).all():
            raise ValueError('Require finite contiguous FP32 input tensors in the declared layout')
    if KIND=='quaternion':
        q=quaternion_rows(inputs);norm2=(q.double()*q.double()).sum(-1)
        if not ((norm2>=2**-20)&(norm2<=2**20)).all():raise ValueError('Quaternion squared norm must be nonzero and in the representable domain')
    else:
        matrix=covariance_matrix(inputs);det=inputs['cov00']*inputs['cov11']-inputs['cov01']*inputs['cov01']
        if not (matrix.abs()<=16).all() or not (det.abs()>=2**-10).all() or not (torch.linalg.cond(matrix)<=64).all():
            raise ValueError('Inverse requires finite bounded and sufficiently nonsingular symmetric matrices')


def reference(inputs):
    validate_inputs(inputs)
    if KIND=='inverse':
        a,b,d=inputs['cov00'],inputs['cov01'],inputs['cov11'];det=a*d-b*b
        return {'inv00':d/det,'inv01':-(b/det),'inv11':a/det}
    q=quaternion_rows(inputs).double();q=q/torch.linalg.vector_norm(q,dim=-1)[:,None]
    w,x,y,z=q.unbind(-1)
    rotation=torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                          2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                          2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).float()
    return {'rotation':rotation.t().contiguous() if LAYOUT=='channels' else rotation}


def validate_reference(inputs,outputs,case=None):
    n=inputs['N']
    if KIND=='quaternion':
        value=outputs['rotation'];shape=(9,n) if LAYOUT=='channels' else (n,9)
        if value.shape!=shape or value.dtype!=torch.float32 or not torch.isfinite(value).all():raise ValueError('Invalid rotation output ABI')
        matrix=(value.t() if LAYOUT=='channels' else value).reshape(n,3,3).double()
        torch.testing.assert_close(matrix.transpose(-1,-2)@matrix,torch.eye(3,dtype=torch.float64).expand(n,3,3),rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(torch.linalg.det(matrix),torch.ones(n,dtype=torch.float64),rtol=1e-5,atol=1e-5)
    else:
        for key in ('inv00','inv01','inv11'):
            value=outputs[key]
            if value.shape!=(n,) or value.dtype!=torch.float32 or not torch.isfinite(value).all():raise ValueError('Invalid inverse output ABI')
        inverse=torch.stack((outputs['inv00'],outputs['inv01'],outputs['inv01'],outputs['inv11']),-1).reshape(n,2,2).double()
        matrix=covariance_matrix(inputs)
        torch.testing.assert_close(inverse,torch.linalg.inv(matrix),rtol=1e-4,atol=1e-4)
        torch.testing.assert_close(matrix@inverse,torch.eye(2,dtype=torch.float64).expand(n,2,2),rtol=1e-4,atol=1e-4)
