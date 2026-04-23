#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
mhc_pre 算子实现

功能：实现 N路流 → 1路 的加权求和操作
公式：out[b,s,d] = Σ_n h[n] × x[b*N+n,s,d]

输入：
  - x: [batch * num_streams, seq, dim] 输入 tensor (N路流)
  - h: [num_streams] 权重系数

输出：
  - out: [batch, seq, dim] 合并后的 tensor
"""

import pypto
import torch
import torch_npu
from numpy.testing import assert_allclose


def mhc_pre_golden(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    mhc_pre 的 Golden 参考实现 (PyTorch 版本)

    实现 N路流 → 1路的加权求和：
    将输入 x [batch * num_streams, seq, dim] 按照系数 h [num_streams] 加权求和得到 [batch, seq, dim]

    Args:
        x: 输入 tensor，形状 [batch * num_streams, seq, dim]
        h: 权重系数，形状 [num_streams]

    Returns:
        输出 tensor，形状 [batch, seq, dim]
    """
    batch_n = x.shape[0]
    num_streams = h.shape[0]
    batch = batch_n // num_streams
    seq = x.shape[1]
    dim = x.shape[2]

    # 将 x 变形为 [batch, num_streams, seq, dim]
    x_reshaped = x.reshape(batch, num_streams, seq, dim)

    # 将 h 扩展为 [1, num_streams, 1, 1] 用于广播
    h_expanded = h.view(1, num_streams, 1, 1)

    # 加权
    weighted = x_reshaped * h_expanded

    # 在 num_streams 维度求和
    out = weighted.sum(dim=1)

    return out


@pypto.jit(
    debug_options={"runtime_debug_mode": 1, "compile_debug_mode": 1},
    runtime_options={"device_sched_mode": 3},
)
def mhc_pre_kernel(x: pypto.Tensor, h: pypto.Tensor, out: pypto.Tensor) -> None:
    batch_n = x.shape[0]
    seq = x.shape[1]
    dim = x.shape[2]
    num_streams = h.shape[0]
    batch = batch_n // num_streams

    pypto.set_vec_tile_shapes(8, 8, 8, 8)

    # 1. 变形 x 到 [batch, num_streams, seq, dim]
    x_reshaped = pypto.reshape(x, [batch, num_streams, seq, dim])

    # 2. 扩展 h 到 [batch, num_streams, seq, dim]
    h_expanded = pypto.reshape(h, [1, num_streams, 1, 1])
    h1 = pypto.concat([h_expanded] * batch, dim=0)
    h2 = pypto.concat([h1] * seq, dim=-2)
    h3 = pypto.concat([h2] * dim, dim=-1)

    # 3. 逐元素乘法
    weighted = pypto.mul(x_reshaped, h3)

    # 4. 在 num_streams 维度求和
    result = pypto.sum(weighted, dim=1)
    pypto.assemble(result, [0, 0, 0], out)


def test_mhc_pre():
    """
    测试 mhc_pre 算子

    验证算子功能正确性和精度
    """

    # 测试配置列表
    test_configs = [
        {"batch": 4, "num_streams": 4, "seq": 512, "dim": 1024, "dtype": torch.float32},
    ]

    for _, config in enumerate(test_configs):

        batch = config["batch"]
        num_streams = config["num_streams"]
        seq = config["seq"]
        dim = config["dim"]
        dtype = config["dtype"]

        # 在 NPU 上生成测试数据
        x = torch.randn(batch * num_streams, seq, dim, dtype=dtype, device='npu')
        h = torch.randn(num_streams, dtype=dtype, device='npu')

        # 在 CPU 上计算 Golden 结果用于对比
        x_cpu = x.cpu()
        h_cpu = h.cpu()
        expected = mhc_pre_golden(x_cpu, h_cpu)

        # 映射 torch dtype 到 pypto dtype
        dtype_map = {
            torch.float32: pypto.DT_FP32,
            torch.float16: pypto.DT_FP16,
            torch.bfloat16: pypto.DT_BF16,
        }
        if x.dtype in dtype_map:
            pto_dtype = dtype_map[x.dtype]
        else:
            pto_dtype = pypto.DT_FP32

        # 创建输出 tensor
        out_shape = (batch, seq, dim)
        out = torch.zeros(out_shape, dtype=dtype, device='npu')

        # 构建 PyPTO 输入输出
        input_tensors = {x: [], h: []}
        output_tensors = {out: []}

        pto_inputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in input_tensors.items()]
        pto_outputs = [pypto.from_torch(tensor, dynamic_axis=axis) for tensor, axis in output_tensors.items()]

        # 执行 PyPTO 算子
        mhc_pre_kernel(*pto_inputs, *pto_outputs)

        # 精度验证
        assert_allclose(expected.cpu().numpy(), out.cpu().numpy(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_mhc_pre()
