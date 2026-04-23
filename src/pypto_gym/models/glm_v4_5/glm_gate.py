#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
GLM-4.5 Gate Module for MoE Expert Routing

This module implements the gate operation that projects hidden states from
the model's main dimension (d_model) to the router-specific dimension (d_router).
This projection is used to compute router logits for expert selection in MoE architectures.

Main Functions:
    - gate: Main gate function for expert routing
    - select_experts_mm_kernel: JIT compiled kernel for matrix multiplication
"""
import os
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose
from torch._subclasses.fake_tensor import FakeTensor
from torch._dynamo import allow_in_graph
import pypto
from utils.get_format import get_format
import pytest


def check_args(
    gate_weight: torch.Tensor,
    hidden_states: torch.Tensor
) -> None:
    """
    Validate input arguments for gate operation.

    Args:
        gate_weight: Gate weight matrix
        hidden_states: Input hidden states
    """
    assert gate_weight.dim() == 2
    assert gate_weight.shape[0] == 160
    assert gate_weight.shape[1] == 5120
    assert get_format(gate_weight) == 'ND'
    assert gate_weight.dtype == torch.float32

    assert hidden_states.dim() == 2
    assert hidden_states.shape[1] == 5120
    assert get_format(hidden_states) == 'ND'
    assert hidden_states.dtype == torch.float32



@pypto.frontend.jit(
    runtime_options={
    "stitch_cfgcache_size": 2500000}
)
def select_experts_mm_kernel(
    hidden_states: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    mm_weight: pypto.Tensor([], pypto.DT_FP32),
    router_logits_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32)
):
    """
    JIT compiled kernel for gate matrix multiplication.

    This kernel performs the matrix multiplication: router_logits = hidden_states @ weight^T
    to project hidden states from d_model to d_router dimension for expert routing.

    Args:
        hidden_states: Input hidden states [num_tokens, hidden_size]
        mm_weight: Gate weight matrix [num_router_experts, hidden_size]
        router_logits_out: Output router logits [num_tokens, num_router_experts]

    Note:
        This function processes inputs in tiles of size 32 to support dynamic batch sizes.
        The computation uses cube tiling for efficient matrix multiplication on NPU.
    """
    # 3. 得到动态tensor的shape
    bs = hidden_states.shape[0]
    ne = mm_weight.shape[0]
    h_num = hidden_states.shape[1]

    view_shape = (32, h_num)

    bs_loop = (bs + view_shape[0] - 1) // view_shape[0]

    # 4. 实现kernel逻辑，循环展开BS动态轴
    for bs_idx in pypto.loop(bs_loop, name="LOOP_MOE_MM_L0", idx_name="bs_idx"):

        # 5. 通过view得到tile_logits
        tile_hidden_states = pypto.view(hidden_states, view_shape,
                                        [bs_idx * view_shape[0], 0],
                                        valid_shape=[(bs - bs_idx * view_shape[0]).min(view_shape[0]),
                                                        h_num])

        pypto.set_cube_tile_shapes([32, 32], [512, 1024], [16, 16])

        res = pypto.matmul(tile_hidden_states, mm_weight, tile_hidden_states.dtype, b_trans=True)

        # 6. 将结果搬运到输出tensor上
        router_logits_out[bs_idx * view_shape[0]:, 0:] = res




@pytest.mark.soc("950", "910")
def test_select_experts_mm():
    # 1. 设置参数
    bs = 64
    ne = 160
    h_num = 5120

    device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
    torch.npu.set_device(device_id)

    # 2. 构造多种shape，测试动态case
    for i in range(0, 1):
        if i == 1:
            bs = 1026
        # 3. 准备测试数据
        torch.manual_seed(0)
        np.random.seed(0)
        hidden_states = torch.rand((bs, h_num), dtype=torch.float32, device=f'npu:{device_id}')
        mm_weight = torch.rand((ne, h_num), dtype=torch.float32, device=f'npu:{device_id}')
        router_logits_out = torch.rand((bs, ne), dtype=torch.float32, device=f'npu:{device_id}')

        # 4. 执行kernel并获取结果
        inputs = [hidden_states, mm_weight, router_logits_out]

        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            gate(*inputs)
        g.replay()

        # 5. 与PyTorch参考实现对比
        result = torch.matmul(hidden_states, mm_weight.t())
        result_list = result.cpu().flatten().tolist()

        # weight result
        assert_allclose(np.array(router_logits_out.cpu().flatten().tolist()),
                        np.array(result_list),
                        rtol=5e-3, atol=5e-3)


@allow_in_graph
def gate(
    hidden_states: torch.Tensor,  # Hidden states of shape (num_tokens, hidden_size).
    gate_weight: torch.Tensor,  # gate matmul weights
    router_logits_out: torch.Tensor
):
    """
    Gate operation for expert routing in MoE architecture.

    This function projects hidden states from the model's main dimension (d_model)
    to the router-specific dimension (d_router) using a learned weight matrix.
    The output router logits are used by the expert selection mechanism to determine
    which experts should process each token.

    Args:
        gate_weight: Gate weight matrix [num_router_experts, hidden_size]
        hidden_states: Input hidden states [num_tokens, hidden_size]
        router_logits_out: Output router logits [num_tokens, num_router_experts]

    Returns:
        router_logits_out: Router logits tensor [num_tokens, num_router_experts]

    Note:
        This function is decorated with @allow_in_graph to enable integration
        with PyTorch's compilation graph.
    """
    if isinstance(hidden_states, FakeTensor):
        return router_logits_out
    check_args(gate_weight, hidden_states)

    inputs = [hidden_states, gate_weight, router_logits_out]
    select_experts_mm_kernel(*inputs)


def main():
    test_select_experts_mm()


if __name__ == "__main__":
    main()
