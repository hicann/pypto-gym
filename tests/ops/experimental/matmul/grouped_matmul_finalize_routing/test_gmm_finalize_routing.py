#!/usr/bin/env python3
# coding: utf-8
"""PyPTO grouped_matmul_finalize_routing operator test.

测试说明：
  - golden 实现来自 gmm_finalize_routing_golden.py
  - kernel 与 host 封装来自 gmm_finalize_routing_impl.py
  - 精度对比使用 numpy.testing.assert_allclose
"""

import argparse
import sys

import pytest
import pypto
import torch
from numpy.testing import assert_allclose

from gmm_finalize_routing_golden import gen_golden, make_group_list
from pypto_gym.ops.pypto_tile.experimental.matmul.grouped_matmul_finalize_routing.gmm_finalize_routing_impl import (
    FinalizeRoutingConfig,
    FinalizeRoutingGoldenInputs,
    FinalizeRoutingInputs,
    gen_pypto,
)

RTOL = 1e-3
ATOL = 1e-3


def test_gmm_finalize_routing(config):
    """单配置测试：构造数据、运行 golden 与 PyPTO，并执行数值对齐校验。"""
    torch_dtype_map = {
        pypto.DT_FP8E4M3: torch.float8_e4m3fn,
        pypto.DT_FP8E5M2: torch.float8_e5m2,
    }
    torch_dtype = torch_dtype_map.get(config.in_dtype, torch.float8_e4m3fn)
    scale_k = (config.k + 63) // 64

    x1 = torch.randn((config.m, config.k), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
    if config.transpose_x2:
        x2 = torch.randn((config.num_experts, config.n, config.k), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
        scale = torch.randn((config.n, scale_k, 2), dtype=torch.float32).uniform_(0, 1)
    else:
        x2 = torch.randn((config.num_experts, config.k, config.n), dtype=torch.float32).uniform_(0, 1).to(torch_dtype)
        scale = torch.randn((scale_k, config.n, 2), dtype=torch.float32).uniform_(0, 1)
    scale = scale.to(torch.float8_e8m0fnu)

    pertoken_scale = torch.randn((config.m, scale_k, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)
    group_list = make_group_list(config.m, config.num_experts, config.group_list_type)
    shared_input = torch.randn((config.batch, config.n), dtype=torch.float32).uniform_(-0.5, 0.5).to(torch.bfloat16)
    logit = torch.randn((config.m,), dtype=torch.float32).uniform_(0, 1)
    row_index = torch.arange(config.m, dtype=torch.int64) % config.batch
    out = torch.zeros((config.batch, config.n), dtype=torch.float32)

    golden = gen_golden(
        FinalizeRoutingGoldenInputs(
            x1=x1,
            x2=x2,
            scale=scale,
            pertoken_scale=pertoken_scale,
            group_list=group_list,
            shared_input=shared_input,
            logit=logit,
            row_index=row_index,
            out=out,
            config=config,
        )
    )

    pypto.set_host_options(compile_monitor_enable=True)
    result = gen_pypto(
        FinalizeRoutingInputs(
            x1=x1,
            x2=x2,
            scale=scale,
            pertoken_scale=pertoken_scale,
            group_list=group_list,
            shared_input=shared_input,
            logit=logit,
            row_index=row_index,
            out=out,
            config=config,
        )
    )

    assert_allclose(golden.cpu().numpy(), result.cpu().numpy(), rtol=RTOL, atol=ATOL)
    print(config.description, "PASSED")


TEST_CONFIGS = [
    FinalizeRoutingConfig(
        batch=128,
        m=768,
        k=6144,
        n=4096,
        num_experts=32,
        m_tile_shape=[128, 128],
        k_tile_shape=[64, 192],
        n_tile_shape=[256, 1024],
        vector_tile_shape=[1, 32, 512, 2],
        in_dtype=pypto.DT_FP8E4M3,
        transpose_x2=True,
        group_list_type=1,
        description="case1 m768 k6144 n4096 e32 (x2 transposed)",
    ),
    FinalizeRoutingConfig(
        batch=256,
        m=768,
        k=8192,
        n=4096,
        num_experts=32,
        m_tile_shape=[128, 128],
        k_tile_shape=[64, 192],
        n_tile_shape=[256, 1024],
        vector_tile_shape=[1, 32, 512, 2],
        in_dtype=pypto.DT_FP8E4M3,
        transpose_x2=True,
        group_list_type=1,
        description="case2 m768 k8192 n4096 e32 (x2 transposed)",
    ),
    FinalizeRoutingConfig(
        batch=64,
        m=128,
        k=5120,
        n=4096,
        num_experts=8,
        m_tile_shape=[128, 128],
        k_tile_shape=[64, 192],
        n_tile_shape=[256, 1024],
        vector_tile_shape=[1, 32, 512, 2],
        in_dtype=pypto.DT_FP8E4M3,
        transpose_x2=True,
        group_list_type=1,
        description="case3 m2048 k6144 n4096 e16",
    ),
    FinalizeRoutingConfig(
        batch=64,
        m=256,
        k=7168,
        n=4096,
        num_experts=16,
        m_tile_shape=[128, 128],
        k_tile_shape=[64, 192],
        n_tile_shape=[256, 1024],
        vector_tile_shape=[1, 32, 512, 2],
        in_dtype=pypto.DT_FP8E5M2,
        transpose_x2=True,
        group_list_type=1,
        description="case4 m2048 k7168 n4096 e16",
    ),
]

test_gmm_finalize_routing = pytest.mark.parametrize("config", TEST_CONFIGS)(test_gmm_finalize_routing)


def main():
    parser = argparse.ArgumentParser(description="PyPTO grouped_matmul_finalize_routing operator test")
    parser.add_argument(
        "case",
        type=int,
        nargs="?",
        metavar="N",
        help="Run single 1-based case index (default: run all)",
    )
    args = parser.parse_args()

    if args.case is not None:
        if not 1 <= args.case <= len(TEST_CONFIGS):
            print(f"ERROR: case must be 1..{len(TEST_CONFIGS)}", file=sys.stderr)
            sys.exit(1)
        test_gmm_finalize_routing(TEST_CONFIGS[args.case - 1])
        return

    for test_config in TEST_CONFIGS:
        test_gmm_finalize_routing(test_config)


if __name__ == "__main__":
    main()
