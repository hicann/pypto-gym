#!/usr/bin/env python3
# coding: utf-8
"""PyPTO grouped_matmul_swiglu_quant operator test.

测试说明：
  - golden 实现来自 gmm_swiglu_quant_golden.py
  - kernel 与 host 封装来自 gmm_swiglu_quant_impl.py
  - 精度对比使用 numpy.testing.assert_allclose
"""

import argparse
import sys
from dataclasses import dataclass

import torch
from numpy.testing import assert_allclose

from gmm_swiglu_quant_golden import gen_golden
from pypto_gym.ops.pypto_tile.experimental.matmul.grouped_matmul_swiglu_quant.gmm_swiglu_quant_impl import (
    GroupedMatmulInputs,
    ShapeConfig,
    TransposeConfig,
    gen_mxfp8,
)

RTOL_OUT = 1e-3
ATOL_OUT = 1
RTOL_QUANT = 1e-4
ATOL_QUANT = 1e-4


@dataclass(frozen=True)
class TestParams:
    """Shape and tiling parameters used by the end-to-end validation entry."""

    m: int
    k: int
    n: int
    group_list: list[int]
    tile_size: int
    m_tile_shape: list[int]
    k_tile_shape: list[int]
    n_tile_shape: list[int]
    vector_tile_shape: list[int]
    transpose: TransposeConfig
    description: str = ""


def test_gmm_mxfp8(params: TestParams):
    """Validate the PyPTO kernel against the PyTorch reference implementation."""
    tile_config = ShapeConfig(
        [params.m, params.k, params.n],
        params.tile_size,
        params.m_tile_shape,
        params.k_tile_shape,
        params.n_tile_shape,
        params.vector_tile_shape,
        params.transpose.a_trans,
        params.transpose.b_trans,
        False,
        False,
        False,
    )

    a = torch.randn((params.m, params.k), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e4m3fn)
    scaled_a = torch.randn((params.m, params.k // 64, 2), dtype=torch.float32).uniform_(0, 1).to(torch.float8_e8m0fnu)

    if params.transpose.b_trans:
        b = (
            torch.randn(
                (len(params.group_list), params.n, params.k),
                dtype=torch.float32,
            ).uniform_(0, 1).to(torch.float8_e4m3fn)
        )
    else:
        b = (
            torch.randn(
                (len(params.group_list), params.k, params.n),
                dtype=torch.float32,
            ).uniform_(0, 1).to(torch.float8_e4m3fn)
        )

    if params.transpose.b_trans:
        scaled_b = torch.randn(
            (len(params.group_list), params.n, params.k // 64, 2),
            dtype=torch.float32,
        ).uniform_(0, 1).to(torch.float8_e8m0fnu)
    else:
        scaled_b = torch.randn(
            (len(params.group_list), params.k // 64, params.n, 2),
            dtype=torch.float32,
        ).uniform_(0, 1).to(torch.float8_e8m0fnu)

    grouped_inputs = GroupedMatmulInputs(
        a=a,
        b=b,
        scaled_a=scaled_a,
        scaled_b=scaled_b,
        group_list=params.group_list,
    )

    golden, golden_quant = gen_golden(grouped_inputs, params.transpose)
    result, result_quant = gen_mxfp8(grouped_inputs, tile_config)

    assert_allclose(golden.cpu().numpy(), result.cpu().numpy(), rtol=RTOL_OUT, atol=ATOL_OUT)
    assert_allclose(golden_quant.cpu().numpy(), result_quant.cpu().numpy(), rtol=RTOL_QUANT, atol=ATOL_QUANT)
    print(params.description, "PASSED")


def get_params(case_name: str) -> TestParams:
    """Get test parameters for specified test case."""
    if case_name == "testcase6":
        return TestParams(
            m=16,
            k=512,
            n=7168,
            group_list=[7, 9],
            tile_size=256,
            m_tile_shape=[9, 9],
            k_tile_shape=[256, 256],
            n_tile_shape=[256, 256],
            vector_tile_shape=[1, 8, 256, 32],
            transpose=TransposeConfig(a_trans=False, b_trans=False),
            description="testcase6 m16 k512 n7168 g[7,9]",
        )
    raise RuntimeError(f"Cannot get parameters for case: {case_name}")


def main():
    parser = argparse.ArgumentParser(description="PyPTO grouped_matmul_swiglu_quant operator test")
    parser.add_argument(
        "case",
        type=str,
        nargs="?",
        default="testcase6",
        help="Case name (default: testcase6)",
    )
    args = parser.parse_args()
    try:
        params = get_params(args.case)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    test_gmm_mxfp8(params)


if __name__ == "__main__":
    main()
