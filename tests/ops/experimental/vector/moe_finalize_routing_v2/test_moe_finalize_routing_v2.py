# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""PyPTO moe_finalize_routing_v2 operator test.

测试说明：
  - 本文件测试 moe_finalize_routing_v2 算子的精度验证
  - golden 实现来自 moe_finalize_routing_v2_golden.py
  - kernel 实现来自 moe_finalize_routing_v2_impl.py
  - 精度对比使用 numpy.testing.assert_allclose
  - 输出三态标记：[PRECISION_PASS] 或 [PRECISION_FAIL]
"""

import os
import sys

# 设置 Python 路径，使 experimental 包可导入
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src'))
sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

import argparse

# 注意：torch、numpy、golden、impl 等模块将在测试函数中延迟导入
# 这样可以避免在 --list 或 --help 时卡住

# 精度容差
RTOL = 0.0078125
ATOL = 0.0001

# ─────────────────────────────────────────────
# 1. 环境工具
# ─────────────────────────────────────────────


def get_device_id():
    """从环境变量获取 TILE_FWK_DEVICE_ID。"""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        return 0
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be int, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return 0


def setup_npu(device_id):
    """设置 NPU 设备。"""
    import torch_npu  # noqa: F401
    import torch
    torch.npu.set_device(device_id)


# ─────────────────────────────────────────────
# 2. 测试函数
# ─────────────────────────────────────────────

def _prepare_optional_inputs(num_rows, K, H, E, has_x1, has_x2, has_bias, has_scales, device):
    x1 = torch.randn(num_rows, H, dtype=torch.bfloat16, device=device) if has_x1 else None
    x2 = torch.randn(num_rows, H, dtype=torch.bfloat16, device=device) if has_x2 else None

    bias = None
    expert_idx = None
    if has_bias:
        if E is None:
            E = 8
        bias = torch.randn(E, H, dtype=torch.bfloat16, device=device)
        expert_idx = torch.randint(0, E, (num_rows, K), dtype=torch.int32, device=device)

    scales = torch.randn(num_rows, K, dtype=torch.bfloat16, device=device) if has_scales else None

    return x1, x2, bias, scales, expert_idx


def _moe_precision_compare_and_assert(result, golden, run_mode):
    import numpy as np
    from numpy.testing import assert_allclose

    result_np = result.cpu().float().numpy()
    golden_np = golden.cpu().float().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    mean_diff = np.abs(result_np - golden_np).mean()
    print(f"  Max diff    : {max_diff:.6e}")
    print(f"  Mean diff   : {mean_diff:.6e}")

    if run_mode == "npu":
        try:
            assert_allclose(result_np, golden_np, rtol=RTOL, atol=ATOL)
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
            raise
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            raise

    print("  ✓ Passed\n")


def run_moe_finalize_routing_v2_test(
    num_rows, K, H, E=None,
    has_x1=False, has_x2=False, has_bias=False, has_scales=False,
    drop_pad_mode=2, seed=42,
    device_id=None, run_mode="npu", test_name=None
):
    import torch
    from moe_finalize_routing_v2_golden import moe_finalize_routing_v2_golden
    from experimental.vector.moe_finalize_routing_v2.moe_finalize_routing_v2_impl import moe_finalize_routing_v2_wrapper

    if test_name is None:
        test_name = f"num_rows={num_rows}, K={K}, H={H}"

    print("=" * 60)
    print(f"Test: moe_finalize_routing_v2 {test_name}")
    print("=" * 60)

    if run_mode == "npu":
        if device_id is None:
            device_id = get_device_id()
        setup_npu(device_id)
        device = f"npu:{device_id}"
    else:
        device = "cpu"

    torch.manual_seed(seed)

    expanded_x = torch.randn(num_rows * K, H, dtype=torch.bfloat16, device=device)
    expanded_row_idx = torch.randint(0, num_rows * K, (num_rows * K,), dtype=torch.int32, device=device)

    x1, x2, bias, scales, expert_idx = _prepare_optional_inputs(
        num_rows, K, H, E, has_x1, has_x2, has_bias, has_scales, device)

    result = moe_finalize_routing_v2_wrapper(
        expanded_x, expanded_row_idx,
        x1, x2, bias, scales, expert_idx,
        drop_pad_mode
    )

    golden = moe_finalize_routing_v2_golden(
        expanded_x.cpu(), expanded_row_idx.cpu(),
        x1.cpu() if x1 is not None else None,
        x2.cpu() if x2 is not None else None,
        bias.cpu() if bias is not None else None,
        scales.cpu() if scales is not None else None,
        expert_idx.cpu() if expert_idx is not None else None,
        drop_pad_mode
    )

    print(f"  Input shape : expanded_x={expanded_x.shape}")
    print(f"  Output shape: {result.shape}")

    _moe_precision_compare_and_assert(result, golden, run_mode)


# ─────────────────────────────────────────────
# 3. Level 0 测试：基础功能验证
# ─────────────────────────────────────────────

def test_moe_finalize_routing_v2_basic(device_id=None, run_mode="npu"):
    """Level 0: 基础功能验证，K=1，无可选参数。"""
    run_moe_finalize_routing_v2_test(
        num_rows=4096, K=1, H=7168,
        has_x1=False, has_x2=False, has_bias=False, has_scales=False,
        drop_pad_mode=2, seed=42,
        device_id=device_id, run_mode=run_mode,
        test_name="num_rows=4096, K=1, H=7168 (Level 0 - Basic)",
    )


def test_moe_finalize_routing_v2_with_scales(device_id=None, run_mode="npu"):
    """Level 1: 路由权重测试，K=2。"""
    run_moe_finalize_routing_v2_test(
        num_rows=4096, K=4, H=7168,
        has_x1=False, has_x2=False, has_bias=False, has_scales=True,
        drop_pad_mode=2, seed=42,
        device_id=device_id, run_mode=run_mode,
        test_name="num_rows=4096, K=4, H=7168, with scales (Level 1)",
    )


def test_moe_finalize_routing_v2_8(device_id=None, run_mode="npu"):
    """Level 0: drop_pad 场景验证，K=1。"""
    run_moe_finalize_routing_v2_test(
        num_rows=8, K=8, H=7168,
        has_x1=False, has_x2=False, has_bias=False, has_scales=False,
        drop_pad_mode=2, seed=200,
        device_id=device_id, run_mode=run_mode,
        test_name="num_rows=8, K=8, H=7168, drop_pad_mode=2 (Level 0)",
    )


def test_moe_finalize_routing_v2_128(device_id=None, run_mode="npu"):
    """Level 0: drop_pad 场景验证，K=1。"""
    run_moe_finalize_routing_v2_test(
        num_rows=128, K=8, H=7168,
        has_x1=False, has_x2=False, has_bias=False, has_scales=False,
        drop_pad_mode=2, seed=200,
        device_id=device_id, run_mode=run_mode,
        test_name="num_rows=128, K=8, H=7168, drop_pad_mode=2 (Level 0)",
    )


def test_moe_finalize_routing_v2_8192(device_id=None, run_mode="npu"):
    """Level 0: drop_pad 场景验证，K=1。"""
    run_moe_finalize_routing_v2_test(
        num_rows=8192, K=8, H=7168,
        has_x1=False, has_x2=False, has_bias=False, has_scales=False,
        drop_pad_mode=2, seed=200,
        device_id=device_id, run_mode=run_mode,
        test_name="num_rows=8192, K=8, H=7168, drop_pad_mode=2 (Level 0)",
    )


def test_moe_finalize_routing_v2_16384(device_id=None, run_mode="npu"):
    """Level 0: drop_pad 场景验证，K=1。"""
    run_moe_finalize_routing_v2_test(
        num_rows=16384, K=8, H=7168,
        has_x1=False, has_x2=False, has_bias=False, has_scales=False,
        drop_pad_mode=2, seed=200,
        device_id=device_id, run_mode=run_mode,
        test_name="num_rows=16384, K=8, H=7168, drop_pad_mode=2 (Level 0)",
    )

# ─────────────────────────────────────────────
# 5. CLI 入口
# ─────────────────────────────────────────────

EXAMPLES = {
    "moe_finalize_routing_v2::test_moe_finalize_routing_v2_basic": {
        "name": "moe_finalize_routing_v2 Basic",
        "description": "test_moe_finalize_routing_v2_basic",
        "function": test_moe_finalize_routing_v2_basic,
    },
    "moe_finalize_routing_v2::test_moe_finalize_routing_v2_with_scales": {
        "name": "moe_finalize_routing_v2 With Scales",
        "description": "test_moe_finalize_routing_v2_with_scales",
        "function": test_moe_finalize_routing_v2_with_scales,
    },
    "moe_finalize_routing_v2::test_moe_finalize_routing_v2_8": {
        "name": "test_moe_finalize_routing_v2_8",
        "description": "test_moe_finalize_routing_v2_8",
        "function": test_moe_finalize_routing_v2_8,
    },
    "moe_finalize_routing_v2::test_moe_finalize_routing_v2_128": {
        "name": "test_moe_finalize_routing_v2_128",
        "description": "test_moe_finalize_routing_v2_128",
        "function": test_moe_finalize_routing_v2_128,
    },
    "moe_finalize_routing_v2::test_moe_finalize_routing_v2_8192": {
        "name": "test_moe_finalize_routing_v2_8192",
        "description": "test_moe_finalize_routing_v2_8192",
        "function": test_moe_finalize_routing_v2_8192,
    },
    "moe_finalize_routing_v2::test_moe_finalize_routing_v2_16384": {
        "name": "test_moe_finalize_routing_v2_16384",
        "description": "test_moe_finalize_routing_v2_16384",
        "function": test_moe_finalize_routing_v2_16384,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO moe_finalize_routing_v2 operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s moe_finalize_routing_v2::test_moe_finalize_routing_v2_basic    Run Level 0 basic test
  %(prog)s moe_finalize_routing_v2::test_moe_finalize_routing_v2_full_params    Run Level 1 full params test
  %(prog)s --list                                           List all cases

Note: Before running, please set environment:
  source /mnt/workspace/gitCode/cann/pypto/env_setup.sh
        """,
    )
    parser.add_argument("example_id", type=str, nargs="?", help="Case ID to run")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument(
        "--run_mode", "--run-mode",
        type=str, default="npu", choices=["npu", "sim"],
        help="Run mode (default: npu)",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable cases:\n")
        for key, info in sorted(EXAMPLES.items()):
            print(f"  {key}  — {info['description']}")
        print("\nNote: Before running, please set environment:")
        print("  source /mnt/workspace/gitCode/cann/pypto/env_setup.sh")
        return

    if args.example_id:
        if args.example_id not in EXAMPLES:
            print(f"ERROR: unknown case '{args.example_id}'")
            print(f"Valid: {', '.join(sorted(EXAMPLES))}")
            raise RuntimeError("Test execution failed")
        to_run = [(args.example_id, EXAMPLES[args.example_id])]
    else:
        to_run = list(sorted(EXAMPLES.items()))

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()

    try:
        for key, info in to_run:
            print(f"\n▸ Running {key}: {info['name']}")
            info["function"](device_id, args.run_mode)
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()