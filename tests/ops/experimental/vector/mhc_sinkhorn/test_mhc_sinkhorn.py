#!/usr/bin/env python3
# coding: utf-8
"""PyPTO mhc_sinkhorn operator test.

测试说明：
  - 本文件测试 mhc_sinkhorn 算子的精度验证
  - golden 实现来自 mhc_sinkhorn_golden.py
  - kernel 实现来自 mhc_sinkhorn_impl.py
  - 精度对比使用 numpy.testing.assert_allclose
  - 输出三态标记：[PRECISION_PASS] 或 [PRECISION_FAIL]
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

# golden 文件从同级目录导入
from mhc_sinkhorn_golden import mhc_sinkhorn_golden
# impl 文件使用包导入方式（假设 pypto_gym 已通过 pip install -e . 安装）
from pypto_gym.ops.pypto_tile.experimental.vector.mhc_sinkhorn.mhc_sinkhorn_impl import mhc_sinkhorn_wrapper

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
    import torch_npu
    torch.npu.set_device(device_id)


# ─────────────────────────────────────────────
# 2. 测试函数
# ─────────────────────────────────────────────


def run_mhc_sinkhorn_test(bs, N, N_out, seed, eps, num_iters, device_id=None, run_mode="npu", test_name=None):
    """通用测试函数：执行 mhc_sinkhorn 算子精度验证

    Args:
        bs: batch size (B*S)
        N: 输入矩阵维度
        N_out: 输出矩阵维度
        seed: 随机种子
        eps: Sinkhorn 算法参数
        num_iters: 迭代次数
        device_id: NPU 设备 ID
        run_mode: 运行模式 ("npu" 或 "sim")
        test_name: 测试名称（可选，用于日志输出）
    """
    if test_name is None:
        test_name = f"B*S = {bs}, N = {N}, N_out = {N_out}"

    print("=" * 60)
    print(f"Test: mhc_sinkhorn {test_name}")
    print("=" * 60)

    if run_mode == "npu":
        if device_id is None:
            device_id = get_device_id()
        setup_npu(device_id)
        device = f"npu:{device_id}"
    else:
        device = "cpu"

    # 固定随机种子以保证可重复性
    torch.manual_seed(seed)

    # 准备测试数据
    shape = (bs, N, N_out)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    # 执行 kernel wrapper
    result = mhc_sinkhorn_wrapper(x, eps, num_iters)

    # 执行 golden（在 CPU 上）
    golden = mhc_sinkhorn_golden(x, eps, num_iters)

    # 精度对比
    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {result.shape}")

    result_np = result.cpu().float().numpy()
    golden_np = golden.cpu().float().numpy()

    max_diff = np.abs(result_np - golden_np).max()
    mean_diff = np.abs(result_np - golden_np).mean()
    print(f"  Max diff    : {max_diff:.6e}")
    print(f"  Mean diff   : {mean_diff:.6e}")

    # 三态判定
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

def test_mhc_sinkhorn_bs8_n4_n4(device_id=None, run_mode="npu"):
    """小数据量基础功能验证（8）。"""
    run_mhc_sinkhorn_test(
        bs=8, N=4, N_out=4,
        seed=0, eps=1e-6, num_iters=20,
        device_id=device_id, run_mode=run_mode,
        test_name="B*S = 8, N = 4, N_out = 4 (Level 0)",
    )

def test_mhc_sinkhorn_bs64_n4_n4(device_id=None, run_mode="npu"):
    """小数据量基础功能验证（64）。"""
    run_mhc_sinkhorn_test(
        bs=64, N=4, N_out=4,
        seed=0, eps=1e-6, num_iters=20,
        device_id=device_id, run_mode=run_mode,
        test_name="B*S = 64, N = 4, N_out = 4 (Level 0)",
    )

def test_mhc_sinkhorn_bs1024_n4_n4(device_id=None, run_mode="npu"):
    """小数据量基础功能验证（1024）。"""
    run_mhc_sinkhorn_test(
        bs=1024, N=4, N_out=4,
        seed=0, eps=1e-6, num_iters=20,
        device_id=device_id, run_mode=run_mode,
        test_name="B*S = 1024, N = 4, N_out = 4 (Level 0)",
    )


def test_mhc_sinkhorn_bs2048_n4_n4(device_id=None, run_mode="npu"):
    """典型场景验证（2048）。"""
    run_mhc_sinkhorn_test(
        bs=2048, N=4, N_out=4,
        seed=42, eps=1e-6, num_iters=20,
        device_id=device_id, run_mode=run_mode,
        test_name="B*S = 2048, N = 4, N_out = 4 (Level 1)",
    )


def test_mhc_sinkhorn_bs4096_n4_n4(device_id=None, run_mode="npu"):
    """大规模验证（4096）。"""
    run_mhc_sinkhorn_test(
        bs=4096, N=4, N_out=4,
        seed=100, eps=1e-6, num_iters=20,
        device_id=device_id, run_mode=run_mode,
        test_name="B*S = 4096, N = 4, N_out = 4 (Level 2)",
    )


# ─────────────────────────────────────────────
# 4. CLI 入口
# ─────────────────────────────────────────────

EXAMPLES = {
    "mhc_sinkhorn::test_mhc_sinkhorn_bs8_n4_n4": {
        "name": "mhc_sinkhorn B*S = 8, N = 4, N_out = 4",
        "description": "test_mhc_sinkhorn_bs8_n4_n4",
        "function": test_mhc_sinkhorn_bs8_n4_n4,
    },
    "mhc_sinkhorn::test_mhc_sinkhorn_bs64_n4_n4": {
        "name": "mhc_sinkhorn B*S = 64, N = 4, N_out = 4",
        "description": "test_mhc_sinkhorn_bs64_n4_n4",
        "function": test_mhc_sinkhorn_bs64_n4_n4,
    },
    "mhc_sinkhorn::test_mhc_sinkhorn_bs1024_n4_n4": {
        "name": "mhc_sinkhorn B*S = 1024, N = 4, N_out = 4",
        "description": "test_mhc_sinkhorn_bs1024_n4_n4",
        "function": test_mhc_sinkhorn_bs1024_n4_n4,
    },
    "mhc_sinkhorn::test_mhc_sinkhorn_bs2048_n4_n4": {
        "name": "mhc_sinkhorn B*S = 2048, N = 4, N_out = 4",
        "description": "test_mhc_sinkhorn_bs2048_n4_n4",
        "function": test_mhc_sinkhorn_bs2048_n4_n4,
    },
    "mhc_sinkhorn::test_mhc_sinkhorn_bs4096_n4_n4": {
        "name": "mhc_sinkhorn B*S = 4096, N = 4, N_out = 4",
        "description": "test_mhc_sinkhorn_bs4096_n4_n4",
        "function": test_mhc_sinkhorn_bs4096_n4_n4,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO mhc_sinkhorn operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s mhc_sinkhorn::test_mhc_sinkhorn_bs2048_n4_n4    Run Level 1 test
  %(prog)s --list                                           List all cases
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
        return

    if args.example_id:
        if args.example_id not in EXAMPLES:
            print(f"ERROR: unknown case '{args.example_id}'")
            print(f"Valid: {', '.join(sorted(EXAMPLES))}")
            sys.exit(1)
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