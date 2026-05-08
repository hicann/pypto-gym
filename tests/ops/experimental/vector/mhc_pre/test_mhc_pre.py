#!/usr/bin/env python3
# coding: utf-8

"""PyPTO mhc_pre operator test.

MHC Pre-processing 算子精度验证测试

测试配置:
  - 测试 B*S = 1024, 2048, 4096
  - atol=0.0001, rtol=0.0078125
  - 使用 NPU 模式验证（若可用）

三态标记:
  - [PRECISION_PASS]: 精度验证通过
  - [PRECISION_FAIL]: 精度验证失败
  - 无标记 + exit ≠ 0: 功能问题
"""


import sys, os; _p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')): _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src')); sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

import os
import sys
import argparse

import torch
import torch_npu  # noqa: F401
import numpy as np
from numpy.testing import assert_allclose

# golden 文件从同级目录导入
from mhc_pre_golden import mhc_pre_golden
# impl 文件使用包导入方式（假设 pypto_gym 已通过 pip install -e . 安装）
from experimental.vector.mhc_pre.mhc_pre_impl import mhc_pre_wrapper


# 精度容差（来自 SPEC.md）
RTOL = 0.0078125
ATOL = 0.0001

# ─────────────────────────────────────────────
# 1. 环境工具
# ─────────────────────────────────────────────

def get_device_id():
    """从环境变量获取 TILE_FWK_DEVICE_ID。"""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID={device_id}")
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

def run_mhc_pre_test(bs, N, D, device_id=None, run_mode="npu", test_name=None):
    """通用测试函数：执行 mhc_pre 算子精度验证
    
    Args:
        bs: batch size
        N: 注意力头数
        D: 隐藏层维度
        device_id: NPU 设备 ID
        run_mode: 运行模式 ("npu" 或 "sim")
        test_name: 测试名称（可选，用于日志输出）
    """
    if test_name is None:
        test_name = f"B*S = {bs}"
    
    print("=" * 60)
    print(f"Test: mhc_pre {test_name}")
    print("=" * 60)

    # 设置设备
    if run_mode == "npu":
        if device_id is None:
            device_id = get_device_id()
        setup_npu(device_id)
        device = f"npu:{device_id}"
    else:
        device = "cpu"

    # 计算派生参数
    N_SQUARED_PLUS_2N = N * N + 2 * N

    # 固定随机种子以保证可重复性
    torch.manual_seed(42)

    # 准备测试数据
    x = torch.randn(bs, N, D, dtype=torch.bfloat16, device=device)
    phi = torch.randn(N_SQUARED_PLUS_2N, N * D, dtype=torch.float32, device=device)
    alpha = torch.randn(3, dtype=torch.float32, device=device)
    bias = torch.randn(N_SQUARED_PLUS_2N, dtype=torch.float32, device=device)

    # 执行 impl
    h_in_impl, h_post_impl, h_res_impl = mhc_pre_wrapper(x, phi, alpha, bias)

    # 执行 golden
    h_in_golden, h_post_golden, h_res_golden = mhc_pre_golden(x.cpu(), phi.cpu(), alpha.cpu(), bias.cpu())

    # 精度对比
    print(f"  Input shape : x {x.shape}, phi {phi.shape}")
    print(f"  Output shape: h_in {h_in_impl.shape}, h_post {h_post_impl.shape}, h_res {h_res_impl.shape}")

    # h_in (BF16 → FP32)
    h_in_impl_np = h_in_impl.cpu().float().numpy()
    h_in_golden_np = h_in_golden.float().numpy()
    max_diff_h_in = np.abs(h_in_impl_np - h_in_golden_np).max()
    print(f"  h_in max diff: {max_diff_h_in:.6e}")

    # h_post (FP32)
    h_post_impl_np = h_post_impl.cpu().numpy()
    h_post_golden_np = h_post_golden.numpy()
    max_diff_h_post = np.abs(h_post_impl_np - h_post_golden_np).max()
    print(f"  h_post max diff: {max_diff_h_post:.6e}")

    # h_res (FP32)
    # impl 输出 h_res 现为 3D [bs, N, N]，与 golden 输出一致
    # 直接对比即可
    h_res_impl_np = h_res_impl.cpu().numpy()
    h_res_golden_np = h_res_golden.numpy()
    max_diff_h_res = np.abs(h_res_impl_np - h_res_golden_np).max()
    print(f"  h_res max diff: {max_diff_h_res:.6e}")

    # 三态判定
    if run_mode == "npu":
        try:
            assert_allclose(h_in_impl_np, h_in_golden_np, rtol=RTOL, atol=ATOL)
            assert_allclose(h_post_impl_np, h_post_golden_np, rtol=RTOL, atol=ATOL)
            assert_allclose(h_res_impl_np, h_res_golden_np, rtol=RTOL, atol=ATOL)
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
            raise
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            raise

    print("  ✓ Passed\n")


def test_mhc_pre_bs128_n4_d5120(device_id=None, run_mode="npu"):
    """测试 case: B*S = 128"""
    run_mhc_pre_test(bs=128, N=4, D=5120, device_id=device_id, run_mode=run_mode, test_name="B*S = 128")

def test_mhc_pre_bs8(device_id=None, run_mode="npu"):
    """测试 case: B*S = 8"""
    run_mhc_pre_test(bs=8, N=4, D=128, device_id=device_id, run_mode=run_mode, test_name="B*S = 8")

def test_mhc_pre_bs256(device_id=None, run_mode="npu"):
    """测试 case: B*S = 256"""
    run_mhc_pre_test(bs=256, N=4, D=128, device_id=device_id, run_mode=run_mode, test_name="B*S = 256")

def test_mhc_pre_bs1024(device_id=None, run_mode="npu"):
    """测试 case: B*S = 1024"""
    run_mhc_pre_test(bs=1024, N=4, D=5120, device_id=device_id, run_mode=run_mode, test_name="B*S = 1024")

def test_mhc_pre_bs4096(device_id=None, run_mode="npu"):
    """测试 case: B*S = 4096"""
    run_mhc_pre_test(bs=4096, N=4, D=2560, device_id=device_id, run_mode=run_mode, test_name="B*S = 4096")


# ─────────────────────────────────────────────
# 3. CLI 入口
# ─────────────────────────────────────────────

EXAMPLES = {
    "mhc_pre::test_mhc_pre_bs128_n4_d5120": {
        "name": "mhc_pre B*S = 128",
        "description": "泛化用例-1精度验证",
        "function": test_mhc_pre_bs128_n4_d5120,
    },
    "mhc_pre::test_mhc_pre_bs8": {
        "name": "mhc_pre B*S = 8",
        "description": "泛化用例-2精度验证",
        "function": test_mhc_pre_bs8,
    },
    "mhc_pre::test_mhc_pre_bs256": {
        "name": "mhc_pre B*S = 256",
        "description": "泛化用例-3精度验证",
        "function": test_mhc_pre_bs256,
    },
    "mhc_pre::test_mhc_pre_bs1024": {
        "name": "mhc_pre B*S = 1024",
        "description": "中等规模精度验证",
        "function": test_mhc_pre_bs1024,
    },
    "mhc_pre::test_mhc_pre_bs4096": {
        "name": "mhc_pre B*S = 4096",
        "description": "大规模精度验证",
        "function": test_mhc_pre_bs4096,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO mhc_pre operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s mhc_pre::test_mhc_pre_bs1024    Run B*S=1024 test
  %(prog)s --list                          List all cases
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
        if device_id is None:
            return

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