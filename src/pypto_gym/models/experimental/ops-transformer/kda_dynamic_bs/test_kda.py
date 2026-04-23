#!/usr/bin/env python3
# coding: utf-8

"""PyPTO KDA operator test.

Usage:
  python3 test_kda.py --list
  python3 test_kda.py
  python3 test_kda.py kda::case_mid
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from numpy.testing import assert_allclose

from kda_golden import kda_golden
from kda_impl import kda_wrapper


def get_device_id() -> int:
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID={device_id}")
        return 0
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be int, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return 0


def generate_inputs(shape: Tuple[int, int, int], device: str) -> Tuple[torch.Tensor, ...]:
    b, s, d = shape
    q = torch.randn(b, s, d, dtype=torch.float32, device=device)
    k = torch.randn(b, s, d, dtype=torch.float32, device=device)
    v = torch.randn(b, s, d, dtype=torch.float32, device=device)
    alpha = torch.sigmoid(torch.randn(b, s, d, dtype=torch.float32, device=device))
    beta = torch.sigmoid(torch.randn(b, s, d, dtype=torch.float32, device=device))
    return q, k, v, alpha, beta


def run_single_case(case_id: str, shape: Tuple[int, int, int], device_id: int | None, run_mode: str) -> None:
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    torch.manual_seed(0)

    q, k, v, alpha, beta = generate_inputs(shape, device)

    impl_out, impl_state = kda_wrapper(q, k, v, alpha, beta, return_state=True)
    golden_out, golden_state = kda_golden(q, k, v, alpha, beta)

    if run_mode == "npu":
        torch.npu.synchronize()

    impl_out_np = impl_out.detach().cpu().numpy()
    golden_out_np = golden_out.detach().cpu().numpy()
    impl_state_np = impl_state.detach().cpu().numpy()
    golden_state_np = golden_state.detach().cpu().numpy()

    assert_allclose(impl_out_np, golden_out_np, rtol=1e-3, atol=1e-3)
    assert_allclose(impl_state_np, golden_state_np, rtol=1e-3, atol=1e-3)
    print(f"{case_id}: shape={shape} max_diff_out={np.abs(impl_out_np - golden_out_np).max():.6e}")


EXAMPLES: Dict[str, Dict[str, object]] = {
    "kda::case_small": {
        "name": "KDA small",
        "description": "B=1,S=16,D=64",
        "shape": (1, 16, 64),
    },
    "kda::case_mid": {
        "name": "KDA mid",
        "description": "B=2,S=31,D=64",
        "shape": (2, 31, 64),
    },
    "kda::case_large": {
        "name": "KDA large",
        "description": "B=4,S=127,D=64",
        "shape": (4, 127, 64),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="PyPTO KDA operator test")
    parser.add_argument("example_id", type=str, nargs="?", help="Case ID to run")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument(
        "--run_mode", "--run-mode",
        type=str,
        default="npu",
        choices=["npu", "sim"],
        help="Run mode (default: npu)",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable cases:\n")
        for key, info in sorted(EXAMPLES.items()):
            print(f"  {key}  — {info['description']}")
        return

    to_run = []
    if args.example_id:
        if args.example_id not in EXAMPLES:
            print(f"ERROR: unknown case '{args.example_id}'")
            print(f"Valid: {', '.join(sorted(EXAMPLES))}")
            sys.exit(1)
        to_run = [args.example_id]
    else:
        to_run = sorted(EXAMPLES.keys())

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        import torch_npu  # noqa: F401
        torch.npu.set_device(device_id)

    try:
        for case_id in to_run:
            shape = EXAMPLES[case_id]["shape"]  # type: ignore[index]
            run_single_case(case_id, shape, device_id, args.run_mode)
        print("[PRECISION_PASS]")
    except AssertionError as err:
        print(f"[PRECISION_FAIL] {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"Runtime error: {err}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
