# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------



"""PyPTO Sigmoid golden reference implementation.

Operator: Sigmoid
Formula: sigma(x) = 1 / (1 + exp(-x))
Confidence: ⭐⭐⭐⭐⭐ (PyTorch built-in API torch.sigmoid)

Template notes:
  - golden must be pure PyTorch, no pypto allowed.
  - Exported function Sigmoid_golden() for test_sigmoid.py.
"""

import torch
from typing import List


def Sigmoid_golden(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid activation function PyTorch reference implementation."""
    return torch.sigmoid(x)


def _validate_typical_cases(rtol, atol):
    """Validate typical cases from operator spec."""
    all_passed = True
    print("\n[Typical case validation]")
    typical_cases = [
        ("perf_P0", [16, 16384]),
        ("func_P0", [1, 16384]),
        ("func_P1", [64, 16384]),
    ]
    for name, shape in typical_cases:
        x = torch.randn(shape, dtype=torch.float32)
        y = Sigmoid_golden(x)
        expected_shape = torch.Size(shape)
        if y.shape != expected_shape:
            print(f"  {name}: shape={shape} ... ✗ FAIL (shape mismatch: {y.shape} vs {expected_shape})")
            all_passed = False
            continue
        if y.dtype != torch.float32:
            print(f"  {name}: shape={shape} ... ✗ FAIL (dtype mismatch: {y.dtype} vs torch.float32)")
            all_passed = False
            continue
        if y.min().item() <= 0.0 or y.max().item() >= 1.0:
            print(f"  {name}: shape={shape} ... ✗ FAIL (range out of bounds: min={y.min():.6f}, max={y.max():.6f})")
            all_passed = False
            continue
        ref = torch.sigmoid(x)
        if not torch.allclose(y, ref, rtol=rtol, atol=atol):
            max_diff = (y - ref).abs().max().item()
            print(f"  {name}: shape={shape} ... ✗ FAIL (inconsistent with torch.sigmoid, max_diff={max_diff:.6e})")
            all_passed = False
            continue
        print(f"  {name}: shape={shape} ... ✓ PASS")
    return all_passed


def _validate_generalization_cases(rtol, atol):
    """Validate generalization cases with varying B dimension."""
    all_passed = True
    print("\n[Generalization case validation]")
    b_values = [1, 32, 128]
    for b in b_values:
        shape = [b, 16384]
        x = torch.randn(shape, dtype=torch.float32)
        y = Sigmoid_golden(x)
        expected_shape = torch.Size(shape)
        if y.shape != expected_shape:
            print(f"  B={b}: shape={shape} ... ✗ FAIL (shape mismatch)")
            all_passed = False
            continue
        ref = torch.sigmoid(x)
        if not torch.allclose(y, ref, rtol=rtol, atol=atol):
            print(f"  B={b}: shape={shape} ... ✗ FAIL (inconsistent with torch.sigmoid)")
            all_passed = False
            continue
        print(f"  B={b}: shape={shape} ... ✓ PASS")
    return all_passed


def _validate_range_check():
    """Validate sigmoid range [0,1] for extreme inputs."""
    print("\n[Range check]")
    x_large_pos = torch.tensor([[100.0] * 16384], dtype=torch.float32)
    y_large_pos = Sigmoid_golden(x_large_pos)
    assert (y_large_pos >= 0).all() and (y_large_pos <= 1).all(), "Large pos: range out of bounds"
    assert torch.allclose(y_large_pos, torch.ones_like(y_large_pos), atol=1e-4), "Large pos: should be close to 1"
    print(f"  Large pos input (x=100) ... ✓ PASS (min={y_large_pos.min():.6f}, max={y_large_pos.max():.6f})")

    x_large_neg = torch.tensor([[-100.0] * 16384], dtype=torch.float32)
    y_large_neg = Sigmoid_golden(x_large_neg)
    assert (y_large_neg >= 0).all() and (y_large_neg <= 1).all(), "Large neg: range out of bounds"
    assert torch.allclose(y_large_neg, torch.zeros_like(y_large_neg), atol=1e-4), "Large neg: should be close to 0"
    print(f"  Large neg input (x=-100) ... ✓ PASS (min={y_large_neg.min():.6f}, max={y_large_neg.max():.6f})")

    x_zero = torch.zeros([2, 16384], dtype=torch.float32)
    y_zero = Sigmoid_golden(x_zero)
    assert (y_zero >= 0).all() and (y_zero <= 1).all(), "Zero: range out of bounds"
    assert torch.allclose(y_zero, torch.full_like(y_zero, 0.5), atol=1e-6), "Zero: sigmoid(0) should be 0.5"
    print(f"  Zero input (x=0) ... ✓ PASS (min={y_zero.min():.6f}, max={y_zero.max():.6f})")


def _validate_numerical_stability():
    """Validate sigmoid for extreme fp32 inputs."""
    all_passed = True
    print("\n[Numerical stability check]")
    x_extreme = torch.tensor([[1e10, -1e10, 1e38, -1e38] + [0.0] * 16380], dtype=torch.float32)
    y_extreme = Sigmoid_golden(x_extreme)
    has_nan = torch.isnan(y_extreme).any().item()
    has_inf = torch.isinf(y_extreme).any().item()
    if has_nan or has_inf:
        print(f"  Extreme input ... ✗ FAIL (NaN={has_nan}, Inf={has_inf})")
        all_passed = False
    else:
        print(f"  Extreme input ... ✓ PASS (no NaN/Inf)")
    return all_passed


def _validate_api_comparison(rtol, atol):
    """Compare with torch.sigmoid."""
    all_passed = True
    print("\n[API comparison]")
    x_rand = torch.randn([16, 16384], dtype=torch.float32)
    y_golden = Sigmoid_golden(x_rand)
    y_ref = torch.sigmoid(x_rand)
    max_diff = (y_golden - y_ref).abs().max().item()
    if torch.allclose(y_golden, y_ref, rtol=rtol, atol=atol):
        print(f"  Compare with torch.sigmoid ... ✓ PASS (max_diff={max_diff:.2e})")
    else:
        print(f"  Compare with torch.sigmoid ... ✗ FAIL (max_diff={max_diff:.2e})")
        all_passed = False
    return all_passed


def _validate_math_properties():
    """Validate monotonicity and symmetry properties."""
    all_passed = True
    print("\n[Mathematical property check]")
    x_mono = torch.randn([4, 16384], dtype=torch.float32)
    x_sorted, _ = torch.sort(x_mono, dim=-1)
    y_sorted = Sigmoid_golden(x_sorted)
    diff = y_sorted[:, 1:] - y_sorted[:, :-1]
    if (diff >= -1e-7).all():
        print(f"  Monotonicity ... ✓ PASS")
    else:
        print(f"  Monotonicity ... ✗ FAIL")
        all_passed = False

    x_sym = torch.randn([4, 16384], dtype=torch.float32)
    y_pos = Sigmoid_golden(x_sym)
    y_neg = Sigmoid_golden(-x_sym)
    if torch.allclose(y_pos + y_neg, torch.ones_like(y_pos), atol=1e-6):
        print(f"  Symmetry sigma(-x) = 1 - sigma(x) ... ✓ PASS")
    else:
        print(f"  Symmetry sigma(-x) = 1 - sigma(x) ... ✗ FAIL")
        all_passed = False
    return all_passed


def _validate_function_signature():
    """Validate function signature matches spec."""
    all_passed = True
    print("\n[Function signature check]")
    import inspect
    sig = inspect.signature(Sigmoid_golden)
    params = list(sig.parameters.keys())
    if params == ["x"]:
        print(f"  Function signature Sigmoid_golden(x) ... ✓ PASS")
    else:
        print(f"  Function signature ... ✗ FAIL (params: {params}, expected: ['x'])")
        all_passed = False
    return all_passed


def _validate():
    """Auto-generated validation function - runs validation report dynamically."""
    op_name = "Sigmoid"
    rtol = 0.001
    atol = 0.001

    print("=" * 60)
    print(f"{op_name}_golden validation report")
    print("=" * 60)

    all_passed = True
    all_passed &= _validate_typical_cases(rtol, atol)
    all_passed &= _validate_generalization_cases(rtol, atol)
    _validate_range_check()
    all_passed &= _validate_numerical_stability()
    all_passed &= _validate_api_comparison(rtol, atol)
    all_passed &= _validate_math_properties()
    all_passed &= _validate_function_signature()

    print("\n" + "=" * 60)
    if all_passed:
        print("All validations passed")
    else:
        print("Some validations failed, check output above")
    print("=" * 60)


if __name__ == "__main__":
    _validate()
