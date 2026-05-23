#!/usr/bin/env python3
# coding: utf-8

"""PyPTO RMSNorm golden reference implementation.

Operator: RMSNorm
Formula: out[b, c, h, w] = x[b, c, h, w] / sqrt(mean(x[b, j, h, w]^2, dim=1) + eps)
Confidence: ⭐⭐⭐⭐⭐ (uses PyTorch built-in operations)

Template notes:
  - golden must be pure PyTorch, no pypto allowed.
  - Exported function RMSNorm_golden() for test_rms_norm.py.
"""

import torch
from typing import Optional

# ─────────────────────────────────────────────
# Golden reference (pure torch)
# ─────────────────────────────────────────────

def RMSNorm_golden(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMS Normalization (Root Mean Square Normalization) PyTorch reference.

    Computes RMS along feature dimension (dim=1) and normalizes input by RMS.
    Unlike LayerNorm, RMSNorm does not subtract mean, only divides by RMS.

    Formula:
        y = x / sqrt(mean(x^2, dim=1, keepdim=True) + eps)

    Args:
        x: Input tensor, shape [B, num_features, H, W], dtype float32.
        eps: Small constant to prevent division by zero, default 1e-5.

    Returns:
        RMS-normalized output tensor, same shape as input [B, num_features, H, W].

    Example:
        >>> x = torch.randn(2, 64, 256, 256)
        >>> y = RMSNorm_golden(x, eps=1e-5)
        >>> y.shape
        torch.Size([2, 64, 256, 256])
    """
    mean_sq = torch.mean(x ** 2, dim=1, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    return x / rms


# ==========================================
# Validation
# ==========================================

def _validate():
    """Auto-generated validation function - runs validation report dynamically."""

    print("=" * 60)
    print("RMSNorm_golden validation report")
    print("=" * 60)

    all_passed = True

    print("\n[Typical case validation]")

    x_perf = torch.randn(16, 64, 256, 256, dtype=torch.float32)
    y_perf = RMSNorm_golden(x_perf, eps=1e-5)
    perf_pass = y_perf.shape == x_perf.shape
    print(f"  perf_P0: shape=[16, 64, 256, 256] ... {'✓ PASS' if perf_pass else '✗ FAIL'}")
    if not perf_pass:
        all_passed = False

    x_func = torch.randn(16, 64, 256, 256, dtype=torch.float32)
    y_func = RMSNorm_golden(x_func, eps=1e-5)
    func_pass = y_func.shape == x_func.shape
    print(f"  func_P0: shape=[16, 64, 256, 256] ... {'✓ PASS' if func_pass else '✗ FAIL'}")
    if not func_pass:
        all_passed = False

    print("\n[Generalization case validation]")

    b_values = [1, 512, 1024]
    for b in b_values:
        x_gen = torch.randn(b, 64, 256, 256, dtype=torch.float32)
        y_gen = RMSNorm_golden(x_gen, eps=1e-5)
        gen_pass = y_gen.shape == x_gen.shape
        print(f"  B={b}: shape=[{b}, 64, 256, 256] ... {'✓ PASS' if gen_pass else '✗ FAIL'}")
        if not gen_pass:
            all_passed = False

    for hw in [(128, 128), (64, 64), (32, 32)]:
        h, w = hw
        x_hw = torch.randn(4, 64, h, w, dtype=torch.float32)
        y_hw = RMSNorm_golden(x_hw, eps=1e-5)
        hw_pass = y_hw.shape == x_hw.shape
        print(f"  B=4, C=64, H={h}, W={w}: shape=[4, 64, {h}, {w}] ... {'✓ PASS' if hw_pass else '✗ FAIL'}")
        if not hw_pass:
            all_passed = False

    print("\n[Range validation]")

    x_val = torch.randn(4, 64, 32, 32, dtype=torch.float32)
    y_val = RMSNorm_golden(x_val, eps=1e-5)
    output_mean_sq = torch.mean(y_val ** 2, dim=1)
    val_pass = torch.allclose(output_mean_sq, torch.ones_like(output_mean_sq), atol=0.01, rtol=0.01)
    print(f"  output mean_sq ≈ 1 (along dim=1) ... {'✓ PASS' if val_pass else '✗ FAIL'}")
    if not val_pass:
        all_passed = False

    for eps_val in [1e-8, 1e-5, 1e-2]:
        x_eps = torch.randn(2, 64, 16, 16, dtype=torch.float32)
        y_eps = RMSNorm_golden(x_eps, eps=eps_val)
        eps_pass = y_eps.shape == x_eps.shape and not torch.any(torch.isnan(y_eps))
        print(f"  eps={eps_val} ... {'✓ PASS' if eps_pass else '✗ FAIL'}")
        if not eps_pass:
            all_passed = False

    print("\n[Numerical stability check]")

    x_large = torch.randn(2, 64, 16, 16, dtype=torch.float32) * 1e4
    y_large = RMSNorm_golden(x_large, eps=1e-5)
    large_pass = not torch.any(torch.isnan(y_large)) and not torch.any(torch.isinf(y_large))
    print(f"  Large input (scale=1e4) ... {'✓ PASS' if large_pass else '✗ FAIL'}")
    if not large_pass:
        all_passed = False

    x_small = torch.randn(2, 64, 16, 16, dtype=torch.float32) * 1e-6
    y_small = RMSNorm_golden(x_small, eps=1e-5)
    small_pass = not torch.any(torch.isnan(y_small)) and not torch.any(torch.isinf(y_small))
    print(f"  Small input (scale=1e-6) ... {'✓ PASS' if small_pass else '✗ FAIL'}")
    if not small_pass:
        all_passed = False

    x_zero = torch.zeros(2, 64, 16, 16, dtype=torch.float32)
    y_zero = RMSNorm_golden(x_zero, eps=1e-5)
    zero_pass = torch.all(y_zero == 0.0)
    print(f"  All-zero input (output should be zero) ... {'✓ PASS' if zero_pass else '✗ FAIL'}")
    if not zero_pass:
        all_passed = False

    x_single = torch.randn(2, 1, 16, 16, dtype=torch.float32)
    y_single = RMSNorm_golden(x_single, eps=1e-5)
    single_pass = y_single.shape == x_single.shape and not torch.any(torch.isnan(y_single))
    print(f"  Single feature dim (C=1) ... {'✓ PASS' if single_pass else '✗ FAIL'}")
    if not single_pass:
        all_passed = False

    print("\n[Mathematical property validation]")

    x_scale = torch.randn(2, 64, 16, 16, dtype=torch.float32)
    k = 3.7
    y_orig = RMSNorm_golden(x_scale, eps=1e-5)
    y_scaled = RMSNorm_golden(k * x_scale, eps=1e-5)
    scale_pass = torch.allclose(y_orig, y_scaled, atol=1e-4, rtol=1e-4)
    print(f"  Scale invariance (k={k}) ... {'✓ PASS' if scale_pass else '✗ FAIL'}")
    if not scale_pass:
        all_passed = False

    print("\n[Function signature validation]")

    import inspect
    sig = inspect.signature(RMSNorm_golden)
    params = list(sig.parameters.keys())
    sig_pass = 'x' in params and 'eps' in params
    print(f"  Signature: RMSNorm_golden{sig} ... {'✓ PASS' if sig_pass else '✗ FAIL'}")
    if not sig_pass:
        all_passed = False

    print("\n[dtype validation]")

    x_f32 = torch.randn(2, 64, 16, 16, dtype=torch.float32)
    y_f32 = RMSNorm_golden(x_f32, eps=1e-5)
    dtype_pass = y_f32.dtype == torch.float32
    print(f"  float32 input -> float32 output ... {'✓ PASS' if dtype_pass else '✗ FAIL'}")
    if not dtype_pass:
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All validations passed")
    else:
        print("Some validations failed, check output above")
    print("=" * 60)


if __name__ == "__main__":
    _validate()
