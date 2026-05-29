#!/usr/bin/env python3
# coding: utf-8
"""E2E test for flash_attention_score — integrated JIT kernel vs golden.

Precision standards (per DESIGN.md §6.2):
  atol=1  rtol=0  MARE<10  MERE<2  RMSE<2
  Exemption: bf16 elements with |golden| < 2^-8 excluded from MERE.

Formulas (IEEE 754-2019 §4.3 / ISO/IEC 10967-2 LIA-2 §5 & §A.3):
  |impl - golden| <= atol + rtol * |golden|          (per-element tolerance)
  MARE = max_i  |d_i| / max(|golden_i|, eps)          (max absolute relative error)
  MERE = max_i  |d_i| / max(|golden_i|, eps)  for |golden_i| >= 2^-8
  RMSE = sqrt( mean( (impl - golden)^2 ) )             (root mean square error)

L0 — canonical shape:  B=1, N=32, N_kv=8, Sq=64, Skv=128, D=128
L1 — tail-block:       B=1, N=8,  N_kv=4, Sq=48, Skv=100, D=128
"""

import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src'))
sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

import torch
import torch_npu  # noqa: F401  required for NPU device init

from experimental.ops_transformer.flash_attention_score.flash_attention_score_impl import flash_attention_score_wrapper
from flash_attention_score_golden import (
    flash_attention_score_golden,
    FlashAttentionInputs,
)


# ============================================================================
# Authoritative precision formulas — no external deps
# ============================================================================

_EPS = torch.finfo(torch.float32).tiny               # ~1.18e-38, underflow guard
_MERE_EXEMPT = 2.0 ** (-8)                            # 0.00390625


def _set_device() -> None:
    """Initialise the NPU device from the standard env var."""
    torch.npu.set_device(int(os.environ["TILE_FWK_DEVICE_ID"]))


def _precision_verify(impl_out, gold_out, tag,
                      atol_bf16=1.0, rtol_bf16=0.0):
    """Authoritative precision check — IEEE 754 / ISO LIA-2 formulas.

    For each of the 3 outputs, applies:
      1. Per-element tolerance:  |impl_i - gold_i| <= atol + rtol * |gold_i|
      2. MARE / MERE / RMSE on bf16 output tensor.
      3. MERE exemption for |gold_i| < 2^-8.

    Returns True iff all checks pass.
    """
    # --- ground on CPU for deterministic compute ---
    gold_out = tuple(t.cpu() for t in gold_out)
    impl_out = tuple(t.cpu() for t in impl_out)

    names   = ("output",      "softmax_max", "softmax_sum")
    atol_rt = [(atol_bf16, rtol_bf16), (1e-5, 1e-5), (1e-5, 1e-5)]

    all_ok = True

    for gold, impl, name, (atol, rtol) in zip(gold_out, impl_out, names, atol_rt):
        g = gold.float()                                # upcast for metric compute
        im = impl.float()
        n_total = g.numel()

        # --- 1. Per-element tolerance (IEEE 754 §4.3) ---
        abs_err = (im - g).abs()
        exceed  = abs_err > (atol + rtol * g.abs())     # bool mask
        n_fail  = exceed.sum().item()
        max_d   = abs_err.max().item()

        passed  = (n_fail == 0)
        print(f"  {tag}_{name}: n={n_total}  fail={n_fail}  "
              f"max_diff={max_d:.2e}  {'PASS' if passed else 'FAIL'}  "
              f"(atol={atol}, rtol={rtol})")

        if not passed:
            all_ok = False

    if not all_ok:
        return False

    # --- 2. MARE / MERE / RMSE on bf16 output tensor only ---
    g     = gold_out[0].cpu().float()
    im    = impl_out[0].cpu().float()
    n_tot = g.numel()

    abs_err = (im - g).abs()
    denom   = g.abs().clamp_min(_EPS)                  # relative-error denominator
    rel_err = abs_err / denom

    # RMSE  (ISO LIA-2 §A.3)
    rmse = abs_err.pow(2).mean().sqrt().item()

    # MARE — over ALL elements
    mare = rel_err.max().item()

    # MERE — restricted to |gold_i| >= 2^-8
    valid_mask = g.abs() >= _MERE_EXEMPT
    n_exempt   = (~valid_mask).sum().item()
    n_valid    = valid_mask.sum().item()
    mere       = rel_err[valid_mask].max().item() if n_valid > 0 else 0.0

    # --- 3. Threshold checks ---
    mare_ok = mare < 10.0
    mere_ok = mere < 2.0
    rmse_ok = rmse < 2.0

    print(f"    MARE = {mare:.6f}  {'PASS' if mare_ok else 'FAIL'}  (threshold < 10)")
    print(f"    MERE = {mere:.6f}  {'PASS' if mere_ok else 'FAIL'}  (threshold <  2)")
    print(f"    RMSE = {rmse:.6f}  {'PASS' if rmse_ok else 'FAIL'}  (threshold <  2)")
    print(f"    exempt (|gold|<2^-8): {n_exempt}/{n_tot}  "
          f"valid: {n_valid}/{n_tot}")

    return mare_ok and mere_ok and rmse_ok


# ============================================================================
# Input generation
# ============================================================================

def _make_inputs(device, seed, shape, pse_type=1, keep_prob=1.0):
    """Build FlashAttentionInputs — all tensors created on device."""
    torch.manual_seed(seed)

    B = shape["B"]
    N = shape["N"]
    N_kv = shape["N_kv"]
    Sq = shape["Sq"]
    Skv = shape["Skv"]
    D_val = shape["D"]

    return FlashAttentionInputs(
        query=torch.randn(B, N, Sq, D_val,
                          dtype=torch.bfloat16, device=device) * 0.1,
        key=torch.randn(B, N_kv, Skv, D_val,
                        dtype=torch.bfloat16, device=device) * 0.1,
        value=torch.randn(B, N_kv, Skv, D_val,
                          dtype=torch.bfloat16, device=device) * 0.1,
        atten_mask=torch.zeros(Sq, Skv,
                               dtype=torch.bfloat16, device=device),
        pse=torch.randn(B, N, Sq, Skv,
                        dtype=torch.bfloat16, device=device) * 0.01,
        drop_mask=torch.ones(Sq, Skv,
                             dtype=torch.bfloat16, device=device),
        pse_type=pse_type,
        keep_prob=keep_prob,
        scale_value=1.0 / (D_val ** 0.5),
    )


# ============================================================================
# Tests
# ============================================================================

def test_l0() -> None:
    """L0: canonical shape (Sq=64, Skv=128) — 2 KV blocks, no tail."""
    _set_device()
    dev = torch.device(f"npu:{int(os.environ['TILE_FWK_DEVICE_ID'])}")

    shape  = {"B": 1, "N": 32, "N_kv": 8, "Sq": 64, "Skv": 128, "D": 128}
    inputs = _make_inputs(dev, seed=42, shape=shape, pse_type=1, keep_prob=1.0)

    print("=" * 60)
    print("L0 — canonical shape (integrated JIT, 2 KV blocks)")
    print(f"  B={shape['B']}, N={shape['N']}, N_kv={shape['N_kv']}, "
          f"Sq={shape['Sq']}, Skv={shape['Skv']}, D={shape['D']}")
    print(f"  pse_type=1, keep_prob=1.0, scale={inputs.scale_value:.6f}")

    gold = flash_attention_score_golden(inputs, npu=True)
    impl = flash_attention_score_wrapper(
        inputs.query, inputs.key, inputs.value,
        inputs.atten_mask, inputs.pse, inputs.drop_mask,
        inputs.pse_type, inputs.keep_prob, inputs.scale_value,
    )

    ok = _precision_verify(impl, gold, tag="L0", atol_bf16=1.0, rtol_bf16=0.0)
    assert ok, "L0: precision check FAILED"
    print()


def test_l1() -> None:
    """L1: tail-block (Sq=48, Skv=100) — partial Q and KV blocks."""
    _set_device()
    dev = torch.device(f"npu:{int(os.environ['TILE_FWK_DEVICE_ID'])}")

    shape  = {"B": 1, "N": 8, "N_kv": 4, "Sq": 48, "Skv": 100, "D": 128}
    inputs = _make_inputs(dev, seed=43, shape=shape, pse_type=1, keep_prob=1.0)

    print("=" * 60)
    print("L1 — tail-block (integrated JIT, Sq=48<64, Skv=100)")
    print(f"  B={shape['B']}, N={shape['N']}, N_kv={shape['N_kv']}, "
          f"Sq={shape['Sq']}, Skv={shape['Skv']}, D={shape['D']}")
    print(f"  pse_type=1, keep_prob=1.0, scale={inputs.scale_value:.6f}")

    gold = flash_attention_score_golden(inputs, npu=True)
    impl = flash_attention_score_wrapper(
        inputs.query, inputs.key, inputs.value,
        inputs.atten_mask, inputs.pse, inputs.drop_mask,
        inputs.pse_type, inputs.keep_prob, inputs.scale_value,
    )

    ok = _precision_verify(impl, gold, tag="L1", atol_bf16=1.0, rtol_bf16=0.0)
    assert ok, "L1: precision check FAILED"
    print()


# ============================================================================
# Runner
# ============================================================================

if __name__ == "__main__":
    print("flash_attention_score E2E test (integrated JIT, no external deps)")
    print("=" * 60)
    print(f"  BLOCK_Q=64, BLOCK_KV=64, D=128")
    print(f"  NPU device: {os.environ.get('TILE_FWK_DEVICE_ID', '0')}")
    print()

    failed = False
    for name, fn in [("L0", test_l0), ("L1", test_l1)]:
        try:
            fn()
            print(f"\N{white heavy check mark} {name} PASSED")
        except Exception as e:
            print(f"\N{cross mark} {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed = True

    print()
    if failed:
        print("[PRECISION_FAIL]")
        sys.exit(1)
    else:
        print("=" * 60)
        print("\N{white heavy check mark} ALL E2E TESTS PASSED")
        print("=" * 60)
        print("[PRECISION_PASS]")
