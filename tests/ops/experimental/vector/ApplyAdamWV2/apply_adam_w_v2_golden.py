"""
Golden reference implementation for apply_adam_w_v2.

Algorithm: AdamW single-step update with bias correction (Loshchilov & Hutter, 2019).

    m_t   = beta1 * m + (1 - beta1) * g
    v_t   = beta2 * v + (1 - beta2) * g * g
    m_hat = m_t / (1 - beta1**t)
    v_hat = v_t / (1 - beta2**t)
    update = m_hat / (sqrt(v_hat) + eps) + weight_decay * weight
    w_new  = weight - lr * update

Inputs:
    weight, grad : [7168, K], bfloat16 OR float32
    m, v         : [7168, K], float32
    beta1, beta2, lr, weight_decay, eps : python float
    step         : python int (>=1)

Outputs (returned as a tuple, also functionally equivalent to in-place update):
    weight_out : same dtype as input weight
    m_out      : float32
    v_out      : float32

Mixed precision contract: weight/grad in bf16 are cast up to fp32 internally,
all math is done in fp32, weight is cast back to its original dtype on return.

Confidence: ***** (direct mapping of the AdamW formula, no algorithmic ambiguity).
"""
from __future__ import annotations

from typing import Tuple

import torch


def apply_adam_w_v2_golden(
    weight: torch.Tensor,
    grad: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    step: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch AdamW single-step update with bias correction."""
    # --- shape / dtype guards ---
    assert weight.shape == grad.shape == m.shape == v.shape, (
        f"shape mismatch: weight={weight.shape} grad={grad.shape} "
        f"m={m.shape} v={v.shape}"
    )
    assert weight.dtype in (torch.bfloat16, torch.float32), (
        f"weight dtype must be bfloat16 or float32, got {weight.dtype}"
    )
    assert grad.dtype == weight.dtype, (
        f"grad dtype {grad.dtype} must match weight dtype {weight.dtype}"
    )
    assert m.dtype == torch.float32, f"m dtype must be float32, got {m.dtype}"
    assert v.dtype == torch.float32, f"v dtype must be float32, got {v.dtype}"
    assert isinstance(step, int) and step >= 1, f"step must be int >= 1, got {step}"

    weight_dtype = weight.dtype

    # --- cast inputs to fp32 accumulators ---
    w_f32 = weight.to(torch.float32)
    g_f32 = grad.to(torch.float32)
    m_f32 = m  # already fp32
    v_f32 = v  # already fp32

    # --- bias correction scalars (host side) ---
    bc1 = 1.0 - (beta1 ** step)
    bc2 = 1.0 - (beta2 ** step)

    # --- moment updates ---
    m_new = beta1 * m_f32 + (1.0 - beta1) * g_f32
    v_new = beta2 * v_f32 + (1.0 - beta2) * (g_f32 * g_f32)

    # --- bias-corrected estimates ---
    m_hat = m_new / bc1
    v_hat = v_new / bc2

    # --- AdamW update (decoupled weight decay) ---
    update = m_hat / (torch.sqrt(v_hat) + eps) + weight_decay * w_f32
    w_new_f32 = w_f32 - lr * update

    # --- cast weight back to original dtype; m/v stay in fp32 ---
    weight_out = w_new_f32.to(weight_dtype)
    m_out = m_new
    v_out = v_new

    return weight_out, m_out, v_out


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
def _smoke_test() -> None:
    torch.manual_seed(0)

    # Small case from SPEC P0: [7168, 2048]
    shape = (7168, 2048)

    for w_dtype in (torch.float32, torch.bfloat16):
        weight = torch.randn(shape, dtype=w_dtype) * 0.02
        grad = torch.randn(shape, dtype=w_dtype) * 0.01
        m = torch.zeros(shape, dtype=torch.float32)
        v = torch.zeros(shape, dtype=torch.float32)

        beta1, beta2 = 0.9, 0.999
        lr, weight_decay, eps = 1e-3, 0.01, 1e-8
        step = 1

        w_out, m_out, v_out = apply_adam_w_v2_golden(
            weight, grad, m, v, beta1, beta2, lr, weight_decay, eps, step
        )

        # --- shape / dtype checks ---
        assert w_out.shape == shape, f"weight_out shape mismatch: {w_out.shape}"
        assert m_out.shape == shape and v_out.shape == shape
        assert w_out.dtype == w_dtype, f"weight_out dtype: {w_out.dtype} != {w_dtype}"
        assert m_out.dtype == torch.float32 and v_out.dtype == torch.float32

        # --- numerical sanity: no NaN / Inf ---
        assert torch.isfinite(w_out.float()).all(), "weight_out contains NaN/Inf"
        assert torch.isfinite(m_out).all(), "m_out contains NaN/Inf"
        assert torch.isfinite(v_out).all(), "v_out contains NaN/Inf"

        # --- analytic check at step=1, m=v=0 ---
        # m_new = (1-b1)*g, v_new = (1-b2)*g^2
        # m_hat = m_new / (1-b1) = g; v_hat = v_new / (1-b2) = g^2
        # update = g / (|g| + eps) + lambda * w
        # w_new = w - lr * update
        g_f32 = grad.to(torch.float32)
        w_f32 = weight.to(torch.float32)
        expected_update = g_f32 / (torch.sqrt(g_f32 * g_f32) + eps) + weight_decay * w_f32
        expected_w = (w_f32 - lr * expected_update).to(w_dtype)
        max_err = (w_out.float() - expected_w.float()).abs().max().item()
        atol = 1e-4 if w_dtype == torch.float32 else 4e-3  # bf16 round-trip
        assert max_err <= atol, (
            f"[{w_dtype}] analytic check failed: max_err={max_err} > atol={atol}"
        )

        print(f"[{w_dtype}] shape={shape} step={step} OK (max_err={max_err:.3e})")

    # --- multi-step bias correction sanity (step=10) ---
    weight = torch.randn(shape, dtype=torch.float32) * 0.02
    grad = torch.randn(shape, dtype=torch.float32) * 0.01
    m = torch.randn(shape, dtype=torch.float32) * 0.001
    v = torch.randn(shape, dtype=torch.float32).abs() * 1e-6
    w_out, m_out, v_out = apply_adam_w_v2_golden(
        weight, grad, m, v, 0.9, 0.999, 1e-3, 0.01, 1e-8, step=10
    )
    assert torch.isfinite(w_out).all() and torch.isfinite(m_out).all() and torch.isfinite(v_out).all()
    print(f"[fp32] step=10 multi-step OK (max|w|={w_out.abs().max().item():.3e})")

    print("All smoke tests passed.")


if __name__ == "__main__":
    _smoke_test()
