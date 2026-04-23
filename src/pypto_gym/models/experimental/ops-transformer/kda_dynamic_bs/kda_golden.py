#!/usr/bin/env python3
# coding: utf-8

"""PyTorch golden for KDA with dynamic B/S axes.

This file exports `kda_golden()` and can run standalone self-check:
    python3 kda_golden.py
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
) -> None:
    if query.ndim != 3:
        raise ValueError(f"query must be 3D [B,S,D], got {query.shape}")
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("key/value shape must match query shape")
    if alpha.shape != query.shape or beta.shape != query.shape:
        raise ValueError("alpha/beta shape must match query shape")
    if query.dtype != torch.float32:
        raise ValueError(f"query dtype must be float32, got {query.dtype}")
    if key.dtype != torch.float32 or value.dtype != torch.float32:
        raise ValueError("key/value dtype must be float32")
    if alpha.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("alpha/beta dtype must be float32")
    if initial_state is not None:
        b, _, d = query.shape
        expected = (b, d, d)
        if initial_state.shape != expected:
            raise ValueError(f"initial_state shape must be {expected}, got {initial_state.shape}")
        if initial_state.dtype != torch.float32:
            raise ValueError("initial_state dtype must be float32")


def kda_golden(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """KDA reference implementation.

    Recurrence:
        state = state * alpha[:, None] + (k[:, None] * v[None, :]) * beta[:, None]
        out   = sum(state * q[:, None], dim=0)

    Args:
        query/key/value/alpha/beta: [B,S,D], float32
        initial_state: optional [B,D,D], float32

    Returns:
        output: [B,S,D], float32
        last_state: [B,D,D], float32
    """
    _validate_inputs(query, key, value, alpha, beta, initial_state)

    b, s, d = query.shape
    output = torch.empty_like(query)
    if initial_state is None:
        state_all = torch.zeros((b, d, d), dtype=torch.float32, device=query.device)
    else:
        state_all = initial_state.clone()

    for b_idx in range(b):
        state = state_all[b_idx]
        for s_idx in range(s):
            q = query[b_idx, s_idx]  # [D]
            k = key[b_idx, s_idx]  # [D]
            v = value[b_idx, s_idx]  # [D]
            a = alpha[b_idx, s_idx]  # [D]
            be = beta[b_idx, s_idx]  # [D]

            outer = k.unsqueeze(1) * v.unsqueeze(0)  # [D,D]
            state = state * a.unsqueeze(1) + outer * be.unsqueeze(1)
            out_row = (state * q.unsqueeze(1)).sum(dim=0)  # [D]
            output[b_idx, s_idx] = out_row
        state_all[b_idx] = state

    return output, state_all


def _validate() -> None:
    torch.manual_seed(0)
    test_cases = [
        (1, 16, 64),
        (2, 31, 64),
        (4, 127, 64),
    ]
    for b, s, d in test_cases:
        q = torch.randn(b, s, d, dtype=torch.float32)
        k = torch.randn(b, s, d, dtype=torch.float32)
        v = torch.randn(b, s, d, dtype=torch.float32)
        alpha = torch.sigmoid(torch.randn(b, s, d, dtype=torch.float32))
        beta = torch.sigmoid(torch.randn(b, s, d, dtype=torch.float32))

        out, last = kda_golden(q, k, v, alpha, beta)
        if out.shape != (b, s, d):
            raise AssertionError(f"output shape mismatch: {out.shape}")
        if last.shape != (b, d, d):
            raise AssertionError(f"last_state shape mismatch: {last.shape}")
        if torch.isnan(out).any() or torch.isnan(last).any():
            raise AssertionError("NaN detected in golden output")
        print(f"[OK] B={b}, S={s}, D={d}, out={tuple(out.shape)}, state={tuple(last.shape)}")

    print("kda_golden validation passed")


if __name__ == "__main__":
    _validate()
