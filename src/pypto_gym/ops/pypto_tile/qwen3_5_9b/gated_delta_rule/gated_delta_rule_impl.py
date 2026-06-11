# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
#
# SPDX-License-Identifier: Apache-2.0
"""gated_delta_rule — chunked multi-head linear-attention kernel for Qwen3.5-9B.

Drop-in PyPTO implementation of the chunk_gated_delta_rule entry point used by
``Qwen3_5GatedDeltaNet`` during prefill. The wrapper mirrors the upstream
``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` signature.

Algorithmic outline (per chunk of L=128 rows, looped over S in steps of L):

  1. L2-normalize q, k.
  2. Build the per-chunk decay mask ``D = exp(g_cum - g_cum^T) * lower_tri``.
  3. ``A0 = -(k_β @ k_n^T * D) * strict_lower_tri``;
     ``A = (I - A0)^-1`` via truncated power series (8 terms).
  4. ``v_out = A @ (v * β)``, ``kcd = A @ (k_β * exp(g_cum))``.
  5. Recurrent state carry:
       ``v_new   = v_out - kcd @ state``
       ``attn   = q_scaled @ k_n^T * D``
       ``out    = (q_scaled * exp(g_cum)) @ state + attn @ v_new``
       ``state' = state * exp(g_last) + k_decay^T @ v_new``

Tile geometry:
  * Chunk size L = 128 (one chunk = one cube `[128,128,128]` matmul).
  * Vector tile shapes set to ``(L, D)``.
  * Cube tile shapes set to ``([128,128], [128,128], [128,128])``.

Scope (caller-enforced via wrapper assertions):
  * B = 1, Nv = 32, head_dim D = 128
  * ``initial_state is None`` (prefill only — decode path goes through the
    upstream recurrent kernel)
  * ``use_qk_l2norm_in_kernel = True``

Out-of-scope shapes raise ``NotImplementedError``. The modeling-layer hook
falls back to the upstream chunk function in that case.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch._dynamo import allow_in_graph

import pypto


_L = 128
_D = 128
_NV = 32
_EPS = 1e-6


_HELPERS: dict = {}


def _get_helpers(device, L: int):
    """Per-device cache of the strict-lower-triangular and identity helpers."""
    key = (str(device), L)
    if key not in _HELPERS:
        lower_strict_LL = torch.ones([L, L], dtype=torch.float32, device=device).tril(diagonal=-1)
        eye_LL = torch.eye(L, dtype=torch.float32, device=device)
        _HELPERS[key] = {
            "lower_strict_LL": lower_strict_LL,
            "eye_LL": eye_LL,
        }
    return _HELPERS[key]


_BN = _NV         # B=1, Nv=32  →  BN = 32 (constant, enforced by wrapper)
_BND = _BN * _D   # 32 * 128 = 4096 (constant)
_INV_SQRT_D = 1.0 / (_D ** 0.5)


@pypto.frontend.jit(runtime_options={
    "stitch_function_max_num": 2,
    "device_sched_parallelism": 8,
})
def _gdr_kernel(
    query:           pypto.Tensor([pypto.DYNAMIC, _D], pypto.DT_BF16),
    key:             pypto.Tensor([pypto.DYNAMIC, _D], pypto.DT_BF16),
    value:           pypto.Tensor([pypto.DYNAMIC, _D], pypto.DT_BF16),
    beta:            pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_BF16),
    g_cum_in:        pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_FP32),
    lower_strict_LL: pypto.Tensor([_L, _L], pypto.DT_FP32),
    eye_LL:          pypto.Tensor([_L, _L], pypto.DT_FP32),
    core_attn_out:   pypto.Tensor([pypto.DYNAMIC, _D], pypto.DT_BF16),
    last_state_data: pypto.Tensor([_BND, _D], pypto.DT_FP32),
):
    L = _L
    D = _D
    BN = _BN
    eps = _EPS
    inv_sqrt_D = _INV_SQRT_D
    # BNL is dynamic; S_pad = BNL // BN derived at runtime.
    BNL = query.shape[0]
    S_pad = BNL // BN

    pypto.set_vec_tile_shapes(L, D)
    pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

    for bn_idx in pypto.loop(BN, name="bn_loop", idx_name="bn_idx", parallel=True):
        q_off_head = bn_idx * S_pad
        s_off_head = bn_idx * D

        state = pypto.full([D, D], 0.0, dtype=pypto.DT_FP32)

        for s_idx in pypto.loop(0, S_pad, L, name="s_loop", idx_name="s_idx", unroll_list=[4, 1]):
            q_off = q_off_head + s_idx

            q_bf16 = pypto.view(query, [L, D], [q_off, 0])
            k_bf16 = pypto.view(key, [L, D], [q_off, 0])
            v_bf16 = pypto.view(value, [L, D], [q_off, 0])
            b_bf16 = pypto.view(beta, [L, 1], [q_off, 0])
            gc = pypto.view(g_cum_in, [L, 1], [q_off, 0]) + 0.0

            q = pypto.cast(q_bf16, pypto.DT_FP32)
            k = pypto.cast(k_bf16, pypto.DT_FP32)
            v = pypto.cast(v_bf16, pypto.DT_FP32)
            bf = pypto.cast(b_bf16, pypto.DT_FP32)

            q_sq = pypto.mul(q, q)
            q_ss = pypto.sum(q_sq, dim=-1, keepdim=True)
            q_se = pypto.add(q_ss, eps)
            q_inv = pypto.rsqrt(q_se)
            q_n = pypto.mul(q, q_inv) + 0.0

            k_sq = pypto.mul(k, k)
            k_ss = pypto.sum(k_sq, dim=-1, keepdim=True)
            k_se = pypto.add(k_ss, eps)
            k_inv = pypto.rsqrt(k_se)
            k_n = pypto.mul(k, k_inv) + 0.0

            q_s = pypto.mul(q_n, inv_sqrt_D)
            v_b = pypto.mul(v, bf)
            k_b = pypto.mul(k_n, bf)

            gc_T = pypto.transpose(gc, 0, 1)
            gc_diff = pypto.sub(gc, gc_T)
            lower_incl_LL = pypto.add(eye_LL, lower_strict_LL)
            gc_masked = pypto.mul(gc_diff, lower_incl_LL)
            decay_pre = pypto.exp(gc_masked)
            decay_mask = pypto.mul(decay_pre, lower_incl_LL)

            kkt = pypto.matmul(k_b, k_n, pypto.DT_FP32, b_trans=True)
            A0_pre = pypto.mul(kkt, decay_mask)
            A0_neg = pypto.mul(A0_pre, -1.0)
            A0 = pypto.mul(A0_neg, lower_strict_LL)

            # (I - A0)^-1 = I + A0 + A0^2 + ... ; A0 strict-lower-tri => series
            # converges; K=8 passes bf16-realistic gate at L=128.
            acc = pypto.add(eye_LL, A0)
            Ak = A0
            K_TERMS = 8
            for _ in range(2, K_TERMS + 1):
                Ak = pypto.matmul(Ak, A0, pypto.DT_FP32)
                acc = pypto.add(acc, Ak)
            A = acc

            v_out = pypto.matmul(A, v_b, pypto.DT_FP32)

            gc_exp = pypto.exp(gc)
            k_b_gexp = pypto.mul(k_b, gc_exp)
            kcd = pypto.matmul(A, k_b_gexp, pypto.DT_FP32)

            qkt = pypto.matmul(q_s, k_n, pypto.DT_FP32, b_trans=True)
            attn_chunk = pypto.mul(qkt, decay_mask)

            v_prime = pypto.matmul(kcd, state, pypto.DT_FP32)
            v_new = pypto.sub(v_out, v_prime)
            q_s_gexp = pypto.mul(q_s, gc_exp)
            attn_inter = pypto.matmul(q_s_gexp, state, pypto.DT_FP32)
            attn_v = pypto.matmul(attn_chunk, v_new, pypto.DT_FP32)
            core_out = pypto.add(attn_inter, attn_v)

            # Reuse gc_exp[L-1] for the per-chunk state decay (saves one exp).
            state_decay = pypto.view(gc_exp, [1, 1], [L - 1, 0]) + 0.0
            g_last = pypto.view(gc, [1, 1], [L - 1, 0]) + 0.0
            g_last_minus_gc = pypto.sub(g_last, gc)
            k_decay = pypto.mul(k_n, pypto.exp(g_last_minus_gc))
            state_scaled = pypto.mul(state, state_decay)
            kdv = pypto.matmul(k_decay, v_new, pypto.DT_FP32, a_trans=True)
            new_state = pypto.add(state_scaled, kdv)
            state[:] = new_state

            core_out_bf16 = pypto.cast(core_out, pypto.DT_BF16)
            pypto.assemble(core_out_bf16, [q_off, 0], core_attn_out)

        pypto.assemble(state, [s_off_head, 0], last_state_data)


@allow_in_graph
def gated_delta_rule_wrapper(
    query: torch.Tensor,
    key:   torch.Tensor,
    value: torch.Tensor,
    *,
    g:     torch.Tensor,
    beta:  torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunk gated delta rule fused kernel for Qwen3.5-9B prefill.

    Mirrors the upstream ``chunk_gated_delta_rule`` signature so it can be
    swapped in by the modeling layer when ``USE_PTO_GATED_DELTA_RULE`` is set.

    Constraints (raises ``NotImplementedError`` otherwise):
      * batch size B = 1
      * value head count Nv = 32
      * head dimension D = 128
      * ``initial_state is None`` (prefill only)
      * ``use_qk_l2norm_in_kernel`` is True
    """
    B, S, Nv, D = query.shape
    L = _L

    unsupported_config = (
        (initial_state is not None)
        or (D != _D)
        or (Nv != _NV)
        or (not use_qk_l2norm_in_kernel)
        or (B != 1)
    )
    if unsupported_config:
        raise NotImplementedError(
            "gated_delta_rule_wrapper requires B=1, Nv=32, D=128, "
            "initial_state=None, use_qk_l2norm_in_kernel=True; "
            f"got B={B}, Nv={Nv}, D={D}, "
            f"initial_state={'set' if initial_state is not None else 'None'}, "
            f"use_qk_l2norm_in_kernel={use_qk_l2norm_in_kernel}"
        )

    pad = (L - S % L) % L
    if pad:
        query = F.pad(query, (0, 0, 0, 0, 0, pad))
        key = F.pad(key, (0, 0, 0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, pad))
    S_pad = S + pad

    BN = B * Nv
    BNL = BN * S_pad
    BND = BN * D

    query_perm = query.to(torch.bfloat16).permute(0, 2, 1, 3).contiguous()
    key_perm = key  .to(torch.bfloat16).permute(0, 2, 1, 3).contiguous()
    value_perm = value.to(torch.bfloat16).permute(0, 2, 1, 3).contiguous()
    beta_perm = beta .to(torch.bfloat16).permute(0, 2, 1).contiguous()
    g_perm = g    .to(torch.float32).permute(0, 2, 1).contiguous()

    g_cum_perm = g_perm.cumsum(dim=-1)

    query_2d = query_perm.view(BNL, D)
    key_2d = key_perm  .view(BNL, D)
    value_2d = value_perm.view(BNL, D)
    beta_2d = beta_perm .view(BNL, 1)
    g_cum_2d = g_cum_perm.view(BNL, 1)

    device = query.device
    helpers = _get_helpers(device, L)

    core_out_2d = torch.empty([BNL, D], dtype=torch.bfloat16, device=device)
    state_2d = torch.empty([BND, D], dtype=torch.float32, device=device)

    _gdr_kernel(query_2d, key_2d, value_2d, beta_2d, g_cum_2d,
                helpers["lower_strict_LL"], helpers["eye_LL"],
                core_out_2d, state_2d)

    core_attn_out = core_out_2d.view(B, Nv, S_pad, D).permute(0, 2, 1, 3).contiguous()
    last_state_data = state_2d.view(B, Nv, D, D).contiguous()

    if pad:
        core_attn_out = core_attn_out[:, :S, :, :].contiguous()

    return core_attn_out, last_state_data if output_final_state else None


__all__ = ["gated_delta_rule_wrapper"]
