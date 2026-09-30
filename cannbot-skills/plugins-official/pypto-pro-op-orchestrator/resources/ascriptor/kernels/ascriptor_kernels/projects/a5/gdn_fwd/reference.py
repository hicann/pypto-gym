# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch reference for the GDN forward-v2: the staged formulas each of the six
kernels answers to, at the precision boundaries the kernels actually use."""

import builtins
import torch

# ----------------------------------------------------------------------------------------------------
# preprocess.py
# Independent Torch stage formula extracted from gdn_preprocess.py; no DSL imports.
# ----------------------------------------------------------------------------------------------------

def _bf16_cube_matmul_fp32_acc(left, right):
    if left.dtype != torch.bfloat16 or right.dtype != torch.bfloat16:
        raise TypeError("bf16 cube matmul reference expects bfloat16 inputs")
    return left.float() @ right.float().transpose(-1, -2)


def gdn_preprocess_v2(key, beta, g, triu):
    """Reference GDN preprocess v2.

    Shape legend:
    B: batch
    H: head count
    C: chunk_size
    L: length_per_chunk
    D: head dimension
    """
    B, H, C, length_per_chunk, head_dim = key.shape
    scalar_shape = (B, H, C, length_per_chunk)
    matrix_shape = (length_per_chunk, length_per_chunk)

    if beta.shape != scalar_shape:
        raise ValueError(f"beta shape must be {scalar_shape}, got {tuple(beta.shape)}")
    if g.shape != scalar_shape:
        raise ValueError(f"g shape must be {scalar_shape}, got {tuple(g.shape)}")
    if triu.shape != matrix_shape:
        raise ValueError(f"triu shape must be {matrix_shape}, got {tuple(triu.shape)}")
    if key.dtype != torch.bfloat16:
        raise TypeError("key must be torch.bfloat16")
    if beta.dtype != torch.float32 or g.dtype != torch.float32 or triu.dtype != torch.float32:
        raise TypeError("beta/g/triu must be torch.float32")

    # Row-vector form: g @ triu is prefix cumsum when triu is upper-triangular ones.
    g_cumsum = (g.reshape(-1, length_per_chunk) @ triu).reshape(B, H, C, length_per_chunk)

    # A5 bf16 cube matmul accumulates to fp32 L0C; keep beta out of the cube input.
    attn_scores = _bf16_cube_matmul_fp32_acc(key, key)

    lower_eq = triu.t()
    strict_lower = torch.tril(torch.ones_like(triu), diagonal=-1)
    decay_mask = (g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2)).exp() * lower_eq
    attn_scores = attn_scores * beta.unsqueeze(-1)
    attn = -(attn_scores * decay_mask * strict_lower)

    return g_cumsum, decay_mask, attn

# ----------------------------------------------------------------------------------------------------
# inverse.py
# Independent Torch stage formula extracted from tril_inverse64.py; no DSL imports.
# ----------------------------------------------------------------------------------------------------

SIZE = 64
BLOCK = 16


def _block(mat: torch.Tensor, row: int, col: int, block: int = BLOCK) -> torch.Tensor:
    r0 = row * block
    c0 = col * block
    return mat[..., r0:r0 + block, c0:c0 + block]


def block_inverse_i_minus_a(mat: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Invert a 64x64 unit lower-triangular matrix with 16x16 blocks.

    The input is expected to be I - A, where A is strictly lower triangular.
    Batched leading dimensions are supported.
    """
    if mat.shape[-2:] != (SIZE, SIZE):
        raise ValueError(f"expected last two dimensions to be {(SIZE, SIZE)}, got {mat.shape[-2:]}")
    if SIZE % block != 0:
        raise ValueError(f"block size {block} must divide matrix size {SIZE}")
    num_blocks = SIZE // block

    eye = torch.eye(block, dtype=mat.dtype, device=mat.device)
    eye = eye.expand(mat.shape[:-2] + (block, block))
    zero = torch.zeros_like(_block(mat, 0, 0, block))

    inv_blocks = [[zero.clone() for _ in builtins.range(num_blocks)] for _ in builtins.range(num_blocks)]
    diag_inv = []
    for i in builtins.range(num_blocks):
        inv_diag = torch.linalg.solve_triangular(_block(mat, i, i, block), eye, upper=False)
        diag_inv.append(inv_diag)
        inv_blocks[i][i] = inv_diag

    for i in builtins.range(num_blocks):
        for j in builtins.range(0, i):
            acc = torch.zeros_like(zero)
            for k in builtins.range(j, i):
                a_ik = -_block(mat, i, k, block)
                acc = acc + a_ik @ inv_blocks[k][j]
            inv_blocks[i][j] = diag_inv[i] @ acc

    rows = [torch.cat(inv_blocks[i], dim=-1) for i in builtins.range(num_blocks)]
    return torch.cat(rows, dim=-2)


def inverse_from_strict_lower_a(a: torch.Tensor) -> torch.Tensor:
    """Build I - A from a strict lower-triangular A and invert it by blocks."""
    if a.shape[-2:] != (SIZE, SIZE):
        raise ValueError(f"expected last two dimensions to be {(SIZE, SIZE)}, got {a.shape[-2:]}")

    eye = torch.eye(SIZE, dtype=a.dtype, device=a.device)
    mat = eye.expand(a.shape[:-2] + (SIZE, SIZE)) - torch.tril(a, diagonal=-1)
    return block_inverse_i_minus_a(mat)

# ----------------------------------------------------------------------------------------------------
# recompute.py
# Independent Torch stage formula extracted from gdn_recompute_wu.py; no DSL imports.
# ----------------------------------------------------------------------------------------------------

def gdn_recompute_wu(key, value, beta, g, attn):
    v_beta = (value.float() * beta.unsqueeze(-1)).to(torch.bfloat16)
    k_beta_g = (key.float() * beta.unsqueeze(-1) * g.exp().unsqueeze(-1)).to(torch.bfloat16)

    value = (attn.float() @ v_beta.float()).bfloat16()
    k_cumdecay = (attn.float() @ k_beta_g.float()).bfloat16()

    return value, k_cumdecay

# ----------------------------------------------------------------------------------------------------
# recurrence.py
# Candidate-local expected outputs for sub-kernel checks. Retained production precision ABI.
#
# Migration source: gdn_fwd/refs/oracle.py
#
# Each expected_* helper computes the oracle's local slice for one sub-kernel
# following the DAG in plan.md.  The expected outputs are derived directly
# from the shared oracle formula without importing it, so each sub can be
# checked in isolation.
# ----------------------------------------------------------------------------------------------------

def expected_sub1(q_i, k_i, decay_mask_i):
    """Oracle attn from M1 alone.

    Accepts a single chunk [B,H,L,D] or full C input [B,H,C,L,D]. The
    sub1->sub2 GM boundary is bf16 in the authored execution path.
    """
    return (q_i.float() @ k_i.float().transpose(-1, -2) * decay_mask_i).bfloat16()


def expected_sub2(attn, q_i, k_i, v_i, k_cumdecay_i, g_i, last_recurrent_state):
    """Oracle fused M2-M5 output over the full C loop."""
    B, H, C, L, D = q_i.shape
    core_attn_out = torch.empty(B, H, C, L, D, dtype=torch.bfloat16, device=q_i.device)
    state = torch.zeros(B, H, D, D, dtype=torch.bfloat16, device=q_i.device)

    for i in range(C):
        q = q_i[:, :, i].float()
        k = k_i[:, :, i].float()
        v = v_i[:, :, i].float()
        k_cumdecay = k_cumdecay_i[:, :, i].float()
        g = g_i[:, :, i].float()

        v_prime = k_cumdecay @ state.float()
        v_new = (v - v_prime).bfloat16()
        attn_inter = (q @ state.float()) * g[:, :, :, None].exp()
        core_attn_out[:, :, i] = (attn_inter + attn[:, :, i].float() @ v_new.float()).bfloat16()

        g_last = g[:, :, -1]
        k_weighted = k * (g_last[:, :, None] - g).exp()[..., None]
        state = (
            state.float() * g_last[:, :, None, None].exp()
            + k_weighted.transpose(-1, -2).bfloat16().float() @ v_new.float()
        ).bfloat16()

    return core_attn_out, state


def expected_sub2_state_history(attn, q_i, k_i, v_i, k_cumdecay_i, g_i, last_recurrent_state):
    """Sub2 oracle that also returns post-update state and saved bwd inputs."""
    B, H, C, L, D = q_i.shape
    core_attn_out = torch.empty(B, H, C, L, D, dtype=torch.bfloat16, device=q_i.device)
    state_after_history = torch.empty(B, H, C, D, D, dtype=torch.bfloat16, device=q_i.device)
    v_new_history = torch.empty(B, H, C, L, D, dtype=torch.bfloat16, device=q_i.device)
    k_weighted_history = torch.empty(B, H, C, L, D, dtype=torch.bfloat16, device=q_i.device)
    exp_delta_history = torch.empty(B, H, C, L, dtype=torch.float32, device=q_i.device)
    state = torch.zeros(B, H, D, D, dtype=torch.bfloat16, device=q_i.device)

    for i in range(C):
        q = q_i[:, :, i].float()
        k = k_i[:, :, i].float()
        v = v_i[:, :, i].float()
        k_cumdecay = k_cumdecay_i[:, :, i].float()
        g = g_i[:, :, i].float()

        v_prime = k_cumdecay @ state.float()
        v_new = (v - v_prime).bfloat16()
        v_new_history[:, :, i] = v_new
        attn_inter = (q @ state.float()) * g[:, :, :, None].exp()
        core_attn_out[:, :, i] = (attn_inter + attn[:, :, i].float() @ v_new.float()).bfloat16()

        g_last = g[:, :, -1]
        exp_delta = (g_last[:, :, None] - g).exp()
        exp_delta_history[:, :, i] = exp_delta
        k_weighted = k * exp_delta[..., None]
        k_weighted_history[:, :, i] = k_weighted.bfloat16()
        state = (
            state.float() * g_last[:, :, None, None].exp()
            + k_weighted.transpose(-1, -2).bfloat16().float() @ v_new.float()
        ).bfloat16()
        state_after_history[:, :, i] = state

    return core_attn_out, state, state_after_history, v_new_history, k_weighted_history, exp_delta_history


# ----------------------------------------------------------------------------------------------------
# reference.py
# Independent generated reference for the current forward-v2 precision ABI.
# ----------------------------------------------------------------------------------------------------

OUTPUTS = ("output", "final_state", "state_after_history", "v_new_history", "k_weighted_history", "exp_delta_history")


def make_inputs(case):
    params = case["parameters"]
    b, h, c = (int(params[name]) for name in ("B", "H", "C"))
    generator = torch.Generator().manual_seed(case["seed"])
    shape = (b, h, c, 64, 128)
    scalar = shape[:-1]
    values = {name: (torch.randn(shape, generator=generator) * params.get("scale", 0.05)).bfloat16() for name in ("query", "key", "value")}
    values["beta"] = torch.rand(scalar, generator=generator)
    if params.get("gate", "logsigmoid") == "slow":
        values["g"] = -torch.rand(scalar, generator=generator) * 0.03
    else:
        values["g"] = torch.nn.functional.logsigmoid(torch.randn(scalar, generator=generator))
    return values


def intermediates(inputs):
    """Every checkpoint the six kernels hand each other, at the precision they hand it at.

    The recurrent tail appears twice on purpose. `plain.*` is the output-only kernel and
    `saved.*` is the one that additionally publishes the four backward-facing histories; they
    run the same recurrence, so a disagreement between them is the saved kernel's extra stores
    corrupting its own state rather than an error in the maths.

    Two invariants hold exactly, not to a tolerance: `state_after_history[:, :, -1]` is
    `final_state`, because the history is post-update, and `exp_delta_history[..., -1]` is 1,
    because the last lane's decay is `exp(g_last - g_last)`.
    """
    q, k, v, beta, g = (inputs[name] for name in ("query", "key", "value", "beta", "g"))
    gc, decay, strict = gdn_preprocess_v2(k, beta, g, torch.triu(torch.ones(64, 64)))
    inverse = inverse_from_strict_lower_a(strict).bfloat16()
    value_wu, key_decay = gdn_recompute_wu(k, v, beta, gc, inverse)
    attention = expected_sub1(q, k, decay)
    initial = torch.zeros(*q.shape[:2], 128, 128, dtype=torch.bfloat16)
    args = (attention, q, k, value_wu, key_decay, gc, initial)
    plain = expected_sub2(*args)
    saved = expected_sub2_state_history(*args)
    return {"preprocess.g_cumsum": gc, "preprocess.decay_mask": decay, "preprocess.strict_lower": strict,
            "inverse.wu_attn": inverse, "recompute.value_wu": value_wu, "recompute.k_cumdecay": key_decay,
            "scores.attention": attention, "plain.output": plain[0], "plain.final_state": plain[1],
            **{"saved." + name: tensor for name, tensor in zip(OUTPUTS, saved)}}


def reference(inputs):
    """The six public outputs: the `saved.*` half of the staged decomposition.

    This forward publishes its saved state as a public output rather than as a debugging
    extra, because gdn_bwd consumes exactly these four histories.
    """
    stages = intermediates(inputs)
    return {name: stages["saved." + name] for name in OUTPUTS}


def reference_stages(inputs):
    return intermediates(inputs)
