# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent Torch reference for the chunked ungated Delta Rule forward: the staged
decomposition each of the four launches answers to, at the precision boundaries they use."""

import builtins
from typing import Optional, Tuple

import torch

# ----------------------------------------------------------------------------------------------------
# inverse.py
# Independent block-triangular reference; FP32 until the caller's BF16 boundary.
#
# Adapted from projects/a5/gdn_fwd/kernels/tril_inverse64.py; pure Torch, no DSL imports.
# ----------------------------------------------------------------------------------------------------

L = 64
D = 128
SIZE = 64
BLOCK = 16
NUM_BLOCKS = 4
ATOL = 2e-3
RTOL = 2e-3

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
# reference.py
# Independent Delta forward references, named precision variants and generated cases.
# ----------------------------------------------------------------------------------------------------

OUTPUTS = {"default": ("core_attn_out", "final_state"),
           "saved_state": ("core_attn_out", "final_state", "state_after_history", "v_new_history")}
STAGE_OUTPUTS = ("preprocess_attn", "wu_attn_bf16", "value_wu", "k_cumdecay", "standalone_scores")


def make_inputs(case: dict) -> dict:
    p = case["parameters"]
    shape = (p["B"], p["H"], p["C"], L, D)
    rng = torch.Generator().manual_seed(case["seed"])
    values = {name: (torch.randn(shape, generator=rng) * p.get("input_scale", 0.05)).bfloat16()
              for name in ("query", "key", "value")}
    values["beta"] = torch.rand(shape[:-1], generator=rng)
    values["scale"] = p.get("query_scale", 1.0)
    values["variant"] = p.get("variant", "default")
    values["inverse_variant"] = p.get("inverse_variant", "block")
    return values


def preprocess(key: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    pair_scores = key.float() @ key.float().transpose(-1, -2)
    return -(pair_scores * beta.unsqueeze(-1)) * torch.tril(torch.ones(L, L), diagonal=-1)


def recompute(key: torch.Tensor, value: torch.Tensor, beta: torch.Tensor, inverse: torch.Tensor) -> tuple:
    key_beta = (key.float() * beta.unsqueeze(-1)).bfloat16()
    value_beta = (value.float() * beta.unsqueeze(-1)).bfloat16()
    return ((inverse.float() @ value_beta.float()).bfloat16(),
            (inverse.float() @ key_beta.float()).bfloat16())


def scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    return ((query.float() @ key.float().transpose(-1, -2)) * torch.tril(torch.ones(L, L))).bfloat16()


def recurrence(query: torch.Tensor, key: torch.Tensor, value_wu: torch.Tensor, key_wu: torch.Tensor) -> dict:
    B, H, C = query.shape[:3]
    state = torch.zeros(B, H, D, D, dtype=torch.bfloat16)
    output, states, v_news = [], [], []
    for c in range(C):
        q, k = query[:, :, c].float(), key[:, :, c].float()
        attn = scores(query[:, :, c], key[:, :, c])
        v_new = (value_wu[:, :, c].float() - key_wu[:, :, c].float() @ state.float()).bfloat16()
        output.append((q @ state.float() + attn.float() @ v_new.float()).bfloat16())
        state = (state.float() + k.transpose(-1, -2) @ v_new.float()).bfloat16()
        states.append(state)
        v_news.append(v_new)
    return {"core_attn_out": torch.stack(output, dim=2), "final_state": state,
            "state_after_history": torch.stack(states, dim=2), "v_new_history": torch.stack(v_news, dim=2)}


def prepare(inputs: dict) -> dict:
    """Every checkpoint the four launches hand each other, at the precision they hand it at.

    The query scaling happens here, rounded back to bf16, because the kernels are handed an
    already-scaled `q`: folding the scale into the fp32 cube operand instead would produce a
    reference the launches cannot match.
    """
    q = (inputs["query"] * inputs["scale"]).bfloat16()
    a = preprocess(inputs["key"], inputs["beta"])
    inverse = inverse_from_strict_lower_a(a).bfloat16()
    value_wu, key_wu = recompute(inputs["key"], inputs["value"], inputs["beta"], inverse)
    return {"preprocess_attn": a, "wu_attn_bf16": inverse, "value_wu": value_wu, "k_cumdecay": key_wu,
            "standalone_scores": scores(q, inputs["key"]), **recurrence(q, inputs["key"], value_wu, key_wu)}


def reference(inputs: dict) -> dict:
    """The public outputs of whichever variant this case names.

    The default forward publishes two. The `saved_state` variant publishes the same two plus
    the post-update state and `v_new` histories that delta_rule_bwd consumes, from the same
    four launches — only the fourth kernel changes.
    """
    result = prepare(inputs)
    return {name: result[name] for name in OUTPUTS[inputs["variant"]]}


def reference_stages(inputs: dict) -> dict:
    return prepare(inputs)
