#!/usr/bin/env python3
# coding: utf-8

"""PyPTO implementation for KDA with dynamic B/S axes.

Exports:
  - kda_wrapper(query, key, value, alpha, beta, return_state=False)
"""

import functools
from typing import Tuple, Union

import pypto
import torch


def _check_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
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
    if not (query.is_contiguous() and key.is_contiguous() and value.is_contiguous() and alpha.is_contiguous() and beta.is_contiguous()):
        raise ValueError("all inputs must be contiguous")
    if not (query.device == key.device == value.device == alpha.device == beta.device):
        raise ValueError("all inputs must be on the same device")


@functools.lru_cache(maxsize=16)
def _build_kda_kernel(d: int):
    if d <= 0:
        raise ValueError(f"D must be > 0, got {d}")
    cube_tile = 16 if d >= 16 else d
    t_dyn = pypto.DYNAMIC
    b1_dyn = pypto.DYNAMIC
    state_dyn = pypto.DYNAMIC

    @pypto.frontend.jit(runtime_options={"stitch_function_max_num": 128})
    def kda_kernel(
        query_2d: pypto.Tensor([t_dyn, d], pypto.DT_FP32),
        key_2d: pypto.Tensor([t_dyn, d], pypto.DT_FP32),
        value_2d: pypto.Tensor([t_dyn, d], pypto.DT_FP32),
        alpha_2d: pypto.Tensor([t_dyn, d], pypto.DT_FP32),
        beta_2d: pypto.Tensor([t_dyn, d], pypto.DT_FP32),
        seq_offsets: pypto.Tensor([b1_dyn], pypto.DT_INT32),
        output_2d: pypto.Tensor([t_dyn, d], pypto.DT_FP32),
        last_state_flat: pypto.Tensor([state_dyn, d], pypto.DT_FP32),
    ):
        b_size = seq_offsets.shape[0] - 1

        for b_idx in pypto.loop(b_size, name="LOOP_B_KDA", idx_name="b_idx"):
            seq_start = seq_offsets[b_idx]
            seq_end = seq_offsets[b_idx + 1]
            seq_len = seq_end - seq_start

            pypto.set_vec_tile_shapes(d, d)
            state = pypto.full(size=[d, d], fill_value=0.0, dtype=pypto.DT_FP32)

            for s_idx in pypto.loop(seq_len, name="LOOP_S_KDA", idx_name="s_idx", unroll_list=[16, 1]):
                seq_ofs = seq_start + s_idx

                pypto.set_vec_tile_shapes(1, d)
                q_row = pypto.view(query_2d, [1, d], [seq_ofs, 0], valid_shape=[1, d])
                k_row = pypto.view(key_2d, [1, d], [seq_ofs, 0], valid_shape=[1, d])
                v_row = pypto.view(value_2d, [1, d], [seq_ofs, 0], valid_shape=[1, d])
                a_row = pypto.view(alpha_2d, [1, d], [seq_ofs, 0], valid_shape=[1, d])
                b_row = pypto.view(beta_2d, [1, d], [seq_ofs, 0], valid_shape=[1, d])

                pypto.set_vec_tile_shapes(d, d)
                k_col = pypto.reshape(k_row, [d, 1], valid_shape=[d, 1])
                a_col = pypto.reshape(a_row, [d, 1], valid_shape=[d, 1])
                b_col = pypto.reshape(b_row, [d, 1], valid_shape=[d, 1])

                k_expand = pypto.expand_clone(k_col, [d, d])
                v_expand = pypto.expand_clone(v_row, [d, d])
                outer = k_expand * v_expand

                a_expand = pypto.expand_clone(a_col, [d, d])
                b_expand = pypto.expand_clone(b_col, [d, d])
                state = state * a_expand + outer * b_expand

                pypto.set_cube_tile_shapes(
                    [cube_tile, cube_tile],
                    [cube_tile, cube_tile],
                    [cube_tile, cube_tile],
                )
                out_row = pypto.matmul(q_row, state, pypto.DT_FP32)  # [1, D]
                pypto.set_vec_tile_shapes(1, d)
                pypto.assemble(out_row, [seq_ofs, 0], output_2d)

            pypto.set_vec_tile_shapes(d, d)
            pypto.assemble(state, [b_idx * d, 0], last_state_flat)

    return kda_kernel


def kda_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    return_state: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """KDA wrapper for PyPTO kernel invocation.

    Args:
        query/key/value/alpha/beta: [B,S,D], float32, contiguous
        return_state: if True, return `(output, last_state)`

    Returns:
        output or (output, last_state)
    """
    _check_inputs(query, key, value, alpha, beta)
    b, s, d = query.shape
    total_tokens = b * s

    query_2d = query.reshape(total_tokens, d).contiguous()
    key_2d = key.reshape(total_tokens, d).contiguous()
    value_2d = value.reshape(total_tokens, d).contiguous()
    alpha_2d = alpha.reshape(total_tokens, d).contiguous()
    beta_2d = beta.reshape(total_tokens, d).contiguous()
    seq_offsets = torch.arange(0, (b + 1) * s, s, dtype=torch.int32, device=query.device)

    output_2d = torch.empty_like(query_2d)
    last_state_flat = torch.empty((b * d, d), dtype=query.dtype, device=query.device)

    kernel = _build_kda_kernel(int(d))
    kernel(query_2d, key_2d, value_2d, alpha_2d, beta_2d, seq_offsets, output_2d, last_state_flat)

    output = output_2d.reshape(b, s, d)
    last_state = last_state_flat.reshape(b, d, d)

    if return_state:
        return output, last_state
    return output
