#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


from __future__ import annotations

import collections
from typing import Any

import torch

MAX_BLOCK_TABLE_COLS = 8

GoldenGatherArgs = collections.namedtuple(
    "GoldenGatherArgs",
    ["key_ref", "value_ref", "seq_offset", "cache_mode", "is_seq_lens_cumsum"],
)

GATHER_GOLDEN_ARG_NAMES = GoldenGatherArgs._fields
GATHER_GOLDEN_DEFAULTS = {
    "key_ref": None,
    "value_ref": None,
    "seq_offset": None,
    "cache_mode": "Norm",
    "is_seq_lens_cumsum": True,
}

CaseResultParts = collections.namedtuple(
    "CaseResultParts",
    ["key_cache", "value_cache", "block_tables", "seq_lens", "key_ref", "value_ref", "seq_offset"],
)

GatherPagesPlan = collections.namedtuple(
    "GatherPagesPlan",
    [
        "q_count",
        "lengths",
        "bases",
        "table_offsets",
        "block_tables",
        "block_size",
        "key_cache",
        "value_cache",
        "key_num_blocks",
        "key_out",
        "value_out",
    ],
)

MakeCaseParams = collections.namedtuple(
    "MakeCaseParams",
    [
        "q_count",
        "total_tokens",
        "block_table_cols",
        "num_blocks",
        "block_size",
        "key_num_heads",
        "value_num_heads",
        "key_dim",
        "value_dim",
    ],
)

CacheTablePlan = collections.namedtuple(
    "CacheTablePlan",
    [
        "num_blocks",
        "block_size",
        "key_num_heads",
        "key_dim",
        "value_num_heads",
        "value_dim",
        "total_tokens",
        "q_count",
        "lengths",
        "table_offsets",
        "block_table_cols",
        "generator",
    ],
)


def _pop_seq_lens_arg(args: tuple, kwargs: dict) -> tuple[torch.Tensor, tuple]:
    if args:
        return args[0], args[1:]
    if "seq_lens" in kwargs:
        return kwargs.pop("seq_lens"), args
    raise TypeError("missing required argument: seq_lens")


def _parse_golden_gather_args(args: tuple, kwargs: dict) -> GoldenGatherArgs:
    if len(args) > len(GATHER_GOLDEN_ARG_NAMES):
        raise TypeError("too many positional arguments")
    values = {**GATHER_GOLDEN_DEFAULTS, **dict(zip(GATHER_GOLDEN_ARG_NAMES, args))}
    values.update({name: kwargs.pop(name) for name in list(kwargs) if name in GATHER_GOLDEN_ARG_NAMES})
    if kwargs:
        raise TypeError(f"unexpected keyword argument(s): {sorted(kwargs)}")
    return GoldenGatherArgs(*(values.get(name) for name in GATHER_GOLDEN_ARG_NAMES))


def _require_cpu_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")


def _validate_cache(
    name: str,
    tensor: torch.Tensor,
) -> tuple[int, int, int, int]:
    _require_cpu_tensor(name, tensor)
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must have dtype torch.bfloat16")
    if tensor.dim() != 4:
        raise ValueError(f"{name} must have rank 4, got shape {tuple(tensor.shape)}")
    num_blocks, block_size, num_heads, head_size = tensor.shape
    if num_blocks <= 0:
        raise ValueError(f"{name} num_blocks must be positive")
    if block_size <= 0 or num_heads <= 0 or head_size <= 0:
        raise ValueError(f"{name} non-batch dimensions must be positive")
    return int(num_blocks), int(block_size), int(num_heads), int(head_size)


def _validate_index_tensor(name: str, tensor: torch.Tensor, dim: int) -> None:
    _require_cpu_tensor(name, tensor)
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if tensor.dim() != dim:
        raise ValueError(f"{name} must have rank {dim}, got shape {tuple(tensor.shape)}")


def _gather_kv_pages(plan: GatherPagesPlan) -> None:
    """Gather KV cache tokens from paged blocks into contiguous output tensors (in-place)."""
    for q_idx in range(plan.q_count):
        seq_len = int(plan.lengths[q_idx].item())
        if seq_len == 0:
            continue
        output_base = int(plan.bases[q_idx].item())
        table_offset = int(plan.table_offsets[q_idx].item())
        required_blocks = (seq_len + plan.block_size - 1) // plan.block_size
        table_end = table_offset + required_blocks
        if table_end > plan.block_tables.shape[1]:
            raise ValueError(
                f"block_tables row {q_idx} is too short: need columns up to {table_end}, "
                f"got {plan.block_tables.shape[1]}"
            )
        for logical_block in range(required_blocks):
            physical_block = int(plan.block_tables[q_idx, table_offset + logical_block].item())
            if physical_block < 0 or physical_block >= plan.key_num_blocks:
                raise ValueError(f"block id {physical_block} outside [0, {plan.key_num_blocks})")
            token_start = logical_block * plan.block_size
            valid_tokens = min(seq_len - token_start, plan.block_size)
            out_start = output_base + token_start
            out_end = out_start + valid_tokens
            plan.key_out[out_start:out_end, :, :] = plan.key_cache[physical_block, :valid_tokens, :, :]
            plan.value_out[out_start:out_end, :, :] = plan.value_cache[physical_block, :valid_tokens, :, :]


def _lengths_and_bases(
    seq_lens: torch.Tensor,
    q_count: int,
    is_seq_lens_cumsum: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    _validate_index_tensor("seq_lens", seq_lens, 1)
    if not is_seq_lens_cumsum:
        raise ValueError("only cumsum seq_lens [Q + 1] is supported")
    seq_lens_i64 = seq_lens.to(torch.int64)
    if seq_lens_i64.numel() != q_count + 1:
        raise ValueError("cumsum seq_lens must have shape [Q + 1]")
    if int(seq_lens_i64[0].item()) != 0:
        raise ValueError("cumsum seq_lens must start with 0")
    lengths = seq_lens_i64[1:] - seq_lens_i64[:-1]
    bases = seq_lens_i64[:-1]
    total_tokens = int(seq_lens_i64[-1].item())
    if bool((lengths < 0).any().item()):
        raise ValueError("seq_lens must describe non-negative lengths")
    return lengths, bases, total_tokens


def _make_golden_gather_plan(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    runtime: GoldenGatherArgs,
) -> GatherPagesPlan:
    key_num_blocks, block_size, key_num_heads, key_dim = _validate_cache("key_cache", key_cache)
    value_num_blocks, value_block_size, value_num_heads, value_dim = _validate_cache("value_cache", value_cache)
    if value_num_blocks != key_num_blocks or value_block_size != block_size:
        raise ValueError("key_cache and value_cache must share num_blocks and block_size")

    _validate_index_tensor("block_tables", block_tables, 2)
    q_count = int(block_tables.shape[0])
    if q_count <= 0:
        raise ValueError("block_tables first dimension must be positive")
    if block_tables.shape[1] <= 0:
        raise ValueError("block_tables second dimension must be positive")

    lengths, bases, total_tokens = _lengths_and_bases(seq_lens, q_count, runtime.is_seq_lens_cumsum)
    table_offsets = _make_table_offsets(runtime.seq_offset, block_size, q_count)
    key_out = _make_ref_output(runtime.key_ref, (total_tokens, key_num_heads, key_dim))
    value_out = _make_ref_output(runtime.value_ref, (total_tokens, value_num_heads, value_dim))
    return GatherPagesPlan(
        q_count, lengths, bases, table_offsets, block_tables, block_size,
        key_cache, value_cache, key_num_blocks, key_out, value_out,
    )


def _make_table_offsets(seq_offset: torch.Tensor | None, block_size: int, q_count: int) -> torch.Tensor:
    if seq_offset is None:
        return torch.zeros(q_count, dtype=torch.int64)
    _validate_index_tensor("seq_offset", seq_offset, 1)
    if seq_offset.numel() != q_count:
        raise ValueError("seq_offset must have shape [Q]")
    seq_offset_i64 = seq_offset.to(torch.int64)
    if bool((seq_offset_i64 < 0).any().item()):
        raise ValueError("seq_offset values must be non-negative")
    if bool((seq_offset_i64 % block_size != 0).any().item()):
        raise ValueError("seq_offset values must be divisible by block_size")
    return seq_offset_i64 // block_size


def _make_ref_output(ref: torch.Tensor | None, shape: tuple[int, int, int]) -> torch.Tensor:
    if ref is not None:
        return ref.clone()
    return torch.empty(shape, dtype=torch.bfloat16)


def gather_pa_kv_cache_golden(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference for the network ND GatherPaKvCache path."""
    seq_lens, args = _pop_seq_lens_arg(args, kwargs)
    runtime = _parse_golden_gather_args(args, kwargs)
    if runtime.cache_mode != "Norm":
        raise NotImplementedError("network sweep path only supports cache_mode='Norm'")
    plan = _make_golden_gather_plan(key_cache, value_cache, block_tables, seq_lens, runtime)
    _gather_kv_pages(plan)
    return plan.key_out, plan.value_out


def _split_total_tokens(total_tokens: int, q_count: int) -> list[int]:
    base = total_tokens // q_count
    rem = total_tokens % q_count
    return [base + (1 if idx < rem else 0) for idx in range(q_count)]


def _validate_make_case_params(params: MakeCaseParams) -> None:
    if params.q_count < 1:
        raise ValueError("q_count must be positive")
    if params.total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    if params.block_table_cols < 1:
        raise ValueError("block_table_cols must be positive")
    if params.num_blocks < 1:
        raise ValueError("num_blocks must be positive")
    invalid_param = (
        params.block_size < 1
        or params.key_num_heads < 1
        or params.value_num_heads < 1
        or params.key_dim < 1
        or params.value_dim < 1
    )
    if invalid_param:
        raise ValueError("cache non-batch dimensions must be positive")


def _make_seq_lens_and_offsets(
    total_tokens: int, q_count: int, block_size: int,
    seq_lens=None, seq_offset=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if seq_lens is None:
        lengths = torch.tensor(_split_total_tokens(total_tokens, q_count), dtype=torch.int32)
    else:
        lengths = torch.as_tensor(seq_lens, dtype=torch.int32, device="cpu")
    if lengths.dim() != 1 or lengths.numel() != q_count:
        raise ValueError("seq_lens must describe exactly Q sequence lengths")
    if int(lengths.sum().item()) != total_tokens:
        raise ValueError("seq_lens must sum to total_tokens")
    if bool((lengths < 0).any().item()):
        raise ValueError("seq_lens values must be non-negative")

    if seq_offset is None:
        seq_offset_tensor = torch.zeros(q_count, dtype=torch.int32)
        table_offsets = torch.zeros(q_count, dtype=torch.int64)
    else:
        seq_offset_tensor = torch.as_tensor(seq_offset, dtype=torch.int32, device="cpu")
        if seq_offset_tensor.dim() != 1 or seq_offset_tensor.numel() != q_count:
            raise ValueError("seq_offset must have shape [Q]")
        if bool((seq_offset_tensor % block_size != 0).any().item()):
            raise ValueError("seq_offset must be divisible by block_size")
        table_offsets = (seq_offset_tensor.to(torch.int64) // block_size)

    return lengths, seq_offset_tensor, table_offsets


def _make_caches_and_block_tables(
    plan: CacheTablePlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    blocks_per_seq = torch.div(
        plan.lengths.to(torch.int64) + plan.block_size - 1, plan.block_size, rounding_mode="floor")
    if int((blocks_per_seq + plan.table_offsets).max().item()) > plan.block_table_cols:
        raise ValueError("block_table_cols is too small for this case")

    key_cache = torch.randn(
        (plan.num_blocks, plan.block_size, plan.key_num_heads, plan.key_dim),
        dtype=torch.float32, generator=plan.generator).to(torch.bfloat16)
    value_cache = torch.randn(
        (plan.num_blocks, plan.block_size, plan.value_num_heads, plan.value_dim),
        dtype=torch.float32, generator=plan.generator).to(torch.bfloat16)
    key_ref = torch.empty((plan.total_tokens, plan.key_num_heads, plan.key_dim), dtype=torch.bfloat16)
    value_ref = torch.empty((plan.total_tokens, plan.value_num_heads, plan.value_dim), dtype=torch.bfloat16)

    block_tables = torch.empty((plan.q_count, plan.block_table_cols), dtype=torch.int32)
    next_block = 0
    for q_idx in range(plan.q_count):
        block_tables[q_idx, :] = 0
        used_cols = int(plan.table_offsets[q_idx].item()) + int(blocks_per_seq[q_idx].item())
        for col_idx in range(used_cols):
            block_tables[q_idx, col_idx] = next_block % plan.num_blocks
            next_block += 1

    return key_cache, value_cache, key_ref, value_ref, block_tables


def _finalize_seq_lens_tensor(lengths: torch.Tensor) -> torch.Tensor:
    seq_lens_tensor = torch.empty(len(lengths) + 1, dtype=torch.int32)
    seq_lens_tensor[0] = 0
    seq_lens_tensor[1:] = torch.cumsum(lengths, dim=0)
    return seq_lens_tensor


def _case_result_dict(
    parts: CaseResultParts,
    total_tokens: int,
    q_count: int,
    is_seq_lens_cumsum: bool,
) -> dict[str, Any]:
    return {
        "key_cache": parts.key_cache,
        "value_cache": parts.value_cache,
        "block_tables": parts.block_tables,
        "seq_lens": parts.seq_lens,
        "key_ref": parts.key_ref,
        "value_ref": parts.value_ref,
        "seq_offset": parts.seq_offset,
        "cache_mode": "Norm",
        "is_seq_lens_cumsum": is_seq_lens_cumsum,
        "total_tokens": total_tokens,
        "q_count": q_count,
    }


def _attach_gather_golden(result: dict[str, Any], compute_golden: bool) -> dict[str, Any]:
    if compute_golden:
        result["golden"] = gather_pa_kv_cache_golden(
            result["key_cache"],
            result["value_cache"],
            result["block_tables"],
            result["seq_lens"],
            result["key_ref"],
            result["value_ref"],
            result["seq_offset"],
            cache_mode="Norm",
            is_seq_lens_cumsum=result["is_seq_lens_cumsum"],
        )
    return result


def make_case(
    *,
    total_tokens: int,
    q_count: int,
    num_blocks: int = 5513,
    block_table_cols: int = MAX_BLOCK_TABLE_COLS,
    block_size: int = 128,
    key_num_heads: int = 1,
    key_dim: int = 512,
    value_num_heads: int = 1,
    value_dim: int = 64,
    seq_lens: list[int] | tuple[int, ...] | torch.Tensor | None = None,
    seq_offset: list[int] | tuple[int, ...] | torch.Tensor | None = None,
    is_seq_lens_cumsum: bool = True,
    compute_golden: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Create a CPU test case matching the network sweep ND shapes."""
    if not is_seq_lens_cumsum:
        raise ValueError("only cumsum seq_lens [Q + 1] is supported")
    _validate_make_case_params(
        MakeCaseParams(
            q_count,
            total_tokens,
            block_table_cols,
            num_blocks,
            block_size,
            key_num_heads,
            value_num_heads,
            key_dim,
            value_dim,
        )
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    lengths, seq_offset_tensor, table_offsets = _make_seq_lens_and_offsets(
        total_tokens, q_count, block_size, seq_lens, seq_offset)

    key_cache, value_cache, key_ref, value_ref, block_tables = _make_caches_and_block_tables(
        CacheTablePlan(
            num_blocks,
            block_size,
            key_num_heads,
            key_dim,
            value_num_heads,
            value_dim,
            total_tokens,
            q_count,
            lengths,
            table_offsets,
            block_table_cols,
            generator,
        )
    )

    seq_lens_tensor = _finalize_seq_lens_tensor(lengths)

    result = _case_result_dict(
        CaseResultParts(key_cache, value_cache, block_tables, seq_lens_tensor, key_ref, value_ref, seq_offset_tensor),
        total_tokens,
        q_count,
        is_seq_lens_cumsum,
    )
    return _attach_gather_golden(result, compute_golden)
