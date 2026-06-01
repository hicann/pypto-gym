from __future__ import annotations

from typing import Any

import torch


MAX_BLOCK_TABLE_COLS = 8


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


def _lengths_and_bases(
    seq_lens: torch.Tensor,
    q_count: int,
    is_seq_lens_cumsum: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    _validate_index_tensor("seq_lens", seq_lens, 1)
    seq_lens_i64 = seq_lens.to(torch.int64)
    if is_seq_lens_cumsum:
        if seq_lens_i64.numel() != q_count + 1:
            raise ValueError("cumsum seq_lens must have shape [Q + 1]")
        if int(seq_lens_i64[0].item()) != 0:
            raise ValueError("cumsum seq_lens must start with 0")
        lengths = seq_lens_i64[1:] - seq_lens_i64[:-1]
        bases = seq_lens_i64[:-1]
        total_tokens = int(seq_lens_i64[-1].item())
    else:
        if seq_lens_i64.numel() != q_count:
            raise ValueError("seq_lens must have shape [Q]")
        lengths = seq_lens_i64
        bases = torch.empty(q_count, dtype=torch.int64)
        bases[0] = 0
        if q_count > 1:
            bases[1:] = torch.cumsum(lengths[:-1], dim=0)
        total_tokens = int(lengths.sum().item())
    if bool((lengths < 0).any().item()):
        raise ValueError("seq_lens must describe non-negative lengths")
    return lengths, bases, total_tokens


def gather_pa_kv_cache_golden(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    key_ref: torch.Tensor | None = None,
    value_ref: torch.Tensor | None = None,
    seq_offset: torch.Tensor | None = None,
    *,
    cache_mode: str = "Norm",
    is_seq_lens_cumsum: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU reference for the network ND GatherPaKvCache path.

    ND layout:
      key_cache   [num_blocks, block_size, key_heads, key_dim]
      value_cache [num_blocks, block_size, value_heads, value_dim]
      key_ref     [total_tokens, key_heads, key_dim]
      value_ref   [total_tokens, value_heads, value_dim]
    """

    if cache_mode != "Norm":
        raise NotImplementedError("network sweep path only supports cache_mode='Norm'")

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

    lengths, bases, total_tokens = _lengths_and_bases(seq_lens, q_count, is_seq_lens_cumsum)

    if seq_offset is None:
        table_offsets = torch.zeros(q_count, dtype=torch.int64)
    else:
        _validate_index_tensor("seq_offset", seq_offset, 1)
        if seq_offset.numel() != q_count:
            raise ValueError("seq_offset must have shape [Q]")
        seq_offset_i64 = seq_offset.to(torch.int64)
        if bool((seq_offset_i64 < 0).any().item()):
            raise ValueError("seq_offset values must be non-negative")
        if bool((seq_offset_i64 % block_size != 0).any().item()):
            raise ValueError("seq_offset values must be divisible by block_size")
        table_offsets = seq_offset_i64 // block_size

    key_out = (
        key_ref.clone()
        if key_ref is not None
        else torch.empty((total_tokens, key_num_heads, key_dim), dtype=torch.bfloat16)
    )
    value_out = (
        value_ref.clone()
        if value_ref is not None
        else torch.empty((total_tokens, value_num_heads, value_dim), dtype=torch.bfloat16)
    )

    for q_idx in range(q_count):
        seq_len = int(lengths[q_idx].item())
        if seq_len == 0:
            continue
        output_base = int(bases[q_idx].item())
        table_offset = int(table_offsets[q_idx].item())
        required_blocks = (seq_len + block_size - 1) // block_size
        table_end = table_offset + required_blocks
        if table_end > block_tables.shape[1]:
            raise ValueError(
                f"block_tables row {q_idx} is too short: need columns up to {table_end}, "
                f"got {block_tables.shape[1]}"
            )
        for logical_block in range(required_blocks):
            physical_block = int(block_tables[q_idx, table_offset + logical_block].item())
            if physical_block < 0 or physical_block >= key_num_blocks:
                raise ValueError(f"block id {physical_block} outside [0, {key_num_blocks})")
            token_start = logical_block * block_size
            valid_tokens = min(seq_len - token_start, block_size)
            out_start = output_base + token_start
            out_end = out_start + valid_tokens
            key_out[out_start:out_end, :, :] = key_cache[physical_block, :valid_tokens, :, :]
            value_out[out_start:out_end, :, :] = value_cache[physical_block, :valid_tokens, :, :]

    return key_out, value_out


def _split_total_tokens(total_tokens: int, q_count: int) -> list[int]:
    base = total_tokens // q_count
    rem = total_tokens % q_count
    return [base + (1 if idx < rem else 0) for idx in range(q_count)]


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
    is_seq_lens_cumsum: bool = False,
    compute_golden: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Create a CPU test case matching the network sweep ND shapes."""

    if q_count < 1:
        raise ValueError("q_count must be positive")
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    if block_table_cols < 1:
        raise ValueError("block_table_cols must be positive")
    if num_blocks < 1:
        raise ValueError("num_blocks must be positive")
    if block_size < 1 or key_num_heads < 1 or value_num_heads < 1 or key_dim < 1 or value_dim < 1:
        raise ValueError("cache non-batch dimensions must be positive")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

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

    blocks_per_seq = torch.div(
        lengths.to(torch.int64) + block_size - 1,
        block_size,
        rounding_mode="floor",
    )
    if int((blocks_per_seq + table_offsets).max().item()) > block_table_cols:
        raise ValueError("block_table_cols is too small for this case")

    key_cache = torch.randn(
        (num_blocks, block_size, key_num_heads, key_dim),
        dtype=torch.float32,
        generator=generator,
    ).to(torch.bfloat16)
    value_cache = torch.randn(
        (num_blocks, block_size, value_num_heads, value_dim),
        dtype=torch.float32,
        generator=generator,
    ).to(torch.bfloat16)
    key_ref = torch.empty((total_tokens, key_num_heads, key_dim), dtype=torch.bfloat16)
    value_ref = torch.empty((total_tokens, value_num_heads, value_dim), dtype=torch.bfloat16)

    block_tables = torch.empty((q_count, block_table_cols), dtype=torch.int32)
    next_block = 0
    for q_idx in range(q_count):
        block_tables[q_idx, :] = 0
        used_cols = int(table_offsets[q_idx].item()) + int(blocks_per_seq[q_idx].item())
        for col_idx in range(used_cols):
            block_tables[q_idx, col_idx] = next_block % num_blocks
            next_block += 1

    if is_seq_lens_cumsum:
        seq_lens_tensor = torch.empty(q_count + 1, dtype=torch.int32)
        seq_lens_tensor[0] = 0
        seq_lens_tensor[1:] = torch.cumsum(lengths, dim=0)
    else:
        seq_lens_tensor = lengths.clone()

    result = {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_tables": block_tables,
        "seq_lens": seq_lens_tensor,
        "key_ref": key_ref,
        "value_ref": value_ref,
        "seq_offset": seq_offset_tensor,
        "cache_mode": "Norm",
        "is_seq_lens_cumsum": is_seq_lens_cumsum,
        "total_tokens": total_tokens,
        "q_count": q_count,
    }
    if compute_golden:
        result["golden"] = gather_pa_kv_cache_golden(
            key_cache,
            value_cache,
            block_tables,
            seq_lens_tensor,
            key_ref,
            value_ref,
            seq_offset_tensor,
            cache_mode="Norm",
            is_seq_lens_cumsum=is_seq_lens_cumsum,
        )
    return result
