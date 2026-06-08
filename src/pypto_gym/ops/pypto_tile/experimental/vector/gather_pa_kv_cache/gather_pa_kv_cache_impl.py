#!/usr/bin/env python3
# coding: utf-8

import torch
import pypto


INT32_MAX = 2**31 - 1


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "device_sched_mode": 1,
        "stitch_function_max_num": 128,
        "ready_on_host_tensors": ["block_tables", "seq_lens", "seq_offset"],
        "valid_shape_optimize": 1,
    },
    pass_options={"vec_nbuffer_setting": {-2: 1, -1: 8}},
    debug_options={"runtime_debug_mode": 1},
)
def _gather_pa_kv_cache_nd_kernel_npu(
    key_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    value_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    block_tables: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_INT32),
    seq_lens: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    key_ref: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    value_ref: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    seq_offset: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
):
    block_size = key_cache.shape[1]
    num_blocks = key_cache.shape[0]
    key_num_heads = key_cache.shape[2]
    key_dim = key_cache.shape[3]
    value_num_heads = value_cache.shape[2]
    value_dim = value_cache.shape[3]
    q_count = block_tables.shape[0]
    key_cache_3d = pypto.reshape(
        key_cache,
        [num_blocks * block_size, key_num_heads, key_dim],
        inplace=True,
    )
    value_cache_3d = pypto.reshape(
        value_cache,
        [num_blocks * block_size, value_num_heads, value_dim],
        inplace=True,
    )

    for q_base, q_unroll in pypto.loop_unroll(
        0, q_count, 1, name="gather_q_loop", idx_name="q_idx", unroll_list=[2, 1]
    ):
        for q_inner in range(q_unroll):
            q_idx = q_base + q_inner
            out_start = seq_lens[q_idx]
            out_start.as_variable()
            seq_len = seq_lens[q_idx + 1] - out_start
            seq_len.as_variable()
            table_offset = seq_offset[q_idx] // block_size
            block_count = pypto.ceildiv(seq_len, block_size)

            for block_idx in pypto.loop(block_count, name="gather_block_loop", idx_name="block_idx", unroll_list=[1]):
                physical_block = block_tables[q_idx, table_offset + block_idx]
                token_offset = block_idx * block_size
                valid_tokens = (seq_len - token_offset).min(block_size)
                out_offset = out_start + token_offset
                cache_offset = physical_block * block_size

                pypto.set_vec_tile_shapes(16, key_num_heads, key_dim)
                key_tile = pypto.view(
                    key_cache_3d,
                    [block_size, key_num_heads, key_dim],
                    [cache_offset, 0, 0],
                    valid_shape=[valid_tokens, key_num_heads, key_dim],
                )
                pypto.assemble(
                    key_tile,
                    [out_offset, 0, 0],
                    key_ref,
                )

                pypto.set_vec_tile_shapes(32, value_num_heads, value_dim)
                value_tile = pypto.view(
                    value_cache_3d,
                    [block_size, value_num_heads, value_dim],
                    [cache_offset, 0, 0],
                    valid_shape=[valid_tokens, value_num_heads, value_dim],
                )
                pypto.assemble(
                    value_tile,
                    [out_offset, 0, 0],
                    value_ref,
                )


@pypto.frontend.jit(
    runtime_options={
        "run_mode": pypto.RunMode.NPU,
        "device_sched_mode": 1,
        "stitch_function_max_num": 128,
        "ready_on_host_tensors": ["block_tables", "seq_lens", "seq_offset"],
        "valid_shape_optimize": 1,
    },
    pass_options={"vec_nbuffer_setting": {-2: 1, -1: 8}},
    debug_options={"runtime_debug_mode": 1},
)
def _gather_pa_kv_cache_nd_large_token_kernel_npu(
    key_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    value_cache: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    block_tables: pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_INT32),
    seq_lens: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    key_ref: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    value_ref: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    seq_offset: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
):
    block_size = key_cache.shape[1]
    num_blocks = key_cache.shape[0]
    key_num_heads = key_cache.shape[2]
    key_dim = key_cache.shape[3]
    value_num_heads = value_cache.shape[2]
    value_dim = value_cache.shape[3]
    q_count = block_tables.shape[0]
    key_cache_3d = pypto.reshape(
        key_cache,
        [num_blocks * block_size, key_num_heads, key_dim],
        inplace=True,
    )
    value_cache_3d = pypto.reshape(
        value_cache,
        [num_blocks * block_size, value_num_heads, value_dim],
        inplace=True,
    )

    for q_base, q_unroll in pypto.loop_unroll(
        0, q_count, 1, name="gather_q_loop_large", idx_name="q_idx", unroll_list=[2, 1]
    ):
        for q_inner in range(q_unroll):
            q_idx = q_base + q_inner
            out_start = seq_lens[q_idx]
            out_start.as_variable()
            seq_len = seq_lens[q_idx + 1] - out_start
            seq_len.as_variable()
            table_offset = seq_offset[q_idx] // block_size
            block_count = pypto.ceildiv(seq_len, block_size)

            for block_idx in pypto.loop(
    block_count,
    name="gather_block_loop_large",
    idx_name="block_idx",
     unroll_list=[1]):
                physical_block = block_tables[q_idx, table_offset + block_idx]
                token_offset = block_idx * block_size
                valid_tokens = (seq_len - token_offset).min(block_size)
                out_offset = out_start + token_offset
                cache_offset = physical_block * block_size

                pypto.set_vec_tile_shapes(8, key_num_heads, key_dim)
                key_tile = pypto.view(
                    key_cache_3d,
                    [block_size, key_num_heads, key_dim],
                    [cache_offset, 0, 0],
                    valid_shape=[valid_tokens, key_num_heads, key_dim],
                )
                pypto.assemble(
                    key_tile,
                    [out_offset, 0, 0],
                    key_ref,
                )

                pypto.set_vec_tile_shapes(8, value_num_heads, value_dim)
                value_tile = pypto.view(
                    value_cache_3d,
                    [block_size, value_num_heads, value_dim],
                    [cache_offset, 0, 0],
                    valid_shape=[valid_tokens, value_num_heads, value_dim],
                )
                pypto.assemble(
                    value_tile,
                    [out_offset, 0, 0],
                    value_ref,
                )


def _require_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _validate_cache_pair(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> tuple[int, int, int, int, int, int, torch.device]:
    _require_tensor("key_cache", key_cache)
    _require_tensor("value_cache", value_cache)
    if key_cache.dtype != torch.bfloat16:
        raise TypeError("key_cache must have dtype torch.bfloat16")
    if value_cache.dtype != torch.bfloat16:
        raise TypeError("value_cache must have dtype torch.bfloat16")
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        raise ValueError("key_cache/value_cache must have rank 4")
    if key_cache.device != value_cache.device:
        raise ValueError("key_cache and value_cache must be on the same device")

    key_num_blocks, key_block_size, key_num_heads, key_dim = key_cache.shape
    value_num_blocks, value_block_size, value_num_heads, value_dim = value_cache.shape
    if key_num_blocks != value_num_blocks or key_block_size != value_block_size:
        raise ValueError(
            "key_cache and value_cache must share num_blocks and block_size"
        )
    if key_num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if key_block_size <= 0:
        raise ValueError("block_size must be positive")
    if key_num_heads <= 0 or value_num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if key_dim <= 0 or value_dim <= 0:
        raise ValueError("head dimensions must be positive")
    return (
        int(key_num_blocks),
        int(key_block_size),
        int(key_num_heads),
        int(key_dim),
        int(value_num_heads),
        int(value_dim),
        key_cache.device,
    )


def _validate_index_tensor(name: str, tensor: torch.Tensor, dim: int) -> None:  # pylint: disable=huawei-too-many-arguments
    _require_tensor(name, tensor)
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if tensor.dim() != dim:
        raise ValueError(f"{name} must have rank {dim}, got {tuple(tensor.shape)}")


def _build_seq_lens_cumsum(
    seq_lens: torch.Tensor,
    q_count: int,
    is_seq_lens_cumsum: bool,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    _validate_index_tensor("seq_lens", seq_lens, 1)
    seq_lens_cpu = seq_lens.detach().cpu().contiguous().to(torch.int64)
    if is_seq_lens_cumsum:
        if seq_lens_cpu.numel() != q_count + 1:
            raise ValueError("seq_lens must have shape [Q + 1] when is_seq_lens_cumsum=True")
        if int(seq_lens_cpu[0].item()) != 0:
            raise ValueError("cumsum seq_lens must start with 0")
        lengths = seq_lens_cpu[1:] - seq_lens_cpu[:-1]
        seq_lens_cumsum = seq_lens_cpu
    else:
        if seq_lens_cpu.numel() != q_count:
            raise ValueError("seq_lens must have shape [Q]")
        lengths = seq_lens_cpu
        seq_lens_cumsum = torch.empty(q_count + 1, dtype=torch.int64)
        seq_lens_cumsum[0] = 0
        seq_lens_cumsum[1:] = torch.cumsum(lengths, dim=0)
    if bool((lengths < 0).any().item()):
        raise ValueError("seq_lens must describe non-negative lengths")
    total_tokens = int(seq_lens_cumsum[-1].item())
    if total_tokens < 0 or total_tokens > INT32_MAX:
        raise ValueError("total token count must fit in int32")
    return seq_lens_cumsum.to(torch.int32), lengths, total_tokens


def _get_seq_lens_meta(
    seq_lens: torch.Tensor,
    q_count: int,
    is_seq_lens_cumsum: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    seq_lens_cumsum, lengths, total_tokens = _build_seq_lens_cumsum(
        seq_lens,
        q_count,
        is_seq_lens_cumsum,
    )
    return seq_lens_cumsum.to(device).contiguous(), lengths, total_tokens


def _prepare_outputs_and_validate(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    key_ref: torch.Tensor | None,
    value_ref: torch.Tensor | None,
    seq_offset: torch.Tensor,
    cache_mode: str,
    is_seq_lens_cumsum: bool,
    run_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if cache_mode != "Norm":
        raise NotImplementedError("network sweep path only supports cache_mode='Norm'")
    if run_mode != "npu":
        raise ValueError("only run_mode='npu' is supported")

    (
        num_blocks,
        block_size,
        key_num_heads,
        key_dim,
        value_num_heads,
        value_dim,
        device,
    ) = _validate_cache_pair(key_cache, value_cache)
    _validate_index_tensor("block_tables", block_tables, 2)
    _validate_index_tensor("seq_offset", seq_offset, 1)
    q_count = int(block_tables.shape[0])
    if q_count <= 0:
        raise ValueError("block_tables first dimension must be positive")
    if block_tables.shape[1] <= 0:
        raise ValueError("block_tables second dimension must be positive")
    if seq_offset.numel() != q_count:
        raise ValueError("seq_offset must have shape [Q]")

    seq_lens_cumsum, lengths, total_tokens = _get_seq_lens_meta(
        seq_lens,
        q_count,
        is_seq_lens_cumsum,
        device,
    )
    expected_key_shape = (total_tokens, key_num_heads, key_dim)
    expected_value_shape = (total_tokens, value_num_heads, value_dim)
    if key_ref is None:
        key_ref = torch.empty(expected_key_shape, dtype=torch.bfloat16, device=device)
    elif key_ref.shape != expected_key_shape or key_ref.dtype != torch.bfloat16 or key_ref.device != device:
        raise ValueError(f"key_ref must have shape {expected_key_shape}, dtype bf16, and device {device}")
    if value_ref is None:
        value_ref = torch.empty(expected_value_shape, dtype=torch.bfloat16, device=device)
    elif value_ref.shape != expected_value_shape or value_ref.dtype != torch.bfloat16 or value_ref.device != device:
        raise ValueError(f"value_ref must have shape {expected_value_shape}, dtype bf16, and device {device}")

    seq_offset_cpu = seq_offset.detach().cpu().contiguous().to(torch.int64)
    if bool((seq_offset_cpu < 0).any().item()):
        raise ValueError("seq_offset values must be non-negative")
    if bool((seq_offset_cpu % block_size != 0).any().item()):
        raise ValueError("seq_offset values must be divisible by block_size")

    block_tables_cpu = block_tables.detach().cpu().contiguous().to(torch.int64)
    table_offsets = seq_offset_cpu // block_size
    for q_idx in range(q_count):
        required_blocks = int((int(lengths[q_idx].item()) + block_size - 1) // block_size)
        table_end = int(table_offsets[q_idx].item()) + required_blocks
        if table_end > block_tables.shape[1]:
            raise ValueError(f"block_tables row {q_idx} is too short")
        if table_end > 0:
            used_blocks = block_tables_cpu[q_idx, int(table_offsets[q_idx].item()):table_end]
            if bool((used_blocks < 0).any().item()) or bool((used_blocks >= num_blocks).any().item()):
                raise ValueError(f"block_tables row {q_idx} contains invalid block ids")

    return key_ref, value_ref, seq_lens_cumsum, total_tokens


def gather_pa_kv_cache_out(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    key_ref: torch.Tensor,
    value_ref: torch.Tensor,
    seq_offset: torch.Tensor,
    *,
    cache_mode: str = "Norm",
    is_seq_lens_cumsum: bool = False,
    run_mode: str = "npu",
) -> tuple[torch.Tensor, torch.Tensor]:
    (
        _,
        _,
        key_num_heads,
        key_dim,
        value_num_heads,
        value_dim,
        _,
    ) = _validate_cache_pair(key_cache, value_cache)
    key_ref, value_ref, seq_lens_cumsum, total_tokens = _prepare_outputs_and_validate(
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        key_ref,
        value_ref,
        seq_offset,
        cache_mode,
        is_seq_lens_cumsum,
        run_mode,
    )
    if total_tokens == 0:
        return key_ref, value_ref

    kernel = _gather_pa_kv_cache_nd_kernel_npu
    if key_num_heads * key_dim > 4096 or value_num_heads * value_dim > 4096:
        kernel = _gather_pa_kv_cache_nd_large_token_kernel_npu

    kernel(
        key_cache.contiguous(),
        value_cache.contiguous(),
        block_tables.contiguous(),
        seq_lens_cumsum,
        key_ref,
        value_ref,
        seq_offset.contiguous(),
    )
    return key_ref, value_ref


def gather_pa_kv_cache_wrapper(
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
    run_mode: str = "npu",
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, _, _, _, _, device = _validate_cache_pair(key_cache, value_cache)
    if seq_offset is None:
        q_count = int(block_tables.shape[0])
        seq_offset = torch.zeros((q_count,), dtype=torch.int32, device=device)
    key_ref, value_ref, _, total_tokens = _prepare_outputs_and_validate(
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        key_ref,
        value_ref,
        seq_offset,
        cache_mode,
        is_seq_lens_cumsum,
        run_mode,
    )
    if total_tokens == 0:
        return key_ref, value_ref
    return gather_pa_kv_cache_out(
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        key_ref,
        value_ref,
        seq_offset,
        cache_mode=cache_mode,
        is_seq_lens_cumsum=is_seq_lens_cumsum,
        run_mode=run_mode,
    )
