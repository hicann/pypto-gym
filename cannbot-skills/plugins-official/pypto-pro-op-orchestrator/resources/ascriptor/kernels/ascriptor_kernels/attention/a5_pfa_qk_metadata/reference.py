# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent HiFloat8 value model, the host metadata generator, and the online-attention oracle.

Every HiFloat8 value is constructed from exponent/fraction classes rather than by calling the
execution codec, so the reference cannot inherit a codec defect from the thing it checks. The
oracle reproduces the kernel's per-interval partial states and their merge rather than one dense
softmax, because what is being checked is that the host-owned key intervals compose back into the
right answer."""

import functools
import math
import torch

# ----------------------------------------------------------------------------------------------------
# hif8.py
# Independent HiFloat8 value model for attention references.
#
# Construct every value from exponent/fraction classes, rather than calling the
# execution codec. Conversion uses monotone midpoint intervals with ties away
# from zero (TA). Hybrid/SSR conversion is deliberately outside this helper.
# ----------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def value_table():
    values = [None] * 128
    values[0] = 0.0
    for code in range(1, 8):
        values[code] = math.ldexp(1.0, code - 23)
    classes = ((0, 0, 3, 0x08), (1, 1, 3, 0x10), (2, 3, 3, 0x20), (4, 7, 2, 0x40), (8, 15, 1, 0x60))
    for low, high, fraction_bits, prefix in classes:
        for exponent_sign in (1, -1) if low else (1,):
            sign_offset = (high - low + 1) * (1 << fraction_bits) if exponent_sign < 0 else 0
            for magnitude in range(low, high + 1):
                for fraction in range(1 << fraction_bits):
                    code = prefix + sign_offset + (magnitude - low) * (1 << fraction_bits) + fraction
                    value = math.ldexp(1.0 + fraction / (1 << fraction_bits), exponent_sign * magnitude)
                    values[code] = value
    values[0x6F] = math.inf
    assert all(value is not None for value in values)
    signed = values + [-value for value in values]
    signed[0x80] = math.nan
    return torch.tensor(signed, dtype=torch.float32)


@functools.lru_cache(maxsize=1)
def _intervals():
    table = value_table()[:128]
    ordered = sorted((float(value), code) for code, value in enumerate(table) if math.isfinite(value))
    levels, codes = zip(*ordered, strict=True)
    boundaries = [(low + high) / 2 for low, high in zip(levels[:-1], levels[1:], strict=True)]
    boundaries.append(40960.0)
    return torch.tensor(boundaries, dtype=torch.float64), torch.tensor((*codes, 0x6F), dtype=torch.uint8)


def decode(codes):
    if codes.dtype != torch.uint8 or codes.device.type != "cpu":
        raise ValueError("HiFloat8 reference expects CPU uint8 carrier bytes")
    return value_table()[codes.long()]


def encode(values, *, saturate=False, nan_to_zero=False):
    """Float32 TA conversion; both zero signs map to zero, NaN normally to0x80."""
    if values.device.type != "cpu":
        raise ValueError("HiFloat8 reference is CPU only")
    source = values.float()
    boundaries, codes = _intervals()
    magnitude = source.abs().double().contiguous()
    intervals = torch.bucketize(magnitude, boundaries, right=True)
    result = codes[intervals]
    if saturate:
        result = torch.where(result == 0x6F, torch.tensor(0x6E, dtype=torch.uint8), result)
    result = torch.where((source < 0) & (result != 0), result | 0x80, result)
    return torch.where(torch.isnan(source), torch.tensor(0 if nan_to_zero else 0x80, dtype=torch.uint8), result)


def quantize(values):
    return decode(encode(values))

# ----------------------------------------------------------------------------------------------------
# metadata.py
# Preserved host metadata tables and builder, independently checked for complete ownership.
# ----------------------------------------------------------------------------------------------------

TILE_M=128
ROWS_SB=64

FA_START_M = [0, 9, 18, 27, 36, 45, 54, 64, 73, 82, 91, 100, 109, 119, 128, 137, 146,
              155, 164, 173, 183, 192, 201, 210, 219, 228, 238, 247]

FA_START_N = [0, 5, 10, 15, 20, 25, 30, 2, 7, 12, 17, 22, 27, 0, 5, 10, 15, 21, 26, 32,
              5, 10, 15, 21, 26, 32, 5, 10]

FA_END_M = [9, 18, 27, 36, 45, 54, 64, 73, 82, 91, 100, 109, 119, 128, 137, 146, 155,
            164, 173, 183, 192, 201, 210, 219, 228, 238, 247, 257]

FA_END_N = [5, 10, 15, 20, 25, 30, 2, 7, 12, 17, 22, 27, 0, 5, 10, 15, 21, 26, 32, 5,
            10, 15, 21, 26, 32, 5, 10, 0]

FIA_TILES_M = 257

FIA_TILES_N = 33

def fia_intervals(tiles_n):
    """Section-17 per-core flat [start, end) intervals (m-major, s2-inner)."""
    n = len(FA_START_M)
    out = []
    for c in range(n):
        fs = FA_START_M[c] * tiles_n + FA_START_N[c]
        if c < n - 1:
            fe = FA_END_M[c] * tiles_n + FA_END_N[c]
        else:
            fe = FIA_TILES_M * tiles_n          # last core ends at the very end (bn2=1)
        out.append((fs, fe))
    return out

def even_intervals(total, num_cores):
    """Generic contiguous near-even flat split (fallback; may land on block edges)."""
    base = total // num_cores
    rem = total % num_cores
    out = []
    cur = 0
    for c in range(num_cores):
        cnt = base + (1 if c < rem else 0)
        out.append((cur, cur + cnt))
        cur += cnt
    return out

def staggered_intervals(total, num_cores, tiles_n):
    """Dev split that deliberately nudges interior boundaries OFF block edges so at least
    some m-blocks are s2-split across cores (exercises the Flash-Decode merge path)."""
    bounds = [0]
    for c in range(1, num_cores):
        b = total * c // num_cores
        if 0 < b < total and b % tiles_n == 0:        # on a block edge -> nudge to force a split
            b += 1
        bounds.append(b)
    bounds.append(total)
    return [(bounds[i], bounds[i + 1]) for i in range(num_cores)]

def even_mblock_intervals(tiles_m, tiles_n, num_cores):
    """Whole-m-block even split: each core owns WHOLE m-blocks (interval aligned to m-block edges,
    flat_start % tiles_n == 0) -> NO s2-split -> NO Flash-Decode needed. The extra m-blocks (rem) go
    to the LAST cores, so for our shape (256 full m-blocks + 1 tiny 24-row M-tail block last) the
    near-even 8/core split lands the cheap tail on the last core (~2.5% imbalance, no FD)."""
    base = tiles_m // num_cores
    rem = tiles_m % num_cores
    out = []
    cur = 0
    for c in range(num_cores):
        cnt = base + (1 if c >= num_cores - rem else 0)
        out.append((cur * tiles_n, (cur + cnt) * tiles_n))
        cur += cnt
    return out

def build_metadata(tiles_m, tiles_n, intervals, M, launch_cores=None):
    """Turn per-core flat intervals into (fa_meta, fd_meta, nws, fd_blocks).

    A block mb is *split* iff some core starts strictly inside it (flat_start % tiles_n
    != 0). FD tasks are the sorted split blocks; task j uses workspace slots 2j (head /
    lower-s2 piece, written by the core whose range ENDS at mb) and 2j+1 (tail / upper-s2
    piece, written by the core whose range STARTS at mb). The merge is symmetric in the
    two pieces, so head/tail order does not affect correctness.

    launch_cores: the number of AIC cores the board actually launches (= GetCubeNum;
    hardcoded 32 for the 950 in kernelbase). The work split (`intervals`) may use FEWER
    cores than launch_cores (e.g. FIA's section-17 28-core split on a 32-core board); the
    extra cores 28..launch_cores-1 are emitted DISABLED so they no-op the FA loop but still
    join the AIV-wide FD barrier. fa_meta has launch_cores rows, fd_meta has 2*launch_cores
    rows (lanes beyond the 2*len(intervals) work lanes / beyond the FD tasks are disabled).
    """
    num_work = len(intervals)
    if launch_cores is None:
        launch_cores = num_work
    if launch_cores < num_work:
        raise ValueError(f"launch_cores {launch_cores} < work cores {num_work}")
    split_blocks = sorted({fs // tiles_n for (fs, fe) in intervals
                           if fs < fe and fs % tiles_n != 0})
    blk_to_task = {mb: j for j, mb in enumerate(split_blocks)}
    nws = 2 * len(split_blocks)

    fa = []
    for c in range(launch_cores):
        if c >= num_work:
            fa.append([0, 0, 0, -1, -1, 0])         # launched-but-idle core
            continue
        fs, fe = intervals[c]
        if fs >= fe:
            fa.append([0, 0, 0, -1, -1, 0])
            continue
        start_m, start_n = fs // tiles_n, fs % tiles_n
        end_m_excl, end_n_excl = fe // tiles_n, fe % tiles_n
        tail_ws, head_ws = -1, -1
        if start_n != 0:                       # first block is the TAIL piece -> 2j+1
            tail_ws = 2 * blk_to_task[start_m] + 1
        if end_n_excl != 0:                    # last block is the HEAD piece -> 2j
            head_ws = 2 * blk_to_task[end_m_excl]
            if start_n != 0 and start_m == end_m_excl:
                raise ValueError(f"3-way split of block {start_m} not supported")
        fa.append([1, fs, fe, tail_ws, head_ws, 0])

    fd = []
    for aiv in range(launch_cores * 2):
        task, row0 = aiv // 2, (aiv % 2) * ROWS_SB
        if task < len(split_blocks):
            mb = split_blocks[task]
            nrows = min(ROWS_SB, M - (mb * TILE_M + row0))
            fd.append([1, mb, 2 * task, 2 * task + 1, row0, nrows] if nrows > 0
                      else [0, 0, 0, 0, 0, 0])
        else:
            fd.append([0, 0, 0, 0, 0, 0])
    return fa, fd, max(1, nws), split_blocks

# ----------------------------------------------------------------------------------------------------
# oracle.py
# Independent online arithmetic over validated host-owned key intervals.
# ----------------------------------------------------------------------------------------------------

def bf16_away(value):
    bits = value.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return ((bits + 0x8000) >> 16).to(torch.int16).view(torch.bfloat16)


def physical(inputs, *, all_hif8):
    m, n = inputs["M"], inputs["N"]
    q, k = decode(inputs["q"]), decode(inputs["k"])
    v = decode(inputs["v"]) if all_hif8 else inputs["v"].float()
    nt = (n + 127) // 128
    groups = [[] for _ in range((m + 127) // 128)]
    for first, stop in inputs["intervals"]:
        for tile in range(first // nt, (stop + nt - 1) // nt):
            begin, end = max(first, tile * nt), min(stop, (tile + 1) * nt)
            if begin < end:
                groups[tile].append((begin % nt, end - tile * nt))
    output = torch.empty((m, 128), dtype=torch.bfloat16)
    for tile, owners in enumerate(groups):
        rows = slice(tile * 128, min((tile + 1) * 128, m))
        query = q[rows]
        partials = []
        for first, stop in owners:
            maximum = torch.full((query.shape[0], 1), -1.0e30)
            denominator = torch.zeros_like(maximum)
            numerator = torch.zeros_like(query)
            for key in range(first, stop):
                cols = slice(key * 128, min((key + 1) * 128, n))
                score = (query @ k[cols].T) * (128 ** -0.5)
                new_max = torch.maximum(maximum, score.amax(dim=-1, keepdim=True))
                weight = torch.exp(maximum - new_max)
                probability = torch.exp(score - (new_max - math.log(16) if all_hif8 else new_max))
                stored = quantize(probability) if all_hif8 else probability.bfloat16().float()
                denominator = denominator * weight + probability.sum(dim=-1, keepdim=True)
                pv = stored @ v[cols]
                numerator = (numerator.double() * weight.double() + pv.double()).float() if all_hif8 else numerator * weight + pv
                maximum = new_max
            partials.append((maximum, denominator, numerator))
        if len(partials) == 1:
            output[rows] = (partials[0][2] / partials[0][1]).bfloat16()
        else:
            (m0, s0, a0), (m1, s1, a1) = partials
            weight = torch.exp(m0 - m1)
            output[rows] = bf16_away((a0 * weight + a1) / (s0 * weight + s1))
    return output


# ----------------------------------------------------------------------------------------------------
# the case surface
# ----------------------------------------------------------------------------------------------------

# Only Q and K are HiFloat8 in this unit: V, the published probabilities and the accumulation
# stay on the bf16/fp32 path, so the oracle takes the bf16 branch of `physical`. Its
# all-HiFloat8 sibling, attention.a5_pfa_hif8_metadata, sets this True.
ALL_HIF8 = False


def schedule(parameters, cores):
    """The host half of this kernel: choose the per-core flat intervals the case names, then turn
    them into the int32 fa_meta/fd_meta tensors the kernel reads. The device decides nothing about
    the grid, so a wrong table here is a wrong answer rather than a slow one.

    The intervals have to partition [0, tiles_m*tiles_n) contiguously and leave one or two
    complete owners per query tile, because the merge reads exactly two workspace slots per split
    block; and `build_metadata` must not produce more than the kernel's NWS_GLOBAL=64 slots, which
    bounds it at 32 launch cores. `fia` is the recorded 28-core cost-aware split from a real run
    and describes exactly one shape (M=32792, N=4099, tiles_n=33); no case here selects it.
    """
    m, n, work = (parameters[name] for name in ("M", "N", "work_cores"))
    mt, nt = (m + 127) // 128, (n + 127) // 128
    choice = parameters["schedule"]
    if choice == "staggered":
        intervals = staggered_intervals(mt * nt, work, nt)
    elif choice == "even":
        intervals = even_intervals(mt * nt, work)
    elif choice == "whole_query":
        intervals = even_mblock_intervals(mt, nt, work)
    elif choice == "fia":
        intervals = fia_intervals(nt)
    else:
        raise ValueError(f"unknown metadata schedule {choice!r}")
    fa, fd, _slots, _splits = build_metadata(mt, nt, intervals, m, launch_cores=cores)
    return (torch.tensor(fa, dtype=torch.int32).flatten(),
            torch.tensor(fd, dtype=torch.int32).flatten(), intervals)


def make_inputs(case):
    """Deterministic Q/K/V, plus the two metadata tensors -- as much an input as the operands are.
    `block_dim` is how many cores are launched and `work_cores` how many are given an interval, so
    a case with work_cores < block_dim launches idle cores that still join the vector-wide barrier.
    The intervals travel on in the returned dict because the oracle reproduces them partial by
    partial rather than computing one dense softmax."""
    p = case["parameters"]
    cores = case["block_dim"]
    fa, fd, intervals = schedule(p, cores)
    generator = torch.Generator().manual_seed(case["seed"])
    result = {**p, "block_dim": cores, "intervals": intervals, "fa_meta": fa, "fd_meta": fd}
    for name in ("q", "k", "v"):
        value = torch.randn((p["M"] if name == "q" else p["N"], 128), generator=generator)
        result[name] = value.bfloat16() if name == "v" and not ALL_HIF8 else encode(value)
    return result


def reference(inputs):
    return {"out": physical(inputs, all_hif8=ALL_HIF8)}
