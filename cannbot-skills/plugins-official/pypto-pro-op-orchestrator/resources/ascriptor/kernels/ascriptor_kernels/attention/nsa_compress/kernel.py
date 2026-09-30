# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""NsaCompress for A5: a sliding-window weighted sum along the KV sequence.

Ported from CANN ops-transformer ef85c0b5, attention/nsa_compress, which targets
A2/A3 only and declares Ascend 950PR unsupported.  The arithmetic is retained
exactly; the implementation is re-derived rather than transliterated, because
what the upstream kernel spends most of its code on is specific to that part:

  Retained   out[i, n, :] = sum_j input[start_i + j, n, :] * weight[j, n],
             accumulated in fp32 and rounded once with CAST_ROUND.
  Re-derived Upstream is KERNEL_TYPE_AIV_ONLY with a 2*ceil(L/d)-1 slot overlap
             state machine (CompressState/SampleHeadState plus a 427-line
             sequence manager) that streams KV in BlocksNums-row chunks so that
             overlapping windows read HBM once.  Every case its own UT exercises
             has compressBlockSize == compressStride, where the windows do not
             overlap and that machinery buys nothing.  Here each output token
             owns its window outright, which needs no state at all.
  Re-derived Upstream broadcasts the weight to an 8-lane block and relies on
             Mul(src1BlkStride=0, src1RepStride=0) to reuse it across the head.
             A5 splats a UB cell across a whole register directly, so the weight
             is expanded once per core into wb and the inner loop is a plain
             full-width multiply-accumulate.

Host work is confined to what the upstream host tiling already does: it reads
actSeqLen to derive kvStartTokenIdx and PerCoreStartOutputOffset, and this port
derives the same window starts there instead.  The device never sees the int64
prefix sum.  See the task's authoring contract for the agreed boundary.
"""

from ascriptor.a5 import *

LANES = 64          # f32 lanes in one A5 vector register

# The DSL dtype and the GM annotation for each supported storage dtype.
DTYPES = {"fp16": (DT.half, f16), "bf16": (DT.bfloat16, bf16)}

# Leave room below the 256 KiB UB for the allocator and for anything the
# lowering adds; the upstream tiling reserves against the same ceiling.
UB_BUDGET = 192 * 1024


def column_tile(L, N, D, element_bytes, budget=UB_BUDGET):
    """The widest column group that fits one UB residency, as `(width, heads)`.

    A group is either whole heads (`heads >= 1`, `width = heads*D`) or a slice of
    one head (`heads == 0`, `width` a divisor of `D`). Whole heads are preferred
    because the window then arrives in one transfer per token; the sub-head form
    exists because the largest corner of the declared domain -- L = 128 with
    D = 256 -- does not fit even one whole head, and refusing it would leave a
    hole in a domain the constraints admit.

    Either way every group is the SAME width, and that width is a compile-time
    constant: a ragged group would make the transfer a partial write into a wider
    tile, and stacking a run-time column extent on a tile window has no
    pypto_pro spelling. This is the same question upstream answers in
    nsa_compress_tiling_general.cpp when it checks NeedUBMemory against ubSize.
    """
    for heads in range(N, 0, -1):
        if N % heads == 0 and _group_bytes(L, heads * D, element_bytes) <= budget:
            return heads * D, heads
    for parts in range(2, D // 16 + 1):
        width = D // parts
        if D % parts == 0 and width >= 16 and _group_bytes(L, width, element_bytes) <= budget:
            return width, 0
    raise ValueError(
        f"no column group fits in {budget} bytes: L={L}, D={D}; the narrowest "
        f"tried was 16 columns at {_group_bytes(L, 16, element_bytes)} bytes"
    )


def _geometry(L, width):
    """Tile shapes for one head group of `width` columns.

    A whole-register access is LANES wide however few columns are live, and
    M10-088 makes that footprint real even under a predicate, so every tile has
    to own what its last chunk reaches.

    The two read-only tiles absorb it in spare ROWS: reading past the end of one
    row lands in the next, which is either written later or is spare. The staged
    products cannot do that -- the folds read and write them in place, so an
    overrunning store would clobber a row that is still live -- so that tile is
    given a LANES-aligned row pitch instead and never crosses a row at all. It
    takes part in no transfer, so widening it costs nothing but bytes.
    """
    chunks = (width + LANES - 1) // LANES
    stage = chunks * LANES                       # products row pitch, LANES-aligned
    spare = max(1, (LANES + width - 1) // width)  # rows an overrunning read needs
    return chunks, stage, spare


def _group_bytes(L, width, element_bytes):
    """The double-buffered window pair, the weights, the products and the output
    pair for one column group, with their overrun backing."""
    chunks, stage, spare = _geometry(L, width)
    return (2 * (L + spare) * width * element_bytes   # xa + xb, the two windows
            + (L + spare) * width * 4                # wb, the expanded weights
            + _staged_rows(L) * stage * 4            # p_ub, the staged products
            + 2 * (1 + spare) * width * element_bytes)  # oa + ob


def fold_plan(L):
    """The same reduction as `reduction_order`, as `(stride, count)` folds.

    Each entry means `p[i] += p[i + stride]` for `i < count`, in order, which is
    what upstream ReduceBlock does to its staged products.  Both fields are
    Python integers, so the loop over this static container unrolls and each
    inner loop over `count` stays a small device loop.

    reference.py states the same order as an expression tree, which reads more
    directly; the demo's bitwise comparison is what checks the two against each
    other.  The tree is not what the kernel emits, because walking it
    depth-first unrolls into straight-line code that bisheng spills -- measured
    at L = 32: 35 vector slots, 8992 bytes against a 6144-byte VF stack.
    """
    if L < 2:
        raise ValueError(f"L={L} must be at least 2")
    align = 1
    while align < L:
        align <<= 1
    align //= 2
    plan = [(align, L - align)]
    while align > 1:
        align >>= 1
        plan.append((align, align))
    return plan


def _staged_rows(L):
    """Rows the staged products need: the first fold happens as they are formed,
    so only `align` of the L terms are ever written down."""
    return fold_plan(L)[0][0]


def make_kernel(dtype, T, TC, L, N, D, block_dim, heads_per_tile=None):
    """Build the kernel for one concrete shape.

    Every dimension is baked in as a Python integer.  The pypto_pro backend
    specialises per scalar valuation anyway, and a literal keeps every UB tile's
    allocation a compile-time constant, which is what RFC-0013 requires; only
    the window start is read at run time, and it is only ever an address.
    """
    element_dtype, gm_dtype = DTYPES[dtype]
    element_bytes = 2
    nd = N * D
    if heads_per_tile:
        if N % heads_per_tile:
            raise ValueError(f"heads_per_tile={heads_per_tile} must divide N={N}")
        width, heads = heads_per_tile * D, heads_per_tile
    else:
        width, heads = column_tile(L, N, D, element_bytes)
    chunks, stage, spare = _geometry(L, width)
    head_chunks = (D + LANES - 1) // LANES   # registers one head's splat needs
    # One outer step per weight expansion, `parts` groups inside it. Whole-head
    # groups expand once per group; a sub-head group reuses one head's expansion
    # across all its parts. Both keep the head index off any division: dividing
    # a run-time scalar would pull in the integer-cast supplement.
    if heads:
        outer, heads_per_outer, outer_span, parts = N // heads, heads, width, 1
    else:
        outer, heads_per_outer, outer_span, parts = N, 1, D, D // width

    @vf()
    def expand_slice_weight_vf(w_ub: Tensor, wb: Tensor, head0: Var):
        """wb[j, 0:width] <- weight[j, head0], for a group inside one head."""
        m32 = MaskReg(DT.float)
        widen = CastConfig(reg_layout=RegLayout.ZERO)
        narrow_reg = Reg(element_dtype)
        wide_reg = Reg(DT.float)
        for j in range(L):
            ub_to_reg_single(narrow_reg, w_ub[j * N + head0])
            cast(wide_reg, narrow_reg, widen, m32)
            for c in range(chunks):
                wb[j * width + c * LANES] <<= wide_reg

    @vf()
    def expand_heads_weight_vf(w_ub: Tensor, wb: Tensor, head0: Var):
        """wb[j, n*D : (n+1)*D] <- weight[j, head0 + n], widened to fp32.

        A splat writes one whole register, so a head wider than LANES needs
        `head_chunks` of them; a head narrower than LANES gets one that overruns
        into the next head's lanes, which that head then overwrites, and the
        last head's overrun lands in the row after it or in the tile's spare
        row.  Both cases are the same loop, and neither needs a predicate.
        """
        m32 = MaskReg(DT.float)
        widen = CastConfig(reg_layout=RegLayout.ZERO)
        narrow_reg = Reg(element_dtype)
        wide_reg = Reg(DT.float)
        for j in range(L):
            for n in range(heads):
                ub_to_reg_single(narrow_reg, w_ub[j * N + head0 + n])
                cast(wide_reg, narrow_reg, widen, m32)
                for c in range(head_chunks):
                    wb[j * width + n * D + c * LANES] <<= wide_reg

    expand_weight_vf = expand_heads_weight_vf if heads else expand_slice_weight_vf
    folds = fold_plan(L)
    align, first_count = folds[0]

    @vf()
    def compress_window_vf(x_ub: Tensor, wb: Tensor, p_ub: Tensor, o_ub: Tensor):
        """o_ub[0:width] <- sum_j x_ub[j, 0:width] * wb[j, 0:width] in fp32.

        The sum follows the upstream ReduceBlock order rather than a serial
        chain, because fp32 addition is not associative and the difference is
        observable: measured against the vendor operator on A2, four of 9216
        elements of the fp16 48x100x1x32 case land close enough to an fp16
        rounding boundary that the order decides them, and all four follow this
        one.  Staging the products and folding them is also what keeps the body
        small -- the same order evaluated as an expression tree unrolls into
        straight-line code that bisheng spills.

        When width is not a multiple of LANES the last chunk reads and writes a
        whole register past it, into the next row and, for the last row, into
        a spare row.  The staged products use a LANES-aligned pitch instead, so
        a fold never writes into the next row -- that one is still live.  Lanes
        do not interact, so no overrun reaches a live result.
        """
        m32 = MaskReg(DT.float)
        m16 = MaskReg(element_dtype)
        widen = CastConfig(reg_layout=RegLayout.ZERO)
        # CAST_ROUND is ties-away-from-zero, matching the upstream writeback.
        narrow = CastConfig(round_mode=RoundMode.AWAY_FROM_ZERO, reg_layout=RegLayout.ZERO)

        row_reg = Reg(element_dtype)
        value_reg = Reg(DT.float)
        weight_reg = Reg(DT.float)
        left = Reg(DT.float)
        right = Reg(DT.float)
        out_reg = Reg(element_dtype)

        for c in range(chunks):
            column = c * LANES
            # The first fold happens as the products are formed, so only `align`
            # of them are ever staged and each paired term costs one fused
            # multiply-add instead of a store, two loads and an add. Fusing is
            # exact here: a bf16 or fp16 product needs at most 22 significand
            # bits, so fp32 holds it without rounding and only the additions
            # round -- which is what makes the order, and only the order, matter.
            for i in range(first_count):
                ub_to_reg_unpack(row_reg, x_ub[i * width + column])
                cast(value_reg, row_reg, widen, m32)
                weight_reg <<= wb[i * width + column]
                mul(left, value_reg, weight_reg)
                ub_to_reg_unpack(row_reg, x_ub[(i + align) * width + column])
                cast(value_reg, row_reg, widen, m32)
                weight_reg <<= wb[(i + align) * width + column]
                muladddst(left, value_reg, weight_reg)
                p_ub[i * stage + column] <<= left
            for i in range(first_count, align):      # terms with no partner
                ub_to_reg_unpack(row_reg, x_ub[i * width + column])
                cast(value_reg, row_reg, widen, m32)
                weight_reg <<= wb[i * width + column]
                mul(left, value_reg, weight_reg)
                p_ub[i * stage + column] <<= left
            # Each fold reads what the one before it wrote. A store inside a vf
            # is not visible to a later load without this, and the simulator
            # does not model that: on the card, without these barriers, every
            # case came back wrong while sim stayed green.
            vf_barrier(VfPipe.STORE, VfPipe.LOAD)
            for stride, count in folds[1:]:
                for i in range(count):
                    left <<= p_ub[i * stage + column]
                    right <<= p_ub[(i + stride) * stage + column]
                    add(left, left, right)
                    p_ub[i * stage + column] <<= left
                vf_barrier(VfPipe.STORE, VfPipe.LOAD)
            left <<= p_ub[column]
            cast(out_reg, left, narrow, m16)
            reg_to_ub_downsample(o_ub[column], out_reg)

    @kernel(mode="vec", block_dim=block_dim)
    def nsa_compress(x: GM[gm_dtype, (T, nd)],
                     w: GM[gm_dtype, (1, L * N)],
                     win_start: GM[i32, (1, TC)],
                     o: GM[gm_dtype, (TC, nd)]):
        # `_geometry` sizes the overrun backing: spare rows for the tiles that
        # are only read, a LANES-aligned pitch for the one folded in place.
        # The window and the output row are doubled so that the next token's
        # load and the last one's writeback can run against this token's
        # arithmetic; the staged products stay single, because both halves of a
        # pair use the vector pipe and would serialise on it anyway.
        xa = Tensor(element_dtype, [L + spare, width], Position.UB, name="xa")
        xb = Tensor(element_dtype, [L + spare, width], Position.UB, name="xb")
        w_ub = Tensor(element_dtype, [1, L * N], Position.UB, name="w_ub")
        wb = Tensor(DT.float, [L + spare, width], Position.UB, name="wb")
        p_ub = Tensor(DT.float, [_staged_rows(L), stage], Position.UB, name="p_ub")
        oa = Tensor(element_dtype, [1 + spare, width], Position.UB, name="oa")
        ob = Tensor(element_dtype, [1 + spare, width], Position.UB, name="ob")

        # Output tokens are the unit of work: each carries its own window, so no
        # core depends on another's partial state.
        tokens_per_core = CeilDiv(TC, GetVecNum())
        first_token = Var(tokens_per_core * GetVecIdx())
        last_token = Min(first_token + tokens_per_core, TC)
        # Two tokens per iteration, so the two slots are Python constants. A
        # rotating slot index would need `k % 2`, and a modulo or divide on a
        # run-time scalar pulls in the integer-cast supplement; a run-time row
        # offset into one doubled tile is the `mem.slice` shape M10-075 does not
        # cover. Naming both tiles avoids each.
        pairs = CeilDiv(last_token - first_token, 2)

        with auto_sync():
            w_ub <<= w
            for h in range(outer):
                expand_weight_vf(w_ub, wb, Var(h * heads_per_outer))
                for q in range(parts):
                    column0 = h * outer_span + q * width
                    # Prologue: the first window, the one load nothing overlaps.
                    # Clamped so that an idle core -- more cores than tokens --
                    # still issues an in-bounds transfer it then never uses.
                    head = Var(Min(first_token, TC - 1))
                    start = Var(0, DT.int)
                    start.GetValueFrom(win_start[0, head:head + 1])
                    xa[0:L, 0:width] <<= x[start:start + L, column0:column0 + width]
                    for k in range(pairs):
                        # Every index is clamped into the core's own range, so an
                        # odd token count recomputes its last token instead of
                        # needing a guard. Writing the same row twice from the
                        # same window is idempotent.
                        i0 = Var(Min(first_token + 2 * k, last_token - 1))
                        i1 = Var(Min(i0 + 1, last_token - 1))
                        i2 = Var(Min(i0 + 2, last_token - 1))

                        nxt = Var(0, DT.int)
                        nxt.GetValueFrom(win_start[0, i1:i1 + 1])
                        xb[0:L, 0:width] <<= x[nxt:nxt + L, column0:column0 + width]
                        compress_window_vf(xa, wb, p_ub, oa)
                        o[i0:i0 + 1, column0:column0 + width] <<= oa[0:1, 0:width]

                        # Only prefetch while there is a pair left to use it.
                        # An unguarded prefetch past the end of this column group
                        # is not merely wasted: it is a second write to `xa` with
                        # no reader, and the next group's prologue write is the
                        # first. Measured on the card, that one won, and the
                        # first output row of every group after the first was
                        # computed from the previous group's columns -- while
                        # sim, pipesim and both backends' synchronisation all
                        # called it correct.
                        if k + 1 < pairs:
                            after = Var(0, DT.int)
                            after.GetValueFrom(win_start[0, i2:i2 + 1])
                            xa[0:L, 0:width] <<= x[after:after + L,
                                                   column0:column0 + width]
                        compress_window_vf(xb, wb, p_ub, ob)
                        o[i1:i1 + 1, column0:column0 + width] <<= ob[0:1, 0:width]
        return o

    return nsa_compress
