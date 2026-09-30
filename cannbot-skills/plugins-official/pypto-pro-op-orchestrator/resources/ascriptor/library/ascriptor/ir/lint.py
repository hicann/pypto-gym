# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hardware lints: warnings for constructs the reference interpreter executes exactly but the hardware treats
differently (D-051). They never change the IR and never fail a compile; ``ascriptor check`` / ``compile``, ``OpExec``
and ``tools/make_golden.py`` print them so the trap is named before the first cannsim or board run.

Two channels, because two of these checks are not traps at all
--------------------------------------------------------------
A trap means the run is wrong or faults: the interpreter is exact and the hardware is not. A ``perf`` lint means both
are correct and one is slower. They are read completely differently - one you must act on, one you weigh - so each
check declares its channel in the tables at the bottom of this file, and the printer prefixes them ``[lint]`` and
``[perf]``. Mixing them cost the nd2nz advice its audience: that message is one of the best in this file (source
line, board-measured multiple, the replacement, and the trap the replacement walks into) and it went out twelve
times in one export, six of them from one line, next to out-of-bounds writes.

Which is the other half. A finding is ``(source line, check)``; one ``@vf`` body expanded into N instances is still
one finding, so :func:`format_lints` collapses them and says how many sites it stood for.

Checks
------
* **unaligned VF load / store** — a contiguous register load or store (``vf.load_cont`` / ``vf.store_cont`` in a
  ``norm*`` mode) whose byte offset is not a multiple of 32: the vector load / store units address 32-byte blocks
  (AscendC's ``LoadAlign`` contract). Probed 2026-08-28 with fp32 loads and stores at byte offsets 4, 8, 16 (the
  board raised an AI core exception, ``aclrtSynchronizeStream`` 507035, for each) and 0, 32 (fine); cannsim and the
  interpreter performed every one element-exactly. Single-element accesses (``.single()`` broadcast loads,
  ``.single_value()`` stores) are not whole-register accesses and are exempt. The usual cause is ``ub[1] <<= reg``
  meant as a row offset (``ub[cols]``).
* **register window past the tile** — an unmasked contiguous load or store moves the whole 256-byte register: a
  store whose window ends past the tile writes over whatever tile the allocator placed next (D-050: that is how the
  64-bit scatter kernels lost their index tile), a load reads the neighbour's bytes into the upper lanes.
* **DMA rows off the tile's pitch** (lowered IR, :func:`lint_lowered`) — a padded GM <-> UB DMA lands every burst on
  a 32-byte-pitched row of UB; a tile whose rows are not that pitch (a [16, 2] uint8 tile: 32 packed bytes to the
  allocator and to a @vf) is read or written past its rows by every burst after the first (D-053).
* **strided block copy through the carrier** — a strided ``ub_to_reg`` / ``reg_to_ub`` of a register dtype the
  compiler header has no ``vsldb`` / ``vsstb`` form for (hif8) prints through the uint8 carrier — a reinterpret-cast of
  the register around the intrinsic, the form that lost the loaded value at the board's -O3 (D-055 / D-057) — guarded
  by an idempotent ``vor``. The guard held on every board run so far; the warning names the case before the next one.
* **mutex credits over the guarded buffer's slots** — a counting semaphore whose credits let the producer retake a
  slot the consumer may still be reading (M10-076); a wrong number rather than a hang.
* **a mutex missing one of its four calls, or unbalanced counts** — ``ready`` pairs with ``wait`` and ``lock`` with
  ``free``, so a role that never appears while its partner does leaves the other side's next call waiting forever.
  This is the shape a hand-written handshake reaches with the producer's half written and the consumer's half not.
* **a slot returned before anything read it** — a ``wait`` followed by its ``free`` with no read of the ``guards=``
  buffer between them. The consuming instructions belong inside that pair; a read placed after the ``free`` is
  ordered by nothing, while the credits still balance, so neither the token oracle nor the credit check speaks.
* **A2-family MMAD accumulation without an M-pipe settle** (lowered IR) — 910B and 910_93 do not interlock two
  short MMADs that update one L0C. A following ``is_init=False`` can read a partially settled accumulator
  unless an M/ALL barrier lies between them. The split-K shortcut inserts that barrier; this check guards
  hand-written MMAD streams and future expansions that bypass it. Both A2 and A3 M=16 boundaries failed on their
  respective boards, so this is a c220-family correctness rule rather than one facade's performance advice. The
  fp32 accumulators that failed there are what was measured, not the extent of the rule: the missing interlock is
  an L0C writeback and does not read the accumulator dtype, so the check reads every dtype (M10-081, 2026-09-18).
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Callable

from .core import Function, Ident, Literal, Module, Op, Value
from .types import BufType, MemType, RegType
from .fixpipe_rules import dual_mode_of, fixpipe_riders
from .verify import Diagnostic

REG_BYTES = 256
BLOCK_BYTES = 32


class HardwareWarning(UserWarning):
    """Raised (as a warning) by the interpreter when a run hits one of the traps at a dynamic address."""

_CONTIGUOUS = {"vf.load_cont": 1, "vf.store_cont": 0}  # opcode -> index of the memory operand


def lint(module: Module) -> list[Diagnostic]:
    """Every hardware lint of ``module`` (surface IR), in op order."""
    out: list[Diagnostic] = []
    for f in module.functions:
        defs = {r.name: op for op in f.walk() for r in op.results}
        for op in f.walk():
            for check, kind in _CHECKS:
                out.extend(_stamp(check, kind, check(f, op, defs)))
    return out


def _stamp(check: Callable[..., Any], kind: str, diags: list[Diagnostic]) -> list[Diagnostic]:
    """The channel and the rule identity come from the table, not from each message."""
    return [replace(d, kind=kind, rule=check.__name__) for d in diags]


def format_lints(diags: list[Diagnostic], *, perf: bool | None = None,
                 seen: set[tuple[str | None, str | None]] | None = None) -> str:
    """The lints as printed: one line per ``(source line, check)``, trap channel first.

    ``perf`` selects whether the performance channel is included; the default reads
    ``ASCRIPTOR_PERF_LINT`` and keeps it on. The trap channel is not optional - a lint that can
    be silenced is a lint that will be, and these are the ones that make a run wrong.

    ``seen`` carries the sites already reported ACROSS calls, and is where the volume actually
    comes from. Measured on this repository's 20-case sparse-attention export: a case emits at
    most one diagnostic per source line, so collapsing within one call saves nothing - the 28
    lines are 4 source sites lowered 20 times, one `lower_kernel` per case, each printing its
    own. A site already stated in full is restated as a single line naming the case, so the
    advice is read once and the per-case attribution survives.
    """
    if perf is None:
        perf = os.environ.get("ASCRIPTOR_PERF_LINT", "1") not in ("0", "false", "no")
    groups: dict[tuple[str | None, str | None], list[Diagnostic]] = {}
    for d in diags:
        if d.kind == "perf" and not perf:
            continue
        groups.setdefault((d.loc, d.rule), []).append(d)
    lines = []
    for kind in ("trap", "perf"):
        for key, hits in groups.items():
            if hits[0].kind != kind:
                continue
            tag = "lint" if kind == "trap" else "perf"
            if seen is not None and key in seen:
                where = f"@{hits[0].function} " if hits[0].function else ""
                lines.append(f"[{tag}] {where}{hits[0].loc}: {hits[0].rule} again (reported above)")
                continue
            if seen is not None:
                seen.add(key)
            more = f" (hit {len(hits)} times)" if len(hits) > 1 else ""
            lines.append(f"[{tag}] {hits[0]}{more}")
    return "\n".join(lines)


#: The sites `lower_kernel` and `ascriptor check` have already stated in full, for this process.
#: An export lowers every case in one process, so without it each case restates the same advice.
REPORTED: set[tuple[str | None, str | None]] = set()


# -- helpers ---------------------------------------------------------------------------------------------------------


def _diag(f: Function, op: Op, msg: str) -> Diagnostic:
    return Diagnostic("warning", msg, f.name, op.id, op.loc.chain[0] if op.loc else None)


def _int_attr(op: Op, name: str, default: int | None) -> int | None:
    v = op.attrs.get(name, default)
    if isinstance(v, Literal):
        v = v.value
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _ident(op: Op, name: str, default: str) -> str:
    v = op.attrs.get(name, default)
    return v.name if isinstance(v, Ident) else str(v)


def _tile_bytes(t: MemType) -> int | None:
    n = 1
    for d in t.dims:
        if not isinstance(d, int):
            return None
        n *= d
    return n * (t.dtype.bits // 8)


# -- checks ----------------------------------------------------------------------------------------------------------


def _contiguous_window(f: Function, op: Op, defs: dict[str, Op]) -> list[Diagnostic]:
    idx = _CONTIGUOUS.get(op.opcode)
    if idx is None or not _ident(op, "mode", "norm").startswith("norm"):
        return []
    off = _int_attr(op, "offset", 0)
    mem = op.operands[idx] if len(op.operands) > idx else None
    if off is None or not (isinstance(mem, Value) and isinstance(mem.type, MemType)):
        return []
    esize = mem.type.dtype.bits // 8
    byte = off * esize
    what = "load" if op.opcode == "vf.load_cont" else "store"
    out: list[Diagnostic] = []
    if byte % BLOCK_BYTES:
        out.append(_diag(f, op, f"{what} at element {off} of {mem.name} is byte {byte}: not 32-byte aligned — the board "
                                f"raises an AI core exception (aclrtSynchronizeStream 507035), cannsim and the interpreter "
                                f"silently {what} element-exactly (a row offset is `{mem.name}[cols]`, not `{mem.name}[1]`)"))
    size = _tile_bytes(mem.type)
    reg = op.operands[1 - idx]
    width = REG_BYTES * getattr(getattr(reg, "type", None), "n", 1)
    if size is not None and "mask" not in op.attrs and byte + width > size:
        over = byte + width - size
        if what == "store":
            out.append(_diag(f, op, f"unmasked store of the whole {width}-byte register at byte {byte} of {mem.name} "
                                    f"({size} bytes): {over} bytes land in whatever tile the allocator placed next"))
        else:
            out.append(_diag(f, op, f"unmasked load of the whole {width}-byte register at byte {byte} of {mem.name} "
                                    f"({size} bytes): the upper lanes hold {over} bytes of the tile that follows"))
    return out


def _tile_of(v: Any, defs: dict[str, Op]) -> MemType | None:
    """The allocation a UB window belongs to, through slices / reinterprets (a reshape is its own layout)."""
    seen = 0
    while isinstance(v, Value) and seen < 16:
        d = defs.get(v.name)
        if d is None or d.opcode in ("mem.alloc", "mem.reshape", "mem.get_buf"):
            break
        if d.opcode in ("mem.slice", "mem.reinterpret") and d.operands:
            v = d.operands[0]
            seen += 1
            continue
        break
    if isinstance(v, Value) and isinstance(v.type, MemType):
        return v.type
    if isinstance(v, Value) and isinstance(v.type, BufType):
        return v.type.elem
    return None


def _narrow_dma_rows(f: Function, op: Op, defs: dict[str, Op]) -> list[Diagnostic]:
    """Lowered IR: a padded GM <-> UB DMA lands every burst on its own 32-byte-pitched row of UB (the descriptor's
    UB stride is rounded up to 32 bytes: `ub_to_gm_pad` / `gm_to_ub_pad`), so a tile whose rows are not that pitch —
    a [16, 2] uint8 tile is 32 packed bytes to the allocator and to a @vf, 16 rows x 32 bytes to the DMA — is read
    or written past its rows (found on the board: the second and later rows came back as the neighbour's bytes).
    Tiles declared with 32-byte rows ([rows, 8] fp32 for a [rows, 1] GM column, as cov2x2_inverse does) are fine."""
    if op.opcode not in ("dma.gm_to_ub.pad", "dma.ub_to_gm.pad"):
        return []
    n, burst = _int_attr(op, "n_burst", 1), _int_attr(op, "burst_len_byte", None)
    ub = op.operands[0 if op.opcode == "dma.gm_to_ub.pad" else 1]
    stride = _int_attr(op, "dst_stride" if op.opcode == "dma.gm_to_ub.pad" else "src_stride", 0)
    if n is None or burst is None or stride is None or n <= 1:
        return []
    tile = _tile_of(ub, defs)
    if tile is None or tile.rank < 2 or not isinstance(tile.dims[-1], int):
        return []
    row = tile.dims[-1] * (tile.dtype.bits // 8)
    pitch = (burst + BLOCK_BYTES - 1) // BLOCK_BYTES * BLOCK_BYTES + stride * BLOCK_BYTES
    if row == pitch:
        return []
    name = ub.name if isinstance(ub, Value) else "?"
    return [_diag(f, op, f"{n} bursts of {burst} bytes through {name} land {pitch} bytes apart in UB (a padded DMA steps the "
                         f"UB side in 32-byte blocks) but the tile's rows are {row} bytes: every burst after the first reads "
                         f"or writes past its row — give the tile 32-byte rows or copy one contiguous block")]


def _carrier_block_copy(f: Function, op: Op, defs: dict[str, Op]) -> list[Diagnostic]:
    if op.opcode not in ("vf.load", "vf.store"):
        return []
    stride = op.attrs.get("blk_stride", 1)
    if isinstance(stride, Literal):
        stride = stride.value
    if stride == 1:  # stride 1 is the contiguous vlds / vsts (D-052)
        return []
    reg = op.operands[0 if op.opcode == "vf.load" else 1] if len(op.operands) > 1 else None
    if not (isinstance(reg, Value) and isinstance(reg.type, RegType)):
        return []
    from ..backends.cce.arch import c310  # the header's native forms; imported here to keep the IR layer free of the backend

    native = c310.VSLDB_ELEM if op.opcode == "vf.load" else c310.VSSTB_ELEM
    dt = reg.type.dtype
    if dt.name in native:
        return []
    what = "load" if op.opcode == "vf.load" else "store"
    return [_diag(f, op, f"strided block {what} of a {dt} register: c310 has no vsldb / vsstb form for {dt}, so it prints "
                         f"through the uint8 carrier — a reinterpret-cast of the register around the intrinsic, the form that "
                         f"lost the loaded value on the board (D-055 / D-057) — with a vor guard after the load; the guard has held "
                         f"on every board run so far, a contiguous copy (stride 1) or an 8-bit view of the data avoids the form")]


def _even_block_stride(f: Function, op: Op, defs: dict[str, Op]) -> list[Diagnostic]:
    """An even block stride on a strided register store (``vsstb``) lands consecutive
    datablocks on conflicting UB banks and the store port serialises. Board-measured
    (D-084 v2 probes: distinct-address store loops, one AIV at 1650 MHz): the contiguous
    ``vsts`` and odd-stride ``vsstb`` both sustain ~1.1 cycles/store; an even stride
    doubles that (stride 2 ~2.1); a multiple of 16 is the cliff (16 and 32 both ~8.7 —
    the bank-ways cap). Padding the row by one datablock makes the stride odd and
    returns to the flat floor. A stride that is a runtime value (a ``Var`` parameter of
    the @vf) gets its own warning: nothing here can prove it is odd, and the kernels that
    pass one had been storing at a multiple of 16 unflagged since they were ported
    (D-225)."""
    if op.opcode != "vf.store":
        return []
    stride = op.attrs.get("blk_stride", 1)
    if isinstance(stride, Literal):
        stride = stride.value
    if not isinstance(stride, int):  # a runtime stride: unjudgeable, which is itself the finding (D-225)
        name = stride.name if isinstance(stride, Value) else "?"
        return [_diag(f, op, f"runtime block stride ({name}) on a strided block store: the stride is not known at compile "
                             f"time, so this lint cannot prove it is odd — an even one lands consecutive datablocks on "
                             f"conflicting UB banks at ~2x an odd stride per store, a multiple of 16 at ~8.7x "
                             f"(board-measured, D-084); pad the destination row and pass an odd pitch (D-225)")]
    if stride <= 1 or stride % 2 == 1:
        return []  # 1 is the contiguous vsts; odd strides steer clear of the banks
    return [_diag(f, op, f"strided block store with an even block stride ({stride}): consecutive datablocks hit "
                         f"conflicting UB banks and the store port serialises (board-measured: ~2x an odd stride, "
                         f"~8x at multiples of 16 — D-084); pad the destination row so the stride is odd "
                         f"({stride + 1}) to restore the flat rate")]


def _ub_to_l1_layout_dma(f: Function, op: Op, defs: dict[str, Op]) -> list[Diagnostic]:
    """``ub_to_l1.nd2nz`` is not one instruction: it composes one MTE3 burst per fractal
    column out of the ND rows. Storing compact NZ from the @vf (a strided ``vf.store``,
    odd block stride) makes the move a single plain ``ub_to_l1`` — board-measured 10x
    faster on a [64, 128] f16 tile (D-084: 450.8 us vs 44.7 us over 512 reps).

    The advice carries the rework's own trap, because the rework walks straight into it:
    the strided store that replaces this move takes the staging tile's row count as its
    block stride, and a tile's row count is 16-aligned by nature — which is the WORST
    rung of the bank ladder, ~8.7 cycles a store against ~1.1. On the two kernels reworked
    in D-226 padding that pitch odd was worth more than the move it rides on (-12.3 and
    -10.8 us of a -17.7 and -28.2 us total), so the number is stated here rather than left
    for `_even_block_stride` to catch afterwards."""
    if op.opcode != "dma.ub_to_l1.nd2nz":
        return []
    return [_diag(f, op, "ub_to_l1.nd2nz composes multiple MTE3 bursts (one per NZ fractal column of the ND rows) — "
                         "not a single instruction, and board-measured 10x slower than the compact-NZ move (D-084); "
                         "store the tile as compact NZ from the "
                         "@vf instead (a strided vf.store writes the fractal directly) and move it with the plain "
                         "ub_to_l1. WHEN YOU DO: pad the staging tile by one row so that store's block stride is ODD. "
                         "The natural stride is the tile's row count, which is 16-aligned by nature and so is the "
                         "worst rung of the bank ladder (~8.7 cycles a store against ~1.1, D-084) — on the two "
                         "kernels reworked in D-226 the padding was worth MORE than this move is (-12.3 and -10.8 us "
                         "of -17.7 and -28.2), and it is worth taking blind: one UB row, and it only ever costs "
                         "nothing when the store pipe is idle anyway")]


def _single_dual_mode(f: Function, op: Op, defs: dict[str, Op]) -> list[Diagnostic]:
    """``l0c_to_ub`` in ``DualMode.SINGLE`` for a move ``SPLITM`` could have carried.

    Every cube core is paired with TWO vector sub-blocks - the ratio holds on all seven device
    profiles this library ships, while the core counts do not (a5 ``950`` is 32c/64v and a5pr
    ``950pr`` is 28c/56v), so the ratio is what the rule rests on. A split mode gives each
    sub-block half of the L0C tile, in its own UB - ``SPLITM`` along M, or ``SPLITN`` when the
    product is transposed and the rows a sub-block owns are the N extent; ``SINGLE`` sends the
    whole tile to one, leaving the other idle and sizing the landing tile for the whole M. Both compile, and
    the difference is roughly two on vector time and two on UB - and the UB half compounds,
    because a landing tile twice the size is what forces the M tile back down. On this
    repository's sparse-attention kernels every drain was written ``SINGLE``, and the two
    factors together were the larger part of a 3.13x gap against the reference.

    No def-use analysis behind it: every ``SINGLE`` that COULD have been split is named, and a
    move that genuinely needs one sub-block to see all M rows answers with one line saying so.
    The alternative - proving the destination is only ever read per row - is a day of work to
    buy silence on a decision that deserves a sentence anyway.

    Lowered rather than surface, because the two spellings only meet there: a written-out
    ``l0c_to_ub`` carries its mode from the frontend, while ``ub <<= l0c`` is a ``dma.copy``
    until ``device_lower`` selects the instruction and materialises the default.
    """
    if op.opcode != "dma.l0c_to_ub":
        return []
    if dual_mode_of(op) != "single":
        return []
    if fixpipe_riders(op):
        return []  # SINGLE is not a choice here -- `fixpipe_rules.split_mode_error` owns that
    return [_diag(f, op, "l0c_to_ub in DualMode.SINGLE carries a same-type plain copy, which a "
                         "split mode also carries: every cube core is paired with two vector "
                         "sub-blocks, and a split hands each of them half of the tile in its own "
                         "UB - about half the vector time and half the UB per sub-block, which "
                         "is often what lets the M tile stay at 128. Split the axis that carries "
                         "the rows a sub-block owns: DualMode.SPLITM for an [M=rows, N=...] "
                         "tile, DualMode.SPLITN when the product is transposed and the rows are "
                         "the N extent (a score^T = K @ Q^T tile is). SINGLE is needed only when "
                         "one sub-block must see every M row (a reduce ACROSS M); if that is the "
                         "case here, say so in a comment so the next reader does not have to "
                         "re-derive it")]


def _is_mmad_to(op: Op, root: str | None, defs: dict[str, Op]) -> bool:
    if op.opcode != "cube.mmad" or len(op.operands) < 3 or root is None:
        return False
    operands = op.operands[:3]
    return (_root_name(operands[0], defs) == root
            and all(isinstance(value, Value) and isinstance(value.type, MemType)
                    for value in operands))


def _mmad_pending_at(f: Function, target: Op, root: str, defs: dict[str, Op]) -> bool:
    """Whether some path reaches ``target`` after an unsettled MMAD to ``root``.

    The state is one bit, so loop closure converges after at most one new carried state.  A
    loop's zero-trip path is retained and both branch arms are joined; an M/ALL barrier clears
    the bit.  This intentionally asks only about MMAD-to-MMAD L0C RAW, not general scheduling.
    """
    hazard = False

    def block_state(block: Any, incoming: set[bool]) -> set[bool]:
        nonlocal hazard
        state = set(incoming)
        for current in block.ops:
            if current is target and True in state:
                hazard = True
            if current.opcode == "sync.barrier" and _ident(current, "pipe", "ALL") in ("M", "ALL"):
                state = {False}
            elif _is_mmad_to(current, root, defs):
                state = {True}

            if not current.regions:
                continue
            if current.opcode == "cf.for":
                entry = set(state)
                while True:
                    body_out = block_state(current.regions[0], entry)
                    widened = entry | body_out  # zero trips plus every repeated iteration
                    if widened == entry:
                        break
                    entry = widened
                state = entry
            else:
                exits = [block_state(region, state) for region in current.regions]
                state = set().union(*exits) if exits else state
        return state

    block_state(f.body, {False})
    return hazard


def unsettled_mmad_accumulate(f: Function, op: Op, defs: dict[str, Op]) -> bool:
    """Whether ``op`` accumulates into an L0C an earlier MMAD may not have written back yet.

    The one analysis behind both consumers of the M10-081 rule: this check, which names the site,
    and ``passes/mmad_settle.py``, which inserts the barrier the site is missing.  Keeping them on
    one function is what makes "the lint is quiet because the pass repaired it" a measurement
    rather than two implementations agreeing by luck.
    """
    if op.opcode != "cube.mmad" or op.attrs.get("is_init", True) is not False or not op.operands:
        return False
    root = _root_name(op.operands[0], defs)
    return (root is not None and _is_mmad_to(op, root, defs)
            and _mmad_pending_at(f, op, root, defs))


def _a2_family_unsettled_mmad(f: Function, op: Op, defs: dict[str, Op]) -> list[Diagnostic]:
    """A hand-written A2-family accumulate whose preceding L0C writer has not settled."""
    if not unsettled_mmad_accumulate(f, op, defs):
        return []
    root = _root_name(op.operands[0], defs)
    return [_diag(
        f,
        op,
        f"MMAD accumulation reads %{root} after an earlier MMAD on at least one path with no PIPE_M/PIPE_ALL "
        "barrier between them: 910B and 910_93 returned wrong values for the M=16 split-K boundary "
        "while the interpreter was exact. Add `barrier(Pipe.M)` after the producing MMAD; `matmul(..., splitk=...)` "
        "already inserts this A2-family settle automatically",
    )]


def _strided_innermost_view(f: Function, op: Op, defs: dict[str, Op]) -> list[Diagnostic]:
    """A ``mem.view`` with a non-unit innermost stride is read-only and gathers per element:
    NDDMA issues one transfer per element (no burst coalescing) on the GM -> UB path, and the
    write engines refuse it outright (RFC-0010 phase 3)."""
    if op.opcode != "mem.view":
        return []
    strides = op.attrs.get("strides") or []
    last = strides[-1] if strides else 1
    if isinstance(last, Literal):
        last = last.value
    if not isinstance(last, int) or last <= 1:
        return []
    return [_diag(f, op, f"strided GM view with a non-unit innermost stride ({last}): NDDMA moves it one element "
                         f"per transfer (no burst coalescing), and only the GM -> UB read path supports it; "
                         f"restructure so the innermost dim is contiguous if this DMA is hot")]


def _mutex_credits_over_slots(f: Function, op: Op, defs: dict[str, Op]) -> list[Diagnostic]:
    """A mutex carrying more credits than the buffer it says it guards has slots.

    ``depth`` is a credit count: the consumer publishes that many up front, so the producer's
    ``lock()`` of cycle *i* blocks on the ``free()`` of cycle *i - depth*. One credit per slot is
    the answer for a mutex that runs once per rotation; more than that lets the producer retake a
    slot the consumer may still be reading, and the result is a wrong number rather than a hang.

    It is a warning and not an error because more credits is not always wrong: a mutex that cycles
    twice per rotation of its buffer legitimately carries twice the slots, and `autosync` computes
    exactly that (``len(locks) * span``). Only the author knows which case this is - which is why
    writing ``guards=`` and ``depth=`` together is worth doing when the answer is not one credit
    per slot. Writing neither is the case this cannot see.

    Surface rather than lowered: ``mem.alloc`` does not survive ``device_lower``, so the slot count
    is only readable here.
    """
    if op.opcode != "sync.mutex":
        return []
    guards = op.attrs.get("guards") or ()
    depth = _int_attr(op, "depth", 2)
    if not guards or depth is None:
        return []
    slots = []
    for name in guards:
        d = defs.get(str(name))
        t = d.results[0].type if d is not None and d.results else None
        if isinstance(t, BufType):
            slots.append(t.slots)
        elif isinstance(t, MemType):
            slots.append(1)
    if not slots or depth <= min(slots):
        return []
    named = ", ".join(f"%{name}" for name in guards)
    return [_diag(f, op, f"mutex {op.attrs.get('id')} carries {depth} credits but guards {named}, which "
                         f"rotates through {min(slots)}: the producer can take a slot {depth - min(slots)} "
                         f"cycle(s) before the consumer frees it, and a slot retaken while it is still "
                         f"being read is a wrong answer, not a hang. This is correct only if the mutex "
                         f"cycles more than once per rotation of that buffer - if it does, that is worth "
                         f"a comment, because the next reader will count slots")]


#: check -> channel. ``trap``: the hardware faults or answers differently from the interpreter.
#: ``perf``: both answer the same and one is slower. `_strided_innermost_view` is a perf lint
#: because the case it would be a trap for cannot arise - a non-unit innermost stride is
#: read-only and the frontend refuses the write outright (`rules_mem.py`), so all this one has
#: left to say is that the read gathers per element.
#: A view chain back to its allocation: ``guards=`` records root names, and a consuming op names a view.
_VIEW_OPS = ("mem.slice", "mem.get_buf", "mem.reinterpret", "mem.reshape", "mem.view")
_MUTEX_CALLS = ("lock", "ready", "wait", "free")


def _root_name(value: Any, defs: dict[str, Op], limit: int = 24) -> str | None:
    name = getattr(value, "name", None)
    for _ in range(limit):
        d = defs.get(name)
        if d is None or d.opcode not in _VIEW_OPS or not d.operands:
            return name
        name = getattr(d.operands[0], "name", None)
    return name


def _blocks(f: Function):
    """Every block of the function, outermost first: one mutex cycle is read inside one block."""
    pending = [f.body]
    while pending:
        block = pending.pop()
        yield block
        for op in block.ops:
            pending.extend(op.regions)


def _mutex_role(op: Op, flag: str | None) -> str | None:
    """Which of the four calls ``op`` is, if it is a call on ``flag``."""
    if not op.opcode.startswith("sync.mutex_") or not op.operands:
        return None
    if getattr(op.operands[0], "name", None) != flag:
        return None
    call = op.opcode.removeprefix("sync.mutex_")
    return call if call in _MUTEX_CALLS else None


def _touches(op: Op, roots: set[str], defs: dict[str, Op]) -> bool:
    """Does ``op``, or anything in its regions, name a view of one of ``roots``?"""
    for value in list(op.operands) + list(op.attr_values()):
        if _root_name(value, defs) in roots:
            return True
    return any(_touches(inner, roots, defs) for block in op.regions for inner in block.ops)


def _mutex_calls_unbalanced(f: Function, op: Op, defs: dict[str, Op]) -> list[Diagnostic]:
    """A mutex missing one of its four calls, or publishing more often than it is waited for.

    One cycle is ``lock`` / ``ready`` on the producer and ``wait`` / ``free`` on the consumer, and
    the counts pair up: ``ready`` with ``wait``, ``lock`` with ``free``. A role that never appears
    while its partner does is the shape a hand-written handshake arrives at - the producer's half
    written and the consumer's half not - and it costs the other side's next call forever, which
    the functional simulator reports as a deadlock naming the mutex and the awaited call.

    Unequal nonzero counts are reported only when every call of the flag sits in one block, since
    a call under a device branch is written once per arm and legitimately outnumbers its partner.
    """
    if op.opcode != "sync.mutex":
        return []
    flag = op.results[0].name if op.results else None
    sites = [(call, c) for c in f.walk() if (call := _mutex_role(c, flag)) is not None]
    if not sites:
        return []
    counts = {call: sum(1 for name, _ in sites if name == call) for call in _MUTEX_CALLS}
    fid = op.attrs.get("id")
    out = []
    for publish, awaits in (("ready", "wait"), ("free", "lock")):
        have, want = counts[publish], counts[awaits]
        if bool(have) != bool(want):
            missing, present = (awaits, publish) if have else (publish, awaits)
            out.append(_diag(f, op, f"mutex {fid} calls `{present}` {max(have, want)} time(s) and `{missing}` "
                                    f"never: the four calls are one cycle, and the side that never runs "
                                    f"`{missing}` leaves the other side's next call waiting forever. The "
                                    f"functional simulator reports that as a deadlock naming this mutex"))
            continue
        if have != want and any(all(c in block.ops for _, c in sites) for block in _blocks(f)):
            out.append(_diag(f, op, f"mutex {fid} runs `{publish}` {have} time(s) against `{awaits}` {want}: "
                                    f"the counts of one cycle pair up, so the extra call either publishes a "
                                    f"token no reader consumes or lets one side overrun the other"))
    return out


def _mutex_slot_returned_unread(f: Function, op: Op, defs: dict[str, Op]) -> list[Diagnostic]:
    """A ``wait`` / ``free`` pair with no read of the guarded buffer between them.

    ``free`` returns the credit after the LAST read of the handed-over buffer, so the consuming
    instructions belong between the pair. A ``wait`` followed straight by its ``free`` hands the
    slot back before anything read it, and the read placed after the ``free`` is then ordered by
    nothing at all - while the credit counts still balance, so neither the token oracle nor the
    credit check says a word. Only ``guards=`` says which buffer the reads should name, so a mutex
    without it is the case this cannot see.

    Ops in a nested region between the two calls count: a read under ``if GetSubBlockIdx() == 0:``
    is still a read. A ``free`` in a different block from its ``wait`` is skipped rather than
    guessed at.
    """
    if op.opcode != "sync.mutex":
        return []
    roots = {str(name) for name in (op.attrs.get("guards") or ())}
    flag = op.results[0].name if op.results else None
    if not roots or flag is None:
        return []
    out = []
    for block in _blocks(f):
        ops = list(block.ops)
        for index, call in enumerate(ops):
            if _mutex_role(call, flag) != "wait":
                continue
            freed = next((j for j in range(index + 1, len(ops)) if _mutex_role(ops[j], flag) == "free"), None)
            if freed is None:
                continue
            if any(_touches(between, roots, defs) for between in ops[index + 1:freed]):
                continue
            named = ", ".join(f"%{name}" for name in sorted(roots))
            out.append(_diag(f, ops[freed], f"mutex {op.attrs.get('id')} returns its slot at this `free` "
                                            f"without {named} having been read since the `wait` at "
                                            f"{ops[index].loc}: `free` belongs after the LAST read of the "
                                            f"handed-over buffer, and a read outside the pair is ordered by "
                                            f"nothing. The credits still balance, so this is a wrong answer "
                                            f"rather than a hang"))
    return out


_CHECKS: tuple[tuple[Callable[[Function, Op, dict[str, Op]], list[Diagnostic]], str], ...] = (
    (_contiguous_window, "trap"), (_carrier_block_copy, "trap"), (_mutex_credits_over_slots, "trap"),
    (_mutex_calls_unbalanced, "trap"), (_mutex_slot_returned_unread, "trap"),
    (_even_block_stride, "perf"), (_strided_innermost_view, "perf"))
_LOWERED_CHECKS: tuple[tuple[Callable[[Function, Op, dict[str, Op]], list[Diagnostic]], str], ...] = (
    (_narrow_dma_rows, "trap"), (_a2_family_unsettled_mmad, "trap"),
    (_ub_to_l1_layout_dma, "perf"), (_single_dual_mode, "perf"))
_CHECK_DEVICES: dict[Callable[..., Any], frozenset[str]] = {
    _a2_family_unsettled_mmad: frozenset({"b1", "b2", "b3", "b4", "a3"}),
}


def lint_lowered(module: Module) -> list[Diagnostic]:
    """The lints that need the lowered geometry (DMA bursts), on a module after the pipeline."""
    out: list[Diagnostic] = []
    for f in module.functions:
        defs = {r.name: op for op in f.walk() for r in op.results}
        for op in f.walk():
            for check, kind in _LOWERED_CHECKS:
                if (devices := _CHECK_DEVICES.get(check)) is not None and module.device not in devices:
                    continue
                out.extend(_stamp(check, kind, check(f, op, defs)))
    return out


__all__ = ["HardwareWarning", "lint", "lint_lowered", "format_lints", "unsettled_mmad_accumulate",
           "REPORTED", "REG_BYTES", "BLOCK_BYTES"]
