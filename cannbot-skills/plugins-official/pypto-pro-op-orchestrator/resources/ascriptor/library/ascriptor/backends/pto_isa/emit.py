# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Lowered IR -> PTO tile-ISA C++ source (RFC-0011).

Batches 1-2: **tile creation and the GM <-> UB movement**. ``mem.alloc`` prints a ``pto::Tile``
declaration and its ``TASSIGN``; the view ops print nothing and ride to their consumer;
``dma.gm_to_ub.pad`` / ``dma.ub_to_gm.pad`` / ``dma.ub_to_ub`` print ``TLOAD`` / ``TSTORE`` /
``TMOV``. Every other opcode raises :class:`PtoIsaGap` with its source location.

The window folding is the ``cce`` backend's (:mod:`..cce.views`): the view chain under a value
folds to one root with element offsets, and both backends read the same ``Geo``. What a window
*becomes* differs: cce displaces an address; here it re-types a tile. **A PTO tile is a view,
not storage** — the same UB address can carry several ``pto::Tile`` objects of different shape
and valid region — so a narrowed or displaced transfer is a second ``Tile`` type bound by its
own ``TASSIGN``, never a `pl`-style ``set_validshape`` bracket.

The burst -> tile bridge (RFC-0011 §2.1) is read out of pto's own a5 templates, which spell it
explicitly (``TLoadVecND2ND`` in ``tload_common.hpp``, ``TStoreVecND`` in ``npu/a5/TStore.hpp``)::

    nBurst   = gShape3                     lenBurst = validCol * esize
    gmStride = gStride3 * esize            ubStride = TileData::Cols * esize

The last one is the load-bearing constraint: **the UB row pitch of a PTO transfer is the tile
type's ``Cols``**, a template argument, so it has to be a compile-time integer. The GM side is
an ordinary runtime ``Stride``.
"""

from __future__ import annotations

import contextlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...ir import FuncRef, Function, Literal, Module, Value
from ...ir.scalar_range import ScalarRanges
from ...ir.types import BufType, CellType, DType, MemType, ScalarType
from ...passes.addr_alloc import ALIGN as SPACE_ALIGN
from ...passes.util import Defs
from ..base import Artifacts
from ...devices import load as _load_device
from ..cce import cpp, views
from ..cce.arch import c310
from ..cce.emit import EVENT_ALIAS, F32, SimtPrinter, VfPrinter, c_ident
from ..cce.host import HostPrinter
from ..cce.host import dim_str as _dim_str
from ..shared import ScalarFolder
from . import types as pt
from .sync import ExplicitLocalSync

#: the scale plane's element type, for the box check's size arithmetic. PTO spells the tile
#: `float8_e8m0_t`; our IR carries the public one-byte carrier (see `FnPrinter._MX_ELEM`).
_MX_DT = DType("e8m0", 8, "float")


class PtoIsaGap(Exception):
    """An op the pto_isa backend has no PTO instruction for (RFC-0011 §6): located, never guessed."""

    def __init__(self, op: Any, why: str, *, owner: str = "unmapped") -> None:
        if owner not in ("ours", "upstream", "unmapped"):
            raise ValueError(f"invalid gap owner: {owner}")
        self.op = op
        self.why = why
        self.owner = owner
        loc = getattr(op, "loc", None) if op is not None else None
        code = getattr(op, "opcode", None) if op is not None else None
        where = f" at {loc}" if loc else ""
        what = f"{code}: " if code else ""
        super().__init__(f"{what}{why}{where}")


@dataclass
class Tile:
    """A declared ``pto::Tile`` and the address it was bound to.

    PTO tiles are views: this record is what a *second* view of the same storage is built from
    when a transfer needs a different shape or valid region.
    """

    name: str  # the C++ identifier of the tile this record addresses
    space: str
    dtype: DType
    shape: tuple[Any, ...]  # declared capacity (rows, cols), in elements
    addr: Any = 0  # the on-chip byte address TASSIGN bound (int or a scalar Value)
    layout: str | None = None
    slots: int = 0  # > 0: a C++ array of tiles (slot buffer)
    step: int = 0  # bytes between consecutive slots (the *aligned* slot, see op_mem_alloc)
    #: the declared C++ tile is ``1 x (rows * cols)``, because the two-dimensional form has no
    #: pto::Tile (see `tile_type`). `shape` still records the geometry the kernel asked for, so a
    #: consumer reasons about the real thing -- it just may not *reuse* this declaration as a shape.
    flat: bool = False


class _Printed(str):
    """A C++ fragment the printer has already rendered.

    The scalar layer folds and prints *IR values*; a helper call like ``SlotOf<2>(idx)`` is not
    one, so it travels as this and ``cexpr`` hands it back unchanged. Keeping it a `str` subclass
    means the ordinary arithmetic helpers below can still compose with it.
    """

    __slots__ = ()


def _SlotOf(index: str, slots: int) -> _Printed:
    """A slot buffer's ring index — `ascrip::SlotOf<N>`, which is `Buff::get`'s arithmetic."""
    return _Printed(f"SlotOf<{slots}>({index})")


def _Mul(a: Any, b: int) -> _Printed:
    return _Printed(f"({a}) * ({b})")


@dataclass(frozen=True)
class Extent:
    """One extent of a transfer: a compile-time integer when it folds, else a C++ expression.

    PTO splits a tile into a **static capacity** (the ``Rows`` / ``Cols`` template arguments) and a
    **valid region** that may be ``DYNAMIC`` and set at run time. So an extent that does not fold
    is not automatically a gap — it is a dynamic valid region (RFC-0011 §7.6). What must still
    fold is the row *pitch*, because that one is the capacity ``Cols``.
    """

    value: int | None
    expr: str

    @property
    def static(self) -> bool:
        return self.value is not None

    def __str__(self) -> str:
        return str(self.value) if self.value is not None else self.expr


@dataclass
class GTensor:
    """A declared ``pto::GlobalTensor`` or a window over one."""

    name: str
    dtype: DType
    shape: tuple[Any, ...]
    strides: tuple[Any, ...] | None = None
    offset: Any = 0
    decl: bool = False  # True once the C++ declaration has been printed


class FnPrinter(HostPrinter, ExplicitLocalSync):
    """Prints one kernel function's body."""

    kind = "kernel"

    def __init__(self, mp: ModulePrinter, fn: Function) -> None:
        self.mp = mp
        self.fn = fn
        self.defs = Defs(Module(fn.name, {}, (fn,)))
        self.lines: list[str] = []
        #: mutex token drains, printed before every return (sync.mutex, §5.3)
        self.epilogue: list[str] = []
        self.indent = 1
        self.names: dict[str, str] = {}
        self.used: set[str] = set()
        self.tiles: dict[str, Tile] = {}
        #: allocation name -> why `op_mem_alloc` refused it, so a later transfer through
        #: that allocation can report the cause rather than the missing record
        self.tile_gaps: dict[str, str] = {}
        self.gms: dict[str, GTensor] = {}
        # event name -> (set_pipe, wait_pipe, ids). A depth-1 event prints as a bare
        # set_flag / wait_flag pair; a deeper one rotates through its ids and needs the counters
        # below, one pair per event, declared where the sync.event is.
        #: names of events whose declaration printed, so a set/wait on one that
        #: refused is a located gap rather than an undefined C++ identifier
        self.events: set[str] = set()
        self.geo_cache: dict[str, views.Geo] = {}
        # a dim written as %div (= scalar.div(tokens, 2)) folds along its SSA chain against the
        # module's valuation; leaves are literals or bound parameters (RFC-0011 §7 batch 1b)
        # Shape proofs refer to the mathematical scalar graph before native
        # arithmetic expansion; value names and memory geometry are unchanged.
        self.folder = ScalarFolder(mp.analysis_functions.get(fn.name, fn), mp.bindings)
        self.ranges = ScalarRanges(mp.analysis_functions.get(fn.name, fn))
        # which core the function runs on: a static property here, so ``core.*`` resolves its
        # AIV / AIC branch at print time instead of testing AscendC's ASCEND_IS_AIV at run time
        self.side = _side_of(fn)
        # the parameters own their C++ names before any SSA value can take one: the signature is
        # printed from ``c_ident(param)`` directly, so a body value that sanitises to the same
        # identifier has to be the one that gets renamed, not the parameter
        for q in fn.params:
            self.name(q)

    def gap(self, op: Any, why: str) -> Exception:
        return PtoIsaGap(op, why)

    # ---------------------------------------------------------------- names, output

    def name(self, v: Value) -> str:
        n = self.names.get(v.name)
        if n is None:
            n = c_ident(v.name)
            while n in self.used:
                n += "_"
            self.used.add(n)
            self.names[v.name] = n
        return n

    def val(self, x: Any, dt: DType | None = None) -> str:
        if isinstance(x, Value):
            return self.name(x)
        if isinstance(x, Literal | bool | int | float):
            return cpp.literal(x, dt)
        raise PtoIsaGap(None, f"cannot print operand {x!r}")

    def bare(self, x: Any, dt: DType | None = None) -> str:
        """Nothing is folded here, so a right-hand side has no outer parentheses to drop."""
        return self.val(x, dt)

    def cexpr(self, e: views.Expr) -> str:
        # a fragment this printer already rendered (a `SlotOf<N>(...)` call, say) is not an IR
        # expression and has nothing left to fold or name
        if isinstance(e, _Printed):
            return str(e)
        return views.cexpr(e, self.name)

    def emit(self, text: str, op: Any = None) -> None:
        tag = f"  // #{op.id}" if op is not None and getattr(op, "id", None) is not None else ""
        self.lines.append("    " * self.indent + text + tag)

    def rec_of(self, op: Any, v: Value) -> Tile:
        """The :class:`Tile` record behind ``v``, or a gap that names the *cause*.

        A missing record always means ``op_mem_alloc`` refused this allocation earlier; that
        refusal is the real reason and it is the one worth reading."""
        root = self.geo(v).root.name
        rec = self.tiles.get(root)
        if rec is not None:
            return rec
        why = self.tile_gaps.get(root)
        raise PtoIsaGap(op, f"{v.name} has no pto::Tile: its allocation was refused — {why}"
                        if why else
                        f"{v.name} is not backed by a mem.alloc this printer has seen")

    def slot_index(self, op: Any, g: views.Geo, rec: Tile) -> Any:
        """The wrapped slot index of a view into a slot buffer, or ``None`` if it has none.

        cce spells a slot buffer as an object whose ``get`` wraps the index --
        ``slot[((i % N) + N) % N]`` (`tensorutils_cce.h:375`). A ring counter that is only ever
        incremented is the normal way our kernels drive one, so the wrap is load-bearing: without
        it the third iteration of a double-buffered loop indexes past the array.
        """
        if g.slot is None:
            return None
        if not rec.slots:
            raise PtoIsaGap(op, f"{rec.name} is indexed as a slot buffer but was not allocated as one")
        c = g.slot if isinstance(g.slot, int) else self.folder.fold(g.slot)
        if c is not None:
            return ((c % rec.slots) + rec.slots) % rec.slots
        if self.ranges.normalized(g.slot, rec.slots):
            return _Printed(self.cexpr(g.slot))
        return _SlotOf(self.cexpr(g.slot), rec.slots)

    def slot_bytes(self, op: Any, g: views.Geo, rec: Tile) -> Any:
        """How far the live slot sits from the allocation's base, in bytes (``0`` if not a ring).

        ``views.byte_offset`` deliberately leaves ``Geo.slot`` out -- cce applies it separately, by
        indexing the ``Buff`` object. Here the slots are a C++ array of tiles one aligned slot
        apart (`op_mem_alloc`), so the slot is arithmetic on the address instead, and a printer
        that forgets it lands *every* transfer on slot 0: right on the first iteration of a
        double-buffered loop and wrong on every one after, which is a silent wrong answer rather
        than a refusal.

        It stays out of the *in-slot* offset because every slot is identical: the 32-byte pitch
        widening, the "does this view fit" overrun check and the whole-tile fast path all measure
        one slot's room, and folding a ring counter into that would turn three static questions
        run-time -- and answer one of them wrong, since the room left would go to zero.
        """
        slot = self.slot_index(op, g, rec)
        if slot is None:
            return 0
        return slot * rec.step if isinstance(slot, int) else _Mul(slot, rec.step)

    def tile_addr(self, op: Any, g: views.Geo, rec: Tile, addr: int, off: int | None,
                  off_b: Any) -> str:
        """The C++ address a ``TASSIGN`` binds: allocation base + live slot + in-slot offset.

        ``off`` is the in-slot offset when it folds and ``None`` when the window origin is
        run-time; ``off_b`` is its expression either way.
        """
        sb = self.slot_bytes(op, g, rec)
        if off is not None and isinstance(sb, int):
            return str(addr + off + sb)
        terms = [self.cexpr(off_b)] if off is None else ([str(off)] if off else [])
        if isinstance(sb, int):
            if sb:
                terms.append(str(sb))
        else:
            terms.append(self.cexpr(sb))
        return f"{addr} + ({' + '.join(terms)})" if terms else str(addr)

    def run_op(self, op: Any) -> None:
        """Dispatch one op, and leave **no output behind** if it refuses.

        A handler may emit before it discovers a gap — §4.12's MX load declares the data move's two
        tiles and only then computes the scale plane's shape, which is where an fp4 operand stops.
        Those lines carry the op's ``// #id`` tag, and `tests/backends/test_cce.py` decides an op
        *printed* by looking for that tag (as the former coverage sweep did), so a
        half-emitted refusal reads as a success: the mxfp4 kernels were reported complete while
        refusing three ops each. Rolling the buffer back is the same invariant `_indented`'s
        ``finally`` keeps (§7.x): a printer's state has to hold on the refusal path, or its own
        measurements become fiction.
        """
        handler = getattr(self, "op_" + op.opcode.replace(".", "_"), None)
        if handler is None:
            raise PtoIsaGap(op, self._UNPRINTED_OP.get(op.opcode) or self._UNPRINTED.get(
                op.opcode.split(".")[0],
                "no printer for this opcode (RFC-0011 §1: this backend's scope is tile creation "
                "and the movement between memories)"))
        mark = len(self.lines)
        try:
            handler(op)
        except PtoIsaGap:
            del self.lines[mark:]
            raise

    #: Why a whole family has no printer, once someone has gone and looked. Written after §7.25's
    #: survey, because "no printer for this opcode" is a true sentence that teaches nothing: the
    #: families left are not left for the same reason, and only one of them is a *PTO* gap.
    _UNPRINTED = {
        "vec": ("vector compute is out of RFC-0011 §1's scope, which is tile creation and "
                "movement; cce's VfPrinter is what prints it (§7.7)"),
    }

    #: A few `vec.*` opcodes are refused for a sharper reason than "out of scope", and saying the
    #: weaker one would hide it.
    _MASK_SPR = ("the vector mask SPR has no PTO spelling, and not because this backend stops "
                 "short: SetVectorMask / ResetMask / SetVectorMaskByCount write state that PTO's "
                 "model *replaces* -- a tile's valid region is where a partial extent lives -- so "
                 "there is nothing for it to set (RFC-0011 §7.25)")
    _UNPRINTED_OP = {
        # in scope (§4.1) and a measured gap: a row here rather than a handler, because an `op_*` attribute is
        # what `HANDLED` is derived from and a handler that only raises is not a capability
        "dma.gm_to_ub.nd": ("the multi-dimensional GM->UB gather (nddma_out_to_ub_*) has no PTO "
                            "template: nothing in the tile ISA walks GM with per-element strides "
                            "(RFC-0011 §4.1). The engine is one-sided"),
        "vec.set_mask": _MASK_SPR,
        "vec.set_mask_by_count": _MASK_SPR,
        "vec.reset_mask": _MASK_SPR,
    }

    # ---------------------------------------------------------------- tile types (RFC-0011 §3)

    def _dims2(self, op: Any, mt: MemType) -> tuple[Any, Any]:
        """A PTO tile is 2-D. Rank 1 becomes [1, n]; rank > 2 is a gap in batch 1."""
        dims = tuple(mt.dims)
        if len(dims) == 1:
            return 1, dims[0]
        if len(dims) == 2:
            return dims[0], dims[1]
        raise PtoIsaGap(op, f"a rank-{len(dims)} tile has no pto::Tile spelling (Tile is 2-D)")

    def _static(self, op: Any, d: Any, what: str) -> int:
        """PTO tile capacity is a template argument: it must be a compile-time integer.

        Shape symbols (``DimValue('M')``) fold through the module's ``bindings`` — the kernel is
        specialised per scalar valuation, exactly as the pypto_pro backend does (D-081), because
        both targets take tile shapes as compile-time quantities."""
        k = self.folder.fold(d)
        if k is not None:
            return k
        bound = "no scalar bindings were given" if not self.mp.bindings else \
            "it does not fold from the bound parameters (a cell, a core id or a loop variable)"
        raise PtoIsaGap(op, f"{what} does not fold to a compile-time int ({d!r}): {bound}. "
                            "pto::Tile capacity is a template argument")

    def _cap_cols(self, op: Any, rows: int, cols: int, esize: int, layout: Any,
                  what: str, room: int | None = None) -> int:
        """The tile capacity ``Cols`` that satisfies pto's row-pitch static_assert.

        ``pto_tile.hpp`` refuses a tile whose row pitch does not land on the 32-byte grid
        (the assert's own line moves between CANN versions -- 1376 in 9.0.0, 1510 on the board)::

            (RowMajor && NoneBox && Cols * sizeof(T) % alignedSize == 0)
            || (ColMajor && NoneBox && Rows * sizeof(T) % alignedSize == 0)
            || (SLayout != NoneBox && Cols % InnerCols == 0)

        which is UB's own rule, not a template quirk: a DMA row has to start on a 32-byte block.
        We would rather refuse at print time than emit a translation unit that cannot compile.

        A **single-row** tile may simply be widened. ``Cols`` is a pitch, and a pitch between one
        row and no other row addresses nothing; meanwhile ``addr_alloc`` has *already* rounded the
        allocation up to 32 bytes (``passes/addr_alloc.py:65``, ``ALIGN["ub"] = 32``), so the bytes
        the wider capacity claims are bytes this tile already owns. The valid region stays the real
        extent, so every transfer still moves exactly the elements it did before.

        With more than one row the pitch is load-bearing: widening it would space the rows out in
        UB, and the packed layout the kernel asked for is then simply not a shape PTO can name.
        That is a genuine gap (it is what the ``unalign_*`` family exists to exercise), so say so.
        """
        blayout, slayout = pt.LAYOUT[layout]
        if slayout != "SLayout::NoneBox":
            return cols  # fractal: the assert bounds Cols against InnerCols, which pt.LAYOUT fixes
        if blayout == "BLayout::ColMajor":
            if rows * esize % pt.ALIGN:
                raise PtoIsaGap(op, f"{what} is column-major with {rows} rows of {esize}-byte "
                                    f"elements ({rows * esize} bytes), which is not a multiple of "
                                    f"{pt.ALIGN}; pto::Tile requires a column-major tile's Rows to "
                                    "land on the 32-byte grid (pto_tile.hpp static_assert)")
            return cols
        if cols * esize % pt.ALIGN == 0:
            return cols
        if rows != 1:
            raise PtoIsaGap(op, f"{what} packs {cols} {esize}-byte elements per row "
                                f"({cols * esize} bytes), which is not a multiple of {pt.ALIGN}, so "
                                f"rows after the first start off the 32-byte grid. A pto::Tile's "
                                "row pitch is its Cols template argument and must be aligned "
                                "(pto_tile.hpp static_assert); widening it here would space the "
                                "rows out and change the layout the kernel asked for")
        grown = -(-cols * esize // pt.ALIGN) * pt.ALIGN // esize
        if room is not None and grown > room:
            raise PtoIsaGap(op, f"{what} needs its capacity widened from {cols} to {grown} elements "
                                f"to put its row pitch on the 32-byte grid (pto_tile.hpp "
                                f"static_assert), but only {room} elements are left in the "
                                "allocation past this window")
        return grown

    def tile_type(self, op: Any, mt: MemType) -> tuple[str, bool]:
        """The ``pto::Tile<...>`` template-argument list for this memory type, and whether it is
        the **flat** reading of it.

        `_cap_cols` refuses a multi-row row-major tile whose row pitch is off the 32-byte grid, and
        rightly: widening ``Cols`` would space the rows out. But there is a third reading it does
        not consider. The storage is *contiguous* -- ``addr_alloc`` lays ``rows * cols`` elements
        down with no gaps -- so ``1 x (rows * cols)`` names exactly the same bytes, has no row pitch
        to get wrong, and, being a single-row tile, may be widened freely.

        That is the same tile only for a consumer that does not read it as two-dimensional, which
        is why the flag travels with the record. The declaration is usable as a **pointer** (the
        ``@vf`` bridge of RFC-0011 §7.7 -- which is all the corpus does with these: they are the
        mxfp8 kernels' ``[16, 2]`` scale-staging buffers, indexed flatly inside the ``@vf`` body).
        A tile-shaped consumer is *not* quietly accommodated: the reuse fast paths skip a flat
        record, so such a consumer builds its own view and meets the identical refusal there.
        """
        if mt.space not in pt.LOC:
            raise PtoIsaGap(op, f"memory space {mt.space!r} is not a pto::Tile location")
        try:
            elem = pt.elem(mt.dtype)
        except KeyError as exc:
            raise PtoIsaGap(op, str(exc.args[0])) from exc
        rows, cols = self._dims2(op, mt)
        r = self._static(op, rows, "tile rows")
        c = self._static(op, cols, "tile cols")
        blayout, slayout = pt.LAYOUT[mt.layout]
        what = f"the {r}x{c} {mt.dtype.name} tile in {mt.space.upper()}"
        flat = False
        try:
            # both of pto_tile.hpp's shape asserts: the fractal divisibility for a boxed tile and
            # the 32-byte row pitch for an unboxed one. They are alternatives -- `_cap_cols`
            # returns early for a boxed layout -- and the allocation must survive whichever applies
            self._box_check(op, r, c, mt.space, slayout, mt.dtype)
            cap = self._cap_cols(op, r, c, cpp.esize(mt.dtype), mt.layout, what)
        except PtoIsaGap:
            if r == 1 and slayout == "SLayout::NoneBox":
                raise  # already flat: there is nothing left to reinterpret
            flat, r, c = True, 1, r * c
            blayout, slayout = pt.LAYOUT[None]
            cap = self._cap_cols(op, r, c, cpp.esize(mt.dtype), None, what)
        args = [pt.LOC[mt.space], elem, str(r), str(cap), blayout, str(r), str(c), slayout, pt.fractal(mt.space)]
        return f"pto::Tile<{', '.join(args)}>", flat

    # ---------------------------------------------------------------- mem.* (RFC-0011 §4.6)

    def op_mem_alloc(self, op: Any) -> None:
        try:
            self._mem_alloc(op)
        except PtoIsaGap as exc:
            # Remember *why*, so the transfers that later look for this tile can say something
            # better than "not backed by a mem.alloc this printer has seen" -- which is the
            # symptom of this refusal, never an independent one.
            self.tile_gaps[op.results[0].name] = str(exc).split(" at loc")[0]
            raise

    def _mem_alloc(self, op: Any) -> None:
        r = op.results[0]
        t = r.type
        if "addr" not in op.attrs:
            raise PtoIsaGap(op, "allocation without an address (run addr_alloc)")
        mt = t.elem if isinstance(t, BufType) else t
        if not isinstance(mt, MemType):
            raise PtoIsaGap(op, f"mem.alloc of a non-memory type {t!r}")
        name = self.name(r)
        decl, flat = self.tile_type(op, mt)
        addr = self.val(op.attrs["addr"])
        rows, cols = self._dims2(op, mt)
        slots = t.slots if isinstance(t, BufType) else 0
        if slots:
            # a slot buffer is an array of tiles; the slots are consecutive, one tile apart --
            # and "one tile" is the *aligned* slot addr_alloc reserved, not the raw element count
            # (passes/addr_alloc.py:65 rounds each slot up to the space's alignment). A tile whose
            # payload is not a whole number of blocks would otherwise have every slot but the
            # zeroth pointing short of where the allocator put it.
            raw = self._static(op, rows, "tile rows") * self._static(op, cols, "tile cols") * cpp.esize(mt.dtype)
            a = SPACE_ALIGN[mt.space]
            step = -(-raw // a) * a
            self.emit(f"{decl} {name}[{slots}];", op)
            self.emit(f"for (int _s = 0; _s < {slots}; ++_s) TASSIGN({name}[_s], ({addr}) + _s * {step});")
        else:
            self.emit(f"{decl} {name};", op)
            # the checked template form when the address is a compile-time integer: PTO's
            # static_asserts SA-0351..SA-0354 bound it against the space capacity for free
            if isinstance(op.attrs["addr"], int):
                self.emit(f"TASSIGN<{addr}>({name});")
            else:
                self.emit(f"TASSIGN({name}, {addr});")
        self.tiles[r.name] = Tile(name=name, space=mt.space, dtype=mt.dtype, shape=(rows, cols),
                                  addr=op.attrs["addr"], layout=mt.layout, slots=slots,
                                  step=step if slots else 0, flat=flat)

    def op_mem_workspace(self, op: Any) -> None:
        r = op.results[0]
        mt = r.type
        if not isinstance(mt, MemType):
            raise PtoIsaGap(op, "mem.workspace with a non-memory result")
        name = self.name(r)
        try:
            elem = pt.elem(mt.dtype)
        except KeyError as exc:
            raise PtoIsaGap(op, str(exc.args[0])) from exc
        offset = op.attrs.get("offset", 0)
        off = self.val(offset) if not isinstance(offset, int) else str(offset)
        self.mp.uses_workspace = True
        self.emit(f"__gm__ {elem}* {name}_p = (__gm__ {elem}*)(workspace + ({off}));", op)
        self.gms[r.name] = GTensor(name=name, dtype=mt.dtype, shape=tuple(mt.dims))
        # The host allocates the workspace from the manifest (HostSpec.workspace_bytes_expr), so a
        # region we print but do not declare would run against zero bytes. Both quantities must
        # therefore reach the manifest; a size the host cannot evaluate is a gap, never a silent 0.
        ws_name = str(op.attrs.get("name", r.name))
        if any(w["name"] == ws_name for w in self.mp.workspaces):
            return
        numel = self.folder.fold(op.attrs.get("numel"))
        offs = self.folder.fold(offset)
        if numel is None or offs is None:
            raise PtoIsaGap(op, f"workspace {ws_name!r}: numel / offset do not fold to compile-time "
                                "ints, and the host tiling function needs them to size the buffer")
        self.mp.workspaces.append({"name": ws_name, "dtype": mt.dtype.name, "numel": numel,
                                   "offset": offs, "dims": [_dim_str(d) for d in mt.dims]})

    # The view family prints nothing: the geometry rides to its consumer, which reads it back
    # through ``geo()`` and turns it into a tile view or a GlobalTensor displacement (§3).
    def _window(self, op: Any) -> None:
        g = self.geo(op.results[0])  # fold now so a malformed chain is caught at its own op
        # zero instructions, but never silently skipped (D-015): the comment lets a reader match
        # the generated source back to the IR op whose geometry rode into the next transfer
        r = op.results[0]
        self.emit(f"// {op.opcode} {self.name(r)} = {self.name(g.root)}"
                  f"{'' if g.strides is None else ' strided'}: folded into its consumer", op)

    op_mem_slice = _window
    op_mem_reshape = _window
    op_mem_view = _window
    op_mem_reinterpret = _window
    op_mem_get_buf = _window

    # ---------------------------------------------------------------- GM <-> UB (RFC-0011 §4.1)

    def _fold(self, op: Any, x: Any, what: str) -> int:
        """A burst quantity that lands in a **template argument**, or a located gap.

        Not every quantity in a transfer has to fold — an *extent* does not, because PTO splits a
        tile into a compile-time capacity and a valid region that may be DYNAMIC (§7.6, and see
        :meth:`_extent`). What must fold is anything that reaches the type: a tile's ``Rows`` /
        ``Cols``, which is to say every **row pitch**, because pto's own templates read the UB
        stride straight off the type (``TStoreVecND``: ``ubStride = TileData::Cols * esize``).
        A run-time pitch has nowhere to go, so it refuses rather than printing a stride the
        instruction would not honour.
        """
        k = self.folder.fold(x)
        if k is None:
            raise PtoIsaGap(op, f"{what} ({x!r}) does not fold to a compile-time int, and it reaches "
                                "a pto::Tile template argument (a row pitch is the tile type's Cols); "
                                "only a transfer *extent* may be run-time (RFC-0011 §7.6)")
        return k

    def _esize(self, op: Any, dt: DType) -> int:
        """The element size in **bytes**, which a packed sub-byte dtype does not have.

        This is byte arithmetic — addresses, row pitches, block counts — and an fp4 element is
        half a byte, so there is no integer to return and no rounding that would be safe. The
        callers that legitimately need a number for fp4 have one that is not this: `_mx_c0` takes
        the group size (32 elements, which both sources agree on) and the MX tile shapes come from
        the operands' own logical extents, where the packing is already accounted for. So this
        refuses, and the refusal is now about units rather than about §8 q6 — **q6 is closed**:
        PTO's ``kStep / KHALF = dstCol / 64`` is one whole 512-byte fractal for the corpus's
        16 x 64 fp4 tile, and both mxfp4 kernels are bit-exact on silicon printing it, where cce's
        ``ceil(n_dst / 32)`` asks for two (RFC-0011 §7.24).
        """
        if cpp.is_packed(dt):
            raise PtoIsaGap(op, f"{dt.name} is a packed sub-byte dtype: an element is half a byte "
                                "and this is byte arithmetic, so there is no size to return (the "
                                "MX path does not need one — §4.12 takes the group size and the "
                                "operands' logical extents instead)")
        return cpp.esize(dt)

    def _extent_and_cap(self, op: Any, x: Any, what: str, capacity: int | None = None) -> tuple[Any, int]:
        """``(valid extent, capacity)`` for a quantity that reaches a tile's *valid region*.

        The capacity has to be a compile-time integer whatever happens (a ``pto::Tile``'s Rows and
        Cols are template arguments, §3), but the valid region does not. When the extent folds the
        two are the same number; when it does not, the capacity is its **ceiling** — see
        :meth:`ScalarFolder.bound`, which reads it off a tail clamp. Without a ceiling there is
        nothing to declare and this refuses, naming the two different things it needed.
        """
        v = self.folder.fold(x)
        if v is not None:
            return v, v
        cap = self.folder.bound(x)
        if cap is None and capacity is not None and capacity > 0:
            # A metadata-loaded extent need not expose a syntactic bound: the
            # operand allocation already supplies capacity. Keep the runtime
            # valid extent unchanged; PTO's validshape checks still apply.
            cap = capacity
        if cap is None:
            raise PtoIsaGap(op, f"{what} ({x}) is neither a compile-time int nor a bounded "
                                "expression, so there is no capacity to declare the tile with "
                                "(its valid region could still be run-time; its Rows/Cols cannot)")
        return _Printed(self.cexpr(x)), cap

    def _extent(self, op: Any, x: Any, what: str) -> Extent:
        """A burst quantity as an :class:`Extent` — folded when it can be, a C++ expression when not.

        The expression names the local the scalar layer printed (§7.5), which is why a runtime
        extent is expressible here at all: before that layer existed there was no identifier to
        refer to.
        """
        k = self.folder.fold(x)
        if k is not None:
            return Extent(k, str(k))
        if not isinstance(x, Value):
            raise PtoIsaGap(op, f"{what} ({x!r}) is neither a constant nor a value")
        return Extent(None, self.name(x))

    def _pitch_from_storage(self, op: Any, v: Value | None, gap_key: str, esz: int) -> int:
        """The UB row pitch of a transfer whose width is run-time: the allocation's ``Cols``.

        ``device_lower`` writes the gap as ``(alloc_cols - width) / C0`` — the allocation's row
        width minus the transfer's, in C0 units (`device_lower.py:228, 239`). So the pitch it
        encodes is ``alloc_cols``, which is compile-time whatever the width does, and which this
        printer already holds as the tile record's shape. Recovering it there instead of adding
        the gap back to the burst is what makes a run-time-narrow window printable at all — and it
        is the same move the single-burst path above already makes.

        **The identity is proved, not assumed.** In bytes, ``gap*C0*esz + burst_bytes`` must come
        to the constant ``alloc_cols*esz``; both sides go through the affine form, so a gap that is
        *not* the pitch-minus-width expression keeps a variable term and refuses here rather than
        silently supplying the wrong ``Cols``.

        One precondition is cce's own rather than this backend's: the gap is a **floor** division
        by C0, so it loses information unless ``C0`` divides ``alloc_cols - width``. The allocator
        32-byte-aligns every UB row, so ``C0 | alloc_cols`` always; what is left is ``C0 | width``,
        and a transfer that fails it has already miscounted its rows *in cce*. This backend does
        not need it — a tile with ``Cols = alloc_cols`` and ``validCol = width`` is right either
        way — so the two agree exactly where cce is correct at all.
        """
        if v is None:
            raise PtoIsaGap(op, "a multi-row transfer with a run-time width needs the UB tile to "
                                "read its row pitch from, and none was passed")
        rec = self.rec_of(op, v)
        cap = self.folder.fold(rec.shape[1])
        if cap is None:
            raise PtoIsaGap(op, f"the width of {rec.name} ({rec.shape}) does not fold, so a run-time "
                                "transfer width has no compile-time row pitch to sit in; a "
                                "pto::Tile's Cols is a template argument")
        c0 = max(32 // esz, 1)
        why = self._gap_encodes(op, gap_key, cap, c0, esz)
        if why is not None:
            raise PtoIsaGap(op, f"the row gap of this transfer cannot be shown to encode "
                                f"{rec.name}'s row pitch of {cap} {rec.dtype.name} elements: {why}")
        return cap

    def _gap_encodes(self, op: Any, gap_key: str, cap: int, c0: int, esz: int) -> str | None:
        """None when ``op``'s row gap is ``(cap - width) / C0`` over this transfer's own width.

        The check is a *shape* match against `device_lower.py:228, 239`, not an algebraic identity,
        and the difference matters. Algebraically ``gap*C0 + width == cap`` is **not** provable:
        the gap is a floor division, so it holds only where ``C0 | (cap - width)``, and with a
        run-time width that is not decidable here. What is decidable is that this gap was *built*
        from this allocation's Cols and this transfer's width — which is the claim actually being
        relied on, since the pitch then comes from the allocation and the floor division never
        enters the printed code at all.
        """
        gap = op.attrs.get(gap_key, 0)
        d = self.folder.defs.get(getattr(gap, "name", ""), None)
        if d is None or d.opcode != "scalar.div":
            return f"{gap_key} is not a division (device_lower writes (Cols - width) / C0)"
        if self.folder.fold(d.operands[1]) != c0:
            return f"{gap_key} divides by {self.folder.fold(d.operands[1])}, not C0 = {c0}"
        n = self.folder.defs.get(getattr(d.operands[0], "name", ""), None)
        if n is None or n.opcode != "scalar.sub":
            return f"the numerator of {gap_key} is not a subtraction"
        if self.folder.fold(n.operands[0]) != cap:
            return (f"the numerator subtracts from {self.folder.fold(n.operands[0])}, "
                    f"not the allocation's {cap}")
        # ... and the subtrahend is this transfer's own width: `burst_len_byte` is that width
        # times the element size, which the affine form settles even when neither folds
        want, have = self.folder.linear(n.operands[1]), self.folder.linear(op.attrs.get("burst_len_byte"))
        if want is None or have is None:
            return "the transfer width is not an affine expression"
        scaled = (want[0] * esz, {k: v * esz for k, v in want[1].items()})
        if scaled != have:
            return f"the width subtracted ({scaled}) is not burst_len_byte ({have})"
        return None

    def _gm_pitch(self, op: Any, gap: Extent, burst: Extent, gap_key: str,
                  esz: int) -> Extent:
        """``gap_bytes + burst_bytes`` as an element count, for a GM side that may be run-time.

        A GM row pitch lands in a ``pto::Stride(...)`` **constructor argument**, so unlike the UB
        one it need not fold. What must hold is that the byte sum divides into whole elements —
        checked on the affine form, since both terms are ``something * sizeof(T)`` when the
        lowering built them, and a rounding here would land every row after the first at the wrong
        address.
        """
        if gap.static and burst.static:
            total = gap.value + burst.value
            if total <= 0:
                raise PtoIsaGap(op, f"a multi-row transfer has a non-positive GM row pitch "
                                    f"({total} B); PTO strides are unsigned element counts")
            if total % esz:
                raise PtoIsaGap(op, f"the GM row pitch ({total} B) is not a whole number of "
                                    f"{esz}-byte elements; PTO strides are element counts")
            return Extent(total // esz, str(total // esz))
        if esz != 1:
            a = self.folder.linear(op.attrs.get(gap_key, 0))
            b = self.folder.linear(op.attrs.get("burst_len_byte"))
            if a is None or b is None:
                raise PtoIsaGap(op, "a run-time GM row pitch is not an affine expression, so it "
                                    "cannot be shown to divide into whole elements")
            if any(t % esz for t in (a[0] + b[0], *a[1].values(), *b[1].values())):
                raise PtoIsaGap(op, f"a run-time GM row gap with a {esz}-byte element size cannot be "
                                    "shown to divide into whole elements; PTO strides count elements")
        expr = f"(({gap.expr}) + ({burst.expr}))"
        return Extent(None, expr if esz == 1 else f"({expr} / {esz})")

    def _transfer(self, op: Any, dt: DType, gm_stride_key: str, ub_stride_key: str,
                  ub_val: Value | None = None) -> tuple[Extent, Extent, Extent, int | None]:
        """``(rows, cols, gm_pitch, ub_pitch)`` in elements, from the burst descriptor.

        The stride formulas are the c310 ones ``tensorutils_cce.h`` implements: the GM side is
        ``stride_bytes + burst_bytes``, the UB side ``align32(stride_blocks * 32 + burst_bytes)``.

        ``rows`` and ``cols`` may be run-time (they become a DYNAMIC valid region, §7.6). The
        **UB pitch may not**: it is the tile type's ``Cols`` template argument, as pto's own
        ``TStoreVecND`` spells it (``ubStride = TileData::Cols * esize``). ``None`` means the
        transfer is a single burst, where no row pitch is ever applied.
        """
        esz = self._esize(op, dt)
        rows = self._extent(op, op.attrs.get("n_burst"), "n_burst")
        burst = self._extent(op, op.attrs.get("burst_len_byte"), "burst_len_byte")
        if burst.static and burst.value % esz:
            raise PtoIsaGap(op, f"burst_len_byte {burst.value} is not a whole number of {dt.name} "
                                "elements; a PTO transfer moves validCol elements per row")
        cols = (Extent(burst.value // esz, str(burst.value // esz)) if burst.static
                else Extent(None, burst.expr if esz == 1 else f"(({burst.expr}) / {esz})"))
        if rows.value == 1:
            # A single burst never applies a row pitch, so neither gap is even read here — which
            # matters, because a narrowed transfer makes both of them run-time. The tile view takes
            # its Cols from the allocation instead (pitch None): the *capacity* stays compile-time
            # while validCol moves.
            return rows, cols, cols, None
        # the GM row gap may be run-time: it lands in a ``pto::Stride(...)`` constructor argument.
        # The UB one may not — that pitch is the tile type's Cols.
        gm_gap = self._extent(op, op.attrs.get(gm_stride_key, 0), gm_stride_key)
        if not burst.static:
            # The pitch is the allocation's Cols, which the gap was *built* from — recover it there
            # rather than adding the gap back to a burst that does not fold (§8 q5 (b2)). This has
            # to precede the fold of `ub_gap` below: a run-time width makes the gap run-time too,
            # so folding it first would refuse the very case this route exists for.
            return (rows, cols, self._gm_pitch(op, gm_gap, burst, gm_stride_key, esz),
                    self._pitch_from_storage(op, ub_val, ub_stride_key, esz))
        ub_gap = self._fold(op, op.attrs.get(ub_stride_key, 0), ub_stride_key)
        ub_pitch_b = (ub_gap * 32 + burst.value + 31) & ~31
        if ub_pitch_b <= 0:
            raise PtoIsaGap(op, f"a multi-row transfer has a non-positive UB row pitch ({ub_pitch_b} B); "
                                "PTO strides are unsigned element counts")
        if ub_pitch_b % esz:
            raise PtoIsaGap(op, f"the UB row pitch ({ub_pitch_b} B) is not a whole number of "
                                f"{dt.name} elements; PTO strides are element counts")
        if gm_gap.static:
            gm_pitch_b = gm_gap.value + burst.value
            if gm_pitch_b <= 0:
                raise PtoIsaGap(op, f"a multi-row transfer has a non-positive GM row pitch "
                                    f"({gm_pitch_b} B); PTO strides are unsigned element counts")
            if gm_pitch_b % esz:
                raise PtoIsaGap(op, f"the GM row pitch ({gm_pitch_b} B) is not a whole number of "
                                    f"{dt.name} elements; PTO strides are element counts")
            gm_pitch = Extent(gm_pitch_b // esz, str(gm_pitch_b // esz))
        elif esz == 1:
            gm_pitch = Extent(None, f"(({gm_gap.expr}) + {burst.value})")
        else:
            # a run-time byte gap only divides into elements if it is a multiple of the element
            # size; that cannot be proven here, so the stride stays a gap rather than a rounding
            gm_pitch = Extent(None, f"((({gm_gap.expr}) + {burst.value}) / {esz})")
            if (burst.value % esz) != 0:
                raise PtoIsaGap(op, f"a run-time GM row gap with a {dt.name} element size cannot be "
                                    "shown to divide into whole elements; PTO strides count elements")
        return rows, cols, gm_pitch, ub_pitch_b // esz

    def _gm_base(self, op: Any, v: Value, elem: str | None = None) -> tuple[str, str]:
        """``(C++ __gm__ pointer expression, element type)`` for a GM value's folded window.

        The pointer is cast to the **window's** element type when that differs from the one the
        parameter (or workspace) was declared with. Two things need it: ``GlobalTensor``'s
        constructor takes `__gm__ DType*` exactly and there is no implicit conversion between
        pointer types, and the offset below is a count of *this window's* elements — added to a
        pointer of some other width it would address the wrong place. A `mem.reinterpret` over a
        `u8` parameter is the ordinary way our kernels hand a byte buffer to a typed transfer,
        so the case is not exotic.
        """
        g = self.geo(v)
        if g.space not in ("gm", "ws"):
            raise PtoIsaGap(op, f"expected a GM operand, got space {g.space!r}")
        if g.slot is not None:
            raise PtoIsaGap(op, "a GM slot-buffer ring (GMBuff) has no GlobalTensor spelling yet")
        if elem is None:
            try:
                elem = pt.elem(g.dtype)
            except KeyError as exc:
                raise PtoIsaGap(op, str(exc.args[0])) from exc
        root = self.gms.get(g.root.name)
        base = f"{root.name}_p" if root is not None else self.name(g.root)
        declared = root.dtype if root is not None else getattr(g.root.type, "dtype", None)
        if declared is None or pt.ELEM.get(declared.name) != elem:
            base = f"((__gm__ {elem}*){base})"
        off_b = views.byte_offset(g)
        esz = self._esize(op, g.dtype)
        off_e = views.div_exact(off_b, esz)
        if off_e is None:
            raise PtoIsaGap(op, "the GM window origin is not a whole number of elements")
        k = self.folder.fold(off_e) if not isinstance(off_e, int) else off_e
        if k is not None:
            return (base if k == 0 else f"({base} + {k})"), elem
        # a run-time origin is ordinary pointer arithmetic on the GM side — nothing in the tile
        # type depends on it (unlike the UB pitch), so it needs no folding at all. It names the
        # locals the scalar layer prints (§7.5).
        return f"({base} + ({self.cexpr(off_e)}))", elem

    def _gtensor(self, op: Any, v: Value, rows: Extent, cols: Extent, pitch: Extent,
                 layout: str = "ND") -> str:
        """Declare a rank-5 ``GlobalTensor`` over ``v``'s window and return its C++ name.

        When any extent is run-time the shape and stride move into constructor arguments over a
        ``Shape<-1,...>`` / ``Stride<-1,...>`` type — pto's own spelling
        (``tests/npu/a5/.../tadd_kernel.cpp``). Both sides have to move together: a5's TSTORE
        asserts ``validCol == gShape4`` at run time (``npu/a5/TStore.hpp:194``), so a dynamic tile
        against a static GM shape would trip that assert rather than transfer.

        ``DN`` describes the **same logical window** as ``ND`` — ``rows`` by ``cols`` — held
        column-major: the pitch moves from ``stride[3]`` to ``stride[4]``, which is precisely what
        ``TLoadCubeND2NZ`` reads for a DN source (`tload_common.hpp:200`). cce says the same thing
        the other way round, "GM holds the transposed matrix: GM [N, M] with row stride N_src"
        (`tensorutils_cce.h:1039`); one description, two conventions for which axis is named first.
        """
        base, elem = self._gm_base(op, v)
        name = f"_g{self.mp.next_id()}"
        one = Extent(1, "1")
        # the leading dims are 1: their strides are unobservable and need only be consistent
        block = (Extent(rows.value * pitch.value, str(rows.value * pitch.value))
                 if rows.static and pitch.static else Extent(None, f"(({rows}) * ({pitch}))"))
        shape = (one, one, one, rows, cols)
        stride = ((block, block, block, one, pitch) if layout == "DN"
                  else (block, block, block, pitch, one))
        if all(x.static for x in (*shape, *stride)):
            ty = (f"pto::GlobalTensor<{elem}, {self._dims5('Shape', shape)}, "
                  f"{self._dims5('Stride', stride)}, pto::Layout::{layout}>")
            self.emit(f"{ty} {name}({base});", op)
            return name
        sh, st = self._dims5("Shape", shape), self._dims5("Stride", stride)
        args = ", ".join(str(x) for x in shape if not x.static)
        sargs = ", ".join(str(x) for x in stride if not x.static)
        ty = f"pto::GlobalTensor<{elem}, {sh}, {st}, pto::Layout::{layout}>"
        self.emit(f"{ty} {name}({base}, {sh}({args}), {st}({sargs}));", op)
        return name

    #: The NZ fractal's row count — ``FRACTAL_NZ_ROW`` in pto, the 16 of cce's ``M_pad * 16``.
    NZ_FRACTAL = 16

    def _gtensor_nz(self, op: Any, v: Value, m_pad: int, n: int, c0: int) -> str:
        """The rank-5 NZ ``GlobalTensor`` an accumulator store writes into: the **plane**.

        ``[1, N/C0, M_pad/16, 16, C0]``, with the strides of the fractal layout it describes: one
        fractal column is ``M_pad * C0`` elements, one fractal ``16 * C0``, one row ``C0``.
        ``staticShape[3] == 16`` and ``staticShape[4]`` are static_asserts (`tstore_common.hpp:70,
        73`), so both are spelled as literals; ``TStoreAccNZ`` reads no stride at all (its one
        stride parameter, ``gStride0``, is unused), so the strides are here to describe the memory
        rather than to be consumed.

        **DIM_2 is the destination plane's fractal-row count, not the transfer's**, and that is a
        deliberate divergence from what pto's own ``PTO_ASSERT(validRow == gShape2 * gShape3)``
        says — see :meth:`op_dma_l0c_to_gm_nz2nz` for why, and why it is the reading that makes
        the emitted instruction cce's.
        """
        base, elem = self._gm_base(op, v)
        name = f"_g{self.mp.next_id()}"
        e = Extent
        rows_f, cols_f = m_pad // self.NZ_FRACTAL, n // c0
        shape = (e(1, "1"), e(cols_f, str(cols_f)), e(rows_f, str(rows_f)),
                 e(self.NZ_FRACTAL, str(self.NZ_FRACTAL)), e(c0, str(c0)))
        col_stride = m_pad * c0
        stride = (e(cols_f * col_stride, str(cols_f * col_stride)), e(col_stride, str(col_stride)),
                  e(self.NZ_FRACTAL * c0, str(self.NZ_FRACTAL * c0)), e(c0, str(c0)), e(1, "1"))
        ty = (f"pto::GlobalTensor<{elem}, {self._dims5('Shape', shape)}, "
              f"{self._dims5('Stride', stride)}, pto::Layout::NZ>")
        self.emit(f"{ty} {name}({base});", op)
        return name

    def _gtensor_dn(self, op: Any, v: Value, m: int, n: int, m_dst: Extent) -> str:
        """The rank-5 ``Layout::NCHW`` ``GlobalTensor`` a *transposed* accumulator store writes to.

        Not a typo, and §4.4's refusal was wrong about it. `TStoreAcc` really has no ``DN`` arm —
        but the arm it has for ``NCHW`` **is** the nz2dn store, term for term with cce's
        (`tstore_common.hpp:196-219` against `tensorutils_cce.h:1468`)::

            nz2dnEn = 1 at Xt[62]                     copy_matrix_cc_to_gm(..., nz2dn = true)
            set_channel_para(loop0SrcStride << 48)    set_channel_para(1 << 48)
              with loop0SrcStride = 1
            set_loop3_para(loop3Num = 1)              set_loop3_para(1)
            mSize = validRow, nSize = validCol        M, N
            dstStride = gStride2                      M_dst
            srcStride = TileData::Rows                align16(M_src)

        An NCHW store *is* a transposed store; the two coincide because there is only one hardware
        instruction under both names. So the shape is the one NCHW's own assert asks for —
        ``validCol == gShape2`` and ``validRow == gShape1 * gShape3 * gShape4`` — and the
        destination row stride goes in ``stride[2]``, which is the only stride the arm reads.
        ``CheckStaticAcc`` lists NCHW among the layouts ``TSTORE(Acc2GM)`` supports
        (`npu/a5/TStore.hpp:126`), so this needs no bypass: it is `TSTORE`, with the layout that
        selects the instruction.
        """
        base, elem = self._gm_base(op, v)
        name = f"_g{self.mp.next_id()}"
        e = Extent
        one = e(1, "1")
        # [1, 1, N, 1, M]: gShape2 = validCol, and gShape1 * gShape3 * gShape4 = validRow
        shape = (one, one, e(n, str(n)), one, e(m, str(m)))
        # only stride[2] is read (`dstStride`); the rest describe the plane so the type is honest
        stride = (e(None, f"(({m_dst}) * {n})") if not m_dst.static
                  else e(m_dst.value * n, str(m_dst.value * n)),
                  e(None, f"(({m_dst}) * {n})") if not m_dst.static
                  else e(m_dst.value * n, str(m_dst.value * n)),
                  m_dst, m_dst, one)
        sh, st = self._dims5("Shape", shape), self._dims5("Stride", stride)
        ty = f"pto::GlobalTensor<{elem}, {sh}, {st}, pto::Layout::NCHW>"
        if all(x.static for x in (*shape, *stride)):
            self.emit(f"{ty} {name}({base});", op)
            return name
        args = ", ".join(str(x) for x in shape if not x.static)
        sargs = ", ".join(str(x) for x in stride if not x.static)
        self.emit(f"{ty} {name}({base}, {sh}({args}), {st}({sargs}));", op)
        return name

    @staticmethod
    def _dims5(kind: str, dims: tuple[Extent, ...]) -> str:
        """``pto::Shape<...>`` / ``pto::Stride<...>``: the literal where it folds, ``-1`` where not.

        Spelled in full at the constructor as well as in the type. ``pto::Shape(a, b, ...)`` on its
        own is CTAD, and since every parameter defaults to ``DYNAMIC`` it deduces the all-dynamic
        specialisation — which is a *different type* from the tensor's ``Shape`` as soon as one dim
        is static, so the constructor argument would not convert.
        """
        inner = ", ".join(str(x.value) if x.static else "-1" for x in dims)
        return f"pto::{kind}<{inner}>"

    def _tile_view(self, op: Any, v: Value, rows: Extent, cols: Extent, pitch: int | None,
                   space: str = "ub", elem: str | None = None) -> str:
        """A ``pto::Tile`` object of the geometry this transfer needs, over ``v``'s storage.

        Reuses the tile ``mem.alloc`` declared when it already has that geometry; otherwise binds
        a second view at the window's address — which is legal because a PTO tile is a view, not
        storage (RFC-0011 §3).

        Capacity and valid region are separate questions (§7.6). The capacity (``Rows`` / ``Cols``)
        is always compile-time: ``Cols`` is the row pitch, ``Rows`` the allocation's. The valid
        region is static when both extents fold, and ``DYNAMIC`` — set from constructor arguments —
        when either does not. Dynamic is all-or-nothing here because that is the form pto's own
        kernels use (``Tile<..., -1, -1>`` then ``t(vRows, vCols)``); a mixed static/dynamic valid
        region is not exercised upstream and is not invented here.
        """
        g = self.geo(v)
        rec = self.rec_of(op, v)
        if rec.space != space:
            raise PtoIsaGap(op, f"this move targets {space.upper()} but {rec.name} lives in "
                                f"{rec.space.upper()}")
        if rec.layout not in (None, "nd", "nz"):
            raise PtoIsaGap(op, f"{rec.name} is laid out {rec.layout!r}, which has no unboxed "
                                "(RowMajor, NoneBox) reading")
        off_b = views.byte_offset(g)
        off = off_b if isinstance(off_b, int) else self.folder.fold(off_b)
        addr = rec.addr if isinstance(rec.addr, int) else self.folder.fold(rec.addr)
        if addr is None:
            raise PtoIsaGap(op, "the tile's own address does not fold, so a displaced view cannot be bound")
        cap_rows = self.folder.fold(rec.shape[0])
        cap_cols = self.folder.fold(rec.shape[1])
        if cap_rows is None or cap_cols is None:
            raise PtoIsaGap(op, f"the capacity of {rec.name} ({rec.shape}) does not fold; a pto::Tile's "
                                "Rows/Cols are template arguments even when its valid region is dynamic")
        # The row pitch IS the view's Cols (pto reads ubStride off the type). A single-burst
        # transfer applies no pitch, so its Cols is free, and the tightest capacity that still
        # holds the valid region is the transfer's own width — which also keeps TASSIGN's static
        # bounds check honest for a *displaced* window. Only when the width is run-time does the
        # capacity have to come from the storage instead: what is left of it past this window.
        view_rows = rows.value if rows.static else cap_rows
        # What the allocation actually holds, in elements: `addr_alloc` rounds every slot up to
        # the space's alignment (addr_alloc.py:65), and those trailing bytes belong to this tile --
        # nothing else was placed in them. Both the widening below and the overrun check further
        # down measure against this rather than the logical shape.
        allocation_esz = cpp.esize(rec.dtype)
        esz = cpp.esize(g.dtype)
        _a = SPACE_ALIGN[rec.space]
        held = (-(-cap_rows * cap_cols * allocation_esz // _a) * _a) // esz
        # A reinterpret changes the element count, not the backing bytes. PTO
        # requires the transfer's tile and GM dtypes to agree, not the dtype
        # originally used to reserve this storage.
        if (cap_cols * allocation_esz) % esz:
            raise PtoIsaGap(op, "the reinterpreted tile row is not a whole number of elements", owner="ours")
        cap_cols = cap_cols * allocation_esz // esz
        if pitch is not None:
            view_cols = pitch
        elif cols.static:
            view_cols = cols.value
        else:
            left = None if off is None else held - off // esz
            view_cols = cap_cols if left is None else min(cap_cols, left)
        # Same 32-byte pitch rule as tile_type. The ceiling is what is left past this window, or
        # the whole slot when the origin is run-time and there is nothing to subtract. The rule
        # itself does not depend on the origin -- `Cols * esize % 32` is a question about the
        # *type* -- so skipping it for a run-time window (which this did) only moved the failure
        # from a located refusal to `pto_tile.hpp`'s static_assert two stages later.
        view_cols = self._cap_cols(op, view_rows, view_cols, esz, None,
                                   f"the {view_rows}x{view_cols} {g.dtype.name} view of "
                                   f"{rec.name}",
                                   room=held if off is None else held - off // esz)
        if view_cols <= 0:
            raise PtoIsaGap(op, f"the tile view would have {view_cols} columns; the window starts "
                                f"past the end of {rec.name} ({cap_rows}x{cap_cols} {rec.dtype.name})")
        if (not rec.slots and not rec.flat and off == 0 and rows.static and cols.static
                and rec.shape == (rows.value, cols.value) and cap_cols == view_cols
                and g.dtype.name == rec.dtype.name and elem in (None, pt.elem(rec.dtype))):
            return rec.name  # the whole tile, exactly this geometry: no second view needed
        # A run-time window origin (or a ring counter) moves only the address. `pl` refuses that
        # outright (coverage §2 #7 — its make_tile(addr=) must be a compile-time integer); TASSIGN
        # has a runtime-address form (docs/isa/TASSIGN.md Form 1) and the scalar layer prints the
        # C++ local the offset names, so the ISA layer expresses what pl could not. The tile's
        # *shape* stays compile-time either way.
        addr_expr = self.tile_addr(op, g, rec, addr, off, off_b)
        if elem is None:
            try:
                elem = pt.elem(g.dtype)
            except KeyError as exc:
                raise PtoIsaGap(op, str(exc.args[0])) from exc
        if cols.static and cols.value > view_cols:
            raise PtoIsaGap(op, f"the transfer moves {cols} elements per row but the tile row pitch is "
                                f"{view_cols}; validCol must not exceed the tile's Cols")
        # the view must fit the storage mem.alloc reserved. cce prints the burst either way and
        # would read past the tile; a Tile is a *type*, so here the overrun would be a lie in the
        # template arguments — refuse instead (PTO would catch it at run time via
        # PTO_ASSERT(validCol == gShape4), and TASSIGN's bounds check only sees the space).
        #
        # A run-time extent cannot be checked here at all; PTO's own run-time assert is what
        # guards it, exactly as it guards a hand-written kernel.
        if off is not None and rows.static:
            have = held - (off // esz)
            need = rows.value * view_cols
            if need > have:
                raise PtoIsaGap(op, f"the transfer needs {need} {g.dtype.name} elements from "
                                    f"{rec.name}, which holds {have} past this window "
                                    f"({cap_rows}x{cap_cols} {rec.dtype.name}, {held} elements once "
                                    "the allocator's alignment padding is counted); the tile view "
                                    "would over-declare its storage")
        name = f"_t{self.mp.next_id()}"
        dynamic = not (rows.static and cols.static)
        valid = ("-1", "-1") if dynamic else (str(rows), str(cols))
        # Read unboxed whatever the allocation is declared as: this move copies bytes, exactly as
        # cce's does (`gm_to_l1_pad` ignores the layout too). A tile is a view, so an NZ operand
        # can be filled through a (RowMajor, NoneBox) view of the same address and read back as NZ
        # by the cube instruction that consumes it (RFC-0011 §3).
        args = [pt.LOC[space], elem, str(view_rows), str(view_cols), "BLayout::RowMajor", *valid,
                "SLayout::NoneBox", pt.fractal(space)]
        ctor = f"({rows}, {cols})" if dynamic else ""
        self.emit(f"pto::Tile<{', '.join(args)}> {name}{ctor};", op)
        # against a run-time origin the static overrun check above cannot run; that is the level
        # of checking cce has always had (it prints the burst and computes the address), and PTO
        # still catches a mismatched extent at run time via PTO_ASSERT
        self.emit(f"TASSIGN({name}, {addr_expr});")
        return name

    def op_dma_gm_to_ub_pad(self, op: Any) -> None:
        dst, src = op.operands[:2]
        if op.attrs.get("pad") is not None:
            raise PtoIsaGap(op, "a pad value rides the tile type's PadValue template argument "
                                "(TLoad reads TileData::PadVal); wiring it is its own step")
        dt = self.geo(dst).dtype
        rows, cols, gm_pitch, ub_pitch = self._transfer(op, dt, "src_stride_byte", "dst_stride", dst)
        tile = self._tile_view(op, dst, rows, cols, ub_pitch)
        gt = self._gtensor(op, src, rows, cols, gm_pitch)
        self.emit(f"TLOAD({tile}, {gt});", op)

    def op_dma_ub_to_gm_pad(self, op: Any) -> None:
        dst, src = op.operands[:2]
        dt = self.geo(src).dtype
        rows, cols, gm_pitch, ub_pitch = self._transfer(op, dt, "dst_stride_byte", "src_stride", src)
        atomic = op.attrs.get("atomic")
        kind = getattr(atomic, "name", atomic)
        # pto's AtomicType (common/type.hpp:300) has AtomicNone and AtomicAdd only — max and min
        # have no spelling there, and a5's TSTORE_IMPL tests for AtomicAdd alone
        if kind not in (None, "none", "add"):
            raise PtoIsaGap(op, f"atomic {kind!r} store: pto::AtomicType has AtomicNone and "
                                "AtomicAdd only (common/type.hpp:300), and a5's TSTORE acts on "
                                "AtomicAdd alone")
        tile = self._tile_view(op, src, rows, cols, ub_pitch)
        gt = self._gtensor(op, dst, rows, cols, gm_pitch)
        targs = (f"<decltype({tile}), decltype({gt}), pto::AtomicType::AtomicAdd>"
                 if kind == "add" else "")
        self.emit(f"TSTORE{targs}({gt}, {tile});", op)

    # ---------------------------------------------------------------- GM -> L1 (RFC-0011 §4.2)

    #: NZ boxes 16 rows to a fractal, so an NZ tile's row capacity is a multiple of 16. cce says
    #: the same number as ``dst_nz_c0_stride = (M_dst + 15) / 16 * 16``.
    NZ_ROWS = 16

    def _mat_tile(self, op: Any, v: Value, rows: Extent, cols: Extent, m_dst: int) -> str:
        """A ``pto::Tile<TileType::Mat, ...>`` in NZ for a GM -> L1 move.

        The seam is narrower here than for GM <-> UB, and states itself: pto's ``TLoadCubeND2NZ``
        sets ``loop3DstStride = TileData::Rows`` — the NZ column-block stride — where cce passes
        ``dst_nz_c0_stride = align16(M_dst)``. So **the tile's Rows template argument is that
        stride**, not merely a capacity, and it has to be ``align16(M_dst)`` exactly. Everything
        else lines up directly: ``nValue = gShape3`` is the GM row count, ``dValue = validCol`` the
        width, ``loop1SrcStride = gStride3 * esize`` the GM row pitch.
        """
        g = self.geo(v)
        rec = self.rec_of(op, v)
        if rec.space != "l1":
            raise PtoIsaGap(op, f"a GM -> {rec.space} move is not §4.2 (which is GM -> L1)")
        if rec.layout != "nz":
            raise PtoIsaGap(op, f"{rec.name} is laid out {rec.layout!r}; an ND2NZ move needs an NZ "
                                "tile (pto's TLoadCubeCheck accepts ND -> (ColMajor, SLayout::"
                                "RowMajor) only)")
        try:
            elem = pt.elem(g.dtype)
        except KeyError as exc:
            raise PtoIsaGap(op, str(exc.args[0])) from exc
        esz = self._esize(op, g.dtype)
        if esz == 8:
            raise PtoIsaGap(op, f"{g.dtype.name} is 8 bytes; pto's TLoadCubeCheck refuses b64 for "
                                "ND2NZ / DN2NZ")
        if m_dst % self.NZ_ROWS:
            raise PtoIsaGap(op, f"the NZ destination has {m_dst} rows, which is not a multiple of "
                                f"{self.NZ_ROWS}; TileData::Rows *is* the NZ column-block stride "
                                "(pto's loop3DstStride), so it must land on a fractal boundary")
        cap_cols = self.folder.fold(rec.shape[1])
        if cap_cols is None:
            raise PtoIsaGap(op, f"the width of {rec.name} ({rec.shape}) does not fold; a pto::Tile's "
                                "Cols is a template argument")
        # pto's own L1 bounds, checked before bisheng sees them (TLoadCubeCheck)
        if m_dst > 16384:
            raise PtoIsaGap(op, f"an L1 tile of {m_dst} rows exceeds pto's limit of 16384")
        if m_dst * cap_cols * esz > 512 * 1024:
            raise PtoIsaGap(op, f"an L1 tile of {m_dst}x{cap_cols} {g.dtype.name} is "
                                f"{m_dst * cap_cols * esz} bytes, over pto's 512KB L1 static bound")
        if cols.static and cols.value > cap_cols:
            raise PtoIsaGap(op, f"the move writes {cols} columns but {rec.name} is {cap_cols} wide")
        blayout, slayout = pt.LAYOUT["nz"]
        dynamic = not (rows.static and cols.static)
        valid = ("-1", "-1") if dynamic else (str(rows), str(cols))
        args = [pt.LOC["l1"], elem, str(m_dst), str(cap_cols), blayout, *valid, slayout,
                pt.fractal("l1")]
        if (not rec.slots and not rec.flat and not dynamic
                and rec.shape == (rows.value, cols.value)
                and m_dst == rows.value and cap_cols == cols.value
                and g.dtype.name == rec.dtype.name and views.byte_offset(g) == 0):
            return rec.name  # the whole tile, exactly this geometry
        name = f"_m{self.mp.next_id()}"
        off_b = views.byte_offset(g)
        off = off_b if isinstance(off_b, int) else self.folder.fold(off_b)
        addr = rec.addr if isinstance(rec.addr, int) else self.folder.fold(rec.addr)
        if addr is None or off is None:
            raise PtoIsaGap(op, "the L1 tile's address does not fold, so a view cannot be bound")
        ctor = f"({rows}, {cols})" if dynamic else ""
        self.emit(f"pto::Tile<{', '.join(args)}> {name}{ctor};", op)
        self.emit(f"TASSIGN({name}, {self.tile_addr(op, g, rec, addr, off, off_b)});")
        return name

    def _gm_to_l1_nz(self, op: Any, gm_layout: str) -> None:
        """``dma.gm_to_l1.nd2nz`` / ``.dn2nz`` — one ``TLOAD`` into an NZ Mat tile.

        These ops state a matrix, not a burst: ``M`` x ``N`` read from GM whose rows are
        ``N_src`` apart, landing in an NZ tile ``M_dst`` rows tall. That is closer to what PTO
        wants than the UB path's burst descriptor, and the mapping is one-to-one.
        """
        dst, src = op.operands[:2]
        rows = self._extent(op, op.attrs.get("M"), "M")
        cols = self._extent(op, op.attrs.get("N"), "N")
        pitch = self._extent(op, op.attrs.get("N_src"), "N_src")
        m_dst = self._fold(op, op.attrs.get("M_dst"), "M_dst")
        tile = self._mat_tile(op, dst, rows, cols, m_dst)
        gt = self._gtensor(op, src, rows, cols, pitch, layout=gm_layout)
        self.emit(f"TLOAD({tile}, {gt});", op)

    def op_dma_gm_to_l1_nd2nz(self, op: Any) -> None:
        self._gm_to_l1_nz(op, "ND")

    def op_dma_gm_to_l1_dn2nz(self, op: Any) -> None:
        """The same ``TLOAD`` as §4.2, over a column-major source.

        ``TLoadCube`` takes ND and DN through one function and separates them in a single line —
        ``loop1SrcStride = GetByteSize<T>(layout == DN ? gStride4 : gStride3)``
        (`tload_common.hpp:196`) — so the whole difference is which stride carries the GM pitch,
        and `_gtensor` puts it where the layout says. Everything else is §4.2 term for term against
        ``gm_to_l1_dn2nz`` (`tensorutils_cce.h:1039`)::

            cce  nValue = M      dValue = N     loop3DstStride = align16(M_dst)
            PTO  nValue = gShape3  dValue = validCol  loop3DstStride = TileData::Rows

        which is the same four bindings `_mat_tile` already makes.
        """
        self._gm_to_l1_nz(op, "DN")

    #: our scale-plane carrier dtype is whatever one-byte type the kernel could build in a `@vf`
    #: body; PTO's MX load asserts the *declared* type is e8m0 on both sides
    #: (``caps::IsFP8E8M0<TileData::DType>() && caps::IsFP8E8M0<GlobalData::RawDType>()``,
    #: `TLoad.hpp:140`). cce reaches the same instruction by casting at the call, so the spelling
    #: is the cast — the same one §4.12's tiles already use.
    def op_dma_gm_to_l1_mx_scale_nd2nz(self, op: Any) -> None:
        """``TLOAD`` with an ``MX_A_ND`` source — the dense e8m0 scale plane into L1.

        cce views the ``[rows, k_groups]`` plane as ``half`` (two e8m0 ride one 16-bit lane) and
        issues a DN2NZ (`tensorutils_cce.h:1056`)::

            dst_half_cols = ceil(k_groups / 2)      src_half_cols = ceil(src_k_groups / 2)
            set_mte2_nz_para(0<<48 | dst_half_cols<<32 | 1<<16 | 1)
            copy_gm_to_cbuf_multi_dn2nz(dst, src, 0, src_half_cols * sizeof(half), 0,
                                        dst_half_cols, rows, 0, false)

        PTO reaches the identical call through ``TLoadMxCubeAND2ZZ`` (`npu/a5/TLoad.hpp:237`),
        which is where the *halving* lives — it is the instruction's own, not something this
        printer does::

            nValue = validCol >> 1        loop3DstStride = TileData::Cols >> 1
            dValue = validRow             loop1SrcStride = GetByteSize<e8m0>(gStride2)

        So the mapping is fixed with nothing left over: the tile is ``rows x k_groups`` in the
        A-side MX order, and the GlobalTensor is the plane read as ``[1, 1, rows, k_groups/2, 2]``
        with ``stride[2] = src_k_groups`` — a row-major ``[rows, src_k_groups]`` byte plane, spelled
        as the pairs the hardware moves. The trailing ``2`` is not a choice either;
        ``TLoadMxCubeCheck`` asserts ``staticShape[4] == 2`` and ``staticShape[0..1] == 1``.

        **A-side, for both operands.** cce's wrapper is one function and writes the same bytes
        whichever operand the plane belongs to; PTO's ``MX_B_ND`` arm reaches the same
        ``copy_gm_to_cbuf_multi_dn2nz`` with the *transposed* tile (``nValue = validRow >> 1``,
        ``loop3DstStride = TileData::Rows >> 1``, `TLoad.hpp:320`), which is the same instruction
        spelled over a tile whose axes are swapped. The A form is the one whose tile shape is the
        ``[rows, k_groups]`` the IR states, so it is the one printed; the side is decided later, by
        the ``dma.l1_to_l0.mx`` that reads the plane (§4.12).
        """
        dst, src = op.operands[:2]
        rows = self._fold(op, op.attrs.get("rows"), "rows")
        k_groups = self._fold(op, op.attrs.get("k_groups"), "k_groups")
        src_k = self._fold(op, op.attrs.get("src_k_groups", op.attrs.get("k_groups")),
                           "src_k_groups")
        for what, n in (("k_groups", k_groups), ("src_k_groups", src_k)):
            if n % 2:
                raise PtoIsaGap(op, f"{what} is {n}: two e8m0 ride one 16-bit lane and both sides "
                                    "count half-lanes, so an odd group count would round the two "
                                    "differently (cce ceils each independently)")
        for v, what in ((dst, "destination"), (src, "source")):
            if self._esize(op, self.geo(v).dtype) != 1:
                raise PtoIsaGap(op, f"the {what} of an MX scale load is one byte per group; "
                                    f"{self.geo(v).dtype.name} is not")
        d_rec = self.rec_of(op, dst)
        if d_rec.space != "l1":
            raise PtoIsaGap(op, f"an MX scale plane lands in L1; {d_rec.name} is in "
                                f"{d_rec.space.upper()}")
        if rows == 1:
            raise PtoIsaGap(op, "a single-row MX scale plane takes TLOAD's vector arm "
                                "(TLoadMxCubeAVector, TLoad.hpp:545), which is a plain burst and "
                                "not the DN2NZ cce issues")
        blayout, slayout = pt.MX_LAYOUT["l0a"]
        self._box_check(op, rows, k_groups, pt.MX_SPACE["l0a"], slayout, _MX_DT)
        addr = d_rec.addr if isinstance(d_rec.addr, int) else self.folder.fold(d_rec.addr)
        if addr is None:
            raise PtoIsaGap(op, f"the address of {d_rec.name} does not fold, so no view can be bound")
        g = self.geo(dst)
        off_b = views.byte_offset(g)
        off = off_b if isinstance(off_b, int) else self.folder.fold(off_b)
        tile = f"_x{self.mp.next_id()}"
        args = ["TileType::Mat", self._MX_ELEM, str(rows), str(k_groups), blayout,
                str(rows), str(k_groups), slayout, pt.FRACTAL_MX]
        self.emit(f"pto::Tile<{', '.join(args)}> {tile};", op)
        self.emit(f"TASSIGN({tile}, {self.tile_addr(op, g, d_rec, addr, off, off_b)});")
        base, _elem = self._gm_base(op, src, elem=self._MX_ELEM)
        gt = f"_g{self.mp.next_id()}"
        shape = f"pto::Shape<1, 1, {rows}, {k_groups // 2}, 2>"
        stride = f"pto::Stride<{rows * src_k}, {rows * src_k}, {src_k}, 2, 1>"
        self.emit(f"pto::GlobalTensor<{self._MX_ELEM}, {shape}, {stride}, "
                  f"pto::Layout::MX_A_ND> {gt}({base});", op)
        self.emit(f"TLOAD({tile}, {gt});", op)

    def op_dma_gm_to_l1_pad(self, op: Any) -> None:
        """A plain burst into L1 — ND2ND, the same descriptor GM <-> UB uses.

        ``gm_to_l1_pad`` and ``gm_to_ub_pad`` compute their strides with identical formulas
        (``tensorutils_cce.h``), so the seam is the one §2.1 already states; only the space
        changes. The destination is read unboxed even when its allocation is declared NZ, which is
        what cce does — the bytes are copied, and the cube instruction that later consumes the
        tile is what reads them as fractals.
        """
        dst, src = op.operands[:2]
        dt = self.geo(dst).dtype
        rows, cols, gm_pitch, l1_pitch = self._transfer(op, dt, "src_stride_byte", "dst_stride")
        tile = self._tile_view(op, dst, rows, cols, l1_pitch, space="l1")
        gt = self._gtensor(op, src, rows, cols, gm_pitch)
        self.emit(f"TLOAD({tile}, {gt});", op)

    # ---------------------------------------------------------------- UB -> L1 (RFC-0011 §4.8)

    #: the element types ``TMOV`` / ``TEXTRACT`` instantiate for: ``is_textract_supported_type``
    #: (`npu/a5/TExtract.hpp:496`, board tree), asserted by ``CommonCheck`` on the *destination*
    #: dtype and then by a second assert that the two tiles agree. Neither the unsigned types nor
    #: the integer widths above 8 bits are in it.
    _TEXTRACT_ELEM = frozenset({"int8_t", "float8_e4m3_t", "float8_e5m2_t", "hifloat8_t", "half",
                                "bfloat16_t", "float", "float4_e2m1x2_t", "float4_e1m2x2_t",
                                "float8_e8m0_t"})
    #: same-width stand-ins for a dtype outside that list, on the same footing as `_FILL_AS`: the
    #: unboxed Vec -> Mat move is a byte copy whose only use of the element type is
    #: ``dstValidRow * dstValidCol * sizeof(T) / 32`` — the block count — so a stand-in of the same
    #: width moves the same bytes. (A *converting* move would not have that licence; this one does
    #: not convert, which is why cce spells it ``copy_ubuf_to_cbuf`` over ``void*``.)
    _EXTRACT_AS = {1: "int8_t", 2: "half", 4: "float"}

    def _extract_elem(self, op: Any, dt: DType) -> str:
        """The element type a ``TMOV`` / ``TEXTRACT`` tile of dtype ``dt`` may be spelled with."""
        try:
            elem = pt.elem(dt)
        except KeyError as exc:
            raise PtoIsaGap(op, str(exc.args[0])) from exc
        if elem in self._TEXTRACT_ELEM:
            return elem
        stand_in = self._EXTRACT_AS.get(self._esize(op, dt))
        if stand_in is None:
            raise PtoIsaGap(op, f"TMOV does not instantiate for {elem} and there is no same-width "
                                "stand-in (TExtract.hpp's is_textract_supported_type)")
        return stand_in

    def op_dma_ub_to_l1(self, op: Any) -> None:
        """``TMOV`` Vec -> Mat — the same ``copy_ubuf_to_cbuf`` cce issues, single-burst only.

        ``TMOV_TILE_IMPL``'s Vec source / Mat destination branch calls ``TExtractVecToMat``
        (`TMov.hpp:682`), whose unboxed-source path is::

            blockLen = dstValidRow * dstValidCol * sizeof(T) / 32
            copy_ubuf_to_cbuf(dstPtr, srcPtr, 0, 1, blockLen, 0, 0)

        against cce's ``copy_ubuf_to_cbuf(dst, src, 0, n_burst, burst_len, src_stride,
        dst_stride)``. Three arguments are constants in PTO's version, so this prints the case
        where cce's are the same constants — and a **gapless multi-burst** counts as that case:
        with both strides 0 the bursts are contiguous, so ``n_burst`` bursts of ``burst_len``
        blocks are one burst of ``n_burst * burst_len``. That is a rewriting of the same transfer,
        not a different one.

        A real gap (either stride non-zero) has no Vec -> Mat spelling at all: the NZ branch two
        lines down does carry a source stride, but it takes it as ``SrcTileData::Rows -
        DstTileData::Rows`` — a constexpr off the two tile types — and pins ``dstStride`` to 0,
        which is `op_dma_ub_to_l1_nz`'s shape (and reached through ``TINSERT``, for the reason
        recorded there). So a strided plain copy refuses.

        The destination's *valid region* is what sets the block count; the source's is not read on
        this path. Both are spelled all the same, since they describe the same bytes.
        """
        dst, src = op.operands[:2]
        dt = self.geo(dst).dtype
        esz = self._esize(op, dt)
        bursts = self._fold(op, op.attrs.get("n_burst"), "n_burst")
        blocks = self._fold(op, op.attrs.get("burst_len"), "burst_len")
        for key in ("src_stride", "dst_stride"):
            gap = self._fold(op, op.attrs.get(key, 0), key)
            if gap:
                raise PtoIsaGap(op, f"{key} is {gap} blocks, and TExtractVecToMat's unboxed path "
                                    "hardcodes both strides to 0 (TExtract.hpp:415); a gapped "
                                    "UB->L1 block copy has no pto::Tile spelling")
        if bursts * blocks >= 1 << 16:
            raise PtoIsaGap(op, f"{bursts} gapless bursts of {blocks} blocks fold to "
                                f"{bursts * blocks}, past the uint16_t lenBurst copy_ubuf_to_cbuf "
                                "takes")
        burst_b = bursts * blocks * 32
        if burst_b % esz:
            raise PtoIsaGap(op, f"{bursts}x{blocks} blocks is not a whole number of {dt.name} "
                                "elements")
        cols = burst_b // esz
        one, c = Extent(1, "1"), Extent(cols, str(cols))
        elem = self._extract_elem(op, dt)
        s = self._tile_view(op, src, one, c, cols, elem=elem)
        d = self._tile_view(op, dst, one, c, cols, space="l1", elem=elem)
        self.emit(f"TMOV({d}, {s});", op)

    #: the element types ``TINSERT`` instantiates for (`npu/a5/TInsert.hpp:598`, board tree). The
    #: list is shorter than pto's general dtype table -- no unsigned, no 16-bit integer -- and a
    #: type outside it is a static_assert, so it is checked here rather than at bisheng.
    _TINSERT_ELEM = frozenset({"half", "bfloat16_t", "float", "int32_t", "float8_e4m3_t",
                               "float8_e5m2_t", "hifloat8_t", "int8_t", "float8_e8m0_t",
                               "float4_e2m1x2_t", "float4_e1m2x2_t"})

    def op_dma_ub_to_l1_nz(self, op: Any) -> None:
        """``TINSERT`` — UB NZ fractals into an L1 NZ tile, the one move that is not a ``TMOV``.

        ``TMOV``'s Vec -> Mat path (``TExtractVecToMat``) cannot express this family: its NZ
        branch hardcodes ``dstStride = 0`` and takes
        ``srcStride = SrcTileData::Rows - DstTileData::Rows`` as a *constexpr*, while cce needs
        ``dst_stride = align16(m_dst) - m_src`` — 64 for the corpus's dominant shape. ``TINSERT``
        is the instruction that has both, and against ``ub_to_l1_nz`` (`tensorutils_cce.h:937`)
        every term agrees (`npu/a5/TInsert.hpp:210`, ``ComputeNZBlockParams``)::

            cce   block_count = ceil(n_src / C0)     PTO  burstNum = ceil(validCol / c0Size)
                  block_len   = m_src                     burstLen = validRow
                  src_stride  = M_src - m_src              srcGap  = SrcTileData::Rows - validRow
                  dst_stride  = align16(m_dst) - m_src     dstGap  = dstRow - validRow

        with ``validRow`` / ``validCol`` read off the **source's valid region** and ``dstRow`` off
        ``DstTileData::Rows`` (`TInsert.hpp:578`). So the mapping fixes both tiles exactly:

        * the source's ``Rows`` **is** cce's ``M_src`` — the fractal-column height, the same role
          ``Rows`` plays for an NZ Mat tile (§4.2) — and its valid row is ``m_src``, which may be
          run-time because ``srcGap`` is computed from the two at run time;
        * the destination's ``Rows`` is ``align16(m_dst)``, the whole tile's NZ height, not this
          transfer's rows.

        The source must therefore keep ``CompactMode::Null``: ``Compact`` is what selects that
        row stride (``Normal`` would substitute ``align16(validRow)``, ``RowPlusOne`` that plus
        one), and only ``Null`` reads ``Rows``. That is a different use of the same template slot
        than §4.3's, which is why `_cube_tile` takes it as an argument.

        ``n_dst`` is not read; cce's own wrapper casts it to ``void``. Window origins stay in the
        address (``indexRow`` / ``indexCol`` are left 0), as everywhere else in this printer.

        **Version skew, and it bites here.** The CANN 9.0.0 tree implements the same instruction
        differently: ``srcGap`` is pinned to the ``TInsertMode`` template argument (0 or 1) and
        ``burstLen`` is rounded up to ``align16(validRow)``. Code written for one tree compiles
        against the other and moves different bytes. This prints for the board's tree — the one
        the goldens are checked against — and RFC-0011 §8 records the divergence.
        """
        dst, src = op.operands[:2]
        m_dst = self._fold(op, op.attrs.get("m_dst"), "m_dst")
        n_src = self._fold(op, op.attrs.get("n_src"), "n_src")
        # `M_src` reaches `SrcTileData::Rows`, a template argument; `m_src` reaches the valid
        # region and may stay run-time, which is the whole reason this batch is expressible.
        M_src = self._fold(op, op.attrs.get("M_src", op.attrs.get("m_src")), "M_src")
        m_src = self._extent(op, op.attrs.get("m_src"), "m_src")
        try:
            elem = pt.elem(self.geo(src).dtype)
        except KeyError as exc:
            raise PtoIsaGap(op, str(exc.args[0])) from exc
        if elem not in self._TINSERT_ELEM:
            raise PtoIsaGap(op, f"TINSERT does not instantiate for {elem} "
                                "(TInsert.hpp's dtype static_assert lists neither the unsigned "
                                "types nor 16-bit integers)")
        if m_src.static and m_src.value > M_src:
            raise PtoIsaGap(op, f"the transfer moves {m_src} rows out of a {M_src}-row source; "
                                "srcGap is SrcTileData::Rows - validRow and is unsigned")
        # `dstRow` is `DstTileData::Rows` and it carries the NZ column-block stride -- cce's
        # `align16(m_dst)`, the whole tile's height, not this transfer's rows.
        d_rows = -(-m_dst // self.NZ_ROWS) * self.NZ_ROWS
        if m_src.static and m_src.value > d_rows:
            raise PtoIsaGap(op, f"the transfer writes {m_src} rows into a {d_rows}-row "
                                "destination; TINSERT asserts indexRow + validRow <= dstRow "
                                "(TInsert.hpp:579)")
        blayout, slayout = pt.LAYOUT["nz"]
        # The source is read as NZ whatever its allocation is declared as -- it holds fractal
        # columns `M_src` rows tall, which is what `Rows` means for a (ColMajor, RowMajor) tile.
        # A tile is a view (§3), and PTO exempts a Vec tile from the fractal-row divisibility,
        # which is what admits the corpus's 33- and 65-row sources.
        s = self._cube_tile(op, src, M_src, n_src, "ub", blayout, slayout,
                            valid=(m_src.value if m_src.static else m_src, n_src), compact=None)
        d = self._cube_tile(op, dst, d_rows, n_src, "l1", blayout, slayout)
        # `indexRow` / `indexCol` have no defaults on the plain overload (`pto_instr.hpp:974`,
        # board tree) -- only the `<TInsertMode>` one defaults them -- so the window origin is
        # spelled even though it is always 0 here: the address already carries it.
        self.emit(f"TINSERT({d}, {s}, 0, 0);", op)

    #: the element types ``TEXPANDS`` instantiates for (`npu/a5/TExpandS.hpp:145`, board tree).
    #: The float8 family is absent, so an fp8 tile is filled through a same-width integer view --
    #: legal because a tile is a view (§3), and identical because ``TExpandSInstrMat`` dispatches
    #: on ``sizeof(DType)`` alone once the value is a bit pattern.
    _TEXPANDS_ELEM = frozenset({"int32_t", "uint32_t", "int16_t", "uint16_t", "int8_t", "uint8_t",
                                "half", "float", "bfloat16_t"})
    #: same-width stand-ins for the dtypes TEXPANDS has no instantiation for
    _FILL_AS = {1: "int8_t", 2: "int16_t", 4: "int32_t"}

    def op_dma_set_constant_to_l1(self, op: Any) -> None:
        """``TEXPANDS`` — fill an L1 tile with a scalar (`create_cbuf_matrix`, the same builtin).

        cce passes the block count; PTO takes it from the tile's own capacity
        (``repeatTimes = Rows * Cols * sizeof(T) / 32``), so the two describe the same transfer
        exactly when the fill covers the whole tile. All 63 corpus ops do — they zero an L1
        operand before a partial matmul writes into it — and anything else refuses by name rather
        than filling a different number of blocks.

        The two encodings of ``repeatConfig`` differ and mean the same thing: cce writes one
        repeat of ``n_blocks`` blocks, PTO ``n_blocks`` repeats of one block with a gap of 0.

        The float8 dtypes have no ``TEXPANDS`` instantiation, so such a tile is filled through a
        same-width **integer** view. That is a reinterpretation of the tile, not of the value: it
        is only taken when the value's bit pattern is what is being written, which here means
        zero — every corpus op — and a non-zero value under a substituted type refuses.
        """
        t = op.operands[0]
        g = self.geo(t)
        rec = self.rec_of(op, t)
        if rec.space != "l1":
            raise PtoIsaGap(op, f"{rec.name} is in {rec.space.upper()}; TEXPANDS's Mat form fills "
                                "an L1 tile (the Vec form is a vector op, out of scope)")
        if self.side == "vec":
            raise PtoIsaGap(op, "create_cbuf_matrix is a cube-side instruction (cce guards its "
                                "wrapper with ASCEND_IS_AIC); this op is on the vector side")
        n_blocks = self._fold(op, op.attrs.get("n_blocks"), "n_blocks")
        rows = self._fold(op, rec.shape[0], "the tile's rows")
        cols = self._fold(op, rec.shape[1], "the tile's cols")
        # The tile is the whole allocation, so the allocation states its width -- not the window,
        # which our kernels commonly re-type (a `u8` view of a bf16 L1 operand). Taking `esz` from
        # the window makes the byte count wrong by the width ratio.
        esz = self._esize(op, rec.dtype)
        total = rows * cols * esz
        held = total // pt.ALIGN
        off_b = views.byte_offset(g)
        off = off_b if isinstance(off_b, int) else self.folder.fold(off_b)
        if off != 0 or held != n_blocks or total % pt.ALIGN:
            raise PtoIsaGap(op, f"the fill covers {n_blocks} blocks at offset {off_b} of a "
                                f"{rows}x{cols} {rec.dtype.name} tile ({total} bytes); TEXPANDS "
                                "takes its repeat count from the tile's own capacity "
                                "(TExpandS.hpp:186) and has no way to say a partial one")
        if not 1 <= held <= 32767:
            raise PtoIsaGap(op, f"{held} blocks is outside TEXPANDS's repeat range of [1, 32767]")
        try:
            elem = pt.elem(rec.dtype)
        except KeyError as exc:
            raise PtoIsaGap(op, str(exc.args[0])) from exc
        val = op.attrs.get("val", 0)
        zero = val if isinstance(val, (int, float)) else self.folder.fold(val)
        if g.dtype.name != rec.dtype.name and zero != 0:
            raise PtoIsaGap(op, f"the fill is written through a {g.dtype.name} window of a "
                                f"{rec.dtype.name} tile, and the value {val!r} is not a known "
                                "zero; the two element types would disagree about what pattern to "
                                "write")
        if elem not in self._TEXPANDS_ELEM:
            if zero != 0:
                raise PtoIsaGap(op, f"TEXPANDS has no instantiation for {elem}, and the fill value "
                                    f"{val!r} is not a known zero, so a same-width integer view "
                                    "would have to reinterpret it rather than carry its bits")
            elem = self._FILL_AS.get(esz)
            if elem is None:
                raise PtoIsaGap(op, f"{g.dtype.name} is {esz} bytes wide; TEXPANDS instantiates "
                                    "for 1-, 2- and 4-byte types only")
        blayout, slayout = pt.LAYOUT[rec.layout]
        name = self._fill_tile(op, rec, rows, cols, elem, blayout, slayout)
        self.emit(f"TEXPANDS({name}, ({elem})({self.val(val, rec.dtype)}));", op)
        # A fill of an L1 tile is *not* ordered against the loads that follow it into the same
        # tile, and the autosync pass cannot know: it models this op on MTE2 because that is what
        # cce's `create_cbuf_matrix` is, so it emits no event between the fill and the
        # `copy_gm_to_cbuf` that overwrites part of the same slot. PTO reaches a different builtin
        # (`pto_create_cbuf_matrix`, `npu/a5/TExpandS.hpp:67`), and on silicon the two reorder:
        # `matmul_chunk_absmax_norm128` came out **non-deterministic** — 1.8 MB of its 2.6 MB
        # output differing between two runs of the same binary — while cce was stable and correct
        # on the identical instruction sequence. With this barrier it is byte-identical to cce
        # (§7.19). The cost is one barrier per fill; the alternative is a kernel that is right on
        # its first iteration and drifts after.
        self.emit("pipe_barrier(PIPE_MTE2);")

    def _fill_tile(self, op: Any, rec: Tile, rows: int, cols: int, elem: str,
                   blayout: str, slayout: str) -> str:
        """The whole-allocation tile TEXPANDS fills — reused when it is already this type."""
        self._box_check(op, rows, cols, rec.space, slayout, rec.dtype)
        if not rec.slots and not rec.flat and pt.elem(rec.dtype) == elem \
                and rec.shape == (rows, cols):
            return rec.name
        addr = rec.addr if isinstance(rec.addr, int) else self.folder.fold(rec.addr)
        if addr is None:
            raise PtoIsaGap(op, f"the address of {rec.name} does not fold, so no view can be bound")
        name = f"_f{self.mp.next_id()}"
        args = [pt.LOC[rec.space], elem, str(rows), str(cols), blayout, str(rows), str(cols),
                slayout, pt.fractal(rec.space)]
        self.emit(f"pto::Tile<{', '.join(args)}> {name};", op)
        self.emit(f"TASSIGN({name}, {self.tile_addr(op, self.geo(op.operands[0]), rec, addr, 0, 0)});")
        return name

    def op_dma_ub_to_l1_nd2nz(self, op: Any) -> None:
        """One ``TINSERT`` per NZ fractal column — the decomposition cce's own wrapper uses.

        PTO has no single ND -> NZ *DMA*: ``TMovToVecNd2Nz`` is Vec -> Vec, a vector-register
        rewrite that would want a scratch UB tile this kernel does not have. But cce does not use
        one instruction either — ``ub_to_l1_nd2nz`` is a loop of ``ceil(n_src / C0)``
        ``copy_ubuf_to_cbuf`` calls, one per fractal column — and ``TInsertNDImpl``'s middle
        branch computes exactly those four burst parameters when the transfer is one column wide::

            cce (per column i)                PTO, validCol = C0 and dstCols = C0
              nBurst   = m_src                  validRow
              lenBurst = 1                      validCol * sizeof(T) / 32
              srcGap   = ceil(N_src/C0) - 1     (Src::Cols - validCol) * sizeof(T) / 32
              dstGap   = 0                      (dstCols  - validCol) * sizeof(T) / 32

        Both tiles still describe the data as it is. The source view is an ``m_src x C0`` strip of
        the ``N_src``-pitch ND tile; the destination view is one NZ fractal column, which is
        contiguous ``m_src x C0`` — ND with a row pitch of C0 — so it is read unboxed, the same
        reading §4.2's byte copies already take of an NZ allocation (§3: a tile is a view).

        The tail is over-copied by both: cce's last iteration moves a whole C0-wide block whatever
        ``n_src % C0`` is, and so does this. Column ``i`` sits at ``i * C0`` in the source and at
        ``i * C0 * align16(m_dst)`` in the destination, both folded into the address as everywhere
        else here, so ``indexRow`` / ``indexCol`` stay 0.

        What must fold is the **trip count** (``n_src``) and the two pitches (``N_src``, which is
        the source tile's ``Cols``, and ``m_dst``, which sets the destination column stride);
        ``m_src`` is ``validRow`` and may be run-time.
        """
        dst, src = op.operands[:2]
        g_src, g_dst = self.geo(src), self.geo(dst)
        s_rec, d_rec = self.rec_of(op, src), self.rec_of(op, dst)
        if s_rec.space != "ub" or d_rec.space != "l1":
            raise PtoIsaGap(op, f"this move is UB -> L1 but the operands are in "
                                f"{s_rec.space.upper()} -> {d_rec.space.upper()}")
        try:
            elem = pt.elem(g_src.dtype)
        except KeyError as exc:
            raise PtoIsaGap(op, str(exc.args[0])) from exc
        if elem not in self._TINSERT_ELEM:
            raise PtoIsaGap(op, f"TINSERT does not instantiate for {elem}")
        esz = self._esize(op, g_src.dtype)
        c0 = pt.ALIGN // esz
        n_src = self._fold(op, op.attrs.get("n_src"), "n_src")
        m_dst = self._fold(op, op.attrs.get("m_dst"), "m_dst")
        n_pitch = self._fold(op, op.attrs.get("N_src", op.attrs.get("n_src")), "N_src")
        m_src = self._extent(op, op.attrs.get("m_src"), "m_src")
        if n_pitch % c0:
            raise PtoIsaGap(op, f"the source row pitch N_src is {n_pitch}, not a multiple of C0 "
                                f"({c0}); cce's srcGap is ceil(N_src/C0)-1 while TInsertNDImpl "
                                "computes (Src::Cols - validCol) * sizeof / 32, and the two agree "
                                "only on the 32-byte grid")
        d_rows = -(-m_dst // self.NZ_ROWS) * self.NZ_ROWS
        cols = -(-n_src // c0)  # cce's own loop count; its tail column is over-copied too
        s_rows = self._fold(op, s_rec.shape[0], "the source tile's rows")
        s_addr = self._view_addr(op, g_src, s_rec)
        d_addr = self._view_addr(op, g_dst, d_rec)
        self._box_check(op, s_rows, n_pitch, "ub", "SLayout::NoneBox", g_src.dtype)
        for i in range(cols):
            sv = f"_n{self.mp.next_id()}"
            args = [pt.LOC["ub"], elem, str(s_rows), str(n_pitch), "BLayout::RowMajor",
                    "-1" if not m_src.static else str(m_src), str(c0), "SLayout::NoneBox",
                    pt.fractal("ub")]
            ctor = f"({m_src}, {c0})" if not m_src.static else ""
            if not m_src.static:
                args[6] = "-1"  # the valid region is all-or-nothing (§7.6)
            self.emit(f"pto::Tile<{', '.join(args)}> {sv}{ctor};", op)
            self.emit(f"TASSIGN({sv}, {self._plus(s_addr, i * c0 * esz)});")
            dv = f"_n{self.mp.next_id()}"
            dargs = [pt.LOC["l1"], elem, str(d_rows), str(c0), "BLayout::RowMajor",
                     str(d_rows), str(c0), "SLayout::NoneBox", pt.fractal("l1")]
            self.emit(f"pto::Tile<{', '.join(dargs)}> {dv};", op)
            self.emit(f"TASSIGN({dv}, {self._plus(d_addr, i * c0 * d_rows * esz)});")
            self.emit(f"TINSERT({dv}, {sv}, 0, 0);", op)

    def _view_addr(self, op: Any, g: views.Geo, rec: Tile) -> Any:
        """The C++ address of ``g``'s window — an int when everything folds, else an expression."""
        off_b = views.byte_offset(g)
        off = off_b if isinstance(off_b, int) else self.folder.fold(off_b)
        addr = rec.addr if isinstance(rec.addr, int) else self.folder.fold(rec.addr)
        if addr is None:
            raise PtoIsaGap(op, f"the address of {rec.name} does not fold, so no view can be bound")
        text = self.tile_addr(op, g, rec, addr, off, off_b)
        return int(text) if text.isdigit() else _Printed(text)

    @staticmethod
    def _plus(addr: Any, delta: int) -> str:
        """``addr + delta`` bytes, folded when the base is a compile-time integer."""
        if isinstance(addr, int):
            return str(addr + delta)
        return str(addr) if delta == 0 else f"({addr}) + {delta}"

    # ---------------------------------------------------------------- L1 -> L0 (RFC-0011 §4.3)

    def _box_check(self, op: Any, rows: int, cols: int, space: str, slayout: str,
                   dt: DType) -> None:
        """pto_tile.hpp's fractal-divisibility assert, in the shape the header actually states it.

        A boxed tile (``SFractal != NoneBox``) is tiled by an inner box whose extents are *not*
        16x16 in general (`pto_tile.hpp:1243`)::

            SFractalSize == fractalCSize   ->  16 x 16                     (the accumulator)
            SFractal     == RowMajor       ->  16 x (32 / sizeof(T))
            SFractal     == ColMajor       ->  (32 / sizeof(T)) x 16

        and the header then asserts ``Cols % InnerCols == 0`` always, but ``Rows % InnerRows == 0``
        only for a tile that is neither ``Loc == Vec`` nor a single row (`pto_tile.hpp:1370`). Both
        exemptions matter here: an NZ *source in UB* is a Vec tile whose row capacity is the
        source fractal-column height — 33 or 65 in the corpus, deliberately one past a fractal
        boundary — and PTO admits it because a Vec tile carries no cube fractal.
        """
        if slayout == "SLayout::NoneBox":
            return
        esz = cpp.esize(dt)
        if pt.fractal(space) == pt.FRACTAL_MX:
            # the MX box is not square, and its rows are exempt the way a Vec tile's are
            # (`pto_tile.hpp:1075, 1526`): the scale plane's row count follows the data tile's
            inner_rows, inner_cols = pt.MX_INNER
        elif pt.fractal(space) == pt.FRACTAL_C:
            inner_rows, inner_cols = self.NZ_ROWS, self.NZ_ROWS
        elif slayout == "SLayout::RowMajor":
            inner_rows, inner_cols = self.NZ_ROWS, pt.ALIGN // esz
        else:
            inner_rows, inner_cols = pt.ALIGN // esz, self.NZ_ROWS
        exempt = (pt.LOC[space] == "TileType::Vec" or rows == 1
                  or pt.fractal(space) == pt.FRACTAL_MX)
        if cols % inner_cols or (rows % inner_rows and not exempt):
            raise PtoIsaGap(op, f"a {rows}x{cols} {dt.name} tile in {space.upper()} boxes into "
                                f"{inner_rows}x{inner_cols} fractals and does not tile whole ones; "
                                "pto_tile.hpp asserts Rows % InnerRows == 0 (Vec and single-row "
                                "tiles excepted) and Cols % InnerCols == 0")

    def _cube_tile(self, op: Any, v: Value, rows: int, cols: int, space: str,
                   blayout: str, slayout: str, valid: tuple[Any, Any] | None = None,
                   compact: str | None = "CompactMode::Normal") -> str:
        """One ``pto::Tile`` over an existing on-chip allocation, with the fractal order given.

        The caller decides the order, because for L0 it is not a property of the allocation: it is
        what selects TMOV's transpose branch. Reading one L1 allocation through two fractal orders
        is legal — a tile is a view, not storage (RFC-0011 §3).

        ``compact`` is what a DYNAMIC valid region is *read through*, and it is not the same
        question for every instruction. ``TExtractToA/B`` only consult the run-time region under
        ``CompactMode::Normal``, so that is the default here. ``TINSERT`` reads it directly
        through ``GetValidRow()`` and uses ``Compact`` for something else entirely — it selects
        the source's row stride (`npu/a5/TInsert.hpp:225`, board tree) — so §4.8 passes ``None``
        and keeps the template default ``CompactMode::Null``, under which that stride is
        ``SrcTileData::Rows``.

        ``valid`` separates the tile's **valid region** from its capacity, which every cube
        transfer but one can leave equal. `TMovCcToUb` is the exception: `TMOV_IMPL` reads
        ``m = src.GetValidRow()`` / ``n = src.GetValidCol()`` off the source
        (`npu/a5/TMov.hpp:481`), so the accumulator's valid region *is* that transfer's M and N
        while its ``Rows`` stays the allocation's."""
        g = self.geo(v)
        rec = self.rec_of(op, v)
        if rec.space != space:
            raise PtoIsaGap(op, f"this move wants {space.upper()} but {rec.name} is in "
                                f"{rec.space.upper()}")
        try:
            elem = pt.elem(g.dtype)
        except KeyError as exc:
            raise PtoIsaGap(op, str(exc.args[0])) from exc
        self._box_check(op, rows, cols, space, slayout, g.dtype)
        off_b = views.byte_offset(g)
        off = off_b if isinstance(off_b, int) else self.folder.fold(off_b)
        addr = rec.addr if isinstance(rec.addr, int) else self.folder.fold(rec.addr)
        if addr is None:
            raise PtoIsaGap(op, f"the address of {rec.name} does not fold, so no view can be bound")
        # A slot buffer driven by a ring counter has a run-time origin. The tile's *shape* is still
        # compile-time -- only the address moves -- and TASSIGN has a run-time address form
        # (docs/isa/TASSIGN.md Form 1), the same one _tile_view uses for a run-time window.
        addr_expr = self.tile_addr(op, g, rec, addr, off, off_b)
        name = f"_c{self.mp.next_id()}"
        v_rows, v_cols = valid if valid is not None else (rows, cols)
        dynamic = not (isinstance(v_rows, int) and isinstance(v_cols, int))
        args = [pt.LOC[space], elem, str(rows), str(cols), blayout,
                "-1" if dynamic else str(v_rows), "-1" if dynamic else str(v_cols),
                slayout, pt.fractal(space)]
        if compact is not None and (dynamic or (space in ("l0a", "l0b") and (v_rows, v_cols) != (rows, cols))):
            # For an extract, a DYNAMIC valid region is only *read* by the Compact form: with the
            # default `CompactMode::Null` the steps come from Rows / Cols and the run-time region
            # would be silently ignored. There the two travel together or not at all.
            args += ["PadValue::Null", compact]
        ctor = f"({v_rows}, {v_cols})" if dynamic else ""
        self.emit(f"pto::Tile<{', '.join(args)}> {name}{ctor};", op)
        self.emit(f"TASSIGN({name}, {addr_expr});")
        return name

    def op_dma_l1_to_l0(self, op: Any) -> None:
        """``TEXTRACT`` from an L1 Mat tile into an L0A (Left) or L0B (Right) operand.

        PTO has no transpose parameter. ``TExtractToLeft`` / ``TExtractToRight`` branch on
        ``Dst::SFractal == Src::SFractal``, and each asserts its own destination order — Left is
        ``(ColMajor, SLayout::RowMajor)``, Right is ``(RowMajor, SLayout::ColMajor)``. So the
        transposing load is spelled by *viewing the source* with the fractal order that picks the
        branch, and the two shapes swap with it. Both sides' arithmetic agree term by term
        (`TExtract.hpp` against `tensorutils_cce.h`'s `l1_to_l0`):

            not transposed   ToA: mStep=dstRow/16  kStep=dstCol/C0  srcStride=srcRow/16
                             cce: m_step=m_dst/16  k_step=n_dst/C0  src_stride=m_src/16
            transposed       ToA: mStep=dstCol/16  kStep=dstRow/C0  srcStride=srcCol/16
                             cce: m_step=m_dst/16  k_step=n_dst/C0  src_stride=m_src/16

        which fixes the tile shapes: (m_dst, n_dst) untransposed, (n_dst, m_dst) transposed — and
        ``TExtractToB``'s untransposed branch *is* ``TExtractToA``'s transposed one, so the L0B
        side swaps once more. Windows need no index: cce folds ``src_row0`` / ``src_col0`` into the
        source view's address (`cce/emit.py`) and so does this.

        Not yet run on silicon — the board was unreachable when this was written. The mapping is
        read off both implementations rather than guessed, but the first board batch is what
        settles it.
        """
        d, sname, _shapes = self._l1_to_l0_tiles(op)
        # TEXTRACT, not TMOV: both reach the same `TExtractToLeft` / `TExtractToRight` with
        # indexRow = indexCol = 0, but `TMOV_TILE_IMPL` first asserts
        # `Src::Rows == Dst::Rows && Src::Cols == Dst::Cols` (`TMov.hpp:640`) -- it moves a whole
        # tile. This op is a *window*: cce passes m_src/n_src and m_dst/n_dst separately, and
        # `Src::Rows` cannot be narrowed to agree because it carries `srcStride` (the L1 tile's NZ
        # column-block stride). TEXTRACT is the window form, TINSERT (§4.8) its write counterpart.
        self.emit(f"TEXTRACT({d}, {sname});", op)

    def _l1_to_l0_tiles(self, op: Any) -> tuple[str, str, tuple[str, int, int, int, int]]:
        """The two tiles of an L1 -> L0 fractal load, and the shapes the MX form needs too."""
        dst, src = op.operands[:2]
        _p = op.attrs.get("dst_position")
        pos = getattr(_p, "name", _p) or ""
        if pos not in pt.L0_LAYOUT:
            raise PtoIsaGap(op, f"dst_position {pos!r} is not an L0 operand side (l0a / l0b)")
        trans = bool(op.attrs.get("src_is_transpose", False))
        # The destination's extents reach `GetValidRow` / `GetValidCol`, not a template argument,
        # so they may be run-time; what must fold is their *ceiling*, which becomes the capacity.
        allocated = self.rec_of(op, dst)
        outer_cap, k_cap = (self.folder.fold(dim) for dim in allocated.shape)
        m_capacity, n_capacity = (k_cap, outer_cap) if trans else (outer_cap, k_cap)
        m_dst, m_dst_cap = self._extent_and_cap(op, op.attrs.get("m_copy", op.attrs.get("m_dst")), "m_copy" if "m_copy" in op.attrs else "m_dst", m_capacity)
        n_dst, n_dst_cap = self._extent_and_cap(op, op.attrs.get("n_dst"), "n_dst", n_capacity)
        # The source's do reach one: `srcStride = SrcTileData::Rows` is constexpr either way.
        m_src = self._fold(op, op.attrs.get("m_src"), "m_src")
        n_src = self._fold(op, op.attrs.get("n_src"), "n_src")
        blayout, slayout = pt.L0_LAYOUT[pos]
        # L0B's own ordering already swaps rows and cols once; a transpose swaps again
        swap = trans != (pos == "l0b")
        d_rows, d_cols = (n_dst_cap, m_dst_cap) if swap else (m_dst_cap, n_dst_cap)
        # Round the requested physical copy to complete destination fractals,
        # without expanding it to the entire reusable phase allocation.
        # Static sub-tiles still use their original tight physical row pitch.
        dt = dst.type.dtype
        if dt.bits >= 8:
            c0 = 32 // cpp.esize(dt)
            row_align, col_align = (16, c0) if pos == "l0a" else (c0, 16)
            grown_rows = -(-d_rows // row_align) * row_align
            grown_cols = -(-d_cols // col_align) * col_align
            raw = [self.folder.fold(dim) for dim in allocated.shape]
            if None not in raw:
                held = -(-(raw[0] * raw[1] * cpp.esize(allocated.dtype)) // SPACE_ALIGN[pos]) * SPACE_ALIGN[pos]
                if grown_rows * grown_cols * cpp.esize(dt) <= held:
                    d_rows, d_cols = grown_rows, grown_cols
        d_valid = (n_dst, m_dst) if swap else (m_dst, n_dst)
        s_rows, s_cols = (n_src, m_src) if swap else (m_src, n_src)
        src_sfrac = pt.OTHER_SFRACTAL[slayout] if trans else slayout
        src_blayout = "BLayout::RowMajor" if src_sfrac == "SLayout::ColMajor" else "BLayout::ColMajor"
        d = self._cube_tile(op, dst, d_rows, d_cols, pos, blayout, slayout, valid=d_valid)
        sname = self._cube_tile(op, src, s_rows, s_cols, "l1", src_blayout, src_sfrac)
        return d, sname, (pos, d_rows, d_cols, s_rows, s_cols)

    #: The dtype PTO's MX tiles are declared with. Our IR carries the plane as the public `u8`
    #: carrier a kernel can build in a `@vf` body — exactly as cce does, which casts
    #: `(__cbuf__ fp8_e8m0_t*)src_mx.addr` at the call (`tensorutils_cce.h:1196`) — so the tile is
    #: spelled `float8_e8m0_t` over whatever one-byte carrier the allocation happens to have.
    _MX_ELEM = "float8_e8m0_t"

    def _mx_c0(self, op: Any, dt: DType) -> int:
        """cce's ``C0 = 32 / sizeof(T)`` over the *data* dtype — the group one e8m0 covers.

        A packed sub-byte dtype takes the same 32, and this is the one place in the fp4 story
        where the two sources agree outright. cce's ``sizeof(T)`` is of the **pair** type, so
        ``C0 = 32`` (`tensorutils_cce.h:1214`); `TExtractToAmx` never looks at the data dtype at
        all — its ``DataType`` is the *scale* tile's e8m0 — and asks only that the plane be
        ``Rows/16`` by ``Cols/2`` steps (`TExtract.hpp:65`). Both land on a 16 x 2 plane for the
        corpus's 16 x 64 fp4 tile, and the arithmetic below reproduces it: 64 / 32 = 2.
        """
        esz = 1 if cpp.is_packed(dt) else self._esize(op, dt)
        if 32 % esz:
            raise PtoIsaGap(op, f"an MX group is 32 bytes and {dt.name} is {esz}")
        return 32 // esz

    def _mx_shape(self, op: Any, side: str, rows: int, cols: int, c0: int,
                  what: str) -> tuple[int, int]:
        """A scale plane's shape from its data tile's: the fractal-inner dimension over C0.

        Left divides columns and Right divides rows, which is what makes the four (side,
        transpose) combinations one rule: `TExtractToAmx` reads ``Dst::Cols`` where
        `TExtractToBmx` reads ``Dst::Rows``, and §4.3's ``swap`` has already put the data tile's
        extents in the order each side wants.
        """
        inner = cols if side == "l0a" else rows
        if inner % c0:
            raise PtoIsaGap(op, f"the {what} tile is {rows}x{cols} and its MX groups run {c0} "
                                f"elements, which does not divide {inner}; one e8m0 covers "
                                "exactly one whole group")
        return (rows, inner // c0) if side == "l0a" else (inner // c0, cols)

    def _mx_addr(self, op: Any, v: Value) -> str:
        """The scale plane of an L0 tile: its address, shifted right by 4.

        Not an allocation this compiler makes. An L0 buffer and its scale plane are mapped to each
        other by the hardware — cce writes ``load_cbuf_to_ca_mx(dst.addr / 16, …)``
        (`tensorutils_cce.h:1196`) and pypto_pro documents ``addr(scale_a) = addr(lhs_tile) >> 4``
        — so the address is arithmetic on the data tile's, including its slot.
        """
        g, rec = self.geo(v), self.rec_of(op, v)
        addr = rec.addr if isinstance(rec.addr, int) else self.folder.fold(rec.addr)
        if addr is None:
            raise PtoIsaGap(op, f"the address of {rec.name} does not fold, so its scale plane has none")
        off_b = views.byte_offset(g)
        off = off_b if isinstance(off_b, int) else self.folder.fold(off_b)
        data = self.tile_addr(op, g, rec, addr, off, off_b)
        try:
            return str(int(data) // pt.MX_ADDR_SHIFT)
        except ValueError:
            return f"(({data}) / {pt.MX_ADDR_SHIFT})"

    def _mx_tile(self, op: Any, side: str, rows: int, cols: int, addr: str) -> str:
        """One L0 MX scale-plane tile, at an address the caller computed."""
        space = pt.MX_SPACE[side]
        blayout, slayout = pt.MX_LAYOUT[side]
        self._box_check(op, rows, cols, space, slayout, _MX_DT)
        name = f"_x{self.mp.next_id()}"
        args = [pt.LOC[space], self._MX_ELEM, str(rows), str(cols), blayout,
                str(rows), str(cols), slayout, pt.fractal(space)]
        self.emit(f"pto::Tile<{', '.join(args)}> {name};", op)
        self.emit(f"TASSIGN({name}, {addr});")
        return name

    def _mx_src_tile(self, op: Any, v: Value, side: str, rows: int, cols: int) -> str:
        """The L1 side of a scale-plane move: a Mat tile in the MX fractal over ``src_mx``.

        Its fractal *order* is the destination's, not a choice: `TExtractToAmx` asserts
        ``(RowMajor, SLayout::RowMajor)`` on **both** tiles and `TExtractToBmx`
        ``(ColMajor, SLayout::ColMajor)`` (`TExtract.hpp:34, 81`). Unlike §4.3's data path there is
        no transpose branch hanging off the order, so there is nothing to select.
        """
        g, rec = self.geo(v), self.rec_of(op, v)
        if rec.space != "l1":
            raise PtoIsaGap(op, f"an MX scale source lives in L1; {rec.name} is in {rec.space.upper()}")
        if cpp.esize(g.dtype) != 1:
            raise PtoIsaGap(op, f"an MX scale plane is one byte per group; {rec.name} is {g.dtype.name}")
        blayout, slayout = pt.MX_LAYOUT[side]
        self._box_check(op, rows, cols, pt.MX_SPACE[side], slayout, _MX_DT)
        addr = rec.addr if isinstance(rec.addr, int) else self.folder.fold(rec.addr)
        if addr is None:
            raise PtoIsaGap(op, f"the address of {rec.name} does not fold, so no view can be bound")
        off_b = views.byte_offset(g)
        off = off_b if isinstance(off_b, int) else self.folder.fold(off_b)
        name = f"_x{self.mp.next_id()}"
        args = ["TileType::Mat", self._MX_ELEM, str(rows), str(cols), blayout,
                str(rows), str(cols), slayout, pt.FRACTAL_MX]
        self.emit(f"pto::Tile<{', '.join(args)}> {name};", op)
        self.emit(f"TASSIGN({name}, {self.tile_addr(op, g, rec, addr, off, off_b)});")
        return name

    def op_dma_l1_to_l0_mx(self, op: Any) -> None:
        """The data move of §4.3, plus a second ``TEXTRACT`` for the scale plane.

        cce does both in one wrapper because the plane is not addressed independently; PTO splits
        them, and the split is the whole difference. The arithmetic agrees term for term once the
        scale tile's shape is read off the data tile's (see `_mx_shape`).

        ``TEXTRACT``, not ``TMOV``, for the same reason as the data path: both reach
        ``TExtractToAmx`` / ``TExtractToBmx``, but ``TMOV_TILE_IMPL`` first asserts the two tiles'
        shapes equal (`TMov.hpp:640`) — and cce's ``mx_src_stride`` comes from ``n_src`` while its
        ``mx_dst_stride`` comes from ``n_dst``, which is exactly the pair that must differ.
        """
        src_mx = op.attrs.get("src_mx")
        if not isinstance(src_mx, Value):
            raise PtoIsaGap(op, "l1_to_l0.mx needs the scale window in 'src_mx'")
        for key in ("src_mx_row0", "src_mx_col0", "src_mx_offset_element"):
            if self._fold(op, op.attrs.get(key, 0), key):
                raise PtoIsaGap(op, f"{key} is not folded into the scale window's address here; "
                                    "cce adds it to the pointer and this printer would have to "
                                    "as well (no corpus op sets it)")
        d, sname, (side, d_rows, d_cols, s_rows, s_cols) = self._l1_to_l0_tiles(op)
        c0 = self._mx_c0(op, self.geo(op.operands[0]).dtype)
        x = self._mx_shape(op, side, d_rows, d_cols, c0, "destination")
        y = self._mx_shape(op, side, s_rows, s_cols, c0, "source")
        d_mx = self._mx_tile(op, side, x[0], x[1], self._mx_addr(op, op.operands[0]))
        s_mx = self._mx_src_tile(op, src_mx, side, y[0], y[1])
        self.emit(f"TEXTRACT({d}, {sname});", op)
        self.emit(f"TEXTRACT({d_mx}, {s_mx});", op)

    def op_dma_l1_to_l0_img2col(self, op: Any) -> None:
        """``TIMG2COL`` per C0 chunk, over a ``pto::ConvTile`` that carries the window.

        Both sides reach ``img2colv2_cbuf_to_ca`` with the same seventeen arguments in the same
        order (`npu/a5/TImg2col.hpp:31` against `tensorutils_cce.h:1254`)::

            cce  stepK = C0                   PTO  stepK = CeilAlignment(dst.GetValidCol(), C0)
                 stepM = m_ext                     stepM = dst.GetValidRow()
                 posK  = k0 + i * C0               posK  = the call's own argument
                 posM  = m0                        posM  = the call's own argument
                 strideW/H, kw/kh, dilW/H          src.GetStrideW() ... GetDilationH()
                 highFilterW/H = false             filterW > 255, filterH > 255
                 transpose = false                 src.GetTranspose()
                 fmatrixCtrl = false                FmatrixMode is an A mode
                 channelSize = c                    src.GetChannelSize()

        so the mapping is fixed by two readings. ``stepK`` must come out ``C0``, which pins the
        destination's **valid column** to ``C0`` — one C0 chunk per instruction, exactly cce's
        loop body — and ``stepM`` pins its valid row to ``m_ext``. The rest of the window is not
        passed at all: it is *stored on the source tile*, and ``FMATRIX_A_AUTO`` is what makes
        ``TIMG2COL`` write it out (`TImg2col.hpp:135`) — ``SetFmatrix`` packs
        ``fmapW | fmapH<<16 | pad[0..3]<<32,40,48,56``, which is cce's ``fmatrix`` word field for
        field with ``padList = (pad_l, pad_r, pad_t, pad_b)``; ``SetRepeat`` packs
        ``repeatStride | repeatTime<<16 | repeatMode<<24 | dstStride<<32 | dstMposition<<48``,
        which is cce's ``rpt = 1<<16 | ceil(m_ext/16)<<32`` once ``dstStride`` is set and the other
        four keep their ConvTile defaults (0, 1, 0, 0); and ``SetPadding`` writes cce's
        ``set_padding(0)`` from the default ``padValue_ = 0``.

        **One difference, and it is in the writes rather than the values.** cce sets ``fmatrix``
        and ``padding`` once before the loop and only ``l3d_rpt`` inside it; PTO's AUTO mode writes
        all three on every ``TIMG2COL``. The words are identical, so every ``img2colv2`` sees the
        same SPR state — the cost is the redundant writes, not the result.

        The loop is printed as a C++ loop rather than unrolled: only the addresses and ``posK``
        move with the chunk, and both are run-time expressions already.
        """
        dst, src = op.operands[:2]
        _p = op.attrs.get("dst_position")
        pos = getattr(_p, "name", _p) or "l0a"
        if pos != "l0a":
            raise PtoIsaGap(op, f"dst_position {pos!r}: Timg2colConvTileCheck asserts the "
                                "destination is TileType::Left (TImg2col.hpp:110), so only the A "
                                "operand has an im2col")
        g_d, g_s = self.geo(dst), self.geo(src)
        if g_d.dtype.name != g_s.dtype.name:
            raise PtoIsaGap(op, f"TIMG2COL asserts the two tiles share a dtype; {g_s.dtype.name} -> "
                                f"{g_d.dtype.name}")
        try:
            elem = pt.elem(g_d.dtype)
        except KeyError as exc:
            raise PtoIsaGap(op, str(exc.args[0])) from exc
        if elem not in self._IMG2COL_ELEM:
            raise PtoIsaGap(op, f"TIMG2COL does not instantiate for {elem} "
                                "(Timg2colConvTileCheck's dtype list, TImg2col.hpp:117)")
        esz = self._esize(op, g_d.dtype)
        c0 = pt.ALIGN // esz
        if self._fold(op, op.attrs.get("c0", c0), "c0") != c0:
            raise PtoIsaGap(op, f"c0 is {op.attrs.get('c0')} where a {g_d.dtype.name} block holds "
                                f"{c0}; PTO derives it as 32 / sizeof(T) (TImg2col.hpp:141)")
        m_ext = self._fold(op, op.attrs.get("m_ext"), "m_ext")
        k_ext = self._fold(op, op.attrs.get("k_ext"), "k_ext")
        for what, kh in (("filter", "kh"), ("filter", "kw")):
            n = self._fold(op, op.attrs.get(kh), kh)
            if n > 0xFF:
                raise PtoIsaGap(op, f"{kh} is {n}: cce passes highFilter{kh[1].upper()} = false "
                                    "unconditionally, while PTO derives it from filter > 255 "
                                    f"(TImg2col.hpp:27) — the two disagree above 255 ({what})")
        d_rec, s_rec = self.rec_of(op, dst), self.rec_of(op, src)
        if s_rec.space != "l1":
            raise PtoIsaGap(op, f"an im2col reads its feature map from L1; {s_rec.name} is in "
                                f"{s_rec.space.upper()}")
        # the ConvTile: the L1 feature map plus the window, in NC1HWC0 (`Timg2colConvTileCheck`)
        cap = self.folder.fold(s_rec.shape[0]), self.folder.fold(s_rec.shape[1])
        if None in cap:
            raise PtoIsaGap(op, f"the shape of {s_rec.name} ({s_rec.shape}) does not fold; a "
                                "ConvTile's buffer size is a template argument")
        cv = f"_v{self.mp.next_id()}"
        chan = self.val(op.attrs["c"])
        shape = f"pto::ConvTileShape<1, -1, -1, -1, {c0}>"
        self.emit(f"pto::ConvTile<TileType::Mat, {elem}, {cap[0] * cap[1] * esz}, "
                  f"pto::Layout::NC1HWC0, {shape}> {cv}("
                  f"(int64_t)((({chan}) + {c0 - 1}) / {c0}), (int64_t)({self.val(op.attrs['h'])}), "
                  f"(int64_t)({self.val(op.attrs['w'])}));", op)
        self.emit(f"TASSIGN({cv}, {self._addr_of(op, src)});")
        self.emit(f"{cv}.SetFmapH({self.val(op.attrs['h'])});")
        self.emit(f"{cv}.SetFmapW({self.val(op.attrs['w'])});")
        # cce's fmatrix word is `w | h<<16 | pad_l<<32 | pad_r<<40 | pad_t<<48 | pad_b<<56` and
        # SetFmatrix's is `fmapW | fmapH<<16 | padList[0..3]<<32,40,48,56`: the order is the list's
        for i, key in enumerate(("pad_l", "pad_r", "pad_t", "pad_b")):
            self.emit(f"{cv}.SetPadList({i}, {self.val(op.attrs.get(key, 0))});")
        for setter, key in (("SetFilterH", "kh"), ("SetFilterW", "kw"),
                            ("SetDilationH", "dil_h"), ("SetDilationW", "dil_w"),
                            ("SetStrideH", "stride_h"), ("SetStrideW", "stride_w")):
            self.emit(f"{cv}.{setter}({self.val(op.attrs.get(key, 1))});")
        self.emit(f"{cv}.SetChannelSize({chan});")
        # cce's `rpt = 1<<16 | ceil(m_ext/16)<<32`: repeatTime 1, dstStride ceil(m_ext/16), and
        # repeatStride / repeatMode / dstMposition 0 — the ConvTile's own defaults for all four
        self.emit(f"{cv}.SetDstStride({-(-m_ext // self.NZ_ROWS)});")
        # the destination: one C0 chunk, because `stepK = CeilAlignment(validCol, C0)` has to come
        # out C0 and `stepM = validRow` has to come out m_ext
        rows = -(-m_ext // self.NZ_ROWS) * self.NZ_ROWS
        blayout, slayout = pt.L0_LAYOUT["l0a"]
        self._box_check(op, rows, c0, "l0a", slayout, g_d.dtype)
        a = f"_a{self.mp.next_id()}"
        args = [pt.LOC["l0a"], elem, str(rows), str(c0), blayout, str(m_ext), str(c0), slayout,
                pt.fractal("l0a")]
        self.emit(f"pto::Tile<{', '.join(args)}> {a};", op)
        # cce's `dst_k_frac_stride = align16(m_ext) * C0` elements per chunk, and `n_chunk` of them
        step = rows * c0 * esz
        k0 = self.val(op.attrs.get("k0", 0))
        i = f"_i{self.mp.next_id()}"
        self.emit(f"for (int {i} = 0; {i} < {-(-k_ext // c0)}; ++{i}) {{")
        self.emit(f"    TASSIGN({a}, ({self._addr_of(op, dst)}) + {i} * {step});")
        self.emit(f"    TIMG2COL<decltype({a}), decltype({cv}), "
                  f"pto::SetFmatrixMode::FMATRIX_A_AUTO>({a}, {cv}, "
                  f"{self.val(op.attrs['m0'])}, ({k0}) + {i} * {c0});")
        self.emit("}")

    #: the element types ``TIMG2COL`` instantiates for (`npu/a5/TImg2col.hpp:117`).
    _IMG2COL_ELEM = frozenset({"int8_t", "uint8_t", "int16_t", "uint16_t", "int32_t", "uint32_t",
                               "half", "bfloat16_t", "float"})

    def _addr_of(self, op: Any, v: Value) -> str:
        """The C++ byte address of a value's window — the argument ``TASSIGN`` takes."""
        g, rec = self.geo(v), self.rec_of(op, v)
        addr = rec.addr if isinstance(rec.addr, int) else self.folder.fold(rec.addr)
        if addr is None:
            raise PtoIsaGap(op, f"the address of {rec.name} does not fold, so no tile can be bound")
        off_b = views.byte_offset(g)
        off = off_b if isinstance(off_b, int) else self.folder.fold(off_b)
        return self.tile_addr(op, g, rec, addr, off, off_b)

    def op_dma_l1_to_bt(self, op: Any) -> None:
        """``TMOV`` from L1 into the bias table — Mat -> Bias, one row, no fractal question.

        ``n`` is cce's *transfer length*; PTO has no such argument. ``TMovToBt`` reads
        ``dstCol = DstTileData::Cols`` (`TMov.hpp:32`) and moves that, and the enclosing
        ``TMOV_TILE_IMPL`` static_asserts the two tiles' shapes equal — so the length is the tile's
        **capacity**, and a clamped ``n`` has to be declared at its ceiling.

        Where the two differ the transfer is *wider* than cce's, which is why the ceiling is
        allowed rather than refused: the source L1 tile is the same width by the equal-shape
        assert, so nothing is read out of bounds, and the bias table beyond the matmul's N is never
        read back. The corpus's one case is a clamp that never bites (`Min(N - n0, 32)` with
        N = 64), so the two lengths agree exactly there — which the board confirms.
        """
        dst, src = op.operands[:2]
        _, n = self._extent_and_cap(op, op.attrs.get("n"), "n")
        d = self._cube_tile(op, dst, 1, n, "bt", "BLayout::RowMajor", "SLayout::NoneBox")
        sname = self._cube_tile(op, src, 1, n, "l1", "BLayout::RowMajor", "SLayout::NoneBox")
        self.emit(f"TMOV({d}, {sname});", op)

    # ---------------------------------------------------------------- L0C -> out (RFC-0011 §4.4)

    #: fixpipe riders that ride PTO template parameters rather than call arguments. None of them
    #: is wired yet, so a transfer carrying one refuses by name rather than dropping it silently.
    #: PTO's scalar-quant mode tables, restated: `GetScalarPreQuantModeGm` (`npu/a5/TStore.hpp:33`)
    #: for a GM destination, `GetScalarPreQuantMode` (`npu/a5/common.hpp:112`) for a UB one. Every
    #: pair here is the mode cce's c310 `fixpipe_quant` reaches for the same pair, which is the
    #: whole mapping: PTO computes the mode from the tile dtypes and cce from the same two dtypes.
    #:
    #: They are restated rather than trusted because a pair a table *misses* returns `NoQuant` —
    #: it compiles, and the scale is silently dropped. So a pair not in this dict is a gap.
    _QUANT_GM = {
        ("f32", "i8"): "QF322B8_PRE", ("f32", "u8"): "QF322B8_PRE",
        ("f32", "f16"): "QF322F16_PRE", ("f32", "bf16"): "QF322BF16_PRE",
        ("f32", "hif8"): "QF322HIF8_PRE", ("f32", "e4m3"): "QF322FP8_PRE",
        ("f32", "f32"): "QF322F32_PRE",
        ("i32", "i8"): "REQ8", ("i32", "u8"): "REQ8",
        ("i32", "f16"): "DEQF16", ("i32", "bf16"): "QS322BF16_PRE",
    }

    #: These two UB selector defects are repaired by passing an explicit mode
    #: to PTO's own TMovCcToUb helper. Both ordinary overloads compile but
    #: silently use the wrong scale, so they need targeted dispatch.
    _QUANT_UB_BROKEN = {
        ("f32", "f32"): "GetScalarPreQuantMode has no float -> float arm at all, so it returns "
                        "NoQuant and the scale is dropped (the GM table does have one)",
        ("f32", "e4m3"): "GetScalarPreQuantMode returns VQF322FP8_PRE for float -> e4m3 — the "
                         "*vector* mode, which reads the FPC scale table, not the scalar SPR the "
                         "same overload just wrote",
    }

    #: The fixpipe riders that still have no spelling here. `scale` and `offset` do (§4.11); these
    #: three do not, each for its own reason:
    _FIX_RIDERS = {
        "relu": "maps 1:1 to ReluPreMode::NormalRelu (common/type.hpp:296) and would be a "
                "one-token change — but no corpus op sets relu on an L0C-out store, so it would "
                "ship unexercised on silicon, and it is *not* free-standing: the old repository "
                "refuses relu on a split-mode l0c_to_ub the same way it refuses requant there "
                "(easyasc stub_functions/cube.py:1897), a rule PTO does not state at all",
        "atomic": "PTO's AtomicType is {AtomicNone, AtomicAdd} (common/type.hpp:301), with no max "
                  "or min; cce prints atomic fixpipe stores for c220 only and refuses them on a5 "
                  "too, so there is nothing to compare against",
        "clip_relu": "a clipped relu has no PTO bound — ReluPreMode carries no threshold",
    }

    def _fixpipe(self, op: Any, dst: Value, src: Value, gm: bool, hybrid_ok: bool = False) -> str | None:
        """The fixpipe riders on an L0C-out store. Returns the ``preQuantScalar`` argument, or
        ``None`` when the store is unquantised and the plain overload applies.

        cce and PTO split the same computation in different places. cce's ``fixpipe_quant``
        returns a *pair* — a ``QuantMode_t`` and a 64-bit ``deq`` word — and its wrapper writes the
        word to the quant SPR with ``set_quant_pre``. PTO's quantised overloads compute the mode as
        a ``constexpr`` from the tile dtypes (which is why it is not an argument) and then call the
        same ``set_quant_pre`` with the word the caller passes. Same mode, from the same dtype
        pair; same SPR, from a word this backend builds with cce's own packing (``deq_scalar``).

        So the only real question is whether PTO's table has an arm for this pair, and the two
        places it does not are checked by name rather than by lookup — a missing arm returns
        ``NoQuant``, which compiles and drops the scale.
        """
        for rider, why in self._FIX_RIDERS.items():
            if op.attrs.get(rider) not in (None, False):
                raise PtoIsaGap(op, f"the fixpipe {rider!r} on an L0C-out store: {why}")
        scale = op.attrs.get("scale")
        offset = op.attrs.get("offset")
        if scale is None:
            if op.attrs.get("hif8_hybrid"):
                raise PtoIsaGap(op, "hif8_hybrid requires an explicit scale on the f32 -> hif8 GM store alone; "
                                    "an unscaled store cannot preserve hybrid rounding")
            if offset not in (None, 0):
                raise PtoIsaGap(op, f"an offset ({offset}) with no scale: cce packs the offset only "
                                    "into a scalar-quant word and reaches no quant mode without a "
                                    "scale, so it drops it silently — this refuses instead")
            return None
        s_dt, d_dt = self.geo(src).dtype.name, self.geo(dst).dtype.name
        if op.attrs.get("hif8_hybrid"):
            # cce's last hif8 arm is `q = hif8_hybrid ? QF322HIF8_PRE_HYBRID : QF322HIF8_PRE`, and
            # neither of PTO's tables has a hybrid arm — but `QuantMode_t` is not PTO's enum. It is
            # the *compiler's* (`cce_aicore_intrinsics.h:180`), both backends name the same
            # `QF322HIF8_PRE_HYBRID = 5`, and `TStoreAcc` takes the mode as a template argument.
            # What PTO lacks is the *selection*, not the mode, so §4.4's store names it (§7.30).
            if not (gm and (s_dt, d_dt) == ("f32", "hif8")):
                raise PtoIsaGap(op, f"hif8_hybrid on a scaled {s_dt} -> {d_dt} "
                                    f"{'store' if gm else 'Acc -> Vec move'}: cce's hybrid arm is "
                                    "the f32 -> hif8 GM store alone (tensorutils_cce.h:1367)")
            # ...and only a caller that goes on to *spell* the mode may have it. `_store_acc_hybrid`
            # is the ND store's; an NZ or DN store would print a plain `TSTORE` and take the
            # constexpr's `QF322HIF8_PRE`, which is the silent wrong rounding this whole arm exists
            # to avoid.
            if not hybrid_ok:
                raise PtoIsaGap(op, f"hif8_hybrid on a {self._OPCODE_LAYOUT.get(op.opcode, op.opcode)} "
                                    "accumulator store: only the ND store names "
                                    "QuantMode_t::QF322HIF8_PRE_HYBRID (§7.30); the others reach "
                                    "TSTORE, whose constexpr mode has no hybrid arm")
            self.hybrid_quant = True
        mode = self._QUANT_GM.get((s_dt, d_dt))
        if mode is None:
            raise PtoIsaGap(op, f"a scaled {s_dt} -> {d_dt} store: PTO's scalar-quant table has no "
                                f"arm for that pair, so GetScalarPreQuantMode{'Gm' if gm else ''} "
                                "would return NoQuant and drop the scale")
        # the two bits cce derives from the same pair: only its 8-bit arms carry the offset, and
        # only a signed 8-bit destination sets bit 46 (`fixpipe_quant`, tensorutils_cce.h:1367)
        b8 = mode in ("QF322B8_PRE", "REQ8")
        signed_b8 = b8 and d_dt == "i8"
        off = self.val(offset) if offset is not None else "0"
        return (f"deq_scalar<{'true' if b8 else 'false'}, {'true' if signed_b8 else 'false'}>"
                f"({self.val(scale, F32)}, {off})")

    def _acc_tile(self, op: Any, v: Value, m_src: int,
                  valid: tuple[Any, Any] | None = None) -> str:
        """The ``pto::Tile<TileType::Acc, ...>`` a store reads from.

        ``TStoreAccND`` takes ``srcStride = TileData::Rows`` — the accumulator's NZ column-block
        stride, the same role ``Rows`` plays for a Mat tile (§4.2) — so it has to be the
        allocation's row count, and 16-aligned like any NZ fractal.

        ``valid`` is a *different* pair: ``mSize`` and ``nSize``, which is to say cce's ``M`` and
        ``N``. A store of part of an accumulator has to say so — the ND path derives
        ``ndNum = validCol / gShape4`` and the NZ path asserts the equality — so the caller passes
        the transfer's extents, not the allocation's.
        """
        g = self.geo(v)
        rec = self.rec_of(op, v)
        if rec.space != "l0c":
            raise PtoIsaGap(op, f"a store from {rec.space.upper()} is not §4.4 (which is L0C out)")
        cap_cols = self.folder.fold(rec.shape[1])
        if cap_cols is None:
            raise PtoIsaGap(op, f"the width of {rec.name} ({rec.shape}) does not fold; a pto::Tile's "
                                "Cols is a template argument")
        if m_src % 16:
            raise PtoIsaGap(op, f"the accumulator has {m_src} rows, not a multiple of 16; "
                                "TileData::Rows is the NZ column-block stride")
        blayout, slayout = pt.LAYOUT["nz"]
        # The declared tile already carries `(m_src, cap_cols)` as its valid region, so a store of
        # exactly that region reuses it -- binding a narrower *view* would be the same tile spelled
        # twice, and `Rows` (the stride) is what must not move, not the view.
        if (valid in (None, (m_src, cap_cols)) and not rec.slots and not rec.flat
                and rec.shape == (m_src, cap_cols) and views.byte_offset(g) == 0):
            return rec.name
        # `CompactMode::Null`, explicitly: `Rows` is this tile's source stride (cce's
        # `align16(M_src)`), and under `Normal` both stores would substitute `align16(validRow)`
        # for it — a different number the moment M and M_src differ.
        return self._cube_tile(op, v, m_src, cap_cols, "l0c", blayout, slayout, valid,
                               compact=None)

    #: set by `_fixpipe` for the one store whose `QuantMode_t` PTO's table cannot select (§7.30)
    hybrid_quant = False
    #: for that refusal's message: which accumulator store the opcode is
    _OPCODE_LAYOUT = {"dma.l0c_to_gm.nz2nz": "NZ", "dma.l0c_to_gm.nz2dn": "DN (transposed)"}

    def _l0c_to_gm(self, op: Any, gm_layout: str) -> None:
        dst, src = op.operands[:2]
        self.hybrid_quant = False
        deq = self._fixpipe(op, dst, src, gm=True, hybrid_ok=True)
        rows = self._extent(op, op.attrs.get("M"), "M")
        cols = self._extent(op, op.attrs.get("N"), "N")
        m_src = self._fold(op, op.attrs.get("M_src"), "M_src")
        pitch_key = "N_dst" if gm_layout == "ND" else "M_pad"
        pitch = self._extent(op, op.attrs.get(pitch_key, op.attrs.get("N")), pitch_key)
        # the accumulator's valid region *is* the transfer: `mSize` / `nSize` are read off it and
        # must agree with the GlobalTensor's own shape, which is these same two extents
        acc = self._acc_tile(op, src, m_src,
                             valid=(rows.value if rows.static else rows,
                                    cols.value if cols.static else cols))
        gt = self._gtensor(op, dst, rows, cols, pitch, layout=gm_layout)
        if self.hybrid_quant:
            self._store_acc_hybrid(op, gt, acc, deq)
            return
        # `atomicType` and `reluPreMode` both default on the `preQuantScalar` overload, and the
        # `STPhase` variants cannot deduce their leading parameter, so the call stays unqualified
        self.emit(f"TSTORE({gt}, {acc});" if deq is None else f"TSTORE({gt}, {acc}, {deq});", op)

    def _store_acc_hybrid(self, op: Any, gt: str, acc: str, deq: str) -> None:
        """``TStoreAcc`` named directly, because ``TSTORE`` picks the quant mode and cannot be told.

        ``TSTORE_IMPL``'s scalar-quant overload is two statements — ``set_quant_pre(word)`` and
        ``TStoreAcc<GlobalData, TileData, quantPre, ...>(dst.data(), src.data(), the five shapes,
        the five strides, GetValidRow(), GetValidCol())`` — with ``quantPre`` a **constexpr**
        ``GetScalarPreQuantModeGm<L0cT, DstT>()`` (`npu/a5/TStore.hpp:367`). Every part of that is
        right for this store except the constexpr, whose table has no hybrid arm, so this restates
        the two statements with the mode named. `CheckStaticAcc` goes with them, so the static
        contract TSTORE would have enforced still is.
        """
        self.emit(f"pto::CheckStaticAcc<decltype({acc}), decltype({gt}), true>();", op)
        self.emit(f"set_quant_pre({deq});")
        self.emit(f"pto::TStoreAcc<decltype({gt}), decltype({acc}), "
                  f"QuantMode_t::QF322HIF8_PRE_HYBRID>({gt}.data(), {acc}.data(), "
                  f"{', '.join(f'{gt}.GetShape(pto::GlobalTensorDim::DIM_{i})' for i in range(5))}, "
                  f"{', '.join(f'{gt}.GetStride(pto::GlobalTensorDim::DIM_{i})' for i in range(5))}, "
                  f"{acc}.GetValidRow(), {acc}.GetValidCol());")

    def op_dma_l0c_to_gm_nz2nd(self, op: Any) -> None:
        """``TSTORE`` an accumulator to GM in ND — `TStoreAcc` dispatches on the GlobalTensor's
        layout, so ND here is what selects ``TStoreAccND`` (`npu/a5/TStore.hpp`), whose
        ``mSize`` / ``nSize`` are the valid region and whose ``dstD`` is the GM row pitch."""
        self._l0c_to_gm(op, "ND")

    #: ``TStoreAccNZ``'s ``c0Size``, read off *this* store's two dtypes (`tstore_common.hpp:91`).
    #: It is the 16 of cce's ``M_pad * 16`` for destinations wider than one byte, so the mapping
    #: holds exactly where the two agree — see :meth:`op_dma_l0c_to_gm_nz2nz`.
    def _nz_c0(self, op: Any, s_dt: DType, d_dt: DType) -> int:
        if self._esize(op, d_dt) == 1:
            raise PtoIsaGap(op, f"an NZ accumulator store to {d_dt.name}: TStoreAccNZ doubles its "
                                "destination stride for a 1-byte destination (dstStride <<= 1, "
                                "tstore_common.hpp:102), as cce does since I039, but no PTO ISA "
                                "store to a 1-byte NZ destination has been measured")
        if self._esize(op, s_dt) == 1:
            raise PtoIsaGap(op, f"an accumulator of {s_dt.name}: TStoreAccNZ reads c0Size = 32 off "
                                "a 1-byte *source* (tstore_common.hpp:92) where cce's stride does "
                                "not depend on the accumulator's width at all")
        # An f32 -> f32 store with gShape4 == 8 is pto's channel-split leg (c0Size = 8,
        # channelSplitEn = 1); cce's nz2nz never sets channel split, so the plane is described
        # with C0 = 16, which its own GM view already uses and which the static_assert's f32
        # clause explicitly allows.
        return self.NZ_FRACTAL

    def op_dma_l0c_to_gm_nz2nz(self, op: Any) -> None:
        """``TSTORE`` an accumulator to GM in NZ — and the one place a pto assert is left behind.

        **The shape.** An NZ ``GlobalTensor`` is the *fractal* decomposition, not the matrix:
        ``staticShape[3] == FRACTAL_NZ_ROW`` and ``staticShape[4] == 32 / sizeof(dst)`` are
        static_asserts (`tstore_common.hpp:70, 73`), so the shape is ``[1, N/C0, ?, 16, C0]`` where
        §4.4's ND path uses ``[1, 1, 1, M, N]``. This printer was emitting the ND shape under
        ``Layout::NZ``, a translation unit that cannot compile; it went unnoticed because the
        compile gate was truncating its diagnostics behind the expected ones (§7.21) *and* because
        the four corpus ops sat in kernels with no golden until §8 q5 closed.

        **What ``?`` is, which is the whole question.** ``dstStride = align16(gShape2 * gShape3) *
        c0Size`` — the distance between adjacent fractal *columns* — and cce passes ``M_pad * 16``,
        the destination **plane's** padded height, as an argument of its own
        (`tensorutils_cce.h:1453`). So ``gShape2 * gShape3`` has to be ``M_pad``. But pto also
        states ``PTO_ASSERT(validRow == gShape2 * gShape3)``, and ``validRow`` is ``mSize``, the
        transfer's rows: pto models the NZ ``GlobalTensor`` as *the transfer*, cce as *the plane*,
        and every corpus op is a strip of a taller one (M = 16 or 32 rows into planes 48, 64 and
        448 rows tall).

        **This prints the plane and leaves the assert behind**, deliberately, and the reason it is
        not a guess is that the resulting instruction is cce's term for term:

        ============ ============================== ==============================
        ``xm``/``xt`` cce                            pto with the plane's shape
        ============ ============================== ==============================
        ``nSize``     ``N``                          ``validCol``            = N
        ``mSize``     ``M``                          ``validRow``            = M
        ``dstStride`` ``M_pad * 16``                 ``align16(M_pad) * c0Size``
        ``srcStride`` ``align16(M_src)``             ``TileData::Rows``
        ============ ============================== ==============================

        The last two are equalities this printer *checks* rather than assumes: ``_acc_tile``
        pins ``Rows`` to ``align16(M_src)`` under ``CompactMode::Null``, ``_nz_c0`` refuses every
        dtype pair where ``c0Size`` is not cce's literal 16, and ``M_pad`` must fold and be
        16-aligned so that ``align16(M_pad) == M_pad``. What is left is one debug assert, which is
        ``((void)0)`` unless ``_DEBUG`` is defined (`common/debug.h:33`) — so the divergence is a
        *model* difference, not a run-time one, and the board is the arbiter. It is bit-exact
        there (§7.23).

        ``validCol == gShape0 * gShape1 * gShape4`` — the other assert — is satisfied, and holding
        it is why ``N`` must divide by ``C0``.
        """
        dst, src = op.operands[:2]
        deq = self._fixpipe(op, dst, src, gm=True)
        m_src = self._fold(op, op.attrs.get("M_src"), "M_src")
        m = self._extent(op, op.attrs.get("M"), "M")
        n = self._fold(op, op.attrs.get("N"), "N")
        m_pad = self._fold(op, op.attrs.get("M_pad", op.attrs.get("M")), "M_pad")
        c0 = self._nz_c0(op, self.geo(src).dtype, self.geo(dst).dtype)
        if m_pad % self.NZ_FRACTAL:
            raise PtoIsaGap(op, f"the destination plane is {m_pad} rows tall, not a multiple of "
                                f"{self.NZ_FRACTAL}: TStoreAccNZ rounds gShape2 * gShape3 up "
                                "(tstore_common.hpp:101) where cce passes M_pad * 16 unrounded, so "
                                "the two strides would differ by the padding")
        if n % c0:
            raise PtoIsaGap(op, f"{n} columns do not divide into C0 = {c0}: TStoreAccNZ asserts "
                                "validCol == gShape0 * gShape1 * gShape4, and gShape1 is the "
                                "fractal-column count")
        acc = self._acc_tile(op, src, m_src, valid=(m.value if m.static else m, n))
        gt = self._gtensor_nz(op, dst, m_pad, n, c0)
        self.emit(f"TSTORE({gt}, {acc});" if deq is None else f"TSTORE({gt}, {acc}, {deq});", op)

    def op_dma_l0c_to_gm_nz2dn(self, op: Any) -> None:
        """``TSTORE`` into a ``Layout::NCHW`` GlobalTensor, which **is** the nz2dn store (§7.30).

        This refused, and the refusal was a good reading of the wrong question. ``TStoreAcc``'s
        dispatch really has no ``DN`` arm and no ``else``, so a ``Layout::DN`` GlobalTensor really
        would expand to nothing — but "does `Layout::DN` work" is not "can PTO issue this
        instruction". ``TStoreAccNCHW`` sets ``nz2dnEn = 1`` at ``Xt[62]``, writes
        ``set_channel_para(1 << 48)`` and ``set_loop3_para(1)``, and is otherwise cce's
        ``l0c_to_gm_nz2dn`` field for field. One instruction, two names; see `_gtensor_dn` for the
        comparison and the shape it fixes.

        What this looks for, and what the earlier reading did not: the *instruction*, by the SPRs
        it writes, rather than the *layout*, by its name.
        """
        dst, src = op.operands[:2]
        deq = self._fixpipe(op, dst, src, gm=True)
        m = self._fold(op, op.attrs.get("M"), "M")
        n = self._fold(op, op.attrs.get("N"), "N")
        m_src = self._fold(op, op.attrs.get("M_src", op.attrs.get("M")), "M_src")
        m_dst = self._extent(op, op.attrs.get("M_dst", op.attrs.get("M")), "M_dst")
        if n % self.NZ_FRACTAL:
            raise PtoIsaGap(op, f"{n} accumulator columns: CheckStaticAcc requires TileData::Cols "
                                f"% 16 == 0 for a non-ND store (npu/a5/TStore.hpp:130)")
        acc = self._acc_tile(op, src, m_src, valid=(m, n))
        gt = self._gtensor_dn(op, dst, m, n, m_dst)
        self.emit(f"TSTORE({gt}, {acc});" if deq is None else f"TSTORE({gt}, {acc}, {deq});", op)

    def _l0c_to_l1_dtypes(self, op: Any, s_dt: DType, d_dt: DType) -> int:
        """The three dtype conditions an L0C -> L1 store holds under, and its ``c0``.

        Each is a place where cce's arithmetic and ``TExtractAccToMat``'s agree only for part of
        the dtype table; see `op_dma_l0c_to_l1` for the term-by-term comparison they come from.
        """
        if s_dt.name == "f32" and d_dt.name == "f32":
            return 8  # TINSERT selects channel splitting for an NZ FP32 Mat tile.
        esz = self._esize(op, d_dt)
        if esz != 2:
            raise PtoIsaGap(op, f"an L0C -> L1 store into {d_dt.name}: PTO's dstStride is "
                                f"DstTileData::Rows * (32 / {esz}) where cce's is align16(M_dst) * "
                                "16 (a literal, not derived from the dtype), so the two agree only "
                                "for a two-byte destination")
        if s_dt.name != "f32":
            raise PtoIsaGap(op, f"a {s_dt.name} -> {d_dt.name} L0C -> L1 store: "
                                "GetCastPreQuantMode reads the destination dtype alone and would "
                                "return F322F16/F322BF16, while cce's q reads both and returns "
                                "NoQuant (tensorutils_cce.h:1492)")
        return pt.ALIGN // esz

    def op_dma_l0c_to_l1(self, op: Any) -> None:
        """Select the Acc -> Mat instruction matching the CCE transfer descriptor.

        FP32 -> FP16/BF16 uses TEXTRACT: TExtractAccToMat keeps the valid M,
        uses a 16-element destination fractal and selects the narrowing mode.
        FP32 -> FP32 uses TINSERT: TInsertAccToMat selects channel splitting
        and an eight-element destination fractal. Unlike TEXTRACT, it copies
        SrcTileData::Rows, so it is valid only for a whole-row transfer.
        The partial-row FP32 form remains a located upstream limitation.

        Both forms round N to destination fractals; reject an unaligned N
        instead of writing padding the IR did not request. Source and
        destination row capacities retain their respective physical strides.
        ReluPreMode is a template parameter of either instruction.
        """
        dst, src = op.operands[:2]
        for rider in ("scale", "offset"):
            if op.attrs.get(rider) not in (None, 0):
                raise PtoIsaGap(op, f"a fixpipe {rider!r} on an L0C -> L1 store: cce's l0c_to_l1 "
                                    "takes relu and nothing else, so there is no cce behaviour for "
                                    "this rider to match")
        f32_insert = self.geo(src).dtype.name == self.geo(dst).dtype.name == "f32"
        c0 = self._l0c_to_l1_dtypes(op, self.geo(src).dtype, self.geo(dst).dtype)
        m, _ = self._extent_and_cap(op, op.attrs.get("M"), "M")
        n = self._fold(op, op.attrs.get("N"), "N")
        if n % c0:
            raise PtoIsaGap(op, f"N is {n}, not a whole number of {c0}-element groups: PTO sends "
                                "ceil(N / c0) * c0 where cce sends N")
        m_dst = self._fold(op, op.attrs.get("M_dst", op.attrs.get("M")), "M_dst")
        m_src = self._fold(op, op.attrs.get("M_src", op.attrs.get("M")), "M_src")
        if f32_insert and (not isinstance(m, int) or m != m_src):
            raise PtoIsaGap(op, "partial-row FP32 L0C -> L1: TINSERT/TMOV select channel splitting but copy SrcTileData::Rows; TEXTRACT keeps valid rows but disables channel splitting (a5/TInsert.hpp, TMov.hpp, TExtract.hpp)", owner="upstream")
        blayout, slayout = pt.LAYOUT["nz"]
        # `DstTileData::Rows` carries cce's `align16(M_dst)` -- the destination's NZ column-block
        # stride, the same role Rows plays in §4.8 -- and the valid region carries M and N.
        d_rows = -(-m_dst // self.NZ_ROWS) * self.NZ_ROWS
        d = self._cube_tile(op, dst, d_rows, n, "l1", blayout, slayout, valid=(m, n))
        # `SrcTileData::Rows` is cce's `align16(M_src)`; `_acc_tile` refuses a source whose rows
        # are not already 16-aligned, since the aligned figure would over-declare the allocation.
        s = self._acc_tile(op, src, m_src, valid=(m, n))
        instruction = "TINSERT" if f32_insert else "TEXTRACT"
        if op.attrs.get("relu"):
            self.emit(f"{instruction}<decltype({d}), decltype({s}), ReluPreMode::NormalRelu>"
                      f"({d}, {s}, 0, 0);", op)
        else:
            self.emit(f"TINSERT({d}, {s}, 0, 0);" if f32_insert else f"TEXTRACT({d}, {s});", op)

    # ---------------------------------------------------------------- cube compute (RFC-0011 §4.7)

    def _mmad_window(self, op: Any, dst: Value) -> None:
        """``dst_row0`` / ``dst_col0`` must be what the destination operand's own view already says.

        cce's ``_mmad`` prints only M, N and K and ignores all four ``dst_*`` attributes, which is
        correct exactly when the window they name is the one the operand is already sliced to.
        Over the corpus it is, 155 ops out of 155 — but that is the kind of agreement that should
        be checked rather than assumed, because if it ever broke the accumulator would be written
        at one address and read at another.
        """
        g = self.geo(dst)
        want = (op.attrs.get("dst_row0", 0), op.attrs.get("dst_col0", 0))
        have = g.offsets[:2] if len(g.offsets) >= 2 else (0, 0)
        for axis, w, h in zip(("row", "col"), want, have, strict=True):
            fw, fh = self.folder.fold(w), self.folder.fold(h)
            same = fw == fh if (fw is not None and fh is not None) else str(w) == str(h)
            if not same:
                raise PtoIsaGap(op, f"dst_{axis}0 says the accumulator window starts at {w}, but the "
                                    f"operand is sliced to {h}; PTO takes the window from the tile, "
                                    "so the two must agree")

    def op_cube_mmad_mx(self, op: Any) -> None:
        """``TMATMUL_MX`` — the same three forms as `op_cube_mmad`, plus the two scale tiles.

        cce's ``mmad_mx`` does not take them: the planes sit at ``l0a.addr / 16`` and
        ``l0b.addr / 16`` and ``mad_mx`` finds them there. Neither does PTO's, in the end —
        ``TMatmulMx`` passes ``mad_mx`` the three data pointers alone. The scale tiles reach
        ``TMATMUL_MX`` only so that ``CheckMadMxValid`` can type-check them, which is worth having:
        it asserts the fp4 / fp8 dtype pairs, ``TileLeft::Cols % 64 == 0`` (K a multiple of 64) and
        all three fractal orders. So they are rebuilt here from the same two addresses rather than
        carried across from §4.3's load — nothing is remembered between ops, and a mismatch would
        be a compile error rather than a silent one.
        """
        self._mmad(op, mx=True)

    def op_cube_mmad(self, op: Any) -> None:
        """``TMATMUL`` / ``TMATMUL_ACC`` / ``TMATMUL_BIAS`` (`common/pto_instr.hpp:501-563`).

        The one compute opcode this backend prints. It is here for a validation reason rather than
        a coverage one: every cube kernel in the corpus needs it, so without it §4.2-§4.4 cannot be
        compared against cce on silicon at all — they would stay a static reading.

        **M, N and K are not parameters.** ``TMATMUL_IMPL`` reads them off the operands —
        ``m = aMatrix.GetValidRow()``, ``k = aMatrix.GetValidCol()``, ``n = bMatrix.GetValidCol()``
        (`npu/a5/TMatmul.hpp:165-167`) — which is the same discipline as TLOAD/TSTORE: the tile
        carries the extent. So this prints the extents by *binding the operands to them*, and the
        instruction has no numeric arguments at all.

        The three forms follow ``is_init`` and ``bias``, the same two attributes cce branches on:

            init, no bias   TMATMUL       cmatrixInitVal = 1, C starts at zero
            not init        TMATMUL_ACC   C accumulates onto itself
            init + bias     TMATMUL_BIAS  C starts from the bias row (Bias tile, 1 x N)

        ``dst_rows`` / ``dst_cols`` are the accumulator's *capacity* and ``M`` / ``N`` its valid
        region — PTO's own split (§7.6) — so the C tile is bound with ``dst_rows``, not ``M``.

        And because the three extents are read off the operands rather than written into their
        types, they may be **run-time**: ``CheckDynamicMmad`` asserts only that each lands in
        [1, 4095] (`TMatmul.hpp:120`). A tail-clamped M binds a tile whose capacity is the clamp's
        ceiling and whose valid region is set at construction — the same shape §4.3 uses.
        """
        self._mmad(op, mx=False)

    def _l0_cap(self, op: Any, v: Value, outer: int, k_cap: int, swap: bool) -> tuple[int, int]:
        """An mmad operand's ``(Rows, Cols)`` — **the operand's own capacity**, not the extents'.

        The extent ceilings from :meth:`_extent_and_cap` bound the *valid region*; they are not the
        tile's capacity, and taking them for it was a second reading of the mistake §4.3 already
        corrected once. `conv2d_relu_n1c2h8w8_fixed` shows what it costs: its ``matmul`` asks for
        K = 18 while everything around it is padded to 32, so §4.5's load declares the L0A tile
        ``Tile<Left, half, 32, 32, ...>`` and the mmad was declaring ``32 x 18`` over the same
        bytes. Two views of one L0 allocation with different ``Rows``/``Cols`` are two different
        fractal grids — here they happened to address alike, because a ``ColMajor`` fractal's
        offset is driven by ``Rows`` alone, but that is luck, and 18 is not a whole number of
        fractals in the first place.

        The IR value carries the allocation's own shape with **K last on both sides** — L0A is
        ``(M, K)`` and L0B is ``(N, K)``, which is why ``swap`` exists — so where it folds it is
        the capacity the load used. Where it does not, the ceiling stands: a run-time dim has no
        capacity of its own and the bound is all there is.
        """
        dims = getattr(v.type, "dims", ()) or ()
        got = [d if isinstance(d, int) else self.folder.fold(d) for d in dims[:2]]
        outer_cap = got[0] if len(got) > 0 and got[0] is not None else outer
        k = got[1] if len(got) > 1 and got[1] is not None else k_cap
        rec = self.rec_of(op, v)
        dtype = v.type.dtype
        if dtype.bits >= 8:
            # A staging view may carry a logical K=3 or M=4 even though the
            # physical load writes complete fractals in its owning slot.
            # Grow only capacity, only inside that reserved byte extent.
            grown_outer = -(-outer_cap // 16) * 16
            c0 = 32 // cpp.esize(dtype)
            grown_k = -(-k // c0) * c0
            raw_dims = [self.folder.fold(dim) for dim in rec.shape]
            if None not in raw_dims:
                held = -(-(raw_dims[0] * raw_dims[1] * cpp.esize(rec.dtype)) // SPACE_ALIGN[rec.space]) * SPACE_ALIGN[rec.space]
                if grown_outer * grown_k * cpp.esize(dtype) <= held:
                    outer_cap, k = grown_outer, grown_k
        return (k, outer_cap) if swap else (outer_cap, k)

    def _mmad(self, op: Any, mx: bool) -> None:
        dst, a, b = op.operands[:3]
        # M, N and K reach `GetValidRow` / `GetValidCol`, not a template argument, so they may be
        # run-time -- `CheckDynamicMmad` asserts only that each lands in [1, 4095]. What must fold
        # is the capacity they are bounded by (§4.3's correction).
        a_capacity = [self.folder.fold(dim) for dim in self.rec_of(op, a).shape]
        b_capacity = [self.folder.fold(dim) for dim in self.rec_of(op, b).shape]
        k_capacity = min(a_capacity[1], b_capacity[1]) if None not in (a_capacity[1], b_capacity[1]) else None
        m, m_cap = self._extent_and_cap(op, op.attrs.get("M"), "M", a_capacity[0])
        n, n_cap = self._extent_and_cap(op, op.attrs.get("N"), "N", b_capacity[0])
        k, k_cap = self._extent_and_cap(op, op.attrs.get("K"), "K", k_capacity)
        d_rows = self._fold(op, op.attrs.get("dst_rows", op.attrs.get("M")), "dst_rows")
        self._mmad_window(op, dst)
        init = bool(op.attrs.get("is_init", False))
        bias = op.attrs.get("bias")
        if bias is not None and not init:
            raise PtoIsaGap(op, "a bias on a non-initialising mmad has no PTO spelling (there is no "
                                "TMATMUL_ACC_BIAS); cce drops it silently, which is worse")
        # L0A is (M, K) and L0B is (K, N) -- the same shapes §4.3 binds them to, and the pair of
        # fractal orders TMovToLeft/ToRight assert
        a_bl, a_sl = pt.L0_LAYOUT["l0a"]
        b_bl, b_sl = pt.L0_LAYOUT["l0b"]
        a_rows, a_k = self._l0_cap(op, a, m_cap, k_cap, swap=False)
        b_k, b_cols = self._l0_cap(op, b, n_cap, k_cap, swap=True)
        at = self._cube_tile(op, a, a_rows, a_k, "l0a", a_bl, a_sl, valid=(m, k))
        bt = self._cube_tile(op, b, b_k, b_cols, "l0b", b_bl, b_sl, valid=(k, n))
        ct = self._acc_tile(op, dst, d_rows)
        # the MX forms take the two scale planes as well -- rebuilt from the operands' own
        # addresses, since nothing carries them across from the load (see `op_cube_mmad_mx`)
        if mx:
            ax = self._mx_shape(op, "l0a", m_cap, k_cap, self._mx_c0(op, self.geo(a).dtype), "A")
            bx = self._mx_shape(op, "l0b", k_cap, n_cap, self._mx_c0(op, self.geo(b).dtype), "B")
            a_mx = self._mx_tile(op, "l0a", ax[0], ax[1], self._mx_addr(op, a))
            b_mx = self._mx_tile(op, "l0b", bx[0], bx[1], self._mx_addr(op, b))
            operands = f"{at}, {a_mx}, {bt}, {b_mx}"
        else:
            operands = f"{at}, {bt}"
        bias_t = None
        if bias is not None:
            bias_t = self._cube_tile(op, bias, 1, n_cap, "bt", "BLayout::RowMajor",
                                     "SLayout::NoneBox", valid=(1, n))
        if mx:
            # There is no `TMATMUL_MX_ACC`: the accumulating form is the six-argument overload
            # that names the accumulator twice, `TMATMUL_MX(cOut, cIn, a, aScale, b, bScale)`
            # (`common/pto_instr.hpp:601`), and its `TMATMUL_MX_IMPL` is the one that passes
            # `cmatrixInitVal = false`.
            if bias_t is not None:
                self.emit(f"TMATMUL_MX({ct}, {operands}, {bias_t});", op)
            elif init:
                self.emit(f"TMATMUL_MX({ct}, {operands});", op)
            else:
                self.emit(f"TMATMUL_MX({ct}, {ct}, {operands});", op)
        elif bias_t is not None:
            self.emit(f"TMATMUL_BIAS({ct}, {operands}, {bias_t});", op)
        elif init:
            self.emit(f"TMATMUL({ct}, {operands});", op)
        else:
            self.emit(f"TMATMUL_ACC({ct}, {operands});", op)

    #: our ``(dual_mode, sub_block_id)`` -> PTO's ``AccToVecMode`` (`common/type.hpp:184`).
    #: cce carries three modes and a *separate* ``sub_block_id`` bool; PTO folds the bool into the
    #: mode, which is why four values face three plus a flag —
    #: ``constexpr bool subBlockId = (mode == AccToVecMode::SingleModeVec1)`` (`TMov.hpp:180`).
    #: So a sub-block id on a dual mode has no spelling at all, and refuses rather than dropping.
    _ACC_TO_VEC = {
        ("single", 0): "AccToVecMode::SingleModeVec0",
        ("single", 1): "AccToVecMode::SingleModeVec1",
        ("splitm", 0): "AccToVecMode::DualModeSplitM",
        ("splitn", 0): "AccToVecMode::DualModeSplitN",
    }

    def op_dma_l0c_to_ub(self, op: Any) -> None:
        """``TMOV`` Acc -> Vec — ``TMovCcToUb`` (`npu/a5/TMov.hpp:173`), the same builtin cce's
        ``l0c_to_ub`` wraps. Four of cce's eight call arguments are not arguments here.

        **M and N come off the accumulator.** ``TMOV_IMPL`` reads ``m = src.GetValidRow()`` and
        ``n = src.GetValidCol()`` (`TMov.hpp:481`), so the source tile's *valid region* is the
        transfer's extent while its ``Rows`` stays the allocation's — the first cube transfer
        where the two must differ, which is why `_cube_tile` takes a ``valid``.

        **The destination contributes only its ``Cols``.** `GetTmovAccDstStride` returns
        ``DstTileData::Cols`` for the unboxed row-major view (`TMov.hpp:114`) and static_asserts it
        to a multiple of 32 — the rule `_cap_cols` already applies — and nothing else about the
        destination reaches the instruction: its valid region is never read. That is what makes the
        dual modes expressible at all. Under ``splitm`` the M rows are shared between the two
        sub-blocks, so a transfer of M=128 lands in a 64-row UB tile; under ``splitn`` the N
        columns are, so N=128 lands in one of pitch 64. Neither M nor N describes the destination.
        So the view is the *allocation's* rows by ``N_dst``, which is the only shape that is true
        of the storage and carries the stride PTO needs.

        **NZ2ND / NZ2DN / NZ2NZ is not chosen**: it follows from the destination's ``isRowMajor``
        and ``SFractal``, and the view built here is the ND one cce's wrapper means.

        **Static partial M keeps the physical source stride.** PTO's
        ``GetTmovAccSrcStride`` uses ``Src::Rows`` when a positive static
        ``ValidRow`` is smaller. Thus ``Rows=M_src, ValidRow=M`` expresses
        the partial transfer, provided its physical footprint fits the slot.
        """
        dst, src = op.operands[:2]
        deq = self._fixpipe(op, dst, src, gm=False)
        m_attr, n_attr = op.attrs.get("M"), op.attrs.get("N")
        rows = self._fold(op, m_attr, "M")
        cols = self._fold(op, n_attr, "N")
        n_dst = self._fold(op, op.attrs.get("N_dst", n_attr), "N_dst")
        src_attr = op.attrs.get("M_src", m_attr)
        m_src = self._fold(op, src_attr, "M_src")
        if m_src != rows and str(src_attr) != str(m_attr):
            geometry = self.geo(src)
            allocation = self.rec_of(op, src)
            held = math.prod(self._fold(op, extent, "L0C capacity") for extent in allocation.shape)
            held = held * self._esize(op, allocation.dtype)
            offset = views.byte_offset(geometry)
            offset = offset if isinstance(offset, int) else self.folder.fold(offset)
            needed = m_src * -(-cols // 16) * 16 * self._esize(op, geometry.dtype)
            if not 0 < rows < m_src or offset is None or offset < 0 or offset + needed > held:
                raise PtoIsaGap(op, "partial M_src/M transfer exceeds the owning L0C capacity "
                                    "or has an unbounded source offset", owner="ours")
        dm = getattr(op.attrs.get("dual_mode"), "name", op.attrs.get("dual_mode")) or "splitm"
        sub = int(op.attrs.get("sub_block_id", 0) or 0)
        if dm != "single":
            # The dual destination control and the fixpipe's scalar path are mutually exclusive in
            # the hardware, and the old repository states the rule in full at its frontend
            # (`easyasc/stub_functions/cube.py:1884`): "The fixpipe scalar quant rides on the
            # deqScalar, which is only available when the dual destination control is off (SINGLE).
            # SPLITM/SPLITN ... support neither relu nor any non-default requant parameter; they
            # only carry the SAME-TYPE plain copy (fp32 -> fp32 or int32 -> int32). Even the
            # deqScalar-free float downcasts (fp32 -> fp16/bf16) are NOT supported in split mode."
            # PTO states only the first half, as a static_assert on the *quantised* overload
            # ("Quant is not support in dual Dst Mode", `TMov.hpp:770`) — its unquantised overload
            # would compile a split-mode fp32 -> half move that the hardware does not do. So the
            # printer carries the old repository's rule, which is the stricter and the correct one.
            s_dt, d_dt = self.geo(src).dtype.name, self.geo(dst).dtype.name
            if deq is not None:
                raise PtoIsaGap(op, f"a quantised Acc -> Vec move in dual_mode {dm!r}: the scalar "
                                    "quant rides the deqScalar, which exists only with the dual "
                                    "destination control off — PTO static_asserts it too "
                                    '("Quant is not support in dual Dst Mode", TMov.hpp:770)')
            if s_dt != d_dt:
                raise PtoIsaGap(op, f"a {s_dt} -> {d_dt} Acc -> Vec move in dual_mode {dm!r}: split "
                                    "mode carries the same-type plain copy only (f32 -> f32 or "
                                    "i32 -> i32); even an unscaled float downcast needs "
                                    "dual_mode=single (easyasc stub_functions/cube.py:1884). PTO "
                                    "would compile this one — its static_assert covers the "
                                    "quantised overload only")
        mode = self._ACC_TO_VEC.get((dm, sub))
        if mode is None:
            raise PtoIsaGap(op, f"dual_mode {dm!r} with sub_block_id {sub} has no AccToVecMode: "
                                "PTO carries the sub-block id inside the mode, and only the single "
                                "modes have one")
        acc = self._acc_tile(op, src, m_src, valid=(rows, cols))
        d_rec = self.rec_of(op, dst)
        d_rows = self.folder.fold(d_rec.shape[0])
        if d_rows is None:
            raise PtoIsaGap(op, f"the row count of {d_rec.name} does not fold; a pto::Tile's Rows "
                                "is a template argument")
        ub = self._tile_view(op, dst, Extent(d_rows, str(d_rows)), Extent(n_dst, str(n_dst)),
                             n_dst, space="ub")
        pair = self.geo(src).dtype.name, self.geo(dst).dtype.name
        if deq is not None and pair in self._QUANT_UB_BROKEN:
            # As for the hybrid GM store, the PTO implementation accepts an
            # explicit QuantMode even though TMOV's dtype selector is wrong.
            self.emit(f"pto::CheckTMovAccValid<decltype({ub}), decltype({acc}), "
                      f"typename decltype({ub})::DType, typename decltype({acc})::DType, true>();", op)
            self.emit(f"set_quant_pre({deq});")
            self.emit(f"pto::TMovCcToUb<decltype({ub}), decltype({acc}), {mode}, "
                      f"QuantMode_t::{self._QUANT_GM[pair]}, ReluPreMode::NoRelu>"
                      f"({ub}.data(), {acc}.data(), {acc}.GetValidRow(), {acc}.GetValidCol());")
            return
        # the tile types precede the mode in the template parameter list; both helpers return a
        # plain variable (never an array subscript), so `decltype` is the declared type
        call = f"TMOV<decltype({ub}), decltype({acc}), {mode}>({ub}, {acc}"
        self.emit(f"{call});" if deq is None else f"{call}, {deq});", op)

    def op_dma_ub_to_ub(self, op: Any) -> None:
        """``TMOV`` Vec->Vec (``TMov.hpp:777`` -> ``TMovVecToVec``), which moves
        ``min(validSrc, validDst)`` — the equal-shape static_assert applies to Mat sources only."""
        dst, src = op.operands[:2]
        dt = self.geo(dst).dtype
        esz = self._esize(op, dt)
        rows = self._fold(op, op.attrs.get("n_burst"), "n_burst")
        blocks = self._fold(op, op.attrs.get("burst_len"), "burst_len")
        s_gap = self._fold(op, op.attrs.get("src_stride", 0), "src_stride")
        d_gap = self._fold(op, op.attrs.get("dst_stride", 0), "dst_stride")
        # the classic 32-byte-block form: burst_len and both strides count 32-byte blocks
        burst_b = blocks * 32
        if burst_b % esz:
            raise PtoIsaGap(op, f"burst_len {blocks} blocks is not a whole number of {dt.name} elements")
        cols = burst_b // esz
        src_pitch = (s_gap * 32 + burst_b) // esz
        dst_pitch = (d_gap * 32 + burst_b) // esz
        r, c = Extent(rows, str(rows)), Extent(cols, str(cols))
        s = self._tile_view(op, src, r, c, src_pitch)
        d = self._tile_view(op, dst, r, c, dst_pitch)
        self.emit(f"TMOV({d}, {s});", op)

    # ---------------------------------------------------------------- debug.* (RFC-0011 §7.20)

    # ---------------------------------------------------------------- synchronisation (§5)

    #: cce's depth aliases, so an event of depth 1..4 prints the name cce prints
    _EVENT_ALIAS = EVENT_ALIAS

    def op_sync_event(self, op: Any) -> None:
        """``SEvent`` / ``DEvent`` / … — cce's event object, restated in the support header.

        PTO's own ``pto::Event<Op, Op>`` is still declined: its ``EventIdCounter`` allocates ids
        and would fight the events pass that owns them (RFC-0011 §5). cce's does not — the ids are
        template arguments — so what is left is exactly the state this backend was keeping by hand.

        Its **destructor** is the load-bearing part. ``release()`` waits, at scope exit, every set
        that was never waited, counting at run time, which is correct at every return including
        one inside a loop or a branch. Printing the bare ``set_flag`` / ``wait_flag`` pair instead
        meant deciding statically how many flags were outstanding: a set-minus-wait analysis over
        the whole function, run-time counters wherever a ``cf.if``'s arms disagreed (226 times in
        the corpus), and a refusal wherever an early return sat inside a region. C++ scoping does
        all three, which is why cce never needed any of it.

        ``preset`` is not decoration: the constructor issues that many ``set_flag`` calls, which is
        what lets the *first* wait of a loop-carried "the buffer is free again" event succeed.
        Without it the pipe waits on a flag nobody set and the kernel deadlocks, which the board
        reports as a task timeout (§7.10).
        """
        r = op.results[0]
        t = r.type
        ids = op.attrs.get("ids")
        if ids is None:
            ids = [] if getattr(t, "id", None) is None else [t.id]
        ids = [int(i) for i in ids]
        if not ids:
            raise PtoIsaGap(op, "event without flag ids (run the events pass)")
        if getattr(t, "set_pipe", None) is None or getattr(t, "wait_pipe", None) is None:
            raise PtoIsaGap(op, "event without pipes")
        preset = op.attrs.get("preset", 0)
        if isinstance(preset, bool):
            preset = len(ids) if preset else 0
        cls = self._EVENT_ALIAS.get(len(ids), "ascrip::Event")
        args = ", ".join([pt.PIPE[t.set_pipe], pt.PIPE[t.wait_pipe], str(int(preset)),
                          *(str(i) for i in ids)])
        self.emit(f"{cls}<{args}> {self.name(r)};", op)
        self.events.add(r.name)

    def _event_name(self, op: Any) -> str:
        """The C++ object a set / wait acts on, or a located gap if its declaration refused."""
        ev = op.operands[0]
        if ev.name not in self.events:
            raise PtoIsaGap(op, f"event {ev.name} was never declared (its sync.event did not print)")
        return self.name(ev)

    def op_sync_barrier(self, op: Any) -> None:
        pipe = op.attrs.get("pipe")
        name = getattr(pipe, "name", pipe) or "ALL"
        if name not in pt.PIPE:
            raise PtoIsaGap(op, f"barrier on pipe {name!r} has no PTO spelling")
        self.emit(f"pipe_barrier({pt.PIPE[name]});", op)

    # ---------------------------------------------------------------- scalars (RFC-0011 §7.5)

    def scalar_ctype(self, t: Any) -> str:
        if isinstance(t, ScalarType | CellType):
            return cpp.ctype(t.dtype)
        raise PtoIsaGap(None, f"not a scalar type: {t}")

    def _def(self, op: Any, expr: str) -> None:
        """One SSA scalar def, one ``const`` local. Unlike cce this never folds a temporary into
        its use: the fold is cosmetic there, and keeping one statement per op means every op keeps
        its own ``// #id`` trace (D-015) without the id-carrying bookkeeping."""
        r = op.results[0]
        self.emit(f"const {self.scalar_ctype(r.type)} {self.name(r)} = {expr};", op)

    def _binop(self, op: Any, sym: str) -> None:
        dt = self._rdt(op)
        a, b = (self.val(x, dt) for x in op.operands[:2])
        self._def(op, f"({a}) {sym} ({b})")

    def op_scalar_min(self, op: Any, kind: str = "min") -> None:
        dt = self._rdt(op)
        a, b = (self.val(x, dt) for x in op.operands[:2])
        lo, hi = (a, b) if kind == "min" else (b, a)  # f32 prints Pro's measured IEEE spelling (RFC-0001 §6.16)
        self._def(op, cpp.f32_extremum(kind, a, b) if dt == F32 else f"(({a}) < ({b}) ? ({lo}) : ({hi}))")

    def op_scalar_ceil_div(self, op: Any) -> None:
        # tensorutils_cce.h's CeilDiv, guard included: b == 0 ? 0 : (a + b - 1) / b
        a, b = (self.val(x) for x in op.operands[:2])
        self._def(op, f"(({b}) == 0 ? 0 : (({a}) + ({b}) - 1) / ({b}))")

    def op_scalar_align(self, op: Any) -> None:
        n = op.attrs.get("n")
        if not isinstance(n, int) or n <= 0:
            raise PtoIsaGap(op, f"scalar.align needs a positive compile-time n, got {n!r}")
        a = self.val(op.operands[0])
        self._def(op, f"((({a}) + {n} - 1) / {n} * {n})")

    def op_scalar_sqrt(self, op: Any) -> None:
        dt = self._rdt(op)
        a = self.val(op.operands[0], dt)
        if dt.name == "bf16":
            raise PtoIsaGap(op, "scalar sqrt requires a target-supported BF16 scalar conversion (M10-055)")
        if dt.name == "f32":
            self._def(op, f"::sqrt({a})")
        elif dt.name == "f16":  # the scalar unit has no half sqrt: through fp32, as cce does
            self._def(op, f"({cpp.ctype(dt)})::sqrt((float)({a}))")
        else:
            raise PtoIsaGap(op, f"scalar sqrt of {dt.name} has no spelling (f32 / f16 / bf16)")

    _CMP = cpp.CMP_OPS

    def op_scalar_cmp(self, op: Any) -> None:
        pred = op.attrs.get("pred")
        pred = getattr(pred, "name", pred)
        if pred not in self._CMP:
            raise PtoIsaGap(op, f"scalar.cmp predicate {pred!r} is not one of {sorted(self._CMP)}")
        a, b = op.operands[:2]
        dt = None
        for x in (a, b):
            if isinstance(x, Value) and isinstance(x.type, ScalarType | CellType):
                dt = x.type.dtype
                break
        self._def(op, f"({self.val(a, dt)}) {self._CMP[pred]} ({self.val(b, dt)})")

    # ---------------------------------------------------------------- gmlist (RFC-0011 §4.14)

    def _gmlist(self, op: Any, v: Value) -> str:
        """The ``ascrip::GMList<T>`` local a list parameter is read through.

        Declared once per function, lazily, right where the first read is — a descriptor read has
        no pipe and no ordering, so there is nowhere it can be too late.
        """
        t = getattr(v, "type", None)
        if not isinstance(t, MemType) or t.space != "gmlist":
            raise PtoIsaGap(op, f"{getattr(v, 'name', v)!r} is not a GM tensor list")
        name = self.name(v)
        local = f"{name}_l"
        if local not in self.mp.gmlists.setdefault(self.fn.name, set()):
            try:
                elem = pt.elem(t.dtype)
            except KeyError as exc:
                raise PtoIsaGap(op, str(exc.args[0])) from exc
            self.emit(f"ascrip::GMList<{elem}> {local}({name});", op)
            self.mp.gmlists[self.fn.name].add(local)
        return local

    def op_list_count(self, op: Any) -> None:
        """``GMList::count()`` — the descriptor's own member count (RFC-0001 §13)."""
        self._def(op, f"{self._gmlist(op, op.operands[0])}.count()")

    def op_list_item_dim(self, op: Any) -> None:
        """One ragged dimension of one member: ``GMList::dim(i, d)``."""
        d = int(op.attrs["dim"])
        self._def(op, f"(int32_t){self._gmlist(op, op.operands[0])}.dim({self.val(op.operands[1])}, {d})")

    def op_list_item(self, op: Any) -> None:
        """One member as a GM window — a **pointer**, which is all a `pto::GlobalTensor` needs.

        This is why the list family is printed at all while `simt.*` is not (§7.25): there is no
        instruction in it. ``ptr(i)`` is a `__gm__ uint64_t` load off the descriptor, and what it
        yields joins §4.1's ordinary GM path through `_gm_base`'s ``<root>_p`` convention — the
        same one `mem.workspace` uses. The tensor's *type* is the member's, so the ragged rank-0
        dimension never reaches a tile: only the transfer's own extents do.
        """
        r = op.results[0]
        mt = r.type
        if not isinstance(mt, MemType):
            raise PtoIsaGap(op, "list.item with a non-memory result")
        try:
            elem = pt.elem(mt.dtype)
        except KeyError as exc:
            raise PtoIsaGap(op, str(exc.args[0])) from exc
        name = self.name(r)
        lst = self._gmlist(op, op.operands[0])
        self.emit(f"__gm__ {elem}* {name}_p = {lst}.ptr({self.val(op.operands[1])});", op)
        self.gms[r.name] = GTensor(name=name, dtype=mt.dtype, shape=tuple(mt.dims))

    def op_scalar_load(self, op: Any) -> None:
        """One element of GM or a manual-mode Vec tile, including the view offset."""
        src, idx = op.operands[:2]
        if src.type.space == "ub":
            base = self._ub_ptr(op, src, src.type.dtype)
        else:
            base, _ = self._gm_base(op, src)
        self._def(op, f"{base}[{self.val(idx)}]")

    def op_scalar_store(self, op: Any) -> None:
        dst, idx, src = op.operands[:3]
        if dst.type.space == "ub":
            base, elem = self._ub_ptr(op, dst, dst.type.dtype), pt.elem(dst.type.dtype)
        else:
            base, elem = self._gm_base(op, dst)
        self.emit(f"{base}[{self.val(idx)}] = ({elem})({self.val(src)});", op)

    # ---------------------------------------------------------------- core identity

    def _core(self, op: Any, vec: str, cube: str) -> None:
        """A core id, resolved for the side this function is printed for.

        ``tensorutils_cce.h`` selects between these two forms with ``ASCEND_IS_AIV`` at run time.
        That macro is an AscendC one and is *not* available here (``pto-inst.hpp`` does not pull
        ``kernel_operator.h`` in — the first board compile proved it). It costs nothing: the side
        is a static property of the function being printed, so the branch is resolved here.
        """
        self._def(op, vec if self.side == "vec" else cube)

    def op_core_cube_idx(self, op: Any) -> None:
        self._def(op, "(int32_t)get_block_idx()")

    def op_core_cube_num(self, op: Any) -> None:
        self._def(op, "(int32_t)get_block_num()")

    def op_core_vec_idx(self, op: Any) -> None:
        self._core(op, "(int32_t)(get_block_idx() * get_subblockdim() + get_subblockid())",
                   "(int32_t)get_block_idx()")

    def op_core_vec_num(self, op: Any) -> None:
        self._core(op, "(int32_t)(get_block_num() * get_subblockdim())", "(int32_t)get_block_num()")

    def op_core_sub_block_idx(self, op: Any) -> None:
        self._core(op, "(int32_t)get_subblockid()", "0")

    #: CTRL register bits, as ``tensorutils_cce.h`` maps them (D-084 verified these on silicon).
    _HF32_BIT = 46

    def _set_ctrl_bit(self, op: Any, bit: int, on: str) -> None:
        """A CTRL-register bit, through the CCE builtins tensorutils_cce.h's SetSatFlag uses.
        These are compiler intrinsics, not AscendC library calls, so they are reachable from a
        translation unit that includes only ``pto/pto-inst.hpp``."""
        self.emit(f"set_ctrl(({on}) ? sbitset1(get_ctrl(), {bit}) : sbitset0(get_ctrl(), {bit}));", op)

    def op_core_set_sat_flag(self, op: Any) -> None:
        self._set_ctrl_bit(op, self._sat_bit(op), self.flag(op, "enable", True))

    def op_core_get_sat_flag(self, op: Any) -> None:
        self._def(op, f"(int32_t)((get_ctrl() >> {self._sat_bit(op)}) & 1ULL)")

    def op_core_set_hf32(self, op: Any) -> None:
        self._set_ctrl_bit(op, self._HF32_BIT, self.flag(op, "enable", True))

    # ---------------------------------------------------------------- control flow

    def op_cf_for(self, op: Any) -> None:
        i = op.results[0]
        lo, hi, step = op.operands[:3]
        T = self.scalar_ctype(i.type)
        n = self.name(i)
        s = self.val(step)
        if isinstance(step, Literal) and int(step.value) > 0:
            cond = f"{n} < {self.val(hi)}"
        else:  # a runtime or negative step: the direction is only known at run time
            cond = f"(({s}) > 0 ? {n} < {self.val(hi)} : {n} > {self.val(hi)})"
        self.emit(f"for ({T} {n} = {self.val(lo)}; {cond}; {n} += {s}) {{", op)
        with self._indented():
            self.run_block(op.regions[0])
        self.emit("}")

    @contextlib.contextmanager
    def _indented(self) -> Any:
        """One level in, and back out even when the body refuses.

        Without the ``finally`` a ``PtoIsaGap`` raised inside a region leaks the indent, and every
        later line — including this printer's own diagnosis of where a `cf.return` sits — is
        measured against a level that never came back. That is how 47 kernels came to report an
        "early return inside a region" that was a plain function-level return."""
        self.indent += 1
        try:
            yield
        finally:
            self.indent -= 1

    def op_cf_if(self, op: Any) -> None:
        self.emit(f"if ({self.val(op.operands[0])}) {{", op)
        with self._indented():
            self.run_block(op.regions[0])
        if len(op.regions) > 1 and len(op.regions[1]) > 0:
            self.emit("} else {")
            with self._indented():
                self.run_block(op.regions[1])
        self.emit("}")

    # ---------------------------------------------------------------- cross-core sync (§5.3)

    #: opcode -> the `tensorutils_ptoisa.h` helper that spells it: cce's own table, so the two
    #: backends' cross-core output is comparable as text. What each expands to (mode 0x4
    #: intra-block for the cube <-> vector pair, FFTS 0x0 / 0x1 for the ALL* and INTRACORE groups,
    #: and the AIC side's N + N+16 asymmetry) is stated once, in the header, rather than at every
    #: call site.
    _XCORE = c310.CROSSCORE

    def _xcore_call(self, opcode: str, pipe: str, fid: str) -> str:
        """The one call a cross-core op prints — `ascrip::CUBE_READY<P>(id)` and its siblings.

        PTO does have cross-core events: `pto::Event` with `IsCrossCore`, whose `Init` is exactly
        ``set_intra_block(p, id); set_intra_block(p, id + 16);`` and whose `Wait` is a single
        ``wait_intra_block`` (`npu/a5/TSync.hpp:98-116`) — the *same* two intrinsics these helpers
        reach. But PTO ties the event to a data-movement op pair (`IsCrossCoreEvent()` is true only
        for `TMOV_A2V` / `TMOV_V2M` / `TEXTRACT_V2M`) while our IR's cross-core flags are
        free-standing, with ids autosync has already assigned — the same reason §5 declines PTO's
        event class for the intra-core flags.

        So the header restates cce's helpers byte for byte and this prints the call. The side guard
        lives inside them, where cce proved it: a vector-only intrinsic is a compile error in the
        cube translation unit and vice versa, and a constexpr-if the unit never odr-uses compiles
        to nothing.
        """
        return f"{self._XCORE[opcode]}<{pipe}>({fid});"

    def _xcore_pipe(self, op: Any, key: str, default: str | None = None) -> str:
        v = op.attrs.get(key, default)
        name = getattr(v, "name", v)
        if name not in pt.PIPE:
            raise PtoIsaGap(op, f"{key} {name!r} is not a pipe this backend spells")
        return pt.PIPE[name]

    def _crosscore(self, op: Any) -> None:
        """``CUBE_READY`` / ``WAIT_VEC`` / ``VEC_READY`` / ``WAIT_CUBE`` and the ALL* groups.

        On a5 the point-to-point pair is asymmetric, and the asymmetry is load-bearing: the **AIC**
        side sets *and* waits both flag N and N+16, because AIV1's N is remapped to N+16, while the
        AIV side handles the single N (`tensorutils_cce.h:492-519`). Getting that wrong deadlocks
        one sub-block rather than producing wrong numbers, which is the harder failure to read.
        """
        pipe = self._xcore_pipe(op, "pipe")
        self.emit(self._xcore_call(op.opcode, pipe, self.val(op.attrs["flag_id"])), op)

    op_sync_crosscore_cube_ready = _crosscore
    op_sync_crosscore_wait_vec = _crosscore
    op_sync_crosscore_vec_ready = _crosscore
    op_sync_crosscore_wait_cube = _crosscore
    op_sync_crosscore_allcube_ready = _crosscore
    op_sync_crosscore_allcube_wait = _crosscore
    op_sync_crosscore_allvec_ready = _crosscore
    op_sync_crosscore_allvec_wait = _crosscore
    op_sync_crosscore_intracore_allvec_ready = _crosscore
    op_sync_crosscore_intracore_allvec_wait = _crosscore

    def op_sync_mutex(self, op: Any) -> None:
        """The prologue / epilogue of a cross-core mutex, both sides of it.

        A mutex is not an instruction, it is a *token count*. The consumer side publishes ``depth``
        free tokens before the body, which is what lets the producer's first lock succeed; the
        producer side drains them before the return, so no flag is left set for the next launch.
        The protocol between the two is the ordinary crosscore ops of the body.

        ``kind``: ``vc`` — the vector cores produce and the cube core consumes; ``cv`` the reverse.
        The op appears on **both** sides and prints differently on each, so what decides is the
        function's own ``side``, not anything on the op.
        """
        kind = getattr(op.attrs.get("kind"), "name", op.attrs.get("kind"))
        if kind not in ("vc", "cv"):
            raise PtoIsaGap(op, f"sync.mutex kind {kind!r} is neither 'vc' nor 'cv'")
        fid = self.val(op.attrs["id"])
        depth = int(op.attrs["depth"])
        side = getattr(self.fn.attrs.get("side"), "name", self.fn.attrs.get("side")) or "vec"
        set_pipe = self._xcore_pipe(op, "dst_end_pipe", "FIX" if kind == "vc" else "MTE3")
        wait_pipe = self._xcore_pipe(op, "src_start_pipe", "S")
        if kind == "vc":
            consumer, publish, drain = "cube", "sync.crosscore.cube_ready", "sync.crosscore.wait_cube"
        else:
            consumer, publish, drain = "vec", "sync.crosscore.vec_ready", "sync.crosscore.wait_vec"
        if side == consumer:
            for _ in range(depth):
                self.emit(self._xcore_call(publish, set_pipe, fid), op)
        else:
            self.emit(f"// sync.mutex {kind} id={fid}: the producer side; its {depth} tokens are "
                      "drained before the return", op)
            for _ in range(depth):
                self.epilogue.append(self._xcore_call(drain, wait_pipe, fid))

    # ---------------------------------------------------------------- the @vf bridge (§1 ruling 1)

    def _ub_ptr(self, op: Any, v: Value, want: DType) -> str:
        """A UB window as a ``__ubuf__`` pointer — the seam between the tile ISA and the register ISA.

        A ``@vf`` body is CCE register intrinsics over raw UB pointers (it never mentions a tile),
        so the call site is where the two layers meet. **In manual mode the seam is just
        ``tile.data()``**: ``MemoryQualifier<TileType::Vec, T>::type`` is ``__ubuf__ T*`` under
        ``#ifndef __PTO_AUTO__`` (``common/memory.hpp:27``), so a manual-mode tile's ``data()``
        already *is* the pointer.

        It is **not** ``__cce_get_tile_ptr(tile.data())``, which the first board build rejected
        with "can only be used inside __tf__ functions". That builtin exists for AUTO mode, where
        the same ``type`` is the handle ``__ubuf__ T`` rather than a pointer; pto's own templates
        (``npu/a5/TAdd.hpp``) call it because they must compile in both modes, and ``__tf__`` is
        what licenses it. We emit manual mode only (the ``#ifdef __PTO_AUTO__`` guard in HEADER
        makes that a compile-time fact), so the conversion is a plain cast.
        """
        g = self.geo(v)
        if g.space != "ub":
            raise PtoIsaGap(op, f"a @vf parameter takes a UB window; {v.name} is in {g.space!r}")
        rec = self.rec_of(op, v)
        try:
            elem = pt.elem(want)
        except KeyError as exc:
            raise PtoIsaGap(op, str(exc.args[0])) from exc
        base = rec.name
        if rec.slots:
            if g.slot is None:
                raise PtoIsaGap(op, f"{v.name} is a slot buffer but the call passes no slot index")
            # the *wrapped* index: cce's Buff::get is `slot[((i % N) + N) % N]`, and our kernels
            # drive a ring with a counter that only ever increments, so the raw index would run
            # off the end of the C++ array within two iterations
            base = f"{base}[{self.cexpr(self.slot_index(op, g, rec))}]"
        same = g.dtype.name == rec.dtype.name and want.name == rec.dtype.name
        ptr = f"{base}.data()" if same else f"(__ubuf__ {elem}*){base}.data()"
        off_b = views.byte_offset(g)
        if isinstance(off_b, int) and off_b == 0:
            return ptr
        # the displacement is applied *after* the cast, in elements of the parameter's dtype —
        # the same arithmetic cce's VfPrinter.ptr() does on its own pointer parameters
        off_e = views.div_exact(off_b, cpp.esize(want))
        if off_e is None:
            raise PtoIsaGap(op, f"the @vf argument {v.name} starts at a byte offset that is not a "
                                f"whole number of {want.name} elements")
        return f"({ptr} + ({self.cexpr(off_e)}))"

    def op_simt_launch(self, op: Any) -> None:
        """``simt::launch<fn>(threads, args...)`` — the same bridge §7.7 draws for ``@vf``.

        A ``@simt`` body is plain C on the *compiler's* SIMT layer over ``__gm__`` / ``__ubuf__``
        pointers: there is no tile in it, and nothing for PTO to say about it. That is precisely
        the argument for printing ``@vf`` bodies with cce's own ``VfPrinter`` rather than
        reprinting them here, and it applies unchanged — so the **body** comes from cce's
        ``SimtPrinter`` and only the launch site is ours. The shim both of them call is restated in
        `tensorutils_ptoisa.h`, on §7.13's boundary.

        ``threads`` is not an argument to the callee: it reaches the *callee's* signature, as
        ``__launch_bounds__``. So the launch site records it on the module printer, and the body is
        rendered afterwards — which is why `emit_module` prints the sides first and the simt
        functions second, the opposite order from ``@vf``.
        """
        callee = op.operands[0]
        if not isinstance(callee, FuncRef):
            raise PtoIsaGap(op, "simt.launch without a function reference")
        fn = self.mp.module.function(callee.name)
        if fn.kind != "simt":
            raise PtoIsaGap(op, f"simt.launch of a {fn.kind} function")
        threads = self._fold(op, op.attrs.get("threads"), "threads")
        # one callee, two launch sites, two thread counts: cce keeps the larger, because
        # `__launch_bounds__` is a ceiling and the smaller launch still fits under it
        self.mp.threads[callee.name] = max(threads, self.mp.threads.get(callee.name, 0))
        args = [self._simt_arg(op, p, a) for p, a in zip(fn.params, op.operands[1:], strict=True)]
        self.mp.simt_called.add(callee.name)
        self.emit(f"simt::launch<{self.mp.fname(callee.name)}>({threads}"
                  f"{''.join(', ' + a for a in args)});", op)

    def _simt_arg(self, op: Any, p: Value, a: Any) -> str:
        """One launch argument. A UB window is the `@vf` bridge's pointer; a GM one is §4.1's."""
        t = p.type
        if isinstance(t, MemType) and t.space == "ub":
            return self._ub_ptr(op, a, t.dtype)
        if isinstance(t, MemType) and t.space in ("gm", "ws"):
            return self._gm_base(op, a, pt.elem(t.dtype))[0]
        if isinstance(t, ScalarType | CellType):
            return f"({cpp.ctype(t.dtype)})({self.val(a, t.dtype)})"
        raise PtoIsaGap(op, f"@simt parameter {p.name!r} of type {t} cannot be passed")

    def op_cf_call(self, op: Any) -> None:
        callee = op.operands[0]
        if not isinstance(callee, FuncRef):
            raise PtoIsaGap(op, "cf.call without a function reference")
        fn = self.mp.module.function(callee.name)
        if fn.kind == "simt":
            raise PtoIsaGap(op, f"@simt function {callee.name!r} reached through cf.call rather "
                                "than simt.launch; a SIMT function is launched, not called")
        if fn.kind != "vf":
            raise PtoIsaGap(op, f"cf.call to a {fn.kind} function is not printed")
        args = []
        for p, a in zip(fn.params, op.operands[1:], strict=True):
            t = p.type
            if isinstance(t, MemType) and t.space == "ub":
                args.append(self._ub_ptr(op, a, t.dtype))
            elif isinstance(t, ScalarType | CellType):
                args.append(f"({cpp.ctype(t.dtype)})({self.val(a, t.dtype)})")
            else:
                raise PtoIsaGap(op, f"@vf parameter {p.name!r} of type {t} cannot be passed")
        self.mp.vf_called.add(callee.name)
        self.emit(f"{self.mp.fname(callee.name)}({', '.join(args)});", op)


class _VfShim:
    """The handful of attributes cce's ``VfPrinter`` reads off its module printer.

    The ``@vf`` body is reused from the ``cce`` backend verbatim (RFC-0011 §7.7): it prints CCE
    register intrinsics over ``__ubuf__`` pointers and contains nothing tile-shaped, so there is
    no PTO work in it at all. Rather than pass our own ``ModulePrinter`` — which happens to carry
    compatible ``fname`` / ``module`` and would silently absorb any *new* attribute cce starts
    reading — this shim states the coupling surface explicitly, so a change on the cce side fails
    loudly with an ``AttributeError`` instead of quietly printing something else.
    """

    #: c310 is a5. cce's printer only tests ``arch == "c220"`` (the a2 V-V hazard tracking of
    #: D-065), so the value matters, not the vocabulary.
    arch = "c310"

    def __init__(self, mp: ModulePrinter) -> None:
        self._mp = mp
        self.module = mp.module
        #: `SimtPrinter.signature` reads it for ``__launch_bounds__``; the launch site filled it
        #: (§7.29). One attribute more than ``VfPrinter`` needs, stated rather than inherited.
        self.threads = mp.threads
        #: `SimtPrinter.op_simt_block_idx` reads both, to derive the CUBE core id from the vec
        #: index (D-161/D-163): in a MIX binary `blockIdx.x` is the vec index, not the cube id the
        #: kernel contract asks for, so it divides by the participants-per-cube ratio -- and an
        #: AIV_ONLY binary, which has no sub-block layer, must not. `mode` this printer already
        #: computes the same way cce does; the ratio comes from the device profile, as there.
        self.mode = mp.mode
        prof = _load_device(mp.device) if mp.device else None
        self.vec_per_cube = max(1, prof.vec_cores // prof.cube_cores) if prof and prof.cube_cores else 2

    def fname(self, name: str) -> str:
        return self._mp.fname(name)


@dataclass
class ModulePrinter:
    """Prints a lowered module as one PTO C++ translation unit.

    ``bindings`` specialises the module for one scalar valuation (shape symbol -> int). PTO tile
    capacities are template arguments, so a kernel whose tiles are shaped by a runtime parameter
    only prints once bound — the same constraint, and the same answer, as the pypto_pro backend
    (D-081): the launcher recompiles per valuation.
    """

    #: c310 is a5, the only family this backend prints (RFC-0011 §1); the shared host layer reads it
    arch = "c310"

    module: Module
    block_dim: int | None = None
    entry: str | None = None
    bindings: dict[str, int] = field(default_factory=dict)
    analysis_functions: dict[str, Function] = field(default_factory=dict)
    uses_workspace: bool = False
    fns: dict[str, list[str]] = field(default_factory=dict)
    vf_called: set[str] = field(default_factory=set)  # @vf functions the body actually calls
    simt_called: set[str] = field(default_factory=set)  # @simt functions the body launches
    threads: dict[str, int] = field(default_factory=dict)  # a launched callee's __launch_bounds__
    _counter: int = 0

    def __post_init__(self) -> None:
        meta = dict(self.module.attrs.get("meta", {}))
        self.kernel = str(meta.get("kernel", self.module.name))
        # vec | cube | mix: drives the entry's task type and which side guards it dispatches under
        m = self.module.attrs.get("mode")
        self.mode = str(getattr(m, "name", m) or "mix")
        self.device = self.module.device or "950"
        self.outputs = [v.name if isinstance(v, Value) else str(v) for v in meta.get("outputs", [])]
        if self.block_dim is None:
            self.block_dim = meta.get("block_dim")
        self.workspaces: list[dict[str, Any]] = []
        #: per-function, the ``ascrip::GMList`` locals already declared (§4.14)
        self.gmlists: dict[str, set[str]] = {}

    def fname(self, name: str) -> str:
        """``gm_view.vec`` -> ``gm_view_vec``, and a name under a renamed entry keeps its own
        spelling — only the *entry* symbol has to match what the host build expects."""
        return c_ident(name)

    def next_id(self) -> int:
        """A module-unique suffix for a generated GlobalTensor / tile-view name."""
        self._counter += 1
        return self._counter

    def fold(self, x: Any) -> int | None:
        """A dim / attribute as a compile-time int, or None. Shape symbols read ``bindings``."""
        if isinstance(x, Literal):
            x = x.value
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        name = getattr(x, "name", None)  # DimValue / Value: both carry the scalar's name
        if isinstance(name, str):
            return self.bindings.get(name)
        return None


#: Every artifact opens with a compile-time refusal of PTO AUTO mode.
#:
#: PTO has two modes. In AUTO (``--cce-pto-auto-enable``, which defines ``__PTO_AUTO__``) the PTO
#: compiler assigns tile addresses and inserts pipe synchronisation itself; in manual mode the
#: program does both. We are necessarily manual: the addresses are ``addr_alloc``'s and the flag
#: ids are ``autosync``'s, and those two passes are the load-bearing part of this compiler
#: (RFC-0011 §2). Building this source with AUTO on would layer a second allocator and a second
#: synchroniser over ours.
#:
#: The guard is not decoration. ``__PTO_AUTO__`` switches implementations *inside pto's own
#: headers* (``npu/*/TSync.hpp``, ``SyncAll.hpp``, the async comm path), so a wrong build flag
#: would change behaviour silently rather than fail to compile. This turns it into an error.
#: the support header this backend prints against, shipped beside the .cpp. Deliberately tiny --
#: it holds only what PTO does not provide and our own lowering decides (the event-flag rotation,
#: a slot buffer's ring index, the cross-core flag pairs, and the two scalar helpers a @vf body
#: reaches for). There is no wrapper here over any T* instruction: PTO already is that layer, and
#: keeping its own names in the output is what lets every line be checked against its headers.
SUPPORT_HEADER = Path(__file__).with_name("include") / "tensorutils_ptoisa.h"

HEADER = (
    '#include "kernel_operator.h"\n'
    '#include "pto/pto-inst.hpp"\n'
    '#include "tensorutils_ptoisa.h"\n'
    "\n"
    "#ifdef __PTO_AUTO__\n"
    '#error "ascriptor emits PTO manual mode: tile addresses come from addr_alloc (TASSIGN) and '
    "pipe events from autosync (set_flag/wait_flag). Building with --cce-pto-auto-enable would "
    'put PTO\'s own allocator and synchroniser on top of them. Drop the flag."\n'
    "#endif\n"
    "\n"
    "using namespace pto;\n"
    "using namespace ascrip;\n"
)

#: how the entry dispatches to a side. PTO kernels are ordinary CCE kernels, so the frame is the
#: cce backend's (RFC-0007 §4) — which is the point: the artifact rides the existing bisheng /
#: aclnn / board chain unchanged. ``kernel_operator.h`` supplies ``GM_ADDR``, these guards and
#: ``KERNEL_TASK_TYPE_DEFAULT``; it coexists with ``pto-inst.hpp`` by design (pto's own
#: ``demos/baseline/add`` includes both), and the aclnn project's CMake already puts
#: ``$CANN/include`` on the path, where ``pto/`` lives — so no template change (D-013).
_SIDE_GUARD = {"vec": "ASCEND_IS_AIV", "cube": "ASCEND_IS_AIC"}


def _side_of(fn: Function) -> str:
    s = fn.attrs.get("side")
    return getattr(s, "name", s) or ("cube" if fn.name.endswith(".cube") else "vec")


def emit_module(module: Module, block_dim: int | None = None, entry: str | None = None,
                bindings: dict[str, int] | None = None) -> Artifacts:
    """One PTO C++ translation unit: a function per side plus the kernel entry.

    Batch 2 prints movement only, so a kernel containing compute still raises ``PtoIsaGap`` from
    the op that has no printer; what this adds is the frame around a kernel whose every op does
    print.
    """
    from ...passes import PassManager
    from ...passes.integer_division import PASS_DEF
    from ...passes.bf16_scalar import PASS_DEF as BF16_SCALAR
    from ...passes.scalar_simplify import PASS_DEF as SCALAR_SIMPLIFY

    analysis_functions = {f.name: f for f in module.functions}
    module = PassManager((PASS_DEF, BF16_SCALAR, SCALAR_SIMPLIFY), options={"compact_integer_mod": True}).run(module)
    mp = ModulePrinter(module, block_dim=block_dim, entry=entry, bindings=dict(bindings or {}),
                       analysis_functions=analysis_functions)
    sides: dict[str, Function] = {}
    vfs: list[Function] = []
    simts: list[Function] = []
    for fn in module.functions:
        if fn.kind == "func":
            sides[_side_of(fn)] = fn
        elif fn.kind == "vf":
            vfs.append(fn)
        elif fn.kind == "simt":
            simts.append(fn)
    if not sides:
        raise PtoIsaGap(None, "no side functions (run split_sides)")

    name = c_ident(entry or module.name)
    params = list(next(iter(sides.values())).params)
    for p in params:
        if not isinstance(p.type, MemType | ScalarType):
            raise PtoIsaGap(None, f"kernel parameter {p.name!r} is a {type(p.type).__name__}, which "
                                  "cannot be passed from the host")

    # Every side is printed before the frame is written: a gap on the cube side must surface as a
    # gap, not as a half-emitted unit.
    #
    # A scalar parameter is an ordinary C++ parameter. `pl` has to fold every one of them at trace
    # time, which is why the pypto backend specialises a whole source file per valuation (D-081);
    # here a runtime scalar only has to fold when it sizes a tile, because *that* is a template
    # argument. RFC-0011 §2 — do not carry `pl`'s constraint into a C++ target.
    side_src: list[str] = []
    for side in ("cube", "vec"):  # definition order: the entry below calls both
        fn = sides.get(side)
        if fn is None:
            continue
        p = FnPrinter(mp, fn)
        p.run_block(fn.body)
        # a body that falls off the end owes nothing: the events are C++ objects, and their
        # destructors run at the closing brace just as they do at a `return`
        sig = ", ".join([*(_param_decl(q) for q in fn.params), "__gm__ uint8_t* workspace"])
        side_src += ["", f"__aicore__ inline void {mp.fname(fn.name)}({sig})", "{",
                     "    (void)workspace;", *p.lines, "}"]

    # The @vf bodies come from cce's own VfPrinter, unmodified (RFC-0011 §7.7): a vf function is
    # CCE register intrinsics over __ubuf__ pointers, which is the same on both backends, so
    # reprinting it here would be duplication with a second chance to be wrong. Only the ones the
    # body actually calls are emitted, and they precede the kernel — C++ needs the definition first.
    shim = _VfShim(mp)
    vf_src: list[str] = []
    for vf in vfs:
        if vf.name not in mp.vf_called:
            continue  # dead after split_sides: this side never calls it
        try:
            vf_src += ["", VfPrinter(shim, vf).render()]
        except Exception as exc:  # noqa: BLE001 - a CceGap (or anything else) becomes our gap
            raise PtoIsaGap(getattr(exc, "op", None),
                            f"@vf function {vf.name!r}: {getattr(exc, 'why', str(exc))}") from exc

    # After the sides, unlike the `@vf` bodies: `__launch_bounds__` is part of the callee's
    # signature and comes from the launch site, so the sides have to be printed before a simt
    # function can be. C++ still needs the definition first, hence the order in `body` below.
    simt_src: list[str] = []
    for fn in simts:
        if fn.name not in mp.simt_called:
            continue  # dead after split_sides: this side never launches it
        try:
            simt_src += ["", SimtPrinter(shim, fn).render()]
        except Exception as exc:  # noqa: BLE001 - a CceGap (or anything else) becomes our gap
            raise PtoIsaGap(None, f"@simt function {fn.name!r}: {exc}") from exc

    body: list[str] = [HEADER, *vf_src, *simt_src, *side_src, "", *_entry(mp, name, sides, params), ""]
    src = "\n".join(body).encode()
    file = f"{name}.cpp"
    files = {file: src, "tensorutils_ptoisa.h": SUPPORT_HEADER.read_bytes(),
             "scalar_math.h": (SUPPORT_HEADER.parents[2] / "shared/include/scalar_math.h").read_bytes()}
    meta = _metadata(mp, name, sides, vfs, simts, params, file, [*files, "manifest.json"])
    files["manifest.json"] = (json.dumps(meta, indent=1) + "\n").encode()
    return Artifacts(files=files, entry=file, metadata=meta)


def _entry(mp: ModulePrinter, name: str, sides: dict[str, Function], params: list[Value]) -> list[str]:
    """The cce-shaped kernel entry (RFC-0007 §4), so the artifact rides the existing aclnn / board
    chain: ``GM_ADDR`` parameters, the tiling struct for scalars, and the AIC / AIV dispatch that
    a mix kernel needs. PTO kernels are ordinary CCE kernels — nothing here is PTO-specific."""
    tensors = cpp.entry_tensors(params, mp.outputs)
    scalars = [q for q in params if isinstance(q.type, ScalarType)]
    pnames = {q.name: c_ident(q.name) for q in params}
    sig = ", ".join([*(f"GM_ADDR {pnames[q.name]}_" for q in tensors), "GM_ADDR workspace", "GM_ADDR tiling"])
    out = [f'extern "C" __global__ __aicore__ void {name}({sig})', "{",
           f"    KERNEL_TASK_TYPE_DEFAULT({c310.TASK_TYPE[mp.mode]});"]
    if scalars:
        out.append("    GET_TILING_DATA(tiling_data, tiling);")
    else:
        out.append("    (void)tiling;")
    out.append("    pipe_barrier(PIPE_ALL);")
    for q in scalars:
        out.append(f"    const {cpp.ctype(q.type.dtype)} {pnames[q.name]} = tiling_data.{cpp.api(q.name)};")
    args = ", ".join([*(f"(__gm__ {_entry_elem(q)}*){pnames[q.name]}_" if isinstance(q.type, MemType)
                        else pnames[q.name] for q in params), "(__gm__ uint8_t*)workspace"])
    for side in ("cube", "vec"):
        fn = sides.get(side)
        if fn is None or mp.mode not in ("mix", side):
            continue
        out += [f"    if {_SIDE_GUARD[side]} {{", f"        {mp.fname(fn.name)}({args});", "    }"]
    return [*out, "}"]


def _metadata(mp: ModulePrinter, name: str, sides: dict[str, Function], vfs: list[Function],
              simts: list[Function], params: list[Value], file: str,
              names: list[str]) -> dict[str, Any]:
    """The manifest ``runtime/project.py``'s ``HostSpec`` reads, in cce's shape — the same
    ``generate_project`` consumes both."""
    out: list[dict[str, Any]] = []
    for q in params:
        t = q.type
        if isinstance(t, MemType):
            out.append({"name": cpp.api(q.name), "ir_name": q.name,
                        "kind": "list" if t.space == "gmlist" else "tensor",
                        "dtype": t.dtype.name, "dims": [_dim_str(d) for d in t.dims],
                        "output": q.name in mp.outputs})
        else:
            out.append({"name": cpp.api(q.name), "ir_name": q.name, "kind": "scalar", "dtype": t.dtype.name})
    return {
        "backend": "pto_isa", "arch": "c310", "kernel": name, "ir_kernel": mp.kernel,
        "device": mp.device, "mode": mp.mode, "task_type": c310.TASK_TYPE[mp.mode],
        "block_dim": mp.block_dim, "params": out,
        "outputs": [cpp.api(o) for o in mp.outputs],
        "workspaces": mp.workspaces, "sides": sorted(sides),
        "vf": [mp.fname(f.name) for f in vfs if f.name in mp.vf_called],
        "simt": {mp.fname(f.name): mp.threads.get(f.name, 1024) for f in simts
                 if f.name in mp.simt_called},
        "ops": sum(1 for _ in mp.module.walk()), "files": sorted(names),
        "entry": file, "bindings": mp.bindings,
    }


def _entry_elem(q: Value) -> str:
    """The element type the entry casts a GM parameter to. A gmlist is bytes (§4.14)."""
    return "uint8_t" if q.type.space == "gmlist" else pt.elem(q.type.dtype)



def _param_decl(q: Value) -> str:
    if isinstance(q.type, ScalarType):
        return f"{cpp.ctype(q.type.dtype)} {c_ident(q.name)}"
    # a gmlist is the ListTensorDesc's own bytes, not the member dtype's (§4.14)
    if isinstance(q.type, MemType) and q.type.space == "gmlist":
        return f"__gm__ uint8_t* {c_ident(q.name)}"
    return f"__gm__ {pt.elem(q.type.dtype)}* {c_ident(q.name)}"


__all__ = ["PtoIsaGap", "FnPrinter", "ModulePrinter", "Tile", "GTensor", "HEADER", "SUPPORT_HEADER", "emit_module"]
