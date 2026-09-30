# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Lowered IR -> PyPTO Pro Python source (`pypto_pro.language`, imported as ``pl``).

``emit_module(module, bindings=...)`` returns one generated kernel module
(``kernel_pypto.py``), a board-side driver (``run_case.py``) and a ``manifest.json`` the
runtime's :class:`HostSpec` reads. ``sync_mode="manual"`` is the default: IR allocation IDs
become Tile group mutex IDs AND the IR's own ``sync.local_mutex_get/release`` are printed as
``pl.system.mutex_lock/mutex_unlock`` under ``@pl.jit(auto_mutex=False)``, so this backend owns
the credits exactly as the cce backend does. Events, cross-core protocols and barriers are
printed verbatim.

``sync_mode="auto_mutex"`` instead emits ``@pl.jit(auto_mutex=True)`` and drops the local mutex
operations, delegating the credits to PyPTO's own insertion over the declared Tile group mutex
IDs. That delegation is UNSOUND when a byte range is reachable through more than one Tile group,
because PyPTO orders accesses per group and this printer mints alias groups for matmul
destinations (see ``_st`` below): the producer then sits on one group and its consumer on
another, and no credit orders them. D-260 is the board-measured case - a cube matmul into L0C
followed by the FIX-pipe copy of that same L0C, reading the previous launch's accumulator. The
mode is kept for comparison and refuses at the alias it cannot cover.

**Specialisation**: PyPTO Pro tile shapes are compile-time, so scalar kernel parameters are
bound to the case's concrete values (``bindings``) and folded — the PyPTO-native pattern
(module-scope constants, re-trace per shape). The launcher recompiles per scalar valuation.

Ops without a mapping raise :class:`PyptoGap` with their source location (mirror of the cce
backend's ``CceGap`` contract; surveyed surface: docs/pypto-pro-coverage.md).
"""

from __future__ import annotations

import builtins
import json
import keyword
import math
import os
import re
import struct
from collections.abc import Iterable
from typing import Any

from ...ir.scalar_range import ScalarRanges
from ...ir.scalar_math import evaluate, integer_divmod, limits, rounding
from ..shared.fold_plan import paren, plan_folds, unparen
from ...ir.core import Block, FuncRef, Function, Literal, Module, Op, Value
from ...ir.saturation import SAT_BITS
from ...ir.types import BufType, CellType, MaskType, MemType, RegType, ScalarType
from ...passes.util import dim_scalar
from ..base import Artifacts
from ..cce.arch.c310 import MEM_BAR as _C310_MEM_BAR
from ..cce.cpp import MATH_NAMES
from .blocks import lane_load
from .cache import clean_dcache
from .capacity import require_static_allocation
from .counters import decrement, decremented
from .names import LocalNames, initializer_names
from .native_sync import TileDecl, TileRegistry, TileSite, validate_mode
from .scalar_cleanup import ScalarLine, cleanup as cleanup_scalars, eligible as scalar_cleanup_eligible
from .supplements import DRIVER_PREFLIGHT, Uses as SupplementUses
from .supplements import declared as declared_supplements
from .sync import raw_flag
from .vf_signature import used_parameters

# ------------------------------------------------------------------ tables

PL_DT = {
    "f16": "pl.DT_FP16", "f32": "pl.DT_FP32", "bf16": "pl.DT_BF16", "hif8": "pl.DT_HF8",
    "e4m3": "pl.DT_FP8E4M3FN", "e5m2": "pl.DT_FP8E5M2", "e8m0": "pl.DT_FP8E8M0",
    "fp4_e2m1": "pl.DT_FP4E2M1", "fp4_e1m2": "pl.DT_FP4E1M2",
    "i4": "pl.DT_INT4", "i8": "pl.DT_INT8", "i16": "pl.DT_INT16", "i32": "pl.DT_INT32",
    "i64": "pl.DT_INT64", "u8": "pl.DT_UINT8", "u16": "pl.DT_UINT16", "u32": "pl.DT_UINT32",
    "u64": "pl.DT_UINT64", "b1": "pl.DT_BOOL",
}

# pto refuses `saturate=OFF` for a cast whose DESTINATION is one of these (board:
# "FP32->FP8 conversion requires saturate=ON (RS_ENABLE)"), so the flag is left unstated
# there and pto's forced saturation stands - see the vf.cast branch, D-125.
_FORCED_SAT_DST = frozenset({"hif8", "e4m3", "e5m2", "e8m0", "fp4_e2m1", "fp4_e1m2"})

# 8-bit float registers pto's VALUE-BLIND vector ops still refuse by dtype - they go through
# their UINT8 carrier, which is the same bytes and the same lane count (see vf.select, D-127).
_FP8_CARRIED = frozenset({"hif8", "e4m3", "e5m2", "e8m0"})

TORCH_DT = {
    "f16": "float16", "f32": "float32", "bf16": "bfloat16",
    "i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64",
    "u8": "uint8", "u16": "uint16", "u32": "uint32", "u64": "uint64", "b1": "bool",
    "e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2",
    # complex32 is torch's `chalf`. Listed not because pypto can print a complex kernel - it
    # has no DT_* constant for one, which is what refuses these - but so that OUR manifest
    # stops being the first refusal and misreporting an upstream absence as our own.
    "c32": "complex32", "c64": "complex64",
}

# spelled through the P alias the preamble defines (P = pl.PipeType) - the sync lines
# dominate a mix kernel's text and the short form keeps them inside one screen line
PIPE = {"MTE1": "P.MTE1", "MTE2": "P.MTE2", "MTE3": "P.MTE3",
        "V": "P.V", "M": "P.M", "FIX": "P.FIX", "S": "P.S"}

BAR = {"M": "bar_m", "MTE1": "bar_mte1", "MTE2": "bar_mte2", "MTE3": "bar_mte3",
       "FIX": "bar_fix", "ALL": "bar_all"}

MEMSPACE = {"ub": "pl.MemorySpace.Vec", "l1": "pl.MemorySpace.Mat", "l0a": "pl.MemorySpace.Left",
            "l0b": "pl.MemorySpace.Right", "l0c": "pl.MemorySpace.Acc", "bt": "pl.MemorySpace.Bias",
            # MX per-group scales do NOT live in L1 on pto: mad_mx reads them from their own
            # stops via the SFractal layout ("scale tiles use E8M0 in ScaleLeft/ScaleRight",
            # _api.py:669). cce keeps them in L1, which is why our IR says l1 - the printer
            # re-homes the scale operands of an mx matmul here.
            "scale_l": "pl.MemorySpace.ScaleLeft", "scale_r": "pl.MemorySpace.ScaleRight"}

# vf: assignment form `dst = vf.<name>(a, b, mask)`
VF_BINARY = {"vf.add": "add", "vf.sub": "sub", "vf.mul": "mul", "vf.div": "div",
             "vf.min": "min", "vf.max": "max", "vf.and": "and_", "vf.or": "or_",
             "vf.xor": "xor", "vf.abssub": "abs_sub", "vf.expsub": "exp_sub",
             "vf.prelu": "prelu", "vf.shiftl": "shift_left", "vf.shiftr": "shift_right",
             "vf.muladddst": "mul_add_dst", "vf.muldstadd": "mul_dst_add"}
VF_UNARY = {"vf.abs": "abs", "vf.exp": "exp", "vf.sqrt": "sqrt", "vf.neg": "neg",
            "vf.relu": "relu", "vf.ln": "ln", "vf.log": "log", "vf.log2": "log2",
            "vf.log10": "log10", "vf.not": "not_", "vf.copy": "move",
            "vf.cadd": "reduce_sum", "vf.cmax": "reduce_max", "vf.cmin": "reduce_min",
            "vf.cpadd": "pair_reduce_sum"}
VF_GROUP_REDUCE = {"vf.cgadd": "reduce_sum", "vf.cgmax": "reduce_max", "vf.cgmin": "reduce_min"}
VF_SCALAR = {"vf.muls": "muls", "vf.adds": "adds", "vf.mins": "mins", "vf.maxs": "maxs",
             "vf.axpy": "axpy", "vf.shiftls": "shift_left", "vf.shiftrs": "shift_right",
             # `leaky_relu(src, scalar, preg, mode=)` is the same (src, scalar, preg) shape the
             # rest of this table prints; `scalar` is the negative slope. It has no
             # register-slope form, so it is deliberately absent from VF_SCALAR_REG and an
             # immediate pl cannot carry falls through to the scalar-parameter spelling.
             "vf.lrelu": "leaky_relu"}
# the register form each of those has, for the exact-constant rescue below
VF_SCALAR_REG = {"vf.muls": "mul", "vf.adds": "add", "vf.mins": "min", "vf.maxs": "max"}
VF_CMP = {"gt": "gt", "lt": "lt", "ge": "ge", "le": "le", "eq": "eq", "ne": "ne"}

# vf.barrier (src, dst) -> pl.MemBarMode. pl documents exactly the twelve AscendC MemType
# combinations the c310 intrinsic takes, under the same names, so this is the cce backend's
# `arch/c310.py::MEM_BAR` table itself (D-138).
MEM_BAR_MODE = _C310_MEM_BAR

SCALAR_BIN = {"scalar.add": "+", "scalar.sub": "-", "scalar.mul": "*", "scalar.mod": "%",
              "scalar.and": "&", "scalar.or": "|", "scalar.xor": "^",
              "scalar.shl": "<<", "scalar.shr": ">>"}
SCALAR_CMP = {"gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "==", "ne": "!="}

#: `direct` prints `grp[<index expression>]`; `named` restores the older `_ix = ...` spelling (see `getitem`).
INDEX_ENV = "ASCRIPTOR_PYPTO_INDEX"

ROUND = {m: f"pl.VFRoundMode.CAST_{m.upper()}" for m in ("rint", "round", "floor", "ceil", "trunc", "odd", "hybrid")}
LOAD_DIST = {"brc_b8": "BRC_B8", "brc_b16": "BRC_B16", "brc_b32": "BRC_B32", "brc": "BRC",
             "unpack4_b8": "UNPK4", "unpack_b8": "UNPK_B8", "unpack_b16": "UNPK_B16",
             "unpack_b32": "UNPK_B32", "dintlv_b8": "DINTLV_B8", "dintlv_b16": "DINTLV_B16",
             "dintlv_b32": "DINTLV_B32", "ds": "DS", "ds_b8": "DS_B8", "ds_b16": "DS_B16",
             "us": "US", "us_b8": "US_B8", "us_b16": "US_B16", "blk": "BLK",
             "e2b": "E2B", "e2b_b16": "E2B_B16", "e2b_b32": "E2B_B32"}
STORE_DIST = {"pack_b16": "PACK", "pack_b32": "PACK", "pack_b64": "PACK", "pack": "PACK",
              "pack4_b32": "PACK4", "pack4": "PACK4",
              "intlv": "INTLV", "intlv_b32": "INTLV_B32",
              "first_element_b8": "FIRST_ELEMENT", "first_element_b16": "FIRST_ELEMENT",
              "first_element_b32": "FIRST_ELEMENT"}
# The DUAL register forms of the same two pl calls. They are separate tables because a
# single-destination `load_align` cannot take a DINTLV distribution and a single-source
# `store_align` cannot take INTLV: pto decides the arity from the dist, so mixing the tables
# would print a call whose argument count does not match its mode.
#
# pto has no INTLV_B8 / INTLV_B16: plain `INTLV` is element-grain and takes its width from the
# REGISTER dtype (the same rule `de_interleave` follows). `INTLV_B32` names its width, as every
# `DINTLV_B*` does. The store handler checks the register against the tag where the mapping is
# lossy -- see the guard there.
LOAD_INTLV_DIST = {"dintlv_b8": "DINTLV_B8", "dintlv_b16": "DINTLV_B16", "dintlv_b32": "DINTLV_B32"}
STORE_INTLV_DIST = {"intlv_b8": "INTLV", "intlv_b16": "INTLV", "intlv_b32": "INTLV_B32"}
_BITS = {"i64": 64, "u64": 64, "f32": 32, "i32": 32, "u32": 32, "f16": 16, "bf16": 16,
         "i16": 16, "u16": 16, "i8": 8, "u8": 8, "hif8": 8, "e4m3": 8, "e5m2": 8, "e8m0": 8,
         "i4": 4, "fp4_e2m1": 4, "fp4_e1m2": 4}

CAST_LAYOUT = {"zero": "pl.CastLayout.ZERO"}  # pto's enum is ZERO/ONE/TWO/THREE (register
# quarter selectors); our even/odd spellings need the meeting-point mapping before use
MASK_W_DT = {"b8": "pl.DT_INT8", "b16": "pl.DT_FP16", "b32": "pl.DT_FP32", "b64": "pl.DT_INT64"}


from ascriptor.backends.cce.emit import CceGap as _CceGap


# Who owns a refusal. The distinction is not bookkeeping: `unmapped` was where the fixpipe
# `scale` of v9_allhif8 sat for weeks, and pl had taken it all along (D-127).
#   upstream - a pl / pto / bisheng absence or refusal we have SEEN (board error, parse error,
#              or a grep of the surveyed surface). Nothing to build on our side.
#   ours     - the route through pl is known and we have not built it (a lowering, not a line).
#   unmapped - OUR printer has no line for this op or attr and nobody has checked whether pl
#              does. The first thing to do with one of these is to go and look.
GAP_OWNERS = ("upstream", "ours", "unmapped")

# Atomic accumulate is a hardware STATE, not a property of one instruction: two SPRs, an op and
# a dtype (cce's runtime header, tensorutils_cce.h:584: "dav_3510 picks the dtype SPR, then the
# op SPR"), and while they are armed EVERY GM write accumulates. That is what our DSL's `with
# atomic_add():` means, and pl has no such state - it folds the mode into `pl.store(atomic=)`.
#
# So the region -> parameter translation is sound only for the stores that can carry the mode
# themselves. Anything else reaching GM inside the region would silently stop accumulating.
#
# cce DOES deliver the region semantics (its markers own the arming and the disarm, and a
# store inside only contributes its own dtype SPR - backends/cce/emit.py `_atomic_wrap`), so
# this is a real difference in reach, not a shared shortfall: a scalar `SetValueTo(gm[...])`
# after a copy accumulates there and cannot here. It refuses rather than passing silently.
ATOMIC_CARRIERS = frozenset({"dma.ub_to_gm.pad", "dma.l0c_to_gm.nz2nd", "dma.l0c_to_gm.nz2dn"})
ATOMIC_MARKERS = frozenset({"atomic.begin", "atomic.end", "atomic.set_type"})

# Opcodes whose absence upstream is already settled, so the catch-all can say `upstream`
# instead of `unmapped`: convolution has no pl entry point and no pto intrinsic (grep of the
# surveyed surface, docs/pypto-pro-coverage.md ladder item 2).
_PROVEN_ABSENT = frozenset({"dma.l1_to_l0.img2col", "cube.conv2d"})

# Read from installed PyPTO source 86ef830 (2026-09-07): scalar_ops.py
# registers min/max/const; abs/sqrt are tile APIs. simt_ops.py explicitly
# rejects its scalar math handlers outside a SIMT function. These are
# kernel/VF scalar surface gaps, not UB getval/setval gaps (M10-061/062).
# Ordinary integer casts use the documented local PyPTO compatibility supplement.
_SCALAR_ABSENT_UPSTREAM = {
    "scalar.abs": "pl.abs requires destination/source Tiles; pl.simt.abs is restricted to SIMT functions",
    "scalar.sqrt": "pl.sqrt requires destination/source Tiles; pl.simt.sqrt is restricted to SIMT functions",
}

# ... and attrs whose upstream absence has been CHECKED against the installed pl, so
# `guard_attrs` reports them as upstream instead of as our own unlooked-at hole. Add a line
# here every time one of those checks is actually run - that is what turns an `unmapped` gap
# into a settled one (the fixpipe `scale` went the other way and became a mapping, D-127).
_ATTR_ABSENT_UPSTREAM = {
    "hif8_hybrid": "pl.store takes relu_pre_mode / scale / order / atomic / phase and no "
                   "rounding selector; `hybrid` appears nowhere in pl (checked 2026-09-02)",
}


class PyptoGap(_CceGap):
    """An op / form the pypto_pro backend has no line for (reported, never guessed)."""

    def __init__(self, op: Op | None, why: str, *, owner: str = "") -> None:
        assert owner in ("",) + GAP_OWNERS, owner
        loc = getattr(op, "loc", None)
        tag = f"[{owner}] " if owner else ""
        super().__init__(op, why)
        self.args = (f"{tag}{why}" + (f" at {loc}" if loc else ""),)
        self.owner = owner


def _pl_dt(dtype_name: str, op: Op | None = None) -> str:
    """The pypto DT_* constant for one of our dtype names, or the gap that says why not.

    Every lookup goes through here so a dtype pl does not HAVE (c32 / c64) is reported as
    pypto's absence rather than as a KeyError, or as a refusal from whichever of our own
    tables happened to be consulted first."""
    try:
        return PL_DT[dtype_name]
    except KeyError:
        extra = ""
        if dtype_name in ("c32", "c64"):
            # D-130: pl's DT_* set is float / int / bool only. There is no complex constant and
            # no complex immediate spelling either, while cce's complex registers ride in an int
            # carrier whose halves are the parts - a mechanism pl cannot name. A decided gap.
            extra = (" - pl's DT_* set is float / int / bool only, with no complex constant and no "
                     "complex immediate spelling, so the whole c32 / c64 family is an upstream "
                     "absence (decided 2026-09-02, not a todo)")
        raise PyptoGap(op, f"dtype {dtype_name} has no pypto DT_* constant{extra}",
                       owner="upstream") from None


_BUILTINS = frozenset(n for n in dir(builtins) if not n.startswith("_"))


def py_ident(name: str) -> str:
    n = re.sub(r"\W", "_", str(name))
    if n[:1].isdigit():
        n = "_" + n
    # our IR names ops after their opcode tail, so a value can land on a python KEYWORD
    # ("or", "and", "not", "is"): the file then fails to import with a SyntaxError far from
    # the line that named it (board: fd_modified's `or = (cmp_1 | cmp_2)`)
    # A BUILTIN is the quiet half of the same trap: `min = pl.min(...)` imports fine and
    # shadows the builtin for the rest of the module, so it costs nothing until the day a
    # printed line calls one. Today none does - the corpus emits `pl.min` / `pl.range` and
    # never a bare builtin - which is an accident of what this printer spells, not a rule.
    return n + "_" if keyword.iskeyword(n) or keyword.issoftkeyword(n) or n in _BUILTINS else n


def fn_ident(name: str) -> str:
    """A vf or SIMT function name: native Pro did not compile SIMT launches named after a <math.h> function or
    ``free`` (A5-UP-042), so those names take a trailing underscore, as the CCE printer's do (I031)."""
    n = py_ident(name)
    return n + "_" if n in MATH_NAMES or n == "free" else n


def _esize_bits(dtype_name: str) -> int:
    from ...ir.types import DTYPES

    return DTYPES[dtype_name].bits


def _lit(v: Any) -> str:
    if isinstance(v, Literal):
        v = v.value
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        return repr(v)
    raise PyptoGap(None, f"immediate {v!r} has no pypto spelling", owner="upstream")


# pypto renders a FLOAT immediate into the CCE with six decimals (std::to_string), so
# 0.004464285714285714 reaches the hardware as 0.004464f (-6.4e-05 relative) and FLT_MIN/2 as
# 0.0, while an INTEGER immediate is printed exactly. A constant that does not survive that
# rendering therefore rides in as its integer bit pattern plus a (free) bit_cast - D-121.
_IMM_CARRIER = {"f32": ("<f", "<i", "i32"), "f16": ("<e", "<h", "i16")}


def _imm_bits(v: float, dtype: str) -> tuple[int, str] | None:
    """The constant's bit pattern in `dtype` and the integer dtype that carries it, or None."""
    try:
        if dtype == "bf16":
            b = struct.unpack("<I", struct.pack("<f", v))[0]
            h = (b + 0x7FFF + ((b >> 16) & 1)) >> 16  # round to nearest even, as the hardware does
            return (h - 0x10000 if h & 0x8000 else h), "i16"
        spec = _IMM_CARRIER.get(dtype)
        if spec is None:
            return None
        fmt, ifmt, carrier = spec
        return struct.unpack(ifmt, struct.pack(fmt, v))[0], carrier
    except (OverflowError, ValueError):
        return None


def _imm_survives(v: float, dtype: str) -> bool:
    """Does pypto's six-decimal rendering of this immediate reach the register unchanged?"""
    printed = float(f"{v:.6f}")
    if printed == v:
        return True
    a, b = _imm_bits(printed, dtype), _imm_bits(v, dtype)
    return a is None or b is None or a == b


# ------------------------------------------------------------------ scalar environment

# the allocator's per-space slot alignment (ascriptor/passes/addr_alloc.py ALIGN)
_SLOT_ALIGN = {"ub": 32, "l1": 32, "bt": 32, "l0a": 512, "l0b": 512, "l0c": 1024}

FOLD_OPS = {"scalar.add", "scalar.sub", "scalar.mul", "scalar.div", "scalar.mod", "scalar.min",
            "scalar.max", "scalar.ceil_div", "scalar.align", "scalar.const", "scalar.shl",
            "scalar.shr", "scalar.and", "scalar.or", "scalar.xor", "scalar.neg",
            # a comparison whose sides both fold is a constant too - without it the cf.if
            # dead-arm rule below never fires and an unreachable arm is held to the geometry
            # of the arm that runs (chunk_row_cumsum's H < 64 arm, D-115)
            "scalar.cmp", "scalar.select"}


class ScalarEnv:
    """Per-function scalar values: parameter bindings fold to ints, the rest materialise as
    Python locals at their def point (SSA def order = evaluation order, so cell reads are
    exact). ``ref`` returns the spelling of a value at a use site."""

    def __init__(self, mp: "ModulePrinter", fn: Function, side: str) -> None:
        self.mp = mp
        self.fn = fn
        self.side = side
        self.defs: dict[str, Op] = {}
        for op in fn.body.walk():
            for r in op.results:
                self.defs[r.name] = op
        self.names: dict[str, str] = {}  # value name -> python spelling (const or local)
        self.initial_names = initializer_names(fn)
        self.cells: set[str] = set()  # value names that are cells (mutable locals)
        self._vars: dict[str, tuple[int, int, int]] = {}  # exact progressions -> (lo, step, count)
        self._spans: dict[str, tuple[int, int]] = {}  # conservative var bounds (superset of _vars)
        self._inflight: set[str] = set()  # lin() recursion guard
        # A temporary whose value is read once, by a line this side prints, is spelled AT that use
        # instead of on a line of its own: pl's parser let-binds a nested expression itself, so a
        # name per IR op only makes the source longer. The guards (purity, single use, nothing that
        # writes what it reads in between, never into a loop bound or a deeper block) are the IR's
        # and live with the CCE printer's, which has always folded this way.
        self.folds = plan_folds(fn)

    def bind_param(self, p: Value) -> None:
        b = self.mp.bindings or {}
        if p.name in b:
            v = b[p.name]
            self.names[p.name] = repr(int(v) if float(v).is_integer() else float(v))
        else:
            raise PyptoGap(None, f"scalar parameter {p.name} has no binding; the pypto backend "
                                 "specialises kernels per scalar valuation (pass bindings)")

    def _cell_written(self) -> set[str]:
        """Names of cells any ``scalar.set`` or visible mask_update decrement (I040) writes: only these have
        loop-varying values; a never-written cell is its init."""
        w = getattr(self, "_cells_w", None)
        if w is None:
            w = {o.operands[0].name for o in self.fn.body.walk()
                 if o.opcode == "scalar.set" and o.operands} | decremented(self.fn)
            self._cells_w = w
        return w

    def _core_range(self, d: Op) -> tuple[int, int, int] | None:
        """Static (lo, step, count) for a runtime core id: the value is per-core but its range
        is known at compile time (sub-block pairs are 2 wide; vec/cube ids need block_dim)."""
        if d.opcode == "core.sub_block_idx":
            return (0, 1, 1 if getattr(self.mp, "mode", "mix") == "vec" else 2)
        if d.opcode == "core.get_sat_flag":
            # a saturation flag is ONE CTRL bit - the printer reads it as
            # pl.get_ctrl_spr(bit, bit) and the interpreter as int(bool(...)) - so the value is
            # runtime but its whole value set is {0, 1}, which is what an enumerated strip needs
            return (0, 1, 2)
        bd = getattr(self.mp, "block_dim", None)
        if not bd:
            return None
        vec_only = getattr(self.mp, "mode", "mix") == "vec"
        if d.opcode == "core.cube_idx":
            return (0, 1, int(bd))
        if d.opcode == "core.vec_idx":
            # in an AIV-only binary block_dim already counts AIVs, so vec_idx spans it exactly
            return (0, 1, int(bd) if vec_only else int(bd) * 2)
        return None

    def fold(self, v: Any) -> int | None:
        """A compile-time int for parameter/constant arithmetic, else None.

        The interval and the affine form both compute in unbounded ints, so the point they
        agree on is checked against the value's own declared type before it is a constant:
        a sum or shift that leaves the type is a wrap on the device, never the exact int."""
        r = self.frange(v)
        if r is None or r[0] != r[1]:
            return None
        dt = getattr(getattr(v, "type", None), "dtype", None)
        if dt is not None and dt.is_integer and not limits(dt)[0] <= r[0] <= limits(dt)[1]:
            return None
        return r[0]

    def lin(self, v: Any) -> tuple[int, dict[str, int]] | None:
        """``(b, {var: coef})`` for ``b + sum(coef * var)`` over loop variables and core ids -
        the linear form keeps correlation, so ``min(begin + n, total) - begin`` folds where
        interval arithmetic loses it. Vars register their static range in ``self._vars``."""
        if isinstance(v, Literal):
            v = v.value
        if isinstance(v, (bool, int)):
            return (int(v), {})
        name = getattr(v, "name", None)
        if name is None or isinstance(v, str):
            return None
        n = self.names.get(name)
        if n is not None:
            try:
                return (int(n), {})
            except ValueError:
                pass
        d = self.defs.get(name)
        if d is None:
            return None
        if name in self._inflight:
            return None
        self._inflight.add(name)
        try:
            return self._lin_def(name, d)
        finally:
            self._inflight.discard(name)

    def _lin_def(self, name: str, d: Op) -> tuple[int, dict[str, int]] | None:
        if d.opcode == "cf.for" and d.results and d.results[0].name == name:
            bs = [self.fold(o) for o in d.operands]
            if all(x is not None for x in bs):
                lo, hi, step = bs
                if step == 0 or (hi - lo) * step <= 0:
                    return None
                cnt = (abs(hi - lo) + abs(step) - 1) // abs(step)
                self._vars[name] = (lo, step, cnt)
                self._spans[name] = (min(lo, lo + step * (cnt - 1)), max(lo, lo + step * (cnt - 1)))
                return (0, {name: 1})
            # per-core bounds (e.g. lo = idx * per_core, hi = min(lo + per_core, total)):
            # when hi - lo is a compile-time constant, the variable is lo + step*k - a
            # single-trip loop IS its lower bound (correlation with the core id survives)
            la = self.lin(d.operands[0])
            lh = self.lin(d.operands[1])
            stp = self.fold(d.operands[2])
            if la is not None and lh is not None and stp is not None and stp > 0:
                dv = dict(lh[1])
                for k2, c2 in la[1].items():
                    dv[k2] = dv.get(k2, 0) - c2
                if not {k2: c2 for k2, c2 in dv.items() if c2}:  # constant trip span
                    c = lh[0] - la[0]
                    if c <= 0:
                        return None  # an empty loop: every use is dead code
                    cnt = -(-c // stp)
                    if cnt == 1:
                        return la
                    kname = name + ".trip"
                    self._vars[kname] = (0, stp, cnt)
                    self._spans[kname] = (0, stp * (cnt - 1))
                    vs2 = dict(la[1])
                    vs2[kname] = vs2.get(kname, 0) + 1
                    return (la[0], vs2)
            rlo = self.frange(d.operands[0])
            rhi = self.frange(d.operands[1])
            if rlo is None or rhi is None or stp is None or stp <= 0 or rhi[1] - 1 < rlo[0]:
                return None
            self._spans[name] = (rlo[0], rhi[1] - 1)
            return (0, {name: 1})
        cr = self._core_range(d)
        if cr is not None:
            self._vars[name] = cr
            self._spans[name] = (cr[0], cr[0] + cr[1] * (cr[2] - 1))
            return (0, {name: 1})
        if d.opcode == "scalar.cell":
            if name not in self._cell_written():
                init = d.attrs.get("init", 0)
                value = self.lin(init)
                # Keep the negative-sentinel guard even after all writes disappear.
                if self.side != "vf" and value is not None and not value[1] and value[0] < 0:
                    return None
                return value
            return None
        if d.opcode in ("core.cube_num", "core.vec_num", "core.sub_block_num"):
            vec_only = getattr(self.mp, "mode", "mix") == "vec"
            if d.opcode == "core.sub_block_num":
                return (1 if vec_only else 2, {})
            bd = getattr(self.mp, "block_dim", None)
            if not bd:
                return None
            # AIV-only: block_dim counts AIVs, so cube_num == vec_num == block_dim
            return (int(bd) * (2 if d.opcode == "core.vec_num" and not vec_only else 1), {})
        if d.opcode == "scalar.const":
            x = d.attrs.get("value")
            return (int(x), {}) if isinstance(x, (int, bool)) else None
        if d.opcode == "scalar.neg":
            a = self.lin(d.operands[0])
            return None if a is None else (-a[0], {k: -c for k, c in a[1].items()})
        if d.opcode in ("scalar.add", "scalar.sub"):
            a, b = (self.lin(o) for o in d.operands)
            if a is None or b is None:
                return None
            sign = 1 if d.opcode == "scalar.add" else -1
            vs = dict(a[1])
            for k, c in b[1].items():
                vs[k] = vs.get(k, 0) + sign * c
            return (a[0] + sign * b[0], {k: c for k, c in vs.items() if c})
        if d.opcode == "scalar.mul":
            a, b = (self.lin(o) for o in d.operands)
            if a is None or b is None:
                return None
            if a[1] and b[1]:
                return None  # a product of two variables is not linear
            k, l = (a, b) if not a[1] else (b, a)
            return (k[0] * l[0], {n2: k[0] * c for n2, c in l[1].items() if k[0] * c})
        if d.opcode in ("scalar.min", "scalar.max"):
            a, b = (self.lin(o) for o in d.operands)
            if a is None or b is None:
                return None
            vs = dict(a[1])
            for k, c in b[1].items():
                vs[k] = vs.get(k, 0) - c
            r = self._lin_range((a[0] - b[0], {k: c for k, c in vs.items() if c}))
            if r[1] <= 0:  # a <= b everywhere
                return a if d.opcode == "scalar.min" else b
            if r[0] >= 0:  # a >= b everywhere
                return b if d.opcode == "scalar.min" else a
            return None
        if d.results:  # outside the linear set (div, align, ...): constants still enter,
            r = self._irange(d.results[0])  # and any bounded value becomes a span-only var -
            if r is not None:  # same-name occurrences still cancel exactly
                if r[0] == r[1]:
                    return (r[0], {})
                self._spans[name] = r
                return (0, {name: 1})
        return None

    def _lin_range(self, l: tuple[int, dict[str, int]]) -> tuple[int, int]:
        lo = hi = l[0]
        for name, c in l[1].items():
            vlo, vhi = self._spans[name]
            x0, x1 = sorted((c * vlo, c * vhi))
            lo += x0
            hi += x1
        return (lo, hi)

    def frange(self, v: Any) -> tuple[int, int] | None:
        """Inclusive bounds for a value when they fold, else None: the linear form when it
        applies (it keeps the correlation the interval recursion loses), the plain interval
        recursion otherwise."""
        nm = getattr(v, "name", None)
        d0 = self.defs.get(nm) if isinstance(nm, str) else None
        if d0 is not None and d0.opcode == "scalar.cmp":
            return self._irange(v)  # a comparison is exact or nothing; lin() would only span it
        l = self.lin(v)
        if l is not None:
            return self._lin_range(l)
        return self._irange(v)

    def fupper(self, v: Any) -> int | None:
        """A sound inclusive UPPER bound, for a consumer that asks only how large a value gets.

        `frange` is two-sided and its recursion discards the whole interval when any operand is
        unbounded. That widening is right for an interval and wrong for a capacity: three forms
        state an upper bound with the other operand unknown.

        * ``min(a, b)`` is at most either side's upper bound, so ONE bounded operand is enough.
        * ``max(a, b)`` needs both, and says so.
        * ``x & m`` with a non-negative constant mask has only ``m``'s bits set whatever ``x``
          is, so it lies in ``[0, m]``.

        The third is the one that unblocks the other two in practice: the front end strength-
        reduces ``%`` into a masked ``and``, so a tail extent ``min(128, N - lag_n0)`` carries
        one operand whose chain passes through it and loses a bound the other operand states
        outright (M10-100). Everything else defers to `frange`, so no value that already
        bounded changes its answer, and this is a bound - never a guess: a caller may declare
        it as capacity because the runtime extent cannot exceed it.
        """
        r = self.frange(v)
        if r is not None:
            return r[1]
        name = getattr(v, "name", None)
        d = self.defs.get(name) if isinstance(name, str) else None
        if d is None:
            return None
        if d.opcode in ("scalar.min", "scalar.max"):
            ups = [self.fupper(o) for o in d.operands]
            known = [u for u in ups if u is not None]
            if d.opcode == "scalar.max":
                return max(known) if len(known) == len(ups) else None
            return min(known) if known else None
        if d.opcode == "scalar.and":
            masks = [self.fold(o) for o in d.operands]
            known = [m for m in masks if isinstance(m, int) and not isinstance(m, bool) and m >= 0]
            return min(known) if known else None
        return None

    def _irange(self, v: Any) -> tuple[int, int] | None:
        """Interval recursion: a ``cf.for`` variable takes its static trip range, so a tail
        clamp the binding renders loop-invariant (for example ``min(SPLIT, N - n0)`` with
        ``N % SPLIT == 0``) still folds to its constant."""
        if isinstance(v, Literal):
            v = v.value
        if isinstance(v, (bool, int)):
            return (int(v), int(v))
        name = getattr(v, "name", None)  # a Value, or a DimValue in a MemType dim
        if name is None or isinstance(v, str):
            return None
        n = self.names.get(name)
        if n is not None:
            try:
                return (int(n), int(n))
            except ValueError:
                pass  # a materialised local: its def chain may still bound it
        d = self.defs.get(name)
        if d is None:
            return None
        if d.opcode == "cf.for" and d.results and d.results[0].name == name:
            bs = [self.fold(o) for o in d.operands]
            if any(x is None for x in bs):
                return None
            lo, hi, step = bs
            if step == 0 or (hi - lo) * step <= 0:  # an empty loop: every use is dead code
                return None
            last = lo + ((hi - lo - (1 if step > 0 else -1)) // step) * step
            return (min(lo, last), max(lo, last))
        cr = self._core_range(d)
        if cr is not None:
            lo, step, count = cr
            return (lo, lo + step * (count - 1))
        if d.opcode not in FOLD_OPS:
            return None
        if d.opcode == "scalar.const":
            x = d.attrs.get("value")
            return (int(x), int(x)) if isinstance(x, (int, bool)) else None
        if d.opcode == "scalar.mod":
            rb = self.frange(d.operands[1])
            ra = self.frange(d.operands[0])
            if ra is not None and rb is not None and ra[0] == ra[1] and rb[0] == rb[1] and rb[0] != 0:
                value = evaluate("mod", [ra[0], rb[0]], d.results[0].type.dtype, rounding=rounding(d))
                return None if value is None else (value, value)
            if rb is not None and rb[0] == rb[1] and rb[0] > 0:
                k = rb[0]
                if ra is not None and 0 <= ra[0] and ra[1] < k:
                    return ra  # the mod never engages
                if rounding(d) == "floor":
                    return (0, k - 1)
                if ra is not None and ra[0] >= 0:
                    return (0, k - 1)
                return (-(k - 1), k - 1)
            return None
        if d.opcode == "scalar.select":
            condition = self.fold(d.operands[0])
            if condition is not None:
                return self.frange(d.operands[1 if condition else 2])
            ranges = [self.frange(o) for o in d.operands[1:]]
            if any(r is None for r in ranges):
                return None
            return (min(r[0] for r in ranges), max(r[1] for r in ranges))
        rs = [self.frange(o) for o in d.operands]
        if any(r is None for r in rs):
            return None
        a = rs[0]
        b = rs[1] if len(rs) > 1 else None
        if d.opcode == "scalar.align":
            divisor = int(d.attrs["n"])
            b = (divisor, divisor)
        if d.opcode == "scalar.cmp":
            # a comparison whose sides both fold IS a constant - and deciding it matters:
            # specialisation makes whole cf.if arms unreachable, and the printer must not
            # hold an unreachable arm to the geometry of the arm that runs
            raw = d.attrs.get("pred", d.attrs.get("mode"))
            mode = str(getattr(raw, "name", raw))
            fn = {"lt": lambda x, y: x < y, "le": lambda x, y: x <= y,
                  "gt": lambda x, y: x > y, "ge": lambda x, y: x >= y,
                  "eq": lambda x, y: x == y, "ne": lambda x, y: x != y}.get(mode)
            if fn is None or b is None or a[0] != a[1] or b[0] != b[1]:
                return None
            k = int(fn(a[0], b[0]))
            return (k, k)
        if all(r[0] == r[1] for r in rs):  # every operand is a constant: the shared exact table
            args = [r[0] for r in rs]
            k = evaluate(d.opcode.removeprefix("scalar."), args, d.results[0].type.dtype,
                         rounding=rounding(d), n=d.attrs.get("n"), pred=d.attrs.get("pred"))
            return None if k is None else (int(k), int(k))
        if d.opcode == "scalar.add":
            return (a[0] + b[0], a[1] + b[1])
        if d.opcode == "scalar.sub":
            return (a[0] - b[1], a[1] - b[0])
        if d.opcode == "scalar.mul":
            xs = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
            return (min(xs), max(xs))
        if d.opcode == "scalar.min":
            return (min(a[0], b[0]), min(a[1], b[1]))
        if d.opcode == "scalar.max":
            return (max(a[0], b[0]), max(a[1], b[1]))
        if d.opcode == "scalar.neg":
            return (-a[1], -a[0])
        if b is not None and b[0] == b[1] and b[0] > 0:  # monotone in a positive constant divisor
            k = b[0]
            if d.opcode == "scalar.div":
                return (integer_divmod(a[0], k, rounding(d))[0], integer_divmod(a[1], k, rounding(d))[0])
            if d.opcode == "scalar.ceil_div":
                return (-(-a[0] // k), -(-a[1] // k))
            if d.opcode == "scalar.align":
                return (-(-a[0] // k) * k, -(-a[1] // k) * k)
            if d.opcode == "scalar.shl":
                return (a[0] << k, a[1] << k)
            if d.opcode == "scalar.shr":
                return (a[0] >> k, a[1] >> k)
        return None

    def faffine(self, v: Any) -> tuple[int, int, int] | None:
        """``(base, step, count)`` when the value set of ``v`` is the arithmetic progression
        of at most one loop variable or core id (constants are ``(c, 0, 1)``): the linear
        form's single-variable case."""
        l = self.lin(v)
        if l is None:
            return None
        b, vs = l
        if not vs:
            return (b, 0, 1)
        if len(vs) > 1:
            return None
        (name, c), = vs.items()
        if name not in self._vars:
            return None  # a span-only variable (per-core loop bounds) cannot flatten
        lo, step, count = self._vars[name]
        return (b + c * lo, c * step, count)
    def ref(self, v: Any) -> str:
        if isinstance(v, Literal) or isinstance(v, (int, float, bool)):
            return _lit(v)
        n = self.names.get(v.name)
        if n is None:
            raise PyptoGap(self.defs.get(v.name), f"scalar {v.name} used before its def was printed")
        return n

    def define(self, v: Value, hint: str | None = None) -> str:
        n = py_ident(hint or self.initial_names.get(v.name, v.name))
        # a value that will be folded into its use never reaches the source, so it takes no name
        # out of the module's namespace; its spelling is replaced by the expression in scalar_op
        n = n if v.name in self.folds else self.mp.unique(n)
        self.names[v.name] = n
        return n

    # -------- op printing; returns the lines to emit at the def point (may be [])

    def cell_snapshot(self, value: Any) -> str:
        text = self.ref(value) if isinstance(value, Value) else _lit(value)
        if (self.side in ("cube", "vec") and isinstance(value, Value)
                and isinstance(value.type, CellType) and value.type.dtype.kind in ("int", "uint")):
            # M10-070: PyPTO can turn a bare cell copy into a loop-argument alias,
            # then commit that argument before its old value is copied. Keep an
            # integer identity as an IR definition until native C++ is emitted.
            # Do not route this through our core-range folding or apply float +0.
            return f"({text} + pl.get_block_idx() * 0)"
        return text

    def scalar_op(self, op: Op, emitter: "SideEmitter | VfPrinter") -> list[str]:
        """The lines this op contributes; a single-use temporary contributes none and is spelled at its use."""
        lines = self.print_scalar_op(op, emitter)
        r = op.results[0] if op.results else None
        if r is None or r.name not in self.folds or len(lines) != 1:
            return lines
        head = f"{self.names.get(r.name)} = "
        if not lines[0].startswith(head):
            return lines  # a form that is not one plain assignment keeps its line
        self.names[r.name] = paren(lines[0][len(head):])
        return []

    def print_scalar_op(self, op: Op, emitter: "SideEmitter | VfPrinter") -> list[str]:
        code = op.opcode
        r = op.results[0] if op.results else None
        if r is not None:
            k = self.fold(r)
            if k is not None:
                self.names[r.name] = repr(k)
                return []
        if code == "scalar.cell":
            init = op.attrs.get("init", 0)
            n = self.define(r, op.attrs.get("name"))
            self.cells.add(r.name)
            value = self.cell_snapshot(init)
            if self.side != "vf" and isinstance(init, int) and init < 0:
                # A mutable negative sentinel can survive after specialized
                # assignments disappear. Keep it a runtime scalar: PyPTO
                # validates constant negative GM offsets even in guarded arms.
                # This preserves the value and guard; it never clamps an address.
                value = f"({value} + pl.get_block_idx() * 0)"
            return [f"{n} = {value}"]
        if code == "scalar.set":
            cell, val = op.operands
            return [f"{self.ref(cell)} = {self.cell_snapshot(val)}"]
        if code == "scalar.const":
            self.names[r.name] = _lit(op.attrs.get("value"))
            return []
        if code == "scalar.add" and self.side in ("cube", "vec"):
            from ...ir.scalar_flow import snapshot_cell
            cell = snapshot_cell(op)
            if cell is not None:
                return [f"{self.define(r)} = {self.cell_snapshot(cell)}"]
        if code == "scalar.cast":
            from .scalar_width import supports_cast
            if not supports_cast(op):
                raise PyptoGap(op, "ordinary pl.cast supports only 8/16/32/64-bit integer scalar conversions",
                               owner="upstream")
            # the scalar overload of pl.cast is a local supplement, not stock pl
            self.mp.supplements.need("integer-cast", op)
            return [f"{self.define(r)} = pl.cast({self.ref(op.operands[0])}, {_pl_dt(r.type.dtype.name)})"]
        if self.side == "vf" and code in ("scalar.div", "scalar.mod"):
            from .scalar_width import vf_divmod_operands
            operands = vf_divmod_operands(op, self.ref, _pl_dt)
            if operands is not None:
                a, b = operands
                # a narrowed operand is spelled with that same overload; a pair that came out as
                # typed literals alone (pl.const is stock) needs nothing
                if any("pl.cast(" in s for s in operands):
                    self.mp.supplements.need("integer-cast", op)
                operator = "//" if code == "scalar.div" else "%"
                return [f"{self.define(r)} = ({a} {operator} {b})"]
        if code in SCALAR_BIN:
            a, b = op.operands
            n = self.define(r)
            return [f"{n} = ({self.ref(a)} {SCALAR_BIN[code]} {self.ref(b)})"]
        if code == "scalar.div":
            a, b = op.operands
            n = self.define(r)
            t = r.type
            fdiv = isinstance(t, ScalarType) and t.dtype.is_float
            return [f"{n} = ({self.ref(a)} {'/' if fdiv else '//'} {self.ref(b)})"]
        if code == "scalar.ceil_div":
            a, b = op.operands
            n = self.define(r)
            return [f"{n} = (({self.ref(a)} + {self.ref(b)} - 1) // {self.ref(b)})"]
        if code == "scalar.align":
            a = op.operands[0]
            b = int(op.attrs["n"])
            n = self.define(r)
            return [f"{n} = ((({self.ref(a)} + {self.ref(b)} - 1) // {self.ref(b)}) * {self.ref(b)})"]
        if code in ("scalar.min", "scalar.max"):
            a, b = op.operands
            n = self.define(r)
            if self.side == "vf":
                # Inside a vf the AIV has no 64-bit scalar ALU, and every pl scalar expression is
                # 64-bit (D-139), so pl.min / pl.max are EMULATED in the vector instruction
                # stream. The compare-and-select spelling of the same value is not: on
                # fd_modified's tail softmax, whose per-key-row `QLANES * Min(Max(valid_n - n, 0),
                # 1)` runs 192 times per call, it measures 13.64 -> 10.64 us (aiv_vec 75 913 ->
                # 45 665 cycles) with the output unchanged, and the same number comes out whether
                # the clamp is respelled as one select or all three ops are - so it is the
                # spelling, not the algebra. Outside a vf the scalar unit runs pl.min / pl.max
                # natively and there is nothing to win, so only the vf side is respelled.
                # The comparison is `a < b` either way, so this is the cce backend's own
                # definition character for character - `Min(a, b) = (a < b) ? a : b` and
                # `Max(a, b) = (a < b) ? b : a` (tensorutils_cce.h:228) - which keeps the two
                # backends' answer identical for a tie and for a NaN operand alike.
                x, y = self.ref(a), self.ref(b)
                then_, else_ = (x, y) if code == "scalar.min" else (y, x)
                return [f"{n} = ({then_} if ({x} < {y}) else {else_})"]
            return [f"{n} = pl.{code[7:]}({self.ref(a)}, {self.ref(b)})"]
        if code == "scalar.neg":
            n = self.define(r)
            return [f"{n} = (0 - {self.ref(op.operands[0])})"]
        if code == "scalar.not":
            n = self.define(r)
            operator = "not " if r.type.dtype.name == "b1" else "~"
            return [f"{n} = ({operator}{self.ref(op.operands[0])})"]
        if code == "scalar.cmp":
            a, b = op.operands
            mode = op.attrs.get("pred", op.attrs.get("mode", ""))
            mode = str(getattr(mode, "name", mode))
            if mode not in SCALAR_CMP:
                raise PyptoGap(op, f"scalar.cmp mode {mode!r} has no python spelling")
            n = self.define(r)
            return [f"{n} = ({self.ref(a)} {SCALAR_CMP[mode]} {self.ref(b)})"]
        if code == "scalar.select":
            c, a, b = op.operands
            n = self.define(r)
            return [f"{n} = ({self.ref(a)} if {self.ref(c)} else {self.ref(b)})"]
        if code in ("scalar.load", "scalar.store"):
            address = getattr(emitter, "scalar_address", None)
            if address is None:
                raise PyptoGap(op, f"{code} requires a kernel side function", owner="ours")
            base, off = address(op, op.operands[0], op.operands[1])
            if code == "scalar.store":
                return [f"pl.setval({base}, {off}, {self.ref(op.operands[2])})"]
            n = self.define(r)
            return [f"{n} = pl.getval({base}, {off})"]
        if code in ("list.count", "list.item_dim"):
            # a GMList is specialised to THIS call's members (D-119): the count and every
            # member's own ragged extent are literals of the translation, exactly as a shape
            # binding is - which is what lets the member loop unroll into real parameters.
            lst = op.operands[0]
            members = self.mp.lists.get(lst.name)
            if members is None:
                raise PyptoGap(op, f"{code} on {lst.name}: this call passed no member shapes to "
                                   "specialise the list with")
            if code == "list.count":
                k = len(members)
            else:
                i = self.fold(op.operands[1])
                if i is None:
                    raise PyptoGap(op, "list.item_dim at a runtime member index: the member loop "
                                       "has to unroll for a list to have any pl spelling at all")
                d = int(op.attrs.get("dim", 0))
                k = int(members[int(i)][d])
            assert r is not None
            self.names[r.name] = str(k)
            return []
        # core indices
        if code.startswith("core."):
            return self.core_op(op)
        if code in _SCALAR_ABSENT_UPSTREAM:
            raise PyptoGap(op, _SCALAR_ABSENT_UPSTREAM[code], owner="upstream")
        raise PyptoGap(op, f"scalar op {code} is outside the pypto surface")

    def core_op(self, op: Op) -> list[str]:
        code = op.opcode
        table = {
            # pto's block queries follow AscendC: inside a Vector section get_block_idx()
            # codegens to block*subblockdim+subblockid = the GLOBAL AIV index
            # (backend_cce_ops.cpp MakeBlockGetBlockIdxCodegenCCE) - never re-derive it
            # with *2+sub, that double-counts and parked the second AIV (D-087). The flip
            # side (D-088): CUBE id asked from a Vector section must divide that global
            # index back down, or the two AIVs of one AIC dispatch as different cubes and
            # skip their crosscore sets - the AIC then waits forever (board: 507014).
            "core.cube_idx": "pl.get_block_idx()" if self.side == "cube"
                             else "(pl.get_block_idx() // pl.get_subblock_num())",
            "core.cube_num": "pl.get_block_num()",
            "core.sub_block_idx": "pl.get_subblock_idx()",
            "core.vec_idx": "pl.get_block_idx()",
            "core.vec_num": "(pl.get_block_num() * pl.get_subblock_num())",
        }
        expr = table.get(code)
        if expr is None:
            raise PyptoGap(op, f"core op {code} is outside the pypto surface")
        n = self.define(op.results[0])
        return [f"{n} = {expr}"]


# ------------------------------------------------------------------ vf functions


_SIMT_MATH = {"exp", "exp2", "log", "log2", "log1p", "sin", "cos", "tanh", "sqrt", "rsqrt",
              "rint", "round", "floor", "ceil", "trunc", "isnan", "isinf", "isfinite",
              "abs", "min", "max", "fmod", "fma", "mul_hi"}
_SIMT_RENAME = {"popc": "popcount"}
_CMP_PY = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}


class SimtEmitter:
    """One ``@simt`` function -> one module-level ``@pl.simt.function`` definition.

    The body is an SSA-direct translation: every op becomes one python assignment, memory
    accesses subscript the parameter through its (binding-folded) declared shape, and the
    launch-site thread count becomes ``max_threads``. pto's simt parser shares the SIMD
    control-flow parser, so ``pl.range`` loops and ``if`` print exactly as in sections."""

    def __init__(self, mp: "ModulePrinter", fn: Function, threads: int,
                 binds: dict[str, int] | None = None) -> None:
        self.mp = mp
        self.fn = fn
        self.threads = threads
        self.lines: list[str] = []
        self.indent = 1
        self.names: dict[str, str] = {}
        self.extra: list[str] = []  # hidden trailing params the launch site must supply
        self.constants: dict[tuple[str, int], str] = {}
        self.env = ScalarEnv(mp, fn, "vec")  # dimension folding only - no core ops inside simt
        for q in fn.params:
            if isinstance(q.type, ScalarType) and q.name in (binds or {}):
                v = binds[q.name]
                self.env.names[q.name] = repr(int(v) if float(v).is_integer() else float(v))

    def emit(self, line: str) -> None:
        self.lines.append("    " * self.indent + line)

    def ref(self, x: Any) -> str:
        if isinstance(x, Literal):
            return _lit(x)
        if isinstance(x, (int, float, bool)):
            return repr(x)
        n = self.names.get(x.name)
        if n is None:
            raise PyptoGap(None, f"simt value {x.name} used before its defining op was printed")
        return n

    def define(self, v: Value, hint: str | None = None) -> str:
        n = py_ident(hint or v.name)
        self.names[v.name] = n
        return n

    def scalar_ref(self, value: Any, kind: Any) -> str:
        """Pass typed constants from the launch site: SIMT rejects pl.const itself."""
        literal = value.value if isinstance(value, Literal) else value
        if isinstance(literal, (int, bool)) and isinstance(kind, ScalarType) and kind.dtype.is_integer:
            key = (kind.dtype.name, int(literal))
            if key not in self.constants:
                used = set(self.names.values()) | {py_ident(v.name) for op in self.fn.walk() for v in op.results}
                name = f"_ascr_const_{len(self.constants)}"
                while name in used:
                    name += "_"
                self.constants[key] = name
            return self.constants[key]
        return self.ref(value)

    def _dims(self, m: Value) -> list[int]:
        out = []
        for d in m.type.dims:
            k = d if isinstance(d, int) else self.env.fold(d)
            if k is None:
                raise PyptoGap(None, f"simt access to {m.name}: dimension {d!r} is not static "
                                     "after binding specialisation")
            out.append(int(k))
        return out

    def index(self, m: Value, i: Any) -> str:
        """Linear element index -> declared-shape subscript."""
        dims = self._dims(m)
        ix = self.ref(i)
        if len(dims) == 1:
            return f"[{ix}]"
        if len(dims) != 2:
            raise PyptoGap(None, f"simt access to {m.name}: rank {len(dims)} unsupported")
        rows, cols = dims
        if rows == 1:
            return f"[0, {ix}]"
        return f"[{paren(unparen(ix))} // {cols}, {paren(unparen(ix))} % {cols}]"

    def render(self) -> tuple[str, str]:
        py = fn_ident(self.fn.name)
        sig = []
        for q in self.fn.params:
            t = q.type
            n = self.define(q)
            if isinstance(t, MemType) and t.space == "gm":
                dims = ", ".join(str(d) for d in self._dims(q))
                sig.append(f"{n}: pl.Tensor[[{dims}], {_pl_dt(t.dtype.name)}]")
            elif isinstance(t, MemType) and t.space == "ub":
                sig.append(n)
            elif isinstance(t, ScalarType):
                sig.append(f"{n}: {_pl_dt(t.dtype.name)}")
            else:
                raise PyptoGap(None, f"simt parameter {q.name} of type {t}: no pl spelling")
        for op in self.fn.body.ops:
            self.op(op)
        sig += [f"_ascr_{e}: pl.DT_INT32" for e in self.extra]
        sig += [f"{name}: {_pl_dt(dt)}" for (dt, _), name in self.constants.items()]
        # a SIMT template is a vector function in mode="simt" (upstream 2b49dbfad, 2026-09-15,
        # replaced the separate pl.simt.function decorator); max_threads stays its own keyword
        head = [f'@pl.vector_function(mode="simt", max_threads={self.threads})',
                f"def {py}({', '.join(sig)}):"]
        body = self.lines or ["    pass"]
        return py, "\n".join(head + body) + "\n\n"

    def op(self, op: Op) -> None:
        code = op.opcode
        if code == "simt.thread_id":
            self.emit(f"{self.define(op.results[0])} = pl.simt.linear_thread_idx()")
            return
        if code == "simt.thread_num":
            self.emit(f"{self.define(op.results[0])} = pl.simt.block_dim().x")
            return
        if code == "simt.block_idx":
            # our simt_block_idx() is the CUBE CORE id (kernel-doc contract), not pto's
            # launch-grid block: it rides in as a hidden trailing parameter
            if "cube_idx" not in self.extra:
                self.extra.append("cube_idx")
            self.emit(f"{self.define(op.results[0])} = _ascr_cube_idx")
            return
        if code == "simt.block_num":
            if "cube_num" not in self.extra:
                self.extra.append("cube_num")
            self.emit(f"{self.define(op.results[0])} = _ascr_cube_num")
            return
        if code == "simt.load":
            m, i = op.operands
            self.emit(f"{self.define(op.results[0])} = {self.ref(m)}{self.index(m, i)}")
            return
        if code == "simt.store":
            m, i, v = op.operands
            self.emit(f"{self.ref(m)}{self.index(m, i)} = {self.ref(v)}")
            return
        if code == "simt.atomic":
            kind = str(op.attrs.get("op"))
            m, i = op.operands[0], op.operands[1]
            # pto's atomics type-check the VALUE operands against the target element dtype
            # strictly (a loop variable arrives as `index` - board reject); route every value
            # through pl.simt.cast, which folds when the dtype already matches
            if kind in ("exch", "cas"):
                # four board rounds: pto's simt scalar typing derives `index` for these value
                # positions and pl.simt.cast refuses index (+0 does not promote it)
                raise PyptoGap(op, f"simt.atomic {kind}: pto types the value operand as index "
                                   "and pl.simt.cast refuses index (board-probed, 4 rounds)",
                               owner="upstream")
            dt = _pl_dt(m.type.dtype.name)
            # a COMPILE-TIME value goes in as a literal: pto types a python int as `index` and
            # pl.simt.cast refuses index -> uint32 (board, simt_atomic_incdec's ring limit)
            rest = ", ".join(str(k) if (k := self.env.fold(o)) is not None
                             else f"pl.simt.cast({self.ref(o)}, {dt})" for o in op.operands[2:])
            r = self.define(op.results[0]) if op.results else None
            call = f"pl.simt.atomic_{kind}({self.ref(m)}{self.index(m, i)}, {rest})"
            self.emit(f"{r} = {call}" if r else call)
            return
        if code in ("simt.threadfence", "simt.threadfence_block"):
            self.emit(f"pl.simt.{code.split('.', 1)[1]}()")
            return
        if code == "simt.barrier":
            # D-??: this branch read `simt.sync_workitems` — an opcode that does not exist. The
            # registry has always spelled the thread barrier `simt.barrier` (legacy name
            # `simt_thread_barrier`), so the branch never matched and every SIMT barrier fell
            # through to the generic gap, while `capabilities()` declared the same wrong name.
            self.emit("pl.simt.syncthreads()")
            return
        if code in ("simt.ffs", "simt.popc"):
            # pl.simt.popcount admits only uint32/uint64: bit-cast signed sources through
            # pl.simt.cast (same width, value-preserving for the bit ops here)
            raise PyptoGap(op, f"{code}: pl.simt.popcount admits only uint32/uint64 and pto's "
                               "simt scalar typing derives `index` for the input here; "
                               "pl.simt.cast refuses index (board-probed, 4 rounds)", owner="upstream")
        if code == "simt.cast":
            x = self.ref(op.operands[0])
            dt = _pl_dt(op.results[0].type.dtype.name)
            self.emit(f"{self.define(op.results[0])} = pl.simt.cast({x}, {dt})")
            return
        if code.startswith("simt."):
            tail = code.split(".", 1)[1]
            name = _SIMT_RENAME.get(tail, tail)
            if tail in _SIMT_MATH or tail in _SIMT_RENAME:
                args = ", ".join(self.ref(o) for o in op.operands)
                self.emit(f"{self.define(op.results[0])} = pl.simt.{name}({args})")
                return
            raise PyptoGap(op, f"simt op {code} has no pl.simt spelling")
        if code in ("scalar.add", "scalar.sub", "scalar.mul", "scalar.mod", "scalar.and"):
            sym = "and" if code == "scalar.and" and op.results[0].type.dtype.name == "b1" else SCALAR_BIN[code]
            a, b = (self.scalar_ref(o, op.results[0].type) for o in op.operands)
            self.emit(f"{self.define(op.results[0])} = ({a}) {sym} ({b})")
            return
        if code == "scalar.div":
            a, b = (self.scalar_ref(o, op.results[0].type) for o in op.operands)
            sym = "/" if op.results[0].type.dtype.kind == "float" else "//"
            self.emit(f"{self.define(op.results[0])} = ({a}) {sym} ({b})")
            return
        if code == "scalar.neg":
            self.emit(f"{self.define(op.results[0])} = -({self.ref(op.operands[0])})")
            return
        if code in ("scalar.min", "scalar.max"):
            if op.results[0].type.dtype.name == "f32":
                raise PyptoGap(op, f"pl.simt.{code[7:]} skips a NaN operand; f32 {code} is IEEE minimum/maximum "
                                   "(RFC-0001 §6.16)", owner="upstream")
            a, b = (self.ref(o) for o in op.operands)
            self.emit(f"{self.define(op.results[0])} = pl.simt.{code.split('.')[1]}({a}, {b})")
            return
        if code == "scalar.cmp":
            pred = _CMP_PY.get(str(op.attrs.get("pred")))
            if pred is None:
                raise PyptoGap(op, f"simt scalar.cmp pred {op.attrs.get('pred')!r} unsupported")
            kind = next((value.type for value in op.operands if isinstance(value, Value)), None)
            a, b = (self.scalar_ref(o, kind) for o in op.operands)
            self.emit(f"{self.define(op.results[0])} = ({a}) {pred} ({b})")
            return
        if code == "scalar.select":
            condition = self.ref(op.operands[0])
            yes, no = (self.scalar_ref(value, op.results[0].type) for value in op.operands[1:])
            self.emit(f"{self.define(op.results[0])} = ({yes} if {condition} else {no})")
            return
        if code == "cf.for":
            start, stop, step = (list(op.operands) + [1])[:3]
            var = self.define(op.results[0], op.attrs.get("name"))
            lo, hi = self.ref(start), self.ref(stop)
            # pto's loop codegen does not tolerate stop < start (D-095), and a SIMT body hits it
            # every time: a strided thread walk gives the late threads a base past the end, and
            # the device answers with an aicore "timeout or trap" (board, simt_atomic_add).
            # pl.max is a Vec-side call, so the SIMT spelling of the clamp is the guard itself.
            sr, tr = self.env.frange(start), self.env.frange(stop)
            k = self.env.fold(step)
            guard = (sr is None or tr is None or tr[0] < sr[1]) and (k is None or k > 0)
            if guard:
                self.emit(f"if {lo} < {hi}:")
                self.indent += 1
            self.emit(f"for {var} in pl.range({lo}, {hi}, {self.ref(step)}):")
            self.indent += 1
            n0 = len(self.lines)
            for inner in op.regions[0].ops:
                self.op(inner)
            if len(self.lines) == n0:
                self.emit("pass")
            self.indent -= 1
            if guard:
                self.indent -= 1
            return
        if code == "cf.if":
            self.emit(f"if {self.ref(op.operands[0])}:")
            self.indent += 1
            n0 = len(self.lines)
            for inner in op.regions[0].ops:
                self.op(inner)
            if len(self.lines) == n0:
                self.emit("pass")
            self.indent -= 1
            if len(op.regions) > 1 and op.regions[1].ops:
                self.emit("else:")
                self.indent += 1
                for inner in op.regions[1].ops:
                    self.op(inner)
                self.indent -= 1
            return
        if code == "cf.return" and not op.operands:
            return
        raise PyptoGap(op, f"simt op {code} has no pl spelling")


class VfPrinter:
    """One ``@vf`` function -> one ``@pl.vector_function`` definition."""

    def __init__(self, mp: "ModulePrinter", fn: Function) -> None:
        self.mp = mp
        self.fn = fn
        self.lines: list[str] = []
        self.indent = 0
        self.names: dict[str, str] = {}
        self.masks_needed: dict[str, str] = {}
        self.aliases: dict[str, tuple[str, str]] = {}  # alias value -> (src name, DT) pending bit_cast
        self.slices: dict[str, tuple[str, str, str]] = {}  # slice value -> (base tile, row, col)
        self.slice_parent: dict[str, Any] = {}  # slice value -> the Value it slices
        self.loops: list[Any] = []  # the cf.for regions currently open (mask SSA, below)
        self.carrier_alias: dict[str, str] = {}  # tile -> its uint8 carrier view
        self.pre_extra: list[str] = []  # declarations that belong at the top of the vf body
        self.param_pos: dict[str, int] = {}  # tile parameter's py name -> its position
        self.carrier_params: list[tuple[int, str, list[int]]] = []  # (param pos, name, shape), D-126
        self.consts_needed: dict[tuple[float, str], str] = {}  # exact-constant registers (D-121)
        self.consts_param: list[str] = []  # exact-constant SCALAR parameters, in signature order
        self._mask_safe: dict[str, bool] = {}
        self.env = ScalarEnv(mp, fn, "vf")

    def emit(self, line: str) -> None:
        self.lines.append("    " * self.indent + line)

    def hoist(self, line: str) -> None:
        """Emit a declaration at the section top: a name assigned inside an if/for goes out
        of scope in pto's DSL codegen (its own tests are written branchless for this)."""
        self.hoisted.append(line)

    def name(self, v: Value) -> str:
        n = self.names.get(v.name)
        if n is None:
            a = self.aliases.pop(v.name, None)
            if a is not None:  # first use of a reinterpret view: materialise the bit_cast here
                src, dt = a
                n = self.mp.unique(py_ident(v.name))
                self.names[v.name] = n
                self.lines.append("    " * self.indent + f"{n} = vf.bit_cast({src}, dtype={dt})")
                return n
            raise PyptoGap(None, f"vf value {v.name} used before its defining op was printed")
        return n

    def define(self, v: Value, hint: str | None = None) -> str:
        n = self.mp.unique(py_ident(hint or v.name))
        self.names[v.name] = n
        return n

    def mask_write(self, op: Op, dst: Value) -> str:
        """The name a mask WRITE binds. Our IR models a mask as a register cell, so the same
        value is assigned again and again; printed literally that makes a LOOP-CARRIED mask,
        and pto's own vf codegen types a loop-carried variable as `float` - the C++ then fails
        with `cannot initialize a variable of type 'float' with an lvalue of type MaskReg`
        (board, fd_modified). A write inside a loop therefore takes a fresh name, which is
        exact exactly when no read of the mask reaches back across a loop boundary."""
        if not self.loops or dst.name not in self.names:
            return self.name(dst) if dst.name in self.names else self.define(dst)
        if not self._mask_self_contained(dst.name):
            raise PyptoGap(op, f"{op.opcode} rewrites mask {dst.name} inside a loop and a read of "
                               "it reaches back across the loop boundary: pto types a "
                               "loop-carried variable as float and a MaskReg does not convert "
                               "(board C++ error)")
        return self.define(dst)

    def _mask_self_contained(self, name: str) -> bool:
        """True when every read of ``name`` is preceded by a write in its own region, with each
        loop body scanned as if nothing came in - i.e. no iteration depends on the previous
        one's mask, so each write may take its own python name."""
        cache = self._mask_safe
        if name in cache:
            return cache[name]
        from ...ir.ops._dsl import REGISTRY

        def scan(ops: Any, written: bool) -> bool:
            for o in ops:
                try:
                    acc = [x.access for x in REGISTRY.get(o.opcode).operands]
                except Exception:
                    acc = []
                hit = [(i, x) for i, x in enumerate(o.operands)
                       if isinstance(x, Value) and x.name == name]
                if not written and (any(i >= len(acc) or acc[i] != "write" for i, _ in hit)
                                    or any(isinstance(v, Value) and v.name == name
                                           for v in o.attrs.values())):
                    return False
                for r in o.regions:
                    if not scan(r.ops, False if o.opcode == "cf.for" else written):
                        return False
                if any(i < len(acc) and acc[i] == "write" for i, _ in hit):
                    written = True
            return True

        cache[name] = scan(self.fn.body.ops, False)
        return cache[name]

    def mask_spill(self, op: Op, code: str) -> None:
        """`vf.mask_to_ub` / `vf.ub_to_mask` - the predicate spill/fill pair (`psts` / `plds`).

        pto carries both, not as their own APIs but as OVERLOADS of the ordinary aligned
        move: `vf.store_align` whose SOURCE is a MaskReg lowers to `psts`, and
        `vf.load_align` whose DESTINATION is a MaskReg lowers to `plds`
        (`backend/backend_cce_vf_ops.cpp` EmitVFStoreAlign / EmitVFLoadAlign, both routed on
        `IsMaskRegVar`; its `vf_ops.cpp` registration records that the separate
        `vf.mask_load`/`vf.mask_store` ops were REMOVED in favour of exactly this dispatch).
        The store side takes no predicate argument, and the default dist is NORM on both -
        the VL/8 = 32-byte physical predicate image cce prints and the simulator models - so
        neither call names one.

        Two properties of that dispatch decide the spelling:

        * The offset is POINTER ARITHMETIC ON THE TILE, not an argument. `psts` is emitted
          with a literal `0` offset unless `post_update` is set, so an offset handed over as
          an argument would be dropped SILENTLY and every slot would land on the first one.
          Displacing the pointer instead prints the same `(__ubuf__ uint32_t*)(base + off)`
          cce does - pto casts the tile pointer to `uint32_t*` itself, so no carrier view is
          needed for a tile whose own dtype is something else.
        * `load_align` is one of pto's UNIFIED ops: when its destination name is FRESH, the
          register kind is inferred from the SOURCES, and a UB tile says RegTensor. A fresh
          name would therefore lower to `vlds` and load data where predicates belong, with no
          diagnostic. The name has to be a MaskReg pto already knows, so a write that
          `mask_write` freshens (the loop case, which exists because pto types a loop-carried
          variable as `float`) declares that fresh name first.
        """
        dst, src = op.operands[:2]
        mem = dst if code == "vf.mask_to_ub" else src
        base, off = self.blk_addr(op, mem)
        addr = base if off == "0" else f"{base} + ({off})"
        if code == "vf.mask_to_ub":
            self.emit(f"vf.store_align({addr}, {self.name(src)})")
            return
        previous = self.names.get(dst.name)
        target = self.mask_write(op, dst)
        if target != previous:
            w = getattr(dst.type, "width", None) or 32
            dt = MASK_W_DT.get(f"b{w}", "pl.DT_FP32")  # width is the INT bit count
            self.emit(f"{target} = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype={dt})")
        self.emit(f"{target} = vf.load_align({addr})")

    def block_carrier(self, op: Op, v: Value, base: str) -> str:
        """The tile a strided block access must name. `vsstb`/`vsldb` have no `__ubuf__ hifloat8*`
        overload - cce prints the same access through the UINT8 carrier, and pto only half-applies
        it: it casts the REGISTER to uint8 but leaves the tile pointer hif8, so the call matches
        no candidate ("no known conversion from '__ubuf__ hifloat8_*'"). A `pl.reinterpret` view
        of the tile fixes exactly that - board-proven that the register and the mask may stay
        hif8, only the tile has to change (the whole v8 pack path compiles and runs)."""
        t = getattr(v, "type", None)
        dt = getattr(getattr(t, "dtype", None), "name", None)
        if dt != "hif8" or not base.isidentifier():
            return base
        got = self.carrier_alias.get(base)
        if got is None:
            dims = [self.env.fold(d) for d in getattr(t, "dims", ())]
            if not dims or any(d is None for d in dims):
                raise PyptoGap(op, f"{op.opcode} through a hif8 tile whose shape does not fold: "
                                   "the uint8 carrier view needs pl.reinterpret's explicit shape")
            got = self.mp.unique(base + "_u8")
            shape = [int(d) for d in dims]
            if base in self.param_pos:
                # D-126: pl.reinterpret of a tile that reached the callee as a RUNTIME-INDEXED
                # tile-group element folds to the group's FIRST slot - pypto binds the view to a
                # compile-time address, so every rotation writes slot 0 and the reader of slot 1
                # gets memory this launch never wrote (board: v8_allhif8's P staging read 0xFF
                # poison on every odd beat). Take the carrier as a parameter and let the caller
                # index a parallel uint8 group, which carries the slot the way pl means it to.
                # THIS IS A WORKAROUND FOR A PYPTO BEHAVIOUR, not for anything in our IR - re-check
                # it whenever the box's pypto moves. The check: emit any kernel whose vf reinterprets
                # a slot buffer (v8_allhif8) and read the JIT's own C++,
                # `build/<kernel>__a5/tk_none/kernel.cpp`. Still broken while
                # `_tg_<name>_grp_tiles_0[` appears ONLY as its declaration and a
                # `__inline_<n>_<name>_u8_0` carries a constant `TASSIGN`; fixed once the inlined
                # body subscripts the group. When it is fixed, this branch and `u8_carrier` /
                # `vf_carriers` / `u8_groups` can go and the view moves back into `pre_extra`.
                self.carrier_params.append((self.param_pos[base], got, shape))
            else:
                self.pre_extra.append(f"{got} = pl.reinterpret({base}, dtype=pl.DT_UINT8, "
                                      f"shape={shape})")
            self.carrier_alias[base] = got
        return got

    def all_mask(self, dtype_name: str) -> str:
        m = self.masks_needed.get(dtype_name)
        if m is None:
            m = f"_pall_{py_ident(dtype_name)}"
            self.masks_needed[dtype_name] = m
        return m

    def const_reg(self, v: float, dtype: str) -> str | None:
        """A register holding `v` exactly, hoisted to the top of the vf body (D-121)."""
        if _imm_bits(v, dtype) is None:
            return None
        n = self.consts_needed.get((v, dtype))
        if n is None:
            n = f"_kconst{len(self.consts_needed)}_{py_ident(dtype)}"
            self.consts_needed[(v, dtype)] = n
        return n

    def imm_reg(self, imm: Any, dtype: str, reg_form: str | None) -> str | None:
        """The exact-constant register a lossy float immediate needs, or None to print it inline."""
        v = imm.value if isinstance(imm, Literal) else imm
        if reg_form is None or not isinstance(v, float):
            return None
        if _imm_survives(v, dtype):
            return None
        return self.const_reg(v, dtype)

    def const_param(self, imm: Any, dtype: str) -> str | None:
        """The jit scalar parameter a lossy float immediate rides in on, or None when the
        literal survives pypto's six-decimal rendering and can simply be printed (D-134).

        Board-probed on this pl: `vf.muls(r, <literal>)` prints `vmuls(..., 0.088388f, ...)` and
        so does `vf.muls(r, pl.const(a) / pl.const(b))` - the parser CONSTANT-FOLDS the ratio
        before codegen, so no arrangement of compile-time constants can carry more than six
        decimals. `vf.muls(r, <parameter>)` prints `vmuls(..., alpha_0, ...)`: a name, exact.
        """
        v = imm.value if isinstance(imm, Literal) else imm
        if not isinstance(v, float) or _imm_survives(v, dtype):
            return None
        if _imm_bits(v, dtype) is None:  # a dtype we cannot spell as a scalar either
            return None
        key = (float(v), dtype)
        n = self.mp.const_scalars.get(key)
        if n is None:
            n = f"_ascr_k{len(self.mp.const_scalars)}"
            self.mp.const_scalars[key] = n
        if n not in self.consts_param:
            self.consts_param.append(n)
        return n

    def check_dtype64(self, op: Op, v: Value) -> None:
        t = v.type
        if isinstance(t, RegType) and t.dtype.name in ("i64", "u64"):
            raise PyptoGap(op, f"{op.opcode} on {t.dtype.name}: PyPTO Pro VF has no native "
                               "i64/u64 arithmetic register type. CCE uses compiler-provided "
                               "vector_2xvl_* overloads, which can expand into multiple hardware "
                               "instructions. PyPTO Pro exposes no equivalent carrier; lo/hi "
                               "arithmetic emulation is outside this backend's contract",
                           owner="upstream")

    def check_reg_groups(self, op: Op) -> None:
        # Check declarations and every use: casts and load/store-only bodies never reach
        # the arithmetic dtype guards, and predicates can arrive through attributes.
        for v in (*op.results, *op.operands, *op.attr_values()):
            t = getattr(v, "type", None)
            if isinstance(t, RegType) and t.n != 1:
                raise PyptoGap(op, f"{op.opcode} uses {t} (reg_num={t.n}): PyPTO Pro VF "
                                   "has no native register-group type or register-count parameter",
                               owner="upstream")
            if isinstance(t, MaskType) and t.n != 1:
                raise PyptoGap(op, f"{op.opcode} uses {t} (reg_num={t.n}): grouped predicates "
                                   "have no PyPTO Pro backend mapping; using only the dtype "
                                   "would discard the doubled lane extent", owner="unmapped")

    def mask_of(self, op: Op, src: Value) -> str:
        m = op.attrs.get("mask")
        if isinstance(m, Value):
            return self.name(m)
        t = src.type
        if isinstance(t, RegType):
            self.pl_dt(op, t.dtype.name)  # refuse an unspellable dtype HERE, where there is a loc
            return self.all_mask(t.dtype.name)
        raise PyptoGap(op, f"cannot infer a mask dtype from {src.name}")

    def dtype_of(self, v: Value) -> str:
        t = v.type
        if isinstance(t, (RegType, MemType)):
            return t.dtype.name
        raise PyptoGap(None, f"{v.name} has no dtype")

    def pl_dt(self, op: Op, dtype_name: str) -> str:
        return _pl_dt(dtype_name, op)

    def sref(self, v: Any) -> str:
        """A scalar operand inside a vf body (loop params, immediates)."""
        if isinstance(v, Literal) or isinstance(v, (int, float, bool)):
            return _lit(v)
        return self.env.ref(v)

    def tile_ref(self, op: Op, v: Value, off: Any = 0) -> tuple[str, str]:
        """A UB tile operand as (base spelling, offset spelling): slices fold into [row, col]."""
        s = self.slices.get(v.name)
        o = self.sref(off)
        if s is None:
            return self.name(v), o
        base, ro, co = s
        if o != "0":
            co = o if co == "0" else f"({co} + {o})"
        return base, f"[{ro}, {co}]"

    def blk_addr(self, op: Op, v: Value) -> tuple[str, str]:
        """A strided block access addresses ``tile + <linear element offset>``.

        The offset is POINTER ARITHMETIC ON THE TILE, not an argument - which is how the old
        easyasc bridge spelled vsstb/vsldb all along, and what the board confirms: a runtime
        offset with a real block stride lands exactly where it should (D-116). The three traps
        D-109 catalogued were all about passing the offset as an ARGUMENT; none of them applies
        here, and the post-update cursor is not needed at all."""
        off = op.attrs.get("offset", 0)
        s = self.slices.get(v.name)
        if s is None:
            return self.name(v), self.sref(off)
        base, ro, co = s
        parts = []
        if ro != "0":
            pv = self.slice_parent.get(v.name)
            dims = [self.env.fold(d) for d in getattr(getattr(pv, "type", None), "dims", ())]
            if not dims or dims[-1] is None:
                raise PyptoGap(op, f"{op.opcode} through a slice of {v.name} whose parent's row "
                                   "pitch does not fold: tile + offset needs one linear element "
                                   "offset, so the row origin has to be multiplied out")
            parts.append(f"({ro}) * {int(dims[-1])}")
        if co != "0":
            parts.append(f"({co})")
        o = self.sref(off)
        if o != "0":
            parts.append(o)
        return base, " + ".join(parts) if parts else "0"

    def render(self) -> str:
        self.mp._names.reset()
        params = []
        for p in self.fn.params:
            if isinstance(p.type, ScalarType):
                n = self.mp.unique(py_ident(p.name))
                self.env.names[p.name] = n
                params.append(n)
            else:
                n = self.mp.unique(py_ident(p.name))
                self.names[p.name] = n
                self.param_pos[n] = len(params)
                params.append(n)
        writers = set()  # dst-first vf ops writing through a bit_cast alias would rebind in pl
        alias_results = {o.results[0].name for o in self.fn.body.walk() if o.opcode == "vf.reinterpret"}
        for o in self.fn.body.walk():
            if o.opcode.startswith("vf.") and o.opcode not in ("vf.reg", "vf.mask", "vf.reinterpret") \
                    and o.operands and isinstance(o.operands[0], Value) \
                    and o.operands[0].name in alias_results \
                    and not o.opcode.startswith(("vf.store", "vf.scatter")):
                raise PyptoGap(o, f"{o.opcode} writes through reinterpret view "
                                  f"{o.operands[0].name}; pl's assignment form would rebind, not alias")
        for op in self.fn.body.ops:
            self.op(op)
        pre = [f"{m} = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype={_pl_dt(dt)})"
               for dt, m in self.masks_needed.items()]
        for (val, cdt), n in self.consts_needed.items():  # D-121: exact constants, loop-invariant
            bits, carrier = _imm_bits(val, cdt)  # type: ignore[misc]  - const_reg checked it
            pre.append(f"{n}_bits = vf.full({bits}, dtype={_pl_dt(carrier)})  # {val!r}")
            pre.append(f"{n} = vf.bit_cast({n}_bits, dtype={_pl_dt(cdt)})")
        body = (pre + self.pre_extra + self.lines) or ["pass"]
        if self.carrier_params:  # D-126: the caller supplies each parameter's uint8 carrier
            params += [n for _, n, _ in self.carrier_params]
            self.mp.vf_carriers[self.fn.name] = [(pos, shape) for pos, _, shape in self.carrier_params]
        if self.consts_param:  # D-134: after the carriers, in the same order at every call site
            params += self.consts_param
            self.mp.vf_consts[self.fn.name] = list(self.consts_param)
        body, report = cleanup_scalars(body, enabled=self.mp.module.attrs.get("scalar_simplify", True))
        self.mp.scalar_cleanup_report[f"vf:{self.fn.name}"] = report
        kept = self.mp.vf_parameters[self.fn.name] = used_parameters(params, body)
        params = [name for i, name in enumerate(params) if i in kept]
        text = ["@pl.vector_function", f"def {fn_ident(self.fn.name)}({', '.join(params)}):"]
        text += [f"    {ln}" for ln in body]
        return "\n".join(text) + "\n"

    # ------------------------------------------------------------ ops

    def op(self, op: Op) -> None:  # noqa: C901 - one printer, one dispatch
        code = op.opcode
        self.check_reg_groups(op)
        if code == "vf.mod":
            raise PyptoGap(op, "vf.mod: PyPTO Pro VF has no integer floor-remainder API",
                           owner="upstream")
        if code == "vf.reg":
            self.define(op.results[0], op.attrs.get("name"))
            return
        if code == "vf.mask":
            n = self.define(op.results[0], op.attrs.get("name"))
            init = str(getattr(op.attrs.get("init", "all"), "name", op.attrs.get("init", "all")))
            pat = "ALLF" if init == "none" else init.upper()
            w = getattr(op.results[0].type, "width", None) or 32
            dt = MASK_W_DT.get(f"b{w}", "pl.DT_FP32")  # width is the INT bit count
            self.emit(f"{n} = vf.create_mask(pattern=pl.MaskPattern.{pat}, dtype={dt})")
            return
        if code.startswith("scalar.") or code.startswith("core.") or code in ("list.count", "list.item_dim"):
            for ln in self.env.scalar_op(op, self):
                self.emit(ln)
                if scalar_cleanup_eligible(op):
                    self.lines[-1] = ScalarLine(self.lines[-1], op)
            return
        if code == "cf.for":
            start, stop, step = (list(op.operands) + [1])[:3]
            var = self.env.define(op.results[0], op.attrs.get("name"))
            lo_txt, stop_txt = self.sref(start), self.sref(stop)
            sr, tr = self.env.frange(start), self.env.frange(stop)
            if sr is None or tr is None or tr[0] < sr[1]:  # same negative-trip clamp as the module printer
                stop_txt = f"pl.max({stop_txt}, {lo_txt})"
            self.emit(f"for {var} in pl.range({lo_txt}, {stop_txt}, {self.sref(step)}):")
            self.indent += 1
            n0 = len(self.lines)
            self.loops.append(op.regions[0])
            for inner in op.regions[0].ops:
                self.op(inner)
            self.loops.pop()
            if len(self.lines) == n0:
                self.emit("pass")
            self.indent -= 1
            return
        if code == "cf.if":
            self.emit(f"if {self.env.ref(op.operands[0])}:")
            self.indent += 1
            n0 = len(self.lines)
            for inner in op.regions[0].ops:
                self.op(inner)
            if len(self.lines) == n0:
                self.emit("pass")
            self.indent -= 1
            if len(op.regions) > 1 and op.regions[1].ops:
                self.emit("else:")
                self.indent += 1
                for inner in op.regions[1].ops:
                    self.op(inner)
                self.indent -= 1
            return
        if code == "cf.return":
            return
        if code == "mem.slice":
            src = op.operands[0]
            offs = op.attrs.get("offsets", [])
            if len(offs) != 2:
                raise PyptoGap(op, f"vf mem.slice with {len(offs)} offsets is outside the surface")
            self.slices[op.results[0].name] = (self.name(src), self.sref(offs[0]), self.sref(offs[1]))
            self.slice_parent[op.results[0].name] = src  # for the linear form of a block access
            return
        if code == "vf.reinterpret":
            src = op.operands[0]
            dt = self.pl_dt(op, self.dtype_of(op.results[0]))
            self.aliases[op.results[0].name] = (self.name(src), dt)
            return
        if code == "vf.load_cont":
            dst, src = op.operands
            mode = str(op.attrs.get("mode", "norm"))
            base, off_s = self.tile_ref(op, src, op.attrs.get("offset", 0))
            dist = ""
            if mode not in ("norm", "norm_b8", "norm_b16", "norm_b32", "norm_b64"):
                pl_dist = LOAD_DIST.get(mode)
                if pl_dist is None:
                    raise PyptoGap(op, f"vf.load_cont mode {mode!r} has no LoadDist mapping")
                dist = f", dist=pl.LoadDist.{pl_dist}"
            self.emit(f"{self.name(dst)} = vf.load_align({base}, {off_s}{dist})")
            return
        if code == "vf.store_cont":
            dst, src = op.operands
            mode = str(op.attrs.get("mode", "norm"))
            base, off_s = self.tile_ref(op, dst, op.attrs.get("offset", 0))
            m = self.mask_of(op, src)
            dist = ""
            src_spell = self.name(src)
            if mode not in ("norm", "norm_b8", "norm_b16", "norm_b32", "norm_b64"):
                pl_dist = STORE_DIST.get(mode)
                if pl_dist is None:
                    raise PyptoGap(op, f"vf.store_cont mode {mode!r} has no StoreDist mapping")
                if mode in ("pack_b32", "pack_b64"):
                    # our pack_bN keeps the element width and drops the odd lanes = c310 PK_BN, but
                    # pto's PACK derives the PK granularity from the REG width (b16 reg -> PK_B16 -
                    # backend_cce_vf_ops.cpp:812) - store through a double-width bit_cast so pto
                    # picks PK_BN; the byte behaviour of vsts only follows the dist code
                    if mode == "pack_b64":
                        raise PyptoGap(op, "vf.store_cont pack_b64: the 64-bit register spelling pto's "
                                           "PACK would need is outside its VF surface")
                    wide = self.mp.unique(py_ident(src.name) + "_pk32")
                    self.emit(f"{wide} = vf.bit_cast({src_spell}, dtype=pl.DT_UINT32)")
                    src_spell = wide
                    # The predicate is a physical bit field, unchanged by the
                    # source register's bit_cast. PK_B32 samples the same bits
                    # in both backends; recreating a typed mask would alter it.
                    if not isinstance(op.attrs.get("mask"), Value):
                        m = self.all_mask("u32")
                dist = f", dist=pl.StoreDist.{pl_dist}"
            tail = f", {off_s}" if off_s != "0" else ""
            self.emit(f"vf.store_align({base}, {src_spell}, {m}{tail}{dist})")
            return
        if code == "vf.load_interleave":
            # pto's DUAL-load: `load_align` with a DINTLV distribution returns the (even, odd)
            # pair rather than one register, so the arity follows the dist. Same op, same call,
            # one more destination -- which is exactly the shape a name-based coverage survey
            # misses (M10-075, the `mask_to_ub` overload).
            d0, d1, src = op.operands[:3]
            mode = str(op.attrs.get("mode", "dintlv_b16"))
            pl_dist = LOAD_INTLV_DIST.get(mode)
            if pl_dist is None:
                raise PyptoGap(op, f"vf.load_interleave mode {mode!r} has no dual-load LoadDist mapping")
            base, off_s = self.tile_ref(op, src, op.attrs.get("offset", 0))
            self.emit(f"{self.name(d0)}, {self.name(d1)} = vf.load_align({base}, {off_s}, "
                      f"dist=pl.LoadDist.{pl_dist})")
            return
        if code == "vf.store_interleave":
            # `store_align(tile, src_even, src_odd, preg, offset)` -- the single-source form with
            # `src` expanded to the pair; `preg` and `offset` keep their positions.
            dst, s0, s1 = op.operands[:3]
            mode = str(op.attrs.get("mode", "intlv_b16"))
            pl_dist = STORE_INTLV_DIST.get(mode)
            if pl_dist is None:
                raise PyptoGap(op, f"vf.store_interleave mode {mode!r} has no dual-store StoreDist mapping")
            if pl_dist == "INTLV":
                # Lossy leg: our `intlv_b8` and `intlv_b16` both print element-grain `INTLV`, whose
                # width pto reads off the register. The two agree only while the register's element
                # size IS the tag's width; otherwise pto would interleave at another grain and the
                # cce build would not, silently.
                want = int(mode.rsplit("_b", 1)[-1])
                have = _BITS.get(self.dtype_of(s0))
                if have is not None and have != want:
                    raise PyptoGap(op, f"vf.store_interleave mode {mode!r} on a {have}-bit register: "
                                       f"pto's INTLV takes the interleave grain from the register dtype, "
                                       f"so the tag and the register disagree")
            base, off_s = self.tile_ref(op, dst, op.attrs.get("offset", 0))
            m = self.mask_of(op, s0)
            tail = f", {off_s}" if off_s != "0" else ""
            self.emit(f"vf.store_align({base}, {self.name(s0)}, {self.name(s1)}, {m}{tail}, "
                      f"dist=pl.StoreDist.{pl_dist})")
            return
        if code == "vf.store":  # strided block store (vsstb)
            dst, src = op.operands
            base, off = self.blk_addr(op, dst)
            base = self.block_carrier(op, dst, base)
            blk = op.attrs.get("blk_stride", 1)  # absent is 1, not Pro's 0 (I036)
            m = self.mask_of(op, src)
            bs = self.sref(blk)
            addr = base if off == "0" else f"{base} + {off}"
            self.emit(f"vf.store_align({addr}, {self.name(src)}, {m}, "
                      f"data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, block_stride={bs})")
            return
        if code == "vf.load":  # strided block load (vsldb): the same tile + offset address
            if lane_load(self, op):  # a predicate that may split a block (I036)
                return
            dst, src = op.operands
            base, off = self.blk_addr(op, src)
            base = self.block_carrier(op, src, base)
            blk = op.attrs.get("blk_stride", 1)
            m = self.mask_of(op, dst)
            bs = self.sref(blk)
            addr = base if off == "0" else f"{base} + {off}"
            self.emit(f"{self.name(dst)} = vf.load_align({addr}, {m}, "
                      f"data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY, block_stride={bs})")
            return
        if code == "vf.ub_cursor":
            # cce declares a mutable __ubuf__ T* cursor; pto keeps its own post-update cursor
            # per pointer expression - but ONLY for a bare tile: a subscripted position
            # materialises an _expr_tmp Var in the parser and the native codegen rejects it
            # Board diagnostic: "GetOrCreateVFTilePtr expects a tile expr, got Var".
            r = op.results[0]
            base, pos = self.tile_ref(op, op.operands[0])
            if pos.startswith("[") and pos not in ("[0, 0]",):
                raise PyptoGap(op, "vf.ub_cursor at a non-zero (runtime) offset: pto's unalign "
                                   "pointer must be a bare tile - a subscripted position "
                                   "materialises a Var its codegen rejects", owner="upstream")
            self.names[r.name] = base
            return
        if code == "vf.unalign":
            r = op.results[0]
            # classify by consumer: the load protocol reads it at load_unalign_pre/load_unalign,
            # the store protocol at store_unalign/store_unalign_post
            used_by_load = any(o.opcode in ("vf.load_unalign_pre", "vf.load_unalign")
                               and any(x.name == r.name for x in o.operands)
                               for o in self.fn.body.walk())
            init = "vf.load_unalign_init()" if used_by_load else "vf.unalign_reg_for_store()"
            self.emit(f"{self.define(r)} = {init}")
            return
        if code == "vf.load_unalign_pre":
            ureg, src = op.operands[:2]
            if self.env.fold(op.attrs.get("offset", 0)) != 0:
                raise PyptoGap(op, "load_unalign_pre with a non-zero offset")
            self.emit(f"vf.load_unalign_pre({self.name(ureg)}, {self.name(src)})")
            return
        if code == "vf.load_unalign":
            dst, src, ureg = op.operands[:3]
            if self.env.fold(op.attrs.get("offset", 0)) != 0:
                raise PyptoGap(op, "load_unalign with a non-zero offset")
            stride = op.attrs.get("stride")
            post = str(op.attrs.get("post_mode", "normal"))
            if stride is not None or post == "update":
                st = self.sref(stride if stride is not None else 0)
                self.emit(f"{self.name(dst)} = vf.load_unalign({self.name(ureg)}, {self.name(src)}, {st})")
            else:
                self.emit(f"{self.name(dst)} = vf.load_unalign({self.name(ureg)}, {self.name(src)})")
            return
        if code == "vf.store_unalign":
            dst, src, ureg = op.operands[:3]
            if self.env.fold(op.attrs.get("offset", 0)) != 0:
                raise PyptoGap(op, "store_unalign with a non-zero offset")
            count = op.attrs.get("count")
            if count is None:
                raise PyptoGap(op, "store_unalign without count")
            # the vstus slot cce fills with count IS pl's `stride` argument (direct pass-through
            # in backend_cce_vf_ops.cpp); cce always advances the cursor -> post_update=True
            self.emit(f"vf.store_unalign({self.name(dst)}, {self.name(src)}, {self.name(ureg)}, "
                      f"{self.sref(count)}, post_update=True)")
            return
        if code == "vf.store_unalign_post":
            dst, ureg = op.operands[:2]
            if self.env.fold(op.attrs.get("offset", 0)) != 0:
                raise PyptoGap(op, "store_unalign_post with a non-zero offset")
            stride = op.attrs.get("stride", 0)
            # The two-argument post prints as vstar, which follows the AR count instead of this cursor;
            # A5 has no flush that leaves the cursor, so every post is post-updating (I042).
            self.emit(f"vf.store_unalign_post({self.name(dst)}, {self.name(ureg)}, "
                      f"{self.sref(stride)}, post_update=True)")
            return
        if code == "vf.pack":
            dst, src = op.operands
            part = {"lowest": "LOWER", "highest": "UPPER"}.get(str(op.attrs.get("part", "lowest")))
            if part is None:
                raise PyptoGap(op, f"vf.pack part {op.attrs.get('part')!r} has no PackPart")
            dt = self.pl_dt(op, self.dtype_of(dst))
            self.emit(f"{self.name(dst)} = vf.pack({self.name(src)}, dtype={dt}, part=pl.PackPart.{part})")
            return
        if code == "vf.unsqueeze":
            dst = op.operands[0]
            m = op.attrs.get("mask")
            if not isinstance(m, Value):
                raise PyptoGap(op, "vf.unsqueeze without a mask register")
            dt = self.pl_dt(op, self.dtype_of(dst))
            self.emit(f"{self.name(dst)} = vf.unsqueeze({self.name(m)}, dtype={dt})")
            return
        if code == "vf.dup":
            dst, imm = op.operands[0], op.operands[1]
            self.check_dtype64(op, dst)
            if isinstance(imm, Value) and isinstance(imm.type, RegType):
                m = self.mask_of(op, imm)
                self.emit(f"{self.name(dst)} = vf.full({self.name(imm)}, {m})")
                return
            dtn = self.dtype_of(dst)
            dt = self.pl_dt(op, dtn)
            # `vf.full`'s scalar mode takes the predicate too, and dropping it printed a masked
            # broadcast as a fill of every lane - a wrong answer on the board with no gap and no
            # error, found by `attention.a5_presence_mask` (its masked `dup` builds a 0/1 output
            # that is compared exactly, so the missing predicate showed as 1.0 everywhere while
            # the `expsub` one line above, which did carry its mask, stayed correct).
            mask = op.attrs.get("mask")
            pred = f"{self.name(mask)}, " if isinstance(mask, Value) else ""
            v = imm.value if isinstance(imm, Literal) else imm
            bits = _imm_bits(v, dtn) if isinstance(v, float) and not _imm_survives(v, dtn) else None
            if bits is not None:  # D-121: spell the constant through its integer bit pattern
                n = self.name(dst)
                # the predicate belongs on the fill: zeroed lanes are zero BITS, which bit_cast
                # carries through as the zero of the destination dtype.
                self.emit(f"{n}_bits = vf.full({bits[0]}, {pred}dtype={_pl_dt(bits[1])})  # {v!r}")
                self.emit(f"{n} = vf.bit_cast({n}_bits, dtype={dt})")
                return
            self.emit(f"{self.name(dst)} = vf.full({self.sref(imm)}, {pred}dtype={dt})")
            return
        if code in VF_BINARY:
            dst, a, b = op.operands
            self.check_dtype64(op, a)
            m = self.mask_of(op, a)
            self.emit(f"{self.name(dst)} = vf.{VF_BINARY[code]}({self.name(a)}, {self.name(b)}, {m})")
            return
        if code in VF_UNARY:
            dst, a = op.operands
            self.check_dtype64(op, a)
            m = self.mask_of(op, a)
            self.emit(f"{self.name(dst)} = vf.{VF_UNARY[code]}({self.name(a)}, {m})")
            return
        if code in VF_GROUP_REDUCE:
            dst, a = op.operands
            self.check_dtype64(op, a)
            m = self.mask_of(op, a)
            self.emit(f"{self.name(dst)} = vf.{VF_GROUP_REDUCE[code]}({self.name(a)}, {m}, datablock=True)")
            return
        if code in VF_SCALAR:
            dst, a, imm = op.operands
            self.check_dtype64(op, a)
            m = self.mask_of(op, a)
            dt = self.dtype_of(a)
            # D-134: a lossy immediate rides in as a scalar PARAMETER and the op keeps its
            # scalar form; only when that is not available does it fall back to D-121's
            # bit-pattern register and the vector-vector form.
            sc = self.const_param(imm, dt)
            if sc is not None:
                self.emit(f"{self.name(dst)} = vf.{VF_SCALAR[code]}({self.name(a)}, {sc}, {m})")
                return
            reg = self.imm_reg(imm, dt, VF_SCALAR_REG.get(code))
            if reg is not None:  # D-121: the immediate would not survive pypto's CCE rendering
                self.emit(f"{self.name(dst)} = vf.{VF_SCALAR_REG[code]}({self.name(a)}, {reg}, {m})")
                return
            self.emit(f"{self.name(dst)} = vf.{VF_SCALAR[code]}({self.name(a)}, {self.sref(imm)}, {m})")
            return
        if code == "vf.mulscast":
            # `muls_cast(src, scalar, preg, dtype, layout=)`: one op for `dst = cast(src * s)`.
            # Sibling of `vf.expsub` in the frontend (rules_reg.py), and like it a fused form
            # whose whole value is being ONE instruction -- printing mul + cast would lose that.
            dst, a, imm = op.operands
            self.check_dtype64(op, a)
            m = self.mask_of(op, a)
            kw = [f"dtype={self.pl_dt(op, self.dtype_of(dst))}"]
            lay = op.attrs.get("layout")
            lay_s = str(getattr(lay, "name", lay)) if lay is not None else None
            if lay_s is not None and lay_s != "zero":
                c = CAST_LAYOUT.get(lay_s)
                if c is None:
                    raise PyptoGap(op, f"vf.mulscast layout {lay_s!r} has no CastLayout spelling")
                kw.append(f"layout={c}")
            # `scalar_dtype` types the immediate for cce's intrinsic; pl takes a Python scalar
            # and reads the width off `src`, so it has nothing to carry here.
            sc = self.const_param(imm, self.dtype_of(a))
            scalar = sc if sc is not None else self.sref(imm)
            self.emit(f"{self.name(dst)} = vf.muls_cast({self.name(a)}, {scalar}, {m}, "
                      f"{', '.join(kw)})")
            return
        if code in ("vf.cmp", "vf.cmps"):
            dst, a, b = op.operands
            self.check_dtype64(op, a)
            mode = str(op.attrs.get("mode", ""))
            if mode not in VF_CMP:
                raise PyptoGap(op, f"{code} mode {mode!r} has no pypto compare")
            if code == "vf.cmp":
                rhs = self.name(b)
            else:
                reg = self.imm_reg(b, self.dtype_of(a), VF_CMP[mode])
                rhs = reg if reg is not None else self.sref(b)
            m = self.mask_of(op, a)
            self.emit(f"{self.name(dst)} = vf.{VF_CMP[mode]}({self.name(a)}, {rhs}, {m})")
            return
        if code == "vf.select":
            dst, a, b = op.operands
            m = op.attrs.get("mask")
            if not isinstance(m, Value):
                raise PyptoGap(op, "vf.select without a mask value")
            dt = self.dtype_of(dst)
            if dt in _FP8_CARRIED:
                # pto's select is typed ("vf.select only supports BOOL / INT8 / UINT8 / INT16 /
                # UINT16 / FP16 / BF16 / INT32 / UINT32 / FP32, got DT_HF8" - board) even though
                # the instruction is a per-lane bit mux that never reads the value. Same UINT8
                # CARRIER as the b8 de-interleave above: the lane count and the mask are the ones
                # an 8-bit register already has, so casting in and back is bit-exact.
                ca = self.mp.unique(py_ident(a.name) + "_u8")
                cb = self.mp.unique(py_ident(b.name) + "_u8")
                self.emit(f"{ca} = vf.bit_cast({self.name(a)}, dtype=pl.DT_UINT8)")
                self.emit(f"{cb} = vf.bit_cast({self.name(b)}, dtype=pl.DT_UINT8)")
                td = self.mp.unique(py_ident(dst.name) + "_u8")
                self.emit(f"{td} = vf.select({ca}, {cb}, {self.name(m)})")
                self.emit(f"{self.define(dst)} = vf.bit_cast({td}, dtype={self.pl_dt(op, dt)})")
                return
            self.emit(f"{self.name(dst)} = vf.select({self.name(a)}, {self.name(b)}, {self.name(m)})")
            return
        if code == "vf.cast":
            dst, src = op.operands
            dts = (self.dtype_of(dst), self.dtype_of(src))
            if "hif8" in dts and "f16" in dts \
                    and str(getattr(op.attrs.get("round"), "name", op.attrs.get("round"))) in ("hybrid", "odd"):
                raise PyptoGap(op, "vf.cast f16<->hif8 with hybrid/odd rounding: pto refuses it "
                                   "upstream ('CAST_ODD/CAST_HYBRID is not supported for "
                                   "src=DT_FP16 dst=DT_HF8' - board parse error; the f32 path "
                                   "supports hybrid and is bit-exact)", owner="upstream")
            if dts[1] == "i4":
                # `i4` is the compiler's `vector_s4x2`, a PACKED PAIR, and its widening casts have
                # their own intrinsic names -- AscendC's own dav_3510 Cast calls `vcvt_s42f16` /
                # `vcvt_s42bf16` / `vcvt_s42s16` rather than the generic `vcvt`. pto prints the
                # generic one with an ordinary 8-bit source register, so the header's `vector_s4x2`
                # overload is never selected: "no matching function for call to 'vcvt'", the
                # candidate at __clang_cce_vector_intrinsics.h:4945 rejected because the argument is
                # a `RegTensor<uint8_t>`, not a `vector_s4x2` (board, cast_i4_widen, 2026-09-05).
                # The NARROWING direction works and is bit-exact, so this is the source side only.
                raise PyptoGap(op, f"vf.cast {dts[1]}->{dts[0]}: pto has no `vector_s4x2` SOURCE "
                                   "register -- it prints the generic `vcvt` with an 8-bit register "
                                   "where the packed-pair widening needs `vcvt_s42f16` / "
                                   "`vcvt_s42bf16` / `vcvt_s42s16` (board: \"no matching function "
                                   "for call to 'vcvt'\", the vector_s4x2 candidate rejected). The "
                                   "narrowing direction to i4 prints and is bit-exact",
                               owner="upstream")
            if 64 in (_BITS.get(dts[0], 0), _BITS.get(dts[1], 0)):
                # c310's 64-bit casts are the argument-free TWO-REGISTER form: `b64_widen` is
                # `vcvt(dst, src)` with no mask, no PART and no MODE, and destination lane k
                # carries source lane k. pto prints the ordinary `vf.astype` with a mask, a round
                # mode and a saturation flag, and gets PART semantics: measured on the board
                # (cast_b64_widen, 2026-09-05, cce bit-exact on the same case) i32 -> i64 read
                # every OTHER source lane -- [0,1,2,3] came back [0,2,4,6] -- and i64 -> i32 wrote
                # its results on alternate destination lanes with zeros between them. It BUILDS and
                # RUNS, so this refusal is the only thing between a caller and a silent wrong
                # answer. (The same mistake was in our own interpreter until the board caught it,
                # D-219.)
                raise PyptoGap(op, f"vf.cast {dts[1]}->{dts[0]}: c310's 64-bit vcvt is the "
                                   "argument-free two-register form (lane k <- lane k), and pto "
                                   "prints the masked PART form -- board-measured, it reads every "
                                   "other source lane widening and writes alternate destination "
                                   "lanes narrowing, while cce is bit-exact on the same case. It "
                                   "compiles, so refusing is what keeps a wrong answer from being "
                                   "returned", owner="upstream")
            if dts == ("i64", "f32"):
                # cce's shape for this pair is `b64_from_f32`: vcvt(dst, src, ROUND, RS) -- the
                # two-register form takes no mask, no PART and no MODE. pto prints the ordinary
                # masked form, so the arguments land one position early.
                raise PyptoGap(op, "vf.cast f32->i64: pto prints the masked "
                                   "vcvt(dst, src, mask, ROUND, PART, MODE) form, but c310's "
                                   "f322s64 is the two-register form and takes (ROUND, RS) -- "
                                   "board static_assert: 'The 5th argument of this vcvt (f322s64) "
                                   "can only be: RS_DISABLE, RS_ENABLE', with PART_EVEN in that "
                                   "position. Same shape as the bf16->f16 bug below",
                               owner="upstream")
            if dts == ("f16", "bf16"):
                raise PyptoGap(op, "vf.cast bf16->f16: pto's bf162f16 printer emits "
                                   "vcvt(dst, src, mask, ROUND_R, MODE_ZEROING) but bisheng's "
                                   "bf162f16 form wants (mask, RS_*, ROUND_*, MODE_*) - one "
                                   "argument short, static_assert refuses (upstream printer bug; "
                                   "f16->bf16 compiles fine)", owner="upstream")
            m = self.mask_of(op, src)
            dt = self.pl_dt(op, self.dtype_of(dst))
            kw = [f"dtype={dt}"]
            rnd = op.attrs.get("round")
            if rnd is not None and str(getattr(rnd, "name", rnd)) in ("none", "None"):
                rnd = None
            if rnd is not None:
                r = ROUND.get(str(getattr(rnd, "name", rnd)))
                if r is None:
                    raise PyptoGap(op, f"vf.cast round mode {rnd!r} has no VFRoundMode spelling")
                kw.append(f"round_mode={r}")
            lay = op.attrs.get("layout")
            lay_s = str(getattr(lay, "name", lay)) if lay is not None else None
            if lay_s is not None and lay_s != "zero":
                c = CAST_LAYOUT.get(lay_s)
                if c is None:
                    raise PyptoGap(op, f"vf.cast layout {lay_s!r} has no CastLayout spelling")
                kw.append(f"layout={c}")
            # The saturation flag is STATED, never left to the default: pl documents
            # `SaturateMode.OFF` as its default but prints `RS_ENABLE` when the kwarg is
            # omitted, while cce prints `RS_ENABLE` only when the op asks for it - and the two
            # differ exactly where a narrowing cast overflows. No a5 corpus op asks to
            # saturate, so every cast the printer can state prints `RS_DISABLE` like cce's.
            # The exception is a float8 / float4 DESTINATION, where pto refuses to be told:
            # "vf.astype: FP32->FP8 conversion requires saturate=ON (RS_ENABLE), OFF is not
            # supported for this path" (board). There the kwarg is left off and pto's forced
            # saturation stands - a divergence from cce confined to values outside the target
            # type's range, which cce sends to infinity and pto clamps (D-125).
            sat = op.attrs.get("saturate")
            sat = sat.value if isinstance(sat, Literal) else sat
            if sat:
                kw.append("saturate=pl.SaturateMode.ON")
            elif self.dtype_of(dst) not in _FORCED_SAT_DST:
                kw.append("saturate=pl.SaturateMode.OFF")
            self.emit(f"{self.name(dst)} = vf.astype({self.name(src)}, {m}, {', '.join(kw)})")
            return
        if code == "vf.barrier":
            # D-138: pl carries the SAME twelve (src -> dst) memory-barrier modes the c310
            # intrinsic does, under the same names. Sending every barrier to VV_ALL - a full
            # vector-to-vector drain - where cce sends a targeted `VST_VLD` costs real time
            # inside a vf that barriers in a loop, and is our translation, not an upstream
            # limit. Same table as the cce backend's `c310.MEM_BAR`.
            key = (str(getattr(op.attrs.get("src"), "name", op.attrs.get("src"))),
                   str(getattr(op.attrs.get("dst"), "name", op.attrs.get("dst"))))
            tag = MEM_BAR_MODE.get(key)
            if tag is None:
                raise PyptoGap(op, f"vf.barrier {key[0]!r} -> {key[1]!r}: pl's MemBarMode has "
                                   "no pair for it (it carries the twelve AscendC MemType "
                                   "combinations; this is not one of them)", owner="upstream")
            self.emit(f"vf.mem_bar(mode=pl.MemBarMode.{tag})")
            return
        if code == "vf.arange":
            dst = op.operands[0]
            self.check_dtype64(op, dst)
            # The start value and the direction ride ATTRS named "v" and "mode" (the IR op
            # declares exactly those, and both the cce printer's vci and the simulator read
            # them). The old lookup asked for a second operand and then for "start", found
            # neither, and fell through to the literal 0 - so EVERY arange began at 0 and
            # every decreasing ramp increased. Board: flash_attn_full_fp8_causal's causal
            # mask built its second 64-column chunk as 0..63 instead of 64..127, leaking one
            # masked key per row into the softmax (row 0 rowsum 1.465 against a golden 1.0).
            start = op.operands[1] if len(op.operands) > 1 else op.attrs.get("v", 0)
            mode = str(getattr(op.attrs.get("mode"), "name", op.attrs.get("mode", "increase")))
            if mode not in ("increase", "decrease"):
                raise PyptoGap(op, f"vf.arange mode={mode!r} has no pl.IndexOrder spelling")
            order = "" if mode == "increase" else ", index_order=pl.IndexOrder.DECREASE_ORDER"
            dt = self.pl_dt(op, self.dtype_of(dst))
            self.emit(f"{self.name(dst)} = vf.arange({self.sref(start)}, dtype={dt}{order})")
            return
        if code == "vf.mask_from_spr":
            dst = op.operands[0]
            self.emit(f"{self.mask_write(op, dst)} = vf.get_mask_spr()")
            return
        if code == "vf.mask_update":
            dst = op.operands[0]
            cnt = op.operands[1] if len(op.operands) > 1 else op.attrs.get("cnt")
            post = op.attrs.get("post_update") or op.attrs.get("post")
            if post:
                raise PyptoGap(op, "vf.mask_update POST_UPDATE decrements its counter in place; "
                                   "pypto update_mask reads a scalar only")
            if isinstance(dst.type, RegType):
                dt = self.pl_dt(op, self.dtype_of(dst))
            else:
                # D-129: the LANE WIDTH is the whole meaning of a count predicate, and pl
                # defaults to 32 when it is not told - `plt_b32` where cce prints `plt_b16`.
                # The mask register's bit layout then addresses every SECOND 16-bit lane, so a
                # count of 64 leaves the odd lanes false and whatever the mask guards silently
                # skips half its work (v9_allhif8: the tail key group never reached the odd
                # queries). The width lives on the mask type, exactly as `vf.mask` reads it.
                w = getattr(dst.type, "width", None)
                dt = MASK_W_DT.get(f"b{w}") if w else None
                if dt is None:
                    raise PyptoGap(op, f"vf.mask_update on a mask of width {w!r}: pl picks the "
                                       "predicate's lane width from dtype= and defaults to 32 "
                                       "when it is unstated, which is a different mask",
                                   owner="ours")
            self.emit(f"{self.mask_write(op, dst)} = vf.update_mask({self.sref(cnt)}, dtype={dt})")
            decrement(self, op)  # pl's update_mask copies its count (I040)
            return
        if code in ("vf.mask_to_ub", "vf.ub_to_mask"):
            self.mask_spill(op, code)
            return
        if code == "vf.deinterleave":
            d0, d1, s0, s1 = op.operands
            if _esize_bits(self.dtype_of(s0)) <= 8:
                # cce prints the b8 form as vdintlv((vector_u8&)...) - the intrinsic exists, the
                # register just has to arrive through its UINT8 CARRIER. pl's de_interleave picks
                # its width from the register dtype and has no hif8/fp8 instantiation, so the
                # same carrier cast is the spelling here: bit_cast in, de-interleave, bit_cast
                # the two results back.
                dt0, dt1 = self.dtype_of(d0), self.dtype_of(d1)
                c0 = self.mp.unique(py_ident(s0.name) + "_u8")
                c1 = self.mp.unique(py_ident(s1.name) + "_u8")
                self.emit(f"{c0} = vf.bit_cast({self.name(s0)}, dtype=pl.DT_UINT8)")
                self.emit(f"{c1} = vf.bit_cast({self.name(s1)}, dtype=pl.DT_UINT8)")
                t0 = self.mp.unique(py_ident(d0.name) + "_u8")
                t1 = self.mp.unique(py_ident(d1.name) + "_u8")
                self.emit(f"{t0}, {t1} = vf.de_interleave({c0}, {c1})")
                self.emit(f"{self.define(d0)} = vf.bit_cast({t0}, dtype={self.pl_dt(op, dt0)})")
                self.emit(f"{self.define(d1)} = vf.bit_cast({t1}, dtype={self.pl_dt(op, dt1)})")
                return
            self.emit(f"{self.name(d0)}, {self.name(d1)} = vf.de_interleave({self.name(s0)}, {self.name(s1)})")
            return
        if code in ("vf.squeeze", "vf.gathermask"):
            dst, src = op.operands
            mask = self.mask_of(op, src)
            dtype = self.dtype_of(src)
            source = self.name(src)
            if _esize_bits(dtype) == 8 and dtype not in ("i8", "u8"):
                carrier = self.mp.unique(py_ident(src.name) + "_sqz_bits")
                packed = self.mp.unique(py_ident(dst.name) + "_sqz_bits")
                self.emit(f"{carrier} = vf.bit_cast({source}, dtype=pl.DT_UINT8)")
                self.emit(f"{packed} = vf.squeeze({carrier}, {mask}, gather_mode=pl.SqueezeMode.NO_STORE_REG, dtype=pl.DT_UINT8)")
                self.emit(f"{self.name(dst)} = vf.bit_cast({packed}, dtype={self.pl_dt(op, self.dtype_of(dst))})")
            else:
                self.emit(f"{self.name(dst)} = vf.squeeze({source}, {mask}, gather_mode=pl.SqueezeMode.NO_STORE_REG, dtype={self.pl_dt(op, self.dtype_of(dst))})")
            return
        if code == "vf.histograms":
            dst, src = op.operands
            group = op.attrs.get("bin_group", 0)
            mode = str(op.attrs.get("mode", "frequency")).lower()
            if group not in (0, 1) or mode not in ("accumulate", "frequency"):
                raise PyptoGap(op, f"unsupported histogram group/mode {group}/{mode}", owner="ours")
            self.emit(f"{self.name(dst)} = vf.histograms({self.name(src)}, {self.mask_of(op, src)}, bin_type=pl.BinType.BIN{group}, hist_type=pl.HistType.{mode.upper()})")
            return
        if code == "vf.mask_and":
            dst, left, right = op.operands
            lhs, rhs = self.name(left), self.name(right)
            gate = op.attrs.get("mask")
            predicate = self.name(gate) if isinstance(gate, Value) else self.all_mask("u8")
            target = self.mask_write(op, dst)
            self.emit(f"{target} = vf.and_({lhs}, {rhs}, {predicate}, mode=pl.MergeMode.ZEROING)")
            return
        if code == "vf.interleave":
            source_dtype = self.dtype_of(op.operands[-2])
            if _esize_bits(source_dtype) == 8 and source_dtype not in ("i8", "u8"):
                destinations, sources = op.operands[:-2], op.operands[-2:]
                carriers = []
                for source in sources:
                    carrier = self.mp.unique(py_ident(source.name) + "_intlv_u8")
                    self.emit(f"{carrier} = vf.bit_cast({self.name(source)}, dtype=pl.DT_UINT8)")
                    carriers.append(carrier)
                results = [self.mp.unique(py_ident(dst.name) + "_intlv_u8") for dst in destinations]
                self.emit(f"{', '.join(results)} = vf.interleave({', '.join(carriers)})")
                for dst, result in zip(destinations, results, strict=True):
                    self.emit(f"{self.name(dst)} = vf.bit_cast({result}, dtype={self.pl_dt(op, self.dtype_of(dst))})")
                return
            if len(op.operands) == 4:
                d0, d1, s0, s1 = op.operands
                self.emit(f"{self.name(d0)}, {self.name(d1)} = vf.interleave({self.name(s0)}, {self.name(s1)})")
            else:
                d0, s0, s1 = op.operands
                self.emit(f"{self.name(d0)} = vf.interleave({self.name(s0)}, {self.name(s1)})")
            return
        if code in ("vf.gather_copy", "vf.gatherb"):
            dst, src, idx = op.operands
            if code == "vf.gather_copy":
                self.check_dtype64(op, dst)  # vgather2 has no 64-bit form; block gatherb does
            if code == "vf.gather_copy" and self.dtype_of(src) != self.dtype_of(dst):
                raise PyptoGap(op, f"vf.gather_copy {self.dtype_of(src)}->{self.dtype_of(dst)}: pypto's "
                                   "b8 gather widening is zero-extend only (board diff on the signed case)",
                               owner="upstream")
            base, off_s = self.tile_ref(op, src, op.attrs.get("offset", 0))
            if off_s != "0":
                raise PyptoGap(op, f"{code} with a nonzero base offset has no pl.gather spelling")
            m = self.mask_of(op, dst)
            kw = ", data_copy_mode=pl.DataCopyMode.DATA_BLOCK_LOAD" if code == "vf.gatherb" else ""
            self.emit(f"{self.name(dst)} = vf.gather({base}, {self.name(idx)}, {m}{kw})")
            return
        if code == "vf.gather":  # Reg -> Reg form
            dst, src, idx = op.operands
            self.emit(f"{self.name(dst)} = vf.gather({self.name(src)}, {self.name(idx)})")
            return
        if code == "vf.scatter_copy":
            dst, src, idx = op.operands
            self.check_dtype64(op, src)
            base, off_s = self.blk_addr(op, dst)
            data = self.name(src)
            if self.dtype_of(src) == "hif8":
                # Scatter copies byte payloads; the upstream VF registry
                # accepts UINT8 but not HF8. Keep identical lanes/predicates
                # and carry both the register and UB pointer as UINT8.
                base = self.block_carrier(op, dst, base)
                data = self.mp.unique("scatter_u8")
                self.emit(f"{data} = vf.bit_cast({self.name(src)}, dtype=pl.DT_UINT8)")
            address = base if off_s == "0" else f"{base} + ({off_s})"
            m = self.mask_of(op, src)
            self.emit(f"vf.scatter({address}, {data}, {self.name(idx)}, {m})")
            return
        raise PyptoGap(op, f"vf op {code} is outside the pypto surface")


# ------------------------------------------------------------------ one side of the kernel


class _Event:
    def __init__(self, set_pipe: str, wait_pipe: str, ids: list[int], preset: int,
                 table: str | None = None, sc: str | None = None, wc: str | None = None) -> None:
        self.set_pipe = set_pipe
        self.wait_pipe = wait_pipe
        self.ids = ids
        self.preset = preset
        self.set_seq = preset  # presets consume the first `preset` set slots
        self.wait_seq = 0
        self.table = table  # python id-table name (multi-id events; ids live as a device array)
        self.sc = sc  # runtime set-call counter variable
        self.wc = wc  # runtime wait-call counter variable


class SideEmitter:
    """One ``func`` (side = cube | vec) -> the statements of its ``pl.section_*`` block."""

    def __init__(self, mp: "ModulePrinter", fn: Function) -> None:
        self.mp = mp
        self.fn = fn
        self.side = str(fn.attrs.get("side", "vec"))
        self.lines: list[str | TileSite] = []
        self.indent = 0
        self.events: dict[str, _Event] = {}
        self.tiles: dict[str, dict[str, Any]] = {}  # value name -> {py, dtype, shape, space}
        # value name -> (tensor py name, per-axis offsets, folded extents or None, source dims)
        self.views: dict[str, tuple[str, list[str], list[int] | None, list[Any]]] = {}
        self.regroups: dict[tuple, str] = {}  # (buf, dtype, shape, space) -> reinterpreted tile-group name
        self.hoisted: list[str | TileSite] = []  # section-top declarations (tile groups born inside branches)
        self.mutex_drain: list[str] = []  # producer-side token drains, flushed after the body
        self.atomic: str | None = None  # inside an atomic.begin / atomic.end region
        self.atomic_dtype: str | None = None  # the legacy atomic.set_type inside it
        self.params: dict[str, str] = {}
        self._ix_cache: dict[str, tuple[str, int, int]] = {}  # idx expr -> (name, indent, gen)
        self._ix_seen: dict[str, int] = {}  # idx expr -> times materialised (function-wide)
        self._ix_gen = 0
        self._last_indent = 0
        self.env = ScalarEnv(mp, fn, self.side)
        self.stable_indices: set[str] = set()
        self.index_spelling = mp.index_spelling
        self.ranges = mp.scalar_ranges[fn.name]
        self.mutex_allocations = {op.results[0].name: op for op in fn.walk()
                                  if op.opcode == "mem.alloc" and op.attrs.get("mutex_ids")}
        # Built in BOTH modes. RFC-0013 requires the allocation IDs to remain in the Tile
        # declarations and the artifact metadata whichever mode prints the credits; manual mode
        # additionally prints the IR's own mutex operations and emits auto_mutex=False, so the
        # declared IDs are documentation there rather than an instruction to PyPTO.
        self.native_tiles = TileRegistry(self.side, PyptoGap) if self.mutex_allocations else None

    def tile_decl(self, op: Op, line: str, *, name: str | None = None, type_expr: str = "",
                  bank: str = "", addresses: tuple[int, ...] = (), size: int = 0,
                  shape: list[int] | None = None, dtype: str = "", group: bool = False,
                  hoisted: bool = False, parent_slots: tuple[tuple[str, int], ...] = ()) -> None:
        """The only side-emitter declaration gateway; unsupported adapters fail closed.

        Manual declarations retain their exact original spelling and placement. Native
        declarations carry physical facts and are rendered after all aliases are known.
        """
        if self.native_tiles is None:
            (self.hoist if hoisted else self.emit)(line)
            return
        if name is None:
            mp_mode = self.mp.sync_mode
            raise PyptoGap(op, f"sync_mode={mp_mode!r}: this backend-created tile adapter has no "
                               "proved physical declaration mapping", owner="ours")
        ids = tuple(op.attrs.get("mutex_ids", ())) if op.opcode == "mem.alloc" else ()
        if bank == "ub" and shape and len(shape) == 2 and shape[0] == 1:
            # The allocator reserves a whole aligned row even for a short UB slot.
            native_size = shape[1] * _esize_bits(dtype) // 8
            if size < native_size <= -(-size // 32) * 32:
                size = native_size
        self.native_tiles.add(TileDecl(name, name if group else self.mp.unique(name + "_native"),
                                       type_expr, bank, addresses, size, tuple(shape or ()),
                                       _esize_bits(dtype), op, singleton=not group, original=line,
                                       mutex_ids=ids, parent_slots=parent_slots))
        (self.hoisted if hoisted else self.lines).append(TileSite(name, 0 if hoisted else self.indent))

    def make_tile(self, op: Op, tt: str, addr: Any, size: Any, shape: list[int], dtype: str, space: str) -> str:
        """``pl.make_tile(tt, addr=...)`` for a tile whose allocation is exactly what its type spans.

        PyPTO derives a tile's byte span from its TileType: ``size=`` was optional (the same default) until upstream
        removed the keyword (9528ff753, 2026-09-17), so the text without it is the one every version reads. The
        allocation must therefore BE that footprint - or, for a one-row Vec tile that `vec_row_align` declared to a
        whole 32-byte block, the block the allocator reserved for it. Anything else has no spelling any more and is
        refused here rather than declared with a span the allocator did not give it. Sub-byte elements count
        packed, as the allocator and upstream since that commit count them.
        """
        footprint = (math.prod(shape) * _esize_bits(dtype) + 7) // 8 if all(type(n) is int for n in shape) else None
        aligned_row = space == "ub" and len(shape) == 2 and shape[0] == 1
        if type(size) is not int or footprint is None or not (
                footprint == size or (aligned_row and size < footprint <= -(-size // 32) * 32)):
            raise PyptoGap(op, f"pl.make_tile over {size} allocated bytes for a {dtype} {list(shape)} tile: PyPTO "
                               f"derives the span from the TileType ({footprint} bytes) and takes no size", owner="ours")
        return f"pl.make_tile({tt}, addr={addr})"

    def emit(self, line: str) -> None:
        if self.indent < self._last_indent:
            self._ix_gen += 1  # a block closed: everything declared inside it is out of scope
        self._last_indent = self.indent
        lhs = line.split(" = ", 1)[0].strip() if " = " in line else None
        if lhs and self._ix_cache:
            # a variable feeding a cached index expression was reassigned (counter step):
            # the cached selection no longer means the same thing
            for key in [k for k in self._ix_cache if lhs in k]:
                del self._ix_cache[key]
        self.lines.append("    " * self.indent + line)

    def _ub_nd2nz(self, op: Op, dst: Value, src: Value) -> None:
        """cce's ub_to_l1_nd2nz reorders ND rows into NZ fractal columns by issuing one
        copy_ubuf_to_cbuf per C0-wide strip: (block_count=m_src, block_len=1,
        src_stride=ceil(N_src/C0)-1, dst_stride=0), the i-th strip reading src[i*C0] and
        writing dst[i * C0 * align16(m_dst)].

        pto reaches the SAME burst through TInsertNDImpl's middle branch (TInsert.hpp:326):
        with an ND (row-major) source, validCol == C0 gives rowBurstLen = 1 and
        srcRowGap = SrcTileData::Cols/C0 - 1, and a DESTINATION DECLARED C0 WIDE gives
        dstRowGap = 0 - parameter for parameter what cce composes by hand. So one
        pl.insert per strip, between
          - a Vec source alias at the strip's column offset, declared [m_src, N_src] (its
            Cols is what sets srcRowGap) and narrowed to [m_src, C0] by set_validshape, and
          - a Mat destination alias of one fractal column, declared [align16(m_dst), C0]
            NZ (a single C0 column block is contiguous, so its bytes ARE the fractal).
        D-089 read TInsertVecToMatImpl's ND branch as 'copies flat' and gapped the op; the
        flat copy is real but it is the branch's FIRST arm (validCol == SrcTileData::Cols),
        which a C0-narrowed source never takes."""
        self.guard_attrs(op, {"m_src", "n_src", "m_dst", "n_dst", "N_src", "dst_row0", "dst_col0"})
        # D-098 gapped four attention kernels by name here because they failed on silicon
        # the first time nd2nz put them there. Both causes were found since and fixed - the
        # rank-aware `order` axis selection and vf.arange's dropped start attr (D-101/D-102)
        # - and the strip composition itself was never implicated: its burst parameters were
        # re-checked against TInsertNDImpl and match cce's ub_to_l1_nd2nz one for one.
        d, sr = self.tiles.get(dst.name), self.tiles.get(src.name)
        if d is None or sr is None:
            raise PyptoGap(op, "ub_to_l1.nd2nz on a tile the printer has not seen")
        # a windowed operand rides on its base: the window's own offsets add to the op's
        # (cce emit.py:960 - dma attrs REPEAT the view offsets, so reading them twice would
        # double-count; the base tile and the extra fractal origin are what the aliases need)
        # a windowed operand rides on its base (cce emit.py:960 - the dma attrs REPEAT the
        # window's offsets, so the record contributes only its base tile, not a second shift)
        def _root(rec: dict, what: str) -> dict:
            if "py" in rec:
                return rec
            if not rec.get("base_py"):
                raise PyptoGap(op, f"ub_to_l1.nd2nz through an unrooted {what} window")
            root = rec.get("root") or {}
            return {**root, "py": rec["base_py"], "shape": rec["base_shape"]}
        # the SOURCE strips are addressed (a tile alias per C0 column), so its window must
        # fold; the DESTINATION row offset rides pl.insert's indexRow, which takes a runtime
        # expression - the fractal-column part still folds (it selects which alias)
        if "py" not in sr:
            for terms in sr["offs_ir"]:
                if self._off_fold(terms) is None:
                    raise PyptoGap(op, "ub_to_l1.nd2nz from a runtime-offset source window: "
                                       "the strip addresses must fold")
        dwin_r = "0"
        dcol_ir = None  # the destination fractal column when it does not fold
        if "py" not in d:
            if self._off_fold(d["offs_ir"][1]) is None:
                dcol_ir = d["offs_ir"][1]
            dwin_r = self._off_expr(op, d["offs_ir"][0])
        d, sr = _root(d, "destination"), _root(sr, "source")
        n_src = self.env.fold(op.attrs.get("n_src"))
        N_src = self.env.fold(op.attrs.get("N_src", op.attrs.get("n_src")))
        m_dst = self.env.fold(op.attrs.get("m_dst"))
        r0 = self.env.fold(op.attrs.get("dst_row0", 0))
        c0_off = self.env.fold(op.attrs.get("dst_col0", 0))
        if None in (n_src, N_src, m_dst) or (c0_off is None and dcol_ir is None):
            raise PyptoGap(op, "ub_to_l1.nd2nz with runtime geometry: the strip count and the "
                               "fractal column must fold (pl.make_tile takes no runtime dims)")
        r0_txt = str(r0) if r0 is not None else self.iexpr(op, op.attrs.get("dst_row0", 0))
        if dwin_r != "0":  # the window record repeats the attr offsets - one source, not two
            r0_txt = dwin_r
            r0 = self._off_fold(self.tiles[dst.name]["offs_ir"][0]) if "py" not in self.tiles.get(dst.name, {"py": 1}) else r0
        esz = _esize_bits(sr["dtype"]) // 8
        c0 = 32 // esz
        if n_src % c0 or N_src % c0:
            raise PyptoGap(op, f"ub_to_l1.nd2nz with n_src={n_src} / N_src={N_src} not a multiple "
                               f"of the {c0}-element datablock: the strip burst would split a block")
        dcol_txt = ""
        if dcol_ir is not None:
            # the column is chosen at runtime, so it may only move in whole fractal columns
            aff = self._off_affine(dcol_ir)
            if aff is None or aff[0] % c0 or aff[1] % c0:
                raise PyptoGap(op, "ub_to_l1.nd2nz into a window at a runtime COLUMN offset that is "
                                   "not an aligned progression: the fractal column selects the "
                                   "destination alias, so it has to step whole C0 columns")
            dcol_txt = self._off_expr(op, dcol_ir)
        if (r0 is not None and r0 % 16) or (c0_off is not None and c0_off % c0):
            raise PyptoGap(op, f"ub_to_l1.nd2nz at dst ({r0}, {c0_off}): a strip alias starts at a "
                               "fractal boundary (row % 16, col % C0)")
        rows16 = -(-m_dst // 16) * 16
        m_src_k = self.env.fold(op.attrs.get("m_src"))
        ms = str(m_src_k) if m_src_k is not None else self.iexpr(op, op.attrs.get("m_src"))
        strips = n_src // c0
        if strips > 64:
            raise PyptoGap(op, f"ub_to_l1.nd2nz of {strips} strips: past the per-op alias budget")
        def _alias(rec: dict, space: str, shape: list[int], byte_off: int, tag: str) -> str:
            """One strip's view of an operand: a plain tile when the base address is static,
            an entry of a narrow tile GROUP over the slot addresses when it rotates (the
            D-091 regroup idiom - a rotating slot has no compile-time address, and pto's
            make_tile refuses an expression one)."""
            layout = "pl.NZ" if space == "l1" else None
            tt = self.mp.tiletype(rec["dtype"], list(shape), space, layout)
            nbytes = shape[0] * shape[1] * (_esize_bits(rec["dtype"]) // 8)
            py = self.mp.unique(rec["py"] + tag)
            # ND source carrier rows preserve the native layout, but only the
            # narrowed m_src-by-C0 window is read. Its parent owns the mutex.
            if rec.get("addr") is not None:
                address = f"({rec['addr']} + {byte_off})"
                self.tile_decl(op, f"{py} = {self.make_tile(op, tt, address, nbytes, shape, rec['dtype'], space)}",
                               name=py, type_expr=tt, bank=space,
                               addresses=((rec["native_addr"] + byte_off) if rec.get("native_addr") is not None else None,),
                               size=nbytes, shape=list(shape), dtype=rec["dtype"],
                               parent_slots=((rec["py"], 0),) if space == "ub" else ())
                return py
            b = self.tiles.get(rec.get("group_of") or "")
            if not b or rec.get("index_py") is None:
                raise PyptoGap(op, "ub_to_l1.nd2nz through a rotating slot with no group handle")
            key = (rec["group_of"], rec["dtype"], tuple(shape), space, byte_off)
            gn = self.regroups.get(key)
            if gn is None:
                gn = self.mp.unique(py_ident(rec["group_of"]) + "_nd")
                base = int(b["addr"])
                addrs = ", ".join(str(base + j * b.get("pitch", b["size"]) + byte_off) for j in range(b["slots"]))
                self.tile_decl(op, f"{gn} = pl.make_tile_group(type={tt}, addrs=[{addrs}], depth={b['slots']})",
                               name=gn, type_expr=tt, bank=space,
                               addresses=tuple(base + j * b.get("pitch", b["size"]) + byte_off for j in range(b["slots"])),
                               size=nbytes, shape=list(shape), dtype=rec["dtype"], group=True, hoisted=True,
                               parent_slots=tuple((b["group_py"], j) for j in range(b["slots"])) if space == "ub" else ())
                self.regroups[key] = gn
            self.getitem(py, gn, rec["index_py"])
            return py

        def _dyn_col(rec: dict, i: int) -> str:
            """The destination fractal column chosen at RUNTIME. pl.make_tile's addr must be a
            literal, but a tile GROUP's addrs are literals and its index is an expression
            (behaviour #25) - so every fractal column of the base gets declared once and the
            runtime column indexes it. A rotating destination flattens to slots x columns."""
            cols = rec["shape"][1]
            if not isinstance(cols, int) or cols % c0:
                raise PyptoGap(op, "ub_to_l1.nd2nz into a runtime column of a destination whose "
                                   "own column count is not a whole number of fractal columns")
            ncol, pitch = cols // c0, rows16 * c0 * esz
            pre = ""
            if rec.get("addr") is not None:
                bases = [int(rec["addr"])]
            else:
                b = self.tiles.get(rec.get("group_of") or "")
                if not b or rec.get("index_py") is None:
                    raise PyptoGap(op, "ub_to_l1.nd2nz into a runtime column of a rotating slot "
                                       "with no group handle")
                bases = [int(b["addr"]) + j * int(b.get("pitch", b["size"])) for j in range(int(b["slots"]))]
                pre = f"{paren(rec['index_py'])} * {ncol} + "
            tt = self.mp.tiletype(rec["dtype"], [rows16, c0], "l1", "pl.NZ")
            key = ("dyncol", rec["py"], rec["dtype"], rows16, c0, pitch, len(bases))
            gn = self.regroups.get(key)
            if gn is None:
                gn = self.mp.unique(py_ident(rec["py"]) + "_fc")
                addrs = ", ".join(str(base + j * pitch) for base in bases for j in range(ncol))
                self.tile_decl(op, f"{gn} = pl.make_tile_group(type={tt}, addrs=[{addrs}], "
                           f"depth={ncol * len(bases)})", name=gn, type_expr=tt, bank="l1",
                           addresses=tuple(base + j * pitch for base in bases for j in range(ncol)),
                           size=pitch, shape=[rows16, c0], dtype=rec["dtype"], group=True, hoisted=True)
                self.regroups[key] = gn
            py = self.mp.unique(rec["py"] + "_fc")
            col = self._ordinal(dcol_txt, 0, c0)
            self.getitem(py, gn, f"{pre}{col} + {i}" if i else f"{pre}{col}")
            return py

        for i in range(strips):
            spy = _alias(sr, "ub", [m_dst, N_src], i * c0 * esz, "_nd")
            self.emit(f"pl.set_validshape({spy}, [{ms}, {c0}])")
            if dcol_ir is not None:
                dpy = _dyn_col(d, i)
            else:
                frac = (c0_off // c0 + i) * rows16 * c0 * esz
                dpy = _alias(d, "l1", [rows16, c0], frac, "_fr")
            # the row origin rides indexRow: TInsertNDImpl adds indexRow * dstCols (= C0)
            # elements, exactly the row's offset inside this one fractal column
            self.emit(f"pl.insert({dpy}, {spy}, [{r0_txt}, 0])")
            # PyPTO CSE can bind the first strip to the original UB tile.
            # Restore its descriptor so the next iteration's GM load fills
            # every column instead of retaining stale data past C0.
            self.emit(f"pl.set_validshape({spy}, [{m_dst}, {N_src}])")

    def u8_carrier(self, op: Op, v: Any, shape: list[int]) -> str:
        """The uint8 carrier tile a vf parameter needs (D-126). A tile that reached the call as a
        RUNTIME-INDEXED group element gets its carrier from a PARALLEL uint8 group over the same
        addresses: reinterpreting the element itself inside the callee folds the slot away and
        every rotation writes slot 0."""
        if not isinstance(v, Value) or v.name not in self.tiles:
            raise PyptoGap(op, f"the uint8 carrier of {v!r} needs a tile argument")
        rec = self.tiles[v.name]
        parent = rec.get("group_of")
        if parent is not None:
            b = self.tiles.get(parent, {})
            key = (parent, tuple(shape))
            g = self.mp.u8_groups.get(key)
            if g is None:
                addr, slots, pitch = b.get("addr"), b.get("slots"), b.get("pitch")
                if addr is None or not slots or not pitch or not str(addr).lstrip("-").isdigit():
                    raise PyptoGap(op, f"the uint8 carrier of {v.name} needs {parent}'s folded "
                                       "slot base, pitch and depth")
                tt = self.mp.tiletype("u8", list(shape), b["space"])
                addrs = ", ".join(str(int(addr) + i * int(pitch)) for i in range(int(slots)))
                g = self.mp.unique(py_ident(parent) + "_u8_grp")
                self.tile_decl(op, f"{g} = pl.make_tile_group(type={tt}, addrs=[{addrs}], depth={int(slots)})",
                               name=g, type_expr=tt, bank=b["space"],
                               addresses=tuple(int(addr) + i * int(pitch) for i in range(int(slots))),
                               size=shape[0] * shape[1], shape=shape, dtype="u8", group=True, hoisted=True)
                self.mp.u8_groups[key] = g
            py = self.mp.unique(py_ident(v.name) + "_u8")
            self.getitem(py, g, rec["index_py"])
            return py
        if rec.get("py") is None:
            raise PyptoGap(op, f"the uint8 carrier of {v.name} needs a named tile")
        py = self.mp.unique(str(rec["py"]) + "_u8")
        self.emit(f"{py} = pl.reinterpret({rec['py']}, dtype=pl.DT_UINT8, shape={list(shape)})")
        return py

    @staticmethod
    def _ordinal(expr: str, first: int = 0, step: int = 1) -> str:
        """``(expr - first) // step`` spelled as a reader would: a zero origin and a unit step
        print nothing, and a bracket that encloses the whole expression is not doubled. The
        argument comes from `_off_expr` or from here, so it is already atomic or bracketed."""
        expr = paren(unparen(expr))
        if first:
            expr = f"({unparen(expr)} - {first})"
        return expr if step == 1 else f"{expr} // {step}"

    @staticmethod
    def _flatten(slot: str, count: int, index: str) -> str:
        """The index of a ``slots x count`` tile group, without the terms that do nothing."""
        if count == 1:
            return paren(unparen(slot)) if index == "0" else f"{paren(slot)} + {index}"
        head = f"{paren(slot)} * {count}"
        return head if index == "0" else f"{head} + {index}"

    def getitem(self, py: str, group: str, idx: str) -> None:
        """``py = group[idx]``.

        Direct by default: `lower_group_subscript` accepts any integer scalar expression and
        let-binds it itself (`_bufidx_N`), so a name of ours adds a line and nothing else.

        `ASCRIPTOR_PYPTO_INDEX=named` restores the older spelling, which materialises the index
        under its own `_ix` name and re-spells a repeat through `(x + k*m) % m`. That shape was
        adopted for a board failure (undeclared `_expr_tmp` / `_ix` in TMATMUL) read as pto CSE-ing
        textually equal index expressions across scopes and dropping the later declaration without
        rewriting its uses. No such pass exists in the pl sources of either qualified version, and
        upstream's own tests repeat identical index text, so the switch exists to measure the two
        spellings against each other on hardware rather than to carry an unexplained workaround."""
        if self.index_spelling == "direct" or idx in self.stable_indices or idx.isdecimal():
            self.emit(f"{py} = {group}[{idx}]")
            return
        key = idx  # the materialised index depends on the expression alone, not the group
        hit = self._ix_cache.get(key)
        if hit is not None and hit[2] == self._ix_gen and hit[1] <= self.indent:
            self.emit(f"{py} = {group}[{hit[0]}]")
            return
        ix = self.mp.unique("_ix")
        spell = idx
        n = self._ix_seen.get(idx, 0)
        if n:
            # pto's native codegen CSEs textually-equal let right-hand sides across block
            # scopes and drops the later declaration without rewriting its uses (board:
            # undeclared _ix in TMATMUL) - and two spellings can even hold DIFFERENT values
            # (a counter stepped in between). Re-spell the repeat through the modulo
            # identity (x + k*m) % m == x % m (counters never go negative), which keeps the
            # value and makes the text function-unique.
            head, _, tail = idx.rpartition(" % ")
            if tail.isdigit():
                m = int(tail)
                spell = f"({head} + {m * n}) % {m}"
        self._ix_seen[idx] = n + 1
        self.emit(f"{ix} = {spell}")
        self._ix_cache[key] = (ix, self.indent, self._ix_gen)
        self.emit(f"{py} = {group}[{ix}]")

    def hoist(self, line: str) -> None:
        """Emit a declaration at the section top: a name assigned inside an if/for goes out
        of scope in pto's DSL codegen (its own tests are written branchless for this)."""
        self.hoisted.append(line)

    # -------- scalar/int helpers

    def iexpr(self, op: Op, x: Any) -> str:
        k = self.env.fold(x)
        if k is not None:
            return str(k)
        if isinstance(x, Literal) or isinstance(x, (int, float, bool)):
            return _lit(x)
        if isinstance(x, Value):
            return self.env.ref(x)
        raise PyptoGap(op, f"value {x!r} has no scalar spelling")

    def ifold(self, op: Op, x: Any, what: str) -> int:
        k = self.env.fold(x)
        if k is None:
            raise PyptoGap(op, f"{what} {x!r} does not fold to a compile-time int "
                               "(pypto tile shapes are compile-time; bind the scalar parameters)")
        return k

    # -------- tiles

    def _loop_ref(self, v: Any, seen: set | None = None) -> bool:
        """True when a scalar value's def chain touches a cf.for variable or a loop-set cell."""
        if not isinstance(v, Value):
            return False
        seen = seen or set()
        if v.name in seen:
            return False
        seen.add(v.name)
        d = self.env.defs.get(v.name)
        if d is None:
            return False
        if d.opcode == "cf.for":
            return True
        if d.opcode == "scalar.cell":
            for o in self.fn.body.walk():
                if o.opcode != "scalar.set" or not o.operands or o.operands[0].name != v.name:
                    continue
                if o not in self.fn.body.ops:  # set below a region boundary = inside a loop/branch
                    return True
            return False
        return any(self._loop_ref(x, seen) for x in d.operands if isinstance(x, Value))

    @staticmethod
    def vec_row_align(space: str, dtype: str, shape: list, size: int) -> list:
        """A Vec tile's ROW is a whole number of 32-byte blocks in pto's TYPE itself
        ("BFractal_ is RowMajor and SFractal_ is NoneBox: Rows must be 32 bytes align",
        pto_tile.hpp:1509), so a [1, 200] fp16 row - 400 bytes - does not compile. A ONE-ROW
        tile has no row transition, so declaring the padded column count names only bytes the
        allocator already reserved for this tile's own slot (it bumps by the same 32-byte
        unit) and leaves every offset into it unchanged."""
        if space != "ub" or len(shape) != 2 or shape[0] != 1 or not isinstance(shape[1], int):
            return shape
        esz = _esize_bits(dtype) // 8
        row = shape[1] * esz
        if row % 32 == 0:
            return shape
        wide = -(-row // 32) * 32 // esz
        return [1, wide] if wide * esz <= -(-size // 32) * 32 else shape

    def deftile(self, op: Op, v: Value, *, space: str, dtype: str, shape: list[int],
                addr_expr: str, size: int, hint: str | None = None, flip_from: bool = False,
                native_addr: int | None = None) -> None:
        if dtype not in PL_DT:
            raise PyptoGap(op, f"tile dtype {dtype} has no pypto DT_* constant")
        if space not in MEMSPACE:
            raise PyptoGap(op, f"memory space {space} has no pypto MemorySpace")
        layout = None
        flip = bool(flip_from)
        if space == "l0b":
            if not flip:
                shape = [shape[1], shape[0]]  # pto's rhs naming: [K, N] (same bytes as our [N, K])
            flip = True
            if isinstance(shape[1], int) and shape[1] % 16:
                # a narrow N is a VALID SHAPE, never a DECLARATION: Tile<Right, half, 32, 2>
                # fails static_assert(Cols % InnerCols == 0). Declare the fractal the fragment
                # lives in and let the matmul's own set_validshape narrow it (the same rule the
                # L0C strip follows). The bytes past N belong to this fragment's own slot.
                wide = -(-shape[1] // 16) * 16
                if wide * shape[0] * _esize_bits(dtype) // 8 <= size:
                    shape = [shape[0], wide]
        elif space == "l1" and not flip and (v.name in getattr(self, "transposed_b", ())
                                             or v.name in getattr(self, "transposed_a", ())):
            shape = [shape[1], shape[0]]  # an order=[1, 0] chain: L1 lives transposed
            flip = True
        elif space == "l1" and v.name in getattr(self, "zn_tiles", ()):
            shape = [shape[1], shape[0]]  # the ZN reading of the same bytes (SFractal gate)
            layout = "pl.ZN"
        shape = self.vec_row_align(space, dtype, shape, size)
        if space == "l1" and v.name in getattr(self.mp, "scale_side", {}):
            # An MX per-group scale is CONSUMED from ScaleLeft/ScaleRight, and pto admits
            # exactly one route into those stops - each leg read off the board:
            #   * pl.load refuses a Scale destination ("dst tile must be in Vec (UB) or Mat
            #     (L1) memory, got ScaleLeft"), so it cannot be filled from GM directly;
            #   * the move whitelist (block_ops.py:642) has Mat->ScaleLeft/ScaleRight and
            #     nothing from Vec, so UB cannot stage it either;
            #   * a Mat tile asserts its fractal columns in the C++ TYPE itself
            #     (pto_tile.hpp:1529 Cols % InnerCols == 0; such a Tile has no GetValidRow),
            #     so the [rows, k_groups] block - 2 or 4 bytes per row - cannot be declared
            #     as a Mat tile at its LOGICAL shape.
            # It can be declared at its byte-equal ALIGNED shape: the load that fills it is
            # one flat burst (cce's gm_to_l1_mx_scale_nd2nz moves rows * k_groups bytes as a
            # unit), so the same bytes read as [1, rows*k_groups] hold identical content and
            # satisfy the fractal rule. The logical shape is kept for the Scale-side tile.
            esz_s = 1  # e8m0
            total = 1
            for x in shape:
                if not isinstance(x, int):
                    total = None
                    break
                total *= x
            if total is None or (total * esz_s) % 32:
                raise PyptoGap(op, f"MX scale block {shape}: its byte count must be a whole "
                                   "number of 32-byte blocks to stage as an aligned Mat tile")
            # The MX scale's route, from pto's own matmul_mx example: the L1 staging tile is
            # a MAT tile of the LOGICAL [rows, k_groups] shape, dtype E8M0, declared
            # layout=pl.ZZ on the lhs side and pl.NN on the rhs - the RowMajor/RowMajor
            # fractal pair TExtractToAmx asserts (TExtract.hpp:29). The GM tensor it is
            # loaded from is declared E8M0 too (our IR calls it a u8 carrier because torch
            # has no e8m0 dtype; the bytes are the same).
            dtype = "e8m0"
            # pto's matmul_mx example stages A's scale as [rows, k_groups] / ZZ and B's as
            # [k_groups, rows] / NN - the second is the transposed READING of the same bytes
            # (both put group g of row r at 2*(r%16)+g%2 inside box (r//16, g//2)), which is
            # what pl.matmul_mx demands: it checks scale_b against the rhs tile's own [K, N]
            # coordinates.
            if self.mp.scale_side[v.name] == "scale_l":
                layout = "pl.ZZ"
            else:
                shape, layout = [shape[1], shape[0]], "pl.NN"
        if space == "ub" and len(shape) == 2 and all(isinstance(x, int) for x in shape) \
                and shape[0] != 1 and (shape[1] * _esize_bits(dtype) // 8) % 32:
            # a Vec tile's ROW is a whole number of 32-byte blocks in pto's type itself
            # (pto_tile.hpp: "BFractal_ is RowMajor and SFractal_ is NoneBox: Rows must be 32
            # bytes align"), so a narrow 2-D UB tile does not compile at all. Its bytes are
            # flat - Vec has no fractal - so the same storage read as one row is legal, but
            # only a consumer that treats it flat may have it.
            flat = [1, shape[0] * shape[1]]
            if (flat[1] * _esize_bits(dtype) // 8) % 32:
                raise PyptoGap(op, f"UB tile declared {shape} ({dtype}): a Vec tile's row must be "
                                   "a whole number of 32-byte blocks (board static-assert) and "
                                   "the flat reading of these bytes is not one either")
            if any(o.opcode.startswith(("vf.", "vec.", "simt."))
                   and any(isinstance(x, Value) and x.name == v.name for x in o.operands)
                   for o in self.fn.body.walk()):
                raise PyptoGap(op, f"UB tile declared {shape} ({dtype}): a Vec tile's row must be "
                                   "a whole number of 32-byte blocks (board static-assert). Its "
                                   "bytes are flat, but a register-level consumer addresses it "
                                   "in rows, so the flat reading is not interchangeable here")
            shape = flat  # flat is the same storage and the only legal declaration
        is_scale = v.name in getattr(self.mp, "scale_side", {})
        # the fractal gates below read an NZ box (16 rows x 32 bytes); a ZN tile is the same
        # box measured the other way round, so its two extents swap roles
        zn = layout == "pl.ZN"
        fr_cols, fr_rows = (shape[0], shape[1]) if zn and len(shape) == 2 else (shape[1] if len(shape) == 2 else None, shape[0])
        if (not is_scale and space == "l1" and len(shape) == 2 and isinstance(fr_cols, int)
                and fr_rows != 1 and (fr_cols * _esize_bits(dtype)) % 256):
            raise PyptoGap(op, f"L1 tile declared {shape} ({dtype}): pto's Mat NZ inner box wants "
                               "cols divisible by 32 bytes (pto_tile.hpp:1529 static-assert; "
                               "cce pads internally)")
        if not is_scale and space == "l1" and len(shape) == 2 and isinstance(fr_rows, int) and fr_rows % 16:
            esz = _esize_bits(dtype) // 8
            if fr_rows == 1 and isinstance(fr_cols, int) and (fr_cols * esz) % 32 == 0:
                # a bias ROW: pto itself demotes a 1-wide Mat tile to ColMajor/DN ("1-wide
                # tile cannot meet the 32-byte column alignment a fractal layout needs"),
                # and its parser rejects an explicit layout override (board: "Mat tiles
                # require layout in {NZ, ZN}, got ND") - declare NO layout, like pto's own
                # test_matmul_bias.py bias_mat group
                pass
            else:
                raise PyptoGap(op, f"L1 tile declared {shape}: pto's Mat tile asserts rows "
                                   "divisible by the 16-row inner box (board static-assert; "
                                   "cce pads internally)")
        py = self.mp.unique(py_ident(hint or v.name))
        self.tiles[v.name] = {"py": py, "dtype": dtype, "shape": shape, "space": space,
                              "addr": addr_expr, "size": size, "layout": layout,
                              "native_addr": native_addr,
                              **({"scale": self.mp.scale_side[v.name]} if is_scale else {}),
                              **({"order_flip": True} if flip else {})}
        # no pad: a Mat tile's pad promises nothing and CORRUPTS its loads (D-123), and the
        # clear it once stood in for is now the real fill of D-124
        tt = self.mp.tiletype(dtype, shape, space, layout)
        self.tile_decl(op, f"{py} = {self.make_tile(op, tt, addr_expr, size, shape, dtype, space)}",
                       name=py, type_expr=tt, bank=space, addresses=(native_addr,),
                       size=size, shape=shape, dtype=dtype)

    def _flat_addr(self, op: Op, v: Value) -> int:
        """The byte address a UB tile or window starts at (compile-time)."""
        rec = self.tiles.get(v.name)
        if rec is None:
            raise PyptoGap(op, f"{v.name} is not a tile the printer knows")
        base = rec.get("addr")
        off = 0
        if "py" not in rec:
            root = rec.get("root") or self.tiles.get(rec.get("group_of") or "") or {}
            if not isinstance(root, dict):
                root = self.tiles.get(str(root)) or {}
            base = rec.get("addr") if rec.get("addr") is not None else root.get("addr")
            cols = list(rec["base_shape"])[1]
            r0, c0 = (self._off_fold(x) for x in rec["offs_ir"])
            if r0 is None or c0 is None or not isinstance(cols, int):
                raise PyptoGap(op, f"the address of {v.name} is not compile-time")
            off = (r0 * cols + c0) * (_esize_bits(rec["dtype"]) // 8)
        if not isinstance(base, int):
            # tile addresses print as small arithmetic strings ("(1024 + 0)"); anything with a
            # name in it is a runtime address and has no compile-time answer
            txt = str(base)
            if not re.fullmatch(r"[\d\s()+\-*/%]+", txt):
                raise PyptoGap(op, f"the address of {v.name} is a runtime expression ({txt})")
            base = int(eval(txt, {"__builtins__": {}}, {}))  # noqa: S307 - digits and operators only
        return base + off

    def flat_row(self, op: Op, v: Value, numel: int | None = None) -> str:
        """A ONE-ROW alias of a UB tile or window, or of its first ``numel`` elements. pto's
        TMrgsort static-asserts that every tile it touches has Rows == 1 ("TMrgsort: the row of
        Destination and Source tile must be 1"), while our sort records live in [rows, cols] UB
        tensors - contiguous, so the same bytes read as [1, rows*cols] are the same records in
        the same order."""
        rec = self.tiles.get(v.name)
        if rec is None:
            raise PyptoGap(op, f"{v.name} is not a tile the printer knows")
        if "py" in rec:
            base_py, addr, shape = rec["py"], rec.get("addr"), list(rec["shape"])
            off = 0
        else:
            root = rec.get("root") or self.tiles.get(rec.get("group_of") or "") or {}
            if not isinstance(root, dict):
                root = self.tiles.get(str(root)) or {}
            addr = rec.get("addr")
            shape = list(rec["shape"])
            cols = list(rec["base_shape"])[1]
            r0, c0 = (self._off_fold(x) for x in rec["offs_ir"])
            if r0 is None or c0 is None or not isinstance(cols, int):
                raise PyptoGap(op, f"a one-row alias of {v.name} needs a compile-time window origin")
            off = (r0 * cols + c0) * (_esize_bits(rec["dtype"]) // 8)
            base_py = rec.get("base_py")
            addr = addr if addr is not None else (root.get("addr") if isinstance(root, dict) else None)
        if addr is None:
            raise PyptoGap(op, f"a one-row alias of {v.name} needs a static UB address "
                               "(this one rides a rotating slot)")
        total = 1
        for d in shape:
            if not isinstance(d, int):
                raise PyptoGap(op, f"a one-row alias of {v.name} needs a compile-time shape")
            total *= d
        if numel is None:
            numel = total
        elif numel > total:
            raise PyptoGap(op, f"{op.opcode} spans {numel} elements of {v.name}, which holds {total}")
        if numel == total and (len(shape) == 1 or shape[0] == 1) and off == 0 and "py" in rec:
            return base_py
        return self._flat_alias(rec, addr, off, numel)

    def _flat_alias(self, rec: dict, addr: Any, off: int, numel: int) -> str:
        cache = getattr(self, "_flat_py", None)
        if cache is None:
            cache = self._flat_py = {}
        key = (str(addr), off, numel, rec["dtype"])
        py = cache.get(key)
        if py is None:
            esz = _esize_bits(rec["dtype"]) // 8
            tt = self.mp.tiletype(rec["dtype"], [1, numel], "ub", None)
            py = cache[key] = self.mp.unique(py_ident(rec.get("py") or rec.get("base_py") or "ub") + "_row")
            a = str(addr) if not off else f"({addr} + {off})"
            self.tile_decl(self._source_op, f"{py} = {self.make_tile(self._source_op, tt, a, numel * esz, [1, numel], rec['dtype'], 'ub')}",
                           name=py, type_expr=tt, bank="ub",
                           addresses=((rec["native_addr"] + off) if rec.get("native_addr") is not None else None,),
                           size=numel * esz, shape=[1, numel], dtype=rec["dtype"])
        return py

    # pl's `expands` gates on the destination's DTYPE alone (block_ops.py _EXPANDS_DTYPES),
    # never on its memory space, so a Mat tile reaches TEXPANDS's cbuf form; these are the
    # element types with an instantiation, and the rest fill through a same-width integer view.
    _EXPANDS_DT = frozenset({"i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64",
                             "f16", "f32", "bf16"})
    _FILL_AS = {1: "u8", 2: "u16", 4: "u32"}

    def fill_l1(self, op: Op) -> None:
        """``set_constant_to_l1`` IS ``pl.expands`` on a Mat tile - pto's ``TEXPANDS`` reaches
        ``pto_create_cbuf_matrix`` there, the SAME builtin cce's ``create_cbuf_matrix`` reaches
        (D-124, correcting D-123: the fill exists, TFillPad was simply the wrong door).

        cce passes the block count; pto takes it from the tile's own capacity
        (``repeatTimes = Rows * Cols * sizeof(T) / 32``), so the two describe the same transfer
        exactly when the fill covers the WHOLE tile - which every corpus op does, each zeroing
        an L1 operand a partial matmul is about to write into. Anything else refuses by name
        rather than filling a different number of blocks.

        The float8 dtypes have no TEXPANDS instantiation, so such a tile fills through a
        same-width INTEGER view. That reinterprets the TILE, not the VALUE: it is taken only
        when the bit pattern is what is being written, which here means zero.
        """
        if self.side != "cube":
            raise PyptoGap(op, "create_cbuf_matrix is a cube-side instruction (cce guards its "
                               "wrapper with ASCEND_IS_AIC); this op is on the vector side")
        t = self.tile(op, op.operands[0])
        if t["space"] != "l1":
            raise PyptoGap(op, f"{op.operands[0].name} is in {t['space'].upper()}; the Mat form of "
                               "TEXPANDS fills an L1 tile")
        size, nb = t.get("size"), self.env.fold(op.attrs.get("n_blocks"))
        if not isinstance(size, int) or not isinstance(nb, int):
            raise PyptoGap(op, f"set_constant_to_l1 of {op.operands[0].name}: the tile size "
                               f"({size!r}) and the block count ({nb!r}) must both fold - pto "
                               "derives the repeat count from the tile's own capacity")
        if nb * 32 != size or size % 32:
            raise PyptoGap(op, f"set_constant_to_l1 clears {nb * 32} of {size} bytes: TEXPANDS "
                               "takes its repeat count from the tile's own capacity "
                               "(TExpandS.hpp:186) and has no way to say a partial one")
        if not 1 <= size // 32 <= 32767:
            raise PyptoGap(op, f"{size // 32} blocks is outside TEXPANDS's repeat range [1, 32767]")
        val = op.attrs.get("val", 0)
        k = val.value if isinstance(val, Literal) else val
        if not isinstance(k, (int, float)) or isinstance(k, bool):
            k = self.env.fold(val)
        if not isinstance(k, (int, float)):
            raise PyptoGap(op, f"set_constant_to_l1 with a runtime value {val!r}: pl.expands takes "
                               "an immediate")
        dtype, py = t["dtype"], t["py"]
        if dtype not in self._EXPANDS_DT:
            if k != 0:
                raise PyptoGap(op, f"TEXPANDS has no instantiation for {dtype} and the fill value "
                                   f"{k!r} is not a known zero, so a same-width integer view would "
                                   "have to reinterpret it rather than carry its bits")
            esz = _esize_bits(dtype)
            elem = self._FILL_AS.get(esz // 8) if esz % 8 == 0 else None
            if elem is None:
                raise PyptoGap(op, f"{dtype} is {esz} bits wide; TEXPANDS instantiates for 1-, 2- "
                                   "and 4-byte types only")
            shape = list(t["shape"])
            if t.get("addr") is None or not all(isinstance(x, int) for x in shape):
                raise PyptoGap(op, f"the {dtype} tile {op.operands[0].name} needs an integer view "
                                   "to fill, and its address or shape does not fold")
            py = self.mp.unique(t["py"] + "_fill")
            tt = self.mp.tiletype(elem, shape, "l1", t.get("layout"))
            self.tile_decl(op, f"{py} = {self.make_tile(op, tt, t['addr'], size, shape, elem, 'l1')}",
                           name=py, type_expr=tt, bank="l1", addresses=(t.get("native_addr"),),
                           size=size, shape=shape, dtype=elem)
        elif not float(k).is_integer() and not _imm_survives(float(k), dtype):
            # the six-decimal rendering of D-121, on a stop no register op can reach: a Mat
            # tile has no bit_cast carrier, so an inexact fill value refuses instead
            raise PyptoGap(op, f"set_constant_to_l1 with {k!r}: pypto renders a float immediate "
                               "with six decimals and an L1 fill has no bit-pattern carrier "
                               "(the vf side's bit_cast needs a register)")
        # A5-UP-038, fixed upstream in 6a652e733 (2026-09-18): the auto-mutex resolver selects
        # MTE2 for a Mat destination and V for a Vec one before inserting native mutexes, so the
        # Mat (including integer aliases) keeps its IR IDs. An older pl asked the CCE backend for
        # `block.expands` and got V, which is not a pipe a Cube mutex can take.
        self.emit(f"pl.expands({py}, {_lit(k)})")
        # A fill of an L1 tile is NOT ordered against the loads that follow it into the same
        # tile, and autosync cannot know: it models this op on MTE2, which is what cce's
        # create_cbuf_matrix is, so it emits no event between the fill and the load that
        # overwrites part of the same slot. pto reaches a different builtin
        # (pto_create_cbuf_matrix, npu/a5/TExpandS.hpp:67) and on silicon the two reorder.
        self.emit("pl.system.bar_mte2()  # the fill and the loads into the same L1 slot reorder")

    def tile(self, op: Op, v: Value) -> dict[str, Any]:
        t = self.tiles.get(v.name)
        if t is None:
            raise PyptoGap(op, f"{v.name} is not a tile the printer knows")
        if "py" not in t:
            raise PyptoGap(op, f"{v.name} is a sliced tile window; {op.opcode} has no window form "
                               "(pl.move reads through an offset, pl.matmul needs a materialised strip)")
        return t

    # -------- sliced windows (mem.slice of an on-chip tile prints nothing: the geometry
    # rides to the consumer as a window record; cce emit.py:960 fixes the convention that
    # dma attrs REPEAT the view's own offsets, so the record is the single source)

    def _off_fold(self, terms: list) -> int | None:
        total = 0
        for t in terms:
            k = self.env.fold(t)
            if k is None:
                return None
            total += k
        return total

    def _off_expr(self, op: Op, terms: list) -> str:
        k = self._off_fold(terms)
        if k is not None:
            return str(k)
        return paren(" + ".join(self.iexpr(op, t) for t in terms))

    def _off_affine(self, terms: list) -> tuple[int, int, int] | None:
        out = (0, 0, 1)
        for t in terms:
            a = self.env.faffine(t)
            if a is None:
                return None
            if out[2] > 1 and a[2] > 1:
                return None  # two progressions: correlation untracked
            out = (out[0] + a[0], out[1] + a[1], max(out[2], a[2]))
        return out

    def _vs_wrap(self, rec: dict) -> tuple[list[str], list[str]]:
        """set_validshape lines narrowing a window's base tile around one consumer, and the
        restoring lines: the base's declared shape stays the loop-body invariant (descriptor-only
        op - no instruction cost)."""
        if not rec.get("base_py") or list(rec["shape"]) == list(rec["base_shape"]):
            return [], []
        win = "[" + ", ".join(str(x) for x in rec["shape"]) + "]"  # ints or bare expressions
        return ([f"pl.set_validshape({rec['base_py']}, {win})"],
                [f"pl.set_validshape({rec['base_py']}, {list(rec['base_shape'])})"])

    def _fix_ub_subview(self, op: Op, name: str, rec: dict) -> dict:
        """Keep FIX's parent UB row pitch in a native, nonallocating Tile subview."""
        rows, cols = rec["base_shape"]
        shape = rec["shape"]
        if not all(isinstance(x, int) for x in [rows, cols, *shape]):
            raise PyptoGap(op, "FIX into a pitched UB window requires static physical and valid shapes", owner="ours")
        esz = _esize_bits(rec["dtype"]) // 8
        affine = [self._off_affine(terms) for terms in rec["offs_ir"]]
        if any(value is None for value in affine):
            raise PyptoGap(op, "FIX UB window offsets do not have a proven aligned range", owner="ours")
        r, c = affine
        origin = (r[0] * cols + c[0]) * esz
        if (origin % 32 or cols * esz % 32
                or (r[2] > 1 and r[1] * cols * esz % 32)
                or (c[2] > 1 and c[1] * esz % 32)):
            raise PyptoGap(op, "FIX UB subview requires a 32-byte-aligned origin and row pitch", owner="ours")
        for (start, step, count), extent, capacity in zip(affine, shape, (rows, cols)):
            low, high = min(start, start + step * (count - 1)), max(start, start + step * (count - 1))
            if low < 0 or high + extent > capacity:
                raise PyptoGap(op, "FIX UB subview exceeds its parent slot", owner="ours")
        if self.env.fold(op.attrs.get("N_dst", cols)) != cols:
            raise PyptoGap(op, "FIX UB subview descriptor N_dst differs from the parent row pitch", owner="ours")
        ro, co = (self._off_expr(op, terms) for terms in rec["offs_ir"])
        py = self.mp.unique(py_ident(name) + "_fix_view")
        self.emit(f"{py} = {rec['base_py']}[{ro}:({ro}) + {shape[0]}, {co}:({co}) + {shape[1]}]")
        # Upstream subview retains the physical TileType and adjusts its pointer
        # and valid shape. In particular, the right half need not over-declare
        # another full-width allocation at its displaced address.
        return {"py": py, "dtype": rec["dtype"], "shape": list(rec["base_shape"]), "space": "ub"}

    def _alias_group_guard(self, op: Op, parent: str) -> None:
        """Refuse ``sync_mode='auto_mutex'`` at the alias group it cannot synchronise.

        Native mode delegates every credit to PyPTO, which orders accesses PER TILE GROUP. A
        second group over bytes an existing managed group already covers therefore has no
        ordering against the first: the producer writes through one and the consumer reads
        through the other, and nothing waits. D-260 is the measured case - a cube matmul into
        ``l0c_tmp_buf_st`` followed by the FIX copy of ``l0c_tmp_buf_grp``, same 16 KiB of L0C,
        which on hardware read the PREVIOUS launch's accumulator. Manual mode (the default) is
        unaffected: it prints the IR's own mutex ops, so the pair is ordered by id, not by group.
        """
        if self.mp.sync_mode != "auto_mutex":
            return
        if parent in getattr(self, "mutex_allocations", {}) or self.tiles.get(parent, {}).get("mutex_ids"):
            raise PyptoGap(op, f"sync_mode='auto_mutex': this printer needs a second tile group over "
                               f"{parent}, whose bytes a managed group already covers. PyPTO orders "
                               f"accesses per group, so a producer on one and a consumer on the other "
                               f"are not ordered (D-260). Use sync_mode='manual' (the default), which "
                               f"prints the IR's own local mutex operations.", owner="ours")

    def _vf_window(self, rec: dict) -> dict | None:
        """The base tile a vector function may take in place of a window, or ``None``.

        A ``@pl.vector_function`` parameter is a bare name. Every address the printed body
        forms is ``base + offset``, and each term of that offset is multiplied out from the
        CALLEE's own declared parameter type - its row pitch - never from the tile object the
        caller hands over (`VfPrinter.tile_ref` / `blk_addr`). A window reaches the callee as
        nothing but a pointer, and when its origin is the base tile's origin and it spans the
        base's full width, that pointer is the base tile's pointer.

        The row count is the difference, and the callee does not read it: a vf walks the rows
        its own scalar parameter names (`for r in range(rows)`), which the kernel passes
        alongside. So the window is fully described by what already crosses the call.

        Only a window `_strip_tile` cannot materialise takes this door, so every window that
        already emitted keeps its strip and its bytes. A runtime extent is exactly the case
        `_strip_tile` refuses - `pl.make_tile` takes a compile-time shape - and the refusal was
        right about the tile and wrong about the call.
        """
        shape, base_shape = rec.get("shape") or [], rec.get("base_shape") or []
        if len(shape) != 2 or len(base_shape) != 2 or not rec.get("base_py"):
            return None
        if all(isinstance(x, int) for x in shape):
            return None  # a static window still materialises, byte for byte as before
        if rec["space"] != "ub" or rec.get("nz") or rec.get("order_flip"):
            return None  # NZ and flipped windows are not a prefix of their base's address order
        if shape[1] != base_shape[1]:
            return None  # a narrower window would let the callee's pitch walk outside it
        if any(self._off_fold(terms) != 0 for terms in rec["offs_ir"]):
            return None  # a displaced window is a different pointer
        return {"py": rec["base_py"], "dtype": rec["dtype"], "shape": list(base_shape),
                "space": rec["space"]}

    def _strip_tile(self, op: Op, name: str, rec: dict) -> dict:
        """Materialise a window record as a real tile at its byte offset. Only windows their
        space's layout keeps contiguous qualify: UB full-width rows / a single row (ND), L0C
        full-height column strips and L0B complete-N K strips / single-K-fractal N strips.
        A dynamically selected slot
        flattens into a slots x strips tile group selected by ``index*strips + strip``."""
        space, shape, dtype = rec["space"], rec["shape"], rec["dtype"]
        if not all(isinstance(x, int) for x in shape):
            raise PyptoGap(op, f"mem.slice of {name}: a runtime-sized window cannot materialise "
                               "as a tile (only set_validshape consumers take it)")
        bits = _esize_bits(dtype)
        if space == "l0b" and bits < 8:
            raise PyptoGap(op, "a packed sub-byte L0B window has no materialised alias", owner="ours")
        esz = bits // 8
        rows, cols = rec["base_shape"]
        ro_t, co_t = rec["offs_ir"]
        k_ro, k_co = self._off_fold(ro_t), self._off_fold(co_t)
        if space == "ub":
            aff_r, aff_c = self._off_affine(ro_t), self._off_affine(co_t)
            if aff_r is None or aff_c is None:
                raise PyptoGap(op, f"mem.slice of {name}: the offsets do not enumerate statically "
                                   "(one affine loop variable at most)")
            if aff_r[2] > 1 and aff_c[2] > 1:
                raise PyptoGap(op, f"mem.slice of {name}: both offsets of the window enumerate; "
                                   "only one of them can pick the strip")
            # a UB tile is ND, so a window is contiguous either when it takes whole rows or when
            # it is a single row. Contiguity alone does not establish the UB
            # port's 32-byte alignment, checked below for every enumerated alias.
            if not ((aff_c[0] == 0 and aff_c[2] == 1 and shape[1] == cols) or shape[0] == 1):
                raise PyptoGap(op, "mem.slice of a UB tile that is not a contiguous window "
                                   "(full-width rows or a single row): the strip would not share the base pitch")
            boff = ((aff_r[0] * cols + aff_c[0]) * esz, (aff_r[1] * cols + aff_c[1]) * esz,
                    max(aff_r[2], aff_c[2]))
            if boff[0] % 32 or (boff[2] > 1 and boff[1] % 32):
                raise PyptoGap(op, f"UB window {name} needs a 32-byte-aligned address: "
                                   f"byte offset {boff[0]}, step {boff[1]}", owner="ours")
            # A short logical row is a valid region inside a physical 32-byte
            # block. Vec<float,1,4> is not a legal PTO tile declaration. The
            # GM consumer's extents narrow this carrier with set_validshape.
            shape = [shape[0], -(-(shape[1] * esz) // 32) * (32 // esz)]
            ord_t, ord_aff = ((co_t, aff_c) if aff_c[2] > 1 else (ro_t, aff_r))
        elif space == "l0a":
            if k_ro != 0 or k_co != 0:
                raise PyptoGap(op, "an L0 load alias requires the slot origin", owner="ours")
            alignment = (16, 32 // esz)
            shape = [-(-extent // align) * align for extent, align in zip(shape, alignment)]
            if any(want > capacity for want, capacity in zip(shape, (rows, cols))):
                raise PyptoGap(op, "an L0 load alias exceeds its slot capacity", owner="ours")
            boff = (0, 0, 1)
            ord_t, ord_aff = ro_t, (0, 0, 1)
        elif space == "l0b":
            # These records already use Right [K, N] coordinates. Logical
            # IR[N, K] offsets/extents are swapped once by mem.slice.
            aff_k, aff_n = self._off_affine(ro_t), self._off_affine(co_t)
            if aff_k is None or aff_n is None:
                raise PyptoGap(op, f"L0B window {name}: offsets do not enumerate statically", owner="ours")
            if aff_k[2] > 1 and aff_n[2] > 1:
                raise PyptoGap(op, f"L0B window {name}: both coordinate axes enumerate", owner="ours")
            c0 = 32 // esz
            if (aff_k[0] % c0 or (aff_k[2] > 1 and aff_k[1] % c0)
                    or aff_n[0] % 16 or (aff_n[2] > 1 and aff_n[1] % 16)):
                raise PyptoGap(op, f"L0B window {name}: origin/step must meet K={c0}, N=16 "
                                   "fractal alignment", owner="ours")
            shape = [-(-shape[0] // c0) * c0, -(-shape[1] // 16) * 16]
            for affine, extent, capacity in zip((aff_k, aff_n), shape, (rows, cols), strict=True):
                first, step, count = affine
                lo = min(first, first + step * (count - 1))
                hi = max(first, first + step * (count - 1)) + extent
                if lo < 0 or hi > capacity:
                    raise PyptoGap(op, f"L0B window {name}: physical extent exceeds its parent "
                                       "slot capacity", owner="ours")
            whole_n = aff_n[0] == 0 and aff_n[2] == 1 and shape[1] == cols
            if not whole_n and shape[0] != c0:
                raise PyptoGap(op, f"L0B window {name}: a partial-N rectangle crossing multiple "
                                   "K fractals cannot preserve its parent pitch in a compact "
                                   "Right alias", owner="ours")
            n_pitch = -(-cols // 16) * 16
            boff = ((aff_k[0] * n_pitch + aff_n[0] * c0) * esz,
                    (aff_k[1] * n_pitch + aff_n[1] * c0) * esz,
                    max(aff_k[2], aff_n[2]))
            ord_t, ord_aff = (ro_t, aff_k) if aff_k[2] > 1 else (co_t, aff_n)
        elif space == "l0c":
            if k_ro != 0 or shape[0] != rows:
                raise PyptoGap(op, f"mem.slice of a L0C tile that is not a full-height column strip "
                                   "(NZ column fractals keep only those contiguous)")
            aff_c = self._off_affine(co_t)
            if aff_c is None:
                raise PyptoGap(op, f"mem.slice of {name}: the column offset does not enumerate statically")
            if aff_c[0] % 16 or (aff_c[2] > 1 and aff_c[1] % 16):
                raise PyptoGap(op, f"L0C column strip at col {aff_c[0]} step {aff_c[1]} width {shape[1]}: "
                                   "the 16-column fractal boundary is not met")
            if shape[1] % 16 and aff_c[2] == 1 and aff_c[0] + -(-shape[1] // 16) * 16 <= cols:
                # a narrow N is a VALID SHAPE in pto, never a tile declaration (the note below):
                # declare the strip at the 16-column fractal it lives in and let the matmul's own
                # set_validshape narrow it back to the columns the kernel asked for. The widening
                # stays inside the parent, so no byte outside this strip's own fractal is named.
                shape = [shape[0], -(-shape[1] // 16) * 16]
            if shape[1] % 16:
                # a narrow N is not a DECLARATION in pto - the fractal width is part of the C++
                # tile type. Board: Tile<Acc, float, 32, 2> and Tile<Right, half, 32, 2> both
                # fail static_assert(Cols % InnerCols == 0) ("Layout cols must be divisible by
                # inner box cols", pto_tile.hpp:1507). It IS a VALID SHAPE: the same probe with
                # 16-wide declarations plus set_validshape(l0b, [32, 2]) / set_validshape(acc,
                # [32, 2]) before pl.matmul computed exactly columns 0..1 (2.9e-06 from the fp32
                # reference) and left the rest of the accumulator untouched. Taking that route
                # means narrowing the WHOLE chain - the L1 Mat window the move reads, the L0B
                # fragment and this strip - not just the destination.
                raise PyptoGap(op, f"L0C column strip of width {shape[1]}: a narrow N is a valid "
                                   "shape in pto, never a tile declaration - a 2-column Acc or "
                                   "Right tile fails static_assert(Cols % InnerCols == 0) "
                                   "('Layout cols must be divisible by inner box cols', "
                                   "pto_tile.hpp:1507, board-verified), while 16-wide "
                                   "declarations narrowed with set_validshape compute exactly "
                                   "those columns; the printer would have to narrow the L1 "
                                   "window, the L0B fragment and this strip together")
            rows16 = -(-rows // 16) * 16
            boff = (aff_c[0] * rows16 * esz, aff_c[1] * rows16 * esz, aff_c[2])
            ord_t, ord_aff = co_t, aff_c
        elif space == "l1" and rows == 1 and shape[0] == 1:
            # a [1, N] bias row is ColMajor-demoted and byte-contiguous: a column window
            # is a plain byte offset (the l1_to_bt split-N source)
            if k_ro not in (0, None):
                raise PyptoGap(op, f"mem.slice of {name}: a row offset on a 1-row L1 tile")
            aff_c = self._off_affine(co_t)
            if aff_c is None:
                raise PyptoGap(op, f"mem.slice of {name}: the column offset does not enumerate statically")
            boff = (aff_c[0] * esz, aff_c[1] * esz, aff_c[2])
            ord_t, ord_aff = co_t, aff_c
        elif (space == "l1" and not rec.get("order_flip") and not rec.get("zn")
              and rec["root"].get("layout") in (None, "pl.NZ")):
            # An NZ full-height column band is contiguous and retains the
            # parent's row/fractal pitch. A partial-height band is not.
            aff_c = self._off_affine(co_t)
            c0 = 32 // esz if esz else 0
            if (k_ro != 0 or shape[0] != rows or rows % 16 or not c0 or aff_c is None
                    or aff_c[0] % c0 or aff_c[1] % c0 or shape[1] % c0):
                raise PyptoGap(op, "L1 alias needs an aligned full-height NZ column band", owner="ours")
            boff = (aff_c[0] * rows * esz, aff_c[1] * rows * esz, aff_c[2])
            ord_t, ord_aff = co_t, aff_c
        else:
            raise PyptoGap(op, f"mem.slice of a {space} tile has no materialised-strip form "
                               f"(a L1 nZ row strip is not contiguous; only pl.move reads it, via offset)")
        b0, bs, n = boff
        size = shape[0] * shape[1] * esz
        root = rec["root"]
        if space in ("ub", "l0b", "l1"):
            capacity = rows * cols * esz
            lo, hi = min(b0, b0 + bs * (n - 1)), max(b0, b0 + bs * (n - 1)) + size
            if lo < 0 or hi > capacity:
                rounding_unit = "32-byte blocks" if space == "ub" else "physical fractals"
                raise PyptoGap(op, f"{space.upper()} window {name} rounded to {rounding_unit} exceeds its "
                                   f"parent allocation ({lo}:{hi} of {capacity} bytes)", owner="ours")
        if root.get("strip"):
            raise PyptoGap(op, f"mem.slice of {name}: a strip of an already materialised strip "
                               "is a later phase")
        if root.get("addr") is not None and n == 1:
            py = self.mp.unique(py_ident(name))
            tt = self.mp.tiletype(dtype, list(shape), space)
            addr = f"({root['addr']} + {b0})" if b0 else root["addr"]
            native_addr = root["native_addr"] + b0 if root.get("native_addr") is not None else None
            self.tile_decl(op, f"{py} = {self.make_tile(op, tt, addr, size, shape, dtype, space)}",
                           name=py, type_expr=tt, bank=space, addresses=(native_addr,),
                           size=size, shape=list(shape), dtype=dtype)
            out = {"py": py, "dtype": dtype, "shape": list(shape), "space": space, "strip": True,
                   "addr": addr, "size": size, "native_addr": native_addr}
        elif root.get("addr") is not None:
            # a loop-varying strip of a statically-addressed tile: the 1-slot case of the
            # group flattening below - one tile group whose n entries are the strip addresses,
            # selected by the strip ordinal
            if n > 256:
                raise PyptoGap(op, f"mem.slice of {name}: {n} strips is past the tile-group cap (256)")
            key = (root.get("py"), dtype, tuple(shape), space, b0, bs, n)
            g2 = self.regroups.get(key)
            if g2 is None:
                self._alias_group_guard(op, root.get("py") or name)
                g2 = self.mp.unique(py_ident(name) + "_st")
                tt = self.mp.tiletype(dtype, list(shape), space)
                addrs = ", ".join(f"({root['addr']} + {b0 + j * bs})" for j in range(n))
                self.tile_decl(op, f"{g2} = pl.make_tile_group(type={tt}, addrs=[{addrs}], depth={n})",
                               name=g2, type_expr=tt, bank=space,
                               addresses=tuple(root["native_addr"] + b0 + j * bs for j in range(n)) if root.get("native_addr") is not None else (),
                               size=size, shape=list(shape), dtype=dtype, group=True, hoisted=True)
                self.regroups[key] = g2
            aff_expr = self._off_expr(op, ord_t)
            first, step = ord_aff[0], ord_aff[1]
            idx = self._ordinal(aff_expr, first, step)
            py = self.mp.unique(py_ident(name))
            self.getitem(py, g2, idx)
            out = {"py": py, "dtype": dtype, "shape": list(shape), "space": space, "strip": True,
                   "addr": None, "size": size, "index_py": idx}
        else:
            alloc = self.tiles[root["group_of"]]
            slots = alloc["slots"]
            if slots * n > 256:
                raise PyptoGap(op, f"mem.slice of {name}: {slots} slots x {n} strips is past the "
                                   "flattened tile-group cap (256)")
            key = (root["group_of"], dtype, tuple(shape), space, b0, bs, n)
            g2 = self.regroups.get(key)
            if g2 is None:
                self._alias_group_guard(op, root["group_of"])
                g2 = self.mp.unique(py_ident(root["group_of"]) + "_st")
                tt = self.mp.tiletype(dtype, list(shape), space)
                base_addr = int(alloc["addr"])
                addrs = ", ".join(str(base_addr + i * alloc.get("pitch", alloc["size"]) + b0 + j * bs)
                                  for i in range(slots) for j in range(n))
                self.tile_decl(op, f"{g2} = pl.make_tile_group(type={tt}, addrs=[{addrs}], depth={slots * n})",
                               name=g2, type_expr=tt, bank=space,
                               addresses=tuple(base_addr + i * alloc.get("pitch", alloc["size"]) + b0 + j * bs for i in range(slots) for j in range(n)),
                               size=size, shape=list(shape), dtype=dtype, group=True, hoisted=True)
                self.regroups[key] = g2
            if n == 1:
                idx = root["index_py"]
            else:
                aff_expr = self._off_expr(op, ord_t)
                # strip ordinal from the running offset expression: (expr - first) // step
                first, step = ord_aff[0], ord_aff[1]
                idx = self._flatten(root["index_py"], n, self._ordinal(aff_expr, first, step))
            py = self.mp.unique(py_ident(name))
            self.getitem(py, g2, idx)
            out = {"py": py, "dtype": dtype, "shape": list(shape), "space": space, "strip": True,
                   "addr": None, "size": size, "group_of": root["group_of"], "index_py": idx}
        if space == "l0b":
            out["order_flip"] = True
        self.tiles[name] = out
        return out

    def _mte1_row_alias(self, op: Op, rec: dict) -> tuple[str, str] | None:
        """Carry an intra-fractal L1 row origin in the address, not TEXTRACT's row/16 field."""
        affine = self._off_affine(rec["offs_ir"][0])
        first, step, count = affine or (0, 1, 16)
        if affine is not None and first % 16 == 0 and (count == 1 or step % 16 == 0):
            return None
        root = rec["root"]
        rotating = root.get("addr") is None
        allocation = self.tiles.get(root.get("group_of")) if rotating else root
        if allocation is None or allocation.get("addr") is None:
            raise PyptoGap(op, "an intra-fractal L1 row origin needs its owning allocation", owner="ours")
        slots = int(allocation.get("slots", 1)) if rotating else 1
        residual = affine is None or slots * count > 256
        if residual:
            first, step, count = 0, 1, 16
        if slots * count > 256:
            raise PyptoGap(op, "L1 row origins exceed PyPTO's 256-entry flattened group capacity", owner="ours")
        shape = list(rec["base_shape"])
        # NZ advances 32 bytes per row inside a column fractal. The alias
        # keeps the original Rows, hence the original column-block pitch.
        tt = self.mp.tiletype(rec["dtype"], shape, "l1", "pl.NZ")
        key = ("mte1_row", root.get("group_of") if rotating else root["py"], tuple(shape), first, step, count)
        group = self.regroups.get(key)
        if group is None:
            group = self.mp.unique(root["py"] + "_rows")
            pitch = int(allocation.get("pitch", allocation["size"]))
            addrs = ", ".join(f"({allocation['addr']} + {slot * pitch + (first + i * step) * 32})"
                              for slot in range(slots) for i in range(count))
            parent = allocation["group_py"] if rotating else allocation["py"]
            base = allocation.get("native_addr")
            addresses = tuple(base + slot * pitch + (first + i * step) * 32
                              for slot in range(slots) for i in range(count)) if base is not None else ()
            self.tile_decl(op, f"{group} = pl.make_tile_group(type={tt}, addrs=[{addrs}], depth={slots * count})",
                           name=group, type_expr=tt, bank="l1", addresses=addresses,
                           size=allocation["size"], shape=shape, dtype=rec["dtype"], group=True, hoisted=True,
                           parent_slots=tuple((parent, slot) for slot in range(slots) for _ in range(count)))
            self.regroups[key] = group
        row = self._off_expr(op, rec["offs_ir"][0])
        index = f"{paren(unparen(row))} % 16" if residual else "0" if count == 1 else self._ordinal(row, first, step)
        if rotating:
            if root.get("index_py") is None:
                raise PyptoGap(op, "a rotating L1 row alias needs its slot selector", owner="ours")
            index = self._flatten(root["index_py"], count, index)
        name = self.mp.unique(root["py"] + "_row")
        self.getitem(name, group, index)
        return name, f"{paren(unparen(row))} - ({paren(unparen(row))} % 16)" if residual else "0"

    def _physical_alias(self, op: Op, rec: dict, shape: list[int], dtype: str,
                        suffix: str, byte_offset: int = 0) -> str:
        """A byte-bounded alias of one fixed or rotating physical slot."""
        size = math.prod(shape) * _esize_bits(dtype) // 8
        if size <= 0 or byte_offset < 0 or byte_offset + size > rec["size"]:
            raise PyptoGap(op, "native alias exceeds its parent slot capacity", owner="ours")
        name = self.mp.unique(rec["py"] + suffix)
        bank = rec["space"]
        tt = self.mp.tiletype(dtype, shape, bank)
        if rec.get("addr") is not None:
            address = rec.get("native_addr")
            if type(address) is not int:
                raise PyptoGap(op, "native alias has no static physical address", owner="ours")
            address += byte_offset
            self.tile_decl(op, f"{name} = {self.make_tile(op, tt, address, size, shape, dtype, bank)}",
                           name=name, type_expr=tt, bank=bank, addresses=(address,),
                           size=size, shape=shape, dtype=dtype)
        else:
            base = self.tiles.get(rec.get("group_of") or "")
            if not base or rec.get("index_py") is None:
                raise PyptoGap(op, "rotating native alias has no physical slot group", owner="ours")
            addresses = tuple(int(base["addr"]) + i * base["pitch"] + byte_offset for i in range(base["slots"]))
            key = ("physical", rec["group_of"], tuple(shape), dtype, byte_offset)
            group = self.regroups.get(key)
            if group is None:
                group = self.mp.unique(name + "_grp")
                self.tile_decl(op, f"{group} = pl.make_tile_group(type={tt}, addrs={list(addresses)}, depth={len(addresses)})",
                               name=group, type_expr=tt, bank=bank, addresses=addresses,
                               size=size, shape=shape, dtype=dtype, group=True, hoisted=True)
                self.regroups[key] = group
            self.getitem(name, group, rec["index_py"])
        return name

    def _nd_unroll(self, op: Op, code: str, ss: list) -> bool:
        """A rank > 2 ND read, emitted as one 2-D `pl.load` per outer index (D-131).

        pl has no rank-3 load: `pl.make_tensor` builds a `TileShape2D` GlobalTensor whatever
        the shape list holds - the extra dims are dropped, board-read out of the generated C++ -
        and `pl.load` then OVERWRITES the last two shape dims from the destination tile
        (`SetShape<DIM_3, DIM_4>(tile.GetValidRow(), tile.GetValidCol())`). pto's own
        `TLoadVecND2ND` does drive five loops, but nothing in pl programs them. The outer loops
        are therefore unrolled here: each iteration is the innermost 2-D window read into its own
        sub-tile of the destination. Returns False when the shape does not qualify, and the
        caller reports the gap."""
        dst, src = op.operands
        sizes = [self.env.fold(x) for x in (op.attrs.get("loop_size") or [])]
        dss = [self.env.fold(x) for x in (op.attrs.get("loop_dst_stride") or [])]
        if code != "dma.gm_to_ub.nd" or len(sizes) != len(ss) or len(dss) != len(ss):
            return False
        if not all(isinstance(x, int) for x in sizes + dss + list(ss)):
            return False
        if ss[0] != 1 or dss[0] != 1:              # the innermost run must be contiguous both sides
            return False
        rows, cols, row_ss, row_ds = sizes[1], sizes[0], ss[1], dss[1]
        rec = self.tiles.get(dst.name)
        if rec is None or "py" not in rec or rec.get("space") != "ub":
            return False
        shape = list(rec["shape"])
        if len(shape) != 2 or not all(isinstance(x, int) for x in shape) or shape[1] != cols:
            return False   # a sub-tile only stays contiguous when it takes whole rows
        if row_ds != cols:
            return False   # the inner window's own rows must already be the tile's rows
        outer = list(zip(sizes[2:], ss[2:], dss[2:]))
        for n, s_st, d_st in outer:
            if n <= 0 or s_st % row_ss or d_st % cols:
                return False   # every outer step has to land on a row of both sides
        esz = _esize_bits(rec["dtype"]) // 8
        base, _ = self.gm(op, src)
        # The view is declared tall enough to hold every unrolled origin: pl BOUNDS-CHECKS the
        # load offset against the declared dim ("load: offsets[0]=6 exceeds tensor dim 0 size 2"),
        # while the window it actually reads comes from the destination tile, so the extra rows
        # cost nothing.
        span = 1
        for n, s_st, _ in outer:
            span += (n - 1) * (s_st // row_ss)
        inner = self.mp.unique(base + "_2d")
        self.emit(f"{inner} = pl.make_tensor({base}, [{span - 1 + rows}, {cols}], [{row_ss}, 1])")
        idx = [0] * len(outer)
        while True:
            s_off = sum(i * s for i, (_, s, _) in zip(idx, outer))
            d_off = sum(i * d for i, (_, _, d) in zip(idx, outer))
            if d_off + rows * cols > shape[0] * cols:
                raise PyptoGap(op, f"{code}: the unrolled window at +{d_off} runs past the "
                                   f"{shape} destination tile")
            sub = (rec["py"] if d_off == 0 and rows == shape[0] else
                   self._physical_alias(op, rec, [rows, cols], rec["dtype"], "_nd", d_off * esz))
            self.emit(f"pl.load({sub}, {inner}, [{s_off // row_ss}, 0])")
            for a in range(len(outer) - 1, -1, -1):   # odometer over the outer indices
                idx[a] += 1
                if idx[a] < outer[a][0]:
                    break
                idx[a] = 0
            else:
                break
        return True

    def gm(self, op: Op, v: Value) -> tuple[str, list[str]]:
        if v.name in self.params:
            t = v.type
            rank = len(t.dims) if isinstance(t, MemType) else 2
            return self.params[v.name], ["0"] * rank
        view = self.views.get(v.name)
        if view is None:
            raise PyptoGap(op, f"GM operand {v.name} is neither a parameter nor a slice of one")
        return view[0], view[1]

    def scalar_address(self, op: Op, value: Value, index: Any) -> tuple[str, str]:
        """Linear element offset into the original GM tensor or Vec tile.

        A one-element window must keep the allocation's row pitch. It is not
        a new narrow tile (whose columns would violate PTO's 32-byte alignment).
        """
        dt = value.type.dtype.name
        if dt not in {"i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "bf16", "f32", "b1"}:
            raise PyptoGap(op, f"pl.getval/setval do not support storage-only dtype {dt}", owner="upstream")
        extra = self.env.ref(index) if isinstance(index, Value) else _lit(index)
        if value.type.space == "ub":
            rec = self.tiles.get(value.name)
            if rec is None:
                raise PyptoGap(op, "scalar UB operand has no tile record", owner="ours")
            if "py" in rec:
                return rec["py"], extra
            if not rec.get("base_py"):
                raise PyptoGap(op, "scalar UB window has no addressable base tile", owner="ours")
            row, col = (self._off_expr(op, terms) for terms in rec["offs_ir"])
            return rec["base_py"], f"(({row}) * {rec['base_shape'][1]} + ({col}) + ({extra}))"
        base, offsets = self.gm(op, value)
        if value.name in self.params:
            return base, extra
        # mem.slice records the source tensor dimensions; offsets address that
        # tensor, not the one-element result shape.
        dims = self.views[value.name][4]
        if len(dims) != len(offsets):
            raise PyptoGap(op, "scalar GM window lacks source dimensions", owner="ours")
        terms = [extra]
        for axis, off in enumerate(offsets):
            if off == "0":
                continue
            stride = " * ".join(self.iexpr(op, d) for d in dims[axis + 1:]) or "1"
            terms.append(f"({off}) * ({stride})")
        return base, "(" + " + ".join(terms) + ")"

    def _mm_narrow(self, op: Op, want: list) -> tuple[list[str], list[str]]:
        """set_validshape lines narrowing each matmul operand to the M/N/K the op asks for, and
        the lines restoring the declared shape after it. Empty when every operand already fills."""
        pre: list[str] = []
        post: list[str] = []
        for tile, dims in want:
            shape = list(tile.get("shape") or ())
            if "py" not in tile or len(shape) != 2 or not all(isinstance(x, int) for x in shape):
                continue
            txt, narrow = [], False
            for want_d, have in zip(dims, shape):
                k = want_d if isinstance(want_d, int) else self.env.fold(want_d)
                if k is not None:
                    if k > have:
                        raise PyptoGap(op, f"cube.mmad asks for {k} of a tile declared {shape}")
                    txt.append(str(k))
                    narrow |= k != have
                    continue
                rng = self.env.frange(want_d)
                if rng is not None and rng[0] == have and rng[1] == have:
                    txt.append(str(have))
                    continue
                e = self.iexpr(op, want_d)
                txt.append(e if rng is not None and rng[1] <= have else f"pl.min({e}, {have})")
                narrow = True
            if narrow:
                pre.append(f"pl.set_validshape({tile['py']}, [{', '.join(txt)}])")
                post.append(f"pl.set_validshape({tile['py']}, {shape})")
        return pre, post

    def _runtime_extents(self, op: Op, view: tuple, tile: dict[str, Any]) -> list[str] | None:
        """The narrowing a TAIL window needs when its extents are runtime Min() expressions.
        None when every extent provably fills the tile (the common full-tile load); otherwise the
        expression list the caller hands to set_validshape. Without this a tail iteration reads a
        whole tile past the end of the tensor."""
        dyn = view[3] if len(view) > 3 else None
        shape = list(tile.get("shape") or ())
        if not dyn or len(dyn) != len(shape) or not all(isinstance(t, int) for t in shape):
            return None
        if tile.get("order_flip"):
            dyn, shape = dyn[::-1], shape[::-1]
        out, partial = [], False
        for d, t in zip(dyn, shape):
            if isinstance(d, int):
                if d > t:
                    return None  # not a narrowing: let the static path report it
                out.append(str(d))
                partial |= d != t
                continue
            txt, rng = d
            if rng is not None and rng[0] == t and rng[1] == t:
                out.append(str(t))
                continue
            # the interval analysis loses the correlation between a tail extent and its own
            # loop variable ((m0 + valid_m) - m0 bounds as [-28, 128] for a 64-row tile), so
            # the extent is CLAMPED rather than vetoed - which is also what makes the
            # narrowing safe if a kernel ever asks for more than the tile holds
            out.append(txt if rng is not None and rng[1] <= t else f"pl.min({txt}, {t})")
            partial = True
        return out if partial else None

    def column_view(self, op: Op, v: Value, tile: dict[str, Any]) -> tuple[str, list[str]] | None:
        """The GM tensor re-described so that a CONTIGUOUS RUN feeds a COLUMN tile, or None.

        ``g[b, h, c, r0:r0 + N]`` is N elements along the tensor's innermost axis, while the UB
        window is ``[N, 1]`` - one element per row, at the base tile's pitch (the AscendC
        ``n_burst=N, burst_len=1 element`` copy). ``pl.load`` maps the tile's dimensions to the
        tensor's LAST TWO axes, so the run has to BECOME the row axis: appending a unit axis to
        the tensor does exactly that and moves no byte - a trailing 1 multiplies no stride, so
        every offset term stays where it was and the row stride becomes 1.

        The obvious alternative, naming the axes in reverse (``order=[rank-1, rank-2]``), is the
        DN form and has no Vec destination: the board answers with pto's own static_assert,
        "Src and dst layout must be same!" (``tload_common.hpp:344``), because a
        ``Tile<TileType::Vec, ..., BLayout::RowMajor>`` only takes a ``Layout::ND`` tensor.
        Both spellings of the unit axis - keeping the parameter's axes and a flat ``[total, 1]``
        view - were board-verified bit-exact on the shared pypto box.
        """
        shape = list(tile.get("shape") or [])
        if len(shape) != 2 or shape[1] != 1 or not isinstance(shape[0], int) or shape[0] < 2:
            return None
        view = self.views.get(v.name)
        if view is None or view[2] is None or len(view) < 5:
            return None
        base, offsets, ext, dims = view[0], list(view[1]), list(view[2]), list(view[4])
        # exactly one non-unit extent, and it must be the innermost one (that is what makes
        # the run contiguous); anything else is a strided gather, not this shape.
        if not ext or ext[-1] != shape[0] or any(e != 1 for e in ext[:-1]):
            return None
        if len(dims) != len(offsets) or not dims:
            return None
        try:
            sdims = [self.iexpr(op, dim_scalar(d)) for d in dims] + ["1"]
        except (TypeError, PyptoGap):
            return None
        strides: list[str] = []
        acc = "1"
        for d in reversed(sdims):
            strides.insert(0, acc)
            acc = d if acc == "1" else f"({d} * {acc})"
        name = self.mp.unique(base + "_col")
        self.emit(f"{name} = pl.make_tensor({base}, [{', '.join(sdims)}], [{', '.join(strides)}])")
        return name, offsets + ["0"]

    def check_extents(self, op: Op, v: Value, tile: dict[str, Any], what: str,
                      axes: list[int] | None = None) -> list[int] | None:
        """None when the GM window matches the tile; the smaller window when the transfer is
        partial (the caller narrows the tile with set_validshape around the op)."""
        view = self.views.get(v.name)
        if view is None:
            return None
        if axes is not None:
            view = (view[0], view[1], None if view[2] is None else [view[2][i] for i in axes],
                    [view[3][i] for i in axes], [view[4][i] for i in axes])
        if view[2] is None:
            return self._runtime_extents(op, view, tile)
        ext = list(view[2])
        if tile.get("scale"):
            # an MX scale block is raw bytes, not a matrix: cce spells the same 32-byte
            # burst as [1, 32] on the GM side and [rows, k_groups] on the L1 side, and the
            # transfer is a flat copy. Equal byte counts is the whole check it needs.
            esz0 = _esize_bits(tile["dtype"]) // 8
            have = 1
            for e in ext:
                have *= int(e)
            want = 1
            for t in tile["shape"]:
                if not isinstance(t, int):
                    return None
                want *= t
            if have == want:
                return None
        while len(ext) > len(tile["shape"]) and ext[0] == 1:
            ext = ext[1:]
        if len(ext) < len(tile["shape"]):
            ext = [1] * (len(tile["shape"]) - len(ext)) + ext  # a 1-D window of a [1, N] tile
        if tile.get("order_flip"):
            ext = ext[::-1]  # the tile lives in order=[1, 0] coordinates
        if ext == list(tile["shape"]):
            return None
        if len(ext) == len(tile["shape"]) and all(e <= t for e, t in zip(ext, tile["shape"])):
            return ext
        raise PyptoGap(op, f"{what} extents {view[2]} do not fit tile shape {tile['shape']}")

    # -------- render

    _TRIVIAL = (0, 0.0, False, None, "", "norm")

    def guard_attrs(self, op: Op, known: set[str]) -> None:
        """Refuse ops carrying semantics the printer does not map (never silently drop)."""
        for k, v in op.attrs.items():
            if k in known or k in ("origin", "loc", "name"):
                continue
            if isinstance(v, Literal):
                v = v.value
            if v in self._TRIVIAL:
                continue
            if isinstance(v, str) and v in ("norm",):
                continue
            note = _ATTR_ABSENT_UPSTREAM.get(k)
            raise PyptoGap(op, f"{op.opcode} attr {k}={v!r} has no pypto mapping (refusing to "
                               f"drop it)" + (f" - {note}" if note else ""),
                           owner="upstream" if note else "unmapped")

    def _atomic_region_guard(self, op: Op) -> None:
        """Refuse a GM write inside an atomic region that pl cannot make accumulate.

        The region arms an SPR, so this is not about the op we came for - it is about every
        OTHER way a store can reach GM inside it (a scalar `SetValueTo`, a SIMT store, the
        unmapped NZ2NZ fixpipe). Registry-driven rather than a hand-kept list, so a GM
        destination added later cannot slip past silently.
        """
        from ...ir.ops._dsl import REGISTRY
        spec = REGISTRY.find(op.opcode)
        writes = ({i for i, o in enumerate(spec.operands) if o.access == "write"}
                  if spec is not None else set(range(len(op.operands))))
        for i, operand in enumerate(op.operands):
            t = getattr(operand, "type", None)
            if i in writes and isinstance(t, MemType) and t.space in ("gm", "gmlist"):
                raise PyptoGap(op, f"{op.opcode} writes GM inside `with atomic_{self.atomic}()` "
                                   "and pl carries the mode on pl.store alone, so it would run "
                                   "plain here while cce's armed SPR accumulates it. Either "
                                   "this store belongs outside the region, or it needs a form "
                                   "that takes the mode", owner="ours")

    def _atomic_kw(self, op: Op, dst: Value) -> str:
        """`pl.store(atomic=)` for a GM store, checked against the enclosing atomic region.

        pl accumulates in the DESTINATION TENSOR's dtype and takes no dtype of its own -
        which is what cce picks too (`self._dtype_of(dst)` in both of its wrappers), so a
        legacy `atomic.set_type` is a consistency check here, never a parameter.
        """
        kind = op.attrs.get("atomic")
        kind = str(kind) if kind is not None else "none"
        if self.atomic is not None and kind == "none":
            raise PyptoGap(op, f"{op.opcode} inside `with atomic_{self.atomic}()` carries no "
                               "atomic attr: pl has no atomic MODE to stand in for it, so the "
                               "accumulate would be dropped", owner="ours")
        if kind == "none":
            return ""
        if self.atomic is not None and kind != self.atomic:
            raise PyptoGap(op, f"{op.opcode} atomic={kind!r} inside `with atomic_{self.atomic}()`: "
                               "pl takes the mode per store and cannot hold both", owner="ours")
        if kind != "add":
            # the same two-member enum pto_isa already names one layer down: pl declares it
            # PYPTO_DECLARE_ENUM(AtomicType, AtomicNone, AtomicAdd) (framework/include/ir/
            # op_attr_types.h:129) and the pybind block exposes exactly those two
            # (python/src/bindings/ir/ir.cpp:528). cce reaches max / min through
            # SetAtomicMax / SetAtomicMin; there is no pl spelling to route them to.
            raise PyptoGap(op, f"atomic {kind!r} store: pl.AtomicType has AtomicNone and "
                               "AtomicAdd only (op_attr_types.h:129), so the max / min "
                               "accumulate cce prints has no pl spelling", owner="upstream")
        dt = getattr(getattr(dst.type, "dtype", None), "name", None)
        if self.atomic_dtype is not None and dt is not None and self.atomic_dtype != dt:
            raise PyptoGap(op, f"atomic accumulate declared {self.atomic_dtype} into a {dt} "
                               "destination: pl.store has no dtype of its own and accumulates "
                               "in the destination tensor's", owner="upstream")
        return ", atomic=pl.AtomicType.AtomicAdd"

    def render(self) -> list[str]:
        for p in self.fn.params:
            if isinstance(p.type, MemType) and p.type.space == "gm":
                self.params[p.name] = py_ident(p.name)
            elif isinstance(p.type, ScalarType):
                self.env.bind_param(p)
        # matmul operand coordinates (variant C, board-proven; D-088). pto's TMOV moves
        # equal shapes only and TEXTRACT adds an offset but never a transpose (master codegen;
        # the dst==src[::-1] contract in its python layer references tests that no longer
        # exist), so every transpose must happen at the GM load (order=[1, 0]) - there is no
        # mte1-transpose spelling. Our cube.mmad stores B as [N, K]; pypto wants Right [K, N]:
        #   implicit B (no .T): the whole chain lives in order=[1, 0] coordinates - the GM
        #     load transposes, L1 tiles and their windows flip (offsets/extents swap);
        #   rhs.T: the data is already [K, N] - untransposed load, straight moves;
        #   lhs.T: the [K, M] data loads with order=[1, 0] so L1 holds [M, K];
        #   an L1 tile produced ON CHIP that still needs a transpose has nowhere to turn.
        self.zn_tiles: set[str] = set()  # on-chip lhs-transpose sources: declared [S, K] ZN
        self.zn_alias: set[str] = set()  # move sources read through a [K, N] ZN alias
        self.zn_alias_ops: set[int] = set()  # one allocation can have differently oriented readers
        self.transposed_b: set[str] = set()  # implicit-B chains in order=[1, 0] coordinates
        self.transposed_a: set[str] = set()  # .T lhs chains in order=[1, 0] coordinates
        self.transpose_absorbed: set[str] = set()  # l1_to_l0 srcs whose .T the coordinates absorb
        defs = {r.name: o for o in self.fn.body.walk() for r in o.results}
        def chain_of(v, acc):
            if not isinstance(v, Value) or v.name in acc:
                return acc
            acc.add(v.name)
            d = defs.get(v.name)
            if d is not None and d.opcode in ("mem.get_buf", "mem.reinterpret", "mem.slice"):
                chain_of(d.operands[0], acc)
            return acc
        gm_loaded: set[str] = set()
        for o in self.fn.body.walk():
            if o.opcode.startswith("dma.gm_to_l1") and isinstance(o.operands[0], Value):
                gm_loaded |= chain_of(o.operands[0], set())
        rhs_t_moves: list[Op] = []
        for o in self.fn.body.walk():
            if o.opcode not in ("dma.l1_to_l0", "dma.l1_to_l0.mx"):
                continue  # the mx move feeds the same L0 fragments and needs the same coordinates
            pos = str(o.attrs.get("dst_position", ""))
            ch = chain_of(o.operands[1], set())
            if pos == "l0b":
                if o.attrs.get("src_is_transpose"):
                    self.transpose_absorbed |= ch  # rhs.T: already [K, N], straight everywhere
                    rhs_t_moves.append(o)  # unless the same tile is ALSO a flipped B - below
                elif not (ch & gm_loaded):
                    # an ON-CHIP produced B (the online-cast chains): there is no GM load to
                    # reverse, but the ZN re-reading needs none - the tile holds NZ(B[N, K]),
                    # which IS ZN(B.T[K, N]), so the move reads a [K, N] + ZN alias of the same
                    # bytes and lands in Right (ZN) as an equal-shape, equal-SFractal copy
                    self.zn_alias.add(o.operands[1].name)
                    self.zn_alias_ops.add(o.id)
                elif ch & gm_loaded:
                    # THE TRANSPOSE BELONGS ON MTE1. Keeping the chain's declared [N, K] and
                    # letting the move into Right do the transposing is what the cce backend does
                    # (l1_to_l0<false> on an [N, K] tile): the A5 SFractal defaults are Mat=NZ /
                    # Right=ZN, so the DEFAULT move is already the transposing form
                    # (TExtract.hpp:532 dispatches on SFractal inequality). The alternative -
                    # variant C, which declares the chain [K, N] and reverses the GM load's axis
                    # order (D-087) - moves the transpose onto MTE2, where the load reads GM
                    # column-wise and the pipe pays for it: mla_hif8 measured 0.902 -> 0.978 and
                    # matmul_f32_tailsafe 0.884 -> 1.022 when the ZN route was taken instead, with
                    # aic_mte2's excess going from +26 % / +164 % to -1 % / +22 % (D-154).
                    #
                    # So the ZN route is the default and the flip is the FALLBACK, for a chain
                    # whose declared shape is not a legal NZ Mat tile: pto's inner box wants
                    # cols * esize a whole number of 32-byte columns and rows a whole number of
                    # 16-row boxes. The two shapes are legal under symmetric conditions, and a
                    # chain that fails as [N, K] may still pass as [K, N] (simt_matmul_transpose:
                    # K = 8 rows) - that one keeps variant C.
                    # NZ(B[N, K]) and ZN(B.T[K, N]) are the SAME BYTES - both put element
                    # (n, k) at n * rowbytes + k * esize inside one fractal - and a ZN Mat tile
                    # measures its fractal the other way round. So the MOVE reads a [K, N] +
                    # layout=ZN ALIAS of the load's own [N, K] storage and lands in Right (ZN as
                    # well) as an equal-shape, equal-SFractal copy. It has to be an alias rather
                    # than the declaration: pl.load has no ZN-destination instantiation
                    # (tload_common.hpp:131 static-assert), so the tile the GM load fills stays NZ.
                    #
                    # This needs no legality test of its own. The tile is the one the GM load
                    # already fills, so it is a legal Mat tile by construction, and the alias is
                    # the same bytes. Variant C - declaring the chain [K, N] and reversing the GM
                    # load's axis order (D-087) - is the one with conditions, because the FLIPPED
                    # shape has to satisfy pto's fractal rules afresh, and it is also the slower
                    # of the two: it puts the transpose on MTE2, where the load reads GM
                    # column-wise. Measured, taking this route instead: mla_hif8 0.902 -> 0.978
                    # and matmul_f32_tailsafe 0.884 -> 1.022, with aic_mte2's excess going from
                    # +26 % / +164 % to -1 % / +22 % (D-154). So the flip is gone from the B path
                    # entirely; a chain that genuinely needs a transposed GM load says so in the
                    # IR (dma.gm_to_l1.dn2nz), which is handled where the load is printed.
                    self.zn_alias.add(o.operands[1].name)
                    self.zn_alias_ops.add(o.id)
            elif pos == "l0a" and o.attrs.get("src_is_transpose") and (ch & gm_loaded):
                # The lhs transpose is the same story as the implicit B above, and takes the same
                # route: keep the chain's declared shape, let the GM load stay straight, and let
                # the MOVE transpose. pto's TEXTRACT dispatches on
                # DstTileData::SFractal != SrcTileData::SFractal and issues the
                # load_cbuf_to_ca(..., transpose=1) form (TExtract.hpp:516) - which is the
                # l1_to_l0<true> the cce backend emits here. An ON-CHIP lhs source reaches that by
                # DECLARING the tile [K, M] + ZN (the branch below); a GM-loaded one cannot,
                # because pl.load has no ZN destination, so it reads a [K, M] + ZN ALIAS of its own
                # [M, K] storage - the same bytes, the same trick as zn_alias uses for B.
                self.zn_alias.add(o.operands[1].name)
                self.zn_alias_ops.add(o.id)
            elif pos == "l0a" and o.attrs.get("src_is_transpose"):
                # The producer writes ordinary NZ storage. Only this reader needs
                # its transposed ZN interpretation: the same on-chip tile may also
                # feed a non-transposed lhs (backward d_score does both).
                # Changing the allocation's layout would transpose that reader too.
                self.zn_alias.add(o.operands[1].name)
                self.zn_alias_ops.add(o.id)
        for o in rhs_t_moves:
            # An `rhs.T` move is "already [K, N]" only while the tile is DECLARED [N, K]. When the
            # SAME L1 tile also feeds a straight B - mla's K and V are one tensor - variant C has
            # already declared it [K, N] for that use, so this one is [N, K] against the
            # declaration and needs the ZN re-reading of the same bytes. Read as a window on the
            # NZ tile instead, pto's transposing extract puts the N offset on the K axis and the
            # product is silently wrong (board: uncorrelated, D-122).
            if chain_of(o.operands[1], set()) & self.transposed_b:
                self.zn_alias.add(o.operands[1].name)
                self.zn_alias_ops.add(o.id)

        # presets fire before any body statement (the WAR credits of iteration 0)
        preset_lines: list[str] = []
        for op in self.fn.body.ops:
            if op.opcode == "sync.event":
                self.op_sync_event(op, preset_lines)
        for ln in preset_lines:
            self.emit(ln)
        self.block(self.fn.body, top=True)
        for ln in self.mutex_drain:
            self.emit(ln)
        if self.native_tiles is not None:
            declarations, replacements, metadata = self.native_tiles.finalize()
            self.mp.native_mutex_map.extend(metadata)
            declared = {d.op.results[0].name for d in self.native_tiles.declarations if d.mutex_ids}
            if declared != self.mutex_allocations.keys():
                raise PyptoGap(None, "IR mutex allocation has no physical Tile declaration", owner="ours")
            for line in self.hoisted + self.lines:
                if isinstance(line, TileSite):
                    replacement = replacements[line.name]
                    if replacement is not None:
                        declarations.append("    " * line.indent + replacement)
                else:
                    declarations.append(line)
            lines = declarations
        else:
            lines = self.hoisted + self.lines
        lines, report = cleanup_scalars(lines, enabled=self.mp.module.attrs.get("scalar_simplify", True))
        self.mp.scalar_cleanup_report[self.side] = report
        return lines

    def block(self, b: Block, top: bool = False) -> None:
        if not top:
            # A loop may update a selector after its first use in the body.
            # An index computed before the loop cannot represent that value
            # on subsequent iterations, even when its declaration dominates.
            self._ix_gen += 1
        n0 = len(self.lines)
        for op in b.ops:
            self.op(op)
        if not top and (len(self.lines) == n0 or all(isinstance(line, TileSite) or line.lstrip().startswith("#") for line in self.lines[n0:])):
            self.emit("pass")

    # -------- ops

    def op(self, op: Op) -> None:
        self._op(op)

    def _op(self, op: Op) -> None:  # noqa: C901 - one printer, one dispatch
        code = op.opcode
        self._source_op = op
        if self.atomic is not None and code not in ATOMIC_MARKERS and code not in ATOMIC_CARRIERS:
            self._atomic_region_guard(op)
        if getattr(self, "_allvec_pending", None) is not None and code != "sync.crosscore.allvec_wait":
            raise PyptoGap(self._allvec_pending, "sync.crosscore.allvec_ready not immediately followed by "
                           "allvec_wait (only the paired barrier form maps to pl.system.sync_all)")
        if code in ("dma.l0c_to_ub", "dma.l0c_to_l1", "dma.l0c_to_gm.nz2nd") \
                and "M_src" in op.attrs and self.env.fold(op.attrs["M_src"]) is None:
            # All mapped FIX destinations share a statically pitched source
            # Tile. Unmapped NZ2DN/NZ2NZ keep their existing refusal reasons.
            raise PyptoGap(op, "dynamic FIX M_src cannot be represented by the static "
                               "PyPTO source Tile pitch; use matching static carriers "
                               "and descriptors in separate branches", owner="ours")
        if code == "sync.event":
            return  # handled in render()
        if code in ("sync.local_mutex_get", "sync.local_mutex_release"):
            if self.mp.sync_mode == "auto_mutex":
                guards = op.attrs.get("guards", ())
                if not guards or any(name not in self.mutex_allocations for name in guards):
                    raise PyptoGap(op, "native local mutex has no IR-owned buffer guard", owner="ours")
            else:
                action = "mutex_lock" if code.endswith("_get") else "mutex_unlock"
                self.emit(f"pl.system.{action}(pipe=pl.PipeType.{op.attrs['pipe']}, mutex_id={self.env.ref(op.attrs['id'])})")
            return
        if code == "vec.mergesort4":
            raise PyptoGap(op, "vec.mergesort4 is unsupported in pypto_pro; its multi-pass "
                               "translation requires temporary storage not allocated in IR", owner="ours")
        if code in ("vec.sort32", "vec.mergesort_2seq"):
            # pto's sort family works on val-idx record TILES, so our repeat/length attrs are
            # carried by the tile shapes rather than by the call.
            if code == "vec.sort32":
                self.guard_attrs(op, {"repeat"})
                dst, src, idx = op.operands[:3]
                self.emit(f"pl.sort32({self.tile(op, dst)['py']}, {self.tile(op, src)['py']}, "
                          f"{self.tile(op, idx)['py']})")
                return
            # two sequences: two adjacent n-record lists are one 4n-word group, and L = n merges
            # its four n-word runs - the halves of both lists, so n must be even (I027). A
            # destination overlapping the span would need a second pass through storage the IR
            # never allocated, so it is refused (RFC-0013 sorting subset). pl.mrgsort2 takes
            # separate sources only in its graph argument order (A5-UP-040), so the one-tile
            # spelling stays.
            self.guard_attrs(op, {"size1", "size2"})
            dst, s0v, s1v = op.operands[:3]
            n1 = self.ifold(op, op.attrs.get("size1", 0), "size1")
            n2 = self.ifold(op, op.attrs.get("size2", 0), "size2")
            if n1 != n2:
                raise PyptoGap(op, f"vec.mergesort_2seq of {n1} + {n2} records: a pl.mrgsort group "
                                   "merges four EQUAL runs, so unequal lists have no one-call spelling")
            if n1 <= 0 or n1 % 2:
                raise PyptoGap(op, f"vec.mergesort_2seq of two {n1}-record lists: the four pl.mrgsort "
                                   "runs are the halves of both lists, and an odd record count splits "
                                   "a record")
            r0 = self.tiles.get(s0v.name)
            a0, a1 = self._flat_addr(op, s0v), self._flat_addr(op, s1v)
            esz = _esize_bits(r0["dtype"]) // 8
            if a1 != a0 + 2 * n1 * esz:
                raise PyptoGap(op, "vec.mergesort_2seq over two runs that are not adjacent in UB: "
                                   "pl.mrgsort merges the adjacent blocks of ONE tile")
            words = 4 * n1
            span, dpy = self._flat_alias(r0, a0, 0, words), self.flat_row(op, dst, words)
            if abs(self._flat_addr(op, dst) - a0) < words * esz:
                raise PyptoGap(op, "vec.mergesort_2seq into a destination overlapping its runs: the "
                                   "second pass would need temporary storage not allocated in IR "
                                   "(RFC-0013 sorting subset)", owner="ours")
            self.emit(f"pl.mrgsort({dpy}, {span}, block_len={n1})")
            return
        if code in ("vec.set_mask", "vec.set_mask_by_count", "vec.reset_mask",
                    "vec.set_mask_count", "vec.set_mask_normal"):
            if code == "vec.set_mask":
                hi = self.ifold(op, op.attrs.get("high", 0), "mask high")
                lo = self.ifold(op, op.attrs.get("low", 0), "mask low")
                self.emit(f"pl.set_vec_mask({hi}, {lo})")
            elif code == "vec.set_mask_by_count":
                # SetVectorMaskByCount is a BIT-mask expansion (tensorutils_cce.h), not the
                # count-mode switch - fold the static count to the two 64-bit halves
                n = self.ifold(op, op.attrs.get("count", 0), "mask count")
                n = max(0, min(128, n))
                lo = (1 << min(n, 64)) - 1
                hi = (1 << max(n - 64, 0)) - 1
                self.emit(f"pl.set_vec_mask({hi}, {lo})")
            elif code == "vec.reset_mask":
                self.emit("pl.reset_mask()")
            elif code == "vec.set_mask_count":
                self.emit("pl.set_mask_count()")
            else:
                self.emit("pl.set_mask_norm()")
            return
        if code in ("core.set_sat_flag", "core.get_sat_flag"):
            bit = SAT_BITS.get(str(op.attrs.get("mode")))
            if bit is None:
                raise PyptoGap(op, f"{code} mode {op.attrs.get('mode')!r}: no CTRL bit assignment")
            if code == "core.set_sat_flag":
                enable = op.attrs.get("enable")
                value = self.env.ref(enable) if isinstance(enable, Value) else str(int(bool(enable)))
                if isinstance(enable, Value) and enable.type.dtype.kind != "bool":
                    value = f"({value} != 0)"  # SetCtrlSpr expects one bit, not an arbitrary nonzero integer.
                self.emit(f"pl.set_ctrl_spr({bit}, {bit}, {value})")
            else:
                n = self.env.define(op.results[0])
                self.emit(f"{n} = pl.get_ctrl_spr({bit}, {bit})")
            return
        if code == "core.clean_dcache":
            clean_dcache(self, op)
            return
        if code.startswith("scalar.") or code.startswith("core.") or code in ("list.count", "list.item_dim"):
            for ln in self.env.scalar_op(op, self):
                self.emit(ln)
                if scalar_cleanup_eligible(op):
                    self.lines[-1] = ScalarLine(self.lines[-1], op)
            return
        if code == "mem.alloc":
            require_static_allocation(op, self.env.fold, PyptoGap)
            t = op.results[0].type
            if isinstance(t, BufType):  # slot buffer: slots materialise at their get_buf
                inner = t.elem
                shape = [self.ifold(op, d, "buffer dim") for d in inner.dims]
                flip = False
                zn = False
                if inner.space == "l0b":
                    shape = [shape[1], shape[0]]  # pto's rhs naming: [K, N]
                    flip = True
                elif inner.space == "l1" and (op.results[0].name in self.transposed_b
                                              or op.results[0].name in self.transposed_a):
                    shape = [shape[1], shape[0]]  # an order=[1, 0] chain: L1 lives transposed
                    flip = True
                elif inner.space == "l1" and op.results[0].name in getattr(self, "zn_tiles", ()):
                    shape = [shape[1], shape[0]]  # the ZN reading of the same bytes
                    zn = True
                numel = 1
                for d in shape:
                    numel *= d
                base = self.ifold(op, op.attrs.get("addr", 0), "buffer addr")
                size = numel * _esize_bits(inner.dtype.name) // 8
                logical_shape = list(shape)
                scale_side = self.mp.scale_side.get(op.results[0].name)
                tile_dtype = inner.dtype.name
                layout = "pl.ZN" if zn else None
                if scale_side:
                    if inner.space != "l1" or size % 32:
                        raise PyptoGap(op, "MX scale slots must be aligned L1 byte blocks")
                    tile_dtype = "e8m0"
                    layout = "pl.ZZ" if scale_side == "scale_l" else "pl.NN"
                    if scale_side == "scale_r":
                        shape = [shape[1], shape[0]]
                # the allocator bumps each SLOT by the space's alignment (addr_alloc.ALIGN),
                # so a tile whose bytes are not a whole number of those units has a PITCH
                # larger than its size - [1, 200] fp16 is 400 bytes and 416 apart. Reading
                # the pitch off the tile put slot 1 at an unaligned address ("Tile address
                # 0x00390 (912) is not 32-byte aligned for memory space Vec" - board).
                _al = _SLOT_ALIGN.get(inner.space, 32)
                shape = self.vec_row_align(inner.space, inner.dtype.name, shape, size)
                rec = {
                    "py": None, "dtype": tile_dtype, "shape": shape, "space": inner.space,
                    "addr": str(base), "size": size, "pitch": -(-size // _al) * _al,
                    "native_addr": base,
                    "slots": t.slots, "group_py": None,
                    **({"order_flip": True} if flip else {}),
                    **({"zn": True} if zn else {}),
                    **({"scale": scale_side, "logical_shape": logical_shape} if scale_side else {}),
                }
                if inner.space in MEMSPACE and inner.dtype.name in PL_DT:
                    # the rotating-group handle: pto's own dynamic-slot mechanism (group[i] is a
                    # runtime GetItemExpr) - declared up front so a loop-carried slot index can use it
                    g = self.mp.unique(py_ident(op.attrs.get("name") or op.results[0].name) + "_grp")
                    pad = None  # see D-123: not on a Mat tile
                    tt = self.mp.tiletype(tile_dtype, shape, inner.space, layout, pad)
                    addrs = ", ".join(str(base + i * rec["pitch"]) for i in range(t.slots))
                    self.tile_decl(op, f"{g} = pl.make_tile_group(type={tt}, addrs=[{addrs}], depth={t.slots})",
                                   name=g, type_expr=tt, bank=inner.space,
                                   addresses=tuple(base + i * rec["pitch"] for i in range(t.slots)),
                                   size=size, shape=shape, dtype=tile_dtype, group=True)
                    rec["group_py"] = g
                self.tiles[op.results[0].name] = rec
                return
            assert isinstance(t, MemType)
            shape = [self.ifold(op, d, "tile dim") for d in t.dims]
            numel = 1
            for d in shape:
                numel *= d
            size = numel * _esize_bits(t.dtype.name) // 8
            addr = self.iexpr(op, op.attrs.get("addr", 0))
            self.deftile(op, op.results[0], space=t.space, dtype=t.dtype.name, shape=shape,
                         addr_expr=addr, size=size, hint=op.attrs.get("name"), native_addr=self.env.fold(op.attrs.get("addr", 0)))
            return
        if code == "mem.get_buf":
            buf, cnt = op.operands
            b = self.tile(op, buf)
            slots = b.get("slots")
            if not slots:
                raise PyptoGap(op, f"mem.get_buf of {buf.name}, which is not a slot buffer")
            k = self.env.fold(cnt)
            # a straight-line counter folds in pypto's own parser (board-proven); a loop-carried
            # index selects through pl.make_tile_group - group[i] is a runtime GetItemExpr, the
            # mechanism pto's own double-buffered FA kernels use (our get_buf is modulo-slots,
            # the group index is used unchanged, so the % prints explicitly)
            if k is None and (self.native_tiles is not None or self._loop_ref(cnt)):
                g = b.get("group_py")
                if g is None:
                    raise PyptoGap(op, f"mem.get_buf of {buf.name}: the slot buffer has no pl tile "
                                       "group (unsupported space or dtype)")
                py = self.mp.unique(py_ident(op.results[0].name))
                if self.ranges.normalized(cnt, slots):
                    idx = self.env.ref(cnt)
                    self.stable_indices.add(idx)
                else:
                    idx = f"{paren(self.env.ref(cnt))} % {slots}"
                self.getitem(py, g, idx)
                self.tiles[op.results[0].name] = {"py": py, "dtype": b["dtype"], "shape": b["shape"],
                                                  "space": b["space"], "addr": None, "size": b["size"],
                                                  "group_of": buf.name, "index_py": idx,
                                                  **({"scale": b["scale"]} if b.get("scale") else {}),
                                                  **({"order_flip": True} if b.get("order_flip") else {})}
                return
            slot = f"({k % slots})" if k is not None else f"({paren(self.env.ref(cnt))} % {slots})"
            addr = f"({b['addr']} + {slot} * {b.get('pitch', b['size'])})"
            self.deftile(op, op.results[0], space=b["space"], dtype=b["dtype"], shape=b.get("logical_shape", b["shape"]),
                         addr_expr=addr, size=b["size"], flip_from=bool(b.get("order_flip")),
                         native_addr=(b["native_addr"] + (k % slots) * b.get("pitch", b["size"])) if k is not None else None)
            return
        if code == "mem.reinterpret":
            src = op.operands[0]
            if src.name in self.params:
                # A GM dtype view (torch has no hif8/fp4, so the public tensor is a byte
                # carrier the kernel re-types). pl.reinterpret is tile-level, but make_tensor
                # takes a dtype= of its own - the same re-description mem.view and
                # mem.workspace already use, with the element type changed instead of the
                # strides. Row-major over the result's own dims, so a size change (b8 -> b4)
                # rides in the shape the IR already computed.
                tt = op.results[0].type
                if not isinstance(tt, MemType):
                    raise PyptoGap(op, f"mem.reinterpret of GM parameter {src.name} into a "
                                       "non-tensor type")
                if tt.dtype.name not in PL_DT:
                    raise PyptoGap(op, f"mem.reinterpret of GM parameter {src.name}: dtype "
                                       f"{tt.dtype.name} has no pypto DT_* constant")
                shape = [self.iexpr(op, dim_scalar(d)) for d in tt.dims]
                strides: list[str] = []
                acc = "1"
                for d in reversed(shape):
                    strides.insert(0, acc)
                    acc = d if acc == "1" else f"({d} * {acc})"
                name = self.mp.unique(py_ident(op.results[0].name))
                self.emit(f"{name} = pl.make_tensor({self.params[src.name]}, "
                          f"[{', '.join(shape)}], [{', '.join(strides)}], dtype={_pl_dt(tt.dtype.name)})")
                self.params[op.results[0].name] = name
                return
            s0 = self.tiles.get(src.name)
            t = op.results[0].type
            assert isinstance(t, MemType)
            lay = op.attrs.get("layout")
            if lay is not None and str(lay) == "nz":
                # the `.nz()` compact spelling: the bytes ARE the NZ fractal stream. The two
                # phase-6 refutations both mis-routed it - a flat copy ignores the row-pitch
                # difference and an ND-declared source sends TINSERT down its flat branch
                # (TInsertVecToMatImpl dispatches on the SOURCE layout). The probe holds the
                # verdict: an NZ-DECLARED Vec alias + pl.insert is bit-exact across equal
                # rows, differing row pitches, and column-block windows. The record passes
                # through untouched (no py: only ub_to_l1.nz may consume it).
                if s0 is None:
                    raise PyptoGap(op, f"mem.reinterpret layout=nz of {src.name}: not a tile "
                                       "the printer knows")
                # This is a marker consumed by the NZ DMA lane, not a Tile
                # declaration. Keep runtime valid extents until that consumer.
                nshape = [self.env.fold(dim_scalar(x)) if self.env.fold(dim_scalar(x)) is not None
                          else self.iexpr(op, dim_scalar(x)) for x in t.dims]
                rec = {k: v for k, v in s0.items() if k != "py"}
                rec.update({"nz": True, "dtype": t.dtype.name, "shape": nshape})
                if "py" in s0:  # bare tile: keep the byte container's geometry for the alias
                    rec["nz_rows"] = s0["shape"][0] if s0.get("shape") else nshape[0]
                elif s0.get("root") is not None:
                    # nz over a window: the address/rotating-group facts live on the
                    # window's base record - surface them for the nz lane
                    broot = s0["root"] if isinstance(s0["root"], dict) else self.tiles.get(s0["root"])
                    if broot:
                        if broot.get("addr") is not None:
                            rec.setdefault("nz_addr", broot["addr"])
                            rec.setdefault("nz_native_addr", broot.get("native_addr"))
                        for kf in ("group_of", "index_py"):
                            if broot.get(kf) is not None:
                                rec.setdefault(kf, broot[kf])
                self.tiles[op.results[0].name] = rec
                return
            shape = []
            for dim in t.dims:
                value = self.env.fold(dim_scalar(dim))
                if value is None and t.space in ("l0a", "l0b") and s0 is not None and "tile" in op.attrs:
                    # A staging view's dynamic valid region is consumed by
                    # move/mmad. Static typed capacity needs a proven bound;
                    # raw byte-carrier axes do not have these element units.
                    # The capacity asks only how LARGE the extent gets, so a one-sided bound
                    # answers it: `min(128, ...)` is capped at 128 even when the other operand
                    # is unbounded, and refusing it was this guard widening, not the contract
                    # narrowing (M10-100, RFC-0013 "static local capacity").
                    bound = self.env.fupper(dim_scalar(dim))
                    if bound is None or bound <= 0:
                        raise PyptoGap(op, "dynamic L0 staging extent has no proven typed capacity; "
                                           "dynamic capacity is unsupported; byte axes cannot bound a typed axis; "
                                           "use a bounded extent or static branches", owner="ours")
                    value = bound
                if value is None:
                    value = self.ifold(op, dim, "reinterpret dim")
                shape.append(value)
            if t.space in ("l0a", "l0b") and "tile" in op.attrs and s0 is not None and t.dtype.bits >= 8:
                # Matrix staging writes complete fractals, including M=1.
                # A one-row Left declaration would make the non-compact
                # TExtract M step zero. The owning raw slot supplies padding.
                c0 = 32 // (_esize_bits(t.dtype.name) // 8)
                physical = [-(-shape[0] // 16) * 16, -(-shape[1] // c0) * c0]
                required = physical[0] * physical[1] * _esize_bits(t.dtype.name) // 8
                held = s0.get("size")
                if s0.get("group_of"):
                    held = self.tiles[s0["group_of"]]["size"]
                if isinstance(held, int) and required <= held:
                    shape = physical
            numel = 1
            for d in shape:
                numel *= d
            size = numel * _esize_bits(t.dtype.name) // 8
            if s0 is not None and s0.get("addr") is None:
                # a dynamically selected slot re-viewed as another dtype/shape (the L0 half-slot
                # idiom): a second tile group over the SAME slot addresses, selected by the same
                # index - the slots are raw bytes, the group is just their typed spelling
                if not s0.get("group_of"):
                    raise PyptoGap(op, f"mem.reinterpret of {src.name}: a slot with no group record")
                b = self.tiles[s0["group_of"]]
                if t.space == "l0b" and len(shape) == 2:
                    shape = [shape[1], shape[0]]  # pto's rhs naming: [K, N] (deftile's rule)
                    if isinstance(shape[1], int) and shape[1] % 16:
                        # a narrow N is a VALID SHAPE, never a DECLARATION (deftile's rule):
                        # declare the fractal the fragment lives in, inside its own slot, and
                        # let the matmul's set_validshape narrow it back
                        wide = -(-shape[1] // 16) * 16
                        w_size = wide * shape[0] * _esize_bits(t.dtype.name) // 8
                        if w_size <= int(b["size"]):
                            shape, size = [shape[0], wide], w_size
                if size > b["size"]:
                    raise PyptoGap(op, f"mem.reinterpret grows {src.name}: {size} > {b['size']} bytes")
                if t.dtype.name not in PL_DT or t.space not in MEMSPACE:
                    raise PyptoGap(op, f"tile dtype {t.dtype.name} / space {t.space} has no pypto spelling")
                key = (s0["group_of"], t.dtype.name, tuple(shape), t.space)
                g2 = self.regroups.get(key)
                if g2 is None:
                    g2 = self.mp.unique(py_ident(s0["group_of"]) + "_re")
                    tt = self.mp.tiletype(t.dtype.name, shape, t.space)
                    base = int(b["addr"])
                    addrs = ", ".join(str(base + i * b.get("pitch", b["size"])) for i in range(b["slots"]))
                    self.tile_decl(op, f"{g2} = pl.make_tile_group(type={tt}, addrs=[{addrs}], depth={b['slots']})",
                                   name=g2, type_expr=tt, bank=t.space,
                                   addresses=tuple(base + i * b.get("pitch", b["size"]) for i in range(b["slots"])),
                                   size=size, shape=shape, dtype=t.dtype.name, group=True, hoisted=True)
                    self.regroups[key] = g2
                py = self.mp.unique(py_ident(op.results[0].name))
                self.getitem(py, g2, s0["index_py"])
                self.tiles[op.results[0].name] = {"py": py, "dtype": t.dtype.name, "shape": shape,
                                                  "space": t.space, "addr": None, "size": size,
                                                  "group_of": s0["group_of"], "index_py": s0["index_py"],
                                                  **({"order_flip": True} if t.space == "l0b" else {})}
                return
            s = self.tile(op, src)
            if size > s["size"]:
                raise PyptoGap(op, f"mem.reinterpret grows {src.name}: {size} > {s['size']} bytes")
            self.deftile(op, op.results[0], space=t.space, dtype=t.dtype.name, shape=shape,
                         addr_expr=s["addr"], size=size, native_addr=s.get("native_addr"))
            return
        if code == "mem.reshape":
            src = op.operands[0]
            if src.name in self.params:
                # a GM parameter reshaped: same bytes, contiguous, new declared shape -
                # the same make_tensor mapping mem.view uses (default row-major strides)
                tt = op.results[0].type
                assert isinstance(tt, MemType)
                shape = [self.ifold(op, d, "reshape dim") for d in tt.dims]
                strides = []
                acc = 1
                for d in reversed(shape):
                    strides.insert(0, acc)
                    acc *= d
                name = py_ident(op.results[0].name)
                self.emit(f"{name} = pl.make_tensor({self.params[src.name]}, {shape}, {strides})")
                self.params[op.results[0].name] = name
                return
            s = self.tile(op, src)
            tt = op.results[0].type
            assert isinstance(tt, MemType)
            shape = [self.ifold(op, d, "reshape dim") for d in tt.dims]
            self.deftile(op, op.results[0], space=s["space"], dtype=s["dtype"], shape=shape,
                         addr_expr=s["addr"], size=s["size"], native_addr=s.get("native_addr"))
            return
        if code == "mem.workspace":
            # a carved GM workspace window: pl.make_tensor over the launcher's u8 workspace
            # parameter, displaced by the byte offset - the same make_tensor mapping mem.view uses
            rt = op.results[0].type
            if not isinstance(rt, MemType):
                raise PyptoGap(op, "a slot-buffer workspace (GMBuff ring) has no pypto spelling yet")
            off = op.attrs.get("offset", 0) or 0
            ws = self.mp.WS_NAME
            ptr = ws if self.env.fold(off) == 0 else f"pl.addptr(pl.make_ptr({ws}), {self.iexpr(op, off)})"
            shape = [self.iexpr(op, dim_scalar(d)) for d in rt.dims]
            strides: list[str] = []
            acc = "1"
            for d in reversed(shape):
                strides.insert(0, acc)
                acc = d if acc == "1" else f"({d} * {acc})"
            try:
                dt = _pl_dt(rt.dtype.name)
            except KeyError:
                raise PyptoGap(op, f"dtype {rt.dtype.name} has no pypto DT_* constant") from None
            name = py_ident(op.results[0].name)
            self.emit(f"{name} = pl.make_tensor({ptr}, [{', '.join(shape)}], [{', '.join(strides)}], dtype={dt})")
            self.params[op.results[0].name] = name
            return
        if code == "mem.view":
            # RFC-0010: the strided GM re-description IS pto's make_tensor - the direct mapping
            # the gap survey asked for. pl.make_tensor(tensor, shape, stride) reuses the source
            # data pointer; an element offset advances a raw pointer first (the documented
            # addptr(make_ptr(x), off) composition, element units).
            src = op.operands[0]
            if src.name not in self.params:
                raise PyptoGap(op, f"mem.view of {src.name}: only a GM parameter view maps to make_tensor")
            shape = [self.iexpr(op, x) for x in op.attrs["shape"]]
            strides = [self.iexpr(op, x) for x in op.attrs["strides"]]
            off_attr = op.attrs.get("offset", 0)
            base = self.params[src.name]
            folded = self.env.fold(off_attr)
            ptr = base if folded == 0 else f"pl.addptr(pl.make_ptr({base}), {self.iexpr(op, off_attr)})"
            rt, st = op.results[0].type, src.type
            assert isinstance(rt, MemType)
            dt_kw = ""
            if isinstance(st, MemType) and rt.dtype != st.dtype:  # a folded same-width reinterpret
                try:
                    dt_kw = f", dtype={_pl_dt(rt.dtype.name)}"
                except KeyError:
                    raise PyptoGap(op, f"dtype {rt.dtype.name} has no pypto DT_* constant") from None
            name = py_ident(op.results[0].name)
            self.emit(f"{name} = pl.make_tensor({ptr}, [{', '.join(shape)}], [{', '.join(strides)}]{dt_kw})")
            self.params[op.results[0].name] = name  # downstream loads/stores treat the view as a GM tensor
            return
        if code == "mem.slice":
            src = op.operands[0]
            if src.name in self.params:
                offsets = [self.iexpr(op, x) for x in op.attrs.get("offsets", [])]
                raw = list(op.attrs.get("extents", []))
                ext = [self.env.fold(x) for x in raw]
                extents = None if any(e is None for e in ext) else [int(e) for e in ext]
                # a tail window's extent is a runtime Min(): keep the EXPRESSION as well, so a
                # partial transfer can still be narrowed with set_validshape (without it the
                # load reads a whole tile past the end of the tensor)
                dyn = [k if k is not None else (self.iexpr(op, x), self.env.frange(x))
                       for k, x in zip(ext, raw)]
                st = src.type
                dims = list(st.dims) if isinstance(st, MemType) else []
                self.views[op.results[0].name] = (self.params[src.name], offsets, extents, dyn, dims)
                return
            base = self.tiles.get(src.name)
            if base is None:
                raise PyptoGap(op, f"mem.slice of {src.name}: neither a GM parameter nor an on-chip tile")
            offs = op.attrs.get("offsets", [])
            ext = op.attrs.get("extents", [])
            if len(offs) != 2 or len(ext) != 2:
                raise PyptoGap(op, f"mem.slice of an on-chip tile with rank {len(offs)} is a later phase")
            shape: list[Any] = []
            for x in ext:  # a dim that does not fold stays symbolic: set_validshape takes it
                k = self.env.fold(x)
                shape.append(k if k is not None else self.iexpr(op, x))
            static_shape = all(isinstance(x, int) for x in shape)
            esz = _esize_bits(base["dtype"]) // 8
            k_ro, k_co = self.env.fold(offs[0]), self.env.fold(offs[1])
            # a Vec tile's ROW must be a whole number of 32-byte blocks (pto_tile.hpp:1509:
            # "BFractal_ is RowMajor and SFractal_ is NoneBox: Rows must be 32 bytes align"),
            # so a narrow window - a single column, one scalar - has no tile of its own; it
            # stays a window record and its consumer narrows the base with set_validshape.
            aligned = static_shape and len(shape) == 2 and (shape[1] * esz) % 32 == 0
            if k_ro is not None and k_co is not None and len(base.get("shape", ())) == 2:
                aligned = aligned and ((k_ro * base["shape"][1] + k_co) * esz) % 32 == 0
            base_shape = base.get("shape", ())
            contiguous = (shape[0] == 1 or (len(base_shape) == 2 and k_co == 0 and shape[1] == base_shape[1]))
            if (static_shape and aligned and contiguous and base["space"] == "ub" and base.get("addr") is not None
                    and "py" in base and k_ro is not None and k_co is not None):
                # the proven static-UB path: a real tile at the window's byte offset. Only a
                # FOLDING offset qualifies - pl.make_tile fixes its address at parse time
                # ("'addr' must be a compile-time integer", board), so a window whose origin is
                # a runtime value (a sub-block id, a loop variable) stays a window record and
                # reaches its consumer through _strip_tile's enumerated tile group instead.
                cols = base["shape"][-1]
                addr = f"({base['addr']} + {(k_ro * cols + k_co) * esz})"
                numel = shape[0] * shape[1]
                self.deftile(op, op.results[0], space="ub", dtype=base["dtype"], shape=shape,
                             addr_expr=addr, size=numel * esz,
                             native_addr=(base["native_addr"] + (k_ro * cols + k_co) * esz) if base.get("native_addr") is not None else None)
                return
            # every other on-chip slice is a window record: nothing prints here; the geometry
            # rides to the consumer, which addresses the base tile through it (dma attrs repeat
            # the same offsets - cce emit.py:960 - so the record is the single source)
            if len(base.get("shape", ())) != 2:
                raise PyptoGap(op, f"mem.slice of {src.name}: only rank-2 base tiles take windows")
            if base.get("order_flip"):
                offs = [offs[1], offs[0]]
                shape = shape[::-1]
            if base.get("nz"):
                # a window of the compact-NZ marker: no base tile narrows - the nz lane
                # aliases the bytes itself (ub_to_l1.nz is the only consumer)
                offs_prev = base.get("offs_ir") or [[], []]
                self.tiles[op.results[0].name] = {
                    "dtype": base["dtype"], "space": base["space"], "shape": shape,
                    "size": (shape[0] * shape[1] * esz) if static_shape else None,
                    "offs_ir": [offs_prev[0] + [offs[0]], offs_prev[1] + [offs[1]]],
                    "nz": True, "nz_addr": base.get("nz_addr", base.get("addr")),
                    "nz_native_addr": base.get("nz_native_addr", base.get("native_addr")),
                    "nz_rows": base.get("nz_rows") or (base["shape"][0] if base.get("shape") else None),
                    **{k: base[k] for k in ("group_of", "index_py") if base.get(k) is not None}}
                return
            if base.get("base_py"):  # a window of a window: stack the offsets on the same root
                offs_ir = [base["offs_ir"][0] + [offs[0]], base["offs_ir"][1] + [offs[1]]]
                root, base_py, base_shape = base["root"], base["base_py"], base["base_shape"]
            else:
                offs_ir = [[offs[0]], [offs[1]]]
                root, base_py, base_shape = base, base["py"], base["shape"]
            rec = {"dtype": base["dtype"], "space": base["space"], "shape": shape,
                   **({"order_flip": True} if base.get("order_flip") else {}),
                   "size": (shape[0] * shape[1] * esz) if static_shape else None, "offs_ir": offs_ir,
                   "base_py": base_py, "base_shape": list(base_shape), "root": root}
            self.tiles[op.results[0].name] = rec
            return
        if code == "dma.gm_to_l1.mx_scale_nd2nz":
            # Dense e8m0 [rows, k_groups] in GM -> the L1 scale fractal. cce spells it with a
            # dedicated MTE2 descriptor; pto's pl.load into a ZZ Mat tile IS that conversion
            # (16-row x 2-group boxes), so the attrs only have to agree with the tile.
            self.guard_attrs(op, {"rows", "k_groups", "src_k_groups"})
            dst, src = op.operands
            t = self.tile(op, dst)
            rows, kg = self.env.fold(op.attrs.get("rows")), self.env.fold(op.attrs.get("k_groups"))
            side = t.get("scale") or self.mp.scale_side.get(dst.name, "scale_l")
            if [rows, kg] != ([t["shape"][1], t["shape"][0]] if side == "scale_r" else list(t["shape"])):
                raise PyptoGap(op, f"mx_scale_nd2nz moves [{rows}, {kg}] into a tile declared "
                                   f"{t['shape']}: pl.load has no partial scale window")
            if kg % 2:
                raise PyptoGap(op, f"mx_scale_nd2nz with an odd k_groups ({kg}): the phase axis "
                                   "pto requires pairs the groups two at a time")
            root = getattr(src, "name", None)
            self.mp.scale_nd_gm[root] = [rows, kg // 2, 2]
            base, offsets = self.gm(op, src)
            if len(offsets) != 2:
                raise PyptoGap(op, f"mx_scale_nd2nz from a rank-{len(offsets)} GM view: the scale "
                                   "load reads a plain [rows, k_groups] block")
            order = "[0, 1]" if side == "scale_l" else "[1, 0]"
            self.emit(f"pl.load({t['py']}, {base}, [{', '.join(offsets)}, 0], order={order})")
            return
        if code in ("dma.gm_to_ub.pad", "dma.gm_to_ub.nd", "dma.gm_to_l1.nd2nz", "dma.gm_to_l1.pad", "dma.gm_to_l1",
                    "dma.gm_to_l1.dn2nz"):
            self.guard_attrs(op, {"n_burst", "burst_len_byte", "src_stride_byte", "dst_stride", "dim",
                                  "M", "N", "M_dst", "N_src", "M_src", "N_dst", "src_stride", "dst_stride_byte",
                                  "loop_src_stride", "loop_dst_stride", "loop_size", "fence",
                                  "loop_left_pad", "loop_right_pad", "constant_value", "pad",
                                  "nearest_value_mode", "config_left_pad", "config_right_pad"})
            for pk in ("pad", "constant_value", "nearest_value_mode", "config_left_pad", "config_right_pad"):
                pv = op.attrs.get(pk)
                if isinstance(pv, Literal):
                    pv = pv.value
                if pv in (None, 0, 0.0, False):
                    continue
                # pl carries a pad MODE, never a pad VALUE. `pl.TilePad` is null / zero / max /
                # min and `pl.TileType(pad=)` takes nothing else ("TileType.pad must be a enum
                # TilePad or compile-time integer 0/1/2/3" for -1.5, "must be one of
                # TilePad.null/zero/max/min" for any other int - both read off the box);
                # `pl.fillpad`'s only knob is NORMAL / EXPAND / INPLACE, which says how the pad
                # region is filled, not with what; and the installed pypto_pro package has zero
                # occurrences of pad_value / padValue / constant_value / fill_value. The C++
                # Tile does carry a runtime pad value (pto_tile.hpp:1332 GetPadValue /
                # SetPadValue), so this is a pl-surface absence, not a hardware one - the same
                # shape as D-104's nz2dn.
                raise PyptoGap(op, f"{code} {pk}={pv!r}: pl carries a pad MODE, not a pad VALUE - "
                                   "pl.TilePad is null/zero/max/min and pl.TileType(pad=) refuses "
                                   "anything else (board: 'TileType.pad must be a enum TilePad or "
                                   "compile-time integer 0/1/2/3'), pl.fillpad only chooses "
                                   "NORMAL/EXPAND/INPLACE, and no pl entry point reaches the C++ "
                                   "Tile's own SetPadValue (pto_tile.hpp:1332)", owner="upstream")
            for pk in ("loop_left_pad", "loop_right_pad"):
                pv = op.attrs.get(pk)
                if pv and any(self.env.fold(x) not in (0, None) or not isinstance(self.env.fold(x), int)
                              or self.env.fold(x) != 0 for x in pv):
                    raise PyptoGap(op, f"{code} {pk}={pv}: NDDMA edge padding has no pl.load spelling")
            lss = op.attrs.get("loop_src_stride")
            if lss is not None:
                ss = [self.env.fold(x) for x in lss]
                if len(ss) >= 2 and ss[0] != 1:
                    # THE UPSTREAM HALF (D-130). a5/TLoad.hpp's TLoadVecND2ND sets
                    # lenBurst = validCol * sizeof(T): the innermost dimension IS a contiguous
                    # burst and pto has no innermost element stride to give it, while cce's
                    # gm_to_ub_nd takes one per loop (NdLoops<5>.loop_src_stride). Nothing to map.
                    raise PyptoGap(op, f"{code} loop_src_stride={ss}: a non-unit INNERMOST stride "
                                       f"({ss[0]}) has no pto spelling - TLoadVecND2ND's inner "
                                       "dimension is a contiguous burst (lenBurst = validCol * "
                                       "sizeof(T)) and there is no per-element stride beside it, "
                                       "where cce's gm_to_ub_nd takes one stride per loop "
                                       "(NdLoops<5>). Board-probed too: the gather form hits pto's "
                                       "Vec ND tile 32B-row assert", owner="upstream")
                if len(ss) > 2 and self._nd_unroll(op, code, ss):
                    return
                if len(ss) != 2:
                    # board-probed: a non-unit innermost stride hits pto's Vec ND tile template
                    # 32B-row assert (gather [4,12]); rank-3 compiled but returned 15 elements
                    # off - only the plain 2D window (contiguous inner rows) round-trips.
                    #
                    # D-130 read the header and the two cases are NOT the same:
                    #  * NON-UNIT INNERMOST STRIDE is an upstream shape limit. a5/TLoad.hpp's
                    #    TLoadVecND2ND sets lenBurst = validCol * sizeof(T) - the inner dimension
                    #    IS a contiguous burst and there is no innermost element stride to give.
                    #    cce's gm_to_ub_nd takes one per loop (NdLoops<5>.loop_src_stride).
                    #  * RANK > 2 with a unit innermost stride IS expressible: the same function
                    #    programs set_loop2_stride_outtoub / set_loop1_stride_outtoub /
                    #    set_loop_size_outtoub from gShape1/2 and gStride1/2 and loops gShape0 by
                    #    hand - five dimensions, exactly cce's shape - and pl feeds them through
                    #    pl.make_tensor(ptr, shape, stride, dtype). The board's "15 elements off"
                    #    was OUR mapping, not a missing capability; retrying it is an open item.
                    raise PyptoGap(op, f"{code} loop_src_stride={ss}: rank {len(ss)} with a unit "
                                       "innermost stride IS expressible and this printer does not "
                                       "build it yet - a5/TLoad.hpp's TLoadVecND2ND programs "
                                       "set_loop2_stride_outtoub / set_loop1_stride_outtoub / "
                                       "set_loop_size_outtoub from gShape1/2 and gStride1/2 and "
                                       "loops gShape0 by hand, five dimensions in all, and pl feeds "
                                       "them through pl.make_tensor(ptr, shape, stride, dtype). Our "
                                       "one attempt came back 15 elements off on silicon (the axis "
                                       "order or the element/byte unit of the strides), and it has "
                                       "not been retried", owner="ours")
            # gm_to_ub.nd's loop strides restate what the strided view / GM slice already
            # carries (device_lower folds them from the SAME geometry): pl.load re-derives its
            # descriptor from the make_tensor strides, so the attrs ride along informationally.
            dst, src = op.operands
            pre: list[str] = []
            post: list[str] = []
            tile = self.tiles.get(dst.name)
            if tile is not None and "py" not in tile:
                if self._off_fold(tile["offs_ir"][0]) != 0 or self._off_fold(tile["offs_ir"][1]) != 0:
                    # pl.load writes from the tile ORIGIN, so an offset window becomes a tile of
                    # its own at that address - the same treatment the store side got in D-099,
                    # including the enumerated tile group when the origin walks a loop or the
                    # sub-block id (that address is not a compile-time integer, so no single
                    # make_tile can carry it).
                    from .load_window import load_window
                    tile, pre, post = load_window(self, op, dst.name, tile)
                else:
                    pre, post = self._vs_wrap(tile)
                    tile = {**tile, "py": tile["base_py"]}
            else:
                tile = self.tile(op, dst)
            flat_scale = False
            if code == "dma.gm_to_l1.pad" and tile.get("scale"):
                # An MX scale filled by the BLOCK route (gm_to_l1_mx_scale): GM already holds
                # the packed 32-byte scale blocks, so this is a flat burst. The fractal tile
                # declared for the Scale move is the wrong window for it - its rows would
                # stride by the GM row length - so load through a one-row byte alias at the
                # same L1 address. Same bytes, and a 1-row Mat tile is the ND form pto takes.
                nb, bl = self.env.fold(op.attrs.get("n_burst")), self.env.fold(op.attrs.get("burst_len_byte"))
                if not isinstance(nb, int) or bl != 32:
                    raise PyptoGap(op, f"MX scale block fill n_burst={nb} burst={bl}: only whole "
                                       "32-byte scale blocks have a flat pl.load spelling")
                apy = self._physical_alias(op, tile, [1, nb * 32], "u8", "_blk")
                tile = {**tile, "py": apy, "shape": [1, nb * 32], "dtype": "u8"}
                flat_scale = True
            base, offsets = self.gm(op, src)
            # dn2nz IS the transposed load, and pl.load's `order` is how pto spells it. BUT
            # order is NOT a transpose flag: it is an AXIS SELECTION, one entry per TILE
            # dimension, each naming a TENSOR axis (pto's own docstring: "which axes of the
            # Tensor the Tile dimensions map to ... Default: last N axes of the Tensor").
            # So the transposed load of a 2-D tile is the last two axes NAMED IN REVERSE -
            # [1, 0] only when the tensor is rank 2. On a rank-3 [B, S, D] tensor [1, 0]
            # reads (axis 1, axis 0) = (S, BATCH): board 507015 on a probe (mha_ifa_256,
            # mha_ifa_fp8_scale_256) or silent garbage when the stride happens to stay
            # inside the allocation (mha_ifa_v2's QK). [2, 1] board-proven bit-equal to the
            # rank-2 [1, 0] on the same bytes.
            dn = code == "dma.gm_to_l1.dn2nz"
            order = ""
            if tile.get("order_flip") or dn:
                rank = len(offsets)
                if rank < 2:
                    raise PyptoGap(op, f"{code}: a transposed load needs a rank-2 or deeper "
                                       f"tensor, got rank {rank}")
                order = f", order=[{rank - 1}, {rank - 2}]"
            col = None
            load_axes = None
            view = self.views.get(src.name)
            if code in ("dma.gm_to_l1.nd2nz", "dma.gm_to_l1.dn2nz") and view is not None:
                extents = view[2] if view[2] is not None else view[3]
                axes = [i for i, extent in enumerate(extents) if extent != 1]
                if len(axes) == 2 and axes != [len(offsets) - 2, len(offsets) - 1]:
                    load_axes = axes
                    ordered = axes[::-1] if tile.get("order_flip") or dn else axes
                    order = f", order={ordered}"
            if code in ("dma.gm_to_ub.pad", "dma.gm_to_ub.nd") and not tile.get("order_flip"):
                col = self.column_view(op, src, tile)
                if col is not None:
                    base, offsets = col
            if not tile.get("order_flip") and not flat_scale and col is None:
                win = self.check_extents(op, src, {**tile, "order_flip": True} if dn else tile, code, axes=load_axes)
                if win is not None:
                    if pre:
                        raise PyptoGap(op, f"{code}: a partial transfer into a tile window "
                                           "stacks two validshapes (a later phase)")
                    pre.append(f"pl.set_validshape({tile['py']}, [{', '.join(str(w) for w in win)}])")
                    post.append(f"pl.set_validshape({tile['py']}, {list(tile['shape'])})")
            for ln in pre:
                self.emit(ln)
            self.emit(f"pl.load({tile['py']}, {base}, [{', '.join(offsets)}]{order})")
            for ln in post:
                self.emit(ln)
            return
        if code in ("dma.ub_to_gm.pad", "dma.l0c_to_gm.nz2nd", "dma.l0c_to_gm.nz2dn"):
            if code == "dma.l0c_to_gm.nz2dn":
                # pto's Acc->GM store HAS a DN destination form in C++ (TStore.hpp accepts
                # !isRowMajor && SFractal == NoneBox; TMov.hpp spells the same enableNz2Dn),
                # but nothing in the pl layer selects it. Both candidate spellings were
                # tried on silicon: declaring the GM parameter pl.Tensor[..., pl.DN] leaves
                # the store untransposed (both boxes, max abs 3.6e+01, and the bytes match
                # neither orientation nor an NZ fractal reading of either), and pl.store's
                # own `order` kwarg is an ASCENDING axis map, not a transpose - it refuses
                # [1, 0] outright ("store: order must be ascending"). Unlike the LOAD side,
                # where order=[rank-1, rank-2] does reach the transposing engine, the store
                # side has no reversed form to ask for.
                # D-130: the ISA layer has no Acc->DN entry point either. a5/TStore.hpp's
                # CheckStaticAcc - the static check on the Acc source path - enumerates the
                # destination layouts it accepts and DN is NOT among them:
                # Diagnostic excerpt: "TSTORE(Acc2GM) only support NZ2ND / NZ2NZ / NZ2NHWC / NZ2NCHW / NZ2NCDHW."
                # DN appears only in CheckStaticVec (a Vec source, !isRowMajor + NoneBox). The
                # hardware HAS the mode - cce reaches it through copy_matrix_cc_to_gm's final
                # flag plus set_channel_para(1 << 48), board-proven - and a stride-swapped
                # make_tensor cannot substitute for it: `layout` is a compile-time property of
                # the tensor type, not a function of the stride values, and the fixpipe burst
                # has no per-element column stride to fake a transpose with.
                raise PyptoGap(op, "dma.l0c_to_gm.nz2dn: the C++ store has a DN destination "
                                   "form but the pl layer cannot select it - the pl.DN "
                                   "annotation leaves the bytes untransposed on silicon and "
                                   "pl.store's order kwarg must be ascending (board: 'order "
                                   "must be ascending, got [1, 0]'), so unlike the load side "
                                   "there is no reversed spelling to ask for. Deeper than the pl "
                                   "layer, pto's own Acc->GM static check lists NZ2ND / NZ2NZ / "
                                   "NZ2NHWC / NZ2NCHW / NZ2NCDHW and no DN at all "
                                   "(a5/TStore.hpp CheckStaticAcc) - DN is a Vec-source "
                                   "destination only, so there is nothing to reach", owner="upstream")
            self.guard_attrs(op, {"n_burst", "burst_len_byte", "src_stride_byte", "dst_stride",
                                  "M", "N", "M_dst", "N_src", "M_src", "N_dst", "src_stride",
                                  "dst_stride_byte", "scale", "offset", "atomic"})
            dst, src = op.operands
            pre = []
            post = []
            tile = self.tiles.get(src.name)
            if tile is not None and "py" not in tile:
                if self._off_fold(tile["offs_ir"][0]) != 0 or self._off_fold(tile["offs_ir"][1]) != 0:
                    # pl.store always reads from the tile ORIGIN, so an offset window has to
                    # become a tile of its own at that address. A runtime-SIZED window still
                    # can: the origin is what needs an address, the extent is what
                    # set_validshape carries - materialise at the base shape, then narrow.
                    shp = list(tile["shape"])
                    if all(isinstance(x, int) for x in shp):
                        tile = self._strip_tile(op, src.name, tile)
                    else:
                        rec = {**tile, "shape": list(tile["base_shape"])}
                        full = self._strip_tile(op, src.name + ".origin", rec)
                        pre, post = self._vs_wrap({**tile, "py": full["py"]})
                        tile = {**full, "shape": shp}
                else:
                    pre, post = self._vs_wrap(tile)
                    tile = {**tile, "py": tile["base_py"]}
            else:
                tile = self.tile(op, src)
            base, offsets = self.gm(op, dst)
            store_axes = None
            order = ""
            view = self.views.get(dst.name)
            if view is not None:
                extents = view[2] if view[2] is not None else view[3]
                axes = [i for i, extent in enumerate(extents) if extent != 1]
                if len(axes) == 2 and axes != [len(offsets) - 2, len(offsets) - 1]:
                    store_axes = axes
                    order = f", order={axes}"
            # A source window may already have runtime rows. Compare the GM
            # transfer against its static capacity, then intersect with that
            # window below. Otherwise runtime rows hide a narrower GM column
            # extent and padded UB columns are incorrectly stored.
            capacity = {**tile, "shape": tile.get("base_shape", tile["shape"])}
            win = self.check_extents(op, dst, capacity, code, axes=store_axes)
            if win is not None and pre:
                # the window already narrows the base tile; the transfer narrows it again.
                # One descriptor holds one shape, so the two fold into their elementwise min -
                # which is what both of them mean (read no more than either allows).
                have = [str(x) for x in tile["shape"]]
                merged = [h if h == w else (w if w.isdigit() and h.isdigit() and int(w) <= int(h)
                                            else (h if w.isdigit() and h.isdigit() else f"pl.min({h}, {w})"))
                          for h, w in zip(have, (str(x) for x in win))]
                pre = [f"pl.set_validshape({tile['py']}, [{', '.join(merged)}])"]
                win = None
            if win is not None:
                if pre:
                    raise PyptoGap(op, f"{code}: a partial transfer from a tile window "
                                       "stacks two validshapes (a later phase)")
                pre.append(f"pl.set_validshape({tile['py']}, [{', '.join(str(w) for w in win)}])")
                post.append(f"pl.set_validshape({tile['py']}, {list(tile['shape'])})")
            kw = ""
            scale = op.attrs.get("scale")
            if scale is not None and isinstance(op.operands[0].type, MemType) \
                    and op.operands[0].type.dtype.name == "bf16" \
                    and self.tile(op, op.operands[1])["dtype"] in ("i32", "f32"):
                raise PyptoGap(op, f"{self.tile(op, op.operands[1])['dtype']}->bf16 fixpipe scale: "
                               "pypto refuses it (its fixpipe dequantizes INT32->FP16 only and "
                               "truncates FP32->BF16 without a scale - board parse error; our c310 "
                               "path scales into bf16)", owner="upstream")
            off_q = self.env.fold(op.attrs.get("offset", 0)) or 0
            if off_q and scale is None:
                raise PyptoGap(op, f"{code} offset={off_q} without a scale: the deqScalar "
                                   "offset field rides only on the quantization path")
            if scale is not None:
                if off_q:
                    # the requant offset lives in deqScalar bits [45:37] (int9). pto's
                    # python layer encodes only the fp32 scale bits - but an INT64 runtime
                    # scalar passes VERBATIM into set_quant_pre(u64) (block_ops
                    # _resolve_scale_param -> TStore.hpp:368), so we pack the full cce
                    # pack_deq_scalar word (tensorutils_cce.h:1348) ourselves
                    sv = self.env.fold(scale) if isinstance(scale, Value) else scale
                    if sv is None or isinstance(sv, Value):
                        raise PyptoGap(op, f"{code} offset={off_q} with a runtime scale: "
                                           "the packed deqScalar needs both at compile time")
                    if isinstance(op.operands[0].type, MemType) \
                            and op.operands[0].type.dtype.name == "u8":
                        raise PyptoGap(op, "u8 fixpipe requant: pto's parser refuses UINT8 "
                                           "outputs (\"hardware fixpipe has no unsigned "
                                           "requantization path\" - board; cce runs the same "
                                           "u8 requant green, an upstream front-end refusal)", owner="upstream")
                    import struct as _struct
                    bits = _struct.unpack("<I", _struct.pack("<f", float(sv)))[0]
                    packed = (bits & 0xFFFFE000) | ((off_q & 0x1FF) << 37)
                    # a BARE pl.const folds to a python int in pto's parser, and
                    # _resolve_scale_param then re-encodes the NUMBER as an f32 pattern
                    # (board: saturated output). A runtime zero term keeps it an INT64
                    # Expr, and the codegen widens integer bits as-is into
                    # set_quant_pre(u64) - offset bits [45:37] ride along (board: exact)
                    kw = (f", scale=(pl.const(0x{packed:x}, pl.DT_INT64)"
                          f" + pl.get_block_idx() * 0)")
                elif isinstance(scale, Value):
                    s = self.tiles.get(scale.name)
                    kw = f", scale={s['py']}" if s else f", scale={self.env.ref(scale)}"
                else:
                    kw = f", scale={_lit(scale)}"
            kw += self._atomic_kw(op, dst)
            for ln in pre:
                self.emit(ln)
            self.emit(f"pl.store({base}, {tile['py']}, [{', '.join(offsets)}]{order}{kw})")
            for ln in post:
                self.emit(ln)
            return
        if code == "dma.set_constant_to_l1":
            self.guard_attrs(op, {"val", "n_blocks"})
            self.fill_l1(op)
            return
        if code == "dma.l1_to_bt":
            # the C2 bias-table burst: pto spells it as a straight Mat->Bias move (an allowed
            # pl.move path); the n= attr repeats the row width the flat burst carries
            self.guard_attrs(op, {"n"})
            n = self.env.fold(op.attrs.get("n"))
            if n is None:
                raise PyptoGap(op, "l1_to_bt with a runtime width: the Bias tile shape must "
                                   "fold statically (pl.make_tile takes no runtime dims)")
            d = self.tile(op, op.operands[0])
            srec = self.tiles.get(op.operands[1].name)
            capacity = [self.env.fold(dim) for dim in op.operands[0].type.dims]
            if n <= 0 or (all(isinstance(dim, int) for dim in capacity) and n > capacity[0] * capacity[1]):
                raise PyptoGap(op, "Bias width exceeds its declared BT slot capacity", owner="ours")
            if srec is not None and "py" not in srec and srec.get("base_py") and not srec.get("nz"):
                # a sliced window of the [1, N] bias row (split-N): a single L1 row is
                # byte-contiguous, so the strip materialises like any other window (a
                # loop-varying offset flattens into a tile group - make_tile addr must be
                # compile-time, board: "'addr' must be a compile-time integer")
                self._strip_tile(op, op.operands[1].name, srec)
            sr = self.tile(op, op.operands[1])
            cols = d["shape"][-1] if d["shape"] else None
            if isinstance(cols, int) and n != cols:
                # lowering allocates the bias table at BT capacity; pto wants the Bias tile
                # declared at its real [1, n] - re-declare a narrow alias at the same address
                esz = _esize_bits(d["dtype"]) // 8
                if d.get("addr") is None:
                    # a rotating slot: alias the whole buffer with a second, narrow tile
                    # group over the SAME slot addresses (the raw-bytes regroup idiom),
                    # selected by the same index
                    b = self.tiles.get(d.get("group_of") or "")
                    if not b or d.get("index_py") is None:
                        raise PyptoGap(op, f"l1_to_bt of {n} elements into a rotating "
                                           f"{cols}-wide slot with no group handle")
                    key = (d["group_of"], n)
                    cache = getattr(self, "_bt_narrow", None)
                    if cache is None:
                        cache = self._bt_narrow = {}
                    gn = cache.get(key)
                    if gn is None:
                        gn = self.mp.unique((b.get("group_py") or "btgrp") + "_n")
                        tt = self.mp.tiletype(d["dtype"], [1, n], "bt")
                        base = int(b["addr"])
                        addrs = ", ".join(str(base + i * b.get("pitch", b["size"])) for i in range(b["slots"]))
                        self.tile_decl(self._source_op, f"{gn} = pl.make_tile_group(type={tt}, addrs=[{addrs}], "
                                       f"depth={b['slots']})", name=gn, type_expr=tt, bank="bt",
                                       addresses=tuple(base + i * b.get("pitch", b["size"]) for i in range(b["slots"])),
                                       size=n * esz, shape=[1, n], dtype=d["dtype"], group=True, hoisted=True)
                        cache[key] = gn
                    py = self.mp.unique(d["py"] + "_n")
                    self.getitem(py, gn, d["index_py"])
                    d = self.tiles[op.operands[0].name] = {**d, "py": py, "shape": [1, n],
                                                           "size": n * esz}
                else:
                    py = self.mp.unique(d["py"] + "_n")
                    tt = self.mp.tiletype(d["dtype"], [1, n], "bt")
                    self.tile_decl(self._source_op, f"{py} = {self.make_tile(self._source_op, tt, d['addr'], n * esz, [1, n], d['dtype'], 'bt')}",
                                   name=py, type_expr=tt, bank="bt", addresses=(d["native_addr"],),
                                   size=n * esz, shape=[1, n], dtype=d["dtype"])
                    d = self.tiles[op.operands[0].name] = {**d, "py": py, "shape": [1, n], "size": n * esz}
            self.emit(f"pl.move({d['py']}, {sr['py']})")
            return
        if code == "dma.ub_to_l1.nz":
            # the compact-NZ publish: cce's ub_to_l1_nz(m_src, n_src, m_dst, n_dst, M_src)
            # == TInsertImpl(validRow=m_src, validCol=n_src, dstRow=m_dst, index=dst offsets,
            # srcGap=M_src-m_src) burst for burst - reached through pl.insert with an
            # NZ-DECLARED Vec source alias (an ND source dispatches to TINSERT's flat
            # branch; probe: equal/rows/colwin all bit-exact)
            self.guard_attrs(op, {"m_src", "n_src", "m_dst", "n_dst", "M_src",
                                  "dst_row0", "dst_col0", "src_row0", "src_col0"})
            dst, src = op.operands
            s0 = self.tiles.get(src.name)
            d0 = self.tiles.get(dst.name)
            if s0 is None or d0 is None:
                raise PyptoGap(op, f"{code}: {src.name if s0 is None else dst.name} is not a "
                                   "tile the printer knows")
            if "py" in d0:
                dp, dr, dc = d0["py"], "0", "0"
            elif d0.get("base_py"):
                dp = d0["base_py"]
                dr = self._off_expr(op, d0["offs_ir"][0])
                dc = self._off_expr(op, d0["offs_ir"][1])
            else:
                raise PyptoGap(op, f"{code}: destination {dst.name} has no materialised base")
            esz = _esize_bits(s0["dtype"]) // 8
            n_src = self.env.fold(op.attrs.get("n_src"))
            M_src = self.env.fold(op.attrs.get("M_src", op.attrs.get("m_src")))
            if M_src is None and n_src is not None:
                source_rows = op.attrs.get("M_src", op.attrs.get("m_src"))
                bounds = self.env.frange(source_rows)
                if bounds is not None and 1 <= bounds[0] <= bounds[1] <= 16 and isinstance(source_rows, Value):
                    # Compact NZ changes its column pitch with the row count.
                    # Enumerate the bounded row types and select the matching
                    # descriptor at runtime; do not substitute the capacity.
                    runtime = self.env.ref(source_rows)
                    saved = dict(self.env.names)
                    for extent in range(bounds[0], bounds[1] + 1):
                        self.emit(f"if {runtime} == {extent}:")
                        self.indent += 1
                        self.env.names[source_rows.name] = str(extent)
                        self.op(op)
                        self.env.names = dict(saved)
                        self.indent -= 1
                    return
            if n_src is None or M_src is None:
                raise PyptoGap(op, f"{code}: the source declared geometry (M_src, n_src) must "
                                   "fold statically (the NZ alias declaration needs it)")
            if s0.get("offs_ir"):
                r_off = self._off_fold(s0["offs_ir"][0])
                c_off = self._off_fold(s0["offs_ir"][1])
                if r_off != 0:
                    raise PyptoGap(op, f"{code}: a compact-NZ source window with a row offset "
                                       "has no alias (rows interleave inside each 32B block)")
                if c_off is None or (c_off * esz) % 32:
                    raise PyptoGap(op, f"{code}: the source column offset must be a static "
                                       "32-byte-block multiple")
                base_addr = s0.get("nz_addr")
            else:
                base_addr = s0.get("nz_addr", s0.get("addr"))
                c_off = 0
            cache = getattr(self, "_nz_alias", None)
            if cache is None:
                cache = self._nz_alias = {}
            if base_addr is None and s0.get("group_of") and s0.get("index_py") is not None:
                # a rotating slot: a second, narrow NZ tile group over the same slot
                # addresses (the raw-bytes regroup idiom), selected by the same index
                g_of = s0["group_of"]
                b = self.tiles.get(g_of) if not isinstance(g_of, dict) else g_of
                if not b or b.get("addr") is None:
                    raise PyptoGap(op, f"{code}: the source slot group has no static base")
                key = ("grp", str(g_of), c_off, M_src, n_src, s0["dtype"])
                gn = cache.get(key)
                if gn is None:
                    tt = self.mp.tiletype(s0["dtype"], [M_src, n_src], "ub", "pl.NZ")
                    base = int(b["addr"]) + c_off * M_src * esz
                    addrs = ", ".join(str(base + i * b.get("pitch", b["size"])) for i in range(b["slots"]))
                    gn = self.mp.unique("nzsrc_grp")
                    self.tile_decl(op, f"{gn} = pl.make_tile_group(type={tt}, addrs=[{addrs}], "
                               f"depth={b['slots']})", name=gn, type_expr=tt, bank="ub",
                               addresses=tuple(base + i * b.get("pitch", b["size"]) for i in range(b["slots"])),
                               size=M_src * n_src * esz, shape=[M_src, n_src], dtype=s0["dtype"], group=True, hoisted=True)
                    cache[key] = gn
                py = self.mp.unique("nzsrc")
                self.getitem(py, gn, s0["index_py"])
            elif base_addr is None:
                raise PyptoGap(op, f"{code}: source {src.name} has no static address")
            else:
                key = (str(base_addr), c_off, M_src, n_src, s0["dtype"])
                py = cache.get(key)
                if py is None:
                    addr = f"({base_addr} + {c_off * M_src * esz})" if c_off else base_addr
                    tt = self.mp.tiletype(s0["dtype"], [M_src, n_src], "ub", "pl.NZ")
                    py = self.mp.unique("nzsrc")
                    native_base = s0.get("nz_native_addr", s0.get("native_addr"))
                    self.tile_decl(op, f"{py} = {self.make_tile(op, tt, addr, M_src * n_src * esz, [M_src, n_src], s0['dtype'], 'ub')}",
                                   name=py, type_expr=tt, bank="ub",
                                   addresses=(native_base + c_off * M_src * esz if native_base is not None else None,),
                                   size=M_src * n_src * esz, shape=[M_src, n_src], dtype=s0["dtype"], hoisted=True)
                    cache[key] = py
            m_src = self.env.fold(op.attrs.get("m_src"))
            ms = str(m_src) if m_src is not None else self.iexpr(op, op.attrs.get("m_src"))
            pre: list[str] = []
            post: list[str] = []
            if not (isinstance(m_src, int) and m_src == M_src):
                pre = [f"pl.set_validshape({py}, [{ms}, {n_src}])"]
                post = [f"pl.set_validshape({py}, [{M_src}, {n_src}])"]
            for ln in pre:
                self.emit(ln)
            self.emit(f"pl.insert({dp}, {py}, [{dr}, {dc}])")
            for ln in post:
                self.emit(ln)
            return
        if code == "dma.l1_to_l0.mx":
            # Scale L1 -> L0 is its own pl.move into ScaleLeft/ScaleRight, and the L0 scale
            # tile's ADDRESS is not free: pto's matmul_mx doc states the hardware locates it
            # implicitly as addr(scale) = addr(data_tile) >> 4, so it is derived from the
            # L0A/L0B fragment this move fills, never allocated.
            mx = op.attrs.get("src_mx")
            nm = str(getattr(mx, "name", mx))
            srec = self.tiles.get(nm)
            if srec is None or "py" not in srec:
                raise PyptoGap(op, f"l1_to_l0.mx whose scale {nm} is not a plain staged tile")
            side = self.mp.scale_side.get(nm, "scale_l")
            drec = self.tiles.get(op.operands[0].name) or {}
            daddr = drec.get("addr")
            shape = list(srec["shape"])
            lay, srcpy = ("pl.ZZ" if side == "scale_l" else "pl.NN"), srec["py"]
            tt = self.mp.tiletype("e8m0", shape, side, lay)
            cache = getattr(self, "_mx_slots", None)
            if cache is None:
                cache = self._mx_slots = {}
            if daddr is None:
                # the data fragment ROTATES, so the scale rotates with it: its address is that
                # slot's own >> 4. One tile group over the shifted slot addresses, indexed by
                # the same counter expression the data slot already materialised.
                base = self.tiles.get(drec.get("group_of") or "")
                baddr = None if not base else base.get("addr")
                if isinstance(baddr, str) and re.fullmatch(r"[\d\s()+\-*/%]+", baddr):
                    baddr = int(eval(baddr, {"__builtins__": {}}, {}))  # noqa: S307 - digits only
                if not base or drec.get("index_py") is None or not isinstance(baddr, int):
                    raise PyptoGap(op, "l1_to_l0.mx into a fragment with neither a static address "
                                       "nor a slot group at a compile-time base: the scale's "
                                       "address is addr(data) >> 4")
                key = (nm, side, baddr, int(base["size"]), int(base["slots"]))
                gn = cache.get(key)
                if gn is None:
                    addrs = ", ".join(str((baddr + j * int(base.get("pitch", base["size"]))) >> 4)
                                      for j in range(int(base["slots"])))
                    gn = self.mp.unique(srec["py"] + "_sgrp")
                    self.tile_decl(self._source_op, f"{gn} = pl.make_tile_group(type={tt}, addrs=[{addrs}], "
                               f"depth={base['slots']})", name=gn, type_expr=tt, bank=side,
                               addresses=tuple((baddr + j * int(base.get("pitch", base["size"]))) >> 4
                                               for j in range(int(base["slots"]))),
                               size=srec["size"], shape=shape, dtype="e8m0", group=True, hoisted=True,
                               parent_slots=tuple((base["group_py"], j) for j in range(int(base["slots"]))))
                    cache[key] = gn
                spy = self.mp.unique(srec["py"] + "_s")
                self.getitem(spy, gn, drec["index_py"])
            else:
                key = (nm, side, str(daddr))
                got = cache.get(key)
                if got is None:
                    spy = self.mp.unique(srec["py"] + "_s")
                    address = self._flat_addr(op, op.operands[0]) >> 4
                    self.tile_decl(self._source_op, f"{spy} = {self.make_tile(self._source_op, tt, f'(({daddr}) >> 4)', srec['size'], shape, 'e8m0', side)}",
                                   name=spy, type_expr=tt, bank=side, addresses=(address,), size=srec["size"],
                                   shape=shape, dtype="e8m0", parent_slots=((drec["py"], 0),))
                    cache[key] = (spy, srcpy)
                    got = cache[key]
                spy, srcpy = got
            self.emit(f"pl.move({spy}, {srcpy})")
            self.mx_scale_py = getattr(self, "mx_scale_py", {})
            self.mx_scale_py[nm] = spy
            code = "dma.l1_to_l0"  # fall through to the data move
        if code in ("dma.l1_to_l0", "dma.l0c_to_ub", "dma.l0c_to_l1", "dma.ub_to_l1",
                    "dma.ub_to_l1.nd2nz", "dma.ub_to_ub"):
            dst, src = op.operands
            if code == "dma.ub_to_l1.nd2nz":
                self._ub_nd2nz(op, dst, src)
                return
            known = {"m_src", "n_src", "m_dst", "n_dst", "src_row0", "src_col0",
                     "src_is_transpose", "dst_position", "n_burst", "burst_len_byte",
                     "burst_len",
                     "src_stride", "dst_stride", "src_stride_byte", "dst_stride_byte",
                     "M", "N", "M_dst", "N_src", "M_src", "N_dst",
                     "dual_mode", "sub_block_id", "dst_row0", "dst_col0", "relu",
                     "src_mx", "src_mx_row0", "src_mx_col0"}
            if code == "dma.l0c_to_ub":
                known.add("scale")   # the fixpipe quantisation scale - pl.move(scale=)
            if code == "dma.l1_to_l0":
                known.add("m_copy")  # explicit paired-fractal physical extent (M10-030)
            self.guard_attrs(op, known)
            drec0 = self.tiles.get(op.operands[0].name) or {}
            srec0 = self.tiles.get(op.operands[1].name) or {}
            if code == "dma.ub_to_l1" and drec0.get("scale") and "py" in drec0:
                # An MX scale staged from the VECTOR side. Its L1 tile is declared as the
                # fractal the Scale stop reads (behaviour #16), and its UB source is flat - a
                # Vec tile has no fractal. For a scale of at most two k-groups the two are the
                # same bytes (the fractal's 16x2 box in row-tile-major order IS the row-major
                # block), so the DESTINATION takes a flat alias for this copy exactly as the GM
                # block route does, and the fractal declaration stays for the L0 move.
                dsh = list(drec0["shape"])
                kg = dsh[0] if drec0["scale"] == "scale_r" else dsh[1]
                if not (isinstance(kg, int) and kg <= 2):
                    raise PyptoGap(op, f"ub_to_l1 into an MX scale of {kg} k-groups: past two, the "
                                       "fractal's box order and the vector side's row-major "
                                       "layout stop being the same bytes")
                total = 1
                for x in dsh:
                    total *= int(x)
                # ... and the carrier is INT8: pto's TMov lists int8_t / hifloat8 / ... and
                # refuses unsigned char ("TMov: Unsupported data type!"), so both sides of the
                # copy name the signed byte over the same storage.
                tt = self.mp.tiletype("i8", [1, total], "l1", None)
                apy = self.mp.unique(drec0["py"] + "_blk")
                if drec0.get("addr") is None:
                    base = self.tiles.get(drec0.get("group_of") or "")
                    if not base or drec0.get("index_py") is None:
                        raise PyptoGap(op, "rotating MX scale copy has no physical slot group")
                    addresses = tuple(int(base["addr"]) + j * base["pitch"] for j in range(base["slots"]))
                    group = self.mp.unique(apy + "_grp")
                    self.tile_decl(op, f"{group} = pl.make_tile_group(type={tt}, addrs={list(addresses)}, depth={base['slots']})",
                                   name=group, type_expr=tt, bank="l1", addresses=addresses, size=total,
                                   shape=[1, total], dtype="i8", group=True, hoisted=True)
                    self.getitem(apy, group, drec0["index_py"])
                else:
                    self.tile_decl(op, f"{apy} = {self.make_tile(op, tt, drec0['addr'], drec0['size'], [1, total], 'i8', 'l1')}",
                                   name=apy, type_expr=tt, bank="l1", addresses=(drec0["native_addr"],),
                                   size=total, shape=[1, total], dtype="i8")
                self.tiles[op.operands[0].name] = {**drec0, "py": apy, "shape": [1, total],
                                                   "dtype": "i8"}
                if "py" in srec0 and srec0.get("addr") is not None:
                    stt = self.mp.tiletype("i8", [1, total], "ub", None)
                    spy2 = self.mp.unique(srec0["py"] + "_i8")
                    self.tile_decl(self._source_op, f"{spy2} = {self.make_tile(self._source_op, stt, srec0['addr'], srec0['size'], [1, total], 'i8', 'ub')}", name=spy2, type_expr=stt, bank="ub",
                                   addresses=(srec0["native_addr"],), size=srec0["size"],
                                   shape=[1, total], dtype="i8")
                    self.tiles[op.operands[1].name] = {**srec0, "py": spy2, "shape": [1, total],
                                                       "dtype": "i8"}
            if "burst_len" in op.attrs:
                # a block-descriptor copy (n_burst bursts of burst_len 32-byte blocks, strided).
                # pl.move copies whole tiles, so it says the same thing only when the descriptor
                # IS the whole tile laid out contiguously - which is what the MX scale's UB -> L1
                # staging is (one 32-byte block, no strides). Anything else keeps its own shape.
                nb = self.env.fold(op.attrs.get("n_burst", 1))
                bl = self.env.fold(op.attrs.get("burst_len"))
                ss = self.env.fold(op.attrs.get("src_stride", 0))
                ds = self.env.fold(op.attrs.get("dst_stride", 0))
                drec = self.tiles.get(op.operands[0].name) or {}
                want = drec.get("size")
                if not (isinstance(nb, int) and isinstance(bl, int) and ss == 0 and ds == 0
                        and isinstance(want, int) and nb * bl * 32 == want):
                    raise PyptoGap(op, f"{code} descriptor n_burst={nb} burst_len={bl} "
                                       f"src_stride={ss} dst_stride={ds} over a {want}-byte tile: "
                                       "pl.move copies a whole tile, so only a contiguous "
                                       "whole-tile descriptor says the same thing")
            if op.attrs.get("src_is_transpose") and src.name not in self.transpose_absorbed:
                s0 = self.tiles.get(src.name) or {}
                if not (src.name in getattr(self, "zn_tiles", ())
                        or src.name in getattr(self, "zn_alias", ()) or s0.get("zn")):
                    raise PyptoGap(op, f"{code} with src_is_transpose on an on-chip-produced "
                                       "tile outside the ZN chain: the only on-chip transpose "
                                       "is the ZN-declared source (SFractal dispatch to the "
                                       "load_cbuf_to_ca transpose form)")
                # the source is declared with the transposed shape + layout=ZN: the move is
                # an EQUAL-shape TMOV and pto's SFractal comparison issues transpose=1
            if str(op.attrs.get("dst_position", "")) == "l0b" \
                    and not op.attrs.get("src_is_transpose"):
                s0 = self.tiles.get(src.name, {})
                if not (s0.get("order_flip") or src.name in self.transpose_absorbed
                        or src.name in getattr(self, "zn_tiles", ())
                        or src.name in getattr(self, "zn_alias", ()) or s0.get("zn")):
                    raise PyptoGap(op, f"{code}: an implicit-B move from an on-chip-produced L1 "
                                       "tile needs the [N, K] -> [K, N] transpose pto cannot spell")
            kw = ""
            if self.env.fold(op.attrs.get("relu", 0)):
                kw += ", relu_pre_mode=pl.ReluPreMode.NormalRelu"
            mode = op.attrs.get("dual_mode")
            if mode is not None:
                m = str(mode)
                if m == "splitm":
                    kw += ", acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM"
                elif m == "splitn":
                    kw += ", acc_to_vec_mode=pl.AccToVecMode.DualModeSplitN"
                elif m == "single":
                    sub = self.env.fold(op.attrs.get("sub_block_id", 0)) or 0
                    kw += f", acc_to_vec_mode=pl.AccToVecMode.SingleModeVec{sub}"
                else:
                    raise PyptoGap(op, f"{code} dual_mode={m} has no pl.AccToVecMode")
            self._fix_scale_kw = False
            sc = op.attrs.get("scale")
            if sc is not None and code == "dma.l0c_to_ub":
                # pl.move's per-tensor deqScalar path. A COMPILE-TIME float only: the runtime
                # forms need the caller to hand over an IEEE-754 bit pattern (an FP32 Scalar is
                # bitcast for you, an INT32/INT64 one must already carry struct.pack("!f", s)),
                # and a per-channel scale is a user-prepared INT64 tile in MemorySpace.Scaling
                # whose data flow - load, move, MTE1->FIX sync - pl does not insert.
                f = sc.value if isinstance(sc, Literal) else sc
                if not isinstance(f, (int, float)) or isinstance(f, bool):
                    f = self.env.fold(f)
                if not isinstance(f, (int, float)) or isinstance(f, bool):
                    raise PyptoGap(op, f"{code} scale={sc!r} is not a compile-time float: pl.move "
                                       "takes a runtime scale only as an FP32 scalar (bitcast) or "
                                       "a pre-encoded INT32/INT64 bit pattern, neither of which "
                                       "this printer builds")
                if float(f) != 1.0:
                    source_dtype = self.tiles[src.name]["dtype"]
                    target_dtype = self.tiles[dst.name]["dtype"]
                    if (source_dtype, target_dtype) in (("f32", "f32"), ("f32", "e4m3")):
                        raise PyptoGap(op, f"{code} scaled {source_dtype}->{target_dtype}: PyPTO emits "
                                           "TMOV's scalar overload, whose PTO dtype selector chooses "
                                           "NoQuant for FP32 or vector quantization for E4M3. The Python "
                                           "move API has no explicit quantization-mode parameter; the "
                                           "FP32 board control returns the unscaled result", owner="upstream")
                    kw += f", scale={float(f)!r}"
                    self._fix_scale_kw = True   # the offset guard below needs to know
            pre: list[str] = []
            post: list[str] = []
            s = self.tiles.get(src.name)
            if s is not None and "py" not in s:
                # a window source: pl.move reads the base tile through offset= (the pto split-K
                # idiom - set_validshape narrows the read window, offset places it; the wrap
                # restores the declared shape so the loop-body invariant holds)
                r0 = self._off_expr(op, s["offs_ir"][0])
                c0 = self._off_expr(op, s["offs_ir"][1])
                a, b = self._vs_wrap(s)
                pre += a
                post += b
                sp = s["base_py"]
            else:
                s = self.tile(op, src)
                sp = s["py"]
                r0 = self.iexpr(op, op.attrs.get("src_row0", 0))
                c0 = self.iexpr(op, op.attrs.get("src_col0", 0))
            if (code == "dma.l1_to_l0" and str(op.attrs.get("dst_position")) == "l0a"
                    and not op.attrs.get("src_is_transpose") and s is not None and "py" not in s
                    and not s.get("order_flip")):
                alias = self._mte1_row_alias(op, s)
                if alias is not None:
                    sp, r0 = alias
                    pre = [f"pl.set_validshape({sp}, [{', '.join(str(x) for x in s['shape'])}])"]
                    post = [f"pl.set_validshape({sp}, {list(s['base_shape'])})"]
                    s = {**s, "py": sp, "base_py": sp}
            if op.id in getattr(self, "zn_alias_ops", ()) and s is not None:
                # the cube-side ZN re-reading of an [N, K] tile (see the classifier): same
                # bytes, [K, N] shape, layout ZN - which makes the move into Right an
                # equal-shape, equal-SFractal copy. Declared here rather than on the tile
                # because pl.load has no ZN destination. A WINDOWED source flips with it:
                # the alias covers the whole base, and the window's extents and origin swap
                # (conv2d_relu's splitn narrows N, which is the alias's ROW extent).
                root = s if "py" in s else {**(s.get("root") or {}), "py": s.get("base_py"),
                                            "shape": s.get("base_shape")}
                if root.get("py") is None:
                    raise PyptoGap(op, f"{code}: the [K, N] ZN re-reading has no source tile")
                cache = getattr(self, "_zn_alias_py", None)
                if cache is None:
                    cache = self._zn_alias_py = {}
                zbase = [root["shape"][1], root["shape"][0]]
                tt = self.mp.tiletype(root["dtype"], zbase, "l1", "pl.ZN")
                if root.get("addr") is not None:
                    sp = cache.get(root["py"])
                    if sp is None:
                        sp = cache[root["py"]] = self.mp.unique(root["py"] + "_zn")
                        self.tile_decl(op, f"{sp} = {self.make_tile(op, tt, root['addr'], root['size'], zbase, root['dtype'], 'l1')}", name=sp, type_expr=tt, bank="l1",
                                  addresses=(root.get("native_addr"),), size=root["size"], shape=zbase, dtype=root["dtype"])
                else:
                    # a ROTATING source: pl.make_tile's addr must be a literal, but a tile
                    # GROUP's addrs are literals and its index is an expression, so the ZN
                    # re-reading of a double buffer is a group over the slot addresses
                    b = self.tiles.get(root.get("group_of") or "")
                    if not b or root.get("index_py") is None:
                        raise PyptoGap(op, f"{code}: the [K, N] ZN re-reading of a rotating slot "
                                           "needs the group handle its slots came from")
                    key = ("zn", root["group_of"], root["dtype"], tuple(zbase))
                    gn = self.regroups.get(key)
                    if gn is None:
                        gn = self.mp.unique(py_ident(root["group_of"]) + "_zn")
                        addrs = ", ".join(str(int(b["addr"]) + j * int(b.get("pitch", b["size"])))
                                          for j in range(int(b["slots"])))
                        self.tile_decl(op, f"{gn} = pl.make_tile_group(type={tt}, addrs=[{addrs}], "
                                   f"depth={b['slots']})", name=gn, type_expr=tt, bank="l1",
                                   addresses=tuple(int(b["addr"]) + j * int(b.get("pitch", b["size"])) for j in range(int(b["slots"]))),
                                   size=b["size"], shape=zbase, dtype=root["dtype"], group=True, hoisted=True)
                        self.regroups[key] = gn
                    sp = self.mp.unique(root["py"] + "_zn")
                    self.getitem(sp, gn, root["index_py"])
                if "py" in s:
                    s = {**s, "py": sp, "shape": zbase}
                else:
                    win = [s["shape"][1], s["shape"][0]]
                    if isinstance(win[1], int) and isinstance(zbase[1], int) and win[1] % 16:
                        # the L0B fragment this feeds is declared at its 16-column fractal (a
                        # narrow N is a valid SHAPE, never a declaration), so the read widens
                        # with it - the columns past N are the operand's own padding and the
                        # matmul's set_validshape keeps them out of the result
                        win = [win[0], min(-(-win[1] // 16) * 16, zbase[1])]
                    wtxt = "[" + ", ".join(str(x) for x in win) + "]"
                    pre = [f"pl.set_validshape({sp}, {wtxt})"] if win != zbase else []
                    post = [f"pl.set_validshape({sp}, {zbase})"] if win != zbase else []
                    r0, c0 = c0, r0
                    s = {**s, "py": sp, "shape": win, "base_py": sp, "base_shape": zbase}
            # destination routing: pl.move is a facade over three engines - TEXTRACT reads a
            # sub-window of a wide SOURCE (offset in source coordinates, handled above), TMOV
            # moves equal declared shapes, and TINSERT writes a small tile into a wide
            # DESTINATION at [row, col] (pl.insert; pto's own move dispatches there when
            # dst > src). A window destination therefore prints as an insert.
            d = self.tiles.get(dst.name)
            dr0 = dc0 = "0"
            insert_dst = False
            if code == "dma.l0c_to_ub" and d is not None and "py" not in d:
                d = self._fix_ub_subview(op, dst.name, d)
                dp = d["py"]
            elif code == "dma.l1_to_l0" and d is not None and "py" not in d and d["space"] == "l0b":
                # Right has no native pitch-preserving subview/TINSERT.
                # A proven compact fractal strip uses its actual byte origin.
                d = self._strip_tile(op, dst.name, d)
                dp = d["py"]
            elif d is not None and "py" not in d:
                dr0 = self._off_expr(op, d["offs_ir"][0])
                dc0 = self._off_expr(op, d["offs_ir"][1])
                if list(d["shape"]) != list(d["base_shape"]) or dr0 != "0" or dc0 != "0":
                    if not all(isinstance(x, int) for x in d["shape"]) and code != "dma.l1_to_l0":
                        raise PyptoGap(op, f"{code} into a runtime-sized window: TINSERT takes "
                                           "the source tile's declared shape")
                    insert_dst = True
                dp = d["base_py"]
            else:
                d = self.tile(op, dst)
                dp = d["py"]
                a_r0 = self.env.fold(op.attrs.get("dst_row0", 0))
                a_c0 = self.env.fold(op.attrs.get("dst_col0", 0))
                ds = list(d.get("shape") or ())
                ss = list(s.get("shape") or ())
                dr0 = str(a_r0) if a_r0 is not None else self.iexpr(op, op.attrs.get("dst_row0", 0))
                dc0 = str(a_c0) if a_c0 is not None else self.iexpr(op, op.attrs.get("dst_col0", 0))
                if ds and ss and ds != ss:
                    static = all(isinstance(x, int) for x in ds + ss)
                    insert_dst = (static and all(x >= y for x, y in zip(ds, ss))) or (
                        code == "dma.l1_to_l0" and not static)
            if code == "dma.l1_to_l0" and insert_dst and dr0 == "0" and dc0 == "0":
                # A small L1 transfer into the start of a larger L0 allocation
                # is a smaller destination tile at the same address. TINSERT
                # is the wrong engine; the later MMAD retains its own extents.
                wanted = list(s.get("shape") or ())
                capacity = list(d.get("base_shape") or d.get("shape") or ())
                source_capacity = list(s.get("base_shape") or s.get("shape") or ())
                if len(wanted) == len(capacity) == len(source_capacity) == 2:
                    wanted = [want if isinstance(want, int) else min(dc, sc)
                              if isinstance(dc, int) and isinstance(sc, int) else want
                              for want, dc, sc in zip(wanted, capacity, source_capacity)]
                if len(wanted) == len(capacity) == 2 and all(isinstance(x, int) for x in wanted + capacity):
                    root = d if "py" in d else d["root"]
                    alias = self._strip_tile(op, dst.name + ".load", {
                        "shape": wanted, "base_shape": capacity, "offs_ir": [[0], [0]],
                        "root": root, "space": d["space"], "dtype": d["dtype"],
                    })
                    dp, d, insert_dst = alias["py"], alias, False
            if code == "dma.l0c_to_l1":
                # pl.move has no Acc->Mat path - the parser prints its whole whitelist:
                # "Mat->Left, Mat->Right, Mat->Scaling, Mat->Bias, Mat->ScaleLeft,
                # Mat->ScaleRight, Acc->Vec, Vec->Vec, Vec->Mat". TINSERT does reach it:
                # pl.insert(mat, acc, [r, c]) compiles and runs on the board (D-116), which
                # is how the L0C->L1 publish keeps a matmul result on chip.
                insert_dst = True
            off = "" if (r0 == "0" and c0 == "0") else f", offset=[{r0}, {c0}]"
            if (off and self._fix_scale_kw and s.get("space") == "l0c" and "py" not in s
                    and self._off_fold(s["offs_ir"][0]) == 0
                    and s["shape"][0] == s["base_shape"][0]):
                # A full-height NZ column strip is an address alias. Read it
                # with scaled TMOV, avoiding the offset facade's unscaled
                # TEXTRACT. The strip helper preserves rotating slot indices.
                alias = self._strip_tile(op, src.name + ".scaled", s)
                sp, off = alias["py"], ""
                pre, post = [], []
            if off and getattr(self, "_fix_scale_kw", False):
                # D-128: pl.move takes `scale` and `offset` together and DROPS THE SCALE.
                # The offset form lowers to TEXTRACT, whose emitted signature has no
                # scale operand, while the offset-free form lowers to TMOV, which does:
                # Emitted form: TMOV<..., SingleModeVec0>(dst, src, static_cast<uint64_t>(1035273459))
                #   TEXTRACT<..., SingleModeVec1>(dst, src, 0, 64)          <-- no scale
                # (0x3db504f3 is the fp32 bit pattern of the scale; read out of the
                # generated kernel.cpp of v9_allhif8, with and without the kwarg). The
                # sub-block that took the offset then reads UNSCALED scores - on v9 that
                # is half the rows of every m-block, off by the softmax scale.
                raise PyptoGap(op, f"{code} with BOTH a fixpipe scale and a source "
                                   f"window (offset=[{r0}, {c0}]): pl accepts the pair "
                                   "and drops the scale - the offset form lowers to "
                                   "TEXTRACT, which has no scale operand, while the "
                                   "offset-free form lowers to TMOV, which does "
                                   "(read out of the generated kernel.cpp, 2026-09-02)",
                               owner="upstream")
            if insert_dst:
                if d.get("space") not in ("l1", "ub"):
                    raise PyptoGap(op, f"{code} into a {d.get('space')} window: TINSERT writes "
                                       "Mat or Vec destinations only (no L0 insert)")
                if off:
                    raise PyptoGap(op, f"{code} reads a source window AND writes a destination "
                                       "window: TEXTRACT and TINSERT do not compose in one move")
                if kw:
                    raise PyptoGap(op, f"{code}: acc_to_vec/scale do not combine with TINSERT")
                if "shape" in s and "py" not in (self.tiles.get(dst.name) or {"py": 1}) \
                        and list(s["shape"]) != list(d["shape"]):
                    raise PyptoGap(op, f"{code}: TINSERT writes the whole source tile, but the "
                                       f"destination window is {d['shape']} while the source is "
                                       f"{s['shape']}")
                for ln in pre:
                    self.emit(ln)
                self.emit(f"pl.insert({dp}, {sp}, [{dr0}, {dc0}])")
                for ln in post:
                    self.emit(ln)
                return
            ds0 = d.get("base_shape") or d.get("shape")
            ss0 = s.get("base_shape") or s.get("shape")
            if code == "dma.l1_to_l0":
                # device_lower already checked the padded source/allocation.
                # Carry its physical rows to TEXTRACT's destination validshape
                # while keeping the allocated tile capacity and pitch intact.
                m = op.attrs.get("m_copy", op.attrs["m_dst"])
                n = op.attrs["n_dst"]
                swap = bool(op.attrs.get("src_is_transpose")) != (str(op.attrs.get("dst_position")) == "l0b")
                physical = (n, m) if swap else (m, n)
                for value, capacity in zip(physical, ds0):
                    extent = self.env.fold(value)
                    if extent is not None and isinstance(capacity, int) and extent > capacity:
                        raise PyptoGap(op, "paired byte-transpose physical extent exceeds the PTO tile capacity", owner="ours")
                shape_text = "[" + ", ".join(self.iexpr(op, value) for value in physical) + "]"
                pre.append(f"pl.set_validshape({dp}, {shape_text})")
                post.append(f"pl.set_validshape({dp}, {list(ds0)})")
            if not off and not kw and ds0 and ss0 and list(ds0) != list(ss0):
                if (all(isinstance(x, int) for x in list(ds0) + list(ss0))
                        and all(a <= b for a, b in zip(ds0, ss0))
                        and d.get("space") in ("l0a", "l0b")):
                    off = ", offset=[0, 0]"  # narrower L0 dst: TEXTRACT's top-left window read
                else:
                    # a reversed-shape Mat -> Right move does NOT compile: TMov.hpp:640
                    # static-asserts equal declared Rows and Cols, and SFractal decides how
                    # the bytes are READ, never what shape they are declared at (D-106
                    # refuted D-103's reading on the board). The narrow-N / narrow-K B chain
                    # goes through the [K, N] ZN alias above instead.
                    raise PyptoGap(op, f"{code} between tiles declared {ss0} -> {ds0}: pto's TMOV "
                                       "statically asserts equal declared shapes, TEXTRACT wants the "
                                       "wide side as the source, TINSERT as the destination - this "
                                       "shape relation fits none of the three")
            for ln in pre:
                self.emit(ln)
            self.emit(f"pl.move({dp}, {sp}{off}{kw})")
            for ln in post:
                self.emit(ln)
            return
        if code in ("cube.mmad.mx", "cube.matmul_mx"):
            self.guard_attrs(op, {"M", "N", "K", "is_init", "dst_row0", "dst_col0",
                                  "dst_rows", "dst_cols", "phase"})
            dst, a_v, b_v = op.operands[:3]
            d = self.tile(op, dst)
            ta, tb = self.tile(op, a_v), self.tile(op, b_v)
            names = [self.mp.l0_scale.get(a_v.name), self.mp.l0_scale.get(b_v.name)]
            if not all(names):
                raise PyptoGap(op, "cube.mmad.mx whose L0 fragments carry no src_mx scale: "
                                   "pl.matmul_mx takes the scales as explicit operands")
            pubs = getattr(self, "mx_scale_py", {})
            if not all(n in pubs for n in names):
                raise PyptoGap(op, "cube.mmad.mx whose scales were never published into "
                                   "ScaleLeft/ScaleRight (no l1_to_l0.mx ran before it)")
            sa, sb = ({"py": pubs[names[0]]}, {"py": pubs[names[1]]})
            fn = "pl.matmul_mx" if op.attrs.get("is_init", True) else "pl.matmul_mx_acc"
            args = [d["py"]] + ([d["py"]] if fn.endswith("_acc") else []) \
                + [ta["py"], tb["py"], sa["py"], sb["py"]]
            self.emit(f"{fn}({', '.join(args)})")
            return
        if code == "cube.mmad":
            self.guard_attrs(op, {"M", "N", "K", "is_init", "dst_row0", "dst_col0", "dst_rows", "dst_cols", "bias"})
            dst, a, b = op.operands[:3]
            rec = self.tiles.get(dst.name)
            if rec is not None and "py" not in rec:
                # a window destination (split-N): pl.matmul has no offset, so the L0C column
                # strip materialises as its own tile (NZ keeps it contiguous); the dst_row0/
                # dst_col0 attrs repeat the window's offsets and are absorbed by it
                d = self._strip_tile(op, dst.name, rec)
            else:
                d = self.tile(op, dst)
                for key in ("dst_row0", "dst_col0"):
                    if (self.env.fold(op.attrs.get(key, 0)) or 0) != 0:
                        raise PyptoGap(op, f"cube.mmad with {key} != 0 (L0C sub-block writes) is a later phase")
            operands = []
            for value in (a, b):
                record = self.tiles.get(value.name)
                operands.append(self._strip_tile(op, value.name, record)
                                if record is not None and "py" not in record
                                else self.tile(op, value))
            ta, tb = operands
            # a TAIL matmul asks for less than the tiles hold (M, N, K attrs): pl.matmul reads
            # the declared shape, so the operands are narrowed with set_validshape around the
            # call and restored after - without it the tail iteration multiplies the padding
            mm = [self.env.fold(op.attrs.get(k)) if self.env.fold(op.attrs.get(k)) is not None
                  else op.attrs.get(k) for k in ("M", "N", "K")]
            pre_mm, post_mm = self._mm_narrow(op, [(d, (mm[0], mm[1])), (ta, (mm[0], mm[2])),
                                                   (tb, (mm[2], mm[1]))])
            for ln in pre_mm:
                self.emit(ln)
            bias = op.attrs.get("bias")
            if bias is not None:
                if not op.attrs.get("is_init", True):
                    raise PyptoGap(op, "cube.mmad bias on an accumulating tile: pl.matmul_acc "
                                       "has no bias channel (pl.matmul adds bias only with init)")
                tbias = self.tile(op, bias)
                self.emit(f"pl.matmul({d['py']}, {ta['py']}, {tb['py']}, {tbias['py']})")
            elif op.attrs.get("is_init", True):
                self.emit(f"pl.matmul({d['py']}, {ta['py']}, {tb['py']})")
            else:
                self.emit(f"pl.matmul_acc({d['py']}, {d['py']}, {ta['py']}, {tb['py']})")
            for ln in post_mm:
                self.emit(ln)
            return
        if code in ("sync.set_flag", "sync.wait_flag"):
            raw_flag(self, op)
            return
        if code in ("sync.set", "sync.wait"):
            ev = self.events.get(op.operands[0].name)
            if ev is None:
                raise PyptoGap(op, f"{code} on an event the printer has not seen")
            n = len(ev.ids)
            what = "set" if code == "sync.set" else "wait"
            if code == "sync.set":
                eid = f"{ev.table}[{ev.sc} % {n}]" if n > 1 else str(ev.ids[0])
                fnname, cnt = "sync_src", ev.sc
            else:
                eid = f"{ev.table}[{ev.wc} % {n}]" if n > 1 else str(ev.ids[0])
                fnname, cnt = "sync_dst", ev.wc
            bump = f"; {cnt} += 1" if n > 1 else ""
            self.emit(f"pl.system.{fnname}(set_pipe={PIPE[ev.set_pipe]}, wait_pipe={PIPE[ev.wait_pipe]}, "
                      f"event_id={eid}){bump}  # {py_ident(op.operands[0].name)}.{what}")
            return
        if code in ("sync.set_all", "sync.release"):
            ev = self.events.get(op.operands[0].name)
            if ev is None:
                raise PyptoGap(op, f"{code} on an event the printer has not seen")
            # set_all arms every slot (counters shift by a full depth: unchanged mod n);
            # release drains what nobody consumed - for autosync's balanced bodies that is
            # exactly the presets (the old transpiler's Release doctrine: a flag left set
            # lets a later, unrelated wait on a reused id fall straight through)
            n = len(ev.ids)
            reps = n if code == "sync.set_all" else int(ev.preset)
            fnname = "sync_src" if code == "sync.set_all" else "sync_dst"
            what = "set_all" if code == "sync.set_all" else "release"
            for k in range(reps):
                eid = str(ev.ids[k % n]) if code == "sync.set_all" else (
                    f"{ev.table}[{ev.wc} % {n}]" if n > 1 else str(ev.ids[0]))
                bump = f"; {ev.wc} += 1" if code == "sync.release" and n > 1 else ""
                self.emit(f"pl.system.{fnname}(set_pipe={PIPE[ev.set_pipe]}, "
                          f"wait_pipe={PIPE[ev.wait_pipe]}, event_id={eid}){bump}"
                          f"  # {py_ident(op.operands[0].name)}.{what} {k + 1}/{reps}")
            return
        if code in ("atomic.begin", "atomic.end", "atomic.set_type"):
            # pl has no atomic MODE. `pl.store` takes the mode per store (`atomic=`), and our
            # frontend already stamps it on every GM-destined copy inside the context
            # (frontend/rules_mem.py) - which is the SAME attr cce's own wrappers read
            # (_fixpipe_atomic / _atomic_wrap), and the only thing the simulator honours
            # (backends/sim/dma_ops.py no-ops all three markers). So the markers carry
            # nothing pl needs; they are tracked here only to catch a store that would
            # silently drop the mode, and to place `atomic.set_type` against the
            # destination dtype pl accumulates in.
            if code == "atomic.begin":
                self.atomic = str(op.attrs.get("op", "add"))
            elif code == "atomic.end":
                self.atomic = self.atomic_dtype = None
            else:
                self.atomic_dtype = str(op.attrs["dtype"])
            return
        if code == "sync.barrier":
            pipe = str(op.attrs.get("pipe", "ALL"))
            bar = BAR.get(pipe)
            if bar is None:
                raise PyptoGap(op, f"sync.barrier on pipe {pipe}: pl.system has no bar_{pipe.lower()} "
                                   "(bar_v is absent upstream)")
            self.emit(f"pl.system.{bar}()")
            return
        if code == "sync.mutex":
            # NOT a bare declaration (cce emit.py op_sync_mutex): the consumer side publishes
            # ``depth`` free tokens BEFORE the body - the first lock's credit - and the
            # producer side drains them at the end so no flag survives the launch. Dropping
            # this deadlocks the first cvmutex.lock on silicon (both sides waiting, D-088).
            kind = str(op.attrs.get("kind", "cv"))
            fid = self.ifold(op, op.attrs.get("id", 0), "mutex id")
            depth = int(op.attrs["depth"])
            consumer = "cube" if kind == "vc" else "vec"
            set_pipe = PIPE[str(op.attrs.get("dst_end_pipe", "FIX" if kind == "vc" else "MTE3"))]
            wait_pipe = PIPE[str(op.attrs.get("src_start_pipe", "S"))]
            if self.side == consumer:
                for _ in range(depth):
                    self.emit(f"pl.system.set_cross_core(pipe={set_pipe}, event_id={fid}, "
                              "sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)")
            else:
                self.mutex_drain.extend(
                    [f"pl.system.wait_cross_core(pipe={wait_pipe}, event_id={fid}, "
                     "sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)"] * depth)
            return
        if code.startswith("sync.crosscore."):
            kind = code[len("sync.crosscore."):]
            if kind == "allvec_ready":
                self._allvec_pending = op
                return
            if kind == "allvec_wait":
                if getattr(self, "_allvec_pending", None) is None:
                    raise PyptoGap(op, "sync.crosscore.allvec_wait without an adjacent allvec_ready "
                                       "(only the paired barrier form maps to pl.system.sync_all)")
                self._allvec_pending = None
                self.emit("pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)")
                return
            table = {"cube_ready": "set_cross_core", "vec_ready": "set_cross_core",
                     "wait_cube": "wait_cross_core", "wait_vec": "wait_cross_core",
                     "intracore_allvec_ready": "set_cross_core",
                     "intracore_allvec_wait": "wait_cross_core"}
            fnname = table.get(kind)
            if fnname is None:
                raise PyptoGap(op, f"{code} (the all-core family) is a later phase")
            pipe = str(op.attrs.get("pipe"))
            if pipe not in PIPE:
                raise PyptoGap(op, f"{code} on pipe {pipe} has no pl.PipeType")
            fid = self.ifold(op, op.attrs.get("flag_id", 0), "crosscore flag id")
            mode = "INTER_SUBBLOCK" if kind.startswith("intracore_allvec_") else "INTRA_BLOCK"
            self.emit(f"pl.system.{fnname}(pipe={PIPE[pipe]}, event_id={fid}, "
                      f"sync_mode=pl.CrossCoreSyncMode.{mode})")
            return
        if code == "cf.for":
            start, stop, step = (op.operands + [1])[:3] if len(op.operands) < 3 else op.operands[:3]
            if any(o.opcode.startswith("list.") for rg in op.regions for o in rg.walk()):
                # the member loop of a GMList: pl has no tensor-list parameter and no runtime
                # member index, so this loop is UNROLLED and each body copy names its own
                # member parameter (D-119). Its bounds fold - list.count is a literal.
                bs = [self.env.fold(x) for x in (start, stop, step)]
                if any(b is None for b in bs) or not bs[2]:
                    raise PyptoGap(op, "a GMList member loop whose bounds do not fold: the arity "
                                       "is part of the specialisation and has to be a constant")
                nm = op.results[0].name
                # every value the body DEFINES is re-evaluated per copy: ScalarEnv folds a
                # result once and then returns the cached binding, so member 1's extents would
                # otherwise be member 0's. Cells live outside the loop and keep their value.
                body = [x.name for rg in op.regions for o in rg.walk() for x in o.results]
                for i in range(int(bs[0]), int(bs[1]), int(bs[2])):
                    for x in body:
                        self.env.names.pop(x, None)
                    self.env.names[nm] = str(i)
                    self.block(op.regions[0])
                self.env.names.pop(nm, None)
                return
            var = self.env.define(op.results[0], op.attrs.get("name"))
            lo_txt, stop_txt = self.iexpr(op, start), self.iexpr(op, stop)
            # pto's loop codegen does not tolerate stop < start: a negative trip count runs
            # as a huge loop on silicon (board 507015 - the flat pfa's idle cores computed
            # my_work < 0 and hung the device). cce is immune (a signed C for-condition).
            # Clamp exactly the loops whose folded stop range can dip below the start.
            sr, tr = self.env.frange(start), self.env.frange(stop)
            if sr is None or tr is None or tr[0] < sr[1]:
                stop_txt = f"pl.max({stop_txt}, {lo_txt})"
            self.emit(f"for {var} in pl.range({lo_txt}, {stop_txt}, {self.iexpr(op, step)}):")
            self.indent += 1
            self.block(op.regions[0])
            self.indent -= 1
            return
        if code == "cf.if":
            # a condition the bindings decide is decided HERE: specialisation makes whole
            # arms unreachable, and printing an unreachable one asks the translation to hold
            # for geometry the kernel never runs (chunk_row_cumsum's H < 64 arm addresses a
            # [8, 64] pad tile with the H = 128 window of the arm that DOES run). cce is
            # immune - bisheng folds the same branch away after type-checking it.
            k = self.env.fold(op.operands[0])
            if k is not None:
                live = op.regions[0] if k else (op.regions[1] if len(op.regions) > 1 else None)
                if live is not None and live.ops:
                    self.block(live)
                return
            self.emit(f"if {self.env.ref(op.operands[0])}:")
            self.indent += 1
            self.block(op.regions[0])
            self.indent -= 1
            if len(op.regions) > 1 and op.regions[1].ops:
                self.emit("else:")
                self.indent += 1
                self.block(op.regions[1])
                self.indent -= 1
            return
        if code == "list.item":
            lst, idx = op.operands[:2]
            i = self.env.fold(idx)
            members = self.mp.lists.get(lst.name)
            if i is None or members is None or not (0 <= int(i) < len(members)):
                raise PyptoGap(op, f"list.item on {lst.name} at a member index that does not fold "
                                   "into this call's arity")
            self.params[op.results[0].name] = f"{py_ident(lst.name)}_{int(i)}"
            return
        if code == "cf.call":
            callee = op.attrs.get("callee") or (op.operands[0] if op.operands else None)
            raw = str(getattr(callee, "name", callee)).lstrip("@")
            name = fn_ident(raw)  # the renamed vf definition (I031, A5-UP-042)
            args = []
            arg_vals = [a for a in op.operands if not isinstance(a, FuncRef) and a is not callee]
            kept = self.mp.vf_parameters.get(raw)
            for i, a in enumerate(arg_vals):
                if kept is not None and i not in kept:
                    continue
                if isinstance(a, Value) and a.name in self.tiles:
                    rec = self.tiles[a.name]
                    if "py" not in rec:  # a window argument: the callee needs a real tile
                        rec = self._vf_window(rec) or self._strip_tile(op, a.name, rec)
                    args.append(rec["py"])
                elif isinstance(a, Value):
                    args.append(self.env.ref(a))
                elif isinstance(a, Literal) or isinstance(a, (int, float, bool)):
                    args.append(_lit(a))
                else:
                    raise PyptoGap(op, f"cf.call argument {a!r} has no pypto spelling")
            carriers = self.mp.vf_carriers.get(raw, ())
            for i, (pos, shape) in enumerate(carriers, len(arg_vals)):  # D-126
                if kept is not None and i not in kept:
                    continue
                if pos >= len(arg_vals):
                    raise PyptoGap(op, f"cf.call to {raw} passes {len(arg_vals)} arguments; its "
                                       f"uint8 carrier names parameter {pos}")
                args.append(self.u8_carrier(op, arg_vals[pos], shape))
            args += [name for i, name in enumerate(self.mp.vf_consts.get(raw, ()),
                                                   len(arg_vals) + len(carriers))
                     if kept is None or i in kept]  # D-134: exact-constant parameters
            self.emit(f"{name}({', '.join(args)})")
            return
        if code == "cf.return":
            return
        if code == "simt.launch":
            callee = getattr(op.operands[0], "name", str(op.operands[0]))
            py = self.mp.simt_names.get(callee)
            if py is None:
                raise PyptoGap(op, f"simt.launch of {callee}: callee not rendered")
            threads = op.attrs.get("threads")
            args = []
            for o in op.operands[1:]:
                t = getattr(o, "type", None)
                if isinstance(t, MemType) and t.space == "gm":
                    a = self.params.get(o.name)
                    if a is None:
                        raise PyptoGap(op, f"simt.launch arg {o.name}: GM tensor unknown here")
                    # "simt.launch Tensor argument must have a static shape" (board): the jit
                    # signature declares DYNAMIC dims, so re-describe the parameter statically
                    # through make_tensor with the binding-folded shape (hoisted, deduplicated)
                    dims = []
                    for dd in t.dims:
                        k = dd if isinstance(dd, int) else self.env.fold(dd)
                        if k is None:
                            raise PyptoGap(op, f"simt.launch arg {o.name}: dimension {dd!r} not "
                                               "static after binding")
                        dims.append(int(k))
                    strides = []
                    acc = 1
                    for dd in reversed(dims):
                        strides.insert(0, acc)
                        acc *= dd
                    if not hasattr(self, "_simt_static"):
                        self._simt_static = {}
                    key = (a, tuple(dims))
                    sv = self._simt_static.get(key)
                    if sv is None:
                        sv = self.mp.unique(py_ident(o.name) + "_s")
                        self.hoist(f"{sv} = pl.make_tensor({a}, {dims}, {strides})")
                        self._simt_static[key] = sv
                    args.append(sv)
                elif isinstance(t, MemType) and t.space == "ub":
                    args.append(self.tile(op, o)["py"])
                else:
                    a = self.env.ref(o)  # scalars AND cells - never drop an argument
                    dt = getattr(getattr(o, "type", None), "dtype", None)
                    if dt is not None and getattr(dt, "kind", None) == "float":
                        try:
                            a = repr(float(a))  # a folded int literal would type as int64
                        except (TypeError, ValueError):
                            pass  # runtime expression: already float-typed upstream
                    args.append(a)
            for e in self.mp.simt_extra.get(callee, []):
                if e == "cube_idx":
                    args.append("(pl.get_block_idx() // pl.get_subblock_num())"
                                if self.side == "vec" else "pl.get_block_idx()")
                elif e == "cube_num":
                    args.append(str(self.mp.block_dim))
            args.extend(f"pl.const({value}, {_pl_dt(dt)})"
                        for dt, value in self.mp.simt_constants.get(callee, ()))
            # the launch is a subscripted call parsed from THIS source (upstream 2b49dbfad replaced
            # pl.simt.launch); arguments are positional only (ir/op/simt_ops.py)
            self.emit(f"{py}[{threads}]({', '.join(args)})")
            return
        if code in ("debug.print", "debug.dump", "debug.print_reg", "debug.assert"):
            return  # observation-only debug ops have no device line (same policy as cce sim aids)
        raise PyptoGap(op, f"op {code} is outside the pypto surface",
                       owner="upstream" if code in _PROVEN_ABSENT else "unmapped")

    def op_sync_event(self, op: Op, preset_lines: list[str]) -> None:
        t = op.results[0].type
        ids = [int(i) for i in (op.attrs.get("ids") or [])]
        if not ids:
            raise PyptoGap(op, "sync.event without allocated ids (autosync must run first)")
        preset = op.attrs.get("preset", 0)
        if isinstance(preset, bool):
            preset = len(ids) if preset else 0
        preset = int(preset)
        set_pipe, wait_pipe = str(t.set_pipe), str(t.wait_pipe)
        if set_pipe not in PIPE or wait_pipe not in PIPE:
            raise PyptoGap(op, f"event pipes {set_pipe}->{wait_pipe} have no pl.PipeType")
        table = sc = wc = None
        if len(ids) > 1:
            # cce's Event<> template rotates ids on RUNTIME call counters inside .set()/.wait();
            # a printed loop body traces once, so a print-time rotation freezes one id and the
            # second iteration's wait starves (board: 507014). Reproduce the counters literally
            # (the old repository's transpiler doctrine, probe b07_eventid_probe): a module-level
            # id table survives to the CCE backend as a real array, the counters are top-level
            # Python locals (declared outside any branch - the DSL scopes branch-born names).
            # short numbered names keep the per-use sync lines on one screen line; the
            # IR event name travels in the declaration comment and on every use's tail
            n = self.mp.unique(f"_ev{len(self.events) + 1}")
            table, sc, wc = n, self.mp.unique(n + "s"), self.mp.unique(n + "w")
            stem = py_ident(op.results[0].name)
            preset_lines.append(f"{table} = {list(ids)}  # {stem}: {set_pipe} -> {wait_pipe}"
                                + (f", preset {preset}" if preset else ""))
            preset_lines.append(f"{sc} = {preset % len(ids)}")
            preset_lines.append(f"{wc} = 0")
        ev = _Event(set_pipe, wait_pipe, ids, preset, table, sc, wc)
        self.events[op.results[0].name] = ev
        for k in range(preset):
            preset_lines.append(f"pl.system.sync_src(set_pipe={PIPE[set_pipe]}, wait_pipe={PIPE[wait_pipe]}, "
                                f"event_id={ids[k % len(ids)]})"
                                f"  # {py_ident(op.results[0].name)}.arm {k + 1}/{preset}")


# ------------------------------------------------------------------ the module

DRIVER = '''\
#!/usr/bin/env python3
"""Board-side driver: read input/*.bin, run the generated pypto kernel on the NPU, write output/*.bin."""
import json
import os
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

import pypto

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)  # pypto JIT artifacts land in ./build relative to CWD

__SUPPLEMENT_PREFLIGHT__

# Before the generated module is even imported: this source may need a local PyPTO-Pro patch, and
# only this box can say whether the installed package has it. Refusing here names the patch;
# importing first would raise PyPTO's own error at the decorator, several layers from the cause.
_SPEC = json.loads((HERE / "input" / "manifest.json").read_text())
if supplement_preflight(_SPEC):
    print("PYPTO_RUN_SKIPPED")
    raise SystemExit(2)

import kernel_pypto  # noqa: E402  (the generated module)


def main() -> int:
    spec = _SPEC
    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)
    args = []
    for p in spec["params"]:
        raw = (HERE / "input" / (p["name"] + ".bin")).read_bytes()
        dt = getattr(torch, p["torch_dtype"])
        t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dt).reshape(p["shape"])
        args.append(t.to(device))
    ws_bytes = int(spec.get("workspace_bytes", 0))
    if ws_bytes:
        args.append(torch.zeros(ws_bytes, dtype=torch.uint8).to(device))
    # exact float constants: pypto's CCE printer renders a float literal with six decimals, so a
    # constant that needs more of them rides in as a trailing scalar parameter instead
    for c in spec.get("const_scalars", []):
        args.append(float(c["value"]))
    fn = getattr(kernel_pypto, spec["entry"])
    block_dim = spec.get("block_dim")
    # ASCRIPTOR_REPEAT > 1 is the PERFORMANCE mode: the same launch, N times, so a profiler
    # collects N task records to take a median over. Every iteration after the first restores
    # the device state from a pristine host copy, so an accumulating kernel computes the same
    # thing each time and the outputs written below are still the single-run outputs.
    reps = max(1, int(os.environ.get("ASCRIPTOR_REPEAT", "1") or 1))
    pristine = ([a.cpu().clone() if torch.is_tensor(a) else None for a in args]
                if reps > 1 else None)
    for r in range(reps):
        if r:
            for a, c in zip(args, pristine):
                if c is not None:
                    a.copy_(c)
        with pypto.options(pass_options={"enable_slice": False}):
            if block_dim:
                fn[None, int(block_dim)](*args)
            else:
                fn(*args)
        torch.npu.synchronize()
    out_dir = HERE / "output"
    out_dir.mkdir(exist_ok=True)
    by_name = {p["name"]: a for p, a in zip(spec["params"], args)}
    for name in spec["outputs"]:
        t = by_name[name].cpu().contiguous()
        (out_dir / (name + ".bin")).write_bytes(t.view(torch.uint8).numpy().tobytes())
    print("PYPTO_RUN_OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


class ModulePrinter:
    def __init__(self, module: Module, block_dim: int | None = None, entry: str | None = None,
                 bindings: dict[str, int] | None = None, max_block_dim: int | None = None,
                 lists: dict[str, list[list[int]]] | None = None,
                 shapes: dict[str, list[int]] | None = None, sync_mode: str = "manual",
                 supplements: Iterable[str] | str | None = None,
                 index_spelling: str | None = None) -> None:
        validate_mode(sync_mode, PyptoGap)
        self.index_spelling = index_spelling or os.environ.get(INDEX_ENV, "direct")
        if self.index_spelling not in ("direct", "named"):
            raise ValueError(f"{INDEX_ENV} must be direct or named, got {self.index_spelling!r}")
        self.sync_mode = sync_mode
        # The local PyPTO-Pro patches the printed lines turn out to need. None is installed by
        # default; `supplements` (or ASCRIPTOR_PYPTO_SUPPLEMENTS) is the caller stating that the
        # installation this source will run on already carries one, which silences its warning
        # and its board-side probe. `supplements.py` owns the mapping.
        self.supplements = SupplementUses(declared_supplements(supplements))
        self.scalar_ranges = {fn.name: ScalarRanges(fn) for fn in module.functions}
        self.native_mutex_map: list[dict] = []
        self.scalar_cleanup_report: dict[str, dict] = {}
        if module.level != "lowered":
            raise PyptoGap(None, f"the pypto_pro backend prints Lowered modules only (got {module.ir!r})")
        self.module = module
        meta = dict(module.attrs.get("meta", {}))
        self.kernel = str(meta.get("kernel", module.name))
        self.entry_name = py_ident(entry or self.kernel)
        self.max_block_dim = max_block_dim
        # ir_name -> the shape the caller will actually pass. The backend already specialises a
        # translation to the call (its scalars, a GMList's arity), so the tensor shapes are known
        # too; declaring them instead of pl.DYNAMIC is what lets pto fold the GM descriptor
        # (D-160). Absent on the signature-only manifest, where there is no call yet.
        self.shapes = dict(shapes or {})
        # vf name -> [(tile parameter position, carrier shape)]: the uint8 carriers its callers
        # have to pass, filled in by VfPrinter and read at every cf.call (D-126)
        self.vf_carriers: dict[str, list[tuple[int, list[int]]]] = {}
        self.vf_parameters: dict[str, frozenset[int]] = {}  # positions in the expanded signature
        # D-134: a float immediate that pypto's six-decimal `std::to_string` would corrupt rides
        # in as a jit SCALAR PARAMETER instead. pl prints a parameter by NAME, so the value
        # reaches the board exactly and the op keeps its `vmuls`/`vadds` scalar form - a register
        # materialised from the bit pattern was exact too, but cost a `vbr` and the vector-vector
        # form. Keyed by (value, register dtype); `vf_consts` is what each callee must be handed.
        self.const_scalars: dict[tuple[float, str], str] = {}
        self.vf_consts: dict[str, list[str]] = {}
        self.u8_groups: dict[tuple[str, tuple[int, ...]], str] = {}  # parallel uint8 tile groups
        # a GMList parameter's members, as THIS call passes them: pl has no tensor-list
        # parameter (behaviour #25), so the arity and the member shapes join the shape and
        # scalar bindings this translation is already specialised for (D-119)
        self.lists: dict[str, list[list[int]]] = dict(lists or {})
        self.mode = str(module.attrs.get("mode", "mix"))
        self.device = module.device
        self.outputs = [v.name if isinstance(v, Value) else str(v) for v in meta.get("outputs", [])]
        from ...runtime.launch_config import launch_block_dim

        block_dim = launch_block_dim(module, block_dim, "the PyPTO Pro printer")
        self.block_dim = block_dim if block_dim is not None else meta.get("block_dim")
        pinned = self.block_dim is not None  # an explicit launch, from the case's own manifest
        if self.block_dim is None:
            # An unspecified launch fills every core. WHICH core count is the one the cce path
            # uses, because that is the only way the two are the same measurement AND the only
            # way this path uses the machine it is on: the generated cce project asks the
            # platform (`SetBlockDim(ascendcPlatform.GetCoreNumAic())`), so an Ascend950PR card
            # gets 28 there. `max_block_dim` is that card count when the runner knows it; the
            # device profile (32 for a5) is the fallback for a run that does not - the
            # interpreter and the recorded goldens. A PINNED block_dim is left alone: the cce
            # project bakes the same literal (`SetBlockDim(32)`), so both paths already agree.
            if max_block_dim:
                self.block_dim = int(max_block_dim)
            else:
                try:
                    from ascriptor.devices import load as _load_profile
                    self.block_dim = int(_load_profile(self.device).cube_cores)
                except Exception:  # noqa: BLE001 - an unknown device just stays dynamic
                    pass
            # An AIV-ONLY binary has no sub-block layer at all - board-probed on this pl:
            # launched <<<4>>>, every lane reports block_num 4, subblock_num 1, subblock_idx 0,
            # and block_idx is the AIV index. So `block_dim` there counts AIVs, `cube_idx` and
            # `vec_idx` are the same number, and the participant count IS the launch. The cce
            # project says the same thing from its side: a vec-only op's tiling function is
            # `SetBlockDim(ascendcPlatform.GetCoreNumAiv())`, i.e. 2x the AIC count.
            if self.mode == "vec" and self.block_dim:
                self.block_dim = int(self.block_dim) * 2
        # A PINNED launch past the card's own core count is no longer a launch: pl rejects a
        # block_dim above the stream's budget (upstream 859743bb4, 2026-09-21), and an all-vec
        # hardware barrier deadlocked on it before that (pto's sync_all waits for every core the
        # launch names; board: simt_atomic_add hung for nine minutes at 64 AIVs on a 56-AIV card).
        # The second wave that over-subscription used to buy is gone, so clamp every pinned launch
        # to the card - and clamp HERE rather than at the launch, so the folded core.cube_num /
        # core.vec_num describe the same split the launch runs.
        if pinned and max_block_dim and self.block_dim and self.block_dim > int(max_block_dim):
            self.block_dim = launch_block_dim(module, int(max_block_dim), "the card core-count clamp")
        self.bindings = dict(bindings or {})
        self._names = LocalNames({"pl", "vf", "P", "pypto", self.entry_name} |
                                 {py_ident(f.name) for f in module.functions if f.kind in ("vf", "simt")})
        self._tiletypes: dict[tuple[str, tuple[int, ...], str], str] = {}
        self.dn_params: set[str] = set()  # GM params loaded through the DN2ZN path (transposed B)
        # MX scale chains. The two backends carry the per-group scale differently: OUR IR
        # (cce's model) hangs it on the l1_to_l0.mx move as a `src_mx` attribute - the scale
        # rides into L0 with the data and mad_mx reads it there - while pto makes it an
        # explicit operand of pl.matmul_mx, read from its own ScaleLeft/ScaleRight stop
        # ("scale tiles use E8M0 in ScaleLeft/ScaleRight", _api.py:669). So: collect each
        # move's src_mx, remember which side it feeds, and hand it to the matmul.
        self.scale_side: dict[str, str] = {}   # scale tile name -> memory space key
        self.scale_logical: dict[str, list] = {}  # its [rows, k_groups] shape before staging
        self.scale_addr: dict[tuple, int] = {}    # one address per (scale, side) in its stop
        self._scale_fill: dict[str, str] = {}     # scale tile -> the value its GM load reads
        self.scale_nd_gm: dict[str, list] = {}    # dense-route scale param -> its rank-3 pto shape
        self.l0_scale: dict[str, str] = {}     # L0 fragment name -> its scale tile name
        def scan(ops: Any) -> None:
            for o in ops:
                if o.opcode.startswith("dma.gm_to_l1") and len(o.operands) >= 2:
                    self._scale_fill[o.operands[0].name] = o.operands[1].name
                if o.opcode == "dma.l1_to_l0.mx":
                    mx = o.attrs.get("src_mx")
                    nm = getattr(mx, "name", mx)
                    if nm:
                        side = "scale_r" if str(o.attrs.get("dst_position")) == "l0b" else "scale_l"
                        self.scale_side[str(nm)] = side
                        if o.operands:
                            self.l0_scale[o.operands[0].name] = str(nm)
                for r in o.regions:
                    scan(r.ops)

        for f in module.functions:
            scan(f.body.ops)
        for f in module.functions:
            for op in reversed(list(f.walk())):
                if op.opcode == "mem.get_buf" and op.results[0].name in self.scale_side:
                    self.scale_side[op.operands[0].name] = self.scale_side[op.results[0].name]
        # (the block route stages pre-packed 32-byte blocks and loads them as plain bytes;
        # only the dense route hands pto a scale TENSOR, and those params are rewritten below)

    def unique(self, n: str) -> str:
        return self._names.unique(n)

    def tiletype(self, dtype_name: str, shape: list[int], space: str, layout: str | None = None,
                 pad: str | None = None) -> str:
        lay = f", layout={layout}" if layout else ""
        pd = f", pad={pad}" if pad else ""
        return (f"pl.TileType(shape={shape}, dtype={_pl_dt(dtype_name)}, "
                f"target_memory={MEMSPACE[space]}{lay}{pd})")

    WS_NAME = "ascriptor_ws"

    def _fold_scalar(self, x: Any) -> int | None:
        if isinstance(x, Literal):
            x = x.value
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, Value):
            return self.bindings.get(x.name)
        return None

    def workspace_bytes(self) -> int:
        """Total launcher workspace the module needs: max(offset + numel * esize) over every
        ``mem.workspace`` (offsets are bytes, numel elements - the HostSpec convention). Only
        static / binding-folded quantities print; anything else is a gap."""
        total = 0
        for f in self.module.functions:
            if f.kind != "func":
                continue
            for op in f.body.walk():
                if op.opcode != "mem.workspace":
                    continue
                numel = self._fold_scalar(op.attrs.get("numel"))
                offset = self._fold_scalar(op.attrs.get("offset", 0))
                if numel is None or offset is None:
                    raise PyptoGap(op, f"mem.workspace {op.attrs.get('name')}: numel/offset do not fold to "
                                       "compile-time ints (a GMBuff ring or derived sizing is a later phase)")
                rt = op.results[0].type
                elem = rt.elem if isinstance(rt, BufType) else rt
                assert isinstance(elem, MemType)
                total = max(total, offset + numel * (max(elem.dtype.bits, 8) // 8))
        return total

    def signature_params(self) -> list[dict[str, Any]]:
        fn = next(f for f in self.module.functions if f.kind == "func")
        params = []
        for p in fn.params:
            t = p.type
            if isinstance(t, MemType) and t.space == "gm":
                params.append({"name": py_ident(p.name), "ir_name": p.name, "kind": "tensor",
                               "dtype": t.dtype.name, "dims": ["?"] * len(t.dims),
                               "shape": self.shapes.get(p.name),
                               "output": p.name in self.outputs})
            elif isinstance(t, ScalarType):
                params.append({"name": py_ident(p.name), "ir_name": p.name, "kind": "scalar",
                               "dtype": t.dtype.name})
            elif isinstance(t, MemType) and t.space == "gmlist":
                # pl has NO tensor-list parameter, and a member pointer read out of a descriptor
                # can never become a tensor (pl.make_ptr refuses a ScalarType - board, and the
                # pypto-gym limitations note says the same). Its own workaround is a fixed
                # maximum arity padded with unused slots; this backend already specialises every
                # translation to the call's shapes and scalars, so the arity is simply THIS
                # call's - no padding, no unused slots to execute (D-119).
                members = self.lists.get(p.name)
                if not members:
                    # the signature-only manifest (module_manifest): no call, so no members yet.
                    # It keeps the parameter as a LIST, which is what the host spec hands the
                    # runtime so it can pass the member shapes back in for the real translation.
                    params.append({"name": py_ident(p.name), "ir_name": p.name, "kind": "list",
                                   "dtype": t.dtype.name, "dims": ["?"] * len(t.dims),
                                   "output": p.name in self.outputs})
                    continue
                for i, shape in enumerate(members):
                    params.append({"name": f"{py_ident(p.name)}_{i}", "ir_name": f"{p.name}#{i}",
                                   "kind": "tensor", "dtype": t.dtype.name, "dims": ["?"] * len(shape),
                                   "shape": [int(x) for x in shape],
                                   "output": p.name in self.outputs, "member_of": p.name,
                                   "member_shape": [int(x) for x in shape]})
            else:
                raise PyptoGap(None, f"parameter {p.name} of type {t}: not in the pypto surface")
        return params

    def _shards_by_vec(self) -> bool:
        """Whether the launch has to be doubled for an AIV-only binary. Our IR counts CUBE cores
        and models a vec-sharded kernel as running over GetVecNum() == 2 * block_dim
        participants, so such a kernel needs 2 * block_dim AIVs. A kernel that never asks which
        vec participant it is just wants block_dim copies, and doubling it doubles its WORK -
        board-proven on simt_atomic_incdec, whose ring atomics landed twice (2 instead of 4 per
        column) until the launch was left alone."""
        for f in self.module.functions:
            for o in f.body.walk():
                if o.opcode in ("core.vec_idx", "core.vec_num",
                                "core.sub_block_idx", "core.sub_block_num"):
                    return True  # asking which SUB-BLOCK you are is asking which vec participant
        return False

    def manifest(self) -> dict[str, Any]:
        params = self.signature_params()
        for p in params:
            if p["kind"] == "tensor" and TORCH_DT.get(p["dtype"]) is None:
                raise PyptoGap(None, f"parameter {p['name']}: dtype {p['dtype']} has no torch spelling")
        ws_bytes = self.workspace_bytes()
        if ws_bytes:
            params = [*params, {"name": self.WS_NAME, "ir_name": self.WS_NAME, "kind": "workspace",
                                "dtype": "u8", "dims": ["?"], "shape": [int(ws_bytes)],
                                "output": False}]
        return {
            "backend": "pypto_pro", "kernel": self.entry_name, "ir_kernel": self.kernel,
            "device": self.device, "mode": self.mode,
            # launch dim: pto's <<<blockDim>>> counts AICs for a mix binary (each brings its
            # two AIVs along) but counts AIVs DIRECTLY for an AIV-only binary. For a vec-only
            # module `block_dim` was already doubled into an AIV count at construction, and the
            # folds agree with it, so the launch is simply that number - the same count the cce
            # project asks for with `SetBlockDim(GetCoreNumAiv())`.
            "block_dim": self.block_dim,
            "params": params, "outputs": [n for o in self.outputs
                                          for n in ([f"{py_ident(o)}_{i}" for i in range(len(self.lists[o]))]
                                                    if o in self.lists else [py_ident(o)])],
            "workspace_bytes": ws_bytes,
            "workspaces": [], "sides": sorted({str(f.attrs.get("side", "vec"))
                                               for f in self.module.functions if f.kind == "func"}),
            "vf": [fn_ident(f.name) for f in self.module.functions if f.kind == "vf"],
            "entry": "kernel_pypto.py",
            "pypto": {"torch_dtypes": {p["name"]: TORCH_DT.get(p["dtype"]) for p in params
                                       if p["kind"] == "tensor"},
                      # a dense MX scale reaches pto as a rank-3 [rows, k_pairs, 2] tensor -
                      # same bytes, but pto's scale load insists on the trailing phase axis
                      "reshape": {p["name"]: self.scale_nd_gm[p["ir_name"]] for p in params
                                  if p["ir_name"] in self.scale_nd_gm},
                      # the local dependency patches this source needs, each with the probe the
                      # driver runs against the installed package (supplements.py); only known
                      # once the body has printed, so `compile` refreshes it
                      "supplements": self.supplements.entries(),
                      "bindings": self.bindings},
        }

    def compile(self) -> Artifacts:
        sides: dict[str, Function] = {}
        vfs: list[Function] = []
        simts: list[Function] = []
        for f in self.module.functions:
            if f.kind == "func":
                sides[str(f.attrs.get("side", "vec"))] = f
            elif f.kind == "vf":
                vfs.append(f)
            elif f.kind == "simt":
                simts.append(f)
            else:
                raise PyptoGap(None, f"function {f.name} of kind {f.kind}: not in the pypto surface")
        # per-callee max launch threads decides max_threads= on the pl.simt.function;
        # folded scalar launch arguments bind the callee's own parameter names so its
        # declared GM shapes specialise (the kernel binds BINS, the callee spells bins)
        launch_threads: dict[str, int] = {}
        callee_binds: dict[str, dict[str, int]] = {}
        by_name = {f.name: f for f in simts}
        for f in sides.values():
            env = ScalarEnv(self, f, str(f.attrs.get("side", "vec")))
            for q in f.params:
                if isinstance(q.type, ScalarType) and q.name in (self.bindings or {}):
                    v = self.bindings[q.name]
                    env.names[q.name] = repr(int(v) if float(v).is_integer() else float(v))
            for o in f.body.walk():
                if o.opcode == "simt.launch":
                    callee = getattr(o.operands[0], "name", str(o.operands[0]))
                    t = o.attrs.get("threads")
                    if not isinstance(t, int):
                        raise PyptoGap(o, "simt.launch threads must be a compile-time int")
                    launch_threads[callee] = max(launch_threads.get(callee, 0), t)
                    cal = by_name.get(callee)
                    if cal is None:
                        continue
                    bs = callee_binds.setdefault(callee, {})
                    for q, a in zip(cal.params, o.operands[1:]):
                        if not isinstance(q.type, ScalarType):
                            continue
                        k = env.fold(a)
                        if k is None or (q.name in bs and bs[q.name] != k):
                            bs.pop(q.name, None)
                        else:
                            bs[q.name] = k
        self.simt_names: dict[str, str] = {}
        self.simt_extra: dict[str, list[str]] = {}
        self.simt_constants: dict[str, list[tuple[str, int]]] = {}
        simt_texts: list[str] = []
        for f in simts:
            t = launch_threads.get(f.name)
            if t is None:
                continue  # a helper never launched from this module: nothing to print
            em = SimtEmitter(self, f, t, callee_binds.get(f.name))
            py, text = em.render()
            self.simt_names[f.name] = py
            self.simt_extra[f.name] = em.extra
            self.simt_constants[f.name] = list(em.constants)
            simt_texts.append(text)
        if not set(sides) <= {"vec", "cube"}:
            raise PyptoGap(None, f"sides {sorted(sides)} are outside the pypto surface")

        meta = self.manifest()
        params = meta["params"]
        for p in params:
            if p["kind"] == "scalar" and p["ir_name"] not in self.bindings:
                raise PyptoGap(None, f"scalar parameter {p['ir_name']} has no binding; the pypto "
                                     "backend specialises kernels per scalar valuation")

        vf_texts = [VfPrinter(self, f).render() for f in vfs]
        self._names.reset({p["name"] for p in params} | set(self.const_scalars.values()))
        section_bodies: list[tuple[str, list[str]]] = []
        for side in ("cube", "vec"):
            if side in sides:
                section_bodies.append((side, SideEmitter(self, sides[side]).render()))
        def _dims(p: dict[str, Any]) -> list[str]:
            # the MX scale's trailing phase axis must be STATICALLY 2 ("MX scale load trailing
            # physical phase axis must be statically equal to 2") - the other two stay dynamic,
            # because the tensor the driver passes is the rank-3 reshape, not this shape
            if p["ir_name"] in self.scale_nd_gm:
                return ["pl.DYNAMIC", "pl.DYNAMIC", "2"]
            # D-160: a GM parameter declared at its real shape lets pto bake the strides into the
            # GlobalTensor instead of carrying `__pypto_dyn_*` into every TASSIGN, which is
            # per-load scalar work OUTSIDE the vec scope (D-149's pipe). Board: cov2x2_inverse
            # 2.114 -> 1.789 us, the l10 probe 1278.6 -> 1244.3 us, both bit-identical.
            s = p.get("shape")
            if s is not None and len(s) == len(p["dims"]):
                return [str(int(d)) for d in s]
            return ["pl.DYNAMIC"] * len(p["dims"])

        sig = ", ".join(
            f"{p['name']}: pl.Tensor[[{', '.join(_dims(p))}], "
            f"{_pl_dt('e8m0' if p['ir_name'] in self.scale_nd_gm else p['dtype'])}"
            f"{', pl.DN' if p['ir_name'] in self.dn_params else ''}]"
            for p in params if p["kind"] in ("tensor", "workspace"))
        if self.const_scalars:  # D-134: trailing scalar parameters, after ascriptor_ws
            sig += (", " if sig else "") + ", ".join(
                f"{n}: {_pl_dt(dt)}" for (_v, dt), n in self.const_scalars.items())
        src = ["# Generated by the ascriptor pypto_pro backend - do not edit.",
               ("# Manual-sync translation: every pl.system.* below is an autosync/device_lower decision."
                if self.sync_mode == "manual" else "# IR-owned Tile mutex IDs; PyPTO inserts local locks. Other IR synchronization is preserved."),
               f"# Specialised for: {json.dumps(self.bindings) if self.bindings else '(no scalar parameters)'}",
               *self.supplements.header(),
               "import pypto  # noqa: F401",
               "import pypto_pro.language as pl",
               "from pypto_pro.language import Vf as vf  # noqa: N813",
               "P = pl.PipeType  # pipe alias for the sync lines",
               ""]
        src += vf_texts
        src += simt_texts
        src += [f"@pl.jit(auto_mutex={self.sync_mode == 'auto_mutex'})", f"def {self.entry_name}({sig}):"]
        for side, body in section_bodies:
            src.append(f"    with pl.section_{'cube' if side == 'cube' else 'vector'}():")
            src += [f"        {ln}" for ln in body]
            if not any(ln.strip() and not ln.lstrip().startswith("#") for ln in body):
                src.append("        pass")
        if not section_bodies:
            src.append("    pass")
        src.append("")
        # the dense MX scale params only become known while the body prints, which is after
        # manifest() ran - fold their rank-3 pto shape in before the metadata is published
        meta["pypto"]["reshape"] = {p["name"]: self.scale_nd_gm[p["ir_name"]] for p in params
                                    if p.get("ir_name") in self.scale_nd_gm}
        # D-134: the exact-constant parameters only become known while the body prints. They are
        # trailing arguments of the launch, after the workspace, in declaration order.
        meta["pypto"]["const_scalars"] = [{"name": n, "dtype": dt, "value": float(v)}
                                          for (v, dt), n in self.const_scalars.items()]
        meta["pypto"].update(sync_mode=self.sync_mode, mutex_map=self.native_mutex_map,
                             mutex_id_source="lowered_ir", scalar_cleanup=self.scalar_cleanup_report)
        # which local dependency patches the printed lines actually needed - known only now
        meta["pypto"]["supplements"] = self.supplements.entries()
        files = {
            "kernel_pypto.py": "\n".join(src).encode(),
            "run_case.py": DRIVER.replace("__SUPPLEMENT_PREFLIGHT__", DRIVER_PREFLIGHT.strip()).encode(),
            "manifest.json": (json.dumps(meta, indent=1) + "\n").encode(),
        }
        # Last, so a failing translation reports its gap rather than a patch it never reached.
        self.supplements.warn(self.entry_name)
        return Artifacts(files=files, entry="kernel_pypto.py", metadata=meta)


def module_manifest(module: Module, block_dim: int | None = None, entry: str | None = None) -> dict[str, Any]:
    """The HostSpec-compatible manifest alone — no body printing, no bindings needed."""
    return ModulePrinter(module, block_dim=block_dim, entry=entry).manifest()


def emit_module(module: Module, block_dim: int | None = None, entry: str | None = None,
                bindings: dict[str, int] | None = None, max_block_dim: int | None = None,
                lists: dict[str, list[list[int]]] | None = None,
                shapes: dict[str, list[int]] | None = None, sync_mode: str = "manual",
                supplements: Iterable[str] | str | None = None,
                index_spelling: str | None = None) -> Artifacts:
    from ...passes import PassManager
    from ...passes.pypto_register_init import PASS_DEF
    from ...passes.integer_division import PASS_DEF as INTEGER_DIVISION
    from ...passes.pypto_l0_staging import PASS_DEF as L0_STAGING

    validate_mode(sync_mode, PyptoGap)
    from .scalar_specialize import specialize
    module = specialize(ModulePrinter(module, block_dim, entry, bindings, max_block_dim, lists, shapes, sync_mode,
                                      index_spelling=index_spelling))
    ranges = {fn.name: ScalarRanges(fn) for fn in module.functions}
    # `//` and `%` are floor for signed integers in every PyPTO this targets (upstream 5866c9b6f,
    # 2026-09-14, is an ancestor of the 289942aa3 minimum), so the correction sequence is the
    # backend's own and expanding it here would apply it twice. Ceiling and alignment still expand.
    module = PassManager((L0_STAGING, INTEGER_DIVISION, PASS_DEF),
                         options={"native_floor_divmod": True}).run(module)
    printer = ModulePrinter(module, block_dim=block_dim, entry=entry, bindings=bindings,
                         max_block_dim=max_block_dim, lists=lists, shapes=shapes, sync_mode=sync_mode,
                         supplements=supplements, index_spelling=index_spelling)
    printer.scalar_ranges = ranges
    return printer.compile()
