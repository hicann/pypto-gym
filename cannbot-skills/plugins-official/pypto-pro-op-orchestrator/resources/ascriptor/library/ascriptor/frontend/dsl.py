# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The DSL surface (RFC-0002 §2): the names a kernel author imports through a device facade.

Nothing here executes kernel code. The classes and functions are *markers*: the AST compiler
recognises them by identity (``_asc_rule``) when it statically evaluates the callee of a call, and
compiles the call to IR ops. Only ``CastConfig`` and the enum-like values are ordinary static
objects. The names keep their ``easyasc`` spellings (D-008): ``DT.float``, ``Position.L1``,
``Tensor``, ``DBuff``, ``Var``, ``Reg``, ``MaskReg``, ``VcMutex``, ``auto_sync``, ``matmul``, ...
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# --------------------------------------------------------------------------- dtypes
from ..dtypes import *  # noqa: F401,F403 - host-side codecs the old facade exported
from ..ir.types import DTYPES, DType


@dataclass(frozen=True)
class DTypeName:
    """A dtype as the DSL names it; ``ir`` is the IR dtype it denotes."""

    name: str
    ir: DType

    def __repr__(self) -> str:
        return f"DT.{self.name}"

    @property
    def bits(self) -> int:
        return self.ir.bits

    @property
    def size(self) -> int:
        """Bytes per element (the old ``Datatype.size``); packed 4-bit dtypes have none."""
        if self.ir.bits < 8:
            raise AttributeError(f"{self} is a packed dtype without a byte size")
        return self.ir.bits // 8

    @property
    def C0(self) -> int:  # noqa: N802 - the old spelling
        """Elements per 32-byte block (the old ``Datatype.C0``)."""
        return 64 if self.ir.bits < 8 else 32 // (self.ir.bits // 8)


class _DT:
    """``DT.<name>``: every old easyasc dtype name plus the neutral IR spellings (D-008 #8)."""

    half = DTypeName("half", DTYPES["f16"])
    float16 = DTypeName("float16", DTYPES["f16"])
    float = DTypeName("float", DTYPES["f32"])
    float32 = DTypeName("float32", DTYPES["f32"])
    int = DTypeName("int", DTYPES["i32"])
    int32 = DTypeName("int32", DTYPES["i32"])
    int8 = DTypeName("int8", DTYPES["i8"])
    uint8 = DTypeName("uint8", DTYPES["u8"])
    int16 = DTypeName("int16", DTYPES["i16"])
    uint16 = DTypeName("uint16", DTYPES["u16"])
    uint32 = DTypeName("uint32", DTYPES["u32"])
    int64 = DTypeName("int64", DTYPES["i64"])
    uint64 = DTypeName("uint64", DTYPES["u64"])
    bfloat16 = DTypeName("bfloat16", DTYPES["bf16"])
    e4m3 = DTypeName("e4m3", DTYPES["e4m3"])
    e5m2 = DTypeName("e5m2", DTYPES["e5m2"])
    mx_e4m3 = e4m3
    mx_e5m2 = e5m2
    hif8 = DTypeName("hif8", DTYPES["hif8"])
    fp4_e2m1 = DTypeName("fp4_e2m1", DTYPES["fp4_e2m1"])
    fp4_e1m2 = DTypeName("fp4_e1m2", DTYPES["fp4_e1m2"])
    e8m0 = DTypeName("e8m0", DTYPES["e8m0"])
    int4 = DTypeName("int4", DTYPES["i4"])
    bool = DTypeName("bool", DTYPES["b1"])
    complex32 = DTypeName("complex32", DTYPES["c32"])
    complex64 = DTypeName("complex64", DTYPES["c64"])
    # neutral spellings
    f16 = half
    bf16 = bfloat16
    f32 = float
    i8 = int8
    i4 = int4
    u8 = uint8
    i16 = int16
    u16 = uint16
    i32 = int
    u32 = uint32
    i64 = int64
    u64 = uint64
    b1 = bool


DT = _DT()
f16, bf16, f32 = DT.half, DT.bfloat16, DT.float
i8, u8, i16, u16, i32, u32, i64, u64, b1 = DT.int8, DT.uint8, DT.int16, DT.uint16, DT.int, DT.uint32, DT.int64, DT.uint64, DT.bool

# --------------------------------------------------------------------------- enum-like values


@dataclass(frozen=True)
class EnumValue:
    family: str
    name: str

    def __repr__(self) -> str:
        return f"{self.family}.{self.name}"


class Position:
    GM = EnumValue("Position", "gm")
    L1 = EnumValue("Position", "l1")
    L0A = EnumValue("Position", "l0a")
    L0B = EnumValue("Position", "l0b")
    L0C = EnumValue("Position", "l0c")
    UB = EnumValue("Position", "ub")
    BT = EnumValue("Position", "bt")


class Pipe:
    MTE1 = EnumValue("Pipe", "MTE1")
    MTE2 = EnumValue("Pipe", "MTE2")
    MTE3 = EnumValue("Pipe", "MTE3")
    M = EnumValue("Pipe", "M")
    V = EnumValue("Pipe", "V")
    FIX = EnumValue("Pipe", "FIX")
    S = EnumValue("Pipe", "S")
    ALL = EnumValue("Pipe", "ALL")


class Layout:
    NZ = EnumValue("Layout", "nz")
    ND = EnumValue("Layout", "nd")
    DEFAULT = EnumValue("Layout", "default")


class RoundMode:
    NONE = EnumValue("RoundMode", "none")
    TO_EVEN = EnumValue("RoundMode", "rint")
    AWAY_FROM_ZERO = EnumValue("RoundMode", "round")
    FLOOR = EnumValue("RoundMode", "floor")
    CEIL = EnumValue("RoundMode", "ceil")
    TRUNC = EnumValue("RoundMode", "trunc")
    ODD = EnumValue("RoundMode", "odd")
    HYBRID = EnumValue("RoundMode", "hybrid")


class RegLayout:
    UNKNOWN = EnumValue("RegLayout", "unknown")
    ZERO = EnumValue("RegLayout", "zero")
    ONE = EnumValue("RegLayout", "one")
    TWO = EnumValue("RegLayout", "two")
    THREE = EnumValue("RegLayout", "three")


class MaskMergeMode:
    ZEROING = EnumValue("MaskMergeMode", "zeroing")
    MERGING = EnumValue("MaskMergeMode", "merging")


@dataclass(frozen=True)
class CastConfig:
    """A static description of a register cast; consumed by ``cast(...)`` / ``reg.astype(...)``."""

    round_mode: EnumValue = RoundMode.TO_EVEN
    reg_layout: EnumValue = RegLayout.ZERO
    saturate: bool = False
    name: str = ""
    merge_mode: EnumValue = MaskMergeMode.ZEROING


class MaskType:
    """Initial pattern of a ``MaskReg``."""

    ALL = EnumValue("MaskType", "all")
    NONE = EnumValue("MaskType", "none")
    LOWEST1 = EnumValue("MaskType", "vl1")
    LOWEST2 = EnumValue("MaskType", "vl2")
    LOWEST3 = EnumValue("MaskType", "vl3")
    LOWEST4 = EnumValue("MaskType", "vl4")
    LOWEST8 = EnumValue("MaskType", "vl8")
    LOWEST16 = EnumValue("MaskType", "vl16")
    LOWEST32 = EnumValue("MaskType", "vl32")
    LOWEST128 = EnumValue("MaskType", "vl128")
    LOWHALF = EnumValue("MaskType", "h")
    LOWQUAT = EnumValue("MaskType", "q")
    MULTI3 = EnumValue("MaskType", "m3")
    MULTI4 = EnumValue("MaskType", "m4")


class CompareMode:
    LT = EnumValue("CompareMode", "lt")
    LE = EnumValue("CompareMode", "le")
    GT = EnumValue("CompareMode", "gt")
    GE = EnumValue("CompareMode", "ge")
    EQ = EnumValue("CompareMode", "eq")
    NE = EnumValue("CompareMode", "ne")


class VfPipe:
    STORE = EnumValue("VfPipe", "vec_store")
    LOAD = EnumValue("VfPipe", "vec_load")
    SCALAR_STORE = EnumValue("VfPipe", "scalar_store")
    SCALAR_LOAD = EnumValue("VfPipe", "scalar_load")
    VEC_ALL = EnumValue("VfPipe", "vec_all")
    SCALAR_ALL = EnumValue("VfPipe", "scalar_all")


class DualMode:
    SINGLE = EnumValue("DualMode", "single")
    SPLITM = EnumValue("DualMode", "splitm")
    SPLITN = EnumValue("DualMode", "splitn")


class PostMode:
    # Unaligned stores and their Post always advance the cursor (RFC-0001 §6.11): NORMAL is accepted on
    # reg_to_ub_unalign / reg_to_ub_unalign_post through 0.1.x and means UPDATE. Loads honour both (I042).
    NORMAL = EnumValue("PostMode", "normal")
    UPDATE = EnumValue("PostMode", "update")


class HighLowPart:
    LOWEST = EnumValue("HighLowPart", "lowest")
    HIGHEST = EnumValue("HighLowPart", "highest")


class LoadDist:
    NORMAL = EnumValue("LoadDist", "normal")
    DOWNSAMPLE = EnumValue("LoadDist", "downsample")
    UPSAMPLE = EnumValue("LoadDist", "upsample")
    SINGLE_VALUE = EnumValue("LoadDist", "single")
    BRCB = EnumValue("LoadDist", "brcb")
    UNPACK = EnumValue("LoadDist", "unpack")
    UNPACK4 = EnumValue("LoadDist", "unpack4")


class StoreDist:
    NORMAL = EnumValue("StoreDist", "normal")
    DOWNSAMPLE = EnumValue("StoreDist", "downsample")
    PACK4 = EnumValue("StoreDist", "pack4")
    SINGLE_VALUE = EnumValue("StoreDist", "single")


class HistBin:
    BIN0, BIN1, BIN2, BIN3, BIN4, BIN5, BIN6, BIN7 = (EnumValue("HistBin", f"bin{i}") for i in range(8))


class HistMode:
    FREQUENCY = EnumValue("HistMode", "frequency")
    ACCUMULATE = EnumValue("HistMode", "accumulate")


@dataclass(frozen=True)
class Conv2D:
    """Static 2-D convolution geometry for ``conv2d`` / ``img2col``."""

    kh: int
    kw: int
    pad: tuple[int, int, int, int] = (0, 0, 0, 0)  # left, right, top, bottom
    stride: tuple[int, int] = (1, 1)
    dilation: tuple[int, int] = (1, 1)

    def out_hw(self, h: Any, w: Any) -> tuple[Any, Any]:
        pl, pr, pt, pb = self.pad
        ho = (h + pt + pb - self.dilation[0] * (self.kh - 1) - 1) // self.stride[0] + 1
        wo = (w + pl + pr - self.dilation[1] * (self.kw - 1) - 1) // self.stride[1] + 1
        return ho, wo


# --------------------------------------------------------------------------- GM parameter annotations


@dataclass(frozen=True)
class GMSpec:
    dtype: DTypeName
    dims: tuple[Any, ...]  # int | str (symbol or product such as "M*K") | "?" (lists only)
    is_list: bool = False
    count: int | None = None

    def __repr__(self) -> str:
        head = "GMList" if self.is_list else "GM"
        return f"{head}[{self.dtype!r}, {self.dims!r}]"


class _GMMeta(type):
    def __getitem__(cls, item: Any) -> GMSpec:
        if not isinstance(item, tuple) or len(item) < 2:
            raise TypeError(f"{cls.__name__}[dtype, dims] needs a dtype and a dims tuple, got {item!r}")
        dtype, dims = item[0], item[1]
        if not isinstance(dtype, DTypeName):
            raise TypeError(f"{cls.__name__}[...]: first argument must be a dtype such as f32, got {dtype!r}")
        if isinstance(dims, (int, str)):
            dims = (dims,)
        if not isinstance(dims, tuple) or not all(isinstance(d, (int, str)) for d in dims):
            raise TypeError(f"{cls.__name__}[...]: dims must be a tuple of ints and symbol strings, got {dims!r}")
        count = item[2] if len(item) > 2 else None
        return GMSpec(dtype, tuple(dims), cls is GMList, count)


class GM(metaclass=_GMMeta):
    """``x: GM[f32, ("M", "K")]`` — a GM tensor parameter with symbolic dims (RFC-0002 §3.2)."""


class GMList(metaclass=_GMMeta):
    """``xs: GMList[bf16, ("?", "D")]`` — a list of GM tensors; ``?`` marks per-member dims."""


class GMTensor:
    """The old bare annotation. Kept as a name so the diagnostic can say what to write instead."""


class GMTensorList:
    """The old bare annotation; see :class:`GMTensor`."""


# --------------------------------------------------------------------------- markers


def marker(rule: str) -> Callable[[Any], Any]:
    def deco(obj: Any) -> Any:
        obj._asc_rule = rule
        return obj

    return deco


def rule_of(obj: Any) -> str | None:
    return getattr(obj, "_asc_rule", None)


@marker("tensor")
class Tensor:
    """``Tensor(dtype, [rows, cols], Position.X, name="")`` — an on-chip tensor; compiled to ``mem.alloc``."""

    def __init__(self, *a: Any, **k: Any) -> None:
        raise TypeError("Tensor is compiled, not executed; call it inside a @kernel / @vf body")


@marker("buf")
class DBuff:
    """Two-slot buffer; ``buf[idx]`` selects modulo 2. ``sync_depth=N`` caps autosync run-ahead without changing slots."""

    slots = 2


@marker("buf")
class TBuff:
    """Three-slot buffer; ``buf[idx]`` selects modulo 3. ``sync_depth=N`` caps autosync run-ahead without changing slots."""

    slots = 3


@marker("buf")
class QBuff:
    """Four-slot buffer; ``buf[idx]`` selects modulo 4. ``sync_depth=N`` caps autosync run-ahead without changing slots."""

    slots = 4


@marker("var")
class Var:
    """``Var(value, dtype=None, name="")`` — a mutable scalar (an IR cell); also the old scalar annotation."""


@marker("reg")
class Reg:
    """``Reg(dtype, name="", reg_num=1)`` — a vector register or a two-register group (vf only)."""


@marker("maskreg")
class MaskReg:
    """``MaskReg(dtype, name="", reg_num=1)`` — predicates for the dtype and register count."""


@marker("unalign_load")
def unalign_reg_for_load(name: str = "") -> None: ...


@marker("unalign_store")
def unalign_reg_for_store(name: str = "") -> None: ...


@marker("mutex_vc")
class VcMutex:
    """``VcMutex(id, *, depth=None, guards=None, src_start_pipe=..., dst_start_pipe=..., src_end_pipe=..., dst_end_pipe=...)``:
    vector producer -> cube consumer.

    Four calls make one cycle, two on each side, and all four are required. The VECTOR side takes a
    slot with ``.lock()``, writes it, and publishes it with ``.ready()``; the CUBE side acquires it
    with ``.wait()``, reads it, and returns the credit with ``.free()`` after its LAST read. The
    consuming instructions - the matmuls that read the handed-over tile - therefore have to sit
    between ``wait`` and ``free``: a ``wait`` followed straight by ``free`` hands the slot back
    before anything read it, and a read placed outside that pair is ordered by nothing, however
    many mutex calls the loop contains. Guard one hand-off buffer per mutex, and keep the counts
    balanced on each side - a ``ready`` no ``wait`` consumes leaves a token behind, and a missing
    ``free`` or ``ready`` stalls the other side's next call. The functional simulator reports such
    a stall as a deadlock naming this mutex, the awaited call and both token counters.

    One of ``depth`` or ``guards`` is REQUIRED, and both are keyword-only. There is no default
    that is right: the credit count belongs to the buffer being handed over, not to the mutex.

    ``depth`` is a CREDIT count, not a pipelining hint: the consumer publishes that many up
    front, so ``.lock()`` of cycle *i* blocks on the ``.free()`` of cycle *i - depth*. Declare
    one credit per slot the guarded buffer rotates through before the producer reaches the
    first one again (a plain ``Tensor`` 1, ``DBuff`` 2, ``TBuff`` 3, ``QBuff`` 4), times the
    number of cycles the mutex runs per rotation. Too many credits lets the producer retake a
    slot the consumer is still reading: a wrong answer, not a hang.

    ``guards=`` names that buffer instead, and the credits follow from it:
    ``VcMutex(0, guards=ub_score, ...)``. It takes one on-chip tensor or buffer, or a tuple of
    them (the tightest rotation wins). The two are not alternatives to choose between - ``guards``
    supplies ``depth`` when it is absent, and a lint compares them when both are written, which is
    how a mutex that deliberately cycles twice per rotation says so. The name also travels into
    the IR, where ``autosync`` otherwise has to infer which buffer an edge belongs to.
    """


@marker("mutex_cv")
class CvMutex:
    """``CvMutex(id, *, depth=None, guards=None, src_start_pipe=..., dst_start_pipe=..., src_end_pipe=..., dst_end_pipe=...)``:
    cube producer -> vector consumer.

    Four calls make one cycle, two on each side, and all four are required. The CUBE side takes a
    slot with ``.lock()``, drains into it, and publishes it with ``.ready()``; the VECTOR side
    acquires it with ``.wait()``, reads it, and returns the credit with ``.free()`` after its LAST
    read. The consuming instructions - the vector work that reads the drained tile - therefore have
    to sit between ``wait`` and ``free``: a ``wait`` followed straight by ``free`` hands the slot
    back before anything read it, and a read placed outside that pair is ordered by nothing,
    however many mutex calls the loop contains. Guard one hand-off buffer per mutex, and keep the
    counts balanced on each side - a ``ready`` no ``wait`` consumes leaves a token behind, and a
    missing ``free`` or ``ready`` stalls the other side's next call. The functional simulator
    reports such a stall as a deadlock naming this mutex, the awaited call and both token counters.

    One of ``depth`` or ``guards`` is REQUIRED, and both are keyword-only. There is no default
    that is right: the credit count belongs to the buffer being handed over, not to the mutex.

    ``depth`` is a CREDIT count, not a pipelining hint: the consumer publishes that many up
    front, so ``.lock()`` of cycle *i* blocks on the ``.free()`` of cycle *i - depth*. Declare
    one credit per slot the guarded buffer rotates through before the producer reaches the
    first one again (a plain ``Tensor`` 1, ``DBuff`` 2, ``TBuff`` 3, ``QBuff`` 4), times the
    number of cycles the mutex runs per rotation. Too many credits lets the producer retake a
    slot the consumer is still reading: a wrong answer, not a hang.

    ``guards=`` names that buffer instead, and the credits follow from it:
    ``CvMutex(0, guards=ub_score, ...)``. It takes one on-chip tensor or buffer, or a tuple of
    them (the tightest rotation wins). The two are not alternatives to choose between - ``guards``
    supplies ``depth`` when it is absent, and a lint compares them when both are written, which is
    how a mutex that deliberately cycles twice per rotation says so. The name also travels into
    the IR, where ``autosync`` otherwise has to infer which buffer an edge belongs to.
    """


@marker("auto_sync")
class auto_sync:
    """``with auto_sync(mode="conservative"):`` — a ``region.autosync``."""


@marker("vec_scope")
class vec_scope:
    """``with vec_scope():`` — ops inside run on the vector side only."""


@marker("cube_scope")
class cube_scope:
    """``with cube_scope():`` — ops inside run on the cube side only."""


def _fn(rule: str, name: str, doc: str) -> Callable[..., Any]:
    def f(*a: Any, **k: Any) -> Any:
        raise TypeError(f"{name} is compiled, not executed; call it inside a kernel body")

    f.__name__ = name
    f.__doc__ = doc
    f._asc_rule = rule  # type: ignore[attr-defined]
    return f


matmul = _fn("matmul", "matmul", "matmul(dst_l0c, a_l1, b_l1, m=, n=, k=, is_init=True, splitn=, splitk=, bias=): dst = a @ b^T")
matmul_mx = _fn("matmul_mx", "matmul_mx", "matmul_mx(dst_l0c, a_l1, b_l1, scale_a, scale_b, m=, n=, k=, is_init=True, splitk=, splitn=, bias=)")
conv2d = _fn("conv2d", "conv2d", "conv2d(l0c, fmap, weight, conv: Conv2D, h, w, c, cout, m0=0, tile_k=None, bias=None)")
img2col = _fn("img2col", "img2col", "img2col(fmap_l1, conv: Conv2D, h, w, c): a compile-time view for l1_to_l0a_img2col")
zero_mxfp8_l1_padding = _fn("zero_mxfp8_l1_padding", "zero_mxfp8_l1_padding", "zero the padding blocks of an MX FP8 L1 tile")
cast = _fn("cast", "cast", "cast(dst_reg, src_reg, cfg: CastConfig, mask=None)")
CeilDiv = _fn("scalar.ceil_div", "CeilDiv", "ceil(a / b)")
Min = _fn("scalar.min", "Min", "min(a, b)")
Max = _fn("scalar.max", "Max", "max(a, b)")
var_add = _fn("scalar.add", "var_add", "a + b (the old var_op spelling; the operators compile the same)")
var_sub = _fn("scalar.sub", "var_sub", "a - b")
var_mul = _fn("scalar.mul", "var_mul", "a * b")
var_div = _fn("scalar.div", "var_div", "Integer floor division; rounding='trunc' explicitly truncates toward zero")
var_mod = _fn("scalar.mod", "var_mod", "Integer floor remainder; rounding='trunc' uses the dividend's sign")
var_and = _fn("scalar.and", "var_and", "a & b")
var_or = _fn("scalar.or", "var_or", "a | b")
var_xor = _fn("scalar.xor", "var_xor", "a ^ b")
var_shl = _fn("scalar.shl", "var_shl", "a << b")
var_shr = _fn("scalar.shr", "var_shr", "a >> b")
var_inv = _fn("scalar.not", "var_inv", "~a")
scalar_sqrt = _fn("scalar.sqrt", "scalar_sqrt", "sqrt(a)")
scalar_abs = _fn("scalar.abs", "scalar_abs", "abs(a)")
Align8 = _fn("align:8", "Align8", "round up to a multiple of 8")
Align16 = _fn("align:16", "Align16", "round up to a multiple of 16")
Align32 = _fn("align:32", "Align32", "round up to a multiple of 32")
Align64 = _fn("align:64", "Align64", "round up to a multiple of 64")
Align128 = _fn("align:128", "Align128", "round up to a multiple of 128")
Align256 = _fn("align:256", "Align256", "round up to a multiple of 256")
GetCubeNum = _fn("core.cube_num", "GetCubeNum", "number of cube cores")
GetCubeIdx = _fn("core.cube_idx", "GetCubeIdx", "index of this cube core")
GetVecNum = _fn("core.vec_num", "GetVecNum", "number of vector cores")
GetVecIdx = _fn("core.vec_idx", "GetVecIdx", "index of this vector core")
GetSubBlockIdx = _fn("core.sub_block_idx", "GetSubBlockIdx", "index of this vector core within its pair")
set_hf32 = _fn("core.set_hf32", "set_hf32", "set_hf32(enable=True)")
set_saturation_flag = _fn("core.set_sat_flag", "set_saturation_flag",
                          "set_saturation_flag(mode, enable): raw CTRL bit; mode is 'float' | 'float8' | 'int' | 'cast' | 'global'")
get_saturation_flag = _fn("core.get_sat_flag", "get_saturation_flag", "get_saturation_flag(mode)")
clean_dcache = _fn("core.clean_dcache", "clean_dcache", "clean_dcache(dst): one GM tensor view")
simt_thread_id = _fn("simt.thread_id", "simt_thread_id", "index of this SIMT thread")
simt_thread_num = _fn("simt.thread_num", "simt_thread_num", "number of SIMT threads")
simt_block_idx = _fn("simt.block_idx", "simt_block_idx", "index of the core running the SIMT launch")
simt_block_num = _fn("simt.block_num", "simt_block_num", "number of cores")
simt_thread_barrier = _fn("simt.barrier", "simt_thread_barrier", "SIMT thread barrier")
simt_atomic_add = _fn("simt.atomic:add", "simt_atomic_add", "simt_atomic_add(tensor[index], value)")
simt_atomic_sub = _fn("simt.atomic:sub", "simt_atomic_sub", "simt_atomic_sub(tensor[index], value)")
simt_atomic_max = _fn("simt.atomic:max", "simt_atomic_max", "simt_atomic_max(tensor[index], value)")
simt_atomic_min = _fn("simt.atomic:min", "simt_atomic_min", "simt_atomic_min(tensor[index], value)")
simt_atomic_cas = _fn("simt.atomic:cas", "simt_atomic_cas", "simt_atomic_cas(tensor[index], compare, value)")
simt_atomic_exch = _fn("simt.atomic:exch", "simt_atomic_exch", "simt_atomic_exch(tensor[index], value)")
simt_atomic_and = _fn("simt.atomic:and", "simt_atomic_and", "simt_atomic_and(tensor[index], value)")
simt_atomic_or = _fn("simt.atomic:or", "simt_atomic_or", "simt_atomic_or(tensor[index], value)")
simt_atomic_xor = _fn("simt.atomic:xor", "simt_atomic_xor", "simt_atomic_xor(tensor[index], value)")
simt_atomic_inc = _fn("simt.atomic:inc", "simt_atomic_inc", "simt_atomic_inc(tensor[index], limit): ring increment, wraps past limit to 0")
simt_atomic_dec = _fn("simt.atomic:dec", "simt_atomic_dec", "simt_atomic_dec(tensor[index], limit): ring decrement, wraps 0 to limit")
simt_threadfence = _fn("simt.threadfence", "simt_threadfence", "device-scope SIMT memory fence")
simt_threadfence_block = _fn("simt.threadfence_block", "simt_threadfence_block", "core-scope SIMT memory fence")
cvt = _fn("scalar.cast", "cvt", "cvt(x, dtype): convert a SIMT scalar")
simt_fma = _fn("simt.fma", "simt_fma", "simt_fma(a, b, c) = a * b + c with one rounding")
simt_ffs = _fn("simt.ffs", "simt_ffs", "index of the lowest set bit (1-based; 0 when empty)")
simt_exp = _fn("simt.exp", "simt_exp", "SIMT scalar e**x (f32)")
simt_exp2 = _fn("simt.exp2", "simt_exp2", "SIMT scalar 2**x (f32)")
simt_log = _fn("simt.log", "simt_log", "SIMT scalar natural log (f32)")
simt_log2 = _fn("simt.log2", "simt_log2", "SIMT scalar log base 2 (f32)")
simt_log1p = _fn("simt.log1p", "simt_log1p", "SIMT scalar log(1 + x) (f32)")
simt_sin = _fn("simt.sin", "simt_sin", "SIMT scalar sine (f32)")
simt_cos = _fn("simt.cos", "simt_cos", "SIMT scalar cosine (f32)")
simt_tanh = _fn("simt.tanh", "simt_tanh", "SIMT scalar tanh (f32)")
simt_rsqrt = _fn("simt.rsqrt", "simt_rsqrt", "SIMT scalar 1/sqrt(x) (f32)")
simt_rint = _fn("simt.rint", "simt_rint", "round to nearest even (f32)")
simt_round = _fn("simt.round", "simt_round", "round half away from zero (f32)")
simt_floor = _fn("simt.floor", "simt_floor", "round toward -inf (f32)")
simt_ceil = _fn("simt.ceil", "simt_ceil", "round toward +inf (f32)")
simt_trunc = _fn("simt.trunc", "simt_trunc", "round toward zero (f32)")
simt_isnan = _fn("simt.isnan", "simt_isnan", "1 when x is NaN else 0")
simt_isinf = _fn("simt.isinf", "simt_isinf", "1 when x is +/-inf else 0")
simt_isfinite = _fn("simt.isfinite", "simt_isfinite", "1 when x is finite else 0")
simt_popc = _fn("simt.popc", "simt_popc", "number of set bits (32-bit)")
simt_mul_hi = _fn("simt.mul_hi", "simt_mul_hi", "high 32 bits of the 64-bit product")
simt_fmod = _fn("simt.fmod", "simt_fmod", "a - trunc(a / b) * b")
static_print = _fn("static_print", "static_print", "print at compile time")
static_assert = _fn("static_assert", "static_assert", "check at compile time")
kernel_print = _fn("debug.print", "kernel_print", "kernel_print(fmt, *args)")
sim_print = _fn("debug.print", "sim_print", "sim_print(*args, pipe=Pipe.S)")
kernel_dump_tensor = _fn("debug.dump", "kernel_dump_tensor", "kernel_dump_tensor(src, desc, dumpSize)")
sim_dump_tensor = _fn("debug.dump", "sim_dump_tensor", "sim_dump_tensor(tensor, filename, pipe)")
print_reg = _fn("debug.print_reg", "print_reg", "print_reg(reg, label='', lanes=8)")
reinterpret = _fn("reinterpret", "reinterpret", "reinterpret(src_tensor, dtype, name=''): view the same bytes as another dtype")
split_workspace = _fn("workspace", "split_workspace", "split_workspace(dtype, shape, name=''): a GM workspace tensor")
GMBuff = _fn("gmbuff", "GMBuff", "GMBuff(dtype, shape, slots=N, name='', per_core=True): a GM workspace ring of N "
             "slot tensors; ws[beat] selects slot beat % N (one slot per cube core when per_core). The lowering checks "
             "the ring algebra: one beat counter, reader lag < slots, every cross-side access inside a mutex window of "
             "depth <= slots (RFC-0009)")
reset_cache = _fn("noop", "reset_cache", "no-op kept for the old spelling")

# events, flags, barriers, cross-core signals
SEvent = _fn("event:1", "SEvent", "SEvent(src_pipe, dst_pipe, preset=False, name=''): a single-slot pipe-to-pipe event")
DEvent = _fn("event:2", "DEvent", "DEvent(src_pipe, dst_pipe, preset=False, name=''): a two-slot event")
TEvent = _fn("event:3", "TEvent", "TEvent(src_pipe, dst_pipe, preset=False, name=''): a three-slot event")
QEvent = _fn("event:4", "QEvent", "QEvent(src_pipe, dst_pipe, preset=False, name=''): a four-slot event")
setflag = _fn("sync.set_flag", "setflag", "setflag(src_pipe, dst_pipe, event_id)")
waitflag = _fn("sync.wait_flag", "waitflag", "waitflag(src_pipe, dst_pipe, event_id)")
barrier = _fn("barrier", "barrier", "barrier(pipe)")
bar_m = _fn("barrier:M", "bar_m", "barrier(Pipe.M)")
bar_v = _fn("barrier:V", "bar_v", "barrier(Pipe.V)")
bar_mte1 = _fn("barrier:MTE1", "bar_mte1", "barrier(Pipe.MTE1)")
bar_mte2 = _fn("barrier:MTE2", "bar_mte2", "barrier(Pipe.MTE2)")
bar_mte3 = _fn("barrier:MTE3", "bar_mte3", "barrier(Pipe.MTE3)")
bar_fix = _fn("barrier:FIX", "bar_fix", "barrier(Pipe.FIX)")
bar_all = _fn("barrier:ALL", "bar_all", "barrier(Pipe.ALL)")
cube_ready = _fn("crosscore:cube_ready", "cube_ready", "cube_ready(flag_id=0, pipe=Pipe.FIX)")
vec_ready = _fn("crosscore:vec_ready", "vec_ready", "vec_ready(flag_id=0, pipe=Pipe.MTE3)")
wait_cube = _fn("crosscore:wait_cube", "wait_cube", "wait_cube(flag_id=0, pipe=Pipe.S)")
wait_vec = _fn("crosscore:wait_vec", "wait_vec", "wait_vec(flag_id=0, pipe=Pipe.S)")
allcube_ready = _fn("crosscore:allcube_ready", "allcube_ready", "allcube_ready(flag_id=0, pipe=Pipe.FIX)")
allcube_wait = _fn("crosscore:allcube_wait", "allcube_wait", "allcube_wait(flag_id=0, pipe=Pipe.S)")
allvec_ready = _fn("crosscore:allvec_ready", "allvec_ready", "allvec_ready(flag_id=0, pipe=Pipe.MTE3)")
allvec_wait = _fn("crosscore:allvec_wait", "allvec_wait", "allvec_wait(flag_id=0, pipe=Pipe.S)")
intracore_allvec_ready = _fn("crosscore:intracore_allvec_ready", "intracore_allvec_ready", "intracore_allvec_ready(flag_id=0, pipe=Pipe.MTE3)")
intracore_allvec_wait = _fn("crosscore:intracore_allvec_wait", "intracore_allvec_wait", "intracore_allvec_wait(flag_id=0, pipe=Pipe.S)")
vf_barrier = _fn("vf.barrier", "vf_barrier", "vf_barrier(src: VfPipe, dst: VfPipe): local memory barrier inside a vf")
atomic_add = _fn("atomic:add", "atomic_add", "with atomic_add(): GM stores inside accumulate")
atomic_max = _fn("atomic:max", "atomic_max", "with atomic_max(): GM stores inside take the maximum")
atomic_min = _fn("atomic:min", "atomic_min", "with atomic_min(): GM stores inside take the minimum")

# explicit cube / DMA instructions (kernel level)
gm_to_l1 = _fn("dma.gm_to_l1", "gm_to_l1", "gm_to_l1(dst, src, n_burst=, burst_len=, src_stride=, dst_stride=)")
gm_to_l1_pad = _fn("dma.gm_to_l1.pad", "gm_to_l1_pad", "gm_to_l1_pad(dst, src, n_burst=, burst_len_element=, src_stride_element=, dst_stride=)")
gm_to_l1_nd2nz = _fn("dma.gm_to_l1.nd2nz", "gm_to_l1_nd2nz", "gm_to_l1_nd2nz(dst, src, M=, N=, N_src=, M_dst=)")
gm_to_l1_dn2nz = _fn("dma.gm_to_l1.dn2nz", "gm_to_l1_dn2nz", "gm_to_l1_dn2nz(dst, src, M=, N=, N_src=, M_dst=)")
gm_to_l1_mx_scale = _fn("dma.gm_to_l1.mx_scale", "gm_to_l1_mx_scale", "gm_to_l1_mx_scale(dst, src, row_tile_idx=0, k_blocks=, src_k_blocks=, k_block_idx=0)")
gm_to_l1_mx_scale_nd2nz = _fn("dma.gm_to_l1.mx_scale_nd2nz", "gm_to_l1_mx_scale_nd2nz", "gm_to_l1_mx_scale_nd2nz(dst, src, rows=, k_groups=, src_k_groups=)")
set_constant_to_l1 = _fn("dma.set_constant_to_l1", "set_constant_to_l1", "set_constant_to_l1(tensor, val, n_blocks=)")
l1_to_l0 = _fn("dma.l1_to_l0", "l1_to_l0", "l1_to_l0(dst, src, m_dst=, n_dst=, m_src=, n_src=)")
l1_to_l0_mx = _fn("dma.l1_to_l0.mx", "l1_to_l0_mx", "l1_to_l0_mx(dst, src, src_mx, m_dst=, n_dst=, m_src=, n_src=, src_mx_offset_element=)")
l1_to_l0a_img2col = _fn("dma.l1_to_l0.img2col", "l1_to_l0a_img2col", "l1_to_l0a_img2col(dst, img2col_window)")
l1_to_bt = _fn("dma.l1_to_bt", "l1_to_bt", "l1_to_bt(dst, src, n=)")
mmad = _fn("cube.mmad", "mmad", "mmad(dst, src_a, src_b, M=, N=, K=, is_init=True, bias=None, unit_flag=0)")
mmad_mx = _fn("cube.mmad.mx", "mmad_mx", "mmad_mx(dst, src_a, src_b, M=, N=, K=, is_init=True, bias=None)")
l0c_to_gm_nz2nd = _fn("dma.l0c_to_gm.nz2nd", "l0c_to_gm_nz2nd", "l0c_to_gm_nz2nd(dst, src, M=, N=, N_dst=, M_src=, relu=False, scale=1.0, offset=0, hif8_hybrid=False)")
l0c_to_gm_nz2nz = _fn("dma.l0c_to_gm.nz2nz", "l0c_to_gm_nz2nz", "l0c_to_gm_nz2nz(dst, src, m_pad=, relu=False, scale=1.0, offset=0, hif8_hybrid=False)")
l0c_to_gm_nz2dn = _fn("dma.l0c_to_gm.nz2dn", "l0c_to_gm_nz2dn", "l0c_to_gm_nz2dn(dst, src, M=, N=, M_dst=, M_src=, relu=False, scale=1.0, offset=0, hif8_hybrid=False)")
l0c_to_l1 = _fn("dma.l0c_to_l1", "l0c_to_l1", "l0c_to_l1(dst, src, M=, N=, M_dst=, M_src=, relu=False)")
l0c_to_ub = _fn("dma.l0c_to_ub", "l0c_to_ub", "l0c_to_ub(dst, src, M=, N=, N_dst=, M_src=, dual_mode=DualMode.SPLITM, sub_block_id=None, relu=False, scale=1.0, offset=0, hif8_hybrid=False)")
gm_to_ub_pad = _fn("dma.gm_to_ub.pad", "gm_to_ub_pad", "gm_to_ub_pad(dst, src, n_burst=, burst_len_element=, src_stride_element=, dst_stride=, pad=)")
gm_to_ub_nd_dma = _fn("dma.gm_to_ub.nd", "gm_to_ub_nd_dma", "gm_to_ub_nd_dma(dst, src, loop_src_stride, loop_dst_stride, loop_size, ...)")
gm_to_ub_nd_dma_transpose = _fn("dma.gm_to_ub.nd:transpose", "gm_to_ub_nd_dma_transpose", "gm_to_ub_nd_dma_transpose(dst, src)")
ub_to_gm_pad = _fn("dma.ub_to_gm.pad", "ub_to_gm_pad", "ub_to_gm_pad(dst, src, n_burst=, burst_len_element=, src_stride=, dst_stride_element=)")
ub_to_ub = _fn("dma.ub_to_ub", "ub_to_ub", "ub_to_ub(dst, src, n_burst=, burst_len=, src_stride=, dst_stride=)")
ub_to_l1 = _fn("dma.ub_to_l1", "ub_to_l1", "ub_to_l1(dst, src, n_burst=, burst_len=, src_stride=, dst_stride=)")
ub_to_l1_nd2nz = _fn("dma.ub_to_l1.nd2nz", "ub_to_l1_nd2nz", "ub_to_l1_nd2nz(dst, src, m_dst=, n_dst=, m_src=, n_src=, N_src=)")
ub_to_l1_nz = _fn("dma.ub_to_l1.nz", "ub_to_l1_nz", "ub_to_l1_nz(dst, src, m_dst=, n_dst=, m_src=, n_src=)")
sort32 = _fn("vec.sort32", "sort32", "sort32(dst, src, idx, repeat=None): 32 fp32 scores + uint32 indices per repeat -> 32 (score, index) records, descending")
mergesort4 = _fn("vec.mergesort4", "mergesort4", "mergesort4(dst, src, length_per_seq, repeat=1): four sorted record lists -> one, per repeat")
mergesort_2seq = _fn("vec.mergesort_2seq", "mergesort_2seq", "mergesort_2seq(dst, src1, src2, size1, size2): two sorted record lists -> one")
set_mask = _fn("vec.set_mask", "set_mask", "set_mask(mask_high, mask_low)")
set_mask_by_count = _fn("vec.set_mask_by_count", "set_mask_by_count", "set_mask_by_count(count)")
set_mask_normal = _fn("vec.set_mask_normal", "set_mask_normal", "set_mask_normal()")
reset_mask = _fn("vec.reset_mask", "reset_mask", "reset_mask()")

# register-level (vf) instructions; the method / operator forms build the same ops
ub_to_reg = _fn("vf.load", "ub_to_reg", "ub_to_reg(dst, src, blk_stride=1, mask=None): 32-byte block copy UB -> register")
reg_to_ub = _fn("vf.store", "reg_to_ub", "reg_to_ub(dst, src, blk_stride=1, mask=None): 32-byte block copy register -> UB")
ub_to_reg_continuous = _fn("vf.load_cont", "ub_to_reg_continuous", "ub_to_reg_continuous(dst, src, loaddist)")
reg_to_ub_continuous = _fn("vf.store_cont", "reg_to_ub_continuous", "reg_to_ub_continuous(dst, src, mask, storedist)")
ub_to_reg_normal = _fn("vf.load_cont:normal", "ub_to_reg_normal", "contiguous load")
ub_to_reg_single = _fn("vf.load_cont:single", "ub_to_reg_single", "broadcast the first element")
ub_to_reg_upsample = _fn("vf.load_cont:upsample", "ub_to_reg_upsample", "each element twice")
ub_to_reg_downsample = _fn("vf.load_cont:downsample", "ub_to_reg_downsample", "every other element")
ub_to_reg_unpack = _fn("vf.load_cont:unpack", "ub_to_reg_unpack", "elements into even lanes")
ub_to_reg_unpack4 = _fn("vf.load_cont:unpack4", "ub_to_reg_unpack4", "elements into every fourth lane")
ub_to_reg_brcb = _fn("vf.load_cont:brcb", "ub_to_reg_brcb", "each element broadcast over a 32-byte block")
ub_to_reg_interleave = _fn("vf.load_interleave", "ub_to_reg_interleave", "ub_to_reg_interleave(dst0, dst1, src)")
reg_to_ub_interleave = _fn("vf.store_interleave", "reg_to_ub_interleave", "reg_to_ub_interleave(dst, src0, src1, mask=None)")
reg_to_ub_normal = _fn("vf.store_cont:normal", "reg_to_ub_normal", "contiguous store")
reg_to_ub_single = _fn("vf.store_cont:single", "reg_to_ub_single", "store lane 0")
reg_to_ub_downsample = _fn("vf.store_cont:downsample", "reg_to_ub_downsample", "store even lanes packed")
reg_to_ub_pack4 = _fn("vf.store_cont:pack4", "reg_to_ub_pack4", "store every fourth lane packed")
ub_to_reg_gather = _fn("vf.gather_copy", "ub_to_reg_gather", "ub_to_reg_gather(dst, src, index, mask=None)")
ub_to_reg_gatherb = _fn("vf.gatherb", "ub_to_reg_gatherb", "ub_to_reg_gatherb(dst, src, index, mask=None)")
reg_to_ub_scatter = _fn("vf.scatter_copy", "reg_to_ub_scatter", "reg_to_ub_scatter(dst, src, index, mask=None)")
gather = _fn("vf.gather", "gather", "gather(dst, src, index): register-to-register gather")
gather_mask = _fn("vf.gathermask", "gather_mask", "gather_mask(dst, src, mask=None)")
squeeze = _fn("vf.squeeze", "squeeze", "squeeze(dst, src, mask=None, store=False)")
unsqueeze = _fn("vf.unsqueeze", "unsqueeze", "unsqueeze(dst, mask=None)")
clear_spr = _fn("vf.clear_spr", "clear_spr", "clear_spr()")
ub_to_mask = _fn("vf.ub_to_mask", "ub_to_mask", "ub_to_mask(dst, src)")
mask_to_ub = _fn("vf.mask_to_ub", "mask_to_ub", "mask_to_ub(dst, src)")
ub_cursor = _fn("vf.ub_cursor", "ub_cursor", "ub_cursor(view, name='')")
ub_to_reg_unalign_pre = _fn("vf.load_unalign_pre", "ub_to_reg_unalign_pre", "ub_to_reg_unalign_pre(ureg, src)")
ub_to_reg_unalign = _fn("vf.load_unalign", "ub_to_reg_unalign", "ub_to_reg_unalign(dst, ureg, src, stride=None, post_mode=None)")
ub_to_reg_unalign_once = _fn("vf.load_unalign:once", "ub_to_reg_unalign_once", "ub_to_reg_unalign_once(dst, src, ureg=None)")
reg_to_ub_unalign = _fn("vf.store_unalign", "reg_to_ub_unalign", "reg_to_ub_unalign(dst, src, ureg, count, post_mode=PostMode.UPDATE)")
reg_to_ub_unalign_post = _fn("vf.store_unalign_post", "reg_to_ub_unalign_post", "reg_to_ub_unalign_post(dst, ureg, stride=0, post_mode=PostMode.UPDATE)")
reg_to_ub_unalign_once = _fn("vf.store_unalign:once", "reg_to_ub_unalign_once", "reg_to_ub_unalign_once(dst, src, count, ureg=None)")
mask_not = _fn("vf.mask_not", "mask_not", "mask_not(dst, src, mask=None)")
mask_and = _fn("vf.mask_and", "mask_and", "mask_and(dst, src1, src2, mask=None)")
mask_or = _fn("vf.mask_or", "mask_or", "mask_or(dst, src1, src2, mask=None)")
mask_xor = _fn("vf.mask_xor", "mask_xor", "mask_xor(dst, src1, src2, mask=None)")
mask_mov = _fn("vf.mask_mov", "mask_mov", "mask_mov(dst, src, mask=None)")
mask_sel = _fn("vf.mask_sel", "mask_sel", "mask_sel(dst, src1, src2, mask=None)")
mask_pack = _fn("vf.mask_pack", "mask_pack", "mask_pack(dst, src, low_part=True)")
mask_unpack = _fn("vf.mask_unpack", "mask_unpack", "mask_unpack(dst, src, low_part=True)")
mask_interleave = _fn("vf.mask_interleave", "mask_interleave", "mask_interleave(dst0, dst1, src0, src1)")
mask_deinterleave = _fn("vf.mask_deinterleave", "mask_deinterleave", "mask_deinterleave(dst0, dst1, src0, src1)")
move_mask_spr = _fn("vf.mask_from_spr", "move_mask_spr", "move_mask_spr(dst)")
update_mask = _fn("vf.mask_update", "update_mask", "update_mask(dst, cnt): the first cnt lanes, then cnt -= lanes")
compare = _fn("vf.cmp", "compare", "compare(dst_mask, src1, src2, mode: CompareMode, mask=None)")
select = _fn("vf.select", "select", "select(dst, src1, src2, mask=None)")
pack = _fn("vf.pack", "pack", "pack(dst, src, low_or_high=HighLowPart.LOWEST)")
interleave = _fn("vf.interleave", "interleave", "interleave(dst0, dst1, src0, src1)")
deinterleave = _fn("vf.deinterleave", "deinterleave", "deinterleave(dst0, dst1, src0, src1)")
dup = _fn("vf.dup", "dup", "dup(dst, src, mask=None)")
arange = _fn("vf.arange", "arange", "arange(dst, start, increase=True)")
expsub = _fn("vf.expsub", "expsub", "expsub(dst, src0, src1, mask=None, layout=RegLayout.ZERO): dst = exp(src0 - src1); a masked-off lane is written 0, not left as it was (docs/api/mask-write-semantics.md)")
abssub = _fn("vf.abssub", "abssub", "abssub(dst, src0, src1, mask=None): dst = |src0 - src1|")
muldstadd = _fn("vf.muldstadd", "muldstadd", "muldstadd(dst, src0, src1, mask=None): dst = dst * src0 + src1")
vmod = _fn("vf.mod", "vmod", "vmod(dst, src0, src1, mask=None): integer floor remainder")
muladddst = _fn("vf.muladddst", "muladddst", "muladddst(dst, src0, src1, mask=None): dst = src0 * src1 + dst")
mulscast = _fn("vf.mulscast", "mulscast", "mulscast(dst, src, value, mask=None, layout=RegLayout.ZERO); a masked-off lane is written 0, not left as it was (docs/api/mask-write-semantics.md)")
histograms = _fn("vf.histograms", "histograms", "histograms(dst, src, bin_group=HistBin.BIN0, mode=HistMode.FREQUENCY, mask=None)")
# arithmetic stub forms: (dst, src...) with a trailing mask
for _name, _op in (("exp", "exp"), ("abs", "abs"), ("sqrt", "sqrt"), ("relu", "relu"), ("ln", "ln"), ("log", "log"), ("log2", "log2"),
                   ("log10", "log10"), ("neg", "neg"), ("vnot", "not"), ("vcopy", "copy")):
    globals()[_name] = _fn(f"vf.{_op}", _name, f"{_name}(dst, src, mask=None)")
for _name, _op in (("adds", "adds"), ("muls", "muls"), ("vmaxs", "maxs"), ("vmins", "mins"), ("lrelu", "lrelu"), ("shiftls", "shiftls"),
                   ("shiftrs", "shiftrs"), ("axpy", "axpy")):
    globals()[_name] = _fn(f"vf.{_op}", _name, f"{_name}(dst, src, value, mask=None)")
for _name, _op in (("add", "add"), ("sub", "sub"), ("mul", "mul"), ("div", "div"), ("vmax", "max"), ("vmin", "min"), ("vand", "and"),
                   ("vor", "or"), ("vxor", "xor"), ("prelu", "prelu"), ("shiftl", "shiftl"), ("shiftr", "shiftr")):
    globals()[_name] = _fn(f"vf.{_op}", _name, f"{_name}(dst, src1, src2, mask=None)")
for _name in ("cadd", "cmax", "cmin", "cgadd", "cgmax", "cgmin", "cpadd"):
    globals()[_name] = _fn(f"vf.{_name}", _name, f"{_name}(dst, src, mask=None)")


@marker("reglist")
class RegList:
    """``RegList(dtype, length, name="")`` — ``length`` registers indexed statically (``rl[i]``)."""


# --------------------------------------------------------------------------- decorated functions


class KernelFn:
    """What ``@kernel`` returns: the entry function plus its device, mode and block_dim."""

    def __init__(self, fn: Callable[..., Any], device: str, mode: str = "mix", block_dim: Any = None) -> None:
        if mode not in ("mix", "vec", "cube"):
            raise ValueError(f"kernel mode must be mix | vec | cube, got {mode!r}")
        self.fn = fn
        self.device = device
        self.mode = mode
        self.block_dim = block_dim
        self.name = fn.__name__
        self._module: Any = None
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__

    def ir(self) -> Any:
        """The Surface IR module (compiled once)."""
        if self._module is None:
            from .compiler import compile_kernel

            self._module = compile_kernel(self)
        return self._module

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(f"{self.name} is a kernel; run it through OpExec / a launcher, or inspect it with .ir()")

    def __repr__(self) -> str:
        return f"<kernel {self.name} device={self.device} mode={self.mode}>"


class VfFn:
    """What ``@vf`` returns; compiled per call-site type signature (RFC-0002 §4.2)."""

    _asc_rule = "call_vf"

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.name = fn.__name__
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

    def __call__(self, *a: Any, **k: Any) -> Any:
        raise TypeError(f"{self.name} is a vf function; call it inside a kernel body")


class SimtFn:
    _asc_rule = "call_simt"
    ALLOWED_THREADS = (64, 128, 256, 512, 1024, 2048)

    def __init__(self, fn: Callable[..., Any], num_threads: int = 1024) -> None:
        if num_threads not in self.ALLOWED_THREADS:
            raise ValueError(f"simt(num_threads=...) must be one of {self.ALLOWED_THREADS}, got {num_threads}")
        self.fn = fn
        self.num_threads = num_threads
        self.name = fn.__name__
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

    def __call__(self, *a: Any, **k: Any) -> Any:
        raise TypeError(f"{self.name} is a simt function; call it inside a kernel body")


class InlineFn:
    """What ``@func`` returns: a helper inlined at each call site that passes a dynamic argument."""

    _asc_rule = "call_inline"

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.name = fn.__name__
        self.__name__ = fn.__name__

    def __call__(self, *a: Any, **k: Any) -> Any:
        return self.fn(*a, **k)


def make_decorators(device: str) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    """The ``kernel`` / ``vf`` / ``simt`` decorators bound to one device (used by the facades)."""

    def kernel(fn: Callable[..., Any] | None = None, *, mode: str = "mix", block_dim: Any = None) -> Any:
        def wrap(f: Callable[..., Any]) -> KernelFn:
            return KernelFn(f, device, mode, block_dim)

        return wrap(fn) if fn is not None else wrap

    def vf(fn: Callable[..., Any] | None = None) -> Any:
        return VfFn(fn) if fn is not None else VfFn

    def simt(fn: Callable[..., Any] | None = None, *, num_threads: int = 1024) -> Any:
        def wrap(f: Callable[..., Any]) -> SimtFn:
            return SimtFn(f, num_threads)

        return wrap(fn) if fn is not None else wrap

    return kernel, vf, simt


def func(fn: Callable[..., Any] | None = None) -> Any:
    """``@func`` / ``@func()``: a helper inlined at each call site (the old ``easyasc.func``)."""
    return InlineFn(fn) if fn is not None else InlineFn


def unroll(*args: int) -> range:
    """``for i in unroll(n)``: a compile-time loop — the frontend copies the body once per iteration.

    ``for i in range(n)`` is the device loop and stays a loop in the generated code even when ``n``
    is a constant (fewer instructions, a little slower); ``unroll`` trades code size for speed and
    needs static bounds (D-026). Outside a DSL function ``unroll`` is just ``range``.
    """
    return range(*args)


# RFC-0012: deliberate compatibility surface; compiler implementation helpers stay private.
__all__ = [
    'Align128', 'Align16', 'Align256', 'Align32', 'Align64', 'Align8',
    'CastConfig', 'CeilDiv', 'CompareMode', 'Conv2D', 'CvMutex', 'DBuff',
    'DEvent', 'DT', 'DualMode', 'E8M0_BIAS', 'E8M0_MIN_VALUE', 'FP4_E1M2_MAX_VALUE',
    'FP4_E2M1_MAX_VALUE', 'GM', 'GMBuff', 'GMList', 'GMTensor', 'GMTensorList',
    'GetCubeIdx', 'GetCubeNum', 'GetSubBlockIdx', 'GetVecIdx', 'GetVecNum', 'HIF8_MAX_FINITE_VALUE',
    'HIF8_MAX_NEGATIVE_NORMAL', 'HIF8_MAX_POSITIVE_NORMAL', 'HIF8_NAN', 'HIF8_NEGATIVE_INF', 'HIF8_OVERFLOW_THRESHOLD', 'HIF8_POSITIVE_INF',
    'HIF8_POSITIVE_ZERO', 'HighLowPart', 'HistBin', 'HistMode', 'Layout', 'LoadDist',
    'MaskMergeMode', 'MaskReg', 'MaskType', 'Max', 'Min', 'Pipe',
    'Position', 'PostMode', 'QBuff', 'QEvent', 'Reg', 'RegLayout',
    'RegList', 'RoundMode', 'SEvent', 'StoreDist', 'TBuff', 'TEvent',
    'Tensor', 'Var', 'VcMutex', 'VfPipe', 'abs', 'abssub',
    'add', 'adds', 'allcube_ready', 'allcube_wait', 'allvec_ready', 'allvec_wait',
    'arange', 'atomic_add', 'atomic_max', 'atomic_min', 'auto_sync', 'axpy',
    'b1', 'bar_all', 'bar_fix', 'bar_m', 'bar_mte1', 'bar_mte2',
    'bar_mte3', 'bar_v', 'barrier', 'bf16', 'cadd', 'cast',
    'cgadd', 'cgmax', 'cgmin', 'clean_dcache', 'clear_spr', 'cmax',
    'cmin', 'compare', 'conv2d', 'cpadd', 'cube_ready', 'cube_scope',
    'cvt', 'deinterleave', 'div', 'dup', 'e8m0_to_fp32', 'exp',
    'expsub', 'f16', 'f32', 'fp16_to_hif8', 'fp16_to_hifloat8',
    'fp32_to_e8m0', 'fp32_to_fp4_e1m2', 'fp32_to_fp4_e2m1', 'fp32_to_hif8', 'fp32_to_hifloat8', 'fp4_e1m2_to_fp32',
    'fp4_e2m1_to_fp32', 'func', 'gather', 'gather_mask', 'get_saturation_flag', 'gm_to_l1',
    'gm_to_l1_dn2nz', 'gm_to_l1_mx_scale', 'gm_to_l1_mx_scale_nd2nz', 'gm_to_l1_nd2nz', 'gm_to_l1_pad', 'gm_to_ub_nd_dma',
    'gm_to_ub_nd_dma_transpose', 'gm_to_ub_pad', 'hif8_to_fp32', 'hifloat8_to_fp32', 'histograms', 'i16',
    'i32', 'i64', 'i8', 'img2col', 'interleave', 'intracore_allvec_ready',
    'intracore_allvec_wait', 'kernel_dump_tensor', 'kernel_print', 'l0c_to_gm_nz2dn', 'l0c_to_gm_nz2nd', 'l0c_to_gm_nz2nz',
    'l0c_to_l1', 'l0c_to_ub', 'l1_to_bt', 'l1_to_l0', 'l1_to_l0_mx', 'l1_to_l0a_img2col',
    'ln', 'log', 'log10', 'log2', 'lrelu', 'mask_and',
    'mask_deinterleave', 'mask_interleave', 'mask_mov', 'mask_not', 'mask_or', 'mask_pack',
    'mask_sel', 'mask_to_ub', 'mask_unpack', 'mask_xor', 'matmul', 'matmul_mx',
    'mergesort4', 'mergesort_2seq', 'mmad', 'mmad_mx', 'move_mask_spr', 'mul',
    'muladddst', 'muldstadd', 'muls', 'mulscast', 'neg', 'pack',
    'pack_int4_to_int32', 'pack_signed_int4', 'pack_signed_int4_to_int32', 'prelu', 'print_reg', 'reg_to_ub',
    'reg_to_ub_continuous', 'reg_to_ub_downsample', 'reg_to_ub_interleave', 'reg_to_ub_normal', 'reg_to_ub_pack4', 'reg_to_ub_scatter',
    'reg_to_ub_single', 'reg_to_ub_unalign', 'reg_to_ub_unalign_once', 'reg_to_ub_unalign_post', 'reinterpret', 'relu',
    'reset_cache', 'reset_mask', 'scalar_abs', 'scalar_sqrt', 'select', 'set_constant_to_l1',
    'set_hf32', 'set_mask', 'set_mask_by_count', 'set_mask_normal', 'set_saturation_flag', 'setflag',
    'shiftl', 'shiftls', 'shiftr', 'shiftrs', 'sim_dump_tensor', 'sim_print',
    'simt_atomic_add', 'simt_atomic_and', 'simt_atomic_cas', 'simt_atomic_dec', 'simt_atomic_exch', 'simt_atomic_inc',
    'simt_atomic_max', 'simt_atomic_min', 'simt_atomic_or', 'simt_atomic_sub', 'simt_atomic_xor', 'simt_block_idx',
    'simt_block_num', 'simt_ceil', 'simt_cos', 'simt_exp', 'simt_exp2', 'simt_ffs',
    'simt_floor', 'simt_fma', 'simt_fmod', 'simt_isfinite', 'simt_isinf', 'simt_isnan',
    'simt_log', 'simt_log1p', 'simt_log2', 'simt_mul_hi', 'simt_popc', 'simt_rint',
    'simt_round', 'simt_rsqrt', 'simt_sin', 'simt_tanh', 'simt_thread_barrier', 'simt_thread_id',
    'simt_thread_num', 'simt_threadfence', 'simt_threadfence_block', 'simt_trunc', 'sort32', 'split_workspace',
    'sqrt', 'squeeze', 'static_assert', 'static_print', 'sub', 'u16',
    'u32', 'u64', 'u8', 'ub_cursor', 'ub_to_gm_pad', 'ub_to_l1',
    'ub_to_l1_nd2nz', 'ub_to_l1_nz', 'ub_to_mask', 'ub_to_reg', 'ub_to_reg_brcb', 'ub_to_reg_continuous',
    'ub_to_reg_downsample', 'ub_to_reg_gather', 'ub_to_reg_gatherb', 'ub_to_reg_interleave', 'ub_to_reg_normal', 'ub_to_reg_single',
    'ub_to_reg_unalign', 'ub_to_reg_unalign_once', 'ub_to_reg_unalign_pre', 'ub_to_reg_unpack', 'ub_to_reg_unpack4', 'ub_to_reg_upsample',
    'ub_to_ub', 'unalign_reg_for_load', 'unalign_reg_for_store', 'unpack_int32_to_int4', 'unpack_int32_to_signed_int4', 'unpack_signed_int4',
    'unroll', 'unsqueeze', 'update_mask', 'vand', 'var_add', 'var_and',
    'var_div', 'var_inv', 'var_mod', 'var_mul', 'var_or', 'var_shl',
    'var_shr', 'var_sub', 'var_xor', 'vcopy', 'vec_ready', 'vec_scope',
    'vf_barrier', 'vmax', 'vmaxs', 'vmin', 'vmins', 'vmod',
    'vnot', 'vor', 'vxor', 'wait_cube', 'wait_vec', 'waitflag',
    'zero_mxfp8_l1_padding',
]
