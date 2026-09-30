# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""C++ spellings the cce printer needs: dtypes, vector register types, pipes, literals, identifiers.

Everything here is a table. The spellings are the compiler's (``half``, ``bfloat16_t``, ``vector_f32``,
``PIPE_MTE2`` …) or the ones ``tensorutils_cce.h`` defines on top of them (``fp8_e4m3fn_t`` …).
"""

from __future__ import annotations

import re

from ...ir import Literal
from ...ir.types import DType, MemType
from .tu_names import NAMES as TU_NAMES

# IR dtype name -> C scalar / element type
CMP_OPS = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}

CTYPE: dict[str, str] = {
    "b1": "bool",
    "i8": "int8_t", "u8": "uint8_t", "i16": "int16_t", "u16": "uint16_t",
    "i32": "int32_t", "u32": "uint32_t", "i64": "int64_t", "u64": "uint64_t",
    "f16": "half", "bf16": "bfloat16_t", "f32": "float",
    "e4m3": "fp8_e4m3fn_t", "e5m2": "fp8_e5m2_t", "hif8": "hifloat8_t", "e8m0": "fp8_e8m0_t",
    "fp4_e2m1": "fp4x2_e2m1_t", "fp4_e1m2": "fp4x2_e1m2_t",
    "c32": "uint32_t", "c64": "uint64_t",  # complex tensors are carried as their bit patterns (no complex C type on c310)
    "i4": "int4b_t",  # packed signed int4 (c220 mmad operand views; the loads move the int32 carriers)
}

# IR dtype name -> the compiler's vector register type (bare VF code)
VTYPE: dict[str, str] = {
    "i8": "vector_s8", "u8": "vector_u8", "i16": "vector_s16", "u16": "vector_u16",
    "i32": "vector_s32", "u32": "vector_u32", "i64": "vector_2xvl_s64", "u64": "vector_2xvl_u64",
    "f16": "vector_f16", "bf16": "vector_bf16", "f32": "vector_f32",
    "e4m3": "vector_f8e4m3", "e5m2": "vector_f8e5m2", "hif8": "vector_hif8", "e8m0": "vector_f8e8m0",
    "i4": "vector_s4x2", "fp4_e2m1": "vector_f4e2m1x2", "fp4_e1m2": "vector_f4e1m2x2",
    "c32": "vector_u32", "c64": "vector_2xvl_u64",  # complex registers travel in their carrier; the halves are the parts
}

def entry_tensors(params: list, outputs: list[str]) -> list:
    """The extern "C" entry's GM_ADDR parameters in the custom op's tensor order: the inputs, then the outputs,
    each in signature order (RFC-0007 §1). The op hands its tensors over in that order, whatever the signature."""
    tensors = [p for p in params if isinstance(p.type, MemType)]
    return [p for p in tensors if p.name not in outputs] + [p for p in tensors if p.name in outputs]


# element width in bytes (packed 4-bit dtypes count their carrier byte)
def esize(dt: DType) -> int:
    return max(dt.bits, 8) // 8


def is_packed(dt: DType) -> bool:
    return dt.bits < 8


def ctype(dt: DType) -> str:
    try:
        return CTYPE[dt.name]
    except KeyError:
        raise KeyError(f"dtype {dt.name} has no C spelling in the cce backend") from None


def vtype(dt: DType) -> str:
    try:
        return VTYPE[dt.name]
    except KeyError:
        raise KeyError(f"dtype {dt.name} has no vector register type in the cce backend") from None


def carrier_vtype(dt: DType) -> str:
    """The unsigned-integer register type of the same width: loads / stores / gathers are bit copies and the
    compiler declares those intrinsics for the integer element types."""
    return {1: "vector_u8", 2: "vector_u16", 4: "vector_u32", 8: "vector_2xvl_u64"}[esize(dt)]


def carrier_ctype(dt: DType) -> str:
    return {1: "uint8_t", 2: "uint16_t", 4: "uint32_t", 8: "uint64_t"}[esize(dt)]


def signed_carrier_vtype(dt: DType) -> str:
    return {1: "vector_s8", 2: "vector_s16", 4: "vector_s32", 8: "vector_2xvl_s64"}[esize(dt)]


def signed_carrier_ctype(dt: DType) -> str:
    return {1: "int8_t", 2: "int16_t", 4: "int32_t", 8: "int64_t"}[esize(dt)]


PIPE: dict[str, str] = {
    "MTE1": "PIPE_MTE1", "MTE2": "PIPE_MTE2", "MTE3": "PIPE_MTE3", "M": "PIPE_M", "V": "PIPE_V",
    "FIX": "PIPE_FIX", "S": "PIPE_S", "ALL": "PIPE_ALL",
}

POS: dict[str, str] = {
    "ub": "Position::UB", "l1": "Position::L1", "l0a": "Position::L0A", "l0b": "Position::L0B",
    "l0c": "Position::L0C", "bt": "Position::BT",
}

# <math.h> for C++17 on the A5 host (RFC-0007 §1): glibc 2.39's math/math.h, bits/mathcalls.h, bits/mathcalls-narrow.h and
# bits/iscanonical.h with the _GNU_SOURCE the C++ driver predefines, and libstdc++'s <math.h> `using std::abs`. Native
# Pro did not compile a SIMT launch of `remainder` (A5-UP-042); CCE did, and reserves these names by decision (I031).
_MATH_TYPES = ("", "f", "l", "f32", "f64", "f32x", "f64x", "f128")  # f128 where the compiler has _Float128 (GCC)
_MATH_FUNCTIONS = """acos asin atan atan2 cos sin tan cosh sinh tanh sincos acosh asinh atanh exp frexp ldexp log log10
    modf exp10 expm1 log1p logb exp2 log2 pow sqrt hypot cbrt ceil fabs floor fmod copysign nan j0 j1 jn y0 y1 yn erf
    erfc lgamma tgamma rint nextafter nextdown nextup remainder scalbn ilogb llogb scalbln nearbyint round trunc remquo
    lrint llrint lround llround fdim fmax fmin fma roundeven fromfp ufromfp fromfpx ufromfpx canonicalize fmaxmag fminmag
    fmaximum fminimum fmaximum_num fminimum_num fmaximum_mag fminimum_mag fmaximum_mag_num fminimum_mag_num totalorder
    totalordermag getpayload setpayload setpayloadsig""".split()
_MATH_CLASSIC = "isinf isnan finite drem significand gamma nexttoward scalb".split()  # double, float, long double only
_MATH_NARROW = ("f", ""), ("f", "l"), ("d", "l"), ("f32", "f32x"), ("f32", "f64"), ("f32", "f64x"), ("f32", "f128"), \
    ("f32x", "f64"), ("f32x", "f64x"), ("f32x", "f128"), ("f64", "f64x"), ("f64", "f128"), ("f64x", "f128")
_MATH_CONSTANTS = "M_E M_LOG2E M_LOG10E M_LN2 M_LN10 M_PI M_PI_2 M_PI_4 M_1_PI M_2_PI M_2_SQRTPI M_SQRT2 M_SQRT1_2".split()
_MATH_MACROS = """fpclassify signbit isfinite isnormal isgreater isgreaterequal isless islessequal islessgreater isunordered
    issignaling issubnormal iszero iscanonical iseqsig FP_NAN FP_INFINITE FP_ZERO FP_SUBNORMAL FP_NORMAL FP_ILOGB0
    FP_ILOGBNAN FP_LLOGB0 FP_LLOGBNAN FP_FAST_FMA FP_FAST_FMAF FP_FAST_FMAL FP_INT_UPWARD FP_INT_DOWNWARD FP_INT_TOWARDZERO
    FP_INT_TONEARESTFROMZERO FP_INT_TONEAREST HUGE_VAL HUGE_VALF HUGE_VALL INFINITY NAN SNAN SNANF SNANL MAXFLOAT
    MATH_ERRNO MATH_ERREXCEPT math_errhandling float_t double_t signgam abs""".split()
MATH_NAMES = frozenset(
    {name + t for name in (*_MATH_FUNCTIONS, *_MATH_CONSTANTS) for t in _MATH_TYPES} | {f"lgamma{t}_r" for t in _MATH_TYPES}
    | {name + t for name in _MATH_CLASSIC for t in _MATH_TYPES[:3]} | {f"HUGE_VAL_{t.upper()}" for t in _MATH_TYPES[3:]}
    | {f"SNAN{t.upper()}" for t in _MATH_TYPES[3:]} | set(_MATH_MACROS)
    | {r + op + a for r, a in _MATH_NARROW for op in ("add", "sub", "mul", "div", "sqrt", "fma")})

C_KEYWORDS = {
    "alignas", "alignof", "and", "asm", "auto", "bool", "break", "case", "catch", "char", "class", "const",
    "constexpr", "continue", "default", "delete", "do", "double", "else", "enum", "explicit", "export", "extern",
    "false", "float", "for", "friend", "goto", "if", "inline", "int", "long", "mutable", "namespace", "new",
    "noexcept", "not", "nullptr", "operator", "or", "private", "protected", "public", "register", "return",
    "short", "signed", "sizeof", "static", "struct", "switch", "template", "this", "throw", "true", "try",
    "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual", "void", "volatile", "while", "xor",
    "half", "workspace", "tiling", "tiling_data", "main",
    # the compiler's SIMT builtin variables: a local of the same name is a redefinition
    "block_idx", "block_num", "blockIdx", "blockDim", "threadIdx", "gridDim", "warpSize",
}
API_KEYWORDS = frozenset(C_KEYWORDS)  # what the custom op's interface respells too (api)
# Kernel translation unit only. f32 extrema call these unqualified (f32_extremum): a local of the same name would
# shadow them. The <math.h> names are reserved by decision, the unit's own functions and macros as measured (I031).
C_KEYWORDS |= {"max", "min", *MATH_NAMES, *TU_NAMES}


def kernel_reserved(s: str) -> bool:
    """Whether the kernel translation unit respells ``s`` (``emit.c_ident``): ``C_KEYWORDS``, or a C++-reserved
    spelling (``__`` or ``_`` and a capital) that ``api`` has not already respelled."""
    return s in C_KEYWORDS or (("__" in s or re.match(r"_[A-Z]", s) is not None) and not s.startswith("__"))


def api(name: str) -> str:
    """``name`` as the custom op's interface spells it: manifest ``name``, OpDef inputs, outputs and attributes, tiling
    fields, aclnn binding and harness. ``emit.c_ident`` also respells the rest of ``C_KEYWORDS`` inside the kernel
    translation unit; the interface keeps those names, because CANN's aclnn generator breaks an output name ending in
    ``_`` and a host binding named ``erf`` or ``max`` only shadows the library declaration (I031)."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not s or s[0].isdigit():
        s = "v" + s
    return s + "_" if s in API_KEYWORDS or s.startswith("__") else s


def bit_pattern(dt: DType, value: int | float | bool) -> int:
    """A scalar's BIT PATTERN in `dt`, zero-extended -- AscendC's ``GetScalarBitcodeValue``.

    The pad-value SPR (`set_mov_pad_val`) takes an encoding, not a number: a 1.0f pad is
    0x3F800000, not 1. Raises for a dtype whose encoding this cannot produce, so a caller must ask
    for a gap rather than send a wrong pattern to the device.
    """
    import struct

    if dt.name in ("f32", "f16", "bf16"):
        scalar = float(value)
        try:
            packed = struct.pack("<f", scalar)
        except OverflowError:
            packed = struct.pack("<f", float("-inf") if scalar < 0 else float("inf"))
        bits = struct.unpack("<I", packed)[0]
        if dt.name == "f32":
            return bits
        # The existing scalar literal convention narrows from FP32, not FP64.
        if dt.name == "f16":
            try:
                return struct.unpack("<H", struct.pack("<e", struct.unpack("<f", packed)[0]))[0]
            except OverflowError:
                return ((bits >> 16) & 0x8000) | 0x7C00
        if (bits & 0x7FFFFFFF) > 0x7F800000:
            return 0x7FC0  # BF16 scalar constructors canonicalize the NaN sign and payload.
        return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    if dt.kind == "int" and dt.bits in (8, 16, 32, 64):
        return int(value) & ((1 << dt.bits) - 1)
    if dt.bits == 8:  # hif8 / e4m3 / e5m2 / e8m0 travel as their byte
        return int(value) & 0xFF
    raise ValueError(f"no scalar bit pattern for {dt}")


def literal(v: Literal | int | float | bool, dt: DType | None = None) -> str:
    """A C literal. Floats are printed as ``float`` literals (the narrower float types take a cast where the
    intrinsic needs the exact type)."""
    x = v.value if isinstance(v, Literal) else v
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        if dt is not None and dt.is_float:
            return f"{float(x)!r}f"
        if x > 2**63 - 1:  # a decimal literal that no signed type holds: LL there is ill-formed C++
            return f"{x}ULL"
        if x > 2**31 - 1 or x < -(2**31):
            return f"{x}LL"
        return str(x)
    if isinstance(x, float):
        if x != x:
            return '__builtin_nanf("")'
        if x == float("inf"):
            return "__builtin_inff()"
        if x == float("-inf"):
            return "(-__builtin_inff())"
        s = repr(x)
        if "e" in s or "E" in s or "." in s:
            return s + "f"
        return s + ".0f"
    raise TypeError(f"not a literal: {v!r}")


def scalar_abs_expr(dtype: DType, value: str, *, arch: str, kind: str) -> str:
    """Target scalar spelling, shared with the PTO host-language printer."""
    if dtype.name.startswith("u") or dtype.name == "b1":
        return value
    if arch != "c310":
        return f"(({value}) < 0 ? -({value}) : ({value}))"
    if dtype.name == "bf16":
        raise ValueError("scalar abs requires a target-supported BF16 scalar conversion (M10-055)")
    if dtype.name not in ("i8", "i16", "i32", "i64", "f16", "f32"):
        raise ValueError(f"scalar abs of {dtype.name} has no C310 spelling")
    callee = "__builtin_fabsf" if kind == "simt" and dtype.is_float else "::abs"
    if dtype.name == "f16":
        return f"(half){callee}((float)({value}))"
    return f"{callee}({value})"


def f32_extremum(kind: str, a: str, b: str) -> str:
    """PyPTO Pro's f32 scalar min/max spelling, which A5 measured as IEEE minimum/maximum (RFC-0001 §6.16).
    Shared with the PTO host-language printer; the ``Min`` / ``Max`` helpers order NaN and zeros differently."""
    return f"{kind}((float)({a}), (float)({b}))"


__all__ = ["CTYPE", "VTYPE", "PIPE", "POS", "C_KEYWORDS", "API_KEYWORDS", "MATH_NAMES", "TU_NAMES", "api", "kernel_reserved", "ctype", "vtype", "carrier_vtype", "carrier_ctype",
           "signed_carrier_vtype", "signed_carrier_ctype", "esize", "is_packed", "literal"]
