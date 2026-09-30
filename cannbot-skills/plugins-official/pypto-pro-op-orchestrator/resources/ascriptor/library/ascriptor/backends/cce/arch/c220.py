# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""c220 (Ascend 910B, ``dav-2201``) facts for the cce printer (RFC-0008 phase B, D-061).

Every table was traced old AscendC handler -> CANN's ``dav_c220`` implementation -> the CCE intrinsic; the trace
with file / line quotes lives in the phase-B working notes (``tmp/a2/c220_mapping.md``, git-ignored) and can be
re-walked from ``<cann>/x86_64-linux/asc/impl/basic_api/dav_c220/kernel_operator_*_impl.h`` plus the typed
prototypes of ``__clang_cce_aicore_functions.h``. Conventions the printer relies on:

* the current VMASK applies to the strided vector ops; the counted forms are printed as CANN's own Level-2
  bracket (D-062): ``set_mask_count(); set_vector_mask(0, n); <op>; set_mask_norm(); set_vector_mask(-1, -1)``,
  no barrier — mask SPR writes dispatch in order with vector instructions on the V queue;
* ``vbrcb`` / ``vgatherb`` reset the mask to FULL themselves and ``vgather`` is printed with a full-mask set
  first (the AscendC impls do exactly this), so the printer re-arms a live explicit mask after them;
* strides are 32-byte blocks unless a table note says bytes / elements / 512-byte fractals;
* ``vec.scatter`` does not exist on c220 (both ``ScatterImpl`` overloads are ``ASCENDC_REPORT_NOT_SUPPORT``).
"""

from __future__ import annotations

# -- the vector ISA ---------------------------------------------------------------------------------------------------

# vec.<kind> -> intrinsic; dtype allow-lists are the dav_c220 static_asserts (IR dtype names).
VEC_BINARY = {
    "add": ("vadd", ("f16", "f32", "i16", "i32")),
    "sub": ("vsub", ("f16", "f32", "i16", "i32")),
    "mul": ("vmul", ("f16", "f32", "i16", "i32")),
    "div": ("vdiv", ("f16", "f32")),
    "max": ("vmax", ("f16", "f32", "i16", "i32")),
    "min": ("vmin", ("f16", "f32", "i16", "i32")),
    "and": ("vand", ("i16", "u16")),
    "or": ("vor", ("i16", "u16")),
    "muladddst": ("vmla", ("f16", "f32")),  # dst accumulates; (dst, src) pairs (f16,f16) (f32,f32) (f32,f16)
}
VEC_UNARY = {  # vec.<kind> -> intrinsic; blk strides print as uint16_t, rep strides as uint8_t (the impl's casts)
    "exp": ("vexp", ("f16", "f32")),
    "ln": ("vln", ("f16", "f32")),
    "abs": ("vabs", ("f16", "f32")),
    "rec": ("vrec", ("f16", "f32")),
    "sqrt": ("vsqrt", ("f16", "f32")),
    "rsqrt": ("vrsqrt", ("f16", "f32")),
    "not": ("vnot", ("i16", "u16")),
    "relu": ("vrelu", ("f16", "f32", "i32")),
}
VEC_SCALAR = {  # vec.<kind> -> (intrinsic, dtypes, trailing args after the strides)
    "adds": ("vadds", ("f16", "f32", "i16", "i32"), ""),
    "muls": ("vmuls", ("f16", "f32", "i16", "i32"), ""),
    "maxs": ("vmaxs", ("f16", "f32", "i16"), ", false, false"),  # two fixed mode bits on c220
    "mins": ("vmins", ("f16", "f32", "i16"), ", false, false"),
    "lrelu": ("vlrelu", ("f16", "f32"), ""),
    "shiftls": ("vshl", ("i16", "u16", "i32", "u32"), ""),
    "shiftrs": ("vshr", ("i16", "u16", "i32", "u32"), None),  # None: vshr takes the round_en flag
    "axpy": ("vaxpy", ("f16", "f32"), ""),  # dst accumulates; scalar in the src dtype
}
VEC_REDUCE = {  # vec.<kind> -> intrinsic; vcadd takes a trailing 0, vcmax / vcmin the ONLY_VALUE order
    "cadd": "vcadd", "cmax": "vcmax", "cmin": "vcmin",  # one result per repeat; dst_rep_stride in ELEMENTS (pairs for max/min)
    "cgadd": "vcgadd", "cgmax": "vcgmax", "cgmin": "vcgmin", "cpadd": "vcpadd",  # per block / pair; dst_rep_stride in blocks
}
VEC_CMP = ("lt", "gt", "eq", "le", "ge", "ne")  # vcmpv_<m> / vcmpvs_<m>; src f16 / f32 (i32: eq only); dst is the bit tensor

# vec.cast: (dst, src) -> (intrinsic base, the rounding suffixes that exist). Suffix by IR mode name below.
# half <- int32 is vconv_deq (needs the DEQSCALE SPR) and u8/i8 <- i16 do not exist: both are gaps.
CAST_INTRINSICS: dict[tuple[str, str], tuple[str, frozenset[str]]] = {
    ("f16", "i8"): ("vconv_s82f16", frozenset({""})),
    ("f16", "u8"): ("vconv_u82f16", frozenset({""})),
    ("f32", "i32"): ("vconv_s322f32", frozenset({"r", "f", "c", "a", "z", ""})),
    ("f32", "f16"): ("vconv_f162f32", frozenset({""})),
    ("i32", "f16"): ("vconv_f162s32", frozenset({"r", "f", "c", "a", "z"})),
    ("i8", "f16"): ("vconv_f162s8", frozenset({"r", "f", "c", "a", "z", ""})),
    ("u8", "f16"): ("vconv_f162u8", frozenset({"r", "f", "c", "a", "z", ""})),
    ("f16", "f32"): ("vconv_f322f16", frozenset({"r", "f", "c", "a", "z", "o", ""})),
    ("i32", "f32"): ("vconv_f322s32", frozenset({"r", "f", "c", "a", "z"})),
    ("i16", "f16"): ("vconv_f162s16", frozenset({"r", "f", "c", "a", "z"})),
    ("f16", "i16"): ("vconv_s162f16", frozenset({"r", "f", "c", "a", "z", ""})),
    ("f32", "f32"): ("vconv_f322f32", frozenset({"r", "f", "c", "a", "z"})),
    ("bf16", "f32"): ("vconv_f322bf16", frozenset({"r", "f", "c", "a", "z"})),
    ("i64", "f32"): ("vconv_f322s64", frozenset({"r", "f", "c", "a", "z"})),
    ("f32", "bf16"): ("vconv_bf162f32", frozenset({""})),
    ("i32", "bf16"): ("vconv_bf162s32", frozenset({"r", "f", "c", "a", "z"})),
    ("i16", "f32"): ("vconv_f322s16", frozenset({"r", "f", "c", "a", "z"})),
    ("f32", "i16"): ("vconv_s162f32", frozenset({""})),
    ("i16", "i32"): ("vconv_s322s16", frozenset({""})),
    ("i64", "i32"): ("vconv_s322s64", frozenset({""})),
    ("f32", "i64"): ("vconv_s642f32", frozenset({"r", "f", "c", "a", "z"})),
    ("i32", "i64"): ("vconv_s642s32", frozenset({""})),
}
CAST_SUFFIX = {"none": "", "rint": "r", "floor": "f", "ceil": "c", "round": "a", "trunc": "z", "odd": "o"}

# dup: vector_dup(dst, scalar, repeat, dstBlkStride, 1, dstRepStride, 0); dtypes:
DUP_DTYPES = ("f16", "bf16", "i16", "u16", "i32", "u32", "f32")
# brcb: ResetMask(); vbrcb through the uint16 / uint32 view of any 2- / 4-byte dtype (b16 -> uint16_t, b32 -> uint32_t).
# gather: full-mask (or the counted bracket); vgather(dst_u16_or_u32, offsets_u32, (uint32_t)src_base_byte_addr, dstRepStride, repeat).
# gatherb: ResetMask(); vgatherb(dst, offsets_u32, (uint32_t)src_base, (uint16_t)dstRepStride, (uint8_t)dstBlkStride, repeat).
# transdata5hd: set_va_reg_sb(VA0..VA3) over 16-entry uint64 address lists, then scatter_vnchwconv_b16 / _b32.
# sort32: vbitsort(dst, src, idx_u32, repeat) — f16 / f32. mergesort: vmrgsort4(dst, addr[4], lens_u64, config_u64).

# -- data movement ----------------------------------------------------------------------------------------------------

# gm <-> ub padded copies: copy_gm_to_ubuf_align_b8|b16|b32 / copy_ubuf_to_gm_align_*; blockLen and the GM-side gap in
# BYTES, the UB-side gap in 32-byte blocks; n_burst <= 4095 (the 12-bit field), blockLen % sizeof(T) == 0 on loads.
ALIGN_SUFFIX = {1: "b8", 2: "b16", 4: "b32", 8: "b32"}  # 8-byte dtypes ride the b32 form with doubled pads
# ub_to_ub: copy_ubuf_to_ubuf(dst, src, 0, nBurst, lenBurst, srcStride, dstStride) — all 32-byte blocks, PIPE_V.
# gm_to_l1 nd2nz: copy_gm_to_cbuf_multi_nd2nz_b8|b16|_b32s(dst, src, 0, ndNum=1, nValue=M, dValue=N, 0, srcDValue=N_src,
#   dstNzC0Stride=align16(M_dst), dstNzNStride=1, 0); fp32 tiles go ND -> ZZ instead: a loop of plain copies (RFC-0008 §5).
# gm_to_l1 plain: copy_gm_to_cbuf(dst, src, 0, nBurst, lenBurst, srcStride, dstStride, (pad_t)0).
# set_constant_to_l1: create_cbuf_matrix[_bf16](dst, ((int64_t)n_blocks << 16) | 1, value) — f32 / i32 through the
#   uint32 bitcode form, i16 / u16 through the half bitcode form.

# l1 -> l0 loads: (dst position, fp32-ZZ L1?, transposed?) -> intrinsic. Units: 512-byte fractals.
LOAD2D = {
    ("l0a", False): "load_cbuf_to_ca",
    ("l0b", False): "load_cbuf_to_cb",
    ("l0a", True): "load_cbuf_to_ca_transpose",  # the dedicated in-fractal transpose loads
    ("l0b", True): "load_cbuf_to_cb_transpose",
}
# l1 -> bt: copy_cbuf_to_bt((uint64_t)bt_byte_offset, src, isEnableConv, 1, lenBurst_64B, 0, 0); conv = f16 L1 -> f32 BT.
# mmad: mad(c, a, b[, bias_bt_addr], m, k, n, unitFlag, false, cmatrixSource, cmatrixInitVal); m rounds up to 16.
MMAD_TUPLES = (("i32", "i8", "i8"), ("f32", "f16", "f16"), ("f32", "f32", "f32"), ("f32", "bf16", "bf16"))  # (+ i4 later)
# l0c -> gm: nz2nd = set_nd_para((1<<32)|(1<<16)|1); [set_quant_pre(deq)]; pipe_barrier(PIPE_FIX);
#   copy_matrix_cc_to_gm(dst, src, 0, n, m, dstStride_elems, srcStride_rows16, unitFlag, QuantMode, relu, false, true).
# nz2nz: no set_nd_para, dstStride in 32-byte units, nz2ndEn false. l0c -> l1: copy_matrix_cc_to_cbuf(...).
FIXPIPE_QUANT = {  # (dst dtype, src dtype) -> QuantMode_t; scalar-quant modes also program set_quant_pre
    ("f32", "f32"): "NoQuant", ("f16", "f32"): "F322F16", ("bf16", "f32"): "F322BF16",
    ("f16", "i32"): "DEQF16", ("i8", "f32"): "QF322B8_PRE", ("u8", "f32"): "QF322B8_PRE", ("i8", "i32"): "REQ8",
    ("i32", "i32"): "NoQuant",
}
# atomics: set_atomic_add|max|min() + set_atomic_<dt>() before the stores, set_atomic_none() after.
ATOMIC_DTYPE = {"f32": "set_atomic_f32", "f16": "set_atomic_f16", "i16": "set_atomic_s16", "i32": "set_atomic_s32",
                "i8": "set_atomic_s8", "bf16": "set_atomic_bf16"}
# set_hf32: set_ctrl(sbitset1(get_ctrl(), 46)) / sbitset0 — bit 46 of CTRL.

# -- sync and entry ---------------------------------------------------------------------------------------------------

# set_flag / wait_flag pairs valid on c220 (HardEvent, kernel_event.h): the printer must keep V-side events out of
# the cube function and M / MTE1 / FIX events out of the vec function.
EVENT_PAIRS_VEC = frozenset({("MTE2", "V"), ("V", "MTE2"), ("MTE3", "V"), ("V", "MTE3"), ("V", "V"), ("S", "V"), ("V", "S"),
                             ("MTE2", "MTE3"), ("MTE3", "MTE2"), ("S", "MTE2"), ("MTE2", "S"), ("S", "MTE3"), ("MTE3", "S")})
EVENT_PAIRS_CUBE = frozenset({("MTE2", "MTE1"), ("MTE1", "MTE2"), ("MTE1", "M"), ("M", "MTE1"), ("M", "FIX"), ("FIX", "M"),
                              ("MTE2", "M"), ("M", "MTE2"), ("MTE2", "FIX"), ("FIX", "MTE2"), ("FIX", "S"), ("M", "S"),
                              ("FIX", "MTE3"), ("MTE1", "FIX"), ("FIX", "MTE1"), ("FIX", "FIX"), ("MTE3", "MTE2"), ("MTE2", "MTE3"),
                              ("MTE3", "MTE1"), ("MTE1", "MTE3"), ("S", "MTE2"), ("MTE2", "S")})
QUE_MAX_EVENT = 8  # event ids 0..7 per (src, dst) kind on 2201

# Cross-core sync (FFTS): ffts_cross_core_sync(pipe, 0x1 | (mode << 4) | (flag_id << 8)); waits are wait_flag_dev(id)
# on the scalar pipe (which is why every cross-core wait's pipe must be S on c220).
FFTS_MODE = {"pair": 0x2, "all": 0x0, "intra": 0x1}  # cube_ready / vec_ready; all*_ready; intracore_allvec_ready
FFTS_RESERVED_FLAGS = (11, 12, 13, 14)  # AscendC SyncAll's; user flags stay 0..7

# Kernel entry: KERNEL_TASK_TYPE_DEFAULT(v) = __builtin_cce_kernel_type_set(v).
KERNEL_TYPE = {"mix": 5, "vec": 0, "cube": 1}  # KERNEL_TYPE_MIX_AIC_1_2 / _AIV_ONLY / _AIC_ONLY
# The side split prints under the compiler predicates __DAV_C220_CUBE__ / __DAV_C220_VEC__; a vector core reads its
# pair index from get_subblockid(); PIPE_FIX spells (pipe_t)10 for the raw set_flag / pipe_barrier forms.

__all__ = ["VEC_BINARY", "VEC_UNARY", "VEC_SCALAR", "VEC_REDUCE", "VEC_CMP", "CAST_INTRINSICS", "CAST_SUFFIX",
           "DUP_DTYPES", "ALIGN_SUFFIX", "LOAD2D", "MMAD_TUPLES", "FIXPIPE_QUANT", "ATOMIC_DTYPE",
           "EVENT_PAIRS_VEC", "EVENT_PAIRS_CUBE", "QUE_MAX_EVENT", "FFTS_MODE", "FFTS_RESERVED_FLAGS", "KERNEL_TYPE"]
