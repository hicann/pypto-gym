# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The c310 (a5) tables: what each Lowered op prints as.

Kernel-level ops print as one call of the wrapper named after them in ``tensorutils_cce.h``; the wrapper
names and argument orders are fixed by the header, the printer only massages attributes (this file lists
the names so the coverage report can enumerate them). ``@vf`` ops print as the compiler's vector
intrinsics on ``vector_*`` registers — the tables below map IR idents to the intrinsic names and the
compile-time tag constants of ``__clang_cce_vector_intrinsics.h`` (CANN's ``dav_c310`` register
implementation is the reference for every pairing).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------- kernel level

TASK_TYPE = {"mix": "KERNEL_TYPE_MIX_AIC_1_2", "vec": "KERNEL_TYPE_AIV_ONLY", "cube": "KERNEL_TYPE_AIC_ONLY"}

# opcode -> wrapper (documentation / coverage; the printer builds the calls)
KERNEL_WRAPPERS = {
    "dma.gm_to_ub.pad": "gm_to_ub_pad", "dma.ub_to_gm.pad": "ub_to_gm_pad",
    "dma.ub_to_ub": "ub_to_ub", "dma.ub_to_l1": "ub_to_l1",
    "dma.ub_to_l1.nd2nz": "ub_to_l1_nd2nz", "dma.ub_to_l1.nz": "ub_to_l1_nz",
    "dma.gm_to_l1": "gm_to_l1", "dma.gm_to_l1.pad": "gm_to_l1_pad",
    "dma.gm_to_l1.nd2nz": "gm_to_l1_nd2nz", "dma.gm_to_l1.dn2nz": "gm_to_l1_dn2nz",
    "dma.gm_to_l1.mx_scale_nd2nz": "gm_to_l1_mx_scale_nd2nz",
    "dma.set_constant_to_l1": "set_constant_to_l1",
    "dma.l1_to_l0": "l1_to_l0", "dma.l1_to_l0.mx": "l1_to_l0_mx",
    "dma.l1_to_l0.img2col": "l1_to_l0_img2col", "dma.l1_to_bt": "l1_to_bt",
    "cube.mmad": "mmad / mmad_bias", "cube.mmad.mx": "mmad_mx / mmad_mx_bias",
    "dma.l0c_to_gm.nz2nd": "l0c_to_gm_nz2nd", "dma.l0c_to_gm.nz2nz": "l0c_to_gm_nz2nz",
    "dma.l0c_to_gm.nz2dn": "l0c_to_gm_nz2dn", "dma.l0c_to_l1": "l0c_to_l1",
    "dma.l0c_to_ub": "l0c_to_ub",
    "sync.event": "SEvent / DEvent / TEvent / QEvent (Event beyond depth 4)", "sync.set": "Event::set", "sync.wait": "Event::wait",
    "sync.set_all": "Event::set_all", "sync.release": "Event::release",
    "sync.set_flag": "SetFlag", "sync.wait_flag": "WaitFlag", "sync.barrier": "PipeBarrier",
    "sync.crosscore.cube_ready": "CUBE_READY", "sync.crosscore.wait_vec": "WAIT_VEC",
    "sync.crosscore.vec_ready": "VEC_READY", "sync.crosscore.wait_cube": "WAIT_CUBE",
    "sync.crosscore.allcube_ready": "ALLCUBE_READY", "sync.crosscore.allcube_wait": "ALLCUBE_WAIT",
    "sync.crosscore.allvec_ready": "ALLVEC_READY", "sync.crosscore.allvec_wait": "ALLVEC_WAIT",
    "sync.crosscore.intracore_allvec_ready": "INTRACORE_ALLVEC_READY",
    "sync.crosscore.intracore_allvec_wait": "INTRACORE_ALLVEC_WAIT",
    "core.cube_idx": "GetCubeIdx", "core.cube_num": "GetCubeNum", "core.vec_idx": "GetVecIdx",
    "core.vec_num": "GetVecNum", "core.sub_block_idx": "GetSubBlockIdx",
    "core.set_hf32": "SetHF32Mode", "core.clean_dcache": "DataCacheCleanAndInvalid",
    "vec.set_mask": "SetVectorMask", "vec.reset_mask": "ResetMask",
    "vec.set_mask_by_count": "SetVectorMaskByCount", "vec.set_mask_count": "SetMaskCount",
    "vec.set_mask_normal": "SetMaskNorm", "vec.set_mask_counter": "SetVectorMask",
    "atomic.begin": "SetAtomicOpAdd / SetAtomicOpMax / SetAtomicOpMin", "atomic.set_type": "SetAtomicType", "atomic.end": "SetAtomicNone",
    "simt.launch": "simt::launch", "cf.call": "(direct call)",
}

CROSSCORE = {
    "sync.crosscore.cube_ready": "CUBE_READY", "sync.crosscore.wait_vec": "WAIT_VEC",
    "sync.crosscore.vec_ready": "VEC_READY", "sync.crosscore.wait_cube": "WAIT_CUBE",
    "sync.crosscore.allcube_ready": "ALLCUBE_READY", "sync.crosscore.allcube_wait": "ALLCUBE_WAIT",
    "sync.crosscore.allvec_ready": "ALLVEC_READY", "sync.crosscore.allvec_wait": "ALLVEC_WAIT",
    "sync.crosscore.intracore_allvec_ready": "INTRACORE_ALLVEC_READY",
    "sync.crosscore.intracore_allvec_wait": "INTRACORE_ALLVEC_WAIT",
}

DUAL_MODE = {"single": 0, "splitm": 1, "splitn": 2}

# ---------------------------------------------------------------------------- vf level

# vf.load_cont mode -> vlds distribution tag
LOAD_DIST = {
    "norm": "NORM", "ds_b8": "DS_B8", "ds_b16": "DS_B16", "us_b8": "US_B8", "us_b16": "US_B16",
    "brc_b8": "BRC_B8", "brc_b16": "BRC_B16", "brc_b32": "BRC_B32", "e2b_b16": "E2B_B16", "e2b_b32": "E2B_B32",
    "unpack_b8": "UNPK_B8", "unpack_b16": "UNPK_B16", "unpack_b32": "UNPK_B32", "unpack4_b8": "UNPK4_B8",
}
# vf.load_interleave mode -> two-register vlds tag
LOAD_INTLV_DIST = {"dintlv_b8": "DINTLV_B8", "dintlv_b16": "DINTLV_B16", "dintlv_b32": "DINTLV_B32"}
# vf.store_cont mode -> vsts distribution tag ("norm" takes the register width)
STORE_DIST = {
    "norm_b8": "NORM_B8", "norm_b16": "NORM_B16", "norm_b32": "NORM_B32",
    "pack_b16": "PK_B16", "pack_b32": "PK_B32", "pack_b64": "PK_B64", "pack4_b32": "PK4_B32",
    "first_element_b8": "ONEPT_B8", "first_element_b16": "ONEPT_B16", "first_element_b32": "ONEPT_B32",
}
STORE_INTLV_DIST = {"intlv_b8": "INTLV_B8", "intlv_b16": "INTLV_B16", "intlv_b32": "INTLV_B32"}

MASK_PATTERN = {
    "all": "PAT_ALL", "none": "PAT_ALLF", "vl1": "PAT_VL1", "vl2": "PAT_VL2", "vl3": "PAT_VL3", "vl4": "PAT_VL4",
    "vl8": "PAT_VL8", "vl16": "PAT_VL16", "vl32": "PAT_VL32", "vl64": "PAT_VL64", "vl128": "PAT_VL128",
    "h": "PAT_H", "q": "PAT_Q", "m3": "PAT_M3", "m4": "PAT_M4",
}

ROUND = {"none": "ROUND_R", "rint": "ROUND_R", "round": "ROUND_A", "floor": "ROUND_F", "ceil": "ROUND_C",
         "trunc": "ROUND_Z", "odd": "ROUND_O", "hybrid": "ROUND_H"}
PART = {"zero": "PART_EVEN", "one": "PART_ODD"}
PART_T = {"zero": "PART_P0", "one": "PART_P1", "two": "PART_P2", "three": "PART_P3"}
MERGE = {"zeroing": "MODE_ZEROING", "merging": "MODE_MERGING"}
HILO = {"lowest": "LOWER", "highest": "HIGHER"}
ORDER = {"increase": "INC_ORDER", "decrease": "DEC_ORDER"}

# (src, dst) of vf.barrier -> mem_bar tag
# Block copies (vsldb / vsstb) print with the register's own element type where the compiler header instantiates the
# form (D-057): a block copy issued through the unsigned carrier — `vsldb((vector_u32&)a, (__ubuf__ uint32_t*)p, …)` for
# a vector_f32 `a` — is type punning around the intrinsic, and at the board's -O3 the consumers of `a` read its previous
# value (the vsldb hazard of D-052 / D-055; AscendC's LoadAlign<T, DATA_BLOCK_COPY> uses T's own overload and is immune).
# dtype name -> the header's pointer element type (`__VF_VSLDB(LT, ST, NUM)`); hif8 and the complex carriers have none.
VSLDB_ELEM: dict[str, str] = {
    "i8": "int8_t", "u8": "uint8_t", "i16": "int16_t", "u16": "uint16_t", "i32": "int32_t", "u32": "uint32_t",
    "f16": "half", "bf16": "bfloat16_t", "f32": "float",
    "e4m3": "float8_e4m3_t", "e5m2": "float8_e5m2_t", "e8m0": "float8_e8m0_t",
    "fp4_e2m1": "float4_e2m1x2_t", "fp4_e1m2": "float4_e1m2x2_t",
}
VSSTB_ELEM: dict[str, str] = {k: v for k, v in VSLDB_ELEM.items() if not k.startswith("fp4")}

MEM_BAR = {
    ("vec_store", "vec_load"): "VST_VLD", ("vec_load", "vec_store"): "VLD_VST", ("vec_store", "vec_store"): "VST_VST",
    ("vec_store", "scalar_load"): "VST_LD", ("vec_store", "scalar_store"): "VST_ST", ("vec_load", "scalar_store"): "VLD_ST",
    ("scalar_store", "vec_load"): "ST_VLD", ("scalar_store", "vec_store"): "ST_VST", ("scalar_load", "vec_store"): "LD_VST",
    ("vec_all", "vec_all"): "VV_ALL", ("vec_all", "scalar_all"): "VS_ALL", ("scalar_all", "vec_all"): "SV_ALL",
}

# opcode -> intrinsic, masked with a merge mode: f(dst, src, mask, MODE)
UNARY = {"vf.exp": "vexp", "vf.ln": "vln", "vf.log": "vln", "vf.abs": "vabs", "vf.sqrt": "vsqrt", "vf.relu": "vrelu",
         "vf.neg": "vneg", "vf.not": "vnot"}
# Intrinsic form: f(dst, src0, src1, mask, MODE)
BINARY = {"vf.add": "vadd", "vf.sub": "vsub", "vf.mul": "vmul", "vf.div": "vdiv", "vf.max": "vmax", "vf.min": "vmin",
          "vf.and": "vand", "vf.or": "vor", "vf.xor": "vxor", "vf.shiftl": "vshl", "vf.shiftr": "vshr",
          "vf.prelu": "vprelu", "vf.abssub": "vabsdif",
          "vf.muldstadd": "vmadd",  # dst = dst * src0 + src1
          "vf.muladddst": "vmula"}  # dst = src0 * src1 + dst
# Intrinsic form: f(dst, src, scalar, mask, MODE)
SCALAR_BINARY = {"vf.adds": "vadds", "vf.muls": "vmuls", "vf.maxs": "vmaxs", "vf.mins": "vmins",
                 "vf.shiftls": "vshls", "vf.shiftrs": "vshrs", "vf.lrelu": "vlrelu", "vf.axpy": "vaxpy"}
# Intrinsic form: f(dst, src, mask, MODE)
REDUCE = {"vf.cadd": "vcadd", "vf.cmax": "vcmax", "vf.cmin": "vcmin", "vf.cgadd": "vcgadd", "vf.cgmax": "vcgmax",
          "vf.cgmin": "vcgmin", "vf.cpadd": "vcpadd"}
CMP = {"lt": "lt", "le": "le", "gt": "gt", "ge": "ge", "eq": "eq", "ne": "ne"}
# mask register ops: f(dst, src0, src1, mask) / f(dst, src, mask)
MASK_BINARY = {"vf.mask_and": "pand", "vf.mask_or": "por", "vf.mask_xor": "pxor"}

# vf.cast argument shapes by (dst dtype, src dtype); the value names the tag sequence after ``mask``:
#   part          (dst, src, mask, PART, MODE)                      widening / same-size, no rounding
#   part_t        (dst, src, mask, PART_Pk, MODE)                   4x widening
#   sat_part      (dst, src, mask, RS, PART, MODE)                  narrowing integers
#   sat_part_t    (dst, src, mask, RS, PART_Pk, MODE)               4x narrowing integers
#   rnd_sat_part  (dst, src, mask, ROUND, RS, PART, MODE)           float -> narrower float / int
#   rnd_sat_part_t(dst, src, mask, ROUND, RS, PART_Pk, MODE)        float -> 8-bit float
#   rnd_sat       (dst, src, mask, ROUND, RS, MODE)                 same-size float -> int
#   sat_rnd       (dst, src, mask, RS, ROUND, MODE)                 bf16 -> f16
#   rnd_part      (dst, src, mask, ROUND, PART, MODE)               f16 -> s32
#   rnd_part_t    (dst, src, mask, ROUND, PART_Pk, MODE)            bf16 -> fp4
#   rnd           (dst, src, mask, ROUND, MODE)                     int -> same-size float, f16 -> bf16
#   b64_from_f32  vcvt(dst, src, ROUND, RS)                         f32 -> s64 (two-register form, no mask)
#   f32_from_b64  vcvt(dst, src, ROUND)                             s64 -> f32
#   b64_widen     vcvt(dst, src)                                    s32 -> s64 / u32 -> u64
CAST_SHAPES: dict[tuple[str, str], str] = {}


def _cast(shape: str, *pairs: tuple[str, str]) -> None:
    for p in pairs:
        CAST_SHAPES[p] = shape


_cast("part", ("u16", "u8"), ("i16", "i8"), ("u32", "u16"), ("u32", "i16"), ("i32", "i16"), ("f32", "f16"),
      ("f32", "bf16"), ("f16", "hif8"), ("f16", "u8"), ("f16", "i8"), ("f32", "i16"))
_cast("part_t", ("u32", "u8"), ("i32", "i8"), ("f32", "hif8"), ("f32", "e4m3"), ("f32", "e5m2"),
      ("bf16", "fp4_e2m1"), ("bf16", "fp4_e1m2"))
_cast("sat_part", ("u8", "u16"), ("u8", "i16"), ("u16", "u32"), ("i16", "u32"), ("u16", "i32"), ("i16", "i32"))
_cast("sat_part_t", ("u8", "u32"), ("u8", "i32"))
_cast("rnd_sat_part", ("i16", "f32"), ("u8", "f16"), ("i8", "f16"), ("i32", "bf16"), ("f16", "f32"), ("bf16", "f32"),
      ("hif8", "f16"))
_cast("rnd_sat_part_t", ("hif8", "f32"), ("e5m2", "f32"), ("e4m3", "f32"))
_cast("rnd_sat", ("i32", "f32"), ("i16", "f16"))
_cast("sat_rnd", ("f16", "bf16"))
_cast("rnd_part", ("i32", "f16"))
_cast("rnd_part_t", ("fp4_e2m1", "bf16"), ("fp4_e1m2", "bf16"))
_cast("rnd", ("f16", "i16"), ("f32", "i32"), ("bf16", "f16"))
# The four-position (b4 / b8 <-> b16 / b32) pairs. `i4` is the compiler's `vector_s4x2`, a PACKED PAIR
# like fp4: the instruction reads and writes two elements at a time, so its layout selector is
# PART_P0..P3 (the `_t` shape family) and a narrowing cast's mask counts even positions (Cast.md's
# last constraint).
# e8m0 <-> bf16 is deliberately NOT here. The Cast API lists both directions for this device and there
# is still no such instruction: bisheng guards `vcvt_bf162e8m0` to __NPU_ARCH__ 9201 / 9202 and
# declares no reverse form at all, and AscendC's own dav_3510 `Cast` synthesises the pair out of
# shifts and the integer part-casts above. This table holds `vcvt` forms, so the recipe lives where a
# recipe belongs -- `tests/kernels/a5/samples/cast_e8m0.py`, board-exact both ways.
_cast("rnd_sat_part_t", ("i4", "f16"))
_cast("sat_part_t", ("i4", "i16"))
_cast("part_t", ("bf16", "i4"), ("f16", "i4"), ("i16", "i4"))

# What the c310 cast matrix does NOT have. Not a header accident and not something a wrapper can add:
# neither the Cast API doc (asc-devkit 9.2.0-beta.2, tables 3 and 6-9, "Ascend 950PR/950DT: 支持") nor
# `asc/include/c_api/reg_compute/reg_convert.h` (whose whole body is `#if __NPU_ARCH__ == 3510`) has a
# row for any of them, and bisheng declares no builtin under any vcvtii / vcvtif / vcvtfi spelling
# (compile probe with a positive control per family, 2026-09-05):
#   unsigned <-> float  u16 -> f16, u32 -> f32, f16 -> u16, f32 -> u16, f32 -> u32
#   narrowing to int8   i16 / i32 / u16 / u32 -> i8
#   mixed-sign widen    u8 -> i16, u16 -> i32, i8 -> u16
# They are CceGaps, and reaching them means going through the signed or the wider type.
_cast("b64_from_f32", ("i64", "f32"))
_cast("f32_from_b64", ("f32", "i64"))
_cast("b64_widen", ("i64", "i32"), ("u64", "u32"), ("u64", "i64"), ("i64", "u64"),
      # This argument-free form discards high bits even in global saturation mode (D-233).
      # AscendC's saturating API uses a different sequence; an explicit SAT request is a gap here.
      ("i32", "i64"))

# The rounding modes the header's static_assert admits for the pairs whose vcvt takes one (bare vcvt on c310 = 3510;
# f32 -> bf16 has two guarded forms, the intersection is kept). Anything else is a CceGap naming these.
_FIVE = frozenset({"ROUND_R", "ROUND_A", "ROUND_F", "ROUND_C", "ROUND_Z"})
CAST_ROUNDS: dict[tuple[str, str], frozenset[str]] = {
    ("f16", "f32"): frozenset({"ROUND_R", "ROUND_A", "ROUND_F", "ROUND_C", "ROUND_Z", "ROUND_O"}),
    ("bf16", "f32"): frozenset({"ROUND_R", "ROUND_A", "ROUND_F", "ROUND_C", "ROUND_Z"}),
    ("hif8", "f16"): frozenset({"ROUND_A", "ROUND_H"}),
    ("hif8", "f32"): frozenset({"ROUND_A", "ROUND_H"}),
    ("e4m3", "f32"): frozenset({"ROUND_R"}),
    ("e5m2", "f32"): frozenset({"ROUND_R"}),
    ("fp4_e2m1", "bf16"): frozenset({"ROUND_R", "ROUND_A", "ROUND_F", "ROUND_C", "ROUND_Z"}),
    ("fp4_e1m2", "bf16"): frozenset({"ROUND_R", "ROUND_A", "ROUND_F", "ROUND_C", "ROUND_Z"}),
    # The thirteen the Cast API doc constrains and this table did not. A pair with no entry here is
    # UNCONSTRAINED in the printer, so leaving them out let ROUND_O / ROUND_H print and fail in the
    # compiler instead of at the gap -- the same shape of bug D-217 fixed for the mask SPR.
    ("f16", "bf16"): _FIVE, ("i32", "bf16"): _FIVE,
    ("bf16", "f16"): _FIVE, ("i16", "f16"): _FIVE, ("i32", "f16"): _FIVE,
    ("i8", "f16"): _FIVE, ("u8", "f16"): _FIVE, ("i4", "f16"): _FIVE,
    ("i16", "f32"): _FIVE, ("i32", "f32"): _FIVE, ("i64", "f32"): _FIVE,
    ("f16", "i16"): _FIVE, ("f32", "i32"): _FIVE, ("f32", "i64"): _FIVE,
}


# (dst, src) pairs whose vcvt the compiler header declares zeroing-only on c310 — its static_assert allows MODE_MERGING
# for DAV_920R1 alone (the __VF_VCVT integer widen / narrow forms, __VF_VCVTFF_RND_SAT_PART / _PP, __VF_VCVTFF_RND_PP).
# Merging casts: there are none on c310. Every one of the 55 rows in the Cast API doc's tables 6-9
# reads `MaskMergeMode::ZEROING`, and a compile probe agrees -- ten pairs, MODE_ZEROING accepted on
# all ten and MODE_MERGING refused on all ten, eight of which the old hand-written list of 27 pairs
# did NOT name, so those eight printed a merging cast that only failed in the compiler. The list was
# a sample of the rule; the rule is every pair.
MERGE_REJECTED: frozenset[tuple[str, str]] = frozenset(CAST_SHAPES)

# every intrinsic / tag name the vf printer can emit (coverage report)
VF_INTRINSICS = sorted({
    "vlds", "vsts", "vsldb", "vsstb", "vldas", "vldus", "vstus", "vstas", "plds", "psts", "pset_b8", "pset_b16",
    "pset_b32", "plt_b8", "plt_b16", "plt_b32", "plt_2xvl_b64", "movp_b16", "movp_b32", "pand", "por", "pxor", "pnot",
    "pmov", "psel", "ppack", "punpack", "pintlv_b8", "pintlv_b16", "pintlv_b32", "pdintlv_b8", "pdintlv_b16",
    "pdintlv_b32", "vsel", "vselr", "vgather2", "vgatherb", "vscatter", "vsqz", "vusqz", "vdup", "vbr", "vmov", "vci",
    "vintlv", "vdintlv", "vpack", "vunpack", "vcvt", "vmod", "vexpdif", "vmulscvt", "dhistv2", "chistv2", "mem_bar", "sprclr",
    *UNARY.values(), *BINARY.values(), *SCALAR_BINARY.values(), *REDUCE.values(),
    *(f"vcmp_{c}" for c in CMP), *(f"vcmps_{c}" for c in CMP),
})

__all__ = ["TASK_TYPE", "KERNEL_WRAPPERS", "CROSSCORE", "DUAL_MODE", "LOAD_DIST", "LOAD_INTLV_DIST", "STORE_DIST",
           "STORE_INTLV_DIST", "MASK_PATTERN", "ROUND", "PART", "PART_T", "MERGE", "HILO", "ORDER", "MEM_BAR", "UNARY",
           "BINARY", "SCALAR_BINARY", "REDUCE", "CMP", "MASK_BINARY", "CAST_SHAPES", "CAST_ROUNDS", "VSLDB_ELEM", "VSSTB_ELEM",
           "VF_INTRINSICS"]
