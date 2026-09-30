# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``vf.*``: register-level micro ops of the a5 vector core (the old ``micro_*`` instructions); a5 only.

They appear only inside ``vf`` functions. Registers and masks are declared with ``vf.reg`` /
``vf.mask`` and written through ``dst`` operands, exactly as the DSL writes them; the predicate
is the optional ``mask`` attribute. Pipe is always V.
"""

from ._dsl import A5, MASK_ATTR, VF, A, N, R, Res, W, op, retire

IV = "int|value"
REG = "reg<*, *>"
UB = "ub<*, *>"
MASK = "mask<*>"


def vf(name: str, operands, attrs=(), results=(), legacy=(), doc: str = "", effects=("memory",)) -> None:
    op(f"vf.{name}", level="both", kinds=VF, side="vec", pipe="V", operands=tuple(operands), attrs=tuple(attrs),
       results=tuple(results), devices=A5, legacy=legacy, effects=effects, doc=doc)


# Declarations
vf("reg", (), (A("name", "str"),), (Res("reg", REG),), "create_reg", "declare a vector register", effects=())
vf("mask", (), (A("name", "str"), A("init", "ident", doc="initial pattern: all | none | vl1..vl128 | h | q | m3 | m4 (default all)")),
   (Res("mask", MASK),), "create_maskreg", "declare a predicate register", effects=())
vf("unalign", (), (A("name", "str"),), (Res("ureg", "unalignreg<*>"),), "create_unalignreg",
   "declare an unaligned-access register; the result type says its role (load | store)", effects=())
vf("reinterpret", (N("src", REG),), (), (Res("reg", REG),), "micro_reg_reinterpret", "view a register as another dtype", effects=())

# UB <-> register movement
vf("load", (W("dst", REG), R("src", UB)), (A("offset", IV, default=0), A("blk_stride", IV), MASK_ATTR), legacy="micro_ub2reg",
   doc="UB -> register (convert on load when dtypes differ)")
vf("store", (W("dst", UB), R("src", REG)), (A("offset", IV, default=0), A("blk_stride", IV), MASK_ATTR), legacy="micro_reg2ub",
   doc="register -> UB (convert on store when dtypes differ)")
vf("load_cont", (W("dst", REG), R("src", UB)), (A("offset", IV, default=0), A("mode", "ident")), legacy="micro_ub2regcont", doc="contiguous load")
vf("store_cont", (W("dst", UB), R("src", REG)), (A("offset", IV, default=0), A("mode", "ident"), MASK_ATTR), legacy="micro_reg2ubcont",
   doc="contiguous store")
vf("load_interleave", (W("dst0", REG), W("dst1", REG), R("src", UB)), (A("offset", IV, default=0), A("mode", "ident")),
   legacy="micro_ub2reginterleave", doc="deinterleaving load into two registers")
vf("store_interleave", (W("dst", UB), R("src0", REG), R("src1", REG)), (A("offset", IV, default=0), A("mode", "ident"), MASK_ATTR),
   legacy="micro_reg2ubinterleave", doc="interleaving store from two registers")
vf("load_unalign_pre", (N("ureg", "unalignreg<load>"), R("src", UB)), (A("offset", IV, default=0),), legacy="micro_ub2reg_unalign_pre",
   doc="prime the unaligned register")
vf("load_unalign", (W("dst", REG), R("src", UB), N("ureg", "unalignreg<load>")), (A("offset", IV, default=0), A("stride", IV), A("post_mode", "ident")),
   legacy="micro_ub2reg_unalign", doc="unaligned load")
vf("store_unalign", (W("dst", UB), R("src", REG), N("ureg", "unalignreg<store>")), (A("offset", IV, default=0), A("count", IV), A("post_mode", "ident")),
   legacy="micro_reg2ub_unalign", doc="unaligned store")
vf("store_unalign_post", (W("dst", UB), N("ureg", "unalignreg<store>")), (A("offset", IV, default=0), A("stride", IV), A("post_mode", "ident")),
   legacy="micro_reg2ub_unalign_post", doc="flush the unaligned register")
vf("ub_cursor", (N("src", UB),), (), (Res("out", "value"),), "micro_ub_cursor", "address cursor into UB", effects=())
vf("mask_to_ub", (W("dst", UB), R("src", MASK)), (A("offset", IV, default=0),), legacy="micro_mask_to_ub", doc="spill a mask to UB")
vf("ub_to_mask", (W("dst", MASK), R("src", UB)), (A("offset", IV, default=0),), legacy="micro_ub_to_mask", doc="load a mask from UB")
vf("clear_spr", (), (), legacy="micro_clear_spr", doc="clear the special-purpose registers")
vf("barrier", (), (A("src", "ident", required=True), A("dst", "ident", required=True)), legacy="micro_vf_barrier",
   doc="local memory barrier between two vf pipes (vec_store | vec_load | scalar_store | scalar_load | vec_all | scalar_all)", effects=("sync",))

# Register arithmetic
for _n, _l in (("add", "micro_vadd"), ("sub", "micro_vsub"), ("mul", "micro_vmul"), ("max", "micro_vmax"), ("min", "micro_vmin"),
               ("and", "micro_vand"), ("or", "micro_vor"), ("xor", "micro_vxor"), ("abssub", "micro_abssub"), ("muladddst", "micro_muladddst"),
               ("muldstadd", "micro_muldstadd"), ("prelu", "micro_vprelu"), ("shiftl", "micro_shiftl"), ("shiftr", "micro_shiftr")):
    vf(_n, (W("dst", REG), R("src1", REG), R("src2", REG)), (MASK_ATTR,), legacy=_l, doc=f"dst = {_n}(src1, src2) per lane")
vf("div", (W("dst", REG), R("src1", REG), R("src2", REG)), (MASK_ATTR, A("config", "any")), legacy="micro_vdiv", doc="lane division")
vf("mod", (W("dst", "reg<T, N>"), R("src1", "reg<T, N>"), R("src2", "reg<T, N>")), (MASK_ATTR,),
   doc="integer floor remainder; a zero divisor produces all-one bits")
vf("axpy", (W("dst", REG), R("src", REG), N("v", "value")), (MASK_ATTR,), legacy="micro_vaxpy", doc="dst += v * src")
for _n, _l in (("abs", "micro_vabs"), ("exp", "micro_vexp"), ("ln", "micro_vln"), ("log", "micro_vlog"), ("log2", "micro_vlog2"),
               ("log10", "micro_vlog10"), ("sqrt", "micro_vsqrt"), ("neg", "micro_vneg"), ("not", "micro_vnot"), ("relu", "micro_vrelu"),
               ("copy", "micro_vcopy")):
    vf(_n, (W("dst", REG), R("src", REG)), (MASK_ATTR,), legacy=_l, doc=f"dst = {_n}(src) per lane")
for _n, _l in (("adds", "micro_vadds"), ("muls", "micro_vmuls"), ("maxs", "micro_vmaxs"), ("mins", "micro_vmins"),
               ("shiftls", "micro_shiftls"), ("shiftrs", "micro_shiftrs"), ("lrelu", "micro_vlrelu")):
    vf(_n, (W("dst", REG), R("src", REG), N("v", "value")), (MASK_ATTR,), legacy=_l, doc=f"dst = {_n}(src, scalar v) per lane")
vf("dup", (W("dst", REG), N("src", "value")), (MASK_ATTR,), legacy="micro_vdup", doc="broadcast a scalar into a register")
vf("cast", (W("dst", REG), R("src", REG)), (MASK_ATTR, A("round", "ident"), A("layout", "ident"), A("saturate", "bool"), A("merge", "ident"), A("config", "any"), A("ddst", "any"), A("dsrc", "any")),
   legacy="micro_cast", doc="lane conversion with rounding mode and register layout")
vf("mulscast", (W("dst", REG), R("src", REG), N("v", "value")), (MASK_ATTR, A("layout", "ident"), A("scalar_dtype", "ident"), A("ddst", "any"), A("dsrc", "any")),
   legacy="micro_mulscast", doc="fused scalar multiply + cast")
vf("expsub", (W("dst", REG), R("src0", REG), R("src1", REG)), (MASK_ATTR, A("layout", "ident"), A("ddst", "any"), A("dsrc", "any")),
   legacy="micro_expsub", doc="fused exp(src0 - src1)")
for _n, _l in (("cadd", "micro_vcadd"), ("cmax", "micro_vcmax"), ("cmin", "micro_vcmin"), ("cgadd", "micro_vcgadd"),
               ("cgmax", "micro_vcgmax"), ("cgmin", "micro_vcgmin"), ("cpadd", "micro_vcpadd")):
    _index = (A("index", "bool", default=False, doc="keep the first extremum lane in lane 1 (RFC-0001)"),) if _n in ("cmax", "cmin") else ()
    vf(_n, (W("dst", REG), R("src", REG)), (MASK_ATTR, *_index), legacy=_l, doc=f"{_n}: lane reduction")
vf("cmp", (W("dst", MASK), R("src1", REG), R("src2", REG)), (MASK_ATTR, A("mode", "ident", required=True), A("dtype", "ident")),
   legacy="micro_compare", doc="lane compare into a mask")
vf("cmps", (W("dst", MASK), R("src1", REG), N("src2", "value")), (MASK_ATTR, A("mode", "ident", required=True), A("dtype", "ident")),
   legacy="micro_compares", doc="lane compare against a scalar")
vf("select", (W("dst", REG), R("src1", REG), R("src2", REG)), (MASK_ATTR,), legacy="micro_select", doc="dst = mask ? src1 : src2")
vf("arange", (W("dst", REG),), (A("v", "any"), A("dtype", "ident"), A("mode", "ident")), legacy="micro_arange", doc="lane index ramp")
vf("pack", (W("dst", REG), R("src", REG)), (A("part", "any"), A("ddst", "any"), A("dsrc", "any")), legacy="micro_pack", doc="narrowing pack")
vf("interleave", (W("dst0", REG), W("dst1", REG), R("src0", REG), R("src1", REG)), (), legacy="micro_interleave", doc="interleave two registers")
vf("deinterleave", (W("dst0", REG), W("dst1", REG), R("src0", REG), R("src1", REG)), (), legacy="micro_dinterleave", doc="deinterleave two registers")
vf("gather", (W("dst", REG), R("src", REG), R("index", REG)), (), legacy="micro_gather", doc="gather from UB by lane index")
vf("gatherb", (W("dst", REG), R("src", UB), R("index", REG)), (A("offset", IV, default=0), MASK_ATTR), legacy="micro_gatherb", doc="block gather from UB")
vf("gathermask", (W("dst", REG), R("src", REG)), (MASK_ATTR,), legacy="micro_gathermask", doc="compress lanes selected by mask")
vf("gather_copy", (W("dst", REG), R("src", UB), R("index", REG)), (A("offset", IV, default=0), MASK_ATTR), legacy="micro_datacopygather",
   doc="UB -> register gather (DataCopyGather)")
vf("scatter_copy", (W("dst", UB), R("src", REG), R("index", REG)), (A("offset", IV, default=0), MASK_ATTR), legacy="micro_datacopyscatter",
   doc="register -> UB scatter (DataCopyScatter)")
vf("squeeze", (W("dst", REG), R("src", REG)), (MASK_ATTR, A("store", "any")), legacy="micro_squeeze", doc="squeeze lanes by mask")
vf("unsqueeze", (W("dst", REG),), (MASK_ATTR,), legacy="micro_unsqueeze", doc="expand lanes by mask")
vf("histograms", (W("dst", REG), R("src", REG)), (MASK_ATTR, A("bin_group", "any"), A("mode", "ident"), A("ddst", "any"), A("dsrc", "any")),
   legacy="micro_histograms", doc="lane histogram")

# Mask register ops
for _n, _l in (("mask_and", "micro_maskand"), ("mask_or", "micro_maskor"), ("mask_xor", "micro_maskxor"), ("mask_sel", "micro_masksel")):
    vf(_n, (W("dst", MASK), R("src1", MASK), R("src2", MASK)), (MASK_ATTR,), legacy=_l, doc=f"{_n} on predicate registers")
vf("mask_not", (W("dst", MASK), R("src", MASK)), (MASK_ATTR,), legacy="micro_masknot", doc="predicate not")
vf("mask_mov", (W("dst", MASK), R("src", MASK)), (MASK_ATTR,), legacy="micro_maskmov", doc="predicate move")
vf("mask_pack", (W("dst", MASK), R("src", MASK)), (A("mode", "ident"),), legacy="micro_maskpack", doc="predicate pack")
vf("mask_unpack", (W("dst", MASK), R("src", MASK)), (A("mode", "ident"),), legacy="micro_maskunpack", doc="predicate unpack")
vf("mask_interleave", (W("dst0", MASK), W("dst1", MASK), R("src0", MASK), R("src1", MASK)), (), legacy="micro_maskinterl", doc="predicate interleave")
vf("mask_deinterleave", (W("dst0", MASK), W("dst1", MASK), R("src0", MASK), R("src1", MASK)), (), legacy="micro_maskdeinterl", doc="predicate deinterleave")
vf("mask_update", (W("dst", MASK),), (A("cnt", IV, required=True),), legacy="micro_updatemask", doc="mask of the first cnt lanes")
vf("mask_from_spr", (W("dst", MASK),), (), legacy="micro_movemaskspr", doc="mask from the special-purpose register")

retire("create_reglist", "a list of registers is a static Python container; nothing to emit")
