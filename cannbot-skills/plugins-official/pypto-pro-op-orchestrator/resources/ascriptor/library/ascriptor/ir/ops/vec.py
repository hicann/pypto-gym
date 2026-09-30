# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``vec.*``: tensor-vector instructions on UB - the a2 family's vector ISA. On a5 only the sort
family and the mask state ops exist (the a5 vector core is programmed through ``vf.*``); the
old stub functions gate each of these with ``A2_DEVICES``, which the registry mirrors. Plus
``atomic.*``. Every op is one V-pipe instruction with repeat/stride attributes."""

from ._dsl import A2, A5, A, N, R, W, op

IV = "int|value"
# count / count_per_rep are attributes of the instruction (D-062): the IR carries no vector-mode state, the c220
# backend materialises the SPR switches (set_mask_count / set_vector_mask / set_mask_norm) around each op that
# needs them — CANN's own dav_c220 Level-2 implementations bracket every counted call exactly so, with no barrier.
STRIDES = tuple(A(n, IV) for n in ("repeat", "dst_blk_stride", "dst_rep_stride", "src_blk_stride", "src_rep_stride",
                                    "src1_blk_stride", "src1_rep_stride", "src2_blk_stride", "src2_rep_stride",
                                    "count", "count_per_rep"))
UB = "ub<*, *>"


def _binary(name: str, legacy, doc: str = "") -> None:
    op(f"vec.{name}", level="both", side="vec", pipe="V", operands=(W("dst", UB), R("src1", UB), R("src2", UB)),
       attrs=STRIDES + (A("mode", "any"),), devices=A2, legacy=legacy, effects=("memory",), doc=doc or f"dst = src1 {name} src2, elementwise")


def _unary(name: str, legacy, doc: str = "") -> None:
    op(f"vec.{name}", level="both", side="vec", pipe="V", operands=(W("dst", UB), R("src", UB)), attrs=STRIDES + (A("mode", "any"),),
       devices=A2, legacy=legacy, effects=("memory",), doc=doc or f"dst = {name}(src), elementwise")


def _unary_scalar(name: str, legacy, doc: str = "", extra: tuple = ()) -> None:
    op(f"vec.{name}", level="both", side="vec", pipe="V", operands=(W("dst", UB), R("src", UB), N("v", "value")),
       attrs=STRIDES + (A("mode", "any"),) + extra, devices=A2, legacy=legacy, effects=("memory",), doc=doc or f"dst = {name}(src, scalar v)")


def _reduce(name: str, legacy, doc: str = "") -> None:
    op(f"vec.{name}", level="both", side="vec", pipe="V", operands=(W("dst", UB), R("src", UB)), attrs=STRIDES + (A("mode", "any"),),
       devices=A2, legacy=legacy, effects=("memory",), doc=doc or f"{name}: reduction over repeats")


for _n, _l in (("add", "add"), ("sub", "sub"), ("mul", "mul"), ("div", "div"), ("max", ("max", "vmax")), ("min", ("min", "vmin")),
               ("and", "vand"), ("or", "vor"), ("muladddst", "muladddst")):
    _binary(_n, _l)
for _n, _l in (("abs", ("abs", "vabs")), ("relu", ("relu", "vec_v_relu")), ("exp", "exp"), ("ln", "ln"), ("rec", "rec"),
               ("sqrt", "sqrt"), ("rsqrt", "rsqrt"), ("not", "vnot")):
    _unary(_n, _l)
for _n, _l in (("adds", "adds"), ("muls", "muls"), ("maxs", "vmaxs"), ("mins", "vmins"), ("lrelu", "lrelu"), ("axpy", "axpy"),
               ("shiftls", "shiftls")):
    _unary_scalar(_n, _l)
_unary_scalar("shiftrs", "shiftrs", "dst = src >> v (arithmetic for signed, logical for unsigned; round_en rounds a signed shift)",
              extra=(A("round_en", "bool"),))
for _n, _l in (("cadd", "cadd"), ("cmax", "cmax"), ("cmin", "cmin"), ("cgadd", "cgadd"), ("cgmax", "cgmax"), ("cgmin", "cgmin"), ("cpadd", "cpadd")):
    _reduce(_n, _l)

op("vec.dup", side="vec", pipe="V", devices=A2, operands=(W("dst", UB), N("src", "value")), attrs=STRIDES, legacy="dup", effects=("memory",),
   doc="broadcast a scalar into dst")
op("vec.brcb", side="vec", pipe="V", devices=A2, operands=(W("dst", UB), R("src", UB)), attrs=STRIDES, legacy="brcb", effects=("memory",),
   doc="broadcast each element of src across a block")
op("vec.cast", side="vec", pipe="V", devices=A2, operands=(W("dst", UB), R("src", UB)), attrs=STRIDES + (A("mode", "ident", required=True),),
   legacy="cast", effects=("memory",), doc="elementwise conversion with a rounding mode")
op("vec.compare", side="vec", pipe="V", devices=A2, operands=(W("dst", UB), R("src1", UB), R("src2", UB)), attrs=STRIDES + (A("mode", "ident", required=True),),
   legacy="compare", effects=("memory",), doc="elementwise compare into a bit mask tensor")
op("vec.compare_scalar", side="vec", pipe="V", devices=A2, operands=(W("dst", UB), R("src1", UB), N("src2", "value")),
   attrs=STRIDES + (A("mode", "ident", required=True),), legacy="compare_scalar", effects=("memory",), doc="compare against a scalar")
op("vec.select", side="vec", pipe="V", devices=A2, operands=(W("dst", UB), R("selmask", UB), R("src1", UB), R("src2", UB)),
   attrs=STRIDES + (A("mode", "any"), A("tmp_addr_buf", "value", pattern=UB, access="write")), legacy="select", effects=("memory",),
   doc="dst = selmask ? src1 : src2")
op("vec.transdata5hd", side="vec", pipe="V", devices=A2, operands=(W("dst", UB), R("src", UB)),
   attrs=(A("repeat", IV), A("dst_rep_stride", IV), A("dst_row_stride", IV), A("src_rep_stride", IV), A("src_row_stride", IV)),
   legacy="transdata5hd", effects=("memory",), doc="NCHW <-> 5HD transpose")
op("vec.sort32", side="vec", pipe="V", operands=(W("dst", UB), R("src", UB), R("idx", UB)), attrs=(A("repeat", IV),), legacy="sort32",
   effects=("memory",), doc="sort 32 elements with indices")
op("vec.mergesort4", side="vec", pipe="V", operands=(W("dst", UB), R("src", UB)), attrs=(A("repeat", IV), A("length_per_seq", IV)),
   legacy="mergesort4", effects=("memory",), doc="4-way merge of sorted sequences")
op("vec.mergesort_2seq", side="vec", pipe="V", operands=(W("dst", UB), R("src1", UB), R("src2", UB)), attrs=(A("size1", IV), A("size2", IV)),
   legacy="mergesort_2seq", effects=("memory",), doc="merge two sorted sequences")
op("vec.topk_radix", side="vec", pipe="V", devices=A5, operands=(W("dst", UB), R("src", UB)),
   attrs=(A("k", IV), A("n", IV), A("aligned_n", IV), A("largest", "bool"), A("sorted", "bool"), A("outter", "any"), A("init_index", "any"),
          A("src_values", "value", pattern=UB, access="read"), A("src_indices", "value", pattern=UB, access="read"),
          A("dst_values", "value", pattern=UB, access="write"), A("dst_indices", "value", pattern=UB, access="write"),
          A("tmp", "value", pattern=UB, access="write")),
   legacy="topk_radix", effects=("memory",), doc="radix top-k")
op("vec.gather", side="vec", pipe="V", devices=A2, operands=(W("dst", UB), R("src", UB), R("offset", UB)),
   attrs=(A("repeat", IV), A("dst_rep_stride", IV), A("count", IV), A("start_idx", IV)), legacy="gather", effects=("memory",), doc="element gather")
op("vec.gather_block", side="vec", pipe="V", devices=A2, operands=(W("dst", UB), R("src", UB), R("offset", UB)),
   attrs=(A("repeat", IV), A("dst_blk_stride", IV), A("dst_rep_stride", IV)), legacy="gather_block", effects=("memory",), doc="block gather")
op("vec.scatter", side="vec", pipe="V", devices=A2, operands=(W("dst", UB), R("src", UB), R("offset", UB)),
   attrs=(A("repeat", IV), A("src_rep_stride", IV), A("count", IV), A("start_idx", IV)), legacy="scatter", effects=("memory",), doc="element scatter")

# Vector mask state (a2 family).
op("vec.set_mask", side="vec", pipe="V", attrs=(A("high", IV, required=True), A("low", IV, required=True)), legacy="set_mask",
   effects=("memory",), doc="set the 128-bit vector mask")
op("vec.set_mask_count", side="vec", pipe="V", legacy="set_mask_count", effects=("memory",), doc="switch to counter mask mode")
op("vec.set_mask_normal", side="vec", pipe="V", legacy="set_mask_normal", effects=("memory",), doc="switch to normal mask mode")
op("vec.set_mask_counter", side="vec", pipe="V", attrs=(A("count", IV, required=True),), legacy="set_mask_counter", effects=("memory",),
   doc="counter mode with a count")
op("vec.set_mask_by_count", side="vec", pipe="V", attrs=(A("count", IV, required=True),), legacy="set_mask_by_count", effects=("memory",),
   doc="set the mask for the first count lanes")
op("vec.reset_mask", side="vec", pipe="V", legacy="reset_mask", effects=("memory",), doc="restore the full mask")

# Atomic GM writes.
op("atomic.begin", side="any", pipe="S", attrs=(A("op", "ident", required=True, doc="add | max | min"),),
   legacy=("atomic_add", "atomic_max", "atomic_min"), effects=("memory",), doc="following GM stores accumulate atomically")
op("atomic.end", side="any", pipe="S", legacy="atomic_end", effects=("memory",), doc="stop atomic accumulation")
op("atomic.set_type", side="vec", pipe="V", attrs=(A("dtype", "ident", required=True),), legacy="set_atomic_type", effects=("memory",),
   doc="dtype for atomic accumulation")
