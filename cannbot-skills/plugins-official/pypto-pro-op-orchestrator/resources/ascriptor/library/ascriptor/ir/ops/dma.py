# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``dma.*``: data movement between GM and the on-chip memories.

Surface IR has one op, ``dma.copy``: the frontend records what the DSL said (``<<=`` between
two typed views plus its flags) and ``device_lower`` selects the instruction from the two
memory types. Lowered ops are one MTE/FIX instruction each; their attribute names are the
old repository's instruction kwargs so the recorded corpus maps one-to-one.
"""

from ._dsl import A5, A, R, W, op

IV = "int|value"

op("dma.copy", level="surface", operands=(W("dst", "mem<*, *>"), R("src", "mem<*, *>")),
   attrs=(A("transpose", "bool", default=False), A("pad", "any"), A("relu", "bool"), A("scale", "any"), A("offset", "any"),
          A("atomic", "ident"), A("hif8_hybrid", "bool"), A("dual_mode", "ident"), A("sub_block_id", "int|value")),
   effects=("memory",), doc="generic copy between two memories (the DSL's <<=); device_lower selects the instruction from the types and riders")

op("dma.gm_to_l1", level="both", side="cube", pipe="MTE2", operands=(W("dst", "l1<T, *>"), R("src", "gm<T, *>")),
   attrs=(A("n_burst", IV, required=True), A("burst_len", IV, required=True), A("src_stride", IV, default=0), A("dst_stride", IV, default=0)),
   legacy="gm_to_l1", effects=("memory",), doc="plain burst copy GM->L1")
op("dma.gm_to_l1.nd2nz", level="both", side="cube", pipe="MTE2", operands=(W("dst", "l1<T, *, nz>"), R("src", "gm<T, *>")),
   attrs=(A("M", IV, required=True), A("N", IV, required=True), A("M_dst", IV, required=True), A("N_src", IV, required=True)),
   legacy="gm_to_l1_nd2nz", effects=("memory",), doc="GM ND -> L1 NZ")
op("dma.gm_to_l1.dn2nz", level="both", side="cube", pipe="MTE2", devices=A5, operands=(W("dst", "l1<T, *, nz>"), R("src", "gm<T, *>")),
   attrs=(A("M", IV, required=True), A("N", IV, required=True), A("M_dst", IV, required=True), A("N_src", IV, required=True)),
   legacy="gm_to_l1_dn2nz", effects=("memory",), doc="GM DN (transposed) -> L1 NZ")
op("dma.gm_to_l1.mx_scale_nd2nz", level="both", side="cube", pipe="MTE2", devices=A5,
   operands=(W("dst", "l1<T, *, nz>"), R("src", "gm<T, *>")),
   attrs=(A("rows", IV, required=True), A("k_groups", IV, required=True), A("src_k_groups", IV, required=True)),
   legacy="gm_to_l1_mx_scale_nd2nz", effects=("memory",), doc="microscaling scale tensor GM -> L1")
op("dma.gm_to_l1.pad", level="both", side="cube", pipe="MTE2", devices=A5, operands=(W("dst", "l1<T, *>"), R("src", "gm<T, *>")),
   attrs=(A("n_burst", IV, required=True), A("burst_len_byte", IV, required=True), A("src_stride_byte", IV, default=0), A("dst_stride", IV, default=0)),
   legacy="gm_to_l1_pad", effects=("memory",), doc="byte-granular GM->L1 with padding")
op("dma.set_constant_to_l1", level="both", side="cube", pipe="MTE2", operands=(W("tensor", "l1<T, *>"),),
   attrs=(A("val", "int|float", required=True), A("n_blocks", IV, required=True)), legacy="set_constant_to_l1",
   effects=("memory",), doc="fill L1 blocks with a constant")

# L1 -> L0 (MTE1, cube side)
op("dma.l1_to_l0", level="both", side="cube", pipe="MTE1", operands=(W("dst", "l0<T, *>"), R("src", "l1<T, *>")),
   attrs=(A("m_src", IV, required=True), A("n_src", IV, required=True), A("m_dst", IV, required=True), A("n_dst", IV, required=True),
          A("src_row0", IV, default=0), A("src_col0", IV, default=0), A("src_is_transpose", "bool", default=False), A("dst_position", "ident"),
          A("m_copy", IV)),
   legacy="l1_to_l0", effects=("memory",), doc="fractal load L1 -> L0A/L0B")
op("dma.l1_to_l0.mx", level="both", side="cube", pipe="MTE1", devices=A5, operands=(W("dst", "l0<T, *>"), R("src", "l1<T, *>")),
   attrs=(A("m_src", IV, required=True), A("n_src", IV, required=True), A("m_dst", IV, required=True), A("n_dst", IV, required=True),
          A("src_row0", IV, default=0), A("src_col0", IV, default=0), A("src_is_transpose", "bool", default=False), A("dst_position", "ident"),
          A("src_mx", "value", pattern="l1<*, *>", access="read", required=True), A("src_mx_row0", IV, default=0),
          A("src_mx_col0", IV, default=0), A("src_mx_offset_element", IV, default=0)),
   legacy="l1_to_l0_mx", effects=("memory",), doc="fractal load with microscaling scales")
op("dma.l1_to_l0.img2col", level="both", side="cube", pipe="MTE1", operands=(W("dst", "l0<T, *>"), R("src", "l1<T, *>")),
   attrs=tuple(A(n, IV) for n in ("h", "w", "c", "c0", "kh", "kw", "k0", "m0", "k_ext", "m_ext", "stride_h", "stride_w", "dil_h", "dil_w",
                                  "pad_t", "pad_b", "pad_l", "pad_r")) + (A("dst_position", "ident"),),
   legacy="l1_to_l0_img2col", effects=("memory",), doc="im2col load for convolution")
op("dma.l1_to_bt", level="both", side="cube", pipe="MTE1", operands=(W("dst", "bt<U, *>"), R("src", "l1<T, *>")),
   attrs=(A("n", IV, required=True),), legacy="l1_to_bt", effects=("memory",), doc="bias L1 -> BT")

# L0C -> out (FIX, cube side)
_FIX_COMMON = (A("M", IV, required=True), A("N", IV, required=True), A("M_src", IV), A("relu", "bool", default=False),
               A("scale", "any"), A("offset", "any"), A("hif8_hybrid", "bool", default=False), A("atomic", "ident"))
op("dma.l0c_to_gm.nz2nd", level="both", side="cube", pipe="FIX", operands=(W("dst", "gm<*, *>"), R("src", "l0c<*, *>")),
   attrs=_FIX_COMMON + (A("N_dst", IV),), legacy="l0c_to_gm_nz2nd", effects=("memory",), doc="L0C -> GM as ND")
op("dma.l0c_to_gm.nz2nz", level="both", side="cube", pipe="FIX", operands=(W("dst", "gm<*, *>"), R("src", "l0c<*, *>")),
   attrs=_FIX_COMMON + (A("M_pad", IV),), legacy="l0c_to_gm_nz2nz", effects=("memory",), doc="L0C -> GM as NZ")
op("dma.l0c_to_gm.nz2dn", level="both", side="cube", pipe="FIX", devices=A5, operands=(W("dst", "gm<*, *>"), R("src", "l0c<*, *>")),
   attrs=_FIX_COMMON + (A("M_dst", IV),), legacy="l0c_to_gm_nz2dn", effects=("memory",), doc="L0C -> GM transposed")
op("dma.l0c_to_l1", level="both", side="cube", pipe="FIX", operands=(W("dst", "l1<*, *>"), R("src", "l0c<*, *>")),
   attrs=(A("M", IV, required=True), A("N", IV, required=True), A("M_src", IV), A("M_dst", IV), A("relu", "bool", default=False)),
   legacy="l0c_to_l1", effects=("memory",), doc="L0C -> L1")
op("dma.l0c_to_ub", level="both", side="cube", pipe="FIX", devices=A5, operands=(W("dst", "ub<*, *>"), R("src", "l0c<*, *>")),
   attrs=_FIX_COMMON + (A("N_dst", IV), A("dual_mode", "ident", default="splitm"), A("sub_block_id", IV)),
   legacy="l0c_to_ub", effects=("memory",), doc="L0C -> UB (a5 fixpipe to the vector core's UB)")

# GM <-> UB (vec side)
op("dma.gm_to_ub.pad", level="both", side="vec", pipe="MTE2", operands=(W("dst", "ub<T, *>"), R("src", "gm<T, *>")),
   attrs=(A("n_burst", IV, required=True), A("burst_len_byte", IV, required=True), A("src_stride_byte", IV, default=0), A("dst_stride", IV, default=0),
          A("pad", "any")),
   legacy="gm_to_ub_pad", effects=("memory",), doc="GM -> UB, byte granular")
op("dma.gm_to_ub.nd", level="both", side="vec", pipe="MTE2", devices=A5, operands=(W("dst", "ub<T, *>"), R("src", "gm<T, *>")),
   attrs=(A("dim", "int", required=True), A("loop_size", "list", required=True), A("loop_src_stride", "list"), A("loop_dst_stride", "list"),
          A("loop_left_pad", "list"), A("loop_right_pad", "list"), A("config_left_pad", IV), A("config_right_pad", IV),
          A("constant_value", "any"), A("nearest_value_mode", "any"), A("fence", "any"), A("asc_optimize", "any")),
   legacy="gm_to_ub_nd_dma", effects=("memory",), doc="multi-dimensional GM -> UB DMA")
op("dma.ub_to_gm.pad", level="both", side="vec", pipe="MTE3", operands=(W("dst", "gm<T, *>"), R("src", "ub<T, *>")),
   attrs=(A("n_burst", IV, required=True), A("burst_len_byte", IV, required=True), A("src_stride", IV, default=0), A("dst_stride_byte", IV, default=0),
          A("atomic", "ident", doc="add | max | min: the store accumulates into GM (the old set_atomic_type before it)")),
   legacy="ub_to_gm_pad", effects=("memory",), doc="UB -> GM, byte granular")
op("dma.ub_to_l1", level="both", side="vec", pipe="MTE3", devices=A5, operands=(W("dst", "l1<T, *>"), R("src", "ub<T, *>")),
   attrs=(A("n_burst", IV, required=True), A("burst_len", IV, required=True), A("src_stride", IV, default=0), A("dst_stride", IV, default=0)),
   legacy="ub_to_l1", effects=("memory",), doc="UB -> L1 burst copy")
op("dma.ub_to_l1.nd2nz", level="both", side="vec", pipe="MTE3", devices=A5, operands=(W("dst", "l1<T, *, nz>"), R("src", "ub<T, *>")),
   attrs=(A("m_src", IV, required=True), A("n_src", IV, required=True), A("m_dst", IV, required=True), A("n_dst", IV, required=True),
          A("N_src", IV), A("dst_row0", IV, default=0), A("dst_col0", IV, default=0)),
   legacy="ub_to_l1_nd2nz", effects=("memory",), doc="UB ND -> L1 NZ")
op("dma.ub_to_l1.nz", level="both", side="vec", pipe="MTE3", devices=A5, operands=(W("dst", "l1<T, *, nz>"), R("src", "ub<T, *>")),
   attrs=(A("m_src", IV, required=True), A("n_src", IV, required=True), A("m_dst", IV, required=True), A("n_dst", IV, required=True),
          A("M_src", IV), A("src_row0", IV, default=0), A("src_col0", IV, default=0), A("dst_row0", IV, default=0), A("dst_col0", IV, default=0)),
   legacy="ub_to_l1_nz", effects=("memory",), doc="UB NZ -> L1 NZ")
op("dma.ub_to_ub", level="both", side="vec", pipe="V", operands=(W("dst", "ub<T, *>"), R("src", "ub<T, *>")),
   attrs=(A("n_burst", IV, required=True), A("burst_len", IV, required=True), A("src_stride", IV, default=0), A("dst_stride", IV, default=0)),
   legacy="ub_to_ub", effects=("memory",), doc="UB -> UB burst copy")
