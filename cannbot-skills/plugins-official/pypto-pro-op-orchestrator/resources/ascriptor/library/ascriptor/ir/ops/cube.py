# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``cube.*``: the matrix unit."""

from ._dsl import A5, RW, A, R, op

IV = "int|value"

op("cube.matmul", level="surface", side="cube", operands=(RW("dst", "l0c<*, *>"), R("a", "l1<T, *>"), R("b", "l1<T, *>")),
   attrs=(A("m", IV), A("n", IV), A("k", IV), A("init", "bool", default=True), A("splitn", "int"), A("splitk", "int"),
          A("bias", "value", pattern="local<*, *>", access="read"), A("a_transpose", "bool", default=False), A("b_transpose", "bool", default=False)),
   effects=("memory",), doc="shortcut: dst (+)= a @ b^T over L1 tiles; desugar expands it into l1_to_l0 + mmad loops")

op("cube.mmad", level="both", side="cube", pipe="M", operands=(RW("dst", "l0c<*, *>"), R("src_a", "l0a<T, *>"), R("src_b", "l0b<T, *>")),
   attrs=(A("M", IV, required=True), A("N", IV, required=True), A("K", IV, required=True), A("is_init", "bool", default=False),
          A("bias", "value", pattern="bt<*, *>", access="read"), A("dst_row0", IV, default=0), A("dst_col0", IV, default=0),
          A("dst_rows", IV), A("dst_cols", IV), A("src_a_carrier_cols", IV), A("src_a_logical_carrier_cols", IV),
          A("src_a_logical_rows", IV), A("src_b_carrier_cols", IV)),
   legacy="mmad", effects=("memory",), doc="one mmad instruction: dst (+)= src_a @ src_b")
op("cube.mmad.mx", level="both", side="cube", pipe="M", devices=A5, operands=(RW("dst", "l0c<*, *>"), R("src_a", "l0a<*, *>"), R("src_b", "l0b<*, *>")),
   attrs=(A("M", IV, required=True), A("N", IV, required=True), A("K", IV, required=True), A("is_init", "bool", default=False),
          A("bias", "value", pattern="bt<*, *>", access="read"), A("dst_row0", IV, default=0), A("dst_col0", IV, default=0),
          A("dst_rows", IV), A("dst_cols", IV)),
   legacy="mmad_mx", effects=("memory",), doc="microscaling mmad")

op("cube.matmul_mx", level="surface", side="cube", devices=A5,
   operands=(RW("dst", "l0c<*, *>"), R("a", "l1<*, *>"), R("b", "l1<*, *>"), R("scale_a", "l1<*, *>"), R("scale_b", "l1<*, *>")),
   attrs=(A("m", IV), A("n", IV), A("k", IV), A("init", "bool", default=True), A("splitn", "int"), A("splitk", "int"),
          A("bias", "value", pattern="local<*, *>", access="read"), A("a_transpose", "bool", default=False), A("b_transpose", "bool", default=False)),
   effects=("memory",), doc="shortcut: dst (+)= (a * 2^scale_a) @ (b * 2^scale_b)^T over MX FP8/FP4 L1 tiles; desugar expands it into l1_to_l0.mx + mmad.mx")
op("cube.conv2d", level="surface", side="cube",  # a2 and a5: the old shortcut ran on both (l1_to_l0a_img2col is a c220 instruction too)
   operands=(RW("dst", "l0c<*, *>"), R("fmap", "l1<T, *>"), R("weight", "l1<T, *>")),
   attrs=(A("h", IV, required=True), A("w", IV, required=True), A("c", IV, required=True), A("cout", IV, required=True),
          A("kh", "int", required=True), A("kw", "int", required=True), A("stride_h", "int", default=1), A("stride_w", "int", default=1),
          A("dil_h", "int", default=1), A("dil_w", "int", default=1), A("pad_t", "int", default=0), A("pad_b", "int", default=0),
          A("pad_l", "int", default=0), A("pad_r", "int", default=0), A("m0", IV, default=0), A("tile_k", IV),
          A("bias", "value", pattern="local<*, *>", access="read")),
   effects=("memory",), doc="shortcut: dst[tile_m, cout] = img2col(fmap)[m0:m0+tile_m, :] @ weight^T (+ bias); desugar expands it into l1_to_l0.img2col + l1_to_l0 + mmad")
