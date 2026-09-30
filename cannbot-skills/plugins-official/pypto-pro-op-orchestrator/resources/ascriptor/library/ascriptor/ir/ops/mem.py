# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``mem.*`` and ``list.*``: allocation, buffers, views, workspace, tensor lists."""

from ._dsl import ALL, KV, A, N, Res, op, retire

op("mem.alloc", kinds=KV, pipe="S", attrs=(A("addr", "int|value", doc="assigned by addr_alloc (Lowered)"),
   A("mutex_ids", "list", doc="A5 implicit local mutex IDs, one per physical slot on this side"),
   A("name", "str"), A("live", "list", doc="[first op id, last op id] of the live range (liveness)"),
   A("sync_depth", "int", doc="maximum automatic event tokens for this slot buffer; autosync adds back-pressure when needed")), results=(Res("t", "value"),), legacy=("create_tensor", "create_dbuf", "create_tbuf", "create_qbuf"),
   effects=("memory",), doc="allocate an on-chip tensor or a slot buffer; the result type carries space, shape, layout, slots")
op("mem.workspace", pipe="S", attrs=(A("name", "str", required=True), A("numel", "int|value", required=True),
   A("offset", "int|value"), A("gmbuff_slots", "int", doc="GMBuff ring: slot count (RFC-0009); the gmbuff pass consumes it"),
   A("gmbuff_per_core", "int", doc="GMBuff ring: 1 = one ring per cube core"),
   A("gmbuff_dims", "list", doc="GMBuff ring: the full workspace dims [cube_num?, slots, rows, cols]")),
   results=(Res("ws", "value"),), legacy="split_workspace",
   doc="carve a region of the launcher-provided GM workspace; a GMBuff ring types the result as a slot buffer")
op("mem.get_buf", kinds=KV, pipe="S", operands=(N("buf", "buf<*>"), N("index", "value")), results=(Res("slot", "mem<*, *>"),),
   legacy="get_buf", doc="select one slot of a slot buffer (index modulo slots); on-chip, or a GMBuff workspace ring")
op("mem.slice", kinds=ALL, pipe="S", operands=(N("src", "mem<*, *>"),),
   attrs=(A("offsets", "list", required=True), A("extents", "list", required=True), A("steps", "list"), A("mask", "list")),
   results=(Res("view", "mem<*, *>"),), legacy=("slice_gm_tensor", "slice_tensor", "micro_slice_tensor"),
   doc="a rectangular view; offsets/extents are ints or scalar values, one per dim of src")
op("mem.reshape", kinds=ALL, pipe="S", operands=(N("src", "mem<*, *>"),), attrs=(A("shape", "list", required=True),),
   results=(Res("view", "mem<*, *>"),), legacy="reshape_gm_tensor", doc="reinterpret the shape of a contiguous view")
op("mem.view", kinds=ALL, pipe="S", operands=(N("src", "gm<*, *>"),),
   attrs=(A("shape", "list", required=True, doc="ints or scalar values, one per dim"),
          A("strides", "list", required=True, doc="element strides, one per dim; the innermost must be 1"),
          A("offset", "int|value", required=True, doc="element offset from the start of src")),
   results=(Res("view", "gm<*, *>"),),
   doc="re-describe a whole GM tensor / workspace with a new shape, explicit strides and an offset "
       "(RFC-0010); zero instructions - only DMA may consume it (strides fold into the burst descriptor)")
op("mem.reinterpret", kinds=ALL, pipe="S", operands=(N("src", "mem<*, *>"),),
   attrs=(A("shape", "list"), A("span", "any"), A("packed_axis", "int"), A("layout", "ident"),
          A("tile", "list", doc="desugar: the [rows, cols] tile an L0 slot holds in the new dtype; the view is that tile at the slot start")),
   results=(Res("view", "mem<*, *>"),), legacy="reinterpret", doc="view the same bytes as another dtype (packed carriers, RFC-0001 §4.1)")

op("list.count", kinds=ALL, pipe="S", operands=(N("list", "gmlist<*, *>"),), results=(Res("out", "i32"),),
   legacy="tensorlist_size", doc="number of members of a tensor list")
op("list.item", kinds=ALL, pipe="S", operands=(N("list", "gmlist<*, *>"), N("index", "value")), results=(Res("t", "gm<*, *>"),),
   legacy="get_gm_tensor_list_item", doc="one member as a GM tensor; '?' dims become runtime scalars")
op("list.item_dim", kinds=ALL, pipe="S", operands=(N("list", "gmlist<*, *>"), N("index", "value")),
   attrs=(A("dim", "int", required=True),), results=(Res("out", "i32"),), legacy="tensorlist_item_numel",
   doc="one dimension of one member, read from the list descriptor")
op("list.load_ptr", level="lowered", kinds=ALL, pipe="S", operands=(N("list", "gmlist<*, *>"), N("index", "value")),
   attrs=(A("offset", "int", required=True), A("stride", "int", required=True)), results=(Res("out", "u64"),),
   doc="descriptor read of a member pointer (RFC-0001 §13)")
op("list.load_dim", level="lowered", kinds=ALL, pipe="S", operands=(N("list", "gmlist<*, *>"), N("index", "value")),
   attrs=(A("dim", "int", required=True), A("offset", "int", required=True), A("stride", "int", required=True)),
   results=(Res("out", "i64"),), doc="descriptor read of a member dimension (RFC-0001 §13)")

retire("create_gm_tensor", "GM tensors are kernel parameters; the signature declares them")
retire("create_gm_tensor_list", "GM tensor lists are kernel parameters; the signature declares them")
