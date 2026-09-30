# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``simt.*``: thread-parallel functions on the vector core.

The old repository recorded SIMT bodies as C statements (``simt_assign``, ``simt_for_range``, ...);
here a ``simt`` function is ordinary structured IR: ``cf.for`` / ``cf.if`` with scalar ops, and
element loads/stores. ``simt.launch`` runs it from the kernel.
"""

from ._dsl import A5, KERNEL, SIMT, VAR, A, N, R, Res, W, op

op("simt.launch", kinds=KERNEL, side="vec", pipe="V", devices=A5, operands=(N("callee", "value"), VAR("args")),
   attrs=(A("threads", "int", required=True), A("read", "list", doc="UB buffers the body reads (autosync)"),
          A("write", "list", doc="UB buffers the body writes (autosync)")), legacy="call_simt", effects=("memory",),
   doc="launch a simt function; operands are the callee then its arguments")
op("simt.thread_id", kinds=SIMT, side="vec", pipe="V", devices=A5, results=(Res("out", "i32"),), doc="index of this thread")
op("simt.thread_num", kinds=SIMT, side="vec", pipe="V", devices=A5, results=(Res("out", "i32"),), doc="number of threads in the launch")
op("simt.load", kinds=SIMT, side="vec", pipe="V", devices=A5, operands=(R("src", "mem<*, *>"), N("index", "value")), results=(Res("out", "scalar"),),
   effects=("memory",), doc="load one element by flat index")
op("simt.store", kinds=SIMT, side="vec", pipe="V", devices=A5, operands=(W("dst", "mem<*, *>"), N("index", "value"), N("src", "value")),
   effects=("memory",), doc="store one element by flat index")
op("simt.barrier", kinds=SIMT, side="vec", pipe="V", devices=A5, effects=("sync",), legacy="simt_thread_barrier", doc="thread barrier")
op("simt.atomic", kinds=SIMT, side="vec", pipe="V", devices=A5,
   operands=(W("dst", "mem<*, *>"), N("index", "value"), N("src", "value"), VAR("compare", "value")),
   attrs=(A("op", "ident", required=True),), results=(Res("old", "scalar"),), effects=("memory",), legacy="simt_atomic",
   doc="atomic read-modify-write on one element (add / sub / max / min / exch / and / or / xor; cas takes `compare` "
       "as a 4th operand; inc / dec take the ring `limit` as the value - CUDA wrap semantics)")

op("simt.block_idx", kinds=SIMT, side="vec", pipe="V", devices=A5, results=(Res("out", "i32"),), legacy="simt_block_idx", doc="index of the core running the launch")
op("simt.block_num", kinds=SIMT, side="vec", pipe="V", devices=A5, results=(Res("out", "i32"),), legacy="simt_block_num", doc="number of cores")

# ---- scalar math on SIMT threads (the dav-c310 SIMT layer; f32). Same surface the PyPTO Pro
# CCE codegen and the CANN dav_c310 SIMT impl sit on: __expf/__logf/... builtins + syntheses.
for _name, _doc in (
    ("exp", "e**x"), ("exp2", "2**x"), ("log", "natural log"), ("log2", "log base 2"),
    ("log1p", "log(1 + x)"), ("sin", "sine"), ("cos", "cosine"), ("tanh", "hyperbolic tangent"),
    ("rsqrt", "1 / sqrt(x)"), ("rint", "round to nearest even"), ("round", "round half away from zero"),
    ("floor", "round toward -inf"), ("ceil", "round toward +inf"), ("trunc", "round toward zero"),
):
    op(f"simt.{_name}", kinds=SIMT, side="vec", pipe="V", devices=A5,
       operands=(N("x", "value"),), results=(Res("out", "f32"),), doc=f"SIMT scalar {_doc} (f32)")
for _name, _doc in (("isnan", "x is NaN"), ("isinf", "x is +/-inf"), ("isfinite", "x is finite")):
    op(f"simt.{_name}", kinds=SIMT, side="vec", pipe="V", devices=A5,
       operands=(N("x", "value"),), results=(Res("out", "i32"),), doc=f"SIMT: 1 when {_doc}, else 0")
op("simt.popc", kinds=SIMT, side="vec", pipe="V", devices=A5, operands=(N("x", "value"),),
   results=(Res("out", "i32"),), doc="number of set bits (32-bit)")
op("simt.ffs", kinds=SIMT, side="vec", pipe="V", devices=A5, operands=(N("x", "value"),),
   results=(Res("out", "i32"),), doc="1-based index of the lowest set bit; 0 when empty")
op("simt.mul_hi", kinds=SIMT, side="vec", pipe="V", devices=A5,
   operands=(N("a", "value"), N("b", "value")), results=(Res("out", "scalar"),),
   doc="high 32 bits of the 64-bit product (signed / unsigned by operand dtype)")
op("simt.fmod", kinds=SIMT, side="vec", pipe="V", devices=A5,
   operands=(N("a", "value"), N("b", "value")), results=(Res("out", "f32"),),
   doc="floating remainder: a - trunc(a / b) * b")
op("simt.fma", kinds=SIMT, side="vec", pipe="V", devices=A5,
   operands=(N("a", "value"), N("b", "value"), N("c", "value")), results=(Res("out", "f32"),),
   legacy="simt_fma", doc="a * b + c with one rounding")
op("simt.threadfence", kinds=SIMT, side="vec", pipe="V", devices=A5, effects=("sync",),
   doc="device-scope memory fence between SIMT threads")
op("simt.threadfence_block", kinds=SIMT, side="vec", pipe="V", devices=A5, effects=("sync",),
   doc="core-scope memory fence between SIMT threads")
