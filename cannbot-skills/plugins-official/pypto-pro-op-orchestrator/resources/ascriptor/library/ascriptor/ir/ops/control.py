# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``cf.*``, ``region.*``, ``sync.*``, ``debug.*``: control flow, regions, synchronisation, debugging."""

from ._dsl import A5, ALL, KERNEL, KV, VAR, A, N, R, Res, op

IV = "int|value"

# Structured control flow (RFC-0001 §7)
op("cf.for", kinds=ALL, pipe="S", operands=(N("lo", "value"), N("hi", "value"), N("step", "value")),
   attrs=(A("name", "str"), A("unroll", "int")), results=(Res("i", "int"),), regions=("body",), effects=("control",),
   legacy=("start_loop", "end_loop", "start_micro_loop"), doc="counted loop; the result is the induction value inside the body")
op("cf.if", kinds=ALL, pipe="S", operands=(N("cond", "b1"),), regions=("then", "else"), effects=("control",),
   legacy=("start_if", "start_elif", "start_else", "end_if"), doc="conditional; the else region may be empty")
op("cf.break", kinds=ALL, pipe="S", terminator=True, effects=("control",), legacy="break_loop", doc="leave the innermost cf.for")
op("cf.continue", kinds=ALL, pipe="S", terminator=True, effects=("control",), legacy="continue_loop", doc="next iteration of the innermost cf.for")
op("cf.return", kinds=ALL, pipe="S", operands=(VAR("values"),), terminator=True, effects=("control",),
   doc="end of a function; a kernel returns its output parameters")
op("cf.call", kinds=KERNEL, side="vec", pipe="V", operands=(N("callee", "value"), VAR("args")),
   attrs=(A("read", "list", doc="UB buffers the callee reads (autosync)"), A("write", "list", doc="UB buffers the callee writes (autosync)")),
   effects=("memory",), legacy="call_micro", doc="call a vf function; operands are the callee then its arguments")

# Regions (Surface only)
op("region.autosync", level="surface", pipe="S", attrs=(A("mode", "ident", default="conservative"),), regions=("body",),
   effects=("sync",), legacy=("start_auto_sync", "end_auto_sync"), doc="events for buffer hazards inside are inserted by autosync")
op("region.side", level="surface", pipe="S", attrs=(A("side", "ident", required=True, doc="cube | vec"),), regions=("body",),
   effects=("control",), legacy=("enter_vec_scope", "end_vec_scope", "enter_cube_scope", "end_cube_scope"),
   doc="ops inside run on the named side only (the old vec_scope / cube_scope)")

# Same-side pipe events
op("sync.event", kinds=KERNEL, pipe="S", attrs=(A("name", "str"), A("preset", "int|bool", default=False, doc="tokens set before the kernel body; True = the depth"),
   A("side", "ident", doc="cube | vec: the side whose pipes set and wait (autosync)"), A("ids", "list", doc="flag ids per slot (events pass)"),
   A("guards", "list", doc="the buffers whose hazards the event guards (autosync)")),
   results=(Res("event", "event<*>"),),
   legacy=("create_sevent", "create_devent", "create_tevent", "create_qevent"), doc="declare an event; the type carries its depth")
op("sync.set", kinds=KERNEL, operands=(N("event", "event<*>"),), attrs=(A("pipe", "ident"),), effects=("sync",), legacy="event_set",
   doc="signal the event from the set pipe")
op("sync.wait", kinds=KERNEL, operands=(N("event", "event<*>"),), attrs=(A("pipe", "ident"),), effects=("sync",), legacy="event_wait",
   doc="block the wait pipe until the event is set")
op("sync.set_all", kinds=KERNEL, operands=(N("event", "event<*>"),), effects=("sync",), legacy="event_setall", doc="set every slot of the event")
op("sync.release", kinds=KERNEL, operands=(N("event", "event<*>"),), effects=("sync",), legacy="event_release", doc="drain the event's slots")
op("sync.set_flag", kinds=KERNEL, attrs=(A("src", "ident", required=True), A("dst", "ident", required=True), A("event_id", IV, required=True)),
   effects=("sync",), legacy="setflag", doc="raw set_flag(src_pipe, dst_pipe, id)")
op("sync.wait_flag", kinds=KERNEL, attrs=(A("src", "ident", required=True), A("dst", "ident", required=True), A("event_id", IV, required=True)),
   effects=("sync",), legacy="waitflag", doc="raw wait_flag(src_pipe, dst_pipe, id)")
op("sync.barrier", kinds=KERNEL, attrs=(A("pipe", "ident", default="ALL"),), effects=("sync",), legacy="barrier", doc="pipe_barrier")

# A5 local buffer locks
for _action in ("get", "release"):
    op(f"sync.local_mutex_{_action}", level="lowered", kinds=KERNEL, devices=A5, pipe="S",
       attrs=(A("id", IV, required=True), A("pipe", "ident", required=True),
              A("side", "ident", required=True), A("mode", "int", required=True),
              A("guards", "list")), effects=("sync",),
       doc=f"mode-zero local buffer mutex {_action}; separate 32-ID namespace per AIC/AIV")

# Cross-core flags
op("sync.mutex", kinds=KERNEL, pipe="S", attrs=(A("kind", "ident", required=True, doc="vc (vec -> cube) | cv (cube -> vec)"),
   A("id", "int", required=True), A("depth", "int", required=True,
     doc="credit count; always explicit - a mutex whose credits are implied is a wrong answer, not a hang"),
   A("src_start_pipe", "ident"), A("dst_start_pipe", "ident"),
   A("src_end_pipe", "ident"), A("dst_end_pipe", "ident"),
   A("guards", "list", doc="the buffers whose hazards the mutex guards, as written by the author (guards=)")),
   results=(Res("flag", "flag"),), effects=("sync",),
   doc="declare a producer/consumer flag pair between the two sides (VcMutex / CvMutex); the consumer side publishes depth tokens up front")
for _n, _doc in (("lock", "producer: take the slot"), ("ready", "producer: publish"), ("wait", "consumer: block until published"),
                 ("free", "consumer: release the slot")):
    op(f"sync.mutex_{_n}", level="surface", kinds=KERNEL, operands=(N("flag", "flag"),), effects=("sync",), doc=_doc)
# Each cross-core primitive belongs to one side, as the old splitter placed them: the cube core publishes with
# cube_ready and waits for its vector cores with wait_vec; the vector cores publish with vec_ready and wait with
# wait_cube; the all-* barriers gather every core of one side.
for _n, _side in (("cube_ready", "cube"), ("wait_cube", "vec"), ("vec_ready", "vec"), ("wait_vec", "cube"),
                  ("allcube_ready", "cube"), ("allcube_wait", "cube"), ("allvec_ready", "vec"), ("allvec_wait", "vec"),
                  ("intracore_allvec_ready", "vec"), ("intracore_allvec_wait", "vec")):
    op(f"sync.crosscore.{_n}", kinds=KERNEL, side=_side, attrs=(A("flag_id", IV, required=True), A("pipe", "ident", required=True)),
       effects=("sync",), legacy=_n, doc=f"cross-core flag ({_side} side): {_n}")

# Debugging
op("debug.print", kinds=ALL, attrs=(A("fmt", "str", required=True), A("args", "list"), A("pipe", "ident")), effects=("debug",),
   legacy=("kernel_print", "sim_print"), doc="device printf (board) / simulator print")
op("debug.dump", kinds=KERNEL, operands=(R("src", "mem<*, *>"),), attrs=(A("desc", "str"), A("size", IV), A("pipe", "ident"), A("filename", "str")),
   effects=("debug",), legacy=("kernel_dump_tensor", "sim_dump_tensor"), doc="dump a tensor (board dump / simulator file)")
op("debug.print_reg", kinds=KV, operands=(R("reg", "reg<*, *>"),), attrs=(A("label", "str"), A("lanes", "int")), effects=("debug",),
   legacy="micro_print_reg", doc="print a vector register (simulator)")
op("debug.assert", kinds=ALL, operands=(N("cond", "b1"),), attrs=(A("msg", "str"),), effects=("debug",), doc="simulator-only assertion")
