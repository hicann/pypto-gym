# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Synchronisation, atomics, workspace, scalar-memory and debug rules of the frontend.

Events (``SEvent`` ... ``QEvent``), explicit flags, pipe barriers, cross-core signals and the
``atomic_add`` / ``atomic_max`` / ``atomic_min`` contexts map one to one onto ``sync.*`` /
``atomic.*`` ops; ``Var.GetValueFrom`` / ``SetValueTo`` become scalar loads and stores.
"""

from __future__ import annotations

import ast
from typing import Any

from ..ir import Ident
from ..ir.types import CellType, EventType, MemType, ScalarType
from . import dsl
from .errors import E_BAD_OPERAND, E_BAD_SIGNATURE, E_UNSUPPORTED
from .values import Dyn, ElemOffset

CROSSCORE_DEFAULT_PIPE = {
    "cube_ready": "FIX", "vec_ready": "MTE3", "wait_cube": "S", "wait_vec": "S",
    "allcube_ready": "FIX", "allcube_wait": "S", "allvec_ready": "MTE3", "allvec_wait": "S",
    "intracore_allvec_ready": "MTE3", "intracore_allvec_wait": "S",
}
EVENT_METHODS = {"set": "sync.set", "wait": "sync.wait", "setall": "sync.set_all", "release": "sync.release"}


class SyncRules:
    """Mixin of :class:`FunctionCompiler`."""

    def rule_event(self, depth: int, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        if self.kind != "kernel":
            raise self.err(E_UNSUPPORTED, "events are declared in kernel functions", node)

        def arg(i: int, key: str, default: Any = None) -> Any:
            return args[i] if len(args) > i else kwargs.get(key, default)

        src, dst = arg(0, "src_pipe"), arg(1, "dst_pipe")
        if not (isinstance(src, dsl.EnumValue) and isinstance(dst, dsl.EnumValue) and src.family == dst.family == "Pipe"):
            raise self.err(E_BAD_SIGNATURE, "an event takes (src_pipe, dst_pipe, preset=False, name='')", node)
        preset = bool(arg(2, "preset", False))
        name = arg(3, "name", "") or "ev"
        attrs: dict[str, Any] = {}
        if arg(3, "name", ""):
            attrs["name"] = name
        if preset:
            attrs["preset"] = True
        return self.emit("sync.event", (), attrs, EventType(depth, src.name, dst.name), name, node)

    def event_method(self, ev: Dyn, name: str, args: list[Any], node: ast.AST) -> Any:
        if name not in EVENT_METHODS:
            raise self.err(E_UNSUPPORTED, f"events have no method {name!r}", node)
        if args:
            raise self.err(E_BAD_SIGNATURE, f"event.{name}() takes no arguments", node)
        self.emit(EVENT_METHODS[name], (ev,), {}, None, None, node)
        return None

    def rule_sync(self, rule: str, callee: Any, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        if self.kind != "kernel":
            raise self.err(E_UNSUPPORTED, f"{callee.__name__} is only available in kernel functions", node)

        def arg(i: int, key: str, default: Any = None) -> Any:
            return args[i] if len(args) > i else kwargs.get(key, default)

        if rule in ("sync.set_flag", "sync.wait_flag"):
            src, dst, event_id = arg(0, "src"), arg(1, "dst"), arg(2, "event_id")
            if not (isinstance(src, dsl.EnumValue) and isinstance(dst, dsl.EnumValue)):
                raise self.err(E_BAD_SIGNATURE, f"{callee.__name__}(src_pipe, dst_pipe, event_id)", node)
            self.emit(rule, (), {"src": Ident(src.name), "dst": Ident(dst.name), "event_id": self.attr(self.rvalue(event_id, node))}, None, None, node)
            return None
        if rule.startswith("barrier"):
            _, _, pipe = rule.partition(":")
            if not pipe:
                p = arg(0, "pipe")
                if not isinstance(p, dsl.EnumValue):
                    raise self.err(E_BAD_SIGNATURE, "barrier(pipe)", node)
                pipe = p.name
            self.emit("sync.barrier", (), {} if pipe == "ALL" else {"pipe": Ident(pipe)}, None, None, node)
            return None
        if rule.startswith("crosscore:"):
            from ..devices import load

            name = rule.split(":")[1]
            flag_id = arg(0, "flag_id", 0)
            pipe = arg(1, "pipe")
            maximum = load(self.fe.kernel.device).crosscore_id_max
            if not isinstance(flag_id, int) or isinstance(flag_id, bool) or not 0 <= flag_id <= maximum:
                raise self.err(E_BAD_OPERAND, f"flag_id must be a static int in 0..{maximum}", node)
            pipe_name = pipe.name if isinstance(pipe, dsl.EnumValue) else CROSSCORE_DEFAULT_PIPE[name]
            self.emit(f"sync.crosscore.{name}", (), {"flag_id": flag_id, "pipe": Ident(pipe_name)}, None, None, node)
            return None
        raise self.err(E_UNSUPPORTED, f"no compile rule for {rule!r}", node)

    def rule_debug(self, rule: str, callee: Any, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        if rule == "debug.print":
            if callee.__name__ == "kernel_print":
                fmt = args[0] if args else kwargs.get("fmt", "")
                if not isinstance(fmt, str):
                    raise self.err(E_BAD_OPERAND, "kernel_print(fmt, *args) needs a static format string", node)
                payload = [self.rvalue(a, node) for a in args[1:]]
            else:
                fmt = " ".join("%s" for _ in args)
                payload = [self.rvalue(a, node) for a in args]
            attrs: dict[str, Any] = {"fmt": fmt}
            if payload:
                attrs["args"] = [self.attr(p) for p in payload]
            pipe = kwargs.get("pipe")
            if isinstance(pipe, dsl.EnumValue):
                attrs["pipe"] = Ident(pipe.name)
            self.emit("debug.print", (), attrs, None, None, node)
            return None
        if rule == "debug.dump":
            src = args[0] if args else kwargs.get("src", kwargs.get("tensor"))
            if isinstance(src, ElemOffset):
                src = src.base
            if not (isinstance(src, Dyn) and isinstance(src.type, MemType)):
                raise self.err(E_BAD_OPERAND, f"{callee.__name__} needs a tensor", node)
            attrs = {}
            if callee.__name__ == "kernel_dump_tensor":
                desc = args[1] if len(args) > 1 else kwargs.get("desc")
                size = args[2] if len(args) > 2 else kwargs.get("dumpSize", kwargs.get("size"))
                if desc is not None:
                    attrs["desc"] = str(self.rvalue(desc, node)) if not isinstance(desc, Dyn) else desc.name
                if size is not None:
                    attrs["size"] = self.attr(self.rvalue(size, node))
            else:
                filename = args[1] if len(args) > 1 else kwargs.get("filename")
                if filename is not None:
                    attrs["filename"] = str(filename)
                pipe = args[2] if len(args) > 2 else kwargs.get("pipe")
                if isinstance(pipe, dsl.EnumValue):
                    attrs["pipe"] = Ident(pipe.name)
            self.emit("debug.dump", (src.plain(),), attrs, None, None, node)
            return None
        raise self.err(E_UNSUPPORTED, f"no compile rule for {rule!r}", node)

    def rule_core_misc(self, rule: str, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        if rule == "core.set_hf32":
            self.emit(rule, (), {"enable": bool(args[0] if args else kwargs.get("enable", True))}, None, None, node)
            return None
        if rule == "core.clean_dcache":
            if len(args) + int("dst" in kwargs) != 1 or set(kwargs) - {"dst"}:
                raise self.err(E_BAD_SIGNATURE, "clean_dcache(dst) takes exactly one target and no cache-policy options", node)
            dst = args[0] if args else kwargs.get("dst")
            if not (isinstance(dst, Dyn) and isinstance(dst.type, MemType) and dst.type.space == "gm"):
                raise self.err(E_BAD_OPERAND, "clean_dcache(dst) needs a GM tensor view", node)
            self.emit(rule, (), {"dst": dst.plain()}, None, None, node)
            return None
        if rule == "noop":
            return None
        raise self.err(E_UNSUPPORTED, f"no compile rule for {rule!r}", node)

    # -- atomic contexts ------------------------------------------------------------------------------

    def compile_atomic_with(self, op: str, ctx: ast.Call, body: list[ast.stmt], node: ast.With) -> None:
        if self.kind != "kernel":
            raise self.err(E_UNSUPPORTED, "atomic contexts are used in kernel functions", node)
        if self.atomic is not None:
            raise self.err(E_UNSUPPORTED, "atomic contexts do not nest", node)
        for kw in ctx.keywords:
            if kw.arg == "cond" and not (isinstance(kw.value, ast.Constant) and kw.value.value is None):
                raise self.err(E_UNSUPPORTED, "atomic_add(cond=...) is not supported (the old DSL rejected it too)", node)
        self.emit("atomic.begin", (), {"op": Ident(op)}, None, None, node)
        self.atomic = Ident(op)
        try:
            with self.region_scope(node):
                self.compile_body(body)
        finally:
            self.atomic = None
        self.emit("atomic.end", (), {}, None, None, node)

    # -- scalar <-> memory ------------------------------------------------------------------------------

    def cell_method(self, cell: Dyn, name: str, args: list[Any], kwargs: dict[str, Any], node: ast.AST) -> Any:
        if name == "set":
            self.emit("scalar.set", (cell, self.rvalue(args[0], node)), {}, None, None, node)
            return None
        if name == "GetValueFrom":
            self.load_scalar(cell, args[0], node)
            return cell
        if name == "SetValueTo":
            self.store_scalar(args[0], cell, node)
            return None
        raise self.err(E_UNSUPPORTED, f"Var has no method {name!r}", node)

    def _element_ref(self, v: Any, node: ast.AST) -> tuple[Dyn, Any]:
        """(tensor, flat index) of the first element of a GM/workspace or UB view."""
        base = v.base if isinstance(v, ElemOffset) else v
        if isinstance(base, Dyn) and isinstance(base.type, MemType) and base.type.space not in ("gm", "ws", "ub"):
            raise self.err(E_BAD_OPERAND, "GetValueFrom / SetValueTo require GM/workspace or UB memory", node)
        if isinstance(v, ElemOffset):
            return v.base.plain(), v.offset
        if isinstance(v, Dyn) and isinstance(v.type, MemType):
            g = self.geom(v, node)
            root = self.named_values.get(g.root)
            if root is None:
                raise self.err(E_BAD_OPERAND, "cannot resolve the tensor behind this view", node)
            index: Any = 0
            stride: Any = 1
            for d in reversed(range(g.rank)):
                index = self.arith(ast.Add(), index, self.arith(ast.Mult(), g.offset[d], stride, node), node)
                stride = self.arith(ast.Mult(), stride, g.shape[d], node)
            return root.plain(), index
        raise self.err(E_BAD_OPERAND, "GetValueFrom / SetValueTo need a one-element tensor view", node)

    def load_scalar(self, cell: Dyn, src: Any, node: ast.AST) -> None:
        if self.kind != "kernel":
            raise self.err(E_UNSUPPORTED, "GetValueFrom is a kernel-level operation", node)
        base, index = self._element_ref(src, node)
        ct = cell.type
        assert isinstance(ct, CellType)
        val = self.emit("scalar.load", (base, index), {}, ScalarType(base.type.dtype), None, node)
        if base.type.dtype != ct.dtype:
            val = self.emit("scalar.cast", (val,), {}, ScalarType(ct.dtype), None, node)
        self.emit("scalar.set", (cell, val), {}, None, None, node)

    def store_scalar(self, dst: Any, cell: Dyn, node: ast.AST) -> None:
        if self.kind != "kernel":
            raise self.err(E_UNSUPPORTED, "SetValueTo is a kernel-level operation", node)
        base, index = self._element_ref(dst, node)
        self.emit("scalar.store", (base, index, cell), {}, None, None, node)
