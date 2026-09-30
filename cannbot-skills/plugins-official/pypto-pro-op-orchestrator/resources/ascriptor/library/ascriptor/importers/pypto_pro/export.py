# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pinned native graph adapter, imported only when a caller requests export."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import threading
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from .schema import FIELDS, PROFILE, SCHEMA, Bundle, ProImportError, require
from .session import profile

# Pro reads tile layout defaults from these while parsing; its jit sets them for the kernel's arch (I017).
ARCH_ENVIRONMENT = {"a5": {"PYPTOPRO_JIT_ARCH": "a5", "PYPTOPRO_NPU_ARCH": "dav-c310"}}
_arch_lock = threading.Lock()
_arch_state = {"users": 0, "saved": {}}


@contextmanager
def architecture(arch):
    """Parse under Pro's environment for ``arch``; the caller's values, including absence, return afterwards.

Concurrent exports share one setting and restore it when the last of them leaves."""
    values = ARCH_ENVIRONMENT[arch]
    with _arch_lock:
        if not _arch_state["users"]:
            _arch_state["saved"] = {key: os.environ.get(key) for key in values}
            os.environ.update(values)
        _arch_state["users"] += 1
    try:
        yield
    finally:
        with _arch_lock:
            _arch_state["users"] -= 1
            if not _arch_state["users"]:
                for key, value in _arch_state["saved"].items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _runtime():
    try:
        pro = importlib.import_module("pypto_pro")
        pypto = importlib.import_module("pypto")
        ir = importlib.import_module("pypto.pypto_impl.ir")
    except ImportError as exc:
        raise ProImportError("Pro export requires the pinned Pro overlay and native metadata helper") from exc
    expected = profile()
    roots = {"pypto_pro": Path(pro.__file__).parent, "pypto": Path(pypto.__file__).parent}
    for package, entries in expected["files"].items():
        for relative, digest in entries.items():
            path = roots[package] / relative
            require(path.is_file() and _sha(path) == digest, f"Pro compatibility mismatch: {package}/{relative}")
    try:
        helper_spec = importlib.util.find_spec("pypto_pro._export_metadata")
        require(helper_spec is not None and helper_spec.origin is not None,
                "Native metadata helper is unavailable")
        require(_sha(helper_spec.origin) == expected["helper_sha256"], "Native metadata helper fingerprint mismatch")
        helper = importlib.import_module("pypto_pro._export_metadata")
    except ImportError as exc:
        raise ProImportError("Native metadata helper could not load against the pinned Pro ABI") from exc
    require(getattr(helper, "schema", None) == "pypto-pro-export-metadata/1", "Native metadata helper version mismatch")
    return ir, helper, {"profile": PROFILE, "fingerprint": expected["fingerprint"], "native_helper": _sha(helper.__file__)}


class _Graph:
    def __init__(self, ir, helper, program, parser, side):
        self.ir, self.helper, self.program, self.parser, self.side = ir, helper, program, parser, side
        self.nodes = []
        self.memo = {}
        self.keep = []
        self.objects = {}
        self.files = {}
        self.metadata = {k: [] for k in ("groups", "mutex", "tuples", "roles", "declarations", "allocations")}

    def attributes(self, kind, obj, location):
        readers = {"Call": self.helper.call_attributes, "Function": self.helper.function_attributes,
                   "ForStmt": self.helper.for_attributes}
        try:
            return dict(readers[kind](obj))
        except (TypeError, RuntimeError) as exc:
            raise ProImportError(f"Cannot export {kind} attributes: {exc}", location) from exc

    def location(self, obj, fallback):
        span = getattr(obj, "span", None)
        # This parser stores Python AST's zero-based columns, even though the
        # native Span.is_valid rejects column zero. Normalize at the boundary.
        if span is None or not span.filename or span.begin_line <= 0 or span.begin_column < 0:
            return fallback, "enclosing" if fallback else "none"
        name = Path(span.filename).name
        require(name not in self.files or self.files[name] == span.filename,
                f"Ambiguous source basename {name!r}; use distinct source filenames")
        self.files[name] = span.filename
        return {"file": name, "line": span.begin_line, "column": span.begin_column + 1}, "native"

    def value(self, obj, location=None):
        if obj is None or type(obj) in (str, int, float, bool):
            return obj
        if isinstance(obj, (list, tuple)):
            return [self.value(x, location) for x in obj]
        if isinstance(obj, dict):
            require(all(isinstance(k, str) for k in obj), "Non-string native metadata key", location)
            return {k: self.value(v, location) for k, v in sorted(obj.items())}
        if isinstance(obj, self.ir.DataType):
            return {"$dtype": obj.to_string(), "code": int(obj), "bits": obj.get_bit()}
        if hasattr(type(obj), "__members__"):
            return {"$enum": type(obj).__name__, "name": obj.name, "value": int(obj.value)}
        kind = type(obj).__name__
        require(kind in FIELDS, f"Unsupported native node {kind}", location)
        previous = self.memo.get(id(obj))
        if previous is not None:
            return {"$ref": previous}
        ident = f"{self.side}/n{len(self.nodes)}"
        self.memo[id(obj)] = ident
        self.keep.append(obj)  # Retain wrappers: Python ids alone are reusable.
        self.objects[ident] = obj
        loc, origin = self.location(obj, location)
        node = {"id": ident, "kind": kind, "fields": {}, "location": loc, "origin": origin}
        self.nodes.append(node)
        for field in FIELDS[kind]:
            if kind == "Call" and field == "kwargs":
                data = self.attributes(kind, obj, loc)
                # A native process counter is replaced by the referenced allocation
                # itself. No physical address or alias relation is inferred here.
                if "memref_id" in data:
                    memory = getattr(obj.type, "memref", None)
                    require(memory is not None, "memref_id without a typed allocation", loc)
                    data["memref_id"] = memory
            elif kind == "Function" and field == "attrs":
                data = self.attributes(kind, obj, loc)
            elif kind == "ForStmt" and field == "attrs":
                data = self.attributes(kind, obj, loc)
            else:
                try:
                    data = getattr(obj, field)
                except AttributeError as exc:
                    raise ProImportError(f"Native ABI missing {kind}.{field}", loc) from exc
                if kind == "Program" and field == "functions":
                    data = sorted(data.values(), key=lambda f: f.name)
            node["fields"][field] = self.value(data, loc)
        if kind == "TupleType":
            debug = self.program.debug_info
            fields = debug.get_tuple_fields(obj)
            name = debug.get_tuple_name(obj)
            if fields is not None or name is not None:
                self.metadata["tuples"].append({"type": {"$ref": ident}, "name": name,
                                                 "fields": self.value(fields, loc)})
        if kind == "AssignStmt":
            producer = obj.value.name if isinstance(obj.value, self.ir.Call) else None
            roles = {"vf.reg_tensor": "register", "vf.create_mask": "predicate"}
            if producer in roles:
                self.metadata["roles"].append({"value": self.value(obj.var, loc), "role": roles[producer]})
            if producer == "block.make_tile":
                key = hashlib.sha256(json.dumps([loc, obj.var.name], sort_keys=True).encode()).hexdigest()
                self.metadata["declarations"].append({"key": key, "value": self.value(obj.var, loc), "location": loc})
        return {"$ref": ident}

    def finish(self):
        root = self.value(self.program)["$ref"]
        fallback = self.nodes[0]["location"]
        for value, (depth, ids, space) in self.parser.tile_group_meta.items():
            self.metadata["groups"].append({"value": self.value(value, fallback), "depth": depth,
                                             "mutex_ids": self.value(ids, fallback), "space": self.value(space, fallback)})
        for value, (ids, candidates) in self.parser._tile_mutex_meta.items():
            self.metadata["mutex"].append({"value": self.value(value, fallback), "ids": self.value(ids, fallback),
                                            "candidates": self.value(candidates, fallback)})
        memories = [(n["id"], self.objects[n["id"]]) for n in self.nodes if n["kind"] == "MemRef"]
        for ident, memory in memories:
            self.metadata["allocations"].append({"value": {"$ref": ident}, "same_allocation": [
                {"$ref": other} for other, candidate in memories if self.ir.MemRef.same_allocation(memory, candidate)]})
        return {"side": self.side, "active": self.parser.matched_target, "root": root,
                "nodes": self.nodes, "metadata": self.metadata}


def export_kernel(kernel, *, directions=None, block_dim=None, concrete_key=None, datatype_consts=None,
                  bound_signature=None, workspace=None) -> Bundle:
    """Export without device allocation, compilation, launcher creation or codegen.

Directions are explicit caller annotations (input/output/inout); unspecified
parameters remain unknown. They are not guessed from the source parameter name.
Workspace entries explicitly mark private Tensor parameters and bind their sizes
to public shapes; export itself never allocates storage.
"""
    ir, helper, producer = _runtime()
    require(getattr(kernel, "arch", None) == "a5", "Expected a Pro jit kernel for arch='a5'")
    with architecture(kernel.arch):
        definition = kernel.to_kernel_def(concrete_key, datatype_consts=datatype_consts)
        require("observer" in inspect.signature(definition.parse_target_program).parameters,
                "Pro parser export observer is unavailable")
        source_file = Path(definition._source_file)
        names = list(inspect.signature(definition._func).parameters)
        directions = directions or {}
        require(set(directions) <= set(names), "Direction metadata names an unknown kernel parameter")
        pipeline = definition._pipeline is not None
        pipeline_config = asdict(definition._pipeline) if pipeline else None
        if pipeline:
            from pypto_pro.runtime.pipeline import transform_pipeline
            from pypto_pro.runtime.pipeline._analyzer import probe_kernel_facts

            facts, types = probe_kernel_facts(definition, bound_signature)
            definition._func_def = transform_pipeline(
                definition._func_def, definition._closure_vars, definition._pipeline,
                if_const_map=facts, var_types=types, tilingkey_consts=definition._tilingkey_consts,
                datatype_consts=definition._datatype_consts)
            definition._pipeline = None
        targets = []
        for side, target in (("cube", ir.SectionKind.Cube), ("vec", ir.SectionKind.Vector)):
            def observe(program, parser, side=side):
                targets.append(_Graph(ir, helper, program, parser, side).finish())
            definition.parse_target_program(target, bound_signature, observer=observe)
    require(len(targets) == 2, "Pro did not invoke both target observers")
    if not any(graph["active"] for graph in targets):
        for graph in targets:
            graph["active"] = True
    datatypes = {k: {"$dtype": v.to_string(), "code": int(v), "bits": v.get_bit()}
                 for k, v in (datatype_consts or {}).items()}
    document = {
        "schema": SCHEMA, "producer": producer, "target": "a5",
        "source": {"name": source_file.name, "sha256": _sha(source_file)},
        "specialization": {"tilingkey": concrete_key or {}, "datatypes": datatypes,
                           "bound_signature": asdict(bound_signature) if bound_signature else None,
                           "pipeline": pipeline, "pipeline_config": pipeline_config, "auto_mutex": definition._auto_mutex},
        "abi": {"block_dim": block_dim, "directions": {n: directions.get(n, "unknown") for n in names},
                "workspace": [] if workspace is None else workspace},
        "targets": targets,
    }
    return Bundle.from_dict(document)
