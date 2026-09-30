# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded instruction import. No ordinary lowering or synchronization planning."""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass, replace

from ...devices import load as load_device
from ...ir import (
    LOWERED,
    REGISTRY,
    Block,
    FuncRef,
    Function,
    Ident,
    Loc,
    Module,
    Rewriter,
    Value,
    VerifyError,
    check,
)
from ...ir.types import CellType, DimValue, MaskType, MemType, RegType, ScalarType, UnalignRegType, dtype
from .ctrl import UNSIGNED, unsigned_result
from .ctrl import convert as ctrl_call
from .schema import Bundle, ProImportError, require
from .session import prepare_import

DTYPES = {"bfloat16": "bf16", "fp16": "f16", "fp32": "f32", "int32": "i32", "int8": "i8", "uint8": "u8", "uint16": "u16", "uint32": "u32", "int64": "i64", "index": "i64", "bool": "b1",
          "fp8e4m3fn": "e4m3", "fp8e5m2": "e5m2", "fp8e8m0": "e8m0"}


@dataclass
class Memory:
    value: Value
    shape: tuple[int | Value, int | Value]  # Pro coordinates; Right is [K,N].
    valid: tuple[int | Value, int | Value] | None
    pitch: int | None = None  # Physical row pitch of an admitted UB subview, or a GM view's row stride.
    root: Memory | None = None  # The GM parameter or allocation that a view or tile alias re-declares.
    origin: int | Value = 0  # Element offset of a GM view in its root.
    transposed: bool = False  # A ZN Mat, typed as the NZ tile of its transpose.
    slot: bool = False  # A mem.get_buf selection of a slot buffer (slots).
    nz: tuple[int, int] | None = None  # The logical [R, C] of an NZ-packed GM parameter's storage (nz_parameters).


@dataclass
class Struct:
    fields: list


class Importer:
    def __init__(self, bundle):
        self.plan = prepare_import(bundle)
        self.document = bundle.document
        self.graphs = [g for g in self.document["targets"] if g["active"]]
        require(bool(self.graphs), "No active source side")
        self.mixed = len(self.graphs) == 2
        self.graph = self.graphs[0]
        self.nodes = {n["id"]: n for g in self.graphs for n in g["nodes"]}
        self.side = self.graph["side"]
        self.rw = Rewriter(Module(self.document["source"]["name"]), "pypto_import")
        self.serial = 0
        self.emitted = []
        self.ledger = {}
        self.functions = []
        self.allocations, self.roots = {}, {}
        self.hoisting, self.hoisted = False, set()  # Control-flow tile re-declarations bound at the prologue (Context.hoist).
        self.device = load_device("950")
        from .cross import declaration_index
        self.declaration_keys, self.declaration_sides = declaration_index(self)
        self.storage_links = {}
        self.simt_sources, self.simt_functions = {}, {}
        self.unsigned_scope = False  # UINT32 scalars are admitted only inside SIMT functions.
        self.simt_launched = False
        self.spr, self.spr_norm = "unset", False  # Vector mask SPR state of the current side (vector_spr).
        self.mx_next, self.mx_pairs, self.mx_split = None, {}, {}  # Statement successors and pending MX moves (mx).

    def node(self, ref):
        return self.nodes[ref["$ref"]] if isinstance(ref, dict) and "$ref" in ref else ref

    def fail(self, node, message):
        raise ProImportError(message, node["location"])

    def need(self, condition, node, message):
        require(condition, message, node["location"])

    def unique(self, hint):
        self.serial += 1
        return f"{hint}_{self.serial}"

    def record(self, node, before):
        entry = self.ledger.setdefault(node["id"], {"source": node["id"], "kind": node["kind"], "target_ids": []})
        entry["target_ids"].extend(i for i in self.emitted[before:] if i not in entry["target_ids"])
        entry["disposition"] = "translated" if entry["target_ids"] else "declaration"

    def dt(self, tagged, node):
        self.need(isinstance(tagged, dict) and tagged.get("$dtype") in DTYPES, node, "Unsupported source dtype")
        result = dtype(DTYPES[tagged["$dtype"]])
        self.need(tagged["bits"] == (8 if tagged["$dtype"] == "bool" else result.bits), node,
                  "Source dtype width contradicts its name")
        return result

    def scalar_type(self, ref):
        node = self.node(ref)
        self.need(node["kind"] == "ScalarType", node, "Expected an admitted scalar type")
        dt = self.dt(node["fields"]["dtype"], node)
        self.need(dt.name in {"i32", "i64", "b1", "f32"} | ({"u32"} if self.unsigned_scope else set()), node,
                  "Narrow scalar legalization is not admitted")
        return ScalarType(dt)

    def literal(self, ref, site):
        node = self.node(ref)
        self.need(node["kind"] in {"ConstInt", "ConstFloat", "ConstBool"}, site, "Static literal required")
        return node["fields"]["value"]

    def shape(self, refs, node):
        result = tuple(self.literal(v, node) for v in refs)
        self.need(len(result) == 2 and all(type(n) is int and n > 0 for n in result), node,
                  "Only fixed positive two-dimensional storage is admitted")
        return result

    def parameter(self, var):
        typ = self.node(var["fields"]["type"])
        if typ["kind"] == "ScalarType":
            return self.scalar_type(typ), None
        self.need(typ["kind"] == "TensorType", var, "Only tensor and admitted scalar kernel parameters are supported")
        f = typ["fields"]
        from .dynamic import parameter_shape
        from .mx import plane_parameter, storage
        from .sort import index_storage
        dt = self.dt(f["dtype"], var)
        shape = plane_parameter(self, var, f["shape"]) if dt.name == "e8m0" else parameter_shape(self, var, f["shape"])
        # UINT32 GM tensors carry sort32 identifiers or SIMT launch storage (sort.guard, simt_exact.launch_storage);
        # FP8 and E8M0 tensors carry microscaling operands and planes (mx.guard).
        view = self.node(f["tensor_view"])["fields"] if f["tensor_view"] else None
        packed = view is not None and view["layout"]["name"] == "NZ"
        self.need(dt.name in {"f16", "bf16", "f32", "i32"} or index_storage(dt) or storage(dt) or packed and dt.name == "i8", var,
                  "Unsupported GM element type")
        self.need(f["memref"] is None, var, "Parameter memref binding requires explicit ABI legalization")
        if view is not None:
            self.need(view["layout"]["name"] in {"ND", "NZ"}, var, "Only ND and NZ-packed tensor parameters are admitted")
            if view["ptr"] is not None:
                pointer = self.node(view["ptr"])
                self.need(pointer["kind"] == "Var" and pointer["fields"]["name"] == var["fields"]["name"] + "_ptr",
                          var, "Noncanonical parameter pointer provenance")
                pointer_type = self.node(pointer["fields"]["type"])
                self.need(pointer_type["kind"] == "PtrType" and pointer_type["fields"]["dtype"] == f["dtype"],
                          var, "Parameter pointer dtype mismatch")
            if view["layout"]["name"] == "NZ":
                from .nz_parameters import declare
                return declare(self, var, dt, shape, view)
            strides = [self.literal(v, var) for v in view["stride"]]
            contiguous = [shape[1], 1] if len(shape) == 2 else [shape[1] * shape[2], shape[2], 1]  # rank 3: MX planes
            self.need(not strides or strides == contiguous, var, "Non-contiguous GM parameter strides are not admitted")
            valid = [self.literal(v, var) for v in view["valid_shape"]]
            self.need(not valid or valid == list(shape), var, "Parameter valid-shape override is not admitted")
        return MemType("gm", dt, shape), shape

    def run_graph(self):
        program = self.nodes[self.graph["root"]]
        functions = [self.node(f) for f in program["fields"]["functions"]]
        kernels = [f for f in functions if f["fields"]["func_type"]["name"] == "Opaque"]
        self.need(len(kernels) == 1, program, "Expected one Pro device kernel")
        source = kernels[0]
        self.simt_sources = {f["fields"]["name"]: f for f in functions if f is not source}
        self.need(all(f["fields"]["func_type"]["name"] in {"SimtVF", "SimtCallee"} for f in self.simt_sources.values())
                  and len(self.simt_sources) == len(functions) - 1, program, "External functions need separate conversion rules")
        self.need(not source["fields"]["attrs"] and not source["fields"]["return_types"], source,
                  "Function attributes or return types need explicit legalization")
        abi = self.document["abi"]
        self.need(type(abi["block_dim"]) is int and abi["block_dim"] > 0, source, "Explicit positive block_dim metadata is required")
        from .workspace import build as build_workspace

        private = {entry['parameter'] for entry in abi['workspace']}
        self.spr, self.spr_norm = "unset", False
        ctx = Context(self)
        self.dynamic_params, self.nz_parameters = {}, {}
        params, outputs = [], []
        seen_scalar = False
        for ref in source["fields"]["params"]:
            var = self.node(ref)
            if var['fields']['name'] in private:
                continue
            typ, shape = self.parameter(var)
            self.need(not (shape and seen_scalar), var, "Interleaved scalar/tensor parameters require ABI legalization")
            seen_scalar |= shape is None
            name = var["fields"]["name"]
            value = Value(name, typ)
            params.append(value)
            actual_shape = tuple(self.dynamic_params[d.name] if isinstance(d, DimValue) else d for d in shape) if shape else None
            nz = self.nz_parameters.get(name)
            ctx.env[name] = Memory(value, actual_shape, actual_shape, nz=nz and tuple(nz)) if shape else value
            direction = abi["directions"].get(name)
            self.need(direction in {"input", "output", "inout"}, var, f"Explicit ABI direction required for {name}")
            if direction in {"output", "inout"}:
                self.need(isinstance(typ, MemType), var, "Scalar output parameters are not admitted")
                outputs.append(value)
        self.need(bool(outputs), source, "At least one tensor output is required")
        from .simt_exact import launch_storage
        launch_storage(self, source, params)
        variables = {self.node(ref)['fields']['name']: self.node(ref) for ref in source['fields']['params']}
        self.workspaces, self.workspace_layout = build_workspace(ctx, source, variables)
        self.need(not set(self.dynamic_params).intersection(p.name for p in params), source,
                  "Dynamic dimension collides with an explicit parameter")
        params.extend(self.dynamic_params.values())
        for name, value in self.dynamic_params.items():
            ctx.env[name] = value
            ctx.bounds[name] = (1, 2**63 - 1)
            ctx.nonnegative.add(name)
        from .slots import plan as slot_plan
        self.slot_tiles, self.slot_cursors, self.group_slots = slot_plan(self, self.graph)
        ctx.prepare_cells(source["fields"]["body"])
        ctx.block(source["fields"]["body"])
        ctx.emit("cf.return", source)
        self.functions.append(Function("func", source["fields"]["name"] + "." + self.side,
                                       tuple(params), {"side": Ident(self.side)}, Block(tuple(ctx.ops))))
        return source, tuple(params), outputs

    def run(self):
        signature = None
        for graph in self.graphs:
            self.graph, self.side = graph, graph["side"]
            self.allocations, self.roots = {}, {}
            source, params, outputs = self.run_graph()
            current = (source["fields"]["name"], params, outputs)
            self.need(signature is None or current == signature, source, "Paired sides have inconsistent parameter ABIs")
            signature = current
        abi = self.document["abi"]
        for op in self.plan.operations:
            self.need(op.id in self.ledger, self.nodes[op.id], f"Unconsumed source operation {op.opcode}")
        attrs = {"device": "950", "ir": LOWERED, "mode": "mix" if self.mixed else self.side, "next_id": self.rw._next_id,
                 "source": self.document["source"]["name"], "source_sha256": self.document["source"]["sha256"],
                 "meta": {"kernel": source["fields"]["name"], "block_dim": abi["block_dim"], "outputs": outputs,
                          "workspace": self.workspaces, "crosscore": [],
                          "directions": {k: v for k, v in abi["directions"].items()
                                         if k not in {e['parameter'] for e in abi['workspace']}}},
                 "import_ledger": list(self.ledger.values()), "import_producer": self.document["producer"]}
        if self.mixed:
            attrs["import_storage_links"] = list(self.storage_links.values())
        if self.dynamic_params:
            attrs["meta"]["pro_dynamic_dimensions"] = list(self.dynamic_params)
        if self.workspace_layout:
            attrs['meta']['pro_workspace_layout'] = self.workspace_layout
        if self.nz_parameters:
            attrs['meta']['pro_nz_parameters'] = self.nz_parameters
        module = Module(source["fields"]["name"], attrs, tuple(self.functions))
        try:
            check(module)
        except VerifyError as exc:
            self.fail(source, f"Imported module violates target IR invariants: {exc}")
        return module


class Context:
    def __init__(self, owner, *, vf=False, depth=0, simt=None, top=False):
        self.o = owner
        self.env = {}
        self.ops = []
        self.vf = vf
        self.top = top  # A VF body outlined from the top level of its side.
        self.simt = simt  # The callee's max_threads inside a SIMT function.
        self.depth = depth
        self.cells = {}
        self.nonnegative = set()
        self.bounds = {}
        self.cell_ranges = {}  # Ranges of reassigned scalars at the current statement (loop_extents).
        self.upper = {}  # Proven symbolic upper bounds, keyed by snapshot name.
        self.expressions, self.tiled_loops, self.offset_limits, self.residuals = {}, {}, {}, {}
        self.reads, self.writes = [], []
        self.written = set()  # Registers/predicates written on every path so far.
        self.whole = None  # Source id of the expression that is an entire assignment value.
        self.unsigned = set()  # Values Pro's CCE holds in uint64 locals (CTRL reads and their arithmetic).
        self.unsigned_cells = set()  # Cells that ever held such a value; shared by every nested context.
        self.vf_memory = None  # Cursor and unaligned-register state of one VF function (vector_memory).
        self.parent = None  # The enclosing context of a branch or loop body.

    def child(self):
        from .descriptors import clone_env

        other = Context(self.o, vf=self.vf, depth=self.depth + 1, simt=self.simt)
        other.env, other.cells, other.parent = clone_env(self.env), self.cells, self
        other.nonnegative = set(self.nonnegative)
        other.unsigned = set(self.unsigned)
        other.unsigned_cells = self.unsigned_cells
        other.bounds = dict(self.bounds)
        other.cell_ranges = dict(self.cell_ranges)
        other.upper = dict(self.upper)
        for name in ('expressions', 'tiled_loops', 'offset_limits', 'residuals'):
            setattr(other, name, dict(getattr(self, name)))
        other.reads, other.writes = self.reads, self.writes
        other.written = set(self.written)
        return other

    def emit(self, opcode, source, operands=(), typ=None, attrs=None, regions=(), result_name=None):
        values = (Value(result_name or self.o.unique("v"), typ),) if typ is not None else ()
        attributes = dict(attrs or {})
        spec = REGISTRY.get(opcode)  # Fixed pipes belong to the registry, not extra attributes.
        loc = source["location"]
        self.o.need(loc is not None, source, "Operation has no source location")
        operation = self.o.rw.make(opcode, operands, results=values, attrs=attributes, regions=regions,
                                   loc=Loc.of(f"{loc['file']}:{loc['line']}:{loc['column']}"),
                                   note=f"Pro {source['id']} ({source['fields'].get('name', source['kind'])})")
        self.ops.append(operation)
        self.o.emitted.append(operation.id)
        # Variadic specs are never register writers; zip only the fixed operand prefix.
        written = [v for v, use in zip(operands, spec.operands, strict=False) if isinstance(v, Value)
                   and isinstance(v.type, (RegType, MaskType)) and use.access in {"write", "readwrite"}]
        self.written.update(written)
        if self.vf and self.o.spr == "live" and any(isinstance(v.type, MaskType) for v in (*values, *written)):
            from .vector_spr import keeps_spr

            if not keeps_spr(opcode, attributes):
                self.o.spr = "predicate"  # Unmeasured predicate producers may rewrite the SPR (vector_spr).
        return values[0] if values else None

    def assigned(self, ref):
        return [self.o.node(n["fields"]["var"])["fields"]["name"] for n in self.walk(ref) if n["kind"] == "AssignStmt"]

    def walk(self, ref):
        node = self.o.node(ref)
        yield node
        kind, f = node["kind"], node["fields"]
        children = f["stmts"] if kind == "SeqStmts" else [f["body"]] if kind in {"ForStmt", "WhileStmt"} else []
        if kind == "IfStmt":
            children = [x for x in (f["then_body"], f["else_body"]) if x]
        for child in children:
            yield from self.walk(child)

    def prepare_cells(self, body):
        from .loop_extents import cell_range
        from .selection import interval

        assignments = [n for n in self.walk(body) if n["kind"] == "AssignStmt"]
        counts = Counter(self.o.node(n["fields"]["var"])["fields"]["name"] for n in assignments)
        for assignment in assignments:
            var = self.o.node(assignment["fields"]["var"])
            name = var["fields"]["name"]
            if (counts[name] <= 1 and name not in self.env) or name in self.cells:
                continue
            producer = self.o.node(assignment["fields"]["value"])
            self.o.need(not (self.vf and producer["kind"] == "Call" and producer["fields"]["name"].startswith("vf.")),
                        assignment, "Repeated VF register/predicate bindings need explicit value-copy legalization")
            typ = self.o.node(var["fields"]["type"])
            self.o.need(typ["kind"] == "ScalarType", assignment, "Rebinding memory or aggregate descriptors in control flow is not admitted")
            scalar = self.o.scalar_type(typ)
            before = len(self.o.emitted)
            cell = self.emit("scalar.cell", assignment, typ=CellType(scalar.dtype))
            if name in self.env:
                initial = self.snapshot(self.env[name], assignment)
                self.emit("scalar.set", assignment, (cell, initial))
                cell_range(self, cell, interval(self, initial))
            self.cells[name] = cell
            self.env[name] = cell
            self.o.record(assignment, before)

    def snapshot(self, value, node, typ=None):
        if isinstance(value, Value) and isinstance(value.type, CellType):
            result = self.emit("scalar.cast", node, (value,), typ=ScalarType(value.type.dtype))
            bounds = self.cell_ranges.get(value.name)
            if bounds is not None:
                self.bounds[result.name] = bounds
                if bounds[0] >= 0:
                    self.nonnegative.add(result.name)
            return result
        if type(value) in (int, float, bool) and typ is not None:
            return self.emit("scalar.const", node, typ=typ, attrs={"value": value})
        return value

    def attrs(self, node, allowed=()):
        attrs = node["fields"]["kwargs"]
        self.o.need(not (set(attrs) - set(allowed)), node,
                    f"Unadmitted attributes for {node['fields']['name']}: {sorted(set(attrs) - set(allowed))}")
        return attrs

    def block(self, ref):
        node = self.o.node(ref)
        self.o.need(node["kind"] == "SeqStmts", node, "Expected a statement sequence")
        for statement in node["fields"]["stmts"]:
            self.statement(statement)

    def statement(self, ref):
        from .loop_extents import cell_range
        from .selection import interval

        node = self.o.node(ref)
        f, kind = node["fields"], node["kind"]
        if kind in {"IfStmt", "ForStmt"} and not self.depth and not self.vf and self.simt is None:
            self.hoist(node)
        before = len(self.o.emitted)
        if kind == "AssignStmt" and (node["id"] in self.o.hoisted or self.simt is not None and simt_declaration(self, f["value"])):
            pass
        elif kind == "AssignStmt":
            var = self.o.node(f["var"])
            self.whole = self.o.node(f["value"])["id"]
            value = self.eval(f["value"])
            name = var["fields"]["name"]
            if name in self.cells:
                typ = self.o.scalar_type(var["fields"]["type"])
                self.o.need(typ.dtype == self.cells[name].type.dtype, node, "Mutable scalar changes dtype")
                self.emit("scalar.set", node, (self.cells[name], value))
                cell_range(self, self.cells[name], interval(self, value))
                if isinstance(value, Value) and value.name in self.unsigned:
                    self.unsigned_cells.add(self.cells[name].name)
            else:
                expression = self.o.node(f["value"])
                register = (RegType, MaskType, UnalignRegType)
                if expression["kind"] == "Var" and isinstance(value, Value) and isinstance(value.type, register):
                    # Pro CCE prints `v = u` as a name alias (native A5 runs); an unaligned register alias is that register.
                    self.o.need(isinstance(value.type, UnalignRegType), node,
                                "VF register/predicate alias bindings need an explicit reference/value contract")
                self.env[name] = self.snapshot(value, node)
        elif kind == "EvalStmt":
            self.eval(f["expr"])
        elif kind == "IfStmt":
            from .descriptors import invalidate, updates

            self.o.need(not f["return_vars"], node, "If yields require SSA-merge legalization")
            cond = self.snapshot(self.eval(f["condition"]), node, ScalarType(dtype("b1")))
            branches, children = [], []
            changed = set()
            for branch in (f["then_body"], f["else_body"]):
                child = self.child()
                if branch:
                    changed.update(updates(child, branch))
                    child.block(branch)
                branches.append(Block(tuple(child.ops)))
                children.append(child)
            self.emit("cf.if", node, (cond,), regions=branches)
            self.written |= children[0].written & children[1].written
            for name in {n for body in (f["then_body"], f["else_body"]) if body for n in self.assigned(body)} & set(self.cells):
                ranges = [child.cell_ranges.get(self.cells[name].name) for child in children]
                cell_range(self, self.cells[name], None if None in ranges else (min(r[0] for r in ranges), max(r[1] for r in ranges)))
            from .slots import join_cursors
            join_cursors(self, children, (f["then_body"], f["else_body"]))
            invalidate(self, changed)
        elif kind == "ForStmt":
            from .descriptors import invalidate, updates
            from .loop_extents import induction_cells, loop

            self.o.need(not f["iter_args"] and not f["return_vars"] and not f["attrs"], node,
                        "Loop iter_args/attributes require explicit legalization")
            lo, hi, step = (self.eval(f[k]) for k in ("start", "stop", "step"))
            self.o.need(type(step) is int and step == 1, node, "Only unit-step counted loops are admitted")
            bound_names = {v["fields"]["name"] for k in ("start", "stop", "step")
                           for v in self.expr_nodes(f[k]) if v["kind"] == "Var"}
            writes = {self.o.node(n["fields"]["var"])["fields"]["name"] for n in self.walk(f["body"]) if n["kind"] == "AssignStmt"}
            self.o.need(not bound_names.intersection(writes), node, "Loop body mutates a loop bound")
            var = self.o.node(f["loop_var"])
            self.o.need(var["fields"]["name"] not in writes, node, "Reassignment of the native loop iterator is not admitted")
            iv = Value(self.o.unique("i"), self.o.scalar_type(var["fields"]["type"]))
            child = self.child()
            changed = updates(child, f['body'])
            invalidate(child, changed)
            child.env[var["fields"]["name"]] = iv
            if type(lo) is int and lo >= 0:
                child.nonnegative.add(iv.name)
            if type(lo) is int and type(hi) is int and lo < hi:
                limit = 1 << (iv.type.dtype.bits - 1)
                self.o.need(-limit <= lo < hi < limit, node, "Loop induction range exceeds its integer width")
                child.bounds[iv.name] = (lo, hi - 1)
            loop(child, iv, lo, hi)
            after = induction_cells(self, child, node, lo, hi)
            child.block(f["body"])
            invalidate(self, changed)
            for cell, bounds in after.items():
                cell_range(self, cell, bounds)
            loc = node["location"]
            op = self.o.rw.make("cf.for", (lo, hi, step), (iv,), {},
                                 regions=(Block(tuple(child.ops)),), loc=Loc.of(f"{loc['file']}:{loc['line']}:{loc['column']}"),
                                 note=f"Pro {node['id']} counted loop")
            self.ops.append(op)
            self.o.emitted.append(op.id)
        elif kind == "SectionStmt":
            self.o.need(f["section_kind"]["name"] == "VF" and not self.vf and self.simt is None, node,
                        "Unsupported nested/target section")
            self.outline(node)
        elif kind in {"BreakStmt", "ContinueStmt"}:
            self.o.need(self.depth > 0 and not f["value"], node, "Unsupported loop exit values")
            self.emit("cf.break" if kind == "BreakStmt" else "cf.continue", node)
        elif kind == "ReturnStmt":
            self.o.fail(node, "Source early return requires control-flow legalization")
        else:
            self.o.fail(node, f"Unsupported statement {kind}")
        self.o.record(node, before)

    def hoist(self, node):
        """Pro declares every tile in the kernel prologue, so a tile re-declared in control flow keeps one descriptor
        across trips and branches (alias_loop_probe on A5): bind those declarations before the statement."""
        for body in (node["fields"].get(k) for k in ("then_body", "else_body", "body")):
            for stmt in self.walk(body) if body else ():
                value = self.o.node(stmt["fields"]["value"]) if stmt["kind"] == "AssignStmt" else None
                if value and value["kind"] == "Call" and value["fields"]["name"] == "block.make_tile":
                    self.o.hoisting = True
                    try:
                        self.statement(stmt)
                    finally:
                        self.o.hoisting = False
                    self.o.hoisted.add(stmt["id"])

    def expr_nodes(self, ref, seen=None):
        seen = set() if seen is None else seen
        if not isinstance(ref, dict) or "$ref" not in ref or ref["$ref"] in seen:
            return
        seen.add(ref["$ref"])
        node = self.o.node(ref)
        yield node
        for key, value in node["fields"].items():
            if key in {"type", "kwargs", "attrs"}:
                continue
            for child in value if isinstance(value, list) else [value]:
                yield from self.expr_nodes(child, seen)

    def eval(self, ref):
        node = self.o.node(ref)
        before = len(self.o.emitted)
        result = self._eval(node)
        self.o.record(node, before)
        return result

    def _eval(self, node):
        from . import scalar_ops
        from .selection import binary_bounds, select

        f, kind = node["fields"], node["kind"]
        if kind.startswith("Const"):
            typ = self.o.scalar_type(f["type"])
            value = f["value"]
            if kind == "ConstFloat":
                try:
                    result = struct.unpack("<f", struct.pack("<f", value))[0]
                except OverflowError:
                    self.o.fail(node, "FP32 constant is outside the finite admitted range")
                self.o.need(struct.pack("<f", scalar_ops.printed_float(value)) == struct.pack("<f", result), node,
                            "Pro CCE prints this FP32 literal with six fractional digits, which changes its value")
                return result
            if kind == "ConstInt":
                bits = typ.dtype.bits
                low, high = (0, 1 << bits) if typ.dtype.kind == "uint" else (-(1 << (bits - 1)), 1 << (bits - 1))
                self.o.need(low <= value < high, node, "Integer literal is outside its declared width")
            return value
        if kind == "Var":
            self.o.need(f["name"] in self.env, node, f"Value {f['name']} is out of the admitted scope")
            return self.env[f["name"]]
        if kind == "MakeTuple":
            return tuple(self.eval(v) for v in f["elements"])
        if kind == "GetItemExpr":
            base, index = self.eval(f["value"]), self.eval(f["slice"])
            values = base.fields if isinstance(base, Struct) else base
            result = select(self, node, values, index)
            return self.snapshot(result, node) if isinstance(base, Struct) else result  # An advancing cursor is a cell.
        if kind == "Call":
            return self.call(node)
        binary ={"Add": "add", "Sub": "sub", "Mul": "mul", "FloorDiv": "div", "FloorMod": "mod", **scalar_ops.BINARY}
        compare = {"Eq": "eq", "Ne": "ne", "Lt": "lt", "Le": "le", "Gt": "gt", "Ge": "ge"}
        if kind in binary or kind in compare:
            a, b = self.eval(f["left"]), self.eval(f["right"])
            typ = self.o.scalar_type(f["type"])
            if kind in scalar_ops.BINARY:
                result = scalar_ops.binary(self, node, kind, a, b, typ)
                binary_bounds(self, kind, a, b, result)
                return unsigned_result(self, node, (a, b), result)
            if kind in {"FloorDiv", "FloorMod"}:
                safe_a = type(a) is int and a >= 0 or isinstance(a, Value) and a.name in self.nonnegative
                self.o.need(safe_a and type(b) is int and b > 0, node,
                            "Signed division/remainder requires a proven nonnegative dividend and positive divisor")
                if type(a) is int:
                    return a // b if kind == "FloorDiv" else a % b
            if kind in compare:
                unsigned_result(self, node, (a, b), None)
                return self.emit("scalar.cmp", node, (a, b), typ=typ, attrs={"pred": Ident(compare[kind])})
            result = self.emit("scalar." + binary[kind], node, (a, b), typ=typ)
            binary_bounds(self, kind, a, b, result)
            self.o.need(typ.dtype.name != "u32" or result.name in self.bounds, node,
                        "UINT32 arithmetic needs a proof that it stays inside [0, 2**32)")
            return unsigned_result(self, node, (a, b), result)
        if kind in scalar_ops.UNARY:
            operand = self.eval(f["operand"])
            self.o.need(kind == "Not" or getattr(operand, "name", None) not in self.unsigned | self.unsigned_cells, node, UNSIGNED)
            return scalar_ops.unary(self, node, kind, operand, self.o.scalar_type(f["type"]))
        if kind == "Cast":
            operand = self.eval(f["operand"])
            result = self.emit("scalar.cast", node, (operand,), typ=self.o.scalar_type(f["type"]))
            cast_bounds(self, operand, result)
            return result
        self.o.fail(node, f"Unsupported expression {kind}")
        return None

    def call(self, node):
        from .nz_parameters import guard as nz_operands
        from .selection import Choice, dispatch

        refs, name = node["fields"]["args"], node["fields"]["name"]
        if name == "block.store" and len(refs) == 4:
            from .nz_parameters import scale
            args = [*(self.eval(v) for v in refs[:3]), scale(self, node, refs[3])]
        else:
            args = [self.eval(v) for v in refs]
        nz_operands(self, node, name, args)
        if not self.vf and any(isinstance(arg, Memory) and arg.slot for arg in args):
            self.o.need(name in {"block.load", "block.store", "block.move"}, node,
                        "Slot-buffer tiles are admitted only as load, store, move and VF operands")
        if any(isinstance(arg, Choice) for arg in args):
            from .mx import routes as mx_routes
            if name == "block.move" and mx_routes(node, args):  # data and plane selections dispatch together (mx)
                return self.call_with_args(node, args)
            self.o.need(name in {"block.load", "block.store", "block.move", "block.matmul", "block.matmul_acc", "block.expands"},
                        node, "This operation needs a separate dynamic descriptor rule")
            dispatch(self, node, args, lambda child, values: child.call_with_args(node, values))
            return None
        return self.call_with_args(node, args)

    def call_with_args(self, node, args):
        from .cross import core_query, source_flag
        from .memory import memory_call
        from .mx import guard as mx_guard
        from .mx import routes as mx_routes
        from .sort import guard as index_guard
        from .synchronization import local_mutex
        from .vector import vector_call
        from .vector_spr import NEUTRAL, WRITES, stale, write

        name = node["fields"]["name"]
        index_guard(self, node, args)
        mx_guard(self, node, args)
        if self.simt is not None:
            from .simt import thread_call
            return thread_call(self, node, args)
        if name.startswith("vf."):
            self.o.need(self.vf, node, "Register operation outside a VF section")
            return vector_call(self, node, args)
        if self.vf and name == "block.subview":
            from .vector_memory import address
            return address(self, node, args)
        if name in WRITES:
            return write(self, node, args)
        self.o.need(not self.vf, node, "Device block/scalar-memory operation inside VF requires an effect rule")
        if name == "simt.launch" or name.startswith("block.") and name not in NEUTRAL:
            stale(self.o, "tile")  # Pro TileOp helpers and SIMT threads may rewrite the vector mask SPR.
        if name == "simt.launch":
            from .simt import launch
            return launch(self, node, args)
        if name in {'set_ctrl_spr', 'get_ctrl_spr', 'get_saturation_flag'}:
            return ctrl_call(self, node, args)
        if name.startswith('debug.'):
            from .debug import convert as debug_call
            return debug_call(self, node, args)
        if name.startswith('ptr.'):
            from .views import pointer_call
            return pointer_call(self, node, args)
        if name in {"get_subblock_idx", "get_block_idx", "get_block_num", "get_subblock_num"}:
            return core_query(self, node, args)
        if name == "system.dcci":
            from .cache import convert as cache_call
            return cache_call(self, node, args)
        if name.startswith(("system.set_cross_core", "system.wait_cross_core", "system.sync_src", "system.sync_dst")):
            return source_flag(self, node, args)
        if name in {"system.mutex_lock_dyn", "system.mutex_unlock_dyn"}:
            return local_mutex(self, node, args)
        if name == "struct.create":
            attrs = self.attrs(node, {"name", "fields"})
            self.o.need(attrs.get("name") == "_TileGroupCursor" and attrs.get("fields") == ["cursor"]
                        and len(args) == 1 and type(args[0]) is int, node, "Only static tile-group cursor structs are admitted")
            if node["id"] in self.o.slot_cursors:
                from .slots import cursor
                return cursor(self, node, args[0])
            return Struct(args)
        if name == "struct.set":
            from .slots import advance
            return advance(self, node, args)
        if name.startswith("system.bar_"):
            self.attrs(node)
            pipe = name.removeprefix("system.bar_").upper()
            self.o.need(pipe in {"ALL", "M", "MTE1", "MTE2", "MTE3", "FIX"} and not args, node, "Unadmitted barrier")
            # ALL is the declared scope default, not a physical issue queue.
            self.emit("sync.barrier", node, attrs={} if pipe == "ALL" else {"pipe": Ident(pipe)})
            return None
        if name in {"block.getval", "block.setval"}:
            from .views import scalar_address

            self.attrs(node)
            self.o.need(len(args) == (2 if name.endswith("getval") else 3) and isinstance(args[0], Memory)
                        and args[0].value.type.space in {"gm", "ub"} and (args[0].pitch is None or args[0].root is not None),
                        node, "Unadmitted scalar storage access")
            storage, index = scalar_address(self, node, args[0], args[1])
            if name.endswith("getval"):
                return self.emit("scalar.load", node, (storage, index), typ=self.o.scalar_type(node["fields"]["type"]))
            self.emit("scalar.store", node, (storage, index, args[2]))
            return None
        if name in {"block.sort32", "block.mrgsort", "block.mrgsort2"}:
            from .sort import convert as sort_call
            return sort_call(self, node, args)
        if mx_routes(node, args):
            from .mx import convert as mx_call
            return mx_call(self, node, args)
        return memory_call(self, node, args)

    def outline(self, node):
        from .selection import Choice, dispatch

        body = node["fields"]["body"]
        nodes = list(self.expr_nodes(body))
        self.o.need(not any(n["kind"] in {"ForStmt", "WhileStmt"} for n in nodes), node,
                    "VF loops require dedicated register-lifetime legalization")
        defined = {self.o.node(n["fields"]["var"])["fields"]["name"] for n in nodes if n["kind"] == "AssignStmt"}
        names = list(dict.fromkeys(n["fields"]["name"] for n in nodes if n["kind"] == "Var" and n["fields"]["name"] not in defined))
        choices = [name for name in names if isinstance(self.env.get(name), Choice)]
        if choices:
            def outline_choice(child, values):
                child.env.update(zip(choices, values, strict=True))
                child.outline(node)
            dispatch(self, node, [self.env[name] for name in choices], outline_choice)
            return
        child = Context(self.o, vf=True, top=self.depth == 0)
        params, args, captures = [], [], {}
        for name in names:
            self.o.need(name in self.env, node, f"Unknown VF capture {name}")
            actual = self.env[name]
            source_var = next(n for n in nodes if n["kind"] == "Var" and n["fields"]["name"] == name)
            capture_type = self.o.scalar_type(source_var["fields"]["type"]) if type(actual) in (int, float, bool) else None
            value = actual.value if isinstance(actual, Memory) else self.snapshot(actual, node, capture_type)
            self.o.need(isinstance(value, Value), node, "VF aggregate/constant capture requires legalization")
            param = Value(self.o.unique("p"), value.type)
            params.append(param)
            args.append(value)
            captures[param.name] = value
            child.env[name] = replace(actual, value=param) if isinstance(actual, Memory) else param
        child.prepare_cells(body)
        child.block(body)
        child.emit("cf.return", node)
        name = self.o.unique("pro_vf")
        self.o.functions.append(Function("vf", name, tuple(params), {}, Block(tuple(child.ops))))
        self.emit("cf.call", node, (FuncRef(name), *args), attrs={
            "read": list(dict.fromkeys(captures[v.name] for v in child.reads)),
            "write": list(dict.fromkeys(captures[v.name] for v in child.writes))})


def simt_declaration(ctx, ref):
    from .simt import declaration

    return declaration(ctx, ref)


def cast_bounds(ctx, operand, result):
    """An integer cast keeps a proven interval that its result type holds."""
    from .selection import interval

    bounds = interval(ctx, operand) if not isinstance(operand, Value) or isinstance(operand.type, ScalarType) else None
    source = operand.type.dtype if isinstance(operand, Value) else None
    target = result.type.dtype
    if bounds is None or (source is not None and source.kind not in {"int", "uint"}) or target.kind not in {"int", "uint"}:
        return
    lo, hi = (0, 1 << target.bits) if target.kind == "uint" else (-(1 << (target.bits - 1)), 1 << (target.bits - 1))
    if lo <= bounds[0] <= bounds[1] < hi:
        ctx.bounds[result.name] = bounds
        if bounds[0] >= 0:
            ctx.nonnegative.add(result.name)


def import_module(bundle: Bundle) -> Module:
    """Convert an admitted Pro graph to verified Lowered IR, or fail at source."""
    return Importer(bundle).run()
