# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Versioned, data-only Pro graph transport (distinct from Ascriptor IR).

Nodes have id/kind/fields/location/origin. Values are JSON primitives, ordered
lists, dictionaries, node references {$ref: id}, or tagged enum/dtype values.
Graphs include parser-owned group/mutex/tuple metadata. Physical addresses and
native allocation identity are separate; native process counters are not IDs.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

SCHEMA = "ascriptor.pypto-pro-export/1"
PROFILE = "cann-pro-fe20d726-export/2"
PROFILES = ("cann-pro-fe20d726-export/1", PROFILE)  # Import checks each identity against profile.json predecessors.
HEX = re.compile(r"[0-9a-f]{64}\Z")

FIELDS = {
    "Program": ("name", "functions"),
    "Function": ("name", "params", "func_type", "entry", "attrs", "return_types", "body"),
    "SeqStmts": ("stmts",), "AssignStmt": ("var", "value"), "EvalStmt": ("expr",),
    "IfStmt": ("condition", "then_body", "else_body", "return_vars"),
    "ForStmt": ("loop_var", "start", "stop", "step", "iter_args", "body", "return_vars", "attrs"),
    "WhileStmt": ("condition", "iter_args", "body", "return_vars"),
    "SectionStmt": ("section_kind", "body"),
    "YieldStmt": ("value",), "ReturnStmt": ("value",), "BreakStmt": ("value",), "ContinueStmt": ("value",),
    "Var": ("name", "type"), "IterArg": ("initValue", "iterVar"),
    "MemRef": ("memory_space", "addr", "size"),
    "Call": ("name", "args", "kwargs", "type"),
    "MakeTuple": ("elements", "type"), "GetItemExpr": ("value", "slice", "type"),
    "ConstInt": ("value", "type"), "ConstFloat": ("value", "type"), "ConstBool": ("value", "type"),
    "ScalarType": ("dtype",), "PtrType": ("dtype",), "TupleType": ("types",),
    "TensorType": ("dtype", "shape", "memref", "tensor_view"),
    "TileType": ("dtype", "shape", "memref", "tile_view", "hardware_info"),
    "TensorView": ("valid_shape", "stride", "layout", "ptr"),
    "TileView": ("valid_shape", "stride", "start_offset"),
    "HardwareInfo": ("blayout", "slayout", "fractal", "pad", "compact"),
    "NoneType": (), "UnknownType": (),
}
for _kind in ("Add Sub Mul FloorDiv FloorMod FloatDiv Min Max Pow Eq Ne Lt Le Gt Ge And Or Xor "
              "BitAnd BitOr BitXor BitShiftLeft BitShiftRight").split():
    FIELDS[_kind] = ("left", "right", "type")
for _kind in "Abs Neg Not BitNot Cast".split():
    FIELDS[_kind] = ("operand", "type")


class ProImportError(ValueError):
    """A version, representation or conversion gap with an optional source site."""

    def __init__(self, message: str, location: dict | None = None):
        self.location = location
        prefix = ""
        if location:
            prefix = f"{location['file']}:{location['line']}:{location['column']}: "
        super().__init__(prefix + message)


def require(condition: bool, message: str, location: dict | None = None) -> None:
    if not condition:
        raise ProImportError(message, location)


def _keys(value, keys, context):
    require(isinstance(value, dict) and set(value) == set(keys), f"Invalid {context} fields")


def _location(value):
    if value is None:
        return
    _keys(value, ("file", "line", "column"), "location")
    require(isinstance(value["file"], str) and bool(value["file"]), "Invalid location filename")
    require(all(type(value[k]) is int and value[k] > 0 for k in ("line", "column")), "Invalid location coordinates")


def _value(value, refs, location=None):
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        require(math.isfinite(value), "Non-finite transport value", location)
    elif isinstance(value, list):
        for item in value:
            _value(item, refs, location)
    elif isinstance(value, dict):
        require(all(isinstance(k, str) for k in value), "Non-string dictionary key", location)
        if "$ref" in value:
            require(set(value) == {"$ref"} and isinstance(value["$ref"], str)
                    and value["$ref"] in refs, f"Dangling or malformed reference {value.get('$ref')!r}", location)
        elif "$enum" in value:
            require(set(value) == {"$enum", "name", "value"} and isinstance(value["$enum"], str)
                    and isinstance(value["name"], str) and type(value["value"]) is int, "Invalid enum", location)
        elif "$dtype" in value:
            require(set(value) == {"$dtype", "code", "bits"} and isinstance(value["$dtype"], str)
                    and type(value["code"]) is int and type(value["bits"]) is int, "Invalid dtype", location)
        else:
            require(not any(k.startswith("$") for k in value), "Unknown transport tag", location)
            for item in value.values():
                _value(item, refs, location)
    else:
        raise ProImportError(f"Non-data transport value {type(value).__name__}", location)


def validate(document: dict) -> None:
    _keys(document, ("schema", "producer", "source", "target", "specialization", "abi", "targets"), "bundle")
    require(document["schema"] == SCHEMA, f"Unsupported export schema {document['schema']!r}")
    require(document["target"] == "a5", "Only the audited a5 export target is supported")
    _keys(document["producer"], ("profile", "fingerprint", "native_helper"), "producer")
    require(document["producer"]["profile"] in PROFILES, "Unsupported Pro producer profile")
    for key in ("fingerprint", "native_helper"):
        require(isinstance(document["producer"][key], str) and HEX.fullmatch(document["producer"][key]) is not None,
                f"Invalid producer {key}")
    _keys(document["source"], ("name", "sha256"), "source")
    require(isinstance(document["source"]["name"], str) and bool(document["source"]["name"]), "Missing source name")
    require(isinstance(document["source"]["sha256"], str) and HEX.fullmatch(document["source"]["sha256"]) is not None,
            "Invalid source hash")
    _keys(document["specialization"], ("tilingkey", "datatypes", "bound_signature", "pipeline", "pipeline_config", "auto_mutex"), "specialization")
    require(all(type(document["specialization"][k]) is bool for k in ("pipeline", "auto_mutex")), "Invalid source modes")
    config = document["specialization"]["pipeline_config"]
    require(document["specialization"]["pipeline"] == (config is not None), "Missing pipeline configuration")
    if config is not None:
        _keys(config, ("preload", "sync_only"), "pipeline configuration")
        require(type(config["preload"]) is int and type(config["sync_only"]) is bool, "Invalid pipeline configuration")
    _value(document["specialization"], set())
    _keys(document["abi"], ("block_dim", "directions", "workspace"), "ABI")
    dim = document["abi"]["block_dim"]
    require(dim is None or (type(dim) is int and dim > 0), "Invalid block_dim")
    require(isinstance(document["abi"]["directions"], dict)
            and all(isinstance(k, str) and v in ("input", "output", "inout", "unknown")
                    for k, v in document["abi"]["directions"].items()), "Invalid ABI directions")
    _value(document["abi"], set())
    workspace = document['abi']['workspace']
    require(isinstance(workspace, list), 'Workspace metadata must be a list')
    names = set()
    for entry in workspace:
        _keys(entry, ('parameter', 'shape', 'alignment', 'init'), 'workspace')
        name = entry['parameter']
        require(isinstance(name, str) and name in document['abi']['directions'] and name not in names,
                'Unknown or duplicate workspace parameter')
        names.add(name)
        require(type(entry['alignment']) is int and entry['alignment'] == 32, 'Workspace alignment must be 32')
        require(entry['init'] == 'uninitialized', 'Only uninitialized private workspace is admitted')
        require(isinstance(entry['shape'], list) and len(entry['shape']) == 2, 'Workspace shape must have two axes')
        for dim in entry['shape']:
            if type(dim) is int:
                require(0 < dim < 2**63, 'Workspace dimensions must be positive signed integers')
            else:
                _keys(dim, ('tensor', 'axis'), 'workspace dimension')
                require(isinstance(dim['tensor'], str) and type(dim['axis']) is int and dim['axis'] in (0, 1),
                        'Invalid workspace dimension reference')
    for entry in workspace:
        require(document['abi']['directions'][entry['parameter']] == 'inout', 'Workspace source direction must be inout')
        for dim in entry['shape']:
            require(type(dim) is int or dim['tensor'] in document['abi']['directions'] and dim['tensor'] not in names,
                    'Workspace shape must reference a public tensor')
    require(isinstance(document["targets"], list) and len(document["targets"]) == 2, "Expected cube and vec graphs")
    require([g.get("side") for g in document["targets"] if isinstance(g, dict)] == ["cube", "vec"], "Invalid target order")
    require(any(g.get("active") is True for g in document["targets"]), "No active target")
    for graph in document["targets"]:
        _keys(graph, ("side", "active", "root", "nodes", "metadata"), "graph")
        require(type(graph["active"]) is bool and isinstance(graph["nodes"], list), "Invalid graph")
        refs = {}
        for node in graph["nodes"]:
            _keys(node, ("id", "kind", "fields", "location", "origin"), "node")
            ident = node["id"]
            require(isinstance(ident, str) and ident.startswith(graph["side"] + "/n")
                    and ident not in refs, f"Invalid or duplicate node ID {ident!r}")
            refs[ident] = node
            _location(node["location"])
            require(isinstance(node["kind"], str) and node["kind"] in FIELDS,
                    f"Unsupported node kind {node['kind']!r}", node["location"])
            _keys(node["fields"], FIELDS[node["kind"]], node["kind"])
            require(node["origin"] in ("native", "enclosing", "none"), "Invalid provenance", node["location"])
            if node["kind"] in ("Call", "AssignStmt", "Function"):
                require(node["location"] is not None, "Missing operation provenance")
        require(isinstance(graph["root"], str) and graph["root"] in refs
                and refs[graph["root"]]["kind"] == "Program", "Invalid Program root")
        for node in graph["nodes"]:
            _value(node["fields"], refs, node["location"])
            _node_shape(node, refs)
        _keys(graph["metadata"], ("groups", "mutex", "tuples", "roles", "declarations", "allocations"), "metadata")
        require(all(isinstance(v, list) for v in graph["metadata"].values()), "Invalid metadata tables")
        _value(graph["metadata"], refs)
        _metadata(graph["metadata"], refs)


def _node_shape(node, nodes):
    fields, kind, loc = node["fields"], node["kind"], node["location"]
    def ref(value):
        require(isinstance(value, dict) and set(value) == {"$ref"}, "Expected a typed node reference", loc)
        return nodes[value["$ref"]]
    for key in ("functions", "params", "return_types", "stmts", "iter_args", "return_vars", "args",
                "elements", "shape", "types", "stride", "valid_shape"):
        if key in fields:
            require(isinstance(fields[key], list), f"Expected {kind}.{key} list", loc)
            for value in fields[key]:
                ref(value)
    for key in ("type", "var", "body", "condition", "then_body", "else_body", "loop_var", "start", "stop",
                "step", "expr", "left", "right", "operand", "slice", "initValue", "iterVar", "addr",
                "memref", "tile_view", "hardware_info", "tensor_view", "ptr", "start_offset"):
        if key in fields and fields[key] is not None:
            target = ref(fields[key])
            if key == "type":
                require(target["kind"].endswith("Type"), "Expression type refers to a non-type node", loc)
    for key in ("attrs", "kwargs"):
        if key in fields:
            require(isinstance(fields[key], dict), f"Expected {kind}.{key} dictionary", loc)
    if "name" in fields:
        require(isinstance(fields["name"], str), f"Invalid {kind} name", loc)
    if kind in ("AssignStmt", "GetItemExpr"):
        ref(fields["value"])
    if kind in ("YieldStmt", "ReturnStmt", "BreakStmt", "ContinueStmt"):
        require(isinstance(fields["value"], list), "Invalid control-flow values", loc)
        for value in fields["value"]:
            ref(value)
    if kind.startswith("Const"):
        expected = {"ConstInt": int, "ConstFloat": float, "ConstBool": bool}[kind]
        require(type(fields["value"]) is expected, "Invalid typed constant", loc)


def _metadata(tables, nodes):
    def ref(value, kinds=None):
        require(isinstance(value, dict) and set(value) == {"$ref"} and value["$ref"] in nodes,
                "Metadata needs a node reference")
        node = nodes[value["$ref"]]
        require(kinds is None or node["kind"] in kinds, "Metadata reference has the wrong node kind")
        return node
    for row in tables["groups"]:
        _keys(row, ("value", "depth", "mutex_ids", "space"), "group")
        ref(row["value"])
        require(type(row["depth"]) is int and row["depth"] > 0, "Invalid group depth")
        ids = row["mutex_ids"]
        if ids is not None:
            require(isinstance(ids, list) and len(ids) == row["depth"], "Mutex slots disagree with group depth")
            require(all(isinstance(slot, list) and slot and all(type(i) is int and 0 <= i < 32 for i in slot)
                        and len(set(slot)) == len(slot) for slot in ids), "Invalid group mutex IDs")
            require(len({len(slot) for slot in ids}) == 1, "Inconsistent per-slot mutex width")
        require(isinstance(row["space"], dict) and row["space"].get("$enum") == "MemorySpace", "Invalid group space")
    for row in tables["mutex"]:
        _keys(row, ("value", "ids", "candidates"), "mutex metadata")
        ref(row["value"])
        require(isinstance(row["ids"], list) and row["ids"], "Missing mutex expressions")
        for value in row["ids"]:
            ref(value)
        require(isinstance(row["candidates"], list)
                and all(type(i) is int and 0 <= i < 32 for i in row["candidates"]), "Invalid mutex candidates")
    for row in tables["tuples"]:
        _keys(row, ("type", "name", "fields"), "tuple metadata")
        node = ref(row["type"], {"TupleType"})
        require(row["name"] is None or isinstance(row["name"], str), "Invalid tuple name")
        require(row["fields"] is None or (isinstance(row["fields"], list)
                and all(isinstance(s, str) for s in row["fields"])
                and len(row["fields"]) == len(node["fields"]["types"])), "Invalid tuple fields")
    for row in tables["roles"]:
        _keys(row, ("value", "role"), "value role")
        ref(row["value"], {"Var"})
        require(row["role"] in ("register", "predicate"), "Invalid value role")
    for row in tables["declarations"]:
        _keys(row, ("key", "value", "location"), "declaration")
        ref(row["value"], {"Var"})
        _location(row["location"])
        require(row["location"] is not None and isinstance(row["key"], str)
                and HEX.fullmatch(row["key"]) is not None, "Invalid declaration identity")
    allocation_ids = set()
    for row in tables["allocations"]:
        _keys(row, ("value", "same_allocation"), "allocation identity")
        node = ref(row["value"], {"MemRef"})
        require(node["id"] not in allocation_ids, "Duplicate allocation metadata")
        allocation_ids.add(node["id"])
        require(isinstance(row["same_allocation"], list) and row["value"] in row["same_allocation"],
                "Allocation identity must include itself")
        for value in row["same_allocation"]:
            ref(value, {"MemRef"})
    require(allocation_ids == {k for k, v in nodes.items() if v["kind"] == "MemRef"}, "Incomplete allocation identities")


@dataclass(frozen=True)
class Bundle:
    """Validated canonical JSON; callers receive fresh copies of the document."""

    _json: str

    @classmethod
    def from_dict(cls, document: dict) -> Bundle:
        validate(document)
        return cls(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False))

    @property
    def document(self) -> dict[str, Any]:
        return json.loads(self._json)


def dumps(bundle: Bundle) -> str:
    validate(bundle.document)
    return bundle._json + "\n"


def loads(text: str) -> Bundle:
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, f"Duplicate JSON key {k!r}")
            result[k] = v
        return result
    try:
        document = json.loads(text, object_pairs_hook=pairs)
    except (ValueError, TypeError) as exc:
        raise ProImportError(f"Invalid export JSON: {exc}") from exc
    return Bundle.from_dict(document)
