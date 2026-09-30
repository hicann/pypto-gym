# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Source graph indexing and conversion accounting; no target IR is fabricated."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .schema import Bundle, ProImportError, require, validate


@dataclass(frozen=True)
class SourceOperation:
    id: str
    opcode: str
    side: str
    location: dict


@dataclass(frozen=True)
class ImportPlan:
    bundle: Bundle
    operations: tuple[SourceOperation, ...]

    def require_converters(self, opcodes) -> None:
        """Reject any unhandled source operation before target construction."""
        for op in self.operations:
            if op.opcode not in opcodes:
                raise ProImportError(f"No instruction conversion for {op.opcode} ({op.id})", op.location)


def _digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def profile() -> dict:
    """The pinned profile. A predecessor must be this profile without its added attributes: export never reads
    the attribute whitelist, so a predecessor export has the same content under another identity (RFC-0015)."""
    result = json.loads(Path(__file__).with_name("profile.json").read_text())
    payload = {k: v for k, v in result.items() if k != "fingerprint"}
    require(result.get("fingerprint") == _digest(payload), "Corrupt Pro compatibility profile")
    base = {k: v for k, v in payload.items() if k != "predecessors"}
    for old in payload.get("predecessors", []):
        attributes = {name: list(values) for name, values in base["op_attributes"].items()}
        for name, added in old["added_attributes"].items():
            require(bool(added) and set(added) <= set(attributes.get(name, ())), "Corrupt Pro compatibility profile")
            attributes[name] = [value for value in attributes[name] if value not in added]
        require(old["fingerprint"] == _digest({**base, "profile": old["profile"], "op_attributes": attributes}),
                "Corrupt Pro compatibility profile")
    return result


def accepted(producer: dict, selected: dict) -> bool:
    """Whether a producer names the selected profile or one of its verified predecessors."""
    identities = [selected, *selected.get("predecessors", [])]
    return any(producer["profile"] == p["profile"] and producer["fingerprint"] == p["fingerprint"] for p in identities)


def same_export(fresh: dict, recorded: dict) -> bool:
    """Whether a fresh export reproduces a recorded one, whose producer may be an accepted predecessor."""
    selected = profile()
    return (fresh["producer"]["profile"] == selected["profile"] and fresh["producer"]["fingerprint"] == selected["fingerprint"]
            and accepted(recorded["producer"], selected)
            and recorded["producer"]["native_helper"] == fresh["producer"]["native_helper"]
            and {k: v for k, v in fresh.items() if k != "producer"} == {k: v for k, v in recorded.items() if k != "producer"})


def prepare_import(bundle: Bundle) -> ImportPlan:
    """Validate compatibility and build an ordered ledger for subsequent rules."""
    document = bundle.document
    validate(document)
    selected = profile()
    require(accepted(document["producer"], selected), "Producer fingerprint mismatch")
    require(document["producer"]["native_helper"] == selected["helper_sha256"], "Native helper fingerprint mismatch")
    operations = []
    for graph in document["targets"]:
        if not graph["active"]:
            continue
        functions = {n["fields"]["name"] for n in graph["nodes"] if n["kind"] == "Function"}
        for node in graph["nodes"]:
            if node["kind"] != "Call":
                continue
            name = node["fields"]["name"]
            require(isinstance(name, str), "Invalid opcode", node["location"])
            require(name in selected["op_attributes"] or name in functions,
                    f"Unsupported Pro operation {name!r}", node["location"])
            if name in selected["op_attributes"]:
                attrs = node["fields"]["kwargs"]
                require(isinstance(attrs, dict), "Invalid operation attributes", node["location"])
                unknown = set(attrs) - set(selected["op_attributes"][name])
                require(not unknown, f"Unsupported attributes for {name}: {sorted(unknown)}", node["location"])
            operations.append(SourceOperation(node["id"], name, graph["side"], node["location"]))
    return ImportPlan(bundle, tuple(operations))
