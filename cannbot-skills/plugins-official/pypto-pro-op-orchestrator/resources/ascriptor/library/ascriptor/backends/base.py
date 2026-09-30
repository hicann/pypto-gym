# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The backend protocol and entry-point discovery.

A backend is a pure function from a lowered module to artifacts. It declares what it can
express (``capabilities``) so the compiler can report gaps with source locations instead of
guessing, and how much on-chip memory it can hand to a kernel on a given device
(``resources`` — a framework may reserve part of the silicon for its own runtime).

Two conformance relations hold for every backend (blueprint §4.6):

* semantics: running its artifacts through a launcher gives the same outputs as ``sim``;
* artifacts (wrapper-level backends only): the intrinsics bisheng derives from its output
  match the intrinsics the ``cce`` backend prints for the same module in manual-sync
  mode. An explicitly selected native synchronization mode may delegate ordering to
  the target compiler under its documented retained-protocol and validation
  requirements; output semantics remain unchanged (RFC-0013).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

ENTRY_POINT_GROUP = "ascriptor.backends"
BUILTIN = {
    "cce": "ascriptor.backends.cce:CceBackend",
    "pypto_pro": "ascriptor.backends.pypto_pro:PyptoProBackend",
    "pto_isa": "ascriptor.backends.pto_isa:PtoIsaBackend",
}


@dataclass(frozen=True)
class ResourceLimits:
    """On-chip capacity, in bytes, a backend can actually hand to a kernel on a device."""

    ub: int
    l1: int = 0
    l0a: int = 0
    l0b: int = 0
    l0c: int = 0
    bt: int = 0


@dataclass(frozen=True)
class Capabilities:
    """What a backend can express. Anything outside is reported as a gap, never guessed."""

    function_kinds: frozenset[str] = frozenset({"kernel"})  # subset of {"kernel", "vf", "simt"}
    opcodes: frozenset[str] = frozenset()  # e.g. {"cube.mmad", "dma.gm_to_l1.nd2nz", ...}
    dtypes: frozenset[str] = frozenset()
    devices: frozenset[str] = frozenset()  # device_type names, e.g. {"950", "950pr"}
    notes: Mapping[str, str] = field(default_factory=dict)  # opcode -> why it is unsupported


@dataclass(frozen=True)
class Artifacts:
    """Files a backend produced, keyed by relative path. Writing them to disk is the runtime's job."""

    files: Mapping[str, bytes]
    entry: str  # relative path of the primary artifact (kernel source or module)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    name: str

    def capabilities(self) -> Capabilities: ...

    def resources(self, device: Any) -> ResourceLimits: ...

    def compile(self, module: Any, options: Mapping[str, Any] | None = None) -> Artifacts: ...


def discover() -> dict[str, Any]:
    """Return every backend registered in the ``ascriptor.backends`` entry-point group.

    Loading is lazy per backend: a broken or heavy backend package must not prevent the
    others from being listed, so failures are recorded under ``notes`` instead of raised.
    """
    found: dict[str, Any] = {}
    for name, target in BUILTIN.items():
        module, symbol = target.split(":")
        try:
            found[name] = getattr(import_module(module), symbol)()
        except Exception as exc:  # noqa: BLE001 - reported at the chosen backend
            found[name] = _BrokenBackend(name, exc)
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name in found:
            continue
        try:
            obj = ep.load()
            backend = obj() if isinstance(obj, type) else obj
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            found[ep.name] = _BrokenBackend(ep.name, exc)
            continue
        found[ep.name] = backend
    return found


def get(name: str) -> Any:
    backends = discover()
    if name not in backends:
        raise KeyError(f"unknown backend {name!r}; registered: {sorted(backends)}")
    backend = backends[name]
    if isinstance(backend, _BrokenBackend):
        raise RuntimeError(f"backend {name!r} failed to load: {backend.error!r}")
    return backend


class _BrokenBackend:
    def __init__(self, name: str, error: BaseException) -> None:
        self.name = name
        self.error = error

    def capabilities(self) -> Capabilities:
        return Capabilities(notes={"*": f"failed to load: {self.error!r}"})
