# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The pass manager: a fixed list of ``Module -> Module`` passes with the verifier between them.

A :class:`Pass` names the IR level it accepts and produces and receives a :class:`PassContext`
carrying the device profile, the options and an :class:`Explain` sink. Every op a pass inserts or
rewrites goes through :class:`~ascriptor.ir.Rewriter` so it carries an ``origin`` entry (RFC-0001
§8); every decision worth showing to a person (a hazard edge, an address, an event channel) is
recorded with :meth:`Explain.note` and printed by ``ascriptor explain`` / ``dump-ir --explain``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..devices import load as profile_for
from ..ir import LOWERED, SURFACE, Module, VerifyError, check


class PassError(RuntimeError):
    """A pass could not lower the module; the message names the op and the source position."""

    def __init__(self, pass_name: str, message: str) -> None:
        super().__init__(f"{pass_name}: {message}")
        self.pass_name = pass_name


@dataclass
class Explain:
    """Decisions of one pass, keyed by op id where there is one (``ascriptor explain --op``)."""

    pass_name: str
    entries: list[dict[str, Any]] = field(default_factory=list)

    def note(self, message: str, *, op: int | None = None, **data: Any) -> None:
        entry: dict[str, Any] = {"pass": self.pass_name, "message": message}
        if op is not None:
            entry["op"] = op
        entry.update(data)
        self.entries.append(entry)

    def for_op(self, op_id: int) -> list[dict[str, Any]]:
        return [e for e in self.entries if e.get("op") == op_id or op_id in e.get("ops", ())]


@dataclass
class PassContext:
    device: Any  # the device profile (ascriptor.devices.profile_for)
    options: Mapping[str, Any]
    explain: Explain

    def option(self, name: str, default: Any = None) -> Any:
        return self.options.get(name, default)


@dataclass(frozen=True)
class Pass:
    name: str
    run: Callable[[Module, PassContext], Module]
    accepts: str = SURFACE  # IR version the pass reads
    produces: str = SURFACE  # IR version it writes
    doc: str = ""
    establishes: tuple[str, ...] = ()  # Lowered invariants (RFC-0001 §10) that hold after the pass


@dataclass
class PassRun:
    name: str
    seconds: float
    module: Module
    explain: Explain


class PassManager:
    """Run passes in order, verifying the module before the first and after every pass."""

    def __init__(self, passes: Iterable[Pass], *, options: Mapping[str, Any] | None = None, verify_between: bool = True) -> None:
        self.passes = list(passes)
        self.options = dict(options or {})
        self.verify_between = verify_between
        self.runs: list[PassRun] = []

    def run(self, module: Module, *, stop_after: str | None = None) -> Module:
        device = profile_for(module.device or "950")
        if self.verify_between:
            _verify(module, "input")
        current = module
        self.runs = []
        for p in self.passes:
            if current.ir != p.accepts:
                raise PassError(p.name, f"expects {p.accepts} IR, the module is {current.ir}")
            explain = Explain(p.name)
            ctx = PassContext(device, self.options, explain)
            t0 = time.perf_counter()
            current = p.run(current, ctx)
            seconds = time.perf_counter() - t0
            if current.ir != p.produces:
                raise PassError(p.name, f"produced {current.ir} IR, declared {p.produces}")
            if self.verify_between:
                _verify(current, p.name)
            self.runs.append(PassRun(p.name, seconds, current, explain))
            if stop_after == p.name:
                break
        return current

    def after(self, name: str) -> Module:
        """The module as it was after the named pass (``dump-ir --after``)."""
        for r in self.runs:
            if r.name == name:
                return r.module
        raise KeyError(name)

    def explanations(self, op_id: int | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in self.runs:
            out.extend(r.explain.entries if op_id is None else r.explain.for_op(op_id))
        return out


def _verify(module: Module, stage: str) -> None:
    try:
        check(module)
    except VerifyError as exc:
        raise PassError(stage, f"the module does not verify after {stage}:\n{exc}") from exc


__all__ = ["Explain", "LOWERED", "Pass", "PassContext", "PassError", "PassManager", "PassRun", "SURFACE"]
