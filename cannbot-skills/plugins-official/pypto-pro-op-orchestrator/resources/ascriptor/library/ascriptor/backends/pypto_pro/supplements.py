# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Local PyPTO-Pro source supplements a printed kernel depends on.

Ascriptor installs none of them. The patches under ``docs/patches`` exist so the
dependency can stay a stock checkout until a developer decides otherwise - every
applied patch is one more difference between the installed package and upstream,
and keeping that difference OPTIONAL is the point. What the printer owes in return
is to say, at the line it prints, which patch that line needs; otherwise the cost
lands on PyPTO's own parser error, several layers away from the cause.

`docs/pypto-pro-supplements.md` owns the patches, their evidence and how to apply
one. This module owns only the emit-side question: which printed form needs which
patch, and the probe the board-side driver runs against the INSTALLED package -
the one place that can answer whether it is actually there (this workstation has no
`pypto_pro` at all, so emit cannot probe and does not pretend to).
"""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ...ir import Op

ENV = "ASCRIPTOR_PYPTO_SUPPLEMENTS"
DOC = "docs/pypto-pro-supplements.md"


class PyptoSupplementWarning(UserWarning):
    """Warned at emit: the printed kernel needs a patch that is not installed by default."""


@dataclass(frozen=True)
class Supplement:
    """One patch under ``docs/patches``, seen from the printer."""

    id: str
    patch: str             # repo-relative path of the source patch
    section: str           # its section of DOC, which owns everything else about it
    enables: str           # the printed form that needs it, in one line
    probe: dict[str, Any]  # ``{"all": [condition, ...]}`` the driver asks the installed package

    @property
    def doc(self) -> str:
        return DOC + self.section


# Only what the PRINTER can need. The export observer belongs to the reverse direction
# (`importers/pypto_pro/export.py` already refuses without it) and the float-immediate
# candidate is one Ascriptor does not need at all - `_imm_survives` hoists an exact scalar
# parameter instead. DOC carries those; this table stays the printer's own list.
SUPPLEMENTS: dict[str, Supplement] = {s.id: s for s in (
    Supplement(
        id="integer-cast",
        patch="docs/patches/pypto-pro-integer-cast.patch",
        section="#1-integer-scalar-cast",
        enables="pl.cast(value, dtype), the integer scalar overload the VF div/mod widths need (M10-090)",
        # Two conditions, because the patch has two halves and half of it is worse than none:
        # the patched `_ir_cast` dispatches Scalar+DataType and therefore cannot keep a rounding
        # mode as its default (the stock one defaults `mode` to RoundMode.CAST_ROUND), and the
        # parser's vector-function whitelist must admit the scalar form (upstream e70d321dd
        # refuses every other `pl.*` inside a vf body).
        probe={"all": [
            {"module": "pypto_pro.ir.op.block_ops", "attr": "_ir_cast",
             "kind": "default_is_none", "parameter": "mode"},
            {"module": "pypto_pro.language.parser._call_parser", "attr": "CallParserMixin",
             "kind": "attr_contains", "attribute": "_VF_SCALAR_PL_OPS", "needles": ["cast"]},
        ]},
    ),
)}

#: Supplements upstream has since adopted, kept only so an environment that still names one in
#: ``ENV`` is accepted instead of refused. They are not printed, warned about or recorded.
RETIRED: dict[str, str] = {"mat-fill-mutex": "upstream 6a652e733 (2026-09-18) selects the MTE2 pipe itself"}


def declared(explicit: Iterable[str] | str | None = None) -> frozenset[str]:
    """The supplements the caller states the target installation already carries.

    An explicit argument REPLACES the environment rather than adding to it, so a caller that
    passes one is not silently widened by a shell variable left over from another task. An
    unknown name raises: a typo that kept warning would quietly defeat the switch it was
    meant to operate.
    """
    raw = os.environ.get(ENV, "") if explicit is None else explicit
    names = [n for n in re.split(r"[,\s]+", raw) if n] if isinstance(raw, str) else [str(n) for n in raw]
    names = [n for n in names if n not in RETIRED]
    unknown = sorted(set(names) - set(SUPPLEMENTS))
    if unknown:
        raise ValueError(f"unknown PyPTO-Pro supplement(s) {', '.join(unknown)}; "
                         f"known: {', '.join(sorted(SUPPLEMENTS))} ({DOC})")
    return frozenset(names)


class Uses:
    """What the module being printed needs, recorded as its lines are printed."""

    SITES = 3  # enough first sites to find the cause, short enough for one warning line

    def __init__(self, declared_ids: Iterable[str] = ()) -> None:
        self.declared = frozenset(declared_ids)
        self._sites: dict[str, list[str]] = {}

    def need(self, supplement_id: str, op: Op | None = None, note: str = "") -> None:
        """Record that the line being printed needs ``supplement_id``, and where."""
        if supplement_id not in SUPPLEMENTS:
            raise ValueError(f"unknown PyPTO supplement: {supplement_id}")
        sites = self._sites.setdefault(supplement_id, [])
        where = getattr(op, "opcode", "")
        loc = getattr(op, "loc", None)
        if where and loc:
            where += f" at {loc}"
        where = ("; ".join(x for x in (where, note) if x)) or ""
        if where and where not in sites and len(sites) < self.SITES:
            sites.append(where)

    def __bool__(self) -> bool:
        return bool(self._sites)

    def entries(self) -> list[dict[str, Any]]:
        """The manifest record: what to apply, whether the caller declared it, and the probe."""
        out = []
        for sid in sorted(self._sites):
            s = SUPPLEMENTS[sid]
            out.append({"id": s.id, "patch": s.patch, "doc": s.doc, "enables": s.enables,
                        "declared": sid in self.declared, "sites": list(self._sites[sid]),
                        "probe": dict(s.probe)})
        return out

    def header(self) -> list[str]:
        """The comment block the generated module carries, so the source itself says what it needs."""
        if not self._sites:
            return []
        lines = [f"# Local PyPTO-Pro supplements this source needs - NOT installed by default ({DOC}):"]
        for sid in sorted(self._sites):
            s = SUPPLEMENTS[sid]
            state = "declared installed" if sid in self.declared else "apply before running"
            lines.append(f"#   {s.id} [{state}]: {s.enables}")
            lines.append(f"#     {s.patch}")
        return lines

    def warn(self, kernel: str, stacklevel: int = 3) -> None:
        """One warning per undeclared supplement. A declared one is silent but stays in the manifest."""
        for sid in sorted(self._sites):
            if sid in self.declared:
                continue
            s = SUPPLEMENTS[sid]
            sites = ", ".join(self._sites[sid])
            warnings.warn(
                f"[pypto-pro supplement] {kernel} needs {s.id}: {s.enables}. Ascriptor does not install "
                f"it - apply {s.patch} to the pypto_pro that will run this kernel ({s.doc}), or set "
                f"{ENV}={s.id} when that installation already carries it"
                + (f". First needed by {sites}" if sites else ""),
                PyptoSupplementWarning, stacklevel=stacklevel)


# Spliced into the generated `run_case.py`, which runs where `pypto_pro` actually is. Emit knows
# what the source needs and the box knows what is installed, so the two halves meet in the
# manifest: the entries carry their own probe and this reads them.
DRIVER_PREFLIGHT = '''
def supplement_preflight(spec):
    """Report every recorded supplement the installed pypto_pro does not carry.

    Answers present / absent / unknown and never guesses: an installation this cannot read
    is reported unknown and the run goes on, because a probe that cannot see is not evidence
    of absence. ASCRIPTOR_PYPTO_SUPPLEMENTS declares one present and skips its probe - the
    same switch the emitter reads.
    """
    import importlib
    import inspect

    said = set(os.environ.get("ASCRIPTOR_PYPTO_SUPPLEMENTS", "").replace(",", " ").split())
    missing = []
    for entry in spec.get("supplements", []):
        if entry["id"] in said or entry.get("declared"):
            continue
        conditions = (entry.get("probe") or {}).get("all") or []
        note = ""
        found = bool(conditions) or None
        for probe in conditions:
            try:
                obj = importlib.import_module(probe["module"])
                for part in probe["attr"].split("."):
                    obj = getattr(obj, part)
                if probe["kind"] == "default_is_none":
                    here = inspect.signature(obj).parameters[probe["parameter"]].default is None
                elif probe["kind"] == "attr_contains":
                    here = all(n in getattr(obj, probe["attribute"]) for n in probe["needles"])
                else:
                    here = None
            except Exception as exc:  # noqa: BLE001 - unreadable is unknown, not absent
                here, note = None, f"{type(exc).__name__}: {exc}"
            if here is None:
                found = None
                break
            found = found and here
        if found is None:
            print(f"PYPTO_SUPPLEMENT_UNKNOWN {entry['id']}: {note or 'no probe for this entry'}")
        elif not found:
            missing.append(entry)
            print(f"MISSING PyPTO-Pro supplement: {entry['id']}")
            print(f"  this generated kernel needs {entry['enables']}")
            print(f"  apply {entry['patch']} to this installation ({entry['doc']}),")
            print(f"  or set ASCRIPTOR_PYPTO_SUPPLEMENTS={entry['id']} if it already carries it")
    return missing
'''

__all__ = ["DOC", "DRIVER_PREFLIGHT", "ENV", "PyptoSupplementWarning", "RETIRED", "SUPPLEMENTS",
           "Supplement", "Uses", "declared"]
