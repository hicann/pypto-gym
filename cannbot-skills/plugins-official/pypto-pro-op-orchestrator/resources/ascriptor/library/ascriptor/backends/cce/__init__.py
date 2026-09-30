# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The cce backend (a5 / c310 and a2 / c220): Lowered IR -> CCE source through the ``tensorutils_cce.h`` wrapper layer.

``compile(module)`` returns the kernel entry ``<kernel>.cpp``, one header per side (``<kernel>_cube.h``,
``<kernel>_vec.h``), one header per ``@vf`` / ``@simt`` function, the support header and a
``manifest.json`` the runtime's project generator reads. Ops the printer has no line for raise
:class:`CceGap` with their source location (RFC-0007).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..base import Artifacts, Capabilities, ResourceLimits
from . import cpp
from .arch import c310
from .emit import HEADER, CceGap, FnPrinter, ModulePrinter, SimtPrinter, VfPrinter, emit_module

DEVICES = frozenset({"950", "950pr", "a5", "a5pr", "b1", "b2", "b3", "b4", "a2", "a3"})


def _handled(cls: type) -> set[str]:
    return {n[3:].replace("_", ".") for n in dir(cls) if n.startswith("op_")}


def _opcodes() -> frozenset[str]:
    from ...ir import REGISTRY

    known = {s.name for s in REGISTRY.all()} if hasattr(REGISTRY, "all") else set()
    handled = _handled(FnPrinter) | _handled(VfPrinter) | _handled(SimtPrinter)
    # method names flatten dots and underscores alike; resolve against the registry when it is available
    if known:
        out = set()
        for name in known:
            if name.replace(".", "_") in {h.replace(".", "_") for h in handled}:
                out.add(name)
        return frozenset(out)
    return frozenset(handled)


class CceBackend:
    name = "cce"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            function_kinds=frozenset({"kernel", "vf", "simt"}),
            opcodes=_opcodes(),
            dtypes=frozenset(cpp.CTYPE),
            devices=DEVICES,
            notes={
                "vf.log2 / vf.log10": "multi-instruction sequences on c310 (vln + vmuls); not a single intrinsic",
                "64-bit vf arithmetic": "vadd/vsub/vand/... have no two-register form; only the ops the compiler provides",
                "bf16 scalar math outside ordinary functions": "ordinary A5 BF16 conversions/abs/sqrt are lowered explicitly; BF16 VF/SIMT scalar math is not newly qualified",
            },
        )

    def resources(self, device: Any) -> ResourceLimits:
        from ... import devices as _devices

        p = device if hasattr(device, "capacities_kb") else _devices.load(str(device))
        kb = p.capacities_kb
        return ResourceLimits(ub=kb.get("ub", 0) * 1024, l1=kb.get("l1", 0) * 1024, l0a=kb.get("l0a", 0) * 1024,
                              l0b=kb.get("l0b", 0) * 1024, l0c=kb.get("l0c", 0) * 1024, bt=kb.get("bt", 0) * 1024)

    def compile(self, module: Any, options: Mapping[str, Any] | None = None) -> Artifacts:
        from ...runtime.launch_config import exported_artifacts

        options = dict(options or {})
        return exported_artifacts(module, "the CCE backend", options.get("block_dim"),
                                  lambda block_dim: emit_module(module, block_dim=block_dim, entry=options.get("entry")))


__all__ = ["CceBackend", "CceGap", "ModulePrinter", "emit_module", "HEADER", "c310", "DEVICES"]
