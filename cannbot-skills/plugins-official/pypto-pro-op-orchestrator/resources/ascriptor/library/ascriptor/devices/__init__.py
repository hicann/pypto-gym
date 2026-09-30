# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device profiles: the silicon facts every layer reads, as data.

A profile is a JSON file under ``profiles/``; nothing in the package hard-codes a core count
or a capacity. Facade aliases map the public names (``a2``, ``a3``, ``a5``, ``a5pr``) to
device types. The public facades ``ascriptor.a2`` / ``.a3`` / ``.a5`` / ``.a5pr`` (M2) only
bind names; importing one never mutates process-wide state.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from types import MappingProxyType

#: A c310 (a5) vector core keeps the top of its 256 KiB UB for SIMT launches: a kernel that launches
#: SIMT code may place its own data below this many KiB only. The c220 family's whole UB is 192 KiB,
#: so the bound never binds there. It is a silicon fact that still lives outside the profiles.
SIMT_UB_CAP_KB = 216

FACADE_ALIASES: Mapping[str, str] = MappingProxyType(
    {"a2": "b3", "a3": "a3", "a5": "950", "a5pr": "950pr"}
)


@dataclass(frozen=True)
class DeviceProfile:
    device_type: str
    family: str  # "a2" | "a5"
    arch: str  # "c220" | "c310" — the CCE intrinsic table to use
    npu_arch: str  # bisheng --npu-arch
    compile_unit: str
    debug_chipset: str
    cube_cores: int
    vec_cores: int
    capacities_kb: Mapping[str, float]  # l1, l0a, l0b, l0amx, l0bmx, l0c, bt, ub
    crosscore_id_max: int = 7  # logical IDs, before the mode-4 AIV1 encoding offset
    crosscore_counter_max: int = 15

    def capacity_bytes(self, position: str) -> int:
        return int(self.capacities_kb[position] * 1024)


def _profile_files() -> dict[str, str]:
    root = resources.files(__package__) / "profiles"
    return {p.name[: -len(".json")]: p.read_text(encoding="utf-8") for p in root.iterdir() if p.name.endswith(".json")}


def available() -> list[str]:
    return sorted(_profile_files())


def load(name: str) -> DeviceProfile:
    """Load a profile by device type (``950``) or facade alias (``a5``)."""
    key = FACADE_ALIASES.get(name, name).lower()
    files = _profile_files()
    if key not in files:
        raise KeyError(f"unknown device {name!r}; known: {sorted(files)} (aliases: {dict(FACADE_ALIASES)})")
    raw = json.loads(files[key])
    raw["capacities_kb"] = MappingProxyType(dict(raw["capacities_kb"]))
    return DeviceProfile(**raw)


__all__ = ["DeviceProfile", "FACADE_ALIASES", "SIMT_UB_CAP_KB", "available", "load"]
