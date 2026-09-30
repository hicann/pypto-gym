# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shorthands for declaring ops. Every table module uses only these names."""

from __future__ import annotations

from typing import Any

from ..registry import REGISTRY, AttrSpec, OperandSpec, OpSpec, ResultSpec

KERNEL = frozenset({"kernel"})
VF = frozenset({"vf"})
SIMT = frozenset({"simt"})
KV = frozenset({"kernel", "vf"})
KS = frozenset({"kernel", "simt"})
ALL = frozenset({"kernel", "vf", "simt"})

A5 = frozenset({"950", "950pr"})
A2 = frozenset({"b1", "b2", "b3", "b4", "a3"})
ALL_DEVICES = None


def R(name: str, pattern: str, doc: str = "") -> OperandSpec:
    return OperandSpec(name, pattern, "read", doc=doc)


def W(name: str, pattern: str, doc: str = "") -> OperandSpec:
    return OperandSpec(name, pattern, "write", doc=doc)


def RW(name: str, pattern: str, doc: str = "") -> OperandSpec:
    return OperandSpec(name, pattern, "readwrite", doc=doc)


def N(name: str, pattern: str, doc: str = "") -> OperandSpec:
    """An operand that is neither read nor written as memory (a scalar, an event, a function)."""
    return OperandSpec(name, pattern, "none", doc=doc)


def VAR(name: str, pattern: str = "value") -> OperandSpec:
    return OperandSpec(name, pattern, "none", variadic=True)


def A(name: str, type: str = "any", *, required: bool = False, default: Any = None, pattern: str | None = None,
      access: str = "none", doc: str = "") -> AttrSpec:
    return AttrSpec(name, type, required, default, pattern, access, doc)


def Res(name: str, pattern: str, doc: str = "") -> ResultSpec:
    return ResultSpec(name, pattern, doc)


def op(name: str, *, level: str = "both", kinds: frozenset[str] = KERNEL, side: str = "any", pipe: str | None = None,
       operands: tuple[OperandSpec, ...] = (), attrs: tuple[AttrSpec, ...] = (), results: tuple[ResultSpec, ...] = (),
       regions: tuple[str, ...] = (), devices: frozenset[str] | None = None, dtypes: frozenset[str] | None = None,
       terminator: bool = False, effects: frozenset[str] | tuple[str, ...] = (), legacy: tuple[str, ...] | str = (),
       doc: str = "") -> OpSpec:
    if isinstance(legacy, str):
        legacy = (legacy,)
    spec = OpSpec(name, level, kinds, side, pipe, operands, attrs, results, regions, devices, dtypes, terminator,
                  frozenset(effects), tuple(legacy), doc)
    return REGISTRY.register(spec)


def retire(old: str, reason: str) -> None:
    REGISTRY.retire(old, reason)


# Attribute bundles shared by several tables.
MASK_ATTR = A("mask", "value", pattern="mask<*>", access="read", doc="predicate; absent means all lanes")
