# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile-time values of the frontend (RFC-0002 §4).

The compiler evaluates a kernel body without running it; every Python expression yields one of
these values (or a plain Python object when it is static):

* :class:`Dyn` -- an IR value, optionally decorated with :class:`Riders` (compile-time flags a
  DSL method attaches to a tensor: ``.T``, ``.relu()``, ``.requant()``, ``.subblk()``, ``.single()``
  ...). The riders are consumed by the next ``<<=`` and never reach the IR on their own.
* :class:`ElemOffset` -- ``t[k]`` on an on-chip tensor: a flat element offset (vf) or one element (simt).
* :class:`RegExpr` -- a deferred register expression (the old ``RegOP``): ``reg.abs() + 1.0`` builds
  a tree that is emitted only when it meets ``<<=``, so the outermost op writes the target
  directly and only inner links need temporaries. The emission order is the old one.
* :class:`RegList` -- a static list of registers (the old ``RegList``): ``rl[i]`` needs a static
  index, whole-list operations expand element by element.
* :class:`Img2col` -- a compile-time description of an ``img2col`` view of an L1 feature map.
* :class:`BoundMethod` -- ``obj.method`` before the call.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..ir import Value
from ..ir.types import Type


@dataclass(frozen=True)
class Riders:
    """Compile-time flags a tensor expression carries into the next ``<<=``."""

    transpose: bool = False
    mode: str | None = None  # ub->reg load distribution: single | brcb | upsample | downsample | unpack | unpack4
    relu: bool = False
    scale: Any = None  # requant scale (int | float | Dyn); None = 1.0
    offset: Any = None  # requant offset (int | Dyn); None = 0
    hif8_hybrid: bool = False
    subblk: Any = None  # 0 | 1 | Dyn

    @property
    def has_requant(self) -> bool:
        return self.relu or self.hif8_hybrid or (self.scale is not None and not (isinstance(self.scale, float) and self.scale == 1.0)
                                                  and not (isinstance(self.scale, int) and not isinstance(self.scale, bool) and self.scale == 1)) \
            or (self.offset is not None and not (isinstance(self.offset, int) and self.offset == 0))


NO_RIDERS = Riders()


@dataclass(frozen=True)
class Dyn:
    """A dynamic value: an IR value. ``type`` tells the compiler what it is."""

    value: Value
    riders: Riders = NO_RIDERS

    @property
    def type(self) -> Type:
        return self.value.type

    @property
    def name(self) -> str:
        return self.value.name

    def with_riders(self, **changes: Any) -> Dyn:
        return replace(self, riders=replace(self.riders, **changes))

    def plain(self) -> Dyn:
        return Dyn(self.value) if self.riders is not NO_RIDERS else self

    def __repr__(self) -> str:
        r = "" if self.riders == NO_RIDERS else f" {self.riders}"
        return f"Dyn({self.value} : {self.type}{r})"


@dataclass(frozen=True)
class ElemOffset:
    """``t[idx]`` on an on-chip tensor: a flat element offset (vf load/store) or one element (simt)."""

    base: Dyn
    offset: Any  # int | Dyn


@dataclass(frozen=True)
class BoundMethod:
    obj: Any
    name: str


@dataclass
class RegList:
    """The old ``RegList``: ``length`` registers of one dtype, indexed statically."""

    elems: list[Dyn]
    dtype: Any
    name: str

    def __len__(self) -> int:
        return len(self.elems)


@dataclass
class RegExpr:
    """A deferred register expression (the old ``RegOP``).

    ``op`` is the ir op suffix (``abs``, ``adds``, ``cast``, ``cmp``...) or a store form
    (``store:pack4``, ``store:downsample``, ``store:single``); ``inputs`` are Dyn registers,
    RegList values or scalars; ``mask`` is a Dyn mask register attached with ``*``.
    """

    op: str
    inputs: tuple[Any, ...]
    mask: Dyn | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    dtype: Any = None  # explicit result dtype (astype)


@dataclass(frozen=True)
class Img2colWindow:
    view: Img2col
    m0: Any
    k0: Any
    m_ext: Any
    k_ext: Any


@dataclass(frozen=True)
class Img2col:
    """``img2col(fmap, conv, h, w, c)``: the compile-time description of an L1 NC1HWC0 feature map."""

    fmap: Dyn
    conv: Any  # dsl.Conv2D
    h: Any
    w: Any
    c: Any

    def window(self, m0: Any, k0: Any, m_ext: Any, k_ext: Any) -> Img2colWindow:
        return Img2colWindow(self, m0, k0, m_ext, k_ext)


def is_dynamic(v: Any) -> bool:
    if isinstance(v, (Dyn, ElemOffset, BoundMethod, RegExpr, RegList, Img2col, Img2colWindow)):
        return True
    if isinstance(v, (list, tuple)):
        return any(is_dynamic(x) for x in v)
    if isinstance(v, dict):
        return any(is_dynamic(x) for x in v.values())
    return False
