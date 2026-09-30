# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Translate IR-owned slot identities into PyPTO Tile metadata (RFC-0013).

This module does not plan synchronization or allocate mutex IDs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod

from ...ir import Op


def validate_mode(mode: str, gap: Callable) -> None:
    if not isinstance(mode, str) or mode not in ("manual", "auto_mutex"):
        raise gap(None, f"sync_mode must be 'manual' or 'auto_mutex', got {mode!r}", owner="ours")


@dataclass(frozen=True)
class TileDecl:
    name: str
    group: str
    type_expr: str
    bank: str
    addresses: tuple[int, ...]
    size: int
    shape: tuple[int, ...]
    bits: int
    op: Op
    singleton: bool = False
    original: str = ""
    mutex_ids: tuple[int, ...] = ()
    parent_slots: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class TileSite:
    """A declaration at its original structured emission site."""

    name: str
    indent: int


class TileRegistry:
    """Copy allocation IDs to their physical aliases within one core side."""

    def __init__(self, side: str, gap: Callable):
        self.side, self.gap = side, gap
        self.declarations: list[TileDecl] = []

    def add(self, d: TileDecl) -> None:
        if (d.bank not in ("ub", "l1", "l0a", "l0b", "l0c", "bt", "scale_l", "scale_r")
                or not d.addresses or any(type(a) is not int or a < 0 for a in d.addresses)
                or type(d.size) is not int or d.size <= 0
                or not d.shape or any(type(n) is not int or n <= 0 for n in d.shape)
                or type(d.bits) is not int or d.bits <= 0
                or any(a + d.size > (1 << 63) - 1 for a in d.addresses)):
            raise self.gap(d.op, f"IR mutex mapping requires known physical bank, addresses and capacity: {d.bank}, {d.addresses}, {d.size}, {d.shape}", owner="ours")
        if d.bank in ("scale_l", "scale_r") and (not d.parent_slots or d.mutex_ids):
            raise self.gap(d.op, "MX scale companions require their L0 data parent, not independent IDs", owner="ours")
        if d.mutex_ids and (len(d.mutex_ids) != len(d.addresses)
                            or len(set(d.mutex_ids)) != len(d.mutex_ids)
                            or any(type(i) is not int or not 0 <= i < 32 for i in d.mutex_ids)):
            raise self.gap(d.op, "mutex_ids must contain one distinct ID in 0..31 per physical slot", owner="ours")
        self.declarations.append(d)

    def finalize(self) -> tuple[list[str], dict[str, str | None], list[dict]]:
        owners = [(d, slot, address, address + d.size, d.mutex_ids[slot])
                  for d in self.declarations if d.mutex_ids for slot, address in enumerate(d.addresses)]
        seen = {}
        for d, slot, start, end, ident in owners:
            if ident in seen:
                raise self.gap(d.op, f"IR mutex ID {ident} is assigned to multiple physical slots on {self.side}", owner="ours")
            seen[ident] = (d, slot)
            if any(other.bank == d.bank and other is not d and start < stop and begin < end
                   for other, _, begin, stop, _ in owners):
                raise self.gap(d.op, "independent IR mutex allocations overlap physically", owner="ours")
        mapped = {}
        for d in self.declarations:
            mapped[d.name] = [sorted({ident for owner, _, begin, end, ident in owners
                                      if owner.bank == d.bank and start < end and begin < start + d.size})
                              for start in d.addresses]
            if d.mutex_ids and mapped[d.name] != [[i] for i in d.mutex_ids]:
                raise self.gap(d.op, "physical slot overlap would change its IR mutex identity", owner="ours")
        # Some native carrier types extend beyond their actual addressed window.
        # Their adapter supplies the proven parent slot rather than that type span.
        by_name = {d.name: d for d in self.declarations}
        resolved, active = set(), set()

        def inherit(d):
            if d.name in resolved:
                return mapped[d.name]
            if d.name in active or len(d.parent_slots) not in (0, len(d.addresses)):
                raise self.gap(d.op, "invalid native alias parent slots", owner="ours")
            active.add(d.name)
            if d.parent_slots:
                slots = []
                for index, (name, slot) in enumerate(d.parent_slots):
                    parent = by_name.get(name)
                    bank = {"scale_l": "l0a", "scale_r": "l0b"}.get(d.bank, d.bank)
                    if parent is None or parent.bank != bank or not 0 <= slot < len(parent.addresses):
                        raise self.gap(d.op, "native alias has no matching physical parent slot", owner="ours")
                    if d.bank in ("scale_l", "scale_r") and (
                            parent.addresses[slot] % 16 or d.addresses[index] != parent.addresses[slot] >> 4
                            or d.size * 16 > parent.size):
                        raise self.gap(d.op, "MX scale companion exceeds its L0 data slot mapping", owner="ours")
                    slots.append(inherit(parent)[slot])
                mapped[d.name] = slots
            active.remove(d.name)
            resolved.add(d.name)
            return mapped[d.name]

        for d in self.declarations:
            inherit(d)
        hoisted, replacements, metadata = [], {}, []
        for d in self.declarations:
            slots = mapped[d.name]
            if not any(slots):
                replacements[d.name] = d.original
                continue
            if not slots[0] or any(len(ids) != len(slots[0]) for ids in slots):
                raise self.gap(d.op, "PyPTO tile group needs the same nonzero mutex count in every slot", owner="ours")
            if prod(d.shape) * max(1, (d.bits + 7) // 8) != d.size:
                raise self.gap(d.op, "IR mutex mapping changes make_tile_group type-derived capacity", owner="ours")
            ids = [slot[0] if len(slot) == 1 else slot for slot in slots]
            line = (f"{d.group} = pl.make_tile_group(type={d.type_expr}, addrs={list(d.addresses)}, "
                    f"mutex_ids={ids}, depth={len(slots)})")
            if d.singleton:
                hoisted.extend((line, f"{d.name} = {d.group}[0]"))
                replacements[d.name] = None
            else:
                replacements[d.name] = line
            metadata.append({"side": self.side, "name": d.name, "bank": d.bank,
                             "addresses": list(d.addresses), "mutex_ids": ids,
                             "source": "ir_allocation" if d.mutex_ids else "parent_slot" if d.parent_slots else "physical_alias", "op": d.op.id})
        return hoisted, replacements, metadata
