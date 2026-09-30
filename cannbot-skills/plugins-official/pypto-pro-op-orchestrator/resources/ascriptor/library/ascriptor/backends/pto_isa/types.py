# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The type mapping between our memory types and ``pto::Tile`` / ``pto::GlobalTensor`` (RFC-0011 §3).

Data only: which ``TileType`` a memory space is, which C++ element type a dtype spells, and how a
layout token becomes the ``(BLayout, SLayout)`` pair. The printer in :mod:`.emit` builds the
template argument lists from these.
"""

from __future__ import annotations

from ...ir.types import DType
from ..cce.cpp import PIPE as _CCE_PIPE

# ---------------------------------------------------------------- locations

#: our memory space -> ``pto::TileType`` (RFC-0011 §3). ``gm`` is not a Tile: it becomes a
#: ``GlobalTensor`` and has no entry here.
LOC: dict[str, str] = {
    "ub": "TileType::Vec",
    "l1": "TileType::Mat",
    "l0a": "TileType::Left",
    "l0b": "TileType::Right",
    "l0c": "TileType::Acc",
    "bt": "TileType::Bias",
    "l0amx": "TileType::ScaleLeft",
    "l0bmx": "TileType::ScaleRight",
}

#: on-chip capacity per space, in bytes, as PTO's own TASSIGN table states it for a5
#: (docs/isa/TASSIGN.md). Cross-checked against ``devices/profiles/950.json``: identical.
#: Kept here so a TASSIGN we print can be bounds-checked before bisheng sees it.
CAPACITY_A5: dict[str, int] = {
    "ub": 256 * 1024, "l1": 512 * 1024, "l0a": 64 * 1024, "l0b": 64 * 1024,
    "l0c": 256 * 1024, "bt": 4 * 1024,
}

#: every space PTO aligns tiles to 32 bytes (same table).
ALIGN = 32

# ---------------------------------------------------------------- element types

#: our dtype -> the C++ element type PTO's templates are instantiated with.
#:
#: These are the spellings that appear in pto's own a5 headers (``npu/a5/datatype.hpp``'s
#: ``TypeGet`` specialisations). They are **not** the spellings our cce backend uses
#: (``cpp.CTYPE``: ``fp8_e4m3fn_t``, ``fp4x2_e2m1_t``, ``fp8_e8m0_t``): both sides name types the
#: bisheng compiler provides for dav-c310, and whether each pair is the same type under two names
#: or two distinct types is **unverified** — pto's headers use these names without defining them.
#: RFC-0011 §8 tracks it; the first bisheng build of batch 1 settles it.
ELEM: dict[str, str] = {
    "b1": "bool",
    "i8": "int8_t", "u8": "uint8_t", "i16": "int16_t", "u16": "uint16_t",
    "i32": "int32_t", "u32": "uint32_t", "i64": "int64_t", "u64": "uint64_t",
    "f16": "half", "bf16": "bfloat16_t", "f32": "float",
    "e4m3": "float8_e4m3_t", "e5m2": "float8_e5m2_t", "hif8": "hifloat8_t",
    "e8m0": "float8_e8m0_t",
    "fp4_e2m1": "float4_e2m1x2_t", "fp4_e1m2": "float4_e1m2x2_t",
    "i4": "int4b_t",
}

#: dtypes with no PTO spelling at all. Complex tensors ride their bit patterns on the cce side
#: (``c32`` -> ``uint32_t``); doing the same here would move data PTO cannot then compute on, so
#: they are refused instead of silently carried.
NO_ELEM: dict[str, str] = {
    "c32": "complex32 has no PTO element type (cce carries the bit pattern; PTO would then have "
            "no instruction that reads it back as complex)",
    "c64": "complex64 has no PTO element type (same reason as c32)",
}


def elem(dt: DType) -> str:
    """The C++ element type for ``dt``, or raise ``KeyError`` with the reason for a refused dtype."""
    if dt.name in NO_ELEM:
        raise KeyError(NO_ELEM[dt.name])
    if dt.name not in ELEM:
        raise KeyError(f"dtype {dt.name} has no PTO element type")
    return ELEM[dt.name]


# ---------------------------------------------------------------- layouts

#: our layout token -> ``(BLayout, SLayout)`` (RFC-0011 §3). ``nd`` is the plain unboxed
#: row-major matrix; ``nz`` is the fractal layout the cube operands and L1 tiles use.
LAYOUT: dict[str | None, tuple[str, str]] = {
    None: ("BLayout::RowMajor", "SLayout::NoneBox"),
    "nd": ("BLayout::RowMajor", "SLayout::NoneBox"),
    "nz": ("BLayout::ColMajor", "SLayout::RowMajor"),
}

#: The L0 operand fractals, which are **not** the same on the two sides and are not the ``nz``
#: our IR marks them with. ``TMovToLeft`` asserts its destination is ``(ColMajor,
#: SLayout::RowMajor)``; ``TMovToRight`` asserts ``(RowMajor, SLayout::ColMajor)``
#: (``npu/a5/TMov.hpp``). So the destination *position* picks the pair, not the layout token.
L0_LAYOUT: dict[str, tuple[str, str]] = {
    "l0a": ("BLayout::ColMajor", "SLayout::RowMajor"),
    "l0b": ("BLayout::RowMajor", "SLayout::ColMajor"),
}

#: The MX scale planes' orders, which are **not** the data operands' and are asserted on *both*
#: sides of the move: ``TExtractToAmx`` wants ``(RowMajor, SLayout::RowMajor)`` for source and
#: destination alike, ``TExtractToBmx`` ``(ColMajor, SLayout::ColMajor)``
#: (`npu/a5/TExtract.hpp:34, 81`). So unlike §4.3's data path there is no fractal-order choice
#: here — no transpose branch hangs off it — and the L1 source is read in the same order it lands.
MX_LAYOUT: dict[str, tuple[str, str]] = {
    "l0a": ("BLayout::RowMajor", "SLayout::RowMajor"),
    "l0b": ("BLayout::ColMajor", "SLayout::ColMajor"),
}

#: which L0 side each MX scale plane belongs to
MX_SPACE = {"l0a": "l0amx", "l0b": "l0bmx"}

#: The scale plane's address is the data tile's, shifted: cce writes
#: ``load_cbuf_to_ca_mx(dst.addr / 16, …)`` (`tensorutils_cce.h:1196`) and pypto_pro's own
#: documentation states it as ``addr(scale_a) = addr(lhs_tile) >> 4``. It is a fixed hardware
#: mapping between the L0 data buffer and its scale plane, not an allocation this compiler makes.
MX_ADDR_SHIFT = 16

#: The other fractal ordering, which is how a transposing load is spelled: PTO branches on
#: ``Dst::SFractal == Src::SFractal`` and has no transpose parameter, so the source is *viewed*
#: with the fractal order that selects the branch we want. Reading one L1 allocation through two
#: fractal orders is legal because a tile is a view (RFC-0011 §3).
OTHER_SFRACTAL = {"SLayout::RowMajor": "SLayout::ColMajor",
                  "SLayout::ColMajor": "SLayout::RowMajor"}

#: PTO's base-tile (fractal) byte sizes, from ``pto::TileConfig``. A/B operand tiles box at 512 B,
#: the accumulator at 1024 B, an MX scale plane at 32; unboxed tiles carry the A/B default and
#: ignore it.
FRACTAL_AB = "TileConfig::fractalABSize"  # 512
FRACTAL_C = "TileConfig::fractalCSize"  # 1024
FRACTAL_MX = "TileConfig::fractalMxSize"  # 32

#: which spaces take which fractal size.
FRACTAL_C_SPACES = frozenset({"l0c"})
FRACTAL_MX_SPACES = frozenset({"l0amx", "l0bmx"})

#: the MX box, which is **not** square: ``fixedMxRowSize`` x ``fixedMxColSize``
#: (`common/pto_tile.hpp:1075`). And an MX tile is exempt from ``Rows % InnerRows == 0`` the same
#: way a Vec tile is (`pto_tile.hpp:1526`) — the scale plane's row count follows the data tile's,
#: which need not tile the box.
MX_INNER = (16, 2)


def fractal(space: str) -> str:
    if space in FRACTAL_C_SPACES:
        return FRACTAL_C
    return FRACTAL_MX if space in FRACTAL_MX_SPACES else FRACTAL_AB


# ---------------------------------------------------------------- pipes

#: our pipe name -> the CCE pipe constant. Synchronisation prints the bare ``set_flag`` /
#: ``wait_flag`` pair (RFC-0011 §5), which is CCE's own spelling and what pto's a5 ST kernels
#: bracket their tile instructions with under ``#ifndef __PTO_AUTO__``.
PIPE: dict[str, str] = _CCE_PIPE
