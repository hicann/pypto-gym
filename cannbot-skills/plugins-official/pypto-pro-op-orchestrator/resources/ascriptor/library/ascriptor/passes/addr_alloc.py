# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``addr_alloc``: an address for every on-chip allocation and an offset for every workspace (RFC-0001 §10 invariant 6).

One bump cursor per memory space, never freed — the model the old kernels were written against
(AscendC's ``TPipe::InitBuffer``, the CCE target's ``CceAddressAllocator``): allocations get
consecutive addresses in program order; a slot buffer is ``slots`` consecutive, individually
aligned allocations. Alignment: 32 bytes for UB / L1 / BT, one fractal for L0A / L0B (512 bytes)
and L0C (1 KB). Static sizes give static addresses, checked against the device capacities here;
an allocation with a runtime-sized dimension gets its size and address computed by scalar ops
placed before it (the old CCE target's run-time cursor), and every allocation after it in the same
space is dynamic too — the simulator checks those against the capacity when it runs. Workspaces
are bumped the same way in bytes from the launcher's user workspace base (no alignment, as the old
``split_workspace`` did). The pass runs before the side split so both sides agree on every address
(on a5 the cube core writes vector UB with ``dma.l0c_to_ub`` and the vector core writes L1).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..devices import SIMT_UB_CAP_KB
from ..ir import Block, Function, Module, Op
from ..ir.builder import Rewriter
from ..ir.types import BufType, MemType
from .manager import Pass, PassContext, PassError
from .util import Emit, Scalar, dim_scalar, function_names

PASS = "addr_alloc"
ALIGN = {"ub": 32, "l1": 32, "bt": 32, "l0a": 512, "l0b": 512, "l0c": 1024}


ZZ_COLUMNS = 16  # a c220 fp32 ZZ band is 16 columns wide, whatever the dtype's C0 says
ROWS = 16  # both layouts give a fractal column 16 rows, so a shorter tile still occupies them


def l1_tile_bytes(rows: Scalar, cols: Scalar, t: MemType, zz: bool, e: Emit) -> Scalar:
    """What a `gm_to_l1` into this tile physically writes, which is what must be reserved.

    Both on-chip layouts pad the same way and differ only in the column granule:

        bytes = align16(rows) * align_up(cols, G) * width      G = 16 for ZZ, else C0 = 32 / width

    NZ stores ceil(cols / C0) fractal columns, each `dstNzC0Stride = align16(rows)` rows of a
    32-byte fractal row (CANN `data_copy_wrapper_nd.h`, `kernel_operator_data_copy_impl.h`), so
    `align_up(cols, C0) * align16(rows) * width` is the same number. ZZ (c220 fp32 only) stores
    `rows/16` bands of `align16(cols) * 16` elements, which is the G = 16 form.

    Reserving the logical size instead is what let two adjacent tiles overlap on an A2 card, with
    the tile loaded first losing its upper half (A2-B32-ND2NZ-L1-OVERWRITE). Nothing below a card
    can object: D-022 makes every DMA a logical window copy, so no simulator models the physical
    layout and an overlapping write is invisible to `sim` and `pipesim` by construction.
    """
    width = t.dtype.bits // 8
    granule = ZZ_COLUMNS if zz else 32 // width
    return e.mul(e.mul(e.align(rows, ROWS), e.align(cols, granule)), width)


def _bytes(e: Emit, t: MemType, family: str = "") -> Scalar:
    dims = [dim_scalar(d) for d in t.dims]
    if t.space == "l1" and t.layout == "nz" and len(dims) >= 2 and t.dtype.bits >= 8:
        # The c220 cube reads fp32 L1 tiles as ZZ and everything else as NZ -- the dtype decides,
        # not its width. `tensorutils_cce.h` writes and reads them on the same predicate; testing
        # `sizeof(T) == 4` there sent i32 down the ZZ route and made a narrow i32 tile spill past
        # its logical end, which is how this was found.
        zz = family == "a2" and t.dtype.name == "f32"
        return l1_tile_bytes(dims[-2], dims[-1], t, zz, e)
    numel = e.prod(dims)
    if t.dtype.bits >= 8:
        return e.mul(numel, t.dtype.bits // 8)
    return e.div(e.add(e.mul(numel, t.dtype.bits), 7), 8)


def allocate(f: Function, capacities: dict[str, int], rw: Rewriter, ctx: PassContext) -> Function:
    cursor: dict[str, Scalar] = {}
    names = function_names(f)
    pre: dict[int, list[Op]] = {}  # alloc op id -> scalar ops computing its size / address
    attrs_of: dict[int, dict[str, Any]] = {}

    def place(op: Op) -> None:
        v = op.results[0]
        t = v.type
        e = Emit(rw, f, op, names)
        if op.opcode == "mem.workspace":
            numel = e.value(op.attrs["numel"])
            if not isinstance(t, MemType):
                raise PassError(PASS, "workspace allocation must have a memory type")
            size = e.mul(numel, t.dtype.bits // 8)
            start = cursor.get("ws", 0)
            cursor["ws"] = e.add(start, size)
            attrs_of[op.id] = {"offset": start}  # type: ignore[index]
            pre[op.id] = e.pre  # type: ignore[index]
            ctx.explain.note(f"%{v.name}: workspace byte offset {start}", op=op.id, kind="addr", space="ws")
            return
        elem = t.elem if isinstance(t, BufType) else t
        if not isinstance(elem, MemType):
            raise PassError(PASS, "allocated value must contain memory tiles")
        space = elem.space
        if space not in ALIGN:
            raise PassError(PASS, f"%{v.name} (#{op.id}) lives in {space}, which addr_alloc does not place")
        align = ALIGN[space]
        slots = t.slots if isinstance(t, BufType) else 1
        slot_bytes = e.align(_bytes(e, elem, ctx.device.family), align)
        start = cursor.get(space, 0)
        end = e.add(start, e.mul(slot_bytes, slots))
        cap = capacities.get(space)
        if isinstance(end, int) and cap is not None and end > cap:
            raise PassError(PASS, f"{space.upper()} overflow allocating %{v.name} (#{op.id}, {op.loc}): {end} bytes needed, {cap} available "
                            f"({start} in use before it)")
        cursor[space] = end
        attrs_of[op.id] = {"addr": start}  # type: ignore[index]
        pre[op.id] = e.pre  # type: ignore[index]
        what = f"0x{start:x} .. 0x{end:x}" if isinstance(start, int) and isinstance(end, int) else f"{start} .. {end} (run-time size)"
        ctx.explain.note(f"%{v.name}: {space} {what}, {slots} x {slot_bytes} bytes", op=op.id, kind="addr", addr=start, space=space)

    for op in f.walk():
        if op.opcode in ("mem.alloc", "mem.workspace"):
            place(op)

    def stamp(block: Block) -> Block:
        out: list[Op] = []
        for op in block.ops:
            if op.id in attrs_of:
                out.extend(pre.get(op.id, []))
                op = replace(op, attrs={**op.attrs, **attrs_of[op.id]})
            if op.regions:
                op = replace(op, regions=tuple(stamp(r) for r in op.regions))
            out.append(op)
        return Block(tuple(out))

    for space, used in sorted(cursor.items(), key=lambda kv: kv[0]):
        ctx.explain.note(f"{space}: {used} bytes of {capacities.get(space, '?')}", kind="usage", space=space, used=used)
    return replace(f, body=stamp(f.body))


def run(module: Module, ctx: PassContext) -> Module:
    caps = {k: int(v * 1024) for k, v in ctx.device.capacities_kb.items()}
    if any(o.opcode == "simt.launch" for o in module.walk()):
        caps["ub"] = min(caps.get("ub", 1 << 30), SIMT_UB_CAP_KB * 1024)  # SIMT keeps the top of UB for itself
    rw = Rewriter(module, PASS)
    functions = [allocate(f, caps, rw, ctx) if f.kind in ("kernel", "func") else f for f in module.functions]
    attrs = dict(module.attrs)
    attrs["next_id"] = rw._next_id
    return Module(module.name, attrs, tuple(functions))


PASS_DEF = Pass(PASS, run, doc="bump addresses per memory space (static or run-time cursor) with fractal alignment and capacity checks",
                establishes=("6",))

__all__ = ["ALIGN", "PASS_DEF", "allocate", "run"]
