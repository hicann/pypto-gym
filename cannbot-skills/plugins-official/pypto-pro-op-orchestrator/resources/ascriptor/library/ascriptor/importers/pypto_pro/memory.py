# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Admitted physical storage and transfer mappings; no address allocation."""

import struct
from math import prod

from ...ir import Ident, Value
from ...ir.types import MemType
from ...passes.addr_alloc import ALIGN
from .lower import Memory
from .mx import PLANE_BYTES, PLANES, Plane, storage
from .mx import tile as mx_tile

SPACES = {"Vec": "ub", "Mat": "l1", "Left": "l0a", "Right": "l0b", "Acc": "l0c", "Bias": "bt"}
LAYOUTS = {"Vec": ("row_major", "none_box", 512), "Mat": ("col_major", "row_major", 512),
           "Left": ("col_major", "row_major", 512), "Right": ("row_major", "col_major", 512),
           "Acc": ("col_major", "row_major", 1024), "Bias": ("row_major", "none_box", 512), **PLANES}
# Measured on A5: a ZN Mat holds the NZ bytes of its transpose; an NZ Vec alias holds compact fractals.
# E8M0 ZZ/NN Mats hold MX scale planes; NN is typed as its transpose, like ZN (mx.tile).
ALTERNATIVES = {("Mat", "row_major", "col_major", 512): "zn", ("Vec", "col_major", "row_major", 512): "nz",
                ("Mat", "row_major", "row_major", 32): "zz", ("Mat", "col_major", "col_major", 32): "nn"}


def allocate(ctx, node, args):
    o = ctx.o
    attrs = ctx.attrs(node, {"dtype", "target_memory", "memref_addr", "memref_size", "memref_id",
                            "blayout", "slayout", "fractal", "pad", "compact"})
    typ = o.node(node["fields"]["type"])
    o.need(typ["kind"] == "TileType", node, "make_tile must produce a TileType")
    f = typ["fields"]
    shape = o.shape(f["shape"], node)
    dt = o.dt(f["dtype"], node)
    memory = o.node(f["memref"])["fields"]
    space = memory["memory_space"]["name"]
    from .sort import index_storage
    o.need(dt.name in {"f16", "bf16", "f32", "i32"} or space == "Vec" and index_storage(dt) or storage(dt), node,
           "Unadmitted local element type")
    o.need(space in LAYOUTS, node, "Unadmitted memory space")
    o.need(o.mixed or (o.side == "vec") == (space == "Vec"), node, "Foreign storage requires both source sides")
    hardware = o.node(f["hardware_info"])["fields"]
    canonical = (hardware["blayout"]["name"], hardware["slayout"]["name"], hardware["fractal"])
    o.need(space != "Left" or canonical[:2] != ("row_major", "row_major"), node,
           "ZZ Left is Pro's A3 default; A5 parses and requires NZ Left, so re-export under the A5 environment (I017)")
    layout = ALTERNATIVES.get((space, *canonical))
    o.need((canonical == LAYOUTS[space] or (space, *canonical) in ALTERNATIVES) and hardware["pad"]["name"] == "null"
           and hardware["compact"]["name"] == "null", node, "Noncanonical tile layout/padding/compact mode is not admitted")
    for key in ("blayout", "slayout", "pad", "compact"):
        if key in attrs:
            actual = attrs[key].get("value") if isinstance(attrs[key], dict) else attrs[key]
            o.need(actual == hardware[key]["value"], node, f"Contradictory tile {key}")
    if "fractal" in attrs:
        o.need(attrs["fractal"] == hardware["fractal"], node, "Contradictory tile fractal")
    addr = o.literal(memory["addr"], node)
    o.need(type(addr) is int and addr >= 0 and addr % 32 == 0, node, "Static aligned local address required")
    size = (prod(shape) * dt.bits // 8 + 31) // 32 * 32
    position = SPACES.get(space)  # None: a declaration-only scale plane (mx).
    root = o.roots.get((position, addr))  # A later declaration at this address re-declares its bytes.
    slot = o.slot_tiles.get(node["id"])
    # Pro declares every tile in the kernel prologue (alias_loop_probe); import hoists Vec re-declarations only (lower.hoist).
    o.need(not o.hoisting or root is not None and position == "ub", node, "Allocations in runtime control flow need lifetime legalization")
    o.need(root is None or "buffer" not in root and slot is None, node, "Tile aliases of slot-buffer slots need a same-buffer rule")
    reserved = memory["size"]
    o.need((reserved == size or root is not None and size <= reserved) and attrs.get("memref_size") == reserved
           and attrs.get("memref_addr") == addr, node, "Reserved size/address needs explicit backing-storage legalization")
    o.need(position is None or addr % ALIGN[position] == 0, node, "Address violates the target bank alignment")
    o.need(addr + size <= (PLANE_BYTES if position is None else o.device.capacity_bytes(position)), node,
           "Physical allocation exceeds target bank capacity")
    from .simt import SIMT_UB_CAP_KB, UB_LIMIT
    o.need(not (o.simt_launched and position == "ub" and addr + size > UB_LIMIT), node,
           f"SIMT launches reserve UB above {SIMT_UB_CAP_KB} KB")
    placed = o.allocations.setdefault(position or space, [])
    o.need(root is not None or all(addr + size <= start or addr >= end for start, end in placed), node,
           "Overlapping physical allocations require alias/backing-storage legalization")
    if root is None:
        placed.append((addr, addr + size))
    o.need(attrs.get("memref_id") == f["memref"] and attrs.get("dtype") == f["dtype"]
           and attrs.get("target_memory") == memory["memory_space"], node, "Inconsistent allocation identity or dtype")
    scaled = storage(dt) or position is None or layout in {"zz", "nn"}
    if scaled:
        mx_tile(o, node, space, dt, shape, layout)
    elif space == "Bias":
        o.need(shape[0] == 1 and dt.name == "f32", node, "Bias tables need one FP32 row")
    elif space != "Vec":
        row = space == "Mat" and shape[0] == 1 and shape[1] % 16 == 0 and layout is None  # a flat bias row (bias_chain)
        o.need((row or all(d % 16 == 0 for d in shape)) and dt.name == ("f32" if space == "Acc" else "f16"), node,
               "Cube import currently requires full 16-aligned FP16 tiles, one-row FP16 Mats and FP32 accumulators")
    view = o.node(f["tile_view"])["fields"]
    o.need(not view["stride"] and (view["start_offset"] is None or o.literal(view["start_offset"], node) == 0), node,
           "Local strided/start-offset descriptors need view legalization")
    declared = tuple(o.literal(v, node) for v in view["valid_shape"])
    # valid_shape=[-1, -1] declares the whole shape and, as any valid-shape metadata, a distinct handle (alias_identical_probe).
    valid = shape if declared == (-1, -1) and space == "Vec" else declared or shape
    o.need(len(valid) == 2 and all(type(d) is int and 0 < d <= n for d, n in zip(valid, shape, strict=True)),
           node, "Unadmitted valid shape")
    o.need(len(args) == 2 and args[0] == shape and (args[1] == declared or not args[1] and valid == shape),
           node, "make_tile operands disagree with its descriptor")
    if scaled:
        o.need(valid == shape and root is None, node, "Microscaling tiles are whole and unaliased")
        if position is None:
            return Plane(space, addr, shape)
    if root is not None:
        o.need(not storage(root["memory"].value.type.dtype), node, "Microscaling tiles are whole and unaliased")
        from .views import alias
        return alias(ctx, node, root, dt, shape, valid, reserved, layout, bool(declared), node["id"] in o.group_slots)
    o.need(layout != "nz", node, "NZ Vec tiles are admitted only as aliases of Vec tiles")
    dims = shape[::-1] if space == "Right" or layout in {"zn", "nn"} else shape
    target_type = MemType(SPACES[space], dt, dims, "nz" if space == "Mat" and shape[0] > 1 else None)
    if slot is not None:
        from .slots import declare
        return declare(ctx, node, slot, target_type, addr, reserved, shape, valid, layout == "zn", position)
    from .cross import storage_name
    name = storage_name(o, node, target_type, addr, size, valid)
    value = ctx.emit("mem.alloc", node, typ=target_type, attrs={"addr": addr}, result_name=name)
    result = Memory(value, shape, valid, transposed=layout in {"zn", "nn"})
    kind = "slot" if node["id"] in o.group_slots else "metadata" if declared else "fold"  # views.alias
    o.roots[(position, addr)] = {"memory": result, "size": reserved, "declared": {(dt, shape, layout): (kind, result)}}
    return result


def half_exact(value):
    """A literal that binary16 holds without rounding or overflow."""
    try:
        return struct.unpack("<e", struct.pack("<e", float(value)))[0] == value
    except OverflowError:
        return False


def gm_window(ctx, node, memory, offsets, extent):
    from .dynamic import extent_type
    from .loop_extents import window
    from .selection import interval

    o = ctx.o
    o.need(isinstance(memory, Memory) and memory.value.type.space in {"gm", "ws"}, node, "Expected a GM tensor or workspace")
    o.need(isinstance(offsets, tuple) and len(offsets) == 2, node, "Expected two GM offsets")
    offsets = tuple(ctx.snapshot(off, node) for off in offsets)
    ranges = [interval(ctx, off) for off in offsets]
    dynamic = any(isinstance(v, Value) for v in (*offsets, *extent, *memory.shape))
    safe = all(window(ctx, off, size, bound) for off, size, bound in zip(offsets, extent, memory.shape, strict=True)) if dynamic else all(
                r is not None and 0 <= r[0] <= r[1] and r[1] + size <= bound
                for r, size, bound in zip(ranges, extent, memory.shape, strict=True))
    o.need(safe,
           node, "Transfer window exceeds the declared GM tensor")
    if offsets == (0, 0) and extent == memory.shape:
        return memory.value
    return ctx.emit("mem.slice", node, (memory.value,), typ=MemType(memory.value.type.space, memory.value.type.dtype, extent_type(extent)),
                    attrs={"offsets": list(offsets), "extents": list(extent)})


def memory_call(ctx, node, args):
    o = ctx.o
    name = node["fields"]["name"]
    if name == "block.make_tile":
        o.need(not ctx.depth, node, "Allocations in runtime control flow need lifetime legalization")
        return allocate(ctx, node, args)
    if name == "block.subview":
        from .selection import interval

        ctx.attrs(node)
        o.need(len(args) == 3 and isinstance(args[0], Memory) and args[0].value.type.space == "ub", node,
               "Only ordinary UB subviews are admitted")
        parent, offsets, extent = args
        o.need(parent.root is None, node, "Subviews of tile aliases need a same-buffer rule")
        o.need(parent.valid is not None, node, 'Descriptor mutation requires a valid-shape reset before subview')
        o.need(all(type(v) is int for v in parent.valid), node, 'Subview of dynamic valid shape is not admitted')
        def static(values):
            if not isinstance(values, tuple):
                return values
            ranges = [interval(ctx, value) for value in values]
            return tuple(r[0] if r is not None and r[0] == r[1] else value
                         for value, r in zip(values, ranges, strict=True))
        offsets, extent = static(offsets), static(extent)
        o.need(isinstance(offsets, tuple) and isinstance(extent, tuple) and len(offsets) == len(extent) == 2
               and all(type(v) is int for v in (*offsets, *extent)), node, "Subview offsets/extents must be static")
        o.need(all(0 <= off and 0 < size and off + size <= valid
                   for off, size, valid in zip(offsets, extent, parent.valid, strict=True)), node,
               "Subview exceeds the parent's valid extent")
        source_type = o.node(node["fields"]["args"][0])["fields"].get("type")
        o.need(node["fields"]["type"] == source_type, node, "Subview must retain its source descriptor type")
        pitch = parent.pitch or parent.shape[1]
        o.need((offsets[0] * pitch + offsets[1]) * parent.value.type.dtype.bits // 8 % 32 == 0, node,
               "Subview base violates ordinary UB alignment")
        value = ctx.emit("mem.slice", node, (parent.value,), typ=MemType("ub", parent.value.type.dtype, extent),
                         attrs={"offsets": list(offsets), "extents": list(extent)})
        return Memory(value, extent, extent, pitch)
    if name == "block.set_validshape":
        from .dynamic import le, positive
        ctx.attrs(node)
        o.need(len(args) == 3 and isinstance(args[0], Memory), node, "Expected a tile and two valid extents")
        tile, rows, cols = args
        o.need(all(positive(ctx, v) and le(ctx, v, bound) for v, bound in zip((rows, cols), tile.shape, strict=True)),
               node, "Invalid or unproved valid shape")
        o.need(tile.value.type.space == 'ub' or all(type(v) is int for v in (rows, cols)), node,
               "Dynamic valid extents are admitted only for UB")
        tile.valid = (rows, cols)
        return None
    if name in {"block.load", "block.store"}:
        if name == "block.load" and len(args) == 3 and isinstance(args[1], Memory) and args[1].nz:
            from .nz_parameters import load
            return load(ctx, node, args[0], args[1], args[2])
        if name == "block.store" and len(args) in {3, 4} and isinstance(args[0], Memory) and args[0].nz:
            from .nz_parameters import store
            return store(ctx, node, args[0], args[1], args[2], args[3] if len(args) == 4 else None)
        kwargs = node["fields"]["kwargs"]
        # order=[1, 0] over two GM axes: tile element (i, j) reads GM (o0 + j, o1 + i).
        transpose = (name == "block.load" and set(kwargs) == {"is_transpose", "tile_dims"}
                     and kwargs["is_transpose"] is True and kwargs["tile_dims"] == [0, 1])
        if not transpose:
            ctx.attrs(node)
        o.need(len(args) == 3, node, "Only plain three-operand load/store is admitted")
        tile, gm = (args[0], args[1]) if name == "block.load" else (args[1], args[0])
        o.need(isinstance(tile, Memory) and isinstance(gm, Memory), node, "Expected typed transfer operands")
        o.need(tile.valid is not None, node, 'Descriptor mutation requires a valid-shape reset before transfer')
        rows, cols = tile.valid
        space, dt = tile.value.type.space, tile.value.type.dtype
        o.need(dt == gm.value.type.dtype or space == "l0c" and dt.name == gm.value.type.dtype.name == "f32", node,
               "Transfer dtype conversion requires a separate rule")
        o.need(not transpose or space == "l1" and tile.value.type.layout == "nz", node,
               "Transposed loads are admitted only into Mat tiles of at least 16 rows")
        o.need(space != "l1" or transpose or not tile.transposed, node, "ZN Mat loads without order=[1, 0] do not compile "
               "natively: PTO's TLoadCubeCheck needs an NZ Mat for an ND GM source (tload_common.hpp:131, mat_zn_plain_probe)")
        window = gm_window(ctx, node, gm, args[2], tile.valid[::-1] if transpose else tile.valid)
        if space == "ub":
            from .dynamic import arithmetic

            pitch = tile.pitch or tile.shape[1]
            single_row = type(rows) is int and rows == 1
            o.need(single_row or type(cols) is int, node, "Dynamic columns require a single-row transfer")
            local_gap = 0 if single_row else (pitch - cols) * dt.bits // 8
            # A load row ending inside a 32-byte block fills its rest with the row's first element, as gm_to_ub.pad
            # does (gm_view_skew_probe, ub_row_tail_probe on A5; I038); the next row starts at the following block.
            o.need(pitch * dt.bits // 8 % 32 == 0 and (local_gap % 32 == 0 or name == "block.load"), node,
                   "UB row pitch cannot be represented by ordinary DMA")
            gm_gap = 0 if single_row else arithmetic(ctx, node, 'mul', arithmetic(ctx, node, 'sub', gm.pitch or gm.shape[1], cols),
                                                     dt.bits // 8)
            attrs = {"n_burst": rows, "burst_len_byte": arithmetic(ctx, node, 'mul', cols, dt.bits // 8)}
            if name == "block.load":
                attrs.update(src_stride_byte=gm_gap, dst_stride=local_gap // 32)
                ctx.emit("dma.gm_to_ub.pad", node, (tile.value, window), attrs=attrs)
            else:
                attrs.update(dst_stride_byte=gm_gap, src_stride=local_gap // 32)
                ctx.emit("dma.ub_to_gm.pad", node, (window, tile.value), attrs=attrs)
        elif space == "l1" and name == "block.load":
            o.need(tile.valid == tile.shape, node, "Partial NZ loads need explicit physical extent validation")
            pitch, (m, n) = gm.pitch or gm.shape[1], tile.value.type.dims
            if tile.value.type.layout != "nz":  # One-row Mats are flat rows (bias_chain on A5).
                ctx.emit("dma.gm_to_l1.pad", node, (tile.value, window),
                         attrs={"n_burst": 1, "burst_len_byte": cols * dt.bits // 8, "src_stride_byte": 0, "dst_stride": 0})
            elif transpose and not tile.transposed:  # GM rows fill NZ columns (mat_transposed_load on A5).
                ctx.emit("dma.gm_to_l1.dn2nz", node, (tile.value, window), attrs={"M": m, "N": n, "M_dst": m, "N_src": pitch})
            else:  # A ZN Mat is typed as its transpose, which order=[1, 0] loads in GM order (mat_zn_load_probe).
                ctx.emit("dma.gm_to_l1.nd2nz", node, (tile.value, window), attrs={"M": m, "N": n, "M_dst": m, "N_src": pitch})
        elif space == "l0c" and name == "block.store":
            o.need(tile.valid == tile.shape, node, "Partial fixpipe windows are not admitted")
            ctx.emit("dma.l0c_to_gm.nz2nd", node, (window, tile.value),
                     attrs={"M": rows, "N": cols, "M_src": tile.shape[0], "N_dst": gm.pitch or gm.shape[1], "relu": False})
        else:
            o.fail(node, "Unadmitted load/store memory-space pair")
        return None
    if name == "block.move":
        o.need(len(args) == 2 and all(isinstance(t, Memory) for t in args), node, "Plain typed move required")
        dst, src = args
        if (dst.value.type.space, src.value.type.space) == ("ub", "l0c"):
            attrs = ctx.attrs(node, {"acc_to_vec_mode"})
            o.need(o.mixed and o.side == "cube" and attrs.get("acc_to_vec_mode") == 2
                   and type(attrs.get("acc_to_vec_mode")) is int, node, "Only explicit split-M AIC-to-both-AIV moves are admitted")
            o.need(dst.root is None, node, "Split-M moves into tile aliases need a same-buffer rule")
            o.need(src.value.type.dtype.name == dst.value.type.dtype.name == "f32"
                   and src.shape[0] % 2 == 0 and dst.shape == (src.shape[0] // 2, src.shape[1])
                   and src.valid == src.shape and dst.valid == dst.shape and dst.pitch is None, node,
                   "Split-M requires exact full FP32 source and per-AIV destination extents")
            ctx.emit("dma.l0c_to_ub", node, (dst.value, src.value), attrs={
                "M": src.shape[0], "N": src.shape[1], "M_src": src.shape[0], "N_dst": dst.shape[1],
                "dual_mode": Ident("splitm"), "relu": False})
            return None
        ctx.attrs(node)
        spaces = (dst.value.type.space, src.value.type.space)
        full = src.shape == dst.shape and src.valid == src.shape and dst.valid == dst.shape
        if spaces == ("bt", "l1"):  # TMOV Mat -> Bias widens a flat FP16 row (bias_chain on A5).
            o.need(o.side == "cube" and full and src.value.type.layout != "nz" and src.value.type.dtype.name == "f16",
                   node, "Bias moves need a whole one-row FP16 Mat on the cube side")
            ctx.emit("dma.l1_to_bt", node, (dst.value, src.value), attrs={"n": src.shape[1]})
            return None
        if spaces == ("l1", "ub"):  # TMOV Vec -> Mat copies bytes into the Mat's NZ blocks, or its flat row
            size = prod(src.shape) * src.value.type.dtype.bits // 8  # (vec_mat_move_probe, vec_mat_row_probe on A5).
            o.need(o.side == "vec" and full and src.root is None and src.pitch is None and not dst.transposed and src.value.type.dtype == dst.value.type.dtype and size % 32 == 0, node,
                   "Vec-to-Mat moves need whole equal-shape Vec and Mat tiles of one dtype on the vector side")
            ctx.emit("dma.ub_to_l1", node, (dst.value, src.value),
                     attrs={"n_burst": 1, "burst_len": size // 32, "src_stride": 0, "dst_stride": 0})
            return None
        o.need(spaces in {("l0a", "l1"), ("l0b", "l1")} and full and src.value.type.layout == "nz"
               and src.value.type.dtype == dst.value.type.dtype, node, "Only full equal-coordinate Mat-to-Left/Right moves are admitted")
        # Extents describe the L1 tile before transpose. Right tiles, and ZN Mats, are typed reversed, so a
        # move transposes exactly when one side is (mat_zn_alias, mat_zn_load_probe on A5).
        m, n = src.value.type.dims
        ctx.emit("dma.l1_to_l0", node, (dst.value, src.value), attrs={
            "m_src": m, "n_src": n, "m_dst": m, "n_dst": n,
            "src_row0": 0, "src_col0": 0, "src_is_transpose": (spaces[0] == "l0b") != src.transposed,
            "dst_position": Ident(spaces[0])})
        return None
    if name == "block.insert":
        from .selection import interval
        from .views import Fractals

        ctx.attrs(node)
        o.need(len(args) == 4, node, "Expected a destination, a source and two insert offsets")
        dst, src, row, col = args
        packed = isinstance(src, Fractals)
        src = src.memory if packed else src
        o.need(isinstance(dst, Memory) and isinstance(src, Memory) and dst.value.type.space == "l1", node,
               "Only paired-side UB-to-L1 and cube-side Acc-to-Mat inserts are admitted")
        acc = src.value.type.space == "l0c"
        o.need(acc and o.side == "cube" or src.value.type.space == "ub" and o.mixed and o.side == "vec", node,
               "Only paired-side UB-to-L1 and cube-side Acc-to-Mat inserts are admitted")
        o.need(src.root is None or packed, node, "Inserts from tile aliases need a same-buffer rule")
        o.need(dst.value.type.layout == "nz" and not dst.transposed and dst.value.type.dtype.name == "f16"
               and dst.valid == dst.shape and src.valid == src.shape and src.pitch is None, node,
               "Inserts need whole sources and a whole FP16 NZ Mat destination")
        if acc:  # TINSERT Acc -> Mat casts whole FP32 rows over the Mat's width (acc_insert on A5; A5-UP-007).
            o.need(src.shape[1] == dst.shape[1], node, "Acc-to-Mat inserts need whole rows of the Mat's width")
        else:  # ND sources need whole 16-aligned tiles; NZ alias fractal columns are M_src rows of C0 elements.
            o.need(src.value.type.dtype.name == "f16" and (src.shape[1] % 16 == 0 if packed else all(n % 16 == 0 for n in src.shape)),
                   node, "UB-to-L1 insert needs full FP16 ND/NZ tiles")
        row = ctx.snapshot(row, node)
        bounds = interval(ctx, row)
        o.need(type(col) is int and col == 0 and bounds is not None and 0 <= bounds[0] <= bounds[1]
               and bounds[1] + src.shape[0] <= dst.shape[0] and src.shape[1] <= dst.shape[1], node,
               "Insert needs a proven in-bounds row window and zero column offset")
        view = ctx.emit("mem.slice", node, (dst.value,), typ=MemType("l1", dst.value.type.dtype, src.shape, "nz"),
                        attrs={"offsets": [row, 0], "extents": list(src.shape)})
        m, n = src.shape
        if acc:
            ctx.emit("dma.l0c_to_l1", node, (view, src.value), attrs={"M": m, "N": n, "M_src": m, "M_dst": dst.shape[0], "relu": False})
        elif packed:  # TINSERT from an NZ alias reads compact fractals (ub_nz_publish on A5).
            ctx.emit("dma.ub_to_l1.nz", node, (view, src.value), attrs={
                "m_src": m, "n_src": n, "m_dst": dst.shape[0], "n_dst": dst.shape[1], "M_src": m, "src_row0": 0, "src_col0": 0,
                "dst_row0": row, "dst_col0": 0})
        else:
            ctx.emit("dma.ub_to_l1.nd2nz", node, (view, src.value), attrs={
                "m_src": m, "n_src": n, "m_dst": dst.shape[0], "n_dst": dst.shape[1], "N_src": n})
        return None
    if name in {"block.matmul", "block.matmul_acc", "block.matmul_bias"}:
        ctx.attrs(node)
        o.need(len(args) == (3 if name == "block.matmul" else 4) and all(isinstance(t, Memory) for t in args),
               node, "Unadmitted matmul operands")
        dst, left, right = (args[0], args[2], args[3]) if name == "block.matmul_acc" else args[:3]
        if name == "block.matmul_acc":
            o.need(args[0].value == args[1].value, node, "matmul_acc needs an in-place accumulator")
        o.need((dst.value.type.space, left.value.type.space, right.value.type.space) == ("l0c", "l0a", "l0b")
               and left.shape[1] == right.shape[0] and dst.shape == (left.shape[0], right.shape[1])
               and all(t.valid == t.shape for t in (dst, left, right)), node, "Matmul shapes/spaces/valid extents disagree")
        attrs = {"M": dst.shape[0], "N": dst.shape[1], "K": left.shape[1], "is_init": name != "block.matmul_acc"}
        if name == "block.matmul_bias":  # TMATMUL_BIAS adds the table to every product row (bias_chain on A5).
            bias = args[3]
            o.need(bias.value.type.space == "bt" and bias.shape == (1, dst.shape[1]) and bias.valid == bias.shape, node,
                   "Bias matmul needs a whole bias table of the product's columns")
            attrs["bias"] = bias.value
        ctx.emit("cube.mmad", node, (dst.value, left.value, right.value), attrs=attrs)
        return None
    if name == "block.expands":
        ctx.attrs(node)
        o.need(len(args) == 2 and isinstance(args[0], Memory) and args[0].value.type.space == "l1"
               and o.side == "cube", node, "Only cube-side Mat fills are admitted")
        tile, value = args
        dt = tile.value.type.dtype
        # TEXPANDS takes its repeat count from the whole tile capacity, never the valid extent. A one-row Mat is a
        # flat row whose fill stops at its own blocks (mat_row_fill_probe on A5).
        o.need(tile.valid == tile.shape, node, "Mat fill needs a valid extent equal to the tile shape")
        o.need(not tile.transposed, node, "Mat fills need an NZ-declared tile or a one-row Mat")
        o.need(type(value) in (int, float), node, "Mat fill value must be a static literal")
        o.need(dt.name == "f16" and half_exact(value), node, "Mat fill literal must be exactly representable in FP16")
        ctx.emit("dma.set_constant_to_l1", node, (tile.value,), attrs={
            "val": value, "n_blocks": prod(tile.shape) * dt.bits // 8 // 32})
        return None
    o.fail(node, f"No instruction conversion for {name}")
    return None
