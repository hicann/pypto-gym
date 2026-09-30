# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Kernel-level op handlers of the reference interpreter beyond the M2 core set: explicit DMA
instructions, fixpipe quantisation on L0C stores, explicit mmad, conv2d, workspace, scalar
memory access, SIMT intrinsics and the synchronisation ops that have a functional effect.

On-chip tensors are logical row-major tensors (D-022), so every DMA is a window copy; the
byte-granular ``*_pad`` / raw block copies operate on the flat byte storage exactly as the old
``pipe_cube.py`` / ``pipe_vec.py`` did, including the 32-byte tail rules.
"""

from __future__ import annotations

from dataclasses import replace
import math
import struct
from typing import Any

import torch

from ...ir.core import Op
from ...ir.types import DType, MemType


def _align32(n: int) -> int:
    return (n + 31) // 32 * 32


def _binary32_sum(old: float, value: float, kind: str) -> float:
    """FP32 atomic add/sub: one binary32 rounding with IEEE zero signs; overflow gives infinity."""
    import numpy as np

    with np.errstate(over="ignore", invalid="ignore"):
        a, b = np.float32(old), np.float32(value)
        return float(a + b if kind == "add" else a - b)


def _binary32_extremum(old: float, value: float, kind: str) -> float:
    """FP32 atomic max/min as measured on A5: a NaN `value` keeps `old`, a NaN `old` takes `value`, and -0 < +0."""
    if value != value or old != old:
        return old if value != value else value
    a, b = (math.copysign(1.0, old), math.copysign(1.0, value)) if old == value else (old, value)
    return value if (b > a if kind == "max" else b < a) else old


class Binary32NaN(float):
    """A NaN read from FP32 memory with its bits; converting it to a binary64 float would quiet a signaling NaN."""

    bits: int

    def __new__(cls, bits: int) -> Binary32NaN:
        nan = super().__new__(cls, "nan")
        nan.bits = bits
        return nan


def f32_element(flat: torch.Tensor, index: int) -> float:
    """Element `index` of FP32 memory; a NaN keeps its bits (RFC-0001 §6.14)."""
    value = flat[index].item()
    return Binary32NaN(flat.view(torch.int32)[index].item() & 0xFFFFFFFF) if value != value else value


def f32_bits(value: float) -> int:
    return value.bits if isinstance(value, Binary32NaN) else torch.tensor(value, dtype=torch.float32).view(torch.int32).item() & 0xFFFFFFFF


def f32_nan_store(flat: torch.Tensor, index: int, value: Any) -> bool:
    """Store a NaN read from FP32 memory with its bits; False leaves any other value to the caller."""
    if not isinstance(value, Binary32NaN) or flat.dtype != torch.float32:
        return False
    flat.view(torch.int32)[index] = value.bits - (value.bits >> 31 << 32)
    return True


def fp19_trunc(scale: float) -> float:
    """The hardware float19 scale: keep the top 19 bits of the fp32 pattern."""
    bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0] & 0xFFFFE000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def fixpipe_quant(decoded: torch.Tensor, dst: DType, scale: Any, offset: Any, hif8_hybrid: bool) -> torch.Tensor:
    """The a5 fixpipe scalar quantisation on an L0C store (old ``_apply_fixpipe_quant``)."""
    x = decoded.to(torch.float32)
    scale = 1.0 if scale is None else float(scale)
    offset = 0 if offset is None else int(offset)
    if dst.name == "hif8":
        from ...dtypes.hif8_codec import fp32_to_hif8

        return fp32_to_hif8(x * fp19_trunc(scale), round_mode="hybrid" if hif8_hybrid else None)
    if dst.name in ("i8", "u8"):
        r = torch.round(x * fp19_trunc(scale))
        r = torch.clamp(r, -256.0, 255.0) + float(offset)
        if dst.name == "i8":
            return torch.clamp(r, -128.0, 127.0).to(torch.int8)
        return torch.clamp(r, 0.0, 255.0).to(torch.uint8)
    if dst.name in ("f16", "bf16", "f32", "e4m3", "e5m2"):
        scaled = x * fp19_trunc(scale)
        result = scaled.to(_torch(dst))
        if dst.name == "e4m3":
            # A5 FIX finite overflow has canonical NaN fields (M10-050).
            # Keep signed finite/underflow bytes and unqualified nonfinite inputs unchanged.
            overflow = torch.isfinite(scaled) & (scaled.abs() > 464)
            result.view(torch.uint8).masked_fill_(overflow, 0x7F)
        return result
    return decoded.to(_torch(dst))


def _torch(dt: DType) -> torch.dtype:
    from .interp import torch_dtype

    return torch_dtype(dt)


class DmaOps:
    """Mixin of :class:`Interp`."""

    # -- helpers --------------------------------------------------------------------------------------

    def _record_l0c_geometry(self, ref: Any, rows: int, cols: int) -> None:
        key = (ref.base, ref.slot, ref.offsets)
        self.l0c_geometry[key] = ((rows + 15) // 16 * 16, cols)

    def _check_l0c_pitch(self, op: Op, ref: Any, rows: int, cols: int) -> None:
        declared = int(self.val(op.attrs["M_src"])) if "M_src" in op.attrs else self.m.resolve_base(ref, self.lane).shape[0]
        declared = (declared + 15) // 16 * 16
        r, c = ref.offsets
        for (base, slot, (written_r, written_c)), (pitch, written_n) in self.l0c_geometry.items():
            if (base, slot) != (ref.base, ref.slot) or pitch == declared:
                continue
            row_overlap = max(r, written_r) < min(r + rows, written_r + pitch)
            later_block = max(c, written_c + 16) < min(c + cols, written_c + written_n)
            if row_overlap and later_block:
                where = f" at {op.loc.chain[0]}" if op.loc else ""
                raise self._err(f"physical L0C pitch remapping is not modeled: FIX M_src={declared}, computed pitch={pitch}{where}", op)

    def _byte_view(self, ref: Any) -> tuple[torch.Tensor, int]:
        """(flat uint8 storage, byte origin of the window)."""
        flat, origin = self.m.storage(ref, self.lane)
        return flat.view(torch.uint8), origin * flat.element_size()

    def _check_ub_ranges(self, op: Op, ref: Any, ranges, *, sub: int | None = None) -> None:
        """Validate actual UB instruction starts and full byte spans before access."""
        if ref.space != "ub":
            return
        flat, _ = self.m.storage(ref, self.lane, sub=sub)
        capacity = flat.numel() * flat.element_size()
        checked = []
        for start, size in ranges:
            if size == 0:
                continue
            if size < 0 or start < 0 or start + size > capacity:
                raise self._err(f"UB physical footprint [{start}, {start + size}) exceeds "
                                f"the {capacity}-byte allocation %{ref.base} at {op.loc}", op)
            if start % 32:
                raise self._err(f"UB access at byte {start} is not 32-byte aligned "
                                f"in %{ref.base} at {op.loc}", op)
            checked.append((start, size))
        if op.opcode.startswith("dma."):
            # The pipe trace consumes the same mapped ranges, not the logical
            # operand view (which may be a narrower descriptor carrier).
            merged = []
            for start, size in sorted(checked):
                if merged and start <= merged[-1][0] + merged[-1][1]:
                    previous, extent = merged[-1]
                    merged[-1] = (previous, max(previous + extent, start + size) - previous)
                else:
                    merged.append((start, size))
            lane = max(self.lane.sub, 0) if sub is None else sub
            self.ub_dma_ranges[(id(op), id(ref), lane)] = merged

    def _check_ub_window(self, op: Op, ref: Any, rows: int, cols: int, *, sub: int | None = None,
                         row_stride: int | None = None) -> None:
        """ND logical extents use their physical row pitch and rounded UB blocks."""
        if ref.space != "ub" or rows == 0 or cols == 0:
            return
        flat, origin = self.m.storage(ref, self.lane, sub=sub)
        base = self.m.resolve_base(ref, self.lane, sub=sub)
        pitch = row_stride if row_stride is not None else (base.stride()[-2] if base.dim() > 1 else base.numel())
        size = _align32(cols * flat.element_size())
        self._check_ub_ranges(op, ref, (((origin + row * pitch) * flat.element_size(), size)
                                       for row in range(rows)), sub=sub)

    def _err(self, msg: str, op: Op) -> Exception:
        from .interp import SimError

        return SimError(f"{msg} (#{op.id})")

    def _riders(self, op: Op) -> dict[str, Any]:
        return {
            "relu": bool(op.attrs.get("relu", False)),
            "scale": self.val(op.attrs["scale"]) if "scale" in op.attrs else None,
            "offset": self.val(op.attrs["offset"]) if "offset" in op.attrs else None,
            "hif8_hybrid": bool(op.attrs.get("hif8_hybrid", False)),
        }

    def _quant(self, decoded: torch.Tensor, dst_dtype: DType, r: dict[str, Any]) -> torch.Tensor:
        if r["relu"]:
            decoded = torch.clamp(decoded.to(torch.float32), min=0)
        return fixpipe_quant(decoded, dst_dtype, r["scale"], r["offset"], r["hif8_hybrid"])

    def _store_gm(self, op: Op, d: torch.Tensor, values: torch.Tensor) -> None:
        """Write ``values`` into the GM window ``d`` honouring an atomic attribute."""
        atomic = op.attrs.get("atomic")
        if atomic is None:
            d.copy_(values.to(d.dtype))
            return
        kind = str(getattr(atomic, "name", atomic))
        with self.m.gm_lock:
            cur = d.to(torch.float32) if d.dtype in (torch.float16, torch.bfloat16) else d.to(torch.int32) if d.dtype in (torch.int8, torch.uint8, torch.int16) else d
            v = values.to(cur.dtype)
            out = cur + v if kind == "add" else torch.maximum(cur, v) if kind == "max" else torch.minimum(cur, v)
            d.copy_(out.to(d.dtype))

    # -- generic copy with riders (dma.copy) --------------------------------------------------------------

    def _copy_l0c(self, op: Op, dst: Any, src: Any) -> None:
        m = self.m
        lane = self.lane
        r = self._riders(op)
        s = m.resolve(src, lane)
        if dst.space == "gm":
            if bool(op.attrs.get("transpose", False)):
                rows, cols = dst.extents[-2] if len(dst.extents) > 1 else 1, dst.extents[-1]
                self._check_l0c_pitch(op, src, cols, rows)
                d = m.resolve(dst, lane).reshape(rows, cols)
                logical = self._quant(s[:cols, :rows], dst.dtype, r).transpose(0, 1)
                self._store_gm(op, d, logical)
                return
            rows, cols = (dst.extents[-2], dst.extents[-1]) if len(dst.extents) > 1 else (1, dst.extents[-1])
            self._check_l0c_pitch(op, src, rows, cols)
            d = m.resolve(dst, lane).reshape(rows, cols)
            self._store_gm(op, d, self._quant(s[:rows, :cols], dst.dtype, r))
            return
        if dst.space == "l1":
            rows, cols = dst.extents
            self._check_l0c_pitch(op, src, rows, cols)
            d = m.resolve(dst, lane)
            v = s[:rows, :cols].to(d.dtype)
            if r["relu"]:
                v = torch.clamp(v, min=0)
            d[:rows, :cols].copy_(v)
            return
        if dst.space == "ub":
            mode = str(self.attr(op, "dual_mode", "splitm"))
            rows, cols = dst.extents
            self._check_l0c_pitch(op, src, rows * (2 if mode == "splitm" else 1), cols * (2 if mode not in ("single", "splitm") else 1))
            subs = (int(self.val(op.attrs["sub_block_id"])),) if mode == "single" else (0, 1)
            for sub in subs:
                self._check_ub_window(op, dst, rows, cols, sub=sub)
            if mode == "single":
                sub = int(self.val(op.attrs["sub_block_id"]))
                d = m.resolve(dst, lane, sub=sub)
                d.copy_(self._quant(s[:rows, :cols], dst.dtype, r).reshape(d.shape))
            elif mode == "splitm":
                for sub in (0, 1):
                    d = m.resolve(dst, lane, sub=sub)
                    d.copy_(s[sub * rows:(sub + 1) * rows, :cols].to(d.dtype))
            else:
                for sub in (0, 1):
                    d = m.resolve(dst, lane, sub=sub)
                    d.copy_(s[:rows, sub * cols:(sub + 1) * cols].to(d.dtype))
            return
        raise self._err(f"dma.copy l0c -> {dst.space} is not implemented", op)

    def op_dma_copy(self, op: Op) -> None:
        dst = self.raw(op.operands[0])
        src = self.raw(op.operands[1])
        pair = (src.space, dst.space)
        m = self.m
        lane = self.lane
        transpose = bool(op.attrs.get("transpose", False))
        if src.space == "l0c":
            self._copy_l0c(op, dst, src)
            return
        if pair in (("gm", "l1"), ("gm", "ub"), ("ws", "l1"), ("ws", "ub")):
            d = m.resolve(dst, lane)
            s = m.resolve(src, lane)
            if transpose:
                rows, cols = src.extents[-1], src.extents[-2] if len(src.extents) > 1 else 1
                self._check_ub_window(op, dst, rows, cols)
                d[:rows, :cols].copy_(s.reshape(cols, rows).transpose(0, 1).to(d.dtype))
            else:
                if src.gm_strides is not None and len(src.extents) > 2:  # a rank > 2 view lands row-major
                    rows = 1
                    for x in src.extents[:-1]:
                        rows *= x
                    cols = src.extents[-1]
                else:
                    rows, cols = (src.extents[-2], src.extents[-1]) if len(src.extents) > 1 else (1, src.extents[-1])
                self._check_ub_window(op, dst, rows, cols)
                d[:rows, :cols].copy_(s.reshape(rows, cols).to(d.dtype))
        elif pair == ("ub", "l1"):
            rows, cols = src.extents
            if src.layout == "nz":  # ``l1 <<= ub.nz()[...]``: the old ub_to_l1_nz with the window's geometry
                M_src = int(m.resolve_base(src, lane).shape[0])
                self._copy_nz_fractals(op, dst, src, rows, cols, M_src, src.offsets[0], src.offsets[1], dst.offsets[0], dst.offsets[1])
            else:
                self._check_ub_window(op, src, rows, cols)
                m.resolve(dst, lane)[:rows, :cols].copy_(m.resolve(src, lane))
        elif pair in (("ub", "gm"), ("ub", "ws")):
            rows, cols = (dst.extents[-2], dst.extents[-1]) if len(dst.extents) > 1 else (1, dst.extents[-1])
            self._check_ub_window(op, src, rows, cols)
            s = m.resolve(src, lane)[:rows, :cols]
            d = m.resolve(dst, lane)
            if tuple(d.shape) != (rows, cols):  # reshape copies a non-contiguous (mem.view) window - only flatten real rank changes
                d = d.reshape(rows, cols)
            self._store_gm(op, d, s)
        elif pair == ("ub", "ub"):
            rows, cols = src.extents
            self._check_ub_window(op, src, rows, cols)
            self._check_ub_window(op, dst, rows, cols)
            m.resolve(dst, lane)[:rows, :cols].copy_(m.resolve(src, lane))
        elif pair == ("l1", "bt"):
            d = m.resolve(dst, lane).reshape(-1)
            s = m.resolve(src, lane).reshape(-1)
            n = d.numel()
            d.copy_(s[:n].to(d.dtype))
        elif pair in (("l1", "l0a"), ("l1", "l0b")):
            s = m.resolve(src, lane)
            d = m.resolve(dst, lane)
            v = s.transpose(0, 1) if transpose else s
            d[: v.shape[0], : v.shape[1]].copy_(v)
        else:
            raise self._err(f"dma.copy {src.space} -> {dst.space} is not implemented", op)

    # -- byte-granular DMA ------------------------------------------------------------------------------

    def _pad_copy(self, op: Op, dst: Any, src: Any, n_burst: int, burst: int, src_step: int, dst_step: int,
                  tail: str, pad: torch.Tensor | None = None) -> None:
        dbytes, dorg = self._byte_view(dst)
        sbytes, sorg = self._byte_view(src)
        if min(n_burst, burst, src_step, dst_step) < 0:
            raise self._err(f"pad copy has a negative extent or step at {op.loc}", op)
        if not n_burst or not burst:
            return
        # Validate all bursts before the first write, including the UB port's
        # full final block. Logical short views do not truncate that block.
        for i in range(n_burst):
            s0, d0 = sorg + i * src_step, dorg + i * dst_step
            if s0 < 0 or d0 < 0 or s0 + burst > sbytes.numel() or d0 + burst > dbytes.numel():
                raise self._err(f"pad copy outside the storage at {op.loc}", op)
        self._check_ub_ranges(op, dst, ((dorg + i * dst_step, _align32(burst)) for i in range(n_burst)))
        self._check_ub_ranges(op, src, ((sorg + i * src_step, _align32(burst)) for i in range(n_burst)))
        element = self.m.storage(src, self.lane)[0].element_size()
        for i in range(n_burst):
            s0 = sorg + i * src_step
            d0 = dorg + i * dst_step
            dbytes[d0: d0 + burst] = sbytes[s0: s0 + burst]
            fill = _align32(burst) - burst
            if fill and d0 + burst + fill <= dbytes.numel():
                if pad is not None:
                    # an explicit pad value: the SPR holds one element's bytes and the tail repeats
                    # them from the burst's end (`set_mov_pad_val`, AscendC's DataCopyPadGm2UBImpl)
                    reps = fill // pad.numel() + 1
                    dbytes[d0 + burst: d0 + burst + fill] = pad.repeat(reps)[:fill]
                elif tail == "zero":
                    dbytes[d0 + burst: d0 + burst + fill] = 0
                elif self.m.profile.family == "a5":  # A5 repeats the burst's first element to the block end (I038)
                    first = sbytes[s0: s0 + min(burst, element)]
                    dbytes[d0 + burst: d0 + burst + fill] = first.repeat(fill // first.numel() + 1)[:fill]
                else:  # the unmeasured A2 rule of the old gm_to_ub_pad: what follows in GM (0x7F beyond its storage)
                    have = max(0, min(fill, sbytes.numel() - (s0 + burst)))
                    if have:
                        dbytes[d0 + burst: d0 + burst + have] = sbytes[s0 + burst: s0 + burst + have]
                    if have < fill:
                        dbytes[d0 + burst + have: d0 + burst + fill] = 0x7F

    def op_dma_gm_to_l1_pad(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        n, burst = int(self.val(op.attrs["n_burst"])), int(self.val(op.attrs["burst_len_byte"]))
        sstride, dstride = int(self.val(op.attrs.get("src_stride_byte", 0))), int(self.val(op.attrs.get("dst_stride", 0)))
        if n and burst:
            self._pad_copy(op, dst, src, n, burst, burst + sstride, _align32(burst) + dstride * 32, "zero")

    def op_dma_gm_to_ub_pad(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        n, burst = int(self.val(op.attrs["n_burst"])), int(self.val(op.attrs["burst_len_byte"]))
        sstride, dstride = int(self.val(op.attrs.get("src_stride_byte", 0))), int(self.val(op.attrs.get("dst_stride", 0)))
        pad = None
        if op.attrs.get("pad") is not None:
            from .interp import torch_dtype

            dt = dst.dtype if hasattr(dst, "dtype") else self._dtype_of(op.operands[0])
            pad = torch.tensor([self.val(op.attrs["pad"])], dtype=torch_dtype(dt)).view(torch.uint8)
        if n and burst:
            self._pad_copy(op, dst, src, n, burst, burst + sstride, _align32(burst) + dstride * 32, "first", pad)

    def op_dma_gm_to_ub_nd(self, op: Op) -> None:
        """The multi-dimensional GM -> UB DMA (AscendC NdDma). Index 0 is the innermost loop; every loop has a size, a
        source and a destination stride in elements and a left / right pad count (``config_left_pad`` /
        ``config_right_pad`` override the pads of every loop). A padded position takes ``constant_value``, or the
        nearest source element of that loop when ``nearest_value_mode``."""
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        dim = int(self.val(op.attrs["dim"]))

        def ints(key: str, default: int) -> list[int]:
            v = op.attrs.get(key)
            if v is None:
                return [default] * dim
            if len(v) != dim:
                raise self._err(f"gm_to_ub_nd {key} has {len(v)} entries for dim {dim}", op)
            return [int(self.val(x)) for x in v]

        size, sstride, dstride = ints("loop_size", 1), ints("loop_src_stride", 1), ints("loop_dst_stride", 1)
        lpad, rpad = ints("loop_left_pad", 0), ints("loop_right_pad", 0)
        for key, pads in (("config_left_pad", lpad), ("config_right_pad", rpad)):
            if op.attrs.get(key) is not None:
                pads[:] = [int(self.val(op.attrs[key]))] * dim
        if op.attrs.get("asc_optimize"):
            raise self._err("gm_to_ub_nd asc_optimize is not modelled", op)
        nearest = bool(self.val(op.attrs.get("nearest_value_mode", False)))
        constant = self.val(op.attrs.get("constant_value", 0))
        totals = [size[i] + lpad[i] + rpad[i] for i in range(dim)]
        if not all(totals):
            return
        dflat, dorg = self.m.storage(dst, self.lane)
        sflat, sorg = self.m.storage(src, self.lane)
        doff = torch.tensor(dorg, dtype=torch.int64)
        soff = torch.tensor(sorg, dtype=torch.int64)
        valid = torch.ones((), dtype=torch.bool)
        for i in range(dim):  # loop i runs along grid axis dim - 1 - i; the grids broadcast to the whole footprint
            shape = [1] * dim
            shape[dim - 1 - i] = totals[i]
            j = torch.arange(totals[i], dtype=torch.int64)
            logical = j - lpad[i]
            inside = (logical >= 0) & (logical < size[i])
            if nearest and size[i]:
                logical, inside = logical.clamp(0, size[i] - 1), torch.ones_like(inside)
            doff = doff + (j * dstride[i]).reshape(shape)
            soff = soff + (logical * sstride[i]).reshape(shape)
            valid = valid & inside.reshape(shape)
        doff, soff, valid = doff.reshape(-1), soff.reshape(-1), valid.reshape(-1)
        row_starts = doff.reshape(-1, totals[0])[:, 0]
        row_bytes = _align32(((totals[0] - 1) * dstride[0] + 1) * dflat.element_size())
        self._check_ub_ranges(op, dst, ((int(start) * dflat.element_size(), row_bytes) for start in row_starts))
        if int(doff.min()) < 0 or int(doff.max()) >= dflat.numel():
            raise self._err("gm_to_ub_nd dst element footprint exceeds the storage", op)
        picks = soff[valid]
        if picks.numel() and (int(picks.min()) < 0 or int(picks.max()) >= sflat.numel()):
            raise self._err("gm_to_ub_nd src element footprint exceeds allocated storage", op)
        values = torch.full(doff.shape, constant if dflat.is_floating_point() else int(constant), dtype=dflat.dtype)
        values[valid] = sflat[picks]
        dflat[doff] = values

    def op_dma_ub_to_gm_pad(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        n, burst = int(self.val(op.attrs["n_burst"])), int(self.val(op.attrs["burst_len_byte"]))
        sstride, dstride = int(self.val(op.attrs.get("src_stride", 0))), int(self.val(op.attrs.get("dst_stride_byte", 0)))
        src_step, dst_step = _align32(burst) + sstride * 32, burst + dstride
        dflat, dorg = self.m.storage(dst, self.lane)
        sflat, sorg = self.m.storage(src, self.lane)
        es = dflat.element_size()
        if min(n, burst, src_step, dst_step) < 0:
            raise self._err(f"ub_to_gm_pad has a negative extent or step at {op.loc}", op)
        if not n or not burst:
            return
        if burst % es:
            raise self._err("ub_to_gm_pad burst is not a multiple of the element size", op)
        self._check_ub_ranges(op, src, ((sorg * sflat.element_size() + i * src_step, _align32(burst)) for i in range(n)))
        for i in range(n):
            d0 = dorg * es + i * dst_step
            if d0 < 0 or d0 + burst > dflat.numel() * es:
                raise self._err(f"ub_to_gm_pad destination outside the storage at {op.loc}", op)
        for i in range(n):
            s0, d0, k = sorg + i * src_step // es, dorg + i * dst_step // es, burst // es
            self._store_gm(op, dflat[d0: d0 + k], sflat[s0: s0 + k])

    def _nz_placement(self, op: Op, ref: Any) -> tuple[torch.Tensor, int] | None:
        """Where a raw block copy's physical bytes sit in an NZ L1 tile's logical storage, and the window's physical
        byte origin (I026). Column block b of an [R, C] tile holds R rows of C0 elements: physical element
        b * R * C0 + r * C0 + j is logical (r, b * C0 + j), as vec_mat_move_probe measured on A5. That is a
        permutation only for 16-aligned R and C0-aligned C; ragged tiles keep storage order, which the online-MX
        scale planes (fractal-32 byte planes typed nz) are read in. c220 4-byte L1 tiles are ZZ (RFC-0008 §5)."""
        esize = ref.dtype.bits // 8
        if (ref.space != "l1" or getattr(op.operands[0].type, "layout", None) != "nz" or ref.dtype.bits % 8
                or ref.shape is not None or ref.gm_strides is not None or esize == 4 and self.m.profile.family == "a2"):
            return None
        base = self.m.resolve_base(ref, self.lane)
        c0 = 32 // esize
        if base.dim() != 2 or base.shape[0] % 16 or base.shape[1] % c0:
            return None
        rows, cols = base.shape
        within = torch.arange(rows * cols) % (rows * c0)
        logical = within // c0 * cols + torch.arange(rows * cols) // (rows * c0) * c0 + within % c0
        (row, col) = ref.offsets
        return (logical[:, None] * esize + torch.arange(esize)).reshape(-1), (col // c0 * rows * c0 + row * c0 + col % c0) * esize

    def _raw_blocks(self, op: Op, dst: Any, src: Any, n_burst: int, burst_len: int, src_stride: int, dst_stride: int) -> None:
        dbytes, dorg = self._byte_view(dst)
        sbytes, sorg = self._byte_view(src)
        placement = self._nz_placement(op, dst)
        if placement is not None:
            order, dorg = placement
        size = burst_len * 32
        if min(n_burst, burst_len, src_stride, dst_stride) < 0:
            raise self._err(f"block copy has a negative extent or stride at {op.loc}", op)
        if not n_burst or not size:
            return
        for i in range(n_burst):
            s0, d0 = sorg + i * (burst_len + src_stride) * 32, dorg + i * (burst_len + dst_stride) * 32
            if s0 < 0 or d0 < 0 or s0 + size > sbytes.numel() or d0 + size > dbytes.numel():
                raise self._err(f"block copy outside the storage at {op.loc}", op)
        self._check_ub_ranges(op, src, ((sorg + i * (burst_len + src_stride) * 32, size) for i in range(n_burst)))
        self._check_ub_ranges(op, dst, ((dorg + i * (burst_len + dst_stride) * 32, size) for i in range(n_burst)))
        for i in range(n_burst):
            s0, d0 = sorg + i * (burst_len + src_stride) * 32, dorg + i * (burst_len + dst_stride) * 32
            if placement is None:
                dbytes[d0: d0 + size] = sbytes[s0: s0 + size]
            else:
                dbytes[order[d0: d0 + size]] = sbytes[s0: s0 + size]

    def op_dma_gm_to_l1(self, op: Op) -> None:
        self._raw_blocks(op, self.raw(op.operands[0]), self.raw(op.operands[1]), int(self.val(op.attrs["n_burst"])), int(self.val(op.attrs["burst_len"])),
                         int(self.val(op.attrs.get("src_stride", 0))), int(self.val(op.attrs.get("dst_stride", 0))))

    def op_dma_ub_to_l1(self, op: Op) -> None:
        self.op_dma_gm_to_l1(op)

    def op_dma_ub_to_ub(self, op: Op) -> None:
        self.op_dma_gm_to_l1(op)

    def op_dma_ub_to_l1_nd2nz(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        m_src, n_src = int(self.val(op.attrs["m_src"])), int(self.val(op.attrs["n_src"]))
        pitch = int(self.val(op.attrs["N_src"])) if "N_src" in op.attrs else self.m.resolve_base(src, self.lane).shape[-1]
        self._check_ub_window(op, src, m_src, n_src, row_stride=pitch)
        d = self.m.resolve(dst, self.lane)
        flat, origin = self.m.storage(src, self.lane)
        s = flat.as_strided((m_src, n_src), (pitch, 1), origin)
        d[:m_src, :n_src].copy_(s[:m_src, :n_src])

    def op_dma_ub_to_l1_nz(self, op: Op) -> None:
        """UB already holds NZ fractals (the vector code packed them): fractal column ``b`` of the source is the
        contiguous run ``src_off0 + b * M_src * c0 + [0, m_src * c0)`` of ``m_src`` rows x ``c0`` columns; the
        logical L1 tile receives it as rows ``dst_row0..`` of columns ``dst_col0 + b * c0 ..`` (old ``_ub_to_l1_nz``)."""
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        m_src, n_src = int(self.val(op.attrs["m_src"])), int(self.val(op.attrs["n_src"]))
        M_src = int(self.val(op.attrs["M_src"])) if "M_src" in op.attrs else m_src
        src_row0, src_col0 = int(self.val(op.attrs.get("src_row0", 0))), int(self.val(op.attrs.get("src_col0", 0)))
        dst_row0, dst_col0 = int(self.val(op.attrs.get("dst_row0", 0))), int(self.val(op.attrs.get("dst_col0", 0)))
        self._copy_nz_fractals(op, dst, src, m_src, n_src, M_src, src_row0, src_col0, dst_row0, dst_col0)

    def _copy_nz_fractals(self, op: Op, dst: Any, src: Any, m_src: int, n_src: int, M_src: int, src_row0: int, src_col0: int,
                          dst_row0: int, dst_col0: int) -> None:
        c0 = 32 // max(dst.dtype.bits // 8, 1)
        base_src, _ = self.m.storage(src, self.lane)  # fractal addressing starts at the allocation's base
        d = self.m.resolve_base(dst, self.lane)
        src_block = M_src * c0
        src_off0 = (src_col0 // c0) * src_block + src_row0 * c0
        es = base_src.element_size()
        self._check_ub_ranges(op, src, (((src_off0 + b * src_block + src_col0 % c0) * es, m_src * c0 * es)
                                       for b in range(-(-n_src // c0))))
        for b in range(-(-n_src // c0)):
            run = base_src[src_off0 + b * src_block: src_off0 + b * src_block + m_src * c0].reshape(m_src, c0)
            d[dst_row0: dst_row0 + m_src, dst_col0 + b * c0: dst_col0 + (b + 1) * c0].copy_(run.to(d.dtype))

    def op_dma_gm_to_l1_nd2nz(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        M, N = int(self.val(op.attrs["M"])), int(self.val(op.attrs["N"]))
        d = self.m.resolve(dst, self.lane)
        s = self.m.resolve(src, self.lane).reshape(-1)[: M * N].reshape(M, N) if M * N == self.m.resolve(src, self.lane).numel() else None
        if s is None:
            N_src = int(self.val(op.attrs["N_src"]))
            flat, origin = self.m.storage(src, self.lane)
            s = flat[origin:].as_strided((M, N), (N_src, 1))
        d[:M, :N].copy_(s.to(d.dtype))

    def op_dma_gm_to_l1_dn2nz(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        M, N, N_src = int(self.val(op.attrs["M"])), int(self.val(op.attrs["N"])), int(self.val(op.attrs["N_src"]))
        flat, origin = self.m.storage(src, self.lane)
        s = flat[origin:].as_strided((M, N), (1, N_src))
        d = self.m.resolve(dst, self.lane)
        d[:M, :N].copy_(s.to(d.dtype))

    def op_dma_set_constant_to_l1(self, op: Op) -> None:
        t = self.raw(op.operands[0])
        flat, origin = self.m.storage(t, self.lane)
        val = self.val(op.attrs["val"])
        n = int(self.val(op.attrs["n_blocks"])) * (32 // flat.element_size())
        flat[origin: origin + n].fill_(val)

    def _l1_to_l0_physical(self, op: Op):
        """Physical byte-transpose source window and destination prefix, separate from logical values."""
        from .interp import SimError

        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        rows, cols = int(self.val(op.attrs["m_dst"])), int(self.val(op.attrs["n_dst"]))
        copied, columns = int(self.val(op.attrs["m_copy"])), _align32(cols)
        if src.dtype.bits != 8 or not op.attrs.get("src_is_transpose") or copied < rows or copied % 32:
            raise SimError(f"invalid byte-transpose m_copy at #{op.id} ({op.loc})")
        base = self.m.resolve_base(src, self.lane)
        if any(o < 0 or o + e > n for o, e, n in zip(src.offsets, (copied, columns), base.shape, strict=True)):
            raise SimError(f"byte-transpose physical load exceeds L1 allocation at #{op.id} ({op.loc})")
        raw_destination = self.lane.group.shared[(dst.base, dst.slot)]
        capacity = (raw_destination.numel() * raw_destination.element_size() + 511) // 512 * 512
        if any(dst.offsets) or copied * columns > capacity:
            raise SimError(f"byte-transpose physical load exceeds L0 slot at #{op.id} ({op.loc})")
        return replace(src, extents=(copied, columns)), dst, copied * columns

    def _l1_to_l0_physical_accesses(self, op: Op):
        from .pipesim import Access

        src, dst, size = self._l1_to_l0_physical(op)
        key = (dst.space, self.lane.group.index, dst.base, dst.slot, 0)
        return self._ranges(src, 0, "read", op.operands[1].name) + [Access("write", key, 0, size, name=op.operands[0].name)]

    def op_dma_l1_to_l0(self, op: Op) -> None:
        if "m_copy" in op.attrs:
            self._l1_to_l0_physical(op)
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        m_dst, n_dst = int(self.val(op.attrs["m_dst"])), int(self.val(op.attrs["n_dst"]))
        s = self.m.resolve(src, self.lane)[:m_dst, :n_dst]
        d = self.m.resolve(dst, self.lane)
        v = s.transpose(0, 1) if bool(op.attrs.get("src_is_transpose", False)) else s
        d[: v.shape[0], : v.shape[1]].copy_(v.to(d.dtype))

    def op_dma_l1_to_bt(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        n = int(self.val(op.attrs["n"]))
        d = self.m.resolve(dst, self.lane).reshape(-1)
        s = self.m.storage(src, self.lane)
        flat, origin = s
        d[:n].copy_(flat[origin: origin + n].to(d.dtype))

    def op_dma_l0c_to_ub(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        self._check_fix_bounds(op, src, dst)
        M, N = int(self.val(op.attrs["M"])), int(self.val(op.attrs["N"]))
        self._check_l0c_pitch(op, src, M, N)
        mode = str(self.attr(op, "dual_mode", "splitm"))
        r = self._riders(op)
        s = self.m.resolve(src, self.lane)
        ub_rows, ub_cols = (M // 2, N) if mode == "splitm" else ((M, N // 2) if mode != "single" else (M, N))
        subs = (int(self.val(op.attrs.get("sub_block_id", 0))),) if mode == "single" else (0, 1)
        pitch = int(self.val(op.attrs["N_dst"])) if "N_dst" in op.attrs else None
        for sub in subs:
            self._check_ub_window(op, dst, ub_rows, ub_cols, sub=sub, row_stride=pitch)
        for sub in subs:
            flat, origin = self.m.storage(dst, self.lane, sub=sub)
            backing = self.m.resolve_base(dst, self.lane, sub=sub)
            stride = pitch if pitch is not None else (backing.stride()[-2] if backing.dim() > 1 else backing.numel())
            if mode == "single":
                values = self._quant(s[:M, :N], dst.dtype, r)
            elif mode == "splitm":
                values = s[sub * ub_rows:(sub + 1) * ub_rows, :N].to(flat.dtype)
            else:
                values = s[:M, sub * ub_cols:(sub + 1) * ub_cols].to(flat.dtype)
            # Use the descriptor that bounds/trace validated, not resolve(dst)'s
            # logical parent pitch. A narrow view supplies the address origin;
            # it neither packs nor masks the explicit physical transfer.
            for row in range(ub_rows):
                start = origin + row * stride
                flat[start:start + ub_cols].copy_(values[row])

    def op_dma_l0c_to_l1(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        self._check_fix_bounds(op, src, dst)
        M, N = int(self.val(op.attrs["M"])), int(self.val(op.attrs["N"]))
        self._check_l0c_pitch(op, src, M, N)
        s = self.m.resolve(src, self.lane)[:M, :N]
        d = self.m.resolve(dst, self.lane)
        v = s.to(d.dtype)
        if bool(op.attrs.get("relu", False)):
            v = torch.clamp(v, min=0)
        d[:M, :N].copy_(v)

    def op_dma_l0c_to_gm_nz2nd(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        self._check_fix_bounds(op, src, dst)
        M, N = int(self.val(op.attrs["M"])), int(self.val(op.attrs["N"]))
        self._check_l0c_pitch(op, src, M, N)
        s = self.m.resolve(src, self.lane)[:M, :N]
        d = self.m.resolve(dst, self.lane).reshape(M, N)
        self._store_gm(op, d, self._quant(s, dst.dtype, self._riders(op)))

    def op_dma_l0c_to_gm_nz2dn(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        self._check_fix_bounds(op, src, dst)
        M, N = int(self.val(op.attrs["M"])), int(self.val(op.attrs["N"]))
        self._check_l0c_pitch(op, src, M, N)
        s = self.m.resolve(src, self.lane)[:M, :N]
        d = self.m.resolve(dst, self.lane).reshape(N, M)
        self._store_gm(op, d, self._quant(s, dst.dtype, self._riders(op)).transpose(0, 1))

    def op_dma_l0c_to_gm_nz2nz(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        self._check_fix_bounds(op, src, dst)
        M, N, M_pad = int(self.val(op.attrs["M"])), int(self.val(op.attrs["N"])), int(self.val(op.attrs["M_pad"]))
        self._check_l0c_pitch(op, src, M, N)
        c0g = max(16, 32 // max(dst.dtype.bits, 8) * 8 // 8) if False else max(16, 32 // (max(dst.dtype.bits, 8) // 8))
        logical = self._quant(self.m.resolve(src, self.lane)[:M, :N], dst.dtype, self._riders(op))
        flat, origin = self.m.storage(dst, self.lane)
        for c1 in range(N // c0g):
            block = logical[:, c1 * c0g:(c1 + 1) * c0g].reshape(-1)
            d0 = origin + c1 * M_pad * c0g
            flat[d0: d0 + block.numel()].copy_(block.to(flat.dtype))

    # -- cube ------------------------------------------------------------------------------------------------

    def _operand_matrix(self, ref: Any, rows: int, cols: int, transpose: bool) -> torch.Tensor:
        t = self.m.resolve(ref, self.lane)
        if ref.dtype.name == "i4":  # a byte window over the carrier: two signed nibbles per byte, low first
            raw = t.to(torch.int16)
            lo, hi = raw & 0xF, (raw >> 4) & 0xF
            v = torch.stack((lo, hi), dim=-1).reshape(*raw.shape[:-1], -1)
            t = (v - (v >= 8).to(v.dtype) * 16).to(torch.int32)
        if ref.dtype.name == "hif8":
            from ...dtypes.hif8_codec import hif8_to_fp32

            t = hif8_to_fp32(t)
        if transpose:
            return t[:cols, :rows].transpose(0, 1)
        return t[:rows, :cols]

    def _accumulate(self, D: torch.Tensor, prod: torch.Tensor, init: bool, bias: torch.Tensor | None) -> None:
        if bias is not None and init:
            prod = prod + bias.to(prod.dtype)
        if not init:
            prod = D.to(prod.dtype) + prod
        D.copy_(prod.to(D.dtype))

    def op_cube_matmul(self, op: Op) -> None:
        dst, a, b = (self.raw(x) for x in op.operands[:3])
        at, bt = bool(op.attrs.get("a_transpose", False)), bool(op.attrs.get("b_transpose", False))
        m = self.attr(op, "m") or (a.extents[1] if at else a.extents[0])
        n = self.attr(op, "n") or (b.extents[1] if bt else b.extents[0])
        k = self.attr(op, "k") or (a.extents[0] if at else a.extents[1])
        if a.dtype.name == "i4" and "k" not in op.attrs:
            k = int(k) * 2  # the i4 window counts bytes; logical K is two nibbles per byte
        m, n, k = int(m), int(n), int(k)
        if a.dtype.name == "i4":
            k = (k + 1) // 2 * 2  # C220 mad_s4 consumes the entire final byte (M10-099).
        self._record_l0c_geometry(dst, m, n)
        init = bool(op.attrs.get("init", True))
        splitn, splitk = self.attr(op, "splitn"), self.attr(op, "splitk")
        dt = torch.int32 if dst.dtype.is_integer else torch.float32
        A = self._operand_matrix(a, m, k, at).to(dt)
        B = self._operand_matrix(b, n, k, bt).to(dt)
        D = self.m.resolve(dst, self.lane)
        bias = None
        if "bias" in op.attrs:
            bref = self.raw(op.attrs["bias"])
            bias = self.m.resolve(bref, self.lane).reshape(-1)[:n]
        nblocks = [(0, n)] if not splitn else [(s, min(n - s, int(splitn))) for s in range(0, n, int(splitn))]
        kblocks = [(0, k)] if not splitk else [(s, min(k - s, int(splitk))) for s in range(0, k, int(splitk))]
        for ns, nw in nblocks:
            first = init
            for ks, kw in kblocks:
                prod = torch.matmul(A[:, ks:ks + kw], B[ns:ns + nw, ks:ks + kw].transpose(0, 1))
                self._accumulate(D[:m, ns:ns + nw], prod, first, bias[ns:ns + nw] if bias is not None else None)
                first = False

    def op_cube_mmad(self, op: Op) -> None:
        dst, a, b = (self.raw(x) for x in op.operands[:3])
        M, N, K = (int(self.val(op.attrs[x])) for x in ("M", "N", "K"))
        if a.dtype.name == "i4":
            K = (K + 1) // 2 * 2
        self._record_l0c_geometry(dst, M, N)
        init = bool(op.attrs.get("is_init", False))
        dt = torch.int32 if dst.dtype.is_integer else torch.float32
        A = self._operand_matrix(a, M, K, False).to(dt)
        B = self._operand_matrix(b, N, K, False).to(dt)
        D = self.m.resolve(dst, self.lane)
        bias = None
        if "bias" in op.attrs:
            bias = self.m.resolve(self.raw(op.attrs["bias"]), self.lane).reshape(-1)[:N]
        prod = torch.matmul(A, B.transpose(0, 1))
        self._accumulate(D[:M, :N], prod, init, bias)  # the operand view carries the offset; dst_row0/dst_col0 are for codegen

    @staticmethod
    def _im2col_window(F: torch.Tensor, h: int, w: int, c: int, c0: int, kh: int, kw: int, stride: tuple[int, int], dil: tuple[int, int],
                       pads: tuple[int, int, int, int], m0: int, k0: int, m_ext: int, k_ext: int) -> torch.Tensor:
        """Rows m0..m0+m_ext, columns k0..k0+k_ext of the virtual im2col matrix of an NC1HWC0 feature map ``F[c1, h, w, c0]``.

        Rows past ``ho * wo`` are the M tail -- the layout padding an NZ output plane must have. The
        hardware does NOT zero them: ``load3dv2`` keeps sliding the window by the same formula, so a
        tail row whose window still lands (partly) inside the feature map produces real data, and only
        a window fully outside it reads as zero. Modelling that is what lets the board be compared on
        every row instead of through a ``rows_valid_per_period`` mask (D-223; RFC-0008 §6 open item)."""
        stride_h, stride_w = stride
        dil_h, dil_w = dil
        pt, pb, pl, pr = pads
        ho = (h + pt + pb - dil_h * (kh - 1) - 1) // stride_h + 1
        wo = (w + pl + pr - dil_w * (kw - 1) - 1) // stride_w + 1
        m_idx = torch.arange(m0, m0 + m_ext)
        k_idx = torch.arange(k0, k0 + k_ext)
        oh, ow = m_idx // wo, m_idx % wo
        kc1 = k_idx // (kh * kw * c0)
        fkh = (k_idx % (kh * kw * c0)) // (kw * c0)
        fkw = (k_idx % (kw * c0)) // c0
        kc0 = k_idx % c0
        ih = oh[:, None] * stride_h - pt + fkh[None, :] * dil_h
        iw = ow[:, None] * stride_w - pl + fkw[None, :] * dil_w
        valid = (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w) & ((kc1 * c0 + kc0)[None, :] < c) & (kc1[None, :] < F.shape[0])
        ihc, iwc = ih.clamp(0, h - 1), iw.clamp(0, w - 1)
        kc1c = kc1.clamp(0, F.shape[0] - 1)
        gathered = F[kc1c[None, :].expand(m_ext, k_ext), ihc, iwc, kc0[None, :].expand(m_ext, k_ext)]
        return torch.where(valid, gathered, torch.zeros_like(gathered))

    def op_dma_l1_to_l0_img2col(self, op: Op) -> None:
        dst, fmap = self.raw(op.operands[0]), self.raw(op.operands[1])
        g = {k: int(self.val(op.attrs[k])) for k in ("h", "w", "c", "c0", "kh", "kw", "k0", "m0", "k_ext", "m_ext")}
        stride = (int(self.val(op.attrs.get("stride_h", 1))), int(self.val(op.attrs.get("stride_w", 1))))
        dil = (int(self.val(op.attrs.get("dil_h", 1))), int(self.val(op.attrs.get("dil_w", 1))))
        pads = tuple(int(self.val(op.attrs.get(k, 0))) for k in ("pad_t", "pad_b", "pad_l", "pad_r"))
        c0 = g["c0"]
        c1 = -(-g["c"] // c0)
        F = self.m.resolve(fmap, self.lane).reshape(-1)[: c1 * g["h"] * g["w"] * c0].reshape(c1, g["h"], g["w"], c0)
        win = self._im2col_window(F, g["h"], g["w"], g["c"], c0, g["kh"], g["kw"], stride, dil, pads, g["m0"], g["k0"], g["m_ext"], g["k_ext"])  # type: ignore[arg-type]
        d = self.m.resolve(dst, self.lane)
        d[: g["m_ext"], : g["k_ext"]].copy_(win.to(d.dtype))

    def op_cube_conv2d(self, op: Op) -> None:
        dst, fmap, weight = (self.raw(x) for x in op.operands[:3])
        g = {k: int(self.val(op.attrs[k])) for k in ("h", "w", "c", "cout", "kh", "kw")}
        stride_h, stride_w = int(op.attrs.get("stride_h", 1)), int(op.attrs.get("stride_w", 1))
        dil_h, dil_w = int(op.attrs.get("dil_h", 1)), int(op.attrs.get("dil_w", 1))
        pt, pb, pl, pr = (int(op.attrs.get(k, 0)) for k in ("pad_t", "pad_b", "pad_l", "pad_r"))
        m0 = int(self.val(op.attrs.get("m0", 0)))
        c0 = 64 if fmap.dtype.bits < 8 else 32 // (fmap.dtype.bits // 8)
        h, w, c, kh, kw = g["h"], g["w"], g["c"], g["kh"], g["kw"]
        c1 = -(-c // c0)
        ho = (h + pt + pb - dil_h * (kh - 1) - 1) // stride_h + 1
        wo = (w + pl + pr - dil_w * (kw - 1) - 1) // stride_w + 1
        K = c1 * kh * kw * c0
        tile_m = dst.extents[0]
        self._record_l0c_geometry(dst, tile_m, min(dst.extents[1], weight.extents[0]))
        cout_p = min(dst.extents[1], weight.extents[0])  # the mmad covers the 16-aligned cout of the L0C tile, padding rows included
        tile_k = int(self.val(op.attrs["tile_k"])) if "tile_k" in op.attrs else self._conv_tile_k(K, dst.extents[1], fmap.dtype)
        # img2col reads the L1 feature map as c1 contiguous [h, w, c0] planes from the tile's start; a tile allocated
        # for a larger static plane (conv_half_large: FM_PLANE rows) simply has unused rows after the c1 planes.
        F = self.m.resolve(fmap, self.lane).reshape(-1)[: c1 * h * w * c0].reshape(c1, h, w, c0)
        W = self._operand_matrix(weight, cout_p, K, False)
        m_idx = torch.arange(m0, m0 + tile_m)
        k_idx = torch.arange(K)
        oh, ow = m_idx // wo, m_idx % wo
        kc1 = k_idx // (kh * kw * c0)
        fkh = (k_idx % (kh * kw * c0)) // (kw * c0)
        fkw = (k_idx % (kw * c0)) // c0
        kc0 = k_idx % c0
        ih = oh[:, None] * stride_h - pt + fkh[None, :] * dil_h
        iw = ow[:, None] * stride_w - pl + fkw[None, :] * dil_w
        # the M tail slides by the formula like the hardware's; see _im2col_window
        valid = (ih >= 0) & (ih < h) & (iw >= 0) & (iw < w) & ((kc1 * c0 + kc0)[None, :] < c)
        ihc, iwc = ih.clamp(0, h - 1), iw.clamp(0, w - 1)
        gathered = F[kc1[None, :].expand(tile_m, K), ihc, iwc, kc0[None, :].expand(tile_m, K)]
        im2col = torch.where(valid, gathered, torch.zeros_like(gathered))
        D = self.m.resolve(dst, self.lane)
        bias = None
        if "bias" in op.attrs:
            bias = self.m.resolve(self.raw(op.attrs["bias"]), self.lane).reshape(-1)[:cout_p]
        first = True
        for k0 in range(0, K, tile_k):
            kk = min(tile_k, K - k0)
            prod = torch.matmul(im2col[:, k0:k0 + kk].to(torch.float32), W[:cout_p, k0:k0 + kk].to(torch.float32).transpose(0, 1))
            self._accumulate(D[:tile_m, :cout_p], prod, first, bias)
            first = False

    @staticmethod
    def _conv_tile_k(K: int, cout_p: int, dt: DType) -> int:
        """The old shortcut's automatic K tile (``_pick_tile_k``): the largest 16-multiple divisor of K
        whose [tile_k, cout_p] L0B tile fits one 32 KB slot."""
        esize = max(dt.bits, 8) // 8
        cap = 32 * 1024 // (cout_p * esize)
        for t in range(min(K, cap // 16 * 16), 0, -16):
            if K % t == 0:
                return t
        return 16

    # -- MX (microscaling): fp8 / fp4 operands with per-32-column power-of-two scales --------------------------------
    #
    # Scale tiles keep the old simulator's byte layout: 32-byte blocks in (row_tile, k_block) order, block = the
    # 16 rows x 2 columns window of the [rows, 2 * ceil(K / 64)] scale matrix, so scale[row, g] lives at byte
    # (row // 16 * k_blocks + g // 2) * 32 + g % 2 + 2 * (row % 16) and decodes to 2^(byte - 127).

    @staticmethod
    def _fp4_decode(carrier: torch.Tensor, dtype_name: str) -> torch.Tensor:
        """[rows, cols] uint8 carriers -> [rows, 2 * cols] fp32 (element 2j in the low nibble of byte j)."""
        from ...dtypes.fp4_fp32 import fp4_e1m2_to_fp32, fp4_e2m1_to_fp32

        decode = fp4_e1m2_to_fp32 if dtype_name == "fp4_e1m2" else fp4_e2m1_to_fp32
        return decode(carrier.contiguous().view(torch.uint8))

    @staticmethod
    def _fp4_encode_transposed(carrier: torch.Tensor) -> torch.Tensor:
        """Transpose a [rows, cols] carrier tile logically ([rows, 2 cols] -> [2 cols, rows]) and repack the nibbles."""
        u = carrier.contiguous().view(torch.uint8)
        n = torch.stack([u & 0x0F, u >> 4], dim=-1).reshape(u.shape[0], u.shape[1] * 2).transpose(0, 1)
        n = n.reshape(n.shape[0], n.shape[1] // 2, 2)
        return (n[..., 0] | (n[..., 1] << 4)).to(torch.uint8)

    def _mx_values(self, ref: Any, rows: int, cols: int, transpose: bool) -> torch.Tensor:
        """An MX operand window as fp32 [rows, cols] (cols = logical K); fp4 carriers are decoded first."""
        t = self.m.resolve(ref, self.lane)
        t = self._fp4_decode(t, ref.dtype.name) if ref.dtype.bits < 8 else t.float()
        return t[:cols, :rows].transpose(0, 1) if transpose else t[:rows, :cols]

    def _mx_scale_bytes(self, ref: Any, byte_offset: int, rows: int, k: int, src_k: int) -> torch.Tensor:
        """The L0 MX buffer (4 KB) filled from a scale tile in L1, as the old ``_copy_mx_scale_blocks`` did."""
        from .interp import MemRef

        base = self.m.resolve_base(ref, self.lane)
        whole = MemRef(ref.space, ref.dtype, ref.base, ref.slot, (0,) * len(ref.offsets), tuple(base.shape), ref.shape,
                       view_offset=ref.view_offset if ref.shape is not None else 0)
        flat, origin = self._byte_view(whole)
        src = flat[origin + byte_offset:]
        row_tiles, k_blocks, src_k_blocks = -(-rows // 16), -(-k // 64), -(-src_k // 64)
        out = torch.zeros(4096, dtype=torch.uint8)
        for rt in range(row_tiles):
            for kb in range(k_blocks):
                s0 = (rt * src_k_blocks + kb) * 32
                d0 = (rt * k_blocks + kb) * 32
                out[d0: d0 + 32] = src[s0: s0 + 32]
        return out

    @staticmethod
    def _mx_scale_codes(buf: torch.Tensor, rows: int, k: int) -> torch.Tensor:
        """The E8M0 code [row, group] of an L0 MX buffer (old ``_decode_mx_scale_buffer``)."""
        k_blocks = -(-k // 64)
        r = torch.arange(rows)[:, None]
        g = torch.arange(-(-k // 32))[None, :]
        return buf.to(torch.int64)[((r // 16) * k_blocks + g // 2) * 32 + (g % 2) + 2 * (r % 16)]

    def _l0_mx_codes(self, ref: Any, rows: int, k: int) -> torch.Tensor:
        buf = self.l0_mx.get((ref.base, ref.slot))
        if buf is None:
            buf = torch.full((4096,), 127, dtype=torch.uint8)  # never loaded: scale 1.0
        return self._mx_scale_codes(buf, rows, k)

    def _l0_mx_scale(self, ref: Any, rows: int, k: int) -> torch.Tensor:
        """scale[row, group] = 2^(code - 127)."""
        return torch.pow(2.0, (self._l0_mx_codes(ref, rows, k) - 127).to(torch.float32))

    @staticmethod
    def _mx_window(values: torch.Tensor, lead: torch.Tensor) -> torch.Tensor:
        """values truncated toward zero to the 24 bits from lead's leading bit, never finer than 2^-149 (I034)."""
        unit = torch.pow(2.0, torch.clamp(torch.frexp(lead)[1].to(torch.float64) - 24, min=-149))
        return torch.trunc(values / unit) * unit

    @staticmethod
    def _mx_product(a: torch.Tensor, ea: torch.Tensor, b: torch.Tensor, eb: torch.Tensor,
                    start: torch.Tensor | None = None) -> torch.Tensor:
        """A [M, K] @ B [N, K].T by K32 groups, each scaled by 2^(ea + eb - 254) with the exponents added first
        (I029), and added in ascending order to start: zero, the prior accumulator or a bias. A subnormal partial
        sum becomes a zero of its sign (I030); code 255 makes its group NaN, stored as 0x7FFFFFFF (mx_code_255).
        Every addition truncates (I034): a group's products, then its sum, then the scaled term and the partial
        sum keep the bits within 24 of the largest addend's leading bit. A5 measured small in-group products and a
        group term below the partial sum's window vanishing; opposite signs, carries and the exact width are
        model rules."""
        k = a.shape[1]
        acc = torch.zeros(a.shape[0], b.shape[0], dtype=torch.float64) if start is None else start.to(torch.float64)
        for g in range(-(-k // 32)):
            cols = slice(32 * g, min(32 * (g + 1), k))
            products = a[:, None, cols].double() * b[None, :, cols].double()
            group = DmaOps._mx_window(products, products.abs().amax(-1, keepdim=True)).sum(-1)
            left, right = ea[:, g, None], eb[None, :, g]
            scale = torch.pow(2.0, (left + right - 254).to(torch.float64))
            term = DmaOps._mx_window(group, group.abs()) * torch.where((left == 255) | (right == 255), torch.nan, scale)
            lead = torch.maximum(acc.abs(), term.abs())
            acc = DmaOps._mx_window(acc, lead) + DmaOps._mx_window(term, lead)
            acc = DmaOps._mx_window(acc, acc.abs())
            acc = torch.where(acc.abs() < torch.finfo(torch.float32).tiny, torch.copysign(torch.zeros_like(acc), acc), acc)
        out = acc.to(torch.float32)
        return torch.where(out.isnan(), torch.tensor(0x7FFFFFFF, dtype=torch.int32).view(torch.float32), out)

    def op_dma_gm_to_l1_mx_scale_nd2nz(self, op: Op) -> None:
        """Whole 16-row boxes of two groups; code 0 fills rows past ``rows`` in the last box (I030, mx_tail_code
        on A5) and, unmeasured, the group past an odd ``k_groups``."""
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        rows, k_groups = int(self.val(op.attrs["rows"])), int(self.val(op.attrs["k_groups"]))
        s = self.m.resolve(src, self.lane)[:rows, :k_groups].to(torch.uint8)
        row_tiles, half_cols = -(-rows // 16), -(-k_groups // 2)
        padded = torch.zeros((16 * row_tiles, 2 * half_cols), dtype=torch.uint8)
        padded[:rows, :k_groups] = s
        flat, origin = self._byte_view(dst)
        for rt in range(row_tiles):
            for hc in range(half_cols):
                d0 = origin + (rt * half_cols + hc) * 32
                flat[d0: d0 + 32] = padded[rt * 16:(rt + 1) * 16, hc * 2:(hc + 1) * 2].reshape(32)

    def op_dma_l1_to_l0_mx(self, op: Op) -> None:
        dst, src, src_mx = self.raw(op.operands[0]), self.raw(op.operands[1]), self.raw(op.attrs["src_mx"])
        m_dst, n_dst, m_src, n_src = (int(self.val(op.attrs[x])) for x in ("m_dst", "n_dst", "m_src", "n_src"))
        transposed = bool(op.attrs.get("src_is_transpose", False))
        scale_rows, scale_k, scale_src_rows, scale_src_k = (n_dst, m_dst, n_src, m_src) if transposed else (m_dst, n_dst, m_src, n_src)
        off = int(self.val(op.attrs.get("src_mx_offset_element", 0)))
        off += int(self.val(op.attrs.get("src_mx_row0", 0))) * 32 + int(self.val(op.attrs.get("src_mx_col0", 0))) * scale_src_rows
        self.l0_mx[(dst.base, dst.slot)] = self._mx_scale_bytes(src_mx, off, scale_rows, scale_k, scale_src_k)
        s = self.m.resolve(src, self.lane)
        d = self.m.resolve(dst, self.lane)
        if src.dtype.bits < 8:
            v = s[:m_dst, : n_dst // 2]
            v = self._fp4_encode_transposed(v) if transposed else v
        else:
            v = s[:m_dst, :n_dst]
            v = v.transpose(0, 1) if transposed else v
        d[: v.shape[0], : v.shape[1]].copy_(v.to(d.dtype))

    def op_cube_mmad_mx(self, op: Op) -> None:
        dst, a, b = (self.raw(x) for x in op.operands[:3])
        M, N, K = (int(self.val(op.attrs[x])) for x in ("M", "N", "K"))
        self._record_l0c_geometry(dst, M, N)
        D = self.m.resolve(dst, self.lane)
        bias = self.m.resolve(self.raw(op.attrs["bias"]), self.lane).reshape(-1)[:N] if "bias" in op.attrs else None
        start = self._mx_start(D[:M, :N], bool(op.attrs.get("is_init", False)), bias)
        D[:M, :N].copy_(self._mx_product(self._mx_values(a, M, K, False), self._l0_mx_codes(a, M, K),
                                         self._mx_values(b, N, K, False), self._l0_mx_codes(b, N, K), start))

    @staticmethod
    def _mx_start(D: torch.Tensor, init: bool, bias: torch.Tensor | None) -> torch.Tensor | None:
        """The first addend of an MX product: the prior accumulator (mx_acc_flush on A5), else a bias row, which
        no Pro form carries (a model rule), else zero."""
        if not init:
            return D
        return None if bias is None else bias.to(torch.float64).expand(D.shape)

    def op_cube_matmul_mx(self, op: Op) -> None:
        """The old shortcut without splits: l1_to_l0.mx for both operands, then one mmad.mx (M4 desugars it the same way)."""
        dst, a, b, sa, sb = (self.raw(x) for x in op.operands[:5])
        if self.attr(op, "splitk") or self.attr(op, "splitn"):
            raise self._err("matmul_mx with splitk / splitn is not implemented in the reference interpreter", op)

        def logical(ref: Any) -> tuple[int, int]:
            return ref.extents[0], ref.extents[1] * (2 if ref.dtype.bits < 8 else 1)

        at, bt = bool(op.attrs.get("a_transpose", False)), bool(op.attrs.get("b_transpose", False))
        la, lb = logical(a), logical(b)
        m = int(self.attr(op, "m") or (la[1] if at else la[0]))
        n = int(self.attr(op, "n") or (lb[1] if bt else lb[0]))
        k = int(self.attr(op, "k") or (la[0] if at else la[1]))
        self._record_l0c_geometry(dst, m, n)

        def operand(ref: Any, scale_ref: Any, rows: int, transpose: bool, shape: tuple[int, int]) -> tuple[Any, Any]:
            src_rows, src_k = (shape[1], shape[0]) if transpose else shape
            off = scale_ref.offsets[0] * 32 + scale_ref.offsets[1] * src_rows
            codes = self._mx_scale_codes(self._mx_scale_bytes(scale_ref, off, rows, k, src_k), rows, k)
            return self._mx_values(ref, rows, k, transpose), codes

        D = self.m.resolve(dst, self.lane)
        bias = self.m.resolve(self.raw(op.attrs["bias"]), self.lane).reshape(-1)[:n] if "bias" in op.attrs else None
        start = self._mx_start(D[:m, :n], bool(op.attrs.get("init", True)), bias)
        D[:m, :n].copy_(self._mx_product(*operand(a, sa, m, at, la), *operand(b, sb, n, bt, lb), start))

    # -- memory ------------------------------------------------------------------------------------------------

    def op_mem_workspace(self, op: Op) -> None:
        from .interp import MemRef, RingRef

        t = op.results[0].type
        if "gmbuff_slots" in op.attrs:  # a GMBuff ring: pieces ws:<name>:<i> were allocated before launch
            key0 = f"ws:{op.attrs['name']}:0"
            if key0 not in self.m.gm:
                raise self._err("workspace ring was not allocated before launch", op)
            elem = t.elem
            self.set_result(op, RingRef(str(op.attrs["name"]), int(op.attrs["gmbuff_slots"]),
                                        bool(op.attrs.get("gmbuff_per_core")), elem, tuple(self.m.gm[key0].shape)))
            return
        if not isinstance(t, MemType):
            raise self._err("workspace result must be a memory type", op)
        key = f"ws:{op.attrs['name']}"
        if key not in self.m.gm:
            raise self._err("workspace was not allocated before launch", op)
        dims = tuple(self.m.gm[key].shape)
        self.set_result(op, MemRef("gm", t.dtype, key, 0, (0,) * len(dims), dims))

    def op_mem_reshape(self, op: Op) -> None:
        from .interp import MemRef

        base = self.raw(op.operands[0])
        shape = tuple(int(self.val(x)) for x in self.attr(op, "shape"))
        origin = self._rebase_origin(op, base, base.dtype)  # row-major from the window's first element (RFC-0010 §10)
        if origin:
            whole = self.m.resolve_base(replace(base, shape=None), self.lane, 0 if base.space == "ub" and self.lane.sub < 0 else None)
            if origin + math.prod(shape) > whole.numel():
                raise self._err(f"mem.reshape to {list(shape)} at element {origin} passes the end of the "
                                f"{whole.numel()}-element storage at {op.loc}", op)
        self.set_result(op, MemRef(base.space, base.dtype, base.base, base.slot, (0,) * len(shape), shape, shape,
                                   view_offset=origin))

    def _check_fix_bounds(self, op: Op, src, dst) -> None:
        from ...ir.fix_bounds import ALIGN, Memory, align, fix_errors

        if self.m.profile.family != 'a5':
            return

        def memory(ref, sub=None):
            backing = self.m.resolve_base(ref, self.lane, sub=sub)
            capacity = backing.numel() * backing.element_size()
            if ref.space in ALIGN:
                capacity = align(capacity, ALIGN[ref.space])
            return Memory(ref.space, ref.dtype, tuple(backing.shape), tuple(ref.offsets),
                          (True,) * len(ref.offsets), capacity,  # a view's origin in bytes; a reshape's backing starts there
                          ref.view_offset * backing.element_size() if ref.gm_strides is not None else 0,
                          ref.gm_strides, ref.layout)

        mode = op.attrs.get('dual_mode', 'splitm')
        mode = getattr(mode, 'name', mode)
        subs = ([int(self.val(op.attrs.get('sub_block_id', 0)))] if mode == 'single' else [0, 1]) if dst.space == 'ub' else [None]
        messages = []
        for sub in subs:
            messages.extend(fix_errors(op, memory(src), memory(dst, sub),
                                      lambda x: None if x is None else int(self.val(x))))
        if messages:
            raise self._err(messages[0] + f' at {op.loc}', op)

    def op_scalar_load(self, op: Op) -> None:
        _, flat, index = self._scalar_address(op)  # an f32 NaN keeps its bits (RFC-0001 §6.16)
        self.set_result(op, f32_element(flat, index) if flat.dtype == torch.float32 else flat[index].item())

    def op_scalar_store(self, op: Op) -> None:
        _, flat, index = self._scalar_address(op)
        value = self.val(op.operands[2])
        if type(value) is int and not flat.dtype.is_floating_point:  # the store converts to the element width
            bits = 8 * flat.element_size()
            value &= (1 << bits) - 1
            value -= (1 << bits) if flat.dtype.is_signed and value >> (bits - 1) else 0
        if not f32_nan_store(flat, index, value):  # an f32 NaN keeps its bits (RFC-0001 §6.16)
            flat[index] = value

    def _scalar_address(self, op: Op):
        """Resolve exactly the physical element both execution and tracing access."""
        ref = self.raw(op.operands[0])
        flat, origin = self.m.storage(ref, self.lane)
        index = origin + int(self.val(op.operands[1]))
        if not 0 <= index < flat.numel():
            raise self._err(f"scalar address {index} outside allocation of {flat.numel()} elements", op)
        return ref, flat, index

    # -- simt -----------------------------------------------------------------------------------------------------

    def op_simt_block_idx(self, op: Op) -> None:
        self.set_result(op, self.lane.cube_idx if self.lane.cube_idx >= 0 else self.lane.vec_idx)

    def op_simt_block_num(self, op: Op) -> None:
        self.set_result(op, self.lane.cube_num if self.lane.cube_idx >= 0 else self.lane.vec_num)

    def op_simt_atomic(self, op: Op) -> None:
        ref = self.raw(op.operands[0])
        index = int(self.val(op.operands[1]))
        v = self.val(op.operands[2])
        kind = str(self.attr(op, "op", "add"))
        flat, origin = self.m.storage(ref, self.lane)
        lock = self.m.gm_lock if ref.space in ("gm", "ws") else self.lane.group.lock
        fp32 = flat.dtype == torch.float32
        with lock:
            old = f32_element(flat, origin + index) if fp32 else flat[origin + index].item()
            if kind in ("add", "sub") and fp32:
                new = _binary32_sum(old, v, kind)
            elif kind in ("max", "min") and fp32:
                new = _binary32_extremum(old, v, kind)
            elif kind == "cas" and fp32:  # RFC-0001 §6.14: compares bits
                new = v if f32_bits(old) == f32_bits(self.val(op.operands[3])) else old
            elif kind == "add":
                new = old + v
            elif kind == "sub":
                new = old - v
            elif kind == "max":
                new = max(old, v)
            elif kind == "min":
                new = min(old, v)
            elif kind == "exch":
                new = v
            elif kind in ("and", "or", "xor"):
                a, b = int(old), int(v)
                new = a & b if kind == "and" else a | b if kind == "or" else a ^ b
            elif kind == "cas":
                compare = self.val(op.operands[3])
                new = v if old == compare else old
            elif kind == "inc":  # CUDA ring semantics: wrap past the limit to 0
                new = 0 if old >= v else old + 1
            elif kind == "dec":  # CUDA ring semantics: wrap 0 (or beyond the limit) to the limit
                new = v if (old == 0 or old > v) else old - 1
            else:
                raise ValueError(f"simt.atomic: unknown op {kind!r}")
            if not flat.dtype.is_floating_point and flat.dtype != torch.bool:  # RFC-0001 §6.14: integers wrap
                bits = 8 * flat.element_size()
                new = int(new) & ((1 << bits) - 1)
                new -= (1 << bits) if flat.dtype.is_signed and new >> (bits - 1) else 0
            if not f32_nan_store(flat, origin + index, new):
                flat[origin + index] = new
        self.set_result(op, old)

    def op_core_set_sat_flag(self, op: Op) -> None:
        mode = str(getattr(op.attrs.get("mode"), "name", op.attrs.get("mode")))
        self.lane.sat_flags[mode] = bool(self.attr(op, "enable", True))
        self.lane.sat_written.add(mode)

    def op_core_get_sat_flag(self, op: Op) -> None:
        mode = str(getattr(op.attrs.get("mode"), "name", op.attrs.get("mode")))
        self.note_ctrl_entry(op, (mode,))
        self.set_result(op, int(self.lane.sat_flags[mode]))

    def op_core_clean_dcache(self, op: Op) -> None:
        """Value-neutral: cache lines are pipesim warnings (RFC-0006 §9, I012). The window must still resolve."""
        ref = self.raw(op.attrs["dst"])
        if getattr(ref, "space", None) != "gm":
            raise self._err("clean_dcache needs a GM window in 'dst'", op)
        self.m.storage(ref, self.lane)

    # -- synchronisation -----------------------------------------------------------------------------------------

    def op_sync_event(self, op: Op) -> None:
        self.set_result(op, ("event", op.id))

    def op_sync_set(self, op: Op) -> None:
        return None

    def op_sync_wait(self, op: Op) -> None:
        return None

    def op_sync_set_all(self, op: Op) -> None:
        return None

    def op_sync_release(self, op: Op) -> None:
        return None

    def op_sync_set_flag(self, op: Op) -> None:
        from ...ir.sync_rules import raw_flag_error

        message = raw_flag_error(op.attrs["src"], op.attrs["dst"], self.val(op.attrs["event_id"]))
        if message is not None:
            raise self._err(message, op)

    def op_sync_wait_flag(self, op: Op) -> None:
        self.op_sync_set_flag(op)

    def op_sync_barrier(self, op: Op) -> None:
        return None

    def op_atomic_begin(self, op: Op) -> None:
        return None

    def op_atomic_end(self, op: Op) -> None:
        return None

    def op_atomic_set_type(self, op: Op) -> None:
        return None

    def op_debug_dump(self, op: Op) -> None:
        return None

    def op_sync_crosscore_cube_ready(self, op: Op) -> None:
        self._cube_ready(int(self.val(op.attrs["flag_id"])))

    def op_sync_crosscore_vec_ready(self, op: Op) -> None:
        self._vec_ready(int(self.val(op.attrs["flag_id"])))

    def op_sync_crosscore_wait_cube(self, op: Op) -> None:
        self._wait_cube(int(self.val(op.attrs["flag_id"])), op)

    def op_sync_crosscore_wait_vec(self, op: Op) -> None:
        self._wait_vec(int(self.val(op.attrs["flag_id"])), op)

    def op_sync_crosscore_allvec_ready(self, op: Op) -> None:
        self._collective("vec", op, True)

    def op_sync_crosscore_allvec_wait(self, op: Op) -> None:
        self._collective("vec", op, False)

    def op_sync_crosscore_intracore_allvec_ready(self, op: Op) -> None:
        self._collective(f"vec@{self.lane.group.index}", op, True)

    def op_sync_crosscore_intracore_allvec_wait(self, op: Op) -> None:
        self._collective(f"vec@{self.lane.group.index}", op, False)

    def op_sync_crosscore_allcube_ready(self, op: Op) -> None:
        self._collective("cube", op, True)

    def op_sync_crosscore_allcube_wait(self, op: Op) -> None:
        self._collective("cube", op, False)

    def _collective(self, scope: str, op: Op, setting: bool) -> None:
        from .interp import _lane_name

        members, condition, produced, consumed = self.m.collectives[scope]
        flag = int(self.val(op.attrs["flag_id"]))
        self._crosscore_id(flag)
        flags = self.m.crosscore_flag_count
        indices = [i * flags + flag for i in members.values()]
        own = members[_lane_name(self.lane)] * flags + flag
        with condition:
            if setting:
                if produced[own] - min(consumed[i] for i in indices) >= self.m.profile.crosscore_counter_max:
                    raise self._err(f"collective flag {flag} exceeds {self.m.profile.crosscore_counter_max} pending generations", op)
                produced[own] += 1
                condition.notify_all()
            else:
                while any(produced[i] <= consumed[own] for i in indices):
                    # the shared wait: a member that never publishes is a deadlock within half a second, and the
                    # time limit with a member still computing is a timeout (this used to wait for the limit either way)
                    self._wait(condition, op, f" ({scope} collective flag {flag}: a member has not published it)")
                consumed[own] += 1
                condition.notify_all()

    # -- vector-unit masks (SPR) -----------------------------------------------------------------------------------

    # -- the sort family: fp32 (score, index) records of 8 bytes, descending by score ----------------------------

    def _records_out(self, dst: torch.Tensor, scores: torch.Tensor, ids: torch.Tensor) -> None:
        """Sort (score, index) pairs by descending score into ``dst`` as interleaved records. The tie order is
        torch's default (unstable) argsort — the old simulator's exact call, which the recorded goldens embody;
        how the silicon vbitsort orders ties is a T3 question (RFC-0008 §6)."""
        order = torch.argsort(scores, descending=True)
        dst[0::2] = scores[order]
        dst[1::2] = ids[order].view(torch.float32)

    def op_vec_sort32(self, op: Op) -> None:
        dst, src, idx = (self.raw(x) for x in op.operands[:3])
        repeat = int(self.val(op.attrs.get("repeat", 0)))
        dflat, dorg = self.m.storage(dst, self.lane)
        sflat, sorg = self.m.storage(src, self.lane)
        iflat, iorg = self.m.storage(idx, self.lane)
        if sflat.dtype != torch.float32 or dflat.dtype != torch.float32 or iflat.element_size() != 4:
            raise self._err("sort32 sorts fp32 scores with 32-bit indices", op)
        if sorg + 32 * repeat > sflat.numel() or iorg + 32 * repeat > iflat.numel() or dorg + 64 * repeat > dflat.numel():
            raise self._err("sort32 footprint exceeds the storage", op)
        for r in range(repeat):
            self._records_out(dflat[dorg + 64 * r: dorg + 64 * r + 64], sflat[sorg + 32 * r: sorg + 32 * r + 32],
                              iflat[iorg + 32 * r: iorg + 32 * r + 32].view(torch.int32))

    def _merge(self, op: Op, dst: torch.Tensor, lists: list[torch.Tensor]) -> None:
        block = torch.cat(lists)
        if block.dtype != torch.float32 or dst.dtype != torch.float32:
            raise self._err("the merge sorts fp32 records", op)
        even, odd = block[0::2].clone(), block[1::2].clone()
        order = torch.argsort(even, descending=True)  # the old simulator's exact call (unstable ties)
        dst[0::2] = even[order]
        dst[1::2] = odd[order]

    def op_vec_mergesort4(self, op: Op) -> None:
        dst, src = self.raw(op.operands[0]), self.raw(op.operands[1])
        lps, repeat = int(self.val(op.attrs["length_per_seq"])), int(self.val(op.attrs.get("repeat", 1)))
        dflat, dorg = self.m.storage(dst, self.lane)
        sflat, sorg = self.m.storage(src, self.lane)
        epr = 8 * lps  # four lists of lps records, two floats each
        if sorg + repeat * epr > sflat.numel() or dorg + repeat * epr > dflat.numel():
            raise self._err("mergesort4 footprint exceeds the storage", op)
        for r in range(repeat):
            self._merge(op, dflat[dorg + r * epr: dorg + (r + 1) * epr], [sflat[sorg + r * epr: sorg + (r + 1) * epr]])

    def op_vec_mergesort_2seq(self, op: Op) -> None:
        dst, src1, src2 = (self.raw(x) for x in op.operands[:3])
        s1, s2 = int(self.val(op.attrs["size1"])), int(self.val(op.attrs["size2"]))
        dflat, dorg = self.m.storage(dst, self.lane)
        f1, o1 = self.m.storage(src1, self.lane)
        f2, o2 = self.m.storage(src2, self.lane)
        if o1 + 2 * s1 > f1.numel() or o2 + 2 * s2 > f2.numel() or dorg + 2 * (s1 + s2) > dflat.numel():
            raise self._err("mergesort_2seq footprint exceeds the storage", op)
        self._merge(op, dflat[dorg: dorg + 2 * (s1 + s2)], [f1[o1: o1 + 2 * s1], f2[o2: o2 + 2 * s2]])

def _lane_label(lane: Any) -> str:
    return f"core{lane.cube_idx if lane.cube_idx >= 0 else lane.vec_idx}/{lane.side}{lane.sub if lane.sub >= 0 else ''}"
