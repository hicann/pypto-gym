# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 FIX descriptor byte ranges, independent of a printer or tensor runtime.

Unknown dimensions/offsets remain unknown. A rejection requires a provable
overrun of the backing allocation, not merely a view or producer-pitch mismatch.
"""

from dataclasses import dataclass, replace
from math import prod

from .core import Function, Ident, Literal, Op, Value
from .types import BufType, CellType, DType, MemType, ScalarType

FIX_OPS = frozenset({'dma.l0c_to_gm.nz2nd', 'dma.l0c_to_gm.nz2dn', 'dma.l0c_to_gm.nz2nz',
                     'dma.l0c_to_l1', 'dma.l0c_to_ub'})
ALIGN = {'l0c': 1024, 'l1': 32, 'ub': 32}


def align(value, size):
    return (value + size - 1) // size * size


def nz_end(rows, cols, pitch, c0):
    """Exclusive element offset of the last active value in an NZ transfer."""
    if rows == 0 or cols == 0:
        return 0
    return (cols - 1) // c0 * pitch * c0 + (rows - 1) * c0 + (cols - 1) % c0 + 1


@dataclass(frozen=True)
class Memory:
    space: str
    dtype: DType
    shape: tuple
    offsets: tuple
    kept: tuple
    capacity: int | None
    base: int | None = 0
    strides: tuple | None = None
    layout: str | None = None

    def origin(self, layout=None):
        """The window's first byte, in `layout` when a caller names one.

        A caller that measured an extent in one coordinate system has to read the origin in the
        same one, so `fix_errors` names it rather than trusting the declaration -- the `MemRef`
        the model builds from carries a layout only for NZ-packed UB windows (I044).
        """
        if self.base is None or any(x is None for x in self.offsets):
            return None
        if self.dtype.bits < 8:
            return None  # packed address coordinates need their own carrier contract
        width = self.dtype.bits // 8
        if not any(self.offsets):
            return self.base
        if layout is None:
            # A5 L0C is fractal however the model's logical tensor is shaped (D-022); L1 and UB
            # are addressed as declared, row-major unless an `.nz()` view says otherwise.
            layout = 'nz' if self.space == 'l0c' else self.layout
        if layout == 'nz' and self.space in ('l0c', 'l1', 'ub'):
            if len(self.offsets) != 2 or self.shape[0] is None:
                return None
            row, col = self.offsets
            pitch = self.shape[0] if self.space == 'ub' else align(self.shape[0], 16)
            c0 = 16 if self.space == 'l0c' else 32 // width
            return self.base + (col // c0 * pitch * c0 + row * c0 + col % c0) * width
        strides = self.strides
        if strides is None:
            if any(x is None for x in self.shape):
                return None
            strides = tuple(prod(self.shape[i + 1:]) for i in range(len(self.shape)))
        if any(x is None for x in strides):
            return None
        return self.base + sum(o * s for o, s in zip(self.offsets, strides, strict=True)) * width


def fix_errors(op: Op, src: Memory, dst: Memory, integer):
    """Check known read/write byte ranges; a zero-sized transfer touches no data."""
    rows, cols = integer(op.attrs.get('M')), integer(op.attrs.get('N'))
    errors = []
    for name, value in (('M', rows), ('N', cols)):
        if value is not None and value < 0:
            errors.append(f'{op.opcode}: {name} must not be negative, got {value}')
    if errors or rows is None or cols is None or rows == 0 or cols == 0:
        return errors

    def check(memory, extent, label, layout=None):
        """`layout` is the coordinate system `extent` was measured in, when that is not the one
        the memory declares; an origin read in another one is not comparable with it."""
        if extent is None or memory.capacity is None:
            return
        origin = memory.origin(layout)
        # Even an unknown valid origin cannot fit a transfer larger than its allocation.
        if extent > memory.capacity or (origin is not None and (origin < 0 or origin + extent > memory.capacity)):
            start = '?' if origin is None else origin
            errors.append(f'{op.opcode}: {label} FIX byte range starting at {start} with extent {extent} '
                          f'exceeds {memory.space.upper()} allocation capacity {memory.capacity}')

    source_pitch = integer(op.attrs.get('M_src')) if 'M_src' in op.attrs else src.shape[0]
    if source_pitch is not None:
        if source_pitch <= 0:
            errors.append(f'{op.opcode}: M_src must be positive for a nonempty transfer')
        else:
            # The FIX descriptor walks the accumulator's fractal, so this is an NZ footprint.
            check(src, nz_end(rows, cols, align(source_pitch, 16), 16) * (src.dtype.bits // 8),
                  'source', 'nz')

    code = op.opcode
    if dst.dtype.bits < 8:
        return errors  # the source is still checked; do not invent packed destination geometry
    width = dst.dtype.bits // 8
    stride_name = 'M_dst' if code in ('dma.l0c_to_l1', 'dma.l0c_to_gm.nz2dn') else 'N_dst'
    default_stride = dst.shape[0] if stride_name == 'M_dst' else dst.shape[-1]
    stride = integer(op.attrs.get(stride_name)) if stride_name in op.attrs else default_stride
    if code == 'dma.l0c_to_gm.nz2nz':
        stride = integer(op.attrs.get('M_pad')) if 'M_pad' in op.attrs else rows
    if stride is None:
        return errors
    if stride <= 0:
        errors.append(f'{op.opcode}: destination pitch must be positive for a nonempty transfer')
        return errors
    if code == 'dma.l0c_to_ub':
        mode = op.attrs.get('dual_mode', 'splitm')
        mode = mode.name if isinstance(mode, Ident) else mode
        if mode == 'splitm':
            rows //= 2
        elif mode == 'splitn':
            cols //= 2
    if code == 'dma.l0c_to_l1':
        extent = nz_end(rows, cols, align(stride, 16), 32 // width) * width
    elif code == 'dma.l0c_to_gm.nz2nz':
        if width == 1:
            return errors  # low-precision NZ FIX packing remains a separate form
        extent = nz_end(rows, cols, stride, 16) * width
    elif code == 'dma.l0c_to_gm.nz2dn':
        extent = ((cols - 1) * stride + rows) * width
    else:
        extent = ((rows - 1) * stride + cols) * width if rows and cols else 0
    # `dma.l0c_to_l1` lands an NZ block in a fractal L1; the other destinations are as their
    # extent above reads them, and GM is linear from the transfer's own start.
    check(dst, extent, 'destination', 'nz' if code == 'dma.l0c_to_l1' else None)
    return errors


class StaticMemory:
    """Read-only constants and allocation geometry for one function."""

    def __init__(self, function: Function):
        self.defs = {v.name: op for op in function.walk() for v in op.results}
        self.constants = {}
        self.memories = {}

    def integer(self, value, seen=frozenset()):
        if isinstance(value, Literal):
            value = value.value
        if isinstance(value, int):
            return value
        name = getattr(value, 'name', None)
        if name is None or name in seen:
            return None
        if isinstance(value, Value) and (isinstance(value.type, CellType) or
                isinstance(value.type, ScalarType) and not value.type.dtype.is_integer):
            return None
        if name in self.constants:
            return self.constants[name]
        op = self.defs.get(name)
        if op is None or not op.opcode.startswith('scalar.'):
            return None
        more = seen | {name}
        if op.opcode == 'scalar.const':
            result = self.integer(op.attrs.get('value'), more)
        else:
            args = [self.integer(x, more) for x in op.operands]
            if any(x is None for x in args):
                return None
            kind = op.opcode[7:]
            result = None
            if len(args) == 2:
                a, b = args
                if kind == 'add': result = a + b
                elif kind == 'sub': result = a - b
                elif kind == 'mul': result = a * b
                elif kind == 'min': result = min(a, b)
                elif kind == 'max': result = max(a, b)
                elif kind in ('div', 'mod', 'ceil_div') and b != 0:
                    from .scalar_math import integer_divmod, rounding
                    result = -(-a // b) if kind == 'ceil_div' else integer_divmod(a, b, rounding(op))[kind == 'mod']
                elif kind == 'and': result = a & b
                elif kind == 'cmp':
                    pred = getattr(op.attrs['pred'], 'name', op.attrs['pred'])
                    result = int({'lt': a < b, 'le': a <= b, 'gt': a > b, 'ge': a >= b,
                                  'eq': a == b, 'ne': a != b}[pred])
            elif len(args) == 3 and kind == 'select':
                result = args[1] if args[0] else args[2]
            elif len(args) == 1 and kind in ('cast', 'neg'):
                result = -args[0] if kind == 'neg' else args[0]
            elif len(args) == 1 and kind == 'align' and op.attrs['n'] > 0:
                result = -(-args[0] // op.attrs['n']) * op.attrs['n']
        dtype = getattr(op.results[0].type, 'dtype', None) if op.results else None
        if result is not None and dtype is not None and dtype.is_integer:
            result %= 1 << dtype.bits
            if dtype.name.startswith('i') and result >= 1 << (dtype.bits - 1):
                result -= 1 << dtype.bits
        self.constants[name] = result
        return result

    def memory(self, value, seen=frozenset()):
        if not isinstance(value, Value) or value.name in seen:
            return None
        if value.name in self.memories:
            return self.memories[value.name]
        kind = value.type.elem if isinstance(value.type, BufType) else value.type
        if not isinstance(kind, MemType):
            return None
        shape = tuple(self.integer(d) for d in kind.dims)
        op = self.defs.get(value.name)
        if op is None or op.opcode in ('mem.alloc', 'mem.workspace', 'list.item'):
            capacity = (prod(shape) * kind.dtype.bits + 7) // 8 if all(d is not None and d >= 0 for d in shape) else None
            if capacity is not None and kind.space in ALIGN:
                capacity = align(capacity, ALIGN[kind.space])
            result = Memory(kind.space, kind.dtype, shape, (0,) * len(shape), (True,) * len(shape),
                            capacity, layout=kind.layout)
        else:
            if not op.operands:
                return None
            base = self.memory(op.operands[0], seen | {value.name})
            if base is None:
                return None
            result = base
            if op.opcode == 'mem.slice':
                offsets, kept = list(base.offsets), list(base.kept)
                selected = [i for i, keep in enumerate(kept) if keep]
                local = op.attrs.get('offsets', ())
                masks = op.attrs.get('mask', [True] * len(local))
                if len(selected) != len(local) or len(masks) != len(local):
                    return None
                for i, offset, keep in zip(selected, local, masks, strict=True):
                    offset = self.integer(offset)
                    offsets[i] = offsets[i] + offset if offsets[i] is not None and offset is not None else None
                    kept[i] = bool(keep)
                result = replace(base, offsets=tuple(offsets), kept=tuple(kept))
            elif op.opcode in ('mem.reshape', 'mem.view') or op.opcode == 'mem.reinterpret' and 'tile' in op.attrs:
                origin = base.origin()
                strides = None
                if op.opcode == 'mem.view':
                    offset = self.integer(op.attrs.get('offset', 0))
                    origin = origin + offset * (kind.dtype.bits // 8) if origin is not None and offset is not None else None
                    strides = tuple(self.integer(x) for x in op.attrs['strides'])
                result = Memory(base.space, kind.dtype, shape, (0,) * len(shape), (True,) * len(shape),
                                base.capacity, origin, strides, kind.layout)
            elif op.opcode == 'mem.reinterpret':
                layout = kind.layout if 'layout' in op.attrs else base.layout
                if base.dtype.bits == kind.dtype.bits:
                    result = replace(base, dtype=kind.dtype, layout=layout)
                else:
                    # A plain reinterpret retains the backing row pitch. Only
                    # explicit reshape/tile operations start new coordinates.
                    old_bits, new_bits = max(base.dtype.bits, 8), max(kind.dtype.bits, 8)
                    selected = [i for i, kept in enumerate(base.kept) if kept]
                    if not base.shape:
                        return None
                    last = selected[-1] if selected else len(base.shape) - 1
                    dims, offsets = list(base.shape), list(base.offsets)
                    dims[last] = dims[last] * old_bits // new_bits if dims[last] is not None else None
                    offsets[last] = offsets[last] * old_bits // new_bits if offsets[last] is not None else None
                    result = replace(base, dtype=kind.dtype, shape=tuple(dims), offsets=tuple(offsets), layout=layout)
            elif op.opcode != 'mem.get_buf':
                return None
        self.memories[value.name] = result
        return result

    def errors(self, op):
        if op.opcode not in FIX_OPS or len(op.operands) != 2:
            return []
        dst, src = (self.memory(x) for x in op.operands)
        if src is None or dst is None or not src.shape or not dst.shape:
            return []
        return fix_errors(op, src, dst, self.integer)
