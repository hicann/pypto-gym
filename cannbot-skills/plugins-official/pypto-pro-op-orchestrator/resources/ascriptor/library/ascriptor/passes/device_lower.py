# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``device_lower``: select one DMA instruction for every ``dma.copy`` (the DSL's ``<<=``).

The rules are the old ``Tensor.__ilshift__`` / ``GMTensor.__ilshift__`` dispatch tables (research
notes ``tensor_sync_flow.md`` §1.3 / §1.4): the pair of memory spaces, the destination layout and the
riders (``transpose``, ``relu``, ``scale`` …) pick the instruction, and the parameters the old stubs
inferred from ``Tensor.shape / offset / span`` are computed from the view geometry (:func:`view_of`).
Dynamic parameters become ``scalar.*`` ops placed before the instruction. Every rewritten op keeps its
id and gains an ``origin`` entry naming this pass.
"""

from __future__ import annotations

from math import prod
from typing import Any

from ..ir import Ident, Module, Op, Value
from ..ir.builder import Rewriter
from ..ir.types import BufType, MemType
from .manager import Pass, PassContext, PassError
from .util import Defs, Emit, Scalar, View, rewrite_module, view_of

PASS = "device_lower"


def _c0(dt: Any) -> int:
    return 32 // max(dt.bits // 8, 1)


def _elem_bytes(dt: Any) -> int:
    return max(dt.bits // 8, 1)


class _Lowerer:
    def __init__(self, ctx: PassContext, rw: Rewriter, defs: Defs) -> None:
        self.ctx = ctx
        self.rw = rw
        self.defs = defs

    def err(self, op: Op, msg: str) -> PassError:
        return PassError(PASS, f"{msg} (#{op.id} at {op.loc})")

    def physical_l0(self, op: Op, e: Emit) -> Op:
        """Record byte-transpose padding before dependency analysis, never inside a printer."""
        if (op.opcode != "dma.l1_to_l0" or self.ctx.device.arch != "c310"
                or not op.attrs.get("src_is_transpose", False) or op.operands[1].type.dtype.bits != 8):
            return op
        rows, cols = e.value(op.attrs["m_dst"]), e.value(op.attrs["n_dst"])
        if isinstance(rows, int) and rows % 32 == 0:
            return op
        copied, columns = e.align(rows, 32), e.align(cols, 32)
        src, dst = (view_of(op.operands[i], self.defs) for i in (1, 0))
        for offset, extent, capacity in zip(src.offsets, (copied, columns), src.shape, strict=True):
            values = tuple(e.value(x) for x in (offset, extent, capacity))
            if all(isinstance(x, int) for x in values) and not 0 <= values[0] <= values[2] - values[1]:
                raise self.err(op, "byte-transpose physical load exceeds the declared L1 allocation; pad its source tile")
        if any(e.value(x) != 0 for x in dst.offsets):
            raise self.err(op, "an expanded byte-transpose load needs an L0 slot origin; offset destinations are not supported")
        allocation = dst.root.type.elem if isinstance(dst.root.type, BufType) else dst.root.type
        if isinstance(allocation, MemType) and all(isinstance(x, int) for x in allocation.dims) and isinstance(copied, int) and isinstance(columns, int):
            capacity = (prod(allocation.dims) * allocation.dtype.bits // 8 + 511) // 512 * 512
            if copied * columns > capacity:
                raise self.err(op, "byte-transpose physical load exceeds the declared L0 allocation; use a padded slot")
        self.ctx.explain.note("byte transpose copies paired16-row fractals; m_copy records the physical rows", op=op.id, kind="expanded")
        return self.rw.rewritten(op, attrs={**op.attrs, "m_copy": copied}, note="paired byte-transpose fractals")

    # -- geometry helpers ----------------------------------------------------------------------

    @staticmethod
    def _view_gather(v: View) -> bool:
        """A mem.view window only NDDMA can move: non-unit innermost stride, or a rank > 2 residue."""
        if v.gm_strides is None:
            return False
        last = v.gm_strides[-1]
        return (isinstance(last, int) and last != 1) or len(v.gm_strides) > 2

    def refuse_view_gather(self, v: View, op: Op, engine: str) -> None:
        if self._view_gather(v):
            raise self.err(op, f"a strided view with a non-unit innermost stride (or rank > 2) is read-only: "
                               f"NDDMA gathers per element on the GM -> UB path, but {engine} bursts contiguous runs")

    def gm_transfer(self, e: Emit, v: View, op: Op) -> tuple[Scalar, Scalar, Scalar]:
        """The old ``_infer_gm_transfer``: (n_burst, burst_len_element, gm_stride_element) of a GM view."""
        rank = v.rank
        if v.gm_strides is not None:
            # a mem.view window (RFC-0010): the row pitch is the explicit stride, not the type's shape
            self.refuse_view_gather(v, op, "this burst engine")
            if rank == 1 and all(v.kept):
                return 1, v.extents[0], 0
            if rank == 2 and all(v.kept):
                return v.extents[0], v.extents[1], e.sub(v.gm_strides[0], v.extents[1])
            raise self.err(op, "a strided GM view used by <<= must stay rank-1 or rank-2 (slice it without indexing a dim away)")
        if rank == 1:
            return 1, v.extents[0], 0
        if rank == 2:
            return v.extents[0], v.extents[1], e.sub(v.shape[1], v.extents[1])
        sliced = v.sliced_dims
        if len(sliced) == 1:
            d = sliced[0]
            if d == rank - 1:
                return 1, v.extents[d], 0
            burst = e.prod(list(v.extents[d + 1:]))
            return v.extents[d], burst, e.sub(e.prod(list(v.shape[d + 1:])), burst)
        if len(sliced) == 2:
            first, second = sliced
            for i in range(second + 1, rank):  # the inner slice must cover the contiguous suffix
                if not (isinstance(v.extents[i], int) and isinstance(v.shape[i], int) and v.extents[i] == v.shape[i]):
                    if v.extents[i] is not v.shape[i]:
                        raise self.err(op, "a GM window with two sliced dims must cover the dims after the second one entirely")
            burst = e.prod(list(v.extents[second:]))
            return v.extents[first], burst, e.sub(e.prod(list(v.shape[second:])), burst)
        raise self.err(op, f"a GM window used by <<= needs one or two sliced dims, this one has {len(sliced)}")

    def nd2nz_params(self, e: Emit, v: View, op: Op, transposed: bool = False) -> tuple[Scalar, Scalar, Scalar]:
        """(M, N, N_src) of a GM source for gm_to_l1_nd2nz (rows/cols of the window, row stride of the tensor)."""
        sliced = v.sliced_dims
        if v.gm_strides is not None:
            self.refuse_view_gather(v, op, "the nd2nz burst engine")
            if v.rank == 2 and all(v.kept):
                if transposed:
                    return v.extents[1], v.extents[0], v.gm_strides[0]
                return v.extents[0], v.extents[1], v.gm_strides[0]
            raise self.err(op, "a strided GM view feeding nd2nz must stay rank-2")
        if len(sliced) == 2:
            first, second = sliced
            if transposed:
                return v.extents[second], v.extents[first], v.shape[second]
            return v.extents[first], v.extents[second], v.shape[second]
        if len(sliced) == 1:
            d = sliced[0]
            if transposed:
                return v.extents[d], 1, v.shape[d]
            return 1, v.extents[d], v.shape[d]
        raise self.err(op, "an un-sliced GM tensor cannot be the source of <<=; write gm[a:b, c:d]")

    def nz2nd_params(self, v: View, op: Op) -> tuple[Scalar, Scalar, Scalar]:
        """(M, N, N_dst) of a GM destination window for l0c_to_gm_nz2nd."""
        sliced = v.sliced_dims
        if v.gm_strides is not None:
            self.refuse_view_gather(v, op, "the fixpipe nz2nd store")
            if v.rank == 2 and all(v.kept):
                return v.extents[0], v.extents[1], v.gm_strides[0]
            raise self.err(op, "a strided GM view taking an L0C tile must stay rank-2")
        if len(sliced) == 2:
            first, second = sliced
            return v.extents[first], v.extents[second], v.shape[second]
        if len(sliced) == 1:
            d = sliced[0]
            return 1, v.extents[d], v.shape[d]
        raise self.err(op, "an un-sliced GM tensor cannot be the target of <<=; write gm[a:b, c:d]")

    def gm_transpose_transfer(self, v: View, op: Op) -> tuple[Scalar, Scalar, Scalar]:
        """The old ``_infer_gm_transpose_transfer``: (rows, cols, row_stride) of a ``gm[...].T`` source."""
        sliced = v.sliced_dims
        if v.gm_strides is not None:
            self.refuse_view_gather(v, op, "the transposed nd load")
            if v.rank == 2 and all(v.kept):
                return v.extents[0], v.extents[1], v.gm_strides[0]
            raise self.err(op, "a transposed strided GM view must stay rank-2")
        if len(sliced) == 2:
            if sliced != (0, 1):
                raise self.err(op, "a transposed GM source needs its two sliced dims to be dims 0 and 1")
            return v.extents[0], v.extents[1], v.shape[1]
        if len(sliced) == 1:
            d = sliced[0]
            if d == v.rank - 1:
                return 1, v.extents[d], v.shape[d]
            return v.extents[d], 1, v.shape[1]
        raise self.err(op, "a transposed GM source needs at least one sliced dim")

    # -- the rewrite ---------------------------------------------------------------------------

    def lower(self, op: Op, e: Emit) -> Op:
        dst_v, src_v = op.operands[0], op.operands[1]
        assert isinstance(dst_v, Value) and isinstance(src_v, Value)
        dst, src = view_of(dst_v, self.defs), view_of(src_v, self.defs)
        ds, ss = dst_v.type.space, src_v.type.space  # type: ignore[union-attr]
        ds = "gm" if ds == "ws" else ds
        ss = "gm" if ss == "ws" else ss
        a = op.attrs
        transpose = bool(a.get("transpose", False))
        riders = {k: a[k] for k in ("relu", "scale", "offset", "hif8_hybrid") if k in a}
        atomic = {"atomic": a["atomic"]} if "atomic" in a else {}

        def out(opcode: str, attrs: dict[str, Any]) -> Op:
            attrs = {k: e.value(v) for k, v in attrs.items()}
            self.ctx.explain.note(f"<<= {ss} -> {ds} is {opcode}", op=op.id, opcode=opcode)
            return self.rw.rewritten(op, note=f"{ss} -> {ds}", opcode=opcode, attrs=attrs)

        if ss == "gm" and ds == "l1":
            if transpose:
                if dst.layout != "nz":
                    raise self.err(op, "a transposed GM source needs an NZ-layout L1 destination (or an ND UB one)")
                m, n, n_src = self.nd2nz_params(e, src, op, transposed=True)
                return out("dma.gm_to_l1.dn2nz", {"M": m, "N": n, "M_dst": dst.shape[0], "N_src": n_src})
            if dst.layout == "nd":
                if any(not (isinstance(o, int) and o == 0) for o in dst.offsets):
                    raise self.err(op, "gm -> L1 (ND) needs an un-sliced L1 destination")
                n_burst, burst, stride = self.gm_transfer(e, src, op)
                size = _elem_bytes(src.dtype)
                if self.ctx.device.family == "a2":
                    # c220 has no byte-granular GM->L1 (DataCopyPad targets UB only): the plain block copy,
                    # 32-B units with exact division — the old DataCopy path's own assumption
                    if isinstance(burst, int) and (burst * size) % 32:
                        raise self.err(op, f"gm -> L1 on a2 copies whole 32-B blocks; a burst of {burst} elements "
                                           f"({burst * size} B) is not block-aligned")
                    return out("dma.gm_to_l1", {"n_burst": n_burst, "burst_len": e.div(e.mul(burst, size), 32),
                                                "src_stride": e.div(e.mul(stride, size), 32), "dst_stride": 0})
                return out("dma.gm_to_l1.pad", {"n_burst": n_burst, "burst_len_byte": e.mul(burst, size),
                                                "src_stride_byte": e.mul(stride, size), "dst_stride": 0})
            m, n, n_src = self.nd2nz_params(e, src, op)
            return out("dma.gm_to_l1.nd2nz", {"M": m, "N": n, "M_dst": dst.shape[0], "N_src": n_src})
        if ss == "gm" and ds == "ub":
            if transpose:
                rows, cols, row_stride = self.gm_transpose_transfer(src, op)
                return out("dma.gm_to_ub.nd", {"dim": 2, "loop_src_stride": [row_stride, 1], "loop_dst_stride": [1, dst.shape[1]],
                                               "loop_size": [rows, cols], "loop_left_pad": [0, 0], "loop_right_pad": [0, 0],
                                               "constant_value": 0, "nearest_value_mode": False, "fence": "all", "asc_optimize": False})
            if self._view_gather(src):
                # per-element gather: the NDDMA read engine walks the explicit strides (RFC-0010 phase 3);
                # loop 0 is innermost, dst strides are the view's own row-major layout
                if not all(src.kept):
                    raise self.err(op, "an NDDMA-gathered view must keep every dim (slice without indexing dims away)")
                dim = len(src.extents)
                if dim > 5:
                    raise self.err(op, f"NDDMA walks at most 5 dims, this view has {dim}")
                size = _elem_bytes(src.dtype)
                if dim > 2:
                    flat = e.prod(list(src.extents[:-1]))
                    ok = (isinstance(flat, int) and isinstance(dst.extents[0], int) and flat == dst.extents[0]
                          and src.extents[-1] == dst.extents[-1]) or not all(isinstance(x, int) for x in (*src.extents, *dst.extents))
                    if not ok or any(not (isinstance(o, int) and o == 0) for o in dst.offsets):
                        raise self.err(op, "a rank > 2 view lands row-major in a [rows*..., cols] UB tile written whole")
                    row = src.extents[-1]
                    dst_strides: list[Scalar] = []
                    acc: Scalar = 1
                    for sdim in reversed(src.extents):
                        dst_strides.append(acc)
                        acc = e.mul(acc, sdim)
                else:
                    row = dst.shape[1]
                    dst_strides = [1, dst.shape[1]]
                # the UB port lands NDDMA rows on 32-byte blocks: a narrower row silently corrupts
                # its neighbours (board-measured on the D-085 gather probe; same port rule as D-053)
                if isinstance(row, int) and (row * size) % 32:
                    raise self.err(op, f"an NDDMA-gathered view needs its UB rows on 32-byte blocks: "
                                       f"the destination row is {row} elements ({row * size} bytes) - pad the tile row")
                return out("dma.gm_to_ub.nd", {"dim": dim,
                                               "loop_src_stride": list(reversed(list(src.gm_strides or ()))),
                                               "loop_dst_stride": dst_strides[:dim],
                                               "loop_size": list(reversed(list(src.extents))),
                                               "loop_left_pad": [0] * dim, "loop_right_pad": [0] * dim,
                                               "constant_value": 0, "nearest_value_mode": False,
                                               "fence": "all", "asc_optimize": False})
            n_burst, burst, stride = self.gm_transfer(e, src, op)
            size = _elem_bytes(src.dtype)
            dst_stride = 0 if src.rank == 1 else e.div(e.sub(dst.shape[1], burst), _c0(dst.dtype))
            attrs = {"n_burst": n_burst, "burst_len_byte": e.mul(burst, size), "src_stride_byte": e.mul(stride, size), "dst_stride": dst_stride}
            if "pad" in a:
                attrs["pad"] = a["pad"]
            return out("dma.gm_to_ub.pad", attrs)
        if ss == "gm":
            raise self.err(op, f"a GM tensor can only be copied into L1 or UB, not {ds}")
        if ss == "ub" and ds == "gm":
            self.refuse_view_gather(dst, op, "ub_to_gm")
            n_burst, burst, dst_stride = self.gm_transfer(e, dst, op)
            size = _elem_bytes(dst.dtype)
            src_stride = 0 if dst.rank == 1 else e.div(e.sub(src.shape[1], burst), _c0(src.dtype))
            return out("dma.ub_to_gm.pad", {"n_burst": n_burst, "burst_len_byte": e.mul(burst, size), "src_stride": src_stride,
                                            "dst_stride_byte": e.mul(dst_stride, size), **atomic})
        if ss == "ub" and ds == "ub":
            c0 = _c0(dst.dtype)
            span1 = dst.span[1]
            return out("dma.ub_to_ub", {"n_burst": dst.span[0], "burst_len": e.ceil_div(span1, c0),
                                        "src_stride": e.ceil_div(e.sub(src.shape[1], span1), c0),
                                        "dst_stride": e.ceil_div(e.sub(dst.shape[1], span1), c0)})
        if ss == "ub" and ds == "l1":
            common = {"m_dst": dst.shape[0], "n_dst": dst.span[1], "m_src": src.span[0], "n_src": src.span[1],
                      "dst_row0": dst.offsets[0], "dst_col0": dst.offsets[1]}
            if src.layout == "nz":
                return out("dma.ub_to_l1.nz", {**common, "M_src": src.shape[0], "src_row0": src.offsets[0], "src_col0": src.offsets[1]})
            return out("dma.ub_to_l1.nd2nz", {**common, "N_src": src.shape[1]})
        if ss == "l0c" and ds == "l1":
            return out("dma.l0c_to_l1", {"M": dst.span[0], "N": dst.span[1], "M_dst": dst.shape[0], "M_src": src.shape[0],
                                         "relu": bool(a.get("relu", False))})
        if ss == "l0c" and ds == "ub":
            mode = str(a.get("dual_mode", Ident("splitm")))
            if mode == "single":
                return out("dma.l0c_to_ub", {"M": dst.span[0], "N": dst.span[1], "N_dst": dst.shape[1], "M_src": src.shape[0],
                                             "dual_mode": Ident("single"), "sub_block_id": a["sub_block_id"], **riders})
            # the riders go through in *both* branches: dropping them here made a split-mode
            # `relu=True` copy lose its relu without a word, and the rule that split mode carries
            # none of them belongs in the verifier, which can say so (ir/verify.py check_hardware)
            return out("dma.l0c_to_ub", {"M": e.mul(dst.span[0], 2), "N": dst.span[1], "N_dst": dst.shape[1], "M_src": src.shape[0],
                                         "dual_mode": Ident("splitm"), "sub_block_id": 0, **riders})
        if ss == "l0c" and ds == "gm":
            if transpose:
                return out("dma.l0c_to_gm.nz2dn", {"M": src.shape[0], "N": src.shape[1], "M_dst": dst.shape[-1], "M_src": src.shape[0],
                                                   **riders, **atomic})
            m, n, n_dst = self.nz2nd_params(dst, op)
            return out("dma.l0c_to_gm.nz2nd", {"M": m, "N": n, "N_dst": n_dst, "M_src": src.shape[0], **riders, **atomic})
        if ss == "l1" and ds in ("l0a", "l0b"):
            return out("dma.l1_to_l0", {"m_dst": src.span[0], "n_dst": src.span[1], "m_src": src.shape[0], "n_src": src.shape[1],
                                        "src_row0": src.offsets[0], "src_col0": src.offsets[1], "src_is_transpose": transpose,
                                        "dst_position": Ident(ds)})
        if ss == "l1" and ds == "bt":
            return out("dma.l1_to_bt", {"n": e.mul(dst.shape[0], dst.shape[1])})
        raise self.err(op, f"no instruction copies {ss} -> {ds}")


# The old CvMutex / VcMutex methods as cross-core primitives: (opcode, which pipe attribute of the declaration).
_MUTEX = {
    "cv": {"lock": ("wait_vec", "src_start_pipe"), "ready": ("cube_ready", "src_end_pipe"),
           "wait": ("wait_cube", "dst_start_pipe"), "free": ("vec_ready", "dst_end_pipe")},
    "vc": {"lock": ("wait_cube", "src_start_pipe"), "ready": ("vec_ready", "src_end_pipe"),
           "wait": ("wait_vec", "dst_start_pipe"), "free": ("cube_ready", "dst_end_pipe")},
}


def _mutex_pipes(kind: str, decl: Op, family: str) -> dict[str, str]:
    """Pipe defaults of the old mutex classes; a5 vector cores release after V, c220 after MTE2, others after MTE3."""
    vec_end = {"a5": "V", "a2": "MTE2"}.get(family, "MTE3")
    defaults = {"cv": {"src_start_pipe": "S", "dst_start_pipe": "S", "src_end_pipe": "FIX", "dst_end_pipe": vec_end},
                "vc": {"src_start_pipe": "S", "dst_start_pipe": "S", "src_end_pipe": "MTE3", "dst_end_pipe": "FIX"}}[kind]
    out = {}
    for k, d in defaults.items():
        v = decl.attrs.get(k)
        out[k] = (v.name if isinstance(v, Ident) else str(v)) if v is not None else d
    return out


def lower_mutex(op: Op, defs: Defs, rw: Rewriter, ctx: PassContext) -> Op:
    flag = op.operands[0]
    decl = defs.op(flag) if isinstance(flag, Value) else None
    if decl is None or decl.opcode != "sync.mutex":
        raise PassError(PASS, f"{op.opcode} #{op.id} ({op.loc}): the flag is not a sync.mutex declaration")
    kind = decl.attrs["kind"]
    kind = kind.name if isinstance(kind, Ident) else str(kind)
    method = op.opcode.removeprefix("sync.mutex_")
    name, pipe_attr = _MUTEX[kind][method]
    pipe = _mutex_pipes(kind, decl, ctx.device.family)[pipe_attr]
    ctx.explain.note(f"{op.opcode} on a {kind} mutex is sync.crosscore.{name} on {pipe}", op=op.id, opcode=f"sync.crosscore.{name}")
    return rw.rewritten(op, opcode=f"sync.crosscore.{name}", operands=(), attrs={"flag_id": int(decl.attrs["id"]), "pipe": Ident(pipe)},
                        note=f"{kind} mutex {method}")


def run(module: Module, ctx: PassContext) -> Module:
    defs = Defs(module)
    rw = Rewriter(module, PASS)
    lowerer = _Lowerer(ctx, rw, defs)

    def visit(op: Op, fn, emit_factory) -> list[Op] | None:
        if op.opcode.startswith("sync.mutex_"):
            return lower_mutex(op, defs, rw, ctx)
        if op.opcode not in ("dma.copy", "dma.l1_to_l0"):
            return None
        e = emit_factory(op)
        new = lowerer.lower(op, e) if op.opcode == "dma.copy" else op
        new = lowerer.physical_l0(new, e)
        return e.pre + [new]

    return rewrite_module(module, rw, visit)


PASS_DEF = Pass(PASS, run, doc="select one DMA instruction per dma.copy from the memory types, layouts and riders",
                establishes=("5",))

__all__ = ["PASS_DEF", "run"]
