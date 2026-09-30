# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The ``pto_isa`` backend (RFC-0011) — Lowered IR -> PTO tile-ISA C++ source.

PTO (Parallel Tile Operation) is the tile-level virtual ISA CANN defines; ``pto-isa`` is its
header-only C++ implementation, the layer *below* PyPTO Pro (``pl`` traces down to these same
templates, which expand to the same CCE builtins the ``cce`` backend prints). This backend is a5
only, and its scope is **tile creation and movement**: the ``mem.*`` and ``dma.*`` families.
Compute, registers and SIMT stay outside it — a ``@vf`` body prints bare CCE intrinsics through
``__cce_get_tile_ptr``, which is what pto's own a5 headers do (RFC-0011 §1).

The groups below are the scoped opcodes by batch; ``HANDLED`` is read off the printer's dispatch table and
a test holds the two equal. Everything else raises :class:`PtoIsaGap` with a source location.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..base import Artifacts, Capabilities, ResourceLimits
from . import types as pt
from .emit import FnPrinter, GTensor, ModulePrinter, PtoIsaGap, Tile, emit_module

DEVICES = frozenset({"950", "950pr", "a5", "a5pr"})  # a5 only (RFC-0011 §1 ruling 2)

#: the opcodes this RFC scopes in, by batch (RFC-0011 §7). Batch 1 prints the creation half;
#: batch 2 the GM <-> UB movement (``dma.gm_to_ub.nd`` is in scope but a measured gap, §4.1).
BATCH1 = frozenset({"mem.alloc", "mem.workspace", "mem.get_buf", "mem.slice", "mem.reshape",
                    "mem.view", "mem.reinterpret"})
BATCH2 = frozenset({"dma.gm_to_ub.pad", "dma.ub_to_gm.pad", "dma.ub_to_ub"})
#: GM -> L1 (RFC-0011 §4.2). These ops state a matrix rather than a burst -- M x N read from GM
#: whose rows are N_src apart, landing in an NZ tile M_dst rows tall -- which is closer to what
#: PTO wants than the UB path's burst descriptor is.
BATCH3 = frozenset({"dma.gm_to_l1.nd2nz", "dma.gm_to_l1.pad", "dma.l1_to_l0", "dma.l1_to_bt"})
#: L0C out (RFC-0011 §4.4): TSTORE from an Acc tile, all three of them. The GlobalTensor's layout
#: picks the arm: the NZ one describes the destination *plane* rather than the transfer, which is
#: what makes every register cce's (§7.23), and the DN one is `TSTORE` into a ``Layout::NCHW``
#: tensor -- ``TStoreAccNCHW`` sets ``nz2dnEn = 1`` and is cce's `l0c_to_gm_nz2dn` field for field,
#: one instruction under two names (§7.30).
BATCH4 = frozenset({"dma.l0c_to_gm.nz2nd", "dma.l0c_to_gm.nz2nz", "dma.l0c_to_gm.nz2dn"})
#: The one compute opcode (RFC-0011 §4.7), admitted for a validation reason rather than a coverage
#: one: every cube kernel in the corpus needs it, so without it §4.2-§4.4 can never be compared
#: against cce on silicon. ``cube.mmad.mx`` is its own batch (12), beside the operand path it needs.
BATCH5 = frozenset({"cube.mmad"})
#: Cross-core synchronisation and the cross-core mutex (RFC-0011 §5.3). The two largest blockers
#: in the corpus. Both are pure synchronisation -- the same class as the intra-core event layer
#: FRAME already prints -- and without them a mix kernel's two sides cannot be sequenced, so its
#: *movement* cannot be validated on the board either.
BATCH6 = frozenset({
    "sync.crosscore.cube_ready", "sync.crosscore.wait_vec",
    "sync.crosscore.vec_ready", "sync.crosscore.wait_cube",
    "sync.crosscore.allcube_ready", "sync.crosscore.allcube_wait",
    "sync.crosscore.allvec_ready", "sync.crosscore.allvec_wait",
    "sync.crosscore.intracore_allvec_ready", "sync.crosscore.intracore_allvec_wait",
    "sync.mutex",
})
#: L0C -> UB (RFC-0011 §4.4 #20): TMOV Acc -> Vec. The one place cce's three-modes-plus-a-bool
#: meets PTO's four-valued AccToVecMode, which carries the sub-block id inside the mode.
BATCH7 = frozenset({"dma.l0c_to_ub"})
#: UB -> L1 (RFC-0011 §4.8): the one movement that is not a TMOV. TMOV's Vec -> Mat path takes its
#: source stride as a constexpr and pins dstStride to 0; TINSERT has both at run time, and carries
#: cce's `src_stride` in a template argument -- the TInsertMode -- instead of a call argument.
BATCH8 = frozenset({"dma.ub_to_l1.nz"})
#: Filling an L1 tile with a constant (RFC-0011 §4.9). One instruction either side, and the
#: same builtin; the block count moves from a call argument into the tile's own capacity.
BATCH9 = frozenset({"dma.set_constant_to_l1"})
#: UB ND -> L1 NZ (RFC-0011 §4.10). PTO has no single ND->NZ DMA and neither does cce: both
#: are a loop of one burst per fractal column, and the two loops agree term for term.
BATCH10 = frozenset({"dma.ub_to_l1.nd2nz"})

#: Batch 12, the microscaling operand path (RFC-0011 §4.12). cce loads a data tile and its e8m0
#: scale plane in one wrapper because the plane is not addressed independently -- it sits at
#: `l0.addr / 16`; PTO splits it into two `TEXTRACT`s over two tiles and then hands the scale tiles
#: to `TMATMUL_MX` for type-checking alone.
BATCH12 = frozenset({"dma.l1_to_l0.mx", "cube.mmad.mx"})
#: Batch 13, the rest of the movement (RFC-0011 §4.13): the plain UB -> L1 block copy, which is
#: `TMOV`'s Vec -> Mat path with three of `copy_ubuf_to_cbuf`'s six arguments fixed at the values
#: cce's callers already pass.
BATCH13 = frozenset({"dma.ub_to_l1", "dma.l0c_to_l1", "dma.gm_to_l1.dn2nz",
                     "dma.gm_to_l1.mx_scale_nd2nz",
                     "dma.l1_to_l0.img2col"})
#: `debug.*` is a comment on both sides (RFC-0011 §7.20): cce's handlers are a `self.comment` each,
#: so these ops change nothing about the kernel that reaches the board. They are declared because a
#: kernel that uses one would otherwise be unable to *run* the movement it also contains.
DEBUG = frozenset({"debug.print", "debug.dump", "debug.assert", "debug.print_reg"})

#: RFC-0011 §4.14 -- the gmlist descriptor reads. Not an ISA family at all: a `GMList` is
#: AscendC's ListTensorDesc read with plain scalar loads, and what it yields is a pointer
#: §4.1's GlobalTensor takes like any other.
LIST = frozenset({"list.count", "list.item", "list.item_dim"})

#: RFC-0011 §7.29 -- the launch site only. A `@simt` body is plain C on the compiler's own
#: SIMT layer and comes from cce's `SimtPrinter`, the same bridge §7.7 draws for `@vf`; its
#: `simt.*` opcodes are therefore absent from this table, exactly as the `vf.*` ones are.
SIMT = frozenset({"simt.launch"})
#: the frame a whole kernel needs around the movement: autosync's lowered sync ops print as the
#: bare ``set_flag`` / ``wait_flag`` pair (RFC-0011 §5) and the return closes the entry. Not a
#: batch of its own — without it no kernel could emit a complete translation unit.
FRAME = frozenset({"sync.event", "sync.set", "sync.wait", "sync.set_all", "sync.release", "sync.barrier", "cf.return",
                   "sync.local_mutex_get", "sync.local_mutex_release", "sync.set_flag", "sync.wait_flag"})
#: the host language layer (RFC-0011 §7.5). None of it is a PTO instruction: scalars are C++
#: expressions, control flow is C++ control flow, and a core id is a CCE builtin — which is
#: exactly why a C++ target does not need `pl`'s trace-time folding of every scalar (§2).
SCALAR = frozenset({"scalar.add", "scalar.sub", "scalar.mul", "scalar.div", "scalar.mod",
                    "scalar.and", "scalar.or", "scalar.xor", "scalar.shl", "scalar.shr",
                    "scalar.min", "scalar.max", "scalar.ceil_div", "scalar.not", "scalar.neg",
                    "scalar.abs", "scalar.sqrt", "scalar.align", "scalar.cmp", "scalar.select",
                    "scalar.cast", "scalar.const", "scalar.cell", "scalar.set",
                    "scalar.load", "scalar.store"})
CONTROL = frozenset({"cf.for", "cf.if", "cf.break", "cf.continue"})
CORE = frozenset({"core.cube_idx", "core.cube_num", "core.vec_idx", "core.vec_num",
                  "core.sub_block_idx", "core.set_sat_flag", "core.get_sat_flag",
                  "core.set_hf32"})
#: the @vf bridge (RFC-0011 §7.7). ``cf.call`` is the only kernel-level op it adds: the vf body
#: itself is printed by cce's own ``VfPrinter``, unmodified, because a vf function is CCE register
#: intrinsics over ``__ubuf__`` pointers and mentions no tile. The 91 ``vf.*`` opcodes therefore
#: never appear in this backend's own dispatch table.
VF = frozenset({"cf.call"})
#: the groups above, as one set: what the RFC scopes in. `HANDLED` is not this union but what the printer
#: dispatches (`FnPrinter.run_op` looks up `op_<opcode>`), so neither can drift from the other unnoticed:
#: `tests/backends/test_host_printer.py` holds the two equal.
SCOPED = (LIST | SIMT | BATCH1 | BATCH2 | BATCH3 | BATCH4 | BATCH5 | BATCH6 | BATCH7 | BATCH8 | BATCH9 | BATCH10
          | BATCH12 | BATCH13
          | FRAME | DEBUG
          | SCALAR | CONTROL | CORE | VF)


def _handled() -> frozenset[str]:
    from ...ir import REGISTRY

    # method names flatten dots and underscores alike, so resolve from the registry, as cce does
    return frozenset(s.name for s in REGISTRY.all() if hasattr(FnPrinter, "op_" + s.name.replace(".", "_")))


HANDLED = _handled()


class PtoIsaBackend:
    name = "pto_isa"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            function_kinds=frozenset({"kernel", "vf", "simt"}),
            opcodes=HANDLED,
            dtypes=frozenset(pt.ELEM),
            devices=DEVICES,
            notes={
                "*": "RFC-0011: tile creation (mem.*), the movement between memories (dma.*) and the host "
                     "frame a whole kernel needs; an opcode outside the dispatch table raises PtoIsaGap, "
                     "and a declared opcode may still refuse a form it cannot express",
                "dma.gm_to_ub.nd": "no PTO template walks GM with element strides (RFC-0011 §4.1)",
                "vf.*": "shared CCE VF printer, including register groups; CCE form gaps "
                        "preserve the source op in PtoIsaGap; complex PTO memory types remain gaps",
            },
        )

    def resources(self, device: Any) -> ResourceLimits:
        from ... import devices as _devices

        p = device if hasattr(device, "capacities_kb") else _devices.load(str(device))
        kb = p.capacities_kb
        return ResourceLimits(ub=kb.get("ub", 0) * 1024, l1=kb.get("l1", 0) * 1024, l0a=kb.get("l0a", 0) * 1024,
                              l0b=kb.get("l0b", 0) * 1024, l0c=kb.get("l0c", 0) * 1024, bt=kb.get("bt", 0) * 1024)

    def compile(self, module: Any, options: Mapping[str, Any] | None = None) -> Artifacts:
        options = dict(options or {})
        # ``bindings`` specialises tile capacities per scalar valuation (RFC-0011 §3); without it
        # a kernel whose tiles are shaped by a parameter refuses rather than guessing a shape.
        from ...runtime.launch_config import exported_artifacts

        return exported_artifacts(module, "the PTO ISA backend", options.get("block_dim"), lambda block_dim: emit_module(
            module, block_dim=block_dim, entry=options.get("entry"), bindings=options.get("bindings")))


__all__ = ["PtoIsaBackend", "PtoIsaGap", "FnPrinter", "ModulePrinter", "Tile", "GTensor",
           "emit_module", "DEVICES", "BATCH1", "BATCH2", "BATCH3", "BATCH4", "BATCH5", "BATCH6", "BATCH7", "BATCH8", "BATCH9", "BATCH10", "BATCH12", "BATCH13", "FRAME", "DEBUG", "LIST", "SIMT", "SCALAR", "CONTROL", "CORE", "VF",
           "SCOPED", "HANDLED"]
