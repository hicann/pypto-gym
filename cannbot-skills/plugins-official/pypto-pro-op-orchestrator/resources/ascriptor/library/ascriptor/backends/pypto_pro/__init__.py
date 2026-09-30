# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The ``pypto_pro`` backend (M7) — Lowered IR -> PyPTO Pro Python source.

Named ``pypto_pro`` on purpose: ``pypto`` is the auto-tiling frontend in the same upstream
repository and means something else. ``compile(module)`` returns a generated
``kernel_pypto.py`` (the ``pl`` DSL module), a board-side ``run_case.py`` driver and a
``manifest.json`` (:class:`HostSpec`-compatible). Native ``auto_mutex`` is enabled by default,
using the Lowered IR's exact slot IDs. Manual emission remains available for
comparison. The backend does not plan synchronization (RFC-0013).
The API overlap lives
in ``docs/pypto-pro-coverage.md``; ops outside the wired surface raise :class:`PyptoGap`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..base import Artifacts, Capabilities, ResourceLimits
from .emit import PL_DT, ModulePrinter, PyptoGap, emit_module, module_manifest
from .supplements import SUPPLEMENTS, PyptoSupplementWarning

DEVICES = frozenset({"950", "950pr", "a5", "a5pr"})  # PyPTO Pro runs on 950PR / 950DT silicon only


class PyptoProBackend:
    name = "pypto_pro"

    def capabilities(self) -> Capabilities:
        """The opcodes this backend prints.

        Written out rather than derived, because the printer dispatches through ``if`` chains and
        parsing them at import time would cost every caller. `tests/backends/test_pypto_support.py`
        holds this list to `tools/pypto_support.py`'s reading of that dispatch, in BOTH directions
        and against the registry, so it cannot drift from the code or name an opcode that does not
        exist. An addition here without a line in `emit.py` fails, and a line without an addition
        fails too.
        """
        from .emit import VF_BINARY, VF_CMP, VF_GROUP_REDUCE, VF_SCALAR, VF_UNARY, SCALAR_BIN

        opcodes = {"mem.alloc", "mem.slice", "mem.get_buf", "mem.reinterpret", "mem.view", "mem.workspace",
                   "dma.gm_to_ub.pad", "dma.gm_to_ub.nd", "dma.ub_to_gm.pad",
                   "dma.gm_to_l1.nd2nz", "dma.gm_to_l1.pad", "dma.gm_to_l1",
                   "dma.l0c_to_gm.nz2nd", "dma.l0c_to_gm.nz2dn", "dma.l1_to_l0.mx", "dma.gm_to_l1.mx_scale_nd2nz", "cube.mmad.mx", "dma.l1_to_l0", "dma.l0c_to_ub", "dma.l0c_to_l1", "dma.l1_to_bt", "dma.set_constant_to_l1",
                   "dma.ub_to_l1", "dma.ub_to_l1.nd2nz", "dma.ub_to_l1.nz", "dma.ub_to_ub",
                   "cube.mmad",
                   "sync.event", "sync.set", "sync.set_all", "sync.release", "sync.wait", "sync.barrier", "sync.mutex",
                   "sync.local_mutex_get", "sync.local_mutex_release",
                   "sync.crosscore.cube_ready", "sync.crosscore.vec_ready",
                   "sync.crosscore.wait_cube", "sync.crosscore.wait_vec",
                   "sync.crosscore.intracore_allvec_ready", "sync.crosscore.intracore_allvec_wait",
                   "cf.call", "cf.return", "cf.for", "cf.if",
                   "core.cube_idx", "core.cube_num", "core.vec_idx", "core.vec_num",
                   "core.sub_block_idx",
                   "scalar.cell", "scalar.set", "scalar.const", "scalar.div", "scalar.ceil_div",
                   "scalar.align", "scalar.min", "scalar.max", "scalar.neg", "scalar.not", "scalar.cmp",
                   "scalar.select", "scalar.load", "scalar.store", "scalar.cast",
                   "vec.sort32", "vec.mergesort_2seq",
                   "debug.print", "debug.dump", "debug.print_reg", "debug.assert",
                   "core.set_sat_flag", "core.get_sat_flag", "core.clean_dcache", "dma.gm_to_l1.dn2nz",
                   "vec.set_mask", "vec.set_mask_by_count", "vec.reset_mask", "vec.set_mask_count",
                   "vec.set_mask_normal", "vf.pack", "vf.unsqueeze", "vf.unalign", "vf.load_unalign_pre",
                   "vf.load_unalign", "vf.store_unalign", "vf.store_unalign_post", "vf.ub_cursor",
                   "vf.reg", "vf.mask", "vf.load_cont", "vf.store_cont", "vf.load", "vf.store",
                   "vf.dup", "vf.cmp", "vf.cmps", "vf.select", "vf.cast", "vf.barrier", "vf.squeeze", "vf.gathermask", "vf.histograms", "vf.mask_and",
                   "vf.arange", "vf.mask_update", "vf.mask_from_spr", "vf.reinterpret",
                   # The predicate spill/fill pair: pto spells them as the MaskReg overloads of
                   # the aligned move (psts / plds), not as their own APIs.
                   "vf.mask_to_ub", "vf.ub_to_mask",
                   # Printed here since the surveyed surface admitted them, but never declared -
                   # a stale declaration understates the backend just as a missing emit overstates
                   # it, and a route planner reading capabilities takes a gap that is not there.
                   "vf.scatter_copy", "vf.gather", "vf.gather_copy", "vf.gatherb",
                   "vf.interleave", "vf.deinterleave"}
        opcodes |= set(VF_BINARY) | set(VF_UNARY) | set(VF_SCALAR) | set(VF_GROUP_REDUCE) | set(SCALAR_BIN)
        # The dual-register overloads of the aligned move (the printer prints them through
        # LOAD_INTLV_DIST / STORE_INTLV_DIST; declaring them is what M10-075 had missed).
        opcodes |= {"vf.load_interleave", "vf.store_interleave",
                    # the raw intra-core flag pair -> pl.system.sync_src / sync_dst
                    "sync.set_flag", "sync.wait_flag",
                    "vf.mulscast"}   # vf.lrelu rides in through VF_SCALAR above
        # the atomic REGION markers. pl has no mode - `pl.store(atomic=)` is per store and
        # the frontend already stamps the mode on each copy inside - so the three carry
        # nothing to print; the emitter tracks them to refuse a store that would drop it.
        opcodes |= {"atomic.begin", "atomic.end", "atomic.set_type"}
        opcodes |= {"simt.launch", "simt.load", "simt.store", "simt.atomic", "simt.thread_id",
                    "simt.thread_num", "simt.block_idx", "simt.block_num", "simt.threadfence",
                    "simt.threadfence_block", "simt.barrier"}
        # `simt.ffs` / `simt.popc` are NOT here: the printer refuses both outright (pl.simt.popcount
        # admits uint32/uint64 and pto derives `index` for the input), and declaring a refusal sends
        # a route planner into a gap at export with no warning beforehand.
        opcodes |= {f"simt.{m}" for m in ("exp", "exp2", "log", "log2", "log1p", "sin", "cos",
                    "tanh", "rsqrt", "rint", "round", "floor", "ceil", "trunc", "isnan",
                    "isinf", "isfinite", "fmod", "fma", "mul_hi")}
        # The GMList trio and the reshape view: printed since the member loop was unrolled
        # (D-119), never declared. `list.count` folds to a literal; `list.item` / `list.item_dim`
        # name the member each unrolled copy carries.
        opcodes |= {"list.count", "list.item", "list.item_dim", "mem.reshape"}
        # The AIV-wide barrier, which maps only as a PAIR: `allvec_ready` immediately followed by
        # `allvec_wait` becomes one `pl.system.sync_all`, and either alone raises.
        opcodes |= {"sync.crosscore.allvec_ready", "sync.crosscore.allvec_wait"}
        return Capabilities(
            function_kinds=frozenset({"kernel", "vf", "simt"}),
            opcodes=frozenset(opcodes),
            dtypes=frozenset(PL_DT),
            devices=DEVICES,
            notes={
                "*": "phase 2: vec + mix kernels, loops/branches, specialised scalar parameters, "
                     "slot buffers, cross-core pair sync (docs/pypto-pro-coverage.md); "
                     "outside forms raise PyptoGap",
                "vf.reg": "reg_num=2 has no native PyPTO Pro VF register-group type",
                "vf.mask": "grouped predicates have no backend mapping",
                "vf.mod": "no PyPTO Pro VF integer floor-remainder API",
                "vf.arange": "i64/u64 has no native PyPTO Pro VF register carrier",
                "vf.cast": "i64/u64 register source or destination has no native VF carrier",
                "scalar.cast": "ordinary integer conversion requires the local pl.cast compatibility supplement, "
                               "which is not installed by default and is warned about at emit "
                               "(supplements.py); float/bool/sub-byte conversions remain gaps",
                "simt.ffs": "pl.simt.popcount admits uint32/uint64 only and pto types the input "
                            "as index; no pl spelling (upstream)",
                "simt.popc": "pl.simt.popcount admits uint32/uint64 only and pto types the input "
                             "as index; no pl spelling (upstream)",
            },
        )

    def resources(self, device: Any) -> ResourceLimits:
        from ... import devices as _devices

        p = device if hasattr(device, "capacities_kb") else _devices.load(str(device))
        kb = p.capacities_kb
        return ResourceLimits(ub=kb.get("ub", 0) * 1024, l1=kb.get("l1", 0) * 1024)

    def compile(self, module: Any, options: Mapping[str, Any] | None = None) -> Artifacts:
        options = dict(options or {})
        if "keep_events" in options:
            raise PyptoGap(None, "keep_events was removed; backend event selection no longer exists", owner="ours")
        return emit_module(module, block_dim=options.get("block_dim"), entry=options.get("entry"),
                           bindings=options.get("bindings"), sync_mode=options.get("sync_mode", "manual"),
                           supplements=options.get("supplements"))


__all__ = ["PyptoProBackend", "PyptoGap", "PyptoSupplementWarning", "SUPPLEMENTS", "ModulePrinter",
           "emit_module", "module_manifest", "DEVICES"]
