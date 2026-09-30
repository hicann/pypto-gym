# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cycle costs of Lowered IR instructions (RFC-0006 §7).

The formulas are the old simulator's (``easyasc/simulator/timing/cycle_model.py``, validated against
the a5 board for DMA, mmad and the VF throughput model) applied to the new opcodes; the constants
live in ``a5_cycle_model.json`` next to this file, unchanged. Two deliberate refinements: padded
GM -> L1 loads, ``img2col`` and ``l1_to_bt`` are costed by their bytes instead of the old 11-cycle
fallback.

* DMA: ``overhead + bytes / bandwidth`` (rounded), no per-instruction head overhead.
* ``mmad``: ``67 + M16 * N16 * Kc0 / macs_per_cycle`` with the rate chosen by the operand width.
* L0 loads: ``ceil(bytes / 128)`` plus the 10-cycle head overhead.
* VF (``cf.call``): ``46 + makespan`` where the makespan is the busiest of the LD / ST / SU / EX issue
  pipes (issue intervals per micro op, bank-conflict penalties for strided loads and stores), never
  less than the longest op latency or 0.46 x the register RAW critical path; plus the head overhead.
* Sync markers cost one cycle; ``set_flag`` / ``wait_flag`` also pay the head overhead.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ....ir import Ident, Op, Value
from ....ir.registry import REGISTRY
from ....ir.types import DType, MemType, RegType

_TABLES = Path(__file__).parent


def _legacy(opcode: str) -> str:
    spec = REGISTRY.find(opcode)
    return spec.legacy[0] if spec is not None and spec.legacy else opcode


def _linear(n: float, bandwidth: float, overhead: float) -> int:
    """The old ``_cycles_from_linear_model``: ``max(1, int(overhead + n / bandwidth + 0.5))``."""
    if n <= 0 and overhead <= 0:
        return 0
    return max(1, int(overhead + n / bandwidth + 0.5))


def _ceil_bw(n: float, bandwidth: float) -> int:
    return max(1, math.ceil(n / bandwidth)) if n > 0 else 0


def _a16(x: int) -> int:
    return -(-x // 16) * 16


def _bytes(dt: DType) -> float:
    return dt.bits / 8


@dataclass
class VfCost:
    cycles: int
    busy: dict[str, float]
    crit: float
    max_latency: float
    counts: dict[str, int]


@dataclass
class CycleModel:
    table: dict[str, Any]
    device_family: str = "a5"
    head_overhead: int = 10
    _lat: dict[str, float] = field(default_factory=dict)
    _ex_override: dict[str, float] = field(default_factory=dict)
    _issue: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = self.table
        self.head_overhead = int(t.get("instruction_head_overhead_cycles", 10))
        self._lat = dict(t.get("vf_micro_latency", {}))
        self._ex_override = dict(t.get("vf_ex_interval_override", {}))
        self._issue = {"LD": 1.14, "ST": 1.0, "SU": 0.48}
        self._issue.update(t.get("vf_pipe_issue_interval", {}))

    def k(self, key: str, default: float = 0.0) -> float:
        return float(self.table.get(key, default))

    # -- kernel-level instructions --------------------------------------------------------------

    def cost(self, op: Op, val: Any, operand_type: Any) -> int:
        """Cycles one instruction occupies its pipe. ``val(x)`` evaluates an int|value attribute or operand;
        ``operand_type(i)`` gives the MemType of operand ``i`` (None if not a memory operand)."""
        oc = op.opcode
        H = self.head_overhead

        def a(name: str, default: Any = 0) -> Any:
            return val(op.attrs.get(name, default))

        def elem(i: int) -> float:
            t = operand_type(i)
            return _bytes(t.dtype) if isinstance(t, MemType) else 1.0

        def c0(i: int) -> int:
            t = operand_type(i)
            return (64 if t.dtype.bits < 8 else 32 * 8 // t.dtype.bits) if isinstance(t, MemType) else 16

        if oc in ("sync.set", "sync.wait", "sync.set_all", "sync.release") or oc.startswith(("sync.crosscore.", "sync.local_mutex_")) or oc == "sync.barrier":
            return int(self.k("sync_marker_cycles", 1))
        if oc in ("sync.set_flag", "sync.wait_flag"):
            return int(self.k("sync_marker_cycles", 1)) + H
        if oc in ("dma.gm_to_l1.nd2nz", "dma.gm_to_l1.dn2nz"):
            M, N, N_src = int(a("M")), int(a("N")), int(a("N_src", a("N")))
            span = ((N - 1) * N_src + M) if oc.endswith("dn2nz") else ((M - 1) * N_src + N)
            return _linear(span * elem(1), self.k("gm_to_l1_nd2nz_bandwidth_bytes_per_cycle", self.k("gm_bandwidth_bytes_per_cycle", 32)),
                           self.k("gm_to_l1_nd2nz_overhead_cycles", 300))
        if oc == "dma.gm_to_l1.mx_scale_nd2nz":
            rows, groups = int(a("rows")), int(a("k_groups"))
            return _linear(rows * groups, self.k("gm_bandwidth_bytes_per_cycle", 32), self.k("gm_to_l1_nd2nz_overhead_cycles", 300))
        if oc in ("dma.gm_to_l1.pad", "dma.gm_to_ub.pad"):
            payload = int(a("n_burst", 1)) * int(a("burst_len_byte", 0))
            return _linear(payload, self.k("gm_to_ub_pad_bandwidth_bytes_per_cycle", 32), self.k("gm_to_ub_pad_overhead_cycles", 300))
        if oc == "dma.gm_to_ub.nd":
            sizes = [int(val(x)) for x in op.attrs.get("loop_size", [])]
            left = [int(val(x)) for x in op.attrs.get("loop_left_pad", [])] or [0] * len(sizes)
            right = [int(val(x)) for x in op.attrs.get("loop_right_pad", [])] or [0] * len(sizes)
            n = 1
            for s, lp, rp in zip(sizes, left, right, strict=False):
                n *= s + lp + rp
            return _linear(n * elem(0), self.k("gm_to_ub_pad_bandwidth_bytes_per_cycle", 32), self.k("gm_to_ub_pad_overhead_cycles", 300))
        if oc == "dma.ub_to_gm.pad":
            payload = int(a("n_burst", 1)) * int(a("burst_len_byte", 0))
            return _linear(payload, self.k("ub_to_gm_pad_bandwidth_bytes_per_cycle", 32), self.k("ub_to_gm_pad_overhead_cycles", 300))
        if oc == "dma.set_constant_to_l1":
            return _linear(int(a("n_blocks", 1)) * 32, self.k("set_constant_to_l1_bandwidth_bytes_per_cycle", 256),
                           self.k("set_constant_to_l1_overhead_cycles", 14))
        if oc in ("dma.l1_to_l0", "dma.l1_to_l0.mx"):
            m, n = int(a("m_copy", a("m_dst"))), int(a("n_dst"))
            return _ceil_bw(_a16(m) * n * elem(1), self.k("l0_bandwidth_bytes_per_cycle", 128)) + H
        if oc == "dma.l1_to_l0.img2col":
            return _ceil_bw(_a16(int(a("m_ext"))) * int(a("k_ext")) * elem(1), self.k("l0_bandwidth_bytes_per_cycle", 128)) + H
        if oc == "dma.l1_to_bt":
            return _ceil_bw(int(a("n", 1)) * elem(0), self.k("l0_bandwidth_bytes_per_cycle", 128)) + H
        if oc in ("cube.mmad", "cube.mmad.mx"):
            M, N, K = int(a("M")), int(a("N")), int(a("K"))
            bits = max((operand_type(i).dtype.bits for i in (1, 2) if isinstance(operand_type(i), MemType)), default=16)
            kc = 256 // max(bits, 1)
            macs = _a16(M) * _a16(N) * (-(-K // kc) * kc)
            rate = self.k("matmul_macs_per_cycle_8bit", 8192) if bits <= 8 else self.k("matmul_macs_per_cycle_16bit", 4096) if bits <= 16 \
                else self.k("matmul_macs_per_cycle_32bit", 256)
            return _linear(macs, rate, self.k("mmad_overhead_cycles", 67))
        if oc.startswith("dma.l0c_to_gm."):
            M, N, M_src = int(a("M")), int(a("N")), int(a("M_src", a("M")))
            src = -(-N // 16) * _a16(M_src) * 16 * elem(1)
            if oc.endswith("nz2dn"):
                dst = ((N - 1) * int(a("M_dst", M)) + M) * elem(0)
            else:
                dst = ((M - 1) * int(a("N_dst", N)) + N) * elem(0)
            return _linear(max(src, dst), self.k("l0c_to_gm_bandwidth_bytes_per_cycle", 32), self.k("l0c_to_gm_overhead_cycles", 300))
        if oc == "dma.l0c_to_l1":
            M, N = int(a("M")), int(a("N"))
            src = -(-N // 16) * _a16(int(a("M_src", M))) * 16 * elem(1)
            dst = -(-N // c0(0)) * _a16(int(a("M_dst", M))) * c0(0) * elem(0)
            return _ceil_bw(max(src, dst), self.k("l0c_to_l1_bandwidth_bytes_per_cycle", 128)) + H
        if oc == "dma.l0c_to_ub":
            M, N = int(a("M")), int(a("N"))
            mode = a("dual_mode", "splitm")
            mode = mode.name if isinstance(mode, Ident) else str(mode)
            payload = (M // 2 if mode == "splitm" else M) * N * elem(0)
            src = -(-N // 16) * _a16(M) * 16 * elem(1)
            return _ceil_bw(max(src, payload), self.k("l0c_to_ub_bandwidth_bytes_per_cycle", 256)) + H
        if oc == "dma.ub_to_l1.nd2nz":
            m, n = int(a("m_src")), int(a("n_src"))
            return _ceil_bw(m * n * elem(1), self.k("ub_to_l1_nd2nz_bandwidth_bytes_per_cycle", 32)) + H
        if oc == "dma.ub_to_l1.nz":
            m, n = int(a("m_src")), int(a("n_src"))
            return _ceil_bw(m * n * elem(1), self.k("ub_to_l1_bandwidth_bytes_per_cycle", 256)) + H
        if oc == "dma.ub_to_ub":
            n = int(a("n_burst", 1)) * int(a("burst_len", 1)) * 32
            return _linear(n, self.k("ub_to_ub_bandwidth_bytes_per_cycle", 256), self.k("ub_to_ub_overhead_cycles", 29))
        if oc == "vec.sort32":
            return _linear(int(a("repeat", 1)) * 32 * 4, self.k("sort32_bandwidth_bytes_per_cycle", 8), self.k("sort32_overhead_cycles", 23))
        if oc == "vec.mergesort4":
            return _linear(int(a("repeat", 1)) * 4 * int(a("length_per_seq", 1)) * 8, self.k("mergesort4_bandwidth_bytes_per_cycle", 16),
                           self.k("mergesort4_overhead_cycles", 38))
        if oc == "vec.mergesort_2seq":
            return _linear((int(a("size1", 0)) + int(a("size2", 0))) * 8, self.k("mergesort_2seq_bandwidth_bytes_per_cycle", 16),
                           self.k("mergesort_2seq_overhead_cycles", 38))
        if oc == "simt.launch":
            return int(a("instruction_count", 0)) * int(self.k("simt_cycles_per_instruction", 4)) + H
        if oc.startswith("vec."):
            biggest = 0.0
            for i, _ in enumerate(op.operands):
                t = operand_type(i)
                if isinstance(t, MemType):
                    n = 1
                    for d in t.dims:
                        n *= int(val(Value(d.name, None)) if not isinstance(d, int) else d)  # type: ignore[arg-type]
                    biggest = max(biggest, n * _bytes(t.dtype))
            inst = int(a("repeat", 0)) or max(1, math.ceil(biggest / self.k("vpipe_bytes_per_instruction", 256)))
            table = self.table.get("vpipe_instruction_cycles", {})
            cpi = float(table.get(_legacy(oc), table.get("default", self.k("vpipe_default_instruction_cycles", 2))))
            return int(inst * cpi) + H
        if oc.startswith("scalar.") or oc.startswith("core.") or oc.startswith("mem.") or oc.startswith("list."):
            return 1
        return int(self.k("sync_marker_cycles", 1)) + H

    # -- VF -----------------------------------------------------------------------------------

    @staticmethod
    def vf_class(op: Op) -> str | None:
        """LD | ST | SU | EX | None (free) for one executed vf op."""
        oc = op.opcode
        if oc in ("vf.load", "vf.load_cont", "vf.load_interleave", "vf.load_unalign", "vf.load_unalign_pre", "vf.gather_copy", "vf.gather",
                  "vf.gathermask", "vf.gatherb"):
            return "LD"
        if oc in ("vf.store", "vf.store_cont", "vf.store_interleave", "vf.store_unalign", "vf.store_unalign_post", "vf.scatter_copy"):
            return "ST"
        if oc.startswith("scalar.") or oc.startswith("list."):
            return "SU"
        if oc in ("vf.reg", "vf.mask", "vf.unalign", "vf.reinterpret", "vf.ub_cursor", "vf.barrier", "mem.alloc", "mem.slice", "mem.view", "mem.reinterpret",
                  "cf.for", "cf.if", "cf.return", "cf.break", "cf.continue"):
            return None
        if oc.startswith("vf."):
            return "EX"
        return None

    def vf_latency(self, op: Op, cls: str) -> float:
        name = _legacy(op.opcode)
        if name in self._lat:
            return float(self._lat[name])
        return float(self._lat.get({"LD": "_default_ld", "ST": "_default_st", "SU": "_default_su"}.get(cls, ""),
                                   self.k("vf_micro_default_latency", 6) if cls == "EX" else {"LD": 9, "ST": 9, "SU": 2}[cls]))

    @staticmethod
    def _ways(stride: int, banks: int) -> int:
        if stride <= 0:
            return 1
        v2 = 0
        while stride % 2 == 0:
            stride //= 2
            v2 += 1
        return min(1 << v2, banks)

    def vf_interval(self, op: Op, cls: str, latency: float, stride: int | None) -> float:
        name = _legacy(op.opcode)
        if cls == "EX":
            if name in self._ex_override:
                return float(self._ex_override[name])
            return max(self.k("vf_ex_issue_floor", 0.1), self.k("vf_ex_issue_slope", 0.039) * latency + self.k("vf_ex_issue_intercept", 0.5039))
        if cls == "SU":
            return float(self._issue.get("SU", 0.48))
        if cls in ("LD", "ST") and op.opcode in ("vf.load", "vf.store") and stride is not None and bool(self.table.get("vf_bank_conflict", True)):
            p = "vf_ld" if cls == "LD" else "vf_st"
            base = self.k(f"{p}_bank_base", 3.5 if cls == "LD" else 3.0)
            pen = self.k(f"{p}_bank_stride_penalty", 2.25 if cls == "LD" else 0.0)
            per_way = self.k(f"{p}_bank_per_way", 2.0 if cls == "LD" else 1.0)
            banks = int(self.k(f"{p}_bank_banks", 4 if cls == "LD" else 8))
            ways = self._ways(stride, banks)
            return max(1.0, base + (pen if stride > 1 else 0.0) + per_way * (ways - 1))
        return float(self._issue.get(cls, 1.0))

    def vf_cost(self, ops: list[tuple[Op, int | None, bool]]) -> VfCost:
        """Cycles of one vf call from its executed ops: ``(op, stride or None, folded)``; ``folded`` marks scalar ops whose
        operands were all compile-time constants (free, like the old SU constant folding)."""
        busy: dict[str, float] = {"LD": 0.0, "ST": 0.0, "SU": 0.0, "EX": 0.0}
        counts: dict[str, int] = {"LD": 0, "ST": 0, "SU": 0, "EX": 0}
        ready: dict[str, float] = {}
        crit = 0.0
        max_lat = 0.0
        for op, stride, folded in ops:
            cls = self.vf_class(op)
            if cls is None or (cls == "SU" and folded):
                continue
            lat = self.vf_latency(op, cls)
            busy[cls] += self.vf_interval(op, cls, lat, stride)
            counts[cls] += 1
            max_lat = max(max_lat, lat)
            spec = REGISTRY.find(op.opcode)
            start = 0.0
            if spec is not None:
                for i, o in enumerate(spec.operands):
                    if i < len(op.operands) and o.access == "read" and isinstance(op.operands[i], Value) and isinstance(op.operands[i].type, RegType):
                        start = max(start, ready.get(op.operands[i].name, 0.0))
                for i, o in enumerate(spec.operands):
                    if i < len(op.operands) and o.access in ("write", "readwrite") and isinstance(op.operands[i], Value) \
                            and isinstance(op.operands[i].type, RegType):
                        ready[op.operands[i].name] = start + lat
            crit = max(crit, start + lat)
        if not any(counts.values()):
            return VfCost(0, busy, crit, max_lat, counts)
        makespan = max(max(busy.values()), max_lat, self.k("vf_crit_slack", 0.46) * crit)
        cycles = int(round(self.k("vf_fixed_overhead_cycles", 46) + makespan)) + self.head_overhead
        return VfCost(cycles, busy, crit, max_lat, counts)

    @property
    def intra_core_latency(self) -> int:
        return int(self.k("intra_core_sync_latency_cycles", 200))


def load_model(device_type: str) -> CycleModel:
    name = "a5_cycle_model.json" if device_type in ("950", "950pr") else "a2_cycle_model.json"
    path = _TABLES / name
    if not path.exists():
        raise FileNotFoundError(f"cycle model table not found: {path}")
    table = json.loads(path.read_text(encoding="utf-8"))
    return CycleModel(table, str(table.get("device_family", "a5")))


__all__ = ["CycleModel", "VfCost", "load_model"]
