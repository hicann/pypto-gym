# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A functional interpreter for Surface IR: the semantic reference of the vertical slice (M2).

It executes a module the way the old simulator did — one thread per core lane (cube, vec0,
vec1), scalars as Python numbers, tensors as torch tensors, cross-core flags as counters — but
without pipes, hazards or cycles (those need Lowered IR and arrive in M4). Semantics that must
be bit-exact with the old simulator (mmad block shapes, DMA extents, register lane counts,
casts, fp4 packing, mutex token protocol) follow the code-referenced facts in
``docs/rfc/0003-functional-goldens.md``'s companion notes and the old ``simulator/`` package.

Layout is logical: on-chip tensors are row-major torch tensors; NZ is a no-op here — except
where the *program* packs fractals itself: a UB window written with ``reg_to_ub(..., blk_stride)``
and copied as ``ub.nz()`` is de-fractalised on its way to L1 (``MemRef.layout``).

M3 extended the model to the whole a5 corpus: the explicit DMA / cube instructions (``dma.*``,
``cube.mmad``, ``conv2d`` im2col), the microscaling family (``mx_scale_nd2nz`` / ``l1_to_l0.mx`` /
``mmad.mx`` / ``matmul_mx`` with the old 32-byte scale-block layout, fp4 carriers decoded through
the host codec of ``ascriptor.dtypes``), every LoadAlign / StoreAlign distribution, integer registers computed exactly in int64
(uint64 through signed views), strided 32-byte block copies, and the cross-core primitives on
the side that owns them (``sync.crosscore.*`` carry ``side`` in the registry). Workspaces are GM
tensors allocated before the fork; the launch takes ``block_dim`` from the recording it replays.
"""

from __future__ import annotations

import math
import warnings
import threading
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from ...ir.lint import BLOCK_BYTES, HardwareWarning

from ...devices import SIMT_UB_CAP_KB, DeviceProfile
from ...devices import load as load_profile
from ...ir import FuncRef, Function, Ident, Literal, Module, Op, Value
from ...ir.registry import REGISTRY
from ...ir.scalar_math import division_error, integer_divmod, rounding
from ...ir.saturation import SAT_BITS, SAT_DEFAULTS, TRUNCATING_CASTS, cast_uses_ctrl
from ...ir.types import (
    DTYPES,
    BufType,
    CellType,
    DimValue,
    DType,
    EventType,
    MaskType,
    MemType,
    Product,
    RegType,
    ScalarType,
    UnalignRegType,
)
from ...dtypes.fp4_fp32 import fp32_to_fp4_e1m2

REG_BYTES = 256
MAX_FLAGS = 11  # storage ceiling; profiles validate the public logical ID range
_TORCH_DTYPES = {
    "b1": torch.bool, "i8": torch.int8, "u8": torch.uint8, "i16": torch.int16, "u16": torch.uint16, "i32": torch.int32,
    "u32": torch.uint32, "i64": torch.int64, "u64": torch.uint64, "f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32,
    "e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2, "hif8": torch.uint8, "fp4_e2m1": torch.uint8, "fp4_e1m2": torch.uint8,
    "e8m0": torch.uint8, "c32": torch.complex32, "c64": torch.complex64,  # complex regs compute in complex64 (_reg_values)
    "i4": torch.uint8,  # packed signed int4: the window resolves as raw bytes, the mmad decodes two nibbles per byte
}


def torch_dtype(dt: DType) -> torch.dtype:
    return _TORCH_DTYPES[dt.name]


def elem_bytes(dt: DType) -> int:
    return 1 if dt.bits <= 8 else dt.bits // 8


def reg_lanes(dt: DType) -> int:
    return REG_BYTES // elem_bytes(dt)


class SimError(RuntimeError):
    pass


class SimDeadlock(SimError):
    pass


class SimTimeout(SimError):
    """The wall-clock limit passed while a lane was still running: a slow run, which is not a stall."""


# --------------------------------------------------------------------------- runtime values


class VecOnly:
    """A value that exists only on vector cores (derived from vec_idx / sub_block_idx).

    The old splitter classified such scalar chains and every loop or branch they control as
    vector-only and dropped them from the cube stream; the reference interpreter reproduces that
    by tainting: on the cube lane these ops produce ``VEC_ONLY`` instead of running, control flow
    they decide is skipped, and a cube-side op that consumes one is an error.
    """

    def __repr__(self) -> str:
        return "<vec-only>"


VEC_ONLY = VecOnly()


class Cell:
    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


@dataclass(frozen=True)
class ListRef:
    """A ``gmlist`` parameter: its members live in ``Machine.gm`` under ``name[i]`` (``Machine.lists``)."""

    name: str


@dataclass
class MemRef:
    """A window onto a tensor: GM (``alloc=None``, ``base`` is the parameter name) or on-chip (``base`` = the allocation name)."""

    space: str
    dtype: DType
    base: str  # GM parameter name, or the allocation name (mem.alloc result)
    slot: int
    offsets: tuple[int, ...]
    extents: tuple[int, ...]
    shape: tuple[int, ...] | None = None  # a reshaped view of the storage (mem.reshape)
    layout: str | None = None  # "nz": a UB window whose bytes are NZ fractals (``.nz()``); copies to L1 de-fractalise
    gm_strides: tuple[int, ...] | None = None  # mem.view (RFC-0010): explicit element strides per dim
    # element offset of the coordinate system's origin in the flat storage: a mem.view's origin, or where a
    # mem.reshape / reinterpret {tile} of an offset window starts (RFC-0010 §10)
    view_offset: int = 0

    def sub(self, offsets: tuple[int, ...], extents: tuple[int, ...]) -> MemRef:
        return MemRef(self.space, self.dtype, self.base, self.slot, tuple(a + b for a, b in zip(self.offsets, offsets, strict=True)), extents,
                      self.shape, self.layout, self.gm_strides, self.view_offset)

    @property
    def rows(self) -> int:
        return self.extents[0]

    @property
    def cols(self) -> int:
        return self.extents[-1]


@dataclass
class BufRef:
    alloc: str  # allocation name
    slots: int
    elem: MemType
    dims: tuple[int, ...]


@dataclass
class RingRef:
    """A GMBuff workspace ring (RFC-0009): slot ``b`` of core ``c`` is the 2D piece
    ``ws:<name>:<c * slots + b>``."""

    name: str
    slots: int
    per_core: bool
    elem: MemType
    dims: tuple[int, ...]  # one slot: (rows, cols)


@dataclass
class RegRef:
    """256 bytes viewed as ``dtype``; ``valid`` lanes carry data after a narrowing pack."""

    bytes: torch.Tensor  # uint8[256]
    dtype: DType
    valid: int

    def tensor(self) -> torch.Tensor:
        return self.bytes.view(torch_dtype(self.dtype))

    @property
    def lanes(self) -> int:
        return self.bytes.numel() // elem_bytes(self.dtype)


@dataclass
class MaskRef:
    """A 256-bit predicate payload with a live typed-lane tensor.

    ``bits`` retains its constructor alias for model callers. Physical reads
    overlay those observable lane bits onto the retained payload; physical
    writes update both. Bits between typed lanes therefore survive routing and
    UB transfers, without making a logical tensor mutation stale.
    """

    bits: torch.Tensor  # bool[lanes]; b32 observes physical bits 0, 4, ..., 252
    _payload: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._payload = torch.zeros(256, dtype=torch.bool, device=self.bits.device)

    @property
    def bit_stride(self) -> int:
        return 256 // self.bits.numel()

    def physical(self) -> torch.Tensor:
        """Snapshot every physical bit, including externally mutated lanes."""
        payload = self._payload.clone()
        payload[::self.bit_stride] = self.bits
        return payload

    def write_physical(self, payload: torch.Tensor) -> None:
        self._payload.copy_(payload)
        self.bits.copy_(self._payload[::self.bit_stride])

    def write_lanes(self, bits: torch.Tensor) -> None:
        """Write generated lane predicates, clearing all other physical bits."""
        payload = torch.zeros_like(self._payload)
        payload[::self.bit_stride][:bits.numel()] = bits
        self.write_physical(payload)


@dataclass
class Flag:
    kind: str  # vc | cv
    id: int
    depth: int  # credit count, always explicit: see the `sync.mutex` declaration


# Appended to every cross-core deadlock: the protocol a blocked mutex call is failing to complete.
CREDIT_RULE = (
    "\n  the four calls are producer `lock` (take a slot) / `ready` (publish) and consumer `wait` (acquire) /"
    " `free` (return the credit); `ready` of cycle i orders `wait` of cycle i, while `lock` of cycle i blocks"
    " on `free` of cycle i-depth. A count that never moves names the call that is missing: no consumer `free`"
    " stalls `lock`, no producer `ready` stalls `wait`, and a `ready` published more often than it is waited"
    " for leaves a token no reader consumes."
    "\n  see library/docs/api/synchronization.md#cross-side-ownership"
)


GRACE = 5.0  # seconds the parent keeps waiting after the limit, so that a lane's own report arrives first


def _timeout_note(timeout: float) -> str:
    """Appended where the limit, not a stall, ended the run: a deadlock is reported within half a second."""
    return (f"\n  the {timeout:g} s limit passed while a lane was still running. No stall was detected - every live"
            " lane blocked with no wait ending is reported as SimDeadlock at once - so this run is slow, or the host"
            " is loaded: raise the timeout.")

# Appended to both register-footprint guards. A narrow row view is the usual cause and supplies no
# implicit mask, so the instruction keeps its full lane footprint (M10-057, M10-060).
NARROW_VIEW_HINT = (
    ". A view narrower than the register does not mask the instruction: an A5 b16 `NORM_B16` store"
    " through a 64-element row still drives 128 lanes, and a full-register load still reads past the"
    " row's end. Pass the matching mask (`MaskReg(..., init_mode=MaskType.LOWHALF)` for the low half)"
    " or widen the allocation to the instruction's footprint"
    "\n  see library/docs/api/registers.md#load-compute-and-store"
)


# --------------------------------------------------------------------------- machine


@dataclass
class CoreGroup:
    """One cube core with its two vector cores: shared L1/L0/BT, per-lane UB, flag counters."""

    index: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    cond: threading.Condition = field(default_factory=threading.Condition)
    shared: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)  # (alloc, slot) -> L1/L0/BT tensor
    ub: dict[tuple[int, int, int], torch.Tensor] = field(default_factory=dict)  # (sub, alloc, slot) -> logical UB tensor
    ub_storage: dict[tuple[int, int, int], torch.Tensor] = field(default_factory=dict)  # independent aligned slot backing
    vec_wait: list[list[int]] = field(default_factory=lambda: [[0] * MAX_FLAGS, [0] * MAX_FLAGS])  # cube -> vec tokens
    vec_ready: list[list[int]] = field(default_factory=lambda: [[0] * MAX_FLAGS, [0] * MAX_FLAGS])  # vec -> cube tokens


@dataclass
class Lane:
    side: str  # cube | vec
    group: CoreGroup
    sub: int  # 0 | 1 for vec lanes, -1 for the cube lane
    cube_idx: int
    vec_idx: int
    cube_num: int
    vec_num: int
    sub_block_num: int
    spr_mask: Any = None  # the vector unit's mask special register (vec.set_mask*)
    sat_flags: dict[str, bool] = field(default_factory=lambda: SAT_DEFAULTS.copy())
    sat_written: set[str] = field(default_factory=set)  # CTRL bits this lane wrote in the launch (RFC-0007 §3)
    local_mutexes: list[tuple[str, int] | None] = field(default_factory=lambda: [None] * 32)


class Machine:
    def __init__(self, module: Module, profile: DeviceProfile | None = None, *, timeout: float = 60.0) -> None:
        self.module = module
        self.profile = profile or load_profile(module.device or "950")
        self.crosscore_flag_count = self.profile.crosscore_id_max + 1
        self.timeout = timeout
        self.deadline = 0.0
        self.mode = _ident(module.attrs.get("mode", "mix"))
        self.functions = {f.name: f for f in module.functions}
        self.hw_warnings: list[tuple[int, str]] = []  # hardware notes met at run time (D-051), warned after the lanes join
        self.hw_seen: set[tuple[int, str]] = set()
        self.hw_lock = threading.Lock()
        self.ctrl_saves: dict[int, bool] | None = None  # core.get_sat_flag id -> its value only restores that bit
        self.ctrl_noted: set[tuple[int | None, tuple[str, ...]]] = set()  # (op id, unwritten bits) already noted
        kernels = [f for f in module.functions if f.kind == "kernel"]
        self.sides: dict[str, Function] = {}
        if kernels:
            if len(kernels) != 1:
                raise SimError("the module must contain exactly one kernel")
            self.kernel = kernels[0]
        else:  # lowered: one func per side, the kernel's attributes in module meta
            funcs = [f for f in module.functions if f.kind == "func"]
            if not funcs:
                raise SimError("the module has neither a kernel nor per-side funcs")
            for f in funcs:
                self.sides[_ident(f.attrs.get("side", "vec"))] = f
            self.kernel = replace(funcs[0], kind="kernel", name=str(module.attrs.get("meta", {}).get("kernel", funcs[0].name)),
                                  attrs=dict(module.attrs.get("meta", {})))
        self.gm: dict[str, torch.Tensor] = {}
        self.lists: dict[str, list[str]] = {}  # gmlist parameter -> the storage keys of its members
        self.scalars: dict[str, Any] = {}
        self.errors: list[BaseException] = []
        self.stopping = False  # set once a lane failed or the limit passed while lanes still run: they leave at their next op
        self.blocked: dict[str, str] = {}
        # deadlock detection across every lane and process: lanes alive, lanes inside a wait, waits left (progress),
        # and the verdict — every lane alive is blocked and nothing left a wait for half a second
        self.live: Any = None
        self.nblocked: Any = None
        self.progress: Any = None
        self.dead: Any = None
        self.count_lock: Any = None
        # one lane of a process runs at a time (see _run_lanes_threaded): the lock is made in each process that runs lanes
        self.turn: threading.Lock | None = None
        self.turn_lane: Lane | None = None
        self.barriers: dict[str, Any] = {}  # side -> barrier over every lane of that side (allvec_wait / allcube_wait)
        self.gm_lock: Any = threading.Lock()
        # simt.launch runs serialized across vec lanes (see op_simt_launch); built in run() from the mp context
        self.simt_turn: Any = None
        self.simt_cond: Any = None
        self.simt_rank: dict[int, int] = {}
        self.tracer: Any = None  # pipesim.Tracer when the pipe-level simulator drives the run

    def capacity(self, space: str) -> int | None:
        """Bytes of one memory space on this device (UB keeps 216 KB when the kernel launches SIMT code)."""
        kb = self.profile.capacities_kb.get(space)
        if kb is None:
            return None
        cap = int(kb * 1024)
        if space == "ub" and any(o.opcode == "simt.launch" for o in self.module.walk()):
            cap = min(cap, SIMT_UB_CAP_KB * 1024)
        return cap

    # -- launch -------------------------------------------------------------------------------

    def run(self, args: dict[str, Any], block_dim: int | None = None, processes: bool | None = None) -> None:
        """Execute the kernel. Core groups run in forked processes (GM tensors in shared memory) when
        ``processes`` is on — the default where ``fork`` exists and there is more than one group — with one
        thread per lane inside each process; otherwise every lane is a thread of this process. The lanes of one
        process take turns (``_run_lanes_threaded``). A tracer's tasks come back from every child through a queue
        (ops by id) and are merged here."""
        import multiprocessing as mp
        import time

        torch.set_num_threads(1)  # the old simulator ran single-threaded BLAS; block shapes and threads decide the bits
        for p in self.kernel.params:
            if p.name not in args:
                raise SimError(f"missing argument {p.name}")
            if isinstance(p.type, MemType) and p.type.space == "gmlist":
                members = args[p.name]
                if not isinstance(members, (list, tuple)) or not all(isinstance(m, torch.Tensor) for m in members):
                    raise SimError(f"argument {p.name} must be a list of torch tensors")
                self.lists[p.name] = [f"{p.name}[{i}]" for i in range(len(members))]
                for key, m in zip(self.lists[p.name], members, strict=True):
                    self.gm[key] = m
            elif isinstance(p.type, MemType):
                t = args[p.name]
                if not isinstance(t, torch.Tensor):
                    raise SimError(f"argument {p.name} must be a torch tensor")
                self.gm[p.name] = t
            else:
                self.scalars[p.name] = args[p.name]
        from ...runtime.launch_config import launch_block_dim, resolve_block_dim

        block_dim = launch_block_dim(self.module, block_dim, "the simulator")
        requested = block_dim if block_dim is not None else self.kernel.attrs.get("block_dim")
        block_dim = resolve_block_dim(requested, self.scalars)
        self.deadline = time.monotonic() + self.timeout
        self.stopping = False
        self._allocate_workspaces(block_dim)
        lanes = self._lanes(block_dim)
        ctx0 = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp
        self.gm_lock = ctx0.Lock()
        # the ordered-launch gate: each vec lane's rank among the lanes actually launched
        vec_ids = sorted(lane.vec_idx for lane in lanes if lane.side == "vec")
        self.simt_rank = {v: i for i, v in enumerate(vec_ids)}
        self.simt_turn = ctx0.Value("i", 0, lock=False)
        self.simt_cond = ctx0.Condition()
        self.live, self.nblocked, self.progress, self.dead = (ctx0.Value("i", 0, lock=False) for _ in range(4))
        self.count_lock = ctx0.Lock()
        self.collectives = {}
        scopes = {side: [lane for lane in lanes if lane.side == side] for side in ("vec", "cube")}
        scopes.update({f"vec@{group}": [lane for lane in lanes if lane.side == "vec" and lane.group.index == group]
                       for group in {lane.group.index for lane in lanes}})
        for scope, members in scopes.items():
            if members:
                self.collectives[scope] = ({_lane_name(lane): i for i, lane in enumerate(members)}, ctx0.Condition(),
                                           ctx0.Array("q", len(members) * self.crosscore_flag_count, lock=False),
                                           ctx0.Array("q", len(members) * self.crosscore_flag_count, lock=False))
        groups: dict[int, list[Lane]] = {}
        for lane in lanes:
            groups.setdefault(lane.group.index, []).append(lane)
        if processes is None:
            processes = len(groups) > 1 and "fork" in mp.get_all_start_methods()
        if not processes:
            self._run_lanes_threaded(lanes)
            return
        for t in self.gm.values():
            t.share_memory_()
        ctx = mp.get_context("fork")
        queue = ctx.Queue()  # one message per child: ("tasks", payload) / ("done", None) / ("error", text), + its notes
        procs = []
        for idx, group_lanes in groups.items():
            proc = ctx.Process(target=self._group_main, args=(group_lanes, queue), name=f"core{idx}", daemon=True)
            proc.start()
            procs.append(proc)
        failures: list[str] = []
        payloads: list[Any] = []
        notes: list[tuple[int, str]] = []
        for _ in procs:  # read before joining: a child blocks on a large message until it is read
            try:
                kind, payload, child_notes = queue.get(timeout=self.timeout + GRACE)
            except Exception:  # noqa: BLE001 - queue.Empty
                break
            notes.extend(child_notes)
            if kind == "error":
                failures.append(payload)
                break  # the other children are blocked or about to fail the same way: do not wait for them
            elif kind == "tasks":
                payloads.append(payload)
        for proc in procs:
            proc.join(timeout=GRACE)
            if proc.is_alive():
                proc.kill()
                failures.append(f"{proc.name}: still running after {self.timeout}s")
        self._warn_hardware(notes)
        if failures:
            if any("SimDeadlock" in f for f in failures):
                raise SimDeadlock("\n".join(failures))
            if any("SimTimeout" in f or "still running" in f for f in failures):
                raise SimTimeout("\n".join(failures) + _timeout_note(self.timeout))
            raise SimError("\n".join(failures))
        if self.tracer is not None:
            ops = {op.id: op for op in self.module.walk() if op.id is not None}
            for payload in payloads:
                self.tracer.absorb(payload, ops)

    def _allocate_workspaces(self, block_dim: int | None = None) -> None:
        """GM workspaces (``mem.workspace``) live beside the parameters; sized from the launch scalars. A
        dimension may also be a direct core-count query (``GetCubeNum()`` and friends): those resolve from
        the launch geometry — the same numbers ``_lanes`` hands every lane."""
        if self.mode == "vec":
            geom = {"core.vec_num": block_dim or self.profile.vec_cores, "core.cube_num": 0}
        else:
            n = block_dim or self.profile.cube_cores
            geom = {"core.cube_num": n, "core.vec_num": 2 * n}
        geom["simt.block_num"] = geom["core.vec_num" if self.mode == "vec" else "core.cube_num"]
        def allocate_for_function(f: Function) -> None:
            """Resolve one function's workspace dimensions against its own scalar defs."""
            defs = {op.results[0].name: op for op in f.body.walk() if op.results and op.results[0].name}
            sets = {op.operands[0].name for op in f.body.walk()
                    if op.opcode == "scalar.set" and op.operands and hasattr(op.operands[0], "name")}

            def host_eval(name: str, at: Any) -> int:
                """A dimension as the host sees it before launch: launch scalars, core-count queries, and
                write-once Vars over scalar arithmetic of those (the cce host tiling rule's mirror)."""
                if name in self.scalars:
                    return int(self.scalars[name])
                d = defs.get(name)
                if d is None:
                    raise SimError(f"workspace #{at.id}: dimension %{name} is not a launch scalar")
                if d.opcode in geom:
                    return int(geom[d.opcode])
                if d.opcode == "scalar.cell" and name not in sets:
                    init = d.attrs.get("init")
                    return host_eval(init.name, at) if hasattr(init, "name") else int(init)
                if d.opcode.startswith("scalar."):
                    kind = d.opcode[7:]
                    if kind == "const":
                        return int(d.attrs["value"])
                    a = [host_eval(o.name, at) if hasattr(o, "name") else int(o) for o in d.operands]
                    fold = {"add": lambda: a[0] + a[1], "sub": lambda: a[0] - a[1], "mul": lambda: a[0] * a[1],
                            "div": lambda: integer_divmod(a[0], a[1], rounding(d))[0],
                            "mod": lambda: integer_divmod(a[0], a[1], rounding(d))[1], "min": lambda: min(a),
                            "max": lambda: max(a), "ceil_div": lambda: -(-a[0] // a[1]), "neg": lambda: -a[0],
                            "cast": lambda: a[0], "shl": lambda: a[0] << a[1], "shr": lambda: a[0] >> a[1]}
                    if kind == "align":
                        n = int(d.attrs["n"])
                        return -(-a[0] // n) * n
                    if kind in fold:
                        return fold[kind]()
                raise SimError(f"workspace #{at.id}: dimension %{name} is not a launch scalar (defined by {d.opcode})")

            for op in f.body.walk():
                if op.opcode != "mem.workspace":
                    continue
                t = op.results[0].type
                if isinstance(t, BufType) and "gmbuff_dims" in op.attrs:
                    # a GMBuff ring (RFC-0009): one 2D piece per (core, slot), so a slot selection
                    # is a plain 2D window and the ring's ideal no-alias semantics hold by storage
                    full = [x if isinstance(x, int) else host_eval(x.name, op) for x in op.attrs["gmbuff_dims"]]
                    pieces = 1
                    for d in full[:-2]:
                        pieces *= d
                    for i in range(pieces):
                        self.gm[f"ws:{op.attrs['name']}:{i}"] = _poison((full[-2], full[-1]), t.elem.dtype)
                    continue
                assert isinstance(t, MemType)
                dims = []
                for d in t.dims:
                    if isinstance(d, int):
                        dims.append(d)
                    elif isinstance(d, DimValue):
                        dims.append(host_eval(d.name, op))
                    else:
                        raise SimError(f"workspace #{op.id}: dimension {d} is not a launch scalar")
                self.gm[f"ws:{op.attrs['name']}"] = _poison(tuple(dims), t.dtype)

        for function in self.module.functions:
            allocate_for_function(function)

    def _group_main(self, lanes: list[Lane], queue: Any) -> None:
        torch.set_num_threads(1)
        try:
            self._run_lanes_threaded(lanes, warn=False)
            if self.tracer is not None:
                queue.put(("tasks", self.tracer.export([_lane_name(lane) for lane in lanes]), self.hw_warnings))
            else:
                queue.put(("done", None, self.hw_warnings))
        except BaseException as exc:  # noqa: BLE001 - reported to the parent
            queue.put(("error", f"{type(exc).__name__}: {exc}", self.hw_warnings))
        queue.close()
        queue.join_thread()
        if self.errors:
            raise SystemExit(1)

    def note_hardware(self, op: Op, msg: str) -> None:
        """A hardware note met at run time (D-051): kept once per op and message, warned from the main thread."""
        key = (op.id if op.id is not None else -1, msg)
        with self.hw_lock:
            if key in self.hw_seen:
                return
            self.hw_seen.add(key)
            where = f" ({op.loc.chain[0]})" if op.loc else ""
            self.hw_warnings.append((key[0], f"#{op.id}{where}: {msg}"))

    def _warn_hardware(self, notes: list[tuple[int, str]] = ()) -> None:
        """Warn every note in op order, whichever lane or core-group process met it first."""
        for note in notes:
            if note not in self.hw_warnings:
                self.hw_warnings.append(tuple(note))
        for _, msg in sorted(self.hw_warnings):
            warnings.warn(msg, HardwareWarning, stacklevel=4)
        self.hw_warnings.clear()

    def ctrl_read_saves(self, op: Op) -> bool:
        """Whether a `core.get_sat_flag` value only restores its own bit, so the read saves CTRL (RFC-0007 §3)."""
        if self.ctrl_saves is None:
            self.ctrl_saves = {k: v for f in self.module.functions for k, v in _ctrl_saves(f).items()}
        return self.ctrl_saves.get(op.id, False)

    def _run_lanes_threaded(self, lanes: list[Lane], *, warn: bool = True) -> None:
        """One thread per lane, taking turns: a lane runs until it waits for another lane (``Interp._wait``, the SIMT
        launch gate). Torch lets go of the interpreter lock for every tensor op, so lanes that ran at once handed it
        to one another at each op and switching threads took most of the run (RFC-0006 §9). Forked core groups still
        run in parallel, and a lane's results and trace are its program order either way."""
        threads = [threading.Thread(target=self._run_lane, args=(lane,), name=_lane_name(lane), daemon=True) for lane in lanes]
        import time

        self.turn, self.turn_lane = threading.Lock(), None
        for t in threads:
            t.start()
        for t in threads:  # one limit for the run: a lane that outlives it buys the next lane no second limit
            t.join(timeout=max(0.0, self.deadline + GRACE - time.monotonic()))
        failure = self.errors[0] if self.errors else None
        if any(t.is_alive() for t in threads):
            # A lane inside a wait fails at the deadline on its own, so a thread alive here is still computing.
            waiting = _blocked_summary(self.blocked)
            failure = failure or SimTimeout("lanes still running" + (f"; waiting for them:{waiting}" if waiting else "")
                                            + _timeout_note(self.timeout))
            # A process that exits under lane threads inside torch ends in `terminate called without an active
            # exception` instead of this report, so the lanes are told to leave at their next op and given a moment.
            self.stopping = True
            leave = time.monotonic() + GRACE
            for t in threads:
                t.join(timeout=max(0.0, leave - time.monotonic()))
        if warn:
            self._warn_hardware()
        if failure is not None:
            raise failure

    def _lanes(self, block_dim: int | None) -> list[Lane]:
        lanes: list[Lane] = []
        if self.mode == "vec":
            n = block_dim or self.profile.vec_cores
            for i in range(n):
                g = CoreGroup(i)
                lanes.append(Lane("vec", g, 0, -1, i, 0, n, 1))
            return lanes
        n = block_dim or self.profile.cube_cores
        for c in range(n):
            g = CoreGroup(c)
            if self.mode in ("mix", "cube"):
                lanes.append(Lane("cube", g, -1, c, -1, n, 2 * n, 2))
            if self.mode == "mix":
                for s in (0, 1):
                    lanes.append(Lane("vec", g, s, c, 2 * c + s, n, 2 * n, 2))
        return lanes

    def take_turn(self, lane: Lane, cond: Any = None) -> None:
        """Become the lane of this process that runs. A caller inside ``cond`` lets go of it until then: the lane
        that has the turn may need ``cond`` to wake another lane, and it gives the turn up only when it waits."""
        if self.turn is None:
            return
        if cond is not None:
            cond.release()
        try:
            self.turn.acquire()
            self.turn_lane = lane
        finally:
            if cond is not None:
                cond.acquire()

    def give_turn(self, lane: Lane) -> None:
        """Let the next lane of this process run; a lane that does not have the turn gives nothing."""
        if self.turn_lane is lane:
            self.turn_lane = None
            self.turn.release()

    def _run_lane(self, lane: Lane) -> None:
        torch.set_num_threads(1)  # OpenMP's thread-count ICV is per thread: pin it in every lane thread
        with self.count_lock:
            self.live.value += 1  # before the turn: a lane that waits for its turn is running, not stalled
        try:
            self.take_turn(lane)
            interp = Interp(self, lane)
            env = {p.name: (self.gm_ref(p) if isinstance(p.type, MemType) else self.scalars[p.name]) for p in self.kernel.params}
            f = self.sides.get(lane.side, self.kernel) if self.sides else self.kernel
            if self.sides and lane.side not in self.sides:
                return  # a side the lowered module does not use (vec / cube mode)
            interp.run_function(f, env)
            held = [(index, state) for index, state in enumerate(lane.local_mutexes) if state is not None]
            if held:
                raise SimError(f"{_lane_name(lane)}: unreleased local mutexes {held}")
        except BaseException as exc:  # noqa: BLE001 - reported by run()
            self.errors.append(exc)
            self.stopping = True  # the run has failed: lanes taking turns leave at their next op, not one by one at their next wait
        finally:
            with self.count_lock:
                self.live.value -= 1
                self.progress.value += 1
            self.give_turn(lane)

    def gm_ref(self, p: Value) -> MemRef | ListRef:
        assert isinstance(p.type, MemType)
        if p.type.space == "gmlist":
            return ListRef(p.name)
        t = self.gm[p.name]
        return MemRef("gm", p.type.dtype, p.name, 0, (0,) * t.dim(), tuple(t.shape))

    # -- memory -------------------------------------------------------------------------------

    def resolve(self, ref: MemRef, lane: Lane, sub: int | None = None) -> torch.Tensor:
        """The torch view of a window. UB windows resolve in the lane's own UB unless ``sub`` says otherwise."""
        base = self.resolve_base(ref, lane, sub)
        if ref.gm_strides is not None:  # mem.view (RFC-0010): a strided window over the flat storage
            flat = base.reshape(-1)
            start = ref.view_offset + sum(o * s for o, s in zip(ref.offsets, ref.gm_strides, strict=True))
            return flat.as_strided(ref.extents, ref.gm_strides, flat.storage_offset() + start)
        idx = tuple(slice(o, o + e) for o, e in zip(ref.offsets, ref.extents, strict=True))
        return base[idx]

    def resolve_base(self, ref: MemRef, lane: Lane, sub: int | None = None) -> torch.Tensor:
        """The whole storage tensor a window lives in, typed and shaped as the window sees it."""
        if ref.space in ("gm", "ws"):
            base = self.gm[ref.base]  # type: ignore[index]
        elif ref.space == "ub":
            s = lane.sub if sub is None else sub
            base = lane.group.ub[(s, ref.base, ref.slot)]  # type: ignore[index]
        else:
            base = lane.group.shared[(ref.base, ref.slot)]  # type: ignore[index]
        return self._typed(base, ref)

    @staticmethod
    def _typed(base: torch.Tensor, ref: MemRef) -> torch.Tensor:
        """Reinterpreted (other dtype of any width) and reshaped views of a storage tensor."""
        want = torch_dtype(ref.dtype)
        if base.dtype != want:
            base = base.reshape(-1).view(want).reshape(*base.shape[:-1], -1) if base.dim() > 1 else base.view(want)
        if ref.shape is not None:  # a reshaped view: contiguous from its origin (a prefix when that is 0)
            flat = base.reshape(-1)
            n, start = math.prod(ref.shape), ref.view_offset
            base = flat[start:start + n].reshape(ref.shape) if start or n != flat.numel() else flat.reshape(ref.shape)
        return base

    def storage(self, ref: MemRef, lane: Lane, sub: int | None = None) -> tuple[torch.Tensor, int]:
        """The flat storage a window lives in and the window's origin in it (register loads/stores index flat)."""
        base = self.resolve_base(ref, lane, sub)
        if ref.gm_strides is not None:
            origin = ref.view_offset + sum(o * s for o, s in zip(ref.offsets, ref.gm_strides, strict=True))
            return base.reshape(-1), origin
        origin = ref.view_offset  # a rebased (reshape / tile) coordinate system starts here
        stride = 1
        for o, n in zip(reversed(ref.offsets), reversed(tuple(base.shape)), strict=True):
            origin += o * stride
            stride *= n
        if ref.space == "ub":
            s = lane.sub if sub is None else sub
            backing = lane.group.ub_storage.get((s, ref.base, ref.slot))
            if backing is not None:
                # The shape/pitch above stays logical; only the owned slot's
                # 32-byte allocation padding extends flat instruction access.
                return backing.view(torch_dtype(ref.dtype)), origin
        if ref.view_offset:  # the allocation from its start, still ending where the rebased system ends
            whole = self.resolve_base(replace(ref, shape=None), lane, sub).reshape(-1)
            return whole[:ref.view_offset + base.numel()], origin
        return base.reshape(-1), origin

    def storage_offset(self, ref: MemRef, lane: Lane, sub: int | None = None) -> int:
        """Where the window's storage starts, in elements of its dtype: the zero of trace footprints."""
        return self.resolve_base(replace(ref, shape=None), lane, sub).storage_offset()


def _lane_name(lane: Lane) -> str:
    return f"core{lane.cube_idx if lane.cube_idx >= 0 else lane.vec_idx}/{lane.side}{lane.sub if lane.sub >= 0 else ''}"


def _blocked_summary(blocked: dict[str, str], exclude: str = "", limit: int = 6) -> str:
    """The blocked lanes, grouped by the state they are in: one line per distinct state.

    Every core of a multi-core run usually blocks identically, so listing them one by one buries the
    one core whose counters differ — and that core is the one worth reading."""
    groups: dict[str, list[str]] = {}
    for name, state in sorted(blocked.items()):
        if name != exclude:
            groups.setdefault(state, []).append(name)
    if not groups:
        return ""
    rows = sorted(groups.items(), key=lambda row: (-len(row[1]), row[1][0]))
    lines = [f"\n  {len(names)} lane(s) [{names[0]}{', …' if len(names) > 1 else ''}] at {state}"
             for state, names in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"\n  and {len(rows) - limit} further distinct state(s)")
    return "".join(lines)


def _ident(v: Any) -> str:
    return v.name if isinstance(v, Ident) else str(v)


def _ctrl_saves(f: Function) -> dict[int, bool]:
    """Each `core.get_sat_flag` of ``f``: whether its value only restores the same bit, directly or through
    `!= 0`, `& 1` or a scalar cast. Such a read saves the launch state rather than depending on it."""
    uses: dict[str, list[tuple[Op, str]]] = {}
    for op in f.walk():
        for v in op.operands:
            if isinstance(v, Value):
                uses.setdefault(v.name, []).append((op, "operand"))
        for v in op.attr_values():
            uses.setdefault(v.name, []).append((op, "enable" if op.attrs.get("enable") is v else "attr"))

    def other(op: Op, name: str) -> Any:
        rest = [x for x in op.operands if not (isinstance(x, Value) and x.name == name)]
        return rest[0].value if len(rest) == 1 and isinstance(rest[0], Literal) else None

    def restores(name: str, mode: str, depth: int = 0) -> bool:
        for user, role in uses.get(name, ()):
            if user.opcode == "core.set_sat_flag" and role == "enable" and _ident(user.attrs.get("mode")) == mode:
                continue
            passes = (user.opcode == "scalar.cast"
                      or user.opcode == "scalar.cmp" and _ident(user.attrs.get("pred")) == "ne" and other(user, name) == 0
                      or user.opcode == "scalar.and" and other(user, name) == 1)
            if not (passes and user.results and depth < 4 and restores(user.results[0].name, mode, depth + 1)):
                return False
        return True

    return {op.id: restores(op.results[0].name, _ident(op.attrs.get("mode"))) for op in f.walk()
            if op.opcode == "core.get_sat_flag" and op.results and op.id is not None}


# --------------------------------------------------------------------------- interpreter


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


class _SimtReturn(Exception):
    pass


from .dma_ops import Binary32NaN, DmaOps  # noqa: E402
from .vf_ops import VfOps, _indexable, _where  # noqa: E402
from .vec_ops import VecOps


class Interp(VecOps, VfOps, DmaOps):
    def __init__(self, machine: Machine, lane: Lane) -> None:
        self.m = machine
        self.lane = lane
        self.env: dict[str, Any] = {}
        self.simt_tid: int | None = None
        self.simt_num: int | None = None
        self.simt_accesses: dict[tuple, set[tuple[int, int]]] | None = None
        self.simt_launch_accesses: dict[tuple, set[tuple[int, int]]] = {}
        self.in_vf = False  # executing a vf callee (its ops are traced into the enclosing cf.call)
        self.seq = 0
        self.l0_mx: dict[tuple[str, int], torch.Tensor] = {}  # L0A/L0B microscaling scale buffers, keyed like the tiles
        self.l0c_geometry: dict[tuple, tuple[int, int]] = {}  # producer view -> compact M pitch and computed N
        self.ub_dma_ranges: dict[tuple[int, int, int], list[tuple[int, int]]] = {}

    # -- helpers ------------------------------------------------------------------------------

    def note_ctrl_entry(self, op: Op, modes: tuple[str, ...], cast: tuple[Any, Any] | None = None) -> None:
        """RFC-0007 §3: warn when this lane uses CTRL bits it has not written in the launch. A5 keeps no
        launch value for them; the model still runs from its deterministic SAT_DEFAULTS."""
        written = getattr(self.lane, "sat_written", None)
        unwritten = tuple(m for m in modes if written is not None and m not in written)
        if not unwritten or self.m is None or (op.id, unwritten) in self.m.ctrl_noted:
            return
        self.m.ctrl_noted.add((op.id, unwritten))
        if op.opcode == "core.get_sat_flag" and self.m.ctrl_read_saves(op):
            return
        use = "core.get_sat_flag reads" if cast is None else f"vf.cast {cast[0]} -> {cast[1]} takes its saturation from"
        bits = " and ".join(f"{SAT_BITS[m]} ({m})" for m in unwritten)
        one = len(unwritten) == 1
        model = ", ".join(f"{m}={int(SAT_DEFAULTS[m])}" for m in unwritten)
        self.m.note_hardware(op, f"{use} CTRL bit{'' if one else 's'} {bits} before the kernel writes "
                                 f"{'it' if one else 'them'} in this launch. The entry state is unknown on A5 "
                                 f"(RFC-0007 §3): the kernel should write, and later restore, the "
                                 f"bit{'' if one else 's'} explicitly. The model assumes {model}.")

    def val(self, x: Any) -> Any:
        """Operand -> runtime value. Cells read their current value."""
        if isinstance(x, Value):
            v = self.env[x.name]
            return v.value if isinstance(v, Cell) else v
        if isinstance(x, Literal):
            return x.value
        if isinstance(x, FuncRef):
            return x.name
        return x

    def raw(self, x: Any) -> Any:
        if isinstance(x, Value):
            return self.env[x.name]
        return self.val(x)

    def attr(self, op: Op, name: str, default: Any = None) -> Any:
        v = op.attrs.get(name, default)
        if isinstance(v, Value):
            return self.val(v)
        if isinstance(v, list):
            return [self.val(x) for x in v]
        return v

    def dim(self, d: Any) -> int:
        if isinstance(d, int):
            return d
        if isinstance(d, DimValue):
            return int(self.val(Value(d.name, ScalarType(_I32))))
        if isinstance(d, Product):
            n = 1
            for f in d.factors:
                n *= self.dim(f)
            return n
        raise SimError(f"cannot evaluate dimension {d}")

    def dims(self, t: MemType) -> tuple[int, ...]:
        return tuple(self.dim(d) for d in t.dims)

    def runs_here(self, op: Op) -> bool:
        spec = REGISTRY.get(op.opcode)
        side = spec.side
        if op.opcode == "dma.copy":
            side = _copy_side_types(op)
        elif op.opcode.startswith("sync.mutex_"):
            flag = self.raw(op.operands[0])
            producer = "vec" if flag.kind == "vc" else "cube"
            consumer = "cube" if flag.kind == "vc" else "vec"
            side = producer if op.opcode in ("sync.mutex_lock", "sync.mutex_ready") else consumer
        if side in ("any", "both"):
            return True
        return side == self.lane.side

    # -- execution ----------------------------------------------------------------------------

    def run_function(self, f: Function, env: dict[str, Any]) -> None:
        self.env = env
        self.run_block(f.body.ops)

    def run_block(self, ops: tuple[Op, ...]) -> None:
        for op in ops:
            if self.m.stopping:
                raise SimError("the run is over; this lane leaves at its next op")
            if not self.runs_here(op):
                continue
            if self.lane.side == "cube" and self._vec_only_taint(op):
                continue
            handler = getattr(self, "op_" + op.opcode.replace(".", "_"), None)
            if handler is None:
                raise SimError(f"the reference interpreter does not implement {op.opcode} (#{op.id})")
            try:
                self.ub_dma_ranges = {}
                handler(op)
            except (_Break, _Continue):
                raise
            except SimError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise SimError(f"{op.opcode} #{op.id} at {op.loc}: {type(exc).__name__}: {exc}") from exc
            if self.m.tracer is not None:
                self._trace(op)

    # -- tracing (pipe-level simulator) --------------------------------------------------------

    def _trace(self, op: Op) -> None:
        tr = self.m.tracer
        lane = _lane_name(self.lane)
        if self.in_vf:
            stride = None
            if op.opcode in ("vf.load", "vf.store") and "blk_stride" in op.attrs:
                try:
                    stride = int(self.val(op.attrs["blk_stride"]))
                except Exception:  # noqa: BLE001
                    stride = 1
            folded = op.opcode.startswith("scalar.") and all(not isinstance(x, Value) or self.env.get("__const__", {}).get(x.name, False)
                                                             for x in op.operands)
            if folded:
                self.env.setdefault("__const__", {})[op.results[0].name] = True if op.results else None
            tr.record_vf(lane, op, stride, folded)
            return
        if self.simt_tid is not None:
            self._record_simt_access(op)
            if self.simt_tid == 0:
                tr.simt_count[lane] = tr.simt_count.get(lane, 0) + 1
            return
        if op.opcode in ("cf.for", "cf.if", "region.autosync", "region.side"):
            return  # the body ops are traced one by one as they execute
        from .pipesim import Task

        pipe = self._pipe_of(op)
        accesses = self._accesses(op)
        cost = 0
        vf = None
        if op.opcode == "cf.call":
            vf = tr.model.vf_cost(tr.end_vf(lane))
            cost = vf.cycles
        elif op.opcode == "simt.launch":
            n = tr.simt_count.pop(lane, 0)
            cost = tr.model.cost(replace(op, attrs={**op.attrs, "instruction_count": n}), self.val, self._operand_type(op))
        else:
            cost = tr.model.cost(op, self.val, self._operand_type(op))
        if op.opcode in ("sync.set_flag", "sync.wait_flag"):
            op = replace(op, attrs={**op.attrs, "event_id": int(self.val(op.attrs["event_id"]))})
        if op.opcode.startswith("sync.local_mutex_"):
            op = replace(op, attrs={**op.attrs, "id": int(self.val(op.attrs["id"]))})
        self.seq += 1
        tr.add(Task(lane, self.lane.group.index, self.lane.side, max(self.lane.sub, 0), self.seq, op, pipe, cost, accesses, vf))

    def _pipe_of(self, op: Op) -> str:
        if op.opcode in ("sync.set_flag", "sync.wait_flag"):
            return _ident(op.attrs["src" if op.opcode == "sync.set_flag" else "dst"])
        p = op.attrs.get("pipe")
        if p is not None and op.opcode.startswith("sync."):
            return _ident(p)
        if op.opcode in ("sync.set", "sync.set_all", "sync.wait", "sync.release") and op.operands:
            t = getattr(op.operands[0], "type", None)  # a user event without the attribute: its declared pipes
            if isinstance(t, EventType):
                return t.set_pipe if op.opcode in ("sync.set", "sync.set_all") else t.wait_pipe
        spec = REGISTRY.find(op.opcode)
        if spec is not None and spec.pipe:
            return spec.pipe
        if op.opcode in ("cf.call", "simt.launch"):
            return "V"
        return "S"

    def _operand_type(self, op: Op) -> Any:
        def get(i: int) -> Any:
            x = op.operands[i] if i < len(op.operands) else None
            return x.type if isinstance(x, Value) else None

        return get

    def _accesses(self, op: Op) -> list:
        if op.opcode == "core.clean_dcache":
            # Stores no value, so its bytes are recorded as "clean" and take part in no hazard; the cache-line
            # check reads them to see which line the writer cleaned (RFC-0006 §9).
            window = op.attrs.get("dst")
            try:
                ref = self.raw(window) if isinstance(window, Value) else None
            except KeyError:
                return []
            return [replace(item, kind="clean") for item in self._ranges(ref, 0, "write", window.name)] \
                if isinstance(ref, MemRef) else []
        if op.opcode.startswith("dma.") and "n_burst" in op.attrs:
            if int(self.val(op.attrs["n_burst"])) == 0 or any(
                    key in op.attrs and int(self.val(op.attrs[key])) == 0
                    for key in ("burst_len", "burst_len_byte")):
                return []
        if op.opcode in ("scalar.load", "scalar.store"):
            from .pipesim import view_accesses

            ref, _, index = self._scalar_address(op)
            sub = max(self.lane.sub, 0) if ref.space == "ub" else 0
            key = ((ref.space, ref.base) if ref.space in ("gm", "ws") else
                   (ref.space, self.lane.group.index, ref.base, ref.slot, sub))
            access = "read" if op.opcode == "scalar.load" else "write"
            return view_accesses(access, key, (1,), (1,), index,
                                 max(elem_bytes(ref.dtype), 1), op.operands[0].name)
        if op.opcode == "dma.l1_to_l0" and "m_copy" in op.attrs:
            return self._l1_to_l0_physical_accesses(op)
        out: list = []
        spec = REGISTRY.find(op.opcode)

        def add(x: Any, access: str, name: str = "") -> None:
            if not isinstance(x, Value):
                return
            try:
                ref = self.raw(x)
            except KeyError:
                return
            if not isinstance(ref, MemRef):
                return
            subs = [max(self.lane.sub, 0)] if ref.space == "ub" else [0]
            if ref.space == "ub" and self.lane.side == "cube":
                mode = _ident(op.attrs.get("dual_mode", "splitm")) if op.opcode == "dma.l0c_to_ub" else "splitm"
                subs = [int(self.val(op.attrs.get("sub_block_id", 0)))] if mode == "single" else [0, 1]
            for sub in subs:
                physical = self.ub_dma_ranges.get((id(op), id(ref), sub)) if ref.space == "ub" else None
                if ref.space in ("gm", "ws") and access == "write" and op.opcode == "dma.ub_to_gm.pad":
                    from .pipesim import view_accesses

                    # The operand supplies an address; the descriptor owns every
                    # written burst, even when its carrier view is smaller/larger.
                    flat, origin = self.m.storage(ref, self.lane)
                    count, size = int(self.val(op.attrs["n_burst"])), int(self.val(op.attrs["burst_len_byte"]))
                    pitch = size + int(self.val(op.attrs.get("dst_stride_byte", 0)))
                    ranges = view_accesses(access, (ref.space, ref.base), (count, size), (pitch, 1),
                                           origin * flat.element_size(), 1, name or x.name)
                elif physical is None:
                    ranges = self._ranges(ref, sub, access, name or x.name)
                else:
                    from .pipesim import view_accesses

                    key = (ref.space, self.lane.group.index, ref.base, ref.slot, sub)
                    ranges = [item for start, size in physical
                              for item in view_accesses(access, key, (size,), (1,), start, 1, name or x.name)]
                atomic = (access == "write" and ref.space == "gm" and op.opcode.startswith("dma.")
                          and op.operands[0] == x and _ident(op.attrs.get("atomic", "")) in ("add", "max", "min"))
                if atomic:
                    out.extend(replace(item, atomic=True) for item in ranges)
                    out.extend(replace(item, kind="read", atomic=True) for item in ranges)
                else:
                    out.extend(ranges)

        if spec is not None:
            for i, o in enumerate(spec.operands):
                if i < len(op.operands) and o.access != "none":
                    for kind in (("read", "write") if o.access == "readwrite" else (o.access,)):
                        add(op.operands[i], kind)
            for a in spec.attrs:
                if a.access != "none" and a.name in op.attrs:
                    v = op.attrs[a.name]
                    for kind in (("read", "write") if a.access == "readwrite" else (a.access,)):
                        add(v, kind)
        if op.opcode == "cf.call":
            declared = False
            for kind in ("read", "write"):
                for v in op.attrs.get(kind, []) or []:
                    add(v, kind)
                    declared = True
            if not declared:  # no access lists: every UB view handed to the vf counts as read and written
                for x in op.operands[1:]:
                    add(x, "read")
                    add(x, "write")
        if op.opcode == "simt.launch":
            from .pipesim import interval_accesses

            for (kind, key, name, atomic), ranges in self.simt_launch_accesses.items():
                out.extend(interval_accesses(kind, key, ranges, name, atomic=atomic))
        return out

    def _record_simt_access(self, op: Op) -> None:
        if self.simt_accesses is None or op.opcode not in ("simt.load", "simt.store", "simt.atomic"):
            return
        from .pipesim import _round_out, access_granularity

        ref: MemRef = self.raw(op.operands[0])
        flat, origin = self.m.storage(ref, self.lane)
        index = origin + int(self.val(op.operands[1]))
        lo, hi = index * flat.element_size(), (index + 1) * flat.element_size()
        sub = max(self.lane.sub, 0)
        key = (ref.space, ref.base) if ref.space in ("gm", "ws") else (ref.space, self.lane.group.index, ref.base, ref.slot, sub)
        atomic = op.opcode == "simt.atomic"
        kinds = ("read", "write") if atomic else ("read",) if op.opcode == "simt.load" else ("write",)
        for kind in kinds:
            granularity = access_granularity(kind, key, atomic=atomic)
            self.simt_accesses.setdefault((kind, key, ref.base, atomic), set()).add(_round_out(lo, hi, granularity))

    def _ranges(self, ref: MemRef, sub: int, kind: str, name: str) -> list:
        from .pipesim import view_accesses

        try:
            start = self.m.storage_offset(ref, self.lane, sub if ref.space == "ub" else None)
        except Exception:  # noqa: BLE001
            return []
        eb = max(elem_bytes(ref.dtype), 1)
        # on-chip memories are per core (and UB per sub-block); GM is shared by every core
        key = (ref.space, ref.base) if ref.space in ("gm", "ws") else (ref.space, self.lane.group.index, ref.base, ref.slot, sub)
        window = self.m.resolve(ref, self.lane, sub if ref.space == "ub" else None)
        origin = int(window.storage_offset() - start)  # from the allocation's start, also for a rebased window
        return view_accesses(kind, key, tuple(window.shape), tuple(window.stride()), origin, eb, name)

    def _vec_only_taint(self, op: Op) -> bool:
        """Cube lane: handle ops that produce or consume vector-only values; True = already dealt with."""
        if op.opcode in ("core.vec_idx", "core.vec_num", "core.sub_block_idx"):
            self.set_result(op, VEC_ONLY)
            return True
        tainted = any(self.val(x) is VEC_ONLY for x in op.operands if isinstance(x, Value)) or \
            any(self.val(v) is VEC_ONLY for v in op.attr_values())
        if not tainted:
            return False
        spec = REGISTRY.get(op.opcode)
        if spec.side == "cube" or (op.opcode == "dma.copy" and _copy_side_types(op) == "cube"):
            raise SimError(f"cube-side {op.opcode} #{op.id} ({op.loc}) consumes a vector-only value")
        if op.opcode == "scalar.set":
            self.env[op.operands[0].name].value = VEC_ONLY  # type: ignore[union-attr]
            return True
        for r in op.results:
            self.env[r.name] = Cell(VEC_ONLY) if isinstance(r.type, CellType) else VEC_ONLY
        return True  # loops and branches decided by vector-only values are vector-only as a whole

    def set_result(self, op: Op, value: Any, i: int = 0) -> None:
        self.env[op.results[i].name] = value

    # -- core / scalar ------------------------------------------------------------------------

    def op_core_cube_num(self, op: Op) -> None:
        self.set_result(op, self.lane.cube_num)

    def op_core_cube_idx(self, op: Op) -> None:
        self.set_result(op, self.lane.cube_idx)

    def op_core_vec_num(self, op: Op) -> None:
        self.set_result(op, self.lane.vec_num)

    def op_core_vec_idx(self, op: Op) -> None:
        self._vec_only(op)
        self.set_result(op, self.lane.vec_idx)

    def op_core_sub_block_idx(self, op: Op) -> None:
        self._vec_only(op)
        self.set_result(op, self.lane.sub)

    def _vec_only(self, op: Op) -> None:
        if self.lane.side != "vec":
            raise SimError(f"{op.opcode} (#{op.id}) is only defined on vector cores")

    # -- gmlist parameters (RFC-0001 §13): the members are GM tensors bound by the launcher ----------------

    def _member(self, op: Op) -> tuple[str, torch.Tensor]:
        ref = self.env[op.operands[0].name]  # type: ignore[union-attr]
        keys = self.m.lists[ref.name]
        i = int(self.val(op.operands[1]))
        if not 0 <= i < len(keys):
            raise SimError(f"list index {i} out of range for {ref.name} with {len(keys)} members (#{op.id})")
        return keys[i], self.m.gm[keys[i]]

    def op_list_count(self, op: Op) -> None:
        ref = self.env[op.operands[0].name]  # type: ignore[union-attr]
        self.set_result(op, len(self.m.lists[ref.name]))

    def op_list_item_dim(self, op: Op) -> None:
        _, t = self._member(op)
        self.set_result(op, int(t.shape[int(self.attr(op, "dim"))]))

    def op_list_item(self, op: Op) -> None:
        key, t = self._member(op)
        mt = op.results[0].type
        assert isinstance(mt, MemType)
        self.set_result(op, MemRef("gm", mt.dtype, key, 0, (0,) * t.dim(), tuple(t.shape)))

    def op_scalar_cell(self, op: Op) -> None:
        init = self.attr(op, "init")
        self.set_result(op, Cell(_coerce(init, op.results[0].type)))

    def op_scalar_set(self, op: Op) -> None:
        cell = self.env[op.operands[0].name]  # type: ignore[union-attr]
        cell.value = _coerce(self.val(op.operands[1]), op.operands[0].type)  # type: ignore[union-attr]

    def _binary(self, op: Op, fn: Any) -> None:
        a, b = self.val(op.operands[0]), self.val(op.operands[1])
        self.set_result(op, _coerce(fn(a, b), op.results[0].type))

    def op_scalar_add(self, op: Op) -> None:
        self._binary(op, lambda a, b: a + b)

    def op_scalar_sub(self, op: Op) -> None:
        self._binary(op, lambda a, b: a - b)

    def op_scalar_mul(self, op: Op) -> None:
        self._binary(op, lambda a, b: a * b)

    def op_scalar_div(self, op: Op) -> None:
        self._integer_division(op, 0)

    def op_scalar_mod(self, op: Op) -> None:
        self._integer_division(op, 1)

    def _integer_division(self, op: Op, part: int) -> None:
        a, b = (self.val(x) for x in op.operands)
        if isinstance(a, float) or isinstance(b, float):
            self.set_result(op, _coerce(a / b if part == 0 else _trunc_mod(a, b), op.results[0].type))
            return
        message = division_error(op, self.val)
        if message:
            raise self._err(message, op)
        self.set_result(op, _coerce(integer_divmod(a, b, rounding(op))[part], op.results[0].type))

    def op_scalar_and(self, op: Op) -> None:
        self._binary(op, lambda a, b: (a and b) if isinstance(a, bool) else a & b)

    def op_scalar_or(self, op: Op) -> None:
        self._binary(op, lambda a, b: (a or b) if isinstance(a, bool) else a | b)

    def op_scalar_xor(self, op: Op) -> None:
        self._binary(op, lambda a, b: a ^ b)

    def _integer_result(self, op: Op) -> DType | None:
        typ = op.results[0].type
        dt = typ.dtype if isinstance(typ, (ScalarType, CellType)) else None
        return dt if dt is not None and dt.is_integer and dt.name != "b1" else None

    def _shift(self, op: Op, left: bool) -> None:
        """C++17 shifts: the count lies in [0, width); a left shift needs a nonnegative operand whose
        value fits the unsigned counterpart, then converts to the declared width (docs/rfc/0015)."""
        a, b = self.val(op.operands[0]), self.val(op.operands[1])
        dt = self._integer_result(op)
        if dt is None or isinstance(a, bool) or isinstance(b, bool):
            self._binary(op, (lambda x, y: x << y) if left else (lambda x, y: x >> y))
            return
        a, b, bits = int(a), int(b), dt.bits
        if not 0 <= b < bits:
            raise self._err(f"{op.opcode} count {b} is outside [0, {bits})", op)
        if not left:
            self.set_result(op, a >> b)
            return
        if a < 0 or (a << b) >> bits:
            raise self._err(f"{op.opcode} of {a} by {b} leaves the unsigned {bits}-bit domain", op)
        value = a << b
        self.set_result(op, value - (1 << bits) if dt.kind == "int" and value >> (bits - 1) else value)

    def op_scalar_shl(self, op: Op) -> None:
        self._shift(op, True)

    def op_scalar_shr(self, op: Op) -> None:
        self._shift(op, False)

    def op_scalar_min(self, op: Op, maximum: bool = False) -> None:
        """RFC-0001 §6.16: f32 is IEEE 754-2019 minimum/maximum with A5's all-ones NaN; other dtypes keep their order."""
        if self.in_vf:  # the verifier refuses an A5 f32 one first; this is for a module that skipped the check
            from ...ir.scalar_math import vf_extremum_error

            message = vf_extremum_error(op, "vf", self.m.profile.family)
            if message:
                raise self._err(message, op)
        a, b = self.val(op.operands[0]), self.val(op.operands[1])
        if getattr(op.results[0].type, "dtype", None) != DTYPES["f32"]:
            self._binary(op, max if maximum else min)
            return
        if a != a or b != b:
            self.set_result(op, Binary32NaN(0x7FFFFFFF))
            return
        if a == b:  # equal values differ at most in the sign of zero
            b = a if (math.copysign(1.0, a) < 0) != maximum else b
        self.set_result(op, _coerce((a if a > b else b) if maximum else (a if a < b else b), op.results[0].type))

    def op_scalar_max(self, op: Op) -> None:
        self.op_scalar_min(op, maximum=True)

    def op_scalar_ceil_div(self, op: Op) -> None:
        message = division_error(op, self.val)
        if message:
            raise self._err(message, op)
        self._binary(op, lambda a, b: -(-int(a) // int(b)))

    def op_scalar_not(self, op: Op) -> None:
        a = self.val(op.operands[0])
        self.set_result(op, (not a) if isinstance(a, bool) else ~a)

    def op_scalar_neg(self, op: Op) -> None:
        a, dt = self.val(op.operands[0]), self._integer_result(op)
        if dt is not None and dt.kind == "int" and not isinstance(a, bool) and int(a) == -(1 << (dt.bits - 1)):
            raise self._err(f"scalar.neg of the signed {dt.bits}-bit minimum is undefined", op)
        self.set_result(op, -a)

    def op_scalar_abs(self, op: Op) -> None:
        self.set_result(op, abs(self.val(op.operands[0])))

    def op_scalar_sqrt(self, op: Op) -> None:
        self.set_result(op, _coerce(math.sqrt(self.val(op.operands[0])), op.results[0].type))

    def op_scalar_align(self, op: Op) -> None:
        message = division_error(op, self.val)
        if message:
            raise self._err(message, op)
        a, n = self.val(op.operands[0]), int(op.attrs["n"])
        self.set_result(op, -(-int(a) // n) * n)

    def op_scalar_cmp(self, op: Op) -> None:
        a, b = self.val(op.operands[0]), self.val(op.operands[1])
        pred = _ident(op.attrs["pred"])
        self.set_result(op, {"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b, "eq": a == b, "ne": a != b}[pred])

    def op_scalar_select(self, op: Op) -> None:
        c, a, b = (self.val(x) for x in op.operands)
        self.set_result(op, a if c else b)

    def op_scalar_cast(self, op: Op) -> None:
        typ = op.results[0].type
        value = _coerce(self.val(op.operands[0]), typ)
        if isinstance(typ, ScalarType) and typ.dtype.is_integer and typ.dtype.name != 'b1':
            modulus = 1 << typ.dtype.bits
            value %= modulus
            if typ.dtype.kind == 'int' and value >= modulus // 2:
                value -= modulus
        self.set_result(op, value)

    def op_scalar_const(self, op: Op) -> None:
        self.set_result(op, _coerce(op.attrs["value"], op.results[0].type))

    # -- memory -------------------------------------------------------------------------------

    def op_mem_alloc(self, op: Op) -> None:
        t = op.results[0].type
        if "addr" in op.attrs:  # lowered: the address is known, check it against the space's capacity
            elem = t.elem if isinstance(t, BufType) else t
            assert isinstance(elem, MemType)
            n = 1
            for d in self.dims(elem):
                n *= d
            nbytes = (n * elem.dtype.bits + 7) // 8 * (t.slots if isinstance(t, BufType) else 1)
            if elem.space == "ub":
                slot_bytes = ((n * elem.dtype.bits + 7) // 8 + 31) // 32 * 32
                nbytes = slot_bytes * (t.slots if isinstance(t, BufType) else 1)
                if nbytes and int(self.val(op.attrs["addr"])) % 32:
                    raise SimError(f"UB allocation address is not 32-byte aligned (#{op.id} at {op.loc})")
            end = int(self.val(op.attrs["addr"])) + nbytes
            cap = self.m.capacity(elem.space)
            if cap is not None and end > cap:
                raise SimError(f"{elem.space.upper()} overflow: %{op.results[0].name} (#{op.id}) ends at byte {end}, capacity {cap}")
        key = op.results[0].name  # the allocation's name: identical on both sides of a lowered module (op ids are not)
        if isinstance(t, BufType):
            dims = self.dims(t.elem)
            for s in range(t.slots):
                self._materialise(t.elem, key, s, dims)  # type: ignore[arg-type]
            self.set_result(op, BufRef(key, t.slots, t.elem, dims))  # type: ignore[arg-type]
            return
        assert isinstance(t, MemType)
        dims = self.dims(t)
        self._materialise(t, key, 0, dims)  # type: ignore[arg-type]
        self.set_result(op, MemRef(t.space, t.dtype, key, 0, (0,) * len(dims), dims))  # type: ignore[arg-type]

    def _materialise(self, t: MemType, alloc: Any, slot: int, dims: tuple[int, ...]) -> None:
        g = self.lane.group
        with g.lock:
            if t.space == "ub":
                subs = (self.lane.sub,) if self.lane.side == "vec" else (0, 1)
                for s in subs:
                    key = (s, alloc, slot)
                    if key not in g.ub:
                        elements = math.prod(dims)
                        padded = (elements * elem_bytes(t.dtype) + 31) // 32 * 32
                        backing = _poison((padded // elem_bytes(t.dtype),), t.dtype)
                        g.ub_storage[key] = backing
                        g.ub[key] = backing[:elements].reshape(dims)
            else:
                key2 = (alloc, slot)
                if key2 not in g.shared:
                    g.shared[key2] = _poison(dims, t.dtype)

    def op_mem_get_buf(self, op: Op) -> None:
        buf = self.raw(op.operands[0])
        idx = int(self.val(op.operands[1]))
        if isinstance(buf, RingRef):
            piece = idx % buf.slots + (self.lane.cube_idx * buf.slots if buf.per_core else 0)
            self.set_result(op, MemRef("gm", buf.elem.dtype, f"ws:{buf.name}:{piece}", 0, (0,) * len(buf.dims), buf.dims))
            return
        self.set_result(op, MemRef(buf.elem.space, buf.elem.dtype, buf.alloc, idx % buf.slots, (0,) * len(buf.dims), buf.dims))

    def op_mem_slice(self, op: Op) -> None:
        base: MemRef = self.raw(op.operands[0])
        offsets = tuple(int(x) for x in self.attr(op, "offsets"))
        extents = tuple(int(x) for x in self.attr(op, "extents"))
        if base.dtype.bits < 8:  # packed view: logical coordinates along the packed (last) axis are halved
            offsets = (*offsets[:-1], offsets[-1] // 2)
            extents = (*extents[:-1], extents[-1] // 2)
        for o, e, n in zip(offsets, extents, base.extents, strict=True):
            if o < 0 or e < 0 or o + e > n:
                raise SimError(f"slice [{o}:{o + e}) outside extent {n} (#{op.id})")
        self.set_result(op, base.sub(offsets, extents))

    def _rebase_origin(self, op: Op, base: MemRef, dt: DType) -> int:
        """Where a view, reshape or tile over ``base`` starts, in elements of ``dt``: the byte address of the
        window's first element, as the printers fold it (RFC-0010 §10, I016)."""
        if not any(base.offsets) and not base.view_offset:
            return 0
        if getattr(op.operands[0].type, "layout", None) == "nz" and any(base.offsets):
            raise SimError(f"{op.opcode} of an NZ-layout window at offsets {list(base.offsets)} is not modelled: "
                           f"the printers address its fractal rows (#{op.id} at {op.loc})")
        sub = 0 if base.space == "ub" and self.lane.sub < 0 else None
        flat, origin = self.m.storage(base, self.lane, sub)
        start, size = origin * flat.element_size(), elem_bytes(dt)
        if start % size:
            raise SimError(f"{op.opcode} starts at byte {start}, not a whole {dt.name} element (#{op.id} at {op.loc})")
        return start // size

    def op_mem_view(self, op: Op) -> None:
        """A strided GM re-description (RFC-0010): same storage, new (shape, strides, offset) from the window's start."""
        base: MemRef = self.raw(op.operands[0])
        if base.space not in ("gm", "ws"):
            raise SimError(f"mem.view applies to GM tensors, not {base.space} (#{op.id})")
        shape = tuple(int(x) for x in self.attr(op, "shape"))
        strides = tuple(int(x) for x in self.attr(op, "strides"))
        offset = int(self.attr(op, "offset", 0))
        if len(strides) != len(shape) or strides[-1] < 1 or any(s < 0 for s in strides) or offset < 0:
            raise SimError(f"mem.view needs one non-negative stride per dim, the innermost positive (#{op.id})")
        rt = op.results[0].type  # the view's dtype: a folded same-width reinterpret may differ from the base
        assert isinstance(rt, MemType)
        origin = self._rebase_origin(op, base, rt.dtype) + offset
        storage = self.m.gm[base.base]
        numel = storage.numel() * storage.element_size() // elem_bytes(rt.dtype)
        last = origin + sum((n - 1) * s for n, s in zip(shape, strides, strict=True)) + 1
        if last > numel:  # the view lies inside its root allocation (§3, §10)
            raise SimError(f"mem.view reaches element {last - 1} of a {numel}-element tensor (#{op.id})")
        self.set_result(op, MemRef(base.space, rt.dtype, base.base, base.slot, (0,) * len(shape), shape,
                                   None, None, gm_strides=strides, view_offset=origin))

    def op_mem_reinterpret(self, op: Op) -> None:
        base: MemRef = self.raw(op.operands[0])
        rt = op.results[0].type
        assert isinstance(rt, MemType)
        layout = _ident(op.attrs["layout"]) if "layout" in op.attrs else base.layout
        layout = layout if layout == "nz" else None  # only NZ-packed UB windows need remembering (nd is the model's layout)
        if "tile" in op.attrs:  # an L0 slot (contiguous bytes) seen as a [rows, cols] tile of the new dtype at its start
            rows, cols = (int(self.val(x)) for x in op.attrs["tile"])
            if rt.dtype.bits < 8:
                cols //= 2  # carrier bytes hold two packed elements
            total = base.extents[0] * base.extents[1] * elem_bytes(base.dtype) // (elem_bytes(rt.dtype) if rt.dtype.bits >= 8 else 1)
            if rows * cols > total:
                raise SimError(f"tile [{rows}, {cols}] does not fit the {total}-element slot (#{op.id})")
            self.set_result(op, MemRef(base.space, rt.dtype, base.base, base.slot, (0, 0), (rows, cols), (rows, cols), layout,
                                       view_offset=self._rebase_origin(op, base, rt.dtype)))
            return
        if base.gm_strides is not None and elem_bytes(base.dtype) != elem_bytes(rt.dtype):
            raise SimError(f"mem.reinterpret changes the element width of a mem.view window, whose strides "
                           f"count {base.dtype.name} elements (#{op.id} at {op.loc})")
        # a plain reinterpret keeps the window's coordinate system: its origin, and a view's strides (RFC-0010 §10)
        if rt.dtype.bits < 8 and rt.dtype.name != "i4" or elem_bytes(base.dtype) == elem_bytes(rt.dtype):
            # packed fp4 keeps the carrier's extents (the dtype says how to decode); same width keeps everything
            self.set_result(op, replace(base, dtype=rt.dtype, layout=layout))
            return
        # i4 switches to BYTE units over the carrier's bytes; other widths rescale the last axis
        num, den = (elem_bytes(base.dtype), 1) if rt.dtype.name == "i4" else (elem_bytes(base.dtype), elem_bytes(rt.dtype))
        if base.view_offset * num % den:
            raise SimError(f"mem.reinterpret of a window rebased at element {base.view_offset} is not a whole "
                           f"{rt.dtype.name} element (#{op.id} at {op.loc})")
        offsets = (*base.offsets[:-1], base.offsets[-1] * num // den)
        extents = (*base.extents[:-1], base.extents[-1] * num // den)
        shape = None if base.shape is None else (*base.shape[:-1], base.shape[-1] * num // den)
        self.set_result(op, MemRef(base.space, rt.dtype, base.base, base.slot, offsets, extents, shape, layout,
                                   view_offset=base.view_offset * num // den))

    # -- dma ----------------------------------------------------------------------------------

    # dma / cube handlers live in dma_ops.DmaOps; register ops beyond the core set in vf_ops.VfOps

    # -- control flow -------------------------------------------------------------------------

    def op_cf_for(self, op: Op) -> None:
        lo, hi, step = (int(self.val(x)) for x in op.operands)
        name = op.results[0].name
        for i in range(lo, hi, step):
            self.env[name] = i
            try:
                self.run_block(op.regions[0].ops)
            except _Continue:
                continue
            except _Break:
                break

    def op_cf_if(self, op: Op) -> None:
        cond = self.val(op.operands[0])
        if cond:
            self.run_block(op.regions[0].ops)
        elif len(op.regions) > 1:
            self.run_block(op.regions[1].ops)

    def op_cf_break(self, op: Op) -> None:
        raise _Break()

    def op_cf_continue(self, op: Op) -> None:
        raise _Continue()

    def op_cf_return(self, op: Op) -> None:
        return None

    def op_region_autosync(self, op: Op) -> None:
        self.run_block(op.regions[0].ops)

    def op_region_side(self, op: Op) -> None:
        if _ident(op.attrs["side"]) == self.lane.side:
            self.run_block(op.regions[0].ops)

    def op_cf_call(self, op: Op) -> None:
        callee = self.m.functions[self.val(op.operands[0])]
        args = [self.raw(x) for x in op.operands[1:]]
        sub = Interp(self.m, self.lane)
        sub.in_vf = True
        env = {}
        for p, a in zip(callee.params, args, strict=True):
            env[p.name] = a.value if isinstance(a, Cell) else a
        if self.m.tracer is not None:
            self.m.tracer.begin_vf(_lane_name(self.lane))
        sub.run_function(callee, env)
        sub.check_store_states(callee, sub.env)  # an unflushed store suffix outlives the vf (RFC-0001 §6.11)

    def op_simt_launch(self, op: Op) -> None:
        # Vec lanes take the launch one at a time, in vec_idx order: GM float atomics are not
        # associative, so free-running lanes made every run (and the goldens) end in different
        # last bits. Serialization pins one canonical order without changing any legal result.
        # It assumes every vec lane reaches the same sequence of launches (SPMD) - a lane that
        # skips one starves the lanes behind it, which the timeout below turns into an error.
        cond = self.m.simt_cond
        rank = self.m.simt_rank[self.lane.vec_idx]
        active = len(self.m.simt_rank)
        with cond:
            while self.m.simt_turn.value % active != rank:
                self.m.give_turn(self.lane)  # the lane whose launch is next may be one of this process
                if not cond.wait(timeout=60):
                    raise SimError(f"simt.launch #{op.id}: the ordered-launch gate timed out; "
                                   "simt.launch must be reached by every vec lane (no lane-dependent branches around it)")
                self.m.take_turn(self.lane, cond)
            try:
                self._simt_launch_body(op)
            finally:
                self.m.simt_turn.value += 1
                cond.notify_all()

    def _simt_launch_body(self, op: Op) -> None:
        callee = self.m.functions[self.val(op.operands[0])]
        args = [self.raw(x) for x in op.operands[1:]]
        threads = int(op.attrs["threads"])
        self.simt_launch_accesses = {}
        programs = []
        for tid in range(threads):
            sub = Interp(self.m, self.lane)
            sub.simt_tid, sub.simt_num = tid, threads
            sub.simt_accesses = self.simt_launch_accesses
            env = {}
            for p, a in zip(callee.params, args, strict=True):
                env[p.name] = a.value if isinstance(a, Cell) else a
            sub.env = env
            programs.append(sub._simt_program(callee.body.ops))
        generation = 0
        while programs:
            arrived = [next(program, None) for program in programs]
            if all(barrier is None for barrier in arrived):
                break
            waiting = [(tid, barrier.id) for tid, barrier in enumerate(arrived) if barrier is not None]
            if len(waiting) != threads or len({ident for _, ident in waiting}) != 1:
                ended = [tid for tid, barrier in enumerate(arrived) if barrier is None]
                raise SimError(f"simt.launch #{op.id}: divergent SIMT barrier generation {generation}; "
                               f"waiting thread/op {waiting[:8]}, exited threads {ended[:8]}")
            generation += 1

    def _simt_program(self, ops):
        """Generator frames are per-thread PCs; ordinary execution still owns ops and traces."""
        try:
            yield from self._simt_steps(ops)
        except _SimtReturn:
            return

    def _simt_steps(self, ops):
        for op in ops:
            if op.opcode == "simt.barrier":
                yield op
                if self.m.tracer is not None:
                    self._trace(op)
            elif op.opcode == "cf.if":
                arm = 0 if self.val(op.operands[0]) else 1
                if arm < len(op.regions):
                    yield from self._simt_steps(op.regions[arm].ops)
            elif op.opcode == "cf.for":
                lo, hi, step = (int(self.val(x)) for x in op.operands)
                for index in range(lo, hi, step):
                    self.env[op.results[0].name] = index
                    try:
                        yield from self._simt_steps(op.regions[0].ops)
                    except _Continue:
                        continue
                    except _Break:
                        break
            elif op.opcode == "cf.return":
                raise _SimtReturn()
            else:
                self.run_block((op,))

    # -- simt -----------------------------------------------------------------------------------

    def op_simt_thread_id(self, op: Op) -> None:
        self.set_result(op, self.simt_tid)

    def op_simt_thread_num(self, op: Op) -> None:
        self.set_result(op, self.simt_num)

    def op_simt_load(self, op: Op) -> None:
        from .dma_ops import f32_element

        ref: MemRef = self.raw(op.operands[0])
        flat, origin = self.m.storage(ref, self.lane)
        idx = origin + int(self.val(op.operands[1]))
        if flat.dtype == torch.float32 and op.results[0].type.dtype.name == "f32":  # NaN bits survive (RFC-0001 §6.14)
            self.set_result(op, f32_element(flat, idx))
        else:
            self.set_result(op, _coerce(flat[idx].item(), op.results[0].type))

    def op_simt_store(self, op: Op) -> None:
        from .dma_ops import f32_nan_store

        ref: MemRef = self.raw(op.operands[0])
        flat, origin = self.m.storage(ref, self.lane)
        idx = origin + int(self.val(op.operands[1]))
        v = self.val(op.operands[2])
        if not f32_nan_store(flat, idx, v):
            flat[idx] = torch.tensor(v, dtype=flat.dtype)

    def op_simt_barrier(self, op: Op) -> None:
        raise SimError(f"simt.barrier #{op.id} requires a SIMT launch context")

    def op_simt_threadfence(self, op: Op) -> None:
        pass  # the sequential thread model is the strongest legal visibility order

    def op_simt_threadfence_block(self, op: Op) -> None:
        pass

    # scalar math on SIMT threads: float32 round-trips so the goldens match the f32 device path
    def _simt_f32(self, op: Op, fn: Any) -> None:
        import numpy as np

        xs = [float(np.float32(self.val(o))) for o in op.operands]
        self.set_result(op, float(np.float32(fn(*xs))))

    def _simt_elementary(self, op: Op, fn: Any, edge: Any = None) -> None:
        """RFC-0001 §6.13: ``fn`` in float64 of the FP32 operand, rounded once to FP32. ``edge`` gives the IEEE
        value where Python's math raises instead; float64 overflow is +inf (I019)."""
        import numpy as np

        x = float(np.float32(self.val(op.operands[0])))
        result = edge(x) if edge is not None else None
        if result is None:
            try:
                result = fn(x)
            except OverflowError:  # exp and exp2 above the float64 range
                result = math.inf
        with np.errstate(over="ignore"):
            self.set_result(op, float(np.float32(result)))

    def op_simt_exp(self, op: Op) -> None:
        self._simt_elementary(op, math.exp)

    def op_simt_exp2(self, op: Op) -> None:
        self._simt_elementary(op, lambda x: math.pow(2.0, x))

    def op_simt_log(self, op: Op) -> None:
        self._simt_elementary(op, math.log, _log_edge(0.0))

    def op_simt_log2(self, op: Op) -> None:
        self._simt_elementary(op, math.log2, _log_edge(0.0))

    def op_simt_log1p(self, op: Op) -> None:
        self._simt_elementary(op, math.log1p, _log_edge(-1.0))

    def op_simt_sin(self, op: Op) -> None:
        self._simt_elementary(op, math.sin, lambda x: math.nan if math.isinf(x) else None)

    def op_simt_cos(self, op: Op) -> None:
        self._simt_elementary(op, math.cos, lambda x: math.nan if math.isinf(x) else None)

    def op_simt_tanh(self, op: Op) -> None:
        self._simt_elementary(op, math.tanh)

    def op_simt_rsqrt(self, op: Op) -> None:
        self._simt_elementary(op, lambda x: 1.0 / math.sqrt(x), _rsqrt_edge)

    def op_simt_rint(self, op: Op) -> None:
        import numpy as np

        self._simt_f32(op, lambda x: float(np.rint(np.float32(x))))

    def _simt_integral_f32(self, op: Op, fn: Any) -> None:
        def rounded(x: float) -> float:
            # Python's integral rounding returns int: it loses -0 and rejects
            # NaN/Inf. The SIMT result is floating point (RFC-0001 §6.3).
            if not math.isfinite(x):
                return x
            result = fn(x)
            return math.copysign(0.0, x) if result == 0 else result

        self._simt_f32(op, rounded)

    def op_simt_round(self, op: Op) -> None:
        self._simt_integral_f32(op, lambda x: math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5))

    def op_simt_floor(self, op: Op) -> None:
        self._simt_integral_f32(op, math.floor)

    def op_simt_ceil(self, op: Op) -> None:
        self._simt_integral_f32(op, math.ceil)

    def op_simt_trunc(self, op: Op) -> None:
        self._simt_integral_f32(op, math.trunc)

    def op_simt_fmod(self, op: Op) -> None:
        import numpy as np

        # RFC-0001 §6.15: the exact remainder never rounds (math.fmod is exact, and a binary32 remainder is
        # binary32). A NaN operand, an infinite dividend or a zero divisor gives quiet NaN 0x7FC00000.
        x, y = (float(np.float32(self.val(o))) for o in op.operands)
        invalid = math.isnan(x) or math.isnan(y) or math.isinf(x) or y == 0
        self.set_result(op, math.nan if invalid else x if math.isinf(y) else math.fmod(x, y))

    def op_simt_fma(self, op: Op) -> None:
        from .vf_ops import _fma32

        a, b, c = (torch.tensor([float(self.val(o))], dtype=torch.float32) for o in op.operands)
        self.set_result(op, float(_fma32(a, b, c)[0]))  # one rounding of the exact value (I021)

    def op_simt_isnan(self, op: Op) -> None:
        self.set_result(op, int(math.isnan(float(self.val(op.operands[0])))))

    def op_simt_isinf(self, op: Op) -> None:
        self.set_result(op, int(math.isinf(float(self.val(op.operands[0])))))

    def op_simt_isfinite(self, op: Op) -> None:
        self.set_result(op, int(math.isfinite(float(self.val(op.operands[0])))))

    def op_simt_popc(self, op: Op) -> None:
        self.set_result(op, bin(int(self.val(op.operands[0])) & 0xFFFFFFFF).count("1"))

    def op_simt_ffs(self, op: Op) -> None:
        x = int(self.val(op.operands[0])) & 0xFFFFFFFF
        self.set_result(op, (x & -x).bit_length() if x else 0)

    def op_simt_mul_hi(self, op: Op) -> None:
        t = op.results[0].type
        signed = getattr(getattr(t, "dtype", None), "kind", "int") == "int"
        a, b = int(self.val(op.operands[0])), int(self.val(op.operands[1]))
        if not signed:
            a &= 0xFFFFFFFF
            b &= 0xFFFFFFFF
        hi = (a * b) >> 32
        if signed:
            hi = ((hi + 0x80000000) & 0xFFFFFFFF) - 0x80000000
        else:
            hi &= 0xFFFFFFFF
        self.set_result(op, hi)

    # -- vf: registers ------------------------------------------------------------------------

    def op_vf_reg(self, op: Op) -> None:
        t = op.results[0].type
        assert isinstance(t, RegType)
        self.set_result(op, RegRef(torch.zeros(REG_BYTES * t.n, dtype=torch.uint8), t.dtype, reg_lanes(t.dtype) * t.n))

    def op_vf_mask(self, op: Op) -> None:
        t = op.results[0].type
        assert isinstance(t, MaskType)
        lanes = t.lanes
        init = _ident(op.attrs.get("init", "all"))
        bits = torch.zeros(lanes, dtype=torch.bool)
        if init == "all":
            bits[:] = True
        elif init == "none":
            pass
        elif init.startswith("vl"):
            bits[: int(init[2:])] = True
        elif init == "h":
            bits[: lanes // 2] = True
        elif init == "q":
            bits[: lanes // 4] = True
        elif init == "m3":
            bits[::3] = True
        elif init == "m4":
            bits[::4] = True
        else:
            raise SimError(f"unknown mask pattern {init!r} (#{op.id})")
        self.set_result(op, MaskRef(bits))

    def op_vf_unalign(self, op: Op) -> None:
        assert isinstance(op.results[0].type, UnalignRegType)
        self.set_result(op, RegRef(torch.zeros(REG_BYTES, dtype=torch.uint8), _U8, REG_BYTES))

    def op_vf_reinterpret(self, op: Op) -> None:
        src: RegRef = self.raw(op.operands[0])
        t = op.results[0].type
        assert isinstance(t, RegType)
        self.set_result(op, RegRef(src.bytes, t.dtype, reg_lanes(t.dtype) * t.n))

    def _mask(self, op: Op, lanes: int) -> torch.Tensor | None:
        mref = op.attrs.get("mask")
        if mref is None:
            return None
        mask: MaskRef = self.env[mref.name]
        bits = mask.bits
        if bits.numel() < lanes:
            raise SimError(f"mask has {bits.numel()} lanes, op needs {lanes} (#{op.id})")
        return bits[:lanes]

    def _block_indices(self, op: Op, n: int, esize: int) -> torch.Tensor | None:
        """Lane -> UB element for the 32-byte block copies: block b sits ``blk_stride`` * b blocks past the base, so
        stride 0 puts every block at block 0 (I035). None is the contiguous stride 1."""
        stride = int(self.attr(op, "blk_stride", 1))
        if stride == 1:
            return None
        c0 = 32 // esize
        lane = torch.arange(n)
        return (lane // c0) * (stride * c0) + lane % c0

    def _load_lanes(self, op: Op, n: int, esize: int, strided: bool) -> torch.Tensor:
        """Lanes a block load reads. A strided vsldb reads block b whole when any of its 32 physical predicate bits
        is set (I035); the contiguous load, and a two-register load the 256-bit predicate cannot cover, read lanes."""
        mask = self._mask(op, n)
        if mask is None:
            return torch.ones(n, dtype=torch.bool)
        if not strided or n * esize != REG_BYTES:
            return mask[:n]
        blocks = self.env[op.attrs["mask"].name].physical().reshape(-1, BLOCK_BYTES).any(dim=1)
        return blocks.repeat_interleave(BLOCK_BYTES // esize)

    def op_vf_load(self, op: Op) -> None:
        dst: RegRef = self.raw(op.operands[0])
        src: MemRef = self.raw(op.operands[1])
        offset = int(self.attr(op, "offset", 0))
        flat, origin = self.m.storage(src, self.lane)
        n = dst.lanes
        reg = dst.tensor()
        idx = self._block_indices(op, n, reg.element_size())
        active = self._load_lanes(op, n, reg.element_size(), idx is not None)
        where = origin + offset + (torch.arange(n) if idx is None else idx)
        if bool(((where[active] < 0) | (where[active] >= flat.numel())).any()):
            raise SimError(f"register load has active lanes outside the UB allocation (#{op.id} at {op.loc})")
        self._check_aligned(op, "load", origin + offset, flat.element_size())
        reg.zero_()
        _indexable(reg)[active] = _indexable(flat)[where[active]].view(flat.dtype).to(reg.dtype).view(_indexable(reg).dtype)
        dst.valid = dst.lanes

    def op_vf_store(self, op: Op) -> None:
        dst: MemRef = self.raw(op.operands[0])
        src: RegRef = self.raw(op.operands[1])
        offset = int(self.attr(op, "offset", 0))
        flat, origin = self.m.storage(dst, self.lane)
        n = src.valid
        values = src.tensor()[:n]
        mask = self._mask(op, n)
        idx = self._block_indices(op, n, values.element_size())
        active = torch.ones(n, dtype=torch.bool) if mask is None else mask[:n]
        where = origin + offset + (torch.arange(n) if idx is None else idx)
        if bool(((where[active] < 0) | (where[active] >= flat.numel())).any()):
            raise SimError(f"register store has active lanes past the end of the buffer (#{op.id} at {op.loc})")
        self._check_aligned(op, "store", origin + offset, flat.element_size())
        lanes = torch.nonzero(active).reshape(-1)
        targets = where[lanes]
        if targets.unique().numel() != targets.numel():  # stride 0: later lanes overwrite earlier ones (I035)
            last = {int(t): i for i, t in enumerate(targets.tolist())}
            keep = torch.tensor(sorted(last.values()), dtype=torch.long)
            lanes, targets = lanes[keep], targets[keep]
        _indexable(flat)[targets] = _indexable(values.to(flat.dtype))[lanes]

    def _check_aligned(self, op: Op, what: str, elem: int, esize: int) -> None:
        """Ordinary register/UB instruction starts use physical 32-byte alignment."""
        byte = elem * esize
        if byte % BLOCK_BYTES:
            raise SimError(f"UB {what} at element {elem} is byte {byte}: not 32-byte aligned "
                           f"(#{op.id} at {op.loc})")

    def op_vf_load_cont(self, op: Op) -> None:
        """LoadAlign with a distribution (the old ``micro_ub2regcont`` modes)."""
        dst: RegRef = self.raw(op.operands[0])
        src: MemRef = self.raw(op.operands[1])
        offset = int(self.attr(op, "offset", 0))
        mode = str(self.attr(op, "mode", "norm"))
        flat, origin = self.m.storage(src, self.lane)
        lanes = dst.lanes
        reg = dst.tensor()
        reg.zero_()
        start = origin + offset
        if mode.split("_")[0] != "brc":  # scalar `.single()` is the one-element exception
            self._check_aligned(op, "load", start, flat.element_size())

        def take(n: int) -> torch.Tensor:
            if start < 0 or start + n > flat.numel():
                raise SimError(
                    f"distributed register load is outside the UB allocation (#{op.id} at {op.loc}): "
                    f"mode {mode} over {lanes} lanes reads elements [{start}, {start + n}) of {flat.numel()} "
                    f"in %{src.base} slot {src.slot}{NARROW_VIEW_HINT}")
            return flat[start: start + n].to(reg.dtype)

        kind = mode.split("_")[0]
        if kind == "norm":
            v = take(lanes)
            reg[: v.numel()] = v
        elif kind == "brc":
            reg.fill_(take(1)[0])
        elif kind == "ds":
            v = take(2 * lanes)[0::2]
            reg[: v.numel()] = v
        elif kind == "us":
            v = take(lanes // 2)
            reg[0: 2 * v.numel(): 2] = v
            reg[1: 2 * v.numel(): 2] = v
        elif kind == "unpack":
            v = take(lanes // 2)
            reg[0: 2 * v.numel(): 2] = v
        elif kind == "unpack4":
            v = take(lanes // 4)
            reg[0: 4 * v.numel(): 4] = v
        elif kind == "e2b":
            c0 = 32 // elem_bytes(dst.dtype)
            v = take(lanes // c0)
            for i in range(v.numel()):
                reg[i * c0: (i + 1) * c0] = v[i]
        else:
            raise SimError(f"vf.load_cont mode {mode!r} is not implemented")
        dst.valid = dst.lanes

    def op_vf_store_cont(self, op: Op) -> None:
        """StoreAlign with a distribution (the old ``micro_reg2ubcont`` modes)."""
        dst: MemRef = self.raw(op.operands[0])
        src: RegRef = self.raw(op.operands[1])
        offset = int(self.attr(op, "offset", 0))
        mode = str(self.attr(op, "mode", "norm"))
        flat, origin = self.m.storage(dst, self.lane)
        lanes = src.lanes
        reg = src.tensor()
        kind = mode.split("_")[0]
        if kind == "norm":
            src_idx = torch.arange(lanes)
        elif kind == "pack":
            src_idx = torch.arange(0, lanes, 2)
        elif kind == "pack4":
            src_idx = torch.arange(0, lanes, 4)
        elif kind == "first":
            src_idx = torch.arange(1)
        else:
            raise SimError(f"vf.store_cont mode {mode!r} is not implemented")
        mask = self._mask(op, lanes)
        element_packed = mode == "pack_b32" and src.dtype.bits == 16
        if mask is not None:
            if kind != "norm" and not element_packed:
                c0 = 32 // elem_bytes(src.dtype)
                mask = mask.view(-1, c0).any(dim=1).repeat_interleave(c0)
            src_idx = src_idx[mask[src_idx]]
        # Measured A5 PK_B32 on 16-bit registers gates each packed element and
        # leaves holes for inactive predicates; it neither rounds to blocks
        # nor compacts the remaining active elements (M10-064).
        dst_idx = src_idx // 2 if element_packed else (torch.arange(src_idx.numel()) if kind != "norm" else src_idx)
        where = origin + offset + dst_idx
        outside = (where < 0) | (where >= flat.numel())
        if bool(outside.any()):
            raise SimError(
                f"distributed register store has active lanes outside the buffer (#{op.id} at {op.loc}): "
                f"mode {mode}{' with an explicit mask' if mask is not None else ' with no mask'} drives "
                f"{int(src_idx.numel())} of {lanes} lanes to elements "
                f"[{int(where.min())}, {int(where.max()) + 1}) of {flat.numel()} in %{dst.base} slot {dst.slot}; "
                f"{int(outside.sum())} of them land outside{NARROW_VIEW_HINT}")
        if kind != "first":  # scalar `.single_value()` may use element alignment
            self._check_aligned(op, "store", origin + offset, flat.element_size())
        values = _indexable(reg)[src_idx].view(reg.dtype).to(flat.dtype)
        _indexable(flat)[where] = _indexable(values)

    def _interleave_span(self, op: Op, what: str, mem: MemRef, first: RegRef, second: RegRef) -> tuple[torch.Tensor, int, int]:
        """(byte storage, aligned start byte, element bytes) of a two-register interleaved transfer (RFC-0001)."""
        esize = elem_bytes(first.dtype)
        prefix = "dintlv" if what == "load" else "intlv"
        where = f"(#{op.id} at {op.loc})"
        if first.dtype != second.dtype or esize not in (1, 2, 4) or what == "load" and first is second:
            raise SimError(f"interleaved {what} needs b8/b16/b32 registers of one dtype"
                           f"{', two distinct destinations' if what == 'load' else ''} {where}")
        if str(self.attr(op, "mode", "")) != f"{prefix}_b{8 * esize}":
            raise SimError(f"interleaved {what} mode {self.attr(op, 'mode', '')!r} does not match {first.dtype} registers {where}")
        flat, origin = self.m.storage(mem, self.lane)
        start = origin * flat.element_size() + int(self.attr(op, "offset", 0)) * esize
        if start % BLOCK_BYTES:
            raise SimError(f"interleaved {what} at byte {start} is not 32-byte aligned {where}")
        return flat.view(torch.uint8), start, esize

    def op_vf_load_interleave(self, op: Op) -> None:
        """De-interleaving load: 2L register elements; element 2i enters lane i of dst0, 2i+1 lane i of dst1."""
        d0, d1, src = (self.raw(x) for x in op.operands[:3])
        raw, start, esize = self._interleave_span(op, "load", src, d0, d1)
        end = start + 2 * d0.lanes * esize
        if start < 0 or end > raw.numel():
            raise SimError(f"interleaved load reads bytes [{start}, {end}) of the {raw.numel()}-byte allocation "
                           f"%{src.base} (#{op.id} at {op.loc})")
        elements = raw[start:end].view(-1, esize).clone()
        d0.bytes.view(-1, esize).copy_(elements[0::2])
        d1.bytes.view(-1, esize).copy_(elements[1::2])
        d0.valid, d1.valid = d0.lanes, d1.lanes

    def op_vf_store_interleave(self, op: Op) -> None:
        """Interleaving store: element 2i from src0 and 2i+1 from src1 for every lane. A5 writes all 2L elements
        whatever the predicate operand (RFC-0001), so the whole footprint must lie inside the allocation."""
        dst, s0, s1 = (self.raw(x) for x in op.operands[:3])
        raw, start, esize = self._interleave_span(op, "store", dst, s0, s1)
        if op.attrs.get("mask") is not None and not isinstance(self.raw(op.attrs["mask"]), MaskRef):
            raise SimError(f"interleaved store predicate is not a mask register (#{op.id} at {op.loc})")
        end = start + 2 * s0.lanes * esize
        if start < 0 or end > raw.numel():
            raise SimError(f"interleaved store writes bytes [{start}, {end}) of the {raw.numel()}-byte allocation "
                           f"%{dst.base} (#{op.id} at {op.loc})")
        values = torch.stack((s0.bytes.view(-1, esize), s1.bytes.view(-1, esize)), dim=1).reshape(-1, esize)
        raw[start:end].view(-1, esize).copy_(values)

    def op_vf_copy(self, op: Op) -> None:
        dst: RegRef = self.raw(op.operands[0])
        src: RegRef = self.raw(op.operands[1])
        self._write_reg(op, dst, src.tensor().clone())
        dst.valid = src.valid

    def _write_reg(self, op: Op, dst: RegRef, values: torch.Tensor) -> None:
        reg = dst.tensor()
        out = values.to(reg.dtype)
        mask = self._mask(op, reg.numel())
        if mask is not None:
            out = _where(mask, out, torch.zeros_like(out))  # inactive lanes are zeroed (old _apply_exec_mask_zero)
        reg.copy_(out)
        dst.valid = dst.lanes

    def _store_masked(self, op: Op, dst: RegRef, values: torch.Tensor) -> None:
        """Write computed numbers (fp32 or int64) into a register of any dtype; inactive lanes are zeroed."""
        mask = self._mask(op, dst.lanes)
        if mask is not None:
            values = torch.where(mask, values, torch.zeros_like(values))
        _store_values(dst, values)
        dst.valid = dst.lanes

    def _unary(self, op: Op, fn: Any) -> None:
        dst: RegRef = self.raw(op.operands[0])
        src: RegRef = self.raw(op.operands[1])
        self._store_masked(op, dst, fn(_reg_values(src)))  # ints compute in int64 (exact), floats in fp32

    def op_vf_abs(self, op: Op) -> None:
        src: RegRef = self.raw(op.operands[1])
        if src.dtype.kind == "complex":  # |z| is real: the result fills the low src.lanes of a real register
            dst: RegRef = self.raw(op.operands[0])
            v = torch.abs(_reg_values(src)[: src.lanes])
            out = torch.zeros(dst.lanes, dtype=v.dtype)
            out[: v.numel()] = v
            self._store_masked(op, dst, out)
            dst.valid = src.lanes
            return
        self._unary(op, torch.abs)

    def op_vf_sqrt(self, op: Op) -> None:
        self._unary(op, torch.sqrt)

    def op_vf_exp(self, op: Op) -> None:
        self._unary(op, torch.exp)

    def op_vf_ln(self, op: Op) -> None:
        self._unary(op, torch.log)

    def op_vf_neg(self, op: Op) -> None:
        self._unary(op, torch.neg)

    def op_vf_relu(self, op: Op) -> None:
        self._unary(op, torch.relu)

    def _scalar_op(self, op: Op, fn: Any) -> None:
        dst: RegRef = self.raw(op.operands[0])
        src: RegRef = self.raw(op.operands[1])
        v = self.val(op.operands[2])
        if src.dtype.kind == "complex":  # the printed scalar is complex32((half)re, (half)im) / complex64(re f, im f):
            part = torch.float16 if src.dtype.name == "c32" else torch.float32  # parts round before the compute
            v = complex(torch.tensor(v.real, dtype=part).item(), torch.tensor(v.imag, dtype=part).item())
        elif src.dtype.name in ("f16", "bf16"):
            v = torch.tensor(v, dtype=torch_dtype(src.dtype)).item()  # immediates round to the register dtype
        elif not src.dtype.is_float:
            v = int(v)
        self._store_masked(op, dst, fn(_reg_values(src), v))

    def op_vf_adds(self, op: Op) -> None:
        self._scalar_op(op, lambda x, s: x + s)

    def op_vf_muls(self, op: Op) -> None:
        self._scalar_op(op, lambda x, s: x * s)

    def op_vf_maxs(self, op: Op) -> None:
        if self._unsigned_minmax(op, maximum=True, scalar=True):
            return
        self._scalar_op(op, lambda x, s: torch.clamp(x, min=s))

    def op_vf_mins(self, op: Op) -> None:
        if self._unsigned_minmax(op, maximum=False, scalar=True):
            return
        self._scalar_op(op, lambda x, s: torch.clamp(x, max=s))

    def _binary_reg(self, op: Op, fn: Any) -> None:
        dst: RegRef = self.raw(op.operands[0])
        a: RegRef = self.raw(op.operands[1])
        b: RegRef = self.raw(op.operands[2])
        self._store_masked(op, dst, fn(_reg_values(a), _reg_values(b)))

    def op_vf_add(self, op: Op) -> None:
        self._binary_reg(op, lambda a, b: a + b)

    def op_vf_sub(self, op: Op) -> None:
        self._binary_reg(op, lambda a, b: a - b)

    def op_vf_mul(self, op: Op) -> None:
        self._binary_reg(op, lambda a, b: a * b)

    def op_vf_div(self, op: Op) -> None:
        self._binary_reg(op, lambda a, b: a / b if a.is_floating_point() or a.is_complex() else torch.div(a, b, rounding_mode="trunc"))

    def op_vf_max(self, op: Op) -> None:
        if self._unsigned_minmax(op, maximum=True):
            return
        self._binary_reg(op, torch.maximum)

    def op_vf_min(self, op: Op) -> None:
        if self._unsigned_minmax(op, maximum=False):
            return
        self._binary_reg(op, torch.minimum)

    def op_vf_dup(self, op: Op) -> None:
        dst: RegRef = self.raw(op.operands[0])
        src = self.raw(op.operands[1])
        if isinstance(src, RegRef):  # dup(dst, reg): lane 0 of the source, as the old micro simulator did
            _store_values(dst, _reg_values(src)[0].expand(dst.lanes).clone())
            mask = self._mask(op, dst.lanes)
            if mask is not None:
                _store_values(dst, torch.where(mask, _reg_values(dst), torch.zeros_like(_reg_values(dst))))
            dst.valid = dst.lanes
            return
        v = self.val(op.operands[1])
        self._write_reg(op, dst, torch.full((dst.lanes,), v, dtype=torch_dtype(dst.dtype)))

    def op_vf_cast(self, op: Op) -> None:
        dst: RegRef = self.raw(op.operands[0])
        src: RegRef = self.raw(op.operands[1])
        if (src.dtype.name, dst.dtype.name) in TRUNCATING_CASTS and op.attrs.get("saturate", False):
            raise SimError(f"vf.cast i64 -> i32 has no saturation selector; clamp before narrowing (#{op.id} at {op.loc})")
        round_mode = _ident(op.attrs.get("round", "rint"))
        layout = _ident(op.attrs.get("layout", "zero"))
        slot = {"zero": 0, "one": 1, "two": 2, "three": 3}.get(layout, 0)
        if dst.dtype.name in ("fp4_e1m2", "fp4_e2m1"):
            if dst.dtype.name != "fp4_e1m2":
                raise SimError("fp4 e2m1 casts are not implemented in the reference interpreter")
            src_lanes = src.lanes
            values = src.tensor()[:src_lanes].to(torch.float32)
            mask = self._mask(op, src_lanes)
            if mask is not None:
                values = _where(mask, values, torch.zeros_like(values))
            # the host codec, which a board run matched byte for byte in all five modes; `none` rounds to nearest even
            carrier = fp32_to_fp4_e1m2(values, nan_to_zero=True, round_mode="rint" if round_mode == "none" else round_mode).reshape(-1)
            work = torch.zeros(REG_BYTES, dtype=torch.uint8)
            work[slot: slot + carrier.numel() * 4: 4] = carrier
            dst.bytes.copy_(work)
            dst.valid = dst.lanes
            return
        if "i4" in (dst.dtype.name, src.dtype.name):
            self._cast_int4(op, dst, src, round_mode, slot)
            return
        dst_lanes, src_lanes = dst.lanes, src.lanes
        d_size, s_size = elem_bytes(dst.dtype), elem_bytes(src.dtype)
        mref = op.attrs.get("mask")
        half_pairs = {("f32", "f16"), ("f16", "f32"), ("f32", "bf16"), ("bf16", "f32")}
        if ((src.dtype.name, dst.dtype.name) in half_pairs and min(dst_lanes, src_lanes) == 64
                and max(dst_lanes, src_lanes) == 128 and merge_mode(op) == "zeroing"
                and (mref is None or mref.type == MaskType(32)
                     or s_size == 2 and mref.type == MaskType(16))):
            # vcvt samples predicate bits at the selected SOURCE element positions.
            # In particular, b32 has no active odd f16/bf16 source positions.
            indices = torch.arange(64)
            source = indices if s_size > d_size else indices * 2 + slot
            destination = indices * 2 + slot if s_size > d_size else indices
            active = (torch.ones(64, dtype=torch.bool) if mref is None
                      else self.env[mref.name].physical()[source * s_size])
            work = torch.zeros(dst_lanes, dtype=torch.float32)
            work[destination[active]] = _convert(_reg_values(src)[source[active]], dst.dtype, round_mode,
                                                saturate=False).float()
            _store_values(dst, work)
            dst.valid = dst.lanes
            return
        mask = self._mask(op, dst_lanes)
        active = torch.arange(dst_lanes) if mask is None else torch.nonzero(mask).reshape(-1)
        # compute in a dtype torch can index (float8 and the narrow unsigned dtypes cannot be index_put)
        compute = torch.float32 if dst.dtype.is_float else torch.int64
        work = torch.zeros(dst_lanes, dtype=compute)
        if merge_mode(op) == "merging":
            work.copy_(_reg_values(src if False else dst).to(compute))
        s = _reg_values(src)
        if src.dtype.kind == "uint" and src.dtype.bits <= 32:
            # Arithmetic uses signed carriers, but a cast must preserve unsigned source values.
            s = src.tensor().to(torch.int64)
        flags = getattr(self.lane, "sat_flags", SAT_DEFAULTS)
        if cast_uses_ctrl(src.dtype, dst.dtype):
            self.note_ctrl_entry(op, ("global", "cast") if flags["global"] else ("global",), (src.dtype, dst.dtype))
        saturate = not flags["cast"] if flags["global"] else bool(op.attrs.get("saturate", False))
        if (src.dtype.name, dst.dtype.name) in TRUNCATING_CASTS:
            saturate = False  # argument-free b64 form always discards high bits, independent of CTRL
        if not dst.dtype.is_integer or (not src.dtype.is_float and dst.dtype.bits >= src.dtype.bits):
            saturate = False
        if dst.dtype.name == "hif8" and src.dtype.name == "f16":
            s = s.half()  # exact; the fp16 encoder rounds from half precision (old fp16_to_hif8 path)
        if 8 in (s_size, d_size):
            # The 64-bit vcvt is the two-register (2xvl) form -- `b64_widen` takes no arguments at
            # all, `b64_from_f32` / `f32_from_b64` take no mask and no PART. With no layout
            # selector there is no half to choose: destination lane k carries source lane k, and
            # the board writes them consecutively rather than on alternate lanes.
            reach = min(dst_lanes, src_lanes)
            sel = active[active < reach]
            work[sel] = _convert(s[sel], dst.dtype, round_mode, saturate=saturate).to(compute)
        elif src_lanes == dst_lanes:
            work[active] = _convert(s[active], dst.dtype, round_mode, saturate=saturate).to(compute)
        elif s_size < d_size:
            ratio = d_size // s_size
            src_idx = active * ratio + slot
            valid = src_idx < src_lanes
            work[active[valid]] = _convert(s[src_idx[valid]], dst.dtype, round_mode, saturate=saturate).to(compute)
        else:
            ratio = s_size // d_size
            selected = (active % ratio) == slot
            dst_idx = active[selected]
            src_idx = dst_idx // ratio
            valid = src_idx < src_lanes
            converted = _convert(s[src_idx[valid]], dst.dtype, round_mode, saturate=saturate).to(compute)
            if src.dtype.name == "f32" and dst.dtype.name in ("e4m3", "e5m2") and round_mode in ("none", "rint"):
                converted = torch.where(torch.isnan(converted), torch.full_like(converted, float("nan")), converted)
            work[dst_idx[valid]] = converted
        _store_values(dst, work)
        dst.valid = dst.lanes

    def _cast_int4(self, op: Op, dst: RegRef, src: RegRef, round_mode: str, slot: int) -> None:
        """`i4` is the compiler's `vector_s4x2`: TWO signed nibbles per byte, the lower index in the
        low nibble, and one byte in every four selected by PART_P0..P3 -- the `_t` shape family, the
        same carrier layout `fp4` uses above. A 16-bit partner therefore trades 128 lanes against 64
        carrier bytes, and the nibble is signed, so widening sign-extends from 4 bits."""
        if dst.dtype.name == "i4":
            values = _reg_values(src)[:src.lanes]
            mask = self._mask(op, src.lanes)
            if mask is not None:
                values = _where(mask, values, torch.zeros_like(values))
            q = _convert(values, DTYPES["i16"], round_mode) if src.dtype.is_float else values
            q = torch.bitwise_and(torch.clamp(q.to(torch.int64), -8, 7), 0xF)
            pairs = q.reshape(-1, 2)
            carrier = torch.bitwise_or(pairs[:, 0], pairs[:, 1] << 4).to(torch.uint8)
            work = torch.zeros(REG_BYTES, dtype=torch.uint8)
            work[slot: slot + carrier.numel() * 4: 4] = carrier[:(REG_BYTES - slot + 3) // 4]
            dst.bytes.copy_(work)
            dst.valid = dst.lanes
            return
        dst_lanes = dst.lanes
        carrier = src.bytes[slot: slot + (dst_lanes // 2) * 4: 4].to(torch.int64)
        lo, hi = torch.bitwise_and(carrier, 0xF), torch.bitwise_and(carrier >> 4, 0xF)
        nibbles = torch.stack([torch.where(lo > 7, lo - 16, lo), torch.where(hi > 7, hi - 16, hi)], dim=1).reshape(-1)
        compute = torch.float32 if dst.dtype.is_float else torch.int64
        work = torch.zeros(dst_lanes, dtype=compute)
        mask = self._mask(op, dst_lanes)
        active = torch.arange(dst_lanes) if mask is None else torch.nonzero(mask).reshape(-1)
        take = active[active < nibbles.numel()]
        work[take] = nibbles[take].to(compute)
        _store_values(dst, work)
        dst.valid = dst.lanes

    def _local_mutex(self, op: Op) -> tuple[int, str]:
        ident = int(self.val(op.attrs["id"]))
        if not 0 <= ident < 32 or op.attrs.get("mode") != 0:
            raise self._err(f"local mutex requires mode=0 and ID in 0..31, got {ident}", op)
        if _ident(op.attrs["side"]) != self.lane.side:
            raise self._err("local mutex belongs to a different core side", op)
        return ident, _ident(op.attrs["pipe"])

    def op_sync_local_mutex_get(self, op: Op) -> None:
        ident, pipe = self._local_mutex(op)
        held = self.lane.local_mutexes[ident]
        if held is not None:
            raise self._err(f"local mutex {ident} acquired twice; held by {held[0]} at #{held[1]}", op)
        self.lane.local_mutexes[ident] = (pipe, op.id)

    def op_sync_local_mutex_release(self, op: Op) -> None:
        ident, pipe = self._local_mutex(op)
        held = self.lane.local_mutexes[ident]
        if held is None or held[0] != pipe:
            raise self._err(f"local mutex {ident} release on {pipe} has no matching get; held={held}", op)
        self.lane.local_mutexes[ident] = None

    def op_sync_mutex(self, op: Op) -> None:
        kind = _ident(op.attrs["kind"])
        flag = Flag(kind, int(op.attrs["id"]), int(op.attrs["depth"]))
        self._crosscore_id(flag.id)
        if not 1 <= flag.depth <= self.m.profile.crosscore_counter_max:
            raise self._err(f"cross-core depth must be in 1..{self.m.profile.crosscore_counter_max}", op)
        self.set_result(op, flag)
        # The old kernelbase prologue: the consumer side publishes `depth` free tokens before the body.
        g = self.lane.group
        if kind == "vc" and self.lane.side == "cube":
            with g.cond:
                for sub in (0, 1):
                    g.vec_wait[sub][flag.id] += flag.depth
                g.cond.notify_all()
        elif kind == "cv" and self.lane.side == "vec":
            with g.cond:
                g.vec_ready[self.lane.sub][flag.id] += flag.depth
                g.cond.notify_all()


    def _cube_ready(self, fid: int) -> None:
        g = self.lane.group
        self._crosscore_id(fid)
        with g.cond:
            if any(g.vec_wait[sub][fid] >= self.m.profile.crosscore_counter_max for sub in (0, 1)):
                raise SimError(f"cross-core flag {fid} exceeds {self.m.profile.crosscore_counter_max} pending tokens")
            for sub in (0, 1):
                g.vec_wait[sub][fid] += 1
            g.cond.notify_all()

    def _vec_ready(self, fid: int) -> None:
        g = self.lane.group
        self._crosscore_id(fid)
        with g.cond:
            if g.vec_ready[self.lane.sub][fid] >= self.m.profile.crosscore_counter_max:
                raise SimError(f"cross-core flag {fid} exceeds {self.m.profile.crosscore_counter_max} pending tokens")
            g.vec_ready[self.lane.sub][fid] += 1
            g.cond.notify_all()

    def _flag_note(self, fid: int, awaited: str, flag: Flag | None = None) -> str:
        """Why a cross-core wait is blocked: which flag, what it waits for, and both token counters.

        A blocked mutex call is attributable only from the counters, so they belong in the message:
        the counter this call decrements is the one whose publisher never ran. `cube->vec` carries a
        `CvMutex`'s `ready` and a `VcMutex`'s `free`; `vec->cube` carries the opposite pair."""
        g = self.lane.group
        who = f"{flag.kind} mutex {fid} (depth={flag.depth})" if flag is not None else f"cross-core flag {fid}"
        return (f"; {who} awaiting {awaited}"
                f"; tokens cube->vec {list(g.vec_wait[sub][fid] for sub in (0, 1))}"
                f", vec->cube {list(g.vec_ready[sub][fid] for sub in (0, 1))}")

    def _wait_cube(self, fid: int, op: Op, flag: Flag | None = None) -> None:
        g = self.lane.group
        self._crosscore_id(fid)
        awaited = "the consumer's `free`" if flag is not None and flag.kind == "vc" else "the producer's `ready`"
        with g.cond:
            while g.vec_wait[self.lane.sub][fid] <= 0:
                self._wait(g.cond, op, self._flag_note(fid, awaited, flag))
            g.vec_wait[self.lane.sub][fid] -= 1

    def _wait_vec(self, fid: int, op: Op, flag: Flag | None = None) -> None:
        g = self.lane.group
        self._crosscore_id(fid)
        awaited = "the producer's `ready`" if flag is not None and flag.kind == "vc" else "the consumer's `free`"
        with g.cond:
            while not (g.vec_ready[0][fid] > 0 and g.vec_ready[1][fid] > 0):
                self._wait(g.cond, op, self._flag_note(fid, awaited + " from both sub-blocks", flag))
            g.vec_ready[0][fid] -= 1
            g.vec_ready[1][fid] -= 1

    def _crosscore_id(self, fid: int) -> None:
        if not 0 <= fid <= self.m.profile.crosscore_id_max:
            raise SimError(f"cross-core flag ID must be in 0..{self.m.profile.crosscore_id_max}, got {fid}")

    def _wait(self, cond: threading.Condition, op: Op, note: str = "") -> None:
        """One bounded wait on ``cond`` (the caller loops on its predicate). Every lane alive blocked while nothing
        left a wait for half a second is a deadlock, reported at once instead of at the timeout.

        ``note`` is the caller's attribution — which flag, what it waits for, both counters — and travels
        into every deadlock message, this lane's and the other lanes' summary.

        The lane gives its process's turn to another lane for the wait and takes it back after leaving the
        wait, so a lane that waits for the turn counts as running, not stalled. Past the limit it comes back
        without the turn: the next call gives the limit's verdict at once instead of after the running lane."""
        import time

        m = self.m
        if m.errors:
            raise SimError("another lane failed")
        here = f"{_lane_name(self.lane)} blocked at {op.opcode} #{op.id} ({op.loc}){note}"
        if m.dead.value:
            raise SimDeadlock(here + CREDIT_RULE)
        remaining = m.deadline - time.monotonic()
        if remaining <= 0:
            with m.count_lock:
                stalled = m.nblocked.value + 1 >= m.live.value  # every other live lane is inside a wait as well
            raise SimDeadlock(here + CREDIT_RULE) if stalled else SimTimeout(here + _timeout_note(m.timeout))
        m.blocked[_lane_name(self.lane)] = f"{op.opcode} #{op.id}{note}"
        with m.count_lock:
            m.nblocked.value += 1
            seen = m.progress.value
        m.give_turn(self.lane)
        try:
            woken = cond.wait(timeout=min(remaining, 0.5))
        finally:
            with m.count_lock:
                m.nblocked.value -= 1
                if woken:
                    m.progress.value += 1  # someone notified: a wait ends for a reason (a timeout is no progress)
                elif m.nblocked.value + 1 == m.live.value and m.progress.value == seen:
                    m.dead.value = 1  # this lane included, every lane alive is blocked and none was notified meanwhile
        if m.dead.value:  # every lane of this process and where it waits (the lanes of other processes report themselves)
            others = _blocked_summary(m.blocked, _lane_name(self.lane))
            m.blocked.pop(_lane_name(self.lane), None)
            raise SimDeadlock(here + (f"\n  other lanes:{others}" if others else "") + CREDIT_RULE)
        m.blocked.pop(_lane_name(self.lane), None)
        if time.monotonic() < m.deadline:
            m.take_turn(self.lane, cond)

    def op_sync_mutex_lock(self, op: Op) -> None:
        flag: Flag = self.raw(op.operands[0])
        if flag.kind == "vc":
            self._wait_cube(flag.id, op, flag)
        else:
            self._wait_vec(flag.id, op, flag)

    def op_sync_mutex_ready(self, op: Op) -> None:
        flag: Flag = self.raw(op.operands[0])
        if flag.kind == "vc":
            self._vec_ready(flag.id)
        else:
            self._cube_ready(flag.id)

    def op_sync_mutex_wait(self, op: Op) -> None:
        flag: Flag = self.raw(op.operands[0])
        if flag.kind == "vc":
            self._wait_vec(flag.id, op, flag)
        else:
            self._wait_cube(flag.id, op, flag)

    def op_sync_mutex_free(self, op: Op) -> None:
        flag: Flag = self.raw(op.operands[0])
        if flag.kind == "vc":
            self._cube_ready(flag.id)
        else:
            self._vec_ready(flag.id)

    # -- debug --------------------------------------------------------------------------------

    def op_debug_print(self, op: Op) -> None:
        args = [self.val(x) for x in op.attrs.get("args", [])]
        print(f"[{_lane_name(self.lane)}] " + str(op.attrs.get("fmt", "")) % tuple(args) if args else f"[{_lane_name(self.lane)}] {op.attrs.get('fmt', '')}")

    def op_debug_assert(self, op: Op) -> None:
        if not self.val(op.operands[0]):
            raise SimError(f"assertion failed at {op.loc}: {op.attrs.get('msg', '')}")


# --------------------------------------------------------------------------- helpers

_I32 = DType("i32", 32, "int")
_U8 = DType("u8", 8, "uint")


def _copy_side_types(op: Op) -> str:
    dst_t, src_t = op.operands[0].type, op.operands[1].type  # type: ignore[union-attr]
    assert isinstance(dst_t, MemType) and isinstance(src_t, MemType)
    return _copy_side_pair(src_t.space, dst_t.space)


def _copy_side_pair(src_space: str, dst_space: str) -> str:
    pair = (src_space, dst_space)
    pair = tuple("gm" if s == "ws" else s for s in pair)  # workspaces are GM memory
    if pair in (("gm", "l1"), ("l0c", "gm"), ("l0c", "ub"), ("l0c", "l1"), ("l1", "l0a"), ("l1", "l0b"), ("l1", "bt")):
        return "cube"
    if pair in (("gm", "ub"), ("ub", "gm"), ("ub", "l1"), ("ub", "ub")):
        return "vec"
    raise SimError(f"no side for a {src_space} -> {dst_space} copy")


def _coerce(v: Any, t: Any) -> Any:
    if isinstance(t, CellType):
        t = ScalarType(t.dtype)
    if v is None or not isinstance(t, ScalarType):
        return v
    dt = t.dtype
    if dt.name == "b1":
        return bool(v)
    if dt.is_float:
        if dt.bits >= 64:
            return float(v)
        if isinstance(v, Binary32NaN) and dt.name == "f32":
            return v  # an f32 NaN keeps its bits (RFC-0001 §6.14, §6.16)
        return torch.tensor(float(v), dtype=torch_dtype(dt)).item()
    return int(v)


def _log_edge(limit: float) -> Any:
    """A logarithm's IEEE edge: -inf at ``limit``, NaN below it (including -inf), None inside its domain."""
    return lambda x: (-math.inf if x == limit else math.nan) if x <= limit else None


def _rsqrt_edge(x: float) -> float | None:
    """1/sqrt(x) at its edges: ±inf at ±0, NaN below zero (including -inf), None inside its domain.
    Measured on A5, a positive subnormal operand gives +inf, as if it were +0 (RFC-0001 §6.13)."""
    if 0.0 < x < 2.0 ** -126:
        return math.inf
    return (math.copysign(math.inf, x) if x == 0 else math.nan) if x <= 0 else None


def _trunc_mod(a: Any, b: Any) -> Any:
    if isinstance(a, float) or isinstance(b, float):
        return math.fmod(a, b)
    r = abs(a) % abs(b)
    return -r if a < 0 else r


def _poison(dims: tuple[int, ...], dt: DType) -> torch.Tensor:
    """Fresh on-chip memory reads as 0xFF bytes (NaN for floats), as in the old simulator."""
    n = 1
    for d in dims:
        n *= d
    raw = torch.full((n * elem_bytes(dt),), 255, dtype=torch.uint8)
    return raw.view(torch_dtype(dt)).reshape(dims)


def merge_mode(op: Op) -> str:
    return _ident(op.attrs.get("merge", "zeroing"))


def _reg_values(reg: RegRef) -> torch.Tensor:
    """A register's lanes as numbers torch can compute on (hif8 decoded, ints widened to int64)."""
    t = reg.tensor()
    if reg.dtype.name == "hif8":
        from ...dtypes.hif8_codec import hif8_to_fp32

        return hif8_to_fp32(t)
    if reg.dtype.name == "e8m0":
        # exponent-only: the byte IS the exponent, and the value it stands for is 2 ** (byte - 127).
        # Without this the lanes read back as the raw code (a 1.0 scale would compute as 127).
        from ...dtypes.e8m0_fp32 import e8m0_to_fp32

        return e8m0_to_fp32(t)
    if reg.dtype.kind == "complex":
        # both widths compute in complex64, like the old simulator's numpy-complex64 data path
        # (c32 lanes are stored as chalf and round back on the store side); bit-exact against its goldens
        return t.to(torch.complex64)
    if reg.dtype.is_float:
        return t.float()
    if reg.dtype.name in ("u16", "u32", "u64"):
        return t.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]).to(torch.int64)
    return t.to(torch.int64)


def _store_values(reg: RegRef, values: torch.Tensor) -> None:
    """Write numbers into a register of any dtype (through the byte views torch cannot index)."""
    t = reg.tensor()
    if reg.dtype.name == "hif8":
        from ...dtypes.hif8_codec import fp32_to_hif8

        t.copy_(fp32_to_hif8(values.float()))
    elif reg.dtype.name == "e8m0":
        t.copy_(_e8m0_codes(values, "floor"))  # already quantised by _convert; floor is exact here
    elif reg.dtype.is_float:
        t.copy_(values.to(t.dtype))
    elif reg.dtype.name in ("u16", "u32", "u64"):
        t.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]).copy_(values.to(torch.int64).to({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]))
    else:
        t.copy_(values.to(t.dtype))


def _e8m0_codes(values: torch.Tensor, round_mode: str) -> torch.Tensor:
    """Positive values -> e8m0 exponent bytes, with the rounding the instruction admits.

    `dtypes.e8m0_fp32.fp32_to_e8m0` floors and refuses a negative or a NaN, which is right for the
    scale-encoding it was written for and wrong for a cast: the hardware admits ROUND_C as well as
    ROUND_Z (the vendor's own 9201 macro), and it has an answer for every input rather than an
    exception. e8m0 carries no sign, so the magnitude is what is encoded."""
    x = values.to(torch.float32).abs()
    e = torch.log2(torch.clamp(x, min=2.0 ** -127))
    e = torch.ceil(e) if round_mode in ("ceil", "CAST_CEIL") else torch.floor(e)
    code = torch.clamp(e + 127.0, 0.0, 255.0)
    return torch.where(torch.isnan(x), torch.full_like(code, 255.0), code).to(torch.uint8)


_DIRECTED = ("floor", "CAST_FLOOR", "ceil", "CAST_CEIL", "to_zero", "trunc", "CAST_TRUNC")
_INT_VIEW = {torch.float16: torch.int16, torch.bfloat16: torch.int16,
             torch.float32: torch.int32, torch.float64: torch.int64}


def _ulp_step(x: torch.Tensor, away: torch.Tensor) -> torch.Tensor:
    """One ULP along the magnitude axis: `away` lanes move away from zero, the rest toward it.

    Done on the bit pattern rather than with `torch.nextafter`, which has no half or bfloat16 form;
    for every IEEE binary format the magnitude bits are a monotone integer, so +-1 on them is the
    neighbour, and it crosses the subnormal boundary and reaches zero without a special case."""
    iview = _INT_VIEW[x.dtype]
    raw = x.contiguous().view(iview)
    bits = torch.finfo(x.dtype).bits
    sign, mag_mask = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    mag = torch.bitwise_and(raw, mag_mask)
    mag = torch.where(away, mag + 1, torch.clamp(mag - 1, min=0))
    return torch.bitwise_or(torch.bitwise_and(raw, sign), mag).view(x.dtype)


def _round_toward(values: torch.Tensor, target: torch.dtype, round_mode: str) -> torch.Tensor:
    """`values` converted to `target` under a DIRECTED rounding mode.

    torch always rounds to nearest, ties to even -- which is ROUND_R, and wrong for three of the
    five modes the `vcvt` admits. `near` is that nearest value and `back` is what it stands for in
    the source's own domain, where the widening is exact, so comparing the two says which side of
    the true value we landed on and one ULP corrects it. Board-measured: an i64 -> f32 asking for
    ROUND_Z was rounding away from zero here (`samples/cast_matrix.py::cast_b64`, 2026-09-05)."""
    if not values.dtype.is_floating_point and round_mode in ('round', 'CAST_ROUND'):
        from .cast_rounding import apply_cast_int_to_float

        return apply_cast_int_to_float(values, target, 'round')
    if values.dtype.is_floating_point and round_mode in ("round", "CAST_ROUND", "odd", "CAST_ODD"):
        from .cast_rounding import apply_cast_float_round

        return apply_cast_float_round(values, target, "odd" if round_mode in ("odd", "CAST_ODD") else "round")
    near = values.to(target)
    if round_mode not in _DIRECTED:
        return near
    if values.dtype.is_floating_point:
        back, exact, over = near.to(values.dtype), values, torch.zeros_like(near, dtype=torch.bool)
    else:
        # int -> float: `near` is integer-valued, so the round trip through float64 is exact unless
        # it rounded onto 2**63, which int64 cannot hold -- those lanes rounded away from zero.
        over = near.abs() >= 2.0 ** 63
        back = torch.where(over, torch.zeros_like(near), near).double().to(torch.int64)
        exact = values
    if round_mode in ("floor", "CAST_FLOOR"):
        return torch.where((back > exact) | (over & (near > 0)), _ulp_step(near, near < 0), near)
    if round_mode in ("ceil", "CAST_CEIL"):
        return torch.where((back < exact) | (over & (near < 0)), _ulp_step(near, near > 0), near)
    step = (back.abs() > exact.abs()) | over
    return torch.where(step, _ulp_step(near, torch.zeros_like(step)), near)


def _convert(values: torch.Tensor, dt: DType, round_mode: str, *, saturate: bool = False) -> torch.Tensor:
    """Convert numbers; integer destinations apply the effective CTRL/RS saturation mode."""
    target = torch_dtype(dt)
    if dt.name == "hif8":
        from ...dtypes.hif8_codec import fp16_to_hif8, fp32_to_hif8, hif8_to_fp32

        mode = "hybrid" if round_mode == "hybrid" else None
        codes = fp16_to_hif8(values, round_mode=mode) if values.dtype == torch.float16 else fp32_to_hif8(values.float(), round_mode=mode)
        return hif8_to_fp32(codes)
    if dt.name == "e8m0":
        from ...dtypes.e8m0_fp32 import e8m0_to_fp32

        return e8m0_to_fp32(_e8m0_codes(values, round_mode))
    if dt.is_float:
        if dt.name in ("e4m3", "e5m2"):
            return values.float().to(target).float()
        return _round_toward(values, target, round_mode)
    if dt.name not in ("u64", "b1") and not values.dtype.is_complex:
        from .cast_saturation import convert_integer

        return convert_integer(values, target, round_mode, saturate)
    if not (values.dtype.is_floating_point or values.dtype.is_complex):
        # int -> int: there is no rounding to do, and the `.float()` below would drop every bit
        # above 2**24 -- i64 -> i32 recorded -268435472 as -268435456, and the board was right.
        return values.to(target)
    if round_mode in ("to_even", "rint", "none", "CAST_RINT"):
        return torch.round(values.float()).to(target)
    if round_mode in ("to_zero", "trunc", "CAST_TRUNC"):
        return torch.trunc(values.float()).to(target)
    if round_mode in ("floor", "CAST_FLOOR"):
        return torch.floor(values.float()).to(target)
    if round_mode in ("ceil", "CAST_CEIL"):
        return torch.ceil(values.float()).to(target)
    return values.to(target)


__all__ = ["Machine", "Interp", "SimError", "SimDeadlock", "SimTimeout", "MemRef", "RegRef", "torch_dtype"]
