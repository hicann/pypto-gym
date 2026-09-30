# Glossary

Fixed vocabulary for code, documentation and conversation. Chinese aliases are listed so an
agent reading either language resolves to the same term.

| Term | Meaning | 中文 |
|---|---|---|
| device | The silicon profile a kernel is compiled for: `a2` (b1–b4), `a3`, `a5` (950), `a5pr` (950pr). Data under `ascriptor/devices/profiles/`. | 设备 |
| arch | The CCE intrinsic family of a device: `c220` (dav-2201) or `c310` (dav-3510). | 架构 |
| backend | A consumer of Lowered IR that produces artifacts: `cce` (primary), `pto_isa` (PTO's tile-level ISA), `pypto_pro` (wrapper-level), `sim` (semantic reference). | 后端 |
| launcher | How an artifact is built and run: `aclnn`, `direct`, `pypto_pro_jit`, `cannsim`, `sim`, board batch runner. Orthogonal to backend. | 启动器 |
| Surface IR | The frontend's output: desugared, typed, structured, device-explicit but before device-specific op selection, side split, event ids and addresses. `ir = "surface/1"`. | 表层 IR |
| Lowered IR | The pass pipeline's output and every backend's input: sides split, every op with pipe, event ids and addresses assigned. `ir = "lowered/1"`. The serialisation boundary. | 下沉 IR |
| op registry | The single source of truth for every opcode: operands, attributes, side, pipe, read/write sets, device availability, dtypes. Derives the verifier, autosync sets, sim routing, capability matrices and docs. | op 注册表 |
| pass | An IR → IR transformation run by the pass manager; verifier before and after; every inserted or rewritten op gets an `origin` entry. | pass |
| origin / provenance | The chain on an op: source location plus which pass changed it and why. | 来源链 |
| function kind | `kernel` (the entry), `vf` (register-level micro function on the vector core), `simt` (thread-parallel function on the vector core). | 函数种类 |
| side | Which core a lowered op runs on: `cube` (AIC) or `vec` (AIV). | 侧 |
| event / channel | A same-side pipe-to-pipe flag: set on one pipe, waited on another; a channel is the ordered pipe pair, with 8 flag ids each; an event of depth *d* owns *d* ids and holds up to *d* tokens (RFC-0005). | 事件 / 通道 |
| carried edge / pre-set event | A hazard between iteration *i* and iteration *i + d* of a loop (slot buffers rotating with a counter); guarded by an event that starts with *d* tokens (`preset`), the old `DEvent(preset=True)`. | 跨迭代边 / 预置事件 |
| slot distance | For a `buf<T, n>` indexed by `cell ± k` with the cell stepping by a constant per iteration: the number of iterations after which the same slot comes round again (`deps.slot_distance`). | 槽距离 |
| vector clock / happens-before | Per-pipe counters carried by ops (statically in `deps`, dynamically in the pipe simulator); an access ordered before another by pipe order, events, barriers or mutex tokens is harmless, everything else is a hazard. | 向量时钟 / 先行发生 |
| slot session | The unit `autosync` plans on the A2 family: a `(side, producer pipe, consumer pipe, capacity class)` identity whose hand-offs one pair of events per block depth orders (RFC-0005 §5.1). | 槽会话 |
| ledger (`ready` / `valid`) | A session's paired events at one depth: `ready` from the producer's pipe with preset 0, `valid` back with one token per credit. The reverse direction *is* the acknowledgement, so nothing needs to bound the producer's run-ahead. | 账本（就绪 / 可用） |
| window / credit | `valid.wait`, producer work, `ready.set` / `ready.wait`, consumer work, `valid.set` — one window per producer/consumer role switch, consecutive same-role work merged into it. Its `W` credits are what the producer may keep in flight. | 窗口 / 信用 |
| capacity / ring | A ledger's credits: `min(slots, sync_depth, ring)`. The ring is the smallest window gap whose accumulated counter advance is a multiple of the slot modulus — zero included, since a counter that does not advance leaves two windows on one slot (RFC-0005 §5.3). | 容量 / 轮转 |
| level (block depth) | What a window is numbered by: every depth that switches roles owns a ledger pair, a block handing over inside its own sub-tree is transparent, and a block opening in a consumer phase gets one credit. | 层级（块深度） |
| retired planner vocabulary | run-ahead bound, armed event, conditioned pair (`mirror`), acknowledgement event — the edge planner's mechanisms, retired with it on 2026-09-17 (RFC-0005 §5.5). Read-only vocabulary for the defect records and migration fragments that measured them. | 已退役的规划器术语 |
| pipe-level simulator | `backends/sim/pipesim.py`: the interpreter's trace replayed on per-pipe FIFOs with the cycle model; reports cycles, hazards, deadlocks (RFC-0006 §9). | pipe 级模拟器 |
| pipe | The hardware pipe an op occupies: `MTE1`, `MTE2`, `MTE3`, `M`, `V`, `FIX`, `S`. | 流水线 |
| position | On-chip memory: `GM`, `L1`, `L0A`, `L0B`, `L0C`, `UB`, `BT`. | 存储位置 |
| layout | `NZ` (cube fractal) or `ND` (row-major) arrangement of a local tensor. | 布局 |
| wrapper layer / `ascrip` | `tensorutils_cce.h`, the thin C++ layer the cce printer calls: `GMTensor` / `Tensor<T, Pos>` / `Buff` / `Event` and one function per Lowered opcode, each a single intrinsic sequence (RFC-0007 §2). | 封装层 |
| gap (`CceGap`) | An op a backend has no line for; reported with the op id and its source location, never worked around (D-039). | 后端缺口 |
| launcher: aclnn / cannsim / board | Build the cce artifact as a CANN custom op and run its aclnn API through the host harness: on the local card, under `cannsim record`, or on this machine's assigned card. | aclnn / cannsim / 板端启动器 |
| `boards.json` | Git-ignored machine facts of the board launcher, derived from `machine_specs.md`; never printed. | 板端配置 |
| autosync | The pass that marks which operations a `region.autosync` owns and hands them to the planner of that device family — slot mutexes on A5, slot sessions on A2/A3 (RFC-0005). It inserts no events of its own. | 自动同步 |
| crosssync | The pass that checks cross-side hazards — mutex coverage (warning) and credits (error) — for EVERY kernel, including one with no `region.autosync` in it. Read-only; it was part of `autosync` and inherited that pass's gate (RFC-0005 §3.1). | 跨侧检查 |
| static evaluation | Frontend rule: a sub-tree whose type is not decided by a DSL value is evaluated by Python at compile time and becomes a constant (RFC-0002). | 静态求值 |
| taint | Whether an expression depends on a DSL value (dynamic) or not (static). | 污点 |
| functional golden | A recorded (inputs, outputs, digests, tolerance) case for one kernel, produced once from the old repository (RFC-0003). | 功能黄金 |
| verification ladder | T1 new sim → T2 local cannsim → T3 shared board, bit-level A/B (RFC-0004). | 验证阶梯 |
| shape symbol | A string naming a dimension in a signature (`("M", "K")`); becomes a scalar kernel parameter whose value the launcher derives from the real tensor. `?` marks a ragged list-member dimension read at runtime. | 形状符号 |
| list descriptor | The GM-resident table describing a `GMTensorList` (count, per-member pointer and dims); an ABI between launcher and kernel (RFC-0001 §13). | 列表描述符 |
| reference interpreter | The functional level of the `sim` backend: executes Surface IR per core lane, no pipes or cycles; the T1 oracle (D-022). | 参考解释器 |
| vector-only value | A scalar derived from `vec_idx` / `sub_block_idx`; exists on vector lanes only, and so do the loops and branches it decides (D-023). | 仅向量侧的值 |
