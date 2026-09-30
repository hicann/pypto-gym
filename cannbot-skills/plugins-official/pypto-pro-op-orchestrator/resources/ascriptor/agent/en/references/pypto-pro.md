# Targeting PyPTO-Pro

Where this backend differs from CCE, in the order the differences bite. The execution entry itself
is in [running what you are writing](development-execution.md).

## What is delivered

"A PyPTO-Pro kernel" here means **the Ascriptor DSL source together with the `kernel_pypto.py` it
emits** (one per shape), plus the manifest, the driver and the board evidence. It does not mean
hand-written `pypto_pro.language` code. How those sit in a package is
[the delivery area](#delivery-area), the last section of this page.

<a id="emit-first"></a>
## Emit first, then run

```python
from ascriptor.runtime import compile_kernel
compile_kernel(kernel, backend="pypto_pro", bindings={"B": 2, "T": 9})
```

Seconds, no data moved, no card needed. **Do this before sim.**

The reason: the PyPTO support tables are per **opcode**, but many gaps are per **operand** — the
same opcode with a different stride or dtype has no pto spelling. Those surface **only at emit**. A
real case: a kernel passed both sim and pipesim, and emit then reported
`dma.gm_to_ub.nd loop_src_stride=[64, 1]: a non-unit INNERMOST stride (64) has no pto spelling`
(that is `gm_to_ub_nd_dma_transpose`), so the whole dataflow orientation had to be rewritten.
Emitting first turns that loss from a rewrite into a design choice.

A `PyptoGap` always carries the op id and the source location.

## Specialised per scalar valuation

Every shape symbol needs a binding, or emit reports `scalar parameter D has no binding`. Write
compile-time constants as literals (`GM[f32, ('B', 'T', 'Nqk', 64)]`) and leave only genuinely
runtime dimensions as symbols.

One emit covers one set of scalar values, so there is **one artifact per case**; PyPTO JIT-compiles
on the first call on the box.

<a id="cores"></a>
## The core count: missing means a hang, not an error

A board entry must state `cube_cores`, this card's **AIC count**. Without it:

- an **unpinned** launch (no `block_dim` on the kernel or the call) falls back to the device
  profile — a5 says 32 — and a vec module doubles that, so **64 AIVs are launched on a 56-AIV
  card**. Measured: the emitted manifest of an undeclared vec kernel says `"block_dim": 64`.
- a **pinned** launch skips the clamp entirely. It is not doubled: `block_dim=8` emits `8`, so
  a pinned number over the card's count is launched as written.

`sync_all` is a hardware barrier that waits for every core the launch names. Cores that never
arrive do not raise — they **hang**; one `simt_atomic_add` sat there for nine minutes before this
was understood. `run_pypto` now refuses instead. An Ascend950PR is 28 AIC / 56 AIV.

## Delivery synchronization mode

New PyPTO-Pro deliveries generate and validate `sync_mode="auto_mutex"` by default,
emitting `@pl.jit(auto_mutex=True)`. Generate manual only when the user explicitly
requests it. The low-level `OpExec(..., launcher="pypto")` and `compile_kernel` defaults
remain manual, so delivery workflows must select auto_mutex explicitly. If native
emission or hardware validation fails, retain the evidence and report the blocker;
do not silently produce manual. See the [delivery synchronization policy](../runtime-and-maintenance.md#sync-closeout).

## Always check against CCE

A successful emit does not prove vendor compilation, and a board pass does not prove the semantics.
This backend has **built, run and been wrong**: 64-bit `vf.cast` read the wrong lanes under pypto
while cce was bit-exact (D-221).

So when a board result surprises you, **run the same case through cce before concluding anything**.
A pypto failure is a porting defect only when cce passes that same case.

## Reading a board failure

`BoardError` carries the tail of the box's own `run.log`. Two common empty-message signatures:

| In the log | Cause |
|---|---|
| `libhccl.so: cannot open shared object file` | The CANN tree is incomplete, or its owner deleted it |
| `F7A008 FILE_ERROR … aarch64 toolchain g++` | The same; pypto resolves the cross toolchain from `ASCEND_HOME_PATH` by canonical path |

The full log is at `out_dir/board_pypto_run.log`.

<a id="delivery-area"></a>
## The delivery area

A PyPTO-Pro task hands over a delivery package, and it **does not enter the gallery** — the
[three more files](development-execution.md#admission) route is for CCE demos. Development
artifacts and raw evidence stay in `custom/<op>/`; the runnable package lives separately in
`delivery/<op>/`. Both are in the local task project, outside the checked-in source snapshot:

```
delivery/<op>/
├── kernels/         PyPTO-Pro sources actually used; one may cover several cases
├── golden_cpu.py    torch only, never imports ascriptor
├── wrapper.py       routes a case to one file in kernels/
├── test.py          the single entry
├── DESIGN.md
├── REPORT.md
└── testing/ etc.     only local modules actually needed by test.py
```

The frozen Scriptor contract, `scriptor/` source, `generated/` export, `.scriptor/` state and raw
`reports/` remain in `custom/<op>/`, outside the package. `delivery/<op>/` must run using only
its own files and declared PyPTO-Pro and Torch/NPU runtime dependencies. Local test helpers and
comparators must be included; `.opencode`, a development checkout or another operator directory
cannot be an implicit dependency. Map each case exported from the selected DSL to a byte-identical
kernel source in the package. Cases may share a source, but every packaged kernel source must be
reached. Prefer flat `kernels/<variant>.py` files; retain subdirectories only when local dependencies
require them. Literal `CASES` in `test.py` has `name`, `kernel`,
`input_shapes`, `input_dtypes`, `output_shapes`, `output_dtypes` and `params` per SPEC P0 case, matching
the selected `generated/export.json`. `wrapper.py` must load from the
packaged `kernels/`, never from the working `generated/` directory.

| File | What it owns | The boundary |
|---|---|---|
| `golden_cpu.py` | `make_inputs(case)`, `reference(inputs)` | The gallery's `reference.py` line exactly: a reference that calls the thing it is checking checks nothing |
| `wrapper.py` | Selects one file in `kernels/` per case | Every branch must be reached by a case |
| `test.py` | Runs the packaged public wrapper on device against an independent CPU golden | Literal `CASES` records the SPEC P0 cases; `--output` writes per-case results |
| `DESIGN.md` | The constraints in force and the structure they forced | Not the trials that failed |
| `REPORT.md` | The minimum content below | Final before `.tmp` is deleted |

**Four steps on one machine.** Generate inputs, compute the golden, run the kernel and
compare in the same interpreter on the local device machine. The CPU golden uses that
machine's CPU. Torch random inputs are not bit-identical across platforms, so a
golden computed on another host cannot be paired with a local kernel run. Keep each
task's work area under its own directory.

**The router must prove it was fully exercised.** `test.py` prints the actual selected file for
each case, writes machine results with `name/status/kernel`, and asserts that every packaged kernel
source was reached. Poison outputs before launch and check that unwritten elements cannot look like
a pass. Working DSL `OpExec` tests still use `seed_outputs=True`; the final PyPTO-Pro package uses
equivalent poisoning. The independent Scriptor verifier tests DSL `OpExec` and the exported wrapper
in the work area. Packaged `test.py` exercises what the user receives: `wrapper.py`.

**REPORT.md carries at least:** the result of every case × every launcher (`sim`/`pipesim`/`pypto`),
including **the cce control column** — a pypto failure is a porting defect only when cce passes that
same case; the measured board latency of the selected synchronization mode (see
[the delivery synchronization policy](../runtime-and-maintenance.md#sync-closeout)); each case's manifest
`block_dim` and op counts; the source identity and the dependency versions actually executed; every
skipped case with its reason, and the reason says what to run instead; and a done list and a
remaining list at the end.
It also records the selected source and export identity and, for each case, its shape, dtype,
parameters, selected file under `kernels/` and acceptance evidence. If the workflow tunes after
an earlier acceptance, keep the earlier report as historical evidence and name only the actually
delivered revision as final in `REPORT.md`; mark failed or reverted candidates as unselected.
Every delivered dtype belongs to the same `CASES` and acceptance scope. An unverified side
directory cannot fill a gap in the delivery entry point.
For Scriptor mode, write the full SHA-256 of the final `generated/export.json` and every
`scriptor/*.py` source into `REPORT.md` for the closeout check. Short hashes do not establish
delivery identity. Mark the final revision with exactly one
`Final export SHA-256: <64-digit hash>` line and one
`Final source SHA-256 (scriptor/file.py): <64-digit hash>` line per source.
Historical candidate hashes may appear later, but not in these final fields.

**The order in which `.tmp` goes.** Emitted artifacts and model results are reproducible — delete
them and re-run if needed. **Board results are not**: the card may not be available again, and a
latency measurement is a one-off. So the order is **REPORT.md final → delete `.tmp` → deliver**.
Reversed, those board numbers can only be written from memory. The package excludes `.tmp/`,
`.scriptor/`, `scriptor/`, `generated/`, `reports/`, `prototype/` and `.DS_Store`; the work area
retains the raw evidence.

Before delivery, inspect `REPORT.md` and the final handoff document for machine
addresses, credentials and private paths. Keep this review with the task evidence.
