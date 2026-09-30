# Compile, run, inspect and extend

`ascriptor.runtime.compile_kernel(entry, backend=..., bindings=...)` lowers a
typed entry and returns an `Artifacts` object. Its `files` mapping contains
relative paths and bytes; `entry` identifies the primary file. This is source
emission. It does not run a vendor compiler or execute an NPU kernel.

Generated sources use shared integer scalar simplification by default:
constant arithmetic and redundant intermediates are removed, slot snapshots
remain explicit, and CCE/PTO print general floor remainder through a typed helper. To inspect
the expanded diagnostic form, lower with
`PassManager(PIPELINE, options={"scalar_simplify": False})` and pass that module
to the desired backend's `compile`. `dump-ir --after scalar_simplify --explain`
shows the IR decisions. These transformations preserve integer semantics;
floating-point reassociation is outside their scope.

PyPTO then removes scalar computations with no executable consumer, including
IDs used only by omitted local mutex operations and unused projected counter loops. Adjacent same-type
in-place counter updates and direct boolean conditions can inline; buffer
snapshots and cross-variable copies remain materialized. Artifact metadata
`pypto.scalar_cleanup` records each removal/inlining and its original op ID.
Generated VF signatures and calls also omit parameters with no reads in the
emitted body; used aliases and the public kernel signature remain intact.
See the [VF interface contract](../rfc/0013-pypto-native-synchronization.md#generated-vf-parameters).

PyPTO emission of explicit integer casts and narrow VF div/mod requires the
[integer `pl.cast` supplement](../pypto-pro-supplements.md). The adapter restores
the operands' declared IR types; it preserves VF scheduling and genuine 64-bit
values. See the [width contract](../rfc/0013-pypto-native-synchronization.md#integer-scalar-width-at-vf-division).

PyPTO explicitly rejects runtime local allocation capacity and dynamic FIX
source pitch. Specialize allocation dimensions or use fixed storage with proven
bounded runtime valid extents; see the [capacity contract](../rfc/0013-pypto-native-synchronization.md#static-local-capacity).

`OpExec(entry, launcher=..., backend=..., device=..., block_dim=...)` returns a
callable that binds concrete tensor shapes and scalars to the entry.

**Every tensor in the entry's signature is an argument of that call, outputs included.** The
caller allocates them and passes them in, in signature order, with explicit scalars after; the
runtime allocates nothing on your behalf, and a call that leaves an output out is refused with
`<entry> needs N tensor argument(s), got M`. Returning an output from the kernel body does not
make it an out-parameter — that return names which buffer the result is in, and the buffer is
still yours. Owning them is also what makes the seeding rules below meaningful.

`out_dir` is where a launcher that *builds* something
writes it — the generated project, the artifacts, the lock. `sim` and `pipesim` build
nothing and write nothing there: their result is the returned tensors and the exception a
failed check raises. Full-overwrite outputs are poisoned by
default; use `seed_outputs=True` when the algorithm consumes their initialized
values. Every repeated check/profile launch must receive the same declared
initialization. Unknown or unsupported combinations must fail explicitly.

### What an omitted block_dim means

`@kernel(..., block_dim=N)` declares the launch; a `block_dim` passed to `OpExec` or
`compile_kernel` overrides it, and an imported kernel refuses any value but the one it was
exported with. Omitted in both places, it means *the whole machine*, and each path fills that
in for itself:

| Path | What an omitted block_dim becomes |
| --- | --- |
| `sim`, `pipesim` | The device profile's vector cores for a vec module (measured: 64 lanes on a5, one row each) |
| `cce` | Nothing is baked: the artifacts record `"block_dim": null` and the generated project's host source calls `SetBlockDim(coreNum)`, so the platform answers on the box |
| `pypto_pro` | A number is printed: the card's count when the runner knows it, otherwise the profile's cube cores, doubled for a vec module (measured: 64 on a5) |

A *declared* block_dim is passed through unchanged everywhere: `block_dim=4` emits `4`, with
no doubling. The doubling applies only to the fallback above, which is why a card with fewer
AIVs than its profile needs `cube_cores` in its board entry — [running the whole unit on the
box](#running-the-whole-unit-on-the-box) has that contract.

The [AXPB example](../../examples/api/axpb) is the shortest thing that runs. Its
`kernel.py` imports no Torch, its `reference.py` no compiler, and `main.py` is
both the entry point and the comparison:

```sh
cd examples/api/axpb
python main.py --list                 # the case ids, and what each one is for
python main.py                        # every case on the functional simulator
python main.py --launcher pipesim     # the same, through the pipe model
python main.py --inspect              # identity and Surface IR, no execution
```

Inspection exposes `.name`, `.device`, `.mode` and `.ir()`; the written Surface
IR retains source locations. The callable entry itself is not a host function.
The entry's `.device` is the canonical device identifier: an `ascriptor.a5`
entry reports `"950"`, while `"a5"` is the facade/launcher alias. Use the
[facade-to-device table](reference.md) when checking this field; comparing it
to the literal alias `"a5"` rejects a correctly bound A5 entry before execution.
The simulator extra supplies Torch/NumPy for independent references; base
facade import and supported source emission do not require them.

Every example is the same four files and the same entry point; there is no shared
runner, no `contract.json` and no generated helper to keep in step. Cases,
comparison rules and destination initialization are in that folder's `main.py`,
where the reader is already looking, and `python tools/api_examples.py --check` is
the gate on the shape. Source emission has no example subcommand: it is
`compile_kernel` as the first paragraph of this page describes it, and
[backend_extension](../../examples/api/backend_extension) is where a folder calls it.

`--launcher board` runs on this machine's own card; the machine identifies itself
through ignored configuration that `ASCRIPTOR_BOARDS` names. CANN, drivers and
vendor tooling are external dependencies. Simulation cycles and CPU timings
are not board latency. Source emission, vendor compilation and board results
must have separate evidence rows.

Proven normalized slot indices are reused by CCE, PTO-ISA and PyPTO. For example,
`slot_1 = iteration & 1` can index a two-slot buffer directly, without another
`% 2`, `SlotOf` call or `_ix` copy. The immutable snapshot remains valid after
`iteration` changes; unknown indices still wrap.

## Running the whole unit on the device machine

The board and PyPTO launchers execute only on the machine running the Python process.
The checked-in `library/` directory must be on `PYTHONPATH`; no wheel or remote
transport is required. On a machine with an assigned card, point `ASCRIPTOR_BOARDS`
to an ignored JSON configuration whose selected entry has `"local": true`, a local
`workspace`, and the card's actual `cube_cores`. Connection fields are rejected.

```sh
PYTHONPATH=<snapshot>/library python -m ascriptor.cli doctor
PYTHONPATH=<snapshot>/library python <snapshot>/library/examples/api/axpb/main.py --launcher pypto --backend pypto_pro
```

Generate inputs, the independent reference, device outputs and their comparison on
that machine. Keep each run's output in an isolated task directory. A local workspace
lock serializes use of the selected card; it does not identify a remote host. CANN,
PyPTO-Pro, drivers and the assigned card remain external environment requirements.
`sim` and `pipesim` can run without a card; neither establishes device acceptance.

## Advanced backend protocol

The accepted extension boundary is `Backend`, `Artifacts`, `Capabilities`,
`ResourceLimits` in `ascriptor.backends.base` and the `ascriptor.backends`
distribution entry-point group. `Backend` declares a name and implements
`capabilities()`, `resources(device)` and `compile(module, options=None)`.
It consumes the declared Lowered IR major and preserves source-located
diagnostics for unsupported operations. Package versions and IR majors are
independent. A backend accepting scalar specialization must pass through
`options["bindings"]`, including zero-valued bindings.

`OpExec(..., launcher="pypto")` defaults to `sync_mode="manual"`, and so do
`emit_module` and `PyptoProBackend.compile`: this backend owns its local credits
rather than delegating them (D-260). Manual mode emits `auto_mutex=False` and
translates the IR's local mutex operations to calls.

```python
manual = OpExec(kernel, launcher="pypto")                            # the default
native = OpExec(kernel, launcher="pypto", sync_mode="auto_mutex")
```

The generated `@pl.jit(auto_mutex=True)` of the other mode delegates local locking
to PyPTO using exactly the IDs assigned to physical slots in Lowered IR. A Tensor
becomes a one-member Tile group; xBuff members preserve their slot order and IDs.
IR `get_buf`/`rls_buf` operations emit neither calls nor comments in that mode;
they remain available in Lowered IR for inspection. Authoring and optimization
start from manual and try `auto_mutex` once as a closeout before delivery.

The `sync_mode` keyword applies only to the PyPTO launcher; other launchers keep
their defaults and reject it.
`compile_kernel` takes no `sync_mode`, so it emits each backend's default — for
`backend="pypto_pro"` that is manual, the same `auto_mutex=False` sources as
`OpExec(..., launcher="pypto")`. Call `PyptoProBackend.compile(module, {"sync_mode":
"auto_mutex"})`, or `emit_module(..., sync_mode="auto_mutex")`, when the other mode is what
you want; both are imported from `ascriptor.backends.pypto_pro`.

The former backend event planner and experimental `keep_events` option are
removed. All events, raw flags, barriers and cross-core protocols still present
in IR are emitted unchanged. No backend ID allocator or event fallback remains.
Aliases inherit the existing IDs of overlapping slots. Unsupported managed Tile
geometry or native operations produce located errors. See
[RFC-0013](../rfc/0013-pypto-native-synchronization.md) for the mapping and migration
contract; hardware qualification remains workload-specific.

`Capabilities` describes function kinds, opcodes, dtypes, devices and notes.
`ResourceLimits` states byte capacities for UB, L1, L0A, L0B, L0C and BT.
These fields describe an implementation's declared surface/resources; they
are not measurements proving every operation works on every device.

The [extension conformance example](../../examples/api/backend_extension)
delegates CCE emission, checks protocol conformance and artifact equality,
adds metadata, and exercises lazy discovery through temporary process-local
distribution metadata. Its `pyproject.toml` shows the installed registration
form. This teaching adapter performs no new instruction lowering and needs
no environment mutation to run its checks. Private compiler/pass APIs used
to inspect a sample are not additional extension contracts.

## Diagnostic helper boundaries

Use IR inspection and [the simulator debugging guide](../diagnosing-sync.md)
to locate operations and inspect execution state. `sim_print`, `kernel_print`
and `print_reg` are diagnostic markers; CCE emits comments for them. The
current `kernel_print` declaration does not promise the historical vendor
`printf` behavior. The dump markers similarly produce CCE comments, and their
current simulator handler does not write tensor files. A captured intermediate
used during diagnosis is not an independent expected result.

`inline`, `EntireType` and `DcciDst` are outside the agreed facade. The current
`clean_dcache(dst)` declaration takes a target view; do not carry old cache
policy parameters into new code. Cache-line maintenance, source allocation
ownership and the simulator's memory-conflict granularity are different
concepts. A small output does not by itself justify cache maintenance.
