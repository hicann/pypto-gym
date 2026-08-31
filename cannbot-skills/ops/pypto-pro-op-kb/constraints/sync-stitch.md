# Synchronization and section handoff

## Rule

Use `make_tile_group(..., auto_mutex=True)` for rotation and intra-kernel
buffer ownership that the target API documents as auto-managed. Do not add
manual synchronization to the same managed dependency without evidence.

Cross-section or AIC/AIV handoff is not implied by `auto_mutex`. When the data
flow crosses engines or sub-blocks, copy the producer/consumer event sequence
from a matching official example for the installed SDK and validate it on the
target.

## On A5: delivery rule and historical evidence

For delivery, follow the current
[manual preload design](../../pypto-pro-op-design/references/cv_fusion_pipeline.md)
and its matching official sample inside one `@pl.jit`. The generated-pipeline
result in [framework-findings §17](../references/pypto-pro-framework-findings.md)
and the split-launch results below are historical evidence, not alternative
delivery contracts.

### Historical failed configuration: one hand-written attention handoff

Measured, repeatedly. A single `@pl.jit` holding a `section_cube()` and a
`section_vector()` with a **hand-written** cross-core handoff on every tile
compiles and then dies with `aicore timeout`. One attention kernel was taken
through **thirteen hypotheses and nine mutation-ladder rungs** on that construct
and never ran once, even though every ingredient passed in isolation — the cube
half (including the transposed NT load and a dual-accumulator contraction) and
the register-level softmax were each proven separately. A reference
implementation written against a different kernel language hit the identical
wall on the identical construct. This rejects that exact event/buffer/loop
organization, not the current manual preload design or every hand-written path.

### Diagnostic-only experiment: decompose into single-sided launches

Each kernel is cube-only or vector-only and contains *no* cross-core
synchronisation at all; ordering comes from the launch boundary. Two operators
were rescued this way:

| operator | fused | single-sided |
|---|---|---|
| attention prolog | 0/20 | 11 launches → 19/20 |
| attention | never ran | 3 launches → 20/20, correct on the second device run |

The cost is the intermediates round-tripping through GM — up to 268 MB of score
matrix on the largest attention case — and that is real. It is also affordable
far more often than it looks, because the benchmark pays `0.3` per accurate case
*before* any performance term: a slow correct kernel scores, a fast one that
never runs does not. Reserve this split for diagnosis or research after the
applicable one-launch design has been tried; never deliver it.

In a diagnostic harness there must still be **no launch inside a host loop**:
use a fixed number of launches, each looping internally. A delivered wrapper
must launch exactly once; otherwise report a design blocker. See
[wrapper-boundary.md](wrapper-boundary.md).

## Review sequence

1. Draw each producer→consumer dependency and the memory space carrying it.
2. Mark which dependencies are covered by tile-group mutexes.
3. For every remaining dependency, cite the official API/example that defines
   the required pipe and event.
4. Allocate event IDs without overlap across simultaneously active pipelines.
5. Match physical slot count to the maximum number of in-flight work items.
6. Validate correctness before adding preload depth or extra buffering.

## Evidence

- auto-managed rotating Vec groups:
  [softmax_impl.py](../examples/samples/softmax/softmax_impl.py)
- retained cube→vector handoff with embedded correctness test:
  [fused_matmul_add_impl.py](../examples/samples/fused_matmul_add/fused_matmul_add_impl.py)
- retained vector→cube handoff with embedded correctness test:
  [vec_cube_abs_sqrt_matmul_impl.py](../examples/samples/vec_cube_abs_sqrt_matmul/vec_cube_abs_sqrt_matmul_impl.py)

These samples demonstrate their recorded environment only. The installed
`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/` documentation and official samples
remain authoritative for pipe/event signatures.

---

## A hang is not evidence of a synchronization bug — locate the PC first

Five ablations of the cross-core edges of one deadlocking kernel all came back
negative: even-vs-odd loop parity, moving a release to a different pipe, deleting
the guarded cube-side wait, deleting the guarded vector-side wait, and deleting
both back edges. Each was a well-formed experiment and each said "not this."

They were all aimed at the wrong resource class. Resolving the two stuck PCs to
source settled it in one step:

| core | parked at | pipe |
|---|---|---|
| AIC, all 28 | `get_buf(PIPE_MTE1, b_right/b_l1)` → `TEXTRACT`, inner K sub-block loop | MTE1 |
| AIV, all 56 | `TSTORE(out_tile)` → `set_loop_size_ubtoout` | MTE3 |

**Neither is a `wait_intra_block` or a `wait_flag`.** There was no flag id to
audit, and no cross-core edge to fix, because the machine was never parked on
one — it was parked on **pipe/buffer resource acquisition**. Every core of both
types at a fixed PC is a structural stall, and `aicore timeout` names the
watchdog, not the mechanism.

**The lesson is about ordering of work.** "All cores hang" invites the
synchronization hypothesis, and the vocabulary of the failure (`aicore timeout`,
`retCode=0x25`) reinforces it. But a hang localises *exactly* — one address per
core type — and that address is cheap to obtain. Get it before enumerating
candidate edges; five ablation rounds is a lot to spend re-deriving "not a wait."

### Getting the source line without an ISA disassembler

The obstacle looks like tooling and is not. No Ascend disassembler was available
(`msobjdump`'s Python module absent, `llvm-objdump -d` prints
`<not available>` for every instruction, no bisheng toolchain `bin/`). **But the
instruction is not what you need — the source line is**, and that needs only
`.debug_line`, which `llvm-symbolizer` and `llvm-dwarfdump --lookup` read without
knowing the ISA.

The recipe, with the two checks that make it trustworthy:

1. The device binary is embedded in the built shared library:
   `objcopy --dump-section .aicore_binary=out.bin <build>/tk_*/call_kernel.so /dev/null`.
   Load base is reported in the fault dump; `vaddr = PC − base`. **Verify the base**
   by checking that the AIV `pc start` equals `base + _mix_aiv` from the symbol
   table — do not assume it.
2. Rebuild with `-g` appended to the compile flags, then **prove `-g` was
   codegen-neutral** before trusting the line table: compare `.text` *bytes*,
   `_mix_aic`/`_mix_aiv` offsets, and the generated `kernel.cpp` against the
   archived binary that actually hung. Byte-identical `.text` is the gate. If it
   shifts, the line table describes different code and the lookup is void.

Two practical notes. A frozen config dataclass can be patched in the probe
process only (`object.__setattr__` on the flags tuple), so nothing is installed
on a shared board. And **the rebuild need not reproduce the hang**: a tiny case
on the *same tiling key* emits the identical binary, because the key selects the
specialization while shapes are runtime tiling parameters — 1.6 s instead of a
570 s watchdog timeout.

### What this did *not* establish

The investigation that produced the rule above also produced a specific
geometric hypothesis (one tile group declared outside both sections, shared
mutex ids), a discriminator that **refuted** it, and a claim about the
vec-to-cube handshake that was later **withdrawn** under isolation. None of it
is carried here: this page is routed as a constraint, and a page an agent reads
as normative must not also carry the arguments that were overturned on the way.
What survives is the rule -- locate the PC before believing a hang is a
synchronization bug -- and the technique for doing it without a disassembler.

## 从累加器回 L1 没有直达路径：`Acc → Vec → Mat`，或走 GM

`pl.move` 的空间表（本 checkout 的 `pypto_pro/language/_api.py:196-206`）只列出
`Acc (L0C) → Vec (UB)`（fix 流水）与 `Vec (UB) → Mat (L1)`（mte3）两条，
**没有 `Acc → Mat` 一行**。

> **证据等级：这是关于那张表的陈述，不是一次完整的能力缺失判定。**
> 本 KB 自己的
> [absence gate](../references/investigation-discipline.md)（§11「Before claiming a
> capability is absent, run the absence gate」）要求 API 文档、安装源码、可组合原语、
> 最小探针与生成代码/板端证据齐备后，才能宣布一项能力不存在。此处只走到了"安装源码里
> 那张空间表没有这一行"。因此下面按"没有直达路径"来规划是**当前最稳妥的默认**，
> 而不是已证成的禁令——若你确有理由需要 `Acc → Mat`，先补齐 absence gate 再下结论，
> 不要把这段当作已经替你跑完了那个 gate。

**设计后果**：不要围绕 `L0C → L1` 直搬来规划 cube→cube 的片上再喂。要么预算
`Acc → Vec → Mat` 这趟往返——两次 move、两个空间，外加一次向量流水触碰，
且必须与 fix→mte3 的次序对齐——要么走 GM 中转。

链式收缩（QK^T 后接 PV，或任意两段 GEMM）最容易踩这条：把累加器直接当作下一个
matmul 的 L1 操作数在 DESIGN 阶段看起来成立，到 Stage 4 才发现没有对应的 `pl.move`
拼写。**在设计期就按上面两条之一预算，不要留到实现期。**

## `phase=` is a hardware handshake, and only `pl.store` can answer it

Added after a hang that was misattributed to `Acc→Vec` for weeks. Full entry:
`references/pypto-pro-dsl-limitations-a5.md` #28.

`phase=` on a `matmul` **turns off** the framework's automatic M↔FixPipe
synchronization and replaces it with a hardware `unit_flag` that the paired
`store(phase=...)` must clear. **`pl.move` has no `phase` parameter**, so a
`matmul(phase=Final)` drained by `pl.move(..., acc_to_vec_mode=...)` arms a
protocol whose other half cannot reply. Nothing rejects it.

Checklist item, before diagnosing any Cube-side hang or L0C fault:

1. For every `matmul` carrying `phase=`, name the drain. If it is not
   `pl.store` / `store_tile` **with `phase=`**, that is the defect — stop here.
2. Do not read buffer depth as causal. A slot buys one iteration; it does not
   clear the flag. Depth 1→2 "fixing" a hang means the test shape stopped
   re-entering a dirty block, not that the race is gone.
3. Do not treat a hang and an ECC fault as different bugs. `phase.md`'s 案例三
   (flag stuck at 1 → the next matmul on that block waits forever) and 案例二
   (read not gated on completion → fixpipe reads unwritten L0C → multi-bit ECC
   `error 171`) are the **same** violation seen from the writer's and the
   reader's side.
4. A passing cut is not a cleared cut. Two cuts here carried the violation and
   passed because `kv_tiles = 1` against a 2-slot accumulator never re-entered a
   dirty block. Re-test at a shape that forces re-entry (`kv_len >= 384`).

An `error 171` multi-bit ECC on an L0C read also drives the card into
driver-level `Alarm` through the RAS path — `8C4BA00C`, "The software has failed
and cannot recover", needing a driver reset. A single probe here cost two of
three cards.

**Do not reproduce a suspected L0C fault on shared hardware.** This is a
default prohibition, not a budgeting exercise — an earlier revision of this page
said "budget cards accordingly", which reads as permission and is how the two
cards were lost. Diagnose from the generated code and the failure signature
first; they are usually sufficient, because the hang and the ECC fault are the
same violation (point 3 above) and the hang side is not destructive.

If a live reproduction is genuinely unavoidable, every one of the following must
hold before it runs, and the probe is forbidden if any is missing:

1. written authorization from the hardware owner for this specific probe;
2. an explicitly named device from an allowlist, never "whatever is free";
3. exclusive possession of that device, with every other job drained first;
4. a single attempt with a hard cap — no retry loop, no sweep;
5. isolation from any shared scheduler or CI pool;
6. a written recovery path (driver reset procedure and who may run it), agreed
   before the probe rather than discovered after it.

Record the outcome so the next reader does not pay for it again.
