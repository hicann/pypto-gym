# PyPTO-Pro DSL limitations on A5 — upstream report

For the framework team. Every entry below was established on **Ascend950PR**
under **CANN 9.2.0**, against the `pypto_pro` overlay in this checkout, by an
on-board probe or by reading the installed source tree. None is inferred from
documentation alone; where a claim is a property of *this* installation rather
than of the DSL in general, the entry says so.

Entries are ordered by severity, and severity here is dominated by one
question: **does the defect announce itself?** A compile error costs an hour. A
silent wrong answer costs a release. The first four entries produce correct-
looking output from incorrect code, with no error at any layer.

**This file continues an earlier report and does not replace it.**
[`docs/pypto-pro-dsl-limitations.md`](../../../../docs/pypto-pro-dsl-limitations.md)
carries 20 entries from a four-operator sweep (`foreach_addcdiv_scalar`,
`swi_glu`, `apply_rotary_pos_emb`, `cummin`) with full citations. Those entries
are **indexed by title at the bottom of this page, not restated** — re-expressing
a measured entry is how citation fidelity drifts. Read both; they do not
overlap.

Narrative context, including the diagnostic paths that produced these, is in
[`pypto-pro-framework-findings.md`](pypto-pro-framework-findings.md) §"A5 probe
session findings (2026-08-07)".

---

## Tier 1 — silent: wrong results, no error at any layer

### 1. `mrgsort2`'s parameter order disagrees between the API and the backend

**Symptom.** The Python declaration and the documentation both specify
`(src0, src1, dst, tmp)`. The IR and the CCE backend consume
`(dst, src0, tmp, src1)`.

**Evidence.** Called in the documented order, the kernel **compiles and
launches successfully** and `dst` retains its previous contents. Called in the
backend order, it sorts correctly.

**Impact — silent, and maximally so.** Every signal available to the author
says the code is correct: the declaration matches, the docs match, the compiler
is satisfied, the launch returns cleanly. Only a value check finds it. This is
the single most expensive item in this report.

**Workaround.** Use the backend order. There is no way to write the documented
order correctly.

**Suggested fix.** Make the Python signature match what the backend consumes,
or reorder at the binding. Either way the two must not disagree — and until
they agree, the documentation should carry the backend order.

### 2. A masked `store_align(dist=INTLV_B32)` ignores its predicate

**Symptom.** With an interleaved store distribution, the mask argument has no
effect: every lane is written, including the ones the predicate excludes.

**Impact — silent.** The store succeeds. The data is wrong only in the lanes
the mask existed to protect, which is precisely where a tail or a boundary
lives, so small shapes pass and the failure surfaces on the cases that matter.

**Related semantics, worth stating in the same place because authors conflate
them:** masked operations on this target are **ZEROING**, not merging — an
inactive lane receives zero rather than retaining the destination's prior
value. Code written against merge semantics is wrong in the inactive lanes.

**Workaround.** Do not rely on a predicate with `INTLV_B32`. Mask the data
before the store, or store unmasked into scratch and combine explicitly.

**Suggested fix.** Honour the predicate, or reject a masked interleaved store
at parse time. Accepting and ignoring it is the worst of the three options.

### 3. `auto_mutex` is a tile mutex, not a GM coherence mechanism

**Symptom.** A multi-pass kernel that round-trips intermediate data through GM
reads stale values in pass `k` that pass `k−1` had already stored.

**Evidence.** Pass `k−1`'s MTE3 store and pass `k`'s MTE2 re-read land on
**different rotation slots**, so no dependency is visible to the framework and
no barrier is emitted. Measured: ~100 of 20000 elements stale, whole rows at a
time, and **a different row on each run** — a race, not an indexing bug.

**Impact — silent and non-deterministic.** The corruption rate (0.5% here) is
low enough to pass a tolerance-based gate on a lucky draw and to look like a
precision problem rather than a synchronization one.

**Workaround.** Bring the store and the re-read into the **same tile**,
degrading to a single slot, so the mutex chain orders them.

**Suggested fix.** The name is the problem as much as the behaviour: authors
read `auto_mutex` as "the framework handles ordering". State in the
documentation that it orders accesses **within one `mutex_id`** and nothing
else, and consider a diagnostic when a GM range is stored and re-read across
slots within one kernel.

*(This is the same root cause as the aliasing case already recorded as entry 16
of `pypto-pro-framework-findings.md`, reached from the opposite direction.)*

### 4. An FP16 scalar-Tensor store is not safe at a 32-byte ownership boundary

**Symptom.** Concurrent blocks writing scalar FP16 results on a 32 B ownership
granularity corrupt each other's neighbouring elements.

**Evidence.** Measured accuracy 0.90 overall, 0.90625 at `block_dim=32`, on a
32 B boundary. **64 B and 128 B boundaries were exact.** A whole-tile
`pl.store` was exact at **all three** boundaries. Probe: 60 cases,
`block_dim ∈ {2,4,8,16,32}` × boundary ∈ {32, 64, 128 B} × scalar and tile
stores × 2 repetitions.

**Impact — silent, and shape-dependent.** A 10% error rate reads as a numerical
problem in the kernel body rather than as a write-granularity problem.

**Workaround.** Use **beat-complete whole-tile stores** (exact at every
boundary tested), or raise scalar ownership granularity to **64 B**.

**Suggested fix.** Document the minimum safe store granularity per dtype, and
reject or warn on a sub-granular concurrent scalar store.

---

## Tier 2 — loud, but the diagnostic points away from the cause

### 5. `vf.addc` is unreachable from Python, and the documented example injects a wrong carry

**Symptom.** `no matching function for call to 'vaddcs'`. The carry operand must
be a `vector_bool`, but the parser types it as `RegTensor<uint32_t>`.

**Compounding defect.** The documentation's own example passes
`vf.create_mask(pattern=ALL)` as `carry_src` — which sets a carry-in of **1 in
every lane**. So the example is not merely unhelpful, it is a correctness bug
that an author copying it inherits *after* working around the compile error.

**Workaround.** `vf.full(1, m_carry, …)`, relying on mask-is-ZEROING to place
the carry.

**Suggested fix.** Type the carry operand correctly at the binding, and correct
the example.

### 6. An int32 working tile must be declared `UINT32`

**Symptom.** A scatter against an `INT32`-declared tile does not compile. The
parser coerces the index register to the tile's dtype and emits
`vscatter(..., (RegTensor<int32_t>&)off, ...)`; the s32 overload exists only for
a u32 index.

**Impact.** Loud, but the error names an intrinsic overload rather than the
declaration that caused it.

**Workaround.** Declare the tile `DT_UINT32` and carry signed data as a bit
pattern (see also entry 9 — `pl.Ptr` does no dtype checking, so the
reinterpretation is free).

**Suggested fix.** Type the index register `uint32` independently of the
payload dtype. *This is the same root cause as entry 14 of the earlier report;
it is listed again only because the scatter side needs the UINT32
**declaration**, which the gather-side workaround does not make obvious.*

### 7. Index primitives bind index width to data width, and b16 has no construction path

**Symptom.** The NORM tables in `scatter.md` / `gather.md`: b16 data accepts
**only** UINT16 indices; INT32/UINT32/FP32 take UINT32; INT64/UINT64 take
UINT32 or UINT64.

**Why it is a limitation rather than a rule.** A b16 source needs UINT16
offsets, and **no `vf` operation constructs them**: `vf.muls` has no 16-bit
row, and `vf.astype` has no b32→u16 narrowing row. The required index type is
therefore unreachable, forcing the entire pipeline to 32 bits and doubling
index traffic.

**Workaround.** Run the whole indexed pipeline at 32 bits.

**Suggested fix.** Decouple index width from payload width, or supply a
b32→u16 narrowing path.

### 8. `vf.lt`'s `cmp_dtype` is a width selector, not a signedness selector

**Symptom.** Read as a signedness switch, it produces **every element wrong**.

**Evidence.** `lt.md:35`, whose own example compares UINT16 data at UINT8
width, establishes that the parameter names the comparison *width*.

**Impact.** Loud in effect (total failure), but the API name invites the wrong
reading and there is no unsigned-compare selector at all.

**Workaround.** Compose unsigned comparisons from `ge` / `le`.

**Suggested fix.** Rename to `cmp_width`, or provide an explicit signedness
parameter.

### 9. `pl.Ptr` parameters are not dtype-checked

**Symptom / capability.** A `pl.Ptr` formal accepts any dtype without
complaint.

**Impact.** Recorded here as a **latent hazard with a useful side effect**: it
removes a class of type safety, but it also means bit reinterpretation belongs
**inside** the kernel and the host wrapper needs no `.view()` — which matters
because host-side tensor ops are measured time and a compatibility risk on the
eval runner.

**Suggested fix.** If the laxity is intentional, document it as the supported
reinterpretation mechanism; if not, check it.

### 10. There is no b64 vector register on this target

**Symptom.** Signed INT64 `vf.scatter` does not exist. The `vlds` candidate
list enumerates s8/u8/s16/u16/s32/u32/**u64**/bf16/f16/f32/f8\*/f4\* — **no
s64**.

**The uint64 fallback is blocked twice over.** `launch`'s dtype check rejects
an int64 tensor against `pl.DT_UINT64`; and
`torch.zeros(dtype=torch.uint32/uint64, device=npu)` fails inside `zero_`
(`ZerosLikeKernelNpuOpApi.cpp:26`, error 161002), so a uint64 **device** tensor
cannot be constructed in the first place.

**Scope.** An observation about this installation (CANN 9.2.0, Ascend950PR,
this overlay), not a claim about the ISA family.

**Workaround.** Two 32-bit words via `vf.interleave`.

**Suggested fix.** The second blocker is a torch_npu gap independent of the
DSL and is worth reporting separately — as it stands there is no way to
*materialize* the type the dtype check would accept.

---

## Tier 3 — documentation defects that cause false "unsupported" verdicts

These cost the most when an automated workflow uses the docs as a capability
oracle: each one makes a working construct look impossible, and the workflow
then redesigns around a limit that does not exist.

### 11. `vf.astype`'s dtype table reads as a whitelist and is not one

`astype.md` lists four rows (FP32→FP16, FP32→INT32, FP16→FP32, INT32→FP32) and
**mentions BF16 nowhere**. The silicon does FP32↔BF16 and UINT16→UINT32 — both
exercised by the official sample
`pro_ops/lightning_indexer/test_quant_lightning_indexer_vf.py` at `:149-150`
and `:193-196`. Judging from the table alone rules out the entire bf16 family
incorrectly.

**Suggested fix.** Complete the table, or label it explicitly as a
non-exhaustive sample.

### 12. `CastLayout.ZERO` / `ONE` select even / odd lanes, not low / high halves

Established by running both readings side by side: the interleaved
interpretation is bit-exact, while "two NORM stores 64 elements apart" returns
every other element. The official sample corroborates it **in its own variable
names** — `test_quant_lightning_indexer_vf.py:193-196` assigns the
`CastLayout.ZERO` result to `c0_even` and the `CastLayout.ONE` result to
`c0_odd` (verified against the installed sample). The enum names invite the
half-register reading; the code that uses them does not.

**Suggested fix.** Document the lane mapping next to `vf.astype`, or rename to
`EVEN` / `ODD`.

### 13. `vf.copy` is undocumented and absent from every official sample

It exists and works — a predicated register move — but has no documentation
page and appears in none of the 13 official samples, so depending on it is
forward-compatibility risk taken unknowingly. `vf.select(x, x, m)` or an
identity `vf.add` against a zero register substitute.

**Suggested fix.** Document it or remove it.

### 14. The exponential and reduction dtype tables are narrower than expected

`vf.exp` covers **FP16 and FP32 only**. `vf.exp_sub` has two rows
(FP16|FP16→FP32, FP32|FP32→FP32) — **no BF16 row in either**. And `vf.reduce_*`
is a **same-type** reduction (`src == dst`; "源与目标数据类型需保持一致"), so
there is **no narrow-input/wide-accumulator form** — a reduction operator must
widen explicitly before the reduce.

**Suggested fix.** A narrow-in/wide-accumulate reduction is the single most
useful addition here; it is what every normalization operator needs and
currently pays an extra UB pass for.

### 15. Structural ceilings worth stating in one place

- Codegen supports **at most a two-dimensional Tile** (`TileType.md`).
- `mutex_ids` must lie in **`[0, 31]`** and be mutually distinct — **32 buffer
  slots** is the hard ceiling.

Neither is hard to work within; both are currently discoverable only by hitting
them.

### 16. The single-kernel rule and multi-dtype support are not in tension

Not a defect — a documented construct that is easy to miss, recorded because
missing it drives authors to a much worse design.
`@pl.jit(tiling_key=…, datatype={'x': 'io_dtype'})` gives one source, several
compiled specializations, and **one launch**. `kernel_function.md:117` defines
`datatype` as "数据类型特化，用于同一 Kernel 支持多种数据类型"; `:85` describes
`tiling_key` as a launch-time dictionary selection compiling one specialized
kernel per mode. The official sample
`pro_ops/fa/test_fa_perf_tkv_preload_dn_vf_bufid_dynrank.py` carries exactly one
`@pl.jit` (`:497-505`, launched outside the loop at `:812`).

**Suggested fix.** Cross-reference this from the multi-dtype discussion; the
fallback authors reach for instead is *N* separate `@pl.jit`s selected by a
host-side dictionary, which is more code and only arguably satisfies a
single-kernel requirement.

### 17. A constant array index cannot be produced by any in-kernel loop

Two separately reasonable restrictions compose into one that is not obvious
until both are hit.

`_control_flow_parser.py:102` sets `_VALID_ITERATORS = {"range"}`, so `pl.range`
is the only accepted loop iterator — a bare Python `range()` is rejected with
`ParserSyntaxError F00002: For loop must use pl.range()`. And `pl.range` is a
**runtime** loop, whose index is therefore never a literal. Separately, indexing
a `tiling` array is a **frontend parse rejection** unless the index is a literal
integer (`test_tc_05_array_index_syntax.py`: "literal integer index is valid…
non-int index are rejected").

Together: **a table lookup over a runtime selector cannot be written as a loop
at all** — it must be unrolled in the source text, one literal line per entry.
Measured on a 128-entry selector, the working form is a runtime `pl.range` loop
with the selector unrolled inside it:

```python
for g in pl.range(0, e):                    # runtime loop — legal
    sel = 0
    if g == 0:   sel = tiling.group_ends[0] # 128 literal lines, constant
    if g == 1:   sel = tiling.group_ends[1] # indices, runtime `if`
    …
```

Scalar assignment under a runtime `if`, read after the `if`, works. The
phi-free arithmetic alternative `start += tiling.group_starts[i] * (g == i)`
does **not**: `ValueError: Operator 'mul' does not accept bool dtype`, so a
scalar comparison cannot be multiplied.

**Suggested fix.** Either accept a compile-time-constant loop construct whose
index can index a `tiling` array, or state the unroll requirement in the array
documentation. The failure today is discovered one rejection at a time, and the
diagnostic for the first one ("must use pl.range()") points at the loop rather
than at the index that motivated it.

---

## Measured and found safe — recorded so it is not re-litigated

### Cross-AIV sharing of one 32-byte beat, under a whole-tile store

Three operators in one session designed *around* the possibility that two AIVs
writing parts of the same 32-byte beat might lose or tear a write — by padding
a workspace, by forcing boundaries to stay aligned, or by collapsing to a
single-AIV epilogue. None of them measured it, because the earlier P0 probe
that swept ownership boundaries had only ever placed those boundaries **on**
32/64/128 B, never straddling.

It has now been measured, with the control proved live in the same run:

- control (one row deliberately dropped): **6240 mismatches**, the required
  count, first at `[33,0] got=-1.0 exp=200.0` — the sentinel showing through
  where the missing row should be, i.e. the checker is demonstrably sensitive
  to exactly the defect class at issue;
- hazard case (`m_off=1, N=65`, so AIV1's first row starts at byte 4290 and
  shares block `[4288,4320)` with AIV0's last byte): **0 mismatches over 100
  launches**;
- aligned control (`m_off=0`): 0, as expected and uninformative — see below.

So a whole-tile `pl.store` is safe across a shared beat. The conservative
designs remain correct; they are simply stricter than the hardware requires.

**The geometry trap that makes this easy to get wrong.** AIV1's first row
starts at `(m_off + 32)·N·esize`. At `m_off = 0` that is a multiple of 32
unconditionally, so **a probe built at `m_off = 0` cannot expose the hazard at
all** — a green result there says the construction was invalid, not that the
path is safe. Reaching the hazard needs an unaligned `m_off`, which occurs
whenever the offset comes from a group boundary rather than a tile multiple.

### 18. The evaluation server runs a newer `pypto_pro` than any inspectable local build, and it renamed a `@pl.jit` kwarg

A submission that passes every local and on-board check can still score zero,
because the graded environment is a generation ahead of the one you can read.

Measured: a package whose kernel is decorated `@pl.jit(auto_mutex=True,
tiling_key=…, timeout=600, name=…)` died on the server at **decoration time,
during import**, before any tiling, any case, or any comparison:

```
AI算子执行失败: jit() got unexpected keyword argument(s): timeout.
  cann_bench/test_<op>.py:572  @pl.jit(..., timeout=600, ...)
  pypto_pro/runtime/jit.py:1309: TypeError
```

Ten local copies of `pypto_pro` were inspected: **all accept `timeout`, none
defines `compile_timeout`, and none performs a strict keyword check**. A second
package submitted in the same batch, decorated with `compile_timeout=300`,
compiled and scored on that same server. So the newer generation **renamed
`timeout` → `compile_timeout` and added strict keyword validation** that the
older generation lacks.

Two consequences worth stating separately:

- **Passing on the board does not imply passing on the server.** The older
  generation silently swallows an unknown keyword, so the local run cannot
  surface the mismatch at all — the check that would catch it does not exist
  locally.
- **The failure mode is total.** It is an import-time `TypeError`, so the whole
  20-case set reports `elapsed_us = 0.00` with `compile=0.0` and
  `function=0.0`. That is *unmeasured*, not "scored 0", and it should be
  reported as such — no case was ever compared.

**Before spending a credit**, grep every implementation file in the package for
`timeout=` inside a `@pl.jit` decorator. In one campaign this found the defect
in a second, unrelated package that would have failed identically.

**Suggested fix upstream.** Either keep `timeout` as a deprecated alias, or —
since strict validation is the right call — make the local and graded builds
reachable from one version string. Today `pypto_pro.__version__` does not
exist, the overlay and the CANN 9.2.0 copy are byte-identical while the
conda-env copy differs, and only a content hash distinguishes them; there is no
way for an author to tell which generation will grade the submission.

---

### 19. Intra-core event ids 8–15 pass frontend validation and fail at bisheng

The frontend accepts twice as many event ids as the backend can emit.

`ir/op/system_ops.py:37` sets `_MAX_EVENT_ID = 16`, so `sync_src`/`sync_dst`
with `event_id` in 8–15 validate and lower without complaint. The CCE converter
that turns them into C++ documents a narrower contract —
`codegen/cce/type_converter.h:82-92`: *"Maps event ID (0-7) … `event_id` The
event ID (must be in range [0, 7])"* — and no `EVENT_ID8`..`EVENT_ID15` symbol
exists anywhere in the tree. The result is a compile failure at bisheng
(`EVENT_ID8` / `EVENT_ID10` undeclared), which is a long way from the line that
chose the id, and reads as a toolchain problem rather than a range error.

Two details that matter for diagnosis:

- **Only intra-core `sync_src`/`sync_dst` lower to `EVENT_ID<n>` literals.**
  Cross-core ids emit through a different path and are unaffected by the
  ceiling, so a kernel can legitimately use high cross-core ids while an
  intra-core 8 fails — which makes "the id is too large" a non-obvious
  hypothesis when both appear in the same file.
- The failure is a **hard compile stop, not a silent miscompile**, so it costs
  a board round-trip but cannot corrupt results.

**Suggested fix.** Set `_MAX_EVENT_ID = 8` for the intra-core path, or have the
frontend reject at the call site with the backend's own bound in the message.
The information needed for a clear diagnostic already exists in
`type_converter.h`'s docstring; it just is not enforced where the id is chosen.

---

### 20. `pl.store` addresses from the declared shape and silently ignores `make_tensor` strides

Measured on board, not inferred. A `make_tensor` view built with a permuted
stride array is written to as though it were contiguous: the store derives its
addresses from the declared shape alone.

The diagnosis is worth keeping because the *pattern of failures* identified the
mechanism where the failure count could not. Of six cases, the three that failed
were exactly the BSND cases with `S > 1` and non-zero data; BNSD passed, `S == 1`
passed, and an all-zeros case passed. Each exception follows from the same rule:
BNSD's permuted strides happen to equal the contiguous ones, `S == 1` never
reaches the row stride, and a permuted view of zeros is still zeros. The
hypothesis was written down and checked against all six outcomes before any code
changed, and it predicted all six.

**A silently mis-addressed view is worse than a rejected one**, and the
documentation advertises transposed/strided views as a feature while warning only
vaguely about backend stride support.

**Workaround that works:** collapse the non-row axes so the row pitch lives in
the *declared shape* — a 3-D contiguous tensor, no custom strides, no `order=`.

**What an offline check cannot catch:** a bijection/coverage proof of the index
arithmetic verifies the arithmetic you intended. It cannot see that the backend
never reads the stride array. Sweeping every `make_tensor` view in the operator
afterwards (11 views × 2 layouts) found all 22 contiguous-consistent, so no
sibling defect survived — that sweep is the check that generalises.

### 21. The axis-order kwarg is renamed wholesale per source tree — `order=` and `tile_dims=` never coexist

> **Corrected 2026-08-08.** This entry first claimed an asymmetry *within* one
> installation — that `pl.load` accepted `order=` while `pl.store` rejected it.
> **That asymmetry does not exist in any tree**, and acting on it would send an
> author looking for a per-op difference that is not there. What is real is a
> per-tree rename. The original observation is preserved at the end as
> unresolved.

Measured across three trees. Each registers **one** hook for **all four** of
`load` / `load_tile` / `store` / `store_tile`, so the spelling is a property of
the tree, never of the op:

| tree | kwarg | hook | `@pl.jit` timeout kwarg | basis |
|---|---|---|---|---|
| board `.41` `cann-9.2.0` | `order=` | `_resolve_order_kwarg` | `compile_timeout` | measured on device |
| local `pypto` | `order=` | `_resolve_order_kwarg` (`block_ops.py:1353-1356`) | `timeout` | source read |
| local `old/pypto_pro_9688` | `tile_dims=` | `_resolve_tile_dims_kwarg` (`:1240-1243`) | `timeout` | source read |

`_ir_store` carries an `order` parameter in the new tree and none in the old
one, and the pre-hook is what translates the surface kwarg — so passing the
*other* tree's spelling reaches the builder untranslated and dies as
`_ir_store() got an unexpected keyword argument`.

Two consequences that cost real time here:

- **A file mixing the two spellings compiles nowhere.** One operator arrived
  with `pl.load(order=)` and `pl.store(tile_dims=)`, plus a source comment
  asserting the board registered different hooks for the two ops. Neither the
  file nor the comment corresponded to any tree.
- **The timeout kwarg tracks the same axis** (§18), so `order=` + `timeout=`
  and `tile_dims=` + `compile_timeout=` are both cross-generation mixtures.

**The portable spelling is to pass neither.** For a rank-3 tensor with a rank-2
tile, `[1,2]` *is* the computed default (`range(ndim_t - ndim_tile, ndim_t)`) in
every tree, and every tree drops the kwarg from the emitted IR when it equals
the default — the IR is byte-identical. Omission is the only spelling valid
across all three.

**Still unresolved, recorded rather than smoothed over.** The `mla` run on the
primary board reported `TypeError: _ir_store() got an unexpected keyword
argument 'order'` while also reporting `order` present in
`inspect.signature(pl.store)`. Under the wholesale-rename model those two
cannot both hold, and the same board later ran `grouped_matmul` with
`order=[1,2]` and `compile_timeout=900` through nine passing tests. The primary
board went unreachable before this could be re-measured. Treat the mla
observation as unexplained — not as evidence for a per-op asymmetry.

### 21.1 A widening `vf.astype` takes a mask of the **source** width

Distinct defect, same operator family, and the failure is silent.

A b16→FP32 widening cast was given `p_even`/`p_odd`, both
`dtype=pl.DT_FP32` — the *destination* width. `astype.md` states the rule:
*"MaskReg 根据输入的源操作数进行有效元素筛选"*. With a destination-width mask
under `layout=ONE`, the cast selects nothing and returns zeros.

It surfaced as a precision failure with an arithmetic fingerprint rather than
as an error: `MERE = 0.4980467` at N=257, against `128/257 = 0.4980544` — every
**odd** output column unscaled. A standalone probe separated it cleanly:

```
remaining=128 b16mask ZERO: MATCH      remaining=37 b16mask ZERO: MATCH
remaining=128 b16mask ONE : MATCH      remaining=37 b16mask ONE : MATCH
remaining=128 b32mask ZERO: MATCH      remaining=37 b32mask ZERO: MATCH
remaining=128 b32mask ONE : MISMATCH   got[:4]=[0.0, 0.0, 0.0, 0.0]
```

Official code confirms the rule in both directions:
`pro_ops/lightning_indexer/test_quant_lightning_indexer_vf.py:193-196` widens
UINT16→UINT32 with `preg_b16`; `:149-150` narrows FP32→BF16 with `preg_b32`.

**This is API misuse, not a framework limit**, and worth stating because the
design had pre-registered this exact path as a `capability_gap` candidate. The
probe was run, the prescribed path works, and only the mask width was wrong —
**no capability gap**. The general form: when an op's mask semantics are keyed
to one operand, a mask of the other width fails *quietly* under `ONE` and
correctly under `ZERO`, so a ZERO-only test cannot see it.

— *while* `order` was present in `inspect.signature(pl.store).parameters` on the
same installation. `pl.load` accepted it; `pl.store` did not.

**The generation matters and makes this sharper, not weaker.** In the local
source tree `ir/op/block_ops.py:207` shows `_ir_store` accepting `order`, as does
`_ir_store_tile` at `:298`. So the board's overlay and the inspectable local
build disagree about this kwarg — the same generation skew as §18, now visible in
a second place and in the opposite direction. The public wrapper advertised a
parameter its own IR builder rejected *in that installation*.

**Consequence for a technique this project relies on.** Signature inspection is
used as a runtime capability probe (for example, to select an AIV block-index
ratio). That technique is sound only where the wrapper and builder are known to
agree; it silently over-reports otherwise. Probe by *calling* inside a
throwaway trace, or keep the probe to parameters already exercised by a working
kernel on the same installation.

---

### 22. A `vf` comparison mis-dispatches to the scalar mnemonic when its 2nd operand is loop-carried

> **Rewritten 2026-08-08.** This entry first blamed a CANN 9.1.0 / 9.2.0 version
> gap and concluded "a board pass does not predict that it compiles." **Both
> halves were wrong for this defect.** It reproduces byte-for-byte on 9.2.0, so
> the board predicts it perfectly. What actually happened is recorded in
> [investigation-discipline.md](investigation-discipline.md) §15: the file was
> submitted without ever being run. The version difference is real (§18, §21)
> but was a red herring here.

`mla` scored **0/20, every case `compile_runtime_error`**:

```
kernel.cpp:749: error: no matching function for call to 'vcmps_gt'
    vcmps_gt(__inline_1_is_new_max_0, __inline_1_score_reg_0,
             __inline_1_max_reg_iter_1, __inline_1_visible_0);
note: no known conversion from 'RegTensor<float>' to 'float' for 3rd argument
      __VF_VCMPS(f32, float)
```

**The mechanism.** The header declares two disjoint families, `LT` being a scalar
C type:

```c
vcmp_##OP (vector_bool &dst, vector_##T  src1, vector_##T src2, vector_bool mask)
vcmps_##OP(vector_bool &dst, vector_##ST src1, LT         src2, vector_bool mask)
```

`gt.md` documents the dispatch as automatic: *"第二个参数可以是标量或
RegTensor，接口自动识别并分发……标量比较走 vcmps_gt，向量比较走 vcmp_gt"*. It
is correct for a scalar and for a plain register. It **breaks on a loop-carried
register**, which codegen emits as an `auto` alias:

```c
740:  auto __inline_1_max_reg_iter_1 = __inline_1_max_reg_0;
748:  vcmps_ge(..., _expr_tmp_78_0, ...);             // scalar   -> correct
749:  vcmps_gt(..., __inline_1_max_reg_iter_1, ...);  // REGISTER -> BROKEN
750:  vsel    (..., __inline_1_max_reg_iter_1, ...);  // same reg -> correct
751:  vsub    (..., __inline_1_max_reg_iter_1, ...);  // same reg -> correct
```

Only the comparison mis-dispatches, **because only it has two mnemonics to
choose between** — `vsel` and `vsub` have no scalar sibling, so there is nothing
to get wrong. That is the whole shape of the bug, and it is a framework defect
worth filing.

**dtype is not the discriminator, and neither is literal-vs-identifier.** Two
plausible rules were tested and both fail:

- *"float32 only"* — the header defines `__VF_VCMP(s32)` **and**
  `__VF_VCMPS(s32,int32_t)`, so every dtype has both families and could hit it.
- *"a literal 2nd argument is safe, an identifier is not"* — disproved by probe.
  9.2.0 never folds `vf.full(0.0)` to a scalar; it emits `vdup` into a register
  and then `vcmp_gt` against it. And mla's own legal line 748 has an
  **identifier** 3rd argument that happens to be a C scalar. The rule yields
  false negatives and false positives.

**The discriminator that works** is the *declared kind* of the 2nd source
operand, which the generated C++ states outright: a real
`RegTensor<T> name;` declaration dispatches correctly, an `auto name_iter_N =
…` alias does not. A scanner over the emitted `kernel.cpp` that builds a symbol
table and flags family/kind disagreement localises **every** site at once, where
a compile reports only the first error in the first failing key. Validated in one
direction — it flags the site the compiler rejected, one RED at exactly line 749
against the compiler's "1 error generated". One instance is not a catalogue, so
lead with the compile and let the scan corroborate.
(`custom/mla/_debug/scan_vcmps_operands.py`.)

**The fix, and why the obvious one is wrong.** Replacing `gt` + `select` with
`vf.max(score_reg, max_reg, visible)` looks cleaner and is **silently
incorrect**: `gt.md` states masked lanes are zeroed, `max.md` documents no
merging mode, and the generated arithmetic confirms `MODE_ZEROING` — so it zeroes
the running max on causally-masked lanes, and the downstream `alpha` lines run
under `preg` rather than `visible`, carrying the corruption into the running sum.
Wrong on exactly the causal cases.

What works is an algebraic rewrite that makes the 2nd operand a fresh register:

```python
score_minus_max = vf.sub(score_reg, max_reg, visible)
is_new_max      = vf.gt(score_minus_max, zero_reg, visible)
next_max        = vf.select(score_reg, max_reg, is_new_max)
```

`a > b ⟺ a − b > 0`, ties false in both forms; on a masked lane `vf.sub` zeroes
the difference, `0 > 0` is false, and `next_max` keeps `max_reg` exactly as
before. Numerically safe **because the sentinel is finite** (`-1.0e30`), so
`score − sentinel` cannot produce `inf − inf`. Measured 7/7 on board with the
scanner green across all keys.

**Where a real operand is genuinely loop-carried there is no rewrite** — the
value has to be copied into a fresh register first, or the comparison
restructured so the carried value is the *first* operand.

---

## CANN Bench harness defects (different upstream owner)

Not DSL issues, but found in the same session and reportable.

### H1. `trace_view` is structurally unreachable on the isolated-subprocess path

`pypto_pro_child.py` hardcodes `profiler_level` to `ProfilerLevel.Level1`, while
`trace_view` requires **Level2** (`bench_registry.py:64`). The
`--profiler-level` flag is plumbed into `perf_eval.py` and
`mc2_distributed_runner.py` only — **it never reaches the isolated child**.

**Consequence.** `elapsed_us = 0.0`, `score_error_code:
'no_npu_kernel_detected'`, and the composite score zeroed by anti-cheat — on a
kernel that ran correctly.

**Inference, flagged as such:** the evaluation server runs the same unpatched
child, so it should be affected identically. That part is reasoning from the
shared source, not a server-side measurement.

### H2. Device binding is broken for a bare-metal single-card host

`evaluator.py:1102` writes the **physical** chip id into
`ASCEND_RT_VISIBLE_DEVICES` and passes **the same integer** to `set_device()`.
But `ASCEND_RT_VISIBLE_DEVICES` names an absolute physical chip *and*
renumbers the visible set from 0 — so the two agree **only at 0**.
`--device-id 4` therefore raises `RuntimeError 107001`, and passing 0 to work
around it is overwritten back to physical npu0. Invisible on k8s, where the
device plugin has already renumbered the allocated card to physical 0.

**Workaround.** Keep an inherited value that is already a single chip
(`ASCEND_RT_VISIBLE_DEVICES=4` with `--device-id 0`), and **prove in the log**
that work landed on the intended physical card — an HBM fingerprint or the
`npu-smi` process table.

### H3. `--source-dir` force-reinstalls into the shared environment

It runs `pip install --force-reinstall --no-deps`
(`package_manager.py:474-508`), which **evicts the previously installed
operator's submission** from a shared conda environment. Alternating
evaluations of two operators silently measure whichever was installed last.

---

## Index of the earlier four-operator report

Titles only. Full text, measurements and citations are in
[`docs/pypto-pro-dsl-limitations.md`](../../../../docs/pypto-pro-dsl-limitations.md).

| # | Title |
|---|---|
| 1 | No TensorList parameter type |
| 2 | No 64-bit integer division; scalars are int32-only in practice |
| 3 | Mask-width conversion exists but is undiscoverable |
| 4 | `vf.load_align`'s "align" means register width (256 B); violation is a bare device fault |
| 5 | `vf.load_unalign` crashes the host process natively |
| 6 | The unaligned load entry points sit ~1400 lines from the aligned ones |
| 7 | The parser cannot resolve body-assigned names in `make_tile_group` address lists |
| 8 | The parser cannot follow Python helper calls inside a `@pl.jit` body |
| 9 | `vf.*` is three-address only — no expression nesting |
| 10 | `pl.cast`'s default rounding disagrees with torch, and the `CAST_RINT` docstring is wrong |
| 11 | `vf.astype` between register widths needs an undeclared `CastLayout` |
| 12 | The build cache is keyed on the kernel's `co_name` |
| 13 | `@pl.jit` reads source off disk via `inspect.getsource` |
| 14 | `vf.gather`/`vf.scatter` type the index register from the *data* dtype |
| 15 | A UB tile row must be a whole number of 32-byte blocks |
| 16 | `@pl.jit` parameter names are emitted verbatim into generated C++ |
| 17 | UB memory ordering inside a vector function is undocumented in two places |
| 18 | `vf.update_mask` is issue-expensive, and placement dominates |
| 19 | A module-level `float("inf")` renders as the undeclared C++ identifier `inff` |
| 20 | `vf.astype` narrowing has no round-to-nearest-even mode on 950PR |

## Consolidated upstream asks

The four from the earlier report stand unchanged (register-level lane shift;
document `load_align`'s alignment requirement; fix or document the
`load_unalign` host crash; co-locate the unaligned entry points), plus the RNE
narrowing ask added by entry 20. From this session, in decreasing order of
value:

1. **Make `mrgsort2`'s signature agree with its backend** (Tier 1 #1). A silent
   wrong answer behind a correct-looking call is the worst failure mode a DSL
   can have.
2. **Honour or reject the predicate on an interleaved `store_align`** (#2) —
   accepting and ignoring it is strictly worse than either alternative.
3. **State `auto_mutex`'s actual scope in its documentation** (#3), and consider
   diagnosing a cross-slot GM store/re-read.
4. **Decouple index register width from payload dtype** (#6, #7) — this is the
   earlier report's ask #14 restated, now with a case (b16) that has *no*
   workaround other than abandoning the narrow pipeline.
5. **Complete or explicitly de-normativize the `vf.astype` dtype table** (#11),
   and fix the `vf.addc` example (#5). Both are one-line documentation changes
   that today cause working constructs to be judged impossible.
6. **Add a narrow-input/wide-accumulator reduction** (#14).
