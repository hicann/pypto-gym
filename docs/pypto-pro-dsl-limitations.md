# PyPTO-Pro DSL — consolidated limitations and improvement requests (A5 / Ascend950PR)

> **Canonical source for Part I.** The KB page
> `cannbot-skills/ops/pypto-pro-op-kb/references/pypto-pro-dsl-limitations-a5.md` carries the
> same Part I entries for agent routing. **Edit Part I here first**, then mirror it there;
> `check_kb_integrity.py`'s `report copies agree` compares the two entry-title sets and
> fails on drift. The two once disagreed about whether a defect was explained, and about
> an entry's own title, with nothing to catch either.

**For the PyPTO-Pro framework team.** This is the single consolidated report for
two independent campaigns that together built eleven operators in the
`pl` / `vf` tile DSL. It supersedes nothing: both source reports
are reproduced here **verbatim**, because re-expressing a measured entry is how
citation fidelity drifts. What is new in this document is the front matter, the
severity index, and the consolidated asks in Part III.

## Provenance

| | |
|---|---|
| Hardware | `Ascend950PR_9579` (28 cube / 56 vector) |
| Toolchain | CANN 9.2.0, bisheng/clang 15.0.5 |
| Operators | `foreach_addcdiv_scalar`, `swi_glu`, `apply_rotary_pos_emb`, `cummin`, `gather`, `scatter`, `softmax`, `top_k`, `quant_matmul`, `grouped_matmul`, `mla` |
| Basis | Every entry was established by an on-board probe or by reading the installed source tree. **None is inferred from documentation alone.** Where a claim is a property of *this installation* rather than of the DSL in general, the entry says so. |

Entry counts: **Part I** carries 29 numbered entries (plus sub-entry 21.1),
**Part II** carries 28 — **57 distinct limitations**, every one measured.

**Part II no longer has a separate source file.** It used to be maintained as
`docs/pypto-pro-dsl-limitations.md` and copied here, which left the two able to
disagree — and they did: that file carried entries 25, 27 and 28, and a longer
upstream-asks list, that this report never received. The two have now been
merged and this document *is* the four-operator report; edit Part II here.
Part I is still copied from
`cannbot-skills/ops/pypto-pro-op-kb/references/pypto-pro-dsl-limitations-a5.md`,
which remains a living document, so Part I stays a snapshot of it.

The assembler this report used to be regenerated with is no longer checked into
this repo, so updates are applied by hand, following the same rule the assembler
enforced — copy the missing entries verbatim from the source, do not paraphrase.
(An earlier snapshot of this report recorded that Part II contained two entries
both numbered 21 and that its title index listed only 20; that numbering has
since been repaired upstream, and the note is withdrawn. Two later snapshots
undercounted: #29 in Part I and #26 in Part II were folded in earlier, and the
merge described above added #25, #27 and #28 — the count stood at 58 while the
sources held 60.)

## How severity is ordered

Severity here is dominated by one question: **does the defect announce itself?**
A compile error costs an hour; a silent wrong answer costs a release. Part I is
ordered on that basis, and its Tier 1 entries all produce correct-looking output
from incorrect code with no error at any layer.

### The silent class, across both reports

These are the entries where every signal available to the author says the code is
correct. They are the reason this report exists.

| Entry | Defect | Why it is silent |
|---|---|---|
| I-1 | `mrgsort2`'s parameter order disagrees between API and backend | Declaration, docs, compiler and launch all agree; only a value check finds it |
| I-2 | A masked `store_align(dist=INTLV_B32)` ignores its predicate | Wrong only in the lanes the mask existed to protect — i.e. tails and boundaries |
| I-3 | `auto_mutex` is a tile mutex, not a GM coherence mechanism | ~0.5% of elements stale, a different row each run; reads as precision, not sync |
| I-4 | FP16 scalar-Tensor store unsafe at a 32-byte ownership boundary | 10% error rate reads as a numerical bug in the kernel body |
| I-20 | `pl.store` addresses from the declared shape, silently ignoring `make_tensor` strides | A strided view is advertised as a feature and is written as though contiguous |
| I-22 | A `vf` comparison mis-dispatches when its 2nd operand is loop-carried | Validates and compiles; the wrong mnemonic is emitted |
| I-23 | One `pl.get_block_idx()` call can acquire two runtime values | Build-dependent, traced to a 64-column offset |
| II-3 | An fp16-width mask fed to an int32 `select` reads the wrong bits | Values correct, **indices garbage** |
| II-10 | `pl.cast`'s default rounding disagrees with torch | Off-by-one-ulp results that pass loose tolerances |

### The hang class

| Entry | Defect |
|---|---|
| I-26 | A zero-iteration `pl.range` carrying a rendezvous deadlocks the AI Core — no error, no timeout, no diagnostic |
| II-5 | `vf.load_unalign` crashes the host process natively |
| II-4 | `vf.load_align` alignment violation is a bare device fault |

### One defect found twice, independently

**I-18 and II-21 are the same defect.** Two rounds, months apart, each lost
builds to `@pl.jit`'s `timeout` keyword being rejected by the deployed
server's newer `pypto_pro`, which renamed it `compile_timeout` and added strict
keyword validation the local builds lack. The first campaign spent three
diagnostic builds localising it; the second hit it again because nothing in
either environment exposes the version difference. That independent
rediscovery is itself the argument for ask #3 in Part III.

---

# Part I — A5 severity-ordered report (29 entries)

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
pattern (see also entry I-9 — `pl.Ptr` does no dtype checking, so the
reinterpretation is free).

**Suggested fix.** Type the index register `uint32` independently of the
payload dtype. *This is the same root cause as entry II-14 of the
four-operator report; it is listed again only because the scatter side
needs the UINT32
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
deployed runtime.

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

### 18. The deployed runtime is a newer `pypto_pro` than any inspectable local build, and it renamed a `@pl.jit` kwarg

A package that passes every local and on-board check can still fail outright,
because the graded environment is a generation ahead of the one you can read.

Measured: a package whose kernel is decorated `@pl.jit(auto_mutex=True,
tiling_key=…, timeout=600, name=…)` died on the server at **decoration time,
during import**, before any tiling, any case, or any comparison:

```
AI算子执行失败: jit() got unexpected keyword argument(s): timeout.
  <package>/test_<op>.py:572  @pl.jit(..., timeout=600, ...)
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

**Before submitting**, grep every implementation file in the package for
`timeout=` inside a `@pl.jit` decorator. In one campaign this found the defect
in a second, unrelated package that would have failed identically.

**Suggested fix upstream.** Either keep `timeout` as a deprecated alias, or —
since strict validation is the right call — make the local and graded builds
reachable from one version string. Today `pypto_pro.__version__` does not
exist, the overlay and the CANN 9.2.0 copy are byte-identical while the
conda-env copy differs, and only a content hash distinguishes them; there is no
way for an author to tell which generation will run the package.

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

### 21. A 2026-07-27 runtime boundary splits three behaviours at once — and installations sit on both sides

> **Third revision.** I first recorded a per-op asymmetry (`pl.load` accepts
> `order=`, `pl.store` does not), then "corrected" it to a wholesale per-tree
> rename after finding one hook for all four ops in two local trees. **The
> correction was wrong.** The asymmetry is real on at least one installation, it
> was documented in an operator's own source two weeks before I recorded either
> version, and my local-tree evidence was about trees that are not the boards.

**One runtime change, three coupled behaviours.** From an operator's in-source
note, written 2026-07-27 and corroborated by everything measured since:

> *The 2026-07-27 runtime changed AIV `get_block_idx()` from logical AIC id to
> `aic_id * subblockdim + subblock_id` and added `get_subblock_num()`. The old
> runtime also exposes `store(tile_dims=...)`, while the new one uses `order=`.*

So §23's AIV index semantics, the `get_subblock_num` op, and the
`tile_dims=`/`order=` spelling are **not three independent findings** — they are
one generation boundary. That is why chasing them separately kept producing
partial models.

**Observed matrix. Record per installation; do not infer the row you have not
measured.**

| installation | `pl.load(order=)` | `pl.store(order=)` | `get_subblock_num` | AIV `get_block_idx()` |
|---|---|---|---|---|
| board A | **accepts** | **rejects** (`_ir_store()` unexpected kwarg) | **absent** | logical AIC id |
| board B | accepts | accepts | present | `aic*dim+sub` |
| local tree (current) | accepts | accepts | — | — |
| local tree (`old/…_9688`) | `tile_dims=` | `tile_dims=` | — | — |

Board A is the case that killed the wholesale-rename model: **load and store
differ there**, exactly as first reported. The local trees do share one hook
across all four ops — that observation was correct and simply did not describe
either board.

**`inspect.signature` is not a capability test, and here is the sharpest form of
it.** On board A, `order` **is** in `inspect.signature(pl.store).parameters` and
`_ir_store()` still raises on it at parse time. The same operator's source states
this outright and warns against using its own runtime probe for the wrong
question:

> *CAVEAT: this probe is only trustworthy for choosing the block-index ratio. It
> is NOT a capability test for the keyword itself.*

**The portable idiom, which is the actionable part.** Rather than call an op that
may be absent, compute the divisor on the host and pass it through tiling:

```python
# host, once:
_NEW_AIV_BLOCK_INDEX = "order" in inspect.signature(pl.store).parameters
...
vector_task_ration = 2 if _NEW_AIV_BLOCK_INDEX else 1     # -> tiling field

# kernel:
vector_core_id = pl.get_block_idx() // tiling.vector_task_ration
```

This is why that operator compiles and runs on the deployed runtime while a
sibling using `pl.get_subblock_num()` directly does not compile on the old-runtime
board at all: **the source never names an op absent from either registry**, and
the generation difference becomes a host-side constant.

The probe is a *proxy* — it tests the `order=` half of the boundary to infer the
index-semantics half. Legitimate only because the two travel together in that one
runtime change, and worth stating as such rather than as a capability query.

**Where the official samples sit.** `pro_ops` uses `pl.get_subblock_num()` in
10+ places (matmul ×4, fa ×4, lightning_indexer), and there is **no
`get_subblock_num.md`** among the three system-variable API pages. So the samples
the workflow designates as the writing authority are written against the new
runtime, while the API documentation and at least one installed board are on the
old one. **Copying a sample verbatim is therefore not sufficient for
portability** — which is the trap that produced a `ParserSyntaxError` reading
`Operator 'block.get_subblock_num' not found in registry`.

**Suggested fix.** Either back-port `get_subblock_num` (it is derivable from
existing exports), or document the boundary and give the samples a portability
note. Failing that, publish which runtime is deployed, since
correctness depends on it and nothing in-band reveals it.

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
> [investigation-discipline.md](../cannbot-skills/ops/pypto-pro-op-kb/references/investigation-discipline.md) §14.2: the file was
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
lead with the compile and let the scan corroborate. The scanner itself is not
retained here; the description above is enough to rebuild it.

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

### 23. One `pl.get_block_idx()` call can acquire two runtime values — and whether it does is build-dependent

**Root cause identified and traced end to end.** This was first recorded as
"unresolved, deliberately not promoted"; it is now measured, with the wrong
output localised to the cell.

The operator calls `pl.get_block_idx()` **once, at kernel top level**, before any
`with pl.section_*()`. Two installations compile that one call two ways:

**Installation A** — split into per-section functions, the call duplicated into
each and lowered per section:

```c
  17: grouped_matmul_..._impl_cube(...)
 102:     auto block_0 = (int32_t)(get_block_idx());
1325: grouped_matmul_..._impl_vector(...)
1445:     auto block_0 = (int32_t)(get_block_idx() * get_subblockdim() + get_subblockid());
```

**Installation B** — one fused function, computed once in the shared prologue,
**outside** the vector guard:

```c
  12: grouped_matmul_..._impl(...)
 115:     auto block_0 = (int32_t)(get_block_idx());     // <- before the guard
 116:     #if defined(__DAV_VEC__)
 117:     auto subblock_0 = (int32_t)(get_subblockid());
```

The lowering rule itself is unconditional —
`backend/backend_cce_ops.cpp:947-955` branches on `ir::SectionKind` alone, reading
nothing about grid width or core count. What differs between installations is
**whether the call gets duplicated into the sections at all**. So the author wrote
one call outside any section and, on installation A, one Python variable holds two
different runtime values on the two core types.

**The wrong output, traced to the cell.** In a case with `m_tiles == 1`, the
owner arithmetic makes `n_off` a direct function of `block`, so `block` differing
by 1 shifts the store by exactly one `n_tile` — 64 columns here:

| writer | `block` | `t` | stores at | cube of that core computed |
|---|---|---|---|---|
| AIV0 of b=0 | 0 | 0 | `n_off=0` | 0 ✓ |
| **AIV0 of b=1** | **2** | **0** | **`n_off=0`** | **64** ✗ |
| AIV1 of b=0 | 1 | 1 | `n_off=64` | 0 ✗ |
| AIV1 of b=1 | 3 | 1 | `n_off=64` | 64 ✓ |

All three observed signatures fall out of that table: the column-64 value
appearing at column 0; column 64 reading `torch.empty` garbage (~1.2e-07) because
only AIV1s reach it and they own different rows; and **the nondeterminism**, because
`[0..31, 0]` has *two* writers and which lands last is a race. Two
identically-configured runs disagreed for exactly this reason.

**How this was settled matters as much as the answer.** Two hypotheses were
tested and killed by measurement before this one survived — that a board was
simply broken for multi-block work (a second operator passed 43/43 on it), and
that a launch grid narrower than the device was to blame (the failing tests'
`total` binds the width, not the cap, so both cap values gave identical grids and
identical failures). The evidence that finally landed it was **the two generated
`kernel.cpp` files side by side**, not a runtime readout: a probe written to
report the runtime values died in its own `pl.load`, and the emitted source
answered the question more reliably anyway.

**Suggested fix.** Lower a top-level `get_block_idx()` once for the whole kernel,
as installation B does; or, if per-section lowering is intended, give the
vector-side form a distinct name so "index of my block" cannot silently become
"index of my subblock within the grid." Either way the two installations must not
disagree about a call the author placed outside every section.

### 24. Evidence rules for two-board measurement（原记的未解缺陷已定位，见文内）

The multi-block precision failures once recorded here are **explained**: they are
§23, on the installation that duplicates the call. What remains is a second,
independent defect and a set of rules that cost real time to learn.

**Two defects, cleanly separated:**

| | trigger | scope | status |
|---|---|---|---|
| **A** — owner-index divergence | `nc > 1` (erased by `% 1` at `nc == 1`) | one installation's build | **§23, root cause identified** |
| **B** — device deadlock, `aicore timeout` `507014` | ≥2 tasks per core | **both boards, both builds** | **已定位**：核停在 pipe/buffer 资源获取上（见下） |

Defect B survived five ablations, two of which were then re-confirmed on the
second board: even `k_outer` with ≥2 tasks/core still hangs, and deleting the
guarded cube-side `wait_cross_core` entirely still hangs. It is **not** §23 — the
board whose build computes `block` once, with no divergence at all, deadlocks
identically. Every core of both types parks at a fixed PC, which is a structural
stall rather than anything data-dependent.

**这条已经不再是未解之谜。** 把两个卡住的 PC 解析回源码一步就定了性——它停在
**pipe/buffer 资源获取**上。上面五轮消融是在解析 PC *之前*做的：遇到
`aicore timeout 507014` 先解析 PC，不要重跑这五轮。KB 侧的同一条目
（`cannbot-skills/ops/pypto-pro-op-kb/references/pypto-pro-dsl-limitations-a5.md` §24）
与本节结论一致，并链到同步约束页。

**Rules that hold regardless of cause:**

- **A multi-block result names a board *and* a build.** Two installations
  compiled identical source into materially different code — one fused function
  versus two per-section functions — and the deadlock's stuck PC differed by 24
  bytes between them. An ablation verdict is a statement about one binary.
- **Capture interpreter, library path and `core_num` per run, not per campaign.**
  One board lost `import pypto_pro` mid-session when a third party rebuilt a
  wheel; the only importable copy afterwards was a different build than the one
  that had produced the earlier results.
- **Generated source can be better evidence than a runtime probe.** The question
  "do these two sections see the same index?" was answered by diffing two emitted
  `kernel.cpp` files. The probe written to answer it at runtime failed in its own
  kernel and was correctly abandoned rather than repaired.
- **Derive launch width from `get_platform_info().core_num`, never a literal** —
  worth doing on its own merits, though it was *not* the cause here.

### 25. The same IR compiles to materially different device code across CANN generations

Quantified, with a control. One generated `kernel.cpp`, two bisheng builds, the
40 compile flags and 4 link args taken from pypto's own `compile_config` rather
than guessed:

| | CANN 9.2.0 (built 2026-07-21) | CANN 9.1.0-beta.3 (built 2026-05-20) |
|---|---|---|
| compile | exit 0 | exit 0 |
| `.aicore_binary` | 17,088 B | **21,048 B (+23 %)** |
| host `.text` | 4,002 B | 4,082 B |
| host mnemonic histogram | 7 `testb`, 2 `testq` | **identical** |

Both toolchains report `clang 15.0.5 (clang-5c68a1cb1231)`; only the build dates
differ. **The control matters**: rebuilding with 9.2.0 from the reconstructed
command line reproduced the stored `.so` byte-for-byte, so the comparison is
between toolchains and not between command lines.

**Read the host diff carefully.** A byte-level `cmp -l` reports 2,747 differing
bytes, and that figure is an artifact: the host section is an 80-byte-larger
launcher stub whose *mnemonic histogram is identical*, so nearly all of it is
positional shift rather than independent change. A raw diff count across two
builds of the same source is worth roughly a tenth of what it appears to be.

**What this does and does not license.** It establishes that device codegen is
generation-dependent and substantially so — the third generation-specific
observation in this project, after the axis-order kwarg (§21) and the comparison
mnemonic (§22), and the first with a magnitude attached. It does **not**
establish that the difference is numerics-relevant: FMA contraction,
reassociation and fast-math questions live in the `.aicore_binary` instruction
stream, and no available tool disassembles it (see §24's note on the same gap).
That is not inferred from the size delta.

**Practical consequence.** "It compiles and runs correctly here" is a statement
about one toolchain build, not about the operator. Where a defect appears only on
the grading environment and has been refuted on the local one, generation-
dependent codegen is a live explanation rather than a last resort — but closing it
requires that generation's `pypto_pro`, which a bare CANN install does not carry
(all three 9.1.0 trees on the shared board have the compiler and no `pypto_pro`).

---

### 26. A zero-iteration `pl.range` loop that carries a rendezvous hangs the AI Core

Clamping a data-dependent loop bound to zero is not a safe way to express "this
work item has nothing to do." Where the loop body performs a per-iteration
rendezvous, a zero-iteration producer skips the rendezvous while its consumer
still waits, and the launch hangs — no error, no timeout, no diagnostic.

Measured on `mla` (Ascend950PR_9579, CANN 9.2.0, npu3), eleven single-case
processes under one lock, with a health case passing before and after each hang
so the card is exonerated:

| shape | dtype | causal | result |
|---|---|---|---|
| `S=192, S_kv=128` | bf16 | yes | **hang** (two independent processes) |
| `S=192, S_kv=128` | fp16 | yes | **hang** — dtype exonerated |
| `S=192, S_kv=128` | bf16 | **no** | pass — differs from the hang by one bit |
| `S=512, S_kv=128` | fp16 | yes | **hang** |
| `S=129, S_kv=128` | — | yes | pass |

The log ends at `kernel launching in eager mode` and emits nothing for 420 s.
The non-causal control is the load-bearing comparison: same shape, same dtype,
one flag apart, and it passes — so this is the bound, not the shape.

**The near-miss worth recording.** The first form of the bound could go
*negative* (`min(128, 0+64+128-512) = -320` at `S=512, S_kv=128`), which was
found by review and fixed by clamping at zero:

```python
kv_lim = kv_len - pl.min(kv_len, query_len - q_off - q_valid)   # clamped at 0
```

The clamp is correct and insufficient — it removed the negative bound and left
the zero-iteration one, which is the case that actually hangs. A fix for a
range-underflow is not automatically a fix for the empty-range semantics.

**Regression, not a pre-existing gap.** The same code path previously ran the
full loop with every lane masked, which is well defined and correct. Bounding
the loop is a real optimisation (geomean 1.114×, up to 1.420×) — but it converted
a slow-and-correct corner into a hang.

**Upstream ask.** Either define `pl.range(0, 0)` as skipping the body *and* its
rendezvous consistently for every participant, or diagnose a loop whose
iteration count is data-dependent and whose body contains a cross-participant
handshake. Silently deadlocking is the worst of the three options. Same family
as the G0 hang and §22's dispatch defect: a construct that validates, compiles,
and then misbehaves at runtime. See `constraints/sync-stitch.md`.

**Practical rule for kernel authors.** When a bound can reach zero, floor the
loop at one all-masked iteration rather than letting it empty — the masked pass
costs one tile and preserves the handshake count the consumer expects.

**The floor's threshold is tile-size dependent, and one contract case already sits
on it.** The zero-trip condition is `S >= S_kv + QTile`, so it moves with the
query tile: `+64` at `QTile=64`, `+128` at `QTile=128`. In `mla`'s contract set,
**case 4 (`Sq=256, Skv=128`) sits exactly at `S = S_kv + 128`** — past the
`QTile=64` threshold — and does not hang only because it carries
`is_causal=False`. That is one flag of luck, not margin. Any later change that
raises `QTile`, or that reaches the causal branch for a non-causal shape, walks
straight back into the hang. Two consequences: the floor constant and the
threshold comment must be re-derived whenever the tile size changes rather than
inherited, and a "no in-contract shape can reach it" argument must be re-checked
per tile size — it is not a property of the contract alone.

---

### 27. A `> 0` guard is a NaN sink, and nine investigations could not see it

Not a DSL defect — a kernel-authoring trap that the DSL's comparison semantics
make easy and that no available diagnostic catches. Recorded here because the
cost was nine investigations.

`mla` failed 10 of 80 evaluated shapes with `NaN位置不匹配`, identically across builds
that differed by a complete decode-path rewrite. The sink was one predicate:

```python
positive   = vf.gt(denom_reg, zero_reg, preg)      # NaN > 0 is FALSE
safe_denom = vf.select(denom_reg, one_reg, positive)
out_reg    = vf.select(quot_reg, zero_reg, positive)   # -> exact 0.0
```

Written for one condition (`den == 0`, a fully-masked row, where writing 0 is
correct and matches the reference's own guard), it is false for **three**:
`den < 0`, `den == 0`, and `den` is NaN. A NaN denominator took the
fully-masked branch and the NaN was replaced by an exact zero.

**The fix is three lines** — invert to an *ordered* comparison so NaN falls
through the divide instead of being selected away:

```python
masked     = vf.le(denom_reg, zero_reg, preg)
safe_denom = vf.select(one_reg, denom_reg, masked)
out_reg    = vf.select(zero_reg, quot_reg, masked)
```

Measured: 48 of 112 CPU configs mismatching → 0, board `X2_nan` 65536 → 0,
device time geomean 1.0000x, instruction count unchanged.

**Why it survived nine investigations.** Every one searched for something that
*produced* a NaN — a write-coverage model over 1872 shapes, two allocation-poison
board sweeps (82 and 108 runs), a sentinel analysis, an int32 audit, a bisheng
diff. A poison sweep cannot find this by construction: it proves every output
element is *written*, and this element was written, with a wrong finite value.
**Establish the direction of a mask mismatch before choosing a mechanism** —
"ours has a NaN theirs doesn't" and "theirs has a NaN ours doesn't" are disjoint
bug classes, and the error string does not distinguish them.

Two measured semantics worth having on record, from a standalone VF probe rather
than from documentation (`vf/gt.md` states no NaN behaviour):
`le(NaN, 0) -> 0`, `gt(NaN, 0) -> 0`, `div(2, NaN) -> NaN`, `ne(NaN, NaN) -> 1`.
So `vf.eq(NaN, 0)` is *also* false — swapping `gt` for `eq` does not fix this,
and the `safe_denom` substitution has to be reworked in the same edit.

**A trap inside the probe that verifies it.** The first version of that VF probe
returned eight rows of plausible floats — `sub(0,0)` read `0.907` — which were
uninitialised `torch.empty`: `auto_mutex` with plain `make_tile` scratch tiles
inserts **no V->MTE3 rendezvous**, so every store was lost. A probe measuring
NaN semantics must carry a control row whose expected value is known, or it will
confidently report garbage as a measurement.

---

### 28. `phase=` disables the automatic M-FixPipe sync, and `pl.move` cannot honour the pairing it then requires

The framework accepts a combination that **cannot** satisfy its own documented
constraint, and the two ways it then fails look like unrelated bugs.

`phase` is not a hint. Per `phase.md` it is a **hardware `unit_flag` handshake
between the Cube M pipe and the FixPipe**: `matmul(phase=Final)` sets the flag to
1 and `store(phase=Final)` reads and clears it. Two rules follow:

- **配置了 phase 则不自动插入同步** — passing `phase` *disables* the framework's
  automatic M↔FixPipe software synchronization.
- **Constraint 1, 配对使用** — if the `matmul` carries `phase`, the corresponding
  `store` / `store_tile` must carry it too.

**The unsatisfiable case.** Write an accumulator with `phase=Final` and drain it
with `pl.move(..., acc_to_vec_mode=DualModeSplitM)` — and **`pl.move` has no
`phase` parameter at all** (`move.md:26`; the word does not appear on that page).
The write half arms the protocol and the read half structurally cannot
participate. Nothing rejects this at parse time or at compile time.

**Both failure modes are documented in `phase.md` and they are not alike:**

| doc case | mechanism | observed symptom |
|---|---|---|
| 案例三 | flag stuck at 1; the next `matmul` on the same L0C block waits forever | **hard hang, no error, no timeout** |
| 案例二 | the read is not gated on matmul completion, so it reads 未完成的数据 | fixpipe reads unwritten L0C → **multi-bit ECC, `error code 171`**, and a driver-level card `Alarm` via the RAS path |

Which one appears depends on whether the stalled party is the writer or the
reader, and therefore on accumulator slot count and `d_tiles`. That is why the
same defect produced a silent hang in one cut and a reported fault in another —
and why "a missing handshake hangs rather than faults" is the wrong heuristic.

**The buffer-depth red herring this explains.** A single-slot accumulator hangs
while two slots run, so depth looks causal. It is not: **a slot buys one
iteration, it does not clear the flag.** Any cut whose loop re-enters a dirty
block will hang at any depth. One kernel spent multiple rounds on depth 1→2
before this was understood, and a KB page recorded the single-slot hang as
unexplained between "auto-mutex ownership, lifetime overlap, or another backend
rule" — it was the `unit_flag`.

**Passing tests do not clear a kernel of this.** Two diagnostic cuts carrying the
identical violation passed, purely because their shape gave `kv_tiles = 1` against
a 2-slot accumulator, so no iteration ever re-entered a dirty block. The
prediction that follows is falsifiable and worth running before trusting any
`phase` + `pl.move` kernel: **the same cut at `kv_len >= 384` should hang.**

**The fix is a deletion.** Remove `phase=` from a matmul whose accumulator is
drained by `pl.move`, and let the framework insert the synchronization — this is
`phase.md`'s own correct-usage example. The cost is the pipeline overlap `phase`
exists to buy, which is unmeasured here and matters when the change was motivated
by performance in the first place.

**Authoring rule worth stating in a design checklist:** `phase=` is legal only
when the drain is `pl.store` / `store_tile`. A kernel that drains an accumulator
to the vector unit must not use it.

**Suggested fix upstream.** Reject `phase=` at parse time when the accumulator's
drain cannot carry it, or give `pl.move` a `phase` parameter. Silently accepting a
protocol only one side can speak is what turns a rule violation into a
multi-week misattribution: the failure surfaces at an `Acc→Vec` boundary, so the
`Acc→Vec` primitive gets blamed. A design document in this repo recorded
`DualModeSplitM` as a framework wall in five places on exactly that reasoning —
the primitive was never broken.

### 29. The axis-order kwarg is renamed wholesale per source tree — `order=` and `tile_dims=` never coexist

> **Superseded in part by §21 — read that entry first.** This entry once
> claimed a per-op asymmetry, was then "corrected" to deny it, and §21's third
> revision reinstates it: on board A `pl.load` accepts `order=` while
> `pl.store` rejects it, documented in an operator's own source before either
> version was written here. The denial was based on local trees that are not
> the boards.
>
> **Both facts hold, and they are about different things.** The per-tree
> rename below is real and is what the measured trees show; the per-op
> asymmetry is real on at least one *board*. Neither generalises to the other.
> Record the row for the installation in front of you and do not infer the one
> you have not measured — §21 carries the observed matrix.

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



---
---

# Part II — Four-operator report (28 entries)

From building four operators end-to-end in the `pl` / `vf` tile DSL:
`foreach_addcdiv_scalar`, `swi_glu`, `apply_rotary_pos_emb` and `cummin`, on
Ascend950PR_9579 / CANN 9.2.0 / bisheng-clang 15.0.5. Later entries (#25 and up)
come from the `mla_prolog` campaign against the same DSL. Every entry states what
was observed, how it was established and what it cost, and ends with the fix that
would have prevented it.

Citations of the form `[FF #N tree:lines]` point into
`FRAMEWORK_FINDINGS.md` on the named operator branch (the
per-operator working trees; entries #22 and up are numbered per-session there and
collide, so entries here are identified by title). Links into
`cannbot-skills/ops/pypto-pro-op-kb/` point at the knowledge-base pages that carry
the full workaround.

One earlier claim is deliberately **not** carried: "mask granularity is not
convertible" was retracted by
the same branch's later finding — see entry II-3, which supersedes it.

## Limitations

**1. No TensorList parameter type.** `runtime/jit.py:354-370` recognises
`TENSOR` / `PTR` / `TILING` / `SCALAR` only, argument arity must match the
declaration exactly, and `ptr.make_ptr` requires a `PtrType` — so a runtime
value (a `data_ptr()` read from a tensor) can never become a pointer, and a
Python list of tensors cannot be passed at all. This requires fixed unrolled
parameters: historical bucket probes reached 385 parameters at arity 64, while
delivery uses one finite maximum-arity signature and pads shorter lists. The host-loop
alternative measured **6.46x** device time at `L = 64`. Fix: a real
list-of-tensors parameter kind, or allow `pl.Ptr` construction from a runtime
address. Full pattern:
[`cannbot-skills/ops/pypto-pro-op-kb/patterns/vec-tensorlist-fixed-arity.md`](../cannbot-skills/ops/pypto-pro-op-kb/patterns/vec-tensorlist-fixed-arity.md).

**2. No 64-bit integer division; scalars are int32-only in practice.**
`Div of bitwidth greater than 32 not supported` — and the failure is
conditional on type inference (for example a `pl.min` result feeding a
divisor widens it), so a latent instance compiles until something unrelated
perturbs the types. Workaround: restructure the loop nest to keep divisor
arithmetic in int32. Fix: lower 64-bit division, or diagnose it at parse time
instead of deep in codegen. [FF #15 cummin:364-386]

**3. Mask-width conversion exists but is undiscoverable.** (Supersedes the
retracted "not convertible" claim.) `vf.interleave` / `vf.de_interleave` with
`dtype=` naming the *finer* width re-space a `MaskReg`'s read stride — the
conversion is one instruction, measured correct on 128/128 lanes with
`DT_UINT16` and wrong on 32/128 with `DT_UINT32`. Two operators independently
audited every function with `mask` in its name and concluded conversion was
impossible; the converter is filed under advanced computation. The real
symptom stands: an fp16-width mask (1 bit per 16-bit lane) fed to an int32
`select` reads the wrong bits — values correct, indices garbage. Converting
did not pay on the measured operator (the narrow-dtype penalty was a
tile-pitch bank conflict, found separately), and the residual cost of the
fp32-widening workaround is two extra UB passes. Fix: name or alias a
mask-width conversion (`vf.mask_astype`), and document the MaskReg read-stride
model next to `vf.select`. Full analysis:
[`cannbot-skills/ops/pypto-pro-op-kb/constraints/vec-mask-width.md`](../cannbot-skills/ops/pypto-pro-op-kb/constraints/vec-mask-width.md).
[cummin FF #28 :750-804; #25 :621-660]

**4. `vf.load_align`'s "align" means register width (256 B); violation is a
bare device fault.** No sub-register lane shift is expressible:
`store_align` + `load_align` at an offset of −k lanes faults (507035) even
when every shift is 32-byte aligned and a `vf.mem_bar` sits between the store
and the load. `vf.shift_left/right` are bit shifts within lanes, not lane
moves. Measured per 64-lane register: gather ~20 ns, scatter ~18 ns,
barriered UB round trip ~16 ns, versus under 1 ns for an aligned load/store
and ~0.3 ns for arithmetic — every way of moving data across lanes costs
15–20 ns, every way of not moving it costs under 1. Log-depth scans die on
this (measured floor 1.537 ns/element/core versus 0.942 shipped). Fix: a
vslide-style register lane shift; a compile-time alignment diagnostic instead
of the device fault. [cummin FF #30 :828-916]

**5. `vf.load_unalign` crashes the host process natively.** The three-call
protocol (`load_unalign_init` / `load_unalign_pre` / `load_unalign`) brings
the host process down with pages of addr2line noise and no Python exception.
Fix or document. [cummin FF :901-909]

**6. The unaligned load entry points sit ~1400 lines from the aligned ones**
(`_vf_api.py:1565-1589` versus `:134`). Two readers independently concluded
no unaligned load exists. Move them adjacent or cross-reference. [cummin FF
:860-867]

**7. The parser cannot resolve body-assigned names in `make_tile_group`
address lists** (`name 'C' is not defined`), so tile geometry must be
module-level literals. Fix: constant-fold body-level integer arithmetic, or
issue a diagnostic that names the restriction. [cummin DESIGN.md §6.5]

**8. The parser does not *execute* Python helper calls inside a `@pl.jit` body.**
The body is parsed as an AST, never executed, so a helper call that would have
produced valid arguments at runtime surfaces instead as a pybind "incompatible
constructor arguments" error far from the actual mistake. Fix: diagnose the
unresolvable call at parse time. [FF #6 :127-146]

> **Boundary, because entries II-8 and II-22 were read as contradicting each
> other.** Three different things are in play:
> 1. a **module-level helper** *is* resolved — it is **inlined and traced** into
>    the body, subject to the single-final-`return` rule (II-22);
> 2. it is **not** host Python that runs once at trace time, so ordinary control
>    flow inside it is a parse error rather than a computation (II-22);
> 3. a **general call** — builtins, comprehensions, arbitrary callables — is not
>    available at all (`F00005: Unsupported function call`).
>
> "The parser cannot follow helper calls" means (2)+(3), never (1). A helper that
> inlines cleanly is a supported spelling.

**9. `vf.*` is three-address only — no expression nesting.** Every producing
call must use the assignment form ("vf.reduce_sum produces a result and must
use the assignment form", F00002). Rediscovered independently in two
operators because the samples obey it silently, which reads as style rather
than a rule. Fix: document as a language rule, or auto-flatten nested
expressions. [FF #11 :243-260]

**10. `pl.cast`'s default rounding disagrees with torch, and the `CAST_RINT`
docstring is wrong.** The default `CAST_ROUND` rounds half away from zero;
torch rounds half to even, producing a 1-ulp scatter (measured 0.4% of bf16
elements, 0.05% of fp16). `CAST_RINT`'s docstring describes something else,
but it *behaves* half-to-even — which is why it matches torch: switching to
it took MERE and MARE to 0 everywhere. Fix: default to RINT or document the
divergence; fix the docstring. [FF #14; arpe FF #23 :536-571]

**11. `vf.astype` between register widths needs an undeclared `CastLayout`,
and `CAST_RINT` lowers to an undeclared enum.** The layout default disagrees
with where `load_align` actually placed the narrow elements, producing wrong
results rather than an error; and `pl.VFRoundMode.CAST_RINT` lowers to
`ROUND_N`, which the CANN 9.2.0 bisheng headers do not declare — a compile
error naming an identifier the user never wrote. Fix: document the layout
contract next to `vf.astype`; ship headers that declare what the DSL emits.
[cummin FF :789-802; DESIGN §6.4]

**12. The build cache is keyed on the kernel's `co_name`.** An edited body
compiled under an unchanged function name can be served the previous binary,
so the run reproduces **byte-identically** — indistinguishable from "my
change was inert" after the fact. Fix: key the cache on a hash of the
rendered source; error on one name with two bodies. Mitigation until then:
stamp a source-hash into every kernel name. [FF #17 :387-418]

**13. `@pl.jit` reads source off disk via `inspect.getsource`.** Kernels
cannot be built from `python -c`, `exec`, or any in-process generation —
launch fails with `OSError: could not get source code`. Fix: accept a source
string, or register generated source with `linecache`. [FF #5 :107-126]

**14. `vf.gather` / `vf.scatter` type the index register from the *data*
dtype, while the underlying intrinsics take `vector_u32` regardless.** Fatal
for int32 payloads ("no known conversion ... vector_u32" at C++ compile).
Workarounds, all measured working: keep payloads in UINT32 tiles and use an
order-preserving XOR `0x80000000` key (compare biased, select raw); widen
int8 through int16 with a **truncating** `vf.pack` (the saturating cast
corrupts −1 to 127); emit int64 as two UINT32 words via `vf.interleave`. Two
zero-scoring cases went to 0.593 / 0.687 with these. Fix: type the index
register `uint32` independent of the payload dtype. [FF #21 :479-495; #26
:662-692; DESIGN §6.1]

**15. A UB tile row must be a whole number of 32-byte blocks**
(`pto_tile.hpp:1444`), so an odd *element* pitch is inexpressible — and an
odd pitch is THE anti-bank-conflict tool for strided gathers. Measured:
gather stride 64 fp32 = 51.0 µs versus stride 65 = 7.0 µs (~5–7x for any
power-of-two stride). Under the 32-byte-block rule the only legal
conflict-free pitches are odd multiples of 32 bytes, and no single pitch
serves both a 4-byte and a 2-byte tile (the sets `8·odd` and `16·odd`
elements are disjoint) — forcing a dual-pitch layout bridged by `pl.cast`.
The wrong pitch held six cases at ~6 GB/s/job against 26.5 achievable. Fix:
allow an arbitrary declared pitch (padding internally), or surface the legal
pitch set in the API. [cummin FF #27 :693-749]

**16. `@pl.jit` parameter names are emitted verbatim into generated C++.** A
tensor parameter named `half` breaks every fp16 kernel — the generated
`(half*)` casts in `call_kernel.cpp` become `expected expression` errors that
are dtype-conditional and point away from the cause. Same class: `float`,
`double`, `int`, `short`, `long`, `signed`, `bool`, `min`, `max`, `data`.
Fix: reject C++ reserved and clashing identifiers at parse time, or mangle
parameter names (`p_` prefix). [swiglu FF #22 :496-535]

**17. UB memory ordering inside a vector function is undocumented in two
places.** (a) Back-to-back `vf.scatter` calls to the same UB address are not
ordered **across calls** (the documented rule covers only within one call):
measured 3/256 mismatches without `vf.mem_bar()` and 0/256 with, at ~1/1000
on realistic draws — passes a small sample, fails a large one. (b) A UB store
followed by a load of the same address needs
`vf.mem_bar(mode=pl.MemBarMode.VST_VLD)` or the vector core faults. Fix:
document the cross-call ordering model; auto-insert or diagnose the missing
barrier. [FF #10 :208-242; :875-882]

**18. `vf.update_mask` is issue-expensive, placement dominates, and the
framework's own reference generator had it wrong.** Hoisting the mask out of
the per-register loop recovered 550.75 → 398.06 µs (the DMA floor) on a 67M
index fill; applied to the reference generator's exp kernels it moved mean
SOL 0.300 → 0.961 across 20/20 cases. Two caveats from the same lever on
another operator: the tail loop must be its **own pass** over rows (nesting
it cost 1.65x on empty tails), and one hoisted variant
(`rope_bcast_interleaved_float32`) **faults the device and poisons the NPU
context** — later cases fail in `copy_between_host_and_device_opapi`; root
cause unestablished. Fix: compiler hoist of loop-invariant `update_mask`;
triage the fault. [cummin FF #29 :805-827; arpe FF #25 :621-659]

**19. A module-level Python `float("inf")` constant renders as the undeclared
C++ identifier `inff`.** A kernel that consumes a module-level
`float("inf")` in a `vf`/`vdup` context generates `kernel.cpp` containing a
bare `inff`, and bisheng fails with `use of undeclared identifier 'inff'`
(kernel.cpp:87; cummin `cummin_agg_float32`, 2026-08-06) — the renderer
appears to append the fp32 literal suffix `f` to the Python repr `inf`
instead of emitting a real C++ spelling for the value. Until fixed, treat
any non-finite module-level float constant referenced inside a jit body as a
compile risk and check the generated `kernel.cpp`. Fix: emit `INFINITY` /
`std::numeric_limits<float>::infinity()` for non-finite float literals.

**20. `vf.astype` narrowing has no round-to-nearest-even mode on 950PR —
`CAST_RINT` fails to compile on both available toolchains.** On CANN 9.2.0
bisheng the enum lowers to the undeclared identifier `ROUND_N`; on CANN
9.1.0-beta.3 the same kernel hits the intrinsic's own static_assert, whose
text settles the question: *"The 4th argument of this vcvt (f322bf16) can
only be: ROUND_R, ROUND_A, ROUND_F, ROUND_C, ROUND_Z"* — the f32→bf16 (and
f32→f16, probed separately) vcvt simply has no RNE variant to lower to.
This bounds the earlier tile-op finding ("`pl.cast` `CAST_RINT` behaves
half-to-even, use it when the golden is torch"): that remains true on the
tile-op lowering path, but a **vf-path kernel cannot narrow with RNE at
all**. The only viable vf narrowing is `CAST_ROUND` (half-away); measured
against a torch-RNE golden on rms_norm this cost 1 ulp on a sub-percent of
bf16/fp16 elements — MARE 7.812e-3 against a 7.81e-2 gate, a 10× margin.
Keep the round mode a single-point switch so a future toolchain that gains
RNE needs a one-line change. (rms_norm, dual-toolchain compile probes on
Ascend950PR_9579, 2026-08-06.)

**21. `pl.jit` keyword arguments are not portable across pypto_pro builds,
and rejecting one kills the run with no attributable error.** Our
development box accepts `@pl.jit(auto_mutex=True, timeout=600)` (the timeout
raises the compile budget on large unrolls); the deployed runtime's
pypto_pro raises `TypeError: jit() got unexpected keyword argument(s):
timeout` at `runtime/jit.py:1309`. Because a package normally imports its
kernel module at package import time, the exception fires inside the
evaluator's own import and surfaces only as a stage-level
`staged_rc_1_missing_report` — a successful wheel build followed by nothing,
on every target. It cost three diagnostic builds to localise, and was
only readable after making the forwarder import the kernel module lazily so
the failure landed inside a case. Fix, in order of preference: keep the
accepted-kwarg set stable across builds, or ignore-with-warning an unknown
kwarg rather than raising, or at minimum name the accepted set in the error.
Consumers: ship only the kwargs the target runtime is known to accept, grep
the emitted module before packaging, and prefer a lazy import in the
forwarder so a runtime mismatch is reported per case rather than as an
infrastructure failure. (foreach_addcdiv_scalar, 950PR, 2026-08-07.)

**22. A module-level helper called from a `@pl.jit` body is inlined and traced,
and may contain exactly one `return` — as its final top-level statement.**
Factoring the tile-geometry arithmetic into `def _slot_plan(w): ...` fails with
`ParserSyntaxError F00002: Inline function '_slot_plan' may only return from its
final top-level statement` (`_call_parser.py:_validate_inline_returns`). The
helper is not host Python that runs once at trace time; it is inlined into the
traced body, so ordinary Python control flow in it is a parse error rather than
a computation. This refines entry II-8 rather than contradicting it: a module-level helper
**is** resolved, by inlining, so II-8's "cannot follow" means "does not execute",
not "rejects". What is unavailable is the *general* call — builtins,
comprehensions and arbitrary callables all fail with `F00005: Unsupported
function call`, as the `list(...)` rejection on the same operator shows. Four ways to express a width-dependent slot count were probed, and
only one survives: a Python `if` on the compile-time tiling-key value
(`if WidthTemplate == 16384: ...`) is resolved at parse time and compiles.
Rejected: list comprehension (`F00005: Unsupported expression type: ListComp`),
module-level dict subscript (`F00001: Unsupported closure variable type: dict`),
and scalar addresses built with `list(range(...))`. The practical consequence
is that any per-width UB map must be written as an explicit `if`/`elif` ladder
of literal address lists — which also means the arithmetic must be arranged so
each branch's literals are computable by hand (here, choosing a ceil-division
tile height so every role occupies the same byte count at every width). Fix:
either execute module-level helpers as host Python before tracing, or document
that a jit body admits no calls and name the compile-time-branch idiom in the
tiling-key docs. (arndq, 2026-08-08.)

**23. The compile subprocess is hard-bounded at 60 s and the timeout is
uncaught, so on any runtime that rejects `timeout=` a slow kernel simply cannot
be built.** `runtime/jit.py` declares `timeout: int = 60` and hands it straight
to `subprocess.run(..., timeout=timeout)` (line 867); `TimeoutExpired` is caught
nowhere in `runtime/*.py` or `runtime/opc/*.py` — the single `except Exception`
(`kernel.py:207`) is in the parser and re-raises as `ParserSyntaxError`. Raising
the budget is only possible via the `timeout=` kwarg, which entry II-21 shows the
deployed runtime's build rejects outright. The two limitations therefore
compose into a real wall: a kernel whose bisheng invocation needs more than
60 s is unbuildable on that runtime, with no supported way to ask for more.
Fix: make the budget an environment variable or a runtime config rather than a
decorator kwarg, so it is settable where the kwarg is not.

The same reading is a **useful diagnostic**, and it is the reason this entry is
worth keeping: an uncaught 60 s bound means *no* multi-hour stall can be blamed
on compilation. Verified by fault injection on Ascend950PR_9579 — a bisheng
shim that wedges, and a second that `setsid`-forks a grandchild holding the
captured stdout/stderr pipes (the case where a naive `subprocess` wait would
block past its timeout) — both failed **loudly at exactly 60 s with a full
traceback naming the command**. Two adjacent facts from the same probe, both
useful when costing a compile: every kernel logs `Generating kernel '<name>'
for static shape signature <dynamic-only>` when all dims are `pl.DYNAMIC` and
tiling arrives as runtime scalars, so there is **one binary per kernel,
shape-independent** — no input shape can provoke an extra compile; and a warm
re-run with the build directory intact still invokes bisheng and still takes
the same ~0.7 s, i.e. there is **no on-disk artifact reuse** to mistake for a
cache hit. (apply_rotary_pos_emb stall investigation, dual-toolchain
compile probes, 2026-08-09.)

**24. Three ways to describe a strided memory view, and two of them are
silently ignored.** Reading `H` elements per row out of a GM tensor whose row
stride is `D` *is* expressible — it is the ordinary tiled load: declare the
tensor `[rows, D]`, declare the tile with `valid_shape=[-1, -1]`, narrow it with
`pl.set_validshape(t, [m, H])`, and `pl.load(t, q2d, [r0, H])` for the second
window. The row stride comes from the declared tensor shape, emitted as a
dynamic template, so `D` may be a runtime dim; bounds validation rejects only
`offset[i] >= shape[i]`, and `offset + access_shape` exceeding the shape is
explicitly permitted (`ir/op/block_ops.py:103-119`). The official FA sample
reads `TD <= D` per row off a `[B,S,N,D]` tensor, so a non-contiguous GM window
is a first-class pattern. What is dangerous is the two *other* things that look
like they express the same idea:

- **`pl.make_tensor(t, shape, stride)`'s `stride` never reaches the CCE
  `GlobalTensor`.** Codegen delegates to `GenerateGlobalTensorTypeDeclaration`,
  which recomputes row-major strides from the view's *shape*
  (`backend/backend_cce_ops.cpp:972-1010`), so
  `pl.make_tensor(q, [rows, H], [D, 1])` is read as a **contiguous**
  `[rows, H]`: wrong data, no diagnostic. Every doc and sample happens to pass a
  stride already equal to the row-major product, which is consistent with the
  argument being a redundant declaration — and is why the bug is invisible in
  the examples.
- **`pl.set_stride` does reach the runtime stride, but offsets do not follow
  it.** It emits `tensor.SetStride<...>` (`backend_cce_block_out_ops.cpp:723-740`),
  yet `load(tile, x, [i, 0])` still scales `i` by the *declared* shape's
  row-major stride: the ST test `datacopy/test_set_stride.py` yields rows `i` and
  `i + s/LINE` — the pointer moved by `i*LINE` while the DMA walked by `s`. Two
  different strides in one access, neither reported.

Also: a **statically declared** `valid_shape` produces no cache entry, and the
fallback emits the tile's *physical* shape (`ResolveEffectiveTileShape`,
`backend_cce_block_out_ops.cpp:265-280`) — so a narrowed second-window load
would silently be asked for the full declared width and run past the row, and
past the tensor end on the final row. Always `[-1, -1]` plus
`pl.set_validshape`. And a doc conflict to be aware of: `load.md:40` states
`set_validshape` requires `compact=1`, but no such check exists in
`block_ops.py` or `framework/src/interface/ir/op/block_ops/`, and the runnable
`element_wise/test_eltwise_dynamic_rank.py` uses `valid_shape=[-1, -1]` on Vec
tiles with no `compact` at all. Fix: make the ignored `stride` argument either
honoured or a hard error; make `set_stride` scale offsets by the stride it set,
or reject offsets while a custom stride is live; and reconcile `load.md:40`
with the implementation. (apply_rotary_pos_emb padded-path redesign,
2026-08-09.)

**25. The frontend can emit C++ that references an identifier it never declares,
and the failure reaches the user as a bare "Failed to compile kernel".** On the
deployed `pypto_pro` release, `mla_prolog`'s `4part_3x3` kernel
generated a `kernel.cpp` whose `tk_1` (r3 tiling key) body contains:

```
kernel.cpp:6553:35: error: use of undeclared identifier 'row_off_1'; did you mean 'row_off_9'?
    auto row_off_iter_4 = row_off_1;
kernel.cpp:6645:25: error: use of undeclared identifier 'row_off_1'; did you mean 'row_off_9'?
    row_off_9 = row_off_1;
kernel.cpp:6551:17: note: 'row_off_9' declared here
```

Two errors, `rc=1`, at `n_heads=64, rows=256, rt=4`. The emitted file is
internally inconsistent: the reference is to definition `_1` inside a scope where
only `_9` exists. The Python source has **one** local named `row_off`, reassigned
across many nested scopes (337 mentions in the module, e.g. `row_off = mt * TM`
in M1 and `row_off = m2_group * m2_group_rows` in M2, each feeding `pl.load` /
`pl.store` offsets), which is the shape of a per-definition numbering pool that
is shared across scopes and then resolved to the wrong member.

**Two things make this expensive out of proportion to the bug.** First,
`runtime/jit.py`'s `_ensure_compiled` raises
`RuntimeError(f"Failed to compile kernel '{name}'")` and **drops the compiler's
stderr entirely**, so the defect is invisible: five shapes failed for
months-equivalent with no reason attached. Recovering it required shipping a
pass-through shim that wraps `subprocess.run`, records non-zero-rc invocations
and attaches their text to the raised message — which is a workaround for missing
diagnostics, not for the codegen. Second, **it is release-specific and therefore
untestable from a dev box**: the same kernel compiles cleanly under the CANN
9.1.0 *backend* locally, and the deployed `pypto_pro` is a different release —
its traceback raises at `jit.py:1607` and calls `_ensure_compiled` at `:1444`,
while the local copy of that file is 1510 lines with the same two statements at
`:1451` and `:1282`. Line 1607 does not exist locally. So the frontend that emits
the kernel is not the frontend you can test.

Fix, in order of value: **make `_ensure_compiled` include the compiler's stderr in
the exception** — that alone would have turned a blind five-case loss into a
one-run diagnosis; and scope the generated-identifier numbering per lexical scope
so a reused source name cannot resolve across scopes. Consumer workaround under
test: give each scope a distinct Python-level name, which is semantically neutral
since the parser only traces it. (mla_prolog, 2026-08-12.)

### 26. Two kernels differing only in integer-literal dims share one build dir, and the second silently runs the first's binary

**Severity: wrong numbers, no diagnostic.** This is the same failure class as entry
24 and belongs at the same priority.

`@pl.jit` isolates compiled artifacts per kernel using the *static signature*:
`runtime/jit.py:1493` keys its in-process cache on
`(static_signature, tilingkey_identity, datatype_identity)`, and `:689-693` derives
the on-disk artifact directory suffix `__shape_{sha256(repr(static_signature))[:12]}`,
applied at `:746` and `:981`. Read from the source alone this looks like it makes a
constant-parameterised kernel factory safe: instantiate the same kernel body twice
with different compile-time axis constants and each gets its own `kernel.cpp`.

**It does not, because `static_signature` retains only dims declared `pl.STATIC`.**
A plain integer literal in an annotation — `pl.Tensor[[pl.DYNAMIC, 1536], pl.DT_BF16]`,
which is how every shipped kernel writes its fixed axes — lowers to a `FixedDim` and
never reaches the static signature. Measured on the board for one such kernel:
**70 Dynamic + 29 Fixed + 0 Static**, so `static_signature` is *empty*, its digest is
the digest of nothing, and three variants differing in four axis constants each
resolved to the identical build directory. The first variant to compile wins; every
later one loads its `.so`. Nothing raises. The kernel simply computes the first
variant's geometry over the second variant's data.

What makes this worse than a stale-cache bug is that the *intended* isolation
mechanism is present and looks load-bearing, so a consumer who reads `jit.py` and
reasons carefully concludes the opposite of the truth. We did: the factory was
specified on exactly that reading and would have shipped a silently-wrong build had
the plan not required proving distinctness — a differing `kernel.cpp` hash per
variant — as an explicit gate. That check is the only thing that caught it.

**Consumer workaround, verified:** pass an explicit distinct `name=` to `pl.jit` per
instantiation. `_make_artifact_build_dir` (`:217-219`) prefers `prog.name`, so a
distinct name separates the directories regardless of the signature.

Fix, in order of value: **include `FixedDim` values in `static_signature`** (or in
whatever feeds the artifact digest) so that two kernels with different fixed axes
cannot collide; failing that, **raise when a cached artifact's recorded fixed dims
disagree with the current instantiation's** — a cheap consistency check at load
time. Documenting the `STATIC`/`FixedDim` distinction would help, but a document
cannot protect a consumer from a cache that returns the wrong binary.
(mla_prolog round twelve, 2026-08-12.)

### 27. Float immediates are emitted with six decimal places, so `1.0/N` is silently wrong for most N

**Severity: wrong numbers, no diagnostic, and invisible to every A/B test.** This is the
worst entry in this file. Entry II-24 corrupts a view; this corrupts *arithmetic*, on
every geometry, in a way that no comparison between two builds can reveal.

Passing a Python float to a `vf.*` op puts it in the generated C++ as a **six-decimal
literal**. Read out of a variant's own `kernel.cpp`:

```cpp
vmuls(..., 0.000977f, ...)      // the value passed was 1.0/1024 = 0.0009765625
```

`0.000977` is **+4.480e-4 relative** to the intended value. Because the scale multiplies
`mean_square` in an RMSNorm, it becomes a constant per-row scale error on *every output
element* — indistinguishable from a legitimate result unless compared against a
higher-precision reference.

**The error is a function of N, and the sign flips**, which is what makes it so easy to
misdiagnose as rounding noise:

| N | exact | emitted | relative error |
|---|---|---|---|
| 256 | 0.00390625 | `0.003906` | −6.4e-5 |
| 512 | 0.001953125 | `0.001953` | −6.4e-5 |
| **1024** | 0.0009765625 | `0.000977` | **+4.5e-4** |
| 2048 | 0.00048828125 | `0.000488` | −5.8e-4 |
| 768 | 0.0013020833… | `0.001302` | −6.4e-5 |
| 1536 | 0.00065104166… | `0.000651` | −6.4e-5 |

So `1.0/N` is exact only while N ≤ 64; from 2⁷ upward the seventh decimal is dropped, and
whether it rounds up or down depends on N. Any N with a factor of 3 or 7 has no exact
decimal at any length and can never be expressed as one immediate.

**How it was found, because the path matters.** Measured `%differ` against fp64 climbed
0.568 → 4.149 → 5.357% as the reduction widened from 2 to 8 to 16 chunks, while the
driver's own fp32 host golden stayed flat at 0.006–0.014% — so the geometry was not the
difficulty. The natural hypothesis was a serial accumulation chain, and it was wrong: a
depth ladder at fixed width showed **zero adds** (`row_sum = c0`) already carrying the
full error, and the kernel's own tree evaluated offline in fp32 reproduced the fp64 sum
to 5.6e-9. The error was in the *scale*, not the sum. A per-row scale fit by exact bf16
interval coverage then left 4–10 residual elements of 65k–131k, localising it to the
denominator before anything was changed.

**Consumer workaround, verified:** split the scale into two immediates that each survive
six decimals and are each exact in binary — e.g. `1.0/64` followed by `64.0/N` — so the
product is exactly `1/N`. Machine-check it rather than trusting it; a four-line predicate
that rejects any immediate whose six-decimal rendering differs from the exact value is
enough, and it correctly refuses 768/1536, which have no exact decimal representation.
For those, pass the scalar as a **runtime argument** instead: the framework already does
this for `rmsnorm_epsilon_*`, so the mechanism exists and is proven.

Measured effect of the fix, c_kv leaf: `mere` 2.243e-04 → **1.05e-06** and `%differ`
4.149% → **0.014%**, against a host control of 0.0061% — the leaf stops climbing with
chunk count and tracks its own control, roughly 40× better than the shipped path.

**Design rule for consumers:** treat any precision-critical scalar handed to a `vf.*` op
as having six significant decimals. Prefer runtime arguments for anything that scales a
whole tensor.

Fix, in order of value: **emit float immediates with enough digits to round-trip
(`%.9g` for fp32, or the hex float literal)** — this is a one-line change in the emitter
and it removes an entire class of silent wrongness; failing that, **reject or warn on any
immediate that does not round-trip** at the current precision. (mla_prolog wave three,
2026-08-14.)

### 28. What a kernel body may close over is narrower than Python allows, and a `str` closure variable fails the whole probe

Two parse-time restrictions, both found by a control that refused to lie:

- **A `lambda` in a kernel body is rejected at parse.**
- **A `str` closure variable is rejected outright:** `ErrCode F00001, Unsupported closure
  variable type: str`, with the diagnostic naming the permitted set — **int, float, bool,
  list, tuple, or IR**. So a mode switch cannot reach a kernel body as a string; encode it
  as a bool or int.

The second one earns an entry for how it fails rather than what it forbids. A seven-rung
diagnostic ladder was built to isolate an unproven primitive, with each rung selected by a
string mode argument. **Every rung failed, including RUNG 0 — the negative control** — so the
ladder reported "nothing is indicted" while in fact nothing was being tested. A probe whose
selector is itself illegal cannot distinguish "the feature is broken" from "the harness never
ran", and the only reason it was caught is that the control was *expected* to pass and didn't.

This is the same class as the tuple-hoisting hazard in entry II-26's neighbourhood: **the set of
Python values a kernel body may capture is smaller than Python's, and violations surface as
parse errors far from the capture site.** Corollary for instrument design: always include a
rung that must pass, and treat its failure as evidence about the instrument rather than the
subject.

(mla_prolog wave three, 2026-08-14.)


## One capability confirmed, not a limitation — pad in the transfer, never on the host

Recorded here because two operators reached for a host repack after concluding this
was not expressible, and on the deployed runtime a host repack is not merely slow
but **unavailable**: `x.new_zeros(...)` dispatches `aclnnInplaceZero`, which the
target's CANN 9.1.0 lacks (error 561103), and strided slice assignments dispatch a
copy of their own. See entry II-24 for the three ways of *describing* a strided view,
two of which are silently ignored.

**A 2-D tile's UB row pitch is its declared width, and a narrow load does not
compact rows.** This was the one claim the design review could not settle from the
docs — every existing ST test is self-consistent under either pitch. Probed
directly on device, **both with and without `compact=1`: DECLARED in both cases.**
`pl.cast` across 2-D tiles of differing dtype with `valid < declared` works, and
runtime GM column offsets work on `load` *and* `store`. So the idiom is: declare
the tile with `valid_shape=[-1, -1]`, narrow it at runtime with
`pl.set_validshape(t, [m, w])`, and give the GM-side offsets to `pl.load` /
`pl.store`. Declared columns must satisfy `W * sizeof(dtype) % 32 == 0`; the valid
count need not be aligned, and `W` must be a compile-time constant.

**A reshape often beats a column offset.** For `apply_rotary_pos_emb`'s half mode
the winning form was not `[rows, D]` with offsets `0` and `H` — which needs two
tiles and `m` strided `H`-element segments — but viewing `q`/`k` as
**`[2*rows, H]`**, a pure reshape of contiguous storage, so a row's two halves
become UB rows `2i` and `2i+1`. One tile, GM column offset `0` only, and the GM
side is **one contiguous run of `m*D` elements: one descriptor**. Interleaved mode
cannot use it (its pairs must stay adjacent) and keeps `[rows, D]` with an offset.
Generality came from a **column-block loop** at a single small pitch (`PTW = 16`),
not from a wider tile: a large single class forces the row count down —
`W = 2048` gives `MROWS = 3`, six elements per tile at `D = 2` — trading a failure
for a timeout.

Two traps found while doing it. `padded ⟺ H % ALIGN ≠ 0` does **not** imply
`H ≤ ALIGN - 1` (`H = 9, D = 18` pads), so a size class cannot be justified by
"H is small". And deleting the full-register pass and the second interleaved store
is valid **only** while `PTW < LANES`; raising the pitch later reintroduces a
zero-lane store, the construct implicated in two known miscompiles, as a silent
wrong answer. Comment such deletions at the site.

Measured result of the rewrite: outputs **bit-identical** to the host-repack build
(50/50 `torch.equal` over 18 padded and 7 aligned shapes), the aligned path
byte-identical across 24/24 kernels, and the padded path's dispatch trace down to
`{144 view, 36 empty_like}` from
`{528 slice, 252 view, 150 copy_, 72 new_zeros, 26 new_empty, 10 clone}`.

## Upstream asks, in decreasing order of value

Recorded near-verbatim from the cummin branch [FF :911-936]; the per-entry
"Fix:" lines above are the complete list, these four are the ones that would
have changed the outcome:

1. **Expose a register-level lane shift (vslide-style).** The substantive
   ask; everything else above is papercuts around it.
2. **Document `vf.load_align`'s register-width alignment requirement**, and
   make violating it a diagnostic rather than device fault 507035 at the next
   synchronize.
3. **Fix or document the `vf.load_unalign` host-side crash.**
4. **Move the unaligned entry points next to the aligned ones**, or
   cross-reference them.
5. **Put the compiler's stderr in the exception `_ensure_compiled` raises.**
   Today it raises a bare `Failed to compile kernel '<name>'` and discards the
   diagnostic, so a codegen defect on a path the dev box cannot reproduce is
   simply invisible. Recovering it cost a bespoke `subprocess.run` shim and a
   shipped build (entry II-25). This is a one-line change with the highest
   diagnostic value in the list.
5b. **Emit float immediates with enough digits to round-trip** — `%.9g` for fp32, or a
   hex float literal. Today a `vf.*` op's scalar reaches the generated C++ with **six
   decimal places**, so `1.0/1024` is emitted as `0.000977`, a **+4.5e-4** relative
   error that becomes a constant per-row scale on every output element (entry II-27). It is
   the worst class in this list — wrong arithmetic, no diagnostic, and invisible to any
   test comparing two runs of the same build, since both carry it. The sign even flips
   with N, so it reads as rounding noise. One line in the emitter removes the whole
   class; failing that, reject or warn on any immediate that does not round-trip.
6. **Make an ignored stride argument an error, not a silent reinterpretation.**
   `pl.make_tensor`'s `stride` is dropped and the view is read as contiguous
   (entry II-24). Tied for the worst failure class in the list — wrong numbers with
   no diagnostic — and the cheapest to fix: honour it, or reject it.
6b. **Include fixed (integer-literal) dims in whatever feeds the artifact
   digest**, so two kernels differing only in their fixed axes cannot share a
   build directory and silently execute each other's binary (entry II-26). Same
   failure class as the stride ask, and arguably more dangerous: the isolation
   mechanism *exists* and reads as sufficient, so careful source reading yields
   the wrong conclusion. A load-time check that a cached artifact's fixed dims
   match the current instantiation would be enough.
7. **Make the compile budget settable outside the decorator** — an environment
   variable or runtime config instead of (or in addition to) `pl.jit(timeout=)`.
   Entries 21 and 23 compose into a wall no consumer can work around: the
   budget defaults to 60 s and `TimeoutExpired` is uncaught, while the only
   documented way to raise it is a kwarg the deployed runtime's build rejects.
   A kernel needing 61 s to compile is then simply unbuildable there.
8. **Either add an RNE narrowing mode to `vf.astype` (if any future ISA
   revision admits it) or make `CAST_RINT` on the vf path a front-end
   diagnostic** — today it surfaces as `use of undeclared identifier
   'ROUND_N'` (9.2.0) or an intrinsic static_assert (9.1.0-beta.3) deep in
   generated C++, far from the Python line that chose the mode (entry II-20).

# Part III — Consolidated asks

Ordered by cost incurred, not by implementation effort. Part references are
`I-n` for the A5 report and `II-n` for the four-operator report.

## Tier A — a wrong answer reaches production

These five produce incorrect results with no signal at any layer. Each is worth
more than everything below it combined.

1. **Make `mrgsort2`'s Python signature agree with its backend** (I-1). The
   declaration says `(src0, src1, dst, tmp)`; the backend consumes
   `(dst, src0, tmp, src1)`. Called as documented it compiles, launches cleanly,
   and leaves `dst` untouched. Until the two agree, the documentation should
   carry the backend order — there is currently no way to write the documented
   order correctly.

2. **Honour or reject `make_tensor` strides in `pl.store`** (I-20). A permuted
   view is written as though contiguous. A silently mis-addressed view is worse
   than a rejected one, and strided views are advertised as a feature.

3. **Honour or reject the predicate on an interleaved `store_align`** (I-2).
   Accepting and ignoring it is strictly worse than either alternative. While
   documenting this, state in the same place that masked operations on this
   target are **ZEROING, not merging** — code written against merge semantics is
   wrong in every inactive lane.

4. **State `auto_mutex`'s actual scope** (I-3). Authors read the name as "the
   framework handles ordering"; it orders accesses within one `mutex_id` and
   nothing else. Consider a diagnostic when a GM range is stored and re-read
   across rotation slots inside one kernel — that is the shape of the bug, and it
   corrupted ~0.5% of elements non-deterministically.

4b. **Emit float immediates with enough digits to round-trip** — `%.9g` for fp32,
   or a hex float literal (II-27). A `vf.*` scalar reaches the generated C++ with
   six decimal places, so `1.0/1024` is emitted as `0.000977`: a +4.5e-4 relative
   error that becomes a constant per-row scale on every output element. Invisible
   to any test comparing two runs of the same build, since both carry it, and the
   sign flips with N so it reads as rounding noise. One line in the emitter
   removes the whole class; failing that, reject or warn on any immediate that
   does not round-trip.

## Tier B — the program stops, or cannot be diagnosed

5. **Define empty-range semantics for `pl.range`, consistently for every
   participant** (I-26). A zero-iteration loop whose body carries a
   cross-participant rendezvous deadlocks the AI Core: the producer skips the
   handshake, the consumer still waits. No error, no timeout, no diagnostic.
   Either define `pl.range(0, 0)` as skipping body *and* rendezvous for all
   participants, or diagnose a data-dependent iteration count over a body
   containing a handshake. Silently deadlocking is the worst of the three.

6. **Expose a `pypto_pro` version string** (I-18, I-21, I-25, II-21). This is
   the meta-ask, and the only one that was paid for twice independently.
   `pypto_pro.__version__` does not exist; the overlay and the CANN 9.2.0 copy
   are byte-identical while the conda-env copy differs; only a content hash
   distinguishes them. Consequently: a `@pl.jit` keyword rename
   (`timeout` → `compile_timeout`) is undetectable locally because the older
   generation silently swallows unknown keywords, and the failure on the graded
   environment is a total import-time `TypeError` — *unmeasured*, not "scored 0".
   A 2026-07-27 runtime boundary additionally moved three coupled behaviours at
   once (AIV `get_block_idx()` semantics, `get_subblock_num()`'s existence, and
   `tile_dims=` → `order=`), and installations sit on both sides of it. Either
   keep deprecated aliases, or make local and graded builds reachable from one
   version string. Today an author cannot tell which generation will run the
   package.

7. **Enforce the intra-core event-id bound where the id is chosen** (I-19).
   `_MAX_EVENT_ID = 16` admits ids 8–15 that the CCE converter cannot emit; its
   own docstring documents `[0, 7]`. The result is an undeclared-identifier
   failure at bisheng, far from the line that chose the id. Set the intra-core
   bound to 8, or reject at the call site with the backend's bound in the message.
   Note that cross-core ids lower through a different path and are unaffected,
   which makes "the id is too large" a non-obvious hypothesis.

8. **Fix or document the `vf.load_unalign` host crash** (II-5), and **document
   `load_align`'s 256-byte alignment requirement** (II-4) — violation is
   currently a bare device fault. **Co-locate the unaligned entry points with the
   aligned ones** (II-6); they sit ~1400 lines apart, which is why the crash is
   met before the requirement is read.

8b. **Put the compiler's stderr in the exception `_ensure_compiled` raises**
   (II-25). It raises a bare `Failed to compile kernel '<name>'` and discards the
   diagnostic, so a codegen defect on a path the dev box cannot reproduce is
   invisible: five shapes failed with no reason attached, and recovering it
   cost a bespoke `subprocess.run` shim. A one-line change with the highest
   diagnostic value in this list.

8c. **Document what a kernel body may close over, at the capture site** (II-28).
   A `str` closure variable is rejected outright (`ErrCode F00001`, permitted set
   int/float/bool/list/tuple/IR) and a `lambda` is rejected at parse, both far
   from the capture. The diagnostic already names the permitted set; the
   documentation does not. A probe whose own selector is illegal cannot
   distinguish "the feature is broken" from "the harness never ran".

## Tier C — capability gaps with no workaround

9. **Decouple index-register width from payload dtype** (I-6, I-7, II-14).
   Restated from the earlier report, now with a case that has *no* workaround: a
   b16 source requires UINT16 offsets and **no `vf` operation constructs them** —
   `vf.muls` has no 16-bit row and `vf.astype` has no b32→u16 narrowing row. The
   only escape is abandoning the narrow pipeline and doubling index traffic.
   Separately, an int32 working tile must be declared `UINT32` for scatter to
   compile at all.

10. **Add a narrow-input / wide-accumulator reduction** (I-14). `vf.reduce_*` is
    same-type only ("源与目标数据类型需保持一致"), so every normalization operator
    pays an extra UB widening pass. This is the single most useful addition in
    this report.

11. **Provide a register-level lane shift** and the remaining three asks carried
    unchanged from Part II's own list.

12. **Add a round-to-nearest-even narrowing mode** for `vf.astype` on 950PR
    (II-20), and note that `CAST_RINT` fails to lower on 950PR toolchains
    (II-10) — the failure is one mode's lowering, not a two-mode surface.

## Tier D — documentation that causes false "unsupported" verdicts

These cost the most when an automated workflow treats the docs as a capability
oracle: each makes a working construct look impossible, and the workflow then
redesigns around a limit that does not exist. All are one-line changes.

13. **Complete or explicitly de-normativize `vf.astype`'s dtype table** (I-11).
    It lists four rows and mentions BF16 nowhere; the silicon does FP32↔BF16 and
    UINT16→UINT32, both exercised by an official sample.

14. **Correct the `vf.addc` example** (I-5) — it passes
    `vf.create_mask(pattern=ALL)` as `carry_src`, injecting a carry-in of 1 in
    every lane. The example is a correctness bug an author inherits *after*
    working around the compile error. Also type the carry operand correctly at
    the binding; `vf.addc` is currently unreachable from Python.

15. **Document `CastLayout.ZERO`/`ONE` as even/odd lane selectors, or rename to
    `EVEN`/`ODD`** (I-12). The enum names invite a low/high-halves reading; the
    official sample's own variable names (`c0_even`, `c0_odd`) contradict it.
    Relatedly, `CastLayout` has four members, not two.

16. **Rename `vf.lt`'s `cmp_dtype` to `cmp_width`** (I-8), or add an explicit
    signedness parameter. Read as a signedness switch it produces every element
    wrong, and there is no unsigned-compare selector at all.

17. **Document or remove `vf.copy`** (I-13). It works, has no documentation page,
    and appears in none of the 13 official samples — so depending on it is
    forward-compatibility risk taken unknowingly.

18. **State the structural ceilings in one place** (I-15): a Tile is at most
    two-dimensional, and `mutex_ids` must lie in `[0, 31]` — 32 buffer slots is a
    hard limit. Both are currently discoverable only by hitting them.

19. **Cross-reference `@pl.jit(datatype=…)` from the multi-dtype discussion**
    (I-16). One source, several specializations, one launch — authors who miss it
    reach for *N* separate `@pl.jit`s selected host-side, which is more code and
    only arguably satisfies a single-kernel requirement.

20. **State the unroll requirement for `tiling` array indexing** (I-17). `pl.range`
    is the only accepted iterator and its index is never a literal, while a
    `tiling` array index must be a literal — so a table lookup over a runtime
    selector cannot be written as a loop at all. Today this is discovered one
    rejection at a time, and the first diagnostic points at the loop rather than
    at the index that motivated it.

## Recorded as measured-safe, so it is not re-litigated

Three operators independently designed *around* the possibility that two AIVs
writing parts of one 32-byte beat could tear a write. It has now been measured
safe under a whole-tile `pl.store` — 0 mismatches over 100 launches at a
genuinely straddling offset, with the control proved live in the same run (6240
mismatches, the required count). The conservative designs remain correct; they
are simply stricter than the hardware requires. Note the geometry trap: at
`m_off = 0` the hazard is unreachable by construction, so a green probe there
says the construction was invalid, not that the path is safe.
