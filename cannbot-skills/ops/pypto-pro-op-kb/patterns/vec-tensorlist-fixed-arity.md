# TensorList under a fixed-parameter DSL — padded fixed arity

**Topology:** `elementwise` (and any topology whose inputs carry `is_list: true`)
**Status:** validated skeleton on Ascend950PR_9579 (CANN 9.2.0). 21 generated
kernels covering `L = 1..64` over three dtypes; the whole ladder compiles, and
`L = 64` at 385 declared parameters launches and returns per-slot-correct data.
**Evidence:** [retained validation record](../examples/validation-records.md), including
the generated-length ladder, dtype coverage and 64-slot launch result.

## When this applies

An input is declared `is_list: true` — a `TensorList` whose **length is a runtime
value**. The `torch._foreach_*` family is the obvious case, but the pattern is about the parameter
boundary rather than the arithmetic: it applies whenever the number of tensors is
not known when the kernel is compiled.

It does **not** apply to a *batched* operator whose batch is an axis of one
tensor. That is a loop bound, not an arity problem.

## The problem, stated exactly

pypto-pro has **no TensorList parameter type**. `runtime/jit.py:354-370`
(`_extract_param_specs`) recognises `TENSOR` / `PTR` / `TILING` / `SCALAR` and
nothing else; `_validate_tensor_arg` requires each tensor argument to be a
`torch.Tensor`; `_validate_args` requires the argument count to match the
declared count exactly. **A Python list cannot be passed.** So the list has to
become *parameters*, and the number of parameters is fixed when the kernel is
compiled — while `L` is not known until it is called.

## Two routes that do not work, with what they actually say

**1. Packed descriptor of device addresses — dead, and not for a fixable reason.**
Build an `int64` table of `data_ptr()`s on the host, pass it as an ordinary
tensor, and turn each entry into a tensor view inside the kernel. Every piece
exists: `pl.Ptr`, `pl.make_ptr`, `pl.make_tensor`, `pl.addptr` are all exported.
It fails on the one step that matters:

```
ptr.make_ptr requires first argument to be a PtrType, but got ScalarType.
Use pl.Ptr[dtype] to annotate pointer parameters.
```

Identical whether the address is read from a GM tensor or from a UB tile, so it
is **not** a GM-access restriction — **a pointer only ever enters a kernel
through a parameter annotation.** A value materialised at runtime can never
become one. The control confirms the rest of the family is sound: a declared
`pl.Ptr[pl.DT_FP32]` parameter, viewed with `pl.make_tensor(p, [1, n], [n, 1])`
and read with `pl.load`, compiles, launches and returns correct data. So the idea
is only dead on the runtime-value step — which is the step the whole scheme needs.

**2. Host loop, one launch per element — forbidden, and expensive.**
[`constraints/wrapper-boundary.md`](../constraints/wrapper-boundary.md)'s last
migration row rules it out by contract. It is worth knowing what the contract is
worth, because the answer is not uniform. The same work twice — one launch of a
padded fixed-arity kernel over all `L` items, versus `L` launches of a
single-item kernel from a host Python loop — on Ascend950PR_9579:

| work | L | 1-launch dev µs | L-launch dev µs | dev ratio | wall ratio |
|---|---|---|---|---|---|
| 2 items, 2.1M elts, fp32 | 2 | 8.36 | 10.17 | 1.22x | 1.36x |
| 4 items, 8.4M elts, bf16 | 4 | 18.68 | 25.87 | 1.39x | 1.80x |
| 2 items, 99.4M elts, bf16 | 2 | 656.26 | 659.29 | 1.00x | 1.00x |
| **64 items x 65,536, fp32** | 64 | **28.26** | **182.55** | **6.46x** | 2.71x |
| **64 items x 4,096, fp32** | 64 | **22.05** | **136.66** | **6.20x** | 2.65x |

(device = summed `Duration(us)` per iteration, the evaluator's own metric.)

Two readings, and the second is the one that generalises:

* The penalty is set by the ratio of launch cost to work **per item**, not by
  total work. At 99.4M elements over 2 items the kernel dwarfs the launch and the
  loop costs nothing; at 64 items of 4,096 elements it costs 6.2x.
* **A host loop does not look obviously wrong on small-`L` cases.** At `L <= 4`
  the penalty is 1.0–1.8x, well inside the range an ordinary tuning change moves
  — and in the task this was measured on, nearly every visible case sat at
  `L <= 2`. The rule earns its keep on the cases a visible run never reaches,
  which is exactly why it has to be a rule rather than a measurement.

## The shape that works: fixed arity, padded slots, one launch

Unroll the list into parameters at **generation** time, over a ladder of bucket
sizes, and pad. Per slot `i`: its tensors, plus its element count `n_i`, plus one
load-balance scalar.

```python
@pl.jit(auto_mutex=True)
def op_{dtype}_b{B}_{hash}(
        x1_0: pl.Tensor[[1, pl.DYNAMIC], dt], ..., y_0: pl.Tensor[[1, pl.DYNAMIC], dt],
        ...                                    # slots 1 .. B-1, identical
        n_0: pl.DT_INT32, r_0: pl.DT_INT32,    # ... one pair per slot
        s: pl.DT_FP32):

    <tile groups declared ONCE here>            # see "UB is bucket-independent"

    with pl.section_vector():
        nc = pl.get_block_num()
        cid = pl.get_block_idx()

        # ---- slot 0, emitted literally by the generator -------------------
        nch_0 = (n_0 + CHUNK - 1) // CHUNK      # ceiling division
        st_0 = (cid + r_0) % nc                 # phase rotation, see below
        for c_0 in pl.range(st_0, nch_0, nc):
            off_0 = c_0 * CHUNK
            vl_0 = pl.min(CHUNK, n_0 - off_0)
            a_0 = g1.next()                     # rotate FIRST
            pl.set_validshape(a_0, [1, vl_0])   # then window the ROTATED tile
            pl.load(a_0, x1_0, [0, off_0])
            ...
        # ---- slot 1 .. slot B-1: the same block, re-emitted ---------------
```

The wrapper picks `bucket = min(b for b in BUCKETS if b >= L)`, fills slots
`L..bucket-1` with **a repeat of slot 0's tensors** at `n_i = 0`, and launches
once. Padding is free at the data level — no allocation, no copy, no extra bytes
moved — because the pointers ride the argument buffer the launch already builds.

Each item is flattened to `[1, n_i]` by a `.view()` on the host. That is a pure
view of a contiguous tensor, so it costs nothing and never appears in the
profile; it is also what lets one kernel cover every rank. **Guard contiguity and
raise** — on a non-contiguous tensor the same call silently materialises a full
copy, a dispatched device kernel inside the measured window.

## The arity ceiling: none was found at or below the declared maximum

Bisected because the argument-buffer limit for this toolchain is documented
nowhere. Each slot carried a distinct value, so a parameter-to-slot mismapping
would have shown as wrong data rather than as silence.

| arity | tensor params | int32 params | total | compile | launch | data |
|---|---|---|---|---|---|---|
| 4 | 16 | 4 | 20 | ok | ok | correct |
| 16 | 64 | 16 | 80 | ok | ok | correct |
| 32 | 128 | 32 | 160 | ok | ok | correct |
| **64** | **256** | **128** | **385** (+1 fp32) | **ok** | **ok** | **correct** |

385 declared parameters compile and launch. Two follow-ups worth carrying:

* **Dynamic dims do not silently inflate the ABI.**
  `_append_tensor_ctype_arg` appends an extra `ctypes.c_int64` only for dims that
  are *named strings*, not for `pl.DYNAMIC`, which is an unnamed marker. Measured
  both ways: the `[pl.DYNAMIC, pl.DYNAMIC]` spelling and the `[1, pl.DYNAMIC]`
  spelling both reach bucket 64. Declaring the axis that is structurally `1` as a
  literal is still tidier, and mixing a literal with `pl.DYNAMIC` is the official
  spelling (`pro_ops/datacopy/test_dual_mode_tail.py`).
* **Compile time is not the limit either.** The 64-slot fp32 kernel compiled in
  3.2 s, against `@pl.jit`'s 60 s default timeout.

## Padding needs no `if` guard — the empty range is genuinely empty

The whole scheme rests on `pl.range(start, 0, nc)` executing zero times. If pypto
lowered `start > stop` as a live loop, every padded slot would read slot 0's
tensor at a garbage offset and write it back **with no error raised**. Measured
directly, at arity 64 with slots 0–31 real and 32–63 padded:

```
real slots  : 32/32 correct
padded slots: 32/32 untouched
smallest real slot had 1 tile(s) against 56 cores
VERDICT: PADDING PATH OK
```

So **write the slot body unconditionally**. A dead slot costs one division, one
addition, one modulo and a loop test, per core. `n_i = 0` is an unambiguous
sentinel wherever a per-dimension size of at least 1 is declared, which is the
usual case.

The same measurement covers a case that is *not* padding: an item smaller than
the core array. One tile against 56 cores gives 55 of them a start past the end,
so the empty range is on the **real** path too, on any short item.

## Load balance: the naive per-slot stride idles the array

The obvious `pl.range(cid, nch_i, nc)` gives every slot the same start, so when
items are small the same few cores take all the work: **with 64 items of 3 tiles
each, cores 0–2 execute every tile and cores 3–55 execute none.** That is not a
corner — it is the regime a long list *is*.

Fix it with a per-slot phase rotation. The host computes a running prefix sum of
tile counts and passes its negated residue as an int32 scalar, so slot `i` starts
where slot `i-1` finished:

```python
# host (pure Python integer arithmetic -- no tensor op, no device work)
prefix = 0
for i in range(L):
    n[i]   = int(x1[i].numel())
    rot[i] = (BLOCKS - (prefix % BLOCKS)) % BLOCKS
    prefix += (n[i] + CHUNK - 1) // CHUNK

# kernel
start_i = (cid + rot_i) % nc
```

**Why this is safe to get wrong.** For any fixed `k >= 0`, `cid -> (cid + k) % nc`
is a bijection on `[0, nc)`. So whatever `rot_i` is, the `nc` cores receive `nc`
distinct starts covering the whole range, and chunk `c` of slot `i` runs on
exactly the one core whose start is `c mod nc` — every chunk once, no chunk
twice. Coverage and disjointness do not depend on `rot_i` at all. **`n_i` is the
only host-computed value correctness depends on; a wrong `rot_i` is slower and
still right.** That also means a host/kernel disagreement about the core count or
the chunk width is a balance bug, never a correctness bug, since the loop bound
and the stride are both read from runtime values inside the kernel.

Two spellings are not equivalent, and the difference is a real bug:
`(cid + nc - phase_i) % nc` goes **negative** if `nc < BLOCKS`, and C++ `%` on a
negative operand yields a negative result, so `start` is negative and the loop
computes a negative offset. `(cid + rot_i) % nc` has a non-negative dividend for
every `nc`. Prefer it; it removes a side condition rather than documenting one.

Keep every scalar in this arithmetic at `DT_INT32`. The backend has no 64-bit
integer division (`Div of bitwidth greater than 32 not supported`), and the
failure is conditional on type inference, so a latent instance compiles until
something unrelated widens the divisor.

## UB is bucket-independent — declare tile groups ONCE

**The single most consequential rule here.** The `make_tile_group` calls belong
in the kernel body *above* the unrolled slots, and every slot body reuses them.
Unrolling 64 slots invites a declaration inside each slot body, and UB usage
silently becomes 64x — megabytes against a quarter-megabyte budget.

There is a tell that per-slot declaration is wrong, and it arrives before the
overflow does: `addrs` lists must be **module-level constants**, because a kernel
body is parsed as an AST and never executed, so a body-local `B = CH * 4` fails
with `Cannot resolve expression '[0 * B, 1 * B]': name 'B' is not defined`.
Per-slot groups would need per-slot address constants, which nothing in the
arithmetic produces. If you find yourself writing `addrs=[SLOT_BASE_i, ...]`, stop.

Sequencing follows for free: slot `i+1`'s first load reuses the buffer slot `i`'s
last store released, and `auto_mutex` serialises that automatically because it is
the same tile group. Correctness never depends on slot ordering.

## Sizing the ladder

`{1, 2, 4, 8, 16, 32, 64}` — 7 buckets per dtype — was chosen so no case at the
common small `L` carries dead slots, at a cost of extra compiles. **That is
insurance, not a measured win, and should be labelled as such:** a padded slot is
measured cheap, but nobody has measured 63 of them on a 5 µs kernel. A sparse
ladder `{4, 16, 64}` caps padding at 4x for a third of the compiles. The
falsification is one run: time the smallest case under the narrowest and the
widest bucket, and collapse the ladder if the delta sits inside the harness's own
control tolerance.

Dense at the bottom is the defensible half regardless: that is where kernels are
shortest, so a fixed prologue is a larger fraction of the total, and it is where
the case mass sits.

## What fails silently, in one list

1. **Tile groups declared per slot** — UB overflows by the bucket factor.
2. **An `if n_i > 0` guard** — harmless but unnecessary; its absence is what makes
   padding free, and adding it suggests the empty-range contract is in doubt when
   it is measured.
3. **`(cid + nc - phase_i) % nc`** — negative start whenever `nc < BLOCKS`.
4. **Per-item tile widths** — a tile sized `[1, n_i]` or `[1, align64(n_i)]`
   re-introduces the 64-lane alignment fault *per slot*, so it passes on aligned
   items and faults on unaligned ones. Keep one fixed-width physical tile and vary
   only the runtime valid window.
5. **A `.contiguous()` in the wrapper** — turns the free flattening view into a
   full copy inside the measured window.
6. **Padded slots aliasing one output buffer into many output parameters** —
   inert given `n_i = 0`, and measured inert, but if a bucket misbehaves in a way
   that is not argument width, pass a shared one-element dummy instead.
