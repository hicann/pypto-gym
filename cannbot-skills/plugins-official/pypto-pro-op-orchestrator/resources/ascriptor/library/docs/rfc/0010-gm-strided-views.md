# RFC-0010: GM strided views (`GMTensor.view`)

Status: **Accepted and implemented (first cut, 2026-08-30)** — the maintainer accepted the
design with three rulings (no layout tokens in `strides`, no GMList members, no negative
strides; §8 records them) and the first cut is live and board-verified (§9).

## 1. Motivation

PyPTO Pro's `make_tensor(ptr, shape, strides)` builds a tensor view over raw GM with
arbitrary strides. Ascriptor has no equivalent: GM tensors are kernel parameters with
declared shapes, and the view ops we have are all shape-preserving in a narrow sense —
`mem.slice` (rectangular sub-view), `mem.reshape` (contiguous only), `mem.reinterpret`
(dtype only). What is missing is exactly one capability: **re-describing an existing GM
region with a new shape, new strides, and an element offset**, without moving data.

Concrete uses from the corpus and the pypto bridge:

- reading a row-padded external tensor (`[S, D]` data in a `[S, D_pad]` allocation);
- walking `[B, H, S, D]` GM in `[S, B*H*D]`-strided order to feed one DMA per row group
  instead of a Python loop of slices;
- carving interleaved per-core lanes out of one workspace without `numel`-arithmetic in
  user code;
- the pypto backend: translating our lowered DMA descriptors into `make_tensor` calls is
  mechanical when the IR itself carries (shape, strides, offset).

## 2. Frontend surface

```python
v = x.view(shape, strides=None, offset=0)
```

- `x` is a GM tensor (a parameter, a `list.item` result, a workspace, or another view).
- `shape`: ints or shape symbols / scalar values.
- `strides`: element strides, one per dim; `None` means row-major contiguous over `shape`.
- `offset`: element offset from the start of `x` (int or scalar value).
- **No `dtype` parameter.** Dtype punning stays in `mem.reinterpret`; compose as
  `x.reinterpret(...).view(...)`. One op, one job.

`view` is GM-only by design. On-chip spaces (UB/L1/L0) have banked / fractal physical
layouts where an arbitrary-stride view is not a free re-description; nothing may create
one.

## 3. IR

```
op("mem.view", kinds=GM-only, pipe="S",
   operands=(N("src", "gm<*, *>"),),
   attrs=(A("shape", "list", required=True),      # ints | scalar values
          A("strides", "list", required=True),    # element units, one per dim
          A("offset", "int|value", required=True)),
   results=(Res("view", "gm<*, *>"),))
```

The result is a **plain** `gm<dtype, [shape]>` value — strides and offset live in the op's
attrs, not in the type. Downstream type checking is untouched; only consumers that walk
back to the defining op (device_lower, interp, autosync) see the strides.

The verifier must check, when all quantities are static: `len(strides) == len(shape)`, and
`offset + max reachable element index < numel(src base)`.

## 4. Lowering: DMA-only consumption

The one-op-one-instruction rule (D-083) applies: a view is zero instructions; strides only
become real as fields of a DMA burst descriptor. Therefore:

- **Only `dma.*` ops may consume a `mem.view` result.** device_lower folds the view's
  (shape, strides, offset) into the burst/stride fields of the descriptor it already
  builds for GM endpoints; a view whose innermost stride is 1 and whose row stride fits
  the descriptor's gap field costs exactly what the equivalent hand-sliced DMA costs.
- The `(1, rows)`-strided column pattern lowers to the `dn2nz` path where the engine has
  one (same selection device_lower does for transposed loads today).
- Any other consumer (a `@vf` load, a compute op, `mem.slice`-of-view is fine but
  vf-consumption of that slice is not) is a **compile error** naming the view op, not a
  silent slow path.

## 5. Interpreter

`torch.as_strided(base_flat, shape, strides, offset)` — both for reads and as the write
target, so aliasing semantics (two views over one base) fall out of torch's own storage
model. No copy at view time.

## 6. Autosync

A view result aliases its base: the access-set walk treats `mem.view` exactly like
`mem.slice` (same base-tracking path, RFC-0005). Two views of one base with disjoint
static footprints may be refined later; the safe default is "same memory".

## 7. Performance lint

- innermost stride != 1 → warning: per-element bursts, the DMA engine cannot coalesce;
  suggest restructuring so the fastest-varying dim is contiguous.
- (existing D-084 lints unaffected.)

## 8. Open questions — resolved (maintainer, 2026-08-30)

1. Layout tokens in `strides` — **rejected**: that is `mem.reinterpret(layout=...)`'s job.
2. `GMList` members (`xs[i].view(...)`) — **not supported**.
3. Negative strides — **not supported**: refused statically at compile time.

## 9. As implemented (first cut, 2026-08-30)

- Frontend `x.view(shape, strides=None, offset=0)` compiles to `mem.view`; the source must
  be a **whole GM parameter or workspace** (a slice, a list member, a reshape/reinterpret
  result or another view is a compile error — composing those is phase 2), rank is 1 or 2,
  every quantity is non-negative, and the **innermost stride is the static 1** (a non-unit
  innermost stride is a compile error for now, not the §7 lint). Shape, the row stride and
  the offset may be runtime scalars; the bounds check runs when everything is static.
- `view_of` treats `mem.view` as a new root carrying `gm_strides`; slices of a view
  compose. device_lower folds the explicit row pitch into every GM descriptor family:
  `gm_to_ub.pad` / `ub_to_gm.pad` (gap = stride − cols), `gm_to_l1.nd2nz` / `.dn2nz`
  (`N_src` = stride, covering the `.T` path), `l0c_to_gm` (`N_dst` = stride).
- The interpreter maps a view to `torch.as_strided` over the base storage (reads and
  writes alias); autosync re-anchors every view access on the base root as a
  whole-tensor access (§6's safe default). The printer emits one
  `const GMTensor<T> v = base[offset];` — zero instructions, as §4 requires.
- Board-verified bit-exact: `tests/kernels/a5/samples/gm_view.py` (padded-row read window
  + shifted write window over a full-write baseline). Unit surface:
  `tests/frontend/test_gm_view.py` (13 cases incl. every refusal),
  `tests/passes/test_device_lower_copies.py::test_strided_view_feeds_nd2nz_with_its_row_stride`.
- **Phase 2, part 1 (2026-08-30): composition and the pypto translation are in.**
  - *Composition*: `v.view(...)` where `v` is itself a view, a slice, a reshape or a
    same-width reinterpret **folds at compile time** to a single `mem.view` on the root —
    the IR still only ever carries root-sourced views, so every pass, the interpreter and
    the printers are untouched. A same-width GM `mem.reinterpret` vanishes, `mem.reshape`
    folds to one run from its window's start (§10); a slice contributes `offset + r·pitch + c`; a
    strided source window requires the folded rows to tile its rows (`offset % cols +
    new_cols <= cols`, row stride a multiple of `cols`) and everything in the chain to be
    static — anything else is a compile error naming the condition. A view of a
    reinterpret keeps the reinterpreted dtype (the one case where `mem.view`'s result
    dtype differs from its operand's).
  - *pypto-pro*: `mem.view` is the direct `make_tensor` mapping the gap survey wanted —
    printed as `pl.make_tensor(x, shape, strides)`, with a non-zero offset spelled through
    the documented `pl.addptr(pl.make_ptr(x), off)` composition (element units) and a
    folded reinterpret as the `dtype=` argument; the result then feeds `pl.load` /
    `pl.store` like any GM parameter.
- **Phase 3 (2026-08-30): the hardware ruling and the rest.**
  - *Non-unit innermost stride* — settled by the ISA: **NDDMA (`nddma_out_to_ub_*`) is the
    one engine that walks GM with arbitrary element strides, and it only exists on the
    GM -> UB read path** (dim <= 5); every write engine (`copy_ubuf_to_gm_align_v2`, the
    nd2nz/fixpipe bursts) moves contiguous runs. So a gather view (innermost stride > 1,
    or a genuine rank > 2 residue) is legal **read-only**: GM -> UB lowers to
    `dma.gm_to_ub.nd` with the view's strides as the loop strides, and a surface lint
    warns that NDDMA issues one transfer per element (no burst coalescing); consuming the
    same view on ub_to_gm / nd2nz / nz2nd / the `.T` path is a lowering error naming the
    engine. One more port rule, board-measured: **NDDMA rows land on the UB port's
    32-byte blocks** — a destination tile whose row is not a 32-byte multiple silently
    corrupts its neighbours (no fault), so lowering refuses it statically ("pad the tile
    row"). Board-verified bit-exact (`gm_view_nd.py`: a column gather into a padded
    64-byte-row tile, and a rank-3 window).
  - *Rank > 2* — up to rank 4 at the surface; adjacent contiguous dims merge at compile
    time (never below the user's declared rank once <= 2), and a genuine rank > 2 residue
    needs a unit innermost stride, lands row-major in a `[rows*..., cols]` UB tile and
    moves on the same NDDMA path.
  - *Dynamic quantities in composed chains* — allowed over a **contiguous** source window
    (the fold is a scalar displacement, emitted as `scalar.add`); a strided source window
    still needs everything static, because the row-tiling conditions cannot be checked at
    run time.
  - *Packed sub-byte dtypes* — **closed as a design ruling, not code**: a packed fp4/int4
    plane is declared as its u8 carrier and the carrier is viewed — the carrier byte is
    the hardware's own addressing unit and the corpus already follows this convention;
    a logical-element packed view would just hide a `/2` in every layer. The refusal
    message now points there.

## 10. Window-sourced views (I016, 2026-09-17)

IR may carry `mem.view`, `mem.reshape` or `mem.reinterpret {tile}` over any window, not only the
root-sourced views §9 folds. Each starts a coordinate system at the byte address of its operand
window's first element. The CCE and PTO ISA printers fold that address (`cce/views.py`), the A5 FIX
bound check composes it, and the emitted instructions use it. A `mem.view` adds `offset` elements
of its dtype and steps by its own strides only; `mem.reshape` and `tile` address row-major. A slice
offsets within its operand's system, and a same-width reinterpret keeps offsets, strides and origin.
Native A5 runs of Pro's `gm_view_rebase_probe` (seeds 7, 203) read a view of a view this way.
Composition rather than refusal is the rule: the frontend already emits reshapes and reinterprets
of offset views, and silicon runs them at this address. A view lies inside its root allocation.
Access analysis compares two reshaped windows only inside one reshape of one window at static
offsets. The verifier refuses `mem.view` of on-chip memory (§2) and a width-changing reinterpret
of a view system, whose printed strides keep the old element unit. The simulator refuses an
NZ-layout window at a non-zero offset under a rebase, which its logical layout does not model.
