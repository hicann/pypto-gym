# PyPTO Pro import: why a form is not admitted

The [import evidence index](pypto-pro-import-support.md) counts the refused fixtures, and
`tools/pypto_import_support.py --json` pairs each one with the located message the importer
actually raises, Pro source span included. Those messages say why *this* form is refused.
They cannot say what *kind* of obstacle it is, and the kind is what decides the next step:
three of the five classes below are finished answers, and two are work items.

Use this page to place a refusal. Use the index for the per-fixture reason, the
[import contract](rfc/0015-pypto-pro-import.md) for the admission rules, `docs/defects/` for
our own implementation faults and [upstream](upstream.md) for Pro's.

## A. A5 does not do it, or native Pro already fails

The importer follows the hardware. Nothing is missing on our side, and admitting the form
would mean printing something the device cannot run. These refusals close.

- FP32 scalar `max`/`min` inside a VF body: refused in the IR, because A5 computes IEEE
  754-2019 maximum/minimum while the surrounding model and printers do not
  (I032, [RFC-0001](rfc/0001-ir.md)).
- Multi-block INT8 NZ stores that do not start at column 0: native Pro refuses to compile them.
- SIMT launches of functions named after libm overloads: native Pro fails to build them
  ([A5-UP-042](upstream.md#a5-up-042--generated-simt-function-names-collide-with-libm-overloads)).
- Mode 0/1 cross-core set from `PipeType.S` on AIV: neither native nor CCE compiles.
- FP32 NZ stores: A5 writes them as 8-element channel-split blocks, which no IR store spells.
- MX Final phase and order-free NN plane loads: A5's read order has no target load form.

## B. An Ascriptor IR contract gap, deliberately not widened

The form is real and A5 runs it, but the target IR has no contract for it. These stay refused
by instruction: the import task does not widen library capability. Each is a candidate for a
separate, owner-level decision, not something to work around in the importer.

- Chained unaligned stores whose `vstur`/`vstar` follow an AR counter. Recorded as future work
  in [RFC-0015](rfc/0015-pypto-pro-import.md); see the visibility regression for the
  measured register-lifetime rules that the admitted subset does rest on.
- Multi-bit and other CTRL fields, and arithmetic on possibly negative CTRL-derived values
  (Pro's CCE keeps the read in a `uint64` `auto`, the IR in `int64`).
- Cross-core collectives: mode 3 UNICAST_BLOCK, IDs 11..15, dynamic IDs and `sync_all`.
- Tile dumps, `pl.trap()`, `%p`, and unsigned conversions with no non-negativity proof.
- Two- and three-dimensional SIMT launches, y/z axis queries, SimtCallee helpers and early
  return — twelve forms in all.
- Slot buffers with aliasing, partial overlap or non-constant-stride cursors; L0 and
  paired-side buffers.

One exception was granted and executed: admitting the scalar pipe as a raw flag's set pipe
(2026-09-18). That was a contract gap rather than a capability extension — the kernels already
ran on A5 and all three printers already spelled the channel.

## C. Profile or ABI decisions

Not a semantic obstacle at all: the pinned compatibility profile or the export ABI simply did
not describe the form. Resolving one is a maintainer decision about revising a pin, and both
known cases are resolved.

- Explicit `vf.mem_bar(mode=...)`: the pinned profile did not declare the attribute although
  Pro's API and emitter accept it. Resolved by revising the profile to `/2`; the earlier
  profile is a registered predecessor, so recorded exports stay checkable.
- NZ-packed GM parameters and slot buffers (I018 regression): resolved earlier.

A profile revision invalidates no fixture, because `session.same_export` accepts a recorded
producer that names an accepted predecessor of the pinned profile.

## D. Measured behaviour contradicts Pro's documentation

The measurement wins, and the difference is reported upstream. The importer follows what the
silicon did, and the refusal — where one remains — protects the documented-but-unobserved case.

- `mem_bar` ordering: all twelve modes translate, but the probes found no ordering effect at
  all, and three scalar cases pointed the opposite way from the documentation.
- `dcci`: the earlier conclusion that dcci does not prevent lost writes was overturned by
  measurement. The protocol is producer-side dcci with publication ordered after the store;
  only S-pipe scalar stores lose, and MTE3 moves never do ([I012](defects/README.md),
  [RFC-0006](rfc/0006-lowering-pipeline.md)).
- FP16 `exp_sub` predicates: documented as valid on even mask bits only, measured on A5 as
  sampled by source position
  ([A5-UP-043](upstream.md#a5-up-043--fp16-exp_sub-predicate-documentation-contradicts-a5)).
- `mrgsort2`: the documented argument order differs from the graph operands
  ([A5-UP-040](upstream.md#a5-up-040--documented-mrgsort2-argument-order-differs-from-its-graph-operands)).

## E. Real forms whose silicon semantics are unmeasured

The only open work item that a future batch can close without an owner decision. Nothing is
wrong; there is simply no reading yet, and RFC-0015 admits a form only with matching measured
semantics. Measure first, admit second — never approximate to make a fixture import.

- BF16 VF block copies, SPR forms and runtime offsets; memory access inside VF control flow;
  runtime offsets, strides and counts; register aliases.
- Partial valid ranges, dynamic valid shapes, and FP16 record forms of `sort`/`mrgsort`.
- `count=0` clean register stores, and stores starting above a residual start without
  crossing a block.

## Reading a refusal

A refusal names the Pro source span, so it points at the line to look at rather than at the
importer:

```text
pro_p6_vf_memory.py:362:9: VF memory access inside VF control flow needs its own cursor and
footprint rule
```

Every refused form is kept as a real Pro export fixture under `tests/importers/fixtures/`
with a test that pins the message, so a later change that silently starts accepting one fails
the suite. Editing a bundle to get past a refusal defeats that, and the
[import playbook](../../agent/en/playbooks/import-pypto-pro.md) rules it out.
