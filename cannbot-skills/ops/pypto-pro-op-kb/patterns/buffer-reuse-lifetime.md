# Buffer lifetime and rotating tile groups

## Applies when

A tiled kernel overlaps iterations or reuses local storage for multiple roles.
The implementation must distinguish:

- **persistent state**, which survives across iterations;
- **rotating tiles**, whose slots alternate between producer and consumer;
- **scratch**, which is dead after the current operation or iteration.

## Safe workflow

1. Draw each tile's live interval from its first write to its final read.
2. Give simultaneously live roles non-overlapping address ranges.
3. Use a multi-slot tile group only when iterations are intentionally rotated.
4. Advance a rotating group once per logical iteration.
5. Keep persistent state outside the loop and do not rotate it.
6. Set the valid shape before every operation that consumes a partial tile.
7. Compile and run the exact schedule on the target SDK; mutex and address
   behavior is version- and target-sensitive.

Do not infer that two roles may alias merely because their source tensors have
the same shape. Alias safety depends on live intervals.

## Failure signatures

- a running value resets or alternates between stale values;
- correctness changes when double buffering is enabled;
- a reduction sees padding or data from a previous iteration;
- local-memory usage exceeds the target budget.

## How many slots: a counting law, and the credit that is not a slot

Step 3 above says to use a multi-slot tile group "only when iterations are
intentionally rotated" and does not say **how many** slots. The count is not a
judgement call:

```text
required slots  ≥  (overlapped beats) × (simultaneously live roles)
```

*Beats* are the iterations in flight at once; *roles* are the distinct things the
storage is being used for while all of them are still live. The law is pure
counting — `B` iterations each needing `K` values that have not yet retired need
`B × K` places to put them — so it does not depend on which DSL or which
hardware. Two overlapped beats with two live roles need **four** slots, and a
two-slot group is not a smaller version of that design; modulo wrapping maps the
next beat straight onto the previous beat's still-live roles.

### Handoff credits and rotation slots: transferred guidance

> **未在 PyPTO-Pro 上验证。** 本节关于 credit 与 slot 的区分、提高 credit 的
> 风险，以及错误 credit 可能不报错的结论均迁移自另一套 DSL，本仓从未复现过。
> **本地验证方法**：用两槽 group 搭一个两拍、两角色的轮转，跑一个会回绕的 shape；
> 在保留复现结果前，把这些结论当作待验证假设，不要当作 PyPTO-Pro 的既定行为。

A producer→consumer handoff also has a depth: how many producer beats may be in
flight before the consumer frees a slot. It is easy to reach for that number when
a rotation misbehaves, and it is the wrong lever.

- **Credits describe the handoff. Slots describe local role separation.** They
  are independent, and only one of them is what a stale-value bug is about.
- **Raising credits above the slot count is actively harmful**: it licenses the
  producer to run onto a slot the consumer still holds. The design was stalling;
  now it corrupts.
- Raise credits **only** when the handoff also rotates through additional slots.

### Why this class of bug may be silent

In the source DSL, the prologue pre-publishes exactly as many free tokens as the
epilogue drains. **Both sides stay balanced whatever the number is**, so a wrong
credit count produces no unbalanced-event warning, assertion, or error; the
values are simply stale.

Two further non-proofs, from the same source:

- **Source order is not retirement.** That a store appears earlier in the Python
  body than the overwrite does not establish that the store's pipe has finished
  reading its source. This is the same fact behind the local rule that a scratch
  store needs an explicit fence before the next vector load
  (`ops/pypto-pro-op-develop/references/vf-reduction-perf.md`).
- **A broad barrier that changes the symptom has not fixed the ownership model.**
  If widening a fence makes the failure move rather than disappear, the slot
  count is still wrong.

### Budgeting it here

`auto_mutex` identities are a *scarce* resource in this DSL, and the identity
and byte budgets are reached by different designs. `mutex_ids` must lie in
`[0, 31]`; under `auto_mutex`, different live buffers or rotation slots need
distinct identifiers. Therefore
**32 is the hard ceiling on simultaneously distinct `auto_mutex` identities,
not on the number of physical buffers or tiles**
([../constraints/tiling.md](../constraints/tiling.md)) — a `B × K` product is
spent against that ceiling as well as against the byte budget. And slot
*identity* is what `auto_mutex` orders against: a slot counter stepping by the
wrong stride aliases two logical buffers onto one physical buffer while the event
machinery still issues two credits
([../references/pypto-pro-framework-findings.md](../references/pypto-pro-framework-findings.md),
A5 probe session) — the same failure this law is meant to prevent, arrived at
from the indexing side.

**Verify it on a shape that wraps.** The law is unobservable at one work item per
core; see
[investigation-discipline §13.2](../references/investigation-discipline.md).

## Validation status

The retained [softmax implementation](../examples/samples/softmax/softmax_impl.py)
shows the group construction, but its recorded 64-row, four-core case gives each
core one loop iteration and does not exercise slot rotation or wrap. The
rotation skeleton remains conceptual until the KB retains a runnable
implementation and a scope-matching passing target result that checks every
active core's output and makes at least one active core wrap by executing more
iterations than the relevant group depth. That result does not validate the
separate transferred credit guidance above.
