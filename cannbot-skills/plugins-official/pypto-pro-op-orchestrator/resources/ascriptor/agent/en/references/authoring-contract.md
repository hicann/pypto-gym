# Contract before implementation

Infer these fields from the user's math/reference, current API and unit source. Record
the result once in the unit contract rather than repeatedly asking intake questions.
An unresolved choice needs user input only when evidence cannot settle observable
semantics. Routine tile, identifier and file-layout choices stay with the implementer.

| Field | Required meaning |
|---|---|
| Mathematics | Operations, parenthesization, constants, reduction axes and edge cases |
| Public ABI | Named IO, order, shapes/symbols, dtype, strides/layout, physical storage |
| Mutation | Output initialization, in-place aliases, forbidden overlap, returned outputs |
| Domain | Minimum/maximum shapes, empty inputs, alignment, tail, supported core counts |
| Precision | Accumulator dtype, each cast/rounding/saturation boundary, non-finite policy |
| Comparison | Named-output exact/bit/numeric rule, tolerance and reason |
| Delivery | Device/backend, runtime launch count, permitted host work |
| Ownership | Workspace producer, consumers, lifetime, initialized and defined extents |
| Verification | Generated seeds/cases, independent reference and exact commands |

For multiple stages also record DAG dependencies, stage signatures, boundary layout,
producer completion/consumer readiness, saved-state schema/version and both per-stage
and end-to-end error budgets. Saved state is an output contract even when it is private
to a forward/backward pair. State whether it is materialized or recomputed and who
owns allocation and release. Required preparation is inside each released unit.

Host preparation must stay within the agreed task: do not move required kernel arithmetic
to the host or silently cast, reorder or transform inputs to make a candidate pass.
For permitted preparation, record its value/dtype/layout effects, execution location and
inclusion in the comparison and timing boundary. Include required preparation in the
delivered unit and validate against the original task inputs with an independent reference.
Even a shape-only transform must preserve the declared ABI and element interpretation;
there is no blanket whitelist of safe preprocessing operations.

Use the [authoring template](../../templates/authoring-contract.md) and
[decomposition template](../../templates/decomposition-plan.md). Neither owner has a
machine-readable contract. A library API example uses the same four files as a kernel demo;
[RFC 0012 section 3](../../../library/docs/rfc/0012-product-contracts.md) specifies them.
What a folder computes is the `formula` or `surface` in
its `metadata.json`, and what that is checked against is the code in its `reference.py`. These
narrative templates introduce a competing schema for neither.

Several sources live in one `kernel.py`, `main.py` selects between them with `--variant` (or
`--pattern`/`--mode` where that folder's axes are named that way), and each case carries the
tolerance its own arithmetic justifies, stated where the comparison is. Two sources that must not
drift apart are written out separately rather than shared, so that editing one cannot quietly edit
the other. A variant whose outputs or precision genuinely differ says so in `main.py`, next to the
comparison that reads it -- `register_groups` and `cast_formats` are the two that do.

`reference.py` exports `make_inputs(case) -> dict` and `reference(inputs) -> dict`, plus
`reference_stages(inputs) -> dict` when the folder is a pipeline; `main.py` holds `execute`, the
comparison, and the `--stages` flag where those intermediates are compared. The reference never
mutates shared inputs and never calls a simulator, inputs are regenerated from the case's seed on
every run, and an unsupported execution combination raises an explicit error instead of falling back
to the reference.

Public source uses typed `GM[dtype, dims]` and documented `GMList[dtype, dims, count]`
where applicable. Bare untyped placeholder syntax is not a new public contract.
Do not invent `shape_bindings`: symbols come from typed dimensions and explicit
scalar arguments in signature order. Verify the selected version's declarations.

Mathematical equality does not imply identical floating evaluation. Outer-axis
partitioning can preserve element ownership; fission at a materialized boundary can
preserve that boundary. Fusion, reduction splitting, recomputation, atomic merging,
cast movement and distributivity still require the declared arithmetic/precision
proof. There is no universal "all atomics are float only" restriction or fixed
minimum matmul tile across architectures and formats.

A bad contract may be corrected with source evidence, an explicit version change and
new regressions. Do not change expected outputs merely to match a failed kernel.
