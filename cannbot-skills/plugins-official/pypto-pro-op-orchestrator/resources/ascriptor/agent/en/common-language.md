# Common language for kernel work

Read once after the router in each kernel context. Use this vocabulary to fill
[preflight](references/authoring-preflight.md); first-time authors also read the
[execution model](concepts.md). Exact operations belong to the [library API](../../library/docs/api/README.md).

| Term | Working meaning |
|---|---|
| Facade / device profile | Authoring vocabulary / compiler and model target; `ascriptor.a5` uses canonical ID `950` |
| Backend / launcher | Source-generation target / execution mechanism |
| Stage / launch | Internal dataflow step / one runtime invocation |
| Core owner / work item | Participant responsible for a logical output region / its assigned work |
| Logical extent / footprint / allocation | Useful elements / actual instruction addresses / owned backing bytes |
| GM / UB / L1 / L0A,B,C | Public global storage / vector local storage / cube staging and operands/accumulators |
| View or reinterpret / cast | Address or bit interpretation / numeric conversion with rounding |
| Var / Reg / MaskReg | Mutable scalar cell / VF register / lane predicate |
| Slot / event credit | Physical storage version / permission to publish or reuse it |
| Producer / last reader | Writer / final physical read that must retire before reuse |
| Pipeline warmup / steady state / drain | Fill the schedule / repeat its regular work-item schedule / consume remaining valid work and publish final outputs |
| Precision boundary | Ordered accumulation, materialization, cast, rounding and saturation choices |
| Criteria met / headroom | Agreed exit condition satisfied / further possible improvement |

For mixed pipelines, derive stage work indices, lookahead and drain from dependencies
and available slots. CVC/VCV name a graph sharing physical resources; the name supplies
neither independent engines nor a buffer depth. Use the [pipeline method](references/pipeline-model.md).
Map physical access with [storage](../../library/docs/api/storage.md) and
[memory boundaries](references/memory-and-tails.md#vector-tail). A narrow view supplies
an address range; choose the instruction predicate and footprint explicitly.

<a id="evidence"></a>
## Evidence and conclusions

| Claim | Evidence to retain |
|---|---|
| Declared operation or supported tuple | Current owner declaration and exact device/dtype/layout/case/backend scope |
| Source generated / vendor compiled / device executed | Separate emit, compile and run records with actual source/dependency identities |
| Correct arithmetic | Independent runtime-generated reference, complete outputs and rejected incorrect-output controls |
| Correct synchronization | Balance plus lowered physical accesses, hazards, deadlock, ownership and last-reader lifetimes |
| Compute overlap | Actual same-core stage/item intervals with DMA and synchronization excluded |
| Hardware performance | Same-device measurements; model cycles, CPU time and diagnostic captures stay in their own units |
| Guidance effect | Scoped fresh-context trials; comparable repeated baseline/candidate trials for an improvement claim |

Use this table when reporting results. Missing evidence stays unknown; a failed
candidate or absent catalogue entry leads to a focused source/probe investigation.
For performance work, separate unique, requested and measured HBM bytes and use
[Roofline](references/roofline.md). Topic references retain their specific constraints
and counterexamples. Each checker or experiment states what it established and
which stages or domains remain untested.
