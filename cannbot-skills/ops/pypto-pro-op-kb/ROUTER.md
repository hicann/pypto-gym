# KB router

Use the matching skill first, then open only the references frozen into the class's
`KB_SELECTION.json`. Installed API documentation and official examples remain authoritative
for signatures and platform-specific behavior.

| Task | Skill | Focused reference |
|---|---|---|
| Select a retained study implementation | `pypto-pro-material-explore` | [examples/kernel-index.md](examples/kernel-index.md) |
| Choose a reusable dataflow | `pypto-pro-op-design` | [patterns/pattern-index.md](patterns/pattern-index.md) |
| Check dtype and cast behavior | `pypto-pro-op-design` / `pypto-pro-op-develop` | [constraints/precision.md](constraints/precision.md) |
| Check tile shapes and memory placement | `pypto-pro-op-design` / `pypto-pro-op-develop` | [constraints/tiling.md](constraints/tiling.md), then [constraints/memory-layout.md](constraints/memory-layout.md) |
| Check vector authoring choices | `pypto-pro-op-develop` | [constraints/vec.md](constraints/vec.md) |
| Check synchronization | `pypto-pro-op-design` / `pypto-pro-op-develop` | [constraints/sync-stitch.md](constraints/sync-stitch.md) |
| Check dynamic tails | `pypto-pro-op-design` / `pypto-pro-op-develop` | [constraints/tail-validshape.md](constraints/tail-validshape.md) |
| Select A5 constraints; use limits only after target detection | design/develop/perf skills | [constraints/arch-a5.md](constraints/arch-a5.md) |
| Decide what may run on the host | `pypto-pro-op-design` / `pypto-pro-op-develop` | [constraints/wrapper-boundary.md](constraints/wrapper-boundary.md) |
| Measure and tune a correct kernel | `pypto-pro-op-perf-tune` | [playbooks/benchmark-scoring.md](playbooks/benchmark-scoring.md) |
| Localise a numerical error | `pypto-pro-op-develop` | [playbooks/numerical-error-localisation.md](playbooks/numerical-error-localisation.md) |
| Quantize per row with a scale that is also an output | `pypto-pro-op-design` | [patterns/vec-per-token-dynamic-quant.md](patterns/vec-per-token-dynamic-quant.md) |
| Hit a framework limit, or debug something that makes no sense | any | [references/pypto-pro-framework-findings.md](references/pypto-pro-framework-findings.md) |
| An API looks unsupported, or a correct-looking call returns stale/wrong data | any | [references/pypto-pro-dsl-limitations-a5.md](references/pypto-pro-dsl-limitations-a5.md) — severity-ordered, silent failures first |
| An investigation keeps failing to converge, or you are about to trust a measurement | any | [references/investigation-discipline.md](references/investigation-discipline.md) |
| A whole run failed at once (0/N), or a change "did nothing", or a number reproduces suspiciously well | any | [playbooks/benchmark-scoring.md](playbooks/benchmark-scoring.md) § traps, then `pypto-pro-environment-check` |
| Several agents are working in parallel on one shared record, or you are merging their branches | any | [references/investigation-discipline.md](references/investigation-discipline.md) §10 |
| Map a Chinese/English hardware, pipe, tiling, or layout term to its meaning | any | [references/terminology.md](references/terminology.md) |
| A page carries a cross-DSL claim and you need its provenance, or you are about to re-derive something a sibling DSL already knew | any | [references/easyasc-port-ledger.md](references/easyasc-port-ledger.md) — provenance record, not a selected pattern or constraint |

Load [constraints/arch-a5.md](constraints/arch-a5.md) when runtime/build selects A5 or when
the workflow default A5 applies; do not use its numerical limits until the exact device and source are confirmed.

## Routing by topology

[`topology-map.json`](topology-map.json) is the only maintained routing source. Route on the
shape of the computation, never on the operator name. Planner, architect, coder and verifier
must not keep their own topology enum or copy of the routing table.

For the selected topology and the properties that actually hold:

1. Collect every matching topology constraint.
2. Collect every matching property constraint.
3. Add the target-gated constraint selected by the explicit target or workflow default.
4. Add every `mandatory_constraints` entry whose `applies_to` roles participate.
5. Deduplicate these into `required_constraints`; do not truncate them.
6. Evaluate every matching pattern and retain it only when its preconditions hold and it
   contributes a distinct, concrete design decision for the class. There is no numeric limit.
7. If no pattern fits, set `no_matching_pattern: true`; this does not remove constraints.

The separation is intentional: patterns are optional reusable designs, while constraints
are obligations. Pattern relevance is enforced by explicit design impact, not by truncation.

## Stage ownership

- **Planner:** reads this map and writes the frozen per-class selection.
- **Architect:** reads the frozen optional patterns and required constraints, then turns each
  into a concrete design invariant. It does not re-route the KB.
- **Coder:** implements those invariants and records implementation locations and claims.
- **Verifier:** independently checks the selection against this map and checks usage claims
  against the real design and code. Only its PASS permits stage completion.

## Selection and usage artifacts

`KB_SELECTION.json` contains topology/properties, all materially applicable optional patterns, every
required constraint, reasons, content hashes and the explicit no-pattern outcome.

`KB_USAGE.json` maps:

    selected reference -> derived invariant -> implementation location -> implementation claim

The coder may claim `implemented`, `deviated` or `not_applicable`; it may not claim
`verified`. The verifier owns the verification verdict. Full field rules are in
[`CONTRACT.md`](CONTRACT.md).
