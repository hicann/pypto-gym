# Documentation map

Start with [current status](status.md), then the document for the layer being changed.

| Need | Document |
| --- | --- |
| **Is this form restricted?** | **Five owners, none of which cites the others.** Upstream defects: [upstream report](upstream.md). Ours: [open defects](defects/README.md). What a backend prints and refuses: [CCE support](cce-support.md), [A5 coverage](a5-backend-coverage.md), [Pro support](pypto-pro-support.md), [Pro import support](pypto-pro-import-support.md). Standing constraints: [decisions](decisions.md). Device and form constraints: the owning RFC's capability table, e.g. [A2/A3](rfc/0008-a2-family.md). All five are indexed by prose subject, so a keyword that finds nothing is not a statement that nothing restricts it — say which you searched. |
| Public authoring and execution APIs | [API guide](api/README.md), [examples](../examples/api/README.md) |
| Compiler, IR and device contracts | [RFC index](rfc/README.md), [product contract](rfc/0012-product-contracts.md) |
| Effective design constraints | [Decisions](decisions.md), [terms](glossary.md) |
| Synchronization or native failures | [Synchronization diagnosis](diagnosing-sync.md), [hardware diagnosis](diagnosing-hardware.md) |
| Current implementation defects | [Open defects](defects/README.md) |
| Backend support and upstream gaps | [A5 coverage](a5-backend-coverage.md), [upstream report](upstream.md) |
| Reproduce local PyPTO-Pro supplements | [Dependency supplements and checks](pypto-pro-supplements.md) |
| Pro reverse-import rules and evidence | [Import support matrix](pypto-pro-import-support.md) |
| Why a Pro form is not admitted | [Import gap classes](pypto-pro-import-gaps.md) |
| Measure or model performance | [Measurement procedure](perf.md), [parameter meanings](performance-parameters.md) |
| Kernel workflows | [Bilingual agent guides](../../agent/README.md) |
| Source release and validation scope | [Current status](status.md), [source manifest](../../sources.json) |

The first public source release provides no general device qualification.
Run the requested workload on its assigned device and retain the result with
the snapshot source ID.

The delivered source snapshot is checked in place with
the plugin snapshot checker in `scripts/sync_ascriptor_sources.py`.
The source index and manifest live at the snapshot root; no wheel build is part of
this author workflow.
