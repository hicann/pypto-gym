# Current design constraints

This page lists constraints relevant to the first public source release.
Exact operation semantics belong to the linked API and RFC pages. A declaration
or design rule does not establish a device result.

| Constraint | Owner |
| --- | --- |
| Library owns compiler, runtime and APIs; kernels owns algorithms and comparisons; agent owns workflow guidance. | [Product contract](rfc/0012-product-contracts.md) |
| Generate inputs and independent references at run time; compare actual returned outputs. | [Reference contract](rfc/0003-functional-goldens.md) |
| Keep simulator findings separate from measured device behavior. | [Synchronization diagnosis](diagnosing-sync.md), [hardware diagnosis](diagnosing-hardware.md) |
| Synchronization must preserve physical capacity, participant ownership and last-reader lifetimes. | [Autosync RFC](rfc/0005-autosync-on-ir.md) |
| UB instruction bases need aligned, owned physical footprints; GM offsets follow separate contracts. | [Storage API](api/storage.md) |
| Device launchers execute on the calling machine and use its local card configuration. | [Execution API](api/execution.md) |
| A PyPTO run requires the card's physical core count before launch. | [Board runtime](../ascriptor/runtime/board.py) |
| Backend output, model checks, vendor compilation and card execution are separate evidence stages. | [Product contract](rfc/0012-product-contracts.md) |

Record a new semantic constraint in its owner specification and verify it on
the affected workload. Keep completed task narratives out of this index.
