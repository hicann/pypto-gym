# Specifications

Code follows the applicable RFC. Amend an incorrect contract with its rationale
before implementing a semantic change. [Current status](../status.md) and
[RFC-0012](0012-product-contracts.md) define the product/support boundary;
implementation notes and dated measurements inside older RFCs retain their original scope.

| RFC | Contract |
| --- | --- |
| [0001](0001-ir.md) | IR types, operations, forms, provenance and verification |
| [0002](0002-frontend-static-subset.md) | Python frontend, signatures and diagnostics |
| [0003](0003-functional-goldens.md) | Generated inputs, independent references and comparisons; supersedes stored goldens |
| [0004](0004-board-batch-runner.md) | Current board execution, isolation, transfer identity and failure boundaries |
| [0005](0005-autosync-on-ir.md) | Synchronization dependencies, lifetimes and physical event limits |
| [0006](0006-lowering-pipeline.md) | Lowering passes and pipeline simulation |
| [0007](0007-cce-backend.md) | CCE backend |
| [0008](0008-a2-family.md) | A2 family implementation; current A2/A3 CCE scope is defined by RFC-0012 |
| [0009](0009-slot-buffers.md) | Slot buffers and workspace lifetimes |
| [0010](0010-gm-strided-views.md) | Strided GM views and DMA consumption |
| [0011](0011-pto-isa-backend.md) | Current PTO ISA geometry, movement, synchronization and refusal contracts |
| [0012](0012-product-contracts.md) | Product ownership, public APIs, installation and release contracts |
| [0013](0013-pypto-native-synchronization.md) | Optional PyPTO-Pro native synchronization and admission limits |
| [0014](0014-register-radix-topk.md) | Register radix TopK composite API and its bounded selection contract |
| [0015](0015-pypto-pro-import.md) | Proposed instruction-preserving PyPTO Pro import into Lowered IR |

For operational board checks use [hardware diagnosis](../diagnosing-hardware.md)
and the [runtime API](../api/execution.md). Backend status belongs in
[A5 coverage](../a5-backend-coverage.md) and [upstream limitations](../upstream.md).
Effective cross-repository constraints are indexed in [decisions](../decisions.md).
