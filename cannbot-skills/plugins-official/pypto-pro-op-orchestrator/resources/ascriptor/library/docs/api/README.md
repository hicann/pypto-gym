# Public authoring API

A2/A3 CCE uses `ascriptor.a2`/`ascriptor.a3`; A5 uses `ascriptor.a5`.
The version is in [pyproject.toml](../../pyproject.toml). Exact operation domains
and device results require a current workload check. Imports do not select a
process-wide device.

Read the [reference](reference.md) for expressions, signatures and boundaries, and the
[example guide](../../examples/api/README.md) for independent runnable checks. The
[manifest](manifest.json) enumerates every accepted facade name, dtype/enum member,
descriptor method and operator family. It distinguishes a tested example, an informational
declaration and a located gap. A declaration is not a hardware support claim.
This snapshot carries the reviewed manifest alongside the facades. For a new support
claim, update the declaration and run a direct source and device check for that form.

The adjacent package `.pyi` files carry concrete parameters and defaults checked against
frontend rules. They describe this AST-compiled language; they do not turn DSL marker values
such as `GM[f32, ("M", "N")]` into ordinary Python generic types. They are informational
declarations, with no `py.typed` or whole-program type-checker certification in this batch.
The four device facade stubs describe the runtime facades in this snapshot.

Execution and source emission use `ascriptor.runtime.OpExec` and `compile_kernel`.
Inspection uses kernel `.ir()` and the documented CLI. The complete compatibility boundary
is [RFC-0012](../rfc/0012-product-contracts.md).

| Topic | Current guide |
| --- | --- |
| Functions, loops and scalar cells | [Authoring](authoring.md) |
| Storage, immutable views, lists and DMA | [Storage](storage.md) |
| Local events and cross-side ownership | [Synchronization](synchronization.md) |
| A5 register expressions and predicates | [Registers](registers.md) |
| Packed formats, casts and independent host codecs | [Formats](formats.md) |
| Cube tiles, bias, quantization, MX and convolution | [Cube](cube.md) |
| Record sorting, register radix TopK and tie validation | [Sorting](sorting.md) |
| SIMT math and atomic operations | [SIMT](simt.md) |
| Launchers, inspection and backend extensions | [Execution](execution.md) |

Device-specific contracts: [A2/A3 vectors](a2-vectors.md). Verify the
selected device, backend and workload before reporting support.

The [example index](../../examples/api/index.json) is generated navigation and records no evidence;
the snapshot index ties source documentation to exact file digests. Current device
measurements require their own workload and environment record.
