# RFC-0012: Product, API and example contracts

Status: first public source release, 0.1.0. This contract describes the
checked-in Ascriptor snapshot and its authoring workflow.

## 1. Ownership

The library owns compiler, runtime, simulator, public API declarations and
minimal teaching examples. The kernel gallery owns complete algorithms,
independent references and case comparisons. Agent guidance owns task routes
and delivery instructions. They ship together in one source snapshot.

The root [sources.json](../../../sources.json) and
[sources-index.json](../../../sources-index.json) identify the delivered bytes.
The package version is declared in [pyproject.toml](../../pyproject.toml).
No sibling checkout, wheel or external release receipt selects the runtime.

## 2. Execution and evidence

Source emission, functional simulation, pipe simulation, vendor compilation
and device execution are separate evidence stages. A generated program is not
a device result. Hardware acceptance requires the declared workload on the
assigned card, an independent reference and comparison of actual outputs.

Every device launcher executes on the machine that invokes it. The selected
`ASCRIPTOR_BOARDS` entry must describe that machine with `"local": true`;
connection fields are refused. Machine paths and credentials belong in ignored
local configuration. See the [execution API](../api/execution.md).

## 3. The example folder

A teaching example is a folder under `examples/api/` with `main.py`, a kernel
module, an independent `reference.py` and `metadata.json`. The kernel module
must not import Torch or NumPy; the reference must not import the compiler,
facade, backend or simulator. `main.py` declares cases, generates inputs,
runs the selected launcher, compares every output and exits nonzero on failure.

Run `python main.py --list` inside the folder to see its cases. `python main.py`
runs the functional simulator; `--launcher pipesim` checks the lowered pipe
model. A device launcher requires the local card environment. Examples state
their output initialization and numerical comparison rules next to the cases.

`metadata.json` and the generated [example index](../../examples/api/index.json)
provide navigation; they do not store acceptance state. A backend that refuses
a form must report the source location. Run an example on the selected backend
and device to establish a scoped result. `python tools/api_examples.py --check`
checks the example layout and index from the library root.

## 4. Source release boundary

This release distributes source under the repository root LICENSE. Backend
extensions and device environments remain independent inputs. A change to a
public API or output comparison requires updating the owning contract and
current examples. Historical release protocols and qualification receipts
are outside this first source release.
