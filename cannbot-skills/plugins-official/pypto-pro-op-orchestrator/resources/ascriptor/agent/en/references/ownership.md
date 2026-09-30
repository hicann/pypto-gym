# Find the fact owner

Use this snapshot's source index to verify the library, agent and kernel gallery
together. Paths below are relative to their named owner.

| Fact | Owner |
|---|---|
| Public authoring names/signatures | Library facades and adjacent declarations |
| Frontend constraints | Library `ascriptor/frontend/rules_*.py` and static-subset RFC |
| IR verification, semantics and provenance | Library `ascriptor/ir/`, `ascriptor/passes/`, RFCs |
| Backend/source emission | Library `ascriptor/backends/` and backend RFC |
| Execution/build/board behavior | Library `ascriptor/runtime/` |
| Functional/pipe model | Library `ascriptor/backends/sim/` |
| Example contract | Library `docs/rfc/0012-product-contracts.md` (section 3) |
| What a kernel computes, its cases and its comparison | The folder, in either owner: `metadata.json`, `main.py`, `reference.py` |
| Whether a kernel runs on a backend | Nothing records it. Run that folder's `main.py` with that launcher, on a machine with the card |
| API teaching examples | Library `examples/`; selected attention records its canonical source |
| Defects | Library `docs/defects/` |
| Routes and task guidance | Agent language router and playbooks |

Guides may show small workflow excerpts generated from an owner's named function.
[Code anchors](../../docs/code-anchors.json) and `tools/sync_snippets.py --check`
verify the excerpt against the selected library. Keep full imports, signatures,
domains, references and executable examples with the owner; do not hand-edit a
second implementation in the guide.

Confirm source origin before using a result:

```bash
python -c 'import ascriptor; print(ascriptor.__version__); print(ascriptor.__file__)'
```

The path should resolve inside this snapshot's `library/ascriptor/` directory.
The lead selects the local Python environment. A facade import binds names
without changing process-global device state; decorated kernels retain their target.
Do not import A5 register helpers to implement A2 tensor-vector operations.

Public execution/inspection uses `ascriptor.runtime.OpExec`, `compile_kernel`,
kernel `.ir()` and documented CLI commands. An `OpExec` return owns the actual returned
output; do not assume the input placeholder was updated on every launcher. Explicit
output seeding matters for in-place/atomic contracts. `sim` is functional simulation and
`pipesim` is the separate lowered simulation route; both are `OpExec` launchers, and a pass
on one is not a pass on the other.

An unprintable operation reports a source-located gap. There is no inline-text IR
operation to bypass verification. Advanced backends use the declared `Backend`,
`Artifacts`, `Capabilities`, `ResourceLimits` and `ascriptor.backends` entry-point
protocol; package and IR versions are independent. Read the library contract before
extending it rather than exporting compiler implementation helpers as DSL APIs.

Local machine configuration stays in ignored external files selected by
`ASCRIPTOR_MACHINE_SPECS` and `ASCRIPTOR_BOARDS`. Do not put its values into guidance,
unit metadata, logs for publication or commit messages. Hardware evidence names
device/profile and relevant versions, not access coordinates.

<a id="coordination"></a>
## Coordinated work

When multiple agents share a task, only the lead may bind or rebind environments,
manage hardware locks and integrate branches/shared changes.
Workers MUST stay within assigned paths, return shared changes to the lead and MUST NOT
spawn workers. Each worker MUST run at most one heavy test process unless the lead
explicitly changes the resource budget. Device runs MUST use isolated output identities.
