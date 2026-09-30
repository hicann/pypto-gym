# Migrate or refactor a project

Preserve the algorithm, precision, ABI and launch topology while giving one isolated demo
folder its own runnable entry and its own independent reference. The old snapshot is a
read-only source of evidence. It cannot remain a runtime import, reference provider or
maintenance owner. Read [ownership](../references/ownership.md) for each layer's fact owner.

1. Inventory every source file and linked dependency, including documentation,
   generators, helpers and old drivers. Read each in full and record source path,
   complete SHA-256 and actual UTC read time. Hashing is not validity review.
2. Assess each file's claims against current source, specification and measured scope.
   Choose retain/adapt/rewrite/retire/regenerate with a specific reason and destination.
   A retired workflow names its replacement; historical measurements stay dated.
3. Recover independent full/stage references and deterministic inputs. Preserve corrected
   kernel bodies, [cast/rounding boundaries](../references/precision.md), in-place initialization
   and [saved-state](decompose.md) semantics. Do not replace a missing reference with interpreter output.
4. Refactor into one demo folder of exactly four files, in the kernels checkout under
   `ascriptor_kernels/<area>/<name>`: `kernel.py` (the DSL and the host-side tables it
   needs, no torch and no launching), `reference.py` (torch only, never imports ascriptor,
   exporting `make_inputs(case)` and `reference(inputs)`), `main.py` (the `OpExec` entry, the
   precision check, and the cases as a literal `CASES` list) and `metadata.json` (navigation
   only). Fold duplicate drivers into that one `main.py`, and keep meaningful algorithm
   variants and stage checks explicit as `--variant` or `--stages` rather than as a second
   entry point. A minimal API teaching example is the other destination, and it is the
   same four files under the example contract that
   [RFC 0012 section 3](../../../library/docs/rfc/0012-product-contracts.md) specifies.
5. [Validate every leaf and composition](implement-decomposition.md), then copy the folder
   alone into scratch and run `python main.py --list` and `python main.py` there against the
   installed declared dependencies. A library API example is copied the same way and has no
   export step. A folder that will not run outside the repository it came from is not
   done. Missing cases or references fail. Expected tensors are generated at run time from the
   case's seed, never shipped as recorded data.
6. Check destination code, exact commands, links and each language counterpart. A full
   source read alone does not close a migration row. Include newly discovered resources
   in the pending queue and report actual checks and remaining work.

For a backward demo, the required forward preparation is local to the folder: it prepares its
own saved state in `reference.py` rather than importing a neighbouring forward project. For
A3, use the shared A2 family with device selection and obtain new A3 board evidence yourself.
For attention, keep the complete variant collection in the gallery. Choose a demo by what its
`metadata.json` says it is for, not by a recorded scope — there is none; any task-local copy
still names the folder it came from.

Use the [migration record template](../../templates/migration-record.md). During coordinated work,
the lead owns source snapshots, workspace switching, shared contracts and indexes,
environment binding, hardware scheduling, release pins and private publication.
