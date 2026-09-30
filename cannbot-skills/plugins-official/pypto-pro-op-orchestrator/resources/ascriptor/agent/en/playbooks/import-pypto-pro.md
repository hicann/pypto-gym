# Import a PyPTO Pro kernel

Convert a specialized Pro device kernel into verified `lowered/1` without re-planning it.
The import preserves typed operands, per-participant order, control flow, physical storage,
numerical modes and synchronization. It is not ordinary lowering: it MUST NOT reallocate
memory, replan synchronization, renumber events or coalesce instructions. The contract is
[RFC-0015](../../../library/docs/rfc/0015-pypto-pro-import.md); what is already admitted is the
generated [import evidence index](../../../library/docs/pypto-pro-import-support.md), whose
direction is Pro → IR and whose baseline is the opposite-direction
[backend support](../../../library/docs/pypto-pro-support.md). There is no command-line route:
`ascriptor dump-ir` reads IR, not a Pro bundle.

1. Export once, import anywhere. Export needs Pro installed and the pinned profile; import
   needs neither Pro nor torch. The bundle JSON is the artifact that crosses that boundary.

   ```python
   from ascriptor.importers.pypto_pro import dumps, export_kernel
   bundle = export_kernel(pro_jit_kernel, directions={"x": "input", "y": "output"}, block_dim=1)
   ```

   `directions` are explicit caller annotations and are never guessed from a parameter name;
   an `inout` parameter forces `seed_outputs=True` on every later executor. The bundle records
   the Pro source file name and its SHA-256, so a stored export stays checkable against the
   source it came from.

   Both `directions` and `block_dim` go into the bundle's `abi` exactly as you pass them.
   Export checks only that a named parameter exists; it derives nothing from the kernel body
   and validates nothing against it, and a parameter you leave out is recorded as `unknown`.
   So they come from how the caller launches and uses this kernel, not from reading its
   source — and a wrong one is not a refusal, it is a bundle that imports cleanly and means
   something else.

2. Import, then read the accounting rather than the printed text.

   ```python
   from ascriptor.importers.pypto_pro import import_kernel, loads, prepare_import
   entry = import_kernel(loads(text))
   ```

   `entry.ir().attrs["import_ledger"]` holds one row per source node with its `target_ids` and
   a `translated` or `declaration` disposition. RFC-0015 requires every admitted operation to
   be accounted for there, so that ledger — not a diff of emitted code — answers where a source
   line went. `prepare_import(bundle).operations` lists the source operations before any target
   exists, and `ImportPlan.require_converters` rejects an unhandled one at its span.

3. A refusal is a result, not an obstacle. `ProImportError` names the Pro span, as in
   `pro_p6_vf_memory.py:362:9: VF memory access inside VF control flow needs its own cursor and
   footprint rule`. MUST NOT edit the exported bundle, relax the verifier or approximate the
   form to get past one. Place it with its owner instead:

   | What the refusal names | Where it belongs |
   |---|---|
   | A Pro form with no IR opcode | Extend owner RFC, registry, verifier, model and backend together through maintain |
   | An IR opcode with no converter | The importer and RFC-0015's acceptance section, with a new export fixture |
   | A real form whose silicon semantics are unmeasured | Keep the refusal; measure first, admit second |
   | Pro's own defect or undocumented behaviour | One `A5-UP-*` entry in `library/docs/upstream.md` |
   | Our own wrong conversion | `library/docs/defects/` through the [debug](debug.md) route |

4. Validate in a fixed order and keep the stages distinct: functional model, pipe model with
   hazard and deadlock reporting, printed backend artifact, then device. One stage never stands
   for another, and a warning about correctness, synchronization or support scope blocks the
   claim it touches until the [debug](debug.md) route resolves it.

   ```python
   from ascriptor.backends.sim.pipesim import simulate
   from ascriptor.runtime import compile_kernel

   entry.executor(launcher="sim")(*tensors)
   simulate(entry.ir(), tensors, processes=False, check_gm=True)
   compile_kernel(entry, backend="cce")
   ```

5. Prove the stored bundle is still the export. Re-export the same Pro kernel and compare with
   `importers.pypto_pro.session.same_export`. Bytewise equality is the wrong check and fails
   every case after a profile revision: the recorded `producer` may name an accepted predecessor
   of the pinned profile, which carries the same content under an earlier identity.

6. Device acceptance is the native Pro kernel against the imported CCE artifact on the same
   generated inputs, under the assigned card and lock, following the hardware-first procedure in
   [runtime](../runtime-and-maintenance.md#hardware-first). Model agreement does not satisfy it.

Admitting a new form requires a real Pro export fixture, accepting and refusing
checks, and a regenerated [support index](../../../library/docs/pypto-pro-import-support.md).
The fixture and its checks must be added to the work area before claiming that form.
A hand-written bundle is evidence
of nothing. Record what stayed refused with its span and reason: the index counts refusals, and
an unexplained one is an open work item rather than a closed boundary.
