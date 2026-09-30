# Ascriptor task execution contract

These rules govern agents using Ascriptor to deliver a user's task. MUST and MUST NOT
mark required actions and boundaries. Apply each gate when its stated trigger occurs.
If a prerequisite is unresolved, stop the work that depends on it, record the gap,
and continue independent investigation. Infer routine choices from the task and
owner contracts; ask the user only for missing intent or constraints that affect the result.

Repository maintenance also follows the shared owner maintenance rules.

## 1. Before starting a task

- For kernel authoring, substantial edits, decomposition, debugging or optimization,
  MUST start with the [compact kernel context](context/kernel-authoring.zh-CN.md)
  once, unless the same guide is already embedded in the initial task message.
  Read it directly or use `python tools/build_kernel_context.py --print` from the
  selected agent directory, whether a sibling checkout or an installed product view.
  Reading the guide requires no source-hash check or refresh. If absent, use the
  router and owner pages below. Included material counts as read;
  do not reopen it without a task-specific reason. The compact guide is a starting
  map, not evidence that a particular operation or backend form is supported.
- A different task-domain context embedded at startup is usable within its declared
  scope after its source identities are confirmed. Do not add a file read merely to
  duplicate already embedded material.
- MUST choose the [English](en/ROUTER.md) or [Chinese](zh-CN/ROUTER.md) route.
  The current compact guide's route map, common language, authoring path and
  general preflight table satisfy those initial readings for ordinary Chinese
  single-kernel authoring. Select that route and complete the task-specific
  preflight decisions without reopening the same router or author page. Other
  routes MUST read their selected playbook, and the English route MUST read its
  English router and playbook. All kernel work still follows triggered references
  and one relevant runnable example; read owner pages when a decision or form needs
  detail, or when the guide is stale.
  MUST use selectors or indexes to locate relevant entries, not preload the full catalog.
- MUST establish the objective, allowed changes, deliverables and completion criteria
  from the user's request. Pure documentation work enters the maintenance route directly.
- Turning an existing PyPTO Pro kernel into Ascriptor IR MUST enter
  [import (English)](en/playbooks/import-pypto-pro.md) or
  [import (Chinese)](zh-CN/playbooks/import-pypto-pro.md). A refusal MUST be reported at the
  Pro source span it names, and MUST NOT be worked around by editing the exported bundle.
- MUST select the source identity from this snapshot's `sources.json` and
  `sources-index.json`. Use library, guidance and gallery from the same tree.
  Historical qualification records do not prove a new claim.
- When coordinating multiple agents, MUST follow the coordination rules in
  [ownership (English)](en/references/ownership.md#coordination) or
  [ownership (Chinese)](zh-CN/references/ownership.md#coordination) before assigning work.

## 2. Before implementing or substantially changing a kernel

- MUST complete [preflight (English)](en/references/authoring-preflight.md) or
  [preflight (Chinese)](zh-CN/references/authoring-preflight.md) before implementation
  or an architecture change. The current compact guide provides its general decision
  table; follow triggered owner links for the actual operation and device. First-time
  authors MUST also read that language's concepts.
- MUST record the applicable decisions and their basis using the task contract in
  ignored `tmp/<task>/`. Unresolved semantics, launch/ABI, ownership, storage, precision
  or synchronization decisions block the implementation that depends on them.
- MUST verify operations and forms against owner API contracts. Existing kernels are
  evidence whose assumptions must be checked, not proof of support for a new form or domain.
- MUST settle tiling and dataflow, then build and validate an incremental candidate.
  Operations, casts, buffers, synchronization edges and data movement MUST have a basis
  in the task contract and device behavior; follow the preflight's implementation rationale.
- MUST NOT alter the task through unagreed host computation, casts, reordering or input
  preprocessing. Permitted preparation MUST be covered by the delivery and comparison
  contract; follow the chosen language's authoring-contract reference.
- MUST follow the preflight's pipeline analysis when repeated mixed pipelines occur,
  including buffer lifetimes, lookahead and drain; retain the reasoning with the task.

## 3. Before executing and while validating

- How to run what you are writing — the `OpExec` calls, how a script reaches a card and what
  `ascriptor doctor` checks — is one page: [English](en/references/development-execution.md) or
  [Chinese](zh-CN/references/development-execution.md). The rules below govern it.
- Before compile/run, MUST verify the assigned accepted Python environment, imported
  library version/path and selected source identity. Source archives and easyasc are
  read-only migration inputs and MUST NOT become runtime dependencies.
- Hardware tasks MUST follow the full-workload hardware-first procedure, its explicit
  cost exception and diagnostic limits in [runtime (English)](en/runtime-and-maintenance.md#hardware-first)
  or [runtime (Chinese)](zh-CN/runtime-and-maintenance.md#hardware-first).
  MUST check assigned device status, hold the required lock and isolate outputs.
  If hardware is unavailable, report that validation as blocked; model results do
  not satisfy hardware acceptance.
- PyPTO-Pro tasks MUST follow the selected language's runtime guide for synchronization
  mode selection and its required closeout attempt before delivery.
- MUST generate inputs and independent references at run time, compare actual returned
  outputs, and retain meaningful numerical and synchronization checks. Simulator runs
  MUST use source files, bounded diagnostics and the guide's valid shape/core selection.
- MUST keep temporary runners and reports in ignored `tmp/<task>/`. Machine values
  belong in external ignored configuration selected by `ASCRIPTOR_MACHINE_SPECS` and
  `ASCRIPTOR_BOARDS`; they MUST NOT enter committed sources, tests, documents or evidence.
  Redact board output before writing shareable evidence.

## 4. When a check fails, warns or a contract is unclear

- Warnings affecting correctness, synchronization or support scope MUST be investigated
  through the debug playbook. MUST NOT suppress a warning to claim success; unresolved
  warnings block the affected acceptance claim and MUST be reported with their scope.
- MUST retain the failure and a located reproducer, then use the debug route to identify
  the owning layer. MUST NOT weaken tolerance, omit required cases, replace the requested
  backend or relax a completion criterion merely to obtain a pass.
- MUST resolve API/specification/defect questions with library, algorithm/comparison
  questions with kernels, and workflow questions with agent. Follow owner contracts;
  do not invent a replacement rule when guidance conflicts with them.
- Missing paths or symbols MUST be checked against current indexes and the selected
  owner sources. MUST NOT guess replacement APIs or infer lack of support from a stale
  link or unsuccessful lookup; record any unresolved gap with its version and location.
- Repairs to library or guidance MUST enter maintenance (English)
  or maintenance (Chinese) and follow the owning repository's
  rules. A conflict that cannot yet be resolved blocks dependent claims, not all investigation.

## 5. Before declaring completion

- MUST verify the delivered artifact against the agreed objective, allowed changes and
  completion criteria, including original full workloads when hardware acceptance is required.
- MUST report artifact/source identity, relevant environment, executed cases and stages,
  observed results, and remaining failures or blocked checks. Keep reference, functional
  simulation, pipe simulation, source emission, vendor compilation and device execution
  distinct. Performance claims require measurements of the selected artifact and workload.
- MUST NOT claim a stage, input domain, backend or target passed beyond the evidence
  actually obtained. Incomplete required validation means the task is not fully verified.
- On handoff, MUST preserve decisions, evidence locations and the next unresolved boundary
  in the scratch [workflow checkpoint](templates/workflow-checkpoint.md).
