# Choose one task

Use this snapshot's [sources.json](../../sources.json) and
[sources-index.json](../../sources-index.json) to identify the complete source tree.
For kernel authoring, substantial edits, decomposition, debugging or optimization,
start with the [compact context](../context/kernel-authoring.zh-CN.md)
as [AGENTS.md](../AGENTS.md) directs. Its shared-language summary supplies the
initial read; if absent or stale, read [common language](common-language.md).
Then choose one playbook. The compact guide is in Chinese, so follow the English
playbook and owner pages for the selected English route.
Before implementation follow [preflight](references/authoring-preflight.md), its
triggered facts and one working example. First-time authors also read
[the execution model](concepts.md). Choose one language; a full RFC archive or
the whole gallery is not the starting baseline. Pure documentation maintenance can use its route directly.

Find operations and signatures through the [API entry](../../library/docs/api/README.md).

**To find an example there is one route, and it starts with one decision.** A primitive goes to the [runnable API examples](../../library/examples/api/README.md); a complete algorithm goes to the [kernel gallery](../../kernels/README.md), whose `index.json` you filter on `topology`, `device`, or words in `formula` and `tags`. Either way the candidate's own `study_for` and `do_not_copy_when` say what is worth studying in it and when not to copy it — read those before you open any source, and read one relevant entry rather than the catalog. When the only word you hold is a phrase or a symbol you just typed, `python tools/select_example.py --query '<phrase>' --device a5` in the agent checkout walks that same route for you through the [topic index](../index/README.md) — name the family you are writing for, since the default is `a5` and a demo declaring another family is listed only under `rejected`; the pages below that offer you an example are pointing back here, not opening a second door.

| Task | Playbook | Result |
|---|---|---|
| Write one kernel from a formula, reference or model | [Author](playbooks/author.md) | One agreed launch with independent reference; for a PyPTO-Pro target also read the topic below and deliver a [delivery area](references/pypto-pro.md#delivery-area) |
| Plan several runtime kernels | [Decompose](playbooks/decompose.md) | Executable reference DAG and stage ABI |
| Implement an existing decomposition | [Implement decomposition](playbooks/implement-decomposition.md) | Checked leaves and composition |
| Diagnose wrong output or a hazard — or decide whether a kernel you were handed is finished | [Debug](playbooks/debug.md) | Minimal reproduction, diagnosis and regression; for a handover, each stage's verdict separately and what is still unestablished |
| Improve an already correct kernel | [Optimize](playbooks/optimize.md) | Comparable measurements under the same contract |
| Fix library implementation or guidance | Maintain | Owning-layer repair and defect update |
| Move or refactor an existing project | [Migrate](playbooks/migrate.md) | Per-file dispositions and isolated unit |
| Turn an existing PyPTO Pro kernel into IR | [Import](playbooks/import-pypto-pro.md) | Verified Lowered IR, a re-checkable export and located refusals |
| Reimplement a vendor AscendC operator on another family | [Port](playbooks/port-vendor-operator.md) | Semantics recovered from three upstream sources and a re-derived kernel with its own evidence |

For a new A5 PyPTO-Pro task limited to functional/precision validation of an official AscendC
source case, use the [Port source-only handoff](playbooks/port-vendor-operator.md#source-only-handoff).
Existing formal Scriptor tasks and requests for the full workflow keep their original entry.
Do not route a source-only handoff to `pypto-pro-op-develop`, which requires a frozen formal design.

A2/A3 CCE uses `ascriptor.a2` and `ascriptor.a3`; read the
[A2/A3 family guide](references/a2-a3.md) before using their tensor-vector
vocabulary. A5 provides register/VF and SIMT interfaces. Check the selected
backend and hardware against the task workload.

Focused knowledge: [device facts and their rules](references/facts-device.md),
[contract](references/authoring-contract.md),
[memory and tails](references/memory-and-tails.md#vector-tail),
[synchronization](references/synchronization.md), [precision](references/precision.md),
[simulator internals](references/simulator-white-box.md),
[ownership and source lookup](references/ownership.md),
[patterns](references/patterns.md), [terms](references/terms.md).
| Current question | Direct evidence path |
|---|---|
| **How do I run this kernel at all, and on hardware?** | [Running what you are writing](references/development-execution.md) — read this one first; it is a page |
| **The numbers are exact — is it done?** | No: [the evidence table](common-language.md#evidence) says what each stage establishes, and correct arithmetic is one row of it |
| **The target is PyPTO-Pro; what is different?** | [Targeting PyPTO-Pro](references/pypto-pro.md) |
| How do attention math, layout and online state fit together? | [Attention authoring](references/attention-authoring.md), then one exact canonical case. It is a long page bound to the MLA demo: if what you want is the online-softmax recurrence and its ordering rules, go straight to [Freeze the numerical sequence](references/attention-authoring.md#freeze-the-numerical-sequence), and for an example of it use the softmax row of [numerical patterns](references/numerical-patterns.md) |
| Where does the time go? | [Roofline and cost template](references/roofline.md) |
| How does a cube result reach the vector side? | [Device facts](references/facts-device.md#the-drain-into-the-vector-side-has-no-safe-default), then [Cube API](../../library/docs/api/cube.md#draining-l0c-into-the-vector-side) |
| Which calls hand a buffer to the other side? | [Cross-side handoff](references/cross-side-handoff.md), then [cross-side ownership](../../library/docs/api/synchronization.md#cross-side-ownership) |
| Is a busy input pipe doing repeated work? | [Traffic and reuse](references/roofline.md#worked-cvc-example-remove-work-as-well-as-schedule-it) |
| How should a repeated CVC/VCV/CVCV/VCVC graph overlap? | [Generic pipeline method](references/pipeline-model.md), then [CVC derivation](references/cube-vector-cube.md) |
| When can this slot be overwritten? | [Synchronization](references/synchronization.md) and the [concrete lifetime table](references/cube-vector-cube.md#concrete-lifetime-table) |
| Which primitive/example fits this boundary? | [Numerical patterns](references/numerical-patterns.md) names the pattern; the [example selector](references/patterns.md#select-one-example) is the one example route above, not a second one |

[中文入口](../zh-CN/ROUTER.md)
