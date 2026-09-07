# Knowledge contract

This contract connects the knowledge base to the PyPTO-Pro Stage 1–4 workflow.
[`topology-map.json`](topology-map.json) is the single machine-readable source for
topologies, property keys, reference classes, role scope and status vocabulary.

- The planner writes `KB_SELECTION.json` before Stage 1 can pass.
- The coder writes implementation claims to `KB_USAGE.json` during Stage 4.
- `pypto-pro-op-verifier` independently validates the artifacts and the referenced code.
- Every stage except Stage 2 requires verifier PASS before completion; Stage 2 requires it only when explicitly requested by the user.

There is no external evaluator dependency in this contract. `contract_version` is **3**.

## Optional input from a driver

`cases.yaml` may be supplied beside the task materials, but it is not a PyPTO-Gym output
and may be absent in a standalone run. A check that depends on it records `n/a` when the
file is absent; absence alone must not fail a class.

## Artifact locations

```
custom/<op>/                     single class
custom/<op>/<class>/             one directory per class when inputs are split
```

When artifacts are flat, `class_id` is the literal `"."`. KB paths in generated JSON are
always relative to the KB root:

| Layout | KB root |
|---|---|
| Repository checkout | `cannbot-skills/ops/pypto-pro-op-kb/` |
| Installed workflow | `$CONFIG_ROOT/pypto-pro-op-kb` |

Never record an absolute path or an installation-specific prefix.

### Cross-package links use the sibling form, which resolves in both layouts

**The rule: write the path that is correct with the KB as a sibling of each skill**, i.e.
the `cannbot-skills/ops/` layout — `../pypto-pro-op-kb/…` from `<skill>/SKILL.md`,
`../../pypto-pro-op-kb/…` from `<skill>/references/`, and the mirror image for KB→skill
links.

An earlier revision of this section said the opposite — that a separate "installed form"
(one `../` deeper) had to be written and would not resolve in a checkout. **That was
wrong, and 17 links written to follow it were broken everywhere.** Skills are installed as
**symlinks** into `cannbot-skills/ops/`, so a `..` traversal out of an installed skill is
resolved by the filesystem *physically* — it lands back in `cannbot-skills/ops/`, where
the KB is a sibling. The sibling form is therefore correct under both a lexical resolver
in a checkout and the physical resolver at runtime; the deeper form is correct under
neither.

Measured over all 41 cross-package links: sibling form 41/41 resolve repo-relative and
41/41 resolve through the installed symlink tree.

Two consequences:

- **A cross-package link that does not resolve in a checkout is broken, not "authored for
  the installed layout".** The previous instruction not to "fix" such links protected the
  defect it was written to prevent.
- **`check_kb_integrity.py` does traverse cross-package links** (`cross-package links
  resolve`). The old prohibition rested on the claim that a repo-relative scan reports
  correct links as broken; it does not — every failure it reports is a link that resolves
  nowhere.

Links *within* this KB are unaffected — the directory moves as a unit, so relative paths
between its own pages hold in either layout.

The orchestrator's `agents/*.md` are a further special case: they install to
`$CONFIG_ROOT/agents/`, which bears no fixed relation to their repo path, so no single
relative spelling works there under any arrangement.

## Reference classes

The routing map separates references by responsibility:

| Class | Meaning | Quantity rule |
|---|---|---|
| `optional_patterns` | Reusable implementation/dataflow patterns selected for this class | No fixed limit; include all and only patterns with a distinct, concrete design effect |
| `required_constraints` | Correctness or boundary rules triggered by topology, properties, target or global policy | All applicable constraints; never truncated |
| `mandatory_constraints` | Constraints applied to every matching task, with explicit `applies_to` roles | Automatically included in `required_constraints`; never truncated |

Pattern selection is controlled by relevance rather than a numeric limit. Every selected
pattern must describe a distinct design decision for the class; a merely similar operator,
an unmet precondition, duplicate guidance or generic background reading is not selectable.
Constraints remain independent of pattern selection and are never suppressed.

## `KB_SELECTION.json`

The planner derives this file from computation topology and dtype/shape properties, never
from the operator name.

| Field | Requirement |
|---|---|
| `schema_version` | Equals `contract.contract_version` |
| `op` | Operator name |
| `class_id` | Class directory, or `"."` for a flat layout; never empty |
| `topologies` | An array, possibly empty, of current keys of `topology-map.json.topologies`; consumers must read the map rather than copy the enum. Record `[]` only after routing was evaluated and the formula matched no declared topology — never force a best-fit category, and never use an empty array to mean unknown or skipped. Every actual match must be recorded and coexists, with no override and no single winner; omitting one is invalid. The union of matched topologies' routed constraints is mandatory; their routed patterns join patterns from applicable property modifiers in the candidate pool and remain subject to the optional-pattern relevance rules below. An empty topology set contributes no topology-routed references, but property, target and mandatory routing still applies |
| `properties` | Facts whose keys come from `contract.property_keys` |
| `optional_patterns` | All and only materially applicable pattern references; each has a KB-relative path, class-specific reason, distinct expected design effect and content hash |
| `required_constraints` | Every applicable constraint, without a quantity limit; each has a KB-relative path, reason and content hash |
| `no_matching_pattern` | `true` exactly when no optional pattern fits; required constraints may still be present |

Stage 1 does not pass until `pypto-pro-op-verifier` confirms the selection against the
current routing map.

## `KB_USAGE.json`

The coder records what each selected reference changed in the implementation. It does not
certify its own work.

| Field | Requirement |
|---|---|
| `schema_version` | Equals `contract.contract_version` |
| `op`, `class_id` | Match `KB_SELECTION.json` |
| `invariants` | At least one entry for every optional pattern and required constraint |
| `reference` | Exactly one path present in the selection artifact |
| `invariant` | Concrete rule derived for this class |
| `implementation.file`, `implementation.symbol` | Real implementation location when status is `implemented` or `deviated` |
| `implementation.status` | One of `contract.implementation_statuses`: `implemented`, `deviated`, `not_applicable` |
| `justification` | Required for `deviated` and `not_applicable` |

`verified` is deliberately not a coder-authored status. The verifier opens each referenced
file and symbol, checks the invariant against the code, and returns PASS or FAIL to the
orchestrator. A selected reference that changed nothing is rejected unless the verifier
accepts a justified `not_applicable` claim.

## Changing the contract

Adding or removing a topology, property key, status, role scope, reference class or quantity
rule is a contract change. Update `topology-map.json`, raise both schema versions, update this
document and the consuming skills, and keep the integrity checks green.
