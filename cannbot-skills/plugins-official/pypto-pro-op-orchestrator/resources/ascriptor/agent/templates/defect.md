# <ID>: <observable defect>

Status: suspected | confirmed | fixed-candidate | released | not-a-library-defect.
Owner: <library layer>. Affected contract/API: <version and domain>.

## Expected and actual

State the exact behavior difference, IO, shape/dtype/layout, initialization, device
and backend. Classify suspected library defect, confirmed defect, kernel error,
upstream gap or untested claim; update the classification as evidence develops.

## Reproduction

Unit-local generated input and independent reference: <source>.
Seed and parameters: <values>. Exact command and actual exit/result: <record>.
Library/package/source revision, Python/dependency and relevant toolchain versions: <versions>.
Imported implementation path: <successor package/checkout, no machine access values>.

## Diagnosis

First incorrect stage/op/source location: <evidence>.
Model-derived observation: <rule, source revision/location, experiment>.
Silicon-measured observation: <separate device/toolchain/domain/result or unmeasured>.
Root cause and affected domains: <analysis>. Temporary instrumentation: <removed or justified>.

## Repair and closure

Workaround: <scope and removal condition>.
Specification/implementation/declaration/example changes: <paths and reason>.
Generated regression and original-case rerun: <commands and actual results>.
Candidate tested combination: <library, kernels, agent revisions>.
Fixed version: <pending until released>. Follow-up owners/links: <records>.
Preserve meaningful failure guards and comparisons; no reduced gate is a fix.
After closure, follow the owner retention rules:
retain regressions and necessary evidence, migrate current rules and remaining
restrictions, update consumers, verify Git recovery and remove the closed record.
Do not reuse its ID.
