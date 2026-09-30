# RFC-0003: Generated-reference verification

Status: the D-241 reference-based contract below supersedes the prototype's
stored-golden design. The filename and section 9 anchor remain stable for existing
references. Superseded sections 1–8 are preserved in Git history
under this document's original path; they are not current recording requirements.

## 9. M10 reference-based verification (D-241, 2026-09-06)

This section governs the library/kernel/agent repositories described in
[RFC-0012](0012-product-contracts.md). Historical prototype recording and replay
protocols do not impose requirements on these repositories.

Each public file/project unit contains input generation, an independent Torch/NumPy reference,
a runner and a comparison contract. Expected outputs are calculated during the run. Saved input/
output tensors, recording indexes and replay archives are not installation, migration or release
prerequisites. Small automatic tests and generators remain public.

Record seeds, shape/dtype parameters, source/toolchain versions and comparison rules with their
reasons. Preserve meaningful aligned/tail, multi-tile, initialization and supported core-count
coverage, plus numerical and synchronization assertions. A missing reference/generator or empty
case set fails the applicable check; do not widen tolerances or claim unexecuted checks as passed.
An interpreter recording alone is not independent evidence of mathematical semantics.

Source migration snapshots preserve code, documentation, references, generators and needed local
source changes, excluding golden payloads, old Git object stores, build caches and bulk raw logs.
No golden backup/readability/replay gate remains. Concise result summaries and defect reproducer
code provide maintenance evidence without retaining complete tensor dumps or recordings.

Kernel comparison/support metadata has one maintained owner in the kernel repository and is
passed explicitly to library execution helpers. Successor checks run from the new source versions
and generated inputs, without importing the retired prototype or requiring recorded datasets.
