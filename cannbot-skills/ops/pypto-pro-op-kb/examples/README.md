# Validated study kernels

[kernel-index.md](kernel-index.md) is the only production selector. Every row
has:

- a general topology or technique;
- a canonical implementation path;
- `validated` state;
- reviewable evidence in the implementation itself or a companion golden.

Do not add failed, work-in-progress, diagnostic, or benchmark-only probes to
the selector. Re-run the retained validation on the actual target before
adopting a sample in production; a historical validation does not establish
compatibility with a different PyPTO/CANN version or platform.
