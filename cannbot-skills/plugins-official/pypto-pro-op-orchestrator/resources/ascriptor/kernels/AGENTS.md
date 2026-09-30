# Kernel gallery maintenance

Start with [README.md](README.md). This repository is a gallery of runnable demos, not a
product surface: it holds kernels and the means to run them, and records no development state.

## The shape, which is the contract

A demo folder is exactly `kernel.py`, `reference.py`, `main.py` and `metadata.json`. Nothing
else — no README, no contract, no receipt, no test directory, no scratch file.

- `kernel.py` is the DSL and the host-side tables it needs. No torch, no launching.
- `reference.py` is torch only and **never imports ascriptor**. That independence is the whole
  value of the comparison: a reference that calls the thing it is checking checks nothing. It
  exports `make_inputs(case)` and `reference(inputs)`, plus `reference_stages(inputs)` when the
  demo is a pipeline.
- `main.py` is the OpExec entry and the precision check, with the cases as a literal `CASES`
  list (or a comprehension over a rule, when the matrix is regular enough that the rule is more
  readable — say so in a comment, and check the generated ids against what you intended).
- `metadata.json` is navigation only: `id`, `title`, `formula`, `device`, `topology`, `tags`,
  `study_for`, `do_not_copy_when`. No status, no evidence, no backend field.

`id` is a stable handle, not a second spelling of the path. Nine of them predate this repository's
shape and do not match their folder — `projects/a5/kda_bwd` is `a5.kda_bwd`, `examples/a2_vector`
is `a2-family.masked_scale` — and they stay that way because release receipts in the library name
them. **Do not rename one to tidy it up**, and do not derive a new one from anything but the
convention its neighbours use. `build_index.py` checks that ids are unique, not that they are
predictable; the path is what always answers "which demo is this".

`python tools/build_index.py` regenerates `index.json` and enforces all of that, including the
`device` and `topology` vocabularies. Run `--check` before you call anything done.

## Rules with reasons

**A demo folder imports nothing from this repository.** Copy it to `/tmp` and run it; if that
works, it is self-contained. A shared helper between two folders is a copy in each, not an
import — repetition is cheaper here than a dependency a reader has to go and find.

**Outputs are poisoned before the launch and seeded into it** (`seed_outputs=True`, destinations
filled with NaN or a byte pattern). A lane the kernel never writes must be distinguishable from
a lane it correctly wrote a zero to.

**A case is never widened, shrunk or deleted to make it pass.** If it cannot run where it is
being run — a full model shape under the simulator, a located disagreement between the model and
silicon — it is skipped with the reason printed, and the reason says what to run instead. A whole
demo may refuse one launcher the same way, printing its reason and exiting zero;
`tools/run_all.py` reports that as `skip`, which is neither a pass nor a failure. A
tolerance describes the arithmetic that justifies it, not the failure it was raised to cover.

**A backend that cannot run a demo gets a comment in that demo's `main.py`** saying what is
refused and where it is located, so the next reader finds it next to the code it is about.

**Flattening modules changes numbers.** When two kernel files become one, a constant with the
same name in both shadows, and a constant with the same *text* in both can be a different number
because it reads a name that differs. Rename per variant before merging; do not accept a merged
file because it parses.

## Boundaries

Compiler, runtime, simulator and specification defects belong to
[library](../library/AGENTS.md). Task routes and playbooks belong to
[agent](../agent/AGENTS.md). Use English for code and comments. Keep task outputs under an
ignored `tmp/<task>/`. Review task output for machine access details before committing or sharing it.
