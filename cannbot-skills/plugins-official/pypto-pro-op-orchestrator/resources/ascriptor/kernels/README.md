# Ascriptor kernels

A gallery of runnable NPU kernels. Every folder is one demo, holds exactly four files, and
imports nothing from this repository — so you can copy a folder anywhere and run it.

```sh
export PYTHONPATH="<snapshot>/library"
cd <snapshot>/kernels/ascriptor_kernels/examples/axpy
python main.py                        # every case, functional simulator
python main.py --list                 # what the cases are for
python main.py --case <id>            # one of them
python main.py --launcher pipesim     # lowered pipeline: events, hazards, deadlock
python main.py --launcher aclnn       # the cce backend, on this machine's card
```

**To find a demo, query `index.json`.** It is generated from every folder's `metadata.json`, and
it is the intended way in: filter it on `topology` (a controlled vocabulary — `cube->vec`,
`vec-only` and the rest), on `device`, or on words in `formula` and `tags`. Each entry carries
`study_for` and `do_not_copy_when`, so you can rule a demo in or out before opening its source.
`tools/build_index.py` regenerates it, and `--check` fails if it is stale — so if you add or
remove a case, run it.

## A demo folder

```
kernel.py       the DSL: @kernel / @vf / @simt bodies and the host-side tables they need
reference.py    torch only, never imports ascriptor: make_inputs(case) and reference(inputs)
main.py         the OpExec entry AND the precision check, with the cases as data
metadata.json   what it computes, what to study in it, and when not to copy it
```

`main.py` is the whole execution story: it builds the kernel, launches it through
`ascriptor.runtime.OpExec` on whichever launcher you ask for, and compares the outputs with the
independent reference. Every launcher runs on the machine that calls it — `sim` and `pipesim`
anywhere, `aclnn`, `cannsim`, `board` and `pypto` on a machine with a card.

Some demos carry more: `--stages` on a multi-launch pipeline compares every intermediate, so a
wrong final answer points at the launch that produced it; `--variant` or `--pattern` on a demo
with several schedules runs one of them across every case.

## What each launcher can see

A green run proves what its launcher models, and no more. The two that need no card divide up
like this, and the division decides which defects a demo can catch for you:

- `sim` interprets the kernel sequentially. It sees wrong arithmetic, a wrong index and a lane
  the kernel never wrote — that last one only because destinations arrive NaN-poisoned. It
  models no pipes and no concurrency at all, so no amount of green here says anything about
  synchronisation.
- `pipesim` runs the lowered pipeline against the event/hazard model over the seven task pipes
  (`S`, `MTE1`, `MTE2`, `MTE3`, `M`, `V`, `FIX`) on every lane. It catches unbalanced events, a
  wait with no set (deadlock), and two accesses to overlapping bytes that no event, barrier or
  mutex orders — including across cores. Remove a `bar_all()` between two passes and it names
  both source lines and refuses the run.
- **`pipesim` cannot see inside a vector function.** A whole `@vf` body is one task on the `V`
  pipe, so the register-level STORE/LOAD ordering a `vf_barrier` establishes is below the
  model's granularity. Delete one and every case still passes. That ordering is checked on a
  card, or not at all.

Whether a hazard is *reached* also depends on the case. A dropped `bar_all()` in
`algorithms/matrix_normalization` is caught by five of its six `row_sum_large` cases and missed
by the largest, whose schedule happens to separate the two accesses. A case matrix with an empty
corner is a check with a blind spot, which is why the cases say in `--list` what each is for.

A case that cannot run where you are says so and says why, instead of being skipped quietly or
dropped. Model-shape attention cases print which full-shape cases they stand in for; the one
case whose simulator answer differs from silicon prints the byte, the value and the reason.

## Galleries

| Directory | Holds |
|---|---|
| `ascriptor_kernels/` | kernels written in the ascriptor DSL |
| `pypto_pro_kernels/` | PyPTO-Pro kernel samples (not yet populated) |

`index.json` lists every demo with its formula, device, topology, tags and case count. It is
generated — `python tools/build_index.py` collects each folder's `metadata.json`, and
`--check` fails if it is stale. Each folder describes itself; nothing describes it twice.

## What this repository does not hold

No validation records, no backend support matrix, no release scope, no defect list, no
development status. To find out whether a kernel works on a backend, run its `main.py` with
that launcher — it takes seconds and it answers for the source in front of you rather than for
a source someone recorded months ago. Where a backend genuinely cannot run a demo, the reason
is a comment in that demo's `main.py`, next to the code it is about.

The compiler, runtime, simulator and their specifications belong to the
[library](../library/README.md). Task routes and playbooks belong to
[agent](../agent/README.md).
