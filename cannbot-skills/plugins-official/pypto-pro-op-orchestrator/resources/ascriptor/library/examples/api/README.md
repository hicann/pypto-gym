# Runnable public API examples

Fifty-eight folders, each one a small complete program with its own independent reference. Pick the
one whose surface you need, read four files, run it.

```bash
cd examples/api/axpb
python main.py                    # every case, on the functional simulator
python main.py --list             # the case ids, and what each one is for
python main.py --launcher pipesim # the lowered pipeline: events, hazards, deadlock
python main.py --launcher aclnn   # the cce backend, on this machine's card
```

[index.json](index.json) is the generated listing: every folder with its surface, devices, topology,
tags, case ids and the refusals it records. It is navigation only — it holds no validation state, and
what a folder does today is what running it says. Each folder's `metadata.json` carries the rest:
what the example is worth studying for, and when not to copy it.

## What a folder holds

| File | What it holds |
| --- | --- |
| `main.py` | The entry point **and** the precision check: the cases, the domain assertions, the launch, the comparison |
| `kernel.py`, or a `kernel/` package | The `@kernel` entry and its `@vf` / `@simt` bodies. It imports no torch |
| `reference.py`, or a `reference/` package | The independent reference: it imports no kernel, facade, backend or simulator |
| `metadata.json` | What the folder is for — surface, devices, topology, tags, and the prose a reader needs |

`python tools/api_examples.py --check` is the gate. It rebuilds `index.json`, refuses a folder holding
anything else, and enforces the two invariants that make a comparison mean something: a reference that
imported the compiler would be checking itself, and a kernel that reached for torch would not be the
thing the reference is checking. Two folders diverge deliberately and the tool records it —
`backend_extension` carries the plugin module and the `pyproject.toml` that publishes its entry point,
because for a protocol example those two *are* the subject.

## Reading main.py

Every `main.py` has the same five parts, in the same order, so the second folder you open is already
familiar:

- **`CASES`** — each case with a `purpose` saying what it is there to catch. Read these first; they
  are where the boundaries are, and a case that could not distinguish a wrong answer is not in the list.
- **`check_domain(inputs, expected)`** — the assertions that make the comparison legitimate. Why the
  arithmetic is exact, or why a tolerance is the size it is; which property of the reference the
  outputs rest on. Widening the domain fails here rather than silently invalidating the comparison.
- **`execute(case, inputs, ...)`** — the launch. Every destination arrives filled with a poison value
  and seeded in, so a lane the kernel never wrote reads back as something the arithmetic cannot produce.
- **`compare(name, got, want)`** — one comparison per named output, with its rule stated. Most are
  byte comparisons; where there is a tolerance, `TOLERANCE` names it per output and the docstring says
  what it is for.
- **`main()`** — case selection and the verdict.

A case that cannot run where it is run is skipped with the reason printed, never widened or deleted; a
backend or launcher that refuses something records it as a `# <name>:` comment beside the code, which
`index.json` collects. Some folders add a flag of their own — `--fault` builds a deliberately broken
kernel and reports which check catches it, `--inspect` prints what the frontend made of a kernel. The
command block at the top of `main.py` names every flag that folder has, and `--help` repeats it.

## Where to start

| Folder | Why first |
| --- | --- |
| [axpb](axpb) | The shortest path from a Python function to something that runs: load two rows, scale, add, store |
| [cube_matmul](cube_matmul) | The same, for the cube: two L1 operands, one L0C accumulator, one `matmul`. It runs on all four facades |
| [scalar_control](scalar_control) | Which parts of a loop are decided at compile time and which at run time, with nothing else in the way |
| [buffer_ring](buffer_ring) | What `auto_sync()` covers, and the one thing it does not: a round trip through GM |
| [strided_views](strided_views) | A logical shape over physical strides, counted in elements rather than bytes |
| [a2_fma](a2_fma) | The A2/A3 entry point: three precision paths into an accumulator that arrives initialized |
| [event_depths](event_depths) | What a green functional run does not cover, demonstrated: `--missing-event` passes on `sim` and is refused by `pipesim` |

## Running from source

Use the checked-in snapshot directly. Set `PYTHONPATH` to its absolute `library/`
directory, then run an example folder's `main.py`. Check `ascriptor.__file__`
against that directory before drawing a result. Torch and NumPy are needed for
numerical examples; device runs need the assigned card and vendor environment.

## What an example is not

An example generates its inputs and its reference at run time. It needs no recorded tensor, no source
archive and no sibling folder. It records no validation state and no backend support: a green `sim` run
is the functional model agreeing, a green `pipesim` run adds the event, hazard and deadlock model, and
neither executes a vendor backend. Source emission establishes neither vendor compilation nor hardware
support. A result on a card applies to the case, device, backend and artifact that produced it and to
nothing else.

The [source manifest](../../../sources.json) owns the complete snapshot identity. The
[API manifest](../../docs/api/manifest.json) separately lists where each public declaration is actually
used, and which have no example yet. An example for one family certifies no other overload, enum or
dtype.
