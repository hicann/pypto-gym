# Running what you are writing

While writing a kernel, use `OpExec`. The demo folder's four files and the gallery's batch
runner are for **admission and batch testing** — they exist so one shape is maintained in one
place and the whole gallery can be re-run. They are not the authoring path, and nothing has to
be wired to them before code is written.

## Here

```python
from ascriptor.runtime import OpExec

out = OpExec(kernel, launcher="sim")(q, k, v, o, B, T)       # arithmetic and surface semantics
out = OpExec(kernel, launcher="pipesim")(q, k, v, o, B, T)   # events, hazards, deadlock
```

Tensors in signature order first, explicit scalars after. **Every tensor in the signature is an
argument, including the outputs** — `o` above is a buffer you allocate and pass in; the runtime
does not create it for you, and a call that omits it fails with `<name> needs N tensor
argument(s), got M`. That is also why outputs are poisoned by default and why
`seed_outputs=True` exists: you own what is in them before the launch. The return value is torch
tensors; compare against those.

`pipesim` also checks event balance, hazards and deadlock. `OpExec` **raises** on any of them, as
does a demo's `--launcher pipesim`, so those two paths cannot reach different verdicts. A runner
that calls `simulate()` directly — the investigation entry point, not a launcher — instead
**returns** them on the result, and a runner that ignores `result.hazards` will report a
hazardous kernel as passing. The verdict is the same; who raises it is not.

**Emit before you simulate.** `compile_kernel` takes seconds, needs no card, and a form that
`sim` accepts but a backend refuses surfaces there instead of after you have built on it. For a
PyPTO-Pro target that is [the first rule of its page](pypto-pro.md#emit-first); the authoring
[playbook](../playbooks/author.md) puts it at the head of the same ladder. The hardware-first rule
in [runtime](../runtime-and-maintenance.md#hardware-first) orders the *card* against the
simulator and does not move this: emitting costs no round trip, so it comes first either way.

## On hardware

Run the script on the machine holding the assigned card with this snapshot's
`library/` on `PYTHONPATH`. The same `OpExec` with `launcher="board"` (CCE) or
`launcher="pypto"` (PyPTO-Pro) uses that machine's card. Inputs, independent
reference, execution and comparison stay in one process.

## The six launchers

`OpExec` takes these six and no others, and **every one runs on the machine that calls it**. No connection-based launcher exists.

| launcher | Where | What a pass establishes |
|---|---|---|
| `sim` | Here, in process | Supported arithmetic and Surface semantics; no backend involved |
| `pipesim` | Here, in process | Lowered event balance / hazards / deadlock; any of the three raises through `OpExec` |
| `aclnn` | This machine's card | Vendor build plus actual execution (CCE through an aclnn custom-op project) |
| `cannsim` | Here | Vendor simulator execution, recorded separately from host pipesim |
| `board` | This machine's card | The same project, built and run on this machine |
| `pypto` | This machine's card | Generated PyPTO-Pro sources executed on this machine's card |

The last three need this machine to **say it is a board** (below); on a workstation they refuse,
while `sim` and `pipesim` are unaffected. `compile_kernel(entry, backend=...)` is not a launcher —
it emits source and executes nothing. It comes from the same module as `OpExec`
(`from ascriptor.runtime import compile_kernel`); the package root does not export it.

## What the box needs first

Two things, **done once when the machine is set up**:

1. **Source import**: run from this snapshot with `PYTHONPATH` including its `library/`
   directory. Check `ascriptor.__file__` and the snapshot's `sources.json` before use.
2. **The machine says which one it is**: a config in the workspace whose entry for this machine is
   marked `"local": true`, and an environment script that exports `ASCRIPTOR_BOARDS` at it.

That entry needs at least `workspace`, `env_script`, `cube_cores` and `"local": true`.
`cube_cores` is this card's **AIC count** (an Ascend950PR is 28, not the device profile's 32) —
without it `pypto` refuses to run, for the reason in [targeting PyPTO-Pro](pypto-pro.md#cores).

`ASCRIPTOR_BOARD` picks one when a machine carries several entries, such as one per card.

The board config describes only this machine. Entries containing connection fields
are refused by the local device runtime.

## Ask doctor first

```sh
PYTHONPATH=<snapshot>/library python -m ascriptor.cli doctor
```

One answer for all of it: which interpreter, where the **actually imported** ascriptor lives and
what version it is, whether torch/numpy/pypto_pro are present, whether this machine counts as a
board, and the card's AIC/AIV counts. On a workstation it says plainly that `board` and `pypto`
will refuse while `sim` and `pipesim` are unaffected.

<a id="admission"></a>
## When to write the other three files

When the kernel is to be **admitted** to the gallery. That is when the script becomes a folder of
exactly four files — `kernel.py`, `reference.py`, `main.py`, `metadata.json` — and the evidence
ladder of the [authoring playbook](../playbooks/author.md) applies. Before that, `OpExec` plus a
comparison you wrote yourself is enough.

Admitted means it passes `python tools/run_all.py` in the kernels checkout, which runs every
demo's `main.py` in its own process and its own working directory, because that is how a reader
runs it. A demo that does not pass there is broken whatever its `metadata.json` says.

A PyPTO-Pro target does not take this route: it hands over a delivery package instead — see
[the delivery area](pypto-pro.md#delivery-area).
