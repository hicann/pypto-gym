# Author one kernel

If the [compact context](../../context/kernel-authoring.zh-CN.md)
was read, use its common-language and general preflight checklist; otherwise read
[common language](../common-language.md) once. Complete
[preflight](../references/authoring-preflight.md) around this code path, expanding
only triggered sections.

<a id="first-code"></a>
## Start with one complete data path

The original `axpb` from [axpb](../../../library/examples/api/axpb) shows signature → allocation → DMA → VF computation → writeback.
`axpb_vf` computes `2*x+y` beside it in `kernel.py`; the independent reference is `reference.py` and the
commands are in `main.py`, which is both the entry point and the comparison.
This anchor has fixed `(1,64)` FP32 tensors and one vector core; re-derive footprint and precision before extending its shape.

<!-- code-anchor:author:start -->
```python
@kernel(mode="vec", block_dim=1)
def axpb(x: GM[f32, (1, 64)], y: GM[f32, (1, 64)], o: GM[f32, (1, 64)]):
    ub_x = Tensor(DT.float, [1, 64], Position.UB, name="ub_x")
    ub_y = Tensor(DT.float, [1, 64], Position.UB, name="ub_y")
    ub_o = Tensor(DT.float, [1, 64], Position.UB, name="ub_o")
    with auto_sync():
        ub_x <<= x
        ub_y <<= y
        axpb_vf(ub_x, ub_y, ub_o)
        o <<= ub_o
    return o
```
<!-- code-anchor:author:end -->

## Implementation steps

1. Derive the [task contract](../references/authoring-contract.md) from the formula/reference: casts, grouping, aliases, shapes, launch count and permitted host work.
2. Choose a primitive from [API examples](../../../library/examples/api/README.md) or a complete algorithm from the [kernel gallery](../../../kernels/README.md), whose generated [index](../../../kernels/index.json) lists every demo with its formula, device, topology, tags and case count.
   Find signatures through the [API entry](../../../library/docs/api/README.md).
   Neither owner's folder declares a scope: read its `metadata.json` for what it teaches and when not to copy it, and run its `main.py` where you need the answer. A measurement is a library receipt, scoped by the release record and never by a folder.
3. Derive physical storage, initialization and per-core output ownership. Start ordinary tails with
   [the short checklist](../references/memory-and-tails.md#vector-tail); reductions, casts, views and slots trigger further preflight sections, and a [cross-side handoff](../references/cross-side-handoff.md) has its own call sequence.
   Repeated mixed graphs use the [pipeline method](../references/pipeline-model.md).
4. Implement the smallest complete candidate, inspect `kernel.ir()` and independently compare the actual `OpExec` return.
   [Running what you are writing](../references/development-execution.md) has the calls, how a script reaches a card and what `ascriptor doctor` answers.
   Follow [hardware first](../runtime-and-maintenance.md#hardware-first): use the full workload on the target device;
   after a failure, reduce shape and cores for a causal diagnostic. A round trip too expensive to iterate on has its own branch there, and [which machine runs which step](../runtime-and-maintenance.md#where-each-step-runs) is decided there too. PyPTO-Pro delivery explicitly selects auto_mutex by default; generate manual only on an explicit user request.
   A one-launch task keeps formula work inside the kernel; the host generates inputs, allocates, dispatches and compares.
5. Check all outputs, valid tails, the final row/slot, initialization and same-core reuse, then lowered hazards/events.
   Requested overlap uses actual same-core stage/item compute intervals; latency work uses [cost analysis](../references/roofline.md).
6. Show the result stands alone: run it from a scratch directory outside the checkouts with the declared dependencies, and report [the corresponding evidence stage](../common-language.md#evidence).
   Until the kernel is admitted anywhere, that is your script — `OpExec` plus the comparison you wrote. A kernel [admitted to the gallery](../references/development-execution.md#admission) becomes the four-file demo folder, and a library API example is the same four files; either travels by being copied, with no export step.
   Either one that still imports something from the tree it came from is not finished. Retain failures and meaningful negative controls; follow the user's objective and stop conditions.

Before final delivery, follow the [PyPTO-Pro delivery synchronization policy](../runtime-and-maintenance.md#sync-closeout)
and verify the emitted decorator, hardware result and selected mode.

The following is a bounded one-core recipe for the fixed small starter. It is not a
prerequisite for a card: your own script reaches one the same way, as
[running what you are writing](../references/development-execution.md) sets out. A gallery
demo walks the same three stages from inside its own folder — `python main.py`, `python main.py --launcher
pipesim`, `python main.py --launcher board` — and each establishes the same thing there: what a
stage establishes is a different question from where a full workload's first launch goes, and
the cost branch above is when to walk them in that order.

Emit comes first: it takes seconds and needs no card, and a form that sim accepts but the
backend refuses is found before anything is built on it — for PyPTO-Pro this is [the page's first rule](../references/pypto-pro.md#emit-first).
Emit is the library CLI, because an example has no `emit` subcommand; name the kernel and the backend:

```bash
ascriptor compile examples/api/axpb/kernel.py::axpb --backend cce -o tmp/agent-author/emit
```

Then the two models, from the library checkout. `main.py` generates the inputs, computes the
independent reference and compares every case, so there is no separate `reference` step:

<!-- checked-command: library -->
```bash
python examples/api/axpb/main.py --launcher sim
python examples/api/axpb/main.py --launcher pipesim
```

Expect complete exact outputs and no pipe imbalance/hazard/deadlock; use the
[real-file execution recipe](../runtime-and-maintenance.md). Follow [debug](debug.md) after a failure,
and a generated-input probe for an uncertain primitive. Attention also follows
[its topic](../references/attention-authoring.md) to connect math, layout and an exact canonical case.
