# Run a source example

Use the self-contained library, agent and kernels directories in this snapshot
as one source identity. Verify it with the snapshot index before a new task.
Numerical simulation needs Torch and NumPy; device execution additionally needs
the assigned card, CANN and PyPTO-Pro. No Ascriptor wheel is installed.

From the snapshot root, keeping the source library on `PYTHONPATH`:

```bash
PYTHONPATH=library python library/examples/api/axpb/main.py
PYTHONPATH=library python library/examples/api/axpb/main.py --launcher pipesim
PYTHONPATH=library python -m ascriptor.cli compile library/examples/api/axpb/kernel.py::axpb --backend cce -o tmp/emit
```

The independent expression is `2*x+y` on one row of 64 float32 lanes, and `main.py` is both the
entry point and the comparison: it generates the inputs, computes that expression in `reference.py`
and compares every output lane exactly, on the functional model and then through the pipe model.
`--list` names its three cases. Emission is separate and reports artifact filenames without
executing an NPU; the same kernel emits supported PTO ISA or PyPTO Pro source by changing
`--backend`.

This starter uses the supported A5 facade. Decorators bind locally; importing another facade
does not retarget an existing kernel. Signatures use `GM[dtype, dimensions]` and shared
string symbols; no explicit scalar is required for a fixed shape. Calls pass tensors in signature
order, then explicit scalars, and consume the actual `OpExec` return. There is no runtime
shape-binding guess or legacy boolean execution flag. `<<=` describes a copy selected from the
memory spaces; same-side autosync does not replace cross-side ownership.

Read the [authoring model](concepts.md), then continue with one [task route](ROUTER.md).
For A5 tails and repeated tiles, open an A5 demo in the [kernel gallery](../../kernels/README.md),
run `python main.py --list` inside that folder to see its cases, and preserve the ownership
and precision boundaries its `metadata.json` and `reference.py` state.


Choose other primitives from the [API examples](../../library/examples/api/README.md), whose generated
[index](../../library/examples/api/index.json) lists every folder with its surface, devices, topology,
tags and cases. A folder records no result: what it does today is what running it says, and a
measurement is a library receipt.
