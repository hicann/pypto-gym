# Ascriptor

An instruction-level Python compiler, runtime and simulator for Ascend NPUs.
This checkout is the first public 0.1.0 source snapshot for the PyPTO-Pro authoring workflow.
Its SHA-256 source identity is recorded at the snapshot root in [sources.json](../sources.json).
The snapshot is unqualified: device, backend and performance claims require fresh checks.
The PyPTO-Gym repository root LICENSE applies to this source snapshot.

The compiler accepts typed kernels through `ascriptor.a2`, `ascriptor.a3` and
`ascriptor.a5`, lowers them
to inspectable IR, and emits supported CCE, PTO ISA or PyPTO Pro code. Backend/device support is
scoped; a generated artifact is not a hardware result.

A2/A3 share one tensor-vector vocabulary and select distinct device profiles.
Their admitted hardware path is CCE; other backends and the separate A5PR facade
are not promoted by this release. Original-scale and interface boundaries remain
explicit in the family support record.

See [the documentation map](docs/README.md) and [product contracts](docs/rfc/0012-product-contracts.md).
API teaching examples live in [examples/api](examples/api/README.md). Complete algorithms live
in this snapshot's [kernel gallery](../kernels/index.json); each demo's
`metadata.json` says what it is worth studying. Task guidance lives in [agent](../agent/README.md).
No stored golden dataset is needed.

Use the checked-in source directly; no Ascriptor wheel or external source
checkout is needed. From the snapshot root:

```sh
PYTHONPATH=library python library/examples/api/axpb/main.py
```

Torch and NumPy are needed for numerical examples. CANN, PyPTO-Pro and NPU drivers are
external requirements for device execution. The root [source manifest](../sources.json)
describes the exact bytes. For a first
example, run `cd examples/api/axpb && python main.py`, which generates its inputs,
computes an independent reference and compares every case on the functional simulator;
`--launcher pipesim` runs the same cases through the pipe model and `--list` names them.
