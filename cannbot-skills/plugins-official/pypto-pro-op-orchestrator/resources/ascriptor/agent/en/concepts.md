# Authoring and execution model

Start with the [small code anchor](playbooks/author.md#first-code). The full owner
example supplies imports, generated inputs and an independent reference.

1. Declare typed GM inputs/outputs and explicit scalar parameters. Allocate output
   tensors on the host; pass arguments in signature order and consume the `OpExec`
   return. For read/modify/write, preserve initialization with `seed_outputs=True`.
2. The decorated Python AST becomes Surface IR. Kernel `range` remains a runtime
   loop even with literal bounds; `unroll` expands during authoring. Static expressions
   and calls with entirely static arguments execute host Python. Keep the requested
   formula inside the kernel and check helpers/control flow in the [authoring API](../../library/docs/api/authoring.md).
3. Allocate local storage with a physical shape. `<<=` selects movement from source
   and destination spaces. Follow GM → local storage → VF/SIMT or cube computation
   → GM; derive valid lanes and the complete instruction footprint before changing a shape.
4. `auto_sync` handles supported same-side dependencies. Reused slots and cross-side
   producers/consumers need an explicit ownership/lifetime plan. Continue through
   the corresponding preflight trigger when those features are present.
5. Inspect `kernel.ir()`, run the independent comparison, then inspect lowered
   hazards and ownership. Interpret each result with the [evidence table](common-language.md#evidence).

Use the [API entry](../../library/docs/api/README.md) for GMList, grouped registers,
scalar cells and exact signatures. A2/A3 CCE and A5 have device-specific scopes:
[pyproject.toml](../../library/pyproject.toml) names the source version;
[status](../../library/docs/status.md) explains validation scope.
For a first cube kernel, use the [complete matmul example](../../library/examples/api/cube_matmul)
and its FP16 16×16 operands/FP32 output contract. How to run what you write, here and on a
card, is [one page](references/development-execution.md).
Use [quickstart](quickstart.md) for installation and [practice](practice.md) for progressive tasks.
