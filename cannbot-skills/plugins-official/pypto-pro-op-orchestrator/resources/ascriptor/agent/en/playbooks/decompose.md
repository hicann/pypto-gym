# Decompose an algorithm

Read [common language](../common-language.md) in a new context and
[preflight](../references/authoring-preflight.md) before implementing or materially changing a kernel.

Use this route when the task permits multiple runtime kernels. Produce one explicit
plan with independent stage references and an independent full reference. A tile loop
inside one matmul does not create a host-level stage or a cross-core reduction merge.

1. Write the public math and [authoring contract](../references/authoring-contract.md).
   Preserve the original reference's operations and casts. For ONNX input, record
   opsets, graph IO, real initializers, external data and dynamic dimensions; random
   replacement weights prove connectivity only. No ONNX conversion tool is promised
   by this library.
2. Define each stage ABI: argument order, input/output shape, dtype, layout and workspace
   ownership. Choose launch order from the dependencies. Every edge names its
   producer/consumers, shape, dtype, layout, initialized extent, alias rules and last
   consumer. Validate unique producers, complete public outputs and an acyclic graph.
3. Record accumulation, cast, saturation and rounding at each stage boundary. Keep
   `plan_tolerance` against the original formula separate from
   `implementation_tolerance` against the staged references; each allowance has a
   numeric rule and reason. Algebraic equivalence alone is not floating equivalence.
4. Specify saved state: version, producer, initialization, semantic meaning, layout,
   dtype, lifetime and backward consumer. A backward unit contains its required
   forward-state generation; it cannot import a neighboring forward project.
5. Use the [plan template](../../templates/decomposition-plan.md). The independent full and
   staged formulas go in one `reference.py` that never imports ascriptor, together with the
   deterministic `make_inputs(case)`; both checks live in the same folder. The
   [worked reference](../../templates/decomposition/reference_example.py) demonstrates
   independent full and staged formulas, named checkpoints and DAG checks.
6. Run every stage and the reference composition. Compare composition directly with
   the original formula under its own budget. Check the DAG, then run `reference.py` alone in
   scratch with ascriptor uninstalled or simply unimported — if it still produces the staged
   and full expectations, its independence is a fact rather than an intention.

```bash
python templates/decomposition/reference_example.py
```

Run that command from the agent checkout with Torch installed. It prints the generated
case and stage counts; it does not execute a DSL kernel or claim hardware support.

The handoff is one demo folder and nothing else: `kernel.py`, `reference.py`, `main.py`,
`metadata.json`. `reference.py` exports `make_inputs`, `reference` and, for a pipeline,
`reference_stages`; `main.py` holds `execute`, the case list, and a `--stages` flag that
compares every intermediate instead of only the final outputs. Use that shape; do not invent
a second protocol. Reference-only planning may leave kernel execution
explicitly pending. A discovered contract error is corrected with evidence and a
versioned change before dependent implementation continues. Once the contract is
accepted, implement each leaf and validate the composition through
[implement-decomposition](implement-decomposition.md).
