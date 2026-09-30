# Implement a verified decomposition

Read [common language](../common-language.md) in a new context and
[preflight](../references/authoring-preflight.md) before implementing or materially changing a kernel.

Implement the accepted stage/DAG and precision contract without hiding computation
in the host reference. Start by rerunning the independent full and stage references.
If they fail or an ABI is incomplete, repair the contract with evidence through
[decomposition](decompose.md); do not tune expected values to match a kernel.

1. Record the contract and accepted library revision. Topologically order the work.
   For each stage, write typed signatures, symbol bindings, output initialization,
   footprint, workspace/saved-state ownership and the declared comparison budget.
2. Implement and check each leaf before relying on it in composition. Include the
   minimum, normal, tail and repeated-slot cases the stage contract allows. Use one
   relevant public example and [memory/sync facts](../references/memory-and-tails.md).
3. Compose in `main.py`'s `execute`: it launches the stages in order and returns named
   outputs, with every destination allocated poisoned by the caller and handed to `OpExec`
   with `seed_outputs=True`, so a lane no kernel wrote stays distinguishable from a lane
   correctly written to zero. Keep required saved-state preparation and every import inside
   the folder. `reference.py` cannot call the simulator; execution cannot silently fall back
   to the reference when a backend or device is unsupported.
4. Check the `--stages` outputs against `reference_stages`, composed execution against the
   staged reference, and final outputs against the independent full formula. Preserve
   both error budgets. Check every output's name, shape, dtype and non-finite policy.
   Without `--stages` a wrong final answer only tells you it is wrong, not which launch
   produced it.
5. Copy the whole folder alone into scratch and run it there: `python main.py --list`, then
   `python main.py`, then `--launcher pipesim` and a device launcher on a machine with the
   card. Correctness comes before any timing. Emit, compile, functional simulation, pipe
   simulation and board results remain separate.

The scaffold is one template per file of the folder:
[`kernel.py.template`](../../templates/kernel.py.template),
[`reference.py.template`](../../templates/reference.py.template),
[`main.py.template`](../../templates/main.py.template) and
[`metadata.json.template`](../../templates/metadata.json.template). Copy all four, drop the
`.template` suffix, fill in the TODOs; every body deliberately raises `NotImplementedError`
until you do, and `kernels/tools/build_index.py` rejects both a fifth file in the folder and a
missing one. `metadata.json.template` carries the nine required keys and leaves `device` and
`topology` as TODO on purpose: they come from controlled vocabularies that
`build_index.py --check` enforces, so choose them rather than guess. For the finished shape,
read the smallest demo, [`examples/axpy`](../../../kernels/ascriptor_kernels/examples/axpy).

For authorized parallel work, assign disjoint stage files and fixed contract/library
versions. The lead owns shared interfaces, composition, environments and hardware.
Workers return source paths, exact commands/results, ABI changes requested and open
issues. Worker completion is input to integration review, not a passing gate.
