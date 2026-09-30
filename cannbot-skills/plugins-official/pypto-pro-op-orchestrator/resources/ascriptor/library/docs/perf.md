# Measure performance under a fixed contract

Use an already-correct example with a fixed formula, input domain, launch, comparison
and measurement objective. Device latency, host dispatch time and simulator cycles
answer different questions. The [example contract](rfc/0012-product-contracts.md)
and the folder's own `main.py` own correctness; the agent's
[optimization method](../../agent/en/playbooks/optimize.md) owns the task sequence.

The earlier prototype corpus sweeps, calibration numbers and corrected diagnoses
are preserved in Git. They are historical
measurements, not current defaults or a fresh performance qualification.

## One shape, one entry

Both repositories hold the same shape: a folder of four files whose `main.py` is the entry point
and the precision check. There is no second entry and no profiling subcommand -- profiling is the
run plus the two environment variables below, and the run checks its outputs first, so a timing
number never comes from a launch nobody compared.

## Capture a scoped device run

Use an assigned healthy device, its configured lock and an isolated output
directory under the [board execution contract](rfc/0004-board-batch-runner.md).
From the kernel gallery checkout, after selecting a board name in the external
`ASCRIPTOR_BOARDS` configuration, this existing demo checks its declared outputs
while collecting repeated task measurements:

<!-- checked-command: kernels -->
```sh
ASCRIPTOR_PROFILE=1 ASCRIPTOR_REPEAT=32 python ascriptor_kernels/examples/axpy/main.py --launcher board --backend cce
```

A folder is self-contained, so `cd ascriptor_kernels/examples/axpy` then
`python main.py --launcher board --backend cce` is the same run with its output beside the
kernel. `main.py` is the whole entry: it builds the kernel, launches it through `OpExec` on the
launcher you name, runs every case unless `--case` selects one, prints each output's `ok`/`FAIL`
with its worst absolute difference — and the tolerance, where the case has one rather than
requiring a bitwise match — then a passed/total line, and exits non-zero if any case failed.
`python main.py --list` prints the case ids. A `check` subcommand and the
`--device` and `--output` options are not part of this entry: the folder fixes its own device, and
its output tree is `tmp/<launcher>` under the working directory, so the working directory is what
keeps one job's outputs isolated from another's. A library example under `examples/api/` is the
same entry with the same flags.

Run that on the box: the board launcher runs on the machine that calls it and reads
which machine that is from `ASCRIPTOR_BOARDS`. The board
launcher reads `ASCRIPTOR_PROFILE`, `ASCRIPTOR_REPEAT` and optional
`ASCRIPTOR_AIC_METRICS` (`PipeUtilization` by default), wraps the actual application
in msprof, and retrieves `perf/op_summary.csv` in each execution's output tree.
See [Board](../ascriptor/runtime/board.py). Repetition count and warmup are choices
for the measurement, not universal acceptance values. In-place or accumulating
workloads must preserve their declared initialization on repeated launches.

Host wall time around a launch is not device kernel latency, even with a board launcher: it
includes build, dispatch and readback. Nothing in either repository records it as a number any more,
and a hand-held `time.perf_counter()` around `OpExec.__call__` measures that whole envelope rather
than the kernel. Use the profiler's `Task Duration(us)` for a latency claim; model timing remains in
model cycles.

## Read the measurement

[The runtime parser](../ascriptor/runtime/perf.py) exposes `kernel_rows` and
`summarise(csv_path, warmup=...)`. It selects the non-AI_CPU operation with the
largest total duration, sorts its launches by start time and reports median,
minimum, maximum, sample count and spread. Verify that the selected operation is
the intended kernel when the application launches multiple operations.

Pipe ratios, grid size and counters come from the retained row closest to the
median duration. With an even sample count the median need not be a real sample;
do not relabel its representative row as a separately measured median launch.
The parser defaults to zero warmup and falls back to all rows if warmup removes
every row. A formal acquisition must therefore check its expected sample count
and retained warmup explicitly instead of silently accepting that fallback.

| Quantity | Interpretation |
| --- | --- |
| `Task Duration(us)` | Device task latency in the recorded environment |
| `Block Num`, `Mix Block Num` | Actual grid; match it for a scheduling-only comparison |
| `aic_*_ratio`, `aiv_*_ratio` | Per-pipe observations from the representative task |
| `aic_total_cycles`, `aiv_total_cycles` | Validate availability before deriving utilization |
| Host wall time | Full launch/dispatch/execution path, reported separately |
| Model cycles | Scheduling/model evidence, not a conversion to board microseconds |

All-zero total-cycle counters for a nonempty compute kernel cannot establish
utilization. A zero counter for an unused individual pipe can be valid. Keep a
valid latency-only result when counters are unavailable. GM request bytes and
model transfer costs do not establish measured HBM bandwidth; use the assumptions
and units in [performance parameters](performance-parameters.md).

## Compare like with like

Keep actual source/artifact, compiler/dependency, device, input identity, layout,
tile, core count, warmup and statistic with each acquisition. Scheduling-only
controls preserve arithmetic and transfer work. Residency, tiling or grid changes
need their own attribution. Use interleaved or bracketing controls when drift could
dominate the difference, and retain failed or unstarted attempts separately.

Remote CCE and PyPTO jobs include an identity derived from the local output
directory. Give independent jobs independent outputs and honor device allocation
and locks; a shared kernel name alone does not identify a local run directory.
Distinct output identities prevent that collision, not arbitrary concurrent access
to an unallocated device.

Store raw traces and profiler files in scratch; keep concise source-scoped results
with the owning unit. There is no current list of accepted kernel performance targets: the
kernel repository's performance scope was removed with the gallery refactor and resolves at
kernels `f79b44b721eba5080b802409c2388f74638405ce:docs/performance-scope.md` as dated provenance
for the measurements it recorded, against the source and dependency identities of that day.
The gallery records no target and no measurement, so a performance statement about a kernel is
made here, by measuring it again. This procedure does not create a new speedup claim or
reinterpret an old minimum as a new median.
