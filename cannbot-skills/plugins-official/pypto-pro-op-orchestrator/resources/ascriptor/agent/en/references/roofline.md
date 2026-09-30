# Cost and Roofline analysis

Use this page when selecting a performance change or explaining where time
goes. Fill [the analysis template](../../templates/performance-analysis.md).
Freeze the objective, allowed changes, comparison scope and stop conditions.
Keep user criteria and remaining headroom as separate conclusions.
## Build the cost picture

1. Write the formula, operand/accumulation precision, dimensions and physical
   tile. Count cube MACs and vector work separately; state the MAC-to-FLOP rule.
2. Derive independent work, allowed/used cores and work per active core. A task
   can deliberately fix one core; do not treat that as a preferred deployment.
3. Count unique logical bytes, issued requests per memory boundary and
   read/write direction, repetitions, per-core copies, padding and layout work.
4. Calculate resident inputs plus all simultaneously live buffer versions at
   each memory level. Reducing traffic competes with deeper buffering for space.
5. Record model work per actual serialized resource and the critical waits.
   Preserve core/lane identity; total vector-participant work is not wall time.
6. Record actual latency and valid hardware counters, cache conditions, clock
   provenance, and the source of each capability estimate. Unknown stays unknown.

The library owns [performance parameters](../../../library/docs/performance-parameters.md),
including model MAC rates and clock provenance. On 2026-09-07 the maintainer
supplied 1.6 TB/s nominal GM bandwidth for A2/A3/A5 and about 80% effective for
a suitable access pattern: 1.28e12 B/s in decimal units. Dispersed reads/writes
are slower by an unspecified amount. This is a planning assumption, not a
measurement of the current kernel, not a per-core bandwidth, and not an
independent A2/A3 support declaration.

If using that value, report a sensitivity scenario rather than an HBM
efficiency unless traffic at HBM and its sustained bandwidth were measured.
A program can reread weights from GM while most requests hit L2. A small or
scattered transfer and a single active core need not reach the estimate.

## Two complementary bounds

For a compute class and memory level with compatible scope:

```text
I_level = operations_of_that_class / bytes_at_that_level
P_bound = min(compute_capability_of_that_class, bandwidth_level * I_level)
T_compute_bound = operations_of_that_class / compute_capability_of_that_class
```

Separate FP16 cube compute from vector arithmetic/conversions. A vector FLOP
total divided by a cube peak is not useful. Vector duration is difficult to
estimate: use per-operation/VF model costs and dependencies or measured stage
time, including reductions, casts, issue limits and memory behavior. State an
unknown vector cost rather than inventing one throughput for all VF operations.

For a fixed lowered workload and timing model:

```text
T_resource_bound = max_resource(sum(non_sync_work_cost_on_that_resource))
T_model >= max(T_resource_bound, known_dependency_bound)
resource_efficiency = T_resource_bound / T_model
rescheduling_only_speedup <= T_model / T_resource_bound
```

Group tasks by their real serialized resource, including core/lane identity.
Both cube stages charge the same cube compute resource. Concurrent vector
lanes have separate work; do not add their overlapping intervals to wall time.
The simple resource bound omits some dependencies and startup/drain, so it is
optimistic. A per-transfer DMA coefficient is not proof of calibrated shared
HBM competition across cores. Model cycles do not predict hardware microseconds.

## Worked CVC example: remove work as well as schedule it

Take `M=2176, K=N=128`, two matmuls and an intervening scale/bias/ReLU with FP16 RNE
materialization. One MAC counts as two FLOPs, and the traffic a resident weight removes
follows from the shape alone:

```text
cube_FLOPs = 4 * 2176 * 128 * 128 = 142606336
tiles = 2176 / 128 = 17
one_weight_bytes = 128 * 128 * 2 = 32768
removable_weight_requests = 2 * (17 - 1) * 32768 = 1048576
```

That much is arithmetic. The cycle counts that would turn it into a utilization figure are a
measurement, and this workspace no longer holds one: the recorded lookahead-versus-resident
diagnostic that used to be cited here is gone, and its numbers are not carried forward.
Produce your own for the pair you care about — run the matching `cvc_pipeline_*` and
`cvc_resident_*` cases of the
[mixed-pipeline demo](../../../kernels/ascriptor_kernels/tutorials/mixed_pipeline) under
`--launcher pipesim`, take the per-pipe and total model cycles from each, and apply the two
ratios above to your own shape.

Read them in that order. While the dominant resource is the input pipe, rescheduling the same
work is bounded by `T_model / T_resource_bound`; once repeated weight loads are removed the
dominant resource may become a different pipe, and the scheduling question has to be asked
again against that new bound. A speedup from removing traffic and a speedup from rescheduling
cannot be multiplied, and a high utilization number can coexist with removable work.

## Choose and explain an experiment

Inspect work allocation, reuse, tile/capacity/layout, scheduling/dependencies,
then stage internals. This is an investigation order, not a mandatory sequence
of modifications. Fix the permitted dimension with the largest justified space.
Predict what work is removed or which wait can be hidden, with units and scope.

For a scheduling-only control keep arithmetic, VF body, layout, transfers,
tile and cores matched. For residency/grid/tiling changes report their changed
work and total benefit. Recompute the cost picture after each material change.

Retain warmup, sample count, distributions, source identities and input hashes.
Interleaved or before/after controls help reveal drift. Invalid total-cycle
samples cannot establish utilization; a valid latency can still be reported
separately. A zero on one unused pipe can be legitimate. Keep hardware ratios
from the same sampling row and retain participant denominators. See
[the optimization playbook](../playbooks/optimize.md) and
[pipeline interval evidence](pipeline-model.md#validate-the-actual-schedule).

## Compare work after retiling

Equal allocated bytes do not establish equal work. One large tile and two
smaller slots can fit the same memory budget while changing loop count,
materialization traffic, state updates and launch/control work. Likewise,
increasing row grouping can remove padded arithmetic but reduce active cores.
Report these changes before attributing the result to scheduling.

To isolate lookahead, match tile sizes, arithmetic, VF bodies, transfers,
buffer counts and core assignment. Keep that control separate from the best
end-to-end alternative. The [attention topic](attention-authoring.md) and its
owner records show why tile selection and pipeline overlap need distinct
comparisons. A recorded negative experiment remains useful evidence even
when another change later improves the full kernel.

## Check explicitly scoped evidence

From the agent checkout run the [analysis helper](../../tools/analyze_performance.py):

```bash
python tools/analyze_performance.py templates/performance-input.json
```

The [example input](../../templates/performance-input.json) contains one historical
interval pair, not the whole overlap trace. The helper audits units/resource
sums, unions same-core intervals and keeps HBM efficiency UNKNOWN without
measured traffic and an explicit `hbm_bandwidth_classification: measured_sustained`.
An assumed bandwidth is reported only as a labeled scenario. Hardware samples
use `time_domain: hardware_us`; model costs use `model_cycles`. The tool checks
provided data; it does not authenticate its provenance or change workflow exit gates.
