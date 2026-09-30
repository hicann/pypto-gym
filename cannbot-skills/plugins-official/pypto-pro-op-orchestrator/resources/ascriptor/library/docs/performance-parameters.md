# Performance parameters and their evidence

This page owns the parameter meanings used by the agent's Roofline analysis.
It does not change device profiles, timing-model coefficients or device support.
The planning values below are assumptions until measured on the selected device.

## Bandwidth assumptions supplied by the maintainer

On 2026-09-07 the maintainer supplied the following planning values for A2, A3
and A5: nominal GM bandwidth **1.6 TB/s**, with approximately **80%** achievable
for a suitable access pattern. Decimal units give **1.28e12 bytes/s** as the
initial effective-bandwidth estimate. Dispersed reads/writes achieve less.

These are maintainer-supplied estimates, not a bandwidth benchmark performed by
this repository. Use them as named assumptions in a sensitivity calculation.
Record the actual device/SKU and the scope of its reported memory interface;
do not multiply the device-wide value by the number of active cores. Do not
assume a single core, small transfer or scattered access sustains that value.
No universal scattered-access discount is known. Measure the relevant pattern
or leave its effective bandwidth unknown.

GM request bytes are program transfers. Repeated requests can hit L2 and are
not measured HBM bytes. A calculation using requested bytes and 1.28 TB/s is
an explicit no-cache traffic scenario, not an observed HBM Roofline efficiency.
For read/write overlap, state whether bandwidth is shared or independently
measured. The simulator's per-transfer bytes/cycle coefficient does not supply
a calibrated whole-device bandwidth or multi-core contention model.

## Cube costs from the current executable model

[Model selection and cost calculation](../ascriptor/backends/sim/timing/cycle_model.py)
select the A5 table for `950`/`950pr` and the A2 table for the other retained
device profiles, including A3. The authoritative rates are read from these files:

| Model | FP16 operand MACs/cycle | MMAD setup cycles | Source |
|---|---:|---:|---|
| A2/A3 model | 2048 | 21 | [A2 table](../ascriptor/backends/sim/timing/a2_cycle_model.json) |
| A5 model | 4096 | 67 | [A5 table](../ascriptor/backends/sim/timing/a5_cycle_model.json) |

The field is `matmul_macs_per_cycle_16bit`. Width-specific fields select other
modeled rates; their existence is not an API or hardware-support claim for
every dtype. The model pads M/N to 16 and K to its operand-width quantum before
charging MAC work. Include the setup cost and lowered instruction count when
estimating a tiled implementation; mathematical FLOPs alone omit padded work.

With one MAC counted as two FLOPs, a **nominal** cube roof is
`2 * MACs_per_cycle * cube_frequency_Hz * active_cube_count`. This combines a
model throughput parameter with a stated clock. It is not measured sustained
throughput, and changing the clock does not convert a complete model trace
into a hardware latency prediction. Multiple cube stages share the same cube
resource; charge all their work to it.

## Clock provenance

The read-only 2026-09-07 query retained these configured/reference values:

| Family / selected SKU | Cube MHz | Vector MHz | Observation scope |
|---|---:|---:|---|
| A2 / Ascend910B3 | 1800 | Unknown | Installed platform file read on the A2 machine |
| A3 / Ascend910_9362 | 1500 | Unknown | Matching catalog entry read elsewhere; fresh A3 connection unavailable |
| A5 / Ascend950PR_9589 | 1650 | 1650 | Installed platform file read on the A5 machine; chip-family query confirmed Ascend950PR |

The SKU selection follows the local inventory; these are not universal family
frequencies or live/sustained clock samples. The exact receipt retains the
failed A3 query and null live-clock fields.

Read the actual machine's installed CANN `platform_config/<SKU>.ini`, including
`cube_freq` and, when present, `vec_freq`. Match the SKU to a read-only device
query. A toolkit contains profiles for devices that are not installed: merely
finding an INI file does not identify the local board.

Record configured/nominal clocks separately from a live clock sample. Clock
queries vary by installed driver; inspect the supported `npu-smi info` commands
and never use a frequency-changing command for this check. Missing vector
frequency is unknown, not implicitly equal to cube frequency. The dated
clock receipt records this task's
queries and their individual limits without access coordinates.

## Vector and mixed-kernel analysis

Vector time is difficult to infer from a scalar operation count. Instruction
selection, casts, reductions, register dependencies, load/store issue, bank
conflicts and overlap all matter. Use the current model's per-operation/VF
costs and dependency trace, or a scoped measurement. Keep vector element work
separate from cube FLOPs; do not divide it by an FP16 cube peak. An unknown
vector duration remains unknown in an estimate.

Sum work on each actual serialized resource, then take the maximum resource
work as a fixed-work lower bound. Preserve core/lane identity: two vector
participants can run concurrently. Dependency critical paths and startup/drain
can strengthen the lower bound. The result has the model's units and scope.

Hardware timing needs valid samples and a stated warmup, aggregation and
participant denominator. For an expected nonempty compute kernel, all-zero
total-cycle counters do not establish utilization. A zero for an individual
unused pipe can be valid. Retain a latency-only result when its timing is
valid but its utilization counters are unavailable. See the agent's
[Roofline method](../../agent/en/references/roofline.md).
