# A5 roofline workflow

## Contents

- [Evidence gate](#evidence-gate)
- [Primary source paths](#primary-source-paths)
- [Derive, do not transcribe](#derive-dont-transcribe)
- [The scoring anchor is not the hardware roofline](#scoring-anchor)
- [Calibration measurements](#calibration-measurements)
- [The vector-only ceiling](#vector-only-ceiling)
- [Symbolic roofline](#symbolic-roofline)
- [Measurement loop](#measurement-loop)
- [A5 原语代价表（每 64 lane 寄存器）](#a5-primitive-costs)
- [Measured levers, ranked by what they actually returned](#measured-levers)
- [Scope of retained examples](#example-scope)
- [When the local levers plateau: structural levers](#structural-levers)
- [When to stop](#when-to-stop)
- [Provenance of the numbers on this page](#provenance)


Load this reference only after the runtime or build configuration confirms an
A5 target (`NpuArch=3510` / `dav-c310`). It must not supply constants or tuning
rules to an unknown or non-A5 target.

## <a id="evidence-gate"></a>Evidence gate

Before calculating a roofline:

1. Record the detected device model and architecture.
2. Record the CANN and PyPTO versions and the resolved `pypto_pro` package path.
3. Read constants from the platform files installed with that exact runtime.
4. Retain the profiler output used for every measured claim.

If any input is unavailable, leave the numerical estimate unverified and use
the profiler result as the only tuning basis.

## <a id="primary-source-paths"></a>Primary source paths

Resolve these under the installed, version-recorded PyPTO/CANN tree; do not copy
values from another checkout:

- `framework/src/platform/parser/simulation_platform/platform_config/950DT_957x.ini`
- the matching `950DT_958x.ini`, `950PR_957x.ini`, or `950PR_958x.ini` for the
  detected SKU
- `framework/src/platform/parser/platforminfo.ini`
- `python/pypto_pro/runtime/compile_config.py`
- `python/pypto_pro/runtime/platform.py`

The files are primary evidence for the matching installed version only. If a
path or key differs, search the installed source and record the replacement;
do not infer a value.

## <a id="derive-dont-transcribe"></a>Derive, do not transcribe

`tools/roofline_a5.py` parses whichever ini matches the detected SKU and derives
the cube, HBM, UB, L1 and L0 figures from it, so no constant needs to live in
this page and detach from the installed version:

```
python tools/roofline_a5.py --op <op> --sku <ini stem>
python tools/roofline_a5.py --selfcheck
```

The derivations it applies, so they can be checked rather than trusted:

```
cube FLOP/s = M*N*K (from [DtypeMKN]) * 2 * cube_core_cnt * cube_freq
HBM  B/s    = cube_core_cnt   * [AICoreMemoryRates]ddr_rate    * cube_freq
            + vector_core_cnt * [VectorCoreMemoryRates]ddr_rate * vec_freq
estimate    = max(bytes / HBM, MACs*2 / cube FLOP/s)
```

**Pick the SKU deliberately.** All four A5 inis share `NpuArch=3510`,
`cube_freq=1650`, and the same L2/UB/L1/L0 sizes, but they differ in core count
*and* by more than a factor of two in `ddr_rate`. A roofline computed against the
wrong one is wrong by that factor. `950DT` and `950PR` at the same core count are
not interchangeable, and `platform.py` reports only `DAV_3510` plus core counts,
which narrows to two SKUs rather than one. Record which ini was used.

## <a id="scoring-anchor"></a>The scoring anchor is not the hardware roofline

`cann-bench` scores each case with a saturating ratio, not a speedup
(`kernel_eval/benches/cann_scoring.py`):

```
score_i = (T_baseline - T_HW) / ((T_cand - T_HW) + (T_baseline - T_HW))
```

Both anchors are per-`(op, case_id)` in `tasks/metadata/<hardware>.json`, and
`<hardware>` comes from `resolve_hardware(torch.npu.get_device_name(...))` —
where **only names beginning `Ascend950PR` map to `950pr`**. A box reporting
anything else finds no metadata, gets no anchors, and earns no performance
points at all. Check that before interpreting a zero.

Two consequences:

- **The anchors travel with the task, not with the box.** Running on hardware
  with more cores or bandwidth than the anchor assumed makes `T_HW` easier to
  approach; less makes it harder. Neither changes the score, only what it means.
- **`T_HW` is theoretical peak and generally unreachable.** A score below 1.0 is
  the normal case; calibrate against measurement, not against the anchor.

## <a id="calibration-measurements"></a>Calibration measurements

Measured ratios, not platform constants, recorded with provenance because the
evidence gate above requires it.

**Model versus published anchor** (`roofline_a5.py --selfcheck` against
`tasks/metadata/950pr.json`, SKU `950PR_957x`):

| operator | model / `t_hw_us` | reading |
|---|---|---|
| `add_rms_norm_dynamic_quant` | 0.97 (0.96–1.00) | anchor is the pure memory roofline |
| `mla` | 1.06 (1.03–1.14) | anchor is memory or cube, whichever binds |
| `dequant_swiglu_quant` | 0.69 (0.62–1.00) | anchor ~1.45x more conservative than bytes alone: it prices vector work (the `exp` in SiLU) |
| `mla_prolog` | 0.62 (0.55–1.14) | anchor ~1.6x more conservative: weight streaming, not peak bandwidth |

Where the anchor is *more* conservative than the byte count, the case has more
attainable headroom than a naive memory roofline suggests.

**The anchors discount causal work.** Adding a bottom-right-aligned causal
fraction to the `mla` model moved it from 1.22 (worst case 2.28) to 1.06 (worst
1.14). The kept fraction for mask `j <= i + (S_kv - S)` is
`(S*(S_kv-S+1) + S*(S-1)/2) / (S*S_kv)` — about 0.5 when `S == S_kv`, about 1.0
when `S << S_kv`. Skipping masked work is therefore required to reach the anchor
on square cases and buys nothing on short-query ones.

**Achievable fraction of peak.** Independent A5 measurements: an aligned-plane
MTE2 read reaches ~1.7 TB/s, and a tuned vector-only Ascend-C
`AddRmsNormDynamicQuant` reaches 1.17 TB/s (8192x8192 fp16 in 401 us), which its
authors describe as sitting at the memory roof. CANN's own kernel for that case
manages 0.69 TB/s. The corresponding `t_hw_us` implies 2.80 TB/s.

## <a id="vector-only-ceiling"></a>The vector-only ceiling

If the per-core `ddr_rate` split is physical, a kernel using only
`section_vector()` can reach at most the vector-core share — roughly half the
aggregate on `950PR`. On `add_rms_norm_dynamic_quant` case 1 that caps the score
near 0.77 however clean the kernel is, because score 0.8 needs 1.59 TB/s against
a ~1.48 TB/s vector-only theoretical ceiling.

Treat this as a hypothesis, not a constant. The same ini block lists
`ddr_rate=31` beside `ub_to_ddr_rate=128` for AICore and `16` beside `40` for
VectorCore; the units are not self-consistent, and the reading above is simply
the one that reproduces the published anchors.

**Settle it by measurement:** a pure-DMA copy of identical bytes, vector-only
versus cube+vector, nothing else varied, with a control variant in the same run.
Design around the answer only afterwards.

## <a id="symbolic-roofline"></a>Symbolic roofline

Use version- and SKU-specific values read from the source above:

```text
cube_time   ≈ total_MACs / detected_cube_throughput
vector_time ≈ vector_work / detected_vector_throughput
move_time   ≈ bytes_per_path / detected_path_bandwidth
estimate    ≈ max(cube_time, vector_time, move_time)
```

This ranks hypotheses; it does not establish actual latency. Reload counts,
launch overhead, dependencies, occupancy, and compiler scheduling can change
the result.

## <a id="measurement-loop"></a>Measurement loop

1. Freeze a passing correctness test.
2. Capture a baseline with the same input, launch geometry, warm-up, and
   profiler configuration used for later variants.
3. Read the dominant measured pipe from the retained profiler artifact —
   `aic_mac_ratio` versus `aic_mte2_ratio` on the cube side, `aiv_vec_ratio`
   versus `aiv_mte2_ratio` on the vector side.
4. Change one relevant factor: work distribution, tile shape, buffering,
   on-chip residency, accumulation structure, or documented dtype.
5. Re-run correctness and profile again.
6. Keep the change only when the measured target metric improves without
   violating the numerical contract.

Use [msprof-guide.md](msprof-guide.md) or
[msprof-op-guide.md](msprof-op-guide.md) for collection, and
[csv_fields_reference.md](csv_fields_reference.md) for the fields supported by
the current parser.

## <a id="a5-primitive-costs"></a>A5 原语代价表（每 64 lane 寄存器）

这张表决定绝大多数 dataflow 决策，**设计阶段就要用**。它从 `pypto-pro-op-perf-tune/SKILL.md`
移到本页，因为它是 target 相关的实测值，而核心 Skill 对任意目标都会加载。

- **target**：A5 `Ascend950PR_9579`，56 vector core
- **测得时间**：2026-07（随 A5 实测批次）
- **适用范围**：仅该 SKU。**换 SKU 数值会变，触发条件与规则不变**——与目标无关的那条结论
  （跨 lane 与否是结构选择）留在 SKILL.md，本表只提供数量级。

| 原语 | 代价 | |
|---|---|---|
| `vf.gather` | **~20 ns** | 跨 lane |
| `vf.scatter` | **~18 ns** | 跨 lane |
| UB 往返（**含必需的 `vf.mem_bar(VST_VLD)`**） | **~16 ns** | 跨 lane |
| `vf.load_align` / `vf.store_align` | **< 1 ns** | 不跨 lane |
| 算术本身 | **~0.3 ns** | 不跨 lane |

**三个跨 lane 原语彼此相差不到 25%**——ISA 不提供寄存器级 lane shift，所有 UB 中转替代品
交同样的税。

两条推论：

- **无 bank 冲突的 gather ≠ 便宜的 gather。** padding pitch 仍然必要（冲突时再差 15×），
  但消除冲突不会让它接近 `load_align`。
- **跨 lane 与不跨 lane 相差 20–35 倍。** 一个内层循环里如果有 1 次 gather + 2 次 scatter，
  即使两种写法 op 数完全相同，**UB 寻址也会占到 83%、算术只占 17%**
  （证据：一个连续扫描算子，64 元素 13 op）。

## <a id="measured-levers"></a>Measured levers, ranked by what they actually returned

From two attention-family operators taken from correct-but-slow to the
bandwidth roof. Each was chosen from a per-kernel profile, never guessed, and
each figure is a device measurement on the same case before and after.

**1. Delete work the shape makes unnecessary.** Padding the token count up to a
whole M tile needs a pad kernel before and a strip kernel after — but only when
`M` is not already a multiple of `TM`. Guarding both launches on `M == Mp` and
letting the cube read and write the real tensors removed **24.5%** of the
largest case outright. Look for this before optimising anything.

**2. Size the transfer, not the tile.** A copy kernel moved 67 MB at **121 GB/s**
against a roof near 1 TB/s purely because its column tile was 512 elements — a
1 KB DMA. Widening to 4096 (8 KB) was the whole fix. Choose the column tile from
the transfer size you want.

**3. Fold the inner dimension into the task index.** The same kernel strided its
outer loop over rows with an inner loop over column tiles. At `M = 1` — the
decode shape, and 8 of 20 cases — that is one row, so one core did everything
and 31 idled. Striding over `(row, column_tile)` pairs fixes it with no change to
the body. Whenever the outer dimension can be small, fold the inner one in.

**4. One task per output row can be pure descriptor overhead.** A RoPE kernel ran
`M * N = 65536` tasks each issuing eight 128-byte DMAs: **506 us**, 33% of the
case. Batching 64 rows into one `[TRR, HALF]` tile per task — same arithmetic,
same bytes, one strided transfer instead of 64 — took it to **46 us**. The
register function only has to become a loop over registers; row boundaries do
not matter to an elementwise pass because each row's operands sit at the same
offset in every tile.

**5. Widen the cube's N tile to lengthen the DMA run.** At `TN = 64` each row of
a B tile is a 128-byte run of a 24576-wide weight, and the matmul ran at
~1 TB/s; `TN = 128` doubled the run and reached **1.72 TB/s**, at the roof.
Widths constrain which kernels can use it, and a partial N tile does **not**
fault — it silently corrupts its own share of the output — so keep the narrow
variant for any width the wide tile does not divide.

**6. Hoist the reused operand out of the inner loop — but watch parallelism.**
Flattening `(m_tile, kv_tile)` into one task index re-loads the query tile once
per KV tile: 2.1 GB of traffic where 134 MB suffices. Hoisting it out entirely
fixed that and immediately broke something else — it leaves only `n_mt` tasks,
and decode shapes have `S = 1`, so one case had **4 tasks for 32 cores** and got
*worse* (16.7 → 37.9 us). The shape that works is a task per (M tile × *group*
of KV tiles), with the host choosing `n_g = ceil(cores / n_mt)`: just enough
groups to refill the cores, so the operand is re-read `n_g` times rather than
`n_kt` times. Final: **15.7 us**.

**7. Larger M tiles cut weight re-reads, if L0A allows.** The K operand is
re-read once per M tile. Doubling `TM` halves that, but a `[64, 512]` narrow
query tile is 64 KB — the entire L0A, with no room for a second operand.
Walking the contraction dimension in blocks instead (two 256-wide blocks, both
resident in **L1** across the loop, moved L1→L0A per use) makes `TM = 64` fit and
keeps GM traffic at once per task. Worth ~10% here — smaller than the traffic
arithmetic predicted, because the re-read was already largely L2-resident, so
what improved was L2 pressure rather than DRAM traffic.

**Know when to stop.** After these, the kernels moved 604 MB in 351 us
(**1.72 TB/s**), 125 MB in 72 us (1.74 TB/s) and 234 MB in 139 us (1.68 TB/s),
against a ~1.6 TB/s weight-anchored roof. At that point nothing further is
available by making a kernel faster — only by making it move fewer bytes. Say so
and stop, rather than continuing to tune.

## <a id="example-scope"></a>Scope of retained examples

The KB's
[BF16 operand-reuse implementation](../../../pypto-pro-op-kb/examples/samples/bf16_matmul_operand_reuse/bf16_matmul_operand_reuse_impl.py)
demonstrates one reuse topology and embeds a correctness test. It does not prove
that an operator is cube-bound or that the topology is faster on another shape
or target. Profile the current kernel.

## <a id="structural-levers"></a>When the local levers plateau: structural levers

Every lever above preserves the algorithm — it re-times, de-duplicates, or
re-lays-out work that already exists. When those plateau against a traffic or
occupancy wall that the byte arithmetic says is still reducible, the remaining
moves change *what work and traffic exist*. Three, transferred from measured
attention-family work on the same silicon (the structure transfers; the
numbers do not):

- **Keep a loop-carried reduction on chip.** An accumulator (or running
  max / denominator) that round-trips a GM workspace every step is often the
  dominant traffic of a low-arithmetic-intensity reduction — a decode-shaped
  attention with few query rows is the canonical case. Keep it resident in UB
  (or L0C) across the loop and touch GM once, at the contract boundary.
  Precondition: it fits the on-chip budget at the target tile — check the real
  headroom, since bring-up kernels often leave UB largely free once wide
  temporaries are gone. Risk: on-chip accumulation is a reduction/cast-order
  change; validate at the new boundary, and prefer UB residency when the cube
  overlap depends on L0C slots.
- **Read each shared operand once across cores.** When every core re-reads the
  same GM region, the aggregate read is `operand_bytes × core_count`. Cheapest
  first: **measure whether L2 already absorbs it** — many cores reading
  identical addresses often hit L2 rather than HBM, and the wall may be far
  smaller than the byte count implies; then re-tile so the shared dimension is
  the inner loop; an explicit broadcast/shared-load scheme comes last, because
  it trades a read wall for a sync wall. Measured lever 6 above is this
  lever's instance, including the failure mode (hoisting fully starved the
  cores; the group form fixed it).
- **Split the reduction dimension to raise occupancy** (the flash-decoding
  shape). When the parallelised dimension is small and further splitting is
  exhausted, split the long reduction axis across cores into mergeable
  partials — for softmax: partial output, running max, running denominator —
  and combine them in a cheap final merge. Preconditions: the reduction admits
  a mergeable partial form, and the merge stays small next to the occupancy
  gained; sweep the split count, since over-splitting makes the merge
  dominate. The merge is its own cast-order change — validate the merged
  result against the reference.

These are larger edits that usually move reduction or cast order: apply one at
a time and re-verify correctness at the new boundary before measuring.

## <a id="when-to-stop"></a>When to stop

Stop at a wall proven with data: the moved bytes are irreducible — each read or
written once, each feeding the contract — **and** occupancy is at the device
limit for the parallelism the algorithm exposes. Record the measurement that
proves it. "`MTE2` is at 98%" alone does not; "`MTE2` is at 98% and every byte it
moves is read exactly once" does.

## <a id="provenance"></a>Provenance of the numbers on this page

The ratios in "Calibration measurements" are reproducible without hardware:
`roofline_a5.py --selfcheck` recomputes them from the installed platform ini and
`tasks/metadata/950pr.json`. They are model-versus-anchor comparisons, not
profiler output.

The achievable-bandwidth figures are cited measurements from other A5 work, kept
because the anchor alone gives no sense of what fraction of it is reachable. They
are scenario-specific: do not treat 1.17 TB/s as a target for a different
operator or shape. Any *new* bandwidth or duration claim added here needs a
version-tagged profiler artifact and the command that produced it.
