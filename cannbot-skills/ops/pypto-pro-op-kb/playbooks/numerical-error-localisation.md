# Locate a numerical error in a multi-stage kernel

Use this when a kernel is structurally correct — it runs, most cases pass — but
a graded output misses the accuracy gate and you do not know which stage is
responsible.

It exists because on one operator (MlaProlog) the obvious statistic pointed at
the wrong stage **twice**, costing two full build-measure-revert cycles. The
technique below found it in one run.

---

## 0. Before anything: confirm the gate is real

A failing report is evidence about *the gate you ran*, not about the kernel.
Two checks first, both cheap, both of which have produced false failures:

- **Use the task's configured comparator; never silently reimplement it.** A hand-written
  approximation once reported **0/20 on kernels that were partly
  fine**, by getting four things wrong at once — the flag threshold (`10 ×
  threshold`, not `threshold`), the bf16 small-value boundary (`2**-8`, not the
  fp16 `2**-11`), the cancel region (absent), and the CPU baseline (absent).
  See [benchmark-scoring.md](benchmark-scoring.md).
- **Judge with the harness verdict, not a summary statistic.** `MARE` is a max
  dominated by the smallest `|golden|`, so it moves *non-monotonically* with
  real accuracy: a strictly more accurate scheme measured a **worse** MARE on
  this operator. Use `passed`.

Only once the gate is trustworthy is a failure worth investigating.

---

## 1. Measure the quantity the gate measures

Match the statistic to the region that is failing. Getting this wrong is the
single most common way to chase the wrong stage.

| the gate bounds | measure |
|---|---|
| relative error in a normal region | **relative** error |
| an absolute bound on near-zero outputs (small-value / cancel regions) | **absolute** error |

On MlaProlog every failure was in the small-value region — an absolute
`2**-16` bound on cancellation-born near-zeros — while the diagnostic reported a
*relative* error cut at `0.1 ×` the tensor RMS to stop cancellation dominating
the statistic. That cut is reasonable for describing a tensor's bulk and
completely wrong for this question: it measured the **body** of the distribution
while the failures lived in the **tail**. It reported the offending stage as
"1.2× the CPU reference" when the stage that actually mattered was 2.8× worse.

Report **percentiles** (p50 / p99 / p99.9 / max), not a single number. The ratio
against the reference at the top of the distribution is the factor a fix must
buy.

---

## 2. Decompose by exact substitution, and check the quadrature sum

The technique. For a chain `A → B → C`, re-run it with each stage replaced by
its exact (fp64) value, and difference against the fp64 truth:

```
inherited = |exact_C( device_B )      − truth|   # everything upstream of C
own       = |exact_C( exact_B )       − truth|   # C's own contribution
total     = |device_C( device_B )     − truth|
```

Independent rounding sources add in quadrature, so

```
total² ≈ inherited² + own²
```

**That identity is the check that the decomposition is trustworthy.** On
MlaProlog it held to three digits — `3.51² + 2.43² = 4.27²` — which is what made
it safe to act on after two earlier misdiagnoses.

Then compare each component against the same decomposition of the CPU reference.
The measured table:

| contribution | kernel | CPU reference | verdict |
|---|---|---|---|
| inherited from the first projection | 3.51e-6 | 1.24e-6 | **2.8× worse — the target** |
| the second projection's own | 2.43e-6 | 2.44e-6 | already reference-quality |

The second stage needed *nothing*. Without the decomposition it looked like the
obvious suspect, and one cycle was spent rewriting it.

---

## 3. Cost the fix before building it

A fix that trades throughput for accuracy can lower the score it was meant to
raise. Price both sides in the benchmark's own units first:

```
gain = Σ_newly_passing (0.3 + 0.5·score_i) / N · 100
cost = Σ_all_cases 0.5·Δscore_i / N · 100
```

Worked example: converting the last failing MlaProlog case needed a matmul in
the cube's fp32 mode, which gains one case (**+1.7 points**) and costs ~2×
runtime across all twenty (**−5 points**). **Net −3.3.** The right call was to
ship 19/20 and record the arithmetic, not to chase the case count.

---

## 4. Back up before restructuring anything that passes

Every kernel rewrite in this class is a candidate revert. Copy the working file
first. Two of three restructures attempted on this operator were reverted, and
the backups are the only reason the passing state survived.

---

## Discipline: three ways this investigation went wrong

**A whole-tensor statistic answered a different question than the one asked.**
See §1. Match the statistic to the failing region before drawing any conclusion
from it.

**A refutation went stale.** An experiment showed a candidate fix changed
nothing, and that was correctly recorded as refuted — but it had been measured
while an *upstream* error was 1.5× larger and swamped the effect. After the
upstream stage was fixed, the same candidate became the dominant remaining term.
The negative result was valid; the conclusion drawn from it outlived its
conditions.

> **Record the conditions a negative result was measured under**, and re-open it
> when the thing that dominated it changes. "Refuted" is a statement about a
> configuration, not a permanent property.

**Reasoning substituted for measurement.** Several rounds of error modelling
produced confident predictions ("~2e-3 relative, well under threshold") that the
device contradicted. The models were not useless — they sized the candidates —
but every one that mattered was decided by a measurement, and the two that were
acted on without one were both wrong.

---

## Settle precision plans offline

Choosing *which* scheme to build needs no accelerator. Replay the kernel's
dataflow in torch with the GM dtype and the operand split depth as parameters,
and grade with the benchmark's real comparator. The whole scheme comparison for
MlaProlog — five candidates across three problem sizes — ran on a laptop with no
NPU and picked the one that worked:

| scheme | M=1 | M=128 | M=512 |
|---|---|---|---|
| bf16 GM, 1 term | FAIL | FAIL | FAIL |
| fp32 GM, 1 term | FAIL | FAIL | FAIL |
| fp32 GM, 2-term split | PASS | FAIL | FAIL |
| **fp32 GM, 3-term split** | **PASS** | **PASS** | **PASS** |

Only the kernel implementing the chosen scheme needs hardware. See
[../constraints/precision.md](../constraints/precision.md) for what the schemes
mean.

---

## Evidence

All figures were measured on Ascend A5 with the comparator recorded by that run,
on MlaProlog and MLA. Results over the session: MlaProlog 0/20 → 19/20
(57.96 → 61.84), MLA never-ran → 20/20. Per-stage numbers in
`custom/mla_prolog/DESIGN.md` §9.2–9.8 and
[../references/pypto-pro-framework-findings.md](../references/pypto-pro-framework-findings.md)
findings 21–24.
