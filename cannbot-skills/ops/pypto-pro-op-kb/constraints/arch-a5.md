# A5 platform constraint discovery

## Applies to

Only a confirmed A5 target (`NpuArch=3510` / `dav-c310`). Do not use this page
for an unknown target or for A2/A3.

## Platform gate

1. Detect the target with
   [`../../ops/pypto-pro-environment-check/scripts/get_npu_arch.py`](../../pypto-pro-environment-check/scripts/get_npu_arch.py)
   or the build configuration.
2. Record device model, SKU, CANN version, PyPTO version, and the resolved
   `pypto_pro.__file__`.
3. Select the platform file matching that exact installed version and SKU.
4. Extract only the constants needed for the current design and cite the file
   and key in `EXPLORE_REPORT.md` / `DESIGN.md`.

Fail closed: if the architecture or matching source cannot be confirmed, do
not substitute remembered A5 values.

## Primary source paths

Resolve under the version-recorded installed PyPTO/CANN tree:

- `framework/src/platform/parser/simulation_platform/platform_config/950DT_957x.ini`
- the matching `950DT_958x.ini`, `950PR_957x.ini`, or `950PR_958x.ini`
- `framework/src/platform/parser/platforminfo.ini`
- `python/pypto_pro/runtime/compile_config.py`
- `python/pypto_pro/runtime/platform.py`

Use these files to obtain core counts, memory budgets, datatype/fractal
settings, paths, and compile target for the detected SKU. Do not duplicate
their numbers in this KB: doing so detaches constraints from the installed
version.

## Design checks

- Derive each live tile's byte size from shape and dtype.
- Sum simultaneous allocations separately for Vec, Mat, Left, Right, and Acc.
- Compare those sums with the detected target's corresponding limits.
- Read dtype-dependent contraction geometry from the matching platform file;
  do not assume one K-fractal for every dtype.
- Read launch width from runtime platform information rather than hard-coding
  a SKU's core count.
- Re-run correctness after any dtype, layout, buffering, or tile-size change.

## The UB capacity dispute is settled, and the settling generalises

Two values were in circulation for A5 UB: **248 KB**, from the installed tutorial
(`.../tile_based_python_programming/multi_core_partitioning_and_Tiling.md`, stated
once as the limit and once inside a worked FP16/FP32 budget example), and
**256 KB / 216 KB-with-SIMT** from a sibling DSL's device profile for the same
silicon.

**The platform file decides it.** `950PR_957x.ini` carries `ub_size=253952`, which
is 248 KB exactly, and so do `950DT_957x`, `950PR_958x` and `950DT_958x`. The
tutorial and the platform config agree; the 256 KB figure does not describe the
budget the toolchain compiles against on this SKU.

Two things about *how* that was settled matter more than the number:

- **The search that concluded "no source settles this" had looked in the wrong
  tree.** It grepped `python/pypto_pro` and found no UB constant — correctly, the
  Python package has none. The budget lives in the platform `.ini` named in
  *this page's own* primary-source list, four sections up. That is
  [investigation-discipline §11](../references/investigation-discipline.md)'s
  first requirement — *where was looked, and for which names* — failing on a real
  question, and the fix was one `find`.
- **Carry the key, not the value.** `ub_size` in the SKU's platform file is what
  a design should resolve at planning time. `248` written into a document is a
  number that detaches from the version that produced it, which is the practice
  the top of this page forbids.

The same file settles the neighbouring rows, which is worth knowing because it
makes the whole capacity question cheap: `l0_a_size`, `l0_b_size`, `l0_c_size`,
`l1_size`, `bt_size`, `cube_core_cnt`, `vector_core_cnt`, and `ubblock_size` — the
32-byte UB quantum that the alignment rules across this KB keep arriving at
independently — are all keys in the same `.ini`. Read them there rather than from
any page, including this one.

**What this does to the sibling DSL's device table.** Checked row by row against
`950PR_957x.ini`, it is right about L0A/L0B (64 KB each), L0C (256 KB), L1
(512 KB), BT (4 KB), the 28/56 core split, and A2/A3's 192 KB UB — and wrong only
about A5 UB. So it is a usable cross-check, and the one row it got wrong is the
one that was quoted most. Verify against the platform file before reusing any row
of it.

## Three cross-DSL notes about this silicon

**未在 PyPTO-Pro 上验证——由 EasyASC 移植的假设 (unverified on PyPTO-Pro — an
assumption ported from EasyASC).** These are statements about the chip rather than
about a DSL, which is why they are here rather than being dropped as another
project's platform data. Two are design bounds and one is a negative.

- **A fused-bias contraction is capped at `N` / `Cout` ≤ 512.** The total is
  anchored — `bt_size=4096` in the SKU's platform file, which is 1024 fp32/int32
  elements. The **unverified** half is the structure: the bias table is described
  as two slots, with a shortcut-matmul or conv bias required to fit *one*, which
  halves the usable width to 512. Check the slot structure before a tiling leans
  on it, but check it *early* — it caps a dimension tile planning otherwise treats
  as free. (For scale: A2/A3's BT is 512 B, so the equivalent bound there is 64.)
- **The 12-bit `n_burst` field is an A2/A3 restriction that A5 does not inherit.**
  On C220, padded GM↔UB transfers encode the burst count in `[0, 4095]` and a
  request for 4096 silently does nothing. This is carried as a **negative**: do
  not budget around a 4095 burst cap on A5, and do not port an A2-era tiling
  workaround that exists only to respect it. Recorded because a silent no-op is
  the kind of rule that gets applied defensively long after it stopped applying.
- **950 and 950PR share the C310 instruction and codegen family**, differing in
  default core count, vector lane count, and the debug SoC string. The platform
  files corroborate the shared family directly — `950PR_957x.ini` and
  `950DT_957x.ini` both declare `dav-c310-cube` — which is why a finding measured
  on one is worth *testing* on the other, and equally why it is not automatically
  valid there, since anything derived from core count is exactly what differs.

## Performance boundary

Platform configuration values are roofline inputs, not measured kernel
performance. Use the target profiler to determine the current bottleneck. See
the platform-gated
[A5 roofline workflow](../../pypto-pro-op-perf-tune/references/a5-roofline-and-levers.md).
