# Quantized matmul with a direct floating-point epilogue

## Applies when

**A5 only.** The dataflow below freezes A5 memory capacities, event mapping and
width gates; the routing table selects this page by operator class, not by
target, so check the target before taking anything here as a constant.

An integer Cube contraction remains in int32 through the full K reduction, then
feeds a floating-point Vector epilogue in the same kernel. The output is
fp16/bf16 rather than a clamped integer, so
`quant-matmul-scaled-mm.md`'s round/clamp skeleton is not the formula oracle.

## Invariants

- Freeze the epilogue order from the golden: integer pre-bias, FP32 scale,
  optional offset, optional per-token scale, floating post-bias, final cast.
- Keep the compact Acc physical window unchanged from the first matmul through
  the handoff.  Pin the selected physical family as `[tile_m, tile_n]`; do not
  hard-code a baseline `TN` after adding a wider tiling key.
- Use `make_tile_group + auto_mutex` for every rotating L1/L0 buffer. Advance
  L1 once per non-empty K1 step and L0 once per non-empty K0 step.
- For direct Acc-to-Vector TMOV, do not use Acc `Final` phase: there is no
  `STPhase.Final` consumer to reset the unit flag.
- Prove UB capacity and GM beat ownership for the selected hardware split. Use
  a single-AIV fallback when an unaligned output stride makes sibling stores
  share a 32-byte beat.
- Balance cross-core READY/FREE credits for repeated same-core reuse; one-tile
  success is insufficient.  On the validated direct path, prime and return FREE
  on MTE3, and place the per-task return after the conditional output store.

## Reusable gated-width pattern

Keep the common TN256 path and add TN512 as a compile-time tiling-key family,
not as a global width change.  The retained A5 gate requires all of:

```text
M >= 257
K >= 3584
N % 512 == 0
per-token scale is absent
baseline P0 tile count >= 56
```

The selected `tile_n` must flow through L1/L0B/Acc shapes, offsets, valid
shapes, UB capacity checks and the JIT key.  The validated high-water marks are
L1 393216 B, L0A 16384 B, L0B 65536 B, L0C 262144 B and UB 203264 B on A5.
Small/decode TN512 is a negative template: it passed codegen inspection and was
bit-exact on every case it was run against, yet the decode-shaped subset measured
**slower than the baseline** — roughly a third slower in geometric mean, and worse
still on the narrowest shapes. Exactness plus a clean CCE does not imply the tile
choice pays on small shapes.

For FP16/BF16 output, two adjacent row stores of 256 columns are universally
32-byte-beat disjoint when their byte gap is at least 32:

```text
gap_bytes = (N - 256) * 2
```

判据只有这一条不等式：**`gap_bytes >= 32` 才有不相交的 32B beat**。按它推：
N=272 给出 `gap_bytes = 32`，**满足**判据；N=271 只有 30 字节，不满足，且有具体的
边界对齐使相邻 AIV store 落在同一 beat 上。

因此可接纳的最小宽度是 **N=272**，不是 273——早前写成 273 是把"不满足判据"与
"恰好落在判据边界"混为一谈。若某个目标上要额外留出余量，把余量写进不等式
（例如要求 `gap_bytes > 32`）并说明理由，不要在正文里另立一个与不等式不一致的门限。

用 dual-AIV SplitM 只限已验证的非 decode family 且 `gap_bytes >= 32`；
更窄的尾巴与 decode 留在单 AIV 回退路径上。

本仓不保留该 A5 study 的 kernel，因此这里不给出可点开的产物路径。**相应地，
上面的相对加速比按 study 证据读——量级可用于判断方向，绝对数值不可复核，
不要当作目标或验收线。** 可复现的是它覆盖到什么程度：4×L1/2×L0 轮转在 K=1/63/64/65/127/128/129/257/513、M/N 尾块、多 tile 复用
与 batch 上通过逐位比较——**覆盖面本身是可检查的判据**（跨越轮转的每个边界），
按这个覆盖面在自己的目标上重做，比引用一个打不开的文件有用。
Use [the develop-skill reference](../../pypto-pro-op-develop/references/cv-matmul-direct-buffering.md)
for resource formulas, synchronization details and promotion gates.

## Validation boundary

The dataflow is an A5 study whose kernel is not retained in this repository,
not a universal code template. Recheck
the target SDK's Acc-to-Vector modes, auto-mutex lowering, event mapping and
memory capacities. Tile width/depth changes require fresh codegen, locked
device exactness and kernel-only profiling.

Do not infer that a target is met from a local geometric speedup alone. In one
recorded run, a ~1.4x local kernel gain on three shapes read as a ~1.5x gain across
the full shape set, yet the end-to-end result improved far less, because startup,
handshake and underfilled-wave costs still dominated several shapes. Quantify that
gap before choosing the next experiment.

The subsequent gated WideN/N-tail study reinforces the same rule.  Its ten
selected main paths improved by 1.071656x under the A5 lock and 1.061559x when
compared with the preceding kernel's device times; the N-tail case improved by
1.497716x locally and 1.381942x on the target.  Despite that agreement, the
end-to-end figure moved less than the kernel gain, because the baseline it is
measured against is not under your control and can change between runs.  Compare
device kernel times for candidate bytes, and use a geometric speedup only
against the baseline measured in the same run.  Correctness closure does not
imply a performance target is closed.
