# M10-088 — Masked stride-one load reads a full register on CCE/PTO

Status: **open**, 2026-09-16. Scope: A5 CCE and PTO-ISA `vf.load` with
`blk_stride=1` and a partial predicate.

The functional model reads only active addresses and zeroes inactive lanes.
The CCE register emitter instead selects an unconditional `vlds(..., NORM)`
for stride one, followed by masked `vand`. PTO-ISA shares this register
emission. The register values agree, but the memory footprints differ.

Minimal source pattern inside a VF, with exactly 64 allocated FP16 elements:

```python
r = Reg(DT.half)
low = MaskReg(DT.half, init_mode=MaskType.LOWHALF)
ub_to_reg(r, source[0:1, 0:64], blk_stride=1, mask=low)
```

The emitted `vlds` reads 128 half elements before `vand` zeros the upper half.
The corresponding model accepts the 64 active elements. Thus a passing
numerical or functional check does not establish an in-allocation load.
This was found while checking the generated HiF8 tail source; no device
fault is asserted. PyPTO's observed native output uses predicate-bearing
`vsldb` for this form and is not included in this confirmed emitter defect.

Locations: [CCE `op_vf_load`](../../ascriptor/backends/cce/emit.py),
[model `op_vf_load`](../../ascriptor/backends/sim/interp.py).
The HiF8 kernel uses an actual half-width UNPK load followed by deinterleave.
This source snapshot contains no device receipt for that scoped implementation;
it does not close this generic masked-load discrepancy.

Closure requires preserving the active-address footprint in native emission,
including typed/carrier register forms, zero masks, the final legal active
element and a first-invalid-active-address counterexample. Keep the existing
bounds checks; do not make the model read outside allocation to match the shortcut.
