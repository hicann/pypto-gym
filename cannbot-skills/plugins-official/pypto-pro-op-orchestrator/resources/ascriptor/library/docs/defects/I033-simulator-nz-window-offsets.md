# I033 — Offset NZ-layout UB windows

Status: open. Owner: `backends/sim/interp.py` (`Machine.storage`), frontend scalar fold.

Take `ub.nz()[1:2, 8:72]` of an FP32 `[4, 128]` UB tile. The simulator starts a `@vf` register
load, a register store, a `dma.ub_to_gm.pad` and a `Var.GetValueFrom` through it at element 136.
The printed register and DMA accesses start at fractal element 40 (RFC-0007 §2), and the scalar
load folds to `scalar.load(%ub, 136)`. No kernel is exposed today: the kernels and examples use
offset `.nz()` windows only as L1 copy sources, which the simulator places correctly.

Next: decide the RFC-0001 §6.4 and RFC-0007 §2 rule for each access, then check it with a native
probe.
