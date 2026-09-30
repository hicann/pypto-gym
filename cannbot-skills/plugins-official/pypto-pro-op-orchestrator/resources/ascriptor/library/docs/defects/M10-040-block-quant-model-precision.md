# M10-040: Block quantization exposes FP32 model reduction differences

Repair characterization (2026-09-07): the newly installed compiler/model still
fails `pack4_reuse` at its exact payload gate. Generated FP16 cube cancellation
captures at K64/K512 do not match FP64-end-rounded accumulation or FP32 chunk
sizes 1/8/16/32/64. The exact basis/orientation control passes. No replacement
accumulation rule is inferred from this finite sample; no comparator changed.
See the repair receipt (`docs/migration/fragments/defect-repairs-20260907.json`) and
generated diagnostic.


Status: open simulation precision gap; source and exact payload reference remain
unchanged. This is not a new hardware failure or an accepted tolerance change.

Current review (2026-09-07): the named functional/pipe precision failure
was reproduced again on the current source, separately from the canonical
board matrix. Both `pack4_reuse` runs retain `payload: exact comparison
failed; max finite absolute error 1.0`, with seed7013 and unchanged dimensions.
The next
experiment should isolate cube accumulation using generated cancellation,
split-K and tile-boundary cases before choosing a target-specific model.
The [active plan](README.md) preserves exact payload checks and separates any
later numerical-contract decision from an implementation repair.

The A5 `matrix_block_quant` source performs K128 FP32 cube stages, then computes
a row scale and stores E5M2 payloads through either the ordinary or pack4 path.
All12 generated references pass. Each host model passes11 cases and fails
`pack4_reuse` at one exact payload byte. The case has M384, K272, one cube and
seed7013. Both modes share the same mismatch; the exact equality gate retains
the failing case and original parameters.

The lead's dual-mode A5 checkpoint on foundation12 passes the unchanged
canonical payload reference for both modes. Instrumentation captures raw FP32
product, actual scale, pre-cast normalized value and payload. It adds private
capture buffers without changing the production math or original event protocol.
The raw, scale, normalized and payload tensors agree between the two store modes.

At row376/column113, the board records raw3.811924457550049,
scale0.17326943576335907 and normalized21.999982833862305, which independently
encodes to byte77. The model produces normalized22.000001907348633 and byte78.
This crosses the E5M2 midpoint22. The actual scale at that point agrees; the
source does not specify the silicon's internal FP32 reduction tree.

Across all49152 product elements, the board differs from FP64-end-rounded
matmul at31500 elements (maximum absolute error1.1444091796875e-5) and from an
independently written K128 FP32 scalar-k recurrence at41768 elements
(maximum3.0517578125e-5). CPU SGEMM or a scalar recurrence must therefore not be
promoted as a bit-exact silicon oracle. The proposed CPU-staged reference is
retained as a rejected provisional experiment.

Independent E5M2 encoding matches every actual normalized board value. Actual
scales equal multiplication by the typed FP32 reciprocal of224; division from
the same raw maxima differs at230 rows. The independent encoder's separate
986 signed midpoint/neighbor controls pass. These facts explain the quantizer
boundary but do not establish a universal cube accumulation model.

Executed library commit: `abe056e202d8b1740525fb59df1a8ee107e735e8`.
Wheel SHA256: `00d5526929b07f81e22b625298179e8de6bc43d143ee4ab348711059049fc18a`.
The publication-equivalent source mapping is in
`docs/migration/source-publication-equivalence.json`.
Checkpoint job: `block-quant-checkpoint-a5-v12`; checkpoint SHA256:
`4fe2d0583400df60b981c82cac942822234ea3817763a745422c88538c05af90`.
The kernel unit and its reference review retain source hashes and command
evidence. Captured tensors are ignored diagnostic outputs, not product goldens.

Continue only with a justified hardware accumulation model and independent
boundary controls. Until then, the unit reports its exact failed model case
and measured board scope separately. Full production board coverage remains
its own gate; a successful checkpoint is not a result for unrun cases.

The final corrected production source subsequently passed all12 A5 board
cases in `a5-release-matrix-block-quant-m42-v20`, using library commit
`70360d5f0b9b1632b37662231ca9355a4d049e41` and wheel SHA256
`e8e96442313da51a9ca7c01f84c8dbb6f0564a8ff769d148dab20c0551ad65f5`.
The exact source and per-case comparisons are in the kernels repository's
`docs/migration/a5-hardware.json`. This completes that production board gate;
the named model precision failure remains open and its equality gate remains
unchanged.
