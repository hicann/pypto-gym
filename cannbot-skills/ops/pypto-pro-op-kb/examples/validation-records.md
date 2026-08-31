# Retained pattern validation records

These scoped records preserve the evidence used by pattern pages without depending on an
external repository, branch name or evaluation harness. They are historical compatibility
evidence, not a substitute for re-running correctness on the current SDK and target.

| Pattern | Recorded target | Validated scope | Recorded result | Explicit exclusions |
|---|---|---|---|---|
| UB strip gather | Ascend950PR_9579, CANN 9.2.0 | Dtype compatibility for twelve recorded UB-strip runs; the exact `C`/`L` case matrix was not retained | 12/12 combinations bit-exact; fp32 path measured end to end | The complete hoisting premises, the second-family `C = Irun` path and the general-modulo path are not established by this record; other targets and SDK versions require revalidation |
| TensorList fixed arity | Ascend950PR_9579, CANN 9.2.0 | Historical `Tensor`-parameter/host-view harness for list lengths 1–64 over three dtypes; one 64-slot run used 32 real and 32 zero-work slots | 21 generated kernels compiled; the 64-slot/385-parameter form returned per-slot-correct data, and the zero-work slots were untouched | The deliverable independent-`Ptr` kernel-view ABI is not validated by this record and requires an end-to-end target run |
| Scatter owner model | Ascend950PR_9579 | Inner-tile owner layout with fp32 update mode | Addressing/owner model bit-exact against torch in the recorded fp32 scope | Owner-batch and rank-1 layouts remained conceptual; narrow-dtype accumulation was not validated |
| Prefix-dependent scan | Ascend950PR_9579 | Both dataflows described by the scan pattern | Both built and compiled; recorded outputs were bit-exact | Performance figures are target-specific and are not reusable targets |
| Row-reduce broadcast | Ascend950PR_9579 | Tile-operation row reduction, over-declared reduction tile and short-axis measurements | Bit-exact across the full recorded case set; TR 1, 2, 3, 4, 8 and 16 produced the same fp32 rounding as legal TR=8 | Vector-function form and the very-short-axis limitation require separate validation |
| Paired-lane rotation | Ascend950PR_9579 | Retained interleaved fp32 sample: one operand pair, static innermost width, pre-broadcast coefficients | Sample is validated for its multicore row-tile loop | Split-half pairing, runtime divisor/rank and table lookup regimes are not validated by that sample |

When a pattern page claims more than the scope recorded above, the narrower scope wins.
