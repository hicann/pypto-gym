# Retained pattern validation records

These scoped records preserve the evidence used by pattern pages without depending on an
external repository, branch name or evaluation harness. They are historical compatibility
evidence, not a substitute for re-running correctness on the current SDK and target.

| Pattern | Recorded target | Validated scope | Recorded result | Explicit exclusions |
|---|---|---|---|---|
| UB strip gather | Ascend950PR_9579, CANN 9.2.0 | Twelve dtype combinations for the UB-strip gather dataflow | 12/12 combinations bit-exact; fp32 path measured end to end | Other targets and SDK versions require revalidation |
| TensorList fixed arity | Ascend950PR_9579, CANN 9.2.0 | Generated fixed-arity kernels for list lengths 1–64 over three dtypes | 21 generated kernels compiled; the 64-slot/385-parameter form launched and returned per-slot-correct data | Runtime-variable arity still requires padding to the compiled maximum |
| Scatter owner model | Ascend950PR_9579 | Inner-tile owner layout with fp32 update mode | Addressing/owner model bit-exact against torch; whole operator passed 14/20 recorded cases | Owner-batch and rank-1 layouts remained conceptual; narrow-dtype accumulation was not validated |
| Prefix-dependent scan | Ascend950PR_9579 | Both dataflows described by the scan pattern | Both built and compiled; recorded outputs were bit-exact | Performance figures are target-specific and are not reusable targets |
| Row-reduce broadcast | Ascend950PR_9579 | Tile-operation row reduction, over-declared reduction tile and short-axis measurements | Recorded 20/20 cases; TR 1, 2, 3, 4, 8 and 16 produced the same fp32 rounding as legal TR=8 | Vector-function form and the very-short-axis limitation require separate validation |
| Paired-lane rotation | Ascend950PR_9579 | Retained interleaved fp32 sample: one operand pair, static innermost width, pre-broadcast coefficients | Sample is validated for its multicore row-tile loop | Split-half pairing, runtime divisor/rank and table lookup regimes are not validated by that sample |

When a pattern page claims more than the scope recorded above, the narrower scope wins.
