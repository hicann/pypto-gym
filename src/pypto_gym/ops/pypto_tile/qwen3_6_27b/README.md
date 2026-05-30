# Qwen3.6-27B fused operators

PyPTO fused-operator implementations targeted at Qwen3.6-27B (`qwen3_5`
architecture in upstream `transformers`). Each operator lives in its own
subdirectory with an `_impl.py` exporting a `*_wrapper` function and a
`README.md` describing the contract.

## Operators

| Operator | Subdirectory | Status |
|----------|--------------|--------|
| `gated_delta_rule` | [`gated_delta_rule/`](gated_delta_rule/) | Available |

## Common shape constraints

Qwen3.6-27B's `qwen3_5` architecture has:

| Symbol | Meaning | Value |
|--------|---------|-------|
| `D`   | Per-head dimension              | 128 |
| `Nv`  | Value head count                | 48  |
| `Nqk` | Query/Key head count            | 16  |

Wrappers enforce these constraints at call time and raise
`NotImplementedError` outside the supported envelope.

## Integration

The kernels are wired into the model through the
[modeling_qwen3_5.py](../../../transformers/qwen3_6_27b/modeling_qwen3_5.py)
shim — see that file for the monkey-patch hook pattern.
