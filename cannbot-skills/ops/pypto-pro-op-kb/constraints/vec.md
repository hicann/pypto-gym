# Vector authoring constraints

## Applies to

PyPTO-Pro vector sections implemented with tile operations or
`@pl.vector_function` / `vf.*`.

## Conditional authoring rule

Select the level that is correct and supported by the installed API:

- Use tile operations when they express the complete dataflow and their runtime
  valid-shape semantics cover the tail.
- Use `vf.*` when explicit lane masks, register-level control, or an instruction
  composition unavailable at tile level is required.
- When both are legal, measure both on the target before choosing for
  performance.

No operator family, shape, or prior benchmark makes either level the universal
default.

## Shared correctness requirements

1. A row reduction writes a row scalar, conventionally a `[rows, 1]` result,
   which is broadcast back over the row by the matching expand operation.
2. Padding lanes must be excluded from reductions and stores.
3. Accumulation dtype follows the numerical contract; use FP32 unless a
   documented, validated alternative is allowed.
4. Input, output, reduction, and workspace tiles must fit the detected
   platform's vector-memory budget.
5. Compare/select mask dtype, layout, offset units, and supported operations are
   API-version-specific. Confirm them in the installed documentation.

## Examples

Tile-operation row softmax:

```python
pl.row_max(row_value, input_tile, workspace)
pl.row_expand_sub(output_tile, input_tile, row_value)
pl.exp(output_tile, output_tile)
pl.row_sum(row_value, output_tile, workspace)
pl.row_expand_div(output_tile, output_tile, row_value)
```

Vector-function load and mask placement:

```python
register = vf.load_align(input_tile, offset)
accumulator = vf.add(accumulator, register, predicate)
result = vf.reduce_sum(accumulator, predicate)
vf.store_align(output_tile + offset, result, predicate)
```

`vf.load_align(input_tile, offset, predicate)` is not a supported mask
placement in the retained guidance.

## Evidence

- Tile-operation implementation and golden:
  [softmax_impl.py](../examples/samples/softmax/softmax_impl.py) and
  [softmax_golden.py](../examples/samples/softmax/softmax_golden.py)
- Vector-function implementation:
  [vf_softmax_impl.py](../examples/samples/vf_vs_tileop/vf_softmax_impl.py)
- Conditional selection and source order:
  [vf-reduction-perf.md](../../pypto-pro-op-develop/references/vf-reduction-perf.md)

The installed `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/` documentation remains the
primary source for signatures and target-version behavior.
