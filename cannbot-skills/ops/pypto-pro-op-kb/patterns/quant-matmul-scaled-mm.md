# Matmul with explicit output quantization

## Applies when

A matmul writes an integer result using an explicitly defined scale, rounding
rule, and saturation range. The golden path must use the same order of
operations:

```text
output = clamp(round((left @ right) * scale), qmin, qmax)
```

This page does not imply support for every quantized operand format or
block-scaled matmul variant.

## Required contract

Record and validate:

- operand and accumulator dtypes;
- whether scaling happens before or after accumulation;
- the rounding rule;
- the saturation range;
- scale granularity and layout;
- the exact conversion or move API supported by the target SDK.

Changing any of these changes the operator contract. Do not substitute a
similarly named quantization API without checking its documented equation.

## Failure signatures

- mismatches cluster at half-integer boundaries: rounding rules differ;
- mismatches cluster at the integer limits: saturation or scale order differs;
- error grows with the reduction dimension: accumulator dtype or quantization
  placement differs from the golden path;
- compilation fails at the accumulator-to-vector boundary: the selected
  conversion mode is unsupported for the target.

## Validation status

The retained [int8-output matmul](../examples/samples/matmul_quant_int8/matmul_quant_int8_impl.py)
is a runnable **study** whose Ascend 950-gated test fixes a 64-by-64 shape,
FP32 inputs and accumulator, scalar scale `16.0`, and INT8 output. The test
expects exact integer equality (`rtol=0`, `atol=0`), but no passing result is
retained. Other shapes, scales, operand formats, scale layouts and block-scaled
forms need their own target-matching reference and result.
