# Kernel selector

All paths are relative to `cannbot-skills/ops/pypto-pro-op-kb/`. Performance metrics are
deliberately excluded: profile again for the actual platform, shape, and
software version.

Two states, and the difference matters:

- **validated** -- the retained file records a passing correctness result: what
  was run, on what target, and to what accuracy. Those records were taken on an
  Ascend 950-family A5 target. On another architecture or an unmatched SDK
  version, treat the row as a candidate and re-run correctness.
- **study** -- the file carries an embedded test but no retained record of it
  having passed. Read it for the shape of the implementation; it is not evidence
  that the implementation is correct, and it does not confer validation on
  anything modelled after it.

A row may claim `validated` only when its file carries a scoped validation
record; `check_kb_integrity.py` enforces that, because the previous blanket
claim covered twelve files that had no record at all.

| topology / technique | dtype | implementation | state | retained evidence |
|---|---|---|---|---|
| single-tile matmul | fp32 | [examples/samples/matmul_float_mmad/matmul_float_mmad_impl.py](samples/matmul_float_mmad/matmul_float_mmad_impl.py) | validated | file validation record + [golden](samples/matmul_float_mmad/matmul_float_mmad_golden.py) |
| transposed-left matmul | fp32 | [examples/samples/matmul_kmkn_fp32_out/matmul_tn_impl.py](samples/matmul_kmkn_fp32_out/matmul_tn_impl.py) | validated | embedded correctness tests |
| quantized matmul | fp32→int8 | [examples/samples/matmul_quant_int8/matmul_quant_int8_impl.py](samples/matmul_quant_int8/matmul_quant_int8_impl.py) | study | embedded correctness test |
| matmul operand reuse | bf16 | [examples/samples/bf16_matmul_operand_reuse/bf16_matmul_operand_reuse_impl.py](samples/bf16_matmul_operand_reuse/bf16_matmul_operand_reuse_impl.py) | study | embedded correctness test |
| cube→vector handoff | fp32 | [examples/samples/fused_matmul_add/fused_matmul_add_impl.py](samples/fused_matmul_add/fused_matmul_add_impl.py) | validated | embedded correctness test |
| matmul + bias | fp32 | [examples/samples/matmul_bias/matmul_bias_impl.py](samples/matmul_bias/matmul_bias_impl.py) | study | embedded correctness test |
| matmul + bias + activation | fp32 | [examples/samples/matmul_bias_relu/matmul_bias_relu_impl.py](samples/matmul_bias_relu/matmul_bias_relu_impl.py) | study | embedded correctness test |
| matmul + activation | fp32 | [examples/samples/matmul_relu/matmul_relu_impl.py](samples/matmul_relu/matmul_relu_impl.py) | study | embedded correctness test |
| matmul + row normalization | fp32 | [examples/samples/matmul_rowwise_norm/matmul_rowwise_norm_impl.py](samples/matmul_rowwise_norm/matmul_rowwise_norm_impl.py) | validated | embedded correctness test |
| matmul + row L2 normalization | fp32 | [examples/samples/matmul_rowwise_l2_norm/matmul_rowwise_l2_norm_impl.py](samples/matmul_rowwise_l2_norm/matmul_rowwise_l2_norm_impl.py) | study | embedded correctness test |
| matmul + row softmax | fp32 | [examples/samples/matmul_softmax/matmul_softmax_impl.py](samples/matmul_softmax/matmul_softmax_impl.py) | validated | embedded correctness test |
| cube→vector with atomic output | fp32 | [examples/samples/cube_vec_atomic_add_two_outputs/cube_vec_atomic_add_two_outputs_impl.py](samples/cube_vec_atomic_add_two_outputs/cube_vec_atomic_add_two_outputs_impl.py) | validated | embedded correctness test |
| vector→cube handoff | fp32 | [examples/samples/vec_cube_abs_sqrt_matmul/vec_cube_abs_sqrt_matmul_impl.py](samples/vec_cube_abs_sqrt_matmul/vec_cube_abs_sqrt_matmul_impl.py) | validated | embedded correctness test |
| row softmax tile ops | fp32 | [examples/samples/softmax/softmax_impl.py](samples/softmax/softmax_impl.py) | validated | file validation record + [golden](samples/softmax/softmax_golden.py) |
| row normalization tile ops | fp32 | [examples/samples/norm_softmax_rms_l2/norm_softmax_rms_l2_impl.py](samples/norm_softmax_rms_l2/norm_softmax_rms_l2_impl.py) | validated | embedded correctness tests |
| layer normalization tile ops | fp32 | [examples/samples/vector_kernels/layernorm_impl.py](samples/vector_kernels/layernorm_impl.py) | validated | embedded correctness test |
| activation + normalization tile ops | fp32 | [examples/samples/vector_kernels/act_layernorm_elementwise_impl.py](samples/vector_kernels/act_layernorm_elementwise_impl.py) | study | embedded correctness tests |
| row sum tile ops | fp32 | [examples/samples/vector_kernels/reduce_sum_impl.py](samples/vector_kernels/reduce_sum_impl.py) | validated | embedded correctness test |
| interleaved rotary embedding | fp32 | [examples/samples/vector_kernels/rope_interleave_impl.py](samples/vector_kernels/rope_interleave_impl.py) | validated | embedded correctness test |
| cumulative sum by contraction | fp32 | [examples/samples/vector_kernels/cumsum_matmul_impl.py](samples/vector_kernels/cumsum_matmul_impl.py) | validated | embedded correctness test |
| vector-function elementwise | fp32 | [examples/samples/vf_vs_tileop/vf_elementwise_impl.py](samples/vf_vs_tileop/vf_elementwise_impl.py) | study | embedded correctness test |
| vector-function L2 norm + activation | fp32 | [examples/samples/vf_vs_tileop/vf_l2norm_silu_impl.py](samples/vf_vs_tileop/vf_l2norm_silu_impl.py) | study | embedded correctness test |
| vector-function layer norm + rotary embedding | fp32 | [examples/samples/vf_vs_tileop/vf_layernorm_rope_impl.py](samples/vf_vs_tileop/vf_layernorm_rope_impl.py) | study | embedded correctness test |
| vector-function normalization and reductions | fp32 | [examples/samples/vf_vs_tileop/vf_rms_silu_gelu_reduce_impl.py](samples/vf_vs_tileop/vf_rms_silu_gelu_reduce_impl.py) | study | embedded correctness test |
| vector-function row softmax | fp32 | [examples/samples/vf_vs_tileop/vf_softmax_impl.py](samples/vf_vs_tileop/vf_softmax_impl.py) | study | embedded correctness test |
