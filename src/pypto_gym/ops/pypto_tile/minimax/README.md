# MiniMax MoE fused operators (M2.7 / M3)

PyPTO fused operators shared by the MiniMax M2.7 and M3 text backbones on Ascend
910B. BF16 input/output, FP32 accumulation.

## Operators

| Operator | File | Description |
|----------|------|-------------|
| `minimax_moe_grouped_gemm` | `minimax_grouped_gemm_impl.py` | All-experts single grouped GEMM. One kernel body serves both variants via the `activation` arg: `"silu"` (M2.7, H=3072/I=1536) or `"swigluoai"` (M3, H=6144/I=3072), each with its own UB-fitting tile defaults. |
| `minimax_m3_msa_indexer` | `minimax_m3_msa_indexer_impl.py` | M3 MiniMax Sparse Attention lightning indexer: selects the top-k key blocks per query. |
| `minimax_m3_msa_sparse_decode` | `minimax_m3_msa_sparse_attention_impl.py` | M3 MSA block-sparse decode attention over the selected key blocks. |

All tile knobs and the swigluoai alpha/limit are env-overridable (`PYPTO_VEC_TILE`,
`PYPTO_CUBE_NBUFFER`, `PYPTO_VEC_NBUFFER`, `PYPTO_L1_REUSE`, `PYPTO_MM*`,
`PYPTO_SWIGLU_ALPHA`, `PYPTO_SWIGLU_LIMIT`). Opt-in via `USE_PTO_GROUPED_GEMM`.

## Tests

```bash
export TILE_FWK_DEVICE_ID=0
python3 -m pytest tests/ops/minimax_m27/test_minimax_m27_grouped_gemm.py   # M2.7 (silu)
python3 -m pytest tests/ops/minimax_m3/                                    # M3 (swigluoai) + MSA
```
