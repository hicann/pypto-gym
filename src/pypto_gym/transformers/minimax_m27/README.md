# MiniMax-M2.7 PyPTO Kernel Integration

In-repo `MiniMaxM2ForCausalLM` model definition (no `trust_remote_code`) with the MoE expert FFN
injected with the PyPTO fused grouped-GEMM operator on Ascend 910B. MiniMax-M2.7 is a
Mixture-of-Experts language model (256 experts, top-8, sigmoid + `e_score_correction_bias`
routing). The PyPTO integration replaces the per-expert Python-loop inference with a single fused
grouped GEMM kernel.

## Integration scope

| Operation | PyPTO integrated | Fallback | Notes |
|-----------|:---:|----------|-------|
| Expert FFN (grouped GEMM) | Yes | Per-expert Python loop | All experts in one kernel (`minimax_m27_moe_grouped_gemm`); FP32 accumulation |
| Router gating | No | PyTorch `MiniMaxM2SparseMoeBlock.route_tokens_to_experts` | sigmoid + `e_score_correction_bias` top-k |
| RMSNorm | No | PyTorch `MiniMaxM2RMSNorm` | Standard implementation |
| Attention (GQA) | No | eager / sdpa | 48/8 heads, head_dim 128, qk-norm |
| RoPE | No | PyTorch `MiniMaxM2RotaryEmbedding` | partial rotary (rotary_dim 64) |

`patch_minimax_m2_moe(block, streaming=...)` rebinds the expert forward of
`MiniMaxM2Experts` to the PyPTO path when `USE_PTO_GROUPED_GEMM` is enabled — a single
cumsum-indexed grouped GEMM over all experts, eliminating the per-expert Python loop. Two weight
paths: **prebuilt** (dequant all experts once) and **streaming** (experts kept FP8 on the host,
only the routed experts dequantized per forward, so the full 62-layer model fits one die).

This module also exposes the checkpoint loaders (`build_model`, `load_streaming_state_dict`,
`attach_expert_fp8`, `materialize_meta_buffers`, `patch_moe`) used by the inference / benchmark
entry scripts under `modeling/transformers/minimax_m27/`, and `bench_grouped_gemm.py`, a standalone
kernel benchmark (eager per-expert loop vs PyPTO grouped GEMM) on the Ascend NPU.

## Switch variable

`USE_PTO_GROUPED_GEMM` (in `src/pypto_gym/ops/pypto_tile/minimax_m27/__init__.py`) — the single
switch gating whether the MoE FFN routes through the fused kernel. Tile parameters are env-exposed
(`PYPTO_VEC_TILE`, `PYPTO_CUBE_NBUFFER`, `PYPTO_VEC_NBUFFER`, `PYPTO_MM1_*`, `PYPTO_MM2_*`);
`PYPTO_VEC_TILE=128` keeps the SwiGLU/cast vector tile within the 910B 192 KB UB.
