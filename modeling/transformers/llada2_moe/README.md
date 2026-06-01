# LLaDA2.0-mini MoE — NPU Migration

| Field | Value |
|-------|-------|
| HuggingFace | inclusionAI/LLaDA2.0-mini (trust_remote_code=True) |
| Precision | BF16 weights, FP32 accumulation |
| PyPTO kernels | gate_select, expert_ffn, grouped_gemm |
| Patched module | `LLaDA2MoeSparseMoeBlock` |
| Generation | Block-wise masked diffusion (gen_length=100, steps=32, block_length=32) |

## E2E Results (Ascend 910B, CANN 9.0.0, NPU 14)

| Metric | Baseline (eager) | PyPTO (grouped_gemm) | Speedup |
|--------|-----------------|---------------------|---------|
| Time/iter | 56.1 ± 3.3 s | 6.08 ± 0.24 s | **9.2x** |
| Throughput | 1.6 ± 0.1 tok/s | 14.7 ± 0.6 tok/s | **9.2x** |
| Peak HBM | 31207 MB | 32672 MB | +4.7% |

Baseline uses a **per-expert Python loop** (256 experts × 20 layers × 32 steps = 163,840 kernel launches/iter).
PyPTO `grouped_gemm` processes all experts in a single kernel call using `pypto.loop` with dynamic trip counts,
reducing launches to 20/iter.

## PyPTO Fused Operators

| Kernel | Replaces | Notes |
|--------|----------|-------|
| `llada2_gate_select` | Gate sigmoid + group top-k + renorm | Fused into 1 kernel |
| `llada2_expert_ffn` | Per-expert SwiGLU FFN (matmul→silu→matmul) | Per-expert baseline |
| `llada2_moe_grouped_gemm` | Python per-expert dispatch loop → single kernel | **9.2x E2E speedup** |

## Quick Start

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
export TILE_FWK_DEVICE_ID=14
export PYTHONPATH=$PWD/src

# Single inference — baseline
python modeling/transformers/llada2_moe/ask_LLaDA2-mini.py \
    --model-path /path/to/LLaDA2.0-mini --device 14

# Single inference — PyPTO
python modeling/transformers/llada2_moe/ask_LLaDA2-mini.py \
    --model-path /path/to/LLaDA2.0-mini --device 14 --use_pypto

# Full benchmark (warmup + 10 measurement iters, baseline vs PyPTO)
MODEL_PATH=/path/to/LLaDA2.0-mini bash modeling/transformers/llada2_moe/bench_LLaDA2-mini.sh
```

## Benchmark Methodology

- **Warmup**: 3 iterations (absorbs JIT compilation overhead)
- **Measurement**: 10 iterations, reporting mean ± std, min, max
- **Generation**: 100 tokens via block-wise masked diffusion (steps=32, block_length=32, temperature=0.0)
- **Input**: "Explain the concept of mixture of experts in neural networks." (11 tokens)
- **Model**: LLaDA2.0-mini, BF16, 256 experts, 20 layers, H=2048, I=512
