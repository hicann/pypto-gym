# MiniMax M2.7 — PyPTO MoE acceleration

End-to-end inference and benchmark for **MiniMax M2.7** on Ascend 910B, with the MoE expert FFN
routed through a self-contained PyPTO fused **grouped GEMM** kernel (910B-adapted; mirrors the
`llada2_moe` contribution layout). The model is defined in-repo (`configuration_minimax_m27.py` +
`modeling_minimax_m27.py`), no `trust_remote_code`.

## Model specs

| Parameter | Value |
|---|---|
| hidden_size (H) | 3072 |
| moe_intermediate_size (I) | 1536 |
| num_experts (E) | 256 |
| num_experts_per_tok (top_k) | 8 |
| Routing | sigmoid + e_score_correction_bias |
| num_attention_heads / num_key_value_heads | 48 / 8 (GQA) |
| head_dim | 128 |
| Activation | SiLU (SwiGLU) |
| Layers / params | 62 / 228.7B |

## What PyPTO replaces

The stock HuggingFace path (`MiniMaxM2Experts.forward`) runs a Python loop over the hit experts,
each a `MiniMaxM2MLP` (`w2(silu(w1(x)) * w3(x))`). At E=256 the dispatch overhead dominates. PyPTO
replaces the whole expert FFN with a single fused grouped GEMM over all experts (stack per-expert
`w1`/`w3` → `[E,2I,H]`, `w2` → `[E,H,I]`, convert to kernel layout `[E*H,2I]`/`[E*I,H]`). The 910B
fix is capping the SwiGLU/cast vector tile to a UB-fitting width (`PYPTO_VEC_TILE=128`); see
[`src/pypto_gym/ops/pypto_tile/minimax_m27/README.md`](../../../src/pypto_gym/ops/pypto_tile/minimax_m27/README.md).

## Results summary (measured, Ascend 910B3 · torch_npu 2.5.1 / pypto 0.2.0 / pto-isa v9.0.0)

### Full-model E2E across 8 dies (pipeline-parallel, 1 rank/die, prefill, best-of-3)

On **real text** (the model's own dynamic routing, ~160/256 experts fire per layer, seq=87):

| MoE impl | forward (best) | prefill | PyPTO speedup |
|---|---|---|---|
| eager (per-expert Python loop) | 8578 ms | 10.1 tok/s | 1× |
| vectorized (torch matmul loop) | 2088 ms | 41.7 tok/s | — |
| **PyPTO (fused grouped GEMM)** | **562 ms** | **154.7 tok/s** | **15.3× vs eager · 3.7× vs vectorized** |

PyPTO also runs the full model via `model.generate` (single-die streaming path) and emits coherent
text. Reproducible to ~1% (re-run 524 vs 527 ms at the fixed-160 load).

### PyPTO vs the strongest baseline (vectorized + NPU-graph) — load crossover

`vec+graph` removes all launch overhead but needs a **static** route (dynamic routing's AICPU
`argsort`/`bincount` can't be captured), so it isn't realizable for real inference. Reproduced as
static routes for a fair placement:

| active experts / layer | eager | vectorized | vec+graph¹ | **PyPTO** | PyPTO vs eager | vs vec+graph |
|---|---|---|---|---|---|---|
| 256 (all experts) | 13754 ms | 3245 ms | 718 ms | **619 ms** | 22.2× | **1.16× (PyPTO wins)** |
| 160 (real-data average, static) | 8909 ms | 2104 ms | 480 ms | 527 ms | 16.9× | 0.91× (static-only) |
| ~160 (real, dynamic) | 8578 ms | 2088 ms | — | **562 ms** | **15.3×** | n/a |

¹ static-route only. PyPTO wins at full load (fused compute); `vec+graph` edges it at ~160 only via
off-graph routing — impossible for real dynamic inference, so **PyPTO is the practical best**.

### Kernel microbenchmark — expert FFN only (E=256, N=2048, H=3072, I=1536, BF16, uniform)

| eager | vectorized | vec+graph | **PyPTO** | speedup vs eager |
|---|---|---|---|---|
| 56976 µs | 53180 µs | 9647 µs | **7536 µs** | **7.56×** |

### Real-data expert load (87-token real prompt, 62 layers)

Mean **~160/256 (62%)** experts fire per layer (early layers 7–22 ~170–194; later ~144–150).
Per-token routing is sparse (8/256), but a prefill batch lights up most experts — so real prefill
sits at the high-load end where PyPTO wins most.

### Streaming MoE cost vs active-expert load (single-die, real FP8 weights, 8 tokens/active-expert)

| active experts | **kernel (PyPTO)** | host FP8→BF16 dequant | stream total |
|---|---|---|---|
| 32 | **1.1 ms** | 604 ms | 605 ms |
| 128 | **3.9 ms** | 2.12 s | 2.13 s |
| 256 | **7.6 ms** | 4.92 s | 4.93 s |

The fused kernel is tiny and linear (~270k tok/s at L=256); the streaming cost is dominated by the
host FP8→BF16 dequant (>99%). A native-FP8 grouped GEMM would remove it.

## FP8 checkpoint — kernel-path status

The released checkpoint is **FP8 block-quantized** (`float8_e4m3fn`, `weight_block_size [128,128]`;
`gate` / `e_score_correction_bias` / `lm_head` kept BF16), and **910B has no native FP8**
(`torch_npu: "Float8_e4m3fn has not been supported"`). On 910B the path is therefore **block-dequant
FP8→BF16 on the host → BF16 grouped GEMM**.

- **Prebuilt BF16** — dequantize every expert once, run the BF16 grouped GEMM (fastest per call;
  best for a single block / when the dequantized model fits).
- **Streaming (single-die full model)** — `patch_minimax_m2_moe(..., streaming=True)` keeps experts
  FP8 on the host and dequantizes only the *routed* experts per forward; HBM peak ~one layer, so the
  full 62-layer model runs on one die. Active-expert count is bucketed (32/128/256) so the kernel
  compiles a bounded shape set.
- **Native FP8 (TODO, production)** — keep weights FP8 via a block-FP8 grouped GEMM so the per-layer
  host dequant disappears.

## Files

| File | Purpose |
|---|---|
| `ask_minimax_m27.py` | Single-prompt inference — full model on one die via streaming + PyPTO |
| `bench_minimax_m27.py` / `bench_minimax_m27.sh` | 8-die pipeline-parallel prefill benchmark (eager / vectorized / PyPTO, `--route`, `--graph`) — produces the *Results* numbers |

The model definition, kernel and the streaming-load helpers live under
`src/pypto_gym/{transformers,ops/pypto_tile}/minimax_m27/`; the kernel correctness test is
`tests/ops/minimax_m27/test_minimax_m27_grouped_gemm.py` (+ `test_cases.json`).

## Usage

```bash
export MODEL_PATH=/path/to/MiniMax-M2.7
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
export PYPTO_VEC_TILE=128            # vector tile must fit the 910B 192 KB UB
export TE_PARALLEL_COMPILER=1        # serial TBE compile (avoids a parallel-compiler SIGSEGV)

# single prompt — full 228.7B model on one die (experts FP8 on host, streamed per layer)
TILE_FWK_DEVICE_ID=0 python ask_minimax_m27.py

# 8-die full-model E2E benchmark — PyPTO (default) vs eager, prefill (reproduces the Results table)
MODEL_PATH="$MODEL_PATH" ./bench_minimax_m27.sh
# or directly (one process per die); default --moe-impl is pypto:
torchrun --nproc_per_node=8 bench_minimax_m27.py --moe-impl pypto
torchrun --nproc_per_node=8 bench_minimax_m27.py --moe-impl eager
# the load crossover (static route, vec+graph capturable):
torchrun --nproc_per_node=8 bench_minimax_m27.py --moe-impl vectorized --graph --route fixed --active 160 --seq 87
```

The *Results* numbers (15.3×, the load crossover) are produced by `bench_minimax_m27.py` above
(8 dies, one process per die over HCCL); see the engineering notes.

## Engineering notes (8-die path)

- **direct-to-flat** expert load: build the PyPTO flat straight from the FP8 checkpoint per layer,
  never holding the per-expert Linears AND the flat at once (that 2× peak OOMs at 8 layers/die);
- **HCCL comms pre-created right after init** (while memory is free) — lazy init after the ~58 GB
  load fails `hcclCommInitRootInfoConfig error code 1`;
- one process per die: single-process device_map can't run the kernel multi-die (torch_npu JIT
  kernels bind to the die that compiled them → `rtBinaryGetFunction 107000`; reproduces with eager);
- `TE_PARALLEL_COMPILER=1` works around a CANN 8.5.2 TBE parallel-compiler SIGSEGV at some shapes.
