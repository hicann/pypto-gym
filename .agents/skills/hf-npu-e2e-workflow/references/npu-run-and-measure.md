# Running an HF model on Ascend NPU + E2E measurement

Net-new harness knowledge for taking a converted/loaded HF model and (a) running it
end-to-end on NPU, (b) measuring throughput across the eager / NPU-friendly(graph) /
PyPTO+graph arms. The full *porting & op-fusion methodology* (golden, op dev, sys.modules
injection, 归档) lives in the `pypto-fused-op-integration` skill — this file does NOT
duplicate it; it owns only the **run-on-NPU + graph-capture + measurement** layer.

---

## 1. Pick the regime — what "tok/s" actually means

Three different numbers all get called "tok/s". They are NOT comparable across regimes —
fix ONE per model and label it.

| regime | tok/s formula | what it measures | when to use |
|---|---|---|---|
| **prefill** | `seq_len / single_forward_time` | input ingestion (one parallel forward, no KV cache) | huge / multi-die / FP8-streaming models where a real decode loop is impractical |
| **decode (AR)** | `N / decode_loop_time` (seq=1 + KV cache, N steps) | real autoregressive output speed | single-die autoregressive LMs |
| **diffusion-generate** | `gen_positions / denoising_time` (block of W, S denoising forwards) | diffusion-LM output throughput | block-diffusion LMs (no AR decode exists) |

Rule of thumb: prefill ≫ decode in tok/s (one big parallel forward vs sequential
memory-bound steps) — never put them in the same column without labelling the regime.

## 2. The 3 arms

- **eager** — stock HF forward / `model.generate`. Baseline. (Use the model's natural attention; SDPA is a fair, even generous, baseline since it's the fastest non-graph attention.)
- **NPU-friendly (graph)** — static-shape `torch.npu.NPUGraph` capture, replayed. For MoE, the repo's vectorized expert loop captured under a static route (`vec+graph`).
- **PyPTO+graph** — the PyPTO fused kernel (grouped_gemm / softmax / GQA) under the *same* capture.

Report **within-machine ratios only** — absolute tok/s is node-dependent (committed PR
numbers from another node are not reproducible, only the ratios are).

## 3. NPUGraph capture gotchas — error → cause → fix (THE table)

NPUGraph capture aborts on any op that does a host sync or uses a side stream. Symptoms
all look like a cryptic ACL error at `capture_end`/replay. Mitigations, in the order you
usually need them:

| symptom | cause | fix |
|---|---|---|
| `107025` at `capture_end` | default **SDPA**'s fused NPU kernel runs on a **side stream** | load with `attn_implementation="eager"` |
| `107025` / `107030` (H2D during capture) | `_prepare_4d_causal_attention_mask_for_sdpa` does a host `torch.all(mask==1)` sync | feed a **prebuilt additive mask** and patch the prepare fn to return it verbatim |
| `107025` | rotary embed uses `torch.autocast` + `@dynamic_rope_update` (host seq-len checks) | positions are fixed → **precompute cos/sin once**, inject as constants (replace `rotary.forward`) |
| `107027` (copy-stream sync) / `[ArgSort] ... AiCpu` | an op runs on a copy/side stream — most often MoE `argsort(int64)` on **AICPU**, but any side-stream op (sliding-window mask build, qk-norm path) qualifies | MoE → **static routing** (fixed assignment, precomputed `cumsum`; under capture only `index_select` + device GEMM run); else **bisect** to find the op |
| `107030` (H2D during capture) | `DynamicCache` host ops / cache-length advance | **capture-safe KV cache**: fixed-slot `index_copy_`, no host ops; advance the write-slot tensor *in place* between replays |
| `aicore 507015` (replay/run) | PyPTO `grouped_gemm` fed a **growing/dynamic** shape | **fix the shape** before capture (single fixed block, or bucket-pad N to a few discrete sizes) |

> These are the empirical findings from gemma4 (decode capture), llada2 (diffusion block
> capture), minimax_m27 (static-route grouped GEMM). Even `pypto-fused-op-integration`'s
> aclgraph step does not list them — this table is the net-new bit.

## 4. Capture-safe forward recipe (generic)

1. **Load** `attn_implementation="eager"`.
2. **Bypass `model.forward`'s mask/rope machinery** — either patch
   `_prepare_4d_causal_attention_mask_for_sdpa` → identity and `rotary.forward` → cached
   `(cos, sin)`, or call the decoder layers directly with a prebuilt 0-mask + precomputed
   `position_embeddings`.
3. **MoE** → install a static route (every token → the first `top_k` experts, fixed
   `cumsum`); compute is unchanged (same N×K token-rows through the GEMMs), only the
   assignment is made deterministic.
4. **Decode regime** → capture-safe KV cache (fixed write slot); advance
   `write_pos / cache_position / position_ids / input_id` *in place* between replays.
5. **Capture**: warm up on a side stream, `with torch.npu.graph(g): out = fwd()`, then
   `g.replay()` in the measurement loop.
6. **Localize a capture failure by bisection** — capture `embed+norm` only (should pass),
   then add attention, then add MoE; the stage that flips OK→FAIL names the culprit op.
   (This is how the SDPA side-stream cause was found.)

## 5. Measurement procedure

- Warmup iters absorb JIT (first PyPTO compile, first capture). Then best-of-N.
- prefill: `tok/s = seq_len / best_forward_s`.
- decode: run the N-step loop, `tok/s = N / best_loop_s`.
- Always log peak HBM (`torch.npu.max_memory_allocated`) and `reserved` — a climbing
  `reserved` across iters is the dynamic-shape-workspace leak (→ fix the shape / bucket).
- Set `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` to defrag when capture warmup
  spikes reserved memory.

## 6. Regime decision (quick)

```
multi-die / 100B+ / FP8 weight-streaming  → prefill   (decode loop impractical)
single-die autoregressive LM              → decode    (real output speed)
block-diffusion LM (LLaDA-style)          → diffusion-generate
dense model (no experts)                  → PyPTO has no MoE GEMM to fuse:
                                            pypto+graph ≈ graph (graph still gives
                                            launch-overhead removal; the grouped-GEMM
                                            win only appears on sparse-MoE models)
```
