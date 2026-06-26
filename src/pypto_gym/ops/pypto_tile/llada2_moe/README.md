# llada2_moe — LLaDA2.0 MoE fusion kernels

Three PyPTO NPU kernels that together replace the forward path of
`LLaDA2MoeSparseMoeBlock`. They follow the layout of
glm_v4_5/`glm_moe_fusion_impl.py`, with quantization removed (BF16 in,
BF16 out, FP32 accumulation).


## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## Algorithm mapping

`LLaDA2MoeSparseMoeBlock.forward` in
`model_hf/LLaDA2.0-mini/modeling_llada2_moe.py` (line 306) runs:

1. `gate(hidden_states)` — sigmoid + expert_bias + group-limited top-k + renorm
2. `moe_infer(x, topk_ids, topk_weight)` — per-expert Python loop over SwiGLU FFNs
3. `shared_experts(identity)` — one dense SwiGLU FFN

This directory fuses (1) and the per-expert part of (2,3) into three kernels:

| Kernel | Responsibility | Corresponding eager code |
|---|---|---|
| `llada2_gate_select_kernel` | gate matmul (FP32) → sigmoid → bias → group-limited top-k → mask → top-k → renorm → scaling | `LLaDA2MoeGate.forward` (line 253) |
| `llada2_expert_ffn_kernel` | BF16 `x @ W13` + SwiGLU (silu) + `@ W2`; one expert at a time | `LLaDA2MoeMLP.forward` (line 201) |
| `llada2_moe_grouped_gemm` | All experts in a single kernel call via `pypto.loop` + `pypto.loop_unroll` | Replaces entire per-expert dispatch loop |

## Math (gate)

```
logits  = x @ gate_weight^T                       # FP32, [N, E]
scores  = sigmoid(logits)                          # FP32
sa      = scores + expert_bias                     # augmented score for group-topk
gs[g]   = sum( top-2(sa[:, g, :]) )                # per-group score
top_g   = top-k_group(gs)                          # group mask
sa[mask=0] = -inf
picks   = top-k(sa)
tw      = gather(scores, picks)                   # gather from the un-biased scores
tw     /= tw.sum(-1, keepdim=True) + 1e-20
tw     *= routed_scaling_factor
```

`-inf` masking matches `modeling_llada2_moe.py:248` exactly (GLM uses `0.0`,
which only agrees when sigmoid+bias > 0 everywhere — false in general).

## Math (expert FFN)

```
gate_up = x @ W13                                  # BF16, FP32 accum, [n_e, 2I]
sw      = silu(gate_up[:, :I]) * gate_up[:, I:]    # BF16
out     = sw @ W2                                  # BF16, FP32 accum, [n_e, H]
```

`W13 = [W_gate; W_up]` is stacked once at model-load time — see
`pypto_gym/transformers/llada2_moe/modeling_llada2_moe.py:_ensure_pypto_weights`.

## Math (grouped GEMM)

```
# sorted_tokens[N_total, H], w13_flat[E*H, 2I], w2_flat[E*I, H], cumsum[E+1]
for e in pypto.loop(0, E, 1):           # dynamic trip count over experts
    start = cumsum[e]
    count = cumsum[e+1] - start
    for t in pypto.loop_unroll(count):   # dynamic trip count over tokens
        x_t = sorted_tokens[start + t]
        gate_up = x_t @ w13[e]           # [2I], FP32 accum
        sw = silu(gate_up[:I]) * gate_up[I:]
        out[start + t] = sw @ w2[e]      # [H], FP32 accum
```

This eliminates the Python per-expert dispatch loop (256 experts × 20 layers × 32 steps
= 163,840 kernel launches/iter → 20 launches/iter), achieving **9.2x E2E speedup**.

## Function signatures

```python
def llada2_gate_select(
    hidden_states: torch.Tensor,   # [N, H] bf16
    gate_weight: torch.Tensor,     # [E, H] fp32
    expert_bias: torch.Tensor,     # [E]    fp32
    topk_weights: torch.Tensor,    # [N, K] fp32  (out)
    topk_ids: torch.Tensor,        # [N, K] int32 (out)
    *,
    top_k: int,
    topk_group: int,
    num_expert_group: int,
    routed_scaling_factor: float,
) -> None
```

```python
def llada2_expert_ffn(
    hidden_states: torch.Tensor,   # [N, H] bf16
    w13: torch.Tensor,             # [H, 2I] bf16
    w2:  torch.Tensor,             # [I, H]  bf16
    ffn_res: torch.Tensor,         # [N, H]  bf16 (out)
) -> None
```

```python
def llada2_moe_grouped_gemm(
    sorted_tokens: torch.Tensor,   # [N_total, H] bf16
    w13_flat: torch.Tensor,        # [E*H, 2*I] bf16
    w2_flat: torch.Tensor,         # [E*I, H] bf16
    expert_cumsum: torch.Tensor,   # [E+1] int32
    result: torch.Tensor,          # [N_total, H] bf16 (out)
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> None
```

## Kernel test results (Ascend 910B, NPU 14)

### grouped_gemm — 9/9 PASS

| Experts | Tokens | Token Distribution | Result |
|---------|--------|-------------------|--------|
| E=4 | N=8 | [2, 2, 1, 3] | PASS |
| E=4 | N=32 | [13, 8, 9, 2] | PASS |
| E=4 | N=64 | [6, 8, 28, 22] | PASS |
| E=4 | N=128 | [57, 15, 49, 7] | PASS |
| E=8 | N=8 | [0, 2, 1, 1, 0, 2, 1, 1] | PASS |
| E=8 | N=32 | [6, 2, 7, 5, 4, 0, 7, 1] | PASS |
| E=8 | N=64 | [13, 2, 11, 4, 7, 14, 9, 4] | PASS |
| E=8 | N=128 | [6, 2, 34, 19, 24, 1, 37, 5] | PASS |
| E=8 (zero-token) | N=16 | [8, 0, 0, 8, 0, 0, 0, 0] | PASS |

BF16 rtol/atol=8e-3. Zero-token experts handled correctly.

### gate_select — bs≤16 PASS

| Batch Size | Result |
|------------|--------|
| bs=1 | PASS (IDs + weights) |
| bs=8 | PASS |
| bs=16 | PASS |

### expert_ffn — n≤32 PASS

| n (tokens) | Result |
|------------|--------|
| n=1~32 | PASS (BF16 rtol/atol=8e-3) |

## Where to call / how to test

- Integration wrapper: [`pypto_gym/transformers/llada2_moe/modeling_llada2_moe.py`](../../../transformers/llada2_moe/modeling_llada2_moe.py)
- Unit tests:
  [`tests/ops/llada2_moe/test_llada2_gate_select.py`](../../../../../tests/ops/llada2_moe/test_llada2_gate_select.py),
  [`tests/ops/llada2_moe/test_llada2_expert_ffn.py`](../../../../../tests/ops/llada2_moe/test_llada2_expert_ffn.py),
  [`tests/ops/llada2_moe/test_llada2_moe_grouped_gemm.py`](../../../../../tests/ops/llada2_moe/test_llada2_moe_grouped_gemm.py)
- E2E benchmark: [`modeling/transformers/llada2_moe/`](../../../../../modeling/transformers/llada2_moe/)
- Kernel benchmark: [`pypto_gym/transformers/llada2_moe/bench_grouped_gemm.py`](../../../transformers/llada2_moe/bench_grouped_gemm.py)
