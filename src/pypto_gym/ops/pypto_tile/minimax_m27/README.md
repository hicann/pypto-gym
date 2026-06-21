# minimax_m27 — MiniMax M2.7 MoE grouped GEMM kernel

Self-contained grouped GEMM (all experts in one call: `pypto.loop` over experts +
`pypto.loop_unroll` over tokens). Same algorithm as llada2_moe but **tiled for the
Ascend 910B (dav-c220)** — the SwiGLU/cast vector tile width is capped to a
UB-fitting value (`PYPTO_VEC_TILE`, default 128).

> **910B status: WORKING (measured).** Verified on Ascend 910B3 with
> torch_npu 2.5.1 / pypto 0.2.0 / pto-isa v9.0.0. BF16 precision tests pass
> (`tests/ops/minimax_m27/test_minimax_m27_grouped_gemm.py` + `test_cases.json`:
> balanced / uneven / zero-token / mixed-width cases; rtol/atol 8e-3). The fused kernel
> is ~7.6× over the eager per-expert loop at full (E=256) load.
>
> **Why vec_tile=128:** 910B has split CUBE/VECTOR cores and a **192 KB UB**. A FP32
> vector tile `[t,W]` double-buffered needs `t*W*4*2` B; `W=128` → 128 KB (fits),
> `W=256` or full intermediate/hidden width → exceeds UB and the `OoOSchedule` pass
> fails. The original llada2 kernel used `W=intermediate_size/hidden_size`, which is
> why it would not compile on 910B. See `ai/910b-hardware.md`.

> **FP8 note:** this grouped GEMM is **BF16**. The MiniMax M2.7 checkpoint ships
> FP8 block-quant weights (`float8_e4m3fn`, block `[128,128]`). The modeling layer
> dequantizes to BF16 for this kernel as a *reference* path, but full-scale E2E
> needs a **native FP8** grouped GEMM (reuse `experimental/matmul/
> grouped_matmul_swiglu_quant`). See the modeling README for status.


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## Model specs

| Parameter | MiniMax M2.7 | LLaDA2.0-mini |
|---|---|---|
| Total params | 230B | — |
| Active params | 10B | — |
| hidden_size (H) | 3072 | 2048 |
| intermediate_size (I) | 1536 | 512 |
| num_experts (E) | 256 | 256 |
| top_k | 8 | 8 |
| Routing | sigmoid + e_score_correction_bias | sigmoid + expert_bias + group-topk |
| Shared experts | None | Yes |
| Attention | GQA (48h, 8kv) | GQA |

## Weight layout conversion

MiniMax M2.7 (HuggingFace `MiniMaxM2Experts`) stores weights in
`F.linear` convention (computes `x @ W^T`):

```
gate_up_proj  [E, 2*I, H]
down_proj     [E, H,   I]
```

The grouped GEMM kernel expects direct-matmul layout (computes `x @ W`):

```
w13_flat  [E*H, 2*I]
w2_flat   [E*I, H]
```

Conversion (one-time at model load):

```python
from pypto_gym.ops.pypto_tile.minimax_m27 import convert_minimax_weights

w13_flat, w2_flat = convert_minimax_weights(gate_up_proj, down_proj)
```

## Tile config for H=3072

The UB (192KB) cannot hold the default `VEC_FIRST=13` at H=3072:

```
CAST FP32->BF16 at H=3072: 3072 * (4 + 2) = 18432 bytes/row
Single-buffer max rows: 196608 / 18432 = 10
```

Optimal setting (910B):

```bash
export PYPTO_VEC_TILE=128     # vector tile width; must fit the 192 KB UB
export PYPTO_VEC_NBUFFER=1
```

## Kernel tests (Ascend 910B3)

`tests/ops/minimax_m27/test_minimax_m27_grouped_gemm.py` iterates `test_cases.json` and compares
the fused kernel against the eager per-expert golden. The 5 cases cover balanced (4 experts), uneven
load, zero-token experts, a larger batch (8 experts × 16 tokens), and mixed per-expert widths that
exercise the token unroll list. BF16 `rtol/atol = 8e-3`; all cases PRECISION_PASS (measured
`max_diff ≈ 3.8e-6` with FP32 accumulation).

## Benchmark: kernel vs eager

`src/pypto_gym/transformers/minimax_m27/bench_grouped_gemm.py` times the fused grouped GEMM against
the eager per-expert `F.linear` loop (matches `MiniMaxM2Experts.forward`) across
E ∈ {4, 8, 16, 32, 64}, N up to 2048. Speedup grows with the expert count — the eager loop is bound
by its per-expert dispatch. At the full M2.7 load (E=256, N=2048, BF16) the fused kernel is **7.56×**
over eager (see the modeling README's kernel microbenchmark).

## M3 forward compatibility

MiniMax M3 changes the attention mechanism (MSA sparse attention) but
keeps the MoE backbone. The grouped GEMM kernel is expected to work
without modification for M3 when weights become available.

## Function signatures

```python
def convert_minimax_weights(
    gate_up_proj: torch.Tensor,  # [E, 2*I, H] BF16
    down_proj: torch.Tensor,     # [E, H,   I] BF16
) -> tuple[torch.Tensor, torch.Tensor]:
    # Returns: (w13_flat [E*H, 2*I], w2_flat [E*I, H])

def minimax_m27_moe_grouped_gemm(
    sorted_tokens: torch.Tensor,   # [N_total, H] BF16
    w13_flat: torch.Tensor,        # [E*H, 2*I] BF16
    w2_flat: torch.Tensor,         # [E*I, H] BF16
    expert_cumsum: torch.Tensor,   # [E+1] INT32
    result: torch.Tensor,          # [N_total, H] BF16 (out)
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> None
```

## Where to call / how to test

- Kernel adapter: [`pypto_gym/ops/pypto_tile/minimax_m27/minimax_m27_grouped_gemm_impl.py`](minimax_m27_grouped_gemm_impl.py)
- Unit tests: [`tests/ops/minimax_m27/test_minimax_m27_grouped_gemm.py`](../../../../../tests/ops/minimax_m27/test_minimax_m27_grouped_gemm.py)
- Kernel benchmark: [`pypto_gym/transformers/minimax_m27/bench_grouped_gemm.py`](../../../transformers/minimax_m27/bench_grouped_gemm.py)
