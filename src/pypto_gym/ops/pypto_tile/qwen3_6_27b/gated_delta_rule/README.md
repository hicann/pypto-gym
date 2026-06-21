# gated_delta_rule

PyPTO fused kernel for the chunk gated delta rule attention used during
Qwen3.6-27B prefill. Replaces the upstream
`fla.ops.gated_delta_rule.chunk_gated_delta_rule` (or the torch fallback)
called from `Qwen3_5GatedDeltaNet.forward`.


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## Parameter glossary

| Symbol | Meaning | Value / constraint |
|--------|---------|-------------------|
| `B`  | Batch size                       | Fixed to 1 |
| `S`  | Sequence length (prefill)        | Any S ≥ 1 (padded to a multiple of L internally) |
| `L`  | Chunk length                     | Fixed to 128 |
| `Nv` | Value head count                 | Fixed to 48 |
| `D`  | Per-head dimension               | Fixed to 128 |

## Function

```python
def gated_delta_rule_wrapper(
    query: torch.Tensor,
    key:   torch.Tensor,
    value: torch.Tensor,
    *,
    g:     torch.Tensor,
    beta:  torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
```

Signature mirrors the upstream `chunk_gated_delta_rule` so the modeling-layer
hook can swap implementations transparently.

### Inputs

| Argument | Shape | Dtype | Notes |
|----------|-------|-------|-------|
| `query`  | `[B, S, Nv, D]` | bfloat16 | not L2-normalized (done in kernel) |
| `key`    | `[B, S, Nv, D]` | bfloat16 | not L2-normalized (done in kernel) |
| `value`  | `[B, S, Nv, D]` | bfloat16 | |
| `g`      | `[B, S, Nv]`    | float32  | per-token gate (pre-cumsum) |
| `beta`   | `[B, S, Nv]`    | bfloat16 | |
| `initial_state` | `[B, Nv, D, D]` or `None` | float32 | must be `None` (prefill only) |
| `output_final_state` | — | bool | if False, second return is `None` |
| `use_qk_l2norm_in_kernel` | — | bool | must be `True` |

### Outputs

| Return | Shape | Dtype |
|--------|-------|-------|
| `core_attn_out` | `[B, S, Nv, D]` | bfloat16 |
| `last_state`    | `[B, Nv, D, D]` | float32 (or `None`) |

### Algorithm

Per chunk of L=128 rows:

1. L2-normalize q, k.
2. Decay mask `D = exp(g_cum - g_cum^T) * lower_tri`.
3. `A0 = -(k_β @ k_n^T * D) * strict_lower_tri`;
   `A = (I - A0)^-1` via 8-term truncated power series.
4. `v_out = A @ (v * β)`, `kcd = A @ (k_β * exp(g_cum))`.
5. Recurrent state carry:
   `v_new = v_out - kcd @ state`,
   `out = (q_scaled * exp(g_cum)) @ state + (q_scaled @ k_n^T * D) @ v_new`,
   `state' = state * exp(g_last) + k_decay^T @ v_new`.

### Out-of-scope behavior

If any constraint is violated, the wrapper raises `NotImplementedError`. The
modeling-layer hook (see `src/pypto_gym/transformers/qwen3_6_27b/modeling_qwen3_5.py`)
falls back to the upstream chunk function in that case.

## Testing

Tests under [`tests/ops/qwen3_6_27b/`](../../../../../../tests/ops/qwen3_6_27b/).
