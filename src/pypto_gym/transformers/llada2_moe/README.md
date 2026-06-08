# LLaDA2-MoE PyPTO Kernel Integration

HuggingFace `LLaDA2MoeForCausalLM` model definition modified to inject PyPTO fused operators on Ascend NPU hardware. LLaDA2-MoE (Ant Group) is a Mixture-of-Experts language model with per-expert FFN layers routed via a learned gating mechanism. The PyPTO integration replaces the per-expert Python-loop inference with a single fused grouped GEMM kernel.

## Integration Scope

| Operation | PyPTO Integrated | Fallback | Notes |
|-----------|:---:|----------|-------|
| Expert FFN (Grouped GEMM) | Yes | Per-expert Python loop | Fused grouped GEMM kernel for MoE inference with `llada2_pto_kernels` |
| Router Gating | No | PyTorch `LLaDA2MoeGate` | Group-limited top-k expert selection |
| RMSNorm | No | PyTorch `LLaDA2MoeRMSNorm` | Standard implementation, not fused |
| Attention (GQA) | No | eager / flash_attn / sdpa | Standard HuggingFace attention with partial rotary |
| Dense MLP | No | PyTorch `LLaDA2MoeMLP` | Standard SwiGLU MLP |
| RoPE | No | PyTorch `LLaDA2MoeRotaryEmbedding` | With `partial_rotary_factor` (0.5) |

The PyPTO injection replaces the `moe_infer()` method on `LLaDA2MoeSparseMoeBlock`. When `USE_PTO_EXPERT_FFN` is enabled, the forward path diverges to `_moe_infer_pypto()` which uses `grouped_gemm` — a single kernel call that processes all experts at once via a cumsum-indexed grouped GEMM, eliminating the per-expert Python loop overhead.

The directory also contains `bench_grouped_gemm.py`, a standalone benchmark script that compares eager (per-expert loop + pre-allocated buffers) vs PyPTO grouped GEMM performance on the Ascend NPU.

## Switch Variables

The kernel module is expected to expose the following in `sys.modules["llada2_pto_kernels"]`:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_PTO_EXPERT_FFN` | `bool` | `False` | Enable PyPTO fused grouped GEMM for MoE expert FFN computation |

## Kernel API Contract

When `USE_PTO_EXPERT_FFN` is `True`, the kernel must provide:

```python
# Called inside LLaDA2MoeSparseMoeBlock._moe_infer_pypto:
def grouped_gemm(
    sorted_x: torch.Tensor,    # (total_tokens, hidden_size) — tokens sorted by expert
    w1_stack: torch.Tensor,    # (num_experts, intermediate_size, hidden_size) — gate+up stacked
    w2_stack: torch.Tensor,    # (num_experts, hidden_size, intermediate_size) — down stacked
    cumsum: torch.Tensor,      # (num_experts + 1,) int32 — cumulative token counts per expert
    K: int,                    # top_k (routing multiplicity)
    I: int,                    # intermediate_size
    H: int,                    # hidden_size
    N: int,                    # total_tokens
    E: int,                    # num_experts
) -> torch.Tensor:             # (N // K, hidden_size) — final MoE output
    ...
```

The `_moe_infer_pypto()` method handles the routing preparation:
1. Sorts tokens by expert assignment (topk_ids argsort)
2. Computes bincount → cumsum for expert token boundaries
3. Stack expert weight matrices into 3D tensors (`_ensure_pypto_weights()`)
4. Calls `grouped_gemm()` with sorted tokens and weight stacks
5. Applies `topk_weight` scaling on the output

The `_ensure_pypto_weights()` method pre-processes expert weights into a single stacked tensor:
```python
self._pypto_w13_stack  # shape (num_experts, 2*intermediate_size, hidden_size) — gate+up fused
self._pypto_w2_stack   # shape (num_experts, hidden_size, intermediate_size) — down projection
```

## Usage

### 1. Load kernel module

```python
import sys
from pypto_gym.ops.pypto_tile.llada2_moe.llada2_moe_grouped_gemm_impl import grouped_gemm

class LLaDA2PTOKernels:
    USE_PTO_EXPERT_FFN = True

    @staticmethod
    def grouped_gemm(*args, **kwargs):
        return grouped_gemm(*args, **kwargs)

sys.modules["llada2_pto_kernels"] = LLaDA2PTOKernels
```

### 2. Instantiate model

```python
from pypto_gym.transformers.llada2_moe.configuration_llada2_moe import LLaDA2MoeConfig
from pypto_gym.transformers.llada2_moe.modeling_llada2_moe import LLaDA2MoeForCausalLM

config = LLaDA2MoeConfig()
model = LLaDA2MoeForCausalLM(config).npu()
```

### 3. Disable at runtime

```python
sys.modules["llada2_pto_kernels"].USE_PTO_EXPERT_FFN = False
# Falls back to per-expert Python loop in moe_infer()
```

### 4. Standalone grouped GEMM benchmark

```bash
source env_setup.sh
PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa \
TILE_FWK_DEVICE_ID=0 \
python src/pypto_gym/transformers/llada2_moe/bench_grouped_gemm.py
```

## HuggingFace `auto_map`

```json
{
  "auto_map": {
    "AutoConfig": "pypto_gym/transformers/llada2_moe/configuration_llada2_moe.LLaDA2MoeConfig",
    "AutoModelForCausalLM": "pypto_gym/transformers/llada2_moe/modeling_llada2_moe.LLaDA2MoeForCausalLM"
  }
}
```

## Architecture Notes

LLaDA2 uses a **group-limited top-k routing** mechanism:
- Experts divided into `n_group` groups (default: 8)
- First, select top-2 experts per group, sum scores, pick top `topk_group` groups (default: 4)
- Then, select top `num_experts_per_tok` experts (default: 2) only from selected groups
- Uses `routed_scaling_factor` (default: 2.5) to boost expert outputs

Each expert is a dense FFN (gate_proj → SiLU → up_proj gate → down_proj). The model uses partial rotary embeddings (`partial_rotary_factor=0.5`) meaning only half of the attention head dimensions receive RoPE.

## File Table

| File | Description |
|------|-------------|
| `configuration_llada2_moe.py` | `LLaDA2MoeConfig` — MoE configuration (routing params, expert counts, group topology) |
| `modeling_llada2_moe.py` | Full model graph (1498 lines) — `LLaDA2MoeForCausalLM`, `LLaDA2MoeSparseMoeBlock` (with PyPTO grouped GEMM at line 382), `LLaDA2MoeGate`, `LLaDA2MoeMLP`, `LLaDA2MoeRMSNorm` |
| `bench_grouped_gemm.py` | Standalone benchmark comparing eager vs PyPTO grouped GEMM performance |

## Related Directories

- **Ops**: `src/pypto_gym/ops/pypto_tile/llada2_moe/` — PyPTO kernels (`llada2_expert_ffn_impl.py`, `llada2_gate_select_impl.py`, `llada2_moe_grouped_gemm_impl.py`)
- **Tests**: `tests/ops/llada2_moe/` — correctness and performance tests
