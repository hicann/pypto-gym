# Qwen3.5-9B PyPTO Kernel Integration

HuggingFace `Qwen3_5ForConditionalGeneration` multimodal model definition modified to inject PyPTO fused operators on Ascend NPU hardware. This model (Qwen3.5-9B) is a vision-language model supporting hybrid attention: standard full attention layers alternate with linear attention layers based on the Gated Delta Rule (GDR).

## Integration Scope

| Operation | PyPTO Integrated | Fallback | Notes |
|-----------|:---:|----------|-------|
| Gated Delta Rule (chunk) | Yes | `chunk_gated_delta_rule` (FLA or torch) | Fused chunk-level GDR kernel for prefill with `qwen3_5_9b_pto_kernels` |
| RMSNorm | No | PyTorch `Qwen3_5RMSNorm` | Standard implementation, not fused |
| Gated RMSNorm | No | `FusedRMSNormGated` (FLA) or `Qwen3_5RMSNormGated` | Post-attention gated normalization |
| Full Attention | No | eager / flash_attn / sdpa | Standard HuggingFace attention |
| MLP (SwiGLU) | No | PyTorch `nn.Linear` | Standard |
| RoPE (MRoPE) | No | PyTorch `Qwen3_5TextRotaryEmbedding` | Interleaved MRoPE for multimodal positions |
| Vision Encoder | No | PyTorch `Qwen3_5VisionModel` | Full vision pipeline (patch embed, ViT blocks, merger) |
| Conv1D (causal) | No | `causal_conv1d_fn` (optional) or torch | Causal convolution in GDR pre-processing |

The Gated Delta Rule injection replaces the tensor-level chunk computation in `Qwen3_5GatedDeltaNet.forward()`. When the kernel switch is active and the model is in prefill mode (not using cached recurrent states), the PyPTO kernel handles the entire Q/K/V/Beta/Gate chunk computation as a single fused kernel call.

## Switch Variables

The kernel module is expected to expose the following attributes in `sys.modules["qwen3_5_9b_pto_kernels"]`:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_PTO_GATED_DELTA_RULE` | `bool` | `False` | Enable PyPTO fused Gated Delta Rule chunk kernel for prefill |

## Kernel API Contract

When `USE_PTO_GATED_DELTA_RULE` is `True`, the kernel must provide:

```python
def gated_delta_rule_wrapper(
    query: torch.Tensor,   # (B, num_k_heads, S, head_k_dim)
    key: torch.Tensor,     # (B, num_k_heads, S, head_k_dim)
    value: torch.Tensor,   # (B, num_v_heads, S, head_v_dim)
    g: torch.Tensor,       # (B, num_v_heads, S) — per-timestep decay
    beta: torch.Tensor,    # (B, num_v_heads, S) — per-timestep gate
    initial_state: torch.Tensor | None,  # (B, num_v_heads, head_k_dim, head_v_dim) or None
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Returns:
    #   core_attn_out: (B, S, num_v_heads, head_v_dim)
    #   last_recurrent_state: (B, num_v_heads, head_k_dim, head_v_dim) or None
    ...
```

The function replaces `chunk_gated_delta_rule` (either the FLA C++ implementation or the pure-torch fallback `torch_chunk_gated_delta_rule`). On `NotImplementedError`, the modeling code gracefully falls back to the original chunk function.

Key constraints:
- Only active during **prefill** (no cached recurrent state, `use_precomputed_states == False`)
- Decode (seq_len=1 with cached state) always uses `fused_recurrent_gated_delta_rule` (FLA) or `torch_recurrent_gated_delta_rule`
- Q/K L2-normalization (`use_qk_l2norm_in_kernel=True`) is expected inside the kernel

## Usage

### 1. Load kernel module

```python
import sys
from pypto_gym.ops.pypto_tensor.qwen3_5_9b.gated_delta_rule import gated_delta_rule_impl

class Qwen3_5_9bPTOKernels:
    USE_PTO_GATED_DELTA_RULE = True

    @staticmethod
    def gated_delta_rule_wrapper(*args, **kwargs):
        return gated_delta_rule_impl(*args, **kwargs)

sys.modules["qwen3_5_9b_pto_kernels"] = Qwen3_5_9bPTOKernels
```

### 2. Instantiate model

```python
from pypto_gym.transformers.qwen3_5_9b.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from pypto_gym.transformers.qwen3_5_9b.configuration_qwen3_5 import Qwen3_5Config

config = Qwen3_5Config()
model = Qwen3_5ForConditionalGeneration(config).npu()
```

### 3. Disable at runtime

```python
sys.modules["qwen3_5_9b_pto_kernels"].USE_PTO_GATED_DELTA_RULE = False
```

## HuggingFace `auto_map`

```json
{
  "auto_map": {
    "AutoConfig": "pypto_gym/transformers/qwen3_5_9b/configuration_qwen3_5.Qwen3_5Config",
    "AutoModelForVision2Seq": "pypto_gym/transformers/qwen3_5_9b/modeling_qwen3_5.Qwen3_5ForConditionalGeneration"
  }
}
```

## Architecture Highlights

The Gated Delta Rule (GDR) replaces standard attention in "linear_attention" layers with a recurrent state-space mechanism:

1. **Preprocessing**: Causal Conv1D over QKV projections for temporal smoothing
2. **Discretization**: Per-head learnable `A_log` + `dt_bias` produce timestep-dependent decay `g`
3. **Gating**: Per-head sigmoid `beta` controls delta magnitude
4. **Chunk Recurrence**: Sequence processed in chunks with intra-chunk attention and inter-chunk state passing
5. **Norm + Gating**: Post-attention norm applied with output gate `z` using `FusedRMSNormGated`

Layer type alternation pattern (default): 3 linear_attention layers → 1 full_attention layer → repeat.

## File Table

| File | Description |
|------|-------------|
| `__init__.py` | Package init (SPDX license header) |
| `configuration_qwen3_5.py` | `Qwen3_5Config`, `Qwen3_5TextConfig`, `Qwen3_5VisionConfig` — multimodal config with linear attention params (conv kernel, key/value head dims, head counts) |
| `modeling_qwen3_5.py` | Full model graph: `Qwen3_5GatedDeltaNet`, `Qwen3_5Attention`, `Qwen3_5DecoderLayer`, `Qwen3_5VisionModel`, `Qwen3_5ForConditionalGeneration` — GDR chunk injection at line ~530 |

## Related Directories

- **Ops**: `src/pypto_gym/ops/pypto_tensor/qwen3_5_9b/` — PyPTO Gated Delta Rule kernel (`gated_delta_rule/`)
- **Tests**: `tests/model_ops/qwen3_5_9b/` — correctness and performance tests
