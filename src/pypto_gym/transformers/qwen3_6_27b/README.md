# Qwen3.6-27B PyPTO Kernel Integration

HuggingFace `Qwen3_5ForConditionalGeneration` multimodal model definition (27B variant) modified to inject PyPTO fused operators on Ascend NPU hardware. This is the 27-billion parameter version of the Qwen3.5 vision-language model, sharing the same codebase as qwen3_5_9b but with a distinct kernel module namespace and larger model dimensions.

## Integration Scope

| Operation | PyPTO Integrated | Fallback | Notes |
|-----------|:---:|----------|-------|
| Gated Delta Rule (chunk) | Yes | `chunk_gated_delta_rule` (FLA or torch) | Fused chunk-level GDR kernel for prefill with `qwen3_6_27b_pto_kernels` |
| RMSNorm | No | PyTorch `Qwen3_5RMSNorm` | Standard implementation, not fused |
| Gated RMSNorm | No | `FusedRMSNormGated` (FLA) or `Qwen3_5RMSNormGated` | Post-attention gated normalization |
| Full Attention | No | eager / flash_attn / sdpa | Standard HuggingFace attention |
| MLP (SwiGLU) | No | PyTorch `nn.Linear` | Standard |
| RoPE (MRoPE) | No | PyTorch `Qwen3_5TextRotaryEmbedding` | Interleaved MRoPE |
| Vision Encoder | No | PyTorch `Qwen3_5VisionModel` | Full vision pipeline |
| Conv1D (causal) | No | `causal_conv1d_fn` (optional) or torch | Causal convolution pre-processing |

The PyPTO injection point is identical in structure to qwen3_5_9b but targets a separate kernel namespace (`qwen3_6_27b_pto_kernels`) to allow independent kernel compilation and tuning for the 27B model's specific tensor shapes and parallelism requirements.

## Switch Variables

The kernel module is expected to expose the following in `sys.modules["qwen3_6_27b_pto_kernels"]`:

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
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    ...
```

Same signature as the 9B variant but expects the 27B model's tensor shapes (larger hidden size, more attention heads, etc.). The modeling code at line ~530 checks `sys.modules.get("qwen3_6_27b_pto_kernels")` and falls back to the original chunk kernel on `NotImplementedError`.

## 27B vs 9B Model Dimensions

The 27B variant uses significantly larger model dimensions. Typical configuration differences:

| Parameter | 9B (typical) | 27B (typical) |
|-----------|-------------|---------------|
| `hidden_size` | 4096 | larger embedding dimension |
| `intermediate_size` | 12288 | larger FFN hidden dim |
| `num_hidden_layers` | 32 | deeper model |
| `num_attention_heads` | 16 | more attention heads |
| `linear_num_key_heads` | 16 | potentially more |
| `linear_num_value_heads` | 32 | potentially more |

The actual config values are read from `Qwen3_5TextConfig` at runtime and passed through to the PyPTO kernel which must handle the corresponding tensor shapes.

## Usage

### 1. Load kernel module

```python
import sys
from pypto_gym.ops.pypto_tile.qwen3_6_27b.gated_delta_rule import gated_delta_rule_impl

class Qwen3_6_27bPTOKernels:
    USE_PTO_GATED_DELTA_RULE = True

    @staticmethod
    def gated_delta_rule_wrapper(*args, **kwargs):
        return gated_delta_rule_impl(*args, **kwargs)

sys.modules["qwen3_6_27b_pto_kernels"] = Qwen3_6_27bPTOKernels
```

### 2. Instantiate model

```python
from pypto_gym.transformers.qwen3_6_27b.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from pypto_gym.transformers.qwen3_6_27b.configuration_qwen3_5 import Qwen3_5Config

config = Qwen3_5Config()
model = Qwen3_5ForConditionalGeneration(config).npu()
```

### 3. Disable at runtime

```python
sys.modules["qwen3_6_27b_pto_kernels"].USE_PTO_GATED_DELTA_RULE = False
```

## HuggingFace `auto_map`

```json
{
  "auto_map": {
    "AutoConfig": "pypto_gym/transformers/qwen3_6_27b/configuration_qwen3_5.Qwen3_5Config",
    "AutoModelForVision2Seq": "pypto_gym/transformers/qwen3_6_27b/modeling_qwen3_5.Qwen3_5ForConditionalGeneration"
  }
}
```

## File Table

| File | Description |
|------|-------------|
| `__init__.py` | Package init (SPDX license header) |
| `configuration_qwen3_5.py` | `Qwen3_5Config`, `Qwen3_5TextConfig`, `Qwen3_5VisionConfig` — multimodal config with 27B-specific defaults (larger hidden_size, wider intermediate, more layers) |
| `modeling_qwen3_5.py` | Full model graph with GDR chunk injection at `sys.modules.get("qwen3_6_27b_pto_kernels")` — identical code structure to qwen3_5_9b but separate kernel namespace |

## Related Directories

- **Ops**: `src/pypto_gym/ops/pypto_tile/qwen3_6_27b/` — PyPTO Gated Delta Rule kernel (`gated_delta_rule/`)
- **Tests**: `tests/model_ops/qwen3_6_27b/` — correctness and performance tests for the 27B variant
