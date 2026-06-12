# GutenOCR-3B PyPTO Kernel Integration

HuggingFace `GutenOcr_3b_VLForConditionalGeneration` vision-language model definition modified to inject PyPTO fused operators on Ascend NPU hardware. GutenOCR-3B is a modified Qwen2.5-VL architecture tuned for OCR tasks, with Huawei-specific PyPTO operator fusion replacing the RMSNorm computation path.

## Integration Scope

| Operation | PyPTO Integrated | Fallback | Notes |
|-----------|:---:|----------|-------|
| RMS LayerNorm | Yes | PyTorch fp32 | Wired through `sys.modules.get("pto_kernels")` at forward time |
| Attention (Q/K/V/O + RoPE) | No | eager / flash_attn / sdpa | Standard HuggingFace attention interface |
| MLP (SwiGLU) | No | PyTorch `nn.Linear` | Standard gate/up/down projection |
| Vision Encoder (ViT) | No | PyTorch `GutenOcr_3b_VisionPatchEmbed` / `GutenOcr_3b_VisionBlock` | Full vision pipeline with flash attention support |
| Patch Merger | No | PyTorch `GutenOcr_3b_VLPatchMerger` | Spatial merge and projection |
| MRoPE (Text) | No | PyTorch `GutenOcr_3b_VLTextRotaryEmbedding` | Multimodal rope with window attention support |
| RoPE (Vision) | No | PyTorch `GutenOcr_3b_VisionRotaryEmbedding` | Standard vision rope |

The RMSNorm integration is the sole PyPTO injection point. Every `GutenOcr_3b_VLRMSNorm` instance in the network (attention Q/K norms in vision, input layernorm, post-attention layernorm, final output norm, patch merger layernorm) checks for the kernel module at each forward call. The `@use_kernel_forward_from_hub("RMSNorm")` decorator is applied to the `GutenOcr_3b_VLRMSNorm` class, allowing HuggingFace Hub to additionally supply a kernel implementation.

## Switch Variables

The kernel module is expected to expose the following attributes in `sys.modules["pto_kernels"]`:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_PTO_RMS_NORM` | `bool` | `False` | Enable PyPTO fused RMSNorm; when `False` or absent, falls back to PyTorch fp32 path |

## Kernel API Contract

When `USE_PTO_RMS_NORM` is `True`, the kernel must provide:

```python
def rms_norm_wrapper(
    hidden_states: torch.Tensor,  # (batch, seq_len, hidden_size)
    weight: torch.Tensor,         # (hidden_size,)
    variance_epsilon: float,
) -> torch.Tensor:                # (batch, seq_len, hidden_size)
    ...
```

The PyTorch fallback computes:
```
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
return weight * hidden_states
```

The `rms_norm_wrapper` naming convention (vs `rms_norm_impl` used by qwen3_1_7b) indicates the kernel expects to handle multiple RMSNorm calls through a single wrapper rather than being a direct 1:1 drop-in replacement.

## Usage

### 1. Load kernel module into `sys.modules`

```python
import sys
from pypto_gym.ops.pypto_tile.gutenocr_3b.rms_norm import rms_norm_wrapper

class GutenOCRPTOKernels:
    USE_PTO_RMS_NORM = True

    @staticmethod
    def rms_norm_wrapper(hidden_states, weight, variance_epsilon):
        return rms_norm_wrapper(hidden_states, weight, variance_epsilon)

sys.modules["pto_kernels"] = GutenOCRPTOKernels
```

Note: gutenocr_3b uses the generic `"pto_kernels"` name (not `"gutenocr_pto_kernels"`). Multiple models (gutenocr_3b, qwen3_vl_8b, spatial_ssrl_3b, phi_3_mini) share this module name; ensure only one is loaded in a given process.

### 2. Instantiate model

```python
from pypto_gym.transformers.gutenocr_3b.configuration_gutenocr_3b_vl import GutenOcr_3b_VLConfig
from pypto_gym.transformers.gutenocr_3b.modeling_gutenocr_3b_vl import GutenOcr_3b_VLForConditionalGeneration

config = GutenOcr_3b_VLConfig()
model = GutenOcr_3b_VLForConditionalGeneration(config).npu()
```

### 3. Disable PyPTO at runtime

```python
sys.modules["pto_kernels"].USE_PTO_RMS_NORM = False
```

## HuggingFace `auto_map`

```json
{
  "auto_map": {
    "AutoConfig": "pypto_gym/transformers/gutenocr_3b/configuration_gutenocr_3b_vl.GutenOcr_3b_VLConfig",
    "AutoModelForVision2Seq": "pypto_gym/transformers/gutenocr_3b/modeling_gutenocr_3b_vl.GutenOcr_3b_VLForConditionalGeneration"
  }
}
```

## Architecture Notes

The GutenOCR model is based on Qwen2.5-VL with OCR-specific tuning. Key architecture features:
- **Deep vision encoder**: 32 ViT blocks with full attention and sliding window layers for the text decoder
- **Spatial merge**: 2x2 spatial patch merger to reduce vision token count
- **MRoPE**: Multimodal rotary position embeddings for text (supporting 3D positions for video)
- **Window attention**: Optional sliding window attention on later decoder layers (controlled by `max_window_layers`)
- **Vision RoPE**: Separate 2D rotary embeddings for vision patches

## File Table

| File | Description |
|------|-------------|
| `configuration_gutenocr_3b_vl.py` | `GutenOcr_3b_VLConfig`, `GutenOcr_3b_VLTextConfig`, `GutenOcr_3b_VLVisionConfig` — multimodal config with vision encoder specs, sliding window params, MRoPE settings |
| `modeling_gutenocr_3b_vl.py` | Full model graph (1776 lines) — `GutenOcr_3b_VLRMSNorm` (with `@use_kernel_forward_from_hub("RMSNorm")` + PyPTO injection), `GutenOcr_3b_VLForConditionalGeneration`, vision/text decoders, attention, MLP, patch embed/merger |

## Related Directories

- **Ops**: `src/pypto_gym/ops/pypto_tile/gutenocr_3b/` — PyPTO kernels (`rms_norm/`, `mrope/`, `swiglu_mlp/`)
- **Tests**: `tests/model_ops/gutenocr_3b/` — correctness and performance tests
