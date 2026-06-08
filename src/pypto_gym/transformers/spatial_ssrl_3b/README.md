# Spatial-SSRL-3B PyPTO Kernel Integration

HuggingFace `Qwen2_5_VLForConditionalGeneration` vision-language model definition modified to inject PyPTO fused operators on Ascend NPU hardware. Spatial-SSRL-3B is a modified Qwen2.5-VL architecture tuned for spatial self-supervised representation learning, with the standard PyPTO RMSNorm fusion replacing the PyTorch normalization path.

## Integration Scope

| Operation | PyPTO Integrated | Fallback | Notes |
|-----------|:---:|----------|-------|
| RMS LayerNorm | Yes | PyTorch fp32 | Wired through `sys.modules.get("pto_kernels")` at forward time |
| Attention (Q/K/V/O + RoPE) | No | eager / flash_attn / sdpa | Standard HuggingFace attention interface |
| MLP (SwiGLU) | No | PyTorch `nn.Linear` | Standard gate/up/down projection |
| Vision Encoder (ViT) | No | PyTorch `Qwen2_5_VisionPatchEmbed` / `Qwen2_5_VisionBlock` | Full vision pipeline with flash attention and window attention |
| Patch Merger | No | PyTorch `Qwen2_5_VLPatchMerger` | Spatial merge with GELU activation |
| MRoPE (Text) | No | PyTorch `Qwen2_5_VLTextRotaryEmbedding` | 3D multimodal RoPE supporting video inputs |
| RoPE (Vision) | No | PyTorch `Qwen2_5_VisionRotaryEmbedding` | Standard 2D vision RoPE |

The integration is identical in structure to gutenocr_3b — both models share the same Qwen2.5-VL codebase with the `Qwen2_5_VLRMSNorm.forward()` method checking `sys.modules.get("pto_kernels")` and calling `pto_kernels.rms_norm_wrapper()` when enabled.

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

The PyTorch fallback is the standard RMSNorm:
```
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
return weight * hidden_states
```

## Usage

### 1. Load kernel module

```python
import sys
from pypto_gym.ops.pypto_tile.spatial_ssrl_3b.rms_norm import rms_norm_wrapper

class PTOKernels:
    USE_PTO_RMS_NORM = True

    @staticmethod
    def rms_norm_wrapper(hidden_states, weight, variance_epsilon):
        return rms_norm_wrapper(hidden_states, weight, variance_epsilon)

sys.modules["pto_kernels"] = PTOKernels
```

Note: spatial_ssrl_3b shares the `"pto_kernels"` module name with gutenocr_3b, phi_3_mini, and qwen3_vl_8b. Ensure only one model loads into `sys.modules["pto_kernels"]` per process.

### 2. Instantiate model

```python
from pypto_gym.transformers.spatial_ssrl_3b.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from pypto_gym.transformers.spatial_ssrl_3b.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

config = Qwen2_5_VLConfig()
model = Qwen2_5_VLForConditionalGeneration(config).npu()
```

### 3. Disable at runtime

```python
sys.modules["pto_kernels"].USE_PTO_RMS_NORM = False
```

## Architecture Notes

Spatial-SSRL-3B inherits the full Qwen2.5-VL architecture:
- **Deep vision encoder**: 32-layer ViT with full_attention on specific block indexes (`fullatt_block_indexes = [7, 15, 23, 31]`), window attention elsewhere
- **Spatial merge**: 2x2 patch merging with GELU MLP to reduce vision tokens
- **Video support**: Temporal patching with 3D convolutions (temporal_patch_size=2) and tokens_per_second video frame rate control
- **MRoPE text embeddings**: 3D position embeddings handling temporal, height, and width dimensions
- **Sliding window text attention**: Optional on later layers, controlled by `max_window_layers`
- **Large vocabulary**: 152,064 tokens

The spatial-SSRL variant is tuned for spatial representation tasks, with PyPTO RMSNorm acceleration on the Ascend NPU providing inference speedup across all normalization points in both the vision and text branches.

## HuggingFace `auto_map`

```json
{
  "auto_map": {
    "AutoConfig": "pypto_gym/transformers/spatial_ssrl_3b/configuration_qwen2_5_vl.Qwen2_5_VLConfig",
    "AutoModelForVision2Seq": "pypto_gym/transformers/spatial_ssrl_3b/modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration"
  }
}
```

## File Table

| File | Description |
|------|-------------|
| `configuration_qwen2_5_vl.py` | `Qwen2_5_VLConfig`, `Qwen2_5_VLTextConfig`, `Qwen2_5_VLVisionConfig` — multimodal config with ViT depth, window attention indices, MRoPE settings, video tokenization params |
| `modeling_qwen2_5_vl.py` | Full model graph — `Qwen2_5_VLRMSNorm` (with PyPTO RMSNorm injection via `sys.modules.get("pto_kernels")` + `@use_kernel_forward_from_hub`), `Qwen2_5_VLForConditionalGeneration`, vision blocks with flash attention, text decoder with sliding window support |

## Related Directories

- **Ops**: `src/pypto_gym/ops/pypto_tile/spatial_ssrl_3b/` — PyPTO kernels (`rms_norm/`, `rope/`)
- **Tests**: `tests/model_ops/spatial_ssrl_3b/` — correctness and performance tests
