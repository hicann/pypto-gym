# Qwen3-VL-8B-Instruct-Unredacted-Max PyPTO Kernel Integration

HuggingFace `Qwen3VLForConditionalGeneration` vision-language model definition modified to inject PyPTO fused operators on Ascend NPU hardware. Qwen3-VL (Alibaba) is a multimodal model supporting images and video inputs with deepstack visual features. The PyPTO integration replaces the RMSNorm computation path with a fused PyPTO kernel.

## Integration Scope

| Operation | PyPTO Integrated | Fallback | Notes |
|-----------|:---:|----------|-------|
| RMS LayerNorm | Yes | PyTorch fp32 | Wired through `sys.modules.get("pto_kernels")` at forward time |
| Attention (Q/K/V/O + RoPE) | No | eager / flash_attn / sdpa | Standard HuggingFace attention with swish gating on Q |
| MLP (SwiGLU) | No | PyTorch `nn.Linear` | Standard gate/up/down projection |
| Vision Encoder (ViT) | No | PyTorch `Qwen3VLVisionModel` | Deepstack visual features from intermediate layers |
| Patch Merger | No | PyTorch `Qwen3VLPatchMerger` | Spatial merge with optional post-shuffle normalization |
| MRoPE (Text) | No | PyTorch `Qwen3VLTextRotaryEmbedding` | Interleaved MRoPE for multimodal positions (T/H/W dims) |
| RoPE (Vision) | No | PyTorch `Qwen3VLVisionRotaryEmbedding` | Standard 2D vision RoPE |

The RMSNorm integration is the sole PyPTO injection point, implemented in `Qwen3VLTextRMSNorm.forward()`. Every normalization instance in the text decoder and vision merger checks `sys.modules.get("pto_kernels")` at each forward call.

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

This is the same wrapper pattern used by gutenocr_3b and spatial_ssrl_3b. The PyTorch fallback is the standard RMSNorm computation.

## Usage

### 1. Load kernel module

```python
import sys
from pypto_gym.ops.pypto_tile.qwen3_vl_8b_instruct_unredacted_max.rms_norm import rms_norm_wrapper

class PTOKernels:
    USE_PTO_RMS_NORM = True

    @staticmethod
    def rms_norm_wrapper(hidden_states, weight, variance_epsilon):
        return rms_norm_wrapper(hidden_states, weight, variance_epsilon)

sys.modules["pto_kernels"] = PTOKernels
```

### 2. Instantiate model

```python
from pypto_gym.transformers.qwen3_vl_8b_instruct_unredacted_max.configuration_qwen3_vl import Qwen3VLConfig
from pypto_gym.transformers.qwen3_vl_8b_instruct_unredacted_max.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)

config = Qwen3VLConfig()
model = Qwen3VLForConditionalGeneration(config).npu()
```

### 3. Disable at runtime

```python
sys.modules["pto_kernels"].USE_PTO_RMS_NORM = False
```

## Architecture Notes

Qwen3-VL architecture features:

- **Deepstack visual features**: Vision encoder outputs from intermediate layers (indices 8, 16, 24 by default) are concatenated to provide multi-scale visual representations to the text decoder
- **Swish-gated attention**: Query projection includes a separate gate dimension (2x head_dim for q_proj), with `torch.sigmoid(gate)` applied before the output projection
- **Interleaved MRoPE**: Position embeddings use a 3D (T/H/W) interleaved layout specific to Qwen3-VL, with `mrope_section` controlling dimension allocation
- **Vision RoPE with 2D positions**: Vision tokens receive 2D rotary embeddings computed from grid positions
- **Variable-length packed attention**: Flash attention with `cu_seqlens` for efficient batched variable-length image/video processing

## HuggingFace `auto_map`

```json
{
  "auto_map": {
    "AutoConfig": "pypto_gym/transformers/qwen3_vl_8b_instruct_unredacted_max/configuration_qwen3_vl.Qwen3VLConfig",
    "AutoModelForVision2Seq": "pypto_gym/transformers/qwen3_vl_8b_instruct_unredacted_max/modeling_qwen3_vl.Qwen3VLForConditionalGeneration"
  }
}
```

## File Table

| File | Description |
|------|-------------|
| `configuration_qwen3_vl.py` | `Qwen3VLConfig`, `Qwen3VLTextConfig`, `Qwen3VLVisionConfig` — multimodal config with deepstack indices, MRoPE params, RoPE theta, vision ViT specs |
| `modeling_qwen3_vl.py` | Full model graph — `Qwen3VLTextRMSNorm` (with PyPTO RMSNorm injection at line ~410 via `sys.modules.get("pto_kernels")`), `Qwen3VLForConditionalGeneration`, vision encoder, text decoder, deepstack feature extraction |

## Related Directories

- **Ops**: `src/pypto_gym/ops/pypto_tile/qwen3_vl_8b_instruct_unredacted_max/` — PyPTO kernels (`rms_norm/`, `rope/`)
- **Tests**: `tests/model_ops/qwen3_vl_8b_instruct_unredacted_max/` — correctness and performance tests
