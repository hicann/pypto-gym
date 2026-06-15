# Spatial-SSRL-3B PyPTO Kernel Integration

HuggingFace `Spatial_ssrl_3b_VLForConditionalGeneration` model definition modified to inject PyPTO fused operators on Ascend NPU hardware. This is a multimodal vision-language model derived from the Qwen2.5-VL family (InternLM Spatial-SSRL variant), with Huawei-specific modifications for operator fusion.

## Integration Scope

| Operation | PyPTO Integrated | Fallback | Notes |
|-----------|:---:|----------|-------|
| RMS LayerNorm | Yes | PyTorch fp32 | Wired through `sys.modules` at forward time via `_apply_rms_norm` helper |
| RoPE (Text) | Yes | PyTorch `apply_multimodal_rotary_pos_emb` | Multimodal rotary position embedding with mrope_section |
| RoPE (Vision) | Yes | PyTorch `apply_rotary_pos_emb_vision` | Vision-specific rotary embedding |
| Attention | No | eager / flash_attn / sdpa | Standard HuggingFace attention interface via `ALL_ATTENTION_FUNCTIONS` |
| MLP (SwiGLU) | No | PyTorch `nn.Linear` | Standard gate/up/down projection |

The RMSNorm and RoPE integrations are the primary PyPTO injection points. Every normalization layer in the network checks for the kernel module at each forward call through the `_apply_rms_norm` wrapper function.

## Switch Variables

The kernel module is expected to expose the following attributes in `sys.modules["spatial_ssrl_3b_pto_kernels"]`:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_PTO_RMS_NORM` | `bool` | `False` | Enable PyPTO fused RMSNorm; when `False` or absent, falls back to PyTorch fp32 path |
| `USE_PTO_ROPE` | `bool` | `False` | Enable PyPTO fused RoPE for both text and vision; when `False` or absent, falls back to PyTorch |

## Kernel API Contract

### RMSNorm

When `USE_PTO_RMS_NORM` is `True`, the kernel must provide:

```python
def rms_norm_pto_wrapper(
    hidden_states: torch.Tensor,  # (batch, seq_len, hidden_size)
    weight: torch.Tensor,         # (hidden_size,)
    variance_epsilon: float,
) -> torch.Tensor:                # (batch, seq_len, hidden_size)
    ...
```

### Text RoPE

When `USE_PTO_ROPE` is `True`, for text attention layers:

```python
def apply_multimodal_rotary_pos_emb_wrapper(
    q: torch.Tensor,              # (batch, heads, seq_len, head_dim)
    k: torch.Tensor,              # (batch, heads, seq_len, head_dim)
    cos: torch.Tensor,            # multimodal cosine embeddings
    sin: torch.Tensor,            # multimodal sine embeddings
    mrope_section: List[int],     # e.g., [16, 24, 24]
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...
```

### Vision RoPE

For vision attention blocks:

```python
def apply_rotary_pos_emb_vision_wrapper(
    q: torch.Tensor,              # (seq_len, num_heads, head_dim)
    k: torch.Tensor,              # (seq_len, num_heads, head_dim)
    cos: torch.Tensor,            # vision cosine embeddings
    sin: torch.Tensor,            # vision sine embeddings
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...
```

## Usage

### 1. Load kernel module into `sys.modules`

```python
import sys
from pypto_gym.ops.pypto_tile.spatial_ssrl_3b.rms_norm import rms_norm_pto_wrapper
from pypto_gym.ops.pypto_tile.spatial_ssrl_3b.rope import (
    apply_multimodal_rotary_pos_emb_wrapper,
    apply_rotary_pos_emb_vision_wrapper,
)

class SpatialSSRL3BPTOKernels:
    USE_PTO_RMS_NORM = True
    USE_PTO_ROPE = True
    
    @staticmethod
    def rms_norm_pto_wrapper(hidden_states, weight, variance_epsilon):
        return rms_norm_pto_wrapper(hidden_states, weight, variance_epsilon)
    
    @staticmethod
    def apply_multimodal_rotary_pos_emb_wrapper(q, k, cos, sin, mrope_section):
        return apply_multimodal_rotary_pos_emb_wrapper(q, k, cos, sin, mrope_section)
    
    @staticmethod
    def apply_rotary_pos_emb_vision_wrapper(q, k, cos, sin):
        return apply_rotary_pos_emb_vision_wrapper(q, k, cos, sin)

sys.modules["spatial_ssrl_3b_pto_kernels"] = SpatialSSRL3BPTOKernels
```

### 2. Instantiate model

```python
from pypto_gym.transformers.spatial_ssrl_3b.modeling_spatial_ssrl_3b import Spatial_ssrl_3b_VLForConditionalGeneration
from pypto_gym.transformers.spatial_ssrl_3b.configuration_spatial_ssrl_3b import Spatial_ssrl_3b_VLConfig

config = Spatial_ssrl_3b_VLConfig()
model = Spatial_ssrl_3b_VLForConditionalGeneration(config).npu()
```

### 3. Disable PyPTO at runtime

```python
sys.modules["spatial_ssrl_3b_pto_kernels"].USE_PTO_RMS_NORM = False  # back to PyTorch for RMSNorm
sys.modules["spatial_ssrl_3b_pto_kernels"].USE_PTO_ROPE = False      # back to PyTorch for RoPE
```

## HuggingFace `auto_map`

This model replaces the standard `transformers` Qwen2.5-VL implementation. Set the following in your model's `config.json` to enable `TrustRemoteCode` loading:

```json
{
  "auto_map": {
    "AutoConfig": "pypto_gym/transformers/spatial_ssrl_3b/configuration_spatial_ssrl_3b.Spatial_ssrl_3b_VLConfig",
    "AutoModel": "pypto_gym/transformers/spatial_ssrl_3b/modeling_spatial_ssrl_3b.Spatial_ssrl_3b_VLModel",
    "AutoModelForCausalLM": "pypto_gym/transformers/spatial_ssrl_3b/modeling_spatial_ssrl_3b.Spatial_ssrl_3b_VLForConditionalGeneration"
  }
}
```

## Model Specifications

- **Architecture**: Qwen2.5-VL (Multimodal Vision-Language)
- **Model Type**: `spatial_ssrl_3b_vl`
- **Hidden Size**: 2048
- **Intermediate Size**: 11008
- **Attention Heads**: 16
- **KV Heads**: 2 (Grouped Query Attention)
- **Layers**: 36
- **Max Position Embeddings**: 128000
- **RoPE Theta**: 1000000.0
- **Vision Encoder**: 32-layer ViT with spatial merge (1280 hidden, 14x14 patches)

## Performance Benchmarks

| Metric | Baseline (PyTorch) | PyPTO (RMS Norm + RoPE) | Improvement |
|--------|-------------------|------------------------|-------------|
| Inference Time | 2.062s | 1.505s | 27% faster |
| Throughput | 14.5 tokens/s | 19.9 tokens/s | **37% higher** |
| Peak Memory | 7208.2MB | 7208.2MB | Same |

Test conditions: Prompt "你好，介绍一下华为昇腾NPU", Output 30 tokens, NPU device

## File Table

| File | Description |
|------|-------------|
| `configuration_spatial_ssrl_3b.py` | `Spatial_ssrl_3b_VLConfig`, `Spatial_ssrl_3b_VLTextConfig`, `Spatial_ssrl_3b_VLVisionConfig` — model configuration classes |
| `modeling_spatial_ssrl_3b.py` | `Spatial_ssrl_3b_VLForConditionalGeneration`, `Spatial_ssrl_3b_VLModel`, decoder layers, vision transformer, attention modules — full model graph with PyPTO kernel dispatch via `sys.modules.get("spatial_ssrl_3b_pto_kernels")` |

## Related Directories

- **Ops**: `src/pypto_gym/ops/pypto_tile/spatial_ssrl_3b/` — contains PyPTO kernel implementations (`rms_norm/`, `rope/`)
- **Tests**: `tests/ops/spatial_ssrl_3b/` and `tests/model_ops/spatial_ssrl_3b/` — correctness and performance tests
- **Original Model**: `/data/h00520348/optimize525/models/spatial_ssrl_3b/` — source model weights and inference scripts

## Citation

```bibtex
@article{liu2025spatial,
  title={Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning},
  author={Liu, Yuhong and Zhang, Beichen and Zang, Yuhang and Cao, Yuhang and Xing, Long and Dong, Xiaoyi and Duan, Haodong and Lin, Dahua and Wang, Jiaqi},
  journal={arXiv preprint arXiv:2510.27606},
  year={2025}
}
```
