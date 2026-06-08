# Gemma4-31B-IT PyPTO Kernel Integration

HuggingFace `Gemma4ForConditionalGeneration` multimodal model definition modified to inject PyPTO fused operators on Ascend NPU hardware. Gemma4 (Google) is a vision-language-audio model with hybrid attention (swin transformer vision, conformer audio, alternating sliding/full attention text layers, MoE experts). This integration targets two specific compute paths: attention softmax and GQA decode attention.

## Integration Scope

| Operation | PyPTO Integrated | Fallback | Notes |
|-----------|:---:|----------|-------|
| Attention Softmax | Yes | `nn.functional.softmax` fp32 | Tiled PyPTO softmax kernel for large attention score tensors |
| GQA Decode Attention | Yes | eager / flash_attn / sdpa | Native Grouped-Query-Attention decode kernel (sliding layers, D=256, Sq=1) |
| RMSNorm | No | PyTorch `Gemma4RMSNorm` | Standard implementation, not fused |
| MoE FFN | No | HuggingFace `use_experts_implementation` | Expert FFN dispatch via Transformer's built-in MoE mechanism |
| Full Attention (prefill) | No | eager / flash_attn / sdpa | Standard HuggingFace attention |
| Vision Encoder | No | PyTorch `Gemma4VisionModel` | Swin transformer blocks with pooling |
| Audio Encoder | No | PyTorch `Gemma4AudioModel` | Conformer-based audio encoder |
| RoPE | No | PyTorch (multidimensional) | Full + sliding attention use different rope types |

The softmax injection happens inside `eager_attention_forward`—the common eager attention path used when flash_attn and sdpa are not available. The GQA decode injection happens in `Gemma4TextAttention.forward` and is only active for decode (sequence length = 1) on sliding window layers (head_dim = 256) in inference mode.

## Switch Variables

The kernel module is expected to expose the following in `sys.modules["gemma4_pto_kernels"]`:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_PTO_SOFTMAX` | `bool` | `False` | Enable tiled PyPTO softmax in eager attention forward |
| `USE_PTO_GQA` | `bool` | `False` | Enable native GQA decode attention for sliding layers (decode-only, D=256, Sq=1) |

## Kernel API Contract

### Softmax

```python
def attn_softmax(
    attn_weights: torch.Tensor,  # (B*N*Sq, Sk) — flattened attention scores
    scale: float = 1.0,
) -> torch.Tensor:               # (B*N*Sq, Sk) — softmax-normalized weights
    ...
```

Replaces `nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)`. The modeling code reshapes before/after:
```python
B, N, Sq, Sk = attn_weights.shape
attn_weights = _pk.attn_softmax(attn_weights.reshape(-1, Sk), scale=1.0).view(B, N, Sq, Sk)
```

### GQA Decode Attention

```python
def gqa_decode_attn(
    query_states: torch.Tensor,   # (B, num_heads, 1, head_dim)
    key_states: torch.Tensor,     # (B, num_kv_heads, S_kv, head_dim)
    value_states: torch.Tensor,   # (B, num_kv_heads, S_kv, head_dim)
    attention_mask: torch.Tensor | None,
    scaling: float,
    layer_kind: str,              # "local" or "global"
) -> torch.Tensor:                # (B, num_heads, 1, head_dim)
    ...
```

The GQA kernel is only called when:
- `query_states.shape[2] == 1` (single-token decode)
- `self.head_dim == 256` (sliding window layers only)
- `not self.training` (inference mode)

Global attention layers (head_dim=512) fall through to the standard `ALL_ATTENTION_FUNCTIONS` interface.

## Usage

### 1. Load kernel module

```python
import sys
from pypto_gym.ops.pypto_tile.gemma4_31b_it.attn_softmax import attn_softmax
from pypto_gym.ops.pypto_tile.gemma4_31b_it.gqa_decode_attn import gqa_decode_attn

class Gemma4PTOKernels:
    USE_PTO_SOFTMAX = True
    USE_PTO_GQA = True

    @staticmethod
    def attn_softmax(attn_weights, scale=1.0):
        return attn_softmax(attn_weights, scale)

    @staticmethod
    def gqa_decode_attn(query, key, value, mask, scaling, layer_kind):
        return gqa_decode_attn(query, key, value, mask, scaling, layer_kind)

sys.modules["gemma4_pto_kernels"] = Gemma4PTOKernels
```

### 2. Instantiate model

```python
from pypto_gym.transformers.gemma4_31b_it.configuration_gemma4 import Gemma4Config
from pypto_gym.transformers.gemma4_31b_it.modeling_gemma4 import Gemma4ForConditionalGeneration

config = Gemma4Config()
model = Gemma4ForConditionalGeneration(config).npu()
```

### 3. Selective disable

```python
sys.modules["gemma4_pto_kernels"].USE_PTO_SOFTMAX = False  # only disable softmax
sys.modules["gemma4_pto_kernels"].USE_PTO_GQA = False       # only disable GQA decode
```

## Architecture Notes

The Gemma4 attention dispatch is complex due to:
- **Swish-gated attention**: Q/K projections also produce a gate applied to attention output
- **Sliding vs full attention**: Alternating layers at 5:1 ratio; sliding uses D=256, full uses D=512
- **KV sharing**: Configurable number of consecutive layers reusing same KV states
- **Bidirectional attention**: Vision tokens can attend bidirectionally
- **MoE blocks**: Optional sparse expert FFN layers with top-k routing
- **Per-layer embeddings (PLE)**: Additional per-layer input embeddings

## HuggingFace `auto_map`

```json
{
  "auto_map": {
    "AutoConfig": "pypto_gym/transformers/gemma4_31b_it/configuration_gemma4.Gemma4Config",
    "AutoModelForVision2Seq": "pypto_gym/transformers/gemma4_31b_it/modeling_gemma4.Gemma4ForConditionalGeneration"
  }
}
```

## File Table

| File | Description |
|------|-------------|
| `configuration_gemma4.py` | `Gemma4Config`, `Gemma4TextConfig`, `Gemma4VisionConfig`, `Gemma4AudioConfig` — multimodal config with sliding/full attention, MoE parameters, KV sharing, PLE dimensions |
| `modeling_gemma4.py` | Complete model graph (3136 lines) — `Gemma4RMSNorm`, `Gemma4TextAttention` (with softmax + GQA patches at lines 974 and 1496), `Gemma4TextExperts`, vision/audio encoders, pooler, and conditional generation head |

## Related Directories

- **Ops**: `src/pypto_gym/ops/pypto_tile/gemma4_31b_it/` — PyPTO kernels (`attn_softmax/`, `gqa_decode_attn/`)
- **Tests**: `tests/ops/gemma4_31b_it/` — correctness and performance tests
