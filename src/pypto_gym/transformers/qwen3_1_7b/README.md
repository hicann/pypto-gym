# Qwen3-1.7B PyPTO Kernel Integration

HuggingFace `Qwen3ForCausalLM` model definition modified to inject PyPTO fused operators on Ascend NPU hardware. This is a pure-text causal language model derived from the Qwen3 family (Alibaba), with Huawei-specific modifications for operator fusion.

## Integration Scope

| Operation | PyPTO Integrated | Fallback | Notes |
|-----------|:---:|----------|-------|
| RMS LayerNorm | Yes | PyTorch fp32 | Wired through `sys.modules` at forward time |
| Attention (Q/K/V/O + RoPE) | No | eager / flash_attn / sdpa | Standard HuggingFace attention interface via `ALL_ATTENTION_FUNCTIONS` |
| MLP (SwiGLU) | No | PyTorch `nn.Linear` | Standard gate/up/down projection |
| Rotary Embedding | No | PyTorch `Qwen3RotaryEmbedding` | Standard rope computation |

The RMSNorm integration is the sole PyPTO injection point. Every `Qwen3RMSNorm` instance in the network (attention Q/K norms, input layernorm, post-attention layernorm, and final output norm) checks for the kernel module at each forward call.

## Switch Variables

The kernel module is expected to expose the following attributes in `sys.modules["qwen3_pto_kernels"]`:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_PTO_RMS_NORM` | `bool` | `False` | Enable PyPTO fused RMSNorm; when `False` or absent, falls back to PyTorch fp32 path |

## Kernel API Contract

When `USE_PTO_RMS_NORM` is `True`, the kernel must provide:

```python
# Signature matching Qwen3RMSNorm.forward:
def rms_norm_impl(
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

The PyPTO kernel should produce numerically equivalent output (within fp16/bp16 tolerance).

## Usage

### 1. Load kernel module into `sys.modules`

```python
import sys
from pypto_gym.ops.pypto_tile.qwen3_1_7b.rms_norm import rms_norm_impl

class Qwen3PTOKernels:
    USE_PTO_RMS_NORM = True

    @staticmethod
    def rms_norm_impl(hidden_states, weight, variance_epsilon):
        # Delegate to the PyPTO-compiled kernel
        return rms_norm_impl(hidden_states, weight, variance_epsilon)

sys.modules["qwen3_pto_kernels"] = Qwen3PTOKernels
```

### 2. Instantiate model

```python
from pypto_gym.transformers.qwen3_1_7b.modeling_qwen3 import Qwen3ForCausalLM
from pypto_gym.transformers.qwen3_1_7b.configuration_qwen3 import Qwen3Config

config = Qwen3Config()
model = Qwen3ForCausalLM(config).npu()
```

### 3. Disable PyPTO at runtime

```python
sys.modules["qwen3_pto_kernels"].USE_PTO_RMS_NORM = False  # back to PyTorch
```

## HuggingFace `auto_map`

This model replaces the standard `transformers` Qwen3 implementation. Set the following in your model's `config.json` to enable `TrustRemoteCode` loading:

```json
{
  "auto_map": {
    "AutoConfig": "pypto_gym/transformers/qwen3_1_7b/configuration_qwen3.Qwen3Config",
    "AutoModelForCausalLM": "pypto_gym/transformers/qwen3_1_7b/modeling_qwen3.Qwen3ForCausalLM"
  }
}
```

## File Table

| File | Description |
|------|-------------|
| `configuration_qwen3.py` | `Qwen3Config` — model configuration (4096 hidden, 32 layers, 32 heads, head_dim=128, Sliding Window + Full Attention alternating) |
| `modeling_qwen3.py` | `Qwen3ForCausalLM`, `Qwen3Model`, `Qwen3DecoderLayer`, `Qwen3Attention`, `Qwen3RMSNorm`, `Qwen3MLP`, `Qwen3RotaryEmbedding` — full model graph, with `Qwen3RMSNorm.forward` dispatching to PyPTO kernel via `sys.modules.get("qwen3_pto_kernels")` |

## Related Directories

- **Ops**: `src/pypto_gym/ops/pypto_tile/qwen3_1_7b/` — contains PyPTO kernel implementations (`rms_norm/`, `rms_norm_rope/`)
- **Tests**: `tests/ops/qwen3_1_7b/` and `tests/model_ops/qwen3_1_7b/` — correctness and performance tests
