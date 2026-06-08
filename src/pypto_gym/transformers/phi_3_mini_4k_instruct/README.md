# Phi-3-Mini-4K-Instruct PyPTO Kernel Integration

HuggingFace `Phi3ForCausalLM` model definition modified to inject PyPTO fused operators on Ascend NPU hardware. Phi-3-Mini (Microsoft) is a 3.8B parameter causal language model with a Llama-style architecture. This integration provides two RMSNorm backend options: a PyPTO tile-based path and an ACL (Ascend Compute Library) graph-based path.

## Integration Scope

| Operation | PyPTO Integrated | Fallback | Notes |
|-----------|:---:|----------|-------|
| RMS LayerNorm | Yes (dual path) | PyTorch fp32 | PyPTO tiled path + ACL graph path, selected via `USE_ACL_GRAPH` |
| Attention (MHA/GQA) | No | eager / flash_attn / sdpa | Standard HuggingFace attention |
| MLP (SiLU) | No | PyTorch `nn.Linear` | Standard gate/up/down projection |
| RoPE (LongRope) | No | PyTorch `Phi3RotaryEmbedding` | Supports long rope scaling with short/long factors |
| SuScaled RoPE | No | PyTorch `Phi3SuScaledRotaryEmbedding` | Used for extended context (128K) variant |

Unlike the simpler single-path integrations, `Phi3RMSNorm` has a two-tier dispatch:
1. If `USE_PTO_RMS_NORM` is `True`, check `USE_ACL_GRAPH`
2. If `USE_ACL_GRAPH` is `True`, use `pto_kernels.rms_norm.rms_norm_pypto()` (ACL graph path)
3. If `USE_ACL_GRAPH` is `False`, use `pto_kernels.rms_norm.rms_norm_pto()` (PyPTO tile path)
4. If no kernel module or `USE_PTO_RMS_NORM` is `False`, fall back to PyTorch fp32 path

## Switch Variables

The kernel module is expected to expose the following in `sys.modules["pto_kernels"]`:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_PTO_RMS_NORM` | `bool` | `False` | Master switch: enable any PyPTO/ACL RMSNorm path |
| `USE_ACL_GRAPH` | `bool` | `False` | Select ACL graph-based RMSNorm (when `True`) vs PyPTO tile-based (when `False`) |

## Kernel API Contract

### ACL Graph Path

```python
# pto_kernels.rms_norm.rms_norm_pypto
def rms_norm_pypto(
    hidden_states: torch.Tensor,  # input tensor
    weight: torch.Tensor,         # RMSNorm weight (1 + weight pattern in Phi-3)
) -> torch.Tensor:                # normalized output
    ...
```

### PyPTO Tile Path

```python
# pto_kernels.rms_norm.rms_norm_pto
def rms_norm_pto(
    hidden_states: torch.Tensor,  # input tensor
    weight: torch.Tensor,         # RMSNorm weight
) -> torch.Tensor:                # normalized output
    ...
```

Note the key difference from other models: Phi-3's RMSNorm follows the Llama convention where weights are initialized to zeros and the effective scale is `(1.0 + weight)`. The PyPTO kernel must handle this internally.

The PyTorch fallback:
```
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
return weight * hidden_states
```

## Usage

### 1. Load kernel module (PyPTO tile path)

```python
import sys
from pypto_gym.ops.pypto_tile.phi_3_mini_4k_instruct.rms_norm import rms_norm_pto

class PTOKernels:
    USE_PTO_RMS_NORM = True
    USE_ACL_GRAPH = False  # Use PyPTO tile path

    class rms_norm:
        @staticmethod
        def rms_norm_pto(hidden_states, weight):
            return rms_norm_pto(hidden_states, weight)

        @staticmethod
        def rms_norm_pypto(hidden_states, weight):
            raise NotImplementedError("ACL graph path not loaded")

sys.modules["pto_kernels"] = PTOKernels
```

### 2. Load kernel module (ACL graph path)

```python
import sys
# ACL graph path requires torch_npu and Ascend runtime
import torch_npu

class PTOKernels:
    USE_PTO_RMS_NORM = True
    USE_ACL_GRAPH = True

    class rms_norm:
        @staticmethod
        def rms_norm_pypto(hidden_states, weight):
            # Use ACL operator for RMSNorm
            return torch_npu.npu_rms_norm(hidden_states, weight)[0]

        @staticmethod
        def rms_norm_pto(hidden_states, weight):
            raise NotImplementedError("PyPTO tile path not loaded")

sys.modules["pto_kernels"] = PTOKernels
```

### 3. Instantiate model

```python
from pypto_gym.transformers.phi_3_mini_4k_instruct.configuration_phi3 import Phi3Config
from pypto_gym.transformers.phi_3_mini_4k_instruct.modeling_phi3 import Phi3ForCausalLM

config = Phi3Config()
model = Phi3ForCausalLM(config).npu()
```

### 4. Runtime toggling

```python
# Switch between paths at runtime
sys.modules["pto_kernels"].USE_ACL_GRAPH = False  # switch to tile path
sys.modules["pto_kernels"].USE_ACL_GRAPH = True   # switch to ACL path
sys.modules["pto_kernels"].USE_PTO_RMS_NORM = False  # disable entirely
```

## HuggingFace `auto_map`

```json
{
  "auto_map": {
    "AutoConfig": "pypto_gym/transformers/phi_3_mini_4k_instruct/configuration_phi3.Phi3Config",
    "AutoModelForCausalLM": "pypto_gym/transformers/phi_3_mini_4k_instruct/modeling_phi3.Phi3ForCausalLM"
  }
}
```

## File Table

| File | Description |
|------|-------------|
| `configuration_phi3.py` | `Phi3Config` — model configuration (3072 hidden, 32 layers, 32 heads, supports long rope scaling with short_factor/long_factor for extended context) |
| `modeling_phi3.py` | Full model graph (1569 lines) — `Phi3ForCausalLM`, `Phi3Model`, `Phi3RMSNorm` (dual-path dispatch at line 92-96), `Phi3Attention`, `Phi3MLP`, `Phi3RotaryEmbedding`, `Phi3SuScaledRotaryEmbedding` |

## Related Directories

- **Ops**: `src/pypto_gym/ops/pypto_tile/phi_3_mini_4k_instruct/` — PyPTO kernels (`rms_norm/`)
- **Tests**: `tests/ops/phi_3_mini_4k_instruct/` and `tests/model_ops/phi_3_mini_4k_instruct/` — correctness and performance tests
