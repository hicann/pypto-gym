# Gemma-4-31B-it PyPTO Kernels

Custom PyPTO fused kernels for `google/gemma-4-31b-it` on Ascend NPU.

## Kernel List

| Directory | Description | Input -> Output | Replaces |
|-----------|-------------|-----------------|----------|
| `attn_softmax/` | 3-pass tiled attention softmax | scores [M, S] bf16 -> [M, S] bf16 | `F.softmax(scores * scale, dim=-1)` |
| `gqa_decode_attn/` | GQA decode attention with KV head averaging (16->4, 75% bandwidth reduction) | q [32,256], kv [16,Skv,256] bf16 -> [32,256] bf16 | Eager GQA w/ KV replication |

## Architecture Parameters

| Parameter | Value |
|-----------|-------|
| hidden_size | 5376 |
| intermediate_size | 21504 |
| num_attention_heads | 32 |
| num_key_value_heads | 16 |
| head_dim | 256 (global: 512) |
| num_hidden_layers | 60 |
| sliding_window | 1024 |

## Unit Tests

Tests are located in `tests/ops/gemma4_31b_it/`:

```bash
cd tests/ops/gemma4_31b_it
python3 test_attn_softmax.py
python3 test_gqa_decode_attn.py
```
