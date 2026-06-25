# Gemma-4-31B-it NPU Migration Notes

| Field | Value |
|-------|-------|
| HuggingFace | google/gemma-4-31b-it |
| Architecture | Gemma4ForConditionalGeneration (VLM, text-only inference supported) |
| Inference Script | `python3 ask_Gemma-4-31B-it.py` |
| transformers Version | 4.52+ |
| PyPTO Kernels | `src/pypto_gym/ops/pypto_tile/gemma4_31b_it/` |
| Kernel Tests | `tests/ops/gemma4_31b_it/` |

## Architecture

| Parameter | Value |
|-----------|-------|
| hidden_size | 5376 |
| intermediate_size | 21504 |
| num_attention_heads | 32 |
| num_key_value_heads | 16 (GQA ratio=2) |
| head_dim | 256 (global: 512) |
| num_hidden_layers | 60 |
| sliding_window | 1024 |
| Layer pattern | 5 sliding + 1 full |
| dtype | bfloat16 |

## PyPTO Kernels

| Kernel | Description | Status |
|--------|-------------|--------|
| K1: Attention SoftMax | 3-pass tiled softmax, S_TILE=64 | Implemented |
| K2: GQA Decode Attn | GQA with KV head averaging (16->4), S2_TILE=64, 75% KV bandwidth reduction | Implemented |

## Usage

```bash
export MODEL_PATH=/path/to/gemma-4-31b-it
export PYTHONPATH=$PWD/../../../src:$PYTHONPATH

python3 ../download_hf_model.py \
    --model-id google/gemma-4-31b-it \
    --output-dir "$MODEL_PATH"

python3 ../runtime_patch.py \
    --model-family gemma4_31b_it \
    --model-path "$MODEL_PATH"

# Baseline (no PyPTO)
python3 ask_Gemma-4-31B-it.py --model-path "$MODEL_PATH"

# With PyPTO fused kernels
python3 ask_Gemma-4-31B-it.py --model-path "$MODEL_PATH" --use_pypto

# Benchmark
DEVICE=0 MODEL_PATH="$MODEL_PATH" bash bench_Gemma-4-31B-it.sh
```

## Kernel Unit Tests

```bash
cd tests/ops/gemma4_31b_it
python3 test_attn_softmax.py
python3 test_gqa_decode_attn.py
```

## Hardware Requirements

- Ascend 910B NPU (~64 GB HBM)
- CANN 8.5.0+
- torch_npu, pypto
