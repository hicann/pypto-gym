# PyPTO-Gym Transformers Inference

End-to-end inference examples for the Qwen3-1.7B model accelerated with PyPTO fused kernels on Ascend NPU.

## Supported Models

| Model | Model ID / Path | Features |
|-------|----------|----------|
| Qwen3-1.7B | `/data/z00885570/models/Qwen3-1.7B` | RoPE, SwiGLU, RMSNorm, GQA, PyPTO K1/K2/K3 fused kernels |

## Quick Start

### Basic Inference

```bash
conda activate pypto2
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa

# PyTorch baseline
python infer.py \
    --model_id /data/z00885570/models/Qwen3-1.7B \
    --device 0 \
    --show_outputs

# With PyPTO fused kernels
python infer.py \
    --model_id /data/z00885570/models/Qwen3-1.7B \
    --use_pypto \
    --device 0 \
    --show_outputs
```

### Using Custom Inputs

```bash
# From file
python infer.py \
    --model_id /data/z00885570/models/Qwen3-1.7B \
    --use_pypto \
    --sentence_file sample_inputs/input_prompt_32K.txt \
    --output_length 100

# From command line
python infer.py \
    --model_id /data/z00885570/models/Qwen3-1.7B \
    --use_pypto \
    --input_text "Explain machine learning" \
    --show_outputs
```

### Manual Decode Loop (per-token timing)

```bash
python infer.py \
    --model_id /data/z00885570/models/Qwen3-1.7B \
    --use_pypto \
    --step \
    --max_new 32
```

## Performance Benchmark

Run benchmark scripts for automated comparison:

```bash
# Qwen3-1.7B benchmark (baseline vs PyPTO)
./bench_qwen3_1_7b.sh
```

### Manual Benchmark

```bash
# PyTorch baseline
python infer.py \
    --model_id /data/z00885570/models/Qwen3-1.7B \
    --device 0 \
    --sentence_file sample_inputs/input_prompt_small.txt \
    --output_length 100 \
    --num_runs 5

# PyPTO fused kernel backend
python infer.py \
    --model_id /data/z00885570/models/Qwen3-1.7B \
    --use_pypto \
    --device 0 \
    --sentence_file sample_inputs/input_prompt_small.txt \
    --output_length 100 \
    --num_runs 5
```

## Docker Support

```bash
# Build the container
./build_docker.sh

# Or build manually (must run from repository root)
docker build -t pypto-gym-transformers -f modeling/transformers/Dockerfile .

# Run interactively
docker run --device=/dev/davinci0 -it pypto-gym-transformers bash
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model_id` | Model path (HuggingFace ID or local path) | (required) |
| `--device` | NPU device ID | `0` |
| `--use_pypto` | Enable PyPTO fused kernel optimization | `False` |
| `--input_text` | Input prompt text | - |
| `--sentence_file` | Input file path | - |
| `--output_length` | Number of tokens to generate | `100` |
| `--batch_size` | Batch size | `1` |
| `--precision` | `bfloat16` or `float32` | `bfloat16` |
| `--num_runs` | Benchmark iterations | `5` |
| `--warmup_runs` | Warmup iterations | `2` |
| `--show_outputs` | Print generated text | `False` |
| `--step` | Manual decode loop with per-token timing | `False` |
| `--summary_file` | File to append summary lines | - |
| `--mock_input_len` | Mock input length for throughput testing | `0` |

## Using Local Models

You can use locally cached models by specifying the path directly:

```bash
python infer.py \
    --model_id /path/to/Qwen3-1.7B \
    --use_pypto \
    --device 0 \
    --show_outputs
```

## Troubleshooting

**NPU Out of Memory**
- The 1.7B model requires ~4GB of HBM. Reduce `--batch_size` if OOM occurs
- Use `--precision bfloat16` (default)

**Import Errors**
- Ensure Ascend toolkit is sourced: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- Ensure `PTO_TILE_LIB_CODE_PATH` is set

**Slow Performance**
- First run includes PyPTO JIT compilation overhead. Use `--warmup_runs 2` for benchmarking
