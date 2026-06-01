# Convert-Model Experiment

PyTorch is the source of truth. Each model is converted to **`onnx`**, **`pt`** (TorchScript), and **`safetensors`**, then the converted artifact is reloaded and its forward output is compared against the original PyTorch output on the same dummy input.

## Models

| id | category | source | repo | size |
|---|---|---|---|---|
| resnet18 | CNN | HuggingFace | microsoft/resnet-18 | ~45MB |
| mobilenet_v2 | CNN | HuggingFace | google/mobilenet_v2_1.0_224 | ~14MB |
| efficientnet_b0 | CNN | timm | efficientnet_b0.ra_in1k | ~21MB |
| mlp_mixer | Transformer-MLP | timm | mixer_b16_224.goog_in21k_ft_in1k | ~240MB |
| gmlp | Transformer-MLP | timm | gmlp_s16_224.ra3_in1k | ~80MB |
| resmlp | Transformer-MLP | timm | resmlp_12_224.fb_in1k | ~60MB |
| switch_base_8 | MoE | HuggingFace | google/switch-base-8 | ~300MB |
| qwen_moe | MoE | HuggingFace | Qwen/Qwen1.5-MoE-A2.7B | ~14GB |
| toy_topk_moe / toy_soft_moe / toy_switch_moe | MoE | local | tiny MoE definitions | <1MB |

## Target formats

`onnx`, `pt` (TorchScript), `safetensors`.

## Environment notes

- Test phase auto-detects an accelerator (NPU > CUDA/ROCm > XPU > MPS > CPU).
  Force one with `CONVERT_EXP_DEVICE=npu|cuda|xpu|mps|cpu`.
- ONNX export pins `dynamo=False` to keep the legacy tracing exporter, so
  data-dependent control flow (e.g. MoE routing) works.
- Float precision: ONNX runs float32, so expect max abs diff ~1e-6 to 1e-4
  vs the PyTorch reference.

## Layout

```
convert_experiment/
├── README.md             # this file
├── HANDOFF.md            # previous-run notes
├── matrix.md             # generated PASS/FAIL matrix
├── logs/                 # master + per-conversion logs
├── models/               # HF / timm download cache
├── outputs/<model>/<fmt>/
├── results/<model>.json
└── scripts/
    ├── registry.py       # the model list
    ├── toy_moe.py        # local MoE definitions
    ├── loaders.py        # load PyTorch model + sample input
    ├── converters.py     # pytorch -> {onnx, pt, safetensors}
    ├── compare.py        # runtime evaluation + diff
    ├── device_util.py    # accelerator auto-detection
    ├── log_util.py       # logging helpers
    ├── run.py            # batch runner over the registry
    └── port.py           # CLI: any-to-any porter (onnx | pt | safetensors)
```

## How to run

### Batch experiment (the matrix)

```bash
source .venv/bin/activate
python scripts/run.py                       # all models in registry
python scripts/run.py mobilenet_v2 toy_moe  # subset
```

### Any-to-any porter

```bash
# pt -> onnx, no extra info needed (TorchScript is self-contained)
python scripts/port.py model.pt out.onnx --input-shape 1,3,224,224

# safetensors -> onnx via an HF repo for the architecture
python scripts/port.py model.safetensors out.onnx --hf-repo google/mobilenet_v2_1.0_224

# onnx -> pt (uses onnx2torch under the hood)
python scripts/port.py model.onnx out.pt --input-shape 1,3,224,224
```

Each port runs the source artifact, runs the converted artifact, and
prints a `max_abs` diff so you immediately know if the round-trip is
numerically safe.
