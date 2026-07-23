# Conversion Matrix

Test device (auto-detected): **npu**

| model | onnx | pt | safetensors |
|---|---|---|---|
| resnet18 | L-FAIL | L-FAIL | L-FAIL |
| mobilenet_v2 | L-FAIL | L-FAIL | L-FAIL |
| efficientnet_b0 | L-FAIL | L-FAIL | L-FAIL |
| mlp_mixer | L-FAIL | L-FAIL | L-FAIL |
| gmlp | L-FAIL | L-FAIL | L-FAIL |
| resmlp | L-FAIL | L-FAIL | L-FAIL |
| switch_base_8 | L-FAIL | L-FAIL | L-FAIL |
| qwen_moe | L-FAIL | L-FAIL | L-FAIL |
| toy_moe | C-FAIL | OK<br/>max_abs=0.00e+00 | OK<br/>max_abs=0.00e+00 |
| toy_soft_moe | C-FAIL | OK<br/>max_abs=0.00e+00 | OK<br/>max_abs=0.00e+00 |
| toy_switch_moe | C-FAIL | OK<br/>max_abs=0.00e+00 | OK<br/>max_abs=0.00e+00 |

Legend: OK=pass, DIFF=converted but output differs, SKIP=not applicable, C-FAIL=convert failed, T-FAIL=test failed, T-SKIP=test skipped, L-FAIL=load failed.