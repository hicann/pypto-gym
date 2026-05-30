# Phi-3-mini-4k-instruct NPU 迁移说明

## 模型信息

| 字段 | 值 |
|------|-----|
| HuggingFace | microsoft/Phi-3-mini-4k-instruct |
| 权重目录 | /npu/s00454010/models/Phi-3-mini-4k-instruct |
| 代码来源 | trust_remote_code (HuggingFace仓库自带) |
| 参数量 | ~3.8B |
| 精度 | FP16 |

## 环境信息

| 字段 | 值 |
|------|-----|
| NPU | Ascend 910B2 |
| torch | 2.7.1 |
| torch_npu | 2.7.1 |
| transformers | 4.57.1 |

## 兼容性修复

为适配 transformers 4.57.x，对 modeling_phi3.py 进行了以下修复：

1. `past_key_values.seen_tokens` → `past_key_values.get_seq_length()` (line 1291)
2. `past_key_values.get_max_length()` → `past_key_values.get_max_cache_shape()` (line 1292)
3. `past_key_values.get_usable_length(kv_seq_len, self.layer_idx)` → `past_key_values.get_seq_length(self.layer_idx)` (lines 333, 451, 741)
4. `past_key_values.get_usable_length(seq_length)` → `past_key_values.get_seq_length()` (line 1063)

## 运行命令

```bash
# 基础推理
python3 scripts/ask_Phi-3-mini-4k-instruct.py --prompt "你好" --device 4

# 从文件读取
python3 scripts/ask_Phi-3-mini-4k-instruct.py --sentence_file scripts/sample_inputs.txt --device 4

# 自定义输出长度
python3 scripts/ask_Phi-3-mini-4k-instruct.py --prompt "你好" --output_length 100 --device 4

# 基准测试
bash scripts/bench_Phi-3-mini-4k-instruct.sh
```
