# GutenOCR-3B 迁移说明

## 基本信息

| 项目 | 值 |
|------|------|
| HuggingFace | [rootsautomation/GutenOCR-3B](https://huggingface.co/rootsautomation/GutenOCR-3B) |
| 权重目录 | /mnt/workspace/gitCode/cann/models/GutenOCR-3B |
| 代码来源 | transformers包（内置，model_type=qwen2_5_vl） |
| transformers版本 | 5.6.0 |
| 运行命令 | `python3 scripts/ask_GutenOCR-3B.py --device 0` |

## 目录结构

```
GutenOCR-3B/
├── config.json
├── model*.safetensors
├── model.safetensors.index.json
├── tokenizer.json / tokenizer_config.json
├── vocab.json / merges.txt
├── generation_config.json
└── scripts/
    ├── ask_GutenOCR-3B.py
    ├── bench_GutenOCR-3B.sh
    ├── sample_inputs.txt
    └── README.md
```

## 使用方法

```bash
# 基本用法
python3 scripts/ask_GutenOCR-3B.py --device 0

# 自定义问题
python3 scripts/ask_GutenOCR-3B.py --device 0 --prompt "介绍一下Python语言"

# 指定模型路径
python3 scripts/ask_GutenOCR-3B.py --device 0 --model-path /custom/path
```
