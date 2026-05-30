# Qwen3-1.7B NPU 迁移说明

| 字段 | 说明 |
|------|------|
| HuggingFace | Qwen/Qwen3-1.7B |
| 权重目录 | /data/h00520348/optimize0524/models/Qwen3-1.7B |
| 代码来源 | transformers包（内置实现） |
| transformers版本 | 4.57.6（复制代码时） |
| 代码位置 | core/modeling_qwen3.py, core/configuration_qwen3.py |
| 修改内容 | 从transformers.models.qwen3复制，导入方式已修复，auto_map已添加到config.json |
| 运行命令 | `python3 scripts/ask_Qwen3-1.7B.py --prompt "你好"` |


## 运行命令

```bash
# 基线推理
python3 scripts/ask_Qwen3-1.7B.py --device 1

# 基准测试
bash scripts/bench_Qwen3-1.7B.sh

# 性能采集
bash scripts/prof_Qwen3-1.7B.sh

# RoPE算子基准测试
python3 scripts/benchmark_pto_rope.py
```

## 目录结构

```
Qwen3-1.7B/
├── core/                          # 模型核心代码
│   ├── modeling_qwen3.py          # 模型实现
│   └── configuration_qwen3.py     # 配置实现
├── scripts/                       # 脚本目录
│   ├── ask_Qwen3-1.7B.py          # 基线推理脚本
│   ├── bench_Qwen3-1.7B.sh        # 基准测试脚本
│   ├── prof_Qwen3-1.7B.sh         # 性能采集脚本
│   └── benchmark_pto_rope.py      # RoPE算子基准测试
├── config.json                    # 模型配置（含auto_map）
├── model-*.safetensors            # 模型权重
└── tokenizer相关文件              # 分词器配置
```
