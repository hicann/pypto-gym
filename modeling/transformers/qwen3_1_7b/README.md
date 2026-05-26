# Qwen3-1.7B NPU 迁移说明

| 字段 | 说明 |
|------|------|
| HuggingFace | Qwen/Qwen3-1.7B |
| 权重目录 | /mnt/workspace/gitCode/cann/models/Qwen3-1.7B |
| 代码来源 | transformers包 (v5.6.1) |
| transformers版本 | 5.6.1 |
| 代码位置 | core/modeling_qwen3.py, core/configuration_qwen3.py |
| 修改内容 | 导入方式修复（相对导入→绝对导入），添加 auto_map 到 config.json |
| 运行命令 | `python3 scripts/ask_Qwen3-1.7B.py --prompt "你好"` |
