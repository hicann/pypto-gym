# Skills

PyPTO-Gym 仓库内置的 AI Agent Skills，用于自动化完成 HuggingFace 大语言模型到昇腾 NPU 的迁移、PyPTO 融合算子整网集成、以及模型格式转换等工作流。

## 本仓库 Skill

| Skill | 版本 | 用途 | 触发词 |
|-------|------|------|--------|
| [`pypto-fused-op-integration`](./pypto-fused-op-integration/SKILL.md) | v2.4 | PyPTO 融合算子整网集成：NPU 迁移 → 基线验证 → 打点采 Golden → 算子开发 → 模型集成 → 端到端验证 → 性能分析 → 归档 | 算子融合、整网集成、replace small ops、模型算子替换、NPU 迁移、昇腾、Ascend |
| [`pypto-convert-model`](./pypto-convert-model/SKILL.md) | v1.0 | PyTorch / ONNX / safetensors 三向自动转换 + round-trip 数值校验 | 模型转换、转 onnx、convert model、port.py |

## 其他 Skills

本仓库 skill 引用的其他 skill（如 `pypto-environment-setup`、`pypto-golden-generate`、`pypto-op-develop` 等）不在本仓库中，位于 PyPTO 主仓库：

```
https://gitcode.com/cann/pypto/tree/master/.agents/skills
```

获取方式：克隆或浏览 [`cann/pypto`](https://gitcode.com/cann/pypto)，skill 文件在 `.agents/skills/` 目录下，每个 skill 对应一个子目录，子目录内含 `SKILL.md`。

## Skill 文件结构

```
.agents/skills/
├── README.md
├── pypto-convert-model/
│   ├── SKILL.md
│   ├── requirements.txt
│   ├── scripts/
│   └── references/
└── pypto-fused-op-integration/
    ├── SKILL.md
    ├── references/
    └── scripts/
```
