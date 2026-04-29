# Skills

PyPTO-Gym 仓库内置的 AI Agent Skills，用于自动化完成模型迁移、算子集成等复杂工作流。

## 本仓库 Skills

| Skill | 版本 | 用途 | 触发词 |
|-------|------|------|--------|
| [`migrate-huggingface-to-npu`](./migrate-huggingface-to-npu/SKILL.md) | v1.3 | HuggingFace大语言模型迁移到华为昇腾NPU环境运行 | NPU、昇腾、Ascend、torch-npu、迁移、推理部署 |
| [`pypto-fused-op-integration`](./pypto-fused-op-integration/SKILL.md) | v2.4 | PyPTO融合算子整网集成：打点→Golden→开发→集成→验证 | 算子融合、整网集成、replace small ops、模型算子替换 |

## 工作流关系

```
migrate-huggingface-to-npu          (上游：模型迁移到NPU)
    │ 产出：ask脚本 + core代码 + 权重目录 + .git
    ▼
pypto-fused-op-integration          (下游：融合算子替换)
    │ 复用上游产出，进行算子融合→集成→验证
    ▼
  [更多下游skills...]
```

## 其他 Skills

以上两个skill引用的其他skill（如 `pypto-environment-setup`、`pypto-golden-generate`、`pypto-op-develop` 等）不在本仓库中，**均位于 PyPTO 主仓库**：

```
https://gitcode.com/cann/pypto/tree/master/.agents/skills
```

获取方式：克隆或浏览 [`cann/pypto`](https://gitcode.com/cann/pypto)，skill文件在 `.agents/skills/` 目录下，每个skill对应一个子目录，子目录内含 `SKILL.md`。

## Skill 文件结构

```
.claude/skills/
├── README.md                                  # 本文件
├── migrate-huggingface-to-npu/
│   ├── SKILL.md                               # skill定义
│   └── scripts/                               # 辅助脚本（generate_ask_script.py等）
└── pypto-fused-op-integration/
    ├── SKILL.md                                # skill定义
    └── references/                             # 参考模板
```
