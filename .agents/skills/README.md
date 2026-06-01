# Skills

PyPTO-Gym 仓库内置的 AI Agent Skills，用于自动化完成模型迁移、算子集成、benchmark 用例转换、算子校验和模型格式转换等工作流。

## 本仓库 Skills

| Skill | 版本 | 用途 | 触发词 |
|-------|------|------|--------|
| [`pypto-fused-op-integration`](./pypto-fused-op-integration/SKILL.md) | v2.4 | PyPTO 融合算子整网集成：打点、Golden、开发、集成、验证 | 算子融合、整网集成、replace small ops、模型算子替换 |
| [`pypto-kernel-validate`](./pypto-kernel-validate/SKILL.md) | v1.0 | PyPTO 算子产物校验：反作弊、精度验证、性能验证 | kernel validate、算子校验、反作弊、benchmark verifier |
| [`pypto-testcase-to-benchmark`](./pypto-testcase-to-benchmark/SKILL.md) | - | 将 PyPTO testcase 接入 pypto-gym benchmark 测试框架 | testcase、benchmark、benchmark case |
| [`pypto-convert-model`](./pypto-convert-model/SKILL.md) | v1.0 | PyTorch / ONNX / safetensors 三向自动转换 + round-trip 数值校验 | 模型转换、转 onnx、convert model、port.py、format convert |

## 工作流关系

```
pypto-fused-op-integration          (融合算子整网集成)
    │ 产出：算子实现 + 集成代码 + 验证报告
    ▼
pypto-kernel-validate               (算子产物校验)
    │ 产出：skill_report.json
    ▼
pypto-gym benchmark                 (统一 correctness + performance 评测)
```

`pypto-testcase-to-benchmark` 用于把已有 testcase 转成 benchmark 框架用例，可独立于上述整网集成工作流使用。

`pypto-convert-model` 用于在 PyTorch / ONNX / safetensors 三种格式间做双向转换并以 dummy forward 比较 max_abs 差，独立工作流。常用于 modeling 调试、checkpoint 格式迁移、ONNX 出口验证等场景。
`pypto-convert-model` provides any-to-any conversion between PyTorch / ONNX / safetensors and verifies round-trip via a dummy-forward `max_abs` diff. Independent workflow; common use cases: modeling debug, checkpoint format migration, ONNX export verification.

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
├── pypto-fused-op-integration/
│   ├── SKILL.md
│   ├── references/
│   └── scripts/
├── pypto-kernel-validate/
│   └── SKILL.md
├── pypto-testcase-to-benchmark/
│   ├── SKILL.md
│   ├── references/
│   └── scripts/
└── pypto-convert-model/
    ├── SKILL.md
    ├── requirements.txt
    ├── scripts/
    └── references/
        ├── README.md
        ├── HANDOFF.md
        ├── matrix.md
        └── results/
```
