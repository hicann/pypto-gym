# Skills

PyPTO-Gym 仓库内置的 AI Agent Skills，用于自动化完成 HuggingFace 大语言模型到昇腾 NPU 的迁移、PyPTO 融合算子整网集成、以及模型格式转换等工作流。

## 本仓库 Skill

| Skill | 版本 | 用途 | 触发词 |
|-------|------|------|--------|
| [`pypto-fused-op-integration`](./pypto-fused-op-integration/SKILL.md) | v2.4 | PyPTO 融合算子整网集成：NPU 迁移 → 基线验证 → 打点采 Golden → 算子开发 → 模型集成 → 端到端验证 → 性能分析 → 归档 | 算子融合、整网集成、replace small ops、模型算子替换、NPU 迁移、昇腾、Ascend |
| [`pypto-convert-model`](./pypto-convert-model/SKILL.md) | v1.0 | PyTorch / ONNX / safetensors 三向自动转换 + round-trip 数值校验 | 模型转换、转 onnx、convert model、port.py |

## 🤖 使用指南

可通过 AI Agent（Sisyphus）实现一键式大模型迁移与入网适配。

### 触发方式

在对话中直接描述需求，Agent 自动识别并调用 Skill：

```
"把 Qwen3-1.7B 迁移到 NPU 并集成 PyPTO 算子"
"将 HuggingFace 上的 microsoft/Phi-3-mini-4k-instruct 部署到昇腾"
"恢复 qwen3_1_7b 的 PyPTO 融合状态"
```

### 完整工作流

Agent 按照 Skill 定义的阶段自动编排：

```
阶段零: NPU 迁移与基线建立  →  模型下载 → 脚本生成 → 基线验证 → Git基线
阶段一: 前置准备            →  需求分析 + 环境验证 + 智能推荐融合点
阶段二: 理解验证            →  打点采集 + Golden编写 + 场景验证
阶段三: 算子开发            →  Benchmark 自动化 或 经典手动开发
阶段四: 模型集成            →  目录结构 + 适配层 + 调用逻辑 + 缓存处理
阶段五: 验证与提交          →  端到端验证 + 性能采集 + 归档 + 提交
```

### 示例 1：快速验证已有归档

```bash
# 对 Sisyphus 说：
"用 pypto-gym 仓归档的代码还原 Qwen3-1.7B 的 PyPTO 融合状态并验证"

# Agent 自动执行：
# 1. 探测 src/pypto_gym/ops/pypto_tile/qwen3_1_7b/ 中的算子
# 2. 探索 src/pypto_gym/transformers/qwen3_1_7b/ 中的模型修改
# 3. 运行 restore_model_patch.sh 完成入网适配
# 4. 验证 baseline 与 PyPTO 双模式均可推理
```

### 示例 2：从零迁移新模型

```bash
# 对 Sisyphus 说：
"把 https://huggingface.co/Qwen/Qwen3-1.7B 部署到 NPU 上"

# Agent 会：
# 1. 下载模型权重
# 2. 检测现有 torch/torch_npu 环境并处理版本冲突
# 3. 生成推理脚本并验证基线可运行
# 4. 分析网络结构推荐可融合算子
# 5. 按用户确认的路线完成算子开发与集成
```

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
