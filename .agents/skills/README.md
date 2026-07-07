# Skills

PyPTO-Gym 仓库内置的 AI Agent Skills，用于自动化完成 HuggingFace 大语言模型到昇腾 NPU 的迁移、PyPTO 融合算子整网集成、以及模型格式转换等工作流。

## 本仓库 Skill

### 大模型入网适配

| Skill | 版本 | 用途 | 触发词 |
|-------|------|------|--------|
| [`pypto-fused-op-integration`](./pypto-fused-op-integration/SKILL.md) | v2.4 | PyPTO 融合算子整网集成：NPU 迁移 → 基线验证 → 打点采 Golden → 算子开发 → 模型集成 → 端到端验证 → 性能分析 → 归档 | 算子融合、整网集成、replace small ops、模型算子替换、NPU 迁移、昇腾、Ascend |
| [`hf-npu-e2e-workflow`](./hf-npu-e2e-workflow/SKILL.md) | — | 编排 HF 模型从 model-id 到昇腾 NPU 实测 E2E 吞吐：下载 → runtime patch → 运行 → 测速，按需调用 `pypto-fused-op-integration` | run HF model on NPU end to end、NPU E2E workflow、prefill/decode tok/s、NPUGraph 捕获测速 |
| [`pypto-convert-model`](./pypto-convert-model/SKILL.md) | v1.0 | PyTorch / ONNX / safetensors 三向自动转换 + round-trip 数值校验 | 模型转换、转 onnx、convert model、port.py |

### 算子开发全流程（9-Agent 团队）

| Skill | 版本 | 用途 | 触发词 |
|-------|------|------|--------|
| [`pypto-orchestration-manual`](./pypto-orchestration-manual/SKILL.md) | — | `pypto-op-orchestrator` 入口，汇总团队原则 / 成员名册 / 强制规则三份控制文档 | 算子开发编排、9-agent 团队 |
| [`pypto-intent-understand`](./pypto-intent-understand/SKILL.md) | — | 将自然语言算子描述转化为结构化需求文档 | 开发/实现/创建某个算子 |
| [`pypto-op-plan`](./pypto-op-plan/SKILL.md) | — | 需求规划：结构相似样例搜索与可行性评估 | 需求规划 |
| [`pypto-op-design`](./pypto-op-design/SKILL.md) | — | 迭代式约束收敛生成 DESIGN.md（API 映射 / 精度路由 / Tiling 推导 / Loop 结构） | 生成设计方案、算子设计、Tiling 策略 |
| [`pypto-op-construct`](./pypto-op-construct/SKILL.md) | — | 模块语义拆解与逐模块构建、校验、golden 清单交叉核对 | 模块拆解、模块构建 |
| [`pypto-op-develop`](./pypto-op-develop/SKILL.md) | — | Impl 编码手册：按 Phase 累积构建 impl，验证通过后整理为最终实现 + README | 实现算子、写 kernel、算子编码 |
| [`pypto-op-verify`](./pypto-op-verify/SKILL.md) | — | 验证 runner 要求、`detailed_tensor_compare` 用法、通过标准与产物结构 | 验证算子实现 |
| [`pypto-op-review`](./pypto-op-review/SKILL.md) | — | 逐算子 PyPTO 调用提取，用于调试 `custom/<op>/` kernel | 调试 kernel 调用 |
| [`pypto-op-perf-tune`](./pypto-op-perf-tune/SKILL.md) | — | 算子性能分析与自动调优：执行/精度校验 → 数据采集 → 分步调优 → 报告生成 | 算子性能调优、泳道图分析 |
| [`pypto-op-knowledge`](./pypto-op-knowledge/SKILL.md) | — | 算子开发经验表 / 问题查找表串行查询 | 查经验表、查问题表 |
| [`pypto-memory-template`](./pypto-memory-template/SKILL.md) | — | 算子级 `custom/<op>/MEMORY.md` 模板：必需章节、机器可读字段、更新节奏 | MEMORY.md 模板 |

### 精度与调试

| Skill | 版本 | 用途 | 触发词 |
|-------|------|------|--------|
| [`pypto-golden-generate`](./pypto-golden-generate/SKILL.md) | — | 基于算子规格生成 torch + torch_npu golden 参考实现 `{op}_golden.py` | 生成 golden、写 golden 函数、reference implementation |
| [`pypto-precision-compare`](./pypto-precision-compare/SKILL.md) | — | 精度对比：文件保存法（`pass_verify_save`）与二分对比法（检查点 tensor） | 调试精度、定位精度差异 |
| [`pypto-precision-debug`](./pypto-precision-debug/SKILL.md) | — | 精度问题排查：用户代码层语法逻辑检查与规避尝试 | 精度验证失败、数值偏差 |
| [`pypto-general-debug`](./pypto-general-debug/SKILL.md) | — | 卡住/不明失败的调试路由，经 `DEBUG_GUIDEBOOK.md` 指向对应主题参考或子 skill | tile-shape/L0/L1/alignment 问题 |

### 资料检索

| Skill | 版本 | 用途 | 触发词 |
|-------|------|------|--------|
| [`pypto-api-explore`](./pypto-api-explore/SKILL.md) | — | 探索 PyPTO API：API 映射、约束检查、Tiling 需求分析 | API 探索、支持什么 dtype、tiling 怎么配 |
| [`pypto-docs-search`](./pypto-docs-search/SKILL.md) | — | 检索 API 文档、错误码排障、教程/安装文档、算子参考实现与 golden | 查文档、按错误码排障、找参考实现 |

### 代码规范

| Skill | 版本 | 用途 | 触发词 |
|-------|------|------|--------|
| [`pypto-static-check-repire`](./pypto-static-check-repire/SKILL.md) | — | 依据门禁检测报表批量定位并修复 Python 静态规范问题，同步记录修改日志 | 静态检测不通过、门禁修复 |

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
# 1. 探测 src/pypto_gym/ops/pypto_tensor/qwen3_1_7b/ 中的算子
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

本仓库 skill 引用的少数 skill（如 `pypto-environment-setup`、`pypto-pr-creator`、`pypto-issue-creator`）是 PyPTO 主仓维护流程专属的能力，不在本仓库中，位于 PyPTO 主仓库：

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
