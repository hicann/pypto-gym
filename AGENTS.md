# AGENTS.md

## 项目概述

本仓库为 **PyPTO-Gym** —— 基于 PyPTO 编程框架（面向华为昇腾 AI 处理器的 Tile 编程模型）构建的样例仓，原为 [cann/pypto](https://gitcode.com/cann/pypto) 仓的 `models` 目录拆分而来，与主仓解耦独立演进。此处的 agent 与 skill 面向**融合算子开发、大模型适配与算子训练场的全流程自动化**。

### 核心功能

- 大模型融合算子开发与整网集成（9-agent 团队编排）
- Golden 生成、精度对比与精度问题排查
- 算子性能分析与自动调优
- HuggingFace 模型到昇腾 NPU 的端到端迁移与模型格式转换

---

## 通用原则

> **严格遵循以下原则**

1. **编码前思考** —— 不要假设，不要隐藏困惑，呈现权衡
   - 明确说明假设——如果不确定，询问而不是猜测
   - 存在歧义时呈现多种解释，不要默默选择
   - 如果存在更简单的方法，适时提出异议
   - 困惑时停下来，指出不清楚的地方并要求澄清
2. **简洁优先** —— 用最少的代码解决问题，不要过度推测
   - 不要添加要求之外的功能；不要为一次性代码创建抽象
   - 不要添加未要求的"灵活性"或"可配置性"；不要为不可能发生的场景做错误处理
   - 检验标准：资深工程师会觉得这过于复杂吗？如果是，简化
3. **精准修改** —— 只碰必须碰的，只清理自己造成的混乱
   - 不要"改进"相邻的代码、注释或格式；不要重构没坏的东西；匹配现有风格
   - 如果注意到无关的死代码，提一下，不要删除它
   - 只删除因自己改动而变得无用的导入/变量/函数，不删除预先存在的死代码
   - 检验标准：每一行修改都应该能直接追溯到用户的请求
4. **目标驱动执行** —— 定义成功标准，循环验证直到达成
   - "添加验证"→"为无效输入编写测试，然后让它们通过"；"修复 bug"→"编写重现 bug 的测试，然后让它通过"
   - 多步骤任务先说明简短计划（每步 → 验证）
5. **先验证，再下结论** —— 结论必须有证据支撑，不能靠猜
   - 代码、文档、日志能查证的，不要凭记忆或直觉判断
   - 明确区分已确认的事实与你的推测，不要把推测当结论说
   - 检验标准：结论要能说清依据来源，经得起追问

---

## 入口路径速查

| 你想做什么 | 第一站 |
|---|---|
| 开发/迁移一个新算子（全流程） | skill `pypto-orchestration-manual`（9-agent 团队入口，驱动 `pypto-op-orchestrator`） |
| 生成 golden / 排查精度问题 | skill `pypto-golden-generate` / `pypto-precision-compare` / `pypto-precision-debug` |
| 分析并调优算子性能 | skill `pypto-op-perf-tune` |
| 查 API / 文档 / golden 用法 | skill `pypto-api-explore` / `pypto-docs-search` |
| 把 HF 模型端到端跑通 NPU | skill `hf-npu-e2e-workflow` / `pypto-fused-op-integration` |
| 模型格式互转（PyTorch/ONNX/safetensors） | skill `pypto-convert-model` |
| 校验算子产物（反作弊+精度+性能，KernelBench 把关） | skill `pypto-kernel-validate`（agent `pypto-kernel-validator`） |
| 完整 skill 索引 | [`cannbot-skills/README.md`](cannbot-skills/README.md) |
