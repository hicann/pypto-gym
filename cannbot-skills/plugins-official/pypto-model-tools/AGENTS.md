---
name: pypto-model-tools
description: "PyPTO 模型迁移、融合算子整网集成与模型格式转换 primary agent。按任务选择一个模型 skill，不进入算子状态机。"
mode: primary
skills:
  - hf-npu-e2e-workflow
  - pypto-convert-model
  - pypto-fused-op-integration
tools:
  read: true
  write: true
  edit: true
  bash: true
---

# PyPTO 模型工具约定

本插件是独立 skill 集，不参与 PyPTO 或 PyPTO-Pro 的算子状态机。每次
任务只加载一个主 skill。需要融合算子文件时，用户应明确指定来源目录。

`init.sh` 同时安装 8 个算子支撑 skill（`pypto-intent-understand`、
`pypto-api-explore`、`pypto-golden-generate`、`pypto-op-design`、
`pypto-op-develop`、`pypto-precision-compare`、`pypto-precision-debug`、
`pypto-op-perf-tune`），这些 skill 仅由 `pypto-fused-op-integration`
在算子开发阶段按需调用，不参与 primary 选择，不引入状态机或 Subagent。
