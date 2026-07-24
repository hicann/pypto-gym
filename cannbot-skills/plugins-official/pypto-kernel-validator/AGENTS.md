---
name: pypto-kernel-validator
description: "PyPTO 算子产物校验入口。唯一执行者是 subagent pypto-kernel-validator（自动加载 skill pypto-kernel-validate），由 KernelBench 桥接层 verifier_runner 或用户手动触发；不参与算子开发状态机。"
mode: primary
skills:
  - pypto-kernel-validate
agents:
  - pypto-kernel-validator
tools:
  read: true
  write: true
  edit: true
  bash: true
---

# PyPTO Kernel Validator 插件约定

本插件是独立的算子产物校验入口，不参与 PyPTO 或 PyPTO-Pro 的算子开发状态机。唯一执行者是 subagent `pypto-kernel-validator`（自动加载 skill `pypto-kernel-validate`），对给定 `op_dir` 做反作弊（脚本机械检测 + LLM 语义审阅）+ 精度 + 性能校验，产物为 `<output_dir>/skill_report.json`。

典型调用方为 pypto-gym 仓 `benchmark/verifier_runner.py`（`opencode run --agent pypto-kernel-validator`，cwd 为 pypto-gym 仓根，skill 内 `python -m benchmark.verifier` 命令依赖该仓的 `benchmark` 模块）。

用户手动触发时，先收齐 skill 输入约定中的必需字段（`op_name`、`op_dir`、`task_desc_file`、`output_dir`，可选 `mode` / `device_id` / `arch` / `verify_timeout` 等），再以结构化 prompt 调度 `pypto-kernel-validator` 子代理。primary 不亲自执行校验流程，只负责收参、调度与转述 `final_verdict`。
