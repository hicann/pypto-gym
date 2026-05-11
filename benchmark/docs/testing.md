# 测试说明

benchmark 的测试入口统一为 pytest：

```bash
python -m pytest benchmark/tests
```

不要新增 shell 测试入口，也不要把内部调试命令写成公开测试方式。

## 覆盖范围

当前 pytest 集合主要覆盖无需 NPU/LLM 的基础契约：

- 公开 CLI 只包含 `python -m benchmark run --config [--foreground] [--no-auto-monitor] …`、
  `python -m benchmark monitor <state_dir>` 和
  `python -m benchmark summary <report_dir>`。
- benchmark 目录中保留 `benchmark/scripts/download_pypto.sh` 和 quick start
  脚本作为公开 shell 入口；KernelBench 数据集使用仓内内置目录。
- YAML 配置可从 benchmark 本地 `configs/` 解析。
- monitor 状态目录逻辑可离线冒烟。
- 机械层反作弊 fixture 和 opencode transcript 导出容错。

完整 benchmark 业务流程依赖 NPU、CANN、`torch_npu`、opencode 和 LLM 配置，
不作为 pytest-only 契约的一部分。

## 新增测试要求

- 新增 benchmark 逻辑时，优先补充 `benchmark/tests` 下的 pytest 用例。
- 新增 KernelBench case loader 行为时，应断言生成的 `SPEC.md` front matter
  和正文关键字段。
- 测试中如需调用公开 benchmark CLI，`run` 使用 `--config configs/xxx.yaml`
  （按用例需要可加 `--foreground` 或 `--no-auto-monitor`），`monitor` 使用状态目录参数，
  `summary` 使用报告目录参数。
- 不要在文档或测试中引入新的 Python CLI 入口、shell runner，或除 `run --foreground`
  / `run --no-auto-monitor` 以外的额外 benchmark CLI 开关。
