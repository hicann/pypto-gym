# 配置说明

`run` 通过 YAML 指定运行配置；可选 `--foreground` 在前台跑完全程，或用
`--no-auto-monitor` 后台运行但不自动进入 monitor TUI（详见
`docs/getting-started.md`）。`monitor` 只接收状态目录，`summary` 只接收报告目录：

```bash
python -m benchmark run --config configs/relu.yaml
python -m benchmark run --config configs/relu.yaml --foreground
python -m benchmark run --config configs/relu.yaml --no-auto-monitor
python -m benchmark monitor <root_dir>/state
python -m benchmark summary <root_dir>/report
```

需要调整 case、设备、并发、报告路径或 verifier 行为时，复制
`configs/relu.yaml` 为本地配置文件并修改 YAML 内容。未填写字段会从
`configs/__default__.yaml` 继承。除 `run` 的前后台/monitor 附着开关外，CLI 不提供
其他参数覆盖这些字段。

配置维护原则：`configs/__default__.yaml` 是默认策略的集中来源；内置示例配置只写
相对默认值的差异。本地或实验 YAML 可以按需覆盖任意字段，但不建议把与默认值完全
相同的字段重复粘贴到多个配置里，避免默认策略调整时产生漂移。

## 常用字段

| 字段 | 说明 |
| --- | --- |
| `bench_dir` | KernelBench 数据集目录；留空时使用内置 `benchmark/KernelBench/` |
| `cases` | case 选择器，必须写明 level，支持序号、闭区间、完整 stem |
| `limit` | 使用 `level=` 选择整个 level 时截取前 N 个 case；`null` 表示不截断 |
| `devices` | 用于 PyPTO 生成和验证的 NPU 设备列表 |
| `concurrency` | 同时运行的 case 数，通常不超过 `devices` 数量 |
| `device_mode` | 设备调度：`normal`（默认，`devices` 静态轮询，`case_loader` 可 find-free）或 `pool`（进程内设备池，分阶段 acquire/release，探针禁用 find-free；Ascend 下启动前 softmax 预检 fail-fast） |
| `pool_preflight_timeout_sec` | 可选顶层字段；`device_mode=pool` 且 `backend=ascend` 时，单卡预检子进程超时秒数，默认 `120` |
| `output.root_dir` | 本次运行的统一产物根目录；非空时直接使用 |
| `output.base_dir` | `output.root_dir` 为空时，在该目录下创建随机 `Task_<uuid>` 作为产物根 |
| `pypto.repo_root` | PyPTO 源码仓根；留空时使用 `benchmark/.cache/pypto/`。路径须存在且为目录，并含 **`.opencode/`**，否则 `_build_cfg` / 后台 `run` 预检会立即失败 |
| `pypto.opencode_model` | 传给 opencode 的模型名；留空时沿用 opencode 默认配置 |
| `pypto.timeout_sec` | 单 case PyPTO 工作流超时时间 |
| `pypto.pref_round` | 写入 opencode initial prompt 的 Stage 7 性能优化轮次上限；默认 3 |
| `pypto.incomplete_workflow_retry` | 状态机未完成且无 failed/blocked/cancelled 阶段时，最多自动重试次数 |
| `pypto.incomplete_workflow_retry_min_gap_sec` | 只有 OpenCode session tree 的 last update 到 PyPTO finished 空窗达到该阈值时，才消耗 incomplete retry；默认 1800 秒 |
| `pypto.skip_pypto_gen` | 是否复用已有 `custom/<op>/` 产物，只复跑 verifier |
| `pypto.force_regen` | 是否强制重跑 PyPTO 生成 |
| `verifier.mode` | 验证范围：`correctness` / `performance` / `full`；`configs/__default__.yaml` 默认为 `performance` |
| `verifier.verifier_mode` | `opencode` 或 `direct`；`direct` 仅用于离线开发调试 |
| `verifier.verify_rtol` / `verifier.verify_atol` | 精度比较阈值 |
| `verifier.keep_artifacts` | 是否保留 verifier 临时脚本和源码副本 |
| `monitor.poll_sec` | `run` 刷新 `state.json` 的间隔 |

## case 选择

`cases` 支持以下形式：

```text
level1=19_ReLU
level1=19
level1=1:40
level1=;level2=;level3=;level4=;pto_case=
level1=
```

即使只跑单个 level，也必须使用 `level=cases` 形式。`level1=` 表示选择
`level1` 下全部 case，并可通过 `limit` 截断。

## 输出位置

`run` 使用统一产物根目录。若配置 `output.root_dir`，直接使用该目录；
否则使用 `<output.base_dir>/Task_<uuid>`。不同用途的产物位于该目录的
固定子目录中：

```text
<root_dir>/
├── logs/
│   ├── benchmark.out      # 默认后台 run：子进程 stdout
│   ├── benchmark.err       # 默认后台 run：子进程 stderr
│   └── <verifier workdirs>
├── report/
│   ├── summary.json
│   ├── summary.md
│   └── <level>/<op>/result.json
└── state/
    └── state.json
```

单 case 常见产物包括：

- `<root_dir>/report/<level>/<op>/pypto_run.log`
- `<root_dir>/report/<level>/<op>/pypto_sessions/attempt_XX/root_full.md`
- `<root_dir>/report/<level>/<op>/pypto_sessions/attempt_XX/nodes/*.md`
- `<root_dir>/report/<level>/<op>/verifier.log`
- `<root_dir>/report/<level>/<op>/verifier_session.md`
- `<root_dir>/report/<level>/<op>/result.json`

## 常见调整

- 更新 PyPTO 源码仓：运行 `bash benchmark/scripts/download_pypto.sh`。
- 只复跑 verifier：设置 `pypto.skip_pypto_gen: true`。
- 调整精度阈值：设置 `verifier.verify_rtol` 和 `verifier.verify_atol`。
- 多 level 运行：在 `cases` 中使用 `level1=...;level2=...;pto_case=...` 形式。
- 指定本次产物根：设置 `output.root_dir`。
- 每次生成随机 session 目录：设置 `output.base_dir`，并保持 `output.root_dir` 为空。
- **设备池模式**：设置 `device_mode: pool` 并配置 `devices`；详见 `docs/getting-started.md` 第 8 节与 `configs/local_pool.yaml`。
