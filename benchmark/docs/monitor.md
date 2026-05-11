# Monitor 子命令

## 默认 `run` 与手动 `monitor`

默认执行 `python -m benchmark run --config ...` 时，进程在 fork 子进程完成预检并
对齐产物根后，会等待 `state.json` 落盘，然后 **在同一终端自动进入** 本看板（TUI）。
若你在另一终端只读状态、或 TUI 退出后需要重连，再使用下面的 `monitor` 子命令。

独立调用方式固定为：

```bash
python -m benchmark monitor <root_dir>/state
```

若父进程在等待 `state.json` 时超时（约 90s），会根据子进程状态和
`<root_dir>/logs/benchmark.err` / `benchmark.out` 尾部打印诊断；典型修复包括修正
YAML、补齐 PyPTO 仓或改用 `--foreground` 查看完整 traceback。

需要后台运行但不自动进入 TUI 时，可使用：

```bash
python -m benchmark run --config configs/relu.yaml --no-auto-monitor
```

此时父进程仍会打印可手动执行的 `monitor_command`。

`monitor` 只读取传入目录下的 `state.json` 并渲染看板，不读取运行
YAML，也不负责启动 benchmark。状态路径为 `<本次产物根>/state/`，产物根由
YAML 的 `output.root_dir`（若为空则 `<output.base_dir>/Task_<uuid>`）解析得到。

## 配置字段

YAML 中只保留 run 写状态所需字段：

| 字段 | 说明 |
| --- | --- |
| `monitor.poll_sec` | `benchmark run` 刷新 `state.json` 的间隔 |

## 状态与报告

`benchmark run` 会在 `<root_dir>/state` 写监控状态，报告与日志分别写入
`<root_dir>/report` 和 `<root_dir>/logs`。典型状态目录如下：

```text
<root_dir>/state/
└── state.json
```

看板读取 `state.json` 中的主进程状态、算子阶段、PyPTO/Verifier PID、
耗时和结果状态。主进程进入 `已完成` 或 `未完成` 后，看板会停止刷新。
