# Getting Started

本文档说明如何在 `pypto-gym` 仓库中运行 KernelBench x PyPTO benchmark。

## 1. 准备环境

先准备可用的 `pypto-gym` benchmark 环境，包括：

- NPU / CANN / `torch_npu` 环境可用。
- `opencode` 和模型配置可用。
- 当前目录位于 `pypto-gym` 仓库根目录。
- PyPTO 源码仓通过 benchmark 脚本独立下载，用于代码生成阶段。

下载或更新 PyPTO master 源码仓：

```bash
bash benchmark/scripts/download_pypto.sh
```

默认下载到：

```text
benchmark/.cache/pypto/
```

## 2. 下载 KernelBench 数据集

```bash
bash benchmark/scripts/download_kernelbench.sh
```

默认会下载到：

```text
benchmark/.cache/KernelBench/
```

benchmark 运行时默认读取：

```text
benchmark/.cache/KernelBench/KernelBench/
```

## 3. 准备本地 YAML 配置

当前 benchmark 只通过 YAML 配置运行。建议复制默认配置：

```bash
cp benchmark/configs/relu.yaml benchmark/configs/local.yaml
```

按需修改 `benchmark/configs/local.yaml`。未填写字段会从
`benchmark/configs/__default__.yaml` 继承（其中 `verifier.mode` 默认为
`performance`，即精度与性能一并验证），例如：

```yaml
bench_dir: ""
cases: "level1=19_ReLU"
devices: [0]
concurrency: 1

output:
  root_dir: "/tmp/pypto_gym_benchmark"
  base_dir: "benchmark_runs"

pypto:
  repo_root: ""
  opencode_model: "alibaba-cn/glm-5"
  timeout_sec: 10800
  skip_pypto_gen: false
  force_regen: false

verifier:
  mode: "performance"
  verifier_mode: "opencode"
  verify_rtol: 1.0e-2
  verify_atol: 2.5e-2

monitor:
  poll_sec: 3
```

`pypto.repo_root` 留空时使用 `benchmark/.cache/pypto/`。解析后的仓根必须存在、为目录，且包含 **`.opencode/`**（用于判定为完整 PyPTO 检出）；默认路径缺失或不完整时，错误信息会提示运行 `bash benchmark/scripts/download_pypto.sh`。若显式配置自定义路径，同样必须满足上述布局。

Stage 7 按 PyPTO 原生工作流执行，benchmark 不提供跳过 Stage 7 的配置项。

`cases` 必须显式写明 level，即使只跑单个 level：

```text
level1=19_ReLU
level1=19
level1=1:21,31,41:50
level1=1,3,10,18,19;level2=9,14,29;level3=43,46;pto_case=1:6
```

其中 `pto_case` 是 PyPTO 自维护 case 在 KernelBench fork 里的 level 名。

## 4. 运行 benchmark

```bash
python -m benchmark run --config configs/local.yaml
```

**默认（后台）**：当前进程先完成预检并与子进程对齐本次产物根目录（若未固定
`output.root_dir`，会生成 `base_dir/Task_<uuid>`），再 `fork` 子进程跑
`run_from_config`；子进程 stdout/stderr 写入 `<root_dir>/logs/benchmark.out` 与
`benchmark.err`。父进程等待 `state/state.json` 出现（最多约 10s）后 **自动进入**
`monitor` TUI。不支持 `os.fork` 的平台会报错并提示使用 `--foreground`。

需要像以前一样在当前终端看完整日志、阻塞到批次结束：

```bash
python -m benchmark run --config configs/local.yaml --foreground
```

若默认模式下等待 `state.json` 超时，终端会输出子进程状态、`benchmark.err` /
`benchmark.out` 尾部摘要，并提示用 `--foreground` 重放；详见 `docs/monitor.md`。

## 5. 查看实时进度

默认 `run` 已自动打开 monitor。若需第二个终端只看板、或 monitor 退出后要重连：

```bash
python -m benchmark monitor /tmp/pypto_gym_benchmark/state
```

（具体路径以本次运行产物根下的 `state/` 为准；后台模式结束时也会打印重连命令。）

`monitor` 会读取 `state/state.json`，展示每个 case 的 PyPTO 阶段、Verifier 阶段、PID、耗时和状态。

## 6. 查看产物

一次运行的产物统一写到 `output.root_dir`，例如：

```text
/tmp/pypto_gym_benchmark/
├── logs/
│   ├── benchmark.out    # 默认后台 run 时子进程 stdout
│   └── benchmark.err    # 默认后台 run 时子进程 stderr
├── report/
│   ├── summary.json
│   ├── summary.md
│   └── <level>/<op>/
│       ├── pypto_run.log
│       ├── pypto_session.md
│       ├── verifier.log
│       ├── verifier_session.md
│       ├── custom/<op>/     # PyPTO custom 产物副本，已排除 output*
│       └── result.json
└── state/
    └── state.json
```

常用产物：

- `report/summary.md`：本批次最终报告。
- `report/<level>/<op>/pypto_run.log`：PyPTO 算子开发日志。
- `report/<level>/<op>/verifier.log`：gym 侧验证日志。
- `report/<level>/<op>/pypto_session.md`：PyPTO 开发阶段完整会话。
- `report/<level>/<op>/verifier_session.md`：验证阶段完整会话。
- `report/<level>/<op>/custom/<op>/`：对应 case 的 PyPTO custom 产物副本，复制时会排除 `output*` 路径以控制体积。
- `benchmark/.cache/pypto/custom/<op>/`：PyPTO 生成的算子开发产物。

若需要分析 PyPTO agent 在批量运行中的共性卡点，可在 benchmark 结束后对
`report/` 执行断裂点综合分析。分析产物写入
`report/fracture-points/`，包括全局报告 `benchmark_fracture_report.md` 和单算子
断裂点报告。详细流程见 `docs/fracture-analysis.md`。

## 7. 运行精选 case 集合

基础 case 跑通后，建议继续运行 `benchmark/configs/pypto.yaml`。该配置包含一批经过筛选的 level1/level2/level3 case，用于更接近批量 benchmark 的真实压力。

```bash
python -m benchmark run --config configs/pypto.yaml
```

运行前按机器资源检查并调整 `benchmark/configs/pypto.yaml` 中的 `devices`、`concurrency` 和 `pypto.timeout_sec`。多 case 批量运行耗时较长，默认 `run` 已拉起 monitor；若使用 `--foreground`，可自行另开终端执行 `python -m benchmark monitor <state>` 观察进度。

## 8. 增加自定义 case

自定义 case 不放在 `pypto-gym` 或 PyPTO 源码仓中。新增 case 应提交到 PyPTO 维护的 KernelBench fork，并放在：

```text
KernelBench/pto_case/<N>_<Name>.py
```

case 必须满足 KernelBench 标准格式：

```python
class Model(nn.Module):
    ...

def get_inputs():
    return [...]

def get_init_inputs():
    return [...]
```

提交到 KernelBench fork 后，重新下载数据集，再通过 YAML 指定：

```yaml
cases: "pto_case=1"
```

然后运行：

```bash
python -m benchmark run --config configs/local.yaml
```
