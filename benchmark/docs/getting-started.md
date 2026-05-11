# Getting Started

本文档说明如何在 `pypto-gym` 仓库中运行 KernelBench x PyPTO benchmark。

## 1. 准备环境

先准备可用的 `pypto-gym` benchmark 环境，包括：

- NPU / CANN / `torch_npu` 环境可用。
- `opencode` 和模型配置可用。
- 当前目录位于 `pypto-gym` 仓库根目录。
- PyPTO 源码仓通过 benchmark 脚本独立下载，用于代码生成阶段。
- KernelBench 完整 case 集已内置在 `benchmark/KernelBench/`。

下载或更新 PyPTO master 源码仓：

```bash
bash benchmark/scripts/download_pypto.sh
```

默认下载到：

```text
benchmark/.cache/pypto/
```

## 2. 准备本地 YAML 配置

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
# 可选；省略时与 __default__ 一致为 normal（静态 devices 列表）.
# device_mode: pool

output:
  root_dir: "/tmp/pypto_gym_benchmark"
  base_dir: "benchmark_runs"

pypto:
  repo_root: ""
  opencode_model: "alibaba-cn/glm-5"
  timeout_sec: 10800
  pref_round: 3
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

Stage 7 按 PyPTO 原生工作流执行；可通过 `pypto.pref_round` 控制 initial prompt
中的性能优化轮次上限，默认 3。

`cases` 必须显式写明 level，即使只跑单个 level：

```text
level1=19_ReLU
level1=19
level1=1:40
level1=;level2=;level3=;level4=;pto_case=
```

内置 case 保留上游原始编号。`level=` 表示选择该 level 下全部 case。

## 3. 运行 benchmark

```bash
python -m benchmark run --config configs/local.yaml
```

**默认（后台）**：当前进程先完成预检并与子进程对齐本次产物根目录（若未固定
`output.root_dir`，会生成 `base_dir/Task_<uuid>`），再 `fork` 子进程跑
`run_from_config`；子进程 stdout/stderr 写入 `<root_dir>/logs/benchmark.out` 与
`benchmark.err`。父进程等待 `state/state.json` 出现（最多约 90s）后 **自动进入**
`monitor` TUI。不支持 `os.fork` 的平台会报错并提示使用 `--foreground`。

需要像以前一样在当前终端看完整日志、阻塞到批次结束：

```bash
python -m benchmark run --config configs/local.yaml --foreground
```

后台运行但不自动进入 monitor（例如 cron/自动化场景）：

```bash
python -m benchmark run --config configs/local.yaml --no-auto-monitor
```

若默认模式下等待 `state.json` 超时，终端会输出子进程状态、`benchmark.err` /
`benchmark.out` 尾部摘要，并提示用 `--foreground` 重放；详见 `docs/monitor.md`。

## 4. 查看实时进度

默认 `run` 已自动打开 monitor。若需第二个终端只看板、或 monitor 退出后要重连：

```bash
python -m benchmark monitor /tmp/pypto_gym_benchmark/state
```

（具体路径以本次运行产物根下的 `state/` 为准；后台模式结束时也会打印重连命令。）

`monitor` 会读取 `state/state.json`，展示每个 case 的 PyPTO 阶段、Verifier 阶段、PID、耗时和状态。

## 5. 查看产物

一次运行的产物统一写到 `output.root_dir`，例如：

```text
/tmp/pypto_gym_benchmark/
├── logs/
│   ├── benchmark.out    # 默认后台 run 时子进程 stdout
│   └── benchmark.err    # 默认后台 run 时子进程 stderr
├── custom/                # 与 report/ 同级，便于单独打包 report/
│   └── <level>/<op>/      # PyPTO custom 产物副本，已排除 output*
├── report/
│   ├── summary.json
│   ├── summary.md
│   └── <level>/<op>/
│       ├── pypto_run.log
│       ├── pypto_sessions/attempt_01/
│       │   ├── root_full.md
│       │   ├── root_full.json
│       │   ├── session_tree.tsv
│       │   └── nodes/*.md
│       ├── verifier.log
│       ├── verifier_session.md
│       └── result.json
└── state/
    └── state.json
```

常用产物：

- `report/summary.md`：本批次最终报告。
- `report/<level>/<op>/pypto_run.log`：PyPTO 算子开发日志。
- `report/<level>/<op>/verifier.log`：gym 侧验证日志。
- `report/<level>/<op>/pypto_sessions/attempt_XX/root_full.md`：PyPTO 开发阶段完整会话，含 subagent。
- `report/<level>/<op>/pypto_sessions/attempt_XX/nodes/*.md`：main/subagent 单独会话。
- `report/<level>/<op>/verifier_session.md`：验证阶段完整会话。
- `custom/<level>/<op>/`：对应 case 的 PyPTO custom 产物副本（与 `report/` 同级目录），复制时会排除 `output*` 路径以控制体积。
- `benchmark/.cache/pypto/custom/<op>/`：PyPTO 生成的算子开发产物。

如果只需要基于已有 `result.json` 重建汇总报告：

```bash
python -m benchmark summary /tmp/pypto_gym_benchmark/report
```

若需要分析 PyPTO agent 在批量运行中的共性卡点，可在 benchmark 结束后对
`report/` 执行断裂点综合分析。分析产物写入
`report/fracture-points/`，包括全局报告 `benchmark_fracture_report.md` 和单算子
断裂点报告。详细流程见 `docs/fracture-analysis.md`。

## 6. 运行 curated PyPTO case 集

基础 case 跑通后，建议继续运行 `benchmark/configs/pypto.yaml`。该配置默认选择
`benchmark/KernelBench/` 下维护的 curated case 集；如需跑完整内置 case 集，可显式
把 `cases` 改为 `level1=;level2=;level3=;level4=;pto_case=`。

```bash
python -m benchmark run --config configs/pypto.yaml
```

运行前按机器资源检查并调整 `benchmark/configs/pypto.yaml` 中的 `devices`、`concurrency` 和 `pypto.timeout_sec`。多 case 批量运行耗时较长，默认 `run` 已拉起 monitor；若使用 `--foreground`，可自行另开终端执行 `python -m benchmark monitor <state>` 观察进度。

## 7. 增加自定义 case

本仓已内置完整 KernelBench case 集。新增本地实验 case 时，可以直接放入
`benchmark/KernelBench/<level>/`，也可以准备一个符合 KernelBench 布局的外部目录，
并在 YAML 中通过 `bench_dir` 指向它，例如：

```text
/data/my-kernelbench/custom/<N>_<Name>.py
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

然后通过 YAML 指定：

```yaml
bench_dir: "/data/my-kernelbench"
cases: "custom=1"
```

然后运行：

```bash
python -m benchmark run --config configs/local.yaml
```

## 8. `device_mode=normal` 与 `pool`

| 模式 | 调度方式 | `devices` / `concurrency` |
| --- | --- | --- |
| `normal`（默认） | 按 case 索引对 `devices` 做静态轮询；`case_loader` 探针可在本机扫描可用设备（find-free）。 | 与历史行为一致。 |
| `pool` | 进程内 **asyncio 队列式设备池**：`prepare`（加载用例 + 探针）、`pypto`、`verifier` 三个阶段各自 **acquire → 使用 → release**；同一阶段同一时刻每张配置卡最多被一个 case 占用。`case_loader` 使用池内已分配卡号，**禁用 find-free**。 | `devices` 列出池内卡 ID；`concurrency` 仍为同时进入「执行槽」的 case 数，实际占卡由池与阶段串行化共同约束。 |

**外部独占假设**：`pool` 只保证 **本 benchmark 进程内** 通过池串行化避免多 case 争用同一调度阶段下的同一 `device_id`。不会在 OS 或集群层面对其它进程加锁；须由部署方保证 `devices` 中的卡在机器上不被其它作业抢占。

**预检 fail-fast（Ascend）**：`backend` 为 `ascend` 且 `device_mode=pool` 时，批次开始前对 `devices` 每张卡各跑一遍子进程 softmax quick test（`torch` + `torch_npu` + `pypto`）。任一卡失败则 **立即 `SystemExit`**，不进入 case 循环。可调顶层 `pool_preflight_timeout_sec`（默认 120）控制单卡探针超时。

不可用设备时，进程退出前会输出类似：

```text
pool 设备预检 (softmax quick test) 失败:
  - device_id='0': <错误摘要>
```

**PyPTO workflow 与 skill**：gym **不修改** PyPTO 仓内 SKILL/agent，也 **不在** `download_pypto.sh` 之后对检出打 patch。`pool` 模式下对「禁止 find-free / 换卡」的约束通过 **`TILE_FWK_DEVICE_ID` 环境变量** 与 **opencode 初始 prompt** 中的独占说明一并传达；无法强行改写黑盒 skill 内部逻辑。若模型仍违背 prompt，需走 PyPTO 上游侧修复或调整模型与运行策略。

**示例配置**（与 `relu.yaml` 同属最小单 case，仅开启 pool）：

```yaml
device_mode: pool
cases: "level1=19_ReLU"
devices: [0]
concurrency: 1
```

仓库已提供 `benchmark/configs/local_pool.yaml`。在有 NPU 的环境下可从仓库根目录执行：

```bash
python -m benchmark run --config configs/local_pool.yaml --foreground
```

**硬件 smoke 验收（有 NPU）**

1. 预检通过且进入 case 后，标准日志中应出现（`device_id` 以配置为准）：
   - `case start (device_mode=pool)`
   - `pool acquire phase=prepare device=…` / `pool release phase=prepare device=…`
   - 随后 PyPTO / verifier 阶段同理可出现 `phase=pypto`、`phase=verifier` 的 acquire/release 行。
2. 在 `report/<level>/<op>/pypto_run.log` 中可检索 **`device_mode=pool`** 或 **`【设备约束 -- device_mode=pool`**，确认初始 prompt 已附带独占约束摘录（runner 会在日志中写入摘录段落）。
3. 故意将 `devices` 设为机器上不存在的卡或占用损坏卡时，应在批次输出中看到以 **`pool 设备预检 (softmax quick test) 失败:`** 开头的报告并 **非零退出**，而非静默进入 case。
