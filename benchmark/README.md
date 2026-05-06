# Benchmark

`benchmark` 用于批量运行 KernelBench case，并串联 PyPTO 算子生成、
验证和报告汇总流程。本文档是对外入口；配置、测试、monitor 和
KernelBench 数据集细节放在 `docs/` 下。

## 公开契约

- 公开 Python CLI 只有 `python -m benchmark`。
- `run` 子命令通过 `--config configs/xxx.yaml` 读取运行配置；可选 `--foreground`
  强制前台运行（默认后台 detached 并自动打开 monitor）。
- `monitor` 子命令只接受状态目录，不读取运行配置。
- benchmark 测试只通过 pytest 暴露。
- PyPTO 源码只通过 `benchmark/scripts/download_pypto.sh` 下载。
- KernelBench 数据集只通过 `benchmark/scripts/download_kernelbench.sh` 下载。
- 运行参数以 YAML 为主；`run` 仅额外支持 `--foreground`，不通过 CLI 覆盖其他配置项。

## 目录结构

```text
benchmark/
├── README.md                  # 对外入口
├── docs/                      # 分主题文档
├── configs/                   # YAML 配置
├── scripts/
│   ├── download_kernelbench.sh
│   └── download_pypto.sh
├── tests/                     # pytest-only 测试与测试 fixture
├── verifier/                  # 内部验证实现
├── __main__.py                # 唯一公开 Python CLI
├── run_kernelbench.py
├── monitor.py
└── report.py
```

## 前置条件

运行完整 benchmark 前需要准备 opencode、LLM、NPU/CANN/`torch_npu`
环境，并下载 PyPTO 源码仓与 KernelBench 数据集：

```bash
bash benchmark/scripts/download_pypto.sh
bash benchmark/scripts/download_kernelbench.sh
```

PyPTO 默认下载到 `benchmark/.cache/pypto/`；更多数据集说明见
`docs/kernelbench.md`。

## 运行方式

```bash
python -m benchmark run --config configs/relu.yaml
python -m benchmark run --config configs/relu.yaml --foreground
python -m benchmark monitor <root_dir>/state
```

默认情况下，`run` 在 **后台 detached** 执行批处理（需 Unix 上的 `os.fork`；
Windows 等不支持 fork 的环境须加 `--foreground`）。父进程在 fork 前会做与真实
运行一致的 **预检**（含 `pypto.repo_root`、KernelBench 路径与用例解析等），
子进程标准输出重定向到 `<root_dir>/logs/benchmark.out` / `benchmark.err`。
`state.json` 出现后会 **自动拉起** `monitor` TUI；退出后终端会打印重连命令。

`--foreground` 保留原先「当前终端阻塞跑完全程、日志直接打到 stdout/stderr」
的行为，便于 CI 或本地排障。

`monitor` 仍可单独启动，只读取传入状态目录并刷新看板。配置字段见
`docs/configuration.md`，monitor 与默认后台行为见 `docs/monitor.md`。

## 测试方式

```bash
python -m pytest benchmark/tests
```

pytest 集合覆盖无需 NPU/LLM 的 CLI 契约和基础逻辑。更多说明见
`docs/testing.md`。

## 更多文档

- `docs/getting-started.md`: 快速开始与端到端运行示例。
- `docs/architecture.md`: 架构边界、端到端流程和 Mermaid 流程图。
- `docs/configuration.md`: YAML 配置字段和输出目录。
- `docs/testing.md`: pytest-only 测试约束。
- `docs/monitor.md`: `monitor` 子命令与状态目录。
- `docs/kernelbench.md`: KernelBench 下载、布局和数据集约束。
- `docs/add-new-case.md`: 新增 KernelBench 风格 case 的规则。
- `docs/development.md`: 内部流程、产物布局和已知约束。
