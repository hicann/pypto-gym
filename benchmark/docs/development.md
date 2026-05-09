# 内部开发说明

本文档记录 benchmark 的内部流程、产物布局和已知约束。对外入口和命令约束
以 `../README.md` 为准。

## 数据流

```text
KernelBench/<level>/{N}_{name}.py
  -> case_loader.load_case
  -> case_loader.write_spec
  -> pypto_runner.run_pypto_workflow
  -> verifier_runner.run_verifier
  -> report.write_summary
```

各阶段职责：

- `case_loader`: 读取 KernelBench case，生成 `CaseSpec` 和 `SPEC.md`。
- `pypto_runner`: 调用 PyPTO 7 阶段工作流生成算子产物。
- `verifier_runner`: 按 YAML 配置选择 opencode verifier 或 direct verifier。
- `report`: 汇总 JSON 和 Markdown 报告。
- `monitor`: 读取 `benchmark run` 写出的状态目录并渲染看板。

## 反作弊设计

反作弊分三层：

- 脚本机械层：`verifier/cheat_detector.py` 通过 AST 和字符串规则检查明显问题。
- LLM 语义层：opencode verifier 识别脚本难以覆盖的绕过、fallback 或 mock 行为。
- 运行时层：`verifier/pypto_adapter.py` 根据 runtime profile 汇总性能和运行诊断。

`verifier/` 是内部实现子包，不作为 benchmark 对外 Python CLI 入口公开。

## 产物布局

PyPTO 生成产物通常落在仓根 `custom/<op>/`：

```text
custom/<op>/
├── SPEC.md
├── API_REPORT.md
├── DESIGN.md
├── <op>_golden.py
├── <op>_impl.py
├── <op>_pypto_impl.py
├── test_<op>.py
└── README.md
```

报告目录通常包含：

```text
<report-dir>/
├── summary.json
├── summary.md
├── fracture-points/              # 可选：断裂点综合分析后处理产物
│   ├── README.md
│   ├── benchmark_fracture_report.md
│   └── <level>_<op>_attempt1.md
└── <level>/<op>/
    ├── pypto_run.log
    ├── pypto_sessions/
    │   └── attempt_XX/
    │       ├── root_full.md
    │       ├── root_full.json
    │       ├── session_tree.tsv
    │       └── nodes/*.md
    ├── verifier.log
    ├── verifier_session.md
    ├── custom/<op>/             # PyPTO custom 产物副本，排除 output*
    ├── skill_report.json
    ├── <op>_task_desc.py
    └── result.json
```

## 目录速览

```text
benchmark/
├── __main__.py                 # 唯一公开 Python CLI: run --config / monitor <dir>
├── case_loader.py              # KernelBench .py -> SPEC.md + task_desc
├── pypto_runner.py             # PyPTO 工作流调用
├── verifier_runner.py          # opencode skill / direct KernelVerifier 调度
├── verifier/                   # 内部验证实现
├── run_kernelbench.py          # 批处理调度实现
├── monitor.py                  # 状态看板渲染和状态聚合辅助
├── report.py                   # JSON / Markdown 汇总
├── configs/__default__.yaml    # 默认配置
├── configs/relu.yaml           # ReLU 示例配置
├── scripts/
│   ├── download_kernelbench.sh
│   └── download_pypto.sh
├── tests/                      # pytest-only 测试与测试 fixture
├── docs/
└── README.md
```

## 已知约束

- 完整 PyPTO 工作流会调用 LLM，单 case 可能耗时较长。
- 当前 benchmark 业务验证面向 NPU/Ascend 后端和 PyTorch 框架。
- `pypto.repo_root` 留空时默认使用 `benchmark/.cache/pypto/`；路径必须含 **`.opencode/`**
  （与 `_build_cfg` / 后台 fork 前预检一致）。运行前通过
  `bash benchmark/scripts/download_pypto.sh` 下载或更新 PyPTO master。
- 内部调试单算子时，`python benchmark/pypto_runner.py <Op> …` 的 `--repo-root` 会与
  `run` 使用同一套合法性检查（存在、目录、含 `.opencode/`）；默认缓存未初始化时会提示下载脚本。
- `bench_dir` 需要指向包含 `KernelBench/<level>/{N}_{name}.py` 的目录；
  PyPTO 自维护 case 的 level 名为 `pto_case`。
- 多卡并发依赖 `TILE_FWK_DEVICE_ID` 隔离，同一时刻每卡仅 1 个 case。

## 失败排查

- `level dir 不存在`: 先确认已下载 KernelBench 数据集，并检查 `bench_dir`
  是否指向包含目标 level 的目录，例如 `level1` / `level2` / `level3` /
  `pto_case`。
- `level_dir 下找不到任何 .py 用例`: 检查 `bench_dir` 是否指到
  `KernelBench/KernelBench/` 这一层。
- opencode 子进程超时：调大 YAML 中的 `pypto.timeout_sec` 或
  `verifier.skill_timeout_sec`，并检查 LLM / opencode 配置。
- PyPTO 产物缺失：查看对应 case 的 `pypto_run.log`。
- `skill_report.json` 未产出：查看对应 case 的 `verifier.log`。
- KernelVerifier 失败：查看对应 case 的 `verifier.log` 和 `result.json`。
- 断裂点综合报告统计异常：检查单算子报告是否包含 `聚合分类`
  和 `分类依据`；缺少分类的断裂点不应计入 Type-1 统计。
- 默认后台 `benchmark run` 若迟迟不出现 `state.json`：stderr 会给出子进程状态和
  `logs/benchmark.err` / `benchmark.out` 尾部；优先查看预检错误或子进程 traceback，
  必要时使用 `python -m benchmark run … --foreground`。
