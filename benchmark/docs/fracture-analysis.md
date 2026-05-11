# 断裂点综合分析

本文档说明如何对一次 benchmark 运行结果执行 PyPTO 断裂点综合分析。该分析不是
`python -m benchmark run` 的内置阶段，而是 benchmark 结束后的报告后处理流程。

## 适用场景

当需要回答以下问题时，使用断裂点综合分析：

- 哪些算子失败是 PyPTO 框架、文档、样例或 agent 流程问题导致的。
- 多个算子是否反复遇到同一类阻塞。
- 哪些问题应该优先修复，以提升后续 benchmark 成功率。

分析对象是一次 benchmark 运行生成的 `report/` 目录，例如：

```text
benchmark_runs/Task_xxx/report/
├── summary.json
├── summary.md
└── level2/<op>/
    ├── pypto_run.log
    ├── pypto_sessions/attempt_XX/root_full.md
    ├── pypto_sessions/attempt_XX/nodes/*.md
    ├── verifier.log
    ├── verifier_session.md
    └── result.json
```

断裂点分析优先读取每个算子的 `pypto_run.log`。若存在
`pypto_run.attempt2.log` 等重试日志，也会作为独立 attempt 分析。

## 使用方式

在 agent 会话中请求：

```text
分析这次 benchmark 的断裂点，report 路径是 benchmark_runs/Task_xxx/report
```

触发的 skill 是 `pypto-benchmark-fracture-aggregator`。它会扫描 `report/`
下除汇总产物外的各 level 子目录，例如 `level1` / `level2` / `level3` /
`level4` / `pto_case`，逐个生成单算子断裂点报告，再汇总全局报告。

## 输出目录

断裂点分析结果写入：

```text
<report-dir>/fracture-points/
├── README.md
├── benchmark_fracture_report.md
├── <level>_<OpName>_attempt1.md
├── <level>_<OpName>_attempt2.md
└── ...
```

主要产物：

- `benchmark_fracture_report.md`：全局综合报告，包含共性断裂点、统计分布和修复建议。
- `README.md`：报告索引和快速概览。
- `<level>_<OpName>_attemptN.md`：单算子单 attempt 的断裂点报告。

若某份日志没有检测到断裂点，也应生成对应单算子报告，断裂点总数为 0。

## 聚合分类

综合报告的所有统计只计入 Type-1 断裂点。单算子报告中的每个断裂点必须标注
`聚合分类` 和 `分类依据`：

| 分类 | 含义 | 统计处理 |
| --- | --- | --- |
| Type-1 | PyPTO 算子开发流程中 agent 碰到的框架、代码、文档、样例或 agent 流程问题 | 计入综合报告统计 |
| Type-2 | benchmark 框架配置、调度、日志记录、超时策略等非 PyPTO 产出问题 | 记录但不计入 Type-1 统计 |
| Type-3 | 硬件故障、集群问题、用户中断等外部因素 | 记录但不计入 Type-1 统计 |

如果旧报告缺少聚合分类，汇总时需要回到单算子报告和原始日志补判；无法补判时，
该断裂点不计入 Type-1 统计，并在综合报告的过滤说明中列出。

## 与常规 benchmark 报告的关系

`summary.json` 和 `summary.md` 是 benchmark 原生运行结果，描述每个 case 的生成、
验证和性能状态。

`fracture-points/` 是面向 PyPTO agent 和框架改进的后处理分析，不改变
`summary.json`、`summary.md` 或单 case `result.json`。
