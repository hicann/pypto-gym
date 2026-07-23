---
name: pypto-pro-op-perf-tune
description: PyPTO-pro 算子性能采集与分析，融合 msprof 算子级瓶颈定位与 kernel-level 对比测试，用于采集 PyPTO 算子性能数据、对比 PyPTO 算子 vs golden 标杆加速比、定位性能瓶颈。当用户在 PyPTO 算子开发过程中提到"上板性能"、"算子性能测试"、"硬件性能验证"、"NPU性能采集"、"NPU profiling"、"性能对比"、"加速比"等场景时触发。
---

# PyPTO-pro 上板性能采集与分析

在真实 NPU 上采集 PyPTO 算子性能数据，系统化解读指标文件，判定性能是否达标，定位瓶颈类型。本技能单独服务于 pypto-pro 框架，为 Stage 7 性能调优提供独立、公正的性能数据采集能力。

本技能基于 **msprof** 工具链，统一入口为两个脚本：
- **`msprof_profile_run.sh`** — 性能采集（标准采集 / 对比测试 / 快速采集 / 批量并行）
- **`msprof_perf_summary.py`** — 结果解析（瓶颈分析 / 对比报告 / 批量汇总）

| 工具 | 流程文档 | 用途 |
|------|----------|------|
| **`msprof_profile_run.sh`** | [`references/msprof-guide.md`](references/msprof-guide.md) | 统一采集入口：标准采集、对比测试、快速采集、批量并行 |
| **`msprof_perf_summary.py`** | [`references/msprof-guide.md`](references/msprof-guide.md) | 统一解析入口：瓶颈分析、对比报告生成、批量汇总 |
| **`perf_summary.py`** | [`references/msprof-op-guide.md`](references/msprof-op-guide.md) | msprof op 归档（需 msopprof） |

---

## 使用方式

### 1. 标准采集模式（深度瓶颈分析）

对 pypto-pro 测试脚本采集 7 组 aic-metrics + sample-based：

```bash
# Step 1: 采集（7 组 aic-metrics + sample-based，warm-up=3）
bash scripts/msprof_profile_run.sh --warm-up=3 --output=./msprof_output -- \
    python3 test_{op}.py

# Step 2: 获取目标 kernel 的 Op Name
#    PyPTO 测试脚本含多个 case + torch.randn，op_summary 中有大量非目标 op。
#    必须指定 --op-name 才能选到正确的 kernel 行。
CSV=$(ls ./msprof_output/PROF_GROUP_*/PROF_PipeUtilization/*/mindstudio_profiler_output/op_summary_*.csv | head -1)
grep -i {kernel_func} "$CSV"   # {kernel_func} = @pl.jit 装饰的函数名

# Step 3: 解析（--op-name = Op Name 列的值，如 _Z14rmsnorm_kernelPfS_iiii）
python3 scripts/msprof_perf_summary.py ./msprof_output/PROF_GROUP_* <ops_dir> --op-name=<Op Name>
```

> **重要**：PyPTO 测试脚本在一次运行中执行多个 case + `torch.randn` 噪声生成，`op_summary_*.csv` 中会包含大量非目标 op（如 `StatelessNormal`）。不指定 `--op-name` 时，解析脚本会选到 Task Duration 最大的非目标 op，导致结果完全错误。必须通过 `--op-name` 指定目标 kernel 的 Op Name（mangled name，可从 Step 2 获取）。

### 2. 对比测试模式（kernel-level 加速比）

从 `GOLDEN_PERF_REPORT.md` 读取 golden kernel duration，用 msprof 采集 PyPTO kernel Task Duration，计算加速比。

行为说明：

- `--warm-up=N` 会在正式采集前真实执行 N 次，默认为 3。
- `--repeats=N` 在对比和快速模式中都生效，默认为 1。N 大于等于 3 时去掉最大值和最小值后取平均；N 小于 3 时取中位数。
- `--seed=N` 通过 `PYPTO_PERF_SEED` 传入测试脚本，同时设置 `PYTHONHASHSEED`。测试脚本应读取 `PYPTO_PERF_SEED` 来固定随机输入。
- msprof 临时数据写入算子目录下的 `.msprof/`，并使用 PID 和时间戳隔离并发任务。

```bash
bash scripts/msprof_profile_run.sh --compare \
    --output-dir=./custom/{op} --warm-up=3 \
    --device=0 --op-name=<Op Name> --repeats=3 --seed=0
```

> **前置条件**：`custom/{op}/GOLDEN_PERF_REPORT.md` 必须已存在（由 `pypto-pro-golden-generate` skill 在 Stage 2 产出）。
>
> **必须指定 `--op-name`**：与标准采集模式同理，PyPTO 测试脚本含多个 case + `torch.randn`，不指定会选到非目标 op。Op Name 获取方法见标准采集模式 Step 2。
>
> **逐 case 限制**：当前没有标准的逐 case 选择协议。如果 `GOLDEN_PERF_REPORT.md` 包含多个 case，工具会明确报错，不会把一次聚合耗时复用到所有 case。请按 case 分别产出报告后再对比。

输出：`performance.json`、`performance.log`、`perf_report.md`。

### 3. 快速模式（不采集 aic-metrics）

只采集 kernel Task Duration，不采集 7 个 aic-metrics，适合快速验证优化效果。`--warm-up`、`--repeats`、`--seed` 和逐 case 限制与对比模式一致。

```bash
bash scripts/msprof_profile_run.sh --quick \
    --output-dir=./custom/{op} --warm-up=3 \
    --device=0 --op-name=<Op Name> --repeats=3 --seed=0
```

输出：`performance.json`、`performance.log`、`perf_report.md`。

### 4. 批量并行模式（多 NPU）

扫描目录下所有算子子目录，多 NPU 并行执行对比测试：

```bash
bash scripts/msprof_profile_run.sh --batch --base-dir=./custom --max-jobs=7 --device-start=1
```

输出：各子目录 `performance.json` + `batch_performance.log` + `batch_report.md` + `batch_summary.json`

---

## 输入目录结构

**pypto-pro 测试文件结构（单文件，参考 `$PYPTO_DEVKIT_DIR/pro_ops/`）：**
```
custom/{op}/
├── test_{op}.py                        # 测试脚本
│   ├── @pl.jit 装饰的 kernel 函数      # PyPTO 实现
│   ├── run_perf_test() / test_{op}()   # 测试函数，内含：
│   │   ├── torch.randn(...) 生成输入    # 直接硬编码 shape
│   │   ├── kernel[None, num_cores](…)  # 调用 kernel
│   │   ├── torch.xxx() 计算 golden      # PyTorch 原生 API
│   │   └── torch.testing.assert_close   # 精度验证
│   └── if __name__ == "__main__": run_perf_test()
├── {op}_golden.py                      # golden 参考实现（pypto-pro-golden-generate 产出）
├── GOLDEN_PERF_REPORT.md               # golden 性能报告（profile_golden.py 产出）
└── SPEC.md                             # 算子规格
```

---

## 输出格式（对比模式）

Markdown 报告包含：
- **对比表**：`Case | Shape | DType | PyPTO算子(us) | 标杆(us) | 加速比`
- **全量汇总**：用例数、平均加速比、PyPTO算子/标杆更优条数
- **按数据类型汇总**：分 dtype 的统计
- **简短分析**：整体趋势结论
- **深度瓶颈分析入口**：提供 msprof 深度分析命令

额外输出：
- `performance.json` — 结构化数据（含 geomean/mean/median/min/max 加速比）
- `performance.log` — 打屏日志

---

## 深度分析：msprof / msprof op

当性能不达标或需要根因分析时，使用深度分析流程。

### 选用哪个工具：决策树

1. **用户显式指定工具**
   - 指定 `msprof op` / msopprof → 加载 [`references/msprof-op-guide.md`](references/msprof-op-guide.md)
   - 指定 `msprof` → 加载 [`references/msprof-guide.md`](references/msprof-guide.md)

2. **用户未指定** — 探测环境：
   - 仅 `msopprof` 可用 → [`references/msprof-op-guide.md`](references/msprof-op-guide.md)
   - 仅 `msprof` 可用 → [`references/msprof-guide.md`](references/msprof-guide.md)
   - 两者皆可用 → 须向用户确认或按项目约定选用其一
   - 两者皆不可用 → 报错，提示检查 CANN / `ASCEND_HOME` 安装

### 参考资源

| 文件 | 内容 | 何时查阅 |
|------|------|---------|
| [`references/msprof-op-guide.md`](references/msprof-op-guide.md) | `msprof op`：构建 / 采集 / 归档 / 判定 / 瓶颈 / 回归 | 选用 `msprof op` 时 |
| [`references/msprof-guide.md`](references/msprof-guide.md) | `msprof`：构建 / 采集 / 归档 / **主 Bound 判定** / 瓶颈 | 选用 `msprof` 时 |
| [`references/csv_fields_reference.md`](references/csv_fields_reference.md) | CSV 字段定义与阈值 | 理解指标含义时 |

---

## 适用场景总览

| 场景 | 推荐命令 | 说明 |
|------|----------|------|
| PyPTO 算子性能采集 | `msprof_profile_run.sh` + `msprof_perf_summary.py` | 标准采集 7 组 aic-metrics + 深度瓶颈分析 |
| PyPTO 算子 vs golden 标杆对比 | `msprof_profile_run.sh --compare` | 从 GOLDEN_PERF_REPORT.md 读 golden 数据 + msprof 采集 PyPTO 算子 → 加速比 |
| 性能问题定位 | `msprof_profile_run.sh` + `msprof_perf_summary.py` | 深度瓶颈分析（pipe ratio / 带宽 / L2Cache / 逐核负载均衡） |
| 优化效果验证 | `msprof_profile_run.sh` + `msprof_perf_summary.py` | 对比优化前后的归档数据（round_NNN/） |
| PyPTO 编排器 Stage 7 基线采集 | `msprof_profile_run.sh` | 集成到 pypto-op-orchestrator Stage 7 性能调优流程 |
| 批量性能测试 | `msprof_profile_run.sh --batch` | 多 NPU 并行批量测试 |
