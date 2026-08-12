---
name: pypto-pro-op-perf-tune
description: 采集、比较和分析 PyPTO-Pro 算子在 NPU 上的性能。当已有可运行的 PyPTO/PyPTO-Pro kernel，需要 kernel profiling、基准对比、瓶颈定位、优化前后回归或批量性能测试时使用。平台专属结论必须先探测目标与 profiler，并由当前 profiler 数据验证；不要用历史数字替代本机测量。
---

# PyPTO-pro 上板性能采集与分析

在真实 NPU 上采集 PyPTO 算子性能数据，系统化解读指标文件，判定性能是否达标，定位瓶颈类型。本技能为 **PyPTO-Pro Stage 4 的按需性能诊断**提供独立、公正的性能数据采集能力（wrapper 边界门禁的动态检查即由此采集 `op_times.device_kernels`）；PyPTO classic 的 Stage 7 调优同样可以调用它。

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

> **前置条件**：仅 `--compare` 模式需要 `custom/{op}/GOLDEN_PERF_REPORT.md`。Stage 2 默认不生成该报告；只有用户明确要求 golden 性能采集或 golden 基线对比时，才先以 `collect_golden_perf=true` 调用 `pypto-pro-golden-generate` 生成报告。若用户只要求 PyPTO kernel 自身的性能采集或瓶颈分析，使用标准模式，不要为了它自动采集 golden。
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
├── GOLDEN_PERF_REPORT.md               # 可选；--compare 前按用户要求由 profile_golden.py 产出
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

### 先确认目标平台

在加载任何平台专属模型前，先运行
[`../pypto-pro-environment-check/scripts/get_npu_arch.py`](../pypto-pro-environment-check/scripts/get_npu_arch.py)
或读取构建配置，记录设备型号、`NpuArch`、CANN/PyPTO 版本和
`pypto_pro.__file__`：

- 仅当结果确认 `NpuArch=3510` / `dav-c310` 时，加载
  [`references/a5-roofline-and-levers.md`](references/a5-roofline-and-levers.md)。
- 对未知平台或非 A5 平台，不应用 A5 核数、容量、带宽、频率或经验结论；
  直接使用该平台的官方资料与本次 profiler 数据。
- 平台探测失败时，将平台模型标记为不可用，但仍可继续做不依赖硬编码常量的
  实测瓶颈分析。

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
| [`references/a5-roofline-and-levers.md`](references/a5-roofline-and-levers.md) | A5 专属、证据门控的 roofline 工作流 | 已确认 `NpuArch=3510` / `dav-c310` 后 |
| [`references/msprof-op-guide.md`](references/msprof-op-guide.md) | `msprof op`：构建 / 采集 / 归档 / 判定 / 瓶颈 / 回归 | 选用 `msprof op` 时 |
| [`references/msprof-guide.md`](references/msprof-guide.md) | `msprof`：构建 / 采集 / 归档 / **主 Bound 判定** / 瓶颈 | 选用 `msprof` 时 |
| [`references/csv_fields_reference.md`](references/csv_fields_reference.md) | CSV 字段定义与阈值 | 理解指标含义时 |

> **性能分析顺序**：① 探测目标平台；② 若目标为 A5，按
> [`references/a5-roofline-and-levers.md`](references/a5-roofline-and-levers.md)
> 建立带来源的假设；③ 用 msprof 实测。非 A5 或未知平台跳过 A5 模型。

---

## 适用场景总览

| 场景 | 推荐命令 | 说明 |
|------|----------|------|
| PyPTO 算子性能采集 | `msprof_profile_run.sh` + `msprof_perf_summary.py` | 标准采集 7 组 aic-metrics + 深度瓶颈分析 |
| PyPTO 算子 vs golden 标杆对比 | `msprof_profile_run.sh --compare` | 从 GOLDEN_PERF_REPORT.md 读 golden 数据 + msprof 采集 PyPTO 算子 → 加速比 |
| 性能问题定位 | `msprof_profile_run.sh` + `msprof_perf_summary.py` | 深度瓶颈分析（pipe ratio / 带宽 / L2Cache / 逐核负载均衡） |
| 优化效果验证 | `msprof_profile_run.sh` + `msprof_perf_summary.py` | 对比优化前后的归档数据（round_NNN/） |
| PyPTO-Pro Stage 4 按需诊断 | `msprof_profile_run.sh` | wrapper 边界门禁取 `aclnn*` 占比；coder 定位瓶颈 |
| PyPTO classic Stage 7 基线采集 | `msprof_profile_run.sh` | 集成到 pypto-op-orchestrator Stage 7 调优流程 |
| 批量性能测试 | `msprof_profile_run.sh --batch` | 多 NPU 并行批量测试 |

## 优化第一顺位：先把 wrapper 清零

**测量口径**：cann-bench 计时覆盖整个提交的可调用对象。profile 里
`op_times.device_kernels` 中以 `aclnn` 开头的条目全部是 wrapper 开销，
以 `_Z` 开头的才是你的 kernel。

因此**在调 kernel 内部之前，先看 wrapper 占比**：

```
wrapper_share = sum(aclnn*) / sum(all device_kernels)
```

`wrapper_share ≥ 35%` 时，任何 tile size / double buffer / VF 改写的收益都
小于直接消除 wrapper。此时唯一正确的第一根杠杆是
`remove-wrapper-ops`：按 [`pypto-pro-op-kb/constraints/wrapper-boundary.md`](../../pypto-pro-op-kb/constraints/wrapper-boundary.md)
把 cast / transpose / slice / pad / concat 搬进 kernel。

两条经反复观察成立的判断，按此排杠杆顺序：

1. **wrapper 占比越过约三分之一时，消除 wrapper 的收益大于任何 kernel 内调优。**
   具体阈值随算子与平台变化，不要照搬——用本 skill 采集自己算子的 `wrapper_share` 再判断。

2. **同一个 kernel 可以变快而分数变低。** kernel 时间与算子分可以反向变动：
   wrapper 的增长会吃掉 kernel 的收益，只看 kernel 时间调优会得出完全相反的结论。
   唯一可用的口径是 wrapper + kernel 合计，即完整提交入口的耗时。

---

# 优化经验：代价表、高效写法、坑

**每条规则按「触发条件 → 规则 → 证据」写。** 触发条件是你能在**自己的 kernel 上直接判定**
的结构特征（tile 宽度、pitch 字节数、pass 数、是否跨 lane……），不需要了解任何具体算子。
算子名只出现在证据括号里，作为**溯源标签**——用来判断这个数字可不可信，不是用来做类比的。
这与 [`pypto-pro-op-kb/ROUTER.md`](../../pypto-pro-op-kb/ROUTER.md) 的路由原则一致：**按计算的形状分类，不按算子名分类，
算子名不能告诉新算子任何可复用的东西。**

## 一、跨 lane 是结构选择，不是调参项

与目标无关的规则：**跨 lane 原语（`vf.gather` / `vf.scatter` / UB 往返）彼此代价相近，
而它们与不跨 lane 的对齐访存、算术相差一到两个数量级。** ISA 不提供寄存器级 lane shift，
所有 UB 中转替代品交同样的税。所以设计阶段要问的不是「有没有更便宜的跨 lane 写法」，
而是**「要不要跨 lane」**。

两条推论（同样与目标无关）：

- **无 bank 冲突的 gather ≠ 便宜的 gather。** padding pitch 仍然必要，但消除冲突不会让它
  接近 `load_align`。
- 一个内层循环里若有少量 gather/scatter，即使两种写法 op 数完全相同，**UB 寻址仍可能占据
  绝大部分时间**，算术只占零头。

**具体到 ns 的代价表是 target 相关的实测值，不放在本页。** 确认目标为 A5
（`NpuArch=3510` / `dav-c310`）后，查
[`references/a5-roofline-and-levers.md`](references/a5-roofline-and-levers.md)
的「A5 原语代价表」一节取数。非 A5 或未知平台不要套用任何 ns 数值。

## 二、高效写法

### 1. mask 提升出寄存器循环

**触发条件**：内层对寄存器的循环里出现 `vf.update_mask`（或等价的逐寄存器 mask 计算），
且 tile 宽度 ≥ 64 lane。

**规则**：把满 64 lane 的 mask 提到循环外。
**收益 ≈（满寄存器承担的工作占比）×（每寄存器 mask 代价 ÷ body 代价）——两项，不是一项。**

**先判 tile 宽度，再判 op 数**：tile 宽度 < 64 lane 时满寄存器循环根本不执行，提升只剩记账开销。

| tile 形态 | body | 实测 |
|---|---|---|
| 128 个满寄存器 | 4 op | **3.20×** |
| 满寄存器为主 | 15 op | 1.86× |
| 满寄存器为主 | 30 op | 1.37× |
| **8 / 16 列（窄于一个寄存器）** | — | **+5%，其中 6 个 case 反而慢 3–4%** |

**tail mask 不总是可以丢。** tile 宽度是 64 整数倍时，`vl` 之后的 lane 算垃圾而 `pl.store`
不搬运——**但写回是 scatter 时方向相反**：失活 lane 读到垃圾*索引*，会写到 tile 内任意位置。
**判据是"失活 lane 的输出是否参与寻址"**：参与就必须保留真实 tail 迭代。
拿不准时提升满 mask 那条路径 + 保留一次 tail 迭代，任何情况下都安全，`nfull == 0` 时不花钱。

### 2. tail 循环放在行循环外面

**触发条件**：二维遍历（外层行、内层列），且列宽**可能整除** lane 数。

**规则**：给 tail 单独一遍对行的遍历，不要嵌在行循环内部。嵌套时宽度整除的形状每一行都要
付一次零次迭代的循环建立开销。

**证据**：同一次改动里，tail 为空的形状 **112.7 → 186.5 µs（劣化 1.65×）**，
而 tail 非空的形状仍改善 1.31%。拆成独立一遍后，同一个提升变成 **1.27× 收益**。

> **一个 case 变好不能证明结构对——必须跑一个 tail 为空的形状。**

### 3. 取最小合法 C：最大 strip、最少 pass

**触发条件**：输出需要多遍写回（`npass > 1`），即 tile 装不下一整行输出。

**规则**：`pl.store` 没有逐元素 mask，所以第一遍之后每一遍都要**整读 + 整写**输出——
代价对 `npass` **线性**；而加宽行带来的带宽收益**在 256 B 就饱和**。
**追求行长度是反方向的**，取最小合法 C。

**证据**：一个索引类算子按"行越长越好"取到 C=1024、**205 遍**，对 134 MiB 输出往返 205 次，
而预算只有 815 µs。

### 4. bank 冲突：按跨 lane 访问真正运行的那个 tile 判

**触发条件**：存在 `vf.gather`/`vf.scatter`，且其 pitch **以字节计**是 32 B 的偶数倍。

**规则**：pitch 要 padding 成 32 B 的**奇数倍**。判定必须用**跨 lane 访问实际发生的那个
tile 的 dtype**——不是输入的 native dtype。窄 dtype 升到 fp32 计算时，两者不同。

**证据**：pitch 80 元素，作为 fp16 是 160 B（奇数倍，正常），**作为 fp32 是 320 B（偶数倍，冲突）**，
而 gather 跑在 fp32 工作 tile 上。改对值 **1.5–2×**。
stride 恰为 64 元素是最坏情况：**~5–7×**（51.0 vs 7.0 µs）。
*（该倍率曾发布为 14.6× 并传播到三个算子；源探针踩了 profiler 部分导出的坑而低报。
效应与设计结论成立，倍率不成立。）*

### 5. job 数有甜点

**触发条件**：strided job，每个 reduction 行发一次 DMA。

**规则**：job 数不是越多越好——过了某点 DMA 次数就主导了。**扫一遍，不要外推。**

**证据**：3.2 GB 的一个形状：job floor 128 → 263 GB/s、**512 → 604**、2048 → 487。
在两个形状上值 2.3–2.4×。

### 6. descriptor 的代价按"遍"算，不按字节算

**触发条件**：考虑把 layout 变换（转置/重排）推进 strided DMA descriptor。

**规则**：**数遍数，不要数字节。** descriptor 本身很快（256 B 31 ns、1 KB 46 ns），
但换 layout 往往让**遍数变多**，那才是代价。加宽 `ROWS` 能降低每元素成本，
却会**同比减少 job 数**——而 job 数通常是另一个绑定约束。

**证据**：转置形态需要三遍 descriptor（读 x、写 values、写 indices），bulk 形态各一遍：
**1.509 vs 0.942 ns/elem/core，差 1.60×**，整体为负收益。

### 7. fold 之前先预算一块独立的工作 tile

**触发条件**：把载入的 tile **直接当作**归约/fold 的工作 tile（别名复用）。

**规则**：别名会让这条路径**失去双缓冲**，无法与下一次载入重叠。fold 前预算独立工作 tile。
**一个能重叠的窄档位可以赢过一个不能重叠的宽档位。**

**证据**：同一次 fold 改善了每一个窄 dtype 档位，却把一个 fp32 形状劣化 **2.24×**
（11.76 → 26.32 µs，SOL 0.810 → 0.254），一次吃光其它全部收益。
原因不是 dtype：窄 dtype 路径保留了独立 tile 因而仍双缓冲。

## 三、坑

每一条都产出过"看起来完全合理、且能复现"的错误结论。

| 触发条件（你在做这件事时） | 会得到的错误结论 | 正确做法 |
|---|---|---|
| **拿整个 kernel 的耗时当它的 DMA floor** | floor 被低估（实测 **2.2×**），并把 cliff 判反：某形状 floor 看起来 22.6 µs、远高于 12.42 µs 的 cliff，实测 **11.49 µs，本来就在下面** | 单独跑只有 load+store 的 sweep。**过了 L2 两条曲线会收敛，*正因为*那里已经 DMA-bound**——这恰好掩盖错误 |
| **看到串行依赖就判定它是瓶颈** | 打断进位链、改从 UB 重载状态，**慢 4%**（吞吐受限，不是延迟受限）。此前两列 unroll 只动 2%，是在治不存在的症状 | **先测再 unroll。** 区分吞吐/延迟受限再动手 |
| **沿用上一轮的收益估算** | 一个 **+5.8** 的估算被带了三轮；阻塞它的 API 缺口解决后，改动在四个目标形状上**全部更慢**并被回退——两轮前的另一个修复已经从别的路径收走了这份价值 | **估算的寿命 = 产生它的那次诊断的寿命。花钱前重新推导** |
| **按档位（tier）给重关联设门** | 同一个 fold 在一个形状精度失败（MERE 2.79e-03，门限 1.22e-04），**同档位**另一个形状以 2.66e-09 通过 | **按 `value_range` 设门。** 跨度大时 `exp(x−max)` 对绝大多数 lane 下溢，把由少数项承担的和跨多行重关联就丢了——差别在取值范围，不在 shape |
| **改了 kernel body、名字没变** | 收到**旧二进制**，结果逐字节复现，读起来就是"我的改动没生效" | 用 body 内容哈希命名。**并确认哈希覆盖编译器真正看到的东西**——只哈希渲染出的 wrapper，会让这个保护对它本该拦的改动完全失效 |
| **按 SOL 排优化队列** | 分数列无法区分"并行度不足"和"每元素慢"，两者长得一样 | **按 ratio-to-floor 排队。** 实测把真正的问题顶到队首：某形状 65 个 job、完全并行，仍**偏离自己的 floor 6.7×**（1.19 vs 另一种拓扑做同样工作的 0.42 ns/elem/core） |
| **把自己 kernel 的测量当框架结论** | 一个 kernel 测到三分之一 roofline 就断言"框架带宽被限制在三分之一"；另一个算子实测 **4382 GB/s** | **一个 kernel 测出的带宽是在描述它自己。** 同理 strided-load 拐点（32 B/行 31%、128 B 88%、平台 ~1.92 TB/s）只对当时那个访问形态成立——另一个算子上有一部分其实是 L2 驻留 |

## 四、什么时候停

**先把界测出来，再决定要不要继续推。** 见
[`pypto-pro-op-kb/references/investigation-discipline.md`](../../pypto-pro-op-kb/references/investigation-discipline.md) §8：

- **sibling bound**——把每个 case 按一个**更简单的算子在相同 shape/tiling 上的实测时间**计分。
  它是带着全部真实开销的"零工作量"参照，比自己搭探针便宜也更可信。**它界定的是任何实现。**
- **floor probe**——只搬运合约字节、不做别的。**它只界定数据搬运**，
  不给语义必需的原语定价（跨 lane 每次 15–20 ns），所以必要不充分。
- **从"我想到的杠杆"求和得到的天花板，是关于你那张清单的陈述**，不是关于算子的。
  实测反例：两次算术上无懈可击的不可能性论证（75、然后 72），floor probe 实测 **88.66**。

分数口径、pole、clamped/raw 两个均值见
[`pypto-pro-op-kb/playbooks/benchmark-scoring.md`](../../pypto-pro-op-kb/playbooks/benchmark-scoring.md)。
