# msprof op 采集与分析

## Contents

- [前提](#prerequisites)
- [Step 1：构建算子](#build-operator)
- [Step 2：采集](#collect-profile)
- [Step 3：归档 + 统计摘要](#archive-summary)
- [Step 4：性能标准判定](#performance-criteria)
- [Step 5：瓶颈定位与优化](#bottleneck-analysis)
- [Step 6：验证优化效果](#validate-optimization)
- [数据目录结构](#data-layout)
- [上板 vs 仿真选择](#target-selection)
- [注意事项](#cautions)
- [相关资源](#resources)


> 标准上板采集流程：依赖 `$ASCEND_HOME/tools/msopprof/bin/msopprof`，产出 8 份独立 CSV + 逐核 `PipeUtilization.csv`，可走完 **构建 → 采集 → 归档 → 判定 → 优化 → 回归** 闭环。

---

## <a id="prerequisites"></a>前提

- 环境中存在 `$ASCEND_HOME/tools/msopprof/bin/msopprof`
- 推荐用法包含 `--warm-up` 以规避 DVFS

---

## <a id="build-operator"></a>Step 1：构建算子

**直调算子**：

```bash
cd ops/{operator_name} && mkdir -p build && cd build && cmake .. && make -j
```

**aclnn 算子**：

```bash
bash build.sh --pkg --soc=ascend910b --ops={operator_name} --vendor_name=custom -j16
./build_out/*.run --install-path=$CANN
bash build.sh --run_example {operator_name} eager cust --vendor_name=custom
```

---

## <a id="collect-profile"></a>Step 2：采集

```bash
# 基本用法
msprof op ./demo

# 推荐用法（含预热 + 指定输出）
msprof op --warm-up=10 --output=./msprof_output ./demo

# 多次运行取均值
msprof op --warm-up=10 --launch-count=5 --output=./msprof_output ./demo
```

### 关键参数

| 参数 | 说明 | 何时使用 |
|------|------|---------|
| `--warm-up=N` | 预热 N 次后再采集 | **始终建议**，避免 DVFS（动态调频）影响首次运行 |
| `--launch-count=N` | 运行 N 次取均值 | 需要统计稳定性时 |
| `--output=<dir>` | 指定输出目录 | 避免结果散落 |
| 无需 `--soc-version` | 上板自动检测硬件 | — |

**输出**：在指定目录或当前目录下生成 `OPPROF_{timestamp}_XXX/` 文件夹。

---

## <a id="archive-summary"></a>Step 3：归档 + 统计摘要

```bash
# 找到最新 OPPROF 目录
OPPROF_DIR=$(ls -td <output_dir>/OPPROF_* | head -1)

# 归档 CSV + 生成摘要（自动创建 docs/perf/round_NNN/）
python3 $CANNBOT_CONFIG_ROOT/skills/pypto-pro-op-perf-tune/scripts/perf_summary.py $OPPROF_DIR ops/{operator_name}
```

脚本会：

1. 在 `ops/{operator_name}/docs/perf/round_NNN/` 创建归档目录（轮次自动递增）
2. 复制全部 8 个 CSV 原始文件到归档目录
3. 生成 `summary.txt`（各指标 min/avg/max，**不做达标判定**）

---

## <a id="performance-criteria"></a>Step 4：性能标准判定

对照下表与各流水占比，判定算子性能是否达标。**性能达标**指：主导流水与算子类型匹配（见 4.3）、表 4.2 中严重项未集中触发，且核间负载、带宽等未同时恶化。

### 4.1 总体判定流程

```
读取 OpBasicInfo.csv → Task Duration、Block Dim
    ↓
读取 PipeUtilization.csv → 各流水占比最高的单元（主导流水）
    ↓
对照 4.2 阈值与 4.3 算子类型预期
    ↓
综合结论
    ├── 指标整体健康、主导合理 → 达标或已接近硬件极限
    ├── 多项警告或主导与类型不符 → 有优化空间 → Step 5
    └── 多项严重项 → 严重瓶颈，须优化 → Step 5
```

### 4.2 各指标达标标准

| 指标 | 达标条件 | 警告条件 | 严重问题 |
|------|---------|---------|---------|
| **核间负载均衡** | 各核 `ai*_time(us)` 差异 <10% | 差异 10-30% | 差异 >30% |
| **Block Dim** | 等于可用核数（910B: 20~40 核） | 远小于可用核数 | Block Dim = 1 |
| **VEC ratio** | 与算子类型匹配（见 4.3） | VEC ratio >80% | VEC ratio >90% 且无优化空间 |
| **MTE2 ratio** | <30%（计算型算子） | 30-50% | >50%（搬运成为瓶颈） |
| **fixpipe_ratio** | <5% | 5-15% | >15%（地址未对齐） |
| **icache_miss_rate** | <5% | 5-15% | >15%（代码量过大） |
| **bank conflict 总占比** | `aiv_vec_total_cflt_ratio` <5% | 5-15% | >15% |
| **L2 Cache 总命中率** | >80% | 50-80% | <50% |
| **头开销** | <总耗时的 10% | 10-30% | >30% |
| **DoubleBuffer 效果** | MTE2/VEC 重叠 >30% | 重叠 10-30% | 重叠 <5% |
| **带宽利用率** | `bw_usage_rate` >60% | 30-60% | <30% |

### 4.2a 这些 ratio 是**重叠占比**，不是时间的划分——不能当屋顶用

`aiv_scalar_ratio` = `aiv_scalar_time / aiv_time`，其余 ratio 同理。各条流水**并发执行**，
所以四个占比**相加通常大于 1**：strided kernel 上实测合计可达 **2.0–2.6**。

后果很直接：**`scalar_ratio = 0.67` 不代表"67% 的时间被标量卡住"**，它只说明标量流水在 67% 的
时间里是忙的——而同一段时间里向量与 MTE 也可能各自忙着。把它读成"标量是瓶颈"是这一族
指标最常见的误判，代价通常不是白做一轮，而是**做反一轮**：按错误归因排出的优化方向，
往往正好避开真正的瓶颈。

**判据**：某条流水是不是瓶颈，看的是**它是否饱和**（接近 1.0）以及**去掉它的工作量会不会变快**，
不是看它在几个占比里最大。可靠的做法是**隔离扫描**——每次只删掉一个构造，测时间变化：

> 一组典型的隔离扫描结果（六个变体各差一个构造）：把**全部算术**删光只快 1.0–4.7%；
> 去掉 int64 索引输出（16 字节里的 8 字节）只快 0.6–5.0%；**打断串行依赖反而更慢**。
> 而把**每个 2-D 描述符从 1 行改成 8 行**直接买到 **1.77–2.31x**，再把状态搬进寄存器到 **1.91–2.63x**。
> 真正的成本是「每个归约行一个 DMA 描述符」——三个 ratio 里没有任何一个指向它。

一个推论：当所有流水都不饱和（本例修完后**无一超过 0.56**），算子既不是计算 bound 也不是带宽
bound，而是**受制于发起的操作数量**（描述符、launch）。这种情形下再调 tile 宽度或算术都不会动，
要动的是"发多少次"。

### 4.2b 高 `vec_ratio` 可能是陷阱，不是"到顶"

上表把 `vec_ratio > 90% 且无优化空间` 列为"严重问题"，这是对的，但它**不能反过来读成"vec_ratio 高就说明算得满、没救了"**。
向量单元 100% 忙，只说明**发射槽被占满**，不说明**槽里装的是有用的工作**。

一个反例：某个 1-D 输入的 case 读到
**`vec_ratio 0.996`，却只跑到 roof 的 0.2%**。原因是该拓扑把 64 条车道分给 64 个不同的**行**，
而这个 case 只有 1 行——它在满负荷地计算 **63 条无用车道**。只看 ratio 会判定"已达计算 bound、
无优化空间"，实际有 **45 倍**的空间：固定 job 数与元素总数、只改变行数，耗时从 1473.98 µs 降到 32.63 µs，
且随 `min(rows, 64)` 线性。

**判据**：`vec_ratio` 高时，必须再问一句**有效车道占用率**——参与计算的车道里有几条产出了会被写回的结果。
可用的验证手段是一次**隔离扫描**：把元素总数与 job 数固定住，只改变能填满车道的那个维度；
如果耗时随之线性下降，那么之前的高 ratio 是空转，不是饱和。

同一轮还给出另一条：`Block Num` 显示的是**launch 网格**，不是真正干活的核数，不能拿它判断占用率。

### 4.2c 饱和的 `mte2_ratio` 不等于带宽 bound——先看每个描述符搬了多少连续字节

`mte2_ratio` 逼近 1.0 常被读作"总线跑满"。**它只说明 DMA 在发描述符，不说明它在搬字节。**

**判据**：把 ratio 与**实测达成带宽**放在一起看。

> 流水已饱和、而达成带宽仍随每行连续字节数上升 ⇒ **描述符 bound**，不是带宽 bound。

**如何自证**（隔离扫描，任何算子都能做）：只改片段宽度这一个 tiling key，保持元素总数、
dtype、算法与 launch 网格不变，量每档的达成带宽。承载结论的是**"随片段宽度单调上升"**，
不是任何单点数值——单点会被单次发射的离散度淹没（同一配置重复读数差 10% 量级并不罕见），
所以引用单调性与量级（典型跨度可达数倍），不要把某一档的接近程度当精度声明。

**必须排除的替代解释**：片段宽度与车道占用率会被同一个旋钮同时改变，两种故事都能拟合。
要分开，就找**片段宽度相同、车道占用率不同**的两点比较，或反过来。

**收敛到的做法**：把多行打进一个 2-D 描述符。

**边界**：片段宽度**除不尽**时会崩——尾巴要独占一个 register row 并走一整趟 strided pass，
可以反而更慢。规则是"越宽越好，直到尾巴变病态"，看的是 `I mod wb`。

### 4.2c-2 描述符尺寸的膝点在 2-4 KB，且这条曲线只在**数值已验证正确**的代码上才作数

**规则**：单靠描述符尺寸就可能有数倍差距，一行新代码都不用写。成本主要是**每个描述符的固定
开销**，因此曲线形如「固定/tile + 每 KB 增量」，膝点落在 **2-4 KB**，之后基本走平。

**如何自证**：固定总字节数、行数与算术，只改每描述符字节数，扫一条完整曲线并拟合。
同一轮加一个"把算术全删光"的对照——若删光后时间几乎不变，说明该路径上算术不是成本。

**流程判据（这条比数值更重要）**：**ladder 的形状只有在数值已验证正确的代码上才作数。**
一个数值错误的原型可以给出看似真实的极值：在坏原型上做的扫描里，同一配置重复两次读数
可能差三成以上，噪声会伪装成拐点，指向的最优点与修好后完全不同。

判定与动作：
- 同一配置重复两次差异过大（例如 30% 量级）⇒ **该轮不足以判定极值**，先降噪再谈形状。
- 极值必须在**最终代码**上复测，不得沿用原型结论。
- 先证明数值正确，再读性能曲线；顺序反了，整轮作废。

### 4.2d strided **store** 比 strided **load** 贵得多——不要拿 load 的价给 store 计价

**规则**：同一几何下 strided store 的单位成本**显著高于** strided load（实测量级 3x 起），
且**几何越窄越糟**（窄几何下可达 7x）。

**如何自证**：在同一次采集里分别测 load-only 与 store-only 探针，几何、dtype、核数一致，
单位统一到 ns/elem/core。

**这条会直接改变设计结论**：一个"一读两写"的三趟 strided 方案，若用实测 *load* 成本给三趟
统一计价，可以算出明显优于现行方案；换成实测 store 重算后，仅已测项就已劣化，方案实际是负的。
**误差不在算术，在把一条流水的价格套给了另一条。**

**同轮可复用的一条**：提高流水线深度（buffer 数）对这条路径几乎无效——每描述符延迟不可重叠时，
深度翻十倍买到的可能不足 1%。先确认延迟能否重叠，再投入深度。

**做法**：strided dataflow 的成本模型里，load 与 store **分别实测，不得互相代入**；
写宽度更大的那一趟（如 int64 索引）尤其不能用窄 dtype 的价去估。

### 4.2e apparatus control 必须踩在被测对象的同一条路径上

**规则**：共享机器**非均匀降级**——同一块卡、同一份代码，某类循环可能只慢 1.4x，
而 bulk/strided 路径同时慢 3-14x。因此"总控制项复现了"**不等于**"这次测量可信"。

**判据**：要给 strided/bulk 路径定价，control 就必须是 strided/bulk 的那一项。
拿一个跑在别的流水上的 control 代签，**等于没有 control**。

**典型失效方式**：设计文档只规定"某个总控制项回到基线即可信"，于是某一轮该控制项通过、
而同轮的 strided 控制项偏离数倍，门槛放行了一次本该作废的测量。

**做法**：
- control 集合要**覆盖被测的每一条流水**，不是一个总闸。
- 全部 control 同轮复现（例如 1.5% 以内）才认这一轮；任一项偏离即整轮作废，不做挑选性采信。
- 把 control 读数与被测读数**记在同一份记录里**，事后才能判断当时是否可信。

### 4.2f 性能扫描的正确性闸门要用容差，绝不能用 bit-exact

**规则**：改 tile 几何会改变归约结合序，**末位差是预期行为**，不是 bug。
用 `torch.equal` 当闸门、并在采样前 `exit`，会把完全有效的实验臂整条丢掉。

**判据**：`ndiff`（不等元素数）**本身分不清"舍入"和"算错"**——不等元素占比可以高达两位数百分比，
而最大绝对误差仍在 1e-07 量级。必须同时报 `maxabs` / `maxrel` 才能判。

**做法**：
- 闸门报 **`ndiff` + `maxabs` + `maxrel` 三个数**，只在**大到不可能是舍入**时才判失败。
- **任何情况下都不要在采样之前退出**——先拿到时间，再判要不要采信。
- 阈值按 dtype 与归约长度定，不要跨 dtype 复用同一个数。


### 4.3 不同算子类型的预期 ratio 分布

| 算子类型 | 主导流水 | 预期 ratio | 异常信号 |
|---------|---------|-----------|---------|
| **Elementwise**（Add/Mul/Relu） | VEC | vec_ratio 50-80% | MTE2 ratio > VEC ratio |
| **Reduction**（ReduceSum/Max） | VEC | vec_ratio 40-70% | scalar_ratio >20% |
| **Activation**（Softmax/Gelu） | VEC | vec_ratio 60-85% | 大量 cast 指令 |
| **MatMul** | CUBE | cube_ratio 40-70% | vec_ratio > cube_ratio |
| **纯搬运**（Transpose/Concat） | MTE2/MTE3 | mte2+mte3 合计 >50% | VEC ratio >30% |

---

## <a id="bottleneck-analysis"></a>Step 5：瓶颈定位与优化

1. **先读 `summary.txt`** — 全局概览  
2. **结合 `csv_fields_reference.md`** — 理解字段含义和阈值  
3. **发现异常时再展开原始 CSV**  
4. **再按下表** — 确认瓶颈类型  

**快速查找**：

| 瓶颈类型 | 判定条件 | 首选优化 |
|---------|---------|---------|
| **VEC Bound** | `aiv_vec_ratio` 最高 | UB 融合、减少 Cast、融合指令 |
| **MTE2 Bound** | `ai*_mte2_ratio` 最高 | 增大搬运粒度 ≥16KB、512B 对齐、L2 CacheMode |
| **CUBE Bound** | `aic_cube_ratio` 最高 | L0C 累加、L1 数据复用 |
| **SCALAR Bound** | `ai*_scalar_ratio` >30% | 缩小 TilingData、减少核数 |
| **核间不均衡** | 各核耗时差异 >10% | 调整 Tiling 切分策略 |
| **Bank Conflict** | `vec_bank_cflt_ratio` >5% | 调整 UB 地址、添加 padding |
| **头开销大** | 头开销占比 >30% | 减少核数、缩小 TilingData、TPipe 外置 |
| **DoubleBuffer 未生效** | MTE2/VEC 无重叠 | 检查 InitBuffer 是否设置 bufNum=2 |
| **流水线气泡** | 多单元均 30-50%，无主导 | 增加 workspace 份数、异步迭代 |

---

## <a id="validate-optimization"></a>Step 6：验证优化效果

每次优化后重新执行 Step 2–3，数据自动归档为 `round_NNN+1`。

```bash
# 对比两轮摘要
diff ops/{operator_name}/docs/perf/round_001/summary.txt ops/{operator_name}/docs/perf/round_002/summary.txt

# 或直接读两个 summary.txt 进行对比分析
```

**对比要点**：

1. Task Duration 是否下降
2. 瓶颈单元的 ratio 是否改善
3. 核间均衡是否改善（aiv_time min/max 差距）
4. 是否引入新的瓶颈

---

## <a id="data-layout"></a>数据目录结构

### 临时输出（采集后、归档前）

```
OPPROF_{timestamp}_XXX/
├── dump/                       # 原始性能数据（无需关注）
├── OpBasicInfo.csv             # 算子基本信息（名称、核数、总耗时、频率）
├── PipeUtilization.csv         # 各流水线单元耗时和占比（最重要）
├── ArithmeticUtilization.csv   # Cube/Vector 指令 cycle 占比和计算量
├── Memory.csv                  # 内存读写带宽和数据搬运量
├── MemoryL0.csv                # L0A/L0B/L0C 读写带宽
├── MemoryUB.csv                # UB 读写带宽（Vector/Scalar）
├── L2Cache.csv                 # L2 Cache 命中率
├── ResourceConflictRatio.csv   # Bank conflict 和资源冲突占比
└── visualize_data.bin          # MindStudio Insight 可视化文件
```

### 持久归档

```
ops/{算子名}/docs/perf/
├── round_001/
│   ├── OpBasicInfo.csv
│   ├── PipeUtilization.csv
│   ├── Memory.csv
│   ├── ResourceConflictRatio.csv
│   ├── L2Cache.csv
│   ├── ArithmeticUtilization.csv
│   ├── MemoryUB.csv
│   ├── MemoryL0.csv
│   └── summary.txt            # 统计摘要（min/avg/max，不含判定）
├── round_002/
└── ...
```

字段含义详见 [`csv_fields_reference.md`](csv_fields_reference.md)。

---

## <a id="target-selection"></a>上板 vs 仿真选择

| 维度 | 上板 (msprof op) | 仿真 (msprof op simulator) |
|------|-----------------|---------------------------|
| 需要 NPU | 是 | 否 |
| 时序精度 | 真实硬件时序 | 周期级模型估算 |
| 输出 | 8 个 CSV 文件 | CSV + trace.json |
| 指令级流水图 | 需加参数或单独仿真 | 默认输出 |
| **资源冲突数据** | **有**（ResourceConflictRatio.csv） | 无 |
| **L2 Cache** | **真实命中率** | 估算 |
| DVFS 影响 | 有（需 warm-up） | 无 |
| 适合阶段 | 性能验收、生产调优 | 早期开发、指令级调试 |

**建议**：开发阶段用仿真快速迭代，验收阶段用上板确认真实性能。

---

## <a id="cautions"></a>注意事项

1. **必须 warm-up**：首次运行受 DVFS 影响，耗时偏高。始终使用 `--warm-up=10`
2. **频率检查**：读取 `OpBasicInfo.csv` 的 `Current Freq` 和 `Rated Freq`，若 Current < Rated，说明芯片未满频运行
3. **MTE2/MTE3 带宽共享**：同时读写 GM 时总带宽共享，评估搬运段负载时宜按 MTE2、MTE3 合并字节量与平台带宽对照
4. **小数据量场景**：数据量很小时头开销占比会很高，这不一定是算子问题，而是数据量不足
5. **多核同地址访问**：多核同时读同一 512B 地址范围会被串行化，导致 MTE2 耗时异常

---

## <a id="resources"></a>相关资源

| 文件 | 内容 |
|------|------|
| [`csv_fields_reference.md`](csv_fields_reference.md) | 8 个 CSV 文件的完整字段定义和阈值 |
| `../scripts/perf_summary.py` | 统计摘要生成 + CSV 归档（Step 3 调用） |
