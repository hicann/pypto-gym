# 按明确契约练习

以下是编写练习，不表示每个公式都已有发布单元。在 ignored `tmp/practice/<task>/` 中使用
已验收的安装包，从[编写](playbooks/author.md)开始，生成输入并写独立 Torch reference。
借鉴模式前查当前 API 覆盖与单元 contract，不依赖旧仓、记录张量或进程级固定 facade。

| 练习 | 公式或精度边界 | 主要问题 |
|---|---|---|
| Scale/bias | `2*x+1` | GM/UB/register 所有权、尾部与分核 |
| Sigmoid | `1/(1+exp(-x))` | 表达式顺序、溢出域和近似误差 |
| 行求和 | `x.sum(-1, keepdim=True)` | 规约结果 lane 与广播布局 |
| 单 tile softmax | 稳定行 max、减法、exp、sum、除法 | 定义的 lane 与 mask 尾部 |
| 基本 matmul | `x @ y.T` | L1/L0 存储与 cube 握手 |
| KM/KN matmul | `x.T @ y`，`x[K,M]`、`y[K,N]` | 在合法消费者处转置并显式声明维度 |
| 大 K matmul | 同一乘积按 K 分段 | 初始化 tile 与后续累加 |
| Bias/ReLU 后处理 | `relu(x.float() @ y.float().T + bias)` | cube-to-vector 交接与 bias 广播 |
| Matmul 行 softmax | `softmax(x.float() @ y.float().T, -1)` | 行统计与输出生命周期 |
| Matmul L2 归一化 | 每行乘积除以自己的范数 | 零范数策略与两遍统计 |
| 分块量化 | block128 absmax、明确 scale 规则与 FP8 输出 | scale 所有权、独立 bit 语义与 carrier 布局 |
| ReLU 后 matmul | `relu(x).half().float() @ y.float().T` | vector-to-cube 发布前保留 half 边界 |
| 两段混合阶段 | `abs((x*2).half().float() @ y.float().T)+1` | 双向 mutex 与独立 checkpoint |
| Decode attention | `softmax(q @ k.T / sqrt(D)) @ v`，L=1,D=128 | online state、lookahead、warmup/drain 与复用槽位 |
| Causal attention | 声明的 attention 公式加 `k_pos <= q_pos` | query 位置约定、因果边界和被 mask 的 PV 访问范围 |

每项练习都要选择精确的 dtype/layout/domain 与启动拓扑。A2/A3 workspace bridge 与 A5
片上 bridge 是不同契约。概率 cast 与归一化分母必须保持 reference 的顺序，画廊索引里的
tag 不能作为移动它们的依据。宿主代码负责分配、按显式 ABI 打包、dispatch 和比较，不能悄悄
替代任务要求在 kernel 中实现的数学阶段。

覆盖对齐/尾部、受支持的多 tile/多核心、适用的输出初值，以及同一核心的槽位复用。
按[精度](references/precision.md)选择 exact 或有依据的容差，并拒绝故意缺失、全零或损坏
的输出。功能模拟、lowered pipe、源码生成、厂商编译和 board 分别记录。未解决的 warning
必须有诊断与限制说明。提升为正式样例前，把该目录单独复制到无关位置，用声明的安装依赖
在那里运行；它若还从原来的树里 import 任何东西，就不是自包含的。最小 API 教学归 library，
完整可运行算法归 kernels 画廊，形态是 `kernel.py` + `reference.py` + `main.py` +
`metadata.json`，不放别的文件。

## 混合流水与指南练习

用[可运行混合流水 demo](../../kernels/ascriptor_kernels/tutorials/mixed_pipeline)练习 CVC、
VCV、CVCV、VCVC 与 CVCVC。在该目录执行 `python main.py --list` 列出 case，
`python main.py --pattern CVC --mode pipeline` 只跑一个图的一种排程。复制 delay/depth 前
先读[通用方法](references/pipeline-model.md)，流式权重与常驻权重分别使用自己的匹配串行
对照——该 demo 给每个图都备了 `serial`、`pipeline`、`resident_serial`、`resident`，正是
为此。合法候选在小 shape 上仍可能没有计算重叠或更慢，应保留实际结果。该目录里没有任何
测量值，速度结论必须来自你自己的测量。

评估文档本身使用独立上下文协议和任务契约，
分别考察数值编写、受限排程与开放性能。一个 kernel 的 case 扫描不等于多次独立生成试验。
本轮范围与结果记录在验证回执。
