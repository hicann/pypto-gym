# 成本与 Roofline 分析

选择性能改动或解释耗时前，填写[分析模板](../../templates/performance-analysis.md)。
先冻结目标、允许改变的维度、比较口径与停止条件。用户准出和剩余空间分别报告。

## 建立成本图景

1. 记录公式、操作数与累加精度、维度和物理 tile。Cube MAC 与 vector 工作分别统计，
   写清一次 MAC 计几个 FLOPs。
2. 推导独立工作项、允许与使用核数、每活跃核工作量。题目可能故意固定单核，不能将其
   当作部署时的优选。
3. 分层、分读写方向统计唯一逻辑字节与实际请求，包含重读、每核副本、padding 和 layout。
4. 按存储层计算常驻输入和全部同时 live 的 buffer 版本；数据驻留和加深流水共同占用容量。
5. 记录各实际串行资源上的模型工作及关键等待。保留 core/lane 身份，参与者工作总和不是墙钟时间。
6. 记录实际时延、有效硬件计数、缓存条件、频率来源和能力估计依据；缺项保留未知。

Library 所有的[性能参数](../../../library/docs/performance-parameters.md)给出 model MAC
速率与频率证据。维护者在 2026-09-07 提供 A2/A3/A5 的 GM 标称带宽 1.6 TB/s，合适访问
模式下实际约为 80%，即十进制 1.28e12 B/s。离散读写会更低，尚无通用折扣系数。这是
规划估算，不是当前 kernel 的测量，也不是每 core 带宽或 A2/A3 发布支持声明。

使用该值时报告敏感性场景；只有实际获得 HBM 流量和持续带宽才能报告 HBM 达成率。
GM 重读可能命中 L2，小搬运、离散访问或单个活跃核不一定达到上述估值。

## 两种互补下界

对于口径一致的计算类别与存储层：

```text
I_level = 此计算类别的运算量 / 该层流量字节
P_bound = min(该计算类别能力, 该层带宽 * I_level)
T_compute_bound = 此计算类别运算量 / 此计算类别能力
```

FP16 cube 与 vector 算术、转换分别建账。Vector 运算量不能除以 cube 峰值。Vector 时间
很难粗估，应使用逐操作/VF 模型成本与依赖，或实测阶段时间，包含 reduction、cast、issue
和访存行为。未知的 vector 成本保持未知，不给所有 VF 操作虚构统一吞吐。

对于固定 lowered 工作和 timing model：

```text
T_resource_bound = max_resource(sum(该资源上的非同步工作成本))
T_model >= max(T_resource_bound, 已知依赖下界)
resource_efficiency = T_resource_bound / T_model
仅重排程的加速上界 <= T_model / T_resource_bound
```

按真实串行资源分组，包含 core/lane 身份。两次 cube 阶段计入同一 cube 计算资源；并行
vector lane 分别建账，重叠区间不重复加到墙钟时间。简单资源下界忽略部分依赖与启动收尾，
因此偏乐观。单条 DMA 的成本参数不证明模型已标定多核共享 HBM 竞争，模型 cycles 不能
当作硬件微秒预测。

## CVC 演算：删除工作与改善排程

取 `M=2176, K=N=128`，两次 matmul 之间是 scale/bias/ReLU，并保留 FP16 RNE 的 materialize
边界。每 MAC 计两个 FLOPs；权重常驻能省掉多少流量，只由 shape 推出：

```text
cube_FLOPs = 4 * 2176 * 128 * 128 = 142606336
tiles = 2176 / 128 = 17
一份权重字节 = 128 * 128 * 2 = 32768
可删除权重请求 = 2 * (17 - 1) * 32768 = 1048576
```

到这里都是算术。把它变成利用率还需要 cycle 数，而那是测量值，本工作区已经没有了：原先
引用的 lookahead 与权重常驻对照记录已删除，它的数字不再往下带。你要的那一对自己测——在
[混合流水 demo](../../../kernels/ascriptor_kernels/tutorials/mixed_pipeline) 里对相对应的
`cvc_pipeline_*` 与 `cvc_resident_*` case 跑 `--launcher pipesim`，各取逐 pipe 与总的模型
cycles，再对你自己的 shape 套用上面两个比值。

按那个顺序读：主要资源还是搬入 pipe 时，保持工作不变只改排程的上界就是
`T_model / T_resource_bound`；去掉重复权重请求后，主要资源可能换成另一条 pipe，排程问题
要对着新的下界重新问一遍。删流量的收益和改排程的收益不能相乘；高占用可以与可删除工作
同时存在。

## 选择并解释一次实验

依次检查工作分配、复用、tile/容量/layout、排程/依赖、阶段内部。这是调查顺序，不要求
每项都修改。选择允许维度中最有依据的空间，预测删除的工作或隐藏的等待，明确单位和范围。

纯排程对照保持算术、VF 本体、layout、搬运、tile、核数匹配；驻留、网格和 tiling 实验
报告改变的工作与整体收益。每个实质改动后重算成本图景。

保留 warmup、样本数、分布、源码身份和输入 hash。交错或前后对照帮助发现漂移。无效
total-cycle 样本不能提供利用率结论，但有效时延可以单独报告；单个未用 pipe 为零可以
合法。硬件比率取同一采样行，保留参与者分母。继续读[优化](../playbooks/optimize.md)和
[流水区间证据](pipeline-model.md#验证实际排程)。

## 重新分块后先比较工作量

分配字节相同不代表工作量相同。一个大 tile 与两个小 slot 可占用相同容量，却改变循环
次数、中间结果搬运、状态更新和发射/控制工作。增加行分组也可能同时删除 padding 计算
并减少活跃核数。归因于排程前，先报告这些变化。

隔离 lookahead 收益时，匹配 tile、算术、VF 本体、搬运、buffer 数量和核间分配；将此
对照与端到端最快的另一实现分别记录。[Attention 专题](attention-authoring.md)及其 owner
记录说明为何 tile 选择与流水重叠需要不同对照。即使其他变化后来改善整体 kernel，
已记录的无收益实验仍是有效证据。

## 检查明确口径的证据

在 agent checkout 运行[分析工具](../../tools/analyze_performance.py)：

```bash
python tools/analyze_performance.py templates/performance-input.json
```

[示例输入](../../templates/performance-input.json)只有历史 trace 的一对区间，不是完整重叠。
工具检查单位/资源求和，对同核区间求并；没有实测流量和明确的
`hbm_bandwidth_classification: measured_sustained` 时，HBM 达成率保持 UNKNOWN。
假设带宽仅报告为标明假设的场景。硬件采样使用 `time_domain: hardware_us`，模型成本
使用 `model_cycles`。工具检查给定数据，不认证其来源，也不改变工作流准出门槛。
