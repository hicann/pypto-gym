# 多阶段单 kernel 的任务展平

当一个融合 kernel 含 Cube → Vector → Cube 等全局依赖阶段时，按 row tile 启动 block
很容易让小 shape 只占用一个 AIC/AIV。先枚举 public shapes 的 `row_tiles`、`block_dim` 和各阶段
独立输出任务数；若输出任务远多于 block，优先展平任务，而不是先改 tile 大小。

## 调度骨架

1. 把每个阶段的互斥输出映射为一维任务：`task = row_tile * output_tiles + output_tile`。
2. 用 `pl.range(core_id, total_tasks, num_cores)` 跨 block 分配；每个 GM 输出区间必须只有一个写者。
3. wrapper 为整次 launch 选择一个固定 block 集，不能按阶段改变参与者。
4. 阶段间用成对的 `pl.system.sync_all(core_type=pl.SyncCoreType.MIX)`：生产端 section
   结束前一次、消费端 section 开始时一次。
5. 所有已启动 block 和 AIV subblock 都必须到达相同顺序、相同数量的 barrier；数据 guard 只能包住
   load/compute/store，不能包住 barrier。

片段见 [`../templates/stage-task-flatten.py.tmpl`](../templates/stage-task-flatten.py.tmpl)。

## 单因素验证纪律

- 第一次拓扑实验只改任务所有权与同步，不改 K 分拆、FP32/BF16 workspace、累加树或舍入模式。
- 静态穷举每个 shape 的写区间，证明无缺口、无重复；同时数 value-producing DSL 调用的种类和数量。
- 真板先跑含 row tail、小 shape、最大 workspace、独立 epsilon、全零输入的集合，再跑完整用例集。
- 本机通过不能替代目标部署环境：设备子型、CANN、PyPTO、Python 版本任一不同，都要在目标环境复验。

## 固定输入、成组输出的 Cube 复用

展平为单输出任务后，如果多个相邻输出 tile 共享同一个左输入，可以把 owner 改成
`(partition, output_wave)`，一个任务保留一份 Mat/Left 并驱动最多四个独立 Acc。某个 staged 链
的 down projection 由此把每个 row tile、每个 K wave 的 token Mat/Left 搬运从 68 次降到
20 次，而 weight Mat/Right、MMA、最终 GM store 和 FP32 加法树均不变。

这类改动的门禁不是“输出结果看起来相同”，而是逐项证明：

1. 每个 `(row_tile, partition, output_tile)` 仍恰好一个 owner；尾 wave 不重复或漏写。
2. 每个 Acc 只累加固定 partition 的 `k = q * partitions + partition`，且 `q` 严格递增。
3. consumer 仍按原树组合 partitions，例如 `(p0 + p1) + (p2 + p3)`。
4. 峰值同时存活的 MatB、Right 与 Acc slot 不超过 TileGroup slot 数；不能把“下一轮会复用”
   当作当前 live interval 已结束。
5. 小 shape 的任务数可能从输出数增至 `partitions * ceil(outputs/group)`；同时核算并行度和
   调度开销，不能只报搬运减少比例。

如果为了复用而把多个 partition 累加进同一个 Acc，或用 atomic 把 partial 提前合并，就已经
改变数值 ABI，不再是本节的低风险调度优化。

片段见 [`../templates/cube-output-wave-reuse.py.tmpl`](../templates/cube-output-wave-reuse.py.tmpl)。

若优化模式必须更换 TileGroup、mutex 或 Cube 指令链，不能把 runtime shape 条件直接包在链外。
提交契约要求单 JIT/单 launch 时，使用编译期 tiling key，并检查每个 key 的 IR 已消除 key
引用。片段见
[`../templates/tiling-key-resource-specialization.py.tmpl`](../templates/tiling-key-resource-specialization.py.tmpl)。

## AIV 兼容性门禁

如果旧目标版本呈现严格的半行错误，先检查 AIV `pl.get_block_idx()` 的可见编号：不同 A5
设备子型/CANN 快照可能分别暴露物理 block id 或交错 AIV id。只让 subblock 0 工作、再直接按
raw `core_id` 分 row/head/chunk，在前一种约定可覆盖全部任务，在后一种约定只覆盖偶数编号。

当前框架提供 `get_subblock_num` 时，AIV 的 `get_block_idx()` 是交错编号，先还原物理 block：

```python
HAS_SUBBLOCK_NUM = hasattr(pl, "get_subblock_num")
raw_core_id = pl.get_block_idx()
if HAS_SUBBLOCK_NUM:
    physical_id = raw_core_id // pl.get_subblock_num()
else:
    physical_id = raw_core_id
```

`HAS_SUBBLOCK_NUM` 必须是模块加载期 Python 常量，使旧 parser 不解析不存在的 API 分支。Cube
与 Vector task loop 都统一用 `physical_id`，避免同一 kernel-scope block-id 表达式在两类 engine
降低后产生不同 owner。含 GM load/store 的纯 Vector 路径只让 `sub_id == 0` 执行，并用
`physical_id` 分配 row/head/chunk；官方
文档明确同一 physical block 的两个 AIV subblock 共享 MTE，不能把它们当作两个独立 GM worker。
所有未选中的 subblock 仍必须参加 MIX barrier。先静态穷举两种编号模型，再在两个目标版本实测。

这不是通用的 PyPTO-Pro 语义保证；必须把失败/通过的设备子型、CANN/PyPTO 版本和 case 窗口
写入报告，不能从一台 A5 外推所有 A5。

## 常见失败

| 现象 | 优先检查 |
|---|---|
| 小 shape 很慢，大 shape 接近基线 | wrapper 的 `block_dim` 是否只跟 row tiles 绑定 |
| barrier 后间歇脏数据或死锁 | barrier 是否在 guard 内；各 section 的调用数/顺序是否一致 |
| `T<=tile` 通过、`T>=2*tile` 严格一半错误 | AIV raw `core_id` 是否只覆盖了偶数逻辑 worker |
| GM ownership 完整但结果偶发脏值 | 是否让同一 physical block 的两个 AIV 独立发起 GM traffic |
| 每隔固定行数丢半块 | subblock 所有权、物理 Tile 高度与 `load_tile` 地址单位 |
| 精度随拓扑变化 | 是否意外改变 K 累加顺序、partial 合并树或最终 cast |
| 本机有 `kernel_details.csv` 但没有 PyPTO 行 | profiler 与当前 PyPTO/CANN 快照不兼容；保留精度证据，性能改用目标 server 或匹配版本的 msprof |
