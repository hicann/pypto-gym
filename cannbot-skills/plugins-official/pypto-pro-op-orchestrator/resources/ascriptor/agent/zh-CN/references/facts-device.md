# 设备事实，以及由它推出的规则

光有数字不会改变一个 kernel。本页只在**某个决定取决于它**时才收一条设备事实，并把那个决定
写在它旁边。容量与核数读自[随库发布的 profile](../../../library/ascriptor/devices/profiles)，
吞吐与时钟读自[性能参数](../../../library/docs/performance-parameters.md)。两者都**不在此复制**
——一个数字的第二份拷贝就是第二个会出错的地方。

## profile 里写的是什么

| 资源 | a5（`950`） | a5pr（`950pr`） | a2（`b3`） |
|---|---:|---:|---:|
| Cube core 数 | 32 | 28 | 20 |
| Vector core 数 | 64 | 56 | 40 |
| **每 cube core 的 vector sub-block 数** | **2** | **2** | **2** |
| UB | 256 KB | 256 KB | 192 KB |
| L0C | 256 KB | 256 KB | 128 KB |
| L0A / L0B | 64 KB | 64 KB | 64 KB |
| L1 | 512 KB | 512 KB | 512 KB |
| BT（bias table） | 4 KB | 4 KB | 512 B |

**要引的是比值，不是核数。** "每 cube core 配两个 vector sub-block"在库里七个 profile 上全部
成立；核数不成立，而 `ascriptor.a5` 是 32/64 这一档，950PR 卡才是 28/56。一条写成"56 AIV"的
规则，在它自己门面所指的设备上就是错的。

## 排到向量侧这一步，没有安全的默认值

[A3 profile](../../../library/ascriptor/devices/profiles/a3.json) 同属 C220，每 die
为 20 个 cube 和 40 个 vector，容量与上表 A2 一致。A2/A3 的 Cube→Vector
经 GM 发布与所有权交接；下列 L0C→UB 选项描述 A5，不是 A2/A3 的传输路径。

`l0c_to_ub(..., dual_mode=)` 决定 L0C tile 的 M 行怎么到达这一对 sub-block，
而两个答案都能编译、都能跑、结果都对。

- **`SPLITM`** —— 前 M/2 行给 0 号、后 M/2 行给 1 号，**各自落进自己的 UB**。
  `GetSubBlockIdx()` 于是成为**地址的一部分**，而不是圈在计算外面的守卫。
  这是 IR 的默认值，也是逐行消费者想要的。
- **`SPLITN`** —— 分的是 N 这一维。这并不是什么偏门情况：**转置过的乘法会把
  「一个 sub-block 该拿的那些行」放到 N 上**。attention kernel 里 score 是按
  `score^T = K @ Q^T` 算的，于是它的 query 行**就是** N 维，排空写成 `SPLITN`、`N_dst`
  取 query 块的一半；而下一页那个 PV 结果是 `[M=query, N=D]`，排空就写成 `SPLITM`。
  两者都是"每个 sub-block 拿 64 行 query"。
- **`SINGLE`** —— 整块 M 行只给一个 sub-block，由 `sub_block_id` 指定是哪个，另一个没有活干。
  **只有当单个 sub-block 必须看到每一行 M 时**才正确——那意味着一个**跨 M** 的归约，
  而不是沿着 M 的。

所以规则不是"用 SPLITM"，而是**分那条承载「本 sub-block 所属行」的轴**，
而那是哪条轴，由 matmul 怎么摆决定。`attention/a5_pfa_qk_metadata` 里两次排空相距不到五十行，
并记着 `SINGLE` 那条路在它的形状上实测 +82 µs；`attention/a5_v8_cube_stage` 把同一块 tile
排空两次（`SPLITN` 与 `SINGLE`）并把两份都发布出去，好让参考实现逐一比对。

在本可以 SPLITM 的地方选 SINGLE 要付两次代价，而第二次是藏起来的：vector 的活不再分摊，
**并且**落地 tile 按整块 M 行而不是一半来开。落地 tile 大一倍，往往正是逼你把 M tile 调小的
那个原因，于是两者相乘。实测于一个全程写成 SINGLE 的稀疏 attention kernel：约 4x，
是它与参考实现 3.13x 差距里的主要成分——而那个参考做的路由选择和它一模一样。

有一条性能 lint 会点名每一处这样的搬运；算子本身见 `docs/api/cube.md`。

## SINGLE 有时是**被迫**的，那时它就不是一个选择

split 模式只承载**同类型平搬**——fp32→fp32 或 int32→int32。不带随路 relu、不带任何非默认
requant，**连无 scale 的浮点降位（fp32→fp16/bf16）都不行**，因为 fixpipe 的标量通路挂在
deqScalar 上，而 deqScalar 只在 dual destination control 关闭时存在。所以一次"顺路做转换"的
排空——大多数 attention kernel 都是——**必须**是 SINGLE，lint 也不会对这些发问。
反方向（split 模式带上述任一随路操作）在到达任何 backend 之前就会被拒：硬件做的是别的事，
而三个 backend 里有两个本来会照打不误。

## cube 的活是量子化的，而且量子很大

A5 模型按每周期 4096 个 FP16 操作数 MAC 计费，**MMAD setup 67 个周期**（A2 是 2048 与 21）。
在计费之前，M 与 N 补齐到 16、K 补齐到其操作数宽度的量子。两条值得带进 tiling 决策的推论：
数学 FLOPs **低估**了一个分块实现，因为补齐出来的活是真活；以及，由很多小 matmul 组成的
调度每次都要付那个 setup，所以在 MAC 总量相同时，宁可少而大。

## `@vf` 的进入代价在函数体跑之前就已经记上了

A5 cycle model 对**每一次** `@vf` 调用计 `vf_fixed_overhead_cycles` 46 加
`instruction_head_overhead_cycles` 10，发生在函数体第一条指令发射**之前**；
而函数体大致一条一个周期（`vf_pipe_issue_interval`：LD 1.14、ST 1.0、SU 0.48）。
所以一个五十条左右指令的 `@vf`，有一半时间花在"进去"。数值在
[`a5_cycle_model.json`](../../../library/ascriptor/backends/sim/timing/a5_cycle_model.json)，
`performance-parameters.md` 也是指向那个模型而不是复述它，本页同理。

规则：**把相邻的 `@vf` 合并，而不是拆开**；写在 tile 循环里的 `@vf` 要按"每个 tile 付一次"来算。
同一笔账也决定"重算还是驻留"——已经在一个 `@vf` 里面时重建一个量很便宜，
要为它单开一个 `@vf` 就很贵。

需要"被看见"的交接更贵：`intra_core_sync_latency_cycles` 在 a5 是 200（a2 是 1000），
所以短函数体尾部的一个 barrier 比它包住的东西还重。
本仓稀疏 attention kernel 上实测：把谓词重建放进一个约七十条指令的**逐 tile** `@vf` 里，
vector 侧 722 μs，而 cube 侧只有 161 μs。

## 会变换 layout 的 DMA 不是一条指令

`ub_to_l1.nd2nz` 是按 ND 行的每个 NZ fractal 列各发一次 MTE3 burst 拼出来的。
cycle model 给它每周期 32 字节，而普通 `ub_to_l1` 是 256；板卡实测约 10x（D-084）。
改成**在 `@vf` 里就把 tile 存成 compact NZ**（一次带 stride 的 `vf.store` 直接写 fractal），
再用普通 `ub_to_l1` 搬。

这个改写自带一个坑，两半要一起拿：带 stride 的 store，其自然 block stride 就是暂存 tile 的行数，
而行数天然 16 对齐——**正是 bank 阶梯上最差的那一档**。把 tile 多垫一行让 stride 变成奇数。
D-226 里改的两个 kernel，垫这一行比它所搭载的那次搬运本身还值钱。
有一条 lint 会在调用点把这两半连同数字一起说出来；本页存在的意义是让这条规则**在 kernel 写出来之前**就读得到。

## 不要把核数写死

`block_dim` 来自 profile，不来自字面量。同一份 kernel 源码要在 32c/64v 和 28c/56v 的部件上都跑，
一个假定了其中一种的 launch，要么让核空转，要么去要并不存在的核。
上面那张表是用来读成本模型的，不是用来粘进 kernel 的。
