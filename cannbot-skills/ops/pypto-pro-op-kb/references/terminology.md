# Hardware and tiling terminology, CN/EN

Fixed vocabulary for reading Ascend documentation, profiler output, and design
discussions that mix Chinese and English. It owns definitions and aliases only
— device limits, API signatures, and layout mechanics belong to the installed
documentation under `$PYPTO_DEVKIT_DIR/docs/pypto_pro/` and to the constraint
pages this KB routes to. Adapted from a sibling Ascend DSL's glossary; every
term kept here is hardware- or convention-level and DSL-neutral, and terms that
were that DSL's own API surface were dropped rather than translated.

## Memory hierarchy

| Term | Meaning | Chinese |
| --- | --- | --- |
| `GM` | off-chip global memory and public tensor storage | 全局内存 |
| `L1` | cube-side local staging memory | 一级本地内存 |
| `L0A`, `L0B` | left/right cube operand buffers | 矩阵操作数缓冲 |
| `L0C` | cube accumulator buffer | 矩阵累加器缓冲 |
| `UB` | vector-side unified buffer | 统一缓冲区 |
| `BT` | matmul bias-table memory | 偏置表缓冲 |
| workspace | kernel-private GM scratch, not a public output | 工作空间 |

Sizes and per-SKU budgets come from the installed platform files — see
[`../constraints/arch-a5.md`](../constraints/arch-a5.md); do not transcribe
them here.

## Pipeline sides and pipes

| Term | Meaning | Chinese |
| --- | --- | --- |
| cube side | matrix/load/fixpipe execution side (`section_cube`) | 矩阵侧 |
| vec side | vector execution side (`section_vector`) | 向量侧 |
| `MTE2` | load pipe (GM → on-chip) | 加载引擎 |
| `MTE1` | L1-to-L0/BT pipe | L1 到 L0/BT 搬运引擎 |
| `M` | matrix-compute pipe | 矩阵计算管道 |
| `FIX` | cube writeback/fixpipe | 写回管道 |
| `MTE3` | store pipe (on-chip → GM) | 存储引擎 |
| `V` | vector-compute pipe | 向量计算管道 |
| `S` | scheduling/control stream | 调度流 |
| pipeline bubble | idle interval within an active pipeline | 流水线空泡 |
| pipe occupancy | active cycles divided by a stated time span (`aic_*_ratio`, `aiv_*_ratio`) | 管道占用率 |

## Tiling terms

| Term | Meaning | Chinese |
| --- | --- | --- |
| `TILE_M`, `TILE_N`, `TILE_K` | local tile sizes along matrix axes | M/N/K 维分块大小 |
| `valid_m`, `valid_n`, `valid_k` | live elements in a boundary tile (pypto-pro: `valid_shape` / `pl.set_validshape`) | 尾块有效元素数 |
| tail / tail tile | final tile that is not full-sized | 尾块 |
| `splitk` | tiling of one core's K loop | 单核 K 维分块 |
| `splitn` | tiling of one core's N loop | 单核 N 维分块 |
| init tile | first K tile overwrites/initializes the accumulator (pypto-pro: `pl.AccPhase` on `matmul`/`matmul_acc`) | 初始化分块 |
| accumulate tile | later K tile adds into the existing accumulator | 累加分块 |

`splitk` / `splitn` describe one core's loop structure; neither creates
cross-core work or a merge.

## Layout and transfer terms

| Term | Meaning | Chinese |
| --- | --- | --- |
| `ND` | dense row-major layout | 稠密行主序 |
| `NZ` | cube fractal layout | NZ 分形布局 |
| `nd2nz`, `nz2nd` | dense-to-fractal / fractal-to-dense conversion | 稠密与分形转换 |
| burst | one contiguous datamove unit | 突发搬运单元 |
| `n_burst` | number of datamove bursts | burst 次数 |
| `burst_len` | payload per burst | burst 长度 |
| stride / step | gap between consecutive burst starts | burst 步长 |

## Two words that are easy to conflate

In kernel prose, *staging* means moving data to a memory level for later use; a
*pipeline stage* is a compute/dataflow phase (pypto-pro: `@pl.pipeline.stage`).
The Chinese 暂存 maps to the first, 流水级 to the second.
