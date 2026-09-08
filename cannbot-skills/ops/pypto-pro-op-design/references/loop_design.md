# 循环与Section结构设计

R4根据R0的Module划分组织循环和Section代码结构。本轮确定各Module如何放入`section_cube`或`section_vector`、循环相对Section的位置、跨Tile状态的生命周期、动态循环上界和分核信息的获取位置。

相关资料：

- 多核任务分配：`$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md`
- 尾块处理：`$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/tile_based_python_programming/tail_block_handling.md`

## 将Module放入Section

R0已经记录每个Module属于Cube还是Vector。R4据此确定源码中的Section结构：

- 相邻的同域Module可以顺序写在同一个Section中；
- Cube Module和Vector Module分别放入`pl.section_cube()`和`pl.section_vector()`；
- 根据数据复用范围决定循环放在Section内，还是由多个Module各自组织循环；
- 两个Section共享的TileGroup在Section之前声明，只在一个Section内使用的Tile或TileGroup放在对应Section内或统一资源声明区；
- Section之间的数据通路随结构一并写清，具体同步点和event_id在R6确定。

Section结构必须与R1确定的API调用链和R3确定的内存空间一致。操作需要切换执行域时，不能为了保留现有循环而放进错误的Section。

## 从结果单元确定循环层次

先确定循环一次要完成哪个输出。这个输出对应一套独立的状态，也可以和其他输出并行计算。例如：

- 逐元素算子通常以一个输出Tile为结果单元；
- 沿N轴做Softmax时，以一个行块为结果单元，行内所有N Tile属于同一个归约；
- Matmul通常以一个`[M_tile, N_tile]`输出块为结果单元，所有K Tile共同更新它的Acc状态。

确定结果单元后，按下面的顺序组织循环：

1. 写出输出空间的Tile坐标，以及哪些轴彼此独立；
2. 把需要连续处理的Tile轴放到结果单元内部，标出完整的遍历范围；
3. 如果多个Tile共用一份状态，写明状态的初始化、更新和使用位置；
4. 根据数据复用关系选择扁平任务循环或嵌套循环；
5. 写出动态维度对应的Tile数和循环上界；
6. 确认每个输出位置只有一个任务写入。

R4只确定逻辑循环，具体由哪个核处理哪个结果单元在R5确定。

对于彼此独立的二维输出Tile，可以把二维坐标展平成一维任务编号：

```python
for task_id in pl.range(core_idx, m_tiles * n_tiles, core_num):
    m_idx = task_id // n_tiles
    n_idx = task_id % n_tiles
    ...
```

如果同一行的多个Tile共享数据或状态，则保留行和列两层循环：

```python
for m_idx in pl.range(core_idx, m_tiles, core_num):
    for n_idx in pl.range(0, n_tiles, 1):
        ...
```

选择循环形式时，主要看GM访问是否连续、状态能否复用、各结果单元的计算量是否接近，以及是否存在多个任务写同一输出。Softmax以完整的一行为归约单元；如果把同一行的N Tile分给不同任务，还要设计局部统计量的合并过程。

## Matmul的K循环

矩阵乘的M、N循环选择输出Tile，K循环放在输出Tile内部并持续更新同一个Acc Tile。具体写法直接参考以下官方文档：

- `$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/tile_based_python_programming/Cube_matrix_computation.md`中的“K维分块累加”，包含K循环、首块`matmul`、后续`matmul_acc`、Tile配置和完整Kernel；
- `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/matrix_computation/matmul_acc.md`，用于核对`matmul_acc`参数、Acc Tile约束和分块累加示例；
- `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/matrix_computation/phase.md`，用于核对`AccPhase`、`STPhase`和`unit_flag`的配对要求。

R4只需记录M/N输出循环与K循环的嵌套关系、K的Tile数，以及采用单次`matmul`还是K分块累加。接口参数和同步要求直接引用上述文档，不在设计资料中重复整理。

## 动态循环上界

R4只处理动态shape对应的循环边界。shape值从Kernel参数或TilingData中取得，再据此计算循环所需的Tile数。例如：

```text
n_tiles = (N + TILE_N - 1) // TILE_N
```

每次遍历都用同一个`n_tiles`覆盖完整计算轴，同时注明规格是否允许`N=0`。尾块的`valid_shape`和mask在R7设计，TileGroup的深度、下标和轮转方式在R2设计。

## 分核信息

分核信息包括当前AI核的编号和参与分工的核数，具体取值取决于Section所在的执行域：

| Kernel与Section | `core_idx` | `core_num` |
|---|---|---|
| 仅Cube | `pl.get_block_idx()` | `pl.get_block_num()` |
| 仅Vector | `pl.get_block_idx()` | `pl.get_block_num()` |
| Cube/Vector混合Kernel的Cube Section | `pl.get_block_idx()` | `pl.get_block_num()` |
| Cube/Vector混合Kernel的Vector Section，按全部AIV分工 | `pl.get_block_idx()` | `pl.get_block_num() * pl.get_subblock_num()` |

混合Kernel的Vector Section中，`pl.get_block_idx()`返回展平后的全局AIV核编号。`pl.get_subblock_idx()`返回当前逻辑Block内的AIV核编号，可用于让多个AIV分别处理同一结果Tile的不同部分。需要在全部AIV核之间分配任务时，使用`pl.get_block_idx()`。具体分核方式在R5确定。

Kernel只包含一个执行域时，分核信息可以在Section之前读取，也可以在Section内读取。Cube和Vector混合时，Cube核和AIV核的编号范围不同，应分别在各自Section中读取。

## 官方指定算子示例

以下路径来自PyPTO Pro工作流维护的官方指定算子清单，均相对于`$PYPTO_DEVKIT_DIR`。实际可用文件以`PRO_MATERIAL_INDEX.md` §B为准。

| 示例 | 适合参考的循环与Section结构 |
|---|---|
| `pro_ops/element_wise/test_add.py` | 单Vector Section、二维Tile循环和按核跨步分工 |
| `pro_ops/matmul/test_matmul_8k_example.py` | 单Cube Section、M/N输出Tile循环和K循环 |
| `pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset_dynamic.py` | Cube Section中的动态M/N/K、K尾块和`phase`放置 |
| `pro_ops/vf_api/test_softmax_tile_group_vf.py` | 单Vector Section、动态N轴和常规三遍Softmax状态的生命周期；该样例不包含在线更新 |
| `pro_ops/fa/test_fa_with_mask.py` | Cube/Vector多Section、多层循环、mask分支和N-buffer轮转；仅在目标算子具有相近结构时参考 |

示例中的Tile尺寸、地址和缓冲深度不直接复用。接口签名和参数约束以当前API文档为准。

## R4输出

在DESIGN.md中记录：

- 结果单元；
- Section代码结构，以及每个Module所在的Section；
- 每个Module内的循环嵌套；
- 跨Tile状态的初始化、更新和使用位置；
- 动态循环上界；
- 分核信息的获取位置；
- 参考的官方指定算子及借鉴的循环结构。
