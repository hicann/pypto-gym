# 片上空间布局

R3负责把R2已经确定的Tile放到具体片上地址，并验证对齐、容量和地址复用。R3不修改Tile的shape、dtype、layout或缓冲深度；布局失败时，先说明哪个空间超限以及相差多少字节，再回到R2调整Tile，必要时回R1更换API链。

接口行为以`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/`下的`resource_management/make_tile.md`、`resource_management/make_tile_group.md`、`basic_data_structures/TileType.md`和`basic_data_structures/MemorySpace.md`为准。容量使用当前`EXPLORE_REPORT.md` §7记录的目标平台数值。

## 输入与交付物

开始R3前，R2应当已经确定：

- 每个Tile的物理规格和MemorySpace；
- 使用`make_tile`还是`make_tile_group`；
- TileGroup的槽位数、mutex配置和访问方式；
- 根据当前数据流推导的Tile生命周期初稿；R4确定循环、R6确定同步后，如生命周期发生变化，必须回查R3的地址复用和峰值容量；
- API要求的额外物理空间。

R3的结果写入DESIGN.md §3，包括逐槽位地址表、逐空间容量结果和地址复用依据；表格格式见[设计模板](../templates/design-template.md)。

## 布局流程

1. 按MemorySpace对Tile分组。
2. 展开`make_tile_group`的全部物理槽位，计算每个槽位的大小和地址区间。
3. 在各自的地址空间内分配地址，并检查每个槽位的首地址对齐。
4. 对有意复用的地址检查生命周期和同步关系。
5. 计算每个空间的最高地址上界，并与当前平台容量比较。
6. 对涉及MX、Scaling、Bias或并行L1访问的设计执行对应专项检查。

如果容量超限，先消除无用地址空洞，再检查能否依据生命周期安全复用。仍然超限时回R2调整Tile大小、缓冲深度或生命周期；由API临时空间导致的超限再回R1调整API链。

## 1. MemorySpace与地址域

不同MemorySpace独立寻址和限容。同一个数值地址出现在不同空间中，不表示它们使用同一块物理存储。

| MemorySpace | 物理位置 | 用途 |
|---|---|---|
| `Vec` | UB | Vector输入、输出和中间量 |
| `Mat` | L1 | GM与Cube本地存储之间的矩阵暂存 |
| `Left` | L0A | Cube左操作数 |
| `Right` | L0B | Cube右操作数 |
| `Acc` | L0C | Cube累加结果 |
| `Scaling` | Fixpipe Buffer | FIX通路的per-channel随路量化或反量化参数 |
| `Bias` | BiasTable Buffer | 矩阵计算的偏置 |
| `ScaleLeft` | L0A的MX scale地址域 | `matmul_mx`左操作数的scale |
| `ScaleRight` | L0B的MX scale地址域 | `matmul_mx`右操作数的scale |

在A5的1:2混合Kernel中，两个AIV各自拥有本地Vec地址域。因此AIV0和AIV1的UB分别检查；两侧使用相同的本地地址不会冲突，也不能把两侧占用相加后与单个AIV的UB容量比较。

## 2. 计算物理槽位

地址必须是非负的编译期整数，并统一写成左闭右开区间`[addr, addr + size)`。两个区间首尾相接不算重叠。

`make_tile_group`使用单个基地址时，第`i`个槽位按下式展开：

```text
slot_size = prod(static_shape) × max(1, ceil(dtype_bits / 8))
slot[i].addr = base + i × slot_size
slot[i].range = [slot[i].addr, slot[i].addr + slot_size)
```

使用地址列表时，列表长度必须等于`depth`，每个元素指定一个槽位的首地址。无论使用哪种写法，地址表都要逐槽位展开。

以下规则会影响物理占用：

- shape的每一维必须是正的编译期整数；
- INT4等亚字节dtype按当前前端规则为每个元素预留至少1字节；
- `valid_shape`只表示有效数据范围，不会缩小物理槽位；
- layout、fractal、pad或compact需要额外空间时，应由R2通过Tile规格表达，并根据对应API文档确认；
- `make_tile_group`没有单独的`size`参数，所需物理空间必须由R2的Tile规格表达；
- `make_tile`未指定`size`时使用相同的缺省大小；只有API明确要求更大预留时才显式指定`size`。

## 3. 检查首地址对齐

当前Pro前端对编译期地址采用以下对齐要求：

| MemorySpace | 首地址对齐 |
|---|---:|
| `Vec` | 32B |
| `Mat` | 32B |
| `Left` | 512B |
| `Right` | 512B |
| `Acc` | 64B |
| `ScaleLeft` | 32B |
| `ScaleRight` | 32B |

TileGroup使用单个基地址时，要检查每个`base + i × slot_size`，不能只检查`base`。如果后续槽位不对齐，应调整Tile物理规格或改用显式地址列表。

`Scaling`和`Bias`还受具体搬运API约束。`Bias`数据需要先从GM加载到Mat，再搬到Bias；`Scaling`只用于FIX通路的per-channel参数，与quant/dequant接口的scale以及MX使用的`ScaleLeft`/`ScaleRight`都不是同一空间。当前A5接口要求`Mat → Scaling`的源Tile为1行、目的dtype为`DT_INT64`或`DT_UINT64`，目的地址和搬运量按128B对齐；`Mat → Bias`的源Tile为1行，目的地址和搬运量按64B对齐。两类单次搬运均不超过4 KiB。使用这些空间时须再核对目标版本API文档。

## 4. 分配与复用

同一MemorySpace内同时存活的物理槽位必须使用不重叠的地址区间。一个`depth=N`的TileGroup包含N个物理槽位，容量按N份计算。

两个逻辑Tile只有同时满足以下条件才能复用地址：

1. 位于同一MemorySpace，复用区间能够容纳两者的完整物理大小；
2. 生命周期不重叠，或者同步能够保证新写入发生在旧数据最后一次读取之后；
3. 核内跨Pipe复用具有正确的mutex关系或显式核内同步；
4. 跨Cube/Vector复用具有R6定义的READY/RELEASE同步。

`mutex_id`是同步元数据，不负责分配地址，也不检查容量。当前取值范围为`[0, 31]`；同一Tile内不能重复，同一TileGroup中每个Tile携带的ID数量应一致。不同Tile可以在确有互斥关系时复用同一个ID。Cross-core的`event_id`由R6独立设计。

不同layout的Tile共用地址不会产生layout转换。只有API允许以目标layout解释同一物理数据，并且大小与生命周期均满足复用条件时，才能把它们设计为别名。

## 5. 容量与A5 L0C→UB通路

每个MemorySpace分别计算：

```text
最高地址上界 = max(addr + size)
最高地址上界 <= EXPLORE_REPORT.md §7中的该空间容量
```

存在地址空洞时，可以同时统计有效预留字节数用于分析，但容量是否通过由最高地址上界决定。

A5使用`pl.move(..., acc_to_vec_mode=...)`将L0C结果搬到UB时，还要分别计算每个AIV在流水稳定阶段的UB峰值。计算应包含该AIV同时存活的全部物理槽位、共享数据、mask/PSE、上下文、临时Tile和API额外空间，不能按尾块`valid_shape`缩减。

如果某个AIV的UB超限，先回R2缩小Tile、减少缓冲槽位或调整`AccToVecMode`。仍无法满足容量，或者`pl.move`不支持当前参数时，回R1改为GM workspace通路，并让Vector按可放入UB的大小分块读取。GM workspace不计入片上空间容量，但要在DESIGN中单独记录，并在R6重新设计同步。

## 6. 专项检查

### MX scale地址

MX数据与scale的地址按槽位满足：

```text
ScaleLeftAddr[i]  = LeftAddr[i]  >> 4
ScaleRightAddr[i] = RightAddr[i] >> 4
```

`ScaleLeft`和`ScaleRight`是独立逻辑地址域，其容量不从`Left`和`Right`的数据空间中扣除。

### L1 Bank冲突

完成地址、对齐、生命周期和容量检查后，对计划并行访问的Mat Tile检查L1 Bank冲突。地址位域和冲突规则见：

`$PYPTO_DEVKIT_DIR/docs/guide/programming_guide/pro/development/tile_based_python_programming/Cube_matrix_computation.md`

只检查实际会并行访问的Tile，并根据真实`load`/`move`关系判断。可以参考相同Tile形状和访问方式的官方样例，但仍要在本算子的地址表上验证。

## 完成条件

R3完成时应能从DESIGN.md §3确认：

- R2中的每个Tile和TileGroup槽位都有唯一记录；
- 每个地址、大小和区间均可计算，且满足对齐；
- 每个空间的最高地址上界不超过当前平台容量；
- 每处地址复用都有生命周期和同步依据；
- 使用A5双AIV、MX、Scaling、Bias或并行L1访问时，对应专项检查已经通过。
