# Module划分

Module是设计文档中对计算流程的逻辑划分，不是PyPTO Pro的语法结构。Module划分主要看数据依赖和Section边界，也需要注意片上资源是否足够。

1. 根据数据依赖找出必须分开的计算步骤；
2. 根据API和数据通路判断Section边界。Cube和Vector计算属于不同Section，也要拆成不同Module。

## 根据数据依赖确定边界

对于相邻的两步A和B，可以按下面的情况判断：

- B必须等A遍历完完整归约轴或其他完整数据范围，并基于A的最终结果开始一次新的遍历或独立计算阶段时，通常划分为两个Module；
- A处理完当前Tile后，B能在同一层循环内立即消费该Tile，并且两步属于同一个Section时，通常放在同一个Module；
- A和B之间只有少量同域计算，没有插入其他Section或独立计算阶段时，也可以放在同一个Module；
- B只是紧接着对A得到的归约结果做汇总或简单处理，不需要重新遍历完整数据范围时，可以继续放在A所在的Module。

这些是常用判断方式，不是硬性规则。A的结果是否需要保存不能单独决定Module边界：同一Module内也可能暂存中间结果，不同Module之间也可能通过片上空间直接传递数据。反过来，即使A和B可以连续执行，如果合并后UB等片上空间放不下同时存活的数据，也需要拆分Module。

### Softmax示例

以沿N轴分块的稳定Softmax为例，常规实现需要三次完整遍历，后两次遍历都依赖前一次得到的最终结果，因此通常按三次遍历划分：

```text
Module 1：遍历全部N Tile，得到每行的m = max(x)
Module 2：重新遍历全部N Tile，得到每行的s = sum(exp(x - m))
Module 3：重新遍历全部N Tile，写出y = exp(x - m) / s
```

对同一个行块，前一个Module遍历完全部N Tile后，下一个Module才能开始。这里的边界来自“基于最终结果重新遍历N轴”，不是因为`m`或`s`需要保存。

采用Online Softmax时，可以在一次遍历中同时更新最大值和指数和。处理完前k个N Tile后，每行维护：

- `m`：当前最大值；
- `s`：以当前`m`为基准的指数和，即`sum(exp(x - m))`。

```text
m_new = max(m, tile_max)
s_new = s * exp(m - m_new) + sum(exp(tile - m_new))
```

最终的`m`和`s`到遍历结束时才能确定，因此写输出时通常要重新读取各Tile：

```text
Module 1：在线遍历全部N Tile，得到最终m和s
Module 2：重新遍历全部N Tile，写出exp(x - m) / s
```

如果后续只对最终的`m`或`s`做一次汇总或简单运算，不再重新遍历N轴，这部分可以留在Module 1中。

Attention的在线归一化还会维护输出累加量。`m`更新时，已有的输出累加量也要按新的`m`缩放，再合入当前Tile。Attention应按完整的在线更新公式划分Module，不能直接套用上面的Softmax公式。

## 根据Section边界划分

数据依赖分析完成后，再根据API使用的执行域、内存空间和硬件通路判断相邻步骤之间是否存在Section边界：

- L1/L0A/L0B/L0C上的矩阵路径放在Cube Section，包括GM→L1、L1→L0A/L0B、矩阵乘和L0C写回；
- Vec（UB）上的数据搬运和向量计算放在Vector Section。使用VF时，外层Kernel负责GM与UB之间的搬运，`@pl.vector_function`负责UB与寄存器之间的数据交换和寄存器计算；
- `pl.quant`和`pl.dequant`使用V流水，输入、scale、offset和输出均为Vec Tile，因此放在Vector Section；
- `pl.move`或`pl.store`对Acc结果做随路量化或反量化时，操作走FIX流水，放在Cube Section。per-channel参数使用`MemorySpace.Scaling` Tile时，参数先从Mat（L1）搬到Scaling，再由FIX使用；
- UB→L1、L0C→UB等跨执行域搬运按当前`move`接口文档选择数据通路。

`Scaling`是量化参数使用的片上缓冲区，不是独立的Section。不能看到`MemorySpace.Scaling`就新建Module；仍要根据使用它的API和数据通路判断属于Cube还是Vector。

一个Module只能属于一个Section。相邻步骤分别属于Cube和Vector时，在两者之间划分Module。R0只记录每个Module属于Cube还是Vector；具体使用几个Section代码块、多个同域Module是否放在同一个Section，以及循环相对Section的位置，在R4确定。

Section接口和编程方式参见：

- `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/operation/controlflow/section_vector_section_cube.md`
- `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/operation/quantization/quant.md`
- `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/operation/quantization/dequant.md`
- `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/move.md`
- `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/SIMD-API/operation/memory_data_movement/store.md`
- `$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/Cube_matrix_computation.md`
- `$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/Reg_vector_computation.md`

### Cube与Vector组合算子示例

Matmul后接Vector后处理时，Section边界同时形成Module边界：

```text
Cube Module：A × B → intermediate
                           ↓
Vector Module：intermediate → 激活、归约或归一化 → output
```

两个Module可以由同一次Kernel启动完成。中间数据可以写到GM workspace，也可以走`move`接口支持的片上通路。R0记录数据的写入位置、读取位置和同步方向；具体同步点和event_id在R6确定，见[跨核同步设计](cross_core_synchronization.md)。

## R0输出

每个Module记录数学目标、Section、输入、输出和前置依赖：

| Module | 数学目标 | Section | 输入 | 输出 | 前置依赖 |
|---|---|---|---|---|---|
| Module 1 | ... | Cube/Vector | ... | ... | ... |

只有一个Module时，说明数据依赖和Section边界为何不需要继续拆分。跨Tile状态及其循环范围在R4补充。
