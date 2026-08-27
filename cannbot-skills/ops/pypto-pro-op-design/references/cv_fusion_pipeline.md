# CV 融合算子手动预加载流水设计

本文介绍 PyPTO Pro 中 Cube/Vector 融合算子的手动流水写法。核心做法是把一次完整计算拆成多个流水阶段，让 Cube 和 Vector 在同一轮中处理不同任务，并使用跨核事件保护共享缓冲区。

## 1. 流水的基本组织方式

手动流水的核心是任务错位。先把计算划分成连续的流水任务，再安排每个阶段相对任务编号的延迟。Cube 和 Vector 在同一轮处理不同编号的任务，同一任务内部仍按原有数据依赖顺序执行。

例如两阶段 `C → V` 流水的执行顺序如下：

```text
流水轮次     0       1       2       3
Cube       C(0)    C(1)    C(2)    C(3)
Vector       -     V(0)    V(1)    V(2)
```

同一任务仍按 `C(0) → V(0)` 的顺序执行；不同任务可以重叠，例如第 1 轮安排 `C(1)` 和 `V(0)`。后文代码中的 `tick` 表示流水轮次。

实现流水时，根据数据路径选择下面的机制：

| 使用条件 | 对应机制 |
|---|---|
| Cube 和 Vector 之间传递数据 | 使用 `set_cross_core` 和 `wait_cross_core` 表示数据就绪和缓冲槽位释放 |
| 多个任务的数据在流水中同时存在 | 为共享数据设置多个缓冲槽位，并按 `task_id` 轮转 |
| 延迟阶段需要前面任务的坐标、偏移、有效 shape 或状态索引 | 使用 `pl.struct_array` 保存任务上下文 |
| Cube 或 Vector 执行区内部使用 TileGroup，并涉及多个流水通道 | 使用 `make_tile_group` 提供多个 Tile 槽位，通过 `next()` 或显式下标选择槽位；`@pl.jit(auto_mutex=True)` 根据 mutex 信息管理执行区内部的跨流水通道依赖 |

因此，任务划分和阶段延迟是手动流水的主体。跨核事件、多槽缓冲、上下文循环缓冲和 `auto_mutex` 按具体数据路径使用。简单的两阶段流水可以直接从 `task_id` 算出地址和槽位，此时不需要单独的上下文循环缓冲。

安排阶段延迟前，先找出第一阶段每次交给下一阶段的那份数据。每产生一份这样的数据，连续编号 `task_id` 加 1。后面的所有阶段都沿用这个编号处理同一份数据。

例如，Cube 每次计算一个 `[TM, TN]` 的输出 Tile，Vector 随后处理这个 Tile：

```python
for m_idx in ...:
    for n_idx in ...:
        cube_compute(m_idx, n_idx)    # 生成一个 [TM, TN] Tile
        vector_compute(m_idx, n_idx)  # 处理同一个 Tile
```

假设每个 `m_idx` 下有 3 个 `n_idx`，那么每个 `(m_idx, n_idx)` 对应一份完整的传递数据，编号如下：

| `task_id` | `m_idx` | `n_idx` | 含义 |
|---:|---:|---:|---|
| 0 | 0 | 0 | 第 0 个输出 Tile |
| 1 | 0 | 1 | 第 1 个输出 Tile |
| 2 | 0 | 2 | 第 2 个输出 Tile |
| 3 | 1 | 0 | 第 3 个输出 Tile |

判断 `task_id` 在哪里递增，可以直接看哪个循环变量变化后会生成下一份需要在 Cube 和 Vector 之间传递的数据。上例采用跨外层循环连续运行的流水，内层 `n_idx` 每推进一次就生成下一份数据；内层结束后，`m_idx` 变化，编号继续递增。采用分段流水时，当前分段的后续阶段全部执行完成后，可以重新从 0 编号。

设计时按下面三个问题确定编号位置：

1. 第一阶段一次生成多大范围的数据，例如一个输出 Tile、一个行块或一个归约块。
2. 哪些循环索引共同确定这份数据在输入和输出中的位置。
3. 哪条代码路径表示这份数据真正进入流水；只在这条路径上将 `task_id` 加 1。

Cube 和 Vector 各自运行自己的 Section 和循环，但两侧必须用相同的遍历顺序生成相同的编号。编号确定以后，阶段错位才能写成“第 `T` 轮中，Cube 处理编号 `T`，Vector 处理编号 `T-1`”。多阶段流水中的同一份数据经过 `s1 → s2 → s3 → ...` 时，始终使用同一个 `task_id`。

循环中存在条件过滤时，无效位置不占用编号：

```python
task_id = 0
for m_idx in ...:
    for n_idx in ...:
        if tile_is_valid(m_idx, n_idx):
            process(task_id, m_idx, n_idx)
            task_id = task_id + 1
```

后续阶段还需要 `m_idx`、`n_idx`、有效 shape 或地址偏移时，按以下方式处理：

- 遍历规则固定，并且坐标可以由 `task_id` 直接反算：后续阶段重新计算坐标。
- 存在动态边界、条件过滤或复杂的多层循环：数据进入流水时，将编号和坐标写入 `pl.struct_array`；后续阶段读取该编号对应的上下文。

## 2. 两阶段 C→V 流水

两阶段流水中，Cube 产生任务结果，Vector 消费同一任务的结果。共享数据通常使用两个槽位轮转：

```text
任务       0    1    2    3
槽位       0    1    0    1
```

每个槽位使用两类事件：

- 就绪事件：Cube 写完后发送，Vector 读取前等待。
- 释放事件：Vector 读完后发送，Cube 再次写入该槽位前等待。

Vector 在主循环开始前先为每个空槽位发送一次释放事件。这些事件表示槽位初始可写。

> 下列代码只表示调度关系。`cube_produce`、`vector_consume` 和 `wait_last_releases` 是目标算子需要展开的计算逻辑，并非 PyPTO API。

```python
DEPTH = 2
READY_IDS = (0, 1)
RELEASE_IDS = (2, 3)


@pl.jit(auto_mutex=True)
def fused_kernel(x, out):
    shared = pl.make_tile_group(
        type=pl.TileType(
            shape=[...], dtype=..., target_memory=pl.MemorySpace.Vec
        ),
        addrs=[...],
        mutex_ids=[...],
    )

    with pl.section_cube():
        for task_id in pl.range(0, task_count):
            slot = task_id % DEPTH

            # 等待 Vector 释放当前槽位。
            pl.system.wait_cross_core(
                pipe=CUBE_WAIT_RELEASE_PIPE,
                event_id=RELEASE_IDS[slot],
            )
            cube_produce(shared[slot], task_id)

            # 通知 Vector 当前槽位的数据已经写好。
            pl.system.set_cross_core(
                pipe=CUBE_SET_READY_PIPE,
                event_id=READY_IDS[slot],
            )

        # 等待最后一批槽位被 Vector 读取完成。
        wait_last_releases(...)

    with pl.section_vector():
        # 两个槽位初始都处于可写状态。
        pl.system.set_cross_core(pipe=VECTOR_SET_RELEASE_PIPE, event_id=RELEASE_IDS[0])
        pl.system.set_cross_core(pipe=VECTOR_SET_RELEASE_PIPE, event_id=RELEASE_IDS[1])

        for task_id in pl.range(0, task_count):
            slot = task_id % DEPTH

            # 等待 Cube 写好当前槽位。
            pl.system.wait_cross_core(
                pipe=VECTOR_WAIT_READY_PIPE,
                event_id=READY_IDS[slot],
            )
            vector_consume(shared[slot], task_id, out)

            # 通知 Cube 当前槽位可以再次写入。
            pl.system.set_cross_core(
                pipe=VECTOR_SET_RELEASE_PIPE,
                event_id=RELEASE_IDS[slot],
            )
```

四个 pipe 分别根据同步点紧邻的数据操作确定：`CUBE_WAIT_RELEASE_PIPE` 对应下一次覆盖槽位前的写操作，`CUBE_SET_READY_PIPE` 对应最后一次写操作，`VECTOR_WAIT_READY_PIPE` 对应第一次读操作，`VECTOR_SET_RELEASE_PIPE` 对应最后一次读操作。同一 Section 中的这些同步点也可能使用不同 pipe；紧邻操作位于同一流水通道时，可以使用相同的 pipe。完整的事件配对规则见[跨核同步](cross_core_synchronization.md)。

## 3. 四阶段 C→V→C→V 流水

当前 FA 手动预加载样例采用以下四个阶段：

```text
QK(Cube) -> P(Vector) -> PV(Cube) -> GU(Vector)
```

四个阶段处理同一任务时存在先后依赖。手动流水让它们在不同轮次启动：

| 阶段 | 执行区 | 相对当前轮次的延迟 | 第 `T` 轮处理的任务 |
|---|---|---:|---|
| QK | Cube | 0 轮 | `T` |
| P | Vector | 1 轮 | `T-1` |
| PV | Cube | 2 轮 | `T-2` |
| GU | Vector | 3 轮 | `T-3` |

这段手动流水的阶段延迟为 `0/1/2/3`。预加载轮数和阶段延迟分别设置：阶段延迟决定当前轮次中各阶段处理哪个任务，预加载轮数决定前级计算可以提前准备多少轮数据。当前样例可以使用 2 轮或 3 轮预加载，当前性能场景下 3 轮效果更好。

采用 3 轮预加载时，流水逐步进入稳定运行：

| 流水轮次 | Cube 执行的阶段 | Vector 执行的阶段 |
|---:|---|---|
| 0 | `QK(0)` | 空闲 |
| 1 | `QK(1)` | `P(0)` |
| 2 | `QK(2)`、`PV(0)` | `P(1)` |
| 3 | `QK(3)`、`PV(1)` | `P(2)`、`GU(0)` |
| 4 | `QK(4)`、`PV(2)` | `P(3)`、`GU(1)` |

括号中的数字是流水任务编号。例如第 3 轮中，Cube 同时推进任务 3 的 QK 和任务 1 的 PV，Vector 同时推进任务 2 的 P 和任务 0 的 GU。

- 第 0～2 轮：流水启动段。
- 第 3 轮起：稳定运行段，四个阶段处理不同任务。
- 最后一个真实 QK 任务之后：继续推进 3 轮，依次完成剩余的 P、PV 和 GU。

### 3.1 保存延迟任务的信息

最大阶段延迟为 3 轮，上下文循环缓冲的深度取 `最大阶段延迟 + 1`，即 4：

```python
ctx_arr = pl.struct_array(
    4,
    "PipelineCtx",
    task_id=0,
    tile_i=0,
    tile_j=0,
    valid_m=0,
    valid_n=0,
    state_slot=0,
    is_valid=0,
)
```

当前轮次将任务信息写入：

```python
current_ctx = ctx_arr[tick % 4]
fill_context(current_ctx, ...)
```

各延迟阶段读取的槽位为：

```text
延迟 1 轮：ctx_arr[(tick + 3) % 4]，对应任务 T-1
延迟 2 轮：ctx_arr[(tick + 2) % 4]，对应任务 T-2
延迟 3 轮：ctx_arr[(tick + 1) % 4]，对应任务 T-3
```

上下文保存后续阶段使用的全部任务信息，包括循环坐标、Tensor 偏移、有效 shape、分支选择、状态槽位、事件槽位和 TileGroup 槽位。Cube 和 Vector 各自维护一份上下文循环缓冲，两侧使用相同的任务生成规则填写内容。

### 3.2 Cube 执行区

Cube 执行区负责当前任务的 QK，以及延迟 2 轮任务的 PV：

```python
tick = 0
ctx_arr = pl.struct_array(4, "CubeCtx", ...)

for task_desc in logical_tasks_with_cube_extra_rounds(2):
    current_ctx = ctx_arr[tick % 4]
    current_ctx.is_valid = 0
    if task_desc.is_real:
        fill_cube_ctx(current_ctx, task_desc, tick)
        current_ctx.is_valid = 1
        compute_qk(current_ctx, ...)

    if tick >= 2:
        pv_ctx = ctx_arr[(tick + 2) % 4]
        if pv_ctx.is_valid:
            compute_pv(pv_ctx, ...)

    tick = tick + 1
```

`is_real`、`is_valid`、`logical_tasks_with_cube_extra_rounds` 和 `fill_cube_ctx` 用于说明控制关系。实际 PyPTO 代码可使用 `ki < real_count`、`tick >= 2` 和增加 2 轮的循环上界表达相同逻辑。

### 3.3 Vector 执行区

Vector 执行区负责延迟 1 轮任务的 P，以及延迟 3 轮任务的 GU：

```python
# QK 和 PV 的共享槽位初始可写。
set_all_release_events(QK_RELEASE_IDS)
set_all_release_events(PV_RELEASE_IDS)

tick = 0
ctx_arr = pl.struct_array(4, "VectorCtx", ...)

for task_desc in logical_tasks_with_vector_extra_rounds(3):
    current_ctx = ctx_arr[tick % 4]
    current_ctx.is_valid = 0
    if task_desc.is_real:
        fill_vector_ctx(current_ctx, task_desc, tick)
        current_ctx.is_valid = 1

    if tick >= 1:
        p_ctx = ctx_arr[(tick + 3) % 4]
        if p_ctx.is_valid:
            compute_p(p_ctx, ...)

    if tick >= 3:
        gu_ctx = ctx_arr[(tick + 1) % 4]
        if gu_ctx.is_valid:
            compute_gu(gu_ctx, ...)

    tick = tick + 1
```

Cube 最后一个阶段是延迟 2 轮的 PV，因此最后一个新任务进入后，Cube 执行区再运行 2 轮。Vector 最后一个阶段是延迟 3 轮的 GU，因此 Vector 执行区再运行 3 轮。这些轮次只完成已经进入流水的后续阶段，不创建新的任务上下文。

## 4. 通用多阶段 CV 并行流水

多阶段流水适用于一次数据处理需要在 Cube 和 Vector 之间多次交接的情况。实现过程分为四步：拆分阶段、计算阶段延迟、按延迟执行各阶段、为跨阶段数据配置缓冲和事件。

### 4.1 拆分阶段

一个阶段包含同一执行域上连续执行的一组计算。数据从 Cube 交给 Vector，或者从 Vector 交回 Cube 时，进入下一个阶段。例如：

```text
s1(Cube)：生成中间数据 A
    ↓ A 交给 Vector
s2(Vector)：读取 A，生成中间数据 B
    ↓ B 交回 Cube
s3(Cube)：读取 B，生成中间数据 C
    ↓ C 交给 Vector
s4(Vector)：读取 C，生成最终结果
```

整理后，阶段链在两个执行域之间交替：

```text
s1(Cube) -> s2(Vector) -> s3(Cube) -> s4(Vector) -> ...
```

相邻计算都在 Cube 上执行时，将它们放入同一个 Cube 阶段；相邻计算都在 Vector 上执行时，将它们放入同一个 Vector 阶段。每个阶段需要记录以下内容：

- 输入来自哪个阶段，输出交给哪个阶段。
- 读取和写入的 Tile 或共享缓冲区。
- 该阶段所属的 Cube 或 Vector Section。
- 数据写完后发送的就绪事件，以及数据读完后发送的释放事件。

同一个 `task_id` 表示同一份数据完整经过所有阶段。例如 `s1(5)` 生成的数据由 `s2(5)` 读取，随后传给 `s3(5)` 和 `s4(5)`。流水并行只改变各阶段开始处理编号 5 的时间，不改变这条数据依赖关系。

### 4.2 计算每个阶段的延迟

`delay` 表示一个阶段比第一阶段晚多少轮处理同一编号的数据。阶段延迟为 0 时，第 `T` 轮处理编号 `T`；阶段延迟为 1 时，第 `T` 轮处理编号 `T-1`。

设预加载轮数 `preload` 为正整数，阶段按声明顺序计算 `delay`：

1. 整条流水的第一个阶段：`delay = 0`。
2. 某个执行域第一次出现：`delay = 上一个阶段的 delay + 1`。
3. 某个执行域后续再次出现：`delay = 同一执行域上一个阶段的 delay + preload`。

对应的计算过程为：

```python
last_delay = {"cube": None, "vector": None}

for stage_idx, stage in enumerate(stages):
    core = stage.core
    if stage_idx == 0:
        stage.delay = 0
    elif last_delay[core] is None:
        stage.delay = stages[stage_idx - 1].delay + 1
    else:
        stage.delay = last_delay[core] + preload
    last_delay[core] = stage.delay
```

第一阶段的延迟为 0。另一个执行域第一次出现时，比前一阶段晚 1 轮。同一执行域再次出现时，与该执行域的上一个阶段间隔 `preload` 轮。

以四阶段流水、`preload=2` 为例：

```text
s1(Cube)   delay = 0
s2(Vector) delay = s1.delay + 1 = 1
s3(Cube)   delay = s1.delay + 2 = 2
s4(Vector) delay = s2.delay + 2 = 3
```

因此第 3 轮中，四个阶段分别处理 `s1(3)`、`s2(2)`、`s3(1)` 和 `s4(0)`。Cube 处理编号 3 和 1，Vector 处理编号 2 和 0。

例如 8 个阶段、`preload=3`：

| 阶段 | 执行域 | 同执行域上一个阶段 | delay |
|---|---|---|---:|
| s1 | Cube | 无 | 0 |
| s2 | Vector | 无 | 1 |
| s3 | Cube | s1 | 3 |
| s4 | Vector | s2 | 4 |
| s5 | Cube | s3 | 6 |
| s6 | Vector | s4 | 7 |
| s7 | Cube | s5 | 9 |
| s8 | Vector | s6 | 10 |

### 4.3 按延迟生成执行时序

第 `T` 轮中，阶段 `si` 处理的迭代编号为：

```text
iteration = T - delay[si]
```

仅当 `0 <= iteration < iteration_count` 时执行该阶段。`iteration < 0` 表示该阶段还没有等到第一份输入；`iteration >= iteration_count` 表示没有新的数据进入该阶段。最后一个新任务进入后，循环继续运行，依次完成已经进入流水的后续阶段。

下面是四阶段、`preload=2`、共 5 份数据时的完整时序：

| 流水轮次 | `s1 Cube, d=0` | `s2 Vector, d=1` | `s3 Cube, d=2` | `s4 Vector, d=3` |
|---:|---|---|---|---|
| 0 | `s1(0)` | — | — | — |
| 1 | `s1(1)` | `s2(0)` | — | — |
| 2 | `s1(2)` | `s2(1)` | `s3(0)` | — |
| 3 | `s1(3)` | `s2(2)` | `s3(1)` | `s4(0)` |
| 4 | `s1(4)` | `s2(3)` | `s3(2)` | `s4(1)` |
| 5 | — | `s2(4)` | `s3(3)` | `s4(2)` |
| 6 | — | — | `s3(4)` | `s4(3)` |
| 7 | — | — | — | `s4(4)` |

第 0～2 轮依次启动后续阶段；第 3～4 轮四个阶段都有有效数据；第 5～7 轮不再向 `s1` 加入新数据，只完成剩余阶段。这个时序转换为调度关系时，每个阶段都使用自己的 `delay` 计算编号。以下伪代码不表示实际的 PyPTO API：

```python
for tick in pl.range(0, iteration_count + max_delay):
    for stage in stages:
        task_id = tick - stage.delay
        if 0 <= task_id and task_id < iteration_count:
            run_stage(stage, task_id)
```

实际代码中 Cube 和 Vector 位于各自的 Section，可以分别保留属于本执行域的阶段。例如 Cube Section 执行 `s1` 和 `s3`，Vector Section 执行 `s2` 和 `s4`；两侧使用相同的 `task_id` 编号规则和阶段延迟表。

设 `max_delay` 为所有阶段的最大延迟：

- 使用统一环形上下文缓冲时，深度取 `max_delay + 1` 可以覆盖所有阶段延迟；按执行域或字段生存期分别设置上下文时，可以单独计算所需深度。
- 流水从第 0 轮开始启动。
- 最后一个新迭代进入后，再推进 `max_delay` 轮，完成剩余阶段。
- 总轮数为 `iteration_count + max_delay`。

### 4.4 按数据编号配置上下文、缓冲和事件

上下文按正在处理的数据编号访问。阶段 `si` 在第 `T` 轮读取编号 `T-delay[si]` 对应的坐标、有效 shape 和缓冲槽位。使用循环上下文缓冲时，槽位可以写成：

```text
context_slot = (T - delay[si]) % context_depth
```

每一条跨执行域的数据边都单独配置共享缓冲和事件。例如 `s1(Cube) → s2(Vector)` 传递数据 A，`s2(Vector) → s3(Cube)` 传递数据 B，则 A 和 B 分别计算自己的槽位数量、就绪事件和释放事件。编号为 `k` 的一次交接按以下顺序执行：

1. 生产者等待 `slot(k)` 的释放事件，然后写入编号 `k` 的数据。
2. 生产者写完后发送 `slot(k)` 的就绪事件。
3. 消费者处理编号 `k` 前等待同一个槽位的就绪事件，然后读取数据。
4. 消费者完成最后一次读取后发送释放事件，允许后续编号复用该槽位。

作为消费者的一侧在进入主循环前为初始空槽发送释放事件；消费者可能是 Vector，也可能是 Cube。事件编号、共享缓冲槽位和上下文槽位都由正在处理的数据编号 `k = T-delay[si]` 推导，两侧因此能够选中同一份数据。

`preload` 改变同一执行域各阶段之间的距离，也会改变 `max_delay`、末尾需要补充的轮数、上下文深度和同时处于流水中的数据量。每组共享数据需要多少槽位，根据该数据从第一次写入到最后一次读取之间的生存时间单独计算。

## 5. 预加载轮数和缓冲深度

预加载轮数表示前级计算可以提前准备多少轮数据。当前四阶段样例可以预加载 2 轮或 3 轮，当前性能场景下 3 轮效果更好。预加载轮数增加后，同时保留的数据更多，片上缓冲和事件槽位按实际数据生存期配置。

缓冲深度根据数据生存期计算。对每组跨阶段数据，记录：

```text
第一次写入 -> 发送就绪事件 -> 消费者读取 -> 发送释放事件
```

同一物理槽位的下一次写入安排在上一轮读取完成之后。预加载轮数和缓冲深度是两个不同的设计量，需要联合校验：阶段延迟决定数据在流水中保留多久，缓冲深度必须覆盖这段时间内可能同时存在的数据数量。

上下文缓冲也单独计算。只保存后续阶段实际使用的信息，深度覆盖从写入到最后一次读取的编号范围。迭代坐标能够由 `task_id` 直接得到时，可以省略上下文缓冲。

所有数据缓冲和状态缓冲同时计入 UB、L1、L0 地址占用。跨核事件按同一时间需要保留的方向和槽位统计，并验证复用后的全部 `event_id` 都在有效范围内。

## 6. 跨外层循环连续运行

相邻外层迭代可以共用已经启动的流水。`task_id` 在外层循环之间持续递增，只在当前物理核负责的最后一个外层迭代之后增加用于完成剩余阶段的轮次：

```python
extra_rounds = 0
if work_id == work_end - 1:
    extra_rounds = SECTION_LAST_DELAY

for inner_tick in pl.range(0, real_inner_tasks + extra_rounds):
    ...
```

Cube Section 的 `SECTION_LAST_DELAY` 取最后一个 Cube 阶段的延迟，Vector Section 则取最后一个 Vector 阶段的延迟。四阶段 `0/1/2/3` 时，Cube 追加 2 轮，Vector 追加 3 轮。

设计可以选择以下一种组织方式：

- 连续流水：相邻外层任务共用流水，上下文和状态按外层任务区分；最后一个外层任务结束后，再完成剩余阶段。
- 分段流水：每个外层任务分别启动，并在进入下一个外层任务前完成当前任务的全部阶段；每段都会重新经历启动过程。

连续流水中，Cube 和 Vector 都在最后一个外层任务结束后执行各自剩余的阶段。分段流水中，两侧在每个外层任务边界完成当前任务的全部阶段。

## 7. 一个 AIC 配两个 AIV 时的编号映射

本文引用的官方混合 Kernel 样例按一个 AIC 配两个 AIV 组织。目标产品和 Kernel 采用这种组织方式时，设计中记录以下映射：

- AIC 和 AIV 使用的物理 `core_id`。当前样例使用 `get_block_idx() // get_subblock_num()`。
- `get_subblock_idx()` 对应的行列切分和共享缓冲区视图。
- 两个 AIV 参与的事件方向及每个事件的执行次数。
- 尾块中某个 AIV 没有有效数据时采用的共同同步路径。

两个 AIV 的任务分支和 AIC 的等待次数保持配对。数据有效性判断可以控制实际计算，跨核事件放在各参与方共同执行的控制路径中。

## 8. DESIGN.md 中记录的内容

| 设计项 | 记录内容 |
|---|---|
| 流水编号 | 第一阶段每次交给下一阶段的数据范围、产生下一份数据的循环索引、`task_id`递增位置及Cube/Vector两侧的编号关系 |
| 阶段链 | 各阶段的声明顺序、所属执行域和跨阶段数据依赖 |
| 延迟计算 | `preload`、同执行域上一个阶段、每个阶段的`delay`及计算式 |
| 阶段时序 | 每轮各阶段处理的数据编号和有效执行条件 |
| 预加载 | 候选预加载轮数、计划采用的轮数及性能选择依据 |
| 上下文循环缓冲（按需） | 使用时记录字段、深度、当前写槽、各阶段读槽表达式和信息生存期；未使用时记录任务信息的直接推导方式 |
| 启动、稳定运行和末尾剩余阶段 | 两侧的启动条件、稳定运行时序、最后一个新任务进入后需要继续执行的轮数和最后输出位置 |
| 共享缓冲区 | MemorySpace、shape、layout、深度、地址、生产和消费 API、槽位表达式 |
| 事件协议 | 就绪/释放方向、初始释放事件、`set`/`wait` 的 pipe、事件编号和槽位复用条件 |
| 递推状态 | 最大值、累加和、输出等状态的索引、初始化、更新和外层任务隔离方式 |
| 边界场景 | 0 个任务、1 个任务、任务数少于预加载轮数、尾块和稀疏跳块的处理 |
| 验证结果 | 精度、最后结果写回、事件配对、性能数据和稳定运行阶段的 Cube/Vector 重叠情况 |

## 9. 检查清单

### 9.1 代码和设计检查

- [ ] Cube 和 Vector 对同一有效迭代使用相同的连续编号。
- [ ] 多阶段链在 Cube 和 Vector 之间交替，相邻同执行域计算已经合并。
- [ ] 每个阶段的`delay`符合第 4.2 节的计算规则。
- [ ] 每个阶段的延迟、执行条件和上下文索引对应明确的数据编号。
- [ ] 使用上下文循环缓冲时，深度覆盖最大阶段延迟，写槽位晚于旧上下文最后一次读取。
- [ ] 每个跨核缓冲槽位都有明确的就绪和释放条件。
- [ ] 首次写槽位前已经发送初始释放事件，或首次写采用单独的启动路径。
- [ ] 最后一个新任务进入后，Cube 和 Vector 分别继续运行各自最后阶段对应的轮数。
- [ ] 两个 AIV 在所有边界分支中的事件次数与 AIC 的等待次数配对。
- [ ] TileGroup 的 `mutex_id` 和跨核同步的 `event_id` 分别规划。

### 9.2 运行检查

- [ ] 覆盖 1 个任务、2 个任务、任务数不超过最大阶段延迟、完整稳定运行和非整除尾块。
- [ ] 覆盖稀疏和条件分支，所有 `set` 与 `wait` 均有对应调用。
- [ ] 最后一个延迟阶段和输出 store 已执行，最后一批槽位完成释放。
- [ ] 同一组输入多次运行，结果和执行状态保持稳定。
- [ ] PMU 或流水图显示稳定运行时 Cube 和 Vector 同时处理不同编号的数据。
- [ ] 性能数据覆盖计划采用的各个预加载轮数。
- [ ] 性能对比使用相同的 shape、核数、预热次数和计时方式。

## 10. 官方样例

以下 4 个用例来自[官方指定算子样例清单](../../pypto-pro-material-explore/references/official_samples.md)。gym 工作流会将它们缓存到 `$PYPTO_DEVKIT_DIR/pro_ops/`，agent 可以直接读取这些路径。

| 官方样例 | 缓存路径 | 参考内容 |
|---|---|---|
| fa_perf_tkv_preload | `pro_ops/fa/test_fa_perf_tkv_preload_dn_vf_bufid_dynrank.py` | 四阶段循环、上下文缓冲和跨核事件 |
| fa_tilingkey_attn_mask | `pro_ops/fa/test_fa_tilingkey_attn_mask.py` | mask、tilingkey 分支与手动跨核同步 |
| fa_with_mask | `pro_ops/fa/test_fa_with_mask.py` | 多缓冲、`auto_mutex`、末尾阶段执行和手动跨核事件 |
| flex_attention | `pro_ops/fa/test_flex_attention.py` | 稀疏任务过滤、外层任务连续流水和末尾阶段执行 |

具体设计选择与目标算子结构最接近的官方样例。阶段延迟、地址、事件编号和缓冲深度根据目标算子的实际数据生存期重新计算。
