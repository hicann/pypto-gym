---
name: pypto-pro-precision-debug
description: 基于 PyPTO Pro Kernel 的设备侧 dump 数据与编译产物定位算子精度问题。用于 @pl.jit 算子最终输出不一致、局部 Shape 或尾块错误、单核正确但多核错误、结果错位、误差随归约长度放大、结果偶发不稳定等场景；通过 pl.dump_data、pl.printf 对齐 GM Tensor、Vec Tile、Acc（L0C）Tile 与写回结果，结合 kernel.cpp 指令级审查和错误规律分析，确定第一个出错的中间结果并定位根因。
---

# PyPTO Pro 精度定位

本技能通过设备侧 dump 数据与编译产物，定位 `@pl.jit` 算子精度问题的第一个出错环节并确定根因。

## 一、排查思路

### 精度问题的两类根因

精度问题最终都可归入以下两类。排查时先判断属于哪一类，能大幅缩小范围。

**逻辑错误**——确定性错误，同一输入每次运行错误位置和数值完全一致。大部分稳定复现的精度问题属于此类。

- 搬运参数错误：load/store/move 的 offset 计算错误，Tile/Tensor 的 shape、stride、order（转置）设置错误
- 数据类型处理错误：累加前过早 cast 回低精度（如 FP16）、Vector 侧手动累加用了 FP16 而非 FP32、rounding mode 不符
- layout 不匹配：ND 与 NZ 混淆、转置场景下 Mat 与 Left/Right 的 layout 配置不一致（A5 上 Left 默认 NZ、Right 默认 ZN；左矩阵转置时 Mat layout 需与 Left 相反（配 ZN），右矩阵转置时 Mat layout 需与 Right 相同（配 ZN），两者均需 `load` 的 `order=[1,0]`）
- Tiling 或 Core 分片错误：`pl.range` 的起始/步长/边界计算错误、`block_dim` 与 `pl.get_block_num()` 不一致、尾块未均衡
- API 参数错误：归约方向（`dim`）、mask、shape 不匹配（Pro 不支持广播，tile-tile 操作要求 shape 完全一致）、`scale` 随路量化比例配置
- 初始化遗漏：K 维累加首块误用 `matmul_acc`（应在已有值上累加）而非 `matmul`（覆盖写）、Tile 未写入即被读取

**同步问题**——错误位置或数值随运行变化，或加入 dump 后现象消失。但部分同步问题在固定输入和调度下也可能稳定复现，不能仅凭"稳定"排除。

- 缺少流水同步：依赖操作之间未插入对应流水的同步（MTE2→V、MTE3→V、M→FIX、FIX→V 等）
- Tile 生命周期错误：Tile 在前一个消费者读取前被复用写入
- mutex 使用错误：多核共享缓冲区的互斥保护缺失或 ID 冲突
- 跨核 GM 写冲突：多个 Core 向重叠的 GM 地址写入

**判断方法**：同一输入连续运行 3 次以上，比较错误位置和数值。有变化 → 同步问题；完全一致 → 大概率逻辑错误，但如果逻辑审查后确认代码正确，仍需排查同步问题（某些同步缺陷在固定调度下也会稳定复现）。加入 `pl.dump_data` 后精度恢复，是同步问题的强信号（详见后文"dump 对执行时序的影响"）。

### 总体流程

```
快速排查（速查表 + dump 时序影响）
    │
    ├─ 未命中已知问题 ──▶ dump 定位流程
    │                       │
    │                       ├─ Step 1：固定复现用例
    │                       ├─ Step 2：缩小 shape 复现 / 推导出错 Core 和 Tile
    │                       ├─ Step 3：关键阶段 dump
    │                       ├─ Step 4：Golden 对齐 + 逐级缩小
    │                       └─ 辅助：错误规律分析 / kernel.cpp 审查 / 间接定位
    │
    └─ 命中已知问题 ──▶ 直接修复
```

## 二、快速排查

进入 dump 定位流程前，先逐项检查下表。这些是 PyPTO Pro 已知的正确性要求，违反会直接导致精度错误甚至设备卡死。

### 已知问题速查表

前 8 项适用于所有 Kernel，后 5 项（sync_all、gather/scatter、phase、matmul_acc、Cube 尾块 compact）仅在特定场景（Vector/Cube）或特定 API 下触发。

| 检查项 | 问题现象 | 规避方法 | 依据 |
|---|---|---|---|
| set_validshape 顺序 | 尾块数据错误、越界读入 | `pl.set_validshape` 必须在 `pl.load` 之前调用；load 之后再设置只影响后续操作，不影响已搬入的数据 | `tutorials/.../tail_block_handling.md` |
| 输出 Tile 漏设 valid_shape | 越界写或写回无效数据 | 输入 Tile、计算结果 Tile 和写回 Tile 对同一逻辑区域使用一致的 valid_shape | 同上 |
| Tile 未初始化即读取 | 随机垃圾值混入计算 | `make_tile`/`make_tile_group` 创建的是裸缓冲区，不自动初始化；必须先 `pl.load` 写入或通过计算写入后再读取 | `tutorials/.../Tile_vector_computation.md` |
| make_tile 缺少同步 | 跨流水数据竞争 | `make_tile`（非 group）不自动插入同步，须手工用 `sync_src`/`sync_dst`；`make_tile_group` + `@pl.jit(auto_mutex=True)` 可自动管理 | 同上 |
| block_dim 与 get_block_num 不一致 | 部分数据未被处理或核间分配不均 | **runtime 是否截断超限的 `block_dim` 随版本而变，不要依赖它**：旧版会静默截断到平台上限，`4c835a12`（9.2.0-beta.2 起）已删除该行为、原值 launch。因此 wrapper 必须自行取 `min(requested, 执行域核数, task 数)`。Kernel 内用 `pl.get_block_num()` 做跨步分片时，若实际启动核数与分片假设不一致，会导致部分 Tile 漏处理或重复处理。版本差异与 caller 规则见 [pypto-pro-op-kb/references/pypto-pro-launch-block-dim.md](../pypto-pro-op-kb/references/pypto-pro-launch-block-dim.md) | `tutorials/.../multi_core_partitioning_and_Tiling.md`、`api/.../get_block_num.md` |
| FP16 归约精度 | 大规模数据归约误差偏大 | FP16 归约的中间累加在 FP32 下进行，但输入量化、累加顺序和输出舍入仍会引入误差；对精度敏感的场景用 FP32 归约 | `api/.../sum.md`、`api/.../reduce_sum.md` |
| 分步乘加精度损失 | 乘法后 cast 再加法导致精度下降 | 优先使用 `pl.mul_add_dst`/`pl.fused_mul_add` 等融合指令，中间乘积不在寄存器中截断 | `api/.../mul_dst_add.md`、`api/.../mul_add_dst.md` |
| load 合轴约束 | 读取错误位置或 padding 区域数据 | 合轴（将多个连续维度合并到一次 load）时，合并的维度必须连续，stride 必须与实际内存排布一致；排查时可改用未合轴的逐维 Tensor 视图对比 | `api/.../load.md` |
| sync_all 核类型 | 纯 vector kernel 同步不存在的 cube 核导致设备错误 | 纯 vector kernel 须指定 `core_type=pl.SyncCoreType.AIV_ONLY` | `api/.../sync_all.md` |
| gather/scatter 越界 | 结果不确定 | 索引值须在有效范围内；scatter 的索引值不重复（无写冲突） | `api/.../gather.md`、`api/.../scatter.md` |
| phase 配对与收尾 | matmul 后 store 读到未完成数据、设备卡死 | matmul 和 store 的 `phase` 参数配对使用，段末用 `Final` 收尾；循环内 `store(Final)` 后不能再有 `matmul`（仅 Cube 场景） | `api/SIMD-API/operation/matrix_computation/phase.md` |
| matmul_acc K 维累加 | K 维分块累加结果错误 | 三个硬性要求缺一不可：①每步 matmul/matmul_acc 都传 `phase`；②L0C 累加器设 `fractal=1024`（FP32）；③cube 段用 `set_mm_layout_transform(enabled=True)` 开启，段末关闭（仅 Cube 场景） | `api/SIMD-API/operation/matrix_computation/matmul_acc.md` |
| Cube 尾块缺少 compact | 尾块计算结果错误、L1/L0 布局错位 | 数据路径涉及 Mat→Left/Right、Acc 搬出等分形转换时，Mat/Left/Right/Acc 的 TileType 须设 `compact=1`，按 `valid_shape` 紧凑解释片上排布；Vec ND 尾块不需要 | `tutorials/.../tail_block_handling.md`、`api/.../CompactMode.md` |

### dump 对执行时序的影响

`pl.dump_data` 会增加运行开销，生成的调试代码还会插入额外的流水同步屏障。**加入 dump 后精度恢复，通常说明原 Kernel 存在同步、依赖、mutex 或缓冲区复用问题。**

遇到这种情况，保留"无 dump 错误、有 dump 正确"的两份结果，然后撤掉 dump，检查对应位置前后的同步和地址复用。也可以临时增加一个 GM 输出，把少量中间结果写回 Host 比较，降低设备打印对执行时序的影响。

以上问题确认后，若仍无法定位，进入下文的 dump 定位流程。

## 三、调试工具

### dump 能力矩阵

| 数据位置 | `MemorySpace` | 是否支持 `pl.dump_data` | 用法 |
|---|---|---|---|
| GM Tensor | Tensor | 支持 | 直接传 Tensor，可打印全量或窗口 |
| UB | `Vec` | 支持 | 直接传 Tile，可打印全量或二维窗口 |
| L1 | `Mat` | 不支持 | A5 不能直接打印 Mat Tile |
| L0A | `Left` | 不支持 | 不能直接打印 Left Tile |
| L0B | `Right` | 不支持 | 不能直接打印 Right Tile |
| L0C | `Acc` | 支持 | 必须提供 GM Tensor 作为 `workspace` |

A5 底层 `TPRINT` 的单参数形式只接受 GM Tensor 和 Vec Tile。`pl.dump_data` API 层面进一步限制 workspace 仅用于 Acc Tile。因此用户通过 `pl.dump_data` 只能对 Acc Tile 使用 workspace，对 Mat、Left、Right 的 dump 会在前端或 CCE 编译阶段失败。

`pl.dump_data` 的输出走设备侧 `cce::printf`，直接打印到终端或设备日志。

### `pl.dump_data` 用法

#### 打印 GM Tensor

```python
pl.dump_data(gm_tensor, loc=True)
pl.dump_data(gm_tensor, offsets=[m_off, n_off], shapes=[8, 8], loc=True)
```

窗口打印（提供 `offsets`/`shapes`）的约束：

- `offsets` 和 `shapes` 须同时提供，长度等于 Tensor rank；
- `shapes` 中的静态值须大于 0；
- 若 Tensor 通过 `make_tensor` 构造了带 stride 的视图，最内维 stride 须为静态 `1`；
- CCE 实现支持 1～5 维 Tensor。

GM Tensor 不使用 `workspace`。

#### 打印 Vec Tile

```python
pl.dump_data(vec_tile, loc=True)
pl.dump_data(vec_tile, offsets=[0, 0], shapes=[8, 8], loc=True)
```

`offsets` 和 `shapes` 可以包含循环变量、`pl.get_block_idx()`、运行时标量或动态 Shape 表达式。

#### 打印 Acc（L0C）Tile

```python
pl.dump_data(acc_tile, workspace=workspace, loc=True)
pl.dump_data(
    acc_tile,
    offsets=[16, 16],
    shapes=[8, 8],
    workspace=workspace,
    loc=True,
)
```

`workspace` 约束：

- 类型为 GM `pl.Tensor`；
- dtype 与 Acc Tile 一致；
- 存储空间须容纳完整物理 Tile（即使只打印窗口，生成代码也会先搬整块 Acc Tile 到 workspace）；
- 须与 Kernel 中其他数据缓冲区不重叠（框架不做编译期检查，需开发者自行确保）。

Kernel 参数是 `pl.Ptr` 时，先用 `pl.make_tensor` 构造 GM Tensor 再作为 workspace 传入。

## 四、dump 定位流程

### Step 1：固定复现用例

精度定位的前提是一个稳定可复现的失败用例。记录以下信息，后续每次 dump 都基于同一用例对照：

- **复现命令**：完整的 Kernel 调用脚本或测试命令
- **输入信息**：Shape、dtype、layout；如果输入是随机生成的，记录随机种子
- **运行配置**：TilingKey、`block_dim`；如果 Kernel 按 TilingKey 走不同代码分支，记录命中的是哪个 Key
- **Golden 来源**：CPU 参考实现的代码位置，或预先生成的期望数据文件路径
- **误差摘要**：输出与 Golden 对比后的 `rtol`/`atol` 阈值、NaN/Inf 个数、最大绝对误差、最大相对误差、首个错误元素的索引
- **复现稳定性**：同一输入连续运行 3 次以上，错误位置和数值是否一致

**先确认 Golden 本身正确**。Golden 来自 CPU 参考实现时，检查参考实现的 dtype、运算顺序和 layout 是否与 Kernel 一致。FP16/BF16 场景下 CPU 默认用 FP32 计算，直接对比会产生虚假误差，需要按 Kernel 的实际精度路径生成 Golden。

**判断逻辑错误还是同步问题**：连续运行 3 次以上，错误位置和数值有变化 → 同步问题；完全一致 → 大概率逻辑错误，但如果逻辑审查后确认代码正确，仍需排查同步问题。

Kernel 异步启动后，在读取输出前完成同步：

```python
kernel[None, block_dim](*args)
torch.npu.synchronize()
```

定位期间保持输入和比较阈值不变。每次新增一个 dump 后，结果才能与原始失败用例直接对照。

### Step 2：确定观察范围

大 case 出错时，可以先尝试缩小 shape 到单个基本块（如 `[16, 16]`）看能否复现。单个基本块的输入会自动在单核上运行，排查更简单。如果单基本块能复现，直接按 Step 3 逐阶段 dump 定位；如果复现不了，需要回到原始出错的 case，推导首个错误位置属于哪个 Core、哪个循环迭代、哪个 Tile，然后针对那个核的对应位置做 dump。

推导步骤：

1. **定位首个错误索引**：Golden 对比结果中通常给出首个错误元素的线性索引。例如输出 shape 为 `[2048, 2048]`，首个错误索引为 `132096`，换算到二维坐标是 `[64, 128]`（`132096 // 2048 = 64`，`132096 % 2048 = 128`）。
2. **映射到 Tile**：根据 Kernel 的 Tile shape 反推该坐标属于哪个 Tile。例如 Tile shape 为 `[16, 16]`，则 `[64, 128]` 落在行方向第 4 个 Tile（`64 // 16 = 4`）、列方向第 8 个 Tile（`128 // 16 = 8`）。
3. **映射到循环迭代**：根据 Kernel 的 `pl.range` 循环结构，反推该 Tile 对应的循环变量值。例如 `for m in pl.range(0, 2048, 16)`，行方向 Tile 4 对应 `m = 64`；`for n in pl.range(0, 2048, 16)`，列方向 Tile 8 对应 `n = 128`。
4. **映射到 Core**：根据多核分片方式反推。如果用 `pl.range(core_id, total_tiles, num_cores)` 跨步分配，通过 Tile 的全局序号和 `num_cores` 计算它属于哪个 Core。
5. **只 dump 目标核的目标迭代**：在 `if` 条件中过滤 Core ID 和循环变量，只打印出错 Tile 的数据窗口（如 `8×8` 或 `16×16`）。

用 `if` 条件同时过滤 Core ID 和循环变量，只打印目标核的目标迭代：

```python
core_id = pl.get_block_idx()
if core_id == target_core and m == target_m and n == target_n:
    pl.printf("core=%d, m=%d, n=%d\n", core_id, m, n, loc=True)
    pl.dump_data(vec_tile, offsets=[0, 0], shapes=[8, 8], loc=True)
```

`pl.printf` 打印 Core、循环变量、GM offset 等上下文信息，方便在日志中定位；`pl.dump_data` 打印实际数据用于和 Golden 对照。

### Step 3：关键阶段 dump

按 Kernel 的实际执行顺序列出中间结果，选择 2～4 个关键位置加入 dump。所有 dump 都加 Step 2 的过滤条件（Core ID + 循环变量）。

Vector Kernel 通常按以下顺序检查：

```text
GM 输入
→ load 后的 Vec Tile
→ 关键计算后的 Vec Tile
→ store 前的 Vec Tile（若与计算后不是同一 Tile）
→ GM 输出
```

Cube Kernel 在 A5 上可直接观察的位置较少：

```text
GM 输入 A/B
→ [L1、L0A、L0B 无法直接 dump]
→ matmul 后的 Acc Tile
→ GM 输出
```

CV Kernel（Cube + Vector 混合 Kernel）可按以下边界检查：

```text
Cube 的 GM 输入
→ Acc Tile
→ Cube 写回的中转 GM 区域
→ Vector load 后的 Vec Tile
→ Vector 计算结果
→ 最终 GM 输出
```

例如 Vector Kernel 先比较 load 后、核心计算后和 store 后；如果 load 后正确而核心计算后错误，下一轮只细查这段计算。

### Step 4：Golden 对齐并逐级缩小

中间 Golden 必须与 Kernel 该位置的语义一致，包括：

- dtype 转换发生的位置；
- 矩阵转置和 layout；
- padding 值和有效 Shape；
- 归约的累加 dtype、矩阵乘的累加 dtype；
- 当前 Core 和当前 Tile 对应的数据范围。

记录每个 dump 的结果：

| 位置 | 实际数据 | Golden | 结论 |
|---|---|---|---|
| load 后 Tile | dump 打印的数值摘要 | 对应输入窗口 | 正确/错误 |
| 计算后 Tile | dump 打印的数值摘要 | 中间 Golden | 正确/错误 |
| store 后 GM | dump 打印的数值摘要 | 输出 Golden | 正确/错误 |

找到相邻的"一处正确、一处错误"后，在两者之间继续增加一个 dump。重复这个过程，直到范围缩小到一次 load、move、计算、cast 或 store。

### 辅助 A：分析 dump 数据的错误规律

拿到 dump 数据后，观察错误的分布规律。不同类型的错误通常对应不同的根因：

| 错误现象 | 典型根因 | 检查方向 |
|---|---|---|
| 全零 | Tile 从未被写入、store 到错误地址、累加器未初始化即被读取、计算被编译器删掉 | store offset、Tile 地址分配、kernel.cpp 中对应指令是否被删掉 |
| 全为同一值 | expands 传错值、广播维度错误、累加器未清零 | expands/广播参数、累加器清零逻辑 |
| 数据看起来是转置的 | load 的 order 参数错误、ND/NZ layout 混淆、Mat 与 Left/Right 的 layout 配置不一致 | load order、TileType layout、move 搬运方向 |
| 整体偏移固定值 | offset 计算差一个常数、Tile 基址错误、stride 差一倍 | GM offset 公式、Tile 基址、stride 计算 |
| 每隔 N 个元素出错 | stride 或 block size 配置错误、mask 覆盖范围不符 | Tensor stride、Tile shape、mask 设置 |
| 误差随 K 或归约长度增大 | Vector 侧手动累加用了 FP16 而非 FP32、累加前过早 cast 回低精度、累加顺序导致误差累积 | 改用 FP32 累加、推迟 cast、检查 matmul_acc 累加路径 |
| 仅特定行/列出错 | `pl.range` 跨步分片边界、尾块、valid_shape 未设置导致 padding 读入越界 | `pl.range` 起始/步长、尾块处理、valid_shape |
| 正确与错误块交替出现 | 跨核 GM 写冲突、Tile group 轮转错位、mutex 缺失 | Core 间地址重叠、Tile group depth 和轮转、mutex ID |
| 前几轮正确，后续错误 | 地址递增错误、累加器未在迭代间清零、Tile 被提前复用 | 循环内地址递增、累加器清零、Tile 生命周期 |
| 数值像随机垃圾 | 缓冲区重叠、读取了未初始化内存、地址完全错误 | Tile 地址分配、workspace 是否与数据缓冲区重叠 |
| 符号翻转或数值翻倍 | 计算符号错误（sub 写成 add）、重复计算、`scale` 随路量化比例配置错误 | API 选择（add/sub）、循环是否多跑一轮、`scale` 配置 |
| 加入 dump 后恢复正常 | dump 插入的 pipe_barrier 带来了额外同步，使原本缺失的同步暂时生效 | 撤掉 dump，检查该位置前后的流水依赖和 Tile 复用 |

观察到规律后，用 `pl.printf` 打印相关变量（offset、stride、循环变量、`pl.get_block_idx()`）验证假设。

### 辅助 B：常见现象速查

| 现象 | 优先检查 |
|---|---|
| GM 输入已经错误 | Host 侧调用 Kernel 时参数顺序、dtype、shape、指针是否正确，Host 预处理逻辑是否有误 |
| GM 正确，Vec load 后错误 | load offset、Tensor stride、layout、尾块 padding、有效 Shape |
| Vec 输入正确，计算后错误 | API 参数、dtype、mask、shape 不匹配（不支持广播）、归约方向、类型转换 |
| 前几轮正确，后续循环错误 | 地址递增、Tile 清零、累加器初始化、Tile 复用、同步和 mutex |
| store 前正确，GM 输出错误 | store offset、有效 Shape、写回 layout、重复写和越界 |
| 单核正确，多核错误 | `block_dim`、`pl.range` 跨步分片参数、`block_dim` 与 `pl.get_block_num()` 不一致、跨核覆盖 |
| 只有尾块错误 | 尾块长度、padding、有效 Shape 和 store 范围 |
| 错误呈转置或按固定块大小分布 | ND/NZ、Left/Right layout、stride 和转置配置 |
| 误差随 K 或归约长度增大 | Vector 侧手动累加用了 FP16、累加前过早 cast、运算顺序和误差阈值 |
| 每次运行错误位置不同 | 同步、mutex、未初始化数据、Tile 生命周期和写冲突 |
| 全零或全同一值 | Tile 未写入、store 地址错误、expands 传错值、累加器未初始化 |
| 数值像随机垃圾 | 缓冲区重叠、workspace 与数据区冲突、地址完全错误 |
| 结果符号翻转或翻倍 | API 选错（add/sub）、循环多跑一轮、scale 配置错误 |
| 仅特定 TilingKey 出错 | 该 Key 选择的代码路径、特化参数、分支条件 |
| 量化/反量化结果异常 | `scale` 随路量化比例 dtype/shape、dequant 路径、fixpipe 量化配置 |
| ND↔NZ 转换后出错 | 转换前后的 shape/stride、layout 标记、搬运 order |
| AtomicAdd 累加结果不对 | 原子写冲突、Core 间 GM 覆盖区域、目标 GM 区域未初始化为零 |
| Golden 本身有误 | CPU 参考实现的 dtype、运算顺序、layout 与 Kernel 不一致 |

### 辅助 C：L1、L0A、L0B 不可直接 dump 时的间接定位

Cube 数据路径为 `GM → L1(Mat) → L0A/L0B(Left/Right) → L0C(Acc)`。A5 无法直接查看中间的 L1、L0A 和 L0B，因此按下面的方法缩小范围。

**1. 先确认 GM 中的输入数据是否正确**

打印 A、B 的目标窗口，同时打印对应的 GM offset、shape、stride 和转置配置。先排除 Host 传入的数据本身、地址计算和 GM 视图的问题。

**2. 打印 matmul 后的 Acc**

使用带 workspace 的 `pl.dump_data(acc_tile, ...)`。如果 GM A/B 正确而 Acc 错误，问题范围就在以下环节：

- GM 到 L1 的 `pl.load`；
- L1 到 L0A/L0B 的 `pl.move`；
- Left/Right 的 layout 或转置配置；
- `pl.matmul` 的 M/N/K、dtype、累加方式（`matmul` 覆盖写 vs `matmul_acc` 累加）或有效 Shape。

**3. 通过构造特殊输入，隔离 A 和 B 哪边的搬运有问题**

Acc = A × B。L1→L0A 和 L1→L0B 这两段搬运都无法直接 dump，但可以通过构造特殊输入来间接判断哪边出错：保持 tile shape、dtype、layout 和搬运参数不变，只改变输入数值。

- **把 B 设为单位矩阵**：此时 Acc = A × I = A，Acc 的结果直接反映 A 的搬运路径是否正确。如果 Acc 与 A 的 GM 数据不一致，说明 A 的 L1→L0A 搬运有问题。
- **把 A 设为单位矩阵**：此时 Acc = I × B = B，Acc 的结果直接反映 B 的搬运路径是否正确。如果 Acc 与 B 的 GM 数据不一致，说明 B 的 L1→L0B 搬运有问题。
- **使用递增值、行号或列号模式**作为输入，便于识别转置错误、分块错位和 stride 错误。

如果两组实验单独都正常、但恢复原始 A/B 后异常，说明 A 和 B 的搬运各自没问题，问题在 matmul 参数、累加或同步。

**4. 检查生成代码和同类工作样例**

在对应 JIT 目录的 `kernel.cpp` 中核对 TLOAD、TMOV 和 TMATMUL 的 shape、offset、layout、有效 Shape 及源/目的 Tile。再与目标版本中已上板通过、数据路径相同的 Matmul 用例逐项比较。

这个方法不能直接显示 L1/L0A/L0B 的内容，但可以把问题缩小到左搬运、右搬运或 matmul 本身。输出中应标注为"间接定位"。

### 辅助 D：检查编译生成的 kernel.cpp

dump 只能看数据，看不到指令。当 dump 缩小了范围但仍无法确认根因时，读编译产物 `kernel.cpp` 可以看到指令级细节。

JIT 编译后，`kernel.cpp` 生成在 `./build/<kernel_name>__<arch>/tk_<packed>/` 目录下（未使用 TilingKey 时为 `tk_none/`），包含编译后最终在设备上执行的 C++ 代码：Tile 声明、搬运指令、计算指令和同步指令。同目录下还有 `call_kernel.cpp`（Host 侧启动代码），当怀疑 `block_dim` 计算错误、TilingKey 分发异常或参数打包有问题时，检查该文件中 Host 侧的启动参数。

重点检查项：

**指令序列**：确认 TLOAD、TMOV、TMATMUL、VADD、VCAST 等指令的排列顺序与预期数据流一致。编译器可能合并或删除操作——例如将 cast 合并到 store 的 fixpipe 路径，或删掉被判定为无效的计算。

**搬运参数**：逐条核对 TLOAD/TMOV 的 offset、shape、stride、layout（ND/NZ/ZN）、源和目的 Tile。前端的 `offsets=[m, n]` 编译后会变成具体的字节或元素偏移，确认转换符合预期。

**同步指令**：跨流水依赖的同步形式取决于 Tile 管理方式——`make_tile` 场景框架插入 `set_flag`/`wait_flag`（SYNC_SRC/SYNC_DST），`make_tile_group` + `auto_mutex` 场景生成 `get_buf`/`rls_buf`，vector 侧全屏障用 `pipe_barrier(PIPE_*)`，性能调优场景用 `phase` 靠硬件 `unit_flag`（无软件同步指令，须检查 `phase` 配对是否正确）。检查以下关键依赖之间是否存在正确的同步：

- MTE2（GM→UB）完成后才能在 V 流水读取 → 需要 MTE2→V 同步
- MTE3（UB→GM）完成后才能在 V 流水复用该 UB → 需要 MTE3→V 同步
- M（matmul）完成后 FIX（fixpipe）才能读 L0C → 需要 M→FIX 同步
- FIX（L0C→UB/GM）完成后 V 才能读取搬出的数据 → 需要 FIX→V 同步；FIX 完成后 M 才能复用同一块 L0C 写入新结果 → 需要 FIX→M 同步
- Cube 流水（MTE2→MTE1→M→FIX）与 Vector 流水（MTE2→V→MTE3）之间的跨流水依赖

**Tile 地址分配**：确认各 Tile 的 UB/L1/L0C 地址不重叠。`make_tile_group` 使用显式地址，地址算错会导致缓冲区覆盖。

**Tile group 轮转**：如果 `make_tile_group` 的 `depth` > 1 或 `mutex_ids` 包含多个 ID，检查轮转逻辑和 mutex 是否正确生成。

**编译器优化导致的行为变化**：对比 PyPTO Pro 源码和生成的 CCE 代码，确认编译器没有做开发者预期之外的优化。常见问题：循环展开导致寄存器拷贝错误、算子融合改变了累加顺序、编译器认为无用而删掉了实际必要的操作。

典型配合方式：dump 定位到某步出错 → 打开 `kernel.cpp` 找到该步对应的指令 → 核对参数和同步 → 用 `pl.printf` 打印可疑变量验证假设。

## 五、排障

### dump 没有输出

依次检查：

1. Kernel 调用后是否执行了同步；
2. Core 和循环过滤条件是否命中；
3. `pl.printf` 标记是否出现；
4. 设备日志的输出位置和级别；
5. 生成的 `kernel.cpp` 中是否有对应打印代码。

### 编译失败

检查：

- 是否尝试直接 dump Mat、Left 或 Right；
- `offsets` 和 `shapes` 是否成对且 rank 一致；
- Tensor 窗口的最内维 stride 是否为静态 `1`；
- Tile 窗口是否为二维；
- Acc workspace 的类型、dtype 和容量是否正确。

### `pl.pto_assert` 的行为限制

`pl.pto_assert` 在条件不满足时仅通过设备侧打印记录信息，**不中止 Kernel、不在 Host 侧抛异常**。越界访问在 assert 记录时已经发生，垃圾值已经参与计算。需要无条件中止时使用 `pl.trap()`。

## 六、修复与验证

1. 保留关键 dump，确认原来第一个出错的位置已经正确。
2. 移除 `pl.dump_data`、`pl.printf`、`pl.pto_assert` 和 `pl.trap`。
3. 确保 Kernel 重新编译。
4. 重新运行原始失败用例。
5. 恢复生产 `block_dim`，覆盖目标 Shape、尾块、dtype 和全部 TilingKey。
6. 重复运行，确认结果稳定。

测试数据、失败 case 和比较阈值保持不变。

## 七、输出模板

```markdown
## 精度定位结论

### 失败用例
- 复现命令：
- Shape / dtype / layout：
- TilingKey / block_dim：
- 比较阈值和误差摘要：

### 第一个出错位置
- 代码位置：
- Core / 循环迭代 / 数据窗口：
- 前一个正确结果：
- 当前错误结果：
- Golden：

### 原因
- 涉及的搬运或计算：
- 参数或状态错误：
- 为什么会产生当前错误：

### 修复与回归
- 修改内容：
- 原始失败用例结果：
- 多 Shape / dtype / TilingKey / 多核回归结果：
```

如果问题仍处于 `GM → L1 → L0A/L0B → L0C` 这段不可直接观察的范围，明确列出已确认正确的 GM 输入、错误的 Acc 输出、A/B 搬运路径的排查结果和下一步验证计划。
