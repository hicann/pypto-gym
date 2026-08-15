## 算子名称

- **op_name**: {operator_name}
- **设计时间**: {timestamp}
- **基于 SPEC**: `custom/{op}/SPEC.md`
- **基于 EXPLORE_REPORT**: `custom/{op}/EXPLORE_REPORT.md`

---

## Knowledge Bindings

> 逐条覆盖 `KB_SELECTION.json.optional_patterns` 与 `required_constraints`。可选模式必须产生
> 真实、独立的设计不变量；若前提不成立、仅属背景或作用重复，回退 Stage 1 删除或替换，
> 不得强行绑定。必需约束不得省略，不适用时写明理由交 verifier 裁定。

| 类型 | KB 相对路径 | 本 class 的具体不变量 | 计划实现位置 | verifier 检查方法 |
|---|---|---|---|---|
| optional pattern / required constraint | `{patterns/... 或 constraints/...}` | `{不是标题复述，而是可落到代码的约束}` | `{文件 + symbol/设计章节}` | `{静态检查/运行/profiling 证据}` |

---

## §0 Module 划分（R0 输出）

> 依据[Module划分](../references/module_partitioning.md)，先按数据依赖确定Module边界，再按Section边界继续拆分。R0记录每个Module的Section归属，具体Section代码结构在§4确定。

### 维度契约

- **kernel 处理维度**：{如 2D [M,N]}
- **host 适配规则**（SPEC 要求多维/1D 时填写）：

| SPEC 输入 shape | host 适配 | kernel 收到 | 输出还原 |
|-----------------|-----------|-------------|----------|
| {如 1D [L]} | reshape [L,1] | [M,N] | reshape 回 [L] |
| {2D [M,N]} | 直接传入 | [M,N] | 直接输出 |

### Wrapper 操作清单

> 默认填“空（仅参数校验、纯 Python 形状运算、输出分配和一次 kernel 启动）”。仅当目标框架确实无法迁移、且本阶段有可核验证据时逐项填写；Stage 4 不得新增或扩大例外。

| API / 操作 | 无法迁入 kernel 的目标版本证据 | 适用条件 | 预期代价预算 | Stage 4 profile 测量方法 |
|---|---|---|---|---|
| {空；或具体 API / 操作} | {文档/官方样例路径、原文片段、目标版本；为空时写 N/A} | {dtype/shape/layout 条件；为空时写 N/A} | {允许的 device kernel / 时间上界；为空时写 0} | {profile 命令及核对项；为空时写确认无 aclnn*} |

### 划分依据

{数据依赖边界 + 各步骤的Section归属 + 按Section边界继续拆分的结果；量化场景写明使用V流水还是FIX路径，以及Scaling参数缓冲的准备位置；若因UB等片上空间不足继续拆分，写明超限对象和拆分位置}

### Module 列表

| Module | 目的 | 输入依赖 | 输出 | Section |
|-------|------|---------|------|---------|
| Module 1 | {purpose} | {dependency} | {output} | {section_vector / section_cube} |
| Module 2 | ... | ... | ... | ... |

> 若只有一个 Module，标注"单 Module"并说明为什么不需要拆分。

### 归约轴容量结论

> 含归约算子时必填；无归约时填"不涉及归约"。

- **归约轴是否可能超单 tile**：{是 / 否，依据 SPEC 动态轴范围}
- **多Tile归约方案**：{常规多遍 / 在线统计加输出两遍 / 其他；逐遍写明遍历范围、状态和依据；单Tile装得下则写“不涉及”}

### Module 级数据流

```
{Module 间的数据流向示意；标注中间数据所在空间、传递API，以及Cube/Vector之间的同步方式}
```

---

## §1 API 映射（R1 输出）

> 按`$CANNBOT_CONFIG_ROOT/references/performance-constraints.md`中的“强制2：Vector默认使用VF”填写。每个Vector步骤冻结唯一实现：
>
> ```
> vector_selection:
>   step:                  <本 §1 中的哪一步>
>   implementation:        vf | tile_op
>   api_sequence:          <vf.* 或 pl.* API 序列>
>   decision_reason:       default_vf | kb_template_required
>   kb_template_evidence:  <default_vf 写 n/a；tile_op 写已选 KB 路径、明确要求的原文/片段>
>   target_version:        <目标软件/框架版本>
>   applicable_conditions: <dtype/shape/layout/算子条件>
> ```
>
### Module 1 API 调用序列

```
Module 1 — {名称}:
  for j in range(n_tile_num):
    {API 调用}
    ...
```

{每个 API 标注来源（API 文档路径或 EXPLORE_REPORT §4 定位的官方指定算子）}

### Module 2 API 调用序列

```
...
```

### 超越函数数值安全边界

> 条件性：仅当 API 链含 exp/log/sqrt/reciprocal/tanh 等超越/非线性函数时填写。

| API | 输入理论范围 | 目标 dtype 上限 | 是否溢出 | 防护措施 |
|-----|-------------|----------------|----------|----------|
| `{超越函数 API}` | {如 x 无上界，x>11 时 exp 溢出} | fp16 ≈ 65504 | exp(11.09)≈65504 → +inf → NaN | 先减最大值再 exp（或归一化省去分母）

---

## §2 Tile 规划（R2 输出）

### 2.1 关键常量定义

列出所有在 kernel 中使用的编译期常量。涉及 Cube 的 tile 尺寸必须满足本次资料探索确认的对齐要求，具体以 `custom/{op}/EXPLORE_REPORT.md` §7 的记录为准。

```python
# ── 算子关键常量（coder 必须逐字使用） ──
TS = {S_tile}           # S 方向 tile 尺寸
TD = {D_tile}           # D 方向 tile 尺寸
TS_HALF = {S_half}      # subblock 半尺寸 (如有 dual_split)
SCALE = 1.0 / sqrt({D_logical})  # 缩放因子（若算子有 scale 步骤）
```

> **注意**：若某维度实际值固定且 < tile 尺寸（如 D=64 对齐到 TD=128），tile shape 用 TD 声明，运行时通过 §7 的 `set_validshape` 限制有效区域。公式常量（如 SCALE）若有依赖，必须使用实际维度值而非 tile 尺寸。

### Tile 属性表

每个 tile 的 shape 由其所参与的 API 操作数要求决定：

| 用途 | 变量名 | shape | dtype | 内存空间 | layout | 大小 | 备注 |
|------|--------|-------|-------|---------|--------|------|------|
| 输入暂存 | `tile_a` | `[64,128]` | FP32 | UB(Vec) | `—` | 32768 | ... |
| ... | ... | ... | ... | {Vec/Mat/Left/Right/Acc} | {— 或具体 layout} | ... | {...} |

> 内存空间取 `Vec`(UB)/`Mat`(L1)/`Left`(L0A)/`Right`(L0B)/`Acc`(L0C)；`—` 表示未显式指定layout。`Vec`没有统一的默认layout，是否需要填写以及允许哪些值由具体API决定；其他空间的默认值也要按目标架构核对`TileType.md`。

### TileGroup槽位访问表

> 所有TileGroup都记录`depth`和逐Tile mutex配置，避免漏掉单Tile多ID以及不配置mutex元数据的情况。运行时下标不会自动取模，采用`group[i]`时必须给出`i`始终位于`[0, depth)`的依据。

| TileGroup | depth | 每个Tile的mutex ID | 访问方式 | 槽位表达式 | 同步及有界/游标说明 |
|---|---:|---|---|---|---|
| `{group_name}` | {1/2/N} | {如 `[[0, 1], [2, 3]]` / 不配置} | {`current()` / `next()` / `group[i]`} | {如 `task_idx % 2`} | {多ID用途；不配置时的手工同步；下标范围；与`next()`混用时的游标状态} |

### tile_dims stride 注意事项

> 仅当使用了 `load_tile` + `tile_dims=[d0, d1]`（覆盖多维度）时填写。大 stride 可能影响性能，具体阈值以官方指定算例经验为准。

| load_tile/store_tile | tile_dims | d0 在张量布局中的 stride | 备注 |
|----------------------|-----------|------------------------|------|
| {tile_a, x, [i,j]} | {如 [0,1]} | {计算值} | {是否需关注} |

---

## §3 片上空间布局（R3 输出）

> tile 按 `target_memory` 落到不同片上空间，**每个空间独立寻址、独立限容**——地址各自从 `0x00000` 起算，同一 addr 在不同空间是不同物理位置（证据 `matmul.md` 调用示例：L0A/L0B/L0C 同用 `addr=0x0000` 互不冲突）。纯 vector 算子只有 UB 一节；含 cube/matmul 的算子须补 L1(Mat)/L0A(Left)/L0B(Right)/L0C(Acc) 各节——`matmul` 操作数空间为硬性约束：`lhs`→L0A、`rhs`→L0B、`dst`→L0C（证据 `matmul.md` 参数类型表）。

### 片上地址映射表（按内存空间分节）

| 内存空间 | 用途 | 变量名 | shape | dtype | layout | 地址 | 大小 | 备注 |
|---------|------|--------|-------|-------|--------|------|------|------|
| UB(Vec) | ... | ... | ... | ... | {— 或具体 layout} | ... | ... | {...} |
| L1(Mat) | ... | ... | ... | ... | {— 默认} | ... | ... | {含 cube 时填} |
| L0C(Acc) | ... | ... | ... | ... | {— 默认} | ... | ... | {含 cube 时填} |

**各空间总用量**（逐空间列出，无对应 tile 的空间可省略）:
- UB(Vec): {∑ 大小} / {EXPLORE_REPORT §7 UB 容量} = {百分比}
- L1(Mat) / L0A / L0B / L0C（如有 cube）: {∑ 大小} / {EXPLORE_REPORT §7 对应容量} = {百分比}——容量取 §7 探测记录，§7 未记录则回退 material-explore 补测，不臆测

### 分配方式选择

> **性能强制**：所有需要 buffer 切换/轮转的 tile 一律用 `make_tile_group` + `auto_mutex`；`make_tile` 仅限单次使用 scratch tile（不参与 buffer 轮转）。禁止用 `make_tile` + 手动 `sync_src`/`sync_dst` 管理 buffer 轮转。

- **方案**: make_tile_group + auto_mutex（由框架自动管理 buffer 切换与 core 内互斥）；若有单次使用 scratch tile 用 make_tile
- **依据算子**: {EXPLORE_REPORT §4 定位的最相似官方指定算子路径}
- **理由**: {基于 R0 Module 划分，说明 auto_mutex 如何覆盖本算子的 buffer 切换/互斥边界}

### double buffer 地址规划

> `make_tile_group` 的 buffer 数 > 1 时地址占用按倍数放大，须在此显式记录。buffer 数取值参照官方指定算子中相似算子的实际配置。

| 项目 | 结论 | 受影响 tile/地址 |
|------|------|-----------------|
| buffer 数 | {1 / 2 / ...} | {tile 名单} |
| PONG 地址 | {无 / 预留} | {地址范围} |
| 槽位访问 | {`next()` / `group[slot_idx]`} | {`slot_idx`的定义与范围} |

---

## §4 循环与 Section 结构（R4 输出）

> 依据[循环与Section结构设计](../references/loop_design.md)，把§0确定的Module放入具体Section代码结构，再组织循环和跨Tile状态，并确定分核信息的获取位置。若API数据通路与§0记录的Section归属冲突，回到R0修正Module划分。

### 参考样例

- **主要参考样例**: {EXPLORE_REPORT.md §4 定位的官方指定算子路径}
- **可复用结构点**: {该样例中可复用的Section / 循环 / 分核信息获取方式}
- **补充参考**（如有）: {`../pypto-pro-material-explore/references/official_samples.md` 清单中其他参考样例路径及参考点}

### 本算子结构说明

- **Module / Section对应关系**: {每个Module放在哪个Section；相邻同域Module是否共用一个Section}
- **结果单元**: {一次完整计算产出的输出范围；如一个输出Tile、一个Softmax行块或一个Matmul `[M_tile, N_tile]`块}
- **跨Tile状态生命周期**: {状态名称、初始化位置、更新方式和使用位置；不涉及则填“不涉及”}
- **Section结构**: {section_vector / section_cube的数量、顺序，以及循环相对Section的位置}
- **循环嵌套**: {参照样例说明 M-tile / N-tile 等循环嵌套关系}
- **动态循环上界**: {如 `n_tiles = (N + TILE_N - 1) // TILE_N`；静态shape则写固定值及来源}
- **分核信息**: {核编号、核数及其获取位置；混合Kernel的Vector Section还要写明AIV核数的计算方式}

> 同步点见 §6（R6），尾块处理见 §7（R7）。

---

## §5 分核策略（R5 输出）

> 📌 **权威依据（必读，官方标准）**：`$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md`（分核方式、负载均衡、核数设置和tiling传参以此为准）。

### 分核方式

- **方案**: strided loop —— {扁平切 `pl.range(core_id, m_tiles*n_tiles, num_cores)` / 二维切 外 `range(core_id, m_tiles, num_cores)`+内 `range(0, n_tiles, 1)`} + {选择理由}
- **host 侧 block_dim**: {仅Vector Kernel使用`vector_core_num`，Cube或混合Kernel使用`core_num`}，`block_dim = min(max_blocks, total_tasks)`

---

## §6 核间同步（R6 输出）

> 带非空`mutex_ids`的TileGroup由`auto_mutex`管理核内跨Pipe依赖；未配置`mutex_ids`时，按§2记录的数据路径手工插入核内同步。两种方式都不能替代Cube与Vector之间的cross_core同步。跨执行域依赖使用`set_cross_core`/`wait_cross_core`，具体规则见[跨核同步](../references/cross_core_synchronization.md)。
>
> **条件性**：Cube与Vector之间有数据依赖，或存在需要`INTER_BLOCK`、`INTER_SUBBLOCK`、`UNICAST_BLOCK`处理的依赖时填写；否则填“不涉及cross_core”。Section数量本身不是判断依据。

### cross_core 涉及判定

- **是否涉及 cross_core**: {是 / 不涉及}
- **依据**: {依赖是否跨Cube/Vector执行域或跨Block/subblock；若跨Block/subblock，说明为何不能用普通核内同步或算法重构处理}

> 若"不涉及"，以下各表填"不涉及"或省略。

### 共享数据与槽位映射

| 共享数据/TileGroup | depth或槽位数 | 生产者/Section | 消费者/Section | 生产者访问 | 消费者访问 | 槽位一致性依据 |
|---|---:|---|---|---|---|---|
| `{group_name}` | {2/N} | {Cube/...} | {Vector/...} | {`group[slot_idx]`} | {`group[slot_idx]`} | {`slot_idx = task_idx % depth`及范围证明} |
| `{workspace_name}` | {1/N} | {Cube/...} | {Vector/...} | {地址/offset表达式} | {地址/offset表达式} | {同一逻辑块映射到同一地址} |

> `group[i]`不读取也不推进轮转游标。若一侧或两侧使用`next()`，必须说明初始游标、调用次数和分支路径为何仍选中同一物理槽位。

### 同步点

> 逐方向、逐槽位记录READY/RELEASE；同步点规则以参考文档为准，官方指定算子用于核对完整调用方式。

| 事件组 | 共享缓冲/槽位表达式 | 方向 | set位置/pipe | wait位置/pipe | `sync_mode` | `event_id` | 复用条件 |
|---|---|---|---|---|---|---:|---|
| `{READY}` | `{group[slot_idx]}` | {生产者→消费者} | {最后一次写后；pipe按紧邻的写操作确定} | {第一次读前；pipe按紧邻的读操作确定} | {INTRA_BLOCK/...} | {ID或表达式} | {旧信号已被wait消费} |
| `{RELEASE}` | `{group[slot_idx]}` | {消费者→生产者} | {最后一次读后；pipe按紧邻的读操作确定} | {下一次覆盖前；pipe按紧邻的写操作确定} | {INTRA_BLOCK/...} | {ID或表达式} | {旧信号已被wait消费} |

### event_id 分配表

> event_id取值范围为`[0, 16)`。记录ID对应的数据方向和物理槽位，动态表达式的全部运行时取值也必须在该范围内。

| 流水组 | 方向 | Tile槽位 | event_id范围或表达式 | 复用条件 |
|---|---|---|---|---|
| {流水组1} | {Cube→Vector} | {`slot_idx`} | {`READY_BASE + slot_idx`} | {对应wait已消费} |
| {流水组1} | {Vector→Cube} | {`slot_idx`} | {`RELEASE_BASE + slot_idx`} | {对应wait已消费} |

---

## §7 尾块处理（R7 输出）

> 📌 **权威依据（必读，官方标准）**：`$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/tail_block_handling.md`（尾块处理以此为准）。

### 尾块处理方案

> 在 §7 填入以下尾块处理代码：

```python
# ceiling division 计算 tile 数（必须向上取整，用 N//TILE 直接整除会漏掉尾块）
m_tile_num = (M + TS - 1) // TS
n_tile_num = (N + TD - 1) // TD

# 循环内计算尾块有效尺寸并告知硬件（set_validshape 有状态，每轮都要重设）
valid_m = pl.min(M - m_off, TS)       # 满 tile = TS, 尾块 = 余数
valid_n = pl.min(N - n_off, TD)
pl.set_validshape(tile_a, [valid_m, valid_n])  # 运行时告知硬件
```

- **compact**: {`None`/`1`/`2`，按数据通路和API约束说明依据；普通Vec ND尾块通常不需要`compact`}
- **无效区域是否会被读取**: {否→不填充；是→按计算语义选择`pad`并执行`fillpad`；矩阵尾块是否填充由具体数据路径和API约束决定}

---

## §8 目标测试 case（R7.5 输出 → 交付 develop）

> 基于 R2 tile 尺寸与 R7 尾块方案确定的具体测试 shape，develop 直接实现这些 case，不再自行重算。
> 若 SPEC/用户已有目标 case，在其基础上追加下述基础 case；无用户指定 case 时直接确定，不询问用户。
> 至少 4 个（单轴算子按 R7.5 例外处理，并在此说明实际 case 数与原因）。

| case 名 | 具体 shape | 覆盖场景 | design 适配确认 |
|---------|-----------|---------|----------------|
| `test_{op}_aligned` | {如 `[TILE_A, TILE_B]`} | 全整除 | ✅ / {不适配→回溯的轮次} |
| `test_{op}_tail` | {如 `[TILE_A + 尾块余数, TILE_B]`} | 单轴尾块 | ✅ / ... |
| `test_{op}_tail2d` | {如 `[TILE_A + 尾块余数, TILE_B - 尾块余数]`} | 双轴尾块 | ✅ / ... |
| `test_{op}_multitile` | {如 `[2~3 × TILE_A + 尾块余数, TILE_B - 尾块余数]`} | 跨多 tile + 尾块（最小规模，勿放大） | ✅ / ... |
| {如用户指定的额外 case} | {shape} | {场景} | ✅ / ... |

---

## §9 综合评估（R8 输出）

### 准确性检查

| 检查项 | 结果 | 证据 |
|--------|------|------|
| API 调用链完整覆盖数学公式 | ✅ / ❌ | {映射验证} |
| §4 已参照官方样例确定循环与Section结构（含参考样例路径与结构说明） | ✅ / ❌ | {回 R4 补充} |
| 数据依赖正确（Module 顺序 + sync 点） | ✅ / ❌ | {依赖分析} |
| dtype 精度满足要求 | ✅ / ❌ | {FP32 matmul 累加 / ...} |
| 归约类 API 的 `[M,1]`/`[1,N]` 输出已设 `layout` | ✅ / ❌ | {回 R2 补 layout} |
| Acc tile 物理 M×N×dtype_bytes ≥ fractal（FP32 ≥ 1024 bytes；动态轴含极小维度时尤需检查） | ✅ / ❌ | {回 R2 pad tile shape} |

### 泛化性检查

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 目标测试 case（≥4，单轴算子按例外）已按 tile 切分确定具体 shape，且逐个验证 design 可适配（见 §8「目标测试 case」表） | ✅ / ❌ | {回 R7.5 补充 / 回溯适配不了的轮次} |
| 是否正确处理了尾块（M/N 尾块） | ✅ / ❌ | {ceiling division + set_validshape 设计} |
| 归约轴可能超单Tile时，已给出覆盖完整归约轴的方案，并说明遍历次数、跨Tile状态及依据 | ✅ / ❌ | {回R0重设归约方案} |
| 循环边界正确 | ✅ / ❌ | {valid_m/valid_n 计算验证} |
| 超越函数在 dtype 范围内无溢出 | ✅ / ❌ | {引用 §1 数值安全边界} |
| 跨 tile 状态初始化/持久化正确 | ✅ / ❌ | {expands 恒等值 / muls 拷贝；回 R0 或 R1 修正} |
| cross_core 同步方案正确（存在跨执行域或跨Block/subblock依赖时：同步点位置与 event_id 隔离参照权威文档和官方指定算子） | ✅ / ❌ | {回 R6 修正} |

### 一致性检查

| 检查项 | 结果 | 说明 |
|--------|------|------|
| R0-R7 输出无矛盾 | ✅ / ❌ | {交叉验证} |
| 所有决策有证据支撑 | ✅ / ❌ | {证据链检查} |
| 各内存空间（UB/L1/L0A/L0B/L0C）tile 总用量分别不超各自容量上限（R3 逐空间验证，含 cube 时须查 L1/L0） | ✅ / ❌ | {回 R3 重排地址 / R2 缩 tile} |
| `tile_dims` 使用时已关注大 stride 对性能的影响 | ✅ / ❌ | {回 R2 调整布局} |
| 条件性检查（若 §6 填“不涉及cross_core”，确认不存在Cube↔Vector或跨Block/subblock的数据依赖） | ✅ / ❌ | {R0 重新评估} |

### 评估结论

- **整体**: {通过 / 需修改}
- **限制条件**: {如不支持尾块、需 M 整除完整 M-tile 尺寸等}
- **修改记录**: {修改内容、轮次、原因}

---

## §10 Tile 数据流全景图

将 R0-R7 各轮产出的片段串联为一张完整的 tile 级数据流图。标注每一块 tile 的流向：从哪里读取、经过哪些操作转换、写入哪里。

```
{按实际算子的 tile 流向绘制}

GM ─[load_tile]→ tile_a ─→ {操作} → {输出tile} → ...
                                     ↓
                              {操作}({输出tile}, {持久化tile})
                                     ↓
                              {持久化tile} → tile_a → {操作} → ... → tile_out
                                                                  ↓
GM ←[store_tile]─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```

### 图例

| 标记 | 含义 |
|------|------|
| `[load_tile]` | MTE2 搬运：GM → UB |
| `[store_tile]` | MTE3 搬运：UB → GM |
| `→` | V 流水线操作，标注 API 名 |
| `---` | 数据跨 Module 持久化（tile 在不同 Module 间不被覆盖） |
| `├─` | 同一 Module 内分支（同一数据被多次使用） |
