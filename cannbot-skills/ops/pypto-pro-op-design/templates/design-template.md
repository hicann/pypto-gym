## 算子名称

- **op_name**: {operator_name}
- **设计时间**: {timestamp}
- **基于 SPEC**: `custom/{op}/SPEC.md`
- **基于 EXPLORE_REPORT**: `custom/{op}/EXPLORE_REPORT.md`

---

## §0 Phase 划分（R0 输出）

### 维度契约

- **kernel 处理维度**：{如 2D [M,N]}
- **host 适配规则**（SPEC 要求多维/1D 时填写）：

| SPEC 输入 shape | host 适配 | kernel 收到 | 输出还原 |
|-----------------|-----------|-------------|----------|
| {如 1D [L]} | reshape [L,1] | [M,N] | reshape 回 [L] |
| {2D [M,N]} | 直接传入 | [M,N] | 直接输出 |

### 划分依据

{数据依赖分析 + Section 分隔判断}

### Phase 列表

| Phase | 目的 | 输入依赖 | 输出 | Section |
|-------|------|---------|------|---------|
| Phase1 | {purpose} | {dependency} | {output} | {section_vector / section_cube} |
| Phase2 | ... | ... | ... | ... |

> 若只有一个 Phase，标注"单 Phase"并说明为什么不需要拆分。

### 归约轴容量结论

> 含归约算子时必填；无归约时填"不涉及归约"。

- **归约轴是否可能超单 tile**：{是 / 否，依据 SPEC 动态轴范围}
- **是否采用多 tile 归约（online / 两遍法）**：{是，方案=... / 否，单 tile 装得下，依据=...}

### Phase 级数据流

```
{Phase 间的数据流向示意，标注 via GM workspace}
```

---

## §1 API 映射（R1 输出）

> **性能强制**：Vector 数值计算步骤须映射到 `vf.*` 指令序列（通过 `@pl.vector_function` 装饰器或 `@pl.inline` + `with pl.section_vf():` 块，在 `section_vector()` 内执行）。Cube 步骤用 `pl.*` Cube API。具体 vf 指令选用以 vf API 文档与官方指定算子为准。
>
> ⚠️ 本标注为性能强制要求，**不可改写为 `pl.*` Vector API 或其他非 vf 措辞**。若 vf.* 路径不可行，须按 R1 流程第 5 步标注 unsupported 并触发回退——不得通过改写本措辞使非 vf 方案"合规"。

### Phase1 API 调用序列

```
Phase 1 — {名称}:
  for j in range(n_tile_num):
    {API 调用}          // sync: {目的}
    ...
```

{每个 API 标注来源（API 文档路径或 EXPLORE_REPORT §4 定位的官方指定算子）}

### Phase2 API 调用序列

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

> 内存空间取 `Vec`(UB)/`Mat`(L1)/`Left`(L0A)/`Right`(L0B)/`Acc`(L0C)；`—` 表示默认布局（`Vec` 为行优先 ND，无需显式指定）；具体 layout 值以 API 文档和 EXPLORE_REPORT §3.3 为准。

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
- **理由**: {基于 R0 Phase 划分，说明 auto_mutex 如何覆盖本算子的 buffer 切换/互斥边界}

### R3↔R6 联合决策 / 待回填项

| 项目 | 当前结论 | 受 buffer 数影响的 tile/地址 | R6 回填结果 |
|------|---------|-----------------------------|------------|
| buffer 数 | {1 / 2 / 待定} | {tile 名单} | {R6 pipeline 深度确认后填写} |
| PONG 地址 | {无 / 预留} | {地址范围} | {是否启用、是否调整} |

---

## §4 循环与 Section 结构（R4 输出）

### 完整伪代码骨架

> R4 只搭循环骨架（含 `// sync: {目的}` 占位注释，沿用 R1 约定）。同步点的位置标注、目的归类与具体 API 统一由 R6 在 §6 完成。尾块处理代码（ceiling division、pl.min、set_validshape）由 R7 填入。

```python
# ⚠️ 编译期常量必须声明在 kernel 函数外（模块级）——
#    写进函数体内会触发编译错误
TS = {S_tile}
TD = {D_tile}
SCALE = 1.0 / sqrt({D_logical})

# 动态维度声明（具体 API 以 docs/ 和官方指定算子样例为准）
# 例：M = <动态维度声明>, N = <动态维度声明>

# jit 装饰器：@pl.jit() 或 @pl.jit(auto_mutex=True)
@pl.jit(auto_mutex=True)
def {op}_kernel(
    # 输入输出参数
):
    # ---- Tile 声明 ----
    ...

    # ---- SPMD 原语获取（位置：{section 内 / section 外，取决于 Phase 数}） ----
    num_cores = pl.get_block_num()   # 多 section 算子放 section 外；单 section 可放 section 内
    core_id = pl.get_block_idx()

    # ---- Section 声明 ----
    with pl.section_{vector/cube}():

        # ---- 尾块处理（R7 填入：ceiling division + pl.min + set_validshape） ----
        m_tile_num = ...
        n_tile_num = ...

        # SPMD M-tile 循环
        for i in pl.range(core_id, m_tile_num, num_cores):
            m_off = i * {M_tile_dim}

            for j in pl.range(0, n_tile_num, 1):
                n_off = j * {N_tile_dim}

                load_tile(tile_a, x, [i, j])       // sync: {目的}
                # ... 后续 compute/store ...
```

> 骨架中的 `// sync: {目的}` 仅为占位；同步点的完整标注与 API 选择见 §6（R6 输出）。

---

## §5 分核策略（R5 输出）

> 📌 权威依据：`$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials/programming_guide/programming_model/AI_Core_SIMD_programming/tile_based_python_programming/multi_core_partitioning_and_Tiling.md`（分核方式 / 负载均衡 / 核数设置 / tiling 传参一切以此为准）。

### 分核方式

- **方案**: strided loop —— {扁平切 `pl.range(core_id, m_tiles*n_tiles, num_cores)` / 二维切 外 `range(core_id, m_tiles, num_cores)`+内 `range(0, n_tiles, 1)`} + {选择理由}
- **host 侧核数**: `num_cores = min(get_platform_info().core_num, total_tiles)`

---

## §6 同步与核间流水（R6 输出）

> 同步规划始终填写；cross_core 流水部分仅当算子涉及跨 section/sub-block 数据传递时填写。单 section + 无跨 Phase 数据依赖的算子仅填写 pipe 内/pipe 间同步，不填写 cross_core。

### 同步 API 填充

在 §4 伪代码骨架的同步点占位处填入具体 API：

| 同步点位置 | 同步 API | 参数 | 目的 | 参考来源 |
|-----------|---------|------|------|---------|
| tile_group buffer 轮转 | `auto_mutex`（无额外手动 sync） | {mutex_ids 覆盖范围} | buffer 切换与互斥（框架自动管理） | {EXPLORE_REPORT §4 定位的官方指定算子} |
| make_tile scratch tile 的 pipe 级依赖 | `pl.system.sync_src`/`sync_dst(...)` | `set_pipe=..., wait_pipe=..., event_id=...` | 等 MTE2/V 等完成（仅限未由 auto_mutex 管理的 tile） | {EXPLORE_REPORT §4 定位的官方指定算子} |
| cross_core 跨核数据传递 | `pl.system.set_cross_core(...)` / `wait_cross_core(...)` | `pipe=..., event_id=...` | 跨核同步 | {EXPLORE_REPORT §4 定位的官方指定算子} |

### Pipeline 深度

- **pipeline 预取深度**（以实际算子为准，如 FA 的 QK_PRELOAD）: {深度}
- **FIFO_SIZE**: {= pipeline 预取深度 + 1}
- **回填 R3 结果**: {地址规划是否满足 double buffer / PONG 需求，哪些地址已按 buffer 数更新}

### event_id 分配表

> event_id 取值范围为 `[0, 16)`（API 文档 `set_cross_core_wait_cross_core.md` 参数范围表）。各流水组占不重叠区段。

| 流水组 | event_id 范围 | 用途 | 参考来源 |
|--------|--------------|------|---------|
| {流水组1} | `{如 [0, FIFO_SIZE)}` | {说明} | {EXPLORE_REPORT §4 定位的官方指定算子} |
| {流水组2} | `{如 [FIFO_SIZE, 2*FIFO_SIZE)}` | ... | ... |
| {流水组3} | `{如 [2*FIFO_SIZE, 3*FIFO_SIZE)}` | ... | ... |

> event_id 总用量须不超过 API 文档标注的上限 16。

### cross_core 同步点

| 位置 | 同步 API | 参数 | produce/consume | 参考来源 |
|------|---------|------|----------------|---------|
| {数据产生后} | `pl.system.set_cross_core(...)` | `pipe=..., event_id=...` | produce | {EXPLORE_REPORT §4 定位的官方指定算子} |
| {数据使用前} | `pl.system.wait_cross_core(...)` | `pipe=..., event_id=...` | consume | {EXPLORE_REPORT §4 定位的官方指定算子} |

---

## §7 尾块处理（R7 输出）

> 📌 权威依据：`$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials/programming_guide/programming_model/AI_Core_SIMD_programming/tile_based_python_programming/tail_block_handling.md`（尾块的完整机制——一切以此为准）。

### 尾块处理方案

> 在 §4 伪代码骨架的"尾块处理"占位处填入以下代码：

```python
# ceiling division 计算 tile 数（必须向上取整，用 N//TILE 直接整除会漏掉尾块）
m_tile_num = (M + TS - 1) // TS
n_tile_num = (N + TD - 1) // TD

# 循环内计算尾块有效尺寸并告知硬件（set_validshape 有状态，每轮都要重设）
valid_m = pl.min(M - m_off, TS)       # 满 tile = TS, 尾块 = 余数
valid_n = pl.min(N - n_off, TD)
pl.set_validshape(tile_a, [valid_m, valid_n])  # 运行时告知硬件
```

- **是否需要尾块填充**: {逐元素→否 / 归约或 matmul→是，具体填充方式见 tail_block_handling.md}

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
| §4 伪代码核心计算步骤无留空（`=` 赋值处不得为 `...`，不得标注"需 coder 实现"等；sync 占位 `// sync: {目的}` 和 R7 尾块占位除外） | ✅ / ❌ | {若有留空，回 R1 补充 API 或标注 unsupported 触发回退} |
| 数据依赖正确（Phase 顺序 + sync 点） | ✅ / ❌ | {依赖分析} |
| dtype 精度满足要求 | ✅ / ❌ | {FP32 matmul 累加 / ...} |
| 归约类 API 的 `[M,1]`/`[1,N]` 输出已设 `layout` | ✅ / ❌ | {回 R2 补 layout} |
| Acc tile 物理 M×N×dtype_bytes ≥ fractal（FP32 ≥ 1024 bytes；动态轴含极小维度时尤需检查） | ✅ / ❌ | {回 R2 pad tile shape} |

### 泛化性检查

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 目标测试 case（≥4，单轴算子按例外）已按 tile 切分确定具体 shape，且逐个验证 design 可适配（见 §8「目标测试 case」表） | ✅ / ❌ | {回 R7.5 补充 / 回溯适配不了的轮次} |
| 支持非对齐 M（M 尾块） | ✅ / ❌ | {ceiling division + set_validshape 设计} |
| 支持非对齐 N（N 尾块） | ✅ / ❌ | {同上} |
| 归约轴可能超单 tile 时已采用 online/两遍法（保证泛化性，R0 已判断） | ✅ / ❌ | {回 R0 重设归约方案} |
| 循环边界正确 | ✅ / ❌ | {valid_m/valid_n 计算验证} |
| 超越函数在 dtype 范围内无溢出 | ✅ / ❌ | {引用 §1 数值安全边界} |
| 跨 tile 状态初始化/持久化正确 | ✅ / ❌ | {expands 恒等值 / muls 拷贝；回 R0 或 R1 修正} |
| 同步策略在动态轴全范围下正确 | ✅ / ❌ | {num_cores / pipeline defer / event 隔离} |

### 一致性检查

| 检查项 | 结果 | 说明 |
|--------|------|------|
| R0-R7 输出无矛盾 | ✅ / ❌ | {交叉验证} |
| 所有决策有证据支撑 | ✅ / ❌ | {证据链检查} |
| 各内存空间（UB/L1/L0A/L0B/L0C）tile 总用量分别不超各自容量上限（R3 逐空间验证，含 cube 时须查 L1/L0） | ✅ / ❌ | {回 R3 重排地址 / R2 缩 tile} |
| `tile_dims` 使用时已关注大 stride 对性能的影响 | ✅ / ❌ | {回 R2 调整布局} |
| 条件性检查（如跳过 R6，确认无跨核数据传递） | ✅ / ❌ | {R0 重新评估} |

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
| `---` | 数据跨 Phase 持久化（tile 在不同 Phase 间不被覆盖） |
| `├─` | 同一 Phase 内分支（同一数据被多次使用） |
