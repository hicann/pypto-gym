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

### Phase 级数据流

```
{Phase 间的数据流向示意，标注 via GM workspace}
```

---

## §1 API 映射（R1 输出）

### Phase1 API 调用序列

```
Phase 1 — {名称}:
  for j in range(n_tile_num):
    load_tile(tile_a, x, [i, j])          // sync: 等MTE2完成
    row_max(redu_col, tile_a, tile_tmp)   // sync: V pipe内依赖
    ...
```

{每个 API 标注来源（API 文档路径或 EXPLORE_REPORT §4 定位的样例）}

### Phase2 API 调用序列

```
...
```

### 超越函数数值安全边界

> 条件性：仅当 API 链含 exp/log/sqrt/reciprocal/tanh 等超越/非线性函数时填写。

| API | 输入理论范围 | 目标 dtype 上限 | 是否溢出 | 防护措施 |
|-----|-------------|----------------|----------|----------|
| `pl.exp` | {如 2t, x>4.3 时 2t>11} | fp16 ≈ 65504 | exp(11.09)≈65504 → +inf → NaN | exp 前 `pl.mins(_, 11.0)` 截断（tanh 精度损失 <5e-5） |

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

| 用途 | 变量名 | shape | dtype | layout | 大小 | 备注 |
|------|--------|-------|-------|--------|------|------|
| 输入暂存 | `tile_a` | `[64,128]` | FP32 | `—` | 32768 | valid_shape=[-1,-1] |
| ... | ... | ... | ... | {— 或 pl.DN} | ... | {valid_shape 等} |

> `—` 表示默认 `pl.ND`（行优先，无需显式指定）；`pl.DN` 表示维度转置布局（归约输出 `[M,1]` 必需，见 `row_max.md:25`）。

### tile_dims stride 安全性检查

> 仅当使用了 `load_tile` + `tile_dims=[d0, d1]`（覆盖多维度）时填写。详见 SKILL.md R2 步骤 4。

| load_tile/store_tile | tile_dims | d0 在张量布局中的 stride | ≤ EXPLORE_REPORT §7 阈值? | 结果 |
|----------------------|-----------|------------------------|---------|------|
| {tile_a, x, [i,j]} | {如 [0,1]} | {计算值} | ✅ / ❌ | {通过 / 需 permute} |

---

## §3 UB 空间布局（R3 输出）

### UB 地址映射表

| 用途 | 变量名 | shape | dtype | layout | 地址 | 大小 | 备注 |
|------|--------|-------|-------|--------|------|------|------|
| ... | ... | ... | ... | {— 或 pl.DN} | ... | ... | {valid_shape / 双视图等} |

**UB 总用量**: {∑ 大小} bytes / {EXPLORE_REPORT §7 UB 容量} bytes = {百分比}

### 双视图对（如有）

> 归约类 API 的 `[M,1]`/`[1,N]` 输出须设 `layout=pl.DN`（证据 `row_max.md:25`）；若该输出后续参与 tile×tile 逐元素运算（需默认 ND），须在同地址建 DN + ND 双视图对（证据 `pro_ops/fa/test_fa_performance.py:478-483`）。

| 双视图对 | DN tile (`layout=pl.DN`) | ND tile (默认) | 共用地址 | 读/写 |
|---------|--------------------------|----------------|---------|-------|
| ... | `{name} [M,1] layout=pl.DN` | `{name} [1,M]` | `0x...` | 仅读 |

### 分配方式选择

> 首选 make_tile_group + auto_mutex；make_tile 作为次选并存。

- **方案**: make_tile_group + auto_mutex（首选，由框架自动管理 buffer 切换与 core 内互斥）{；如有单实例 tile 用 make_tile}
- **依据样例**: {EXPLORE_REPORT §4 定位的最相似 pro_ops 样例路径}
- **理由**: {基于 R0 Phase 划分，说明 auto_mutex 如何覆盖本算子的 buffer 切换/互斥边界；若混用 make_tile，说明哪些 tile 归入该路径}

### R3↔R6 联合决策 / 待回填项

| 项目 | 当前结论 | 受 buffer 数影响的 tile/地址 | R6 回填结果 |
|------|---------|-----------------------------|------------|
| buffer 数 | {1 / 2 / 待定} | {tile 名单} | {R6 pipeline 深度确认后填写} |
| PONG 地址 | {无 / 预留} | {地址范围} | {是否启用、是否调整} |

---

## §4 循环与 Section 结构（R4 输出）

### 完整伪代码骨架

> R4 只搭循环骨架 + 同步点占位。尾块处理代码（ceiling division、pl.min、set_validshape）由 R7 填入。

```python
# ⚠️ 编译期常量必须声明在 kernel 函数外（模块级）——
#    写进函数体内会触发 `Unsupported kwarg type for key: memref_size`（见 develop pitfalls §1.2）
TS = {S_tile}
TD = {D_tile}
SCALE = 1.0 / sqrt({D_logical})

# DynVar 声明
M = pl.DynVar('M')
N = pl.DynVar('N')

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

                load_tile(tile_a, x, [i, j])       // sync: {位置和目的}
                # ... 后续 compute/store ...
```

### 同步点位置标注

| 位置 | 同步目的 | 参考来源 |
|------|---------|---------|
| 每个 tile 迭代开头 | {如"等前序 MTE3 store + V compute 完成"} | 样例: {path} |
| load_tile 之后 | {如"等 MTE2 搬运完成"} | ... |

> 具体同步 API（bar_v / sync_src / sync_dst / set_cross_core / wait_cross_core）由 R5/R6 填入。

---

## §5 分核策略（R5 输出）

### 分核方式

- **方案**: strided loop（`pl.range(core_id, m_tile_num, num_cores)`）
- **bar_all**: {不需要 / 需要，条件性——仅当核间有数据依赖时}

---

## §6 同步与核间流水（R6 输出）

> 同步规划始终填写；cross_core 流水部分仅当算子涉及跨 section/sub-block 数据传递时填写。单 section + 无跨 Phase 数据依赖的算子仅填写 pipe 内/pipe 间同步，不填写 cross_core。

### 同步 API 填充

在 §4 伪代码骨架的同步点占位处填入具体 API：

| 同步点位置 | 同步 API | 参数 | 目的 | 参考来源 |
|-----------|---------|------|------|---------|
| load_tile 后 | `pl.system.sync_dst(...)` | `set_pipe=MTE2, wait_pipe=V, event_id=...` | 等 MTE2 完成 | {EXPLORE_REPORT §4 定位样例} |
| V pipe 内计算间 | `pl.system.bar_v()` | — | V pipe 内依赖 | {EXPLORE_REPORT §4 定位样例} |
| tile_group 切换处 | `{auto_mutex / 无额外同步}` | {覆盖范围说明} | {buffer 切换与互斥} | {EXPLORE_REPORT §4 定位样例} |

### Pipeline 深度

- **QK_PRELOAD**: {深度}
- **FIFO_SIZE**: {= pipeline 预取深度 + 1}
- **回填 R3 结果**: {地址规划是否满足 double buffer / PONG 需求，哪些地址已按 buffer 数更新}

### event_id 分配表

| 流水组 | event_id 范围 | max_event_id | 用途 | 参考来源 |
|--------|--------------|-------------|------|---------|
| QK | `[0, FIFO_SIZE)` | FIFO_SIZE | {说明} | {EXPLORE_REPORT §4 定位样例} |
| P | `[FIFO_SIZE, 2*FIFO_SIZE)` | 2*FIFO_SIZE | ... | ... |
| PV | `[2*FIFO_SIZE, 3*FIFO_SIZE)` | 3*FIFO_SIZE | ... | ... |

> `assert 3 * FIFO_SIZE <= {EXPLORE_REPORT §7 event_id 上限}`（cross_core event_id 上限）

### cross_core 同步点

| 位置 | 同步 API | 参数 | produce/consume | 参考来源 |
|------|---------|------|----------------|---------|
| {数据产生后} | `pl.system.set_cross_core(...)` | `pipe=..., event_id=..., max_event_id=...` | produce | {EXPLORE_REPORT §4 定位样例} |
| {数据使用前} | `pl.system.wait_cross_core(...)` | `pipe=..., event_id=..., max_event_id=...` | consume | {EXPLORE_REPORT §4 定位样例} |

---

## §7 尾块处理（R7 输出）

### 尾块处理方案

> 在 §4 伪代码骨架的"尾块处理"占位处填入以下代码：

```python
# ceiling division 计算 tile 数
m_tile_num = (M + TS - 1) // TS
n_tile_num = (N + TD - 1) // TD

# 循环内计算尾块有效尺寸并告知硬件
valid_m = pl.min(M - m_off, TS)       # 满 tile = TS, 尾块 = 余数
valid_n = pl.min(N - n_off, TD)
pl.set_validshape(tile_a, valid_m, valid_n)  # 运行时告知硬件
```

### 跨迭代恒等值初始化

- **初始化方案**: {如 `expands(gmax, -1e9)` 消除首迭代分支}
- **条件分支（如需）**: {如 `is_tail = pl.min(1, TS - actual)` 标志位 + 模块级常量}

### 与 R6 的耦合（如涉及核间流水）

- **尾块 shape 流转**: {如"actual_sq 随 ctx_arr 在 event_id FIFO 中流转，consume 端错位取 ctx"}
- **参考来源**: {如"EXPLORE_REPORT §4 定位的深预计算样例中 ctx_arr 错位取尾块 shape 模式"}

---

## §8 综合评估（R8 输出）

### 准确性检查

| 检查项 | 结果 | 证据 |
|--------|------|------|
| API 调用链完整覆盖数学公式 | ✅ / ❌ | {映射验证} |
| 数据依赖正确（Phase 顺序 + sync 点） | ✅ / ❌ | {依赖分析} |
| dtype 精度满足要求 | ✅ / ❌ | {FP32 matmul 累加 / ...} |

### 泛化性检查

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 支持非对齐 M（M 尾块） | ✅ / ❌ | {ceiling division + set_validshape 设计} |
| 支持非对齐 N（N 尾块） | ✅ / ❌ | {同上} |
| 循环边界正确 | ✅ / ❌ | {valid_m/valid_n 计算验证} |
| 超越函数在 dtype 范围内无溢出 | ✅ / ❌ | {引用 §1 数值安全边界} |
| 跨 tile 状态初始化/持久化正确 | ✅ / ❌ | {expands / muls 拷贝} |
| 同步策略在动态轴全范围下正确 | ✅ / ❌ | {num_cores / pipeline defer / event 隔离} |

### 一致性检查

| 检查项 | 结果 | 说明 |
|--------|------|------|
| R0-R7 输出无矛盾 | ✅ / ❌ | {交叉验证} |
| 所有决策有证据支撑 | ✅ / ❌ | {证据链检查} |
| `tile_dims` 最外层维度 stride 不超过 EXPLORE_REPORT §7 探测阈值（R2 步骤 4 已检查） | ✅ / ❌ | {回 R2 调整布局} |
| 条件性检查（如跳过 R6，确认无跨核数据传递） | ✅ / ❌ | {R0 重新评估} |

### 评估结论

- **整体**: {通过 / 需修改}
- **限制条件**: {如不支持尾块、需 M 整除完整 M-tile 尺寸等}
- **修改记录**: {修改内容、轮次、原因}

---

## §9 Tile 数据流全景图

将 R0-R7 各轮产出的片段串联为一张完整的 tile 级数据流图。标注每一块 tile 的流向：从哪里读取、经过哪些操作转换、写入哪里。

```
{按实际算子的 tile 流向绘制}

GM ─[load_tile]→ tile_a ─→ {操作} → {输出tile} → ...
                                     ↓
                              {操作}({输出tile}, {持久化tile})
                                     ↓
                              {持久化tile} → [双视图] {col_view}
                                                      ↓
                              tile_a → {操作}({col_view}) → ... → tile_out
                                                                      ↓
GM ←[store_tile]─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
```

### 图例

| 标记 | 含义 |
|------|------|
| `[load_tile]` | MTE2 搬运：GM → UB |
| `[store_tile]` | MTE3 搬运：UB → GM |
| `→` | V 流水线操作，标注 API 名 |
| `[双视图]` | 同地址双视图转换（`pl.DN` ↔ 默认 `pl.ND`），仅标注如有 |
| `---` | 数据跨 Phase 持久化（tile 在不同 Phase 间不被覆盖） |
| `├─` | 同一 Phase 内分支（同一数据被多次使用） |
