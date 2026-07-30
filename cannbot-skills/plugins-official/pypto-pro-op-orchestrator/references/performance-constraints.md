# PyPTO-Pro 两条性能强制

> **用途**：PyPTO-Pro 工作流的性能约束。被 `pypto-pro-material-explore`（探索阶段 API 映射）、`pypto-pro-op-design`（设计阶段落实）、`pypto-pro-op-develop`（实现阶段遵守）三个 skill 共同引用。各 skill 不重复定义，统一指向本文件。
>
> **安装路径**：源码位于 `plugins-official/pypto-pro-op-orchestrator/references/`，由 `init.sh` 软链到 `.opencode/references/`。各 skill 引用时写「见 `.opencode/references/performance-constraints.md`」。

以下两条是性能约定，探索阶段的 API 映射与样例参考、设计阶段的 tile 规划、实现阶段的 kernel 编码均须遵守。否则产出的算子性能可能不可接受。

---

## 强制 1：buffer 管理用 `make_tile_group` + `auto_mutex`

| 项目 | 规则 |
|------|------|
| buffer 轮转 tile（含 double buffer） | `make_tile_group` 创建，`auto_mutex=True` 自动管理轮转与互斥 |
| `make_tile` 合法用途 | 仅单次 scratch tile（写入一次、读取一次、不参与轮转循环，如一次性中间结果暂存、不迭代的归约标量结果）|
| 禁止 | `make_tile` + 手动 `sync_src`/`sync_dst` 管理轮转 |

**同步分工**（auto_mutex 管核内，跨核另用手动 API，命名空间独立）：

| 范围 | 机制 | 标识符 | 说明 |
|------|------|--------|------|
| 核内 pipe 间 | `auto_mutex=True`（框架自动插 `mutex_lock`/`mutex_unlock`） | `mutex_id` ∈ [0,31] | 覆盖 `make_tile_group` tile 的 buffer 互斥；禁止在其上叠加手动 `sync_src`/`sync_dst`，否则依赖环导致 AICore timeout 死锁 |
| 跨核（AIC↔AIV） | `set_cross_core`/`wait_cross_core`（手动） | `event_id` ∈ [0,16) | auto_mutex 不覆盖跨核依赖 |

> `mutex_id` 与 `event_id` 是两个独立命名空间（FA 样例中数值重叠共存无报错），不构成冲突。

---

## 强制 2：Vector 数值计算用 `vf.*` 手写

| 项目 | 规则 |
|------|------|
| Vector 数值计算（逐元素/归约/广播/非线性） | `vf.*` 指令手写，在 `section_vector()` 内通过 `@pl.vector_function` 执行 |
| 禁止 | `pl.*` 计算 API 用于 Vector 数值计算（无论在何处编写）|
| 不受此限（数据搬运与控制流） | `pl.load`/`pl.store`/`pl.load_tile`/`pl.store_tile`/`pl.range`/`pl.get_block_idx` 等 |
| 等价写法① | `@pl.vector_function` 装饰器（通常模块级，从 `section_vector()` 调用，见 FA；也可工厂函数内定义，见 lightning）|
| 等价写法② | `@pl.inline` + `with pl.section_vf():` 块（见 vf API 文档示例）|

> **vf.\* 在 VF 寄存器中完成计算，寄存器不占 UB 空间**——VF 路径的 UB 占用不会高于 Memory 级路径。UB 超限不是放弃 VF 的理由（UB 超限处理见 design SKILL R3 重排地址 / R2 缩 tile）。
