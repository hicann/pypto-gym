---
name: pypto-pro-op-develop
description: 实现、调试并验证 PyPTO-Pro 算子 kernel，覆盖纯 Vector、纯 Cube 与融合数据流。当 DESIGN.md 和 Module 契约已通过 Stage 3 门禁，需要根据 DESIGN.md 编写实现、补齐测试、排查编译或精度问题，或交付可运行的算子代码时使用。L0 路径（纯 vec/纯 cube）一口气产出 `test_{op}.py`；L1 路径（融合算子）逐 Module 产出 staged 文件 `modules/test_{op}_module{suffix}.py`。产物交给独立 verifier；不要改写 SPEC 或自行认证 Stage 完成。
---

# PyPTO-Pro 算子 Kernel 实现

生成完整的 PyPTO-Pro kernel 实现文件（一个 `.py` 文件，含 kernel 函数 + 测试函数），并本地跑通验证。运行结果若暴露 DESIGN.md 问题，返回 `design_violation` 证据并由编排器回退 Stage 3；本阶段不直接修改上游设计。

> **角色说明**：本 skill 描述 Stage 4 的**实现方法**——承担者需自行完成 **开发 → 自验证 → 发现问题 → 分析根因 → 解决问题 → 再自验证** 的闭环，直到自认为可交付。**但门禁判定不由实现者做**：交付后由独立的 verifier 裁决，verifier 只检查不改代码。自验证通过不等于通过门禁，不要据此跳过或简化交付前自检。

> **方法论优先**：本 skill 以**思维方法指导**为主，不教具体写法——具体 API 用法、vf 指令组合、tile 配置、同步写法等请查阅 API 文档（`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/`）、教学文档（`$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials`）、官方指定算子（`PRO_MATERIAL_INDEX.md` §B），理解后据实实现。

---

## 实现约束

> 实现阶段只执行已冻结的 DESIGN.md，不重新裁定实现层级。
> 1. 所有需要 buffer 切换/轮转的 tile（含 double buffer）一律用 `make_tile_group` + `auto_mutex`，由框架自动管理 buffer 切换与互斥。`make_tile` 仅限**单次使用 scratch tile**（写入一次、读取一次、不参与 buffer 切换/轮转循环，如一次性中间结果暂存、不迭代的归约标量结果）。**禁止用 `make_tile` + 手动 `sync_src`/`sync_dst` 管理 buffer 轮转**。
>
>    同步方案按 DESIGN.md §6 施工——auto_mutex 管核内 pipe 互斥（禁止在其管理的 tile 上叠加 `sync_src`/`sync_dst`，否则死锁）；跨核同步用 `set_cross_core`/`wait_cross_core`（手动）。具体同步点与 event_id 分配已在 §6 确定。
>
> 2. **Vector 数值计算按 DESIGN.md 已冻结的 `vector_selection` 施工**，不得生成双候选或改写 DESIGN。

---

## 知识库

按 [`pypto-pro-op-kb/ROUTER.md`](../../pypto-pro-op-kb/ROUTER.md) 每次只打开一个与当前实现决策相关的
补充参考。只有 [pattern selector](../../pypto-pro-op-kb/patterns/pattern-index.md) 标为
`validated skeleton` 的代码片段可作为起点；`conceptual only` 条目必须回到目标

>
> **`validated skeleton` 保证代码存在，不保证它在你的工作区。** 本仓把知识库与
> 验证它的算子树拆在不同分支：pattern 页会写明产物留存在哪个分支。把它当作代码起点
> 之前，先确认能取到——`git cat-file -e <branch>:<path>`，取不到就当 `conceptual only` 用，
> 不要凭页面描述照写。
SDK 的官方文档和样例补齐验证。平台专属参考须先确认目标平台。

**写第一版之前先看一眼原语代价表**：
[`pypto-pro-op-perf-tune`](../pypto-pro-op-perf-tune/SKILL.md) 的「优化经验」一节。
它是实测的——`vf.gather` ~20 ns、`vf.scatter` ~18 ns、UB 往返（含必需的 `mem_bar`）~16 ns，
而 `load_align`/`store_align` **< 1 ns**、算术 ~0.3 ns。**跨 lane 比不跨 lane 贵 20–35 倍，
且没有更便宜的写法**，所以"要不要跨 lane"是写代码时就定死的结构选择，不是事后能调回来的
参数。同一节还列了 mask 提升、tail 循环位置、最小合法 C、pitch padding 等写法，
以及每一条对应的实测反例。这属于**实现期**知识，不必等到调优阶段才读。

---

## 输入

### 主要依据（必读）

| 来源 | 内容 | 用途 |
|------|------|------|
| `custom/<op>/DESIGN.md` | Module 划分（§0）、API 映射（§1）、Tile 规划（§2）、UB 空间布局（§3）、循环与 Section（§4）、分核/流水/尾块（§5-7）、目标测试 case（§8）、全景图（§10） | 已通过 Stage 3 的实现合同；发现设计问题时返回证据并回退，不在本阶段修改 |
| `custom/<op>/EXPLORE_REPORT.md` | API 约束（§3）、相似样例与可复用模式（§4）、教程指导（§5）、Tile/同步策略建议（§6） | **编码参考**——API 约束速查、样例写法定位、教程设计指导 |

### 按需取用（EXPLORE_REPORT.md 不足时查阅）

| 来源 | 内容 | 用途 |
|------|------|------|
| `custom/<op>/PRO_MATERIAL_INDEX.md` | API 文档（§A）、官方指定算子（§B）、教程（§C）的精确路径索引 | **路径定位**——先查索引找到文档/样例路径，再读取原文 |

## 参考文件

| 文件 | 用途 | 加载时机 |
|------|------|----------|
| [templates/pure_vec_impl_template.py.tmpl](templates/pure_vec_impl_template.py.tmpl) | 纯 vec 通用骨架；使用前按目标 API 与 DESIGN.md 核对 | 纯 vec 算子生成代码前 |
| `$PYPTO_DEVKIT_DIR/pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset_dynamic.py` | **纯 cube 算子候选实现起点**——两级 K 分块 + move offset + 嵌套四分支 K 累加 + 尾块处理 + ASW 蛇形调度；仅在该路径存在且目标版本匹配时使用 | 纯 cube 算子生成代码前 |
| [templates/impl_template.py.tmpl](templates/impl_template.py.tmpl) | kernel 文件骨架（tile 声明 + section + Module + 测试函数）——**通用**（含 CV 融合 / 多 Module / 跨核流水）；CV 融合算子当前无专用模板，走此通用骨架 + 步骤 0~8 | 生成代码前必读 |
| [references/debugging-methodology.md](references/debugging-methodology.md) | 调试方法论：**先确认失败可信** → 症状快查表 → 分层 review → 定位技术（最小复现/消融/单原语替换/差异属性）→ 升级切换 → **修正后验证** + 诊断纪律 | 验证失败进入 debug 状态时必读 |
| [references/vf-reduction-perf.md](references/vf-reduction-perf.md) | Vector reduction 的数值安全、正确 API 形态与性能注意事项 | 实现或调优 Vector reduction 时 |
| [../../pypto-pro-op-kb/ROUTER.md](../../pypto-pro-op-kb/ROUTER.md) | 按任务选择一个补充约束、pattern 或 validated study kernel | API 文档与官方样例不足时 |
| [scripts/list_idle_chip_ids.sh](scripts/list_idle_chip_ids.sh) | 查找空闲 NPU chip | 运行前按需执行 |


---

## 开发流程

### 实现策略与步骤适用范围

所有算子（纯vec / 纯cube / CV融合）都必须完整走步骤 0~8 及 debug 流程——**不存在"跳过步骤直接套模板"**。各步骤的适用范围与参考来源如下：

| 步骤 | 纯vec / 纯cube | CV融合 | 说明 |
|------|---------------|--------|------|
| 0：响应编排器 dispatch | 所有算子统一 | 同左 | L0/L1 路径由编排器决定，coder 不自行判断 |
| 1~4：确认输入 → API 确认 → tile 声明 → section 骨架 | 所有算子统一 | 同左 | 模板不能替代——模板只提供骨架，参数和结构来自对 DESIGN.md 的确认 |
| 5~6：编写实现 + 测试函数 | **模板可作为起点**：按模板内置骨架填充（CONFIG 常量 + `>>> FILL` 标记），模板已内置同步、尾块处理，但须先按目标 API、DESIGN.md 与当前平台核对 | **非模板优先**：以官方 c-v 融合算子（PRO_MATERIAL_INDEX.md §B）为优先参考，纯vec/cube模板仅作局部写法参考 | 步骤 5~6 按算子类型分叉 |
| 7~8 + debug：本地验证 → 自修复闭环 → 交付前自检 | 所有算子统一 | 同左 | 照常执行 |

**兜底路径**：步骤 5~6 套用模板 / 参考样例后运行失败时，按步骤 7 的 debug 自修复闭环修复。

### 步骤 0：响应编排器 dispatch（L0/L1 路径由编排器决定）

> **L0/L1 路径决策权在编排器**——编排器在 Stage 3 完成时读 `module_interfaces.yaml` 的 `is_fusion`，调 `plan_stage4` 设置 `stage4_path`。coder 不自行判断路径，由 dispatch prompt 的 module 参数决定行为。

**L0 路径**（dispatch prompt **不带** module 参数，`is_fusion == false`）：

纯 vec / 纯 cube 算子，一口气开发完毕。按下面步骤 1→7 一次走完（步骤 3-7 各处理全部 Module），产出 `test_{op}.py`。简单算子首跑失败面本就不大，无需增量。

**L1 路径**（dispatch prompt **带** module_k 参数，`is_fusion == true`）：

融合算子（cube+vec），逐 Module 开发。每次 dispatch 只产一个 staged 文件 `modules/test_{op}_module<suffix_k>.py`：

> **⚠️ 单 kernel 铁律（L1 路径核心约束，违反即失败）**：每个 staged 文件中**只允许一个 `@pl.jit` kernel 函数**。逐 Module 开发是指在**同一个 kernel 函数内增量追加** Module k 的 section/tile/计算逻辑，**不是为每个 Module 新建一个 kernel**。Module k 的 staged 文件 = 复制 Module k-1 的 staged 文件 → 在同一个 kernel 函数内追加 Module k 的实现。如果文件中出现多个 `@pl.jit`，视为严重违规。

> **⚠️ suffix_k 命名规则（必须严格遵守）**：`suffix_k` = **累积 Module 序号拼接**，不是当前 Module 序号。Module 1 → `module1`，Module 2 → `module12`，Module 3 → `module123`，Module 5 → `module12345`。**禁止**用 `module2`、`module3` 这种非累积命名。wrapper 函数名同理：`{op}_wrapper_module<suffix_k>`（如 `{op}_wrapper_module12`）。

1. **staged 文件是完整可运行的算子**：实现 Module 1..k，输出 Module k 的结果作为该文件的最终输出。test 函数导入 `modules/{op}_golden_stage<suffix_k>`（由 mathematician 一次性产出），跑 kernel，用 `_assert_precision` 比对 kernel 输出 vs golden 输出
2. **生成 Module k 时参考 Module k-1**：参考前一轮的 `test_{op}_module<suffix_{k-1}>.py`（已验证通过），在**同一个 kernel 函数内**追加 Module k 的实现，不推翻前序已跑通的代码
3. **步骤 1-2 照常**（全局确认输入与 API 参数）；**步骤 3-7 只处理到当前 Module k**——只声明该 Module 用到的 tile、只搭到该 Module 的 section / 循环、只填该 Module 的实现
4. **wrapper 函数名**：`{op}_wrapper_module<suffix_k>`（签名与 `primary_inputs` 一致，输出 Module k 的结果）
5. 验证失败时在本轮内走步骤 7 的 debug 迭代流程直到 PASS

> **为什么逐 Module 有效**：分轮后，进入 Module k+1 时前 k 个 Module 已确认正确，失败面收缩到"新增 Module + 衔接"，与 [references/debugging-methodology.md](references/debugging-methodology.md) 的"最小复现"思路一致。增量以**追加**为主，不推翻重写。

> ⚠️ **L1 路径不在此做 cleanup**——所有 Module 验证通过后，编排器调度 cleanup（从最后一个 staged 文件生成 `test_{op}.py`，staged 文件链全部保留）。coder 在 L1 路径的每次 dispatch 只负责产出一个 staged 文件。

### 步骤 1：确认输入齐全

读取 DESIGN.md，确认施工所需信息完整——各信息对应位置：

- **§0** Module 划分与数据依赖、维度契约
- **§1** 每个 Module 的 API 映射序列
- **§2/§3** Tile 规划（shape/dtype/layout）与片上地址映射
- **§4** 循环与 Section 结构
- **§5-7** 分核 / 流水 / 尾块策略
- **§8** 目标测试 case（直接按此实现测试，不自行重算 shape）
- **§10** Tile 数据流全景图（确认全局理解）
- **EXPLORE_REPORT.md §3-5**：API 约束 / 相似样例 / 教程，步骤 2/5 按需查阅

信息不足时优先查 EXPLORE_REPORT.md / API 文档 / 官方指定算子核对；仍缺失（尤其 §0/§1/§2/§3/§8 关键项）或运行证据推翻 DESIGN.md 时，停止施工并返回 `design_violation`，由编排器回退 Stage 3 修订并重新过门禁。coder 不猜测、不直接改写 DESIGN.md。

> **L1 路径额外确认**：
> - 读取 `module_interfaces.yaml`，确认当前 Module k 的 `inputs`（来源：`primary` 或 `module_<j>`）、`outputs`（Module k 的输出张量及 shape/dtype）、`golden_steps`（该 Module 的数学步骤）
> - 确认前一轮的 staged 文件 `modules/test_{op}_module<suffix_{k-1}>.py` 已存在且已验证通过（Module 1 无前序，跳过此项）
> - 确认对应的 per-Module golden `modules/{op}_golden_stage<suffix_k>.py` 已由 mathematician 产出
> - 确认当前 Module k 的 section 类型（cube / vector），以及与前一 Module 的数据衔接方式

### 步骤 2：逐 API 确认参数

对 DESIGN.md §1 中的每一个 API 调用，**必须**确认三个信息：

1. **参数名是否为关键字参数**：大部分是位置参数，少数（如 `set_pipe=`、`wait_pipe=`、`event_id=`）是关键字参数。参数传参方式（位置 vs 关键字）以 API 文档原文为准，EXPLORE_REPORT §3.3 已汇总部分
2. **参数顺序**：如 `pl.matmul(dst, a, b, phase=...)`（dst 先于操作数），`pl.load_tile(tile, tensor, [i, j])`（tile 先于 tensor）。具体顺序以 API 文档为准
3. **约束条件**：dtype 限制、layout 要求、MemorySpace 约束——以 API 文档原文为准，EXPLORE_REPORT §3.3 已汇总部分

**确认方式**（先汇总后原文，减少阅读量）：
1. **先查 EXPLORE_REPORT.md §3**（API 映射结果 + API 约束 + MemorySpace 约束已汇总）——大部分信息在此可直接获得，无需读原文
2. **§3 不足时**，再通过 `PRO_MATERIAL_INDEX.md` §A 定位对应 API 文档路径，读取文档原文的"函数原型"和"参数范围"两个表补充

**产出要求**：步骤 2 完成后，应形成一份覆盖 DESIGN.md §1 全部 API 的参数速查清单（vec 步骤为 `vf.*`、cube 步骤为 `pl.*`），后续步骤直接复用，不再重复确认。

> **模式提示**：以下步骤 3-7 以**L0 路径**为默认写法。L1 路径按步骤 0 的 Module 分轮收窄处理范围。

### 步骤 3：编写 tile 声明

> **模板选择**：按上方「实现策略与步骤适用范围」节判定算子类型并选择对应起点。纯vec 填写 CONFIG 常量（N / ROWS / TILE_ROWS / NUM_CORES 等，来自 DESIGN.md §2），按 §3 确认地址无重叠；纯cube 的 tile 声明（Mat 4-buffer / Left 2-buffer / Right 2-buffer / Acc 单 buffer）照抄官方样例 `make_tile_group` 配置，按 §2/§3 调整 shape/dtype/addr；CV 融合走通用模板 `impl_template.py.tmpl`。

骨架与格式参考 [templates/impl_template.py.tmpl](templates/impl_template.py.tmpl)（tile 声明 / section / 循环 / 测试的完整结构）。

1. **编译期常量（TS、TD、SCALE 等）复制到模块级**——从 DESIGN.md §2.1 取值；⚠️ 必须声明在 kernel 函数**外部**（模块级）。写进 kernel 函数体内会被 JIT 当作 IR 语句处理，触发编译错误。
2. **动态维度声明**——从 DESIGN.md §0 的维度契约取动态轴名称，按 `docs/` API 文档和官方指定算子样例中的正确声明方式编写（声明方式以文档/样例为准，不臆测）。循环内通过 `tensor.shape[i]` 获取动态维度值
3. **逐条写 tile 声明**——按 DESIGN.md §2 Tile 属性表 + §3 地址映射表，精确复制每个 tile 的 shape / dtype / layout / addr / size；需要 buffer 切换/轮转的 tile 用 `make_tile_group` + `auto_mutex`，单次使用 scratch tile 用 `make_tile`；地址无重叠、归约输出等 layout 约束均已在 DESIGN §2/§3 定好，此处照抄即可。运行验证发现地址/属性有误时返回 `design_violation`，不得在 Stage 4 私改设计。

### 步骤 4：编写 section 和循环框架

> **纯vec算子**：`pure_vec_impl_template.py.tmpl` 可提供 section、stride
> 循环与 tile-group 骨架；按 DESIGN.md 选择的 tile-op 或 `vf.*` 数据流填充，
> 不把模板当作 API 或性能证据。

> **纯cube算子**：仿照 `test_matmul_perf_asw_4k_dn_move_offset_dynamic.py` 搭建 `section_cube()` + K 分块循环（外层 KL1 wide load、内层 KL0 sub-tile move）+ `set_mm_layout_transform(enabled=True)` 骨架。tile 声明（Mat 4-buffer / Left 2-buffer / Right 2-buffer / Acc 单 buffer）照抄官方样例的 `make_tile_group` 配置，按 DESIGN.md §2/§3 调整 shape / dtype / addr。ASW 蛇形调度为可选性能优化，简单场景用 2D tile 迭代（`idx // N_TILES` + `idx % N_TILES`）即可。

> **纯cube补充**：
> [`pypto-pro-op-kb/patterns/cube-only.md`](../../pypto-pro-op-kb/patterns/cube-only.md)
> 只有 single-block 代码是 validated skeleton；K-loop phase 仅为概念规则，
> 必须以目标 SDK 的官方 K-loop 样例核对后实现。

> **CV融合算子**：无专用模板，按 DESIGN.md §4 搭建 `section_cube()` + `section_vector()` 两段式 section 骨架，Cube/Vector 间的数据传递与跨核同步（set_cross_core/wait_cross_core）按 DESIGN.md §6 施工；具体写法参考官方指定算子中的 c-v 融合算子（通过 PRO_MATERIAL_INDEX.md §B 定位）。

按 DESIGN.md §4（循环与 Section 结构）搭出 kernel 骨架：section 声明、SPMD 原语获取位置（多 section 在 section 外闭包共享，单 section 在内）、循环嵌套、同步点占位注释。此时**不填入具体 API 调用和尾块代码**（步骤 5 完成）。骨架结构见 [templates/impl_template.py.tmpl](templates/impl_template.py.tmpl)。

### 步骤 5：编写 Module 内部实现

> **纯vec算子**：按 DESIGN.md §1 已冻结的 `vector_selection` 实现。使用 `vf.*` 时先读
> [references/vf-reduction-perf.md](references/vf-reduction-perf.md) 并逐项核对
> 当前 API 文档；使用 tile-op 时同样核对 dtype、layout 与 valid-shape 行为。

> **纯cube算子**：按官方样例 `test_matmul_perf_asw_4k_dn_move_offset_dynamic.py` 的嵌套四分支 K 累加编写——首块用 `pl.matmul`（`gsub==0`），其余块用 `pl.matmul_acc(acc, acc, ...)`；末块传 `phase=pl.AccPhase.Final`，中间块传 `Partial`，`gsub==0 && gsub==last_sub`（K_BLOCKS==1）传 `Final`。move 前扩展 valid_shape 到 full tile（fixpipe 必须看 whole tile），store 前 `set_validshape(acc, [valid_m, valid_n])` 缩小到有效窗口并传 `phase=pl.STPhase.Final`，store 后 `set_mm_layout_transform(enabled=False)`。

> **CV融合算子**：非模板优先——以 PyPTO-Pro 文档和官方指定算子中的 c-v 融合算子（PRO_MATERIAL_INDEX.md §B）为优先参考。Cube 段（matmul K累加）和 Vector 段（`vf.*` 后处理）可分别参考纯cube官方样例和纯vec模板的局部写法，但不作为权威参考——两段间的数据流与同步（set_cross_core/wait_cross_core）按 DESIGN.md §1 API 序列 + §6 同步策略施工，整体写法不能完全套用纯vec/cube模板。

将步骤 4 骨架的占位逐 Module 翻译成实际代码，对照 DESIGN.md §10 全景图确认每个 Module 的输入/输出 tile：计算 API 序列取自 §1（参数用步骤 2 已确认的结论）、tile 变量取自 §3、同步取自 §4-6、尾块处理取自 §7。写法不确定时先查 EXPLORE_REPORT.md §4 的可复用模式，不足时按 §B 定位官方指定算子原文。

### 步骤 6：编写测试函数

在同一文件中编写测试函数，结构照 [templates/impl_template.py.tmpl](templates/impl_template.py.tmpl)（含完整多 case 骨架）。输入 shape/dtype 取自 DESIGN.md。以下是必须守住的规则：

> **L0 路径**：golden 签名取自 `{op}_golden.py`，测试 case 取自 DESIGN.md §8，wrapper 名为 `{op}_wrapper`。
> **L1 路径**：golden 导入 `modules/{op}_golden_stage<suffix_k>`（由 mathematician 一次性产出），wrapper 名为 `{op}_wrapper_module<suffix_k>`，测试 case 仍取自 DESIGN.md §8 但只验证到当前 Module k 的输出（与 `{op}_golden_stage<suffix_k>` 的返回值比对）。

- **测试 case 直接取自 DESIGN.md §8「目标测试 case」表**，不自行重算 shape。每个 case 拆为独立 `def test_` 函数、命名沿用 §8；orchestrator 门禁以 `def test_` 数量 ≥ 4 为泛化性判据（覆盖整除 / 单轴尾块 / 双轴尾块 / 跨多 tile+尾块）。§8 已确认这些 case 均可适配；若某 case 因实现翻译错误跑不通，在步骤 7 修正 kernel；若证据表明设计本身不适配，返回 `design_violation`。**不得删改 case 迁就实现**。§8 缺失或不足 4 个时回调 Stage 3。
- **⚠️ 设备必须与 golden 一致**：kernel 异步执行，设备不一致会让 NPU 错误污染 golden、traceback 误指向 torch。**不要硬编码 `npu:0`**，从 `{op}_golden.py` 导入 `_get_device()`（golden 模板已通过 `TILE_FWK_DEVICE_ID` 环境变量选择设备）。
- **⚠️ atol 取值有据**：精度阈值由 `precision_compare.py` 按 dtype 自动查表（方案A混合容差标准），**禁止自定义 atol/rtol**。当前仓内可执行事实源是 `scripts/precision_compare.py` 的 `_THRESHOLDS`、`_REQUIRED_MATCHED_RATIO` 与计算逻辑；在权威标准文件未纳入仓库前，不得声称已与某个不可核验章节自动同步。
- **精度验证使用 `precision_compare.check_precision` + `{op}_golden_cpu`**（方案A混合容差标准）。模板已内置 `_assert_precision` 辅助函数，test 函数只需调用 `_assert_precision(output, *inputs, label="...")`，内部自动完成 CPU golden 计算 + 精度对比。**精度对比的参考实现必须是 `{op}_golden_cpu`（CPU FP32 更高精度），禁止用 `{op}_golden`（NPU 同 dtype）做精度对比**——`{op}_golden` 用于 Stage 2 NPU 参考实现验证、可选性能采集和提供 `_get_device()`。
  - **L1 路径例外**：staged 文件的 test 导入 `modules/{op}_golden_stage<suffix_k>`（纯 torch CPU FP32，由 mathematician 从 `{op}_golden_cpu.py` 切分产出），用 `_assert_precision` 比对 kernel 输出 vs `golden_stage<suffix_k>` 输出。`_assert_precision` 内部的 import 仍写在函数体内（import 安全）。
- **⚠️ 复制精度对比脚本（仅 dev 态）**：编写 test 文件前，将 `scripts/precision_compare.py` 复制到算子目录 `custom/<op>/`（与 `test_<op>.py` 同级）。该脚本仅用于本地 dev 自测，供 `_assert_precision` 在 dev 环境运行时 import。
- **⚠️ 交付态 import 安全（硬性要求）**：算子的**交付单元仅含 `test_{op}.py` + `{op}_golden.py` 两个文件**——`precision_compare.py`、`{op}_golden_cpu.py` 是 dev-only 自测工具，**只在 `custom/<op>/` 本地自测时使用，不进入交付单元**。交付单元被作为模块加载时会执行其全部顶层代码——若 `precision_compare`、`{op}_golden_cpu` 的 import 写在模块顶层，此时会直接 `ModuleNotFoundError`，导致交付态全部 case 0 分。因此这两个 dev-only 依赖的 import **必须写在函数体内**（仿 [impl_template.py.tmpl](templates/impl_template.py.tmpl) 的 `_assert_precision`，import 在函数内 → 模块加载不触发 → 安全），**或在顶层用 `try/except ImportError` 容错**（仿已交付的 rms_norm）。**禁止裸顶层 `from precision_compare import` / `from {op}_golden_cpu import`**。`test_{op}.py` 必须能在仅含 `test_{op}.py` + `{op}_golden.py` 两文件的环境下被作为模块加载通过。
  - **L1 路径**：staged 文件的 dev-only 依赖（`precision_compare.py`、`{op}_golden_stage<suffix_k>.py`）同样 import 在函数体内。staged 文件不是交付单元，import 安全要求相对宽松，但仍建议遵循函数内 import 惯例。
- **host 维度适配**：若 DESIGN.md §0 维度契约要求 kernel 只处理 2D 而 SPEC 需 1D/多维，在调用侧做 reshape 适配（模板见 impl_template）。
- **通过 wrapper 调 kernel**：test 函数必须通过 wrapper 调 kernel，不直接调 kernel。wrapper 默认只做「三件事」（参数校验与纯 Python 形状运算 / 用 torch.empty 分配输出 / 启动一次 kernel）；dtype 转换等张量整形仅可执行 DESIGN.md 在 Stage 3 已裁定的逐项例外（见本文「强制规则：wrapper 只做三件事」），核心计算始终集中在单一 kernel 函数内（核心原则 #2/#3）。
  - **L0 路径**：wrapper 名为 `{op}_wrapper`。
  - **L1 路径**：wrapper 名为 `{op}_wrapper_module<suffix_k>`。

### 步骤 7：本地验证与自修复闭环

必须严格按照以下指令格式运行算子脚本，不要进行额外的环境配置，默认环境可用：

> **L0 路径**：运行 `python custom/<op>/test_<op>.py`
> **L1 路径**：运行 `python custom/<op>/modules/test_<op>_module<suffix_k>.py`（当前 Module k 的 staged 文件）

```bash
# L0 路径
python custom/<op>/test_<op>.py

# L1 路径
python custom/<op>/modules/test_<op>_module<suffix_k>.py
```

- 确认输出 `PASS`， 编译报错 / NPU 错误 / 精度不通过 时，进入 debug 状态
- **环境问题例外**：若报错指向环境而非算子代码（如 `torch_npu` / `pypto_pro` 导入失败、`npu-smi` 无响应、NPU 设备不可见、CANN 未配置、运行超时疑似设备 hang 等），**不进入下方的 debug 自修复循环**——你被硬性规则禁止碰环境，无法自行修复。此时**加载 skill `pypto-pro-environment-check`** 走其环境检测流程（Step 1 官方 VF smoke 事实验证 → 必要时 Step 2 脚本诊断；怀疑设备 hang 时按照要求走该 skill 的「设备 hang 评定」三段式），据评定结论连同报错原文一并反馈给编排器，由编排器决定换卡或进一步反馈给用户。**禁止用临时自写超短超时测试（如 `set_device+randn+add+synchronize`）判定设备 hang**——编译耗时可达数分钟，hang 评定须按 skill 的三段式流程执行。

### 进入 debug 状态后的迭代流程

验证失败（编译报错 / NPU 错误 / 精度不通过）即进入 debug 状态。此时你负责完整的自修复闭环——**不得把问题抛回 orchestrator**（它只做产物验收，不参与调试），须自行定位、修复、重跑，直到 `PASS`。

> **这是一个迭代循环，不是走一遍就结束**：每碰到**一个**具体问题，都完整走一遍下面①→④；解决问题 A 后若重跑又暴露问题 B，就针对 B 重新从①开始再走一轮，直到所有问题清零、最终 `PASS`。不要试图一次性想清所有问题，也不要跳步。

> **⚠️ 遇到任何 API 报错或编译失败时，首先必须先查官方文档和样例确认正确用法，禁止直接归因为"框架不支持"**：
> - 查 `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/` 对应 API 文档，确认参数名、参数顺序、约束条件（如 layout 参数、pipe 类型、dtype 限制）
> - 查 `$PYPTO_DEVKIT_DIR/pro_ops/` 官方算子样例，搜索同类用法的 working example，确认正确的调用方式
> - 查 `$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials/` 教程，确认设计指导
> - **常见误区**：coder 遇到编译错误后倾向于归因为"框架能力不足"。**必须先查文档+样例确认，再下结论**

**① 定位方向**：按 [references/debugging-methodology.md](references/debugging-methodology.md) 排查（**先确认失败现象可信** → 症状快查表 → 分层 review → 定位技术（最小复现/消融/单原语替换/差异属性） → 升级与切换 → **修正后验证**）。该文档是调试的完整指引，此处不复述。**不要跳过第零步**：一次设备故障会污染 NPU 上下文，使后续 case 全部连坐；名字没变的 kernel 可能收到旧二进制。失败计数是证据，不是结论——先分清「一个真故障」和「N 个独立故障」，否则整轮排查会被引到错误的层级。

**② 按根因归属处理**：根因仅在当前 Stage 的代码翻译、参数抄录或实现细节时直接修复 kernel；根因涉及 DESIGN.md 的维度合同、API 序列、tile 属性/地址、循环、同步、尾块或 Vector 实现选择时，停止修改并返回 `design_violation`，附最小复现、错误原文和建议修改点。编排器必须回退 Stage 3，由 architect 修订并重新通过 stage3-check。coder 不得直接编辑 DESIGN.md。
**③ 记录修正**：当前 Stage 的实现修正在代码注释或 MEMORY.md 中记录；设计根因只在 verdict 中记录证据，不提前改写上游产物。
**④ 重跑验证**：修复后重新运行。若 `PASS` 且无告警，debug 结束；若暴露新问题，针对新问题回到 ① 再走一轮。
**⑤ 诚实失败退出（capability_gap）**：若在②中穷尽受支持的 tile-op、
`vf.*` 组合与 kernel 内循环方案后，根因确属框架能力不足，**不得以 host 端
核心计算绕过**。返回 `capability_gap` verdict、编译/精度证据、已尝试方案及
失败原因，由 orchestrator 回退 Stage 3 或上报用户。

### 步骤 8：交付前自检

步骤 7 跑通 `PASS` 后、交付给 orchestrator 前，对照 orchestrator 验收清单逐项自检——把能在本地完成的检查全部做完，避免交付后被静态门禁打回返工。任一项不通过须当场修正并重跑，不得带病交付。

| 检查项 | 自检方式 |
|--------|---------|
| `test_{op}.py` 为最终单文件 | 确认文件存在；L1 路径的 staged 文件链（`modules/test_{op}_module*.py`）保留不动，cleanup 由编排器调度（从最后一个 staged 文件生成 `test_{op}.py`） |
| 设备与 golden 一致 | 确认已导入 `{op}_golden._get_device()`，未在各 test 中硬编码分散的 `npu:` 设备号 |
| atol 有据 | 确认 test 使用 `_assert_precision`（非 `torch.testing.assert_close`），无自定义 atol/rtol；golden 从 `{op}_golden_cpu` 导入（CPU 更高精度），非 `{op}_golden`（NPU 同 dtype） |
| 核心计算在单 kernel 内（含语义反作弊） | **`grep -c "@pl.jit" <file>` 必须返回 1**——确认文件中仅一个 kernel 函数（`@pl.jit` 装饰）。L1 路径同样如此：逐 Module 是在同一个 kernel 内增量追加，不是每 Module 新建 kernel。多个 `@pl.jit` → 严重违规，必须修为单 kernel。**所有核心计算逻辑集中在单一 kernel 函数内**，host 端不做核心计算；`.contiguous()`/`.to()`/非连续上的 `.permute()` 会派发真实 kernel，默认必须搬进 kernel，仅 DESIGN.md 在 Stage 3 已裁定的 wrapper 例外可保留；**禁止在循环中调用 kernel**（host 端循环 launch kernel 分担计算视为作弊）。**语义自检**：host 端无值依赖变换（输出依赖于输入数值大小关系的操作，如排序/选择/去重/索引重排）；kernel 输出是最终结果而非中间/候选（host 端不从中筛选/提取）；无 `assert`/`if` 砍算子定义声明的维度/dtype/参数支持范围到单点；**判定标准是计算实质不是命名**——"post-processing"/"extraction" 等措辞不改变 host 端做核心计算的事实 |
| ≥4 个独立 test 且与 §8 一致 | 确认 `def test_` 数量 ≥ 4，命名与覆盖对齐 DESIGN.md §8「目标测试 case」，非临时另造 |
| Vector 实现合规 | 与 DESIGN.md 已冻结的 `vector_selection` 一致，不自行切换 VF/tile-op 层级 |
| **入口函数命名合规** | **L0 路径**：确认文件中暴露了名为 `{op_name}_wrapper` 的可调用入口函数（签名与算子 schema 一致）。**L1 路径**：确认文件中暴露了名为 `{op_name}_wrapper_module<suffix_k>` 的可调用入口函数（签名与 `primary_inputs` 一致，输出 Module k 的结果）。外部调用方按命名约定查找 `{op_name}_wrapper` 和 `{op_name}`，推荐 `_wrapper` 后缀以与 kernel `_kernel` 配对。wrapper **只调用一次 kernel**；`.contiguous()`/`.to()`/非连续上的 `.permute()`/`.transpose()` 默认全部搬进 kernel，只有 DESIGN.md 在 Stage 3 已裁定的逐项例外可保留。**纯视图除外**——张量本就连续时的 `.reshape`/`.view` 只改元数据、不产生 `aclnn*` 条目，是允许的（判据见 `pypto-pro-op-kb/constraints/wrapper-boundary.md`）。test 通过 wrapper 调 kernel 而非直接调 kernel。**optional 参数必须带默认值**：若 `cases.yaml` 中存在省略某个输入参数的 case（该参数的 `input_shape` 位置为 `null` 或列表更短），则 wrapper 签名中该参数必须设 `=None` 默认值，否则外部调用方省略该参数时会触发 `TypeError` |
| **实现偏差已声明** | 若实现与 DESIGN.md 任何关键常量、算法步骤、tile 布局有偏离，确认已在回复中显式列出偏离点 + 原因 + 是否需回退 Stage 3。**静默偏离视为违规** |
| **测试输入与算子定义一致** | 确认 test 函数的输入 shape/dtype/value_range 取自 DESIGN.md §8「目标测试 case」，未偷换数据分布以规避算法弱点。若 SPEC/DESIGN 指定了 value_range 或数据分布，test 须沿用，不得自行替换 |

> L1 路径每次 dispatch 产出一个 staged 文件（完整独立算子），自检针对该 staged 文件。最终 `test_{op}.py` 的自检在 cleanup 后由编排器调度 stage4-check 覆盖。

---

## 核心原则

1. **DESIGN.md 是 Stage 4 的施工合同**：严格按已通过 Stage 3 门禁的设计实现；运行证据推翻设计时返回 `design_violation`，由编排器回退 Stage 3 修订和复核，coder 不直接改写 DESIGN.md
2. **只写一个 .py 文件、只含一个 `@pl.jit` kernel**：kernel + 测试在同一文件中，**所有的核心计算逻辑集中在单一 kernel 函数内**（本工作流生成的 Pro 算子只需一个 `@pl.jit`，不需拆分为多个 kernel）。L0 路径产出 `test_{op}.py`；L1 路径每次 dispatch 产出一个 staged 文件 `modules/test_{op}_module<suffix_k>.py`（**同样单 kernel——逐 Module 是在同一个 kernel 内增量追加，不是每 Module 新建 kernel**），最终由编排器 cleanup 合并为 `test_{op}.py`
3. **入口函数**：L0 路径命名为 `{op_name}_wrapper`；L1 路径命名为 `{op_name}_wrapper_module<suffix_k>`。参数和返回值与算子定义一致，默认只做本文「强制规则：wrapper 只做三件事」允许的三件事；只有 DESIGN.md 在 Stage 3 已裁定的 wrapper 例外可额外保留，Stage 4 不得新增。test 必须通过 wrapper 调 kernel，**wrapper 只能调用一次 kernel**，**禁止在循环中调用 kernel**（host 端循环多次 launch kernel 分担本应在单次 kernel 内完成的计算视为作弊）。
4. **实现约束不可违背**：buffer 轮转用 `make_tile_group` + `auto_mutex`；Vector 执行 DESIGN.md 的 `vector_selection`
5. **官方指定算子是写法参考来源**：不确定时先查 EXPLORE_REPORT.md §4 可复用模式，不足时按 §B 定位官方指定算子原文
6. **不确定时不猜**：先查 EXPLORE_REPORT.md §3 的 API 约束，不足时按 §A 定位 API 文档原文确认
7. **测完整除 + 尾块两种 case**：泛化性验证
8. **不随意更改环境配置**：环境已在进入 Stage 4 前验证可用，出问题先查算子代码；确属环境问题则停下反馈（详见步骤 7 环境问题例外）

## 知识使用契约（coder）：产出 `KB_USAGE.json`

实现前**必须**读取 `KB_SELECTION.json`，并读完 `optional_patterns` 与
`required_constraints` 中的每一条参考。
实现后**必须**在同目录写出 `KB_USAGE.json`。它记录 coder 的实现声明，随后由
本仓 verifier（contract v2）独立校验；缺失即该 class 不通过；
校验项见 [`pypto-pro-op-kb/CONTRACT.md`](../../pypto-pro-op-kb/CONTRACT.md)。

它记录一条链路：

    选中的参考 → 推导出的不变量 → 实现位置 → 实现声明

```json
{
  "schema_version": 2,
  "op": "<算子名>",
  "class_id": "<class 子目录名>",
  "invariants": [
    {
      "reference": "constraints/precision.md",
      "invariant": "fp16 输入在平方之前用 vf.astype 升到 fp32，整条归约链保持 fp32，避免中间值溢出",
      "implementation": {
        "file": "test_<算子名>.py",
        "symbol": "<vf 子函数名>",
        "status": "implemented",
        "evidence": "vf.mul 之前的 vf.astype(dtype=pl.DT_FP32)"
      }
    }
  ]
}
```

`implementation.status` 取值：`implemented` / `deviated` / `not_applicable`。
coder **不得写 `verified`**；`verified` 是 verifier 检查真实代码后给 orchestrator 的
PASS verdict，不是实现者的自我声明。

硬性约束：

- `optional_patterns` 与 `required_constraints` 中的**每一条**参考都必须至少出现一次。
  一条没带来任何不变量的可选模式本就不该被选中；必需约束不适用时须写
  `not_applicable` 和 justification，交 verifier 裁定。
- **反过来同样成立：`reference` 只能引用 `KB_SELECTION.json` 里已经选中的路径。**
  用到的知识必须是选过的知识——校验器会逐条比对两个文件，引用一条没在选择里的路径
  直接判该 class 不通过。若写代码时发现真正需要的是另一条参考，回到 Stage 1 把它加进
  `KB_SELECTION.json`（含 `sha256` 与 `reason`）后再写进来，不要只在 usage 里补。
  强制类参考（如 [`constraints/wrapper-boundary.md`](../../pypto-pro-op-kb/constraints/wrapper-boundary.md)）
  必须以 KB 相对路径出现在 `required_constraints` 中。
- `reference` 的路径形式与 `KB_SELECTION.json` 完全一致：**相对 KB 根**，
  形如 `constraints/wrapper-boundary.md`。不要写成你打开文件时用的那个路径
  （`.opencode/skills/<skill>/pypto-pro-op-kb/constraints/wrapper-boundary.md` 会被判为不存在），
  也不要凭印象拼路径。
- `implementation.file` 必须真实存在，`symbol` 必须真实出现在该文件中。
  指向不存在的符号会被判定为伪造。
- `op`、`class_id` 必填，`invariants` 必须是**列表**；文件必须是合法 JSON。
  写完自己 `json.load` 读一遍，再跑一遍
  [`pypto-pro-op-plan`](../pypto-pro-op-plan/SKILL.md) 收尾自检里那段
  `KB_SELECTION.json` 检查（按 `$CANNBOT_CONFIG_ROOT` 解析 kb 根，不要抄绝对路径）——
  这一步失败，整个 class（含已经写好的 kernel）会被丢弃且无法被分发到。

  > **不要去找 `auto_pipeline.kb_contract`。** 这里曾指向那个模块，它**在任何可见
  > checkout 里都不存在**（见 [`pypto-pro-op-kb/CONTRACT.md`](../../pypto-pro-op-kb/CONTRACT.md) 开头），照做只会
  > 拿到 `ModuleNotFoundError`。契约由 `pypto-pro-op-plan` 的收尾自检与
  > `pypto-pro-op-verifier` 强制，没有第三个入口。**报告门禁结果时贴你实际跑的命令，
  > 不要写"contract self-check PASS"这类无法复核的措辞**——本会话已有一次这样的报告，
  > verifier 复现不出来，只能自己重写一个检查。

  这些本地检查只是格式预检，不代替 orchestrator 随后调度的 `stage4-check`
  verifier；verifier 未 PASS 时不得完成 Stage 4。
- `deviated` 与 `not_applicable` 必须写 `justification`，说明为什么本 class
  可以偏离。偏离本身是允许的，不写理由不允许。

## 强制规则：wrapper 只做三件事

公开 callable 是实现边界的一部分，host 侧张量操作会派发额外 device kernel。
必读 [`pypto-pro-op-kb/constraints/wrapper-boundary.md`](../../pypto-pro-op-kb/constraints/wrapper-boundary.md)；它属于
`required_constraints`，不占可选 pattern 名额。

wrapper 默认只允许做这三件事：

1. 参数校验与形状推导（纯 Python 标量运算；连续张量上的 `.reshape`/`.view` 纯视图不派发 kernel，允许）；
2. 用 `torch.empty` 分配输出；
3. 启动一次 `@pl.jit` kernel。

唯一扩展是 DESIGN.md 的 wrapper 操作清单在 Stage 3 已逐项记录并通过门禁的例外；
coder 只能原样实现并在 `KB_USAGE.json` 记录 `deviated`，不得在 Stage 4 新增或扩大范围。

### 默认禁止出现在 wrapper 里的调用

`.to()` / `.contiguous()` / `.permute()` / `.movedim()` / `.transpose()` /
`.repeat_interleave()` / `.expand()` / `.broadcast_to()` / `torch.cat` /
`torch.stack` / `torch.nn.functional.pad` / `torch.zeros` / `torch.zeros_like` /
`torch.arange` / `.narrow()` / `.chunk()` / `.split()` /
**`torch.npu.synchronize()`**

`torch.npu.synchronize()` 不产生 device 算子，但它会在 wrapper 内引入整条流等待，
而同步应由调用方负责。完成 wrapper 后必须检查并移除非必要的显式同步。

对应的做法见 wrapper-boundary.md 的迁移表：dtype 转换用 tile load/store 时的
`pl.cast`（tile 级）或 `vf.astype`（寄存器级）；轴变换用 stride/offset 索引；
尾块用 `pl.set_validshape`；
索引构造在 kernel 里算。

`.reshape()` / `.view()` 在张量本来就连续时是纯视图，不产生 device 算子，可以
保留；但只要为了让它成立而先调用了 `.contiguous()`，就已经付费了。

### TensorList 输入：kernel launch 必须在循环外

输入声明为 `is_list: true` 时，**wrapper 里不允许出现「遍历 list、每个元素启动一次
kernel」的写法**：

```python
# 禁止：N 个张量 → N 次 launch
for a_i, b_i in zip(a, b):
    out_i = torch.empty_like(a_i)
    my_kernel[None, cores](a_i, b_i, out_i)
    out.append(out_i)
```

这类算子的基线（`torch._foreach_*` 一族）本身就是**一次融合调用**——它们存在的全部
理由就是消除 per-tensor 的 launch 开销。逐个 launch 等于把基线特意省掉的开销又加了
回来，list 越长差距越大，且**只有 list 长度为 1 时才可能追平基线**。

正确形态是**一次 launch 覆盖整个 list**：把各元素的地址与 shape 打包成 tiling 参数
（或按总元素数展平成一维 work item），核内再遍历。分核按「所有元素的 tile 总数」而不是
按「list 长度」来算，否则元素少时核用不满、元素多时又要多轮 launch。

判据（可自查）：wrapper 里 `_kernel[` 出现的位置必须在任何 `for` 之外；输出一次性
分配好，不在循环里逐个 `empty_like`。

跑一次 profile，看 `op_times.device_kernels`：

- 名字以 `_Z` 开头的是你的 kernel；
- 名字以 `aclnn` 开头的**全部**是 wrapper 开销。

`aclnn*` 的总和应当为 0。不为 0 时，先逐项核对 DESIGN.md 的 wrapper 操作清单：
只有 Stage 3 已裁定的例外才可保留，并须在 `KB_USAGE.json` 对应不变量中记录
`implementation.status: deviated` + `justification`；实现自行新增的操作须移入 kernel，
若新证据表明确实无法迁移则返回 `design_violation`，由编排器回退 Stage 3。不得在
Stage 4 通过补写 DESIGN.md 或事后说明使其合规。

典型反例是 wrapper 做成 cast→transpose→kernel→transpose→cast：一次 kernel 启动被四个计时算子包住，wrapper 可以占掉过半的 device 时间。
完整的迁移对照表（host 做什么 → kernel 里怎么写）见 [`pypto-pro-op-kb/constraints/wrapper-boundary.md`](../../pypto-pro-op-kb/constraints/wrapper-boundary.md)；确有无法迁移的变换时，须由 Stage 3 在 DESIGN.md 预先记录原因、目标版本证据、适用条件、预期代价预算和测量方法；Stage 4 只负责实测核验，并在 `KB_USAGE.json` 以 `implementation.status: deviated` + justification 说明。
