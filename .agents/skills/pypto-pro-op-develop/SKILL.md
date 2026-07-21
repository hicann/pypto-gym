---
name: pypto-pro-op-develop
description: PyPTO-Pro 算子 kernel 实现编码手册。所有算子（纯vec/纯cube/CV融合）均完整走步骤0~8及debug流程，不存在跳过步骤直接套模板——步骤0~4统一走，步骤5~6纯vec/cube模板优先（填DESIGN.md参数到已验证模板），CV融合非模板优先（以Pro文档和官方c-v融合算子样例为优先参考，纯vec/cube模板仅作局部写法参考）。步骤7~8+debug统一走。以运行验证为最终准则。触发词：实现算子、写 kernel、编写实现、写 impl、kernel 实现、pypto-pro develop。
---

# PyPTO-Pro 算子 Kernel 实现

生成完整的 PyPTO-Pro kernel 实现文件（一个 `.py` 文件，含 kernel 函数 + 测试函数），并本地跑通验证。运行验证为最终准则——当运行结果与 DESIGN.md 冲突时，以实际 API 文档、教学文档、官方指定算子为准修正 DESIGN.md 的失误。

> **角色说明**：PyPTO-Pro 的 Stage 4 由单个 `general` 子代理全包完成，暂时没有拆分为 coder / verifier / debugger 等多个专用代理。因此本 skill 的承担者需独自完成 **开发 → 验证 → 发现问题 → 分析根因 → 解决问题 → 再验证** 的完整闭环，直到能够交付符合全部要求的算子代码。

> **方法论优先**：本 skill 以**思维方法指导**为主，不教具体写法——具体 API 用法、vf 指令组合、tile 配置、同步写法等请查阅 API 文档（`$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/`）、教学文档（`$PYPTO_DEVKIT_DIR/docs/pypto_pro/guide`）、官方指定算子（`PRO_MATERIAL_INDEX.md` §B），理解后据实实现。

---

## 两条性能强制（实现阶段须遵守）

> 完整定义见 `pypto-pro-material-explore` SKILL「两条性能强制」节。实现阶段须遵守：
> 1. 所有需要 buffer 切换/轮转的 tile（含 double buffer）一律用 `make_tile_group` + `auto_mutex`，由框架自动管理 buffer 切换与互斥。`make_tile` 仅限**单次使用 scratch tile**（写入一次、读取一次、不参与 buffer 切换/轮转循环，如一次性中间结果暂存、不迭代的归约标量结果）。**禁止用 `make_tile` + 手动 `sync_src`/`sync_dst` 管理 buffer 轮转**。
>
>    同步方案按 DESIGN.md §6 施工——auto_mutex 管核内 pipe 互斥（禁止在其管理的 tile 上叠加 `sync_src`/`sync_dst`，否则死锁）；跨核同步用 `set_cross_core`/`wait_cross_core`（手动）。具体同步点与 event_id 分配已在 §6 确定。
>
> 2. Vector 数值计算用 `vf.*` 指令手写，在 `section_vector()` 内通过 `@pl.vector_function` 装饰器或 `@pl.inline` + `with pl.section_vf():` 块执行（两种写法均可以 vf API 文档与官方指定算子为准）。`pl.*` 计算API不得用于 Vector 数值计算——无论在何处编写。

- **vf 写法学习路径**：见 `pypto-pro-material-explore` SKILL「两条性能强制」节。

---

## 输入

### 主要依据（必读）

| 来源 | 内容 | 用途 |
|------|------|------|
| `custom/<op>/DESIGN.md` | Phase 划分（§0）、API 映射（§1）、Tile 规划（§2）、UB 空间布局（§3）、循环与 Section（§4）、分核/流水/尾块（§5-7）、目标测试 case（§8）、全景图（§10） | **初步设计**——kernel 实现的初始主要依据；运行验证发现问题时可据实修正 |
| `custom/<op>/EXPLORE_REPORT.md` | API 约束（§3）、相似样例与可复用模式（§4）、教程指导（§5）、Tile/同步策略建议（§6） | **编码参考**——API 约束速查、样例写法定位、教程设计指导 |

### 按需取用（EXPLORE_REPORT.md 不足时查阅）

| 来源 | 内容 | 用途 |
|------|------|------|
| `custom/<op>/PRO_MATERIAL_INDEX.md` | API 文档（§A）、官方指定算子（§B）、教程（§C）的精确路径索引 | **路径定位**——先查索引找到文档/样例路径，再读取原文 |

## 参考文件

| 文件 | 用途 | 加载时机 |
|------|------|----------|
| [templates/pure_vec_impl_template.py](templates/pure_vec_impl_template.py) | **纯vec算子首选实现起点**——Pattern A/B/B+/C 四种子模式覆盖整个纯vec设计空间，内置固定骨架（auto_mutex + double-buffer + stride分核 + 尾块处理）+ VF API 快速参考 + 两条性能强制声明。填 CONFIG 常量 + VF chain + 测试 case 即可生成完整 kernel | 纯vec算子生成代码前必读 |
| [$PYPTO_DEVKIT_DIR/pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset_dynamic.py]($PYPTO_DEVKIT_DIR/pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset_dynamic.py) | **纯cube算子首选实现起点**——两级K分块 + move offset + 嵌套四分支K累加 + 尾块处理 + ASW蛇形调度 | 纯cube算子生成代码前必读 |
| [templates/impl_template.py](templates/impl_template.py) | kernel 文件骨架（tile 声明 + section + Phase + 测试函数）——**通用**（含 CV 融合 / 多 Phase / 跨核流水）；CV 融合算子当前无专用模板，走此通用骨架 + 步骤 0~8 | 生成代码前必读 |
| [references/debugging-methodology.md](references/debugging-methodology.md) | 调试方法论（症状快查表 + 分层 review 顺序） | 验证失败进入 debug 状态时必读 |
| [scripts/list_idle_chip_ids.sh](scripts/list_idle_chip_ids.sh) | 查找空闲 NPU chip | 运行前按需执行 |


---

## 开发流程

### 实现策略与步骤适用范围

所有算子（纯vec / 纯cube / CV融合）都必须完整走步骤 0~8 及 debug 流程——**不存在"跳过步骤直接套模板"**。区别仅在于**步骤 5~6 的参考来源**：

- **步骤 0~4（所有算子统一）**：判断实现模式 → 确认输入齐全 → 逐 API 确认参数 → tile 声明 → section 骨架。模板不能替代这些步骤——模板只提供代码骨架，骨架填充所需的参数和结构来自步骤 0~4 对 DESIGN.md 的确认。
- **步骤 5~6（按算子类型分叉）**：
  - **纯vec / 纯cube**：**模板优先**——步骤 5 按模板内置骨架填充 VF chain / matmul K累加（填 CONFIG 常量 + `>>> FILL` 标记），步骤 6 按模板测试结构。模板已内置同步、尾块处理，填充时遵守模板内嵌的两条性能强制。
  - **CV融合**：**非模板优先**——以 PyPTO-Pro 文档和官方指定算子中的 c-v 融合算子（PRO_MATERIAL_INDEX.md §B）为优先参考。Cube 段（matmul K累加）和 Vector 段（`vf.*` 后处理）可分别参考纯cube官方样例和纯vec模板的**局部写法**，但**不作为权威参考**——CV融合涉及跨核搬运与同步（set_cross_core/wait_cross_core），整体数据流和衔接写法与纯vec/cube模板不能完全一致。
- **步骤 7~8 + debug（所有算子统一）**：本地验证 → 自修复闭环 → 交付前自检，照常执行。

**兜底路径**：步骤 5~6 套用模板 / 参考样例后运行失败时，按步骤 7 的 debug 自修复闭环修复。

### 步骤 0：判断实现模式（强制前置检查点）

> ⚠️ **此步骤为强制前置检查点**——必须在开始任何编码工作之前完成，并在回复中**明确输出**你的模式选择结果（"直接模式"或"增量模式"）及判据（DESIGN.md §0 的 Phase 数 / 是否含 cross_core）。**未输出模式选择就直接开始写代码视为违规**。orchestrator 在 Stage 4 dispatch prompt 中会显式要求此检查点。

正式开始前，先按 DESIGN.md §0 判断走哪种实现模式：

- **判据**：Phase 数 **≥ 3**，或算子涉及 **cross_core 跨核流水** → 走**增量模式**；否则（逐元素、单 / 双 Phase 的简单归约等）走**直接模式**。判据直接从 DESIGN.md §0 的 Phase 列表长度与 Section 划分读出，不必自己揣测。
- **增量模式禁止一次性写完所有 Phase**：必须按 Phase 顺序逐轮推进，每轮只实现到当前 Phase 并验证通过后才能进入下一轮。一次性写完所有 Phase 再首跑会导致失败面极大、debug 在多个 Phase 问题间跳跃、context window 耗尽——这是已验证的高风险反模式。

**直接模式**：按下面步骤 1→7 一次走完（步骤 3-7 各处理全部 Phase）。简单算子首跑失败面本就不大，无需增量。

**增量模式**：步骤 1-2 照常（全局确认输入与 API 参数）；**步骤 3-7 改为按 DESIGN.md §0 的 Phase 顺序分轮重复**，每轮只处理到当前 Phase：

1. **逐 Phase 推进**：第 k 轮只实现到 Phase k——只声明该 Phase 用到的 tile（步骤 3）、只搭到该 Phase 的 section / 循环（步骤 4）、只填该 Phase 的实现（步骤 5），把该 Phase 的**中间结果临时 store 出来**，与 golden 的**对应中间量**比对，并按 DESIGN.md §8 的**完整目标 case**（对齐 / 尾块 / 多 tile）验证跑通（步骤 6-7，验证失败则在本轮内走步骤 7 的 debug 迭代流程直到 PASS）。该轮全部 case PASS 后再进入 Phase k+1，**在前序基础上追加**新 tile / section / 实现与 Phase 间衔接（cross_core / 中间量传递），前序已跑通的代码不推翻。
2. **golden 中间量**：为比对各 Phase 中间结果，临时给 golden 增加暴露中间量的辅助函数（如 `{op}_golden_stage1`）。golden 是纯 torch，加中间返回不违反其约束。

控制流（外层按 Phase 分轮，内层每轮验证失败走步骤 7 的 debug 迭代）：

```
轮1（Phase1）    : 步骤3-5 实现 → 步骤6 写测试 → 步骤7 跑 §8 完整 case
                                                  ├─ PASS → 进轮2
                                                  └─ 失败 → debug 迭代(①→④循环) → PASS → 进轮2
轮2（Phase1+2）  : 步骤3-5 追加 → 步骤6 → 步骤7 跑 §8 完整 case
                                                  ├─ PASS → 进轮3
                                                  └─ 失败 → debug 迭代 → PASS → 进轮3
轮k（+Phase k）  : ... 同上，直到最后一个 Phase
最后一轮 PASS 后 : 清理临时产物（中间量 store / golden 辅助函数）→ 合并为最终单文件 → 重跑确认 PASS
```

> **为什么增量有效**：分轮后，进入 Phase k+1 时前 k 个 Phase 已确认正确，失败面收缩到"新增 Phase + 衔接"，与 [references/debugging-methodology.md](references/debugging-methodology.md) 的"最小复现"思路一致。增量以**追加**为主，不推翻重写。

> ⚠️ **收尾清理（增量模式必做）**：全部目标 case PASS 后，清理所有临时产物（中间量 store / golden 辅助函数），合并为最终单文件 `test_{op}.py`（含 §8 完整 case），重跑确认仍 `PASS`。

### 步骤 1：确认输入齐全

读取 DESIGN.md，确认施工所需信息完整——各信息对应位置：

- **§0** Phase 划分与数据依赖、维度契约
- **§1** 每个 Phase 的 API 映射序列
- **§2/§3** Tile 规划（shape/dtype/layout）与片上地址映射
- **§4** 循环与 Section 结构
- **§5-7** 分核 / 流水 / 尾块策略
- **§8** 目标测试 case（直接按此实现测试，不自行重算 shape）
- **§10** Tile 数据流全景图（确认全局理解）
- **EXPLORE_REPORT.md §3-5**：API 约束 / 相似样例 / 教程，步骤 2/5 按需查阅

信息不足时优先查 EXPLORE_REPORT.md / API 文档 / 官方指定算子补全，仍缺失（尤其 §0/§1/§2/§3/§8 关键项）时回调 Stage 3，不做猜测。运行验证发现的 DESIGN.md 失误则据实修正。

### 步骤 2：逐 API 确认参数

对 DESIGN.md §1 中的每一个 API 调用，**必须**确认三个信息：

1. **参数名是否为关键字参数**：大部分是位置参数，少数（如 `set_pipe=`、`wait_pipe=`、`event_id=`）是关键字参数。参数传参方式（位置 vs 关键字）以 API 文档原文为准，EXPLORE_REPORT §3.3 已汇总部分
2. **参数顺序**：如 `pl.matmul(dst, a, b, phase=...)`（dst 先于操作数），`pl.load_tile(tile, tensor, [i, j])`（tile 先于 tensor）。具体顺序以 API 文档为准
3. **约束条件**：dtype 限制、layout 要求、MemorySpace 约束——以 API 文档原文为准，EXPLORE_REPORT §3.3 已汇总部分

> **性能强制影响本轮**：Vector 数值计算步骤须确认 `vf.*` 指令的签名（非高层 `pl.*` vec API）。Cube 步骤确认 `pl.*` Cube API 签名。

**确认方式**（先汇总后原文，减少阅读量）：
1. **先查 EXPLORE_REPORT.md §3**（API 映射结果 + API 约束 + MemorySpace 约束已汇总）——大部分信息在此可直接获得，无需读原文
2. **§3 不足时**，再通过 `PRO_MATERIAL_INDEX.md` §A 定位对应 API 文档路径，读取文档原文的"函数原型"和"参数范围"两个表补充

**产出要求**：步骤 2 完成后，应形成一份覆盖 DESIGN.md §1 全部 API 的参数速查清单（vec 步骤为 `vf.*`、cube 步骤为 `pl.*`），后续步骤直接复用，不再重复确认。

> **模式提示**：以下步骤 3-7 以**直接模式**为默认写法。增量模式按步骤 0 的 Phase 分轮收窄处理范围。

### 步骤 3：编写 tile 声明

> **模板选择**：按上方「实现策略与步骤适用范围」节判定算子类型并选择对应起点。纯vec 填写 CONFIG 常量（N / ROWS / TILE_ROWS / NUM_CORES 等，来自 DESIGN.md §2），按 §3 确认地址无重叠；纯cube 的 tile 声明（Mat 4-buffer / Left 2-buffer / Right 2-buffer / Acc 单 buffer）照抄官方样例 `make_tile_group` 配置，按 §2/§3 调整 shape/dtype/addr；CV 融合走通用模板 `impl_template.py`。

骨架与格式参考 [templates/impl_template.py](templates/impl_template.py)（tile 声明 / section / 循环 / 测试的完整结构）。

1. **编译期常量（TS、TD、SCALE 等）复制到模块级**——从 DESIGN.md §2.1 取值；⚠️ 必须声明在 kernel 函数**外部**（模块级）。写进 kernel 函数体内会被 JIT 当作 IR 语句处理，触发编译错误。
2. **动态维度声明**——从 DESIGN.md §0 的维度契约取动态轴名称，按 `docs/` API 文档和官方指定算子样例中的正确声明方式编写（声明方式以文档/样例为准，不臆测）。循环内通过 `tensor.shape[i]` 获取动态维度值
3. **逐条写 tile 声明**——按 DESIGN.md §2 Tile 属性表 + §3 地址映射表，精确复制每个 tile 的 shape / dtype / layout / addr / size；需要 buffer 切换/轮转的 tile 用 `make_tile_group` + `auto_mutex`，单次使用 scratch tile 用 `make_tile`（性能强制）；地址无重叠、归约输出等 layout 约束均已在 DESIGN §2/§3 定好，此处照抄即可。运行验证发现地址/属性有误时据实修正。

### 步骤 4：编写 section 和循环框架

> **纯vec算子**：`pure_vec_impl_template.py` 已内置 `section_vector()` + stride 循环 + `auto_mutex` 同步骨架，无需额外编写，直接进入步骤 5 填充 VF chain。Pattern A/B/B+ 为单轮循环 + 单 VF 函数；**Pattern C 为 3 轮循环（sum→variance→normalize）+ 指针算术 + UB tile 累加器，模板已内置 3 轮骨架和 init/acc 分离 VF 函数对，按 `>>> FILL` 标注填写即可**。

> **纯cube算子**：仿照 `test_matmul_perf_asw_4k_dn_move_offset_dynamic.py` 搭建 `section_cube()` + K 分块循环（外层 KL1 wide load、内层 KL0 sub-tile move）+ `set_mm_layout_transform(enabled=True)` 骨架。tile 声明（Mat 4-buffer / Left 2-buffer / Right 2-buffer / Acc 单 buffer）照抄官方样例的 `make_tile_group` 配置，按 DESIGN.md §2/§3 调整 shape / dtype / addr。ASW 蛇形调度为可选性能优化，简单场景用 2D tile 迭代（`idx // N_TILES` + `idx % N_TILES`）即可。

> **CV融合算子**：无专用模板，按 DESIGN.md §4 搭建 `section_cube()` + `section_vector()` 两段式 section 骨架，Cube/Vector 间的数据传递与跨核同步（set_cross_core/wait_cross_core）按 DESIGN.md §6 施工；具体写法参考官方指定算子中的 c-v 融合算子（通过 PRO_MATERIAL_INDEX.md §B 定位）。

按 DESIGN.md §4（循环与 Section 结构）搭出 kernel 骨架：section 声明、SPMD 原语获取位置（多 section 在 section 外闭包共享，单 section 在内）、循环嵌套、同步点占位注释。此时**不填入具体 API 调用和尾块代码**（步骤 5 完成）。骨架结构见 [templates/impl_template.py](templates/impl_template.py)。

### 步骤 5：编写 Phase 内部实现

> **纯vec算子**：填充 `@pl.vector_function` 内的 VF chain（`>>> FILL` 标记处），使用 `vf.*` API（`vf.load_align` / `vf.store_align` / `vf.add` / `vf.reduce_max` 等）。VF chain 来源与下方一致：DESIGN.md §1 API 序列 + §10 全景图 + EXPLORE_REPORT.md §4 可复用模式。纯vec模板已内置尾块处理（`set_validshape`）和同步（`auto_mutex`），无需手动编写。**Pattern C 还需填写 kernel body 中的 `>>> FILL` 标注（指针算术 / gamma-beta offset / 循环轮次参数），见模板 Pattern C 段落。**

> **纯cube算子**：按官方样例 `test_matmul_perf_asw_4k_dn_move_offset_dynamic.py` 的嵌套四分支 K 累加编写——首块用 `pl.matmul`（`gsub==0`），其余块用 `pl.matmul_acc(acc, acc, ...)`；末块传 `phase=pl.AccPhase.Final`，中间块传 `Partial`，`gsub==0 && gsub==last_sub`（K_BLOCKS==1）传 `Final`。move 前扩展 valid_shape 到 full tile（fixpipe 必须看 whole tile），store 前 `set_validshape(acc, [valid_m, valid_n])` 缩小到有效窗口并传 `phase=pl.STPhase.Final`，store 后 `set_mm_layout_transform(enabled=False)`。

> **CV融合算子**：非模板优先——以 PyPTO-Pro 文档和官方指定算子中的 c-v 融合算子（PRO_MATERIAL_INDEX.md §B）为优先参考。Cube 段（matmul K累加）和 Vector 段（`vf.*` 后处理）可分别参考纯cube官方样例和纯vec模板的局部写法，但不作为权威参考——两段间的数据流与同步（set_cross_core/wait_cross_core）按 DESIGN.md §1 API 序列 + §6 同步策略施工，整体写法不能完全套用纯vec/cube模板。

将步骤 4 骨架的占位逐 Phase 翻译成实际代码，对照 DESIGN.md §10 全景图确认每个 Phase 的输入/输出 tile：计算 API 序列取自 §1（参数用步骤 2 已确认的结论）、tile 变量取自 §3、同步取自 §4-6、尾块处理取自 §7。写法不确定时先查 EXPLORE_REPORT.md §4 的可复用模式，不足时按 §B 定位官方指定算子原文。

> **性能强制（影响本轮）**：Vector 数值计算用 `vf.*` 手写、Cube 用 `pl.*`（详见上方「两条性能强制」节）。

### 步骤 6：编写测试函数

在同一文件中编写测试函数，结构照 [templates/impl_template.py](templates/impl_template.py)（含完整多 case 骨架）。输入 shape/dtype 取自 DESIGN.md，golden 签名取自 `{op}_golden.py`。以下是必须守住的规则：

- **测试 case 直接取自 DESIGN.md §8「目标测试 case」表**，不自行重算 shape。每个 case 拆为独立 `def test_` 函数、命名沿用 §8；orchestrator 门禁以 `def test_` 数量 ≥ 4 为泛化性判据（覆盖整除 / 单轴尾块 / 双轴尾块 / 跨多 tile+尾块）。§8 已确认这些 case 均可适配；若某 case 跑不通属 design 失误，按步骤 7 据实修正 kernel，**不得删改 case 迁就实现**。§8 缺失或不足 4 个时回调 Stage 3。
- **⚠️ 设备必须与 golden 一致**：kernel 异步执行，设备不一致会让 NPU 错误污染 golden、traceback 误指向 torch。**不要硬编码 `npu:0`**，从 `{op}_golden.py` 导入 `_get_device()`（golden 模板已通过 `TILE_FWK_DEVICE_ID` 环境变量选择设备）。
- **⚠️ atol 取值有据**：精度阈值由 `precision_compare.py` 按 dtype 自动查表（方案A混合容差标准），**禁止自定义 atol/rtol**。阈值表见 `.agents/skills/pypto-pro-op-develop/scripts/precision_compare.py`（与《生态算子精度标准》§2.2 一致）。
- **精度验证使用 `precision_compare.check_precision`**（方案A混合容差标准）。模板已内置 `_assert_precision` 辅助函数，test 函数只需调用 `_assert_precision(output, *inputs, label="...")`，内部自动完成 CPU golden 计算 + 精度对比。
- **⚠️ 复制精度对比脚本（仅 dev 态）**：编写 test 文件前，将 `.agents/skills/pypto-pro-op-develop/scripts/precision_compare.py` 复制到算子目录 `custom/<op>/`（与 `test_<op>.py` 同级）。该脚本仅用于本地 dev 自测，供 `_assert_precision` 在 dev 环境运行时 import。
- **⚠️ 交付态 import 安全（硬性要求）**：算子的**交付单元仅含 `test_{op}.py` + `{op}_golden.py` 两个文件**——`precision_compare.py`、`{op}_golden_cpu.py` 是 dev-only 自测工具，**只在 `custom/<op>/` 本地自测时使用，不进入交付单元**。交付单元被作为模块加载时会执行其全部顶层代码——若 `precision_compare`、`{op}_golden_cpu` 的 import 写在模块顶层，此时会直接 `ModuleNotFoundError`，导致交付态全部 case 0 分。因此这两个 dev-only 依赖的 import **必须写在函数体内**（仿 [impl_template.py](templates/impl_template.py) 的 `_assert_precision`，import 在函数内 → 模块加载不触发 → 安全），**或在顶层用 `try/except ImportError` 容错**（仿已交付的 rms_norm）。**禁止裸顶层 `from precision_compare import` / `from {op}_golden_cpu import`**。`test_{op}.py` 必须能在仅含 `test_{op}.py` + `{op}_golden.py` 两文件的环境下被作为模块加载通过。
- **host 维度适配**：若 DESIGN.md §0 维度契约要求 kernel 只处理 2D 而 SPEC 需 1D/多维，在调用侧做 reshape 适配（模板见 impl_template）。
- **通过 wrapper 调 kernel**：test 函数必须通过 `{op_name}_wrapper` 调 kernel，不直接调 `{op_name}_kernel`。wrapper 内部做必要的 host 适配（输出分配 / dtype 转换 / num_cores 计算）后**只调用一次 kernel**——host 端预处理尽可能少，核心计算全部集中在单一 kernel 函数内（核心原则 #2/#3）。

### 步骤 7：本地验证与自修复闭环

必须严格按照以下指令格式运行算子脚本，不要进行额外的环境配置，默认环境可用：

```bash
python custom/<op>/test_<op>.py
```

- 确认输出 `PASS`， 编译报错 / NPU 错误 / 精度不通过 时，进入 debug 状态
- **环境问题例外**：若报错指向环境而非算子代码（如 `torch_npu` / `pypto_pro` 导入失败、`npu-smi` 无响应、NPU 设备不可见、CANN 未配置等），**不进入下方的 debug 自修复循环**——你被硬性规则禁止碰环境，无法自行修复。此时应收集证据并反馈给编排器：可参考 `.agents/skills/pypto-pro-environment-check` 跑其 Step 1 smoke 测试（`timeout 300 python .agents/skills/pypto-pro-environment-check/scripts/test_matmul_8k_example.py`）做归因判定——smoke PASS 说明环境正常、报错仍属算子代码问题，回到 debug 循环；smoke FAIL 确认环境异常，连同报错原文与 smoke 结果一并反馈给编排器，由编排器决定换卡或进一步反馈给用户

### 进入 debug 状态后的迭代流程

验证失败（编译报错 / NPU 错误 / 精度不通过）即进入 debug 状态。此时你负责完整的自修复闭环——**不得把问题抛回 orchestrator**（它只做产物验收，不参与调试），须自行定位、修复、重跑，直到 `PASS`。

> **这是一个迭代循环，不是走一遍就结束**：每碰到**一个**具体问题，都完整走一遍下面①→④；解决问题 A 后若重跑又暴露问题 B，就针对 B 重新从①开始再走一轮，直到所有问题清零、最终 `PASS`。不要试图一次性想清所有问题，也不要跳步。

**① 定位方向**：按 [references/debugging-methodology.md](references/debugging-methodology.md) 排查（症状快查表 → 分层 review → 最小复现 → 升级与切换）。该文档是调试的完整指引，此处不复述。
**② 定位到哪一层就修哪一层**：遵循「先修模型、再修 kernel」——根因在模型层（DESIGN.md / 维度契约）时先改正 DESIGN，别在错误设计上给 kernel 打补丁；根因在 kernel 实现时直接改代码。修改 DESIGN.md 后直接在 develop 内继续，不需回 Stage 3。以实际证据为准（查阅方式见步骤 1-2），可调整 API 序列、tile 属性/地址、循环结构、同步策略、尾块处理等，只要最终 PASS 且不违反两条性能强制与 kernel 结构约束。**⚠️ Vector 数值计算遇到当前 vf 方案不可行时（如 NPU 崩溃、精度不达标），优先尝试用其他 vf API 组合 + 循环结构手动实现替代方案，而非直接退化成 `pl.*` 标量写法——退化到 pl 会严重牺牲性能。**
**③ 记录修正**：在代码注释或 MEMORY.md 中记录"DESIGN.md 原方案 → 实际修正方案及依据"，便于后续追溯。
**④ 重跑验证**：修复后重新运行。若 `PASS` 且无告警，debug 结束；若暴露新问题，针对新问题回到 ① 再走一轮。

### 步骤 8：交付前自检

步骤 7 跑通 `PASS` 后、交付给 orchestrator 前，对照 orchestrator 验收清单逐项自检——把能在本地完成的检查全部做完，避免交付后被静态门禁打回返工。任一项不通过须当场修正并重跑，不得带病交付。

| 检查项 | 自检方式 |
|--------|---------|
| `test_{op}.py` 为最终单文件 | 确认文件存在；增量模式的临时中间产物（中间量 store / golden 辅助函数）已清理 |
| 设备与 golden 一致 | 确认已导入 `{op}_golden._get_device()`，未在各 test 中硬编码分散的 `npu:` 设备号 |
| atol 有据 | 确认 test 使用 `_assert_precision`（非 `torch.testing.assert_close`），无自定义 atol/rtol；golden 从 `{op}_golden_cpu` 导入（CPU 更高精度），非 `{op}_golden`（NPU 同 dtype） |
| 核心计算在单 kernel 内 | 确认文件中仅一个 kernel 函数（`@pl.jit` 装饰）；**所有的核心计算逻辑集中在单一 kernel 函数内**；host 端只做必要适配（reshape/cast/输出分配/num_cores 计算），不做核心计算；**禁止在循环中调用 kernel**（host 端循环 launch kernel 分担计算视为作弊） |
| ≥4 个独立 test 且与 §8 一致 | 确认 `def test_` 数量 ≥ 4，命名与覆盖对齐 DESIGN.md §8「目标测试 case」，非临时另造 |
| Vector 数值计算用 vf.* 手写 | 确认核心计算用 `vf.*` 指令手写（在 `@pl.vector_function` 内执行），非 `pl.*` 级计算 API。若 DESIGN.md §1 将 vec 步骤映射到 `pl.*` 而非 `vf.*`，视为 DESIGN.md 失误，须修正为 `vf.*` 并重新实现 |
| **入口函数命名合规** | 确认文件中暴露了名为 `{op_name}_wrapper` 的可调用入口函数（签名与算子 schema 一致）。外部调用方按命名约定查找 `{op_name}_wrapper` 和 `{op_name}`，推荐 `_wrapper` 后缀以与 kernel `_kernel` 配对。wrapper **只调用一次 kernel**，host 端预处理尽可能少（仅 reshape/cast/输出分配/num_cores 计算），test_{op_name}_* 通过 wrapper 调 kernel 而非直接调 `{op_name}_kernel` |

> 增量模式须在**清理临时产物、合并为最终单文件之后**再自检——自检针对交付态，不针对中间态。

---

## 核心原则

1. **DESIGN.md 是初步设计、运行验证是最终准则**：以 DESIGN.md 为初始依据，运行验证发现失误时据实修正（查阅方式见步骤 1-2）
2. **只写一个 .py 文件、只含一个 `@pl.jit` kernel**：kernel + 测试在同一文件中，**所有的核心计算逻辑集中在单一 kernel 函数内**（本工作流生成的 Pro 算子只需一个 `@pl.jit`，不需拆分为多个 kernel）
3. **入口函数**：命名为 `{op_name}_wrapper`，参数和返回值与算子定义一致，内部做必要的 host 适配后调用 kernel，**host端的预处理应该尽可能少**。`test_{op_name}_*` 必须通过 wrapper 调 kernel，**wrapper 只能调用一次 kernel**，**禁止在循环中调用 kernel**（host 端循环多次 launch kernel 分担本应在单次 kernel 内完成的计算视为作弊）。
4. **两条性能强制不可违背**：buffer 轮转用 `make_tile_group` + `auto_mutex`，Vector 部分计算用 `vf.*` 手写（详见上方「两条性能强制」节）
5. **官方指定算子是写法参考来源**：不确定时先查 EXPLORE_REPORT.md §4 可复用模式，不足时按 §B 定位官方指定算子原文
6. **不确定时不猜**：先查 EXPLORE_REPORT.md §3 的 API 约束，不足时按 §A 定位 API 文档原文确认
7. **测完整除 + 尾块两种 case**：泛化性验证
8. **不随意更改环境配置**：环境已在进入 Stage 4 前验证可用，出问题先查算子代码；确属环境问题则停下反馈（详见步骤 7 环境问题例外）
