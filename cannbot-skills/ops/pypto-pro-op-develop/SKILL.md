---
name: pypto-pro-op-develop
description: 实现、调试并自验证 PyPTO-Pro 算子 kernel。用于 Stage 3 已通过门禁，需按 DESIGN.md 和 Module 合同开发纯 Vector、纯 Cube 或融合算子；L0 交付 test_{op}.py，L1 按编排器指定的 Module 交付 staged 文件。发现设计、能力或环境问题时返回有证据的分类结论，不修改上游设计或自行认证 Stage 完成。
---

# PyPTO-Pro 算子 Kernel 实现

把已冻结的 `DESIGN.md` 翻译为一个可运行、可测试的 PyPTO-Pro kernel。完成本地开发与自验证闭环后交给独立 verifier；本 skill 不修改 SPEC、DESIGN、Module 合同或编排状态。

## 输入与输出

先读取以下输入：

- `custom/<op>/DESIGN.md`：Stage 4 的施工合同，包含 Module、API、tile、地址、循环、同步、尾块和测试设计。
- `custom/<op>/module_interfaces.yaml`：L1 当前 Module 的输入来源、输出与 `golden_steps`。
- `custom/<op>/EXPLORE_REPORT.md`：已核对的 API 约束和相似样例。
- `custom/<op>/PRO_MATERIAL_INDEX.md`：需要回看原文时的 API、官方样例和教程路径。
- `custom/<op>/KB_SELECTION.json` 及其中选中的所有参考。

按 dispatch 交付：

| 路径 | dispatch | 本轮产物 |
|---|---|---|
| L0 | 无 `module_k` | `custom/<op>/test_<op>.py` |
| L1 | 带 `module_k` | `custom/<op>/modules/test_<op>_module<suffix_k>.py` |

两条路径都必须在 `KB_SELECTION.json` 所在的 class 目录更新 `KB_USAGE.json`；L1 不把它写进 `modules/`。L1 的最终 cleanup 由编排器调用 `scripts/gen_cleanup.py` 完成，本轮 coder 不提前生成或改写最终文件。

## 实现合同

施工时始终满足以下本阶段职责：

1. 一个交付文件只含一个 `@pl.jit` kernel，核心计算全部在该 kernel 内；wrapper 只启动一次 kernel，且启动不在 host 循环内。
2. 严格执行 DESIGN.md 已冻结的 Module 边界、API 序列、tile 属性/地址、循环、同步、尾块和 `vector_selection`。本轮功能或精度测试表明设计有误时返回 `design_violation`，不得在 Stage 4 静默改设计。
3. 轮转 tile 使用 `make_tile_group` + `auto_mutex`；`make_tile` 只用于不参与轮转的单次 scratch。不要在 `auto_mutex` 管理的 tile 上叠加手动 `sync_src`/`sync_dst`。
4. wrapper 默认只做参数检查与 Python 形状推导、`torch.empty` 输出分配、一次 kernel 启动。完整边界、迁移方式和例外条件只以 [wrapper-boundary.md](../../pypto-pro-op-kb/constraints/wrapper-boundary.md) 为准。
5. 测试通过 wrapper 调用 kernel；不得删改 DESIGN.md §8 的 case 来迁就实现，也不得把核心计算移到测试或 host 代码。

## 按需读取的资源

本表是本 skill 资源索引的唯一来源（含加载时机）。

| 资源 | 用途 | 加载时机 |
|---|---|---|
| [templates/pure_vec_impl_template.py.tmpl](templates/pure_vec_impl_template.py.tmpl) | 纯 vec 通用骨架；使用前按目标 API 与 DESIGN.md 核对 | 纯 vec 算子生成代码前 |
| `$PYPTO_DEVKIT_DIR/pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset_dynamic.py` | **纯 cube 算子候选实现起点**——两级 K 分块 + move offset + 嵌套四分支 K 累加 + 尾块处理 + ASW 蛇形调度；仅在该路径存在且目标版本匹配时使用 | 纯 cube 算子生成代码前 |
| [templates/impl_template.py.tmpl](templates/impl_template.py.tmpl) | kernel 文件骨架（tile 声明 + section + Module + 测试函数）——**通用**（含 CV 融合 / 多 Module / 跨核流水）；CV 融合算子当前无专用模板，走此通用骨架，按「开发流程」各步执行 | 生成代码前必读 |
| [references/debugging-methodology.md](references/debugging-methodology.md) | 调试方法论：**先确认失败可信** → 症状快查表 → 分层 review → 定位技术（最小复现/消融/单原语替换/差异属性）→ 升级切换 → **修正后验证** + 诊断纪律 | 验证失败进入 debug 状态时必读 |
| [references/vf-reduction-perf.md](references/vf-reduction-perf.md) | Vector reduction 的数值安全、正确 API 形态与性能注意事项 | 实现或调优 Vector reduction 时 |
| [references/cv-matmul-direct-buffering.md](references/cv-matmul-direct-buffering.md) | Cube 累加器在同一 launch 内被 Vector epilogue 消费时的轮转缓冲与 `auto_mutex` 形态 | 实现 CV 直连 matmul 时 |
| [templates/cv_matmul_direct_buffering.py.tmpl](templates/cv_matmul_direct_buffering.py.tmpl) | 上一行的可替换代码骨架 | 同上 |
| [templates/fp32-chain-precision-fragments.py.tmpl](templates/fp32-chain-precision-fragments.py.tmpl) | fp32 链路的精度安全片段 | 需要与 CPU 参考逐位对齐时 |
| [../../pypto-pro-op-kb/ROUTER.md](../../pypto-pro-op-kb/ROUTER.md) | 按任务选择一个补充约束、pattern 或 validated study kernel | API 文档与官方样例不足时 |
| [scripts/list_idle_chip_ids.sh](scripts/list_idle_chip_ids.sh) | 查找空闲 NPU chip | 运行前按需执行 |
| [pypto-pro-op-perf-tune](../pypto-pro-op-perf-tune/SKILL.md) | wrapper 动态反作弊检查：只检查 profile 中的 device op，不评价性能 | 需要验证 wrapper 未绕过 kernel 时 |
| [KB CONTRACT](../../pypto-pro-op-kb/CONTRACT.md) | KB 选择与使用格式的完整合同，唯一来源 | 写 `KB_SELECTION.json` / `KB_USAGE.json` 前 |

参考之间是互补关系：DESIGN.md 决定“实现什么”，纯 Vector 模板或 Cube 官方样例提供“如何写”的主要起点，`KB_SELECTION.json` 已选参考补充必须落实的 pattern 和约束；三者不得相互替代。模板和样例不是 API 或性能事实源，使用时仍须核对目标版本 API 文档；与 DESIGN.md 或已选 KB 冲突的参考片段直接弃用，只有上游合同本身无法同时落实时才按根因分流，Stage 4 不自行改合同。标为 conceptual 的片段不得直接复制成交付代码。

## 开发流程

### 1. 锁定本轮范围

识别 dispatch 是否包含 `module_k`：

- L0 一次实现全部设计，产出最终 `test_<op>.py`。
- L1 每轮只扩展当前 Module，但 staged 文件必须是实现 Module 1..k 的完整可运行算子。复制上一轮已验证文件生成新 suffix 文件后，只修改这个新文件；此前的 staged 文件全部只读。例如开发 `test_<op>_module123.py` 时只能修改该文件，`test_<op>_module1.py` 和 `test_<op>_module12.py` 保持不变。可在新文件的 kernel 中调整复制过来的前序 Module 连接实现，但不得回改前序文件，也不新建第二个 kernel。

L1 的 `suffix_k` 是累积序号：1 → `1`，2 → `12`，3 → `123`。文件和 wrapper 分别命名为 `test_<op>_module<suffix_k>.py` 与 `<op>_wrapper_module<suffix_k>`。确认前一 staged 文件和对应 `<op>_golden_stage<suffix_k>.py` 已存在；Module 1 没有前序文件。

### 2. 检查施工信息

从 DESIGN.md 逐项确认：

- §0：I/O、动态维度、Module 和数据依赖；
- §1：API 序列及 Vector `vector_selection`；
- §2/§3：tile shape、dtype、layout、地址和各空间容量；
- §4：section 与循环；
- §5–§7：分核、流水、同步和尾块；
- §8：至少 4 个目标测试 case；
- §10：完整数据流。

L1 还要用 `module_interfaces.yaml` 核对当前 Module 的 `inputs`、`outputs`、`golden_steps` 和 section 类型。缺少关键合同，或上下游产物互相矛盾时停止编码并返回 `design_violation` 证据。

### 3. 核对全部 API

对 DESIGN.md §1 的每个 API 核对函数原型、位置/关键字参数、参数顺序、dtype、layout、MemorySpace 与目标平台支持：

1. 先读 EXPLORE_REPORT.md §3 的汇总结论；
2. 信息不足时，通过 PRO_MATERIAL_INDEX.md 定位并读取目标版本 API 原文；
3. 对不确定写法，在官方指定算子中找到同平台 working example。

形成覆盖本轮全部 API 的简短速查清单，后续直接复用。不要把编译错误直接解释为框架不支持。

### 4. 搭建 tile、section 与循环

把 DESIGN.md 的编译期常量放在 kernel 外部。逐条复制 tile 属性和地址，确认空间不重叠；按 §4–§6 搭建 section、SPMD 原语位置、循环和同步；按 §7 为整除与尾块路径预留正确的 valid-shape、填充和 store 处理。

- 纯 Vector（包括 L1 的 Vector Module）：以 `pure_vec_impl_template.py.tmpl` 为主要编码参考。`vector_selection: vf` 时只保留一个匹配当前计算依赖的 Pattern，删除其他 Pattern；`vector_selection: tile_op` 时只借用模板中与当前合同一致的文件、tile/section 和测试组织结构，计算写法严格执行 DESIGN.md 和已选 KB pattern。
- 纯 Cube（包括 L1 的 Cube Module）：以上表的目标版本官方 matmul 样例为主要编码参考，使用其 tile、K 循环、phase 和尾块的已验证写法，再按 DESIGN.md 调整并落实已选 KB 约束。
- CV 融合或 L1 分 Module 开发：每轮按当前 Module 的 Vector/Cube section 复用上述对应方式，在同一 kernel 内从 Module 1..k 逐步扩展。若 PRO_MATERIAL_INDEX.md §B 有与当前数据流匹配的官方 CV 融合样例，优先核对其 Module 衔接和同步写法。通用模板仅提供文件骨架，不把纯 Vector/Cube 完整算子直接拼接，也不为每个 Module 新建 kernel。

L1 的逐 Module 交付只规定开发与验证顺序，不等于运行时整段串行。连接相邻 Vector/Cube section 时，只在 `KB_SELECTION.json` 已选参考或 PRO_MATERIAL_INDEX.md 定位的目标版本 PyPTO-Pro 文档、源码和样例中查找与冻结 DESIGN.md 匹配的连接方式：有匹配方式就优先在本轮新 staged 文件中采用，使上下游处理不同 tile/基本块时能够重叠；没有则执行 DESIGN.md 的现有连接兜底。可以调整新文件中复制过来的前序 Module 连接实现，但不得改动历史 staged 文件，也不得改变数学、接口、支持范围或自行发明连接协议。

若本轮功能或精度测试证明 tile、地址、循环、同步或实现选择本身错误，按设计问题分流，不在代码中私自改合同。

### 5. 填写 kernel 实现

按 Module 把 DESIGN.md §1 的 API 序列翻译为代码，以 §10 校验每一步输入、输出和数据所有权。同步点严格取自 §6；尾块严格取自 §7。所有 dtype 转换、轴变换、padding、索引和计算都在 kernel 内完成，除非 DESIGN.md 已列出并通过门禁的 wrapper 例外。

实现与任何冻结常量、算法步骤或布局不一致时，不得静默交付。先判断是抄录错误还是设计错误：前者修代码，后者返回 `design_violation`。

### 6. 编写 wrapper、测试和 KB 使用记录

入口命名是硬合同：L0 必须命名为 `<op>_wrapper`，L1 staged 入口必须命名为 `<op>_wrapper_module<suffix_k>`。wrapper 的签名与算子 schema 一致；L1 使用 `primary_inputs`，输出当前 Module 的结果。optional 参数若可被调用方省略，必须提供相应默认值。wrapper 只执行 [wrapper-boundary.md](../../pypto-pro-op-kb/constraints/wrapper-boundary.md) 允许的三类动作；只有 DESIGN.md 已冻结的逐项例外可原样实现，并在 `KB_USAGE.json` 中记录 `deviated` 和理由。

TensorList（`is_list: true`）同样只能启动一次 kernel，且启动不得位于 host 循环内。参数展开、固定 arity、地址/shape 传递与 work-item 映射必须原样执行 DESIGN.md 及 `KB_SELECTION.json` 已选 TensorList pattern；Stage 4 不自行发明新的打包协议或改写支持范围。

完成 wrapper 后用 `pypto-pro-op-perf-tune` 采集 profile，仅作动态反作弊/边界检查：`op_times.device_kernels` 中不得出现未经 DESIGN 批准的 `aclnn*`。命中时把对应计算迁入 kernel；确因目标版本能力受限则返回 `design_violation`，不得事后补写设计。这一步不比较耗时或判定性能，性能验收仍由 Stage 5 完成。

在同一文件实现 DESIGN.md §8 的全部 case：

- 至少覆盖整除、尾块和跨多 tile 等设计分支；每个 case 使用独立 `def test_...`。
- 从 `<op>_golden._get_device()` 获取设备，不硬编码卡号。
- 测试前把本 skill 的 `scripts/precision_compare.py` 复制到 `custom/<op>/`，用模板 `_assert_precision` 对比 CPU FP32 golden，不自定义阈值。
- `precision_compare` 与 `<op>_golden_cpu` 是 dev-only；import 放在函数体内，保证交付单元仅含 test 与 NPU golden 时可安全导入。
- L1 改为对比当前 `<op>_golden_stage<suffix_k>`；它同样属于 dev-only 依赖，必须在 `_assert_precision` 或测试函数体内 import，禁止顶层 import。

读取 `KB_SELECTION.json` 中每个 `optional_patterns` 和 `required_constraints`，再按 [KB CONTRACT](../../pypto-pro-op-kb/CONTRACT.md) 生成 `KB_USAGE.json`。本阶段只声明选中参考如何落到真实文件和符号；不得引用未选路径，也不得写 `verified`。`deviated` 或 `not_applicable` 必须说明理由。写完用标准库 `json.load` 做格式预检，verifier 才负责最终裁决。

若实现中发现必须使用未选参考，返回 `kb_selection_invalid` 请求回 Stage 1 更新选择；不得只在 `KB_USAGE.json` 中补路径。

### 7. 运行并调试

只使用当前环境运行本轮文件：

```bash
# L0
python custom/<op>/test_<op>.py

# L1
python custom/<op>/modules/test_<op>_module<suffix_k>.py
```

成功标准是全部 case 输出 PASS 且没有未解释告警。失败时读取 [debugging-methodology.md](references/debugging-methodology.md)，先确认失败可信，再按症状和层级定位；用最小复现或消融隔离根因，修复后重跑目标 case、相反分支和全量 case。

按根因返回或继续：

| 根因 | 动作 |
|---|---|
| 当前实现的代码翻译、参数或局部细节 | 修复本轮文件并重跑 |
| DESIGN 的维度、API 序列、tile、循环、同步、尾块或 `vector_selection` | 返回 `design_violation`，附错误原文、最小复现和对应合同位置；不改 DESIGN |
| 已核对文档、官方样例并穷尽 DESIGN 允许路径后确认框架能力缺口 | 返回 `capability_gap`，附目标版本、原始错误、尝试路径和各自失败证据；不在 host 端绕过 |
| 导入、CANN、设备不可见或疑似 hang | 加载 `pypto-pro-environment-check` 按其流程评定并把证据交给编排器；不自行改环境 |

### 8. 交付前自检

交付前执行一次紧凑自检：

- 产物路径和 L0/L1 命名正确；L1 staged 链保留。
- `@pl.jit` 恰好一个，wrapper 启动 kernel 恰好一次且不在循环内。
- 实现逐项符合 DESIGN.md，Vector 使用冻结的 `vector_selection`。
- L1 只修改本轮新 staged 文件，历史 staged 文件未变；异构 Module 连接符合 DESIGN.md 和所用参考。
- wrapper 只做三类允许动作；任何例外已在 DESIGN 和 `KB_USAGE.json` 对应。
- DESIGN.md §8 全部 case 已实际运行并通过；测试数据、dtype、shape、value range 未被偷换。
- dev-only import 位于函数内，交付态模块可安全导入。
- `KB_USAGE.json` 可由 `json.load` 读取，覆盖全部选中参考且文件、符号真实存在。
- 返回运行命令、原始结果摘要、产物路径以及任何偏差或分类 verdict；不要声称 verifier PASS。

满足以上条件后交给 orchestrator。门禁与状态推进仍由独立 verifier 和编排器完成。
