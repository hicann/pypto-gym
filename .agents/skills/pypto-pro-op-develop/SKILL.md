---
name: pypto-pro-op-develop
description: PyPTO-Pro 算子 kernel 实现编码手册。以 DESIGN.md（初步设计）和 EXPLORE_REPORT.md（API 约束/样例模式/教程指导）为初始主要依据，以运行验证为最终准则，完成完整的 PyPTO-Pro kernel 代码并跑通验证。PRO_MATERIAL_INDEX.md 中的 API 文档 / pro_ops 样例 / 教程作为按需取用的补充查阅来源。触发词：实现算子、写 kernel、编写实现、写 impl、kernel 实现、pypto-pro develop。
---

# PyPTO-Pro 算子 Kernel 实现

基于 DESIGN.md（初步设计）生成完整的 PyPTO-Pro kernel 实现文件（一个 `.py` 文件，含 kernel 函数 + 测试函数），并本地跑通验证。运行验证为最终准则——当运行结果与 DESIGN.md 冲突时，以实际 API 文档、教程、pro_ops 样例为准修正 DESIGN.md 的失误。

> **角色说明**：PyPTO-Pro 的 Stage 4 由单个 `general` 子代理全包完成，暂时没有拆分为 coder / verifier / debugger 等多个专用代理。因此本 skill 的承担者需独自完成 **开发 → 验证 → 发现问题 → 分析根因 → 解决问题 → 再验证** 的完整闭环，直到能够交付符合全部要求的算子代码。

---

## 输入

### 主要依据（必读）

| 来源 | 内容 | 用途 |
|------|------|------|
| `custom/<op>/DESIGN.md` | Phase 划分（§0）、API 映射（§1）、Tile 规划（§2）、UB 空间布局（§3）、循环与 Section（§4）、分核/流水/尾块（§5-7）、全景图（§9） | **初步设计**——kernel 实现的初始主要依据；运行验证发现问题时可据实修正 |
| `custom/<op>/EXPLORE_REPORT.md` | API 约束（§3）、相似样例与可复用模式（§4）、教程指导（§5）、Tile/同步策略建议（§6） | **编码参考**——API 约束速查、样例写法定位、教程设计指导 |

### 按需取用（EXPLORE_REPORT.md 不足时查阅）

| 来源 | 内容 | 用途 |
|------|------|------|
| `custom/<op>/PRO_MATERIAL_INDEX.md` | API 文档（§A）、pro_ops 样例（§B）、教程（§C）的精确路径索引 | **路径定位**——先查索引找到文档/样例路径，再读取原文 |

## 参考文件

| 文件 | 用途 | 加载时机 |
|------|------|----------|
| [templates/impl_template.py](templates/impl_template.py) | kernel 文件骨架（tile 声明 + section + Phase + 测试函数） | 生成代码前必读 |
| [references/pitfalls.md](references/pitfalls.md) | 常见编译错误 / 精度问题的症状与修复 | 发现参考已有文档和算子用例无法解决的编译错误与精度问题时必读 |
| [scripts/list_idle_chip_ids.sh](scripts/list_idle_chip_ids.sh) | 查找空闲 NPU chip | 运行前按需执行 |


---

## 开发流程

### 步骤 1：确认输入齐全

读取 DESIGN.md，确认以下信息完整：

| 信息 | 所在位置 | 缺失时处理 |
|------|---------|-----------|
| Phase 划分与数据依赖 | DESIGN.md §0 | 优先查 EXPLORE_REPORT.md / pro_ops 样例补全；仍缺失时回调 Stage 3 |
| API 映射表（每个 Phase 内的 pl.* 序列） | DESIGN.md §1 | 优先查 API 文档 / pro_ops 样例补全；仍缺失时回调 Stage 3 |
| Tile 规划（shape/dtype/layout） | DESIGN.md §2 | 优先查 pro_ops 样例补全；仍缺失时回调 Stage 3 |
| UB 地址映射表（所有 tile 的 shape/dtype/地址） | DESIGN.md §3 | 优先查 pro_ops 样例补全；仍缺失时回调 Stage 3 |
| 循环结构（M-tile / N-tile 嵌套、ceiling division） | DESIGN.md §4 | 可与步骤 2 同步推进 |
| 分核/流水/尾块策略 | DESIGN.md §5-7 | 参考 pro_ops 样例确认 |
| Tile 数据流全景图 | DESIGN.md §9 | 确认理解全局数据流 |
| API 约束 / 相似样例 / 教程指导 | EXPLORE_REPORT.md §3-5 | 步骤 2/5 按需查阅 |

信息不足时优先查 EXPLORE_REPORT.md / API 文档 / pro_ops 样例补全，仍缺失时向用户反馈。不做猜测，但运行验证发现的 DESIGN.md 失误应据实修正。

### 步骤 2：逐 API 确认参数

对 DESIGN.md §1 中的每一个 API 调用，**必须**确认三个信息：

1. **参数名是否为关键字参数**：大部分是位置参数，少数（如 `set_pipe=`、`wait_pipe=`、`event_id=`）是关键字参数。参数传参方式（位置 vs 关键字）以 API 文档原文为准，EXPLORE_REPORT §3.3 已汇总部分
2. **参数顺序**：如 `pl.row_max(dst[M,1], src[M,N], tmp[M,N])`（dst 先于 src），`pl.load_tile(tile, tensor, [i, j])`（tile 先于 tensor）。具体顺序以 API 文档为准
3. **约束条件**：dtype 限制、layout 要求、MemorySpace 约束——以 API 文档原文为准，EXPLORE_REPORT §3.3 已汇总部分

**确认方式**（先汇总后原文，减少阅读量）：
1. **先查 EXPLORE_REPORT.md §3**（API 映射结果 + API 约束 + MemorySpace 约束已汇总）——大部分信息在此可直接获得，无需读原文
2. **§3 不足时**，再通过 `PRO_MATERIAL_INDEX.md` §A 定位对应 API 文档路径，读取文档原文的"函数原型"和"参数范围"两个表补充

**产出要求**：步骤 2 完成后，应形成一份覆盖 DESIGN.md §1 全部 API 的参数速查清单，后续步骤直接复用，不再重复确认。

### 步骤 3：编写 tile 声明

1. **先从 DESIGN.md §2.1 复制编译期常量**（TS、TD、TS_HALF、SCALE 等）到模块级——这些是 tile 尺寸和公式常量的来源；若运行验证证明有误，以 API 文档 / pro_ops 样例为准修正——⚠️ 必须声明在 kernel 函数**外部**，写进函数体内会触发编译错误 `Unsupported kwarg type for key: memref_size`（见 pitfalls §1.2）
2. **按 DESIGN.md §2 Tile 属性表 + §3 UB 地址映射表，逐条写出 tile 声明**——§2 提供 shape/dtype/layout，§3 提供地址分配；运行验证发现地址/属性有误时据实修正
**首选 `pl.make_tile_group()` + `auto_mutex`**，由框架自动管理 buffer 切换和同步；`pl.make_tile()` 手动分配为次选。每条声明必须精确复制表中的 shape / dtype / layout / addr / size。

**格式规范**（参考 pro_ops 样例）：

```python
# 首选方案：make_tile_group + auto_mutex（参考 EXPLORE_REPORT §4 定位的 element-wise 样例）；次选 make_tile
a_db = pl.make_tile_group(type=tile_type, addrs=0x00000, mutex_ids=[0, 1])

**核对清单**：
- [ ] 每个 tile 的 shape / dtype / addr / size 与 DESIGN.md 一致
- [ ] 数据 tile 标注了 `valid_shape=[-1, -1]`
- [ ] 双视图 tile 对地址相同、size 相同
- [ ] layout 约束（如归约类 API 输出 `[M,1]` 须设 `layout=pl.DN`，见 `row_max.md:25`）以 EXPLORE_REPORT §3.3 / API 文档为准
- [ ] `make_tile_group` 的参数传参方式以 API 文档为准
- [ ] 地址连续、无重叠

### 步骤 4：编写 section 和循环框架

按 DESIGN.md §4（循环与 Section 结构）搭出 kernel 骨架：section 声明、SPMD 原语获取位置、循环嵌套、同步点占位。**不填入具体 API 调用和尾块代码**（步骤 5 完成）。
SPMD 原语获取位置、循环嵌套、同步点位置均已在 DESIGN.md §4-6 中确定，直接按其翻译即可。

```python
# SPMD 原语获取位置按 DESIGN.md §4（多 section 在 section 外闭包共享，单 section 在 section 内）
num_cores = pl.get_block_num()
core_id = pl.get_block_idx()

with {SECTION_TYPE}():  # 来自 DESIGN.md §0
    for work_id in pl.range(core_id, {TOTAL_WORK}, num_cores):  # 分核方式来自 DESIGN.md §5
        # sync: {同步点占位} — 按 DESIGN.md §4-6 标注，步骤 5 填入具体 pl.system.* API
        # Phase 实现 — 步骤 5 逐行翻译
        ...
```

### 步骤 5：编写 Phase 内部实现

将步骤 4 骨架中的占位填入实际代码。逐 Phase 翻译，每个 Phase 对照 DESIGN.md 全景图（§9）确认输入/输出 tile：

| 填入内容 | 来源 |
|---------|------|
| `pl.*` API 调用序列 | DESIGN.md §1（参数直接使用步骤 2 已确认的结论） |
| tile 变量名 | DESIGN.md §3 地址映射表 |
| 同步 API（`pl.system.bar_all/bar_v/sync_src/sync_dst/set_cross_core/wait_cross_core`） | DESIGN.md §4 同步标注 + §5/§6 策略 |
| 尾块处理（`pl.min` + `set_validshape`） | DESIGN.md §7 |
| 写法参考 | 先查 EXPLORE_REPORT.md §4（可复用模式），不足时按 §B 定位 pro_ops 样例原文 |

### 步骤 6：编写测试函数

kernel 函数完成后，在同一个文件中编写测试函数。从 DESIGN.md 获取算子输入张量的 shape 和 dtype，从 `{op}_golden.py` 确认 golden 函数的签名。

**host 维度适配**（DESIGN.md §0 维度契约要求时）：若 kernel 只处理 2D 而 SPEC 要求 1D/多维，在测试/调用侧做 reshape 适配：

```python
def _run_kernel(inp, out):
    orig_shape = inp.shape
    if inp.dim() == 1:                    # 1D → 2D
        inp2d = inp.reshape(-1, 1)
        out2d = out.reshape(-1, 1)
    else:
        inp2d, out2d = inp, out
    {op}_kernel[None, block_dim](inp2d, out2d)
    torch.npu.synchronize()
    return out2d.reshape(orig_shape)      # 还原
```

> ⚠️ **设备选择必须与 golden 一致**：Kernel 执行是异步的，设备不一致会导致 NPU 错误污染 golden 执行，traceback 指向 torch 而非 kernel。**不要硬编码 `npu:0`**——从 `{op}_golden.py` 导入 `_get_device()` 函数（golden 模板已通过 `TILE_FWK_DEVICE_ID` 环境变量选择设备）。
>
> ⚠️ **atol 必须合理**：参考官方 PyPTO-Pro 教程与 pro_ops 样例，根据 golden 对比结果收紧。

```python
# 示例：{input_shapes} 和 {dtype} 替换为 DESIGN.md 中的实际规格
from {op}_golden import {op}_golden, _get_device

def test_{op}():
    device = _get_device()               # 与 golden 使用同一设备
    torch.manual_seed(42)

    inp = torch.randn({input_shapes}, device=device, dtype={dtype})
    out = torch.zeros({output_shapes}, device=device, dtype={dtype})

    {op}_kernel[None, {block_dim}](inp, out)
    torch.npu.synchronize()

    out_ref = {op}_golden(inp)
    torch.testing.assert_close(out, out_ref, rtol={rtol}, atol={atol})
    # atol 取值：FP16 matmul / 融合类算子可先用 1e-2，纯逐元素默认 1e-5
    logging.info("{op} PASS")
```

**测试 shape 选择**：至少覆盖两个 case——所有 tile 维度均可整除的 shape（如 `[TILE_A, TILE_B]`），和至少一个 tile 维度存在尾块的 shape（如 `[TILE_A + 22, TILE_B - 30]`），具体取值以 DESIGN.md 中的 tile 尺寸为基准。**建议拆为两个独立的 test 函数**（如 `test_{op}_aligned` 整除 case + `test_{op}_tail` 尾块 case），orchestrator 门禁以 `def test_` 数量 ≥ 2 作为泛化性判据。

### 步骤 7：本地验证与自修复闭环

必须严格按照以下指令格式运行算子脚本，不要进行额外的环境配置，默认环境可用，如果环境确实出现问题，应收集证据并反馈给用户：

```bash
python custom/<op>/test_<op>.py
```

- 确认输出 `PASS`
- 如编译报错，根据错误信息回到对应步骤修正（大多数错误是 API 参数类型/顺序/关键字问题）
- 如运行报错（NPU 错误），检查 tile 地址是否重叠、UB 总量是否超出、sync 是否遗漏
- 如精度不通过，对照 `{op}_golden.py` 逐 Phase 排查计算逻辑差异

**Debug 状态下的决策权限**：

当运行验证发现问题（编译报错 / NPU 错误 / 精度不通过）进入 debug 状态时，**你拥有最高决策权，不受 DESIGN.md 约束**：

1. **正视 DESIGN.md 可能存在的问题**：DESIGN.md 是初步设计，其 API 映射、Tile 规划、UB 地址、循环结构等均可能在运行中暴露错误。不得因"DESIGN.md 这么写"而拒绝修正。
2. **以实际证据为最终准则**：以 EXPLORE_REPORT.md、PRO_MATERIAL_INDEX.md 为路径索引，查阅具体的 API 文档（约束 / 函数原型 / 参数范围）、指导教学文档、官方 pro_ops 样例的实际写法，据此修正 DESIGN.md 的失误。
3. **修正范围**：可调整 API 调用序列、tile shape / dtype / layout / 地址、循环结构、同步策略、尾块处理等，只要最终运行验证 PASS 且不违反 kernel 结构约束。
4. **记录修正**：在代码注释或 MEMORY.md 中记录"DESIGN.md 原方案 → 实际修正方案及依据"，便于后续追溯。

**你负责完整自修复闭环**：遇到任何失败不得将问题抛回给 orchestrator（orchestrator 只做产物验收，不参与调试）。必须自行参照本 skill 各步骤以及 [references/pitfalls.md](references/pitfalls.md) 定位问题、修复代码、重新运行，直到 `PASS`。

---

## 核心原则

1. **DESIGN.md 是初步设计、运行验证是最终准则**：初始实现以 DESIGN.md 为主要依据，API 约束/样例模式/教程指导先查 EXPLORE_REPORT.md；不足时再按 PRO_MATERIAL_INDEX.md 定位 API 文档/pro_ops 样例/教程原文。运行验证发现 DESIGN.md 失误时，以实际 API 文档 / pro_ops 样例为准修正，不受 DESIGN.md 约束
2. **只写一个 .py 文件**：kernel + 测试在同一文件中，不需拆分
3. **pro_ops 样例是写法字典**：不确定怎么写的，先查 EXPLORE_REPORT.md §4 的可复用模式，不足时按 §B 定位 pro_ops 样例原文
4. **不确定时不猜**：先查 EXPLORE_REPORT.md §3 的 API 约束，不足时按 §A 定位 API 文档原文确认
5. **测完整除 + 尾块两种 case**：泛化性验证
6. **不随意更改环境配置**：当流程进入 Stage 4 就说明环境已经验证可用，出现问题时绝对不能归咎于环境，应该多检查、分析和纠正算子代码写法，如果确实发现是环境问题，应该停下并反馈给用户。

## 禁止事项

- 不看 API 文档就写参数——必须逐 API 确认签名和顺序
- 不看 DESIGN.md 就凭空编造 tile 地址——初始实现须依据 DESIGN.md 地址映射表；运行验证暴露错误时以实际 API 文档 / pro_ops 样例为准修正
- 抄 pypto（非 Pro）的 API 用法——`pl.*` vs `pypto.*` 是两套东西
- 跳过测试就跑——写完必须本地验证通过