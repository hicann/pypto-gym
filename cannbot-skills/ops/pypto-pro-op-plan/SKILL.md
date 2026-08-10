---
name: pypto-pro-op-plan
description: 算子规划：串行组织需求理解与资料探索，产出算子规格与可复用的资料索引。用于在设计开始前把自然语言需求与可用 API/样例整理成结构化依据。首先加载 pypto-pro-intent-understand 产出 SPEC.md，之后加载 pypto-pro-material-explore 构建 PRO_MATERIAL_INDEX.md 全量资料索引并产出 EXPLORE_REPORT.md，初始化 MEMORY.md。
---

# PyPTO-Pro 复杂 Kernel — Stage 1 规划

## 规划流程

> 先需求、后探索：资料探索须基于 SPEC.md 中的算子公式、shape、dtype 等明确需求进行，确保探索具有目的性，避免盲目搜索。
>
> **执行方式**：本 skill 由单个子代理在同一 session 内**串行加载** `pypto-pro-intent-understand` 和 `pypto-pro-material-explore`，非嵌套 dispatch 子子代理。先完成 intent-understand 产出 SPEC.md，再加载 material-explore 基于 SPEC.md 进行探索。

### Step 1：需求总结

**必须**加载 skill `pypto-pro-intent-understand`，按其流程产出 `SPEC.md`。

> `pypto-pro-intent-understand` 是 PyPTO-Pro 专属的需求理解组件，产出的是**通用 SPEC**（公式、dtype、shape、关键特性、优先级等）。它不覆盖 kernel 层的若干契约字段——这些由下面的 Step 1.5 在其产物基础上补齐。

### Step 1.5：Pro kernel 契约补充

在 Step 1 产出通用 SPEC.md **之后**，基于它补齐 kernel 层特有的契约信息。产物**追加到 SPEC.md 末尾的「## kernel 契约补充」节**（同时在 MEMORY.md 记一份裁定摘要），供 Step 2 探索与后续 design / develop 消费。

#### 1. 字段归属裁定（三档）

对每个契约字段判定「由谁定」，避免子代理在 Stage 1 凭空猜测本该由用户或 design 决定的事：

| 档位 | 含义 | 处理 |
|------|------|------|
| **ASK** | 公式无法消解、须用户拍板 | 已由 Step 1 的 intent-understand 采集；此处只核对是否齐全 |
| **MAY-DESIGN** | 既不猜也不问用户，交由 design 阶段设计 | 在补充节标注「留待 design」，**不在 Stage 1 定值** |
| **MAY-ASSUME** | 可取仓库 / 对齐默认 | 取默认并注明 |

逐字段归属：

- **ASK**（Step 1 应已确认）：目标公式、输入/输出 dtype。
- **MAY-DESIGN**（标注留待 design，不问用户）：输入张量 **shape**、**拓扑**（纯 vector / cube→vec / …，由 design R0 Module/Section 决定）、**尾块行为**（design R7/R7.5）、**tile 族 / 切分 / 片上地址**（design R2/R3）。
- **MAY-ASSUME**（取默认并注明）：**设备**默认 a5；**matmul 累加 dtype** 默认 float（与 golden 的 `.float()` 累加对齐）。

#### 2. 三个 kernel 契约字段（SPEC 通用模板未覆盖，此处必补）

这三项直接影响 tile 规划、精度与多核写回，通用 SPEC 模板无对应字段，须在补充节显式写出（无则注明「不涉及」）：

| 字段 | 说明 | 缺省判断依据 |
|------|------|-------------|
| **辅助张量暂存策略** | bias / mask / scale 向量等：调用侧预展开成满 tensor，还是 kernel 内用 vf 广播指令实现 | 影响 tile 规划；歧义时归 ASK 向用户确认。|
| **cast 边界链** | input → matmul 累加 → 后处理 → 输出 各段 dtype 及降精度时点 | 精度正确性关键；累加段默认 float |
| **累加语义** | 跨多次启动 / 多核：覆盖写 vs 原子累加（后者需输出预清零） | 多核归约场景必判，默认覆盖写 |

#### 3. 无用户应答时（自动化运行）

用户无法实时回复时，ASK 字段无法当场确认：仍按 intent-understand 写出该问的问题，再为每个未决字段取最佳推测默认值，并在 SPEC 补充节与 MEMORY.md **显式记录每个假设**，供后续追溯与用户事后复核。MAY-DESIGN 字段无需在此处理，正常流转到 design。

### Step 2：资料探索

**必须**加载 skill `pypto-pro-material-explore`，先构建覆盖全流程的资料索引 `PRO_MATERIAL_INDEX.md`（后续所有 Stage 均以该索引为权威目录，不再依赖盲目 grep），再基于索引从三个方向依次探索：API 文档（公式分解、`pl.*`/`vf.*` API 映射、约束验证 dtype / layout / MemorySpace / tile shape）、官方指定算子（典型算子实现参考）、教学文档（设计模式与关键约束），产出 `EXPLORE_REPORT.md`。

### 必要规划文件

`custom/<算子名称>/MEMORY.md` 最迟在 Step 1.5 创建（用于承接契约裁定摘要），全流程持续追加。Stage 1 结束时必须包含：

- 任务摘要
- **契约裁定摘要**（来自 Step 1.5：三档归属结论 + 三个 kernel 契约字段的取值 / 假设）
- 参考位置（包括 `PRO_MATERIAL_INDEX.md` 的路径及索引中的官方指定算子路径）
- **PyPTO-Pro API 映射**（来自 Step 2）：每个数学步骤 → `pl.*` API 调用链
- 规范化 golden 状态
- 已冻结条目
- 尝试历史
- 阻塞列表

### Step 3：知识选择（产出 `KB_SELECTION.json`）

Stage 1 结束前，必须在该 class 的目录下写出 `KB_SELECTION.json`：
cases 未切分时是 `custom/<算子名称>/`（此时 `class_id` 写字面量 `"."`），
切分时是 `custom/<算子名称>/<class>/`。

这是 contract v2 的必选产物。planner 负责产出和做格式预检，
`pypto-pro-op-verifier` 在 `stage1-check` 中独立校验；verifier 未返回 PASS 时，
orchestrator 不得宣布 Stage 1 完成。校验项与目录约定见
[`pypto-pro-op-kb/CONTRACT.md`](../../pypto-pro-op-kb/CONTRACT.md)。

按 [`pypto-pro-op-kb/ROUTER.md`](../../pypto-pro-op-kb/ROUTER.md) 的「Routing by topology」一节选择，
数据源是 [`pypto-pro-op-kb/topology-map.json`](../../pypto-pro-op-kb/topology-map.json)。

**按计算形状路由，不按算子名路由。** 算子名不能迁移到新算子；拓扑可以。

```json
{
  "schema_version": 2,
  "op": "<算子名>",
  "class_id": "<class 子目录名>",
  "topology": "row-reduction",
  "properties": {
    "dtypes": ["float16"], "ranks": [2],
    "unaligned_shapes": true, "tail_blocks": true,
    "long_axis": false, "mixed_precision": true,
    "index_dtypes": [], "dynamic_dims": false
  },
  "optional_patterns": [
    {"path": "patterns/vec-row-reduce-broadcast.md",
     "sha256": "sha256:<该文件内容哈希>",
     "reason": "每行归约成标量再广播回该行，与本 class 的数据流一致"}
  ],
  "required_constraints": [
    {"path": "constraints/vec.md",
     "sha256": "sha256:<该文件内容哈希>",
     "reason": "row-reduction 拓扑要求使用 vector 约束"},
    {"path": "constraints/wrapper-boundary.md",
     "sha256": "sha256:<该文件内容哈希>",
     "reason": "所有 class 都必须遵守公开 callable 的单 kernel 边界"}
  ],
  "failure_signatures": [
    {"id": "bf16-cancellation", "reason": "本 class 存在相近量相减，需 fp32 累加"}
  ],
  "no_matching_pattern": false
}
```

硬性约束：

- `optional_patterns` **不设数量上限**，但只允许放入对当前 class 有明确、独立设计作用的
  `patterns/` 复用模式。每条候选必须同时满足：适用前提与当前 topology/properties 一致；
  `reason` 能指出具体数据流问题和预期设计影响；与已选 pattern 不重复。
- `required_constraints` 收集 topology、property、target gate 和
  `mandatory_constraints` 触发的**全部**约束，不限数量，不得截断。
- 两类中的每条都必须写明 `reason`：为什么**这个 class** 需要它。
- 以下候选必须丢弃，不得为了增加参考数量而选入：仅算子名相似；适用前提不成立；
  只提供通用背景而不能改变设计；与已选 pattern 作用重复；与官方 API 或必需约束冲突。
- 路径必须**相对于 KB 根**（源码树 `cannbot-skills/ops/pypto-pro-op-kb/`，安装后 `$CONFIG_ROOT/pypto-pro-op-kb`），
  即以 `patterns/` / `constraints/` / `examples/` / `references/` 等 KB 一级目录开头。
  下面这些都会被判为「路径不存在」而使该 class 不通过：
  - 机器绝对路径：任何以 `/` 开头的完整路径（含用户目录、挂载点等）
  - **带安装前缀的相对路径**：`.opencode/skills/<skill>/pypto-pro-op-kb/constraints/wrapper-boundary.md`
    ——你读文件时用的是这个路径，但**写进 JSON 的必须是 `constraints/wrapper-boundary.md`**。
    这是最常见的失败原因：把「我怎么打开它」当成了「它在契约里的名字」。
  - 凭印象拼出的路径：`pro_ops/vf_api/test_softmax_tile_group.py` 之类并不存在的条目。
- **每条路径必须真实存在**。写入前用 `ls` 逐条确认（相对 KB 根解析），
  不确认就写等于让该 class 失败。宁可 `no_matching_pattern: true`，也不要写一条猜的路径。
- 每条必须带 `sha256`，取所引用文件的内容哈希：校验器据此判定引用是否在选择之后被改动过。占位符（`sha256:...`）不算，会被判为 stale。
- 确实没有匹配 pattern 时（**以 `topology-map.json` 当前内容为准，不要凭本文举例判断**——
  该文件是唯一数据源，pattern 会随交付增补），
  置 `no_matching_pattern: true` 并留空 `optional_patterns`，但仍须保留全部
  `required_constraints`；**不要**为了填满而选一条勉强相关的 pattern。
- `properties` 必须来自真实事实（`cases.yaml` 存在时以其为准，否则以 Step 1.5 的契约裁定为准），不得臆测。
- `topology` 必须是 [`pypto-pro-op-kb/topology-map.json`](../../pypto-pro-op-kb/topology-map.json) 当前
  `topologies` 对象中的键，**不能为 null、不能自造，也不得在本 skill 中维护枚举副本**。
- `op`、`class_id`、`topology`、`properties` 四个字段**必填**；
  `optional_patterns`、`required_constraints`、`failure_signatures` 必须是**列表**
  （没有内容就写 `[]`，不能写 `null` 或对象）。文件必须是能被 `json.load` 读通的合法
  JSON——写完自己读一遍。

**收尾预检（Stage 1 子代理返回前必做）**：下面只负责尽早发现 JSON、字段和路径错误，
不产生 PASS verdict，也不能代替随后由 orchestrator 调度的 `stage1-check` verifier。

```bash
python3 -c "
import json, pathlib, sys
op = pathlib.Path('custom/<op>')
kb = pathlib.Path('<kb 根>')
sel = json.loads((op / 'KB_SELECTION.json').read_text())
bad = []
for key in ('schema_version', 'op', 'class_id', 'topology', 'properties',
            'optional_patterns', 'required_constraints', 'no_matching_pattern'):
    if key not in sel:
        bad.append(f'缺字段 {key}')
patterns = sel.get('optional_patterns') or []
constraints = sel.get('required_constraints') or []
if patterns and sel.get('no_matching_pattern'):
    bad.append('no_matching_pattern=true 时 optional_patterns 必须为空')
refs = patterns + constraints
for r in refs:
    path = r['path'] if isinstance(r, dict) else r
    if not (kb / path).is_file():
        bad.append(f'引用不存在：{path}')
print('OK' if not bad else 'FAIL: ' + '; '.join(bad))
sys.exit(0 if not bad else 1)
"
```

预检输出不是 `OK` 时就地修正；输出 `OK` 后把产物交给 verifier，只有 verifier PASS
才能进入 Stage 2。
