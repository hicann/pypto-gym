---
name: pypto-pro-op-plan
description: 编排 PyPTO-Pro Stage 1 规划。用于依次完成需求规格化、目标版本资料探索、kernel 合同补充、MEMORY.md 初始化和 KB_SELECTION.json 冻结；输入是已授权算子需求，输出是 Stage 2/3 可直接消费的一组规划产物。不要用于 golden、Module/tile 设计、kernel 实现或性能调优。
---

# PyPTO-Pro Stage 1 规划

本 skill 只定义 Stage 1 的顺序、交接和完成条件。需求理解与资料探索的具体方法分别由
`pypto-pro-intent-understand` 和 `pypto-pro-material-explore` 负责，不在这里重复。

## 输入与产物

输入：用户已授权的算子需求，以及 orchestrator 已装配的 `$PYPTO_DEVKIT_DIR`。

在 `custom/<op>/` 生成：

| 产物 | 所有者 | 下游用途 |
|------|--------|----------|
| `SPEC.md` | intent-understand | 数学语义、公开接口、P0 cases |
| `PRO_MATERIAL_INDEX.md` | material-explore | 本次目标版本资料目录 |
| `EXPLORE_REPORT.md` | material-explore | API 可行性、约束和证据 |
| `MEMORY.md` | plan | Stage 间的裁定摘要，不复制长篇规则 |
| `KB_SELECTION.json` | plan | 冻结当前 class 的 KB 路由结果 |

## 串行流程

### 1. 冻结需求语义

在同一 agent session 中加载 `pypto-pro-intent-understand`，生成并验证 `SPEC.md`。
不要 dispatch 子代理，也不要在本 skill 中重做其确认流程。

### 2. 创建 MEMORY 并补充 kernel 交接合同

语义冻结后立即创建或追加 `MEMORY.md`，先记录任务、确认状态与 `SPEC.md` 路径；后续每步
只追加新裁定、阻塞、尝试与产物指针，探索完成后再按第 4 步收敛，避免到末尾才补写历史。

在 `SPEC.md` 末尾增加 `## kernel 契约补充`，只记录后续阶段必须知道、但通用需求模板
不表达的边界：

| 字段 | Stage 1 处理 |
|------|--------------|
| 辅助张量语义 | 确认是公开输入、模型参数还是内部临时量；实现位置交 Stage 3 |
| cast 边界链 | 记录输入、累加、后处理、输出各段的语义 dtype |
| 累加/写回语义 | 确认覆盖写、跨块累加或原子累加的数学要求 |
| 目标设备 | 运行时或 build 配置已指定 target 时使用指定值；未指定时默认 A5，并在 SPEC/MEMORY 标注这是默认假设 |
| topology/tile/同步 | 标记“由 Stage 3 设计”，不得在 Stage 1 猜测 |

若该字段会改变数学语义或公开接口，必须回到 intent-understand；若只是硬件实现选择，留给
Stage 3，不问用户。

### 3. 探索目标版本资料

加载 `pypto-pro-material-explore`，以包含 kernel 交接合同的 `SPEC.md` 为输入，重新扫描资料
索引并生成 `PRO_MATERIAL_INDEX.md` 与 `EXPLORE_REPORT.md`。探索不能反向静默改变 SPEC；
若发现公式、接口或 P0 case 有问题，返回 intent-understand 修订并重新确认。

### 4. 收敛 MEMORY

整理持续追加的 `MEMORY.md`，只保留下游需要快速恢复的索引信息：

- 任务和确认状态摘要；
- kernel 契约补充的裁定及未决项；
- 三个 Stage 1 产物路径；
- 公式步骤到 API 候选的简短映射；
- 已冻结事实、阻塞项和尝试历史。

详细公式留在 SPEC，详细证据留在 EXPLORE_REPORT，KB 规则留在 KB；不要复制全文。

### 5. 冻结知识选择

阅读安装态 KB 根下的 `CONTRACT.md`、`ROUTER.md` 与 `topology-map.json`，为每个 class
逐 class 生成 `KB_SELECTION.json`：flat 布局落在 `custom/<op>/KB_SELECTION.json`，此时
`class_id` 必须为字面量 `"."`；split 布局逐一落在
`custom/<op>/<class>/KB_SELECTION.json`，`class_id` 必须等于该 class 目录名。安装态路径为
`$CANNBOT_CONFIG_ROOT/pypto-pro-op-kb/`；源码中
skill 链接使用 `../../pypto-pro-op-kb/` 是有意的安装布局。

执行规则：

1. 从公式的计算拓扑和已确认 properties 路由，禁止按算子名称猜选。
2. **允许命中零个或多个拓扑，不做唯一选择、不做覆盖**：融合算子同时符合
   `multi-phase-fusion` 与其组成部分（如 `cube-matmul`、`row-reduction`）时全部选入。
   若公式不符合任何已声明拓扑，必须如实记录 `topologies: []`，不得强行选择最接近的类别。
   `[]` 只表示已完成公式路由且没有当前键命中，不能表示未知、未分析或跳过；信息不足时必须
   继续补充分析依据，任何实际命中都不得遗漏。
   数组中的每个元素都必须是 `topology-map.json.topologies` 的当前键；不维护本地枚举副本。
3. 收集全部命中 `topologies` 的并集（空并集合法）、properties、已确认 target 和 mandatory
   触发的全部 constraints（去重合并，不得截断、不得只收其一）。拓扑数组为空时，后三类
   路由仍须正常执行。
4. optional pattern 候选来自全部命中拓扑与适用 property modifier 路由结果的并集；只保留
   适用前提成立且会产生独立、具体设计作用的条目，不限数量。
5. 无适用 pattern 时设置 `no_matching_pattern: true`，但不得删除 required constraints。
6. 每条引用使用 KB 根相对路径，记录 class-specific reason 与当前文件 SHA-256；不得记录
   安装前缀、绝对路径或占位哈希。
7. `properties` 只来自 SPEC、cases 或已确认环境事实。target 未指定时按默认 A5
   触发 `constraints/arch-a5.md`；已明确为非 A5 时不触发。

布局、`class_id` 与完整字段合同以 [KB CONTRACT](../pypto-pro-op-kb/CONTRACT.md) 为唯一规范，路由算法以
[KB ROUTER](../pypto-pro-op-kb/ROUTER.md) 和
[`topology-map.json`](../pypto-pro-op-kb/topology-map.json) 为唯一数据源。

### 6. 收尾自检

每个 class 在返回前执行下面的预检（将 `<selection>` 和 `<kb_root>` 替换为实际路径）。它只提前发现
JSON、必填字段、pattern 标记、路径和哈希错误，不判断 pattern 是否适用，不产生 verifier PASS。

```bash
python -c '
import hashlib, json, pathlib, sys
selection_path = pathlib.Path("<selection>")
kb = pathlib.Path("<kb_root>")
selection = json.loads(selection_path.read_text(encoding="utf-8"))
mapping = json.loads((kb / "topology-map.json").read_text(encoding="utf-8"))
bad = []
required = {"schema_version", "op", "class_id", "topologies", "properties",
            "optional_patterns", "required_constraints", "no_matching_pattern"}
bad += [f"missing field: {key}" for key in sorted(required - set(selection))]
if "topology" in selection:
    bad.append("legacy field topology is not allowed; use topologies")
if selection.get("schema_version") != mapping["contract"]["contract_version"]:
    bad.append("schema_version does not match topology-map contract")
topologies = selection.get("topologies")
if not isinstance(topologies, list):
    bad.append("topologies must be an array")
else:
    for topo in topologies:
        if not isinstance(topo, str):
            bad.append(f"topology must be a string: {topo!r}")
        elif topo not in mapping["topologies"]:
            bad.append(f"topology is not declared in topology-map: {topo}")
expected_class = "." if selection_path.parent.parent.name == "custom" else selection_path.parent.name
if selection.get("class_id") != expected_class:
    bad.append(f"class_id must be {expected_class}")
properties = selection.get("properties")
property_keys = set(mapping["contract"]["property_keys"])
if not isinstance(properties, dict):
    bad.append("properties must be an object")
elif set(properties) - property_keys:
    bad.append(f"unknown property keys: {sorted(set(properties) - property_keys)}")
patterns = selection.get("optional_patterns")
constraints = selection.get("required_constraints")
if not isinstance(patterns, list) or not isinstance(constraints, list):
    bad.append("optional_patterns and required_constraints must be lists")
else:
    if selection.get("no_matching_pattern") is not (not patterns):
        bad.append("no_matching_pattern is inconsistent")
    for refs, namespace in ((patterns, "patterns"), (constraints, "constraints")):
        namespace_root = (kb / namespace).resolve()
        for ref in refs:
            if not isinstance(ref, dict):
                bad.append("reference must be an object")
                continue
            if not isinstance(ref.get("reason"), str) or not ref["reason"].strip():
                bad.append("reference reason must be non-empty")
            path = ref.get("path")
            target = (kb / path).resolve() if isinstance(path, str) else None
            try:
                if not isinstance(path, str) or not path.startswith(namespace + "/"):
                    raise ValueError
                target.relative_to(namespace_root)
            except (TypeError, ValueError):
                bad.append(f"path escapes {namespace}/ namespace: {path}")
                continue
            if not target.is_file():
                bad.append(f"referenced file does not exist: {path}")
                continue
            expected = "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
            if ref.get("sha256") != expected:
                bad.append(f"stale sha256: {path}")

print("OK" if not bad else "FAIL: " + "; ".join(bad))
sys.exit(0 if not bad else 1)
'
```

对 flat 布局检查 `<op>/KB_SELECTION.json`；对 split 布局逐个检查
`<op>/<class>/KB_SELECTION.json`。输出非 `OK` 时就地修正，然后再交 orchestrator 调度 `stage1-check`。

## 完成条件

- 五个 Stage 1 产物均存在；
- SPEC 校验通过且没有被探索阶段静默改写；
- INDEX 的 §A/§B/§C 与本次缓存一致；
- EXPLORE_REPORT 没有未解决的 `unsupported` 阻断；
- MEMORY 只记录摘要和指针，含 kernel 合同裁定；
- KB_SELECTION 的 `schema_version` 等于 `topology-map.json` 中当前的
  `contract.contract_version`，所有路径和哈希真实可复核；
- 本 agent 只做预检，不得自称 verifier PASS。

返回上述产物路径、未决风险和预检结果，交 orchestrator 调度 `stage1-check`。
