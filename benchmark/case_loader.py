#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""KernelBench 用例加载器 (上游 PyTorch 扁平布局).

内置数据集位于 ``pypto-gym/benchmark/KernelBench/``，内容来自
PyPTO 支持的 KernelBench 仓库并保留上游原始目录/编号。布局为
``KernelBench/<level>/{N}_{name}.py`` 的扁平结构, 每个 .py 内含一个
``class Model(nn.Module)`` + ``get_inputs()`` + ``get_init_inputs()``.

负责把一个 KernelBench 用例 .py 解析成两个产物:

1. ``task_desc``: 原始源码字符串, 直接喂给 ``KernelVerifier``
   (作为 ``framework_code`` 参数).
2. ``REQUIRE.md``: 自然语言 + 半结构化的算子需求文档, 作为 pypto
   Stage 1 的用户需求输入.

设计要点:
- 优先 AST 解析 (无副作用); 形状/dtype 推断走"在子进程中真实执行
  ``get_inputs()`` 并打印 shape/dtype" 以避免 torch / numpy 表达式自行求值
  的复杂度, 同时不污染主进程.
- 对解析失败的字段做 best-effort fallback: 即便没拿到 shape, REQUIRE.md 仍可
  落地, 让 pypto 工作流自己按源码推断.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from benchmark.constants import NPU_SMI_INFO_TIMEOUT_SEC


logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# 数据模型
# ────────────────────────────────────────────────────────────

@dataclass
class TensorSpec:
    """单个输入/输出张量的规格."""
    name: str = ""
    shape: Optional[List[int]] = None
    dtype: str = ""


@dataclass
class CaseSpec:
    """KernelBench 用例派生的结构化规格."""
    op_name: str
    case_id: str                           # 上游文件 stem, 例如 "19_relu"
    source_file: str                       # 绝对路径
    task_desc: str                         # 原始源码 (KernelBench 风格)
    level: str = ""                        # KernelBench level, 例如 level1 / custom
    framework_module: str = "torch"        # 上游 KernelBench 一律 torch; 探针时若 import 不同, 会被覆盖
    init_source: str = ""                  # Model.__init__ 的源码片段
    forward_source: str = ""               # Model.forward / __call__ 的源码片段
    init_args_repr: str = "[]"             # get_init_inputs() 的 repr
    inputs: List[TensorSpec] = field(default_factory=list)
    outputs: List[TensorSpec] = field(default_factory=list)
    supported_dtypes: List[str] = field(default_factory=lambda: ["float32"])
    p0_shapes: List[List[int]] = field(default_factory=list)
    tolerance: Dict[str, float] = field(
        default_factory=lambda: {"rtol": 1e-3, "atol": 1e-3}
    )
    dynamic_axis: Optional[List[str]] = None
    formula: str = ""
    p1_shapes: Optional[List] = None       # 来自 case 顶层 CASES 全局变量, 解析后的泛化用例列表


# ────────────────────────────────────────────────────────────
# AST 解析
# ────────────────────────────────────────────────────────────

def _detect_framework(tree: ast.Module) -> str:
    """根据 import 语句推断框架; 默认 numpy."""
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch":
                    return "torch"
                if alias.name == "mindspore":
                    return "mindspore"
                if alias.name == "numpy":
                    return "numpy"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("torch"):
                return "torch"
            if node.module and node.module.startswith("mindspore"):
                return "mindspore"
    return "numpy"


def _extract_model_method_source(tree: ast.Module, source: str,
                                 *method_names: str) -> str:
    """提取 ``Model`` 类里指定方法的源码片段."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in method_names:
                    return ast.get_source_segment(source, item) or ""
    return ""


def _has_kernelbench_layout(tree: ast.Module) -> List[str]:
    """返回缺失的关键组件列表; 空列表表示合规."""
    has_model = False
    has_inputs = False
    has_init = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            has_model = True
        if isinstance(node, ast.FunctionDef):
            if node.name == "get_inputs":
                has_inputs = True
            elif node.name == "get_init_inputs":
                has_init = True
    missing: List[str] = []
    if not has_model:
        missing.append("class Model")
    if not has_inputs:
        missing.append("def get_inputs")
    if not has_init:
        missing.append("def get_init_inputs")
    return missing


def _extract_new_interface_globals(tree: ast.Module) -> tuple[str, Optional[List[str]], Optional[List]]:
    """提取新增 KernelBench case 顶层接口: ``FORMULA`` / ``DYNAMIC_AXIS`` / ``CASES``.

    旧 case 没有这些全局变量时保持空值, REQUIRE.md 渲染时不会输出对应字段。
    ``CASES`` 的字符串内容会被解析并校验为结构化列表后写入 front-matter 的 ``p1_shapes`` 字段。
    """
    formula = ""
    dynamic_axis: Optional[List[str]] = None
    cases: Optional[List] = None
    for node in tree.body:
        targets: List[ast.expr]
        value_node: Optional[ast.expr]
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value
        else:
            continue
        if value_node is None:
            continue

        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if not names:
            continue

        try:
            value = ast.literal_eval(value_node)
        except (ValueError, SyntaxError):
            continue

        if "FORMULA" in names and isinstance(value, str):
            formula = value.strip()
        if "DYNAMIC_AXIS" in names and isinstance(value, (list, tuple)):
            axis = [str(item) for item in value]
            if axis:
                dynamic_axis = axis
        if "CASES" in names and isinstance(value, str):
            parsed = _parse_cases_string(value)
            if parsed is not None:
                cases = parsed
    return formula, dynamic_axis, cases


def _parse_cases_string(raw: str) -> Optional[List]:
    """将 CASES 全局变量的字符串值解析为结构化列表.

    支持两种写法 (零新增依赖, 仅用 ``json.loads``):

    - JSON flow style: ``"[[[2, 3], [3]], [[4, 5], [5]]]"``
    - 多行块写法 (每行一个 JSON case)::

        CASES = \"\"\"
        - [[2, 3], [3]]
        - [[4, 5], [5]]
        \"\"\"

    多行块会先去掉 ``-`` 前缀再拼成 JSON 数组解析;
    解析结果不符合 ``[case, ...]`` / ``case=[shape, ...]`` / ``shape=[dim, ...]``
    结构时返回 ``None``.
    """
    text = textwrap.dedent(raw).strip()
    if not text:
        return None
    # 先尝试直接 JSON 解析 (覆盖 flow style)
    try:
        result = _validate_cases_structure(json.loads(text))
        if result is not None:
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    if not re.search(r"^\s*-\s+", text, flags=re.MULTILINE):
        logger.warning("CASES 字符串无法解析为合法 p1_shapes, 忽略: %.200s", text)
        return None
    # 预处理多行块写法: 去掉 "-" 前缀, 补齐成 JSON 数组
    stripped = re.sub(r"^\s*-\s+", "", text, flags=re.MULTILINE)
    stripped = re.sub(r"\]\s*\n\s*\[", "], [", stripped)
    try:
        result = _validate_cases_structure(json.loads(f"[{stripped}]"))
        if result is not None:
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    logger.warning("CASES 字符串无法解析为合法 p1_shapes, 忽略: %.200s", text)
    return None


def _validate_cases_structure(
    cases: object,
    expected_input_count: Optional[int] = None,
) -> Optional[List]:
    """校验 p1_shapes 结构: 外层 case 列表, 每个 case 含每个输入的 shape."""
    if not isinstance(cases, list) or not cases:
        return None
    for case in cases:
        if not isinstance(case, list) or not case:
            return None
        if expected_input_count is not None and len(case) != expected_input_count:
            return None
        for shape in case:
            if not isinstance(shape, list):
                return None
            for dim in shape:
                if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
                    return None
    return cases


# ────────────────────────────────────────────────────────────
# 子进程探针: 跑 get_inputs() / get_init_inputs() 拿到 shape & dtype
# ────────────────────────────────────────────────────────────

_PROBE_TEMPLATE = r"""
import importlib.util, json, sys
PROBE_OUTPUTS = {probe_outputs!r}
spec = importlib.util.spec_from_file_location("kb_case", {path!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def _shape_dtype(obj):
    shape = list(getattr(obj, "shape", ()))
    shape = [int(s) for s in shape]
    dtype = str(getattr(obj, "dtype", type(obj).__name__))
    return shape, dtype

def _flatten_outputs(obj):
    if isinstance(obj, dict):
        return list(obj.items())
    if isinstance(obj, (list, tuple)):
        return list(enumerate(obj))
    return [(0, obj)]

inputs_info = []
inputs = []
try:
    inputs = mod.get_inputs()
    for i, t in enumerate(inputs):
        shp, dt = _shape_dtype(t)
        inputs_info.append({{"name": f"x{{i}}", "shape": shp, "dtype": dt}})
except Exception as e:
    inputs_info.append({{"error": str(e)}})

init_repr = "[]"
init_args = []
try:
    init_args = mod.get_init_inputs()
    init_repr = repr(init_args)
except Exception as e:
    init_repr = f"<unavailable: {{e}}>"

outputs_info = []
try:
    if PROBE_OUTPUTS and inputs_info and "error" not in inputs_info[0]:
        model = mod.Model(*init_args)
        outputs = model(*inputs)
        for key, t in _flatten_outputs(outputs):
            shp, dt = _shape_dtype(t)
            name = f"y{{key}}" if isinstance(key, int) else str(key)
            outputs_info.append({{"name": name, "shape": shp, "dtype": dt}})
except Exception as e:
    outputs_info.append({{"error": str(e)}})

print("__PROBE_RESULT__")
print(json.dumps({{"inputs": inputs_info, "outputs": outputs_info, "init_args_repr": init_repr}}))
"""


def _parse_idle_chip_ids(npu_smi_output: str) -> List[str]:
    """从 ``npu-smi info`` 输出中解析当前空闲 chip id。"""
    all_chips: set[int] = set()
    used_chips: set[int] = set()
    proc_section = False
    current_npu: Optional[int] = None
    mode = 1

    for raw_line in npu_smi_output.splitlines():
        line = raw_line.strip()
        if line.startswith("| NPU     Chip"):
            proc_section = True
            continue
        fields = [item for item in line.split() if item != "|"]
        if not fields or not fields[0].isdigit():
            continue

        if not proc_section:
            if len(fields) >= 2 and not fields[1].isdigit():
                current_npu = int(fields[0])
                all_chips.add(current_npu)
                continue
            if len(fields) >= 2 and fields[1].isdigit():
                mode = 2
                local_chip = int(fields[1])
                all_chips.add((current_npu or 0) * 2 + local_chip)
            continue

        if len(fields) >= 3 and fields[1].isdigit() and fields[2].isdigit():
            npu_id = int(fields[0])
            local_chip = int(fields[1])
            global_chip = npu_id * 2 + local_chip if mode == 2 else npu_id
            used_chips.add(global_chip)

    return [str(chip) for chip in sorted(all_chips) if chip not in used_chips]


def _list_idle_chip_ids() -> List[str]:
    """返回当前空闲 NPU chip id 列表；探测失败时返回空列表。"""
    try:
        proc = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=NPU_SMI_INFO_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    return _parse_idle_chip_ids(proc.stdout)


def _run_probe_subprocess(
    case_path: Path,
    timeout_sec: int,
    probe_outputs: bool,
    chip_id: Optional[str] = None,
) -> tuple[List[TensorSpec], List[TensorSpec], str]:
    """执行一次 probe 子进程；失败时返回空结果。"""
    script = _PROBE_TEMPLATE.format(path=str(case_path), probe_outputs=probe_outputs)
    env = os.environ.copy()
    if chip_id is not None:
        env["TILE_FWK_DEVICE_ID"] = str(chip_id)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [], [], "[]"

    if proc.returncode != 0:
        return [], [], "[]"

    marker = "__PROBE_RESULT__"
    if marker not in proc.stdout:
        return [], [], "[]"
    payload = proc.stdout.split(marker, 1)[1].strip()
    try:
        data = json.loads(payload.splitlines()[0])
    except (ValueError, IndexError):
        return [], [], "[]"

    def _to_specs(entries: list[dict]) -> List[TensorSpec]:
        specs: List[TensorSpec] = []
        for entry in entries:
            if "error" in entry:
                continue
            specs.append(TensorSpec(
                name=entry.get("name", ""),
                shape=entry.get("shape"),
                dtype=str(entry.get("dtype", "")),
            ))
        return specs

    inputs = _to_specs(data.get("inputs", []))
    outputs = _to_specs(data.get("outputs", []))
    return inputs, outputs, str(data.get("init_args_repr", "[]"))


def _probe_io_specs(
    case_path: Path,
    timeout_sec: int = 30,
    max_output_attempts: int = 3,
    output_probe_device_id: Optional[str] = None,
    allow_find_free: bool = True,
) -> tuple[List[TensorSpec], List[TensorSpec], str]:
    """探测输入和输出规格。

    输入/init 参数探针不绑定设备；输出规格通过真实执行 ``Model.forward`` 获取。
    若给定 ``output_probe_device_id``，仅在该设备上探测输出，不调用空闲卡发现。
    否则在 ``allow_find_free`` 为真时，forward 前从 ``npu-smi info`` 解析的空闲 chip 中选卡，
    失败后换其他空闲 chip 重试，最多 ``max_output_attempts`` 次。
    ``allow_find_free`` 为假且未指定显式设备时，跳过输出探针（pool 模式由外部固定设备）。
    所有失败都降级为空输出规格，不阻断 REQUIRE.md 生成。
    """
    inputs, _, init_repr = _run_probe_subprocess(
        case_path,
        timeout_sec=timeout_sec,
        probe_outputs=False,
    )

    if not inputs or max_output_attempts <= 0:
        return inputs, [], init_repr

    if output_probe_device_id is not None:
        chip = str(output_probe_device_id).strip()
        if chip:
            _, outputs, _ = _run_probe_subprocess(
                case_path,
                timeout_sec=timeout_sec,
                probe_outputs=True,
                chip_id=chip,
            )
            return inputs, outputs, init_repr

    if not allow_find_free:
        return inputs, [], init_repr

    attempted: set[str] = set()
    for _ in range(max_output_attempts):
        idle_ids = [chip for chip in _list_idle_chip_ids() if chip not in attempted]
        if not idle_ids:
            break
        chip_id = idle_ids[0]
        attempted.add(chip_id)
        _, outputs, _ = _run_probe_subprocess(
            case_path,
            timeout_sec=timeout_sec,
            probe_outputs=True,
            chip_id=chip_id,
        )
        if outputs:
            return inputs, outputs, init_repr

    return inputs, [], init_repr


def _probe_inputs(case_path: Path, timeout_sec: int = 30) -> tuple[List[TensorSpec], str]:
    """兼容旧调用方：仅返回输入规格和 init 参数。"""
    inputs, _, init_repr = _run_probe_subprocess(
        case_path,
        timeout_sec=timeout_sec,
        probe_outputs=False,
    )
    return inputs, init_repr


# ────────────────────────────────────────────────────────────
# op 名规范化
# ────────────────────────────────────────────────────────────

_OP_NAME_RE = re.compile(r"\W+")


def derive_op_name(case_id: str) -> str:
    """从 ``19_relu`` / ``1_square_matrix_multiplication_`` 等 case id 推一个合法 Python 标识符.

    - 去掉前缀数字 + 下划线 (``19_relu`` → ``relu``)
    - 非法字符替换为 ``_``
    - 收尾下划线裁掉
    - 若以数字开头则前缀 ``op_``
    """
    s = case_id.strip()
    s = re.sub(r"^\d+_", "", s)
    s = _OP_NAME_RE.sub("_", s).strip("_")
    if not s:
        s = "op"
    if s[0].isdigit():
        s = f"op_{s}"
    return s


# ────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────

def load_case(case_path: Path, op_name: Optional[str] = None,
              case_id: Optional[str] = None,
              probe_timeout_sec: int = 30,
              max_output_probe_attempts: int = 3,
              output_probe_device_id: Optional[str] = None,
              allow_find_free: bool = True) -> CaseSpec:
    """加载并解析一个 KernelBench 用例 (上游 PyTorch 扁平布局).

    Args:
        case_path: 用例 ``.py`` 绝对/相对路径; 上游 KernelBench 中即
            ``KernelBench/<level>/{N}_{name}.py``.
        op_name: 可选, 指定算子名; 缺省时按 ``case_id`` 推导.
        case_id: 可选, 用例标识; 缺省时取 ``case_path.stem``
            (上游扁平布局下文件名即标识).
        probe_timeout_sec: 子进程执行 ``get_inputs()`` 的超时.
        max_output_probe_attempts: 执行 ``Model.forward`` 探测输出规格的最大选卡
            尝试次数。每次尝试都从空闲 chip 列表中选择一张未尝试过的卡
            （未指定 ``output_probe_device_id`` 且 ``allow_find_free`` 为真时生效）。
        output_probe_device_id: 显式指定输出探针使用的设备 id（``TILE_FWK_DEVICE_ID``），
            pool 模式传入配置卡号；指定后不再做空闲卡发现。
        allow_find_free: 为假时禁止通过 ``_list_idle_chip_ids`` 扫空闲卡做输出探针；
            需配合 ``output_probe_device_id`` 或由调用方接受空输出规格。

    Raises:
        FileNotFoundError: 文件不存在.
        ValueError: 文件不符合 KernelBench 格式 (缺 ``Model`` /
            ``get_inputs`` / ``get_init_inputs``).
    """
    case_path = case_path.resolve()
    if not case_path.exists():
        raise FileNotFoundError(f"KernelBench case not found: {case_path}")

    source = case_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    missing = _has_kernelbench_layout(tree)
    if missing:
        raise ValueError(
            f"{case_path} 不符合 KernelBench 格式, 缺少: {', '.join(missing)}"
        )

    if not case_id:
        case_id = case_path.stem
    if not op_name:
        op_name = derive_op_name(case_id)

    framework = _detect_framework(tree)
    init_src = _extract_model_method_source(tree, source, "__init__")
    forward_src = _extract_model_method_source(tree, source, "__call__", "forward")
    formula, dynamic_axis, p1_shapes = _extract_new_interface_globals(tree)
    inputs, outputs, init_repr = _probe_io_specs(
        case_path,
        timeout_sec=probe_timeout_sec,
        max_output_attempts=max_output_probe_attempts,
        output_probe_device_id=output_probe_device_id,
        allow_find_free=allow_find_free,
    )
    if p1_shapes is not None and inputs:
        expected_input_count = len(inputs)
        validated_p1_shapes = _validate_cases_structure(p1_shapes, expected_input_count)
        if validated_p1_shapes is None:
            logger.warning(
                "CASES 结构与探针输入数量不匹配, 忽略 p1_shapes: expected_inputs=%s, cases=%.200s",
                expected_input_count,
                json.dumps(p1_shapes, ensure_ascii=False),
            )
        p1_shapes = validated_p1_shapes
    supported_dtypes, p0_shapes, tolerance = _derive_front_matter_fields(inputs)

    return CaseSpec(
        op_name=op_name,
        case_id=case_id,
        source_file=str(case_path),
        task_desc=source,
        level=case_path.parent.name,
        framework_module=framework,
        init_source=init_src,
        forward_source=forward_src,
        init_args_repr=init_repr,
        inputs=inputs,
        outputs=outputs,
        supported_dtypes=supported_dtypes,
        p0_shapes=p0_shapes,
        tolerance=tolerance,
        dynamic_axis=dynamic_axis,
        formula=formula,
        p1_shapes=p1_shapes,
    )


# ────────────────────────────────────────────────────────────
# REQUIRE.md 渲染
# ────────────────────────────────────────────────────────────

_REQUIRE_TEMPLATE = """\
---
schema_version: 1
op_name: {op_name}
supported_dtypes: {supported_dtypes_json}
p0_shapes: {p0_shapes_json}
{p1_shapes_front_matter}tolerance: {tolerance_json}
{dynamic_axis_front_matter}---

# {op_name} 算子需求规格 (派生自上游 KernelBench)

> 本 REQUIRE 由 ``benchmark.case_loader`` 自动生成, 用作
> ``pypto-op-orchestrator`` Stage 1 的用户需求输入.
>
> 数据集来源: 内置 benchmark/KernelBench
> (github.com/zwx2238/KernelBench @ 5bb8dda, 保留上游原始编号)

## 元数据

- **算子名 (op_name)**: `{op_name}`
- **来源用例 (case_id)**: `{case_id}`
- **来源 level**: `{level}`
- **来源文件**: `{source_file}`
- **参考框架 (framework)**: `{framework}`

{formula_section}
## 输入输出规格

**输入规格**:
{inputs_section}

**输出规格**:
{outputs_section}

## Shape 约束

{shape_constraints_section}

## 数据类型支持

{dtype_section}

## 初始化参数 (get_init_inputs)

```python
init_args = {init_args_repr}
```

## KernelBench 调用约定

```python
model = Model(*get_init_inputs())
outputs = model(*get_inputs())
```

- `get_init_inputs()` 与 `get_inputs()` 是两段不同的调用面.
- 若 PyPTO wrapper 需要消费 init 参数, 应由 `ModelNew.__init__` 保存, 并在
  `ModelNew.forward()` 内部按正确顺序转发给 wrapper.
- 禁止要求下游验证器把 init 参数和 forward 输入拍平成一个外部调用接口.

## 构造逻辑 (Model.__init__ 参考实现)

```python
{init_source}
```

## 计算逻辑 (Model 参考实现)

```python
{forward_source}
```

## 精度要求

- 默认: ``rtol={rtol}, atol={atol}`` (由输入 dtype 自动推导; FP16/BF16 使用更宽容差).
- 验证通过条件: ``test_{op_name}.py`` 输出 ``[PRECISION_PASS]``.
- 桥接层 KernelVerifier 端按 ``mode=correctness`` 复测.

## 算子开发约束

1. 必须导出 ``{op_name}_wrapper(...) -> torch.Tensor``.
2. 对外桥接后的调用约定必须与 KernelBench 一致:
   ``ModelNew(*get_init_inputs()).forward(*get_inputs())`` 必须可用.
3. 若存在 init 参数, 不得要求外部把 init_args 和 forward inputs 错误拍平后再调用 wrapper.
4. 本算子由外部 KernelBench 桥接消费; 调用方会在 prompt 中要求额外产出
   ``{op_name}_pypto_impl.py`` (含 ``ModelNew`` 类), 文件契约以 prompt 为准,
   本 REQUIRE 不重复声明.
5. golden / impl / test 三文件分离.
6. 输入/输出 dtype 必须与原 KernelBench 用例一致.

## 原始 KernelBench 任务描述 (task_desc)

```python
{task_desc}
```
"""


def _format_shape(shape: Optional[List[int]]) -> str:
    if shape is None:
        return "unknown"
    return "x".join(str(s) for s in shape) or "scalar"


def _render_tensor_section(specs: List[TensorSpec], kind: str) -> str:
    if not specs:
        return (f"> {kind} shape/dtype 探针执行失败 (子进程超时、环境缺依赖或 forward 不可执行). "
                "请由 pypto-intent-understand 从下方 task_desc 自行推断.\n")
    lines = ["| # | name | shape | dtype |", "|---|------|-------|-------|"]
    for i, spec in enumerate(specs):
        lines.append(f"| {i} | `{spec.name}` | `{_format_shape(spec.shape)}` | `{spec.dtype}` |")
    return "\n".join(lines) + "\n"


def _render_shape_constraints_section(
    inputs: List[TensorSpec],
    outputs: List[TensorSpec],
    dynamic_axis: Optional[List[str]],
) -> str:
    lines = ["- **来源**: `get_inputs()` 与 `Model(*get_init_inputs()).forward(*get_inputs())` 探针结果。"]
    if inputs:
        lines.append("- **P0 输入 Shape**:")
        for spec in inputs:
            lines.append(f"  - `{spec.name}`: `{_format_shape(spec.shape)}`")
    else:
        lines.append("- **P0 输入 Shape**: 未探测到。")
    if outputs:
        lines.append("- **P0 输出 Shape**:")
        for spec in outputs:
            lines.append(f"  - `{spec.name}`: `{_format_shape(spec.shape)}`")
    else:
        lines.append("- **P0 输出 Shape**: 未探测到。")
    if dynamic_axis:
        lines.append(f"- **动态轴**: `{', '.join(dynamic_axis)}` (来自 case 顶层 `DYNAMIC_AXIS`)。")
    else:
        lines.append("- **动态轴**: 未声明 (case 顶层未提供 `DYNAMIC_AXIS`)。")
    return "\n".join(lines) + "\n"


def _render_dtype_section(supported_dtypes: List[str], tolerance: Dict[str, float]) -> str:
    lines = ["| Dtype | 支持 | atol | rtol | 备注 |", "|-------|------|------|------|------|"]
    for dtype in supported_dtypes:
        lines.append(
            f"| {dtype} | 是 | {tolerance['atol']} | {tolerance['rtol']} | 自动探测 |"
        )
    return "\n".join(lines) + "\n"


def _render_dynamic_axis_front_matter(dynamic_axis: Optional[List[str]]) -> str:
    if not dynamic_axis:
        return ""
    return f"dynamic_axis: {json.dumps(dynamic_axis, ensure_ascii=False)}\n"


def _render_p1_shapes_front_matter(p1_shapes: Optional[List]) -> str:
    """渲染 p1_shapes front-matter 行; 单行 JSON 序列化以保证 YAML 合法性."""
    if not p1_shapes:
        return ""
    return f"p1_shapes: {json.dumps(p1_shapes, ensure_ascii=False)}\n"


def _render_formula_section(formula: str) -> str:
    formula = textwrap.dedent(formula or "").strip()
    if not formula:
        return ""
    return (
        "### 1.3 数学公式\n\n"
        "```text\n"
        f"{formula}\n"
        "```\n\n"
    )


def _normalize_dtype(dtype: str) -> str:
    """把 ``torch.float32`` / ``numpy.float32`` 等归一成 front matter dtype."""
    value = str(dtype or "").strip()
    if not value:
        return ""
    value = value.replace("torch.", "")
    value = value.replace("mindspore.", "")
    value = value.replace("numpy.", "")
    match = re.search(
        r"float(?:16|32|64)|bfloat16|int(?:8|16|32|64)|uint8|bool",
        value,
    )
    return match.group(0) if match else value


def _derive_front_matter_fields(
    inputs: List[TensorSpec],
) -> tuple[List[str], List[List[int]], Dict[str, float]]:
    """从探针输入规格派生 REQUIRE.md YAML front matter 字段."""
    supported_dtypes: List[str] = []
    seen_dtypes = set()
    for spec in inputs:
        dtype = _normalize_dtype(spec.dtype)
        if dtype and dtype not in seen_dtypes:
            seen_dtypes.add(dtype)
            supported_dtypes.append(dtype)
    if not supported_dtypes:
        supported_dtypes = ["float32"]

    p0_shapes = [
        [int(dim) for dim in spec.shape]
        for spec in inputs
        if spec.shape
    ]

    has_low_precision = any(dt in ("float16", "bfloat16") for dt in supported_dtypes)
    tolerance = (
        {"rtol": 4e-3, "atol": 4e-3}
        if has_low_precision else
        {"rtol": 1e-3, "atol": 1e-3}
    )
    return supported_dtypes, p0_shapes, tolerance


def render_require_md(case: CaseSpec) -> str:
    """把 ``CaseSpec`` 渲染成 REQUIRE.md 文本."""
    supported_dtypes, p0_shapes, tolerance = _derive_front_matter_fields(
        case.inputs
    )
    return _REQUIRE_TEMPLATE.format(
        op_name=case.op_name,
        supported_dtypes_json=json.dumps(supported_dtypes, ensure_ascii=False),
        p0_shapes_json=json.dumps(p0_shapes, ensure_ascii=False),
        p1_shapes_front_matter=_render_p1_shapes_front_matter(case.p1_shapes),
        tolerance_json=json.dumps(tolerance, ensure_ascii=False),
        dynamic_axis_front_matter=_render_dynamic_axis_front_matter(case.dynamic_axis),
        case_id=case.case_id,
        level=case.level,
        source_file=case.source_file,
        framework=case.framework_module,
        formula_section=_render_formula_section(case.formula),
        inputs_section=_render_tensor_section(case.inputs, "输入"),
        outputs_section=_render_tensor_section(case.outputs, "输出"),
        shape_constraints_section=_render_shape_constraints_section(
            case.inputs, case.outputs, case.dynamic_axis
        ),
        dtype_section=_render_dtype_section(supported_dtypes, tolerance),
        rtol=tolerance["rtol"],
        atol=tolerance["atol"],
        init_args_repr=case.init_args_repr,
        init_source=textwrap.dedent(case.init_source).strip() or "# (未提取到 __init__ 源码)",
        forward_source=textwrap.dedent(case.forward_source).strip() or "# (未提取到 forward 源码)",
        task_desc=case.task_desc.strip(),
    )


def write_require(case: CaseSpec, workdir: Path) -> Path:
    """把 REQUIRE.md 写到 ``workdir/{op}/REQUIRE.md`` 并返回路径.

    若文件已存在且内容一致, 不重写以利于断点续跑.
    """
    op_dir = workdir / case.op_name
    op_dir.mkdir(parents=True, exist_ok=True)
    require_path = op_dir / "REQUIRE.md"
    new_content = render_require_md(case)
    if require_path.exists() and require_path.read_text(encoding="utf-8") == new_content:
        return require_path
    require_path.write_text(new_content, encoding="utf-8")
    return require_path


def write_task_desc(case: CaseSpec, workdir: Path) -> Path:
    """把原始 KernelBench task_desc 缓存到 ``workdir/{op}/task_desc.py``.

    给 ``verifier_runner`` 直接读取使用, 避免再次 IO 原 KernelBench 路径.
    """
    op_dir = workdir / case.op_name
    op_dir.mkdir(parents=True, exist_ok=True)
    out = op_dir / "task_desc.py"
    out.write_text(case.task_desc, encoding="utf-8")
    return out


# ────────────────────────────────────────────────────────────
# CLI (调试用)
# ────────────────────────────────────────────────────────────

def _main_cli() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Inspect a KernelBench case → CaseSpec / REQUIRE.md")
    parser.add_argument("case_file", type=Path, help="Path to KernelBench .py")
    parser.add_argument("--op-name", type=str, default=None)
    parser.add_argument("--write", type=Path, default=None,
                        help="写出 REQUIRE.md 到此目录的 {op}/REQUIRE.md")
    parser.add_argument("--probe-timeout", type=int, default=30)
    args = parser.parse_args()

    case = load_case(args.case_file, op_name=args.op_name,
                     probe_timeout_sec=args.probe_timeout)
    payload = asdict(case)
    payload["task_desc"] = f"<{len(case.task_desc)} chars>"
    sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    if args.write:
        require_path = write_require(case, args.write)
        td_path = write_task_desc(case, args.write)
        logger.info("Wrote: %s", require_path)
        logger.info("Wrote: %s", td_path)
    return 0


if __name__ == "__main__":
    sys.exit(_main_cli())
