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
"""Tests for KernelBench REQUIRE.md YAML front matter generation."""

from __future__ import annotations

import json
import textwrap

from benchmark import case_loader
from benchmark.case_loader import (
    CaseSpec,
    TensorSpec,
    load_case,
    render_require_md,
    write_require,
)


def _parse_front_matter(markdown: str) -> dict:
    assert markdown.startswith("---\n")
    block = markdown.split("---", 2)[1].strip()
    out = {}
    for line in block.splitlines():
        key, value = line.split(": ", 1)
        out[key] = value
    out["schema_version"] = int(out["schema_version"])
    out["supported_dtypes"] = json.loads(out["supported_dtypes"])
    out["p0_shapes"] = json.loads(out["p0_shapes"])
    out["tolerance"] = json.loads(out["tolerance"])
    if "dynamic_axis" in out:
        out["dynamic_axis"] = json.loads(out["dynamic_axis"])
    if "p1_shapes" in out:
        out["p1_shapes"] = json.loads(out["p1_shapes"])
    return out


def test_render_require_front_matter_from_tensor_specs() -> None:
    case = CaseSpec(
        op_name="Foo",
        case_id="1_Foo",
        source_file="/tmp/1_Foo.py",
        task_desc="def get_inputs(): return []",
        inputs=[
            TensorSpec(name="x0", shape=[2, 3], dtype="torch.float16"),
            TensorSpec(name="x1", shape=[4], dtype="float16"),
            TensorSpec(name="x2", shape=None, dtype="torch.float32"),
        ],
        outputs=[TensorSpec(name="y0", shape=[2, 3], dtype="torch.float16")],
    )

    markdown = render_require_md(case)
    front_matter = _parse_front_matter(markdown)

    assert front_matter["schema_version"] == 1
    assert front_matter["op_name"] == "Foo"
    assert front_matter["supported_dtypes"] == ["float16", "float32"]
    assert front_matter["p0_shapes"] == [[2, 3], [4]]
    assert front_matter["tolerance"] == {"rtol": 0.004, "atol": 0.004}
    assert "## 输入输出规格" in markdown
    assert "**输出规格**" in markdown
    assert "`y0` | `2x3` | `torch.float16`" in markdown


def test_render_require_front_matter_probe_fallback() -> None:
    case = CaseSpec(
        op_name="Fallback",
        case_id="2_Fallback",
        source_file="/tmp/2_Fallback.py",
        task_desc="def get_inputs(): return []",
    )

    front_matter = _parse_front_matter(render_require_md(case))

    assert front_matter["supported_dtypes"] == ["float32"]
    assert front_matter["p0_shapes"] == []
    assert front_matter["tolerance"] == {"rtol": 0.001, "atol": 0.001}
    assert "dynamic_axis" not in front_matter
    assert "### 1.3 数学公式" not in render_require_md(case)


def test_render_require_new_interface_globals_when_present() -> None:
    case = CaseSpec(
        op_name="DynamicAxisAdd",
        case_id="101_DynamicAxisAdd",
        source_file="/tmp/101_DynamicAxisAdd.py",
        task_desc="FORMULA = 'out = x + bias'",
        inputs=[TensorSpec(name="x0", shape=[2, 4, 8], dtype="float32")],
        dynamic_axis=["B", "S"],
        formula="out[b, s, d] = x[b, s, d] + bias[d]",
    )

    markdown = render_require_md(case)
    front_matter = _parse_front_matter(markdown)

    assert front_matter["dynamic_axis"] == ["B", "S"]
    assert "### 1.3 数学公式" in markdown
    assert "out[b, s, d] = x[b, s, d] + bias[d]" in markdown


def test_parse_idle_chip_ids_without_shell_helper() -> None:
    npu_smi_output = textwrap.dedent(
        """
        | 0 910C |
        | 0 0 |
        | 0 1 |
        | 1 910C |
        | 1 0 |
        | 1 1 |
        | NPU     Chip     PID |
        | 0       1        222 |
        | 1       0        333 |
        """
    )

    assert case_loader._parse_idle_chip_ids(npu_smi_output) == ["0", "3"]


def test_load_case_populates_front_matter_fields_and_write_require(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(case_loader, "_list_idle_chip_ids", lambda: ["0"])
    case_file = tmp_path / "19_Softmax.py"
    case_file.write_text(
        textwrap.dedent(
            """
            class Model:
                def forward(self, x):
                    return x

                def __call__(self, x):
                    return self.forward(x)

            class FakeTensor:
                shape = (16, 256, 256)
                dtype = "float32"

            def get_inputs():
                return [FakeTensor()]

            def get_init_inputs():
                return []
            """
        ).strip(),
        encoding="utf-8",
    )

    case = load_case(case_file, case_id="19_Softmax")
    require_path = write_require(case, tmp_path / "custom")
    front_matter = _parse_front_matter(require_path.read_text(encoding="utf-8"))

    assert case.supported_dtypes == ["float32"]
    assert case.p0_shapes == [[16, 256, 256]]
    assert case.tolerance == {"rtol": 0.001, "atol": 0.001}
    assert case.outputs[0].shape == [16, 256, 256]
    assert front_matter["op_name"] == "Softmax"
    assert front_matter["supported_dtypes"] == ["float32"]
    assert front_matter["p0_shapes"] == [[16, 256, 256]]


def test_load_case_extracts_formula_and_dynamic_axis_globals(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(case_loader, "_list_idle_chip_ids", lambda: ["0"])
    monkeypatch.setattr(
        case_loader,
        "_run_probe_subprocess",
        lambda case_path, timeout_sec, probe_outputs, chip_id=None: (
            [TensorSpec(name="x0", shape=[2, 4, 8], dtype="float32")],
            [TensorSpec(name="y0", shape=[2, 4, 8], dtype="float32")] if probe_outputs else [],
            "[8]",
        ),
    )
    case_file = tmp_path / "101_DynamicAxisAdd.py"
    case_file.write_text(
        textwrap.dedent(
            """
            FORMULA = "out[b, s, d] = x[b, s, d] + bias[d]"
            DYNAMIC_AXIS = ["B", "S"]

            class Model:
                def __init__(self, hidden_size):
                    self.hidden_size = hidden_size

                def forward(self, x, bias):
                    return x + bias

                def __call__(self, x, bias):
                    return self.forward(x, bias)

            class FakeTensor:
                shape = (2, 4, 8)
                dtype = "float32"

            def get_inputs():
                return [FakeTensor(), FakeTensor()]

            def get_init_inputs():
                return [8]
            """
        ).strip(),
        encoding="utf-8",
    )

    case = load_case(case_file, case_id="101_DynamicAxisAdd")
    markdown = render_require_md(case)
    front_matter = _parse_front_matter(markdown)

    assert case.formula == "out[b, s, d] = x[b, s, d] + bias[d]"
    assert case.dynamic_axis == ["B", "S"]
    assert case.outputs[0].shape == [2, 4, 8]
    assert front_matter["dynamic_axis"] == ["B", "S"]
    assert "### 1.3 数学公式" in markdown


def test_load_case_leaves_outputs_empty_without_idle_chip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(case_loader, "_list_idle_chip_ids", lambda: [])
    case_file = tmp_path / "102_NoIdleChip.py"
    case_file.write_text(
        textwrap.dedent(
            """
            class Model:
                def forward(self, x):
                    raise RuntimeError("forward should not run without idle chip")

                def __call__(self, x):
                    return self.forward(x)

            class FakeTensor:
                shape = (4, 8)
                dtype = "float32"

            def get_inputs():
                return [FakeTensor()]

            def get_init_inputs():
                return []
            """
        ).strip(),
        encoding="utf-8",
    )

    case = load_case(case_file, case_id="102_NoIdleChip")

    assert case.inputs[0].shape == [4, 8]
    assert case.outputs == []


def test_load_case_parses_cases_global_into_p1_shapes(tmp_path, monkeypatch) -> None:
    """CASES 全局变量应被解析为列表并以合法单行 JSON 写入 front-matter.

    覆盖: JSON flow / 多行块 / 缺省 / 非法值四种场景.
    """
    monkeypatch.setattr(case_loader, "_list_idle_chip_ids", lambda: ["0"])
    monkeypatch.setattr(
        case_loader,
        "_run_probe_subprocess",
        lambda case_path, timeout_sec, probe_outputs, chip_id=None: (
            [
                TensorSpec(name="x0", shape=[2, 3], dtype="float32"),
                TensorSpec(name="x1", shape=[3], dtype="float32"),
            ],
            [TensorSpec(name="y0", shape=[2, 3], dtype="float32")] if probe_outputs else [],
            "[]",
        ),
    )

    model_body = textwrap.dedent(
        """
        class Model:
            def forward(self, x, y): return x
            def __call__(self, x, y): return self.forward(x, y)
        def get_inputs(): return [object(), object()]
        def get_init_inputs(): return []
        """
    ).lstrip()

    def _write_case(filename: str, cases_decl: str) -> None:
        (tmp_path / filename).write_text(cases_decl + model_body, encoding="utf-8")

    def _load_p1_shapes(filename: str):
        case = load_case(tmp_path / filename, case_id=filename[:-3])
        markdown = render_require_md(case)
        front_matter = _parse_front_matter(markdown)
        return case.p1_shapes, front_matter

    expected = [[[2, 3], [3]], [[4, 5], [5]]]

    # 1) JSON flow style
    _write_case("200_CasesJson.py", 'CASES = "[[[2, 3], [3]], [[4, 5], [5]]]"\n')
    p1_shapes, front_matter = _load_p1_shapes("200_CasesJson.py")
    assert p1_shapes == expected
    assert front_matter["p1_shapes"] == expected

    # 2) 多行块写法: 每行一个 JSON case
    _write_case("201_CasesBlock.py", 'CASES = """\n- [[2, 3], [3]]\n- [[4, 5], [5]]\n"""\n')
    p1_shapes, front_matter = _load_p1_shapes("201_CasesBlock.py")
    assert p1_shapes == expected
    assert front_matter["p1_shapes"] == expected

    # 3) 缺省 CASES
    _write_case("202_NoCases.py", "")
    p1_shapes, front_matter = _load_p1_shapes("202_NoCases.py")
    assert p1_shapes is None
    assert "p1_shapes" not in front_matter

    # 4) 非法 CASES (非列表) — 静默忽略, 不渲染 p1_shapes
    _write_case("203_BadCases.py", 'CASES = "not a list"\n')
    p1_shapes, front_matter = _load_p1_shapes("203_BadCases.py")
    assert p1_shapes is None
    assert "p1_shapes" not in front_matter

    # 5) 结构非法或输入数量不匹配: 不渲染 p1_shapes
    for filename, cases_decl in [
        ("204_BadCasesScalar.py", 'CASES = "[1, 2]"\n'),
        ("205_BadCasesMissingOuterCase.py", 'CASES = "[[2, 3], [3]]"\n'),
        ("206_BadCasesWrongArity.py", 'CASES = "[[[2, 3]]]"\n'),
    ]:
        _write_case(filename, cases_decl)
        p1_shapes, front_matter = _load_p1_shapes(filename)
        assert p1_shapes is None
        assert "p1_shapes" not in front_matter


def test_load_case_retries_other_idle_chips(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_run_probe(case_path, timeout_sec, probe_outputs, chip_id=None):
        if not probe_outputs:
            return [TensorSpec(name="x0", shape=[2, 3], dtype="float32")], [], "[]"
        calls.append(chip_id)
        if chip_id == "0":
            return [], [], "[]"
        return [], [TensorSpec(name="y0", shape=[2, 3], dtype="float32")], "[]"

    monkeypatch.setattr(case_loader, "_list_idle_chip_ids", lambda: ["0", "1"])
    monkeypatch.setattr(case_loader, "_run_probe_subprocess", fake_run_probe)
    case_file = tmp_path / "103_Retry.py"
    case_file.write_text(
        textwrap.dedent(
            """
            class Model:
                def forward(self, x):
                    return x

            def get_inputs():
                return []

            def get_init_inputs():
                return []
            """
        ).strip(),
        encoding="utf-8",
    )

    case = load_case(case_file, case_id="103_Retry")

    assert calls == ["0", "1"]
    assert case.outputs[0].name == "y0"


def test_load_case_limits_output_probe_attempts(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_run_probe(case_path, timeout_sec, probe_outputs, chip_id=None):
        if not probe_outputs:
            return [TensorSpec(name="x0", shape=[2, 3], dtype="float32")], [], "[]"
        calls.append(chip_id)
        return [], [], "[]"

    monkeypatch.setattr(case_loader, "_list_idle_chip_ids", lambda: ["0", "1", "2", "3"])
    monkeypatch.setattr(case_loader, "_run_probe_subprocess", fake_run_probe)
    case_file = tmp_path / "104_AttemptLimit.py"
    case_file.write_text(
        textwrap.dedent(
            """
            class Model:
                def forward(self, x):
                    return x

            def get_inputs():
                return []

            def get_init_inputs():
                return []
            """
        ).strip(),
        encoding="utf-8",
    )

    case = load_case(case_file, case_id="104_AttemptLimit")

    assert calls == ["0", "1", "2"]
    assert case.outputs == []
