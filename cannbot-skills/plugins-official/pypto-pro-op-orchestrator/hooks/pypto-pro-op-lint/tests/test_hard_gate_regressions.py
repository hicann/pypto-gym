#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression tests for AST checks and fail-closed hard-gate behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

LINT_ROOT = Path(__file__).resolve().parents[1]
if str(LINT_ROOT) not in sys.path:
    sys.path.insert(0, str(LINT_ROOT))

from pypto_pro_op_lint import core  # noqa: E402
from pypto_pro_op_lint import observability  # noqa: E402
from pypto_pro_op_lint.checks import d2_artifact  # noqa: E402
from pypto_pro_op_lint.cli import cmd_check_gate, cmd_check_module_gate  # noqa: E402
from pypto_pro_op_lint.core import (  # noqa: E402
    HOOK_INPUT_ENV,
    CheckContext,
    _load_rules,
    _run_checks,
)
from pypto_pro_op_lint.hooks import hook_post_edit, hook_stop  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_observability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_dir = tmp_path / "lint-state"
    monkeypatch.setattr(observability, "LOGS_DIR", str(state_dir))
    monkeypatch.setattr(
        observability, "LOGS_EVENTS_FILE", str(state_dir / "lint_events.jsonl")
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _context(op_dir: Path, stage: int = 4) -> CheckContext:
    return CheckContext(
        op_dir=str(op_dir),
        op_name="demo",
        stage=stage,
        rules=_load_rules(),
    )


def _run(op_dir: Path, rule_id: str, stage: int = 4):
    findings, _ = _run_checks(_context(op_dir, stage), [rule_id])
    return findings[0]


def _python_comment(content: str) -> str:
    return f"# {content}"


def _write_state(op_dir: Path, stage: int = 4, module_count: int | None = None) -> None:
    state: dict[str, object] = {
        "operator_name": "demo",
        "max_stage": 4,
        "current_stage": stage,
    }
    if module_count is not None:
        state["stage4_modules"] = {"module_count": module_count}
    _write(op_dir / ".orchestrator_state.json", json.dumps(state))


def _valid_spec(op_name: str = "demo") -> str:
    return f'''```json machine-contract
{{
  "schema_version": 1,
  "op_name": "{op_name}",
  "formula": "y = x",
  "supported_dtypes": ["float32"],
  "inputs": [{{"name": "x", "shape": [8, 16], "dtype": "float32", "value_range": [-4, 4]}}],
  "outputs": [{{"name": "y", "shape": [8, 16], "dtype": "float32", "value_range": [-4, 4]}}],
  "default_params": {{}},
  "tolerance": {{"atol": 0.001, "rtol": 0.001}},
  "dynamic_axes_ranges": {{}},
  "shape_constraints": [],
  "p0_cases": [{{"name": "p0", "params": {{}}, "input_shapes": {{"x": [8, 16]}}, "output_shapes": {{"y": [8, 16]}}}}]
}}
```

## 语义说明
y = x
'''


def _build_valid_operator(op_dir: Path) -> None:
    _write_state(op_dir)
    _write(op_dir / "SPEC.md", _valid_spec())
    _write(op_dir / "DESIGN.md", "# DESIGN\n")
    _write(op_dir / "module_interfaces.yaml", "is_fusion: false\n")
    _write(op_dir / "demo_golden.py", "import torch\n\ndef demo_golden(x): return x\n")
    _write(
        op_dir / "demo_golden_cpu.py",
        "import torch\n\ndef demo_golden_cpu(x): return x.float()\n",
    )
    _write(
        op_dir / "test_demo.py",
        """import pypto_pro.language as pl
from demo_golden_cpu import demo_golden_cpu

@pl.jit(auto_mutex=True)
def demo_kernel():
    pass

def _assert_precision(actual, expected):
    return actual == expected

def test_one(): pass
def test_two(): pass
def test_three(): pass
def test_four(): pass
""",
    )


def test_pl03_accepts_canonical_contract_from_source_layout(tmp_path: Path) -> None:
    _write(tmp_path / "SPEC.md", _valid_spec())
    assert _run(tmp_path, "PL03", stage=1).status == "PASS"


def test_pl03_rejects_invalid_contract_and_operator_mismatch(tmp_path: Path) -> None:
    _write(tmp_path / "SPEC.md", _valid_spec().replace("demo", "{{OP_NAME}}"))
    assert _run(tmp_path, "PL03", stage=1).status == "FAIL"

    _write(tmp_path / "SPEC.md", _valid_spec("another_op"))
    finding = _run(tmp_path, "PL03", stage=1)
    assert finding.status == "FAIL"
    assert "op_name" in finding.message


def test_pl03_loads_canonical_contract_from_installed_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_root = tmp_path / "config"
    installed_validator = (
        config_root / "skills/pypto-pro-intent-understand/scripts/validate_spec.py"
    )
    source_validator = (
        Path(__file__).resolve().parents[5]
        / "ops/pypto-pro-intent-understand/scripts/validate_spec.py"
    )
    _write(installed_validator, source_validator.read_text(encoding="utf-8"))
    fake_check = (
        config_root
        / "hooks/pypto-pro-op-lint/pypto_pro_op_lint/checks/d2_artifact.py"
    )
    monkeypatch.setattr(d2_artifact, "__file__", str(fake_check))
    _write(tmp_path / "SPEC.md", _valid_spec())
    assert _run(tmp_path, "PL03", stage=1).status == "PASS"


def test_pl03_fails_closed_when_canonical_validator_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        d2_artifact,
        "__file__",
        str(tmp_path / "isolated/hooks/lint/package/checks/d2_artifact.py"),
    )
    _write(tmp_path / "SPEC.md", _valid_spec())
    finding = _run(tmp_path, "PL03", stage=1)
    assert finding.status == "FAIL"
    assert "canonical SPEC validator" in finding.message


def test_pl01_rejects_from_import_of_classic_pypto(tmp_path: Path) -> None:
    _write(
        tmp_path / "test_demo.py",
        """import pypto_pro.language as pl
from pypto import frontend

@pl.jit
def kernel():
    pass
""",
    )
    assert _run(tmp_path, "PL01").status == "FAIL"


@pytest.mark.parametrize(
    "source",
    [
        "import importlib\nimportlib.import_module('pypto.frontend')\n",
        "import importlib as il\nil.import_module('pypto')\n",
        "from importlib import import_module as load\nload('pypto')\n",
        "__import__('pypto.frontend')\n",
    ],
)
def test_pl01_rejects_literal_dynamic_classic_imports(
    tmp_path: Path, source: str,
) -> None:
    _write(
        tmp_path / "test_demo.py",
        "import pypto_pro.language as pl\n@pl.jit\ndef kernel(): pass\n" + source,
    )
    assert _run(tmp_path, "PL01").status == "FAIL"


def test_pl02_ignores_decorator_text_in_comments_and_strings(tmp_path: Path) -> None:
    _write(
        tmp_path / "test_demo.py",
        '''import pypto_pro.language as pl
TEXT = "@pl.jit"
# @pl.jit

@pl.jit(auto_mutex=True)
def kernel():
    pass
''',
    )
    assert _run(tmp_path, "PL02").status == "PASS"


def test_pl02_fails_closed_on_invalid_python(tmp_path: Path) -> None:
    _write(tmp_path / "test_demo.py", "@pl.jit\ndef kernel(:\n    pass\n")
    finding = _run(tmp_path, "PL02")
    assert finding.status == "FAIL"
    assert "syntax error" in finding.message


@pytest.mark.parametrize(
    "statement",
    ["from pypto_pro import language", "from pypto import frontend"],
)
def test_pl11_rejects_from_imports(tmp_path: Path, statement: str) -> None:
    _write(tmp_path / "demo_golden.py", f"{statement}\n")
    assert _run(tmp_path, "PL11").status == "FAIL"


@pytest.mark.parametrize(
    "statement",
    [
        "import importlib\nimportlib.import_module('pypto_pro.language')",
        "__import__('pypto')",
    ],
)
def test_pl11_rejects_literal_dynamic_imports(
    tmp_path: Path, statement: str,
) -> None:
    _write(tmp_path / "demo_golden.py", f"{statement}\n")
    assert _run(tmp_path, "PL11").status == "FAIL"


@pytest.mark.parametrize(
    "statement",
    [
        "from modules.test_demo_module1 import kernel",
        "from modules import test_demo_module1",
        "import modules.test_demo_module1",
    ],
)
def test_pl13_rejects_qualified_staged_imports(tmp_path: Path, statement: str) -> None:
    _write(tmp_path / "modules" / "demo_golden_stage1.py", f"{statement}\n")
    assert _run(tmp_path, "PL13").status == "FAIL"


def test_pl13_rejects_literal_dynamic_staged_import(tmp_path: Path) -> None:
    _write(
        tmp_path / "modules" / "demo_golden_stage1.py",
        "import importlib\nimportlib.import_module('modules.test_demo_module1')\n",
    )
    assert _run(tmp_path, "PL13").status == "FAIL"


def test_pl13_post_edit_only_checks_current_golden_stage(tmp_path: Path) -> None:
    clean = tmp_path / "modules" / "demo_golden_stage1.py"
    polluted = tmp_path / "modules" / "demo_golden_stage12.py"
    _write(clean, "import torch\n")
    _write(polluted, "from modules.test_demo_module12 import kernel\n")
    ctx = _context(tmp_path)
    ctx.file_scope = str(clean)
    findings, _ = _run_checks(ctx, ["PL13"])
    assert findings[0].status == "PASS"


def test_stage2_golden_rules_ignore_downstream_stage_files(tmp_path: Path) -> None:
    _write(tmp_path / "demo_golden.py", "import torch\n")
    _write(tmp_path / "demo_golden_cpu.py", "import torch\n")
    _write(
        tmp_path / "modules" / "demo_golden_stage1.py",
        "import pypto_pro\n",
    )
    assert _run(tmp_path, "PL11", stage=2).status == "PASS"
    assert _run(tmp_path, "PL11", stage=4).status == "FAIL"


def test_pl15_ignores_assert_close_in_comments_and_strings(tmp_path: Path) -> None:
    source = "\n".join([
        'TEXT = "assert_close"',
        _python_comment("assert_close(x, y)"),
        "def _assert_precision(actual, expected):",
        "    return actual == expected",
    ])
    _write(tmp_path / "test_demo.py", source)
    assert _run(tmp_path, "PL15").status == "PASS"


def test_pl15_does_not_accept_assert_precision_string(tmp_path: Path) -> None:
    _write(tmp_path / "test_demo.py", 'TEXT = "_assert_precision"\n')
    assert _run(tmp_path, "PL15").status == "FAIL"


def test_pl15_does_not_accept_assert_precision_variable(tmp_path: Path) -> None:
    _write(tmp_path / "test_demo.py", "_assert_precision = object()\n")
    assert _run(tmp_path, "PL15").status == "FAIL"


def test_missing_checker_is_s0_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _context(tmp_path)
    monkeypatch.delitem(core.CHECKERS, "PL01")
    findings, _ = _run_checks(ctx, ["PL01"])
    finding = findings[0]
    assert finding.status == "FAIL"
    assert finding.severity == "S0"


def test_rules_checker_drift_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = _load_rules()
    _write(tmp_path / "rules.json", json.dumps({"rules": rules}))
    monkeypatch.setattr(core, "SCRIPT_DIR", str(tmp_path))
    monkeypatch.delitem(core.CHECKERS, "PL01")
    with pytest.raises(core.LintConfigurationError, match="规则缺少检查器"):
        _load_rules()


def test_missing_gate_trigger_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = _load_rules()
    for rule in rules:
        if rule["id"] == "PL03":
            rule["triggers"] = []
    _write(tmp_path / "rules.json", json.dumps({"rules": rules}))
    monkeypatch.setattr(core, "SCRIPT_DIR", str(tmp_path))
    with pytest.raises(core.LintConfigurationError, match="缺少合法 triggers"):
        _load_rules()


def test_unmapped_declared_gate_trigger_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = _load_rules()
    for rule in rules:
        if rule["id"] == "PL03":
            rule["stages"].append(2)
            rule["triggers"].append("gate:S2")
    _write(tmp_path / "rules.json", json.dumps({"rules": rules}))
    monkeypatch.setattr(core, "SCRIPT_DIR", str(tmp_path))
    with pytest.raises(core.LintConfigurationError, match="未进入对应 Stage 门禁"):
        _load_rules()


def test_missing_rule_definition_is_s0_failure(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    ctx.rules = [rule for rule in ctx.rules if rule["id"] != "PL01"]
    findings, _ = _run_checks(ctx, ["PL01"])
    assert findings[0].status == "FAIL"
    assert findings[0].severity == "S0"


def test_rules_load_failure_raises_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "SCRIPT_DIR", str(tmp_path))
    with pytest.raises(core.LintConfigurationError):
        _load_rules()


def test_empty_rules_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path / "rules.json", json.dumps({"rules": []}))
    monkeypatch.setattr(core, "SCRIPT_DIR", str(tmp_path))
    with pytest.raises(core.LintConfigurationError):
        _load_rules()


def test_module_resolution_failure_blocks(tmp_path: Path) -> None:
    assert cmd_check_module_gate(str(tmp_path), "1") == 2


def test_module_gate_rejects_wrong_stage(tmp_path: Path) -> None:
    assert cmd_check_module_gate(str(tmp_path), "1", stage=3) == 2


def test_module_gate_rejects_missing_staged_file(tmp_path: Path) -> None:
    _write(
        tmp_path / ".orchestrator_state.json",
        json.dumps({
            "operator_name": "demo",
            "stage4_modules": {"module_count": 1},
        }),
    )
    assert cmd_check_module_gate(str(tmp_path), "1") == 2


def test_unknown_stage_gate_blocks(tmp_path: Path) -> None:
    assert cmd_check_gate(str(tmp_path), 99) == 2


def test_stage4_golden_rules_are_enabled() -> None:
    rule_by_id = {rule["id"]: rule for rule in _load_rules()}
    assert 4 in rule_by_id["PL04"]["stages"]
    assert 4 in rule_by_id["PL05"]["stages"]
    assert 4 in rule_by_id["PL11"]["stages"]
    assert 4 in rule_by_id["PL13"]["stages"]
    assert "PL04" in core.GATE_RULES_BY_STAGE[4]
    assert "PL05" in core.GATE_RULES_BY_STAGE[4]
    assert "PL11" in core.GATE_RULES_BY_STAGE[4]
    assert "PL13" in core.GATE_RULES_BY_STAGE[4]


@pytest.mark.parametrize("missing", ["demo_golden.py", "demo_golden_cpu.py"])
def test_stage4_gate_rejects_deleted_required_golden(
    tmp_path: Path, missing: str,
) -> None:
    _build_valid_operator(tmp_path)
    (tmp_path / missing).unlink()
    assert cmd_check_gate(str(tmp_path), 4) == 2


def test_removed_rules_are_not_registered_or_referenced() -> None:
    removed = {"PL06", "PL12", "PL14", "PL17"}
    rule_ids = {rule["id"] for rule in _load_rules()}
    referenced = set(
        core.TEST_RULE_IDS
        + core.GOLDEN_RULE_IDS
        + core.MODULE_GATE_RULE_IDS
        + core.POST_EDIT_TEST_RULES
        + core.POST_EDIT_GOLDEN_RULES
    )
    for stage_rules in core.GATE_RULES_BY_STAGE.values():
        referenced.update(stage_rules)
    assert removed.isdisjoint(rule_ids)
    assert removed.isdisjoint(core.CHECKERS)
    assert removed.isdisjoint(referenced)


def test_all_rules_have_registered_checkers_and_consistent_gate_triggers() -> None:
    rules = _load_rules()
    rule_by_id = {rule["id"]: rule for rule in rules}
    assert set(rule_by_id) == set(core.CHECKERS)
    for stage, rule_ids in core.GATE_RULES_BY_STAGE.items():
        for rule_id in rule_ids:
            assert stage in rule_by_id[rule_id]["stages"]
            assert f"gate:S{stage}" in rule_by_id[rule_id]["triggers"]
    for rule_id in core.MODULE_GATE_RULE_IDS:
        assert "gate:M" in rule_by_id[rule_id]["triggers"]
    for rule_id in core.POST_EDIT_TEST_RULES + core.POST_EDIT_GOLDEN_RULES:
        assert "post-edit" in rule_by_id[rule_id]["triggers"]


@pytest.mark.parametrize("stage", [1, 2, 3, 4])
def test_complete_stage_gate_passes_valid_operator(tmp_path: Path, stage: int) -> None:
    _build_valid_operator(tmp_path)
    assert cmd_check_gate(str(tmp_path), stage) == 0


def test_stage2_gate_does_not_require_optional_performance_report(
    tmp_path: Path,
) -> None:
    _build_valid_operator(tmp_path)
    assert not (tmp_path / "GOLDEN_PERF_REPORT.md").exists()
    assert cmd_check_gate(str(tmp_path), 2) == 0


def _post_edit_output(
    tmp_path: Path,
    source: str,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> dict:
    _write_state(tmp_path)
    test_file = tmp_path / "test_demo.py"
    _write(test_file, source)
    payload = {"tool_input": {"file_path": str(test_file)}}
    monkeypatch.setenv(HOOK_INPUT_ENV, json.dumps(payload))
    assert hook_post_edit() == 0
    return json.loads(capfd.readouterr().out)


@pytest.mark.parametrize(
    ("source", "expected_decision"),
    [
        pytest.param(
            """import pypto_pro.language as pl
from pypto import frontend
@pl.jit
def kernel(): pass
""",
            "block",
            id="ast-import-violation",
        ),
        pytest.param(
            """import pypto_pro.language as pl
@pl.jit
def kernel(): pass
""",
            "allow",
            id="valid-file",
        ),
    ],
)
def test_post_edit_hook_returns_expected_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    source: str,
    expected_decision: str,
) -> None:
    output = _post_edit_output(tmp_path, source, monkeypatch, capfd)
    assert output["hookSpecificOutput"]["decision"] == expected_decision
    assert "test_demo.py test_demo.py:" not in output["hookSpecificOutput"]["reason"]


def test_stop_hook_passes_valid_stage4_operator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    _build_valid_operator(tmp_path)
    monkeypatch.setenv(HOOK_INPUT_ENV, json.dumps({"cwd": str(tmp_path)}))
    assert hook_stop() == 0
    assert capfd.readouterr().out == ""
