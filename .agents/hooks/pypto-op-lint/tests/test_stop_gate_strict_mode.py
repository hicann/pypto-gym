# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

from pathlib import Path

from .helpers import build_stateless_op_dir, load_lint_module, run_rule


def test_ol39_and_ol40_pass_when_structured_docs_valid(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    f39 = run_rule(mod, op_dir, "OL39")
    f40 = run_rule(mod, op_dir, "OL40")
    assert f39.status == "PASS"
    assert f40.status == "PASS"


def test_ol39_fails_when_missing_front_matter(tmp_path: Path):
    mod = load_lint_module()
    op_dir = build_stateless_op_dir(tmp_path, "demo")
    spec_path = op_dir / "SPEC.md"
    spec_path.write_text("# SPEC only\n", encoding="utf-8")
    finding = run_rule(mod, op_dir, "OL39")
    assert finding.status == "FAIL"
