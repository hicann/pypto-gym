#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Regression checks for the Stage-1 skill boundaries and hand-off contract."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path


OPS_ROOT = Path(__file__).resolve().parents[2]
INTENT = OPS_ROOT / "pypto-pro-intent-understand"
MATERIAL = OPS_ROOT / "pypto-pro-material-explore"
PLAN = OPS_ROOT / "pypto-pro-op-plan"
KB = OPS_ROOT / "pypto-pro-op-kb"
VERIFIER = (
    OPS_ROOT.parent / "plugins-official" / "pypto-pro-op-orchestrator"
    / "agents" / "pypto-pro-op-verifier.md"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _ref(path: Path, rel: str) -> dict[str, str]:
    return {
        "path": rel,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        "reason": "specific design reason",
    }


def _report_with_paths(template: str, samples: list[str], tutorials: list[str]) -> str:
    report = re.sub(r"\{[^{}\r\n]+\}", "filled", template).replace(
        "op_name: filled", "op_name: demo",
    )
    report = re.sub(r"(?m)^\|[^\n]*`filled`[^\n]*\n?", "", report)
    sample_rows = "\n".join(
        f"| {index} | `{path}` | `vf.add` |" for index, path in enumerate(samples, 1)
    )
    tutorial_rows = "\n".join(f"| `{path}` | tutorials | pattern | applicable |" for path in tutorials)
    report = report.replace(
        "### 4.1 全量样例参考（按 cube/vec 组成分类）",
        "### 4.1 全量样例参考（按 cube/vec 组成分类）\n" + sample_rows,
        1,
    )
    return report.replace(
        "### 5.1 适用的设计模式",
        "### 5.1 适用的设计模式\n" + tutorial_rows,
        1,
    )


class Stage1SkillContractTests(unittest.TestCase):
    def test_skills_are_concise_and_have_clear_boundaries(self) -> None:
        boundaries = {
            INTENT: ("SPEC.md", "不要选择 PyPTO-Pro API"),
            MATERIAL: ("PRO_MATERIAL_INDEX.md", "不要修改需求语义"),
            PLAN: ("KB_SELECTION.json", "不要用于 golden"),
        }
        for directory, required in boundaries.items():
            text = _read(directory / "SKILL.md")
            self.assertLessEqual(len(text.splitlines()), 500, directory.name)
            for phrase in required:
                self.assertIn(phrase, text, directory.name)

    def test_plan_preserves_all_stage1_deliverables_and_order(self) -> None:
        text = _read(PLAN / "SKILL.md")
        for artifact in (
            "SPEC.md", "PRO_MATERIAL_INDEX.md", "EXPLORE_REPORT.md", "MEMORY.md",
            "KB_SELECTION.json",
        ):
            self.assertIn(artifact, text)
        positions = [
            text.index("### 1. 冻结需求语义"),
            text.index("### 2. 创建 MEMORY 并补充 kernel 交接合同"),
            text.index("### 3. 探索目标版本资料"),
            text.index("### 4. 收敛 MEMORY"),
            text.index("### 5. 冻结知识选择"),
        ]
        self.assertEqual(positions, sorted(positions))

    def test_material_indexes_only_current_tutorials_tree(self) -> None:
        for path in (
            MATERIAL / "SKILL.md",
            MATERIAL / "templates" / "pro_material_index.md",
            MATERIAL / "templates" / "explore_report.md",
        ):
            text = _read(path)
            self.assertIn("tutorial", text, path.name)
            self.assertNotIn("guide/", text, path.name)
            self.assertNotIn("{guide", text, path.name)

    def test_spec_template_has_no_operator_specific_defaults(self) -> None:
        text = _read(INTENT / "templates" / "spec-template.md")
        for forbidden in ("bfloat16", "1024", "eps", "min_v", "max_v"):
            self.assertNotIn(forbidden, text)
        self.assertEqual(text.count("```json machine-contract"), 1)
        self.assertRegex(text, r'"formula":\s*"\{\{[^{}]+\}\}"')
        self.assertRegex(text, r'"supported_dtypes":\s*\[\]')
        self.assertNotIn('"p0_shapes"', text)
        self.assertRegex(text, r'"default_params":\s*\{\}')
        self.assertTrue(re.search(r"\{\{[^{}]+\}\}", text))

    def test_stage1_commands_use_canonical_python_entrypoint(self) -> None:
        paths = list(INTENT.rglob("*.md")) + list(MATERIAL.rglob("*.md")) + list(PLAN.rglob("*.md"))
        for path in paths:
            self.assertNotIn("python3", _read(path), path.as_posix())

    def test_interface_and_kb_handoffs_are_explicit(self) -> None:
        intent = _read(INTENT / "SKILL.md")
        plan = _read(PLAN / "SKILL.md")
        material = _read(MATERIAL / "SKILL.md")
        self.assertIn("公开接口的 rank、shape", intent)
        self.assertIn("rank-0", intent)
        self.assertIn('class_id` 必须为字面量 `"."`', plan)
        self.assertIn("custom/<op>/<class>/KB_SELECTION.json", plan)
        self.assertLess(plan.index("立即创建或追加 `MEMORY.md`"), plan.index("### 3. 探索"))
        self.assertIn("单一 VF API", material)

    def test_kb_documents_define_zero_or_multi_topology_semantics(self) -> None:
        plan = _read(PLAN / "SKILL.md")
        router = _read(KB / "ROUTER.md")
        router_flat = " ".join(router.split())
        contract = _read(KB / "CONTRACT.md")
        selection_rule = json.loads(
            _read(KB / "topology-map.json")
        )["selection_rule"]
        verifier = _read(VERIFIER)

        self.assertIn("Route zero or more topologies matched by the formula", router)
        self.assertIn("record `topologies: []` rather than force a best-fit category", router)
        self.assertIn("property, target and mandatory routing still runs", router)
        self.assertIn("does not mean unknown or skipped", router)
        self.assertIn("every actual topology match must be recorded", router_flat)
        self.assertIn("union of constraints routed by every matched topology", router)
        self.assertIn("applicable property modifier", router)
        self.assertIn("contains topologies/properties", router)
        self.assertNotIn("For the selected topology", router)
        self.assertIn("possibly empty", contract)
        self.assertIn("never force a best-fit category", contract)
        self.assertIn("never use an empty array to mean unknown or skipped", contract)
        self.assertIn("omitting one is invalid", contract)
        self.assertIn("property, target and mandatory routing still applies", contract)
        self.assertIn("routed constraints is mandatory", contract)
        self.assertIn("applicable property modifiers in the candidate pool", contract)
        self.assertIn("optional-pattern relevance rules", contract)
        self.assertIn("may match zero or more topologies", selection_rule)
        self.assertIn("omitting an actual match is invalid", selection_rule)
        self.assertIn("must not mean unknown or skipped", selection_rule)
        self.assertIn(
            "then add every applicable constraint from property modifiers, "
            "the target gate and mandatory constraints",
            selection_rule,
        )
        self.assertIn("matched topologies and applicable property modifiers", selection_rule)
        self.assertIn("retain all and only candidates", selection_rule)
        self.assertIn("零命中和多命中都是正常结果", verifier)
        self.assertIn("若实际命中任何拓扑却写 `[]` 或遗漏，判 FAIL", verifier)
        self.assertIn("适用 property modifier 路由结果的并集", verifier)
        self.assertIn("拓扑数组为空时后三类仍须检查", verifier)
        self.assertIn("不能表示未知、未分析或跳过", plan)
        topology_map = json.loads(_read(KB / "topology-map.json"))
        self.assertTrue(topology_map["property_modifiers"]["is_list"]["patterns"])
        self.assertTrue(topology_map["target_gated"])
        self.assertTrue(topology_map["mandatory_constraints"])
        self.assertNotRegex(plan, r"contract v\d+")

    def test_plan_uses_one_stage1_validator(self) -> None:
        plan = _read(PLAN / "SKILL.md")
        validator_path = PLAN / "scripts" / "validate_stage1.py"
        self.assertTrue(validator_path.is_file())
        self.assertIn("scripts/validate_stage1.py", plan)
        self.assertIn("未指定时默认 A5", plan)
        self.assertNotIn("python -c '", plan)
        planner = _read(VERIFIER.with_name("pypto-pro-op-planner.md"))
        self.assertIn("validate_stage1.py", planner)

    def test_kb_validator_accepts_flat_split_and_dynamic_topologies(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_stage1.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb, selection = self._create_kb_fixture(root)
            op_dir = root / "custom" / "demo"
            op_dir.mkdir(parents=True)

            flat = op_dir / "KB_SELECTION.json"
            flat.write_text(json.dumps(selection), encoding="utf-8")
            self.assertEqual(validator.check_kb(op_dir, kb, "demo"), [])

            flat.unlink()
            split = op_dir / "class_a" / "KB_SELECTION.json"
            split.parent.mkdir()
            selection["class_id"] = "class_a"
            split.write_text(json.dumps(selection), encoding="utf-8")
            self.assertEqual(validator.check_kb(op_dir, kb, "demo"), [])

            selection["topologies"] = []
            selection["optional_patterns"] = []
            selection["no_matching_pattern"] = True
            split.write_text(json.dumps(selection), encoding="utf-8")
            self.assertEqual(validator.check_kb(op_dir, kb, "demo"), [])

    def test_kb_validator_rejects_stale_unknown_and_escaping_references(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_stage1.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb, selection = self._create_kb_fixture(root)
            op_dir = root / "custom" / "demo"
            op_dir.mkdir(parents=True)
            path = op_dir / "KB_SELECTION.json"
            selection["topologies"] = ["made-up"]
            selection["properties"] = {"unknown": True}
            selection["optional_patterns"][0]["reason"] = ""
            selection["optional_patterns"][0]["path"] = "patterns/../constraints/demo.md"
            selection["required_constraints"][0]["sha256"] = "sha256:stale"
            path.write_text(json.dumps(selection), encoding="utf-8")
            errors = "\n".join(validator.check_kb(op_dir, kb, "demo"))
            for expected in (
                "unknown, duplicate, or non-string topologies", "unknown properties", "reason must be non-empty",
                "escapes patterns/", "stale sha256",
            ):
                self.assertIn(expected, errors)

            path.write_text('{"op":"demo","op":"other"}', encoding="utf-8")
            self.assertIn("duplicate JSON key", "\n".join(validator.check_kb(op_dir, kb, "demo")))

            (kb / "topology-map.json").write_text(
                '{"contract":[],"topologies":{}}', encoding="utf-8",
            )
            with self.assertRaises(validator.Stage1ConfigurationError):
                validator.check_kb(op_dir, kb, "demo")

    def test_kb_validator_requires_boolean_flags_and_integer_contract_versions(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_stage1.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb, selection = self._create_kb_fixture(root)
            op_dir = root / "custom/demo"
            op_dir.mkdir(parents=True)
            path = op_dir / "KB_SELECTION.json"
            for flag in (0, 1, None, "false", True):
                with self.subTest(flag=flag):
                    selection["no_matching_pattern"] = flag
                    path.write_text(json.dumps(selection), encoding="utf-8")
                    self.assertEqual(
                        validator.check_kb(op_dir, kb, "demo"),
                        ["KB_SELECTION.json no_matching_pattern is invalid"],
                    )
            mapping_path = kb / "topology-map.json"
            mapping = json.loads(_read(mapping_path))
            for version in (True, False, 1.0, "1", None):
                with self.subTest(contract_version=version):
                    mapping["contract"]["contract_version"] = version
                    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
                    self.assertRaisesRegex(
                        validator.Stage1ConfigurationError, "contract_version must be an integer",
                        validator.check_kb, op_dir, kb, "demo",
                    )

    def test_report_validator_uses_template_and_index_contents(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_stage1.py")
        op_dir, paths, report, index = self._create_report_fixture()
        self.assertEqual(validator.check_report(op_dir, "demo", index), [])
        tutorial_row = f"| `{paths[1]}` | tutorials | pattern | applicable |"
        misplaced = report.replace(tutorial_row, "").replace("### 5.2 来自教程的关键约束与建议", "")
        (op_dir / "EXPLORE_REPORT.md").write_text(
            misplaced.replace("## 6. Stage 3 设计事实输入", "## 6. Stage 3 设计事实输入\n" + tutorial_row),
            encoding="utf-8",
        )
        errors = validator.check_report(op_dir, "demo", index)
        self.assertTrue(any("tutorial paths not covered" in error for error in errors))
        (op_dir / "EXPLORE_REPORT.md").write_text(
            report.replace(f"| `{paths[1]}` |", "| `unrelated.md` |") + "\n" + paths[1],
            encoding="utf-8",
        )
        errors = validator.check_report(op_dir, "demo", index)
        self.assertTrue(any("tutorial paths not covered" in error for error in errors))
        report = _report_with_paths(
            _read(MATERIAL / "templates/explore_report.md"),
            [paths[0], "pro_ops/a5/vector/unlisted.py"],
            [paths[1], "docs/pypto_pro/tutorials/unlisted.md"],
        )
        (op_dir / "EXPLORE_REPORT.md").write_text(report, encoding="utf-8")
        errors = "\n".join(validator.check_report(op_dir, "demo", index))
        self.assertIn("sample table paths absent", errors)
        self.assertIn("tutorial table paths absent", errors)

    def test_report_validator_parses_frontmatter(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_stage1.py")
        op_dir, _, report, index = self._create_report_fixture()
        for field, values, valid in (
            ("op_name: demo", ('"demo"', "'demo' # operator"), True),
            ("schema_version: 2", ('"2" # version', "'2'", "2 # version"), True),
            ("feasibility: filled", ('"feasible # literal"', "'it''s feasible'", "feasible#literal"), True),
            ("op_name: demo", ('"other"', '"demo', "'demo\"", "demo\nop_name: demo"), False),
            ("schema_version: 2", ("999", "2#not-a-comment"), False),
            ("feasibility: filled", ("", "# no value", '""', "' ' # empty", '"bad\\q"'), False),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    replacement = field.split(":", 1)[0] + ": " + value
                    (op_dir / "EXPLORE_REPORT.md").write_text(
                        report.replace(field, replacement), encoding="utf-8",
                    )
                    errors = validator.check_report(op_dir, "demo", index)
                    self.assertEqual(not errors, valid, errors)
        (op_dir / "EXPLORE_REPORT.md").write_text(
            report.replace("op_name: demo", "# note\n\nop_name: 'it''s # feasible' # comment"),
            encoding="utf-8",
        )
        self.assertEqual(validator.check_report(op_dir, "it's # feasible", index), [])

    def test_report_validator_limits_table_coverage_to_its_section(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_stage1.py")
        op_dir, paths, report, index = self._create_report_fixture()
        tutorial_row = f"| `{paths[1]}` | tutorials | pattern | applicable |"
        for heading, valid in (("#### detail", True), ("### next", False), ("## next", False), ("# next", False)):
            with self.subTest(heading=heading):
                (op_dir / "EXPLORE_REPORT.md").write_text(
                    report.replace(tutorial_row, heading + "\n" + tutorial_row), encoding="utf-8",
                )
                errors = validator.check_report(op_dir, "demo", index)
                covered = not any("tutorial paths not covered" in error for error in errors)
                self.assertEqual(covered, valid)

    def test_stage1_validator_accepts_a_complete_fixture(self) -> None:
        validator = self._load_module(PLAN / "scripts/validate_stage1.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            op_dir, devkit = root / "custom/demo", root / "devkit"
            (devkit / "docs/pypto_pro/api").mkdir(parents=True)
            (devkit / "docs/pypto_pro/api/index.md").write_text("api", encoding="utf-8")
            (devkit / "docs/pypto_pro/tutorials").mkdir(parents=True)
            (devkit / "docs/pypto_pro/tutorials/guide.md").write_text("guide", encoding="utf-8")
            manifest = _read(MATERIAL / "references" / "official_samples.md")
            for relative in re.findall(r"`(pro_ops/[^`|]+\.py)`", manifest):
                target = devkit / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("# sample", encoding="utf-8")

            op_dir.mkdir(parents=True)
            contract = {
                "schema_version": 1, "op_name": "demo", "formula": "y = x",
                "supported_dtypes": ["float32"],
                "inputs": [{"name": "x", "shape": [4], "dtype": "float32", "value_range": [-1, 1]}],
                "outputs": [{"name": "y", "shape": [4], "dtype": "float32", "value_range": [-1, 1]}],
                "default_params": {}, "tolerance": {"atol": 0.001, "rtol": 0.001},
                "dynamic_axes_ranges": {}, "shape_constraints": [],
                "p0_cases": [{"name": "p0", "params": {}, "input_shapes": {"x": [4]}, "output_shapes": {"y": [4]}}],
            }
            (op_dir / "SPEC.md").write_text(
                "```json machine-contract\n" + json.dumps(contract) +
                "\n```\n\n## kernel 契约补充\n已确认。\n", encoding="utf-8",
            )
            material = self._load_module(MATERIAL / "scripts/build_material_index.py")
            index, _ = material.build_index(devkit, material.manifest_path())
            (op_dir / "PRO_MATERIAL_INDEX.md").write_text(index, encoding="utf-8")
            template = _read(MATERIAL / "templates" / "explore_report.md")
            covered = re.findall(
                r"`((?:pro_ops/[^`]+\.py|docs/pypto_pro/tutorials/[^`]+\.md))`", index,
            )
            report = _report_with_paths(
                template,
                [path for path in covered if path.startswith("pro_ops/")],
                [path for path in covered if path.startswith("docs/")],
            )
            (op_dir / "EXPLORE_REPORT.md").write_text(
                report, encoding="utf-8",
            )
            (op_dir / "MEMORY.md").write_text(
                "SPEC.md PRO_MATERIAL_INDEX.md EXPLORE_REPORT.md KB_SELECTION.json", encoding="utf-8",
            )
            kb, selection = self._create_kb_fixture(root)
            (op_dir / "KB_SELECTION.json").write_text(json.dumps(selection), encoding="utf-8")
            results = validator.validate(op_dir, devkit, kb)
            self.assertTrue(all(not errors for _, errors in results), results)

    def _load_module(self, path: Path):
        spec = importlib.util.spec_from_file_location(f"_stage1_test_{path.stem}", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _create_report_fixture(self) -> tuple[Path, tuple[str, str], str, str]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        op_dir = Path(temporary.name) / "demo"
        op_dir.mkdir()
        paths = ("pro_ops/a5/vector/test_demo.py", "docs/pypto_pro/tutorials/guide.md")
        report = _report_with_paths(_read(MATERIAL / "templates/explore_report.md"), [paths[0]], [paths[1]])
        (op_dir / "EXPLORE_REPORT.md").write_text(report, encoding="utf-8")
        index = "\n".join(f"- `{path}`" for path in paths)
        return op_dir, paths, report, index

    def _create_kb_fixture(self, root: Path) -> tuple[Path, dict[str, object]]:
        production_version = json.loads(
            _read(KB / "topology-map.json")
        )["contract"]["contract_version"]
        contract_version = production_version + 1000
        kb = root / "config" / "pypto-pro-op-kb"
        files = {
            "patterns/demo.md": b"pattern\n",
            "constraints/demo.md": b"constraint\n",
        }
        for relative, content in files.items():
            target = kb / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        (kb / "topology-map.json").write_text(json.dumps({
            "contract": {
                "contract_version": contract_version,
                "property_keys": ["dtypes", "is_list"],
            },
            "topologies": {"elementwise": {}, "row-reduction": {}},
        }), encoding="utf-8")
        selection = {
            "schema_version": contract_version,
            "op": "demo",
            "class_id": ".",
            "topologies": ["elementwise", "row-reduction"],
            "properties": {"dtypes": ["float16"]},
            "optional_patterns": [
                _ref(kb / "patterns/demo.md", "patterns/demo.md")
            ],
            "required_constraints": [
                _ref(kb / "constraints/demo.md", "constraints/demo.md")
            ],
            "no_matching_pattern": False,
        }
        return kb, selection

if __name__ == "__main__":
    unittest.main()
