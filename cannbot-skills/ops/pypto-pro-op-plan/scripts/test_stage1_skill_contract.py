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
import json
import re
import subprocess
import sys
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


def _run_precheck(code: str, root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code], cwd=root,
        text=True, capture_output=True, check=False,
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
        self.assertIn("若实际命中任何已声明拓扑却写 `[]` 或遗漏该命中，判 FAIL", verifier)
        self.assertIn("适用 property modifier 路由结果的并集", verifier)
        self.assertIn("拓扑数组为空时后三类仍须检查", verifier)
        self.assertIn("不能表示未知、未分析或跳过", plan)
        topology_map = json.loads(_read(KB / "topology-map.json"))
        self.assertTrue(topology_map["property_modifiers"]["is_list"]["patterns"])
        self.assertTrue(topology_map["target_gated"])
        self.assertTrue(topology_map["mandatory_constraints"])
        self.assertNotRegex(plan, r"contract v\d+")

    def test_plan_defaults_to_a5_and_has_an_executable_kb_precheck(self) -> None:
        plan = _read(PLAN / "SKILL.md")
        code_template = self._extract_kb_precheck(plan)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb, selection = self._create_kb_fixture(root)
            code = code_template.replace(
                "<selection>", "custom/demo/KB_SELECTION.json"
            ).replace("<kb_root>", kb.as_posix())
            op_dir = root / "custom" / "demo"
            op_dir.mkdir(parents=True)
            self._assert_flat_selection(root, op_dir, code, selection)
            split, code = self._assert_split_selection(
                root, op_dir, code_template, kb, selection,
            )
            self._assert_invalid_selection(root, split, code, selection)

    def test_kb_precheck_validates_topologies_array(self) -> None:
        plan = _read(PLAN / "SKILL.md")
        code_template = self._extract_kb_precheck(plan)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb, selection = self._create_kb_fixture(root)
            op_dir = root / "custom" / "demo"
            op_dir.mkdir(parents=True)
            selection_path = op_dir / "KB_SELECTION.json"
            code = code_template.replace(
                "<selection>", "custom/demo/KB_SELECTION.json"
            ).replace("<kb_root>", kb.as_posix())

            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            result = _run_precheck(code, root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.strip(), "OK")

            no_topology = json.loads(json.dumps(selection))
            no_topology["topologies"] = []
            no_topology["properties"] = {"is_list": True}
            no_topology["optional_patterns"] = [
                _ref(kb / "patterns" / "list.md", "patterns/list.md")
            ]
            no_topology["required_constraints"] = [
                _ref(kb / "constraints" / "list.md", "constraints/list.md"),
                _ref(kb / "constraints" / "target.md", "constraints/target.md"),
                _ref(kb / "constraints" / "mandatory.md", "constraints/mandatory.md"),
            ]
            no_topology["no_matching_pattern"] = False
            selection_path.write_text(json.dumps(no_topology), encoding="utf-8")
            result = _run_precheck(code, root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stdout.strip(), "OK")

            self._assert_topology_rejections(root, selection_path, code, selection)

    def _assert_topology_rejections(
        self, root: Path, selection_path: Path, code: str,
        selection: dict[str, object],
    ) -> None:
        invalid_cases = (
            ("scalar", "elementwise", "topologies must be an array"),
            ("null", None, "topologies must be an array"),
            ("non-string", ["elementwise", {}], "topology must be a string"),
            ("unknown", ["elementwise", "made-up"], "topology is not declared"),
        )
        for name, topologies, message in invalid_cases:
            with self.subTest(name=name):
                candidate = json.loads(json.dumps(selection))
                candidate["topologies"] = topologies
                selection_path.write_text(json.dumps(candidate), encoding="utf-8")
                result = _run_precheck(code, root)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stdout)
                self.assertEqual(result.stderr, "")

        candidate = json.loads(json.dumps(selection))
        candidate["topology"] = "made-up-legacy-value"
        selection_path.write_text(json.dumps(candidate), encoding="utf-8")
        result = _run_precheck(code, root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("legacy field topology is not allowed", result.stdout)
        self.assertEqual(result.stderr, "")

    def _extract_kb_precheck(self, plan: str) -> str:
        self.assertIn("未指定时默认 A5", plan)
        self.assertIn("constraints/arch-a5.md", plan)
        router = _read(KB / "ROUTER.md")
        topology_map = json.loads(_read(KB / "topology-map.json"))
        self.assertIn("workflow default A5", router)
        self.assertIn("workflow default A5", topology_map["target_gated"]["constraints/arch-a5.md"])
        self.assertIn("### 6. 收尾自检", plan)
        match = re.search(r"python -c '\n(.*?)\n'\n```", plan, re.DOTALL)
        self.assertIsNotNone(match)
        return match.group(1)

    def _create_kb_fixture(self, root: Path) -> tuple[Path, dict[str, object]]:
        production_version = json.loads(
            _read(KB / "topology-map.json")
        )["contract"]["contract_version"]
        contract_version = production_version + 1000
        config = root / "config"
        kb = config / "pypto-pro-op-kb"
        pattern = kb / "patterns" / "demo.md"
        list_pattern = kb / "patterns" / "list.md"
        constraint = kb / "constraints" / "demo.md"
        list_constraint = kb / "constraints" / "list.md"
        target_constraint = kb / "constraints" / "target.md"
        mandatory_constraint = kb / "constraints" / "mandatory.md"
        pattern.parent.mkdir(parents=True)
        constraint.parent.mkdir(parents=True)
        pattern.write_bytes(b"pattern\n")
        list_pattern.write_bytes(b"list pattern\n")
        constraint.write_bytes(b"constraint\n")
        list_constraint.write_bytes(b"list constraint\n")
        target_constraint.write_bytes(b"target constraint\n")
        mandatory_constraint.write_bytes(b"mandatory constraint\n")
        (kb / "topology-map.json").write_text(json.dumps({
            "contract": {
                "contract_version": contract_version,
                "property_keys": ["dtypes", "is_list"],
            },
            "topologies": {"elementwise": {}, "row-reduction": {}},
            "property_modifiers": {
                "is_list": {
                    "patterns": ["patterns/list.md"],
                    "constraints": ["constraints/list.md"],
                },
            },
            "target_gated": {"constraints/target.md": "fixture target"},
            "mandatory_constraints": {
                "constraints/mandatory.md": {
                    "applies_to": ["architect", "coder", "verifier"],
                },
            },
        }), encoding="utf-8")
        selection = {
            "schema_version": contract_version,
            "op": "demo",
            "class_id": ".",
            "topologies": ["elementwise", "row-reduction"],
            "properties": {"dtypes": ["float16"]},
            "optional_patterns": [_ref(pattern, "patterns/demo.md")],
            "required_constraints": [_ref(constraint, "constraints/demo.md")],
            "no_matching_pattern": False,
        }
        return kb, selection

    def _assert_flat_selection(
        self, root: Path, op_dir: Path, code: str, selection: dict[str, object],
    ) -> None:
        flat = op_dir / "KB_SELECTION.json"
        flat.write_text(json.dumps(selection), encoding="utf-8")
        result = _run_precheck(code, root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), "OK")

    def _assert_split_selection(
        self, root: Path, op_dir: Path, code_template: str,
        kb: Path, selection: dict[str, object],
    ) -> tuple[Path, str]:
        split_dir = op_dir / "class_a"
        split_dir.mkdir()
        split = split_dir / "KB_SELECTION.json"
        selection["class_id"] = "class_a"
        split.write_text(json.dumps(selection), encoding="utf-8")
        code = code_template.replace(
            "<selection>", "custom/demo/class_a/KB_SELECTION.json"
        ).replace("<kb_root>", kb.as_posix())
        result = _run_precheck(code, root)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return split, code

    def _assert_invalid_selection(
        self, root: Path, split: Path, code: str, selection: dict[str, object],
    ) -> None:
        selection["optional_patterns"][0]["sha256"] = "sha256:ABC"
        selection["optional_patterns"][0]["reason"] = ""
        selection["optional_patterns"][0]["path"] = "patterns/../constraints/demo.md"
        selection["required_constraints"][0]["sha256"] = "sha256:ABC"
        selection["schema_version"] += 1
        selection["topologies"] = ["made-up"]
        selection["class_id"] = "wrong"
        selection["properties"] = {"unknown": True}
        split.write_text(json.dumps(selection), encoding="utf-8")
        result = _run_precheck(code, root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stale sha256", result.stdout)
        self.assertIn("reference reason must be non-empty", result.stdout)
        self.assertIn("schema_version does not match", result.stdout)
        self.assertIn("topology is not declared", result.stdout)
        self.assertIn("class_id must be class_a", result.stdout)
        self.assertIn("unknown property keys", result.stdout)
        self.assertIn("path escapes patterns/ namespace", result.stdout)


if __name__ == "__main__":
    unittest.main()
