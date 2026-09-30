# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The delivered compiler is a checked-in source snapshot, not an installed package."""
from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest

PLUGIN = Path(__file__).resolve().parents[2]
SNAPSHOT = PLUGIN / "resources/ascriptor"
sys.path.insert(0, str(PLUGIN / "scripts/scriptor-runtime"))
from scriptorlib.common import ContractError, atomic_json, load_module
from scriptorlib.sources import check_references, refresh, verify

installer = load_module(PLUGIN / "scripts/install_opencode.py", "scriptor_source_installer_test")


class SnapshotTests(unittest.TestCase):
    def test_checked_in_snapshot_is_complete(self):
        self.assertTrue(refresh(SNAPSHOT, check=True)["synchronized"])
        index = verify(SNAPSHOT)
        self.assertIn("library/ascriptor/runtime/board.py", index["files"])
        self.assertNotIn("library/tools/build_wheel.py", index["files"])
        manifest = json.loads((SNAPSHOT / "sources.json").read_text())
        self.assertEqual(manifest["provenance"]["kind"], "self-contained-snapshot")
        self.assertNotIn("sources", manifest)
        api = json.loads((SNAPSHOT / "library/docs/api/manifest.json").read_text())
        names = {row["name"] for row in api["entries"] if row.get("scope") == "runtime"}
        self.assertEqual(names, {"OpExec", "OpExec.__call__", "compile_kernel", "Board", "BoardError"})

    def test_modified_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "snapshot"
            shutil.copytree(SNAPSHOT, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            scratch = target / "library/examples/api/axpb/tmp"
            scratch.mkdir()
            (scratch / "run.log").write_text("local result")
            (target / "tmp").mkdir()
            (target / "tmp/emit.py").write_text("generated")
            (target / "library/README.md").chmod(0o644)
            verify(target)
            (target / "library/ascriptor/__init__.py").write_text("changed\n")
            with self.assertRaises(ContractError):
                verify(target)

    def test_inline_command_and_json_reference_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("library", "agent", "kernels"):
                (root / name).mkdir()
            (root / "agent/README.md").write_text("Run `python tools/missing.py` now.\n")
            for relative, value in (("library/examples/api/index.json", {"entries": []}),
                                    ("kernels/index.json", {"entries": []}),
                                    ("agent/index/kernels.json", {"candidates": []}),
                                    ("library/docs/api/manifest.json", {"entries": [
                                        {"name": "missing", "example": "examples/api/missing/main.py"}]})):
                atomic_json(root / relative, value)
            with self.assertRaisesRegex(ContractError, "missing command"):
                check_references(root, {"agent/README.md": {}})
            (root / "agent/README.md").write_text("Ready.\n")
            with self.assertRaisesRegex(ContractError, "missing JSON path"):
                check_references(root, {"agent/README.md": {}})

    def test_navigation_fallback_targets_exist(self):
        navigation = load_module(SNAPSHOT / "agent/tools/select_example.py", "scriptor_navigation_test")
        for language in ("en", "zh-CN"):
            for row in navigation.fallback(SNAPSHOT / "agent", language):
                self.assertTrue(Path(row["path"]).is_file(), row)

    def test_opencode_registers_without_copying_compiler(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            result = installer.install(project, sys.executable)
            config = project / ".opencode"
            receipt = json.loads((config / "scriptor-install.json").read_text())
            self.assertEqual(receipt["schema"], "cannbot.scriptor-install/1")
            self.assertEqual(Path(receipt["source_root"]), SNAPSHOT)
            self.assertEqual(result["doctor"]["source_root"], str(SNAPSHOT))
            self.assertFalse((config / "resources/ascriptor").exists())
            self.assertTrue((config / "scriptor/scripts/scriptor.py").is_file())
            self.assertTrue((config / "skills/pypto-pro-scriptor-develop/SKILL.md").is_file())

    def test_unmanaged_project_instructions_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "AGENTS.md").write_text("my own instructions")
            with self.assertRaises(ContractError):
                installer.install(project, sys.executable)
            self.assertFalse((project / ".opencode").exists())



if __name__ == "__main__":
    unittest.main()
