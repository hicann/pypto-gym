# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One repository license, no nested Ascriptor licenses, and Python source headers."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest
import xml.etree.ElementTree as ET

PLUGIN = Path(__file__).resolve().parents[2]
REPO = PLUGIN.parents[2]
VIEW = PLUGIN / "resources/ascriptor"


class LicenseScopeTests(unittest.TestCase):
    def test_snapshot_uses_repository_license(self):
        root_license = (REPO / "LICENSE").read_text()
        self.assertIn("CANN Open Software License Agreement Version 2.0", root_license)
        index = json.loads((VIEW / "sources-index.json").read_text())
        for owner in ("library", "agent", "kernels"):
            self.assertNotIn(owner + "/LICENSE", index["files"])
            self.assertFalse((VIEW / owner / "LICENSE").exists())
        self.assertNotIn("license-files", (VIEW / "library/pyproject.toml").read_text())
        self.assertFalse((VIEW / "library/ascriptor/runtime/aclnn/template").exists())

    def test_oat_uses_one_repository_rule(self):
        root = ET.parse(REPO / "OAT.xml").getroot()
        self.assertEqual(root.findtext(".//licensefile"), "LICENSE")
        rules = [item for item in root.findall(".//policyitem")
                 if item.attrib.get("type") == "license"]
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].attrib["name"], "CANN-2.0")
        self.assertEqual(rules[0].attrib["path"], ".*")

    def test_plugin_python_files_have_repository_header(self):
        git = shutil.which("git")
        self.assertIsNotNone(git)
        paths = subprocess.check_output([git, "-C", str(REPO), "ls-files", "-z", "--", "*.py"])
        for name in filter(None, paths.decode().split("\0")):
            if not name.startswith("cannbot-skills/plugins-official/pypto-pro-op-orchestrator/"):
                continue
            path = REPO / name
            if not path.is_file():
                continue
            lines = path.read_text(encoding="utf-8").splitlines()[:25]
            header = "\n".join(lines)
            with self.subTest(path=name):
                self.assertIn("CANN Open Software License Agreement Version 2.0", header)
                self.assertTrue(any(line.startswith("# Copyright") for line in lines))
                end = next(i for i, line in enumerate(lines)
                           if line.startswith(("# See LICENSE", "# See the License")))
                self.assertTrue(lines[end + 1].startswith("# ---"))


if __name__ == "__main__":
    unittest.main()
