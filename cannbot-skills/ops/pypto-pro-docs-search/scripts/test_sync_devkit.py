#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exercise the standalone Pro cache CLI against a local source fixture."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("sync_devkit.py")
API_PAGE = "SIMD-API/basic_data_structures/TileType.md"
CODE_BLOCK = '```python\ndiagram = "[tile](../basic_data_structures/TileType.md)"\n```\n'
INLINE_CODE = '`[tile](../basic_data_structures/TileType.md)`'


def _snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


class ProDevkitTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="pro cache test ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source with spaces"
        self.cache = self.root / "cache with spaces"
        self.samples = self.root / "official samples.md"
        self.samples.write_text(
            "# Official samples\n\n"
            "| Sample | Path |\n|---|---|\n"
            "| add | `pro_ops/element_wise/test_add.py` |\n"
            "| matmul | `pro_ops/matmul/test_matmul_8k_example.py` |\n",
            encoding="utf-8",
        )
        self.expected_samples = {
            "element_wise/test_add.py", "matmul/test_matmul_8k_example.py",
        }
        self._create_source()

    def test_provision_is_standalone_and_preserves_only_pro_resources(self) -> None:
        standalone = self.root / "standalone" / "sync_devkit.py"
        standalone.parent.mkdir()
        shutil.copy2(SCRIPT, standalone)
        before = _snapshot(self.source)
        self._assert_success(self._run(script=standalone))
        self.assertEqual(_snapshot(self.source), before)
        self.assertFalse((self.cache / "docs").is_symlink())
        self.assertFalse((self.cache / "pro_ops").is_symlink())
        self.assertTrue((self.cache / ".pro-docs-v2").is_file())
        actual = {
            path.relative_to(self.cache / "pro_ops").as_posix() for path in (self.cache / "pro_ops").rglob("*.py")
        }
        self.assertEqual(actual, self.expected_samples)
        for relative in (
            "docs/api/tensor_api", "docs/install", "docs/contribute",
            "docs/guide/programming_guide/tensor", "docs/guide/quick_start/tensor",
            "docs/guide/introduction_tensor.md", "docs/guide/introduction_pro.md",
        ):
            self.assertFalse((self.cache / relative).exists(), relative)
        for relative in (
            "docs/api/pro_api/index.md", "docs/pypto_pro/api/index.md",
            "docs/guide/programming_guide/pro/index.md",
            "docs/guide/quick_start/pro/index.md", "docs/guide/introduction.md",
        ):
            self.assertTrue((self.cache / relative).is_file(), relative)
        native = self.cache / "docs/api/pro_api" / API_PAGE
        self.assertEqual(native.read_bytes(), (self.source / "docs/zh/api/pro_api" / API_PAGE).read_bytes())
        copied = self.cache / "docs/pypto_pro/api" / API_PAGE
        text = copied.read_text(encoding="utf-8")
        self.assertIn(CODE_BLOCK, text)
        self.assertIn(INLINE_CODE, text)
        for label in ("layout", "Guide"):
            match = re.search(r"\[" + label + r"\]\(([^)]+)\)", text)
            self.assertIsNotNone(match)
            self.assertTrue((copied.parent / match.group(1)).is_file(), label)
        self.assertEqual(
            (self.cache / "docs/guide/figures/pro/layout.png").read_bytes(),
            (self.source / "docs/zh/guide/figures/pro/layout.png").read_bytes(),
        )
        result = self._run("--check", script=standalone, source=self.root / "absent source")
        self._assert_success(result)
        self.assertEqual(result.stdout.strip(), "READY")
        for relative in ("guide/programming_guide/pro", "guide/quick_start/pro"):
            index = self.cache / "docs" / relative / "index.md"
            saved = index.read_bytes()
            try:
                index.unlink()
                self._assert_not_ready()
            finally:
                index.write_bytes(saved)

    def test_missing_sample_never_marks_new_cache_ready(self) -> None:
        (self.source / "python/tests/st/pypto_pro/frontend/element_wise/test_add.py").unlink()
        before = _snapshot(self.source)
        result = self._run()
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertFalse((self.cache / ".pro-docs-v2").exists())
        self.assertEqual(_snapshot(self.source), before)
        self._assert_not_ready()

    def test_failed_refresh_preserves_existing_docs_and_samples(self) -> None:
        self._assert_success(self._run())
        before_docs = _snapshot(self.cache / "docs")
        before_samples = _snapshot(self.cache / "pro_ops")
        (self.source / "python/tests/st/pypto_pro/frontend/element_wise/test_add.py").unlink()
        result = self._run()
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertEqual(_snapshot(self.cache / "docs"), before_docs)
        self.assertEqual(_snapshot(self.cache / "pro_ops"), before_samples)
        self.assertFalse((self.cache / ".pro-docs-v2").exists())
        self._assert_not_ready()

    def test_missing_guide_sources_are_rejected(self) -> None:
        for relative in (
            "docs/zh/guide/programming_guide/pro",
            "docs/zh/guide/quick_start/pro",
            "docs/zh/guide/introduction.md",
            "docs/zh/guide/programming_guide/pro/index.md",
            "docs/zh/guide/quick_start/pro/index.md",
        ):
            with self.subTest(source=relative):
                source = self.source / relative
                saved = self.root / "missing guide source"
                source.rename(saved)
                try:
                    result = self._run()
                    self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
                    self.assertFalse((self.cache / ".pro-docs-v2").exists())
                    self._assert_not_ready()
                finally:
                    saved.rename(source)

    def test_check_rejects_wrong_names_even_when_counts_match(self) -> None:
        self._assert_success(self._run())
        original = self.cache / "pro_ops/matmul/test_matmul_8k_example.py"
        for name in ("test_matmul_8K_example.py", "unlisted_replacement.py"):
            with self.subTest(filename=name):
                renamed = original.with_name(name)
                original.rename(renamed)
                try:
                    self.assertEqual(len(list((self.cache / "pro_ops").rglob("*.py"))), 2)
                    self._assert_not_ready()
                finally:
                    renamed.rename(original)

    def test_invalid_explicit_source_does_not_fall_back_to_download(self) -> None:
        result = self._run(source=self.root / "absent source")
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertFalse((self.cache / ".pro-docs-v2").exists())
        self._assert_not_ready()

    def test_cache_cannot_replace_local_source(self) -> None:
        for number, relative in enumerate((".", "docs/source", "pro_ops/source")):
            with self.subTest(source=relative):
                self.cache = self.root / f"overlap cache {number}"
                source = self.cache / relative
                shutil.copytree(self.source, source)
                before = _snapshot(source)
                result = self._run(source=source)
                self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
                self.assertEqual(_snapshot(source), before)
                self.assertFalse((self.cache / ".pro-docs-v2").exists())

    def test_samples_argument_is_required(self) -> None:
        result = self._run(samples=False)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("--samples", result.stderr)

    def test_pinned_download_with_empty_git_template(self) -> None:
        template = self.root / "empty git template"
        template.mkdir()
        for arguments in (
            ["init", "--template=" + str(template)],
            ["add", "."],
            ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
             "-c", "commit.gpgsign=false", "commit", "-m", "source fixture"],
        ):
            subprocess.run(["git", *arguments], cwd=self.source, capture_output=True, check=True)
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.source, text=True,
        ).strip()
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--samples", str(self.samples), "--pin", revision],
            cwd=self.root, capture_output=True, text=True, encoding="utf-8", timeout=30,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONPATH": "",
                 "PYPTO_DEVKIT_DIR": str(self.cache), "PYPTO_SRC_URL": self.source.as_uri(),
                 "GIT_TEMPLATE_DIR": str(template), "GIT_ALLOW_PROTOCOL": "file"},
        )
        self._assert_success(result)
        self.assertEqual(
            {path.relative_to(self.cache / "pro_ops").as_posix()
             for path in (self.cache / "pro_ops").rglob("*.py")},
            self.expected_samples,
        )
        self.assertFalse((self.cache / "docs/api/tensor_api").exists())
        self.assertEqual(
            (self.cache / "docs/api/pro_api" / API_PAGE).read_bytes(),
            (self.source / "docs/zh/api/pro_api" / API_PAGE).read_bytes(),
        )

    def _create_source(self) -> None:
        files = {
            "docs/zh/api/pro_api/index.md": f"# Pro API\n\n[Tile]({API_PAGE})\n",
            f"docs/zh/api/pro_api/{API_PAGE}": (
                "# Tile\n\n"
                "![layout](../../../../guide/figures/pro/layout.png)\n\n"
                "[Guide](../../../../guide/programming_guide/pro/development/tiling.md)\n\n"
                + CODE_BLOCK + "\n" + INLINE_CODE + "\n"
            ),
            "docs/zh/guide/programming_guide/pro/index.md": "# Programming\n\n[Tiling](development/tiling.md)\n",
            "docs/zh/guide/programming_guide/pro/development/tiling.md": (
                "# Tiling\n\n![layout](../../../figures/pro/layout.png)\n"
            ),
            "docs/zh/guide/quick_start/pro/index.md": "# Quick start\n\n[Hello](hello.md)\n",
            "docs/zh/guide/quick_start/pro/hello.md": "# Hello Pro\n",
            "docs/zh/guide/introduction.md": "# PyPTO\n\nShared introduction for Tensor and Pro.\n",
            "docs/zh/api/tensor_api/index.md": "# Excluded Tensor API\n",
            "docs/zh/guide/programming_guide/tensor/index.md": "# Excluded Tensor guide\n",
            "docs/zh/guide/quick_start/tensor/index.md": "# Excluded Tensor quick start\n",
            "docs/zh/guide/introduction_tensor.md": "# Excluded Tensor introduction\n",
            "docs/zh/install/index.md": "# Excluded installation\n",
            "docs/zh/contribute/index.md": "# Excluded contribution\n",
            "python/tests/st/pypto_pro/frontend/element_wise/test_add.py": "def test_add():\n    pass\n",
            "python/tests/st/pypto_pro/frontend/matmul/test_matmul_8k_example.py": "def test_matmul():\n    pass\n",
            "python/tests/st/pypto_pro/frontend/unlisted/test_extra.py": (
                "raise AssertionError('not a designated sample')\n"
            ),
        }
        for relative, text in files.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        (self.source / "docs/zh/guide/figures/pro").mkdir(parents=True)
        (self.source / "docs/zh/guide/figures/pro/layout.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")

    def _run(
        self, *arguments: str, script: Path = SCRIPT,
        source: Path | None = None, samples: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        environment = {
            **os.environ,
            "PYPTO_DEVKIT_DIR": str(self.cache),
            "PYPTO_SRC": str(self.source if source is None else source),
            "PYPTO_SRC_URL": "https://example.invalid/pypto.git",
            "PYTHONPATH": "",
            "PYTHONUTF8": "1",
        }
        command = [sys.executable, str(script), *arguments]
        if samples:
            command.extend(["--samples", str(self.samples)])
        return subprocess.run(
            command, cwd=self.root, env=environment,
            text=True, encoding="utf-8", capture_output=True, check=False, timeout=30,
        )

    def _assert_success(self, result: subprocess.CompletedProcess[str]) -> None:
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def _assert_not_ready(self) -> None:
        result = self._run("--check")
        self.assertEqual(result.returncode, 4, result.stdout + result.stderr)
        self.assertIn("NEED_PROVISION", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
