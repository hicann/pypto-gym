#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Negative controls for `check_kb_integrity.py`.

Every case here is a defect that once passed a full green run. That is the point: a
check is only worth its `ok` line if something makes it go red, and each of these was
found by review *after* the check it exercises had been written and manually "verified".

Each test injects one defect into a real KB page, runs the single check that should
catch it, and restores the page. Run with::

    python -m unittest discover -s cannbot-skills/ops/pypto-pro-op-kb/tests
"""

from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path

KB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = KB_ROOT.parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_kb_integrity", KB_ROOT / "check_kb_integrity.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_kb_integrity"] = mod
    spec.loader.exec_module(mod)
    return mod


kb = _load()


class _Injects(unittest.TestCase):
    """Append text to a page, run a check, always put the page back."""

    def _with(self, page: Path, extra: str, check):
        original = page.read_text(encoding="utf-8")
        try:
            page.write_text(original + extra, encoding="utf-8")
            return [e for e in check() if not e.startswith("SKIP:")], [
                e for e in check() if e.startswith("SKIP:")]
        finally:
            page.write_text(original, encoding="utf-8")


class TestArtifactCitations(_Injects):
    PAGE = KB_ROOT / "references" / "terminology.md"

    def test_brace_citation_needs_every_member(self):
        """One present member must not vouch for an absent one.

        `tools/verify_codegen.py` genuinely exists, so an `any()` resolver passes the
        pair. This is the exact shape that slipped through review.
        """
        errs, _ = self._with(
            self.PAGE, "\n\nsee `tools/{verify_codegen.py,NEVER_EXISTED.py}`.\n",
            kb.check_reference_artifact_citations)
        self.assertTrue(any("NEVER_EXISTED" in e for e in errs), errs)

    def test_docs_directory_is_scanned(self):
        """`docs/` carries citations too; excluding it hid a dangling path."""
        docs = REPO_ROOT / "docs"
        if not docs.is_dir():
            self.skipTest("no docs/ in this checkout")
        page = sorted(docs.glob("*.md"))[0]
        errs, _ = self._with(
            page, "\n\nsee `custom/does_not_exist/nothing.py`.\n",
            kb.check_reference_artifact_citations)
        self.assertTrue(any("does_not_exist" in e for e in errs), errs)

    def test_unverifiable_branch_is_not_silently_passed(self):
        """An unfetched and a fabricated ref are indistinguishable -> SKIP, never ok."""
        _, skips = self._with(
            self.PAGE,
            "\n\nretained `custom/nope/absent.py` on branch `totally/made-up-branch`\n",
            kb.check_reference_artifact_citations)
        self.assertTrue(any("made-up-branch" in s for s in skips), skips)


class TestSectionCitations(_Injects):
    PAGE = KB_ROOT / "references" / "terminology.md"

    def test_section_number_inside_link_text(self):
        errs, _ = self._with(
            self.PAGE, "\n\nsee [ROUTER §99](../ROUTER.md).\n",
            kb.check_section_citations)
        self.assertTrue(any("§99" in e for e in errs), errs)

    def test_every_number_in_a_list_is_checked(self):
        """`§1, §99` -- taking only the first left the second unchecked."""
        errs, _ = self._with(
            self.PAGE, "\n\nsee [terminology](terminology.md) §1, §99.\n",
            kb.check_section_citations)
        self.assertTrue(any("§99" in e for e in errs), errs)


# Matched here rather than imported from the checker: a test that reaches into another
# module's protected names is coupled to its internals, and this one only needs to find a
# document with two or more Parts.
_PART_HEADING_RE = re.compile(r"^# Part ([IVX]+)\b", re.M)


class TestMultiPartEntries(_Injects):
    def test_prefix_naming_an_absent_part_is_reported(self):
        """A Part the document does not define was skipped, not flagged."""
        page = self._multipart_page()
        errs, _ = self._with(page, "\n\nsee entry IV-5.\n",
                             kb.check_multi_part_entry_citations)
        self.assertTrue(any("IV-5" in e for e in errs), errs)

    def test_bare_entry_still_requires_a_part(self):
        page = self._multipart_page()
        errs, _ = self._with(page, "\n\nsee entry 7 for details.\n",
                             kb.check_multi_part_entry_citations)
        self.assertTrue(any("entry 7" in e for e in errs), errs)

    def _multipart_page(self) -> Path:
        docs = REPO_ROOT / "docs"
        candidates = [
            page
            for page in (sorted(docs.glob("*.md")) if docs.is_dir() else [])
            if len(_PART_HEADING_RE.findall(page.read_text(encoding="utf-8"))) >= 2
        ]
        if not candidates:
            self.skipTest("no multi-Part document in this checkout")
        return candidates[0]


class TestValidatedPatternEvidence(_Injects):
    """The path that guards `validated` rows -- the strongest claim in the KB."""

    def test_brace_citation_needs_every_member(self):
        """`any()` was fixed in one checker and missed here for a release."""
        errs, _ = self._with(
            self._validated_page(),
            "\n\nretained `tools/{verify_codegen.py,NEVER_EXISTED.py}`\n",
            kb.check_pattern_validated_claims_cite_evidence)
        self.assertTrue(any("NEVER_EXISTED" in e for e in errs), errs)

    def test_fabricated_branch_is_not_silently_passed(self):
        _, skips = self._with(
            self._validated_page(),
            "\n\nretained `custom/nope/absent.py` on branch `totally/made-up-branch`\n",
            kb.check_pattern_validated_claims_cite_evidence)
        self.assertTrue(any("made-up-branch" in s for s in skips), skips)

    def _validated_page(self) -> Path:
        page = KB_ROOT / "patterns" / "vec-row-reduce-broadcast.md"
        if not page.is_file():
            self.skipTest("expected validated pattern page is absent")
        return page


class TestCrossPackageLinks(_Injects):
    def test_a_link_out_of_the_package_must_resolve(self):
        """The deeper "installed form" resolved under neither resolver."""
        page = KB_ROOT / "references" / "terminology.md"
        errs, _ = self._with(
            page,
            "\n\nsee [perf](../../../pypto-pro-op-perf-tune/SKILL.md).\n",
            kb.check_cross_package_links)
        self.assertTrue(any("pypto-pro-op-perf-tune" in e for e in errs), errs)


class TestReportCopiesAgree(_Injects):
    """Two copies of the same entries drifted twice before anything compared them."""

    KB_COPY = KB_ROOT / "references" / "pypto-pro-dsl-limitations-a5.md"

    def test_a_renamed_entry_title_is_caught(self):
        """One copy said "evaluation server" where the other said "deployed runtime"."""
        if not self.KB_COPY.is_file():
            self.skipTest("KB report copy is absent")
        original = self.KB_COPY.read_text(encoding="utf-8")
        marker = "### 18. "
        if marker not in original:
            self.skipTest("entry 18 is not in this copy")
        try:
            self.KB_COPY.write_text(
                original.replace(marker, "### 18. DRIFTED ", 1), encoding="utf-8")
            errs = [e for e in kb.check_report_copies_agree() if not e.startswith("SKIP:")]
        finally:
            self.KB_COPY.write_text(original, encoding="utf-8")
        self.assertTrue(any("DRIFTED" in e for e in errs), errs)


class TestCleanTree(unittest.TestCase):
    def test_the_repository_passes_every_check(self):
        failures = {}
        for label, check in kb.CHECKS:
            errs = [e for e in check() if not e.startswith("SKIP:")]
            if errs:
                failures[label] = errs
        self.assertEqual(failures, {}, failures)


if __name__ == "__main__":
    unittest.main()
