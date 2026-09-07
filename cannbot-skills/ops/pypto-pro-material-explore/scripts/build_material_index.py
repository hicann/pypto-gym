#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Generate PRO_MATERIAL_INDEX.md from the prepared PyPTO-Pro devkit."""

from __future__ import annotations

import argparse
import logging
import os
import re
import string
import sys
import tempfile
from pathlib import Path


SAMPLE_HEADER = ("#", "算子名称", "缓存相对路径", "类型", "描述")
SAMPLE_PATH_RE = re.compile(r"`(pro_ops/[^`|]+\.py)`")
TEMPLATE_FIELDS = (
    "api_count", "api_index", "api_rows", "sample_count", "sample_rows",
    "tutorial_count", "tutorial_rows",
)
TEMPLATE_ORDER = (
    "## §A API 文档", "- `{api_index}`", "{api_rows}",
    "## §B 官方指定算子样例", "{sample_rows}",
    "## §C 教程与设计指南", "{tutorial_rows}",
)
LOGGER = logging.getLogger(__name__)


class MaterialIndexError(ValueError):
    """Raised when the material cache or manifest is incomplete."""


def _configure_logging() -> None:
    formatter = logging.Formatter("%(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    LOGGER.handlers = [stdout_handler, stderr_handler]
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in LOGGER.handlers:
        handler.setFormatter(formatter)


def _scan(root: Path, devkit: Path, label: str) -> list[str]:
    if not root.is_dir():
        raise MaterialIndexError(f"{label} directory is missing: {root}")
    files = sorted(
        path.relative_to(devkit).as_posix()
        for path in root.rglob("*.md")
        if path.is_file()
    )
    if not files:
        raise MaterialIndexError(f"{label} contains no Markdown files: {root}")
    return files


def _table_cells(line: str) -> list[str]:
    line = line.strip()
    if not line.startswith("|") or not line.endswith("|"):
        return []
    return [cell.strip() for cell in line[1:-1].split("|")]


def _samples(manifest: Path, devkit: Path) -> tuple[list[str], list[str]]:
    if not manifest.is_file():
        raise MaterialIndexError(f"official sample manifest is missing: {manifest}")
    lines = manifest.read_text(encoding="utf-8").rstrip().splitlines()
    headers = [index for index, line in enumerate(lines) if tuple(_table_cells(line)) == SAMPLE_HEADER]
    if len(headers) != 1:
        raise MaterialIndexError("official sample manifest must contain one canonical table")
    header = headers[0]
    separator = _table_cells(lines[header + 1]) if header + 1 < len(lines) else []
    if len(separator) != 5 or any(not re.fullmatch(r":?-{3,}:?", cell) for cell in separator):
        raise MaterialIndexError(f"malformed sample table separator at line {header + 2}")

    rows = []
    paths = []
    numbers = []
    seen = set()
    for line_number, line in enumerate(lines[header + 2:], start=header + 3):
        cells = _table_cells(line)
        path_match = SAMPLE_PATH_RE.fullmatch(cells[2]) if len(cells) == 5 else None
        if not path_match or not re.fullmatch(r"[0-9]+", cells[0]):
            raise MaterialIndexError(f"malformed sample row at line {line_number}")
        path = path_match.group(1)
        if path in seen:
            raise MaterialIndexError(f"duplicate official sample path at line {line_number}: {path}")
        seen.add(path)
        numbers.append(int(cells[0]))
        paths.append(path)
        rows.append(line.strip())
    if not rows or numbers != list(range(1, len(rows) + 1)):
        raise MaterialIndexError("official sample rows must be contiguous from 1")

    pro_ops = devkit / "pro_ops"
    if not pro_ops.is_dir():
        raise MaterialIndexError(f"official sample directory is missing: {pro_ops}")
    actual = {
        path.relative_to(devkit).as_posix()
        for path in pro_ops.rglob("*.py")
        if path.is_file()
    }
    missing = [path for path in paths if path not in actual]
    if missing:
        raise MaterialIndexError("official sample is missing: " + ", ".join(missing))
    return rows, paths


def build_index(devkit: Path, manifest: Path) -> tuple[str, tuple[int, int, int]]:
    devkit = devkit.resolve()
    api = _scan(devkit / "docs/pypto_pro/api", devkit, "API")
    tutorials = _scan(devkit / "docs/pypto_pro/tutorials", devkit, "tutorial")
    sample_rows, sample_paths = _samples(manifest, devkit)
    api_index = "docs/pypto_pro/api/index.md"
    if api_index not in api:
        raise MaterialIndexError(f"API root index is missing: {api_index}")

    template = _template_path().read_text(encoding="utf-8")
    try:
        parsed_fields = list(string.Formatter().parse(template))
    except ValueError as error:
        raise MaterialIndexError(f"invalid material index template: {error}") from error
    fields = []
    for _, field, format_spec, conversion in parsed_fields:
        if field is None:
            continue
        if field not in TEMPLATE_FIELDS or format_spec or conversion:
            raise MaterialIndexError(f"invalid material index template field: {field!r}")
        fields.append(field)
    invalid_fields = [field for field in TEMPLATE_FIELDS if fields.count(field) != 1]
    if invalid_fields:
        raise MaterialIndexError("template fields must appear exactly once: " + ", ".join(invalid_fields))
    template_lines = template.splitlines()
    if any(template_lines.count(item) != 1 for item in TEMPLATE_ORDER):
        raise MaterialIndexError("template sections and row fields must appear exactly once")
    positions = [template_lines.index(item) for item in TEMPLATE_ORDER]
    if positions != sorted(positions):
        raise MaterialIndexError("template sections and row fields are out of order")
    try:
        content = template.format(
            api_count=len(api), api_index=api_index,
            api_rows="\n".join(f"- `{path}`" for path in api if path != api_index),
            sample_count=len(sample_paths), sample_rows="\n".join(sample_rows),
            tutorial_count=len(tutorials),
            tutorial_rows="\n".join(f"- `{path}`" for path in tutorials),
        )
    except (IndexError, KeyError, ValueError) as error:
        raise MaterialIndexError(f"invalid material index template: {error}") from error
    return content, (len(api), len(sample_paths), len(tutorials))


def manifest_path() -> Path:
    """Return the canonical official-sample manifest used to build material indexes."""
    return Path(__file__).resolve().parents[1] / "references/official_samples.md"


def _template_path() -> Path:
    return Path(__file__).resolve().parents[1] / "templates/pro_material_index.md"


def _write_index(output: Path, content: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devkit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    _configure_logging()
    try:
        devkit = Path(args.devkit).resolve()
        output = Path(args.output).resolve()
        if not devkit.is_dir():
            raise MaterialIndexError(f"devkit directory is missing: {devkit}")
        if output.is_dir():
            raise MaterialIndexError(f"output must be a file path: {output}")
        if output == devkit or devkit in output.parents:
            raise MaterialIndexError(f"output must be outside devkit: {output}")
        protected = {Path(__file__).resolve(), manifest_path().resolve(), _template_path().resolve()}
        if output in protected:
            raise MaterialIndexError(f"output must not overwrite generator inputs: {output}")

        content, counts = build_index(devkit, manifest_path())
        if args.check:
            if not output.is_file() or output.read_text(encoding="utf-8") != content:
                raise MaterialIndexError(f"material index is stale: {output}")
        else:
            _write_index(output, content)
        LOGGER.info("api=%d samples=%d tutorials=%d", *counts)
        return 0
    except (MaterialIndexError, OSError, RuntimeError, UnicodeError) as error:
        LOGGER.error("ERROR: %s", error)
        return 2


if __name__ == "__main__":
    sys.exit(main())
