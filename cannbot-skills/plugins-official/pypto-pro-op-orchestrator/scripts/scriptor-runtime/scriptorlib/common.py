# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Filesystem and evidence primitives shared by the CLI and OpenCode adapter."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


class ContractError(ValueError):
    """A request or artifact does not satisfy the workflow contract."""


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text: str) -> Any:
    def nonfinite(value):
        raise ContractError(f"non-finite JSON value: {value}")
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=nonfinite)


def read_json(path: Path) -> Any:
    return parse_json(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with tempfile.TemporaryDirectory(prefix=f".{path.name}.", dir=path.parent) as temporary:
        candidate = Path(temporary) / path.name
        candidate.touch(mode=0o600, exist_ok=False)
        with candidate.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(candidate, path)


def confined(root: Path, relative: str, *, must_exist: bool = False) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ContractError(f"expected a non-empty relative path: {relative!r}")
    base = root.resolve()
    target = (base / relative).resolve()
    if not target.is_relative_to(base) or target == base:
        raise ContractError(f"path escapes its artifact directory: {relative!r}")
    if must_exist and not target.is_file():
        raise ContractError(f"required artifact is missing: {relative}")
    return target


def json_pointer(value: Any, pointer: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ContractError("metric pointer must be an absolute JSON pointer")
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(f"{label} must be a finite number")
    if positive and value <= 0:
        raise ContractError(f"{label} must be positive")
    return float(value)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load Python module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def config_root(value: str | Path | None = None) -> Path:
    if value:
        result = Path(value).resolve()
    else:
        # Installed at .opencode/scriptor/scripts/scriptorlib/common.py.
        result = Path(__file__).resolve().parents[3]
    if not (result / "scriptor-install.json").is_file():
        raise ContractError("scriptor is not installed here; pass --config-root or run init.sh")
    return result
