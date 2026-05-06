#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for verifier source file materialization."""

from __future__ import annotations

import tempfile
from pathlib import Path

from benchmark.verifier.__main__ import _collect_pypto_source_files as cli_collect
from benchmark.verifier.kernel_verifier import KernelVerifier
from benchmark.verifier_runner import collect_pypto_source_files as runner_collect


def _write_case(op_dir: Path) -> None:
    (op_dir / "LongOp_impl.py").write_text(
        "def LongOp_wrapper(x):\n    return x\n",
        encoding="utf-8",
    )
    (op_dir / "LongOp_pypto_impl.py").write_text(
        "from LongOp_impl import \\\n"
        "    LongOp_wrapper\n\n"
        "class ModelNew:\n"
        "    def forward(self, x):\n"
        "        return LongOp_wrapper(x)\n",
        encoding="utf-8",
    )


def _assert_collects_original_sources(collect_func) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        op_dir = Path(tmp)
        _write_case(op_dir)

        source_files = collect_func(op_dir, "LongOp")

    assert set(source_files) == {"LongOp_impl.py", "LongOp_pypto_impl.py"}
    assert "from LongOp_impl import \\" in source_files["LongOp_pypto_impl.py"]


def test_collect_pypto_source_files_preserves_imports() -> None:
    _assert_collects_original_sources(runner_collect)
    _assert_collects_original_sources(cli_collect)


def test_kernel_verifier_writes_multiple_source_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        verify_dir = Path(tmp) / "verify"
        verifier = KernelVerifier(
            op_name="LongOp",
            framework_code="class Model: pass\n",
            config={"log_dir": tmp},
        )
        task_info = {
            "source_files": {
                "LongOp_impl.py": "def LongOp_wrapper(x):\n    return x\n",
                "LongOp_pypto_impl.py": "from LongOp_impl import LongOp_wrapper\n",
            }
        }

        verifier._write_source_artifacts(task_info, verify_dir)

        assert (verify_dir / "LongOp_torch.py").read_text(encoding="utf-8") == "class Model: pass\n"
        assert (verify_dir / "LongOp_impl.py").is_file()
        assert (verify_dir / "LongOp_pypto_impl.py").is_file()
