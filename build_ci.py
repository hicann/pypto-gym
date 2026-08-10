#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compatibility entry point for the PyPTO build controller.

The authoritative build implementation lives in the sibling ``pypto``
repository. This wrapper preserves existing commands issued from pypto-gym
while ensuring that PyPTO's own build process is always used.
"""

import os
import sys
from pathlib import Path


def main() -> None:
    gym_root = Path(__file__).resolve().parent
    pypto_root = gym_root.parent / "pypto"
    pypto_build_script = pypto_root / "build_ci.py"

    if not pypto_build_script.is_file():
        raise FileNotFoundError(
            f"PyPTO build script not found: {pypto_build_script}. "
            "Clone the PyPTO repository next to pypto-gym first."
        )

    os.chdir(pypto_root)
    os.execv(
        sys.executable,
        [sys.executable, str(pypto_build_script), *sys.argv[1:]],
    )


if __name__ == "__main__":
    main()
