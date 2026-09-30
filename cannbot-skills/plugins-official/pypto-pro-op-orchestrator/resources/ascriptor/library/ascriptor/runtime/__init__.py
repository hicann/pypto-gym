# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Launchers and ``OpExec``. Every one of them runs on the machine that calls it.

* :mod:`.project` generates the aclnn custom-op project from a cce artifact (D-013 template directory).
* :mod:`.harness` generates the host test program and writes / reads its argument files.
* :mod:`.build` builds the package and the harness and runs it (no NPU idle check by decision of the
  maintainer for M5; a workspace-local ``flock`` serialises runs).
* :mod:`.board` holds what a box is: its environment script, its card, its lock, its core count.
  ``Board.local()`` answers "which of these am I", and the device launchers ask only that.
* :mod:`.opexec` ties them together behind ``OpExec(kernel, launcher=...)``.

No launcher reaches another machine: run the source snapshot on the card machine
so inputs, reference, execution and comparison remain together.
"""

from .board import Board, BoardError
from .opexec import LAUNCHERS, OpExec, compile_kernel, run_case

__all__ = ["OpExec", "LAUNCHERS", "Board", "BoardError", "compile_kernel", "run_case"]
