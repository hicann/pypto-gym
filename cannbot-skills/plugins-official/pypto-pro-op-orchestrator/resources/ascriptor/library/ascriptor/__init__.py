# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ascriptor — a Python-embedded, instruction-level language for the Ascend AI Core.

The package is organised as a compiler:

    frontend  ->  Surface IR  ->  passes  ->  Lowered IR  ->  backends  ->  runtime
                  (ir/)          (passes/)   (ir/)           (backends/)   (runtime/)

The IR is the product; see ``docs/rfc/0001-ir.md``. Milestone status lives in
``docs/decisions.md``.
"""

__version__ = "0.1.0"

IR_VERSIONS = {"surface": "surface/1", "lowered": "lowered/1"}

__all__ = ["__version__", "IR_VERSIONS"]
