# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""IR -> IR passes and the pass manager.

Every pass reads and writes IR only, runs the verifier before and after, and records an
``origin`` entry on every op it inserts or rewrites (RFC-0001 §8). The canonical pipeline,
Surface IR to Lowered IR, is: cellfold -> desugar -> device_lower -> gmbuff -> crosssync ->
autosync -> events -> addr_alloc -> split_sides -> events_restamp -> dce -> local_mutex ->
mutex_coalesce -> scalar_simplify -> liveness.
Backends may append backend passes.
"""

from .manager import Explain, Pass, PassContext, PassError, PassManager, PassRun
from .pipeline import PIPELINE, lower, pass_named

__all__ = ["PIPELINE", "Explain", "Pass", "PassContext", "PassError", "PassManager", "PassRun", "lower", "pass_named"]
